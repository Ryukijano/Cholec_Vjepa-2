"""
Trainer for V-JEPA 2.1 World Model with Multi-Scale Prediction + DETR/ReID
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import time
from collections import defaultdict
import logging

from ..models.vjepa_world_model import VJEPAWorldModel
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.metrics import AverageMeter, MetricLogger


class WorldModelTrainer:
    """
    Trainer for the complete V-JEPA World Model architecture.
    Handles:
    - Multi-scale future prediction loss
    - DETR detection loss
    - ReID embedding loss
    - Curriculum learning (gradual ReID weight increase)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[VJEPAWorldModel] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda'
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Build model
        if model is None:
            self.model = self.build_model(config)
        else:
            self.model = model
        
        self.model.to(self.device)
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer and scheduler
        self.optimizer = self.build_optimizer(config)
        self.scheduler = self.build_scheduler(config)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Setup output directory
        self.output_dir = Path(config['meta']['folder'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Trainer initialized. Output: {self.output_dir}")
    
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def build_model(self, config: Dict[str, Any]) -> VJEPAWorldModel:
        """Build the world model from config."""
        model_cfg = config.get('model', {})
        supervised_cfg = config.get('supervised', {})
        predictor_cfg = config.get('predictor', {})
        data_cfg = config.get('data', {})
        detr_cfg = supervised_cfg.get('detr_head', {})
        reid_cfg = supervised_cfg.get('reid_head', {})
        
        encoder_checkpoint = supervised_cfg.get('pretrained_checkpoint')
        
        # Determine encoder type from checkpoint name
        if encoder_checkpoint:
            if 'dinov2' in encoder_checkpoint.lower() or 'dino' in encoder_checkpoint.lower():
                encoder_type = 'dinov2'
                model_name = 'dinov2_vitb14'
            else:
                encoder_type = 'vjepa'
                model_name = 'vit_base'
            self.logger.info(f"Using local checkpoint: {encoder_checkpoint} ({encoder_type})")
        else:
            # Default to V-JEPA
            encoder_type = 'vjepa'
            model_name = 'vit_base'
            self.logger.warning("No pretrained encoder specified! Using random initialization.")
        
        # Hierarchical layers
        layer_indices = model_cfg.get('layer_indices', [4, 8, 12, -1])
        
        return VJEPAWorldModel(
            encoder_type=encoder_type,
            encoder_checkpoint=encoder_checkpoint,
            model_name=model_name,
            encoder_dim=model_cfg.get('embed_dim', 768),
            neck_dim=detr_cfg.get('neck_dim', 256),
            num_frames=data_cfg.get('dataset_fpcs', [16])[0],
            img_size=data_cfg.get('crop_size', 384),
            layer_indices=layer_indices,
            predictor_hidden_dim=predictor_cfg.get('hidden_dim', 768),
            predictor_num_layers=predictor_cfg.get('num_layers', 8),
            predictor_num_heads=predictor_cfg.get('num_heads', 12),
            prediction_horizons=predictor_cfg.get('horizons', [1, 4, 16]),
            num_tools=detr_cfg.get('num_tools', 7),
            num_queries=detr_cfg.get('num_queries', 16),
            num_decoder_layers=detr_cfg.get('num_decoder_layers', 5),
            detr_nheads=detr_cfg.get('nheads', 8),
            detr_dropout=detr_cfg.get('dropout', 0.15),
            detr_class_weight=detr_cfg.get('class_weight', 1.0),
            detr_bbox_weight=detr_cfg.get('bbox_weight', 5.0),
            detr_giou_weight=detr_cfg.get('giou_weight', 2.0),
            detr_focal_alpha=detr_cfg.get('focal_alpha', 0.25),
            detr_focal_gamma=detr_cfg.get('focal_gamma', 2.0),
            reid_embedding_dim=reid_cfg.get('embedding_dim', 256),
            reid_dropout=reid_cfg.get('dropout', 0.15),
            reid_supcon_temperature=reid_cfg.get('supcon_temperature', 0.07),
            reid_supcon_weight=reid_cfg.get('supcon_weight', 1.0),
            reid_cross_consistency_weight=reid_cfg.get('cross_consistency_weight', 0.1),
            feature_weight=config.get('world_model', {}).get('feature_weight', 1.0),
            rollout_weight=config.get('world_model', {}).get('rollout_weight', 0.3),
            temporal_consistency_weight=config.get('world_model', {}).get('temporal_consistency_weight', 0.0),
        )
    
    def build_optimizer(self, config: Dict[str, Any]) -> torch.optim.Optimizer:
        """Build optimizer."""
        opt_cfg = config.get('optimization', {})
        lr = opt_cfg.get('lr', 1e-4)
        weight_decay = opt_cfg.get('weight_decay', 0.01)
        
        # Freeze the predictor to prevent overfitting on reconstruction
        if hasattr(self.model, 'world_model') and hasattr(self.model.world_model, 'predictor'):
            for param in self.model.world_model.predictor.parameters():
                param.requires_grad = False
            self.logger.info("Predictor frozen (no_grad)")
        
        # Only optimize trainable parameters (neck, DETR, ReID)
        # Encoder and predictor are frozen
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        return AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    
    def build_scheduler(self, config: Dict[str, Any]):
        """Build learning rate scheduler."""
        opt_cfg = config.get('optimization', {})
        epochs = opt_cfg.get('epochs', 10)
        lr = opt_cfg.get('lr', 1e-4)
        
        # Cosine annealing with warmup
        return CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=lr * 0.01
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Update curriculum
        max_epochs = self.config['optimization'].get('epochs', 10)
        self.model.update_curriculum(epoch, max_epochs)
        
        metric_logger = MetricLogger()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                current_video=batch['current_video'],
                future_videos=batch.get('future_videos'),
                detr_targets=batch.get('detr_targets'),
                reid_labels=batch.get('reid_labels'),
                mode='train'
            )
            
            # Get loss
            loss = outputs['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            # Gradient accumulation if configured
            accum_steps = self.config['optimization'].get('gradient_accumulation_steps', 1)
            loss = loss / accum_steps
            loss.backward()
            
            # Gradient clipping
            clip_norm = self.config['optimization'].get('gradient_clip_norm', 1.0)
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=clip_norm
                )
            
            # Optimizer step (only every accum_steps)
            if (batch_idx + 1) % accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Log metrics
            metric_logger.update(
                loss=loss.item() * accum_steps,
                **outputs.get('loss_dict', {})
            )
            
            # Log periodically
            log_freq = self.config['optimization'].get('log_freq', 10)
            if batch_idx % log_freq == 0:
                self.logger.info(
                    f"Epoch [{epoch}/{max_epochs}] "
                    f"Batch [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item() * accum_steps:.4f}"
                )
        
        # Scheduler step
        self.scheduler.step()
        
        return metric_logger.avg_dict()
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        metric_logger = MetricLogger()
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = self.model(
                current_video=batch['current_video'],
                future_videos=batch.get('future_videos'),
                detr_targets=batch.get('detr_targets'),
                reid_labels=batch.get('reid_labels'),
                mode='train'
            )
            
            metric_logger.update(
                val_loss=outputs['total_loss'].item(),
                **{f"val_{k}": v for k, v in outputs.get('loss_dict', {}).items()}
            )
        
        return metric_logger.avg_dict()
    
    def train(self, num_epochs: Optional[int] = None):
        """Main training loop."""
        if num_epochs is None:
            num_epochs = self.config['optimization'].get('epochs', 10)
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.logger.info(f"Epoch {epoch} - Train: {train_metrics}")
            
            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate(epoch)
                self.logger.info(f"Epoch {epoch} - Val: {val_metrics}")
                
                # Save best checkpoint
                val_loss = val_metrics.get('val_loss', float('inf'))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['meta'].get('save_every_freq', 1) == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        self.logger.info("Training complete!")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step
        }
        
        if is_best:
            path = self.output_dir / 'best_checkpoint.pth.tar'
        else:
            path = self.output_dir / f'checkpoint_epoch_{epoch}.pth.tar'
        
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


class MetricLogger:
    """Simple metric logger."""
    
    def __init__(self):
        self.meters = defaultdict(AverageMeter)
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                self.meters[k].update(v)
    
    def avg_dict(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self.meters.items()}
