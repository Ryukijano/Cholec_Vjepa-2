"""
Main entry point for V-JEPA 2.1 World Model Training
"""
import argparse
import yaml
import sys
from pathlib import Path
import torch

# Enable cuDNN for proper 3D convolution support
# The aten::slow_conv3d_forward error occurs when cuDNN is disabled
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_app.trainers.world_model_trainer import WorldModelTrainer
from core_app.data.video_dataset import CholecDataset, collate_fn
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train V-JEPA 2.1 World Model with DETR/ReID tracking'
    )
    parser.add_argument(
        '--fname', 
        type=str, 
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--devices',
        nargs='+',
        default=['cuda:0'],
        help='GPU devices to use (e.g., cuda:0 cuda:1)'
    )
    parser.add_argument(
        '--debugmode',
        type=bool,
        default=False,
        help='Enable debug mode with reduced dataset'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_dataloaders(config: dict, debug: bool = False):
    """Build training and validation dataloaders."""
    data_cfg = config['data']
    
    # Training dataset
    train_dataset = CholecDataset(
        data_root=data_cfg['datasets'][0],
        split='train',
        clip_length=data_cfg.get('dataset_fpcs', [16])[0],
        prediction_horizons=config.get('predictor', {}).get('horizons', [1, 4, 16]),
        img_size=data_cfg.get('crop_size', 384),
        fps=data_cfg.get('fps', 4),
        training=True
    )
    
    # Validation dataset
    val_dataset = CholecDataset(
        data_root=data_cfg['datasets'][0],
        split='val',
        clip_length=data_cfg.get('dataset_fpcs', [16])[0],
        prediction_horizons=config.get('predictor', {}).get('horizons', [1, 4, 16]),
        img_size=data_cfg.get('crop_size', 384),
        fps=data_cfg.get('fps', 4),
        training=False
    )
    
    if debug:
        # Use small subset for debugging
        train_dataset.clips = train_dataset.clips[:32]
        val_dataset.clips = val_dataset.clips[:16]
        print(f"Debug mode: using {len(train_dataset.clips)} train clips, "
              f"{len(val_dataset.clips)} val clips")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.get('batch_size', 4),
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=data_cfg.get('pin_mem', True),
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.get('batch_size', 4),
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=data_cfg.get('pin_mem', True),
        collate_fn=collate_fn,
        drop_last=False
    )
    
    return train_loader, val_loader


def main():
    args = parse_args()
    
    # Load config
    print(f"Loading config from: {args.fname}")
    config = load_config(args.fname)
    
    # Set device
    device = args.devices[0]
    print(f"Using device: {device}")
    
    # Build dataloaders
    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(config, debug=args.debugmode)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Build trainer
    print("Building trainer...")
    trainer = WorldModelTrainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("Starting training...")
    num_epochs = config['optimization'].get('epochs', 10)
    trainer.train(num_epochs=num_epochs)
    
    print("Training complete!")


if __name__ == '__main__':
    main()
