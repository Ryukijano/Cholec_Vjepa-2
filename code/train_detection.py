#!/usr/bin/env python3
"""
Detection training on CholecTrack20 with V-JEPA2 encoder + LightweightQueryDecoder.

Loads SSL-finetuned encoder (CholecTrack20 e20.pt with LoRA), freezes it,
and trains only the detection decoder + SupCon head.

Key design choices:
  - LightweightQueryDecoder: periphery priors, spatial-biased cross-attention, 
    two-step refinement (coarse→ROI→refined)
  - Last-frame tokens (not temporal mean) — GT boxes supervise last frame only
  - Cost-sensitive Hungarian matching with inverse-frequency class weights
  - SupCon loss to pull same-class tools together across videos
  - Mid-backbone (layer 12) + final layer feature fusion

Usage:
  python train_detection.py --config configs/detect_cholectrack20.yaml
"""

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Path setup (run from code/ or project root)
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
vjepa2_path = project_root / "vjepa2"
src_path = project_root / "vjepa2" / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(vjepa2_path))
sys.path.insert(0, str(current_dir))

from vjepa2.app.vjepa.utils import init_video_model
from vjepa2.app.vjepa_cholec80.lora import apply_lora_to_encoder, load_checkpoint_into_lora_model

from models import LightweightQueryDecoder, MidBackboneHook, TokenSpatialClassifier
from losses import (
    detection_loss, detection_loss_focal, SupConProjectionHead, SupConLoss,
    DEFAULT_CLASS_WEIGHTS, hungarian_match
)
from dataset import CholecDetectDataset, collate_fn
from detection_viz import (
    run_detection_visualization, make_recall_bar_chart
)

try:
    from torchvision.ops import box_iou
except ImportError:
    box_iou = None

from torch.utils.data import WeightedRandomSampler
import wandb
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO, format='[%(levelname)-8s][%(name)-20s] %(message)s')
logger = logging.getLogger(__name__)

# Tool names for logging
TOOL_NAMES = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]


def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5*w, cy - 0.5*h, cx + 0.5*w, cy + 0.5*h], dim=-1)


def evaluate(encoder, head, mid_hook, val_loader, device, num_classes, epoch):
    """Compute per-class recall@0.5 on validation set."""
    head.eval()
    
    per_class_tp = torch.zeros(num_classes)
    per_class_gt = torch.zeros(num_classes)
    
    with torch.no_grad():
        for clips, boxes, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
            clips = clips.to(device, non_blocking=True)
            if clips.size(2) < 2:
                clips = torch.cat([clips, clips], dim=2)
            
            with torch.amp.autocast(device_type='cuda', enabled=True):
                full_tokens = encoder([clips])[0]
                mid_features = mid_hook.get_features()
                
                # Model handles temporal pooling internally
                logits, pred_boxes, _, _ = head(full_tokens.float(), mid_features.float())
            
            probs = logits.softmax(-1).cpu()
            scores, pred_cls = probs[..., :-1].max(-1)
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes.cpu())
            
            for b in range(len(boxes)):
                gt_boxes_xyxy = box_cxcywh_to_xyxy(boxes[b])
                gt_cls = labels[b]
                
                # Count GT per class
                for c in gt_cls:
                    per_class_gt[c] += 1
                
                if gt_cls.numel() == 0:
                    continue
                
                # Score-based ordering
                b_scores = scores[b]
                b_cls = pred_cls[b]
                b_boxes = pred_boxes_xyxy[b]
                order = b_scores.argsort(descending=True)
                
                matched_gt = set()
                for qi in order:
                    pc = b_cls[qi].item()
                    if pc >= num_classes:
                        continue
                    if b_scores[qi] < 0.3:
                        continue
                    
                    if box_iou is not None and gt_boxes_xyxy.numel() > 0:
                        ious = box_iou(b_boxes[qi:qi+1], gt_boxes_xyxy)[0]
                        for gi in range(len(gt_cls)):
                            if gi in matched_gt:
                                continue
                            if ious[gi] > 0.5 and pc == gt_cls[gi].item():
                                per_class_tp[pc] += 1
                                matched_gt.add(gi)
                                break
    
    per_class_recall = per_class_tp / (per_class_gt + 1e-8)
    overall_recall = per_class_tp.sum() / (per_class_gt.sum() + 1e-8)
    
    return per_class_recall, overall_recall, per_class_gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to SSL or detection checkpoint (encoder_lora or encoder key)')
    parser.add_argument('--train_dir', type=str,
                        default='../cholec_dataset/Training',
                        help='CholecTrack20 Training folder')
    parser.add_argument('--val_dir', type=str,
                        default='../cholec_dataset/Validation',
                        help='CholecTrack20 Validation folder')
    parser.add_argument('--out_dir', type=str,
                        default='../outputs/detection-cholectrack20',
                        help='Where to save checkpoints (best.pt, last.pt, e{N}.pt)')
    
    # Model
    parser.add_argument('--num_queries', type=int, default=100)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--decoder_nheads', type=int, default=8)
    parser.add_argument('--internal_dim', type=int, default=256,
                        help='Internal decoder dim (bottleneck from 1024). 0=no bottleneck')
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--temporal_pooling', type=str, default='last', choices=['last', 'mean'],
                        help='How to pool temporal tokens (last frame or temporal mean)')
    parser.add_argument('--use_aux_loss', action='store_true', help='Use auxiliary losses on intermediate decoder layers')
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--mid_layer_idx', type=int, default=12)
    
    # LoRA (must match SSL checkpoint)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--lora_start_layer', type=int, default=12)
    
    # Training
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--decoder_lr', type=float, default=5e-4)
    parser.add_argument('--lora_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--supcon_weight', type=float, default=0.5)
    parser.add_argument('--supcon_temp', type=float, default=0.07)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss instead of CE')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--use_plateau_scheduler', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=15)
    parser.add_argument('--class_balanced', action='store_true', help='Use class-balanced sampling')
    parser.add_argument('--resume', action='store_true', help='Resume from last.pt checkpoint if exists')
    parser.add_argument('--class_head_warmup', type=int, default=1, help='Epochs to train only class head (freeze bbox heads)')
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # --- Logging ---
    if args.use_wandb:
        wandb.init(project="vjepa2-cholec20-detection", config=vars(args))
    tb_writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    
    # --- Datasets ---
    logger.info("Loading datasets...")
    train_ds = CholecDetectDataset(args.train_dir, clip_len=args.clip_len)
    val_ds = CholecDetectDataset(args.val_dir, clip_len=args.clip_len)
    
    # Infer classes + distribution
    from collections import Counter
    all_labels = [l for item in train_ds.items for l in item['labels']]
    num_classes = max(all_labels) + 1 if all_labels else 7
    logger.info(f"Classes: {num_classes}, Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    dist = Counter(all_labels)
    for c in range(num_classes):
        name = TOOL_NAMES[c] if c < len(TOOL_NAMES) else f"class_{c}"
        logger.info(f"  {name}: {dist.get(c, 0)} ({100*dist.get(c,0)/len(all_labels):.1f}%)")
    
    # Class-balanced sampling: oversample rare tools
    sampler = None
    shuffle = True
    if args.class_balanced:
        label_counts = Counter(all_labels)
        total = len(all_labels)
        class_weight_map = {c: total / (len(label_counts) * cnt) for c, cnt in label_counts.items()}
        sample_weights = []
        for item in train_ds.items:
            # Use max weight among tools in this frame (upsample rare tools)
            w = max(class_weight_map.get(l, 1.0) for l in item['labels'])
            sample_weights.append(w)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        shuffle = False
        logger.info(f"Class-balanced sampling enabled. Weights: {class_weight_map}")
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn, persistent_workers=True)
    
    # --- Encoder ---
    logger.info("Initializing V-JEPA2 encoder...")
    encoder, predictor = init_video_model(
        device=device, model_name='vit_large', patch_size=16,
        max_num_frames=args.clip_len, tubelet_size=2, crop_size=224,
        pred_depth=12, pred_embed_dim=384,
        use_mask_tokens=True, use_sdpa=True)
    
    # Apply LoRA (must match SSL checkpoint)
    logger.info(f"Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}, start={args.lora_start_layer})")
    apply_lora_to_encoder(encoder, rank=args.lora_rank, alpha=args.lora_alpha, 
                          start_layer=args.lora_start_layer)
    
    # Load encoder (SSL or previous detection checkpoint)
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'encoder' in ckpt:
        load_checkpoint_into_lora_model(encoder, ckpt['encoder'])
    elif 'encoder_lora' in ckpt:
        load_checkpoint_into_lora_model(encoder, ckpt['encoder_lora'])
    else:
        raise KeyError("Checkpoint must contain 'encoder' or 'encoder_lora'")
    encoder.to(device).eval()
    
    # Freeze encoder base weights, keep LoRA trainable
    lora_params = []
    for name, param in encoder.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    logger.info(f"LoRA trainable params: {sum(p.numel() for p in lora_params):,}")
    
    # Mid-backbone hook
    mid_hook = MidBackboneHook(layer_idx=args.mid_layer_idx)
    mid_hook.register(encoder.backbone.blocks)
    
    # --- Detection Head ---
    head = LightweightQueryDecoder(
        embed_dim=1024, num_classes=num_classes,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        nheads=args.decoder_nheads,
        dim_feedforward=args.dim_feedforward, dropout=0.1, grid_size=14,
        internal_dim=args.internal_dim,
        temporal_pooling=args.temporal_pooling
    ).to(device)
    logger.info(f"Decoder params: {sum(p.numel() for p in head.parameters()):,}")
    logger.info(f"Decoder internal_dim={head.internal_dim}, layers={head.num_decoder_layers}")
    
    # SupCon head — match decoder output dim
    sc_input_dim = head.internal_dim
    supcon_head = SupConProjectionHead(embed_dim=sc_input_dim, hidden_dim=min(512, sc_input_dim*2), proj_dim=128).to(device)
    supcon_loss_fn = SupConLoss(temperature=args.supcon_temp)
    
    # Token spatial classifier — direct encoder supervision
    token_spatial_head = TokenSpatialClassifier(
        embed_dim=1024, num_classes=num_classes, hidden_dim=256, grid_size=14
    ).to(device)
    logger.info(f"TokenSpatialClassifier params: {sum(p.numel() for p in token_spatial_head.parameters()):,}")
    
    # --- Optimizer ---
    param_groups = [
        {'params': list(head.parameters()), 'lr': args.decoder_lr, 'name': 'decoder'},
        {'params': list(supcon_head.parameters()), 'lr': args.decoder_lr, 'name': 'supcon'},
        {'params': list(token_spatial_head.parameters()), 'lr': args.decoder_lr, 'name': 'token_spatial'},
    ]
    if lora_params:
        param_groups.append({'params': lora_params, 'lr': args.lora_lr, 'name': 'lora'})
    
    optimizer = optim.AdamW(param_groups, lr=args.decoder_lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    # LR scheduler
    if args.use_plateau_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5)
        logger.info("Using ReduceLROnPlateau scheduler")
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        logger.info("Using CosineAnnealingLR scheduler")
    
    total_trainable = sum(p.numel() for pg in param_groups for p in pg['params'])
    logger.info(f"Total trainable: {total_trainable:,}")
    logger.info(f"LR: decoder={args.decoder_lr}, lora={args.lora_lr}")
    
    # --- Checkpoint resume ---
    start_epoch = 0
    resume_path = out_dir / 'last.pt'
    if args.resume and resume_path.exists():
        logger.info(f"Resuming from {resume_path}")
        ckpt_resume = torch.load(resume_path, map_location='cpu')
        if 'head' in ckpt_resume:
            head.load_state_dict(ckpt_resume['head'])
        if 'supcon_head' in ckpt_resume:
            supcon_head.load_state_dict(ckpt_resume['supcon_head'])
        if 'token_spatial_head' in ckpt_resume:
            token_spatial_head.load_state_dict(ckpt_resume['token_spatial_head'])
        if 'optimizer' in ckpt_resume:
            optimizer.load_state_dict(ckpt_resume['optimizer'])
        if 'encoder_lora' in ckpt_resume:
            lora_state = ckpt_resume['encoder_lora']
            encoder_state = encoder.state_dict()
            encoder_state.update(lora_state)
            encoder.load_state_dict(encoder_state)
        if 'epoch' in ckpt_resume:
            start_epoch = ckpt_resume['epoch'] + 1
        logger.info(f"Resumed: start_epoch={start_epoch}")
    elif args.resume:
        logger.info("--resume requested but no last.pt found. Starting fresh.")
    
    # --- Training ---
    best_recall = 0.0
    epochs_no_improve = 0
    class_weights = DEFAULT_CLASS_WEIGHTS[:num_classes]
    
    for epoch in range(start_epoch, args.epochs):
        head.train()
        supcon_head.train()
        token_spatial_head.train()
        
        # === CLASSIFICATION HEAD WARMUP ===
        # For first N epochs, freeze bbox heads and only train class head
        if epoch < args.class_head_warmup:
            logger.info(f"Class head warmup epoch {epoch+1}/{args.class_head_warmup}: freezing bbox heads")
            # Freeze box heads
            for param in head.box_head_coarse.parameters():
                param.requires_grad = False
            for param in head.box_head_refine.parameters():
                param.requires_grad = False
            # Keep class head trainable
            for param in head.class_head.parameters():
                param.requires_grad = True
        else:
            # Unfreeze all
            for param in head.box_head_coarse.parameters():
                param.requires_grad = True
            for param in head.box_head_refine.parameters():
                param.requires_grad = True
        
        meters = {'ce': 0., 'l1': 0., 'giou': 0., 'sc': 0., 'tok_cls': 0., 'tok_fg': 0., 'total': 0.}
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        # Linear warmup
        if epoch < args.warmup_epochs:
            warmup_lr_scale = (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                if pg['name'] == 'lora':
                    pg['lr'] = args.lora_lr * warmup_lr_scale
                else:
                    pg['lr'] = args.decoder_lr * warmup_lr_scale
            logger.info(f"Warmup {epoch+1}/{args.warmup_epochs}: decoder_lr={args.decoder_lr * warmup_lr_scale:.2e}, lora_lr={args.lora_lr * warmup_lr_scale:.2e}")
        
        for clips, boxes, labels in pbar:
            clips = clips.to(device, non_blocking=True)
            if clips.size(2) < 2:
                clips = torch.cat([clips, clips], dim=2)
            
            targets_boxes = [b.to(device) for b in boxes]
            targets_labels = [l.to(device) for l in labels]
            B = clips.shape[0]
            
            # Extract features (encoder frozen except LoRA)
            with torch.amp.autocast(device_type='cuda', enabled=True):
                full_tokens = encoder([clips])[0]
                mid_features = mid_hook.get_features()
                
                # Model handles temporal pooling internally now
                logits, pred_boxes, query_features, aux_outputs = head(full_tokens, mid_features)
                
                # Main loss
                if args.use_focal_loss:
                    ce_loss, l1_loss, giou_loss = detection_loss_focal(
                        logits, pred_boxes, targets_labels, targets_boxes,
                        num_classes=num_classes, class_weights=class_weights,
                        focal_gamma=args.focal_gamma)
                else:
                    ce_loss, l1_loss, giou_loss = detection_loss(
                        logits, pred_boxes, targets_labels, targets_boxes,
                        num_classes=num_classes, class_weights=class_weights)
                
                # Auxiliary losses
                aux_loss = torch.tensor(0., device=device)
                if args.use_aux_loss and aux_outputs:
                    for aux in aux_outputs:
                        if args.use_focal_loss:
                            ace, al1, agiou = detection_loss_focal(
                                aux['logits'], aux['boxes'], targets_labels, targets_boxes,
                                num_classes=num_classes, class_weights=class_weights,
                                focal_gamma=args.focal_gamma)
                        else:
                            ace, al1, agiou = detection_loss(
                                aux['logits'], aux['boxes'], targets_labels, targets_boxes,
                                num_classes=num_classes, class_weights=class_weights)
                        aux_loss += ace + 10.0 * al1 + 5.0 * agiou
                    aux_loss = aux_loss / len(aux_outputs)
                
                # === Token Spatial Loss (DIRECT encoder supervision) ===
                # This bypasses the decoder and gives direct gradients to LoRA
                token_cls_loss = torch.tensor(0., device=device)
                token_fg_loss = torch.tensor(0., device=device)
                # Get spatially-pooled encoder tokens (before decoder processing)
                spatial_tokens_for_loss = full_tokens
                N_spatial = 14 * 14
                if spatial_tokens_for_loss.dim() == 3 and spatial_tokens_for_loss.shape[1] > N_spatial:
                    D_st = spatial_tokens_for_loss.shape[-1]
                    T_st = spatial_tokens_for_loss.shape[1] // N_spatial
                    spatial_tokens_for_loss = spatial_tokens_for_loss.view(B, T_st, N_spatial, D_st)
                if spatial_tokens_for_loss.dim() == 4:
                    spatial_tokens_for_loss = spatial_tokens_for_loss[:, -1]  # last frame
                token_cls_loss, token_fg_loss = token_spatial_head(
                    spatial_tokens_for_loss, targets_boxes, targets_labels)
                
                # SupCon loss on matched query features
                sc_loss = torch.tensor(0., device=device)
                if args.supcon_weight > 0:
                    sc_embeds, sc_labels = [], []
                    for b_idx in range(B):
                        tl = targets_labels[b_idx]
                        tb = targets_boxes[b_idx]
                        if tl.numel() == 0:
                            continue
                        mq, mt = hungarian_match(
                            logits[b_idx].detach(), pred_boxes[b_idx].detach(),
                            tl, tb, class_weights=class_weights)
                        if len(mq) > 0:
                            for qi, ti in zip(mq, mt):
                                sc_embeds.append(query_features[b_idx, qi])
                                sc_labels.append(tl[ti])
                    
                    if len(sc_embeds) >= 2:
                        sc_embeds = torch.stack(sc_embeds)
                        sc_labels = torch.stack(sc_labels)
                        sc_proj = supcon_head(sc_embeds.float())
                        sc_loss = supcon_loss_fn(sc_proj, sc_labels)
                
                # Token spatial loss weight: high (2.0) because this is the direct encoder signal
                token_spatial_weight = 2.0
                loss = ce_loss + 10.0 * l1_loss + 5.0 * giou_loss + \
                       args.supcon_weight * sc_loss + 0.5 * aux_loss + \
                       token_spatial_weight * (token_cls_loss + token_fg_loss)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for pg in param_groups for p in pg['params']], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            meters['ce'] += ce_loss.item()
            meters['l1'] += l1_loss.item()
            meters['giou'] += giou_loss.item()
            meters['sc'] += sc_loss.item() if isinstance(sc_loss, torch.Tensor) else sc_loss
            meters['tok_cls'] += token_cls_loss.item()
            meters['tok_fg'] += token_fg_loss.item()
            meters['total'] += loss.item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f"{meters['total']/n_batches:.3f}",
                'ce': f"{meters['ce']/n_batches:.3f}",
                'tok': f"{(meters['tok_cls']+meters['tok_fg'])/n_batches:.3f}",
            })
        
        # Log training metrics
        avg = {k: v/max(n_batches, 1) for k, v in meters.items()}
        tb_writer.add_scalar("train/loss", avg['total'], epoch)
        tb_writer.add_scalar("train/ce", avg['ce'], epoch)
        tb_writer.add_scalar("train/l1", avg['l1'], epoch)
        tb_writer.add_scalar("train/giou", avg['giou'], epoch)
        tb_writer.add_scalar("train/supcon", avg['sc'], epoch)
        tb_writer.add_scalar("train/token_cls", avg['tok_cls'], epoch)
        tb_writer.add_scalar("train/token_fg", avg['tok_fg'], epoch)
        if args.use_wandb:
            wandb.log({f"train/{k}": v for k, v in avg.items()}, step=epoch)
        
        # Validation
        per_class_recall, overall_recall, per_class_gt = evaluate(
            encoder, head, mid_hook, val_loader, device, num_classes, epoch+1)
        
        logger.info(
            f"Epoch {epoch+1} | Loss: {avg['total']:.3f} (ce={avg['ce']:.3f} l1={avg['l1']:.3f} "
            f"giou={avg['giou']:.3f} sc={avg['sc']:.3f} tok={avg['tok_cls']+avg['tok_fg']:.3f}) | Recall@0.5: {overall_recall:.3f}")
        
        for c in range(num_classes):
            name = TOOL_NAMES[c] if c < len(TOOL_NAMES) else f"class_{c}"
            gt_count = int(per_class_gt[c].item())
            logger.info(f"  {name}: recall={per_class_recall[c]:.3f} (GT={gt_count})")
            tb_writer.add_scalar(f"val/recall_{name}", per_class_recall[c].item(), epoch)
        
        tb_writer.add_scalar("val/recall_overall", overall_recall.item(), epoch)
        if args.use_wandb:
            log_dict = {"val/recall_overall": overall_recall.item(), "epoch": epoch}
            for c in range(num_classes):
                name = TOOL_NAMES[c] if c < len(TOOL_NAMES) else f"class_{c}"
                log_dict[f"val/recall_{name}"] = per_class_recall[c].item()
            wandb.log(log_dict, step=epoch)
        
        # Visualizations (every 5 epochs + first epoch)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Recall bar chart
            recall_chart = make_recall_bar_chart(
                per_class_recall, per_class_gt, overall_recall, epoch+1, num_classes)
            if args.use_wandb:
                wandb.log({"viz/recall_chart": wandb.Image(recall_chart)}, step=epoch)
            tb_writer.add_image("viz/recall_chart", recall_chart, global_step=epoch, dataformats='HWC')
            
            # Full detection visualizations (bbox overlays, attention, confusion matrix, query assignment)
            run_detection_visualization(
                encoder=encoder, head=head, mid_hook=mid_hook,
                dataset=val_ds, device=device, epoch=epoch+1,
                wandb_run=wandb if args.use_wandb else None,
                tb_writer=tb_writer, num_classes=num_classes,
                num_samples=8, score_thresh=0.3)
        
        # Scheduler step AFTER validation
        if epoch >= args.warmup_epochs:
            if args.use_plateau_scheduler:
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(overall_recall.item())
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr < old_lr:
                    logger.info(f"  LR reduced: {old_lr:.2e} → {new_lr:.2e}")
            else:
                scheduler.step()
        
        # Save best + early stopping
        if overall_recall > best_recall:
            best_recall = overall_recall
            epochs_no_improve = 0
            save_dict = {
                'epoch': epoch,
                'head': head.state_dict(),
                'supcon_head': supcon_head.state_dict(),
                'token_spatial_head': token_spatial_head.state_dict(),
                'encoder_lora': {k: v for k, v in encoder.state_dict().items() if 'lora' in k.lower()},
                'val_recall': overall_recall.item(),
                'per_class_recall': {TOOL_NAMES[c]: per_class_recall[c].item() for c in range(num_classes)},
                'args': vars(args),
            }
            torch.save(save_dict, out_dir / 'best.pt')
            logger.info(f"  ★ New best recall: {overall_recall:.3f} → saved best.pt")
        else:
            epochs_no_improve += 1
            logger.info(f"  No improvement ({epochs_no_improve}/{args.early_stop_patience})")
        
        if epochs_no_improve >= args.early_stop_patience:
            logger.info(f"Early stopping after {epochs_no_improve} epochs without improvement")
            break
        
        # Save last.pt every epoch (crash-safe resume)
        state = {
            'epoch': epoch,
            'head': head.state_dict(),
            'supcon_head': supcon_head.state_dict(),
            'token_spatial_head': token_spatial_head.state_dict(),
            'encoder_lora': {k: v for k, v in encoder.state_dict().items() if 'lora' in k.lower()},
            'optimizer': optimizer.state_dict(),
            'val_recall': overall_recall.item(),
            'epochs_no_improve': epochs_no_improve,
        }
        torch.save(state, out_dir / 'last.pt')
        
        # Periodic numbered checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save(state, out_dir / f'e{epoch+1}.pt')
    
    mid_hook.remove()
    logger.info(f"\nTraining complete! Best Recall@0.5: {best_recall:.3f}")


if __name__ == '__main__':
    main()
