#!/usr/bin/env python3
"""
Fast Re-ID training on pre-extracted features.
Trains only the Re-ID head (no encoder forward pass).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# === WINDOWS PATH FIXES ===
current_dir = Path(__file__).parent
project_root = current_dir.parent

sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from reid_head import ReIDHead, TripletLoss, TrackContrastiveLoss

logging.basicConfig(level=logging.INFO, format='[%(levelname)-8s][%(name)-20s] %(message)s')
logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None


class CachedFeatureDataset(Dataset):
    """Dataset that loads pre-extracted features from disk."""
    
    def __init__(self, cache_file):
        logger.info(f"Loading cached features from {cache_file}...")
        cache = torch.load(cache_file, map_location='cpu')
        
        self.features = cache['features']
        self.boxes = cache['boxes']
        self.labels = cache['labels']
        self.tids = cache['tids']
        
        logger.info(f"Loaded {len(self)} samples from cache")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            self.features[idx]['feats_a'],
            self.boxes[idx]['boxes_a'],
            self.labels[idx]['labels_a'],
            self.tids[idx]['tids_a'],
            self.features[idx]['feats_b'],
            self.boxes[idx]['boxes_b'],
            self.labels[idx]['labels_b'],
            self.tids[idx]['tids_b'],
        )


def collate_cached(batch):
    """Collate function for cached features."""
    feats_a, boxes_a, labels_a, tids_a = [], [], [], []
    feats_b, boxes_b, labels_b, tids_b = [], [], [], []
    
    for fa, ba, la, ta, fb, bb, lb, tb in batch:
        feats_a.append(fa)
        boxes_a.append(ba)
        labels_a.append(la)
        tids_a.append(ta)
        feats_b.append(fb)
        boxes_b.append(bb)
        labels_b.append(lb)
        tids_b.append(tb)
    
    # Stack features (they're already tensors)
    feats_a = torch.stack(feats_a, dim=0)
    feats_b = torch.stack(feats_b, dim=0)
    
    return (feats_a, boxes_a, labels_a, tids_a,
            feats_b, boxes_b, labels_b, tids_b)


def evaluate_cached(reid_head, val_loader, device):
    """Evaluate on cached features."""
    reid_head.eval()
    correct_pos = 0
    correct_neg = 0
    total_pos = 0
    total_neg = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Re-ID Eval", leave=False):
            feats_a, boxes_a, labels_a, tids_a, feats_b, boxes_b, labels_b, tids_b = batch
            
            feats_a = feats_a.to(device, non_blocking=True)
            feats_b = feats_b.to(device, non_blocking=True)
            boxes_a = [b.to(device) for b in boxes_a]
            boxes_b = [b.to(device) for b in boxes_b]
            
            with autocast():
                emb_a = reid_head(feats_a, boxes_a)
                emb_b = reid_head(feats_b, boxes_b)
            
            for i in range(len(tids_a)):
                for j in range(len(tids_b)):
                    if tids_a[i] == tids_b[j]:
                        total_pos += 1
                        if torch.cosine_similarity(emb_a[i], emb_b[j], dim=0) > 0.5:
                            correct_pos += 1
                    else:
                        total_neg += 1
                        if torch.cosine_similarity(emb_a[i], emb_b[j], dim=0) < 0.5:
                            correct_neg += 1
    
    pos_acc = correct_pos / max(total_pos, 1)
    neg_acc = correct_neg / max(total_neg, 1)
    return pos_acc, neg_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, default='../cache/reid_features')
    parser.add_argument('--out_dir', type=str, default='../outputs/reid-phase2-fast')
    
    # Model
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--clip_len', type=int, default=8)
    
    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)  # Can use much larger batch!
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--loss_type', type=str, default='both', choices=['triplet', 'contrastive', 'both'])
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--val_freq', type=int, default=5)
    parser.add_argument('--use_wandb', action='store_true')
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.use_wandb and wandb is not None:
        wandb.init(project="vjepa2-reid-fast", config=vars(args))
    
    # --- Load cached datasets ---
    cache_dir = Path(args.cache_dir)
    train_cache = cache_dir / 'train' / 'features_cache.pt'
    val_cache = cache_dir / 'val' / 'features_cache.pt'
    
    if not train_cache.exists():
        logger.error(f"Train cache not found: {train_cache}")
        logger.error("Run: python extract_features.py --detection_checkpoint <path> --cache_dir <dir>")
        return
    
    logger.info("Loading cached datasets...")
    train_ds = CachedFeatureDataset(train_cache)
    val_ds = CachedFeatureDataset(val_cache)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
        collate_fn=collate_cached, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
        collate_fn=collate_cached)
    
    # --- Re-ID Head only ---
    logger.info("Initializing Re-ID head...")
    reid_head = ReIDHead(
        embed_dim=1024,  # ViT-L outputs 1024-D
        grid_size=14, hidden_dim=512
    ).to(device)
    
    logger.info(f"ReIDHead params: {sum(p.numel() for p in reid_head.parameters()):,}")
    
    # Compile for extra speed
    if hasattr(torch, 'compile'):
        logger.info("Compiling ReID head...")
        reid_head = torch.compile(reid_head, mode='reduce-overhead')
    
    # --- Losses & Optimizer ---
    triplet_loss_fn = TripletLoss(margin=args.margin)
    contrastive_loss_fn = TrackContrastiveLoss(temperature=args.temperature)
    
    optimizer = optim.AdamW(
        reid_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    logger.info(f"Total trainable: {sum(p.numel() for p in reid_head.parameters()):,}")
    logger.info(f"Batches per epoch: {len(train_loader)}")
    
    # --- Training ---
    best_pos_acc = 0.0
    
    for epoch in range(args.epochs):
        reid_head.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for feats_a, boxes_a, labels_a, tids_a, feats_b, boxes_b, labels_b, tids_b in pbar:
            feats_a = feats_a.to(device, non_blocking=True)
            feats_b = feats_b.to(device, non_blocking=True)
            boxes_a = [b.to(device) for b in boxes_a]
            boxes_b = [b.to(device) for b in boxes_b]
            
            optimizer.zero_grad()
            
            with autocast():
                emb_a = reid_head(feats_a, boxes_a)
                emb_b = reid_head(feats_b, boxes_b)
                
                all_emb = torch.cat(emb_a + emb_b, dim=0)
                all_tids = torch.cat(tids_a + tids_b, dim=0)
                
                loss = 0.0
                if args.loss_type in ['triplet', 'both']:
                    loss += triplet_loss_fn(all_emb, all_tids)
                if args.loss_type in ['contrastive', 'both']:
                    loss += contrastive_loss_fn(all_emb, all_tids)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(reid_head.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{total_loss/max(n_batches,1):.3f}'})
        
        # Validation
        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            pos_acc, neg_acc = evaluate_cached(reid_head, val_loader, device)
        else:
            pos_acc, neg_acc = 0.0, 0.0
        
        logger.info(f"Epoch {epoch+1} | Loss: {total_loss/max(n_batches,1):.3f} | "
                   f"Pos Acc: {pos_acc:.3f} | Neg Acc: {neg_acc:.3f}")
        
        if args.use_wandb and wandb is not None:
            wandb.log({
                "train/loss": total_loss/max(n_batches,1),
                "val/pos_acc": pos_acc,
                "val/neg_acc": neg_acc,
                "epoch": epoch
            }, step=epoch)
        
        # Save best
        if pos_acc > best_pos_acc:
            best_pos_acc = pos_acc
            torch.save({
                'epoch': epoch,
                'reid_head': reid_head.state_dict(),
                'best_pos_acc': pos_acc,
                'args': vars(args),
            }, out_dir / 'best.pt')
            logger.info(f"  ★ New best pos acc: {pos_acc:.3f}")
        
        scheduler.step()
    
    logger.info(f"\nTraining complete! Best positive accuracy: {best_pos_acc:.3f}")


if __name__ == '__main__':
    main()
