#!/usr/bin/env python3
"""
Pre-extract V-JEPA2 encoder features for fast Re-ID training.
Since encoder is frozen (only LoRA trains), extract once and cache to disk.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import pickle

# === WINDOWS PATH FIXES ===
current_dir = Path(__file__).parent
project_root = current_dir.parent
vjepa2_path = project_root / "vjepa2"
src_path = project_root / "vjepa2" / "src"

sys.path.insert(0, str(vjepa2_path))
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vjepa2.app.vjepa.utils import init_video_model
from vjepa2.app.vjepa_cholec80.lora import apply_lora_to_encoder, load_checkpoint_into_lora_model
from dataset_tracking import CholecTrackPairDataset, collate_fn

logging.basicConfig(level=logging.INFO, format='[%(levelname)-8s][%(name)-20s] %(message)s')
logger = logging.getLogger(__name__)


def extract_features(encoder, dataloader, device, cache_dir):
    """Extract and cache encoder features for all samples."""
    encoder.eval()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    all_features = []
    all_boxes = []
    all_labels = []
    all_tids = []
    
    logger.info(f"Extracting features for {len(dataloader.dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting")):
            clips_a, boxes_a, labels_a, tids_a, clips_b, boxes_b, labels_b, tids_b = batch
            
            clips_a = clips_a.to(device, non_blocking=True)
            clips_b = clips_b.to(device, non_blocking=True)
            
            # Extract features
            with torch.cuda.amp.autocast():
                feats_a = encoder([clips_a])[0]  # [B, T*N, D]
                feats_b = encoder([clips_b])[0]
            
            # Move to CPU and store
            for i in range(len(boxes_a)):
                all_features.append({
                    'feats_a': feats_a[i].cpu(),
                    'feats_b': feats_b[i].cpu(),
                })
                all_boxes.append({
                    'boxes_a': boxes_a[i],
                    'boxes_b': boxes_b[i],
                })
                all_labels.append({
                    'labels_a': labels_a[i],
                    'labels_b': labels_b[i],
                })
                all_tids.append({
                    'tids_a': tids_a[i],
                    'tids_b': tids_b[i],
                })
            
            # Periodic saving to avoid memory overflow
            if (batch_idx + 1) % 500 == 0:
                logger.info(f"Processed {batch_idx + 1} batches, saving checkpoint...")
    
    # Save final cache
    cache_data = {
        'features': all_features,
        'boxes': all_boxes,
        'labels': all_labels,
        'tids': all_tids,
    }
    
    cache_file = cache_dir / 'features_cache.pt'
    logger.info(f"Saving cache to {cache_file} ({len(all_features)} samples)")
    torch.save(cache_data, cache_file)
    
    logger.info("Feature extraction complete!")
    return cache_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_checkpoint', type=str, required=True)
    parser.add_argument('--train_dir', type=str, default='../cholec_dataset/Training')
    parser.add_argument('--val_dir', type=str, default='../cholec_dataset/Validation')
    parser.add_argument('--cache_dir', type=str, default='../cache/reid_features')
    parser.add_argument('--clip_len', type=int, default=8)  # Reduced from 16
    parser.add_argument('--batch_size', type=int, default=32)  # Can use larger batch for extraction
    parser.add_argument('--num_workers', type=int, default=0)
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- Datasets ---
    logger.info("Loading datasets...")
    train_ds = CholecTrackPairDataset(args.train_dir, clip_len=args.clip_len, max_frame_gap=3)
    val_ds = CholecTrackPairDataset(args.val_dir, clip_len=args.clip_len, max_frame_gap=3)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,  # No shuffle for extraction
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn, persistent_workers=False)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn, persistent_workers=False)
    
    # --- Encoder ---
    logger.info("Initializing V-JEPA2 encoder...")
    encoder, _ = init_video_model(
        device=device, model_name='vit_large', patch_size=16,
        max_num_frames=args.clip_len, tubelet_size=2, crop_size=224,
        pred_depth=12, pred_embed_dim=384,
        use_mask_tokens=True, use_sdpa=True)
    
    apply_lora_to_encoder(encoder, rank=16, alpha=16.0, start_layer=12)
    
    logger.info(f"Loading detection checkpoint from {args.detection_checkpoint}")
    ckpt = torch.load(args.detection_checkpoint, map_location='cpu')
    load_checkpoint_into_lora_model(encoder, ckpt['encoder_lora'])
    encoder.to(device).eval()
    
    # Freeze all
    for param in encoder.parameters():
        param.requires_grad = False
    
    # --- Extract features ---
    cache_dir = Path(args.cache_dir)
    
    logger.info("=== Extracting TRAIN features ===")
    train_cache = extract_features(encoder, train_loader, device, cache_dir / 'train')
    
    logger.info("=== Extracting VAL features ===")
    val_cache = extract_features(encoder, val_loader, device, cache_dir / 'val')
    
    logger.info(f"\nAll features cached!")
    logger.info(f"Train: {train_cache}")
    logger.info(f"Val: {val_cache}")
    logger.info(f"\nNow run: python train_reid_fast.py --cache_dir {args.cache_dir}")


if __name__ == '__main__':
    main()
