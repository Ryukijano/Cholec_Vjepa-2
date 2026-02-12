#!/usr/bin/env python3
"""
Phase 2: Train Re-ID embedding head for surgical tool tracking.

Uses V-JEPA2 encoder (with LoRA from Phase 1) to extract features,
then trains a Re-ID head with contrastive loss on CholecTrack20 track IDs.

The Re-ID head learns to produce embeddings where:
- Same tool instance across frames → close embeddings
- Different tool instances → far apart embeddings

Usage:
  python train_reid.py \
    --detection_checkpoint outputs/detection-hardened-v2/best.pt \
    --out_dir outputs/reid-phase2
"""

import argparse
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

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vjepa2')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vjepa2.app.vjepa.utils import init_video_model
from vjepa2.app.vjepa_cholec80.lora import apply_lora_to_encoder, load_checkpoint_into_lora_model

from cholectrack_vjepa2_training.reid_head import ReIDHead, TripletLoss, TrackContrastiveLoss
from cholectrack_vjepa2_training.dataset_tracking import CholecTrackPairDataset, collate_track_pairs

try:
    import wandb
except ImportError:
    wandb = None

from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO, format='[%(levelname)-8s][%(asctime)s][%(name)-20s] %(message)s')
logger = logging.getLogger(__name__)

TOOL_NAMES = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]


def evaluate_reid(encoder, reid_head, val_loader, device, epoch):
    """Evaluate Re-ID quality: consistency accuracy across frame pairs."""
    reid_head.eval()
    
    total_pairs = 0
    correct_pairs = 0
    total_neg_pairs = 0
    correct_neg_pairs = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
            clips_a, boxes_a, labels_a, tids_a, clips_b, boxes_b, labels_b, tids_b = batch
            
            clips_a = clips_a.to(device, non_blocking=True)
            clips_b = clips_b.to(device, non_blocking=True)
            
            if clips_a.size(2) < 2:
                clips_a = torch.cat([clips_a, clips_a], dim=2)
            if clips_b.size(2) < 2:
                clips_b = torch.cat([clips_b, clips_b], dim=2)
            
            with torch.amp.autocast(device_type='cuda', enabled=True):
                tokens_a = encoder([clips_a])[0]
                tokens_b = encoder([clips_b])[0]
                
                embs_a = reid_head(tokens_a, [b.to(device) for b in boxes_a])
                embs_b = reid_head(tokens_b, [b.to(device) for b in boxes_b])
            
            B = clips_a.shape[0]
            for b in range(B):
                ea = embs_a[b].cpu()
                eb = embs_b[b].cpu()
                ta = tids_a[b]
                tb = tids_b[b]
                
                if ea.shape[0] == 0 or eb.shape[0] == 0:
                    continue
                
                # For each detection in frame A, find closest in frame B
                sim = torch.mm(ea, eb.T)  # [Ma, Mb]
                
                for i in range(ea.shape[0]):
                    best_j = sim[i].argmax().item()
                    
                    # Check if matched track ID is correct
                    if ta[i].item() in tb.tolist():
                        # There's a matching track in frame B
                        gt_j = (tb == ta[i]).nonzero(as_tuple=True)[0]
                        if len(gt_j) > 0:
                            total_pairs += 1
                            if best_j == gt_j[0].item():
                                correct_pairs += 1
                    
                    # Negative: closest match should NOT be a different track
                    if ta[i].item() != tb[best_j].item():
                        total_neg_pairs += 1
                    else:
                        correct_neg_pairs += 1
    
    pos_acc = correct_pairs / max(total_pairs, 1)
    neg_acc = correct_neg_pairs / max(correct_neg_pairs + total_neg_pairs, 1)
    
    return pos_acc, neg_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_checkpoint', type=str, required=True,
                        help='Path to Phase 1 detection checkpoint (best.pt)')
    parser.add_argument('--ssl_checkpoint', type=str,
                        default='/teamspace/studios/this_studio/outputs/vjepa2-cholec-pretrain/latest.pt',
                        help='Path to SSL pretrained checkpoint')
    parser.add_argument('--train_dir', type=str,
                        default='/teamspace/studios/this_studio/cholec_dataset/Training')
    parser.add_argument('--val_dir', type=str,
                        default='/teamspace/studios/this_studio/cholec_dataset/Validation')
    parser.add_argument('--out_dir', type=str,
                        default='/teamspace/studios/this_studio/outputs/reid-phase2')
    
    # Re-ID head
    parser.add_argument('--reid_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    
    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lora_lr', type=float, default=1e-6,
                        help='Very low LR for LoRA (already adapted in Phase 1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--max_frame_gap', type=int, default=5,
                        help='Max frame gap for positive pairs')
    parser.add_argument('--loss_type', type=str, default='both',
                        choices=['triplet', 'contrastive', 'both'])
    parser.add_argument('--triplet_margin', type=float, default=0.3)
    parser.add_argument('--contrastive_temp', type=float, default=0.07)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--track_perspective', type=str, default='intraoperative',
                        choices=['intraoperative', 'intracorporeal', 'visibility'])
    
    # LoRA (must match Phase 1)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--lora_start_layer', type=int, default=12)
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # --- Logging ---
    if args.use_wandb and wandb is not None:
        wandb.init(project="vjepa2-cholec20-reid", config=vars(args))
    tb_writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    
    # --- Dataset ---
    logger.info("Loading track pair datasets...")
    train_ds = CholecTrackPairDataset(
        args.train_dir, clip_len=args.clip_len, max_frame_gap=args.max_frame_gap,
        track_perspective=args.track_perspective)
    val_ds = CholecTrackPairDataset(
        args.val_dir, clip_len=args.clip_len, max_frame_gap=args.max_frame_gap,
        track_perspective=args.track_perspective)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_track_pairs, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_track_pairs, persistent_workers=True)
    
    # --- Encoder (from Phase 1) ---
    logger.info("Initializing V-JEPA2 encoder...")
    encoder, predictor = init_video_model(
        device=device, model_name='vit_large', patch_size=16,
        max_num_frames=args.clip_len, tubelet_size=2, crop_size=224,
        pred_depth=12, pred_embed_dim=384,
        use_mask_tokens=True, use_sdpa=True)
    
    # Apply LoRA
    apply_lora_to_encoder(encoder, rank=args.lora_rank, alpha=args.lora_alpha,
                          start_layer=args.lora_start_layer)
    
    # Load SSL base weights
    ssl_ckpt = torch.load(args.ssl_checkpoint, map_location='cpu')
    load_checkpoint_into_lora_model(encoder, ssl_ckpt['encoder'])
    
    # Load Phase 1 LoRA weights (adapted for detection)
    det_ckpt = torch.load(args.detection_checkpoint, map_location='cpu')
    if 'encoder_lora' in det_ckpt:
        lora_state = det_ckpt['encoder_lora']
        encoder_state = encoder.state_dict()
        encoder_state.update(lora_state)
        encoder.load_state_dict(encoder_state)
        logger.info(f"Loaded Phase 1 LoRA weights ({len(lora_state)} params)")
    
    encoder.to(device).eval()
    
    # Keep LoRA trainable but with very low LR
    lora_params = []
    for name, param in encoder.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    logger.info(f"LoRA trainable params: {sum(p.numel() for p in lora_params):,}")
    
    # --- Re-ID Head ---
    reid_head = ReIDHead(
        embed_dim=1024, hidden_dim=args.hidden_dim,
        reid_dim=args.reid_dim, grid_size=14
    ).to(device)
    logger.info(f"ReIDHead params: {sum(p.numel() for p in reid_head.parameters()):,}")
    
    # --- Loss ---
    triplet_loss_fn = TripletLoss(margin=args.triplet_margin)
    contrastive_loss_fn = TrackContrastiveLoss(temperature=args.contrastive_temp)
    
    # --- Optimizer ---
    param_groups = [
        {'params': list(reid_head.parameters()), 'lr': args.lr, 'name': 'reid'},
    ]
    if lora_params:
        param_groups.append({'params': lora_params, 'lr': args.lora_lr, 'name': 'lora'})
    
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    total_trainable = sum(p.numel() for pg in param_groups for p in pg['params'])
    logger.info(f"Total trainable: {total_trainable:,}")
    
    # --- Training ---
    best_pos_acc = 0.0
    
    for epoch in range(args.epochs):
        reid_head.train()
        
        meters = {'triplet': 0., 'contrastive': 0., 'total': 0.}
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            clips_a, boxes_a, labels_a, tids_a, clips_b, boxes_b, labels_b, tids_b = batch
            
            clips_a = clips_a.to(device, non_blocking=True)
            clips_b = clips_b.to(device, non_blocking=True)
            
            if clips_a.size(2) < 2:
                clips_a = torch.cat([clips_a, clips_a], dim=2)
            if clips_b.size(2) < 2:
                clips_b = torch.cat([clips_b, clips_b], dim=2)
            
            B = clips_a.shape[0]
            boxes_a_dev = [b.to(device) for b in boxes_a]
            boxes_b_dev = [b.to(device) for b in boxes_b]
            
            with torch.amp.autocast(device_type='cuda', enabled=True):
                tokens_a = encoder([clips_a])[0]
                tokens_b = encoder([clips_b])[0]
                
                embs_a = reid_head(tokens_a, boxes_a_dev)
                embs_b = reid_head(tokens_b, boxes_b_dev)
                
                # Collect all embeddings + track IDs across batch
                all_embs = []
                all_tids = []
                
                for b in range(B):
                    if embs_a[b].shape[0] > 0:
                        all_embs.append(embs_a[b])
                        all_tids.append(tids_a[b].to(device))
                    if embs_b[b].shape[0] > 0:
                        all_embs.append(embs_b[b])
                        all_tids.append(tids_b[b].to(device))
                
                if len(all_embs) == 0:
                    continue
                
                all_embs = torch.cat(all_embs, dim=0)  # [N_total, reid_dim]
                all_tids = torch.cat(all_tids, dim=0)   # [N_total]
                
                # Compute losses
                trip_loss = torch.tensor(0., device=device)
                cont_loss = torch.tensor(0., device=device)
                
                if args.loss_type in ('triplet', 'both'):
                    trip_loss = triplet_loss_fn(all_embs, all_tids)
                
                if args.loss_type in ('contrastive', 'both'):
                    cont_loss = contrastive_loss_fn(all_embs, all_tids)
                
                loss = trip_loss + cont_loss
            
            if loss.item() == 0:
                continue
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for pg in param_groups for p in pg['params']], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            meters['triplet'] += trip_loss.item()
            meters['contrastive'] += cont_loss.item()
            meters['total'] += loss.item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f"{meters['total']/n_batches:.3f}",
                'trip': f"{meters['triplet']/n_batches:.3f}",
                'cont': f"{meters['contrastive']/n_batches:.3f}",
            })
        
        scheduler.step()
        
        # Log
        avg = {k: v / max(n_batches, 1) for k, v in meters.items()}
        for k, v in avg.items():
            tb_writer.add_scalar(f"train/{k}", v, epoch)
        if args.use_wandb and wandb is not None:
            wandb.log({f"train/{k}": v for k, v in avg.items()}, step=epoch)
        
        # Evaluate
        pos_acc, neg_acc = evaluate_reid(encoder, reid_head, val_loader, device, epoch+1)
        
        logger.info(
            f"Epoch {epoch+1} | Loss: {avg['total']:.3f} "
            f"(trip={avg['triplet']:.3f} cont={avg['contrastive']:.3f}) | "
            f"Pos Acc: {pos_acc:.3f} | Match Acc: {neg_acc:.3f}")
        
        tb_writer.add_scalar("val/pos_accuracy", pos_acc, epoch)
        tb_writer.add_scalar("val/match_accuracy", neg_acc, epoch)
        if args.use_wandb and wandb is not None:
            wandb.log({"val/pos_accuracy": pos_acc, "val/match_accuracy": neg_acc}, step=epoch)
        
        # Save best
        if pos_acc > best_pos_acc:
            best_pos_acc = pos_acc
            save_dict = {
                'epoch': epoch,
                'reid_head': reid_head.state_dict(),
                'encoder_lora': {k: v for k, v in encoder.state_dict().items() if 'lora' in k.lower()},
                'pos_accuracy': pos_acc,
                'args': vars(args),
            }
            torch.save(save_dict, out_dir / 'best_reid.pt')
            logger.info(f"  ★ New best pos accuracy: {pos_acc:.3f} → saved best_reid.pt")
        
        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            state = {
                'epoch': epoch,
                'reid_head': reid_head.state_dict(),
                'encoder_lora': {k: v for k, v in encoder.state_dict().items() if 'lora' in k.lower()},
                'optimizer': optimizer.state_dict(),
                'pos_accuracy': pos_acc,
            }
            torch.save(state, out_dir / f'reid_e{epoch+1}.pt')
        
        # Save last
        torch.save({
            'epoch': epoch,
            'reid_head': reid_head.state_dict(),
            'encoder_lora': {k: v for k, v in encoder.state_dict().items() if 'lora' in k.lower()},
            'optimizer': optimizer.state_dict(),
            'pos_accuracy': pos_acc,
        }, out_dir / 'last.pt')
    
    logger.info(f"\nPhase 2 Complete! Best Pos Accuracy: {best_pos_acc:.3f}")


if __name__ == '__main__':
    main()
