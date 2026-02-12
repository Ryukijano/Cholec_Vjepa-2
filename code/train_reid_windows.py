#!/usr/bin/env python3
"""
Windows-compatible Re-ID training script with anti-collapse fixes.

Implementation checklist (see also reid_head.py for 1–4):
  5. Dynamic loss weighting: epoch 0 → 0.7*triplet + 0.3*contrastive;
     by end → 0.5*triplet + 0.5*contrastive (balanced).
  6. Progress bar shows: loss, trip, cont, temperature, bank size.

Also: gradient accumulation for effective large batches on 4090.
"""

import argparse
import copy
import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path

# === WINDOWS PATH FIXES ===
current_dir = Path(__file__).parent
project_root = current_dir.parent
vjepa2_path = project_root / "vjepa2"
src_path = project_root / "vjepa2" / "src"

sys.path.insert(0, str(vjepa2_path))
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

if mp.current_process().name == "MainProcess":
    print(f"Added paths:")
    print(f"  vjepa2: {vjepa2_path}")
    print(f"  src: {src_path}")
    print(f"  project_root: {project_root}")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from vjepa2.app.vjepa.utils import init_video_model
from vjepa2.app.vjepa_cholec80.lora import apply_lora_to_encoder, load_checkpoint_into_lora_model

from reid_head import ReIDHead, TripletLoss, TrackContrastiveLoss, EmbeddingMemoryBank
from tracker import SurgicalTracker
from dataset_tracking import CholecTrackPairDataset, collate_fn

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import wandb
except ImportError:
    wandb = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)-8s][%(name)-20s] %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dynamic loss weighting
# ---------------------------------------------------------------------------
def get_loss_weights(epoch: int, total_epochs: int):
    """Return (triplet_weight, contrastive_weight) that evolve over training.

    Early epochs: triplet dominates (0.7) to break collapse fast.
    Later epochs: contrastive catches up (0.5/0.5) for fine-grained separation.
    """
    progress = epoch / max(total_epochs - 1, 1)
    triplet_w = 0.7 - 0.2 * progress   # 0.7 -> 0.5
    contrastive_w = 0.3 + 0.2 * progress  # 0.3 -> 0.5
    return triplet_w, contrastive_w


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_reid(reid_head, encoder, val_loader, device):
    """Evaluate Re-ID accuracy (positive/negative pair accuracy)."""
    reid_head.eval()
    correct_pos = 0
    correct_neg = 0
    total_pos = 0
    total_neg = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Re-ID Eval", leave=False):
            clips_a, boxes_a, labels_a, tids_a, clips_b, boxes_b, labels_b, tids_b = batch

            clips_a = torch.stack(clips_a).to(device, non_blocking=True)
            clips_b = torch.stack(clips_b).to(device, non_blocking=True)
            boxes_a = [b.to(device) for b in boxes_a]
            boxes_b = [b.to(device) for b in boxes_b]

            with autocast(device_type='cuda', enabled=(device == 'cuda')):
                feats_a = encoder.backbone(clips_a)
                feats_b = encoder.backbone(clips_b)

                emb_a = reid_head(feats_a, boxes_a)
                emb_b = reid_head(feats_b, boxes_b)

            for i in range(len(emb_a)):
                for j in range(len(emb_b)):
                    for ii in range(len(tids_a[i])):
                        for jj in range(len(tids_b[j])):
                            tid_a = tids_a[i][ii].item()
                            tid_b = tids_b[j][jj].item()
                            if tid_a == tid_b:
                                total_pos += 1
                                if torch.cosine_similarity(emb_a[i][ii], emb_b[j][jj], dim=0) > 0.5:
                                    correct_pos += 1
                            else:
                                total_neg += 1
                                if torch.cosine_similarity(emb_a[i][ii], emb_b[j][jj], dim=0) < 0.5:
                                    correct_neg += 1

    pos_acc = correct_pos / max(total_pos, 1)
    neg_acc = correct_neg / max(total_neg, 1)
    return pos_acc, neg_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_checkpoint', type=str, required=True,
                        help='Path to detection checkpoint (best.pt)')
    parser.add_argument('--train_dir', type=str,
                        default='../cholec_dataset/Training')
    parser.add_argument('--val_dir', type=str,
                        default='../cholec_dataset/Validation')
    parser.add_argument('--out_dir', type=str,
                        default='../outputs/reid-phase2')

    # Model
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--clip_len', type=int, default=8)
    parser.add_argument('--roi_size', type=int, default=7)

    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=28)
    parser.add_argument('--lr', type=float, default=3.2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--persistent_workers', type=int, default=1)
    parser.add_argument('--grad_accum_steps', type=int, default=2)

    # Losses
    parser.add_argument('--loss_type', type=str, default='both',
                        choices=['triplet', 'contrastive', 'both'])
    parser.add_argument('--margin', type=float, default=0.7,
                        help='Triplet margin (0.7 helps avoid Re-ID embedding collapse)')
    parser.add_argument('--temp_start', type=float, default=0.15,
                        help='Contrastive temperature at epoch 0 (warm)')
    parser.add_argument('--temp_final', type=float, default=0.07,
                        help='Contrastive temperature after warmup')
    parser.add_argument('--temp_warmup_epochs', type=int, default=5)

    # Memory bank
    parser.add_argument('--memory_bank_size', type=int, default=8192,
                        help='Number of past embeddings to keep for extra negatives')

    # Periodic numbered checkpoint
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')

    # Performance
    parser.add_argument('--compile_backbone', type=int, default=0)
    parser.add_argument('--compile_reid_head', type=int, default=0)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from (loads epoch, models, best_acc)')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # --- Logging ---
    if args.use_wandb and wandb is not None:
        wandb.init(project="vjepa2-reid", config=vars(args))
    tb_writer = None
    if SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    # --- Datasets ---
    logger.info("Loading track pair datasets...")
    train_ds = CholecTrackPairDataset(args.train_dir, clip_len=args.clip_len, max_frame_gap=3)
    val_ds = CholecTrackPairDataset(args.val_dir, clip_len=args.clip_len, max_frame_gap=3)
    logger.info(f"Train pairs: {len(train_ds)}, Val pairs: {len(val_ds)}")

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, **loader_kwargs)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        **loader_kwargs)

    # --- Encoder ---
    logger.info("Initializing V-JEPA2 encoder...")
    encoder, predictor = init_video_model(
        device=device, model_name='vit_large', patch_size=16,
        max_num_frames=args.clip_len, tubelet_size=2, crop_size=224,
        pred_depth=12, pred_embed_dim=384,
        use_mask_tokens=True, use_sdpa=True)

    logger.info("Applying LoRA (rank=16, alpha=16, start=12)")
    apply_lora_to_encoder(encoder, rank=16, alpha=16.0, start_layer=12)

    logger.info(f"Loading detection checkpoint from {args.detection_checkpoint}")
    ckpt = torch.load(args.detection_checkpoint, map_location='cpu')
    load_checkpoint_into_lora_model(encoder, ckpt['encoder_lora'])
    encoder.to(device).eval()
    if hasattr(torch, "compile") and args.compile_backbone:
        logger.info("Compiling encoder backbone...")
        encoder.backbone = torch.compile(encoder.backbone, mode="reduce-overhead")

    # Freeze encoder except LoRA
    lora_params = []
    for name, param in encoder.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    # --- Re-ID Head ---
    reid_head = ReIDHead(
        embed_dim=1024, grid_size=14, hidden_dim=512, reid_dim=args.embed_dim,
    ).to(device)
    if hasattr(torch, "compile") and args.compile_reid_head:
        logger.info("Compiling ReID head...")
        reid_head = torch.compile(reid_head, mode="reduce-overhead")
    logger.info(f"ReIDHead params: {sum(p.numel() for p in reid_head.parameters()):,}")

    # --- Memory Bank ---
    memory_bank = EmbeddingMemoryBank(
        capacity=args.memory_bank_size, embed_dim=args.embed_dim, device=device)
    logger.info(f"Memory bank capacity: {args.memory_bank_size}")

    # --- Losses ---
    triplet_loss_fn = TripletLoss(margin=args.margin)
    contrastive_loss_fn = TrackContrastiveLoss(
        temp_start=args.temp_start,
        temp_final=args.temp_final,
        warmup_epochs=args.temp_warmup_epochs,
    )

    # --- Optimizer ---
    optimizer = optim.AdamW([
        {'params': list(reid_head.parameters()), 'lr': args.lr},
        {'params': lora_params, 'lr': args.lr * 0.1},
    ], weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=(device == 'cuda'))
    accum_steps = max(1, args.grad_accum_steps)

    logger.info(f"Total trainable: {sum(p.numel() for p in reid_head.parameters()) + sum(p.numel() for p in lora_params):,}")
    logger.info(f"LR: reid={args.lr}, lora={args.lr * 0.1}")
    logger.info(f"Micro-batch: {args.batch_size}, Grad accum: {accum_steps}, "
                f"Effective batch: {args.batch_size * accum_steps}")
    logger.info(f"Temp schedule: {args.temp_start} -> {args.temp_final} over {args.temp_warmup_epochs} epochs")
    logger.info(f"Loss weights: dynamic (triplet 0.7->0.5, contrastive 0.3->0.5)")

    # --- Resume from checkpoint ---
    start_epoch = 0
    best_pos_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')

        # Strip _orig_mod. prefix added by torch.compile
        def _strip_compiled(sd):
            return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

        reid_sd = _strip_compiled(ckpt['reid_head'])
        # Filter out keys with shape mismatches so strict=False doesn't crash
        current_sd = reid_head.state_dict()
        compatible_sd = {}
        skipped = []
        for k, v in reid_sd.items():
            if k in current_sd and current_sd[k].shape == v.shape:
                compatible_sd[k] = v
            else:
                skipped.append(k)
        if skipped:
            logger.warning(f"Resume: skipped {len(skipped)} keys with shape mismatch (architecture changed)")
            logger.warning(f"  Skipped: {skipped[:8]}")
        missing, unexpected = reid_head.load_state_dict(compatible_sd, strict=False)
        if missing:
            logger.warning(f"Resume: {len(missing)} missing keys — training from scratch for those layers")
        loaded_count = len(compatible_sd)
        logger.info(f"Resume: loaded {loaded_count}/{len(reid_sd)} ReIDHead params")

        if 'encoder_lora' in ckpt:
            lora_sd = _strip_compiled(ckpt['encoder_lora'])
            load_checkpoint_into_lora_model(encoder, lora_sd)

        start_epoch = ckpt.get('epoch', -1) + 1
        best_pos_acc = ckpt.get('best_pos_acc', 0.0)
        logger.info(f"Resumed from epoch {start_epoch - 1}, best pos acc {best_pos_acc:.3f}")
        # Step scheduler to resume at correct LR
        for _ in range(start_epoch):
            scheduler.step()

    # --- Training ---
    for epoch in range(start_epoch, args.epochs):
        reid_head.train()
        contrastive_loss_fn.set_epoch(epoch)
        trip_w, cont_w = get_loss_weights(epoch, args.epochs)

        total_loss = 0.0
        total_trip = 0.0
        total_cont = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (clips_a, boxes_a, labels_a, tids_a,
                    clips_b, boxes_b, labels_b, tids_b) in enumerate(pbar, start=1):

            clips_a = torch.stack(clips_a).to(device, non_blocking=True)
            clips_b = torch.stack(clips_b).to(device, non_blocking=True)
            boxes_a = [b.to(device) for b in boxes_a]
            boxes_b = [b.to(device) for b in boxes_b]

            with autocast(device_type='cuda', enabled=(device == 'cuda')):
                feats_a = encoder.backbone(clips_a)
                feats_b = encoder.backbone(clips_b)

                emb_a = reid_head(feats_a, boxes_a)
                emb_b = reid_head(feats_b, boxes_b)

                # Flatten batch embeddings + track IDs
                all_emb = torch.cat(emb_a + emb_b, dim=0)
                all_tids = torch.cat(tids_a + tids_b, dim=0)

                if all_emb.shape[0] < 2:
                    continue

                # Fetch memory bank entries
                bank_embs, bank_tids = memory_bank.get()

                # Compute losses with memory bank
                loss = torch.tensor(0., device=device)
                trip_loss_val = 0.0
                cont_loss_val = 0.0

                if args.loss_type in ['triplet', 'both']:
                    trip_loss = triplet_loss_fn(all_emb, all_tids, bank_embs, bank_tids)
                    loss = loss + trip_w * trip_loss
                    trip_loss_val = trip_loss.item()

                if args.loss_type in ['contrastive', 'both']:
                    cont_loss = contrastive_loss_fn(all_emb, all_tids, bank_embs, bank_tids)
                    loss = loss + cont_w * cont_loss
                    cont_loss_val = cont_loss.item()

            # Enqueue current batch into memory bank (before backward, detached)
            memory_bank.enqueue(all_emb, all_tids)

            scaler.scale(loss / accum_steps).backward()
            if (step % accum_steps == 0) or (step == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(reid_head.parameters()) + lora_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            total_trip += trip_loss_val
            total_cont += cont_loss_val
            n_batches += 1

            # Progress: loss, trip, cont, temperature, bank size
            pbar.set_postfix({
                'loss': f'{(total_loss/n_batches):.3f}',
                'trip': f'{(total_trip/n_batches):.3f}',
                'cont': f'{(total_cont/n_batches):.3f}',
                'temp': f'{contrastive_loss_fn.temperature:.3f}',
                'bank_sz': memory_bank.size(),
            })

        # Validation
        pos_acc, neg_acc = evaluate_reid(reid_head, encoder, val_loader, device)

        avg_loss = total_loss / max(n_batches, 1)
        avg_trip = total_trip / max(n_batches, 1)
        avg_cont = total_cont / max(n_batches, 1)

        logger.info(
            f"Epoch {epoch+1} | Loss: {avg_loss:.3f} "
            f"(trip={avg_trip:.3f} w={trip_w:.2f}, cont={avg_cont:.3f} w={cont_w:.2f}) | "
            f"Temp: {contrastive_loss_fn.temperature:.3f} | Bank: {memory_bank.size()} | "
            f"Pos Acc: {pos_acc:.3f} | Neg Acc: {neg_acc:.3f}")

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", avg_loss, epoch)
            tb_writer.add_scalar("train/triplet", avg_trip, epoch)
            tb_writer.add_scalar("train/contrastive", avg_cont, epoch)
            tb_writer.add_scalar("train/temperature", contrastive_loss_fn.temperature, epoch)
            tb_writer.add_scalar("train/bank_size", memory_bank.size(), epoch)
            tb_writer.add_scalar("val/pos_acc", pos_acc, epoch)
            tb_writer.add_scalar("val/neg_acc", neg_acc, epoch)
        if args.use_wandb and wandb is not None:
            wandb.log({
                "train/loss": avg_loss,
                "train/triplet": avg_trip,
                "train/contrastive": avg_cont,
                "train/temperature": contrastive_loss_fn.temperature,
                "train/bank_size": memory_bank.size(),
                "train/triplet_weight": trip_w,
                "train/contrastive_weight": cont_w,
                "val/pos_acc": pos_acc,
                "val/neg_acc": neg_acc,
                "epoch": epoch,
            }, step=epoch)

        # Save best
        if pos_acc > best_pos_acc:
            best_pos_acc = pos_acc
            torch.save({
                'epoch': epoch,
                'reid_head': reid_head.state_dict(),
                'encoder_lora': {k: v for k, v in encoder.state_dict().items() if 'lora' in k},
                'best_pos_acc': pos_acc,
                'args': vars(args),
            }, out_dir / 'best.pt')
            logger.info(f"  ★ New best pos acc: {pos_acc:.3f}")

        # Periodic numbered checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'reid_head': reid_head.state_dict(),
                'encoder_lora': {k: v for k, v in encoder.state_dict().items() if 'lora' in k},
                'best_pos_acc': pos_acc,
                'args': vars(args),
            }, out_dir / f'e{epoch+1}.pt')
            logger.info(f"  Checkpoint saved: e{epoch+1}.pt")

        scheduler.step()

    logger.info(f"\nRe-ID training complete! Best positive accuracy: {best_pos_acc:.3f}")


if __name__ == '__main__':
    main()
