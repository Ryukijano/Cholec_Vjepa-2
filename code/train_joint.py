#!/usr/bin/env python3
"""
Joint Detection + Re-ID training on CholecTrack20 with V-JEPA2 encoder.

Trains both the LightweightQueryDecoder (detection) and ReIDHead (re-identification)
simultaneously on a shared V-JEPA encoder with LoRA fine-tuning.

Design principles (FairMOT / JDE best practices):
  - Single encoder forward pass per batch (halves compute vs two-stage)
  - Re-ID uses GT boxes for ROI pooling during training (predicted boxes at inference)
  - Curriculum warmup: detection-only for first N epochs, then joint
  - Gradient management: Re-ID gradients on encoder can be detached or scaled
  - Memory bank (8192 entries) for rich negative mining in Re-ID losses
  - Dynamic loss weighting for triplet vs contrastive (0.7/0.3 -> 0.5/0.5)

Supports loading from:
  - SSL checkpoint (encoder key)
  - Detection-only checkpoint (encoder_lora + head keys)
  - Previous joint checkpoint (all keys)

Speed (Windows + CUDA):
  - DataLoader: prefetch_factor, num_workers, pin_memory (defaults tuned for Windows).
  - torch.compile(encoder/decoder/reid): use --compile_encoder --compile_decoder --compile_reid
    (PyTorch 2+; first epoch may be slower due to compilation).
  - Flash SDP for attention is enabled by default (--no_flash_sdp to disable).
  - On Windows, for Triton-backed kernels with torch.compile, install: pip install triton-windows

Usage:
  python train_joint.py --checkpoint ../outputs/ssl/CholecTrack20_e20.pt \\
      --train_dir ../cholec_dataset/Training \\
      --out_dir ../outputs/joint-det-reid \\
      --compile_encoder --compile_decoder --compile_reid \\
      --epochs 80 --reid_warmup_epochs 5 --margin 0.7
"""

import argparse
import copy
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup (run from code/ or project root)
# ---------------------------------------------------------------------------
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
    DEFAULT_CLASS_WEIGHTS, hungarian_match,
)
from reid_head import ReIDHead, TripletLoss, TrackContrastiveLoss, EmbeddingMemoryBank
from dataset import CholecDetectDataset, collate_fn
from detection_viz import run_detection_visualization, make_recall_bar_chart

try:
    from torchvision.ops import box_iou
except ImportError:
    box_iou = None

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

TOOL_NAMES = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def get_reid_loss_weights(epoch: int, total_epochs: int):
    """Dynamic triplet/contrastive weighting (same as train_reid_windows.py)."""
    progress = epoch / max(total_epochs - 1, 1)
    triplet_w = 0.7 - 0.2 * progress    # 0.7 -> 0.5
    contrastive_w = 0.3 + 0.2 * progress  # 0.3 -> 0.5
    return triplet_w, contrastive_w


def get_reid_lambda(epoch: int, reid_warmup_epochs: int, reid_ramp_epochs: int,
                    reid_weight: float) -> float:
    """Curriculum schedule for Re-ID loss weight.

    Returns 0 during warmup, then linearly ramps to ``reid_weight``
    over ``reid_ramp_epochs``.
    """
    if epoch < reid_warmup_epochs:
        return 0.0
    elapsed = epoch - reid_warmup_epochs
    ramp = min(1.0, elapsed / max(reid_ramp_epochs, 1))
    return ramp * reid_weight


# ---------------------------------------------------------------------------
# Validation: Detection Recall@0.5
# ---------------------------------------------------------------------------
def evaluate_detection(encoder, head, mid_hook, val_loader, device,
                       num_classes, epoch):
    """Per-class Recall@0.5 on the validation set (detection only)."""
    head.eval()
    per_class_tp = torch.zeros(num_classes)
    per_class_gt = torch.zeros(num_classes)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Det-Val]", leave=False):
            # Unpack – val loader may return 3 or 4 items depending on load_track_ids
            clips = batch[0].to(device, non_blocking=True)
            boxes_list = batch[1]
            labels_list = batch[2]

            if clips.size(2) < 2:
                clips = torch.cat([clips, clips], dim=2)

            with torch.amp.autocast(device_type='cuda', enabled=True):
                full_tokens = encoder([clips])[0]
                mid_features = mid_hook.get_features()
                logits, pred_boxes, _, _ = head(full_tokens.float(), mid_features.float())

            probs = logits.softmax(-1).cpu()
            scores, pred_cls = probs[..., :-1].max(-1)
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes.cpu())

            for b in range(len(boxes_list)):
                gt_boxes_xyxy = box_cxcywh_to_xyxy(boxes_list[b])
                gt_cls = labels_list[b]

                for c in gt_cls:
                    per_class_gt[c] += 1

                if gt_cls.numel() == 0:
                    continue

                b_scores = scores[b]
                b_cls = pred_cls[b]
                b_boxes = pred_boxes_xyxy[b]
                order = b_scores.argsort(descending=True)

                matched_gt = set()
                for qi in order:
                    pc = b_cls[qi].item()
                    if pc >= num_classes or b_scores[qi] < 0.3:
                        continue
                    if box_iou is not None and gt_boxes_xyxy.numel() > 0:
                        ious = box_iou(b_boxes[qi:qi + 1], gt_boxes_xyxy)[0]
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


# ---------------------------------------------------------------------------
# Validation: Re-ID Positive / Negative Accuracy (within-batch)
# ---------------------------------------------------------------------------
def evaluate_reid(encoder, reid_head, val_loader, device):
    """Evaluate Re-ID accuracy by checking cosine similarity between detections
    sharing the same track ID (positive) vs different track IDs (negative)
    across the batch.
    """
    reid_head.eval()
    correct_pos = 0
    correct_neg = 0
    total_pos = 0
    total_neg = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Re-ID Val", leave=False):
            clips = batch[0].to(device, non_blocking=True)
            boxes_list = batch[1]
            # labels_list = batch[2]  # not needed for Re-ID eval
            if len(batch) < 4:
                continue  # no track IDs in this loader
            tids_list = batch[3]

            if clips.size(2) < 2:
                clips = torch.cat([clips, clips], dim=2)

            with torch.amp.autocast(device_type='cuda', enabled=True):
                full_tokens = encoder([clips])[0]
                boxes_dev = [b.to(device) for b in boxes_list]
                embs = reid_head(full_tokens, boxes_dev)  # list of [M_i, 128]

            # Flatten all embeddings + track IDs across batch
            all_emb, all_tid = [], []
            for i in range(len(embs)):
                if embs[i].shape[0] == 0:
                    continue
                all_emb.append(embs[i].cpu())
                all_tid.append(tids_list[i])
            if len(all_emb) == 0:
                continue

            all_emb = torch.cat(all_emb, dim=0)   # [N_total, 128]
            all_tid = torch.cat(all_tid, dim=0)    # [N_total]

            # Only evaluate detections with valid track IDs (>= 0)
            valid = all_tid >= 0
            if valid.sum() < 2:
                continue
            all_emb = all_emb[valid]
            all_tid = all_tid[valid]

            # Pairwise cosine similarity
            sim = torch.mm(all_emb, all_emb.T)  # already L2-normed
            N = all_emb.shape[0]
            for i in range(N):
                for j in range(i + 1, N):
                    if all_tid[i] == all_tid[j]:
                        total_pos += 1
                        if sim[i, j] > 0.5:
                            correct_pos += 1
                    else:
                        total_neg += 1
                        if sim[i, j] < 0.5:
                            correct_neg += 1

    pos_acc = correct_pos / max(total_pos, 1)
    neg_acc = correct_neg / max(total_neg, 1)
    return pos_acc, neg_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Joint Detection + Re-ID training on CholecTrack20")

    # --- Paths ---
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to SSL, detection, or joint checkpoint')
    parser.add_argument('--train_dir', type=str,
                        default='../cholec_dataset/Training')
    parser.add_argument('--val_dir', type=str,
                        default='../cholec_dataset/Validation')
    parser.add_argument('--out_dir', type=str,
                        default='../outputs/joint-det-reid')

    # --- Detection model ---
    parser.add_argument('--num_queries', type=int, default=100)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--decoder_nheads', type=int, default=8)
    parser.add_argument('--internal_dim', type=int, default=256)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--temporal_pooling', type=str, default='last',
                        choices=['last', 'mean'])
    parser.add_argument('--use_aux_loss', action='store_true')
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--mid_layer_idx', type=int, default=12)

    # --- LoRA ---
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--lora_start_layer', type=int, default=12)

    # --- Re-ID model ---
    parser.add_argument('--reid_dim', type=int, default=128,
                        help='Re-ID embedding dimension')

    # --- Training ---
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Per-step batch size (micro-batch if grad_accum_steps > 1). '
                             'On RTX 4090 (24GB): try 8–12 with clip_len=16; reduce to 4–6 if OOM.')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps. Effective batch = batch_size * grad_accum_steps. '
                             'Use 2–4 on 4090 to get larger effective batch without OOM.')
    parser.add_argument('--decoder_lr', type=float, default=5e-4)
    parser.add_argument('--reid_lr', type=float, default=3.2e-4)
    parser.add_argument('--lora_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='DataLoader prefetch batches per worker (2-4 on Windows).')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='LR warmup epochs (both tasks)')
    parser.add_argument('--class_balanced', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last.pt if it exists')

    # --- Detection losses ---
    parser.add_argument('--supcon_weight', type=float, default=0.5)
    parser.add_argument('--supcon_temp', type=float, default=0.07)
    parser.add_argument('--use_focal_loss', action='store_true')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--class_head_warmup', type=int, default=1)

    # --- Re-ID losses ---
    parser.add_argument('--margin', type=float, default=0.7,
                        help='Triplet margin (0.7 avoids Re-ID collapse)')
    parser.add_argument('--temp_start', type=float, default=0.15)
    parser.add_argument('--temp_final', type=float, default=0.07)
    parser.add_argument('--temp_warmup_epochs', type=int, default=5)
    parser.add_argument('--memory_bank_size', type=int, default=8192)
    parser.add_argument('--reid_loss_type', type=str, default='both',
                        choices=['triplet', 'contrastive', 'both'])

    # --- Joint training ---
    parser.add_argument('--reid_warmup_epochs', type=int, default=5,
                        help='Epochs of detection-only before Re-ID kicks in')
    parser.add_argument('--reid_ramp_epochs', type=int, default=5,
                        help='Epochs to linearly ramp Re-ID weight after warmup')
    parser.add_argument('--reid_weight', type=float, default=1.0,
                        help='Final Re-ID loss multiplier (after ramp)')
    parser.add_argument('--reid_encoder_grad', action='store_true',
                        help='Allow Re-ID loss to backprop into encoder '
                             '(default: detach encoder tokens for Re-ID)')
    parser.add_argument('--use_plateau_scheduler', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=15)
    parser.add_argument('--det_val_weight', type=float, default=0.7,
                        help='Weight of detection recall in combined val metric')

    # --- Speed (Windows + CUDA) ---
    parser.add_argument('--compile_encoder', action='store_true',
                        help='torch.compile(encoder.backbone) for faster forward (PyTorch 2+).')
    parser.add_argument('--compile_decoder', action='store_true',
                        help='torch.compile(decoder) for faster forward.')
    parser.add_argument('--compile_reid', action='store_true',
                        help='torch.compile(reid_head) for faster forward.')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode. reduce-overhead good for training.')
    parser.add_argument('--force_compile_decoder_reduce_overhead', action='store_true',
                        help='Force decoder compile in reduce-overhead mode. '
                             'Not recommended on Windows due to CUDA graph overwrite errors.')
    parser.add_argument('--flash_sdp', action='store_true', default=True,
                        help='Use Flash Attention SDP when available (PyTorch 2.0+).')
    parser.add_argument('--no_flash_sdp', action='store_false', dest='flash_sdp',
                        help='Disable Flash SDP (use if unstable).')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # Flash / memory-efficient attention (PyTorch 2.0+); falls back to math if unavailable
    if args.flash_sdp and device == 'cuda':
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                logger.info("Flash SDP and memory-efficient SDP enabled for attention.")
            except Exception as e:
                logger.warning(f"Could not enable Flash SDP: {e}")

    # --- Logging ---
    if args.use_wandb and wandb is not None:
        wandb.init(project="vjepa2-joint-det-reid", config=vars(args))
    tb_writer = None
    if SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    # ===================================================================
    # Datasets  (with track IDs for Re-ID)
    # ===================================================================
    logger.info("Loading datasets (with track IDs)...")
    train_ds = CholecDetectDataset(
        args.train_dir, clip_len=args.clip_len, load_track_ids=True)
    val_ds = CholecDetectDataset(
        args.val_dir, clip_len=args.clip_len, load_track_ids=True)

    all_labels = [l for item in train_ds.items for l in item['labels']]
    num_classes = max(all_labels) + 1 if all_labels else 7
    logger.info(f"Classes: {num_classes}, Train: {len(train_ds)}, Val: {len(val_ds)}")

    from dataset import IdentitySampler
    # For Triplet Loss to work, we MUST have positive pairs in each batch.
    # IdentitySampler ensures each batch contains multiple frames from the same video.
    train_sampler = IdentitySampler(train_ds, batch_size=args.batch_size, num_instances=4)
    logger.info(f"Using IdentitySampler: batch_size={args.batch_size}, num_instances=4")

    _loader_kw = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        _loader_kw['prefetch_factor'] = max(2, getattr(args, 'prefetch_factor', 4))
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, **_loader_kw)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, **_loader_kw)

    # ===================================================================
    # Encoder
    # ===================================================================
    logger.info("Initializing V-JEPA2 encoder...")
    encoder, _predictor = init_video_model(
        device=device, model_name='vit_large', patch_size=16,
        max_num_frames=args.clip_len, tubelet_size=2, crop_size=224,
        pred_depth=12, pred_embed_dim=384,
        use_mask_tokens=True, use_sdpa=True)

    logger.info(f"Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}, "
                f"start={args.lora_start_layer})")
    apply_lora_to_encoder(encoder, rank=args.lora_rank, alpha=args.lora_alpha,
                          start_layer=args.lora_start_layer)

    # Load encoder weights from checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'encoder' in ckpt:
        load_checkpoint_into_lora_model(encoder, ckpt['encoder'])
    elif 'encoder_lora' in ckpt:
        load_checkpoint_into_lora_model(encoder, ckpt['encoder_lora'])
    else:
        raise KeyError("Checkpoint must contain 'encoder' or 'encoder_lora'")
    encoder.to(device).eval()

    # Freeze base weights, keep LoRA trainable
    lora_params = []
    for name, param in encoder.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    logger.info(f"LoRA trainable params: {sum(p.numel() for p in lora_params):,}")

    # Mid-backbone hook (register before compile so hook stays on inner blocks)
    mid_hook = MidBackboneHook(layer_idx=args.mid_layer_idx)
    mid_hook.register(encoder.backbone.blocks)

    # Optional: torch.compile encoder backbone (PyTorch 2+; on Windows use default Inductor)
    if getattr(args, 'compile_encoder', False) and hasattr(torch, 'compile'):
        try:
            encoder.backbone = torch.compile(
                encoder.backbone, mode=args.compile_mode, fullgraph=False)
            logger.info(f"  Encoder backbone compiled (mode={args.compile_mode}).")
        except Exception as e:
            logger.warning(f"  Encoder compile failed: {e}")

    # ===================================================================
    # Detection Head
    # ===================================================================
    head = LightweightQueryDecoder(
        embed_dim=1024, num_classes=num_classes,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        nheads=args.decoder_nheads,
        dim_feedforward=args.dim_feedforward, dropout=0.1, grid_size=14,
        internal_dim=args.internal_dim,
        temporal_pooling=args.temporal_pooling,
    ).to(device)
    logger.info(f"Decoder params: {sum(p.numel() for p in head.parameters()):,}")

    # SupCon head
    sc_input_dim = head.internal_dim
    supcon_head = SupConProjectionHead(
        embed_dim=sc_input_dim, hidden_dim=min(512, sc_input_dim * 2),
        proj_dim=128).to(device)
    supcon_loss_fn = SupConLoss(temperature=args.supcon_temp)

    # Token spatial classifier
    token_spatial_head = TokenSpatialClassifier(
        embed_dim=1024, num_classes=num_classes, hidden_dim=256, grid_size=14
    ).to(device)

    # Load detection head weights if present in checkpoint
    if 'head' in ckpt:
        head.load_state_dict(ckpt['head'])
        logger.info("  Loaded detection head from checkpoint")
    if 'supcon_head' in ckpt:
        supcon_head.load_state_dict(ckpt['supcon_head'])
        logger.info("  Loaded SupCon head from checkpoint")
    if 'token_spatial_head' in ckpt:
        token_spatial_head.load_state_dict(ckpt['token_spatial_head'])
        logger.info("  Loaded TokenSpatialClassifier from checkpoint")

    # Optional: torch.compile decoder
    if getattr(args, 'compile_decoder', False) and hasattr(torch, 'compile'):
        try:
            decoder_mode = args.compile_mode
            # Known issue on Windows: decoder in reduce-overhead mode can hit
            # CUDA graph output overwrite errors during backward.
            if (
                os.name == 'nt'
                and decoder_mode == 'reduce-overhead'
                and not args.force_compile_decoder_reduce_overhead
            ):
                logger.warning(
                    "  Skipping decoder compile in reduce-overhead mode on Windows "
                    "(known CUDA graph overwrite issue). "
                    "Use --force_compile_decoder_reduce_overhead to override."
                )
            else:
                head = torch.compile(head, mode=decoder_mode, fullgraph=False)
                logger.info(f"  Decoder compiled (mode={decoder_mode}).")
        except Exception as e:
            logger.warning(f"  Decoder compile failed: {e}")

    # ===================================================================
    # Re-ID Head + Memory Bank + Losses
    # ===================================================================
    reid_head = ReIDHead(
        embed_dim=1024, grid_size=14, hidden_dim=512, reid_dim=args.reid_dim,
    ).to(device)
    logger.info(f"ReIDHead params: {sum(p.numel() for p in reid_head.parameters()):,}")

    # Load Re-ID head if present in checkpoint
    if 'reid_head' in ckpt:
        def _strip_compiled(sd):
            return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        reid_sd = _strip_compiled(ckpt['reid_head'])
        current_sd = reid_head.state_dict()
        compatible_sd = {k: v for k, v in reid_sd.items()
                        if k in current_sd and current_sd[k].shape == v.shape}
        skipped = [k for k in reid_sd if k not in compatible_sd]
        if skipped:
            logger.warning(f"  Re-ID head: skipped {len(skipped)} keys with shape mismatch")
        reid_head.load_state_dict(compatible_sd, strict=False)
        logger.info(f"  Loaded Re-ID head ({len(compatible_sd)}/{len(reid_sd)} params)")

    # Optional: torch.compile Re-ID head
    if getattr(args, 'compile_reid', False) and hasattr(torch, 'compile'):
        try:
            reid_head = torch.compile(reid_head, mode=args.compile_mode, fullgraph=False)
            logger.info(f"  Re-ID head compiled (mode={args.compile_mode}).")
        except Exception as e:
            logger.warning(f"  Re-ID head compile failed: {e}")

    memory_bank = EmbeddingMemoryBank(
        capacity=args.memory_bank_size, embed_dim=args.reid_dim, device=device)

    triplet_loss_fn = TripletLoss(margin=args.margin)
    contrastive_loss_fn = TrackContrastiveLoss(
        temp_start=args.temp_start, temp_final=args.temp_final,
        warmup_epochs=args.temp_warmup_epochs)

    # ===================================================================
    # Optimizer (separate param groups for different LRs)
    # ===================================================================
    param_groups = [
        {'params': list(head.parameters()), 'lr': args.decoder_lr, 'name': 'decoder'},
        {'params': list(supcon_head.parameters()), 'lr': args.decoder_lr, 'name': 'supcon'},
        {'params': list(token_spatial_head.parameters()), 'lr': args.decoder_lr, 'name': 'token_spatial'},
        {'params': list(reid_head.parameters()), 'lr': args.reid_lr, 'name': 'reid'},
    ]
    if lora_params:
        param_groups.append({'params': lora_params, 'lr': args.lora_lr, 'name': 'lora'})

    optimizer = optim.AdamW(param_groups, lr=args.decoder_lr,
                            weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    if args.use_plateau_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5)
        logger.info("Using ReduceLROnPlateau scheduler")
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)
        logger.info("Using CosineAnnealingLR scheduler")

    accum_steps = max(1, getattr(args, 'grad_accum_steps', 1))
    total_trainable = sum(p.numel() for pg in param_groups for p in pg['params'])
    logger.info(f"Total trainable: {total_trainable:,}")
    logger.info(f"LR: decoder={args.decoder_lr}, reid={args.reid_lr}, lora={args.lora_lr}")
    logger.info(f"Batch: micro={args.batch_size}, grad_accum={accum_steps}, "
                f"effective={args.batch_size * accum_steps}")
    logger.info(f"Joint config: reid_warmup={args.reid_warmup_epochs}, "
                f"reid_ramp={args.reid_ramp_epochs}, reid_weight={args.reid_weight}, "
                f"reid_encoder_grad={args.reid_encoder_grad}")

    # ===================================================================
    # Resume from last.pt
    # ===================================================================
    start_epoch = 0
    best_combined = 0.0
    epochs_no_improve = 0

    resume_path = out_dir / 'last.pt'
    if args.resume and resume_path.exists():
        logger.info(f"Resuming from {resume_path}")
        ckpt_r = torch.load(resume_path, map_location='cpu')
        if 'head' in ckpt_r:
            head.load_state_dict(ckpt_r['head'])
        if 'supcon_head' in ckpt_r:
            supcon_head.load_state_dict(ckpt_r['supcon_head'])
        if 'token_spatial_head' in ckpt_r:
            token_spatial_head.load_state_dict(ckpt_r['token_spatial_head'])
        if 'reid_head' in ckpt_r:
            reid_head.load_state_dict(ckpt_r['reid_head'])
        if 'optimizer' in ckpt_r:
            optimizer.load_state_dict(ckpt_r['optimizer'])
        if 'encoder_lora' in ckpt_r:
            lora_state = ckpt_r['encoder_lora']
            enc_state = encoder.state_dict()
            enc_state.update(lora_state)
            encoder.load_state_dict(enc_state)
        start_epoch = ckpt_r.get('epoch', -1) + 1
        best_combined = ckpt_r.get('best_combined', 0.0)
        epochs_no_improve = ckpt_r.get('epochs_no_improve', 0)
        logger.info(f"Resumed: start_epoch={start_epoch}, best_combined={best_combined:.3f}")
    elif args.resume:
        logger.info("--resume requested but no last.pt found. Starting fresh.")

    # ===================================================================
    # Training loop
    # ===================================================================
    class_weights = DEFAULT_CLASS_WEIGHTS[:num_classes]

    # Use standard DataLoader for validation (no IdentitySampler needed)
    val_loader_standard = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, **_loader_kw)

    for epoch in range(start_epoch, args.epochs):
        head.train()
        supcon_head.train()
        token_spatial_head.train()
        reid_head.train()

        # Re-ID curriculum
        reid_lambda = get_reid_lambda(
            epoch, args.reid_warmup_epochs, args.reid_ramp_epochs, args.reid_weight)
        reid_active = reid_lambda > 0
        contrastive_loss_fn.set_epoch(epoch)
        trip_w, cont_w = get_reid_loss_weights(epoch, args.epochs)

        logger.info(f"--- Epoch {epoch + 1}/{args.epochs} | "
                    f"reid_lambda={reid_lambda:.3f} {'(active)' if reid_active else '(warmup)'} ---")

        # Classification head warmup
        if epoch < args.class_head_warmup:
            for p in head.box_head_coarse.parameters():
                p.requires_grad = False
            for p in head.box_head_refine.parameters():
                p.requires_grad = False
            for p in head.class_head.parameters():
                p.requires_grad = True
        else:
            for p in head.box_head_coarse.parameters():
                p.requires_grad = True
            for p in head.box_head_refine.parameters():
                p.requires_grad = True

        # LR warmup
        if epoch < args.warmup_epochs:
            warmup_scale = (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                if pg['name'] == 'lora':
                    pg['lr'] = args.lora_lr * warmup_scale
                elif pg['name'] == 'reid':
                    pg['lr'] = args.reid_lr * warmup_scale
                else:
                    pg['lr'] = args.decoder_lr * warmup_scale

        meters = {
            'det_ce': 0., 'det_l1': 0., 'det_giou': 0., 'det_sc': 0.,
            'tok_cls': 0., 'tok_fg': 0.,
            'reid_trip': 0., 'reid_cont': 0., 'reid_total': 0.,
            'total': 0.,
        }
        n_batches = 0

        accum_steps = max(1, getattr(args, 'grad_accum_steps', 1))
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(pbar, start=1):
            clips = batch[0].to(device, non_blocking=True)
            targets_boxes = [b.to(device) for b in batch[1]]
            targets_labels = [l.to(device) for l in batch[2]]
            # Track IDs (may contain -1 for detections without valid track IDs)
            targets_tids = [t.to(device) for t in batch[3]]
            B = clips.shape[0]

            if clips.size(2) < 2:
                clips = torch.cat([clips, clips], dim=2)

            # Mark step boundary for torch.compile(CUDA graphs) to avoid buffer overwrite errors
            if (getattr(args, 'compile_decoder', False) or getattr(args, 'compile_encoder', False)) and hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                torch.compiler.cudagraph_mark_step_begin()

            with torch.amp.autocast(device_type='cuda', enabled=True):
                # === Single encoder forward pass ===
                full_tokens = encoder([clips])[0]
                mid_features = mid_hook.get_features()

                # -------------------------------------------------------
                # DETECTION BRANCH
                # -------------------------------------------------------
                logits, pred_boxes, query_features, aux_outputs = head(
                    full_tokens, mid_features)

                if args.use_focal_loss:
                    ce_loss, l1_loss, giou_loss = detection_loss_focal(
                        logits, pred_boxes, targets_labels, targets_boxes,
                        num_classes=num_classes, class_weights=class_weights,
                        focal_gamma=args.focal_gamma)
                else:
                    ce_loss, l1_loss, giou_loss = detection_loss(
                        logits, pred_boxes, targets_labels, targets_boxes,
                        num_classes=num_classes, class_weights=class_weights)

                # Auxiliary decoder layer losses
                aux_loss = torch.tensor(0., device=device)
                if args.use_aux_loss and aux_outputs:
                    for aux in aux_outputs:
                        if args.use_focal_loss:
                            ace, al1, agiou = detection_loss_focal(
                                aux['logits'], aux['boxes'],
                                targets_labels, targets_boxes,
                                num_classes=num_classes, class_weights=class_weights,
                                focal_gamma=args.focal_gamma)
                        else:
                            ace, al1, agiou = detection_loss(
                                aux['logits'], aux['boxes'],
                                targets_labels, targets_boxes,
                                num_classes=num_classes, class_weights=class_weights)
                        aux_loss += ace + 10.0 * al1 + 5.0 * agiou
                    aux_loss = aux_loss / len(aux_outputs)

                # Token spatial loss (direct encoder supervision)
                N_spatial = 14 * 14
                spatial_tokens = full_tokens
                if spatial_tokens.dim() == 3 and spatial_tokens.shape[1] > N_spatial:
                    D_st = spatial_tokens.shape[-1]
                    T_st = spatial_tokens.shape[1] // N_spatial
                    spatial_tokens = spatial_tokens.view(B, T_st, N_spatial, D_st)
                if spatial_tokens.dim() == 4:
                    spatial_tokens = spatial_tokens[:, -1]
                token_cls_loss, token_fg_loss = token_spatial_head(
                    spatial_tokens, targets_boxes, targets_labels)

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

                # Total detection loss
                det_loss = (ce_loss + 10.0 * l1_loss + 5.0 * giou_loss
                            + args.supcon_weight * sc_loss
                            + 0.5 * aux_loss
                            + 2.0 * (token_cls_loss + token_fg_loss))

                # -------------------------------------------------------
                # RE-ID BRANCH
                # -------------------------------------------------------
                reid_loss = torch.tensor(0., device=device)
                trip_loss_val = 0.0
                cont_loss_val = 0.0

                if reid_active:
                    # Choose whether Re-ID gradients flow through encoder
                    if args.reid_encoder_grad:
                        reid_tokens = full_tokens
                    else:
                        reid_tokens = full_tokens.detach()

                    # Compute Re-ID embeddings using GT boxes
                    emb_list = reid_head(reid_tokens, targets_boxes)

                    # Flatten embeddings + track IDs across batch
                    all_emb, all_tid = [], []
                    for b_idx in range(B):
                        if emb_list[b_idx].shape[0] == 0:
                            continue
                        all_emb.append(emb_list[b_idx])
                        all_tid.append(targets_tids[b_idx])

                    if len(all_emb) > 0:
                        all_emb = torch.cat(all_emb, dim=0)
                        all_tid = torch.cat(all_tid, dim=0)

                        # Filter out detections with invalid track IDs
                        valid_mask = all_tid >= 0
                        if valid_mask.sum() >= 2:
                            valid_emb = all_emb[valid_mask]
                            valid_tid = all_tid[valid_mask]

                            bank_embs, bank_tids = memory_bank.get()

                            if args.reid_loss_type in ['triplet', 'both']:
                                trip_loss = triplet_loss_fn(
                                    valid_emb, valid_tid, bank_embs, bank_tids)
                                reid_loss = reid_loss + trip_w * trip_loss
                                trip_loss_val = trip_loss.item()

                            if args.reid_loss_type in ['contrastive', 'both']:
                                cont_loss = contrastive_loss_fn(
                                    valid_emb, valid_tid, bank_embs, bank_tids)
                                reid_loss = reid_loss + cont_w * cont_loss
                                cont_loss_val = cont_loss.item()

                            # Enqueue into memory bank
                            memory_bank.enqueue(valid_emb, valid_tid)

                # -------------------------------------------------------
                # TOTAL LOSS
                # -------------------------------------------------------
                loss_full = det_loss + reid_lambda * reid_loss
                loss_scaled = loss_full / accum_steps

            # Backward (accumulate gradients)
            scaler.scale(loss_scaled).backward()
            # Step only every accum_steps (or at end of epoch)
            if (step % accum_steps == 0) or (step == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in param_groups for p in pg['params']], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Meters
            meters['det_ce'] += ce_loss.item()
            meters['det_l1'] += l1_loss.item()
            meters['det_giou'] += giou_loss.item()
            meters['det_sc'] += sc_loss.item() if isinstance(sc_loss, torch.Tensor) else sc_loss
            meters['tok_cls'] += token_cls_loss.item()
            meters['tok_fg'] += token_fg_loss.item()
            meters['reid_trip'] += trip_loss_val
            meters['reid_cont'] += cont_loss_val
            meters['reid_total'] += reid_loss.item()
            meters['total'] += loss_full.item()
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{meters['total'] / n_batches:.3f}",
                'det': f"{(meters['det_ce'] + meters['det_l1'] + meters['det_giou']) / n_batches:.3f}",
                'reid': f"{meters['reid_total'] / n_batches:.3f}",
                'lam': f"{reid_lambda:.2f}",
            })

        # ---------------------------------------------------------------
        # Epoch-level logging
        # ---------------------------------------------------------------
        avg = {k: v / max(n_batches, 1) for k, v in meters.items()}
        logger.info(
            f"Epoch {epoch + 1} | Total: {avg['total']:.3f} | "
            f"Det(ce={avg['det_ce']:.3f} l1={avg['det_l1']:.3f} giou={avg['det_giou']:.3f} "
            f"sc={avg['det_sc']:.3f} tok={avg['tok_cls'] + avg['tok_fg']:.3f}) | "
            f"ReID(trip={avg['reid_trip']:.3f} cont={avg['reid_cont']:.3f} "
            f"lam={reid_lambda:.3f} bank={memory_bank.size()})")

        if tb_writer is not None:
            for k, v in avg.items():
                tb_writer.add_scalar(f"train/{k}", v, epoch)
            tb_writer.add_scalar("train/reid_lambda", reid_lambda, epoch)
            tb_writer.add_scalar("train/reid_temp", contrastive_loss_fn.temperature, epoch)
            tb_writer.add_scalar("train/memory_bank_size", memory_bank.size(), epoch)
        if args.use_wandb and wandb is not None:
            log = {f"train/{k}": v for k, v in avg.items()}
            log["train/reid_lambda"] = reid_lambda
            log["train/reid_temp"] = contrastive_loss_fn.temperature
            log["train/memory_bank_size"] = memory_bank.size()
            log["epoch"] = epoch
            wandb.log(log, step=epoch)

        # ---------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------
        per_class_recall, overall_recall, per_class_gt = evaluate_detection(
            encoder, head, mid_hook, val_loader_standard, device, num_classes, epoch + 1)

        logger.info(f"  Detection Recall@0.5: {overall_recall:.3f}")
        for c in range(num_classes):
            name = TOOL_NAMES[c] if c < len(TOOL_NAMES) else f"class_{c}"
            logger.info(f"    {name}: recall={per_class_recall[c]:.3f} "
                        f"(GT={int(per_class_gt[c].item())})")
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/recall_{name}", per_class_recall[c].item(), epoch)

        if tb_writer is not None:
            tb_writer.add_scalar("val/recall_overall", overall_recall.item(), epoch)

        # Re-ID validation (only when Re-ID is active)
        pos_acc, neg_acc = 0.0, 0.0
        if reid_active:
            pos_acc, neg_acc = evaluate_reid(
                encoder, reid_head, val_loader_standard, device)
            logger.info(f"  Re-ID Pos Acc: {pos_acc:.3f} | Neg Acc: {neg_acc:.3f}")
            if tb_writer is not None:
                tb_writer.add_scalar("val/reid_pos_acc", pos_acc, epoch)
                tb_writer.add_scalar("val/reid_neg_acc", neg_acc, epoch)

        if args.use_wandb and wandb is not None:
            wlog = {
                "val/recall_overall": overall_recall.item(),
                "val/reid_pos_acc": pos_acc,
                "val/reid_neg_acc": neg_acc,
                "epoch": epoch,
            }
            for c in range(num_classes):
                name = TOOL_NAMES[c] if c < len(TOOL_NAMES) else f"class_{c}"
                wlog[f"val/recall_{name}"] = per_class_recall[c].item()
            wandb.log(wlog, step=epoch)

        # Detection visualizations (every 5 epochs + first)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            recall_chart = make_recall_bar_chart(
                per_class_recall, per_class_gt, overall_recall, epoch + 1, num_classes)
            if args.use_wandb and wandb is not None:
                wandb.log({"viz/recall_chart": wandb.Image(recall_chart)}, step=epoch)
            if tb_writer is not None:
                tb_writer.add_image("viz/recall_chart", recall_chart,
                                    global_step=epoch, dataformats='HWC')
            run_detection_visualization(
                encoder=encoder, head=head, mid_hook=mid_hook,
                dataset=val_ds, device=device, epoch=epoch + 1,
                wandb_run=wandb if args.use_wandb else None,
                tb_writer=tb_writer, num_classes=num_classes,
                num_samples=8, score_thresh=0.3)

        # Scheduler
        if epoch >= args.warmup_epochs:
            if args.use_plateau_scheduler:
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(overall_recall.item())
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr < old_lr:
                    logger.info(f"  LR reduced: {old_lr:.2e} -> {new_lr:.2e}")
            else:
                scheduler.step()

        # ---------------------------------------------------------------
        # Combined metric for best-model saving
        # ---------------------------------------------------------------
        det_w = args.det_val_weight
        reid_w = 1.0 - det_w
        combined = det_w * overall_recall.item() + reid_w * pos_acc

        if combined > best_combined:
            best_combined = combined
            epochs_no_improve = 0
            save_dict = {
                'epoch': epoch,
                'head': head.state_dict(),
                'supcon_head': supcon_head.state_dict(),
                'token_spatial_head': token_spatial_head.state_dict(),
                'reid_head': reid_head.state_dict(),
                'encoder_lora': {k: v for k, v in encoder.state_dict().items()
                                 if 'lora' in k.lower()},
                'val_recall': overall_recall.item(),
                'val_reid_pos_acc': pos_acc,
                'best_combined': combined,
                'per_class_recall': {
                    (TOOL_NAMES[c] if c < len(TOOL_NAMES) else f"class_{c}"): per_class_recall[c].item()
                    for c in range(num_classes)
                },
                'args': vars(args),
            }
            torch.save(save_dict, out_dir / 'best.pt')
            logger.info(f"  * New best combined: {combined:.3f} "
                        f"(recall={overall_recall:.3f}, reid_pos={pos_acc:.3f}) -> saved best.pt")
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
            'reid_head': reid_head.state_dict(),
            'encoder_lora': {k: v for k, v in encoder.state_dict().items()
                             if 'lora' in k.lower()},
            'optimizer': optimizer.state_dict(),
            'val_recall': overall_recall.item(),
            'val_reid_pos_acc': pos_acc,
            'best_combined': best_combined,
            'epochs_no_improve': epochs_no_improve,
        }
        torch.save(state, out_dir / 'last.pt')

        # Periodic numbered checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save(state, out_dir / f'e{epoch + 1}.pt')

    mid_hook.remove()
    logger.info(f"\nJoint training complete! Best combined: {best_combined:.3f}")


if __name__ == '__main__':
    main()
