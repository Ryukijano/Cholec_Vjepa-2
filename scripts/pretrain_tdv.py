#!/usr/bin/env python3
"""
TDV pretraining script for surgical video domain adaptation.

Trains a TDV (Temporal Difference in Vision) model on the leak-free
Cholec80 SSL corpus, producing a domain-adapted DINOv2 encoder for
downstream surgical tool detection.

Usage:
    python scripts/pretrain_tdv.py --config configs/train_mot/dinov2/tdv-pretrain.yaml

The script supports:
    - Single-GPU and DDP training
    - EMA teacher updates
    - WandB logging
    - Checkpoint saving (best + latest)
    - L2-SP regularization (anchor to pretrained DINOv2 weights)
    - ExPLoRA-style progressive layer unfreezing
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import math
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# -- Debugging env vars (must be set before importing torch)
os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
os.environ.setdefault('TORCH_SHOW_CPP_STACKTRACE', '1')
os.environ.setdefault('NCCL_DEBUG', 'WARN')
# NCCL_P2P_DISABLE=1: L40S PCIe has no NVLink; P2P DMA segfaults.
os.environ.setdefault('NCCL_P2P_DISABLE', '1')
# NCCL_CUMEM_ENABLE=0: work around NCCL version mismatch (system NCCL 2.28
# vs PyTorch-bundled 2.26) causing segfault in CUDA memory registration.
os.environ.setdefault('NCCL_CUMEM_ENABLE', '0')
# Force socket transport for single-node (avoid IB/NET plugin issues).
os.environ.setdefault('NCCL_IB_DISABLE', '1')
os.environ.setdefault('NCCL_SOCKET_IFNAME', 'lo')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core_app.models.tdv_model import TDVModel
from core_app.tdv_dataloader import (
    Cholec80TDVDataset,
    parse_ssl_video_list,
    build_tdv_dataloader,
)


def cosine_lr_schedule(step, max_steps, warmup_steps, peak_lr, min_lr_scale=10):
    """Linear warmup → cosine decay."""
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    min_lr = peak_lr / min_lr_scale
    return min_lr + (peak_lr - min_lr) * cosine


def get_param_groups(model: TDVModel, weight_decay: float, l2sp_weight: float = 0.0):
    """Get parameter groups with layer-wise LR and optional L2-SP regularization.

    L2-SP anchors fine-tuned weights to their pretrained values:
        L_l2sp = λ * ||W - W_pretrained||^2
    """
    # Separate parameters: frame encoder (potentially unfrozen) vs motion encoder + heads
    frame_encoder_params = []
    other_params = []
    no_decay_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'frame_encoder' in name:
            frame_encoder_params.append(p)
        elif 'bias' in name or 'norm' in name or 'LayerNorm' in name:
            no_decay_params.append(p)
        else:
            other_params.append(p)

    param_groups = [
        {'params': frame_encoder_params, 'weight_decay': weight_decay, 'lr_scale': 0.1},
        {'params': other_params, 'weight_decay': weight_decay, 'lr_scale': 1.0},
        {'params': no_decay_params, 'weight_decay': 0.0, 'lr_scale': 1.0},
    ]

    return param_groups


def l2sp_loss(model: TDVModel, pretrained_encoder_sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    """L2-SP regularization: penalize drift from pretrained weights."""
    loss = torch.tensor(0., device=next(model.parameters()).device)
    current_sd = model.frame_encoder.encoder.state_dict()
    for name, param in current_sd.items():
        if name in pretrained_encoder_sd and pretrained_encoder_sd[name].shape == param.shape:
            diff = param - pretrained_encoder_sd[name].to(param.device)
            loss = loss + (diff ** 2).sum()
    return loss


def progressive_unfreeze(model: TDVModel, epoch: int, unfreeze_schedule: List[Dict]):
    """Progressively unfreeze encoder layers based on a schedule.

    Args:
        unfreeze_schedule: list of dicts with 'epoch' and 'num_blocks' keys,
            e.g. [{'epoch': 0, 'num_blocks': 0}, {'epoch': 5, 'num_blocks': 4}, ...]
    """
    # Find the current schedule entry
    current_blocks = 0
    for entry in sorted(unfreeze_schedule, key=lambda x: x['epoch']):
        if epoch >= entry['epoch']:
            current_blocks = entry['num_blocks']

    # Unfreeze the last N blocks of the frame encoder
    blocks = list(model.frame_encoder.encoder.blocks)
    total_blocks = len(blocks)
    blocks_to_unfreeze = current_blocks

    for i, block in enumerate(blocks):
        if i >= total_blocks - blocks_to_unfreeze:
            for p in block.parameters():
                p.requires_grad = True
        else:
            for p in block.parameters():
                p.requires_grad = False

    trainable = sum(p.numel() for p in model.frame_encoder.parameters() if p.requires_grad)
    print(f"[Progressive Unfreeze] Epoch {epoch}: {blocks_to_unfreeze}/{total_blocks} blocks trainable "
          f"({trainable / 1e6:.1f}M params in frame encoder)")


def train_tdv(config: dict, args: argparse.Namespace):
    if args.ddp:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank = 0
        world_size = 1
    print(f"Device: {device} | rank={rank} | world_size={world_size}")

    # -- Parse SSL video list
    splits_path = PROJECT_ROOT / config['splits_path']
    video_names, extras = parse_ssl_video_list(str(splits_path))
    print(f"SSL corpus: {len(video_names)} Cholec80 videos + {len(extras)} CT20 extras")

    # -- Build dataloader
    frames_root = config.get('frames_root', '/scratch/kcwp264/datasets_cholec/cholec80/cholec80/frames')
    dataloader = build_tdv_dataloader(
        frames_root=frames_root,
        video_names=video_names,
        batch_size=config.get('batch_size', 4),
        num_frames=config.get('num_frames', 4),
        img_size=config.get('img_size', 224),
        stride=config.get('stride', 1),
        num_workers=config.get('num_workers', 4),
        distributed=args.ddp,
    )

    # -- Build model
    model_cfg = config.get('model', {})
    model = TDVModel(
        backbone_type=model_cfg.get('backbone_type', 'dinov2'),
        backbone_size=model_cfg.get('backbone_size', 'base'),
        pretrained=model_cfg.get('pretrained', True),
        unfreeze_frame_encoder=model_cfg.get('unfreeze_frame_encoder', False),
        img_size=model_cfg.get('img_size', 224),
        patch_size=model_cfg.get('patch_size', 14),
        encoder_checkpoint=model_cfg.get('encoder_checkpoint', None),
        motion_encoder_depth=model_cfg.get('motion_encoder_depth', 4),
        motion_encoder_heads=model_cfg.get('motion_encoder_heads', 12),
        remove_motion_encoder=model_cfg.get('remove_motion_encoder', False),
        use_ema=model_cfg.get('use_ema', True),
        ema_momentum=model_cfg.get('ema_momentum', 0.996),
        use_fixed_dino_teacher=model_cfg.get('use_fixed_dino_teacher', False),
        use_dino_head=model_cfg.get('use_dino_head', True),
        dino_head_prototype_dim=model_cfg.get('dino_head_prototype_dim', 65536),
        use_separate_ibot_head=model_cfg.get('use_separate_ibot_head', False),
        use_mse_loss=model_cfg.get('use_mse_loss', True),
        mse_loss_weight=model_cfg.get('mse_loss_weight', 1.0),
        use_dino_loss=model_cfg.get('use_dino_loss', True),
        dino_loss_weight=model_cfg.get('dino_loss_weight', 1.0),
        use_ibot_loss=model_cfg.get('use_ibot_loss', False),
        ibot_loss_weight=model_cfg.get('ibot_loss_weight', 1.0),
        use_motion_loss=model_cfg.get('use_motion_loss', True),
        motion_loss_weight=model_cfg.get('motion_loss_weight', 0.1),
        min_embed_diff_per_pixel_diff=model_cfg.get('min_embed_diff_per_pixel_diff', 0.0),
        use_dino_augmentation=model_cfg.get('use_dino_augmentation', True),
        rollout_n_frames=model_cfg.get('rollout_n_frames', 1),
        rgb_diff_threshold=model_cfg.get('rgb_diff_threshold', 0.0),
        use_only_cls_token=model_cfg.get('use_only_cls_token', False),
        log_var_covar=model_cfg.get('log_var_covar', True),
        log_baseline_losses=model_cfg.get('log_baseline_losses', True),
    ).to(device)

    # -- Sync all ranks before DDP wrap (no device_ids — avoids NCCL guessing
    # segfault on some NCCL versions; the barrier itself is lightweight)
    if args.ddp:
        torch.distributed.barrier()
        if rank == 0:
            print(f"[DDP] All ranks ready, model loaded on device.")
            print(f"[DDP] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
            print(f"[DDP] Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # -- Save pretrained encoder state for L2-SP
    pretrained_encoder_sd = None
    l2sp_weight = config.get('l2sp_weight', 0.0)
    if l2sp_weight > 0:
        pretrained_encoder_sd = {
            k: v.clone() for k, v in model.frame_encoder.encoder.state_dict().items()
        }
        print(f"L2-SP regularization enabled (weight={l2sp_weight})")

    # -- Progressive unfreezing
    unfreeze_schedule = config.get('unfreeze_schedule', None)

    # -- Optimizer
    opt_cfg = config.get('optimizer', {})
    param_groups = get_param_groups(model, opt_cfg.get('weight_decay', 0.01), l2sp_weight)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=opt_cfg.get('peak_lr', 1e-4),
        betas=(opt_cfg.get('beta1', 0.9), opt_cfg.get('beta2', 0.999)),
    )

    # -- Training config
    max_steps = config.get('max_steps', 50000)
    warmup_steps = config.get('warmup_steps', 1000)
    peak_lr = opt_cfg.get('peak_lr', 1e-4)
    grad_clip = config.get('gradient_clip_val', 1.0)
    ema_update_interval = config.get('ema_update_interval', 1)
    save_interval = config.get('save_interval', 1000)
    log_interval = config.get('log_interval', 50)
    eval_interval = config.get('eval_interval', 1000)

    # -- Output directory
    output_dir = Path(config.get('output_dir', 'outputs/tdv_pretrain'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- WandB (only on rank 0)
    use_wandb = config.get('use_wandb', False) and not args.no_wandb and rank == 0
    if use_wandb:
        import wandb
        wandb.init(
            project=config.get('wandb_project', 'tdv-cholec'),
            name=config.get('run_name', 'tdv-pretrain'),
            config=config,
        )

    # -- DDP
    if args.ddp:
        if rank == 0:
            print(f"[DDP] Wrapping model with DistributedDataParallel...")
        # find_unused_parameters=False is safe here because all frozen params
        # (frame_encoder, teacher_*) have requires_grad=False and are skipped by DDP.
        # find_unused_parameters=True can cause SIGSEGV with xFormers custom kernels.
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False,
        )
        raw_model = model.module
        if rank == 0:
            print(f"[DDP] DDP wrapping complete.")
    else:
        raw_model = model

    # -- Training loop
    step = 0
    epoch = 0
    best_loss = float('inf')

    if rank == 0:
        print(f"Starting TDV pretraining for {max_steps} steps...")
        print(f"  batch_size={config.get('batch_size', 4)}, num_frames={config.get('num_frames', 4)}")
        print(f"  peak_lr={peak_lr}, warmup={warmup_steps}, grad_clip={grad_clip}")

    while step < max_steps:
        dataloader.sampler.set_epoch(epoch) if hasattr(dataloader.sampler, 'set_epoch') else None

        for batch in dataloader:
            if step >= max_steps:
                break

            # Progressive unfreezing
            if unfreeze_schedule is not None:
                progressive_unfreeze(raw_model, epoch, unfreeze_schedule)

            frame_sequences = batch.to(device)  # (B, T, C, H, W)

            # LR schedule
            lr = cosine_lr_schedule(step, max_steps, warmup_steps, peak_lr)
            for pg in optimizer.param_groups:
                pg['lr'] = lr * pg.get('lr_scale', 1.0)

            # Forward
            if step == 0 and rank == 0:
                print(f"[DDP] First forward pass starting... input shape={frame_sequences.shape}")
            outputs = model(frame_sequences)
            if step == 0 and rank == 0:
                print(f"[DDP] First forward pass complete. loss={outputs['loss'].item():.4f}")
            loss = outputs['loss']

            # L2-SP
            if l2sp_weight > 0 and pretrained_encoder_sd is not None:
                l2sp = l2sp_loss(raw_model, pretrained_encoder_sd)
                loss = loss + l2sp_weight * l2sp
                outputs['l2sp_loss'] = l2sp.detach()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # EMA update
            if step % ema_update_interval == 0:
                raw_model.ema_update()

            # Logging
            if step % log_interval == 0 and rank == 0:
                log_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                           for k, v in outputs.items() if k != 'loss'}
                log_dict['loss'] = loss.item()
                log_dict['lr'] = lr
                log_dict['step'] = step
                log_dict['epoch'] = epoch

                print(f"[step {step}/{max_steps}] loss={loss.item():.4f} lr={lr:.2e}")
                for k, v in sorted(log_dict.items()):
                    if k not in ('loss', 'lr', 'step', 'epoch'):
                        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

                if use_wandb:
                    wandb.log(log_dict, step=step)

            # Checkpoint (only rank 0)
            if step > 0 and step % save_interval == 0 and rank == 0:
                ckpt_path = output_dir / 'latest.pth.tar'
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'loss': loss.item(),
                }, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_path = output_dir / 'best.pth.tar'
                    torch.save({
                        'step': step,
                        'epoch': epoch,
                        'model_state_dict': raw_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config,
                        'loss': best_loss,
                    }, best_path)
                    print(f"  New best loss: {best_loss:.4f} → {best_path}")

            step += 1

        epoch += 1

    # -- Final checkpoint (only rank 0)
    if rank == 0:
        final_path = output_dir / 'final.pth.tar'
        torch.save({
            'step': step,
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'loss': loss.item() if 'loss' in locals() else 0.0,
        }, final_path)
        print(f"Training complete. Final checkpoint: {final_path}")

        # -- Extract frame encoder for downstream detection
        encoder_path = output_dir / 'tdv_frame_encoder.pth'
        torch.save(raw_model.get_frame_encoder_state_dict(), encoder_path)
        print(f"Frame encoder weights saved to: {encoder_path}")

    if use_wandb:
        wandb.finish()

    if args.ddp:
        torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="TDV pretraining on Cholec80")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--ddp', action='store_true', help='Enable DDP')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # -- Initialize DDP process group BEFORE building DistributedSampler
    if args.ddp:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        # Do NOT pass device_id here — PyTorch 2.7 + NCCL eager init segfaults
        # in graph/topo.cc:785 (nullptr paths). See pytorch/pytorch#146118.
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
        )
        print(f"Initialized DDP: rank={torch.distributed.get_rank()}, "
              f"world_size={torch.distributed.get_world_size()}")

    train_tdv(config, args)


if __name__ == '__main__':
    main()
