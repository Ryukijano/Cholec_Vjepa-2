#!/usr/bin/env python3
"""
RF-DETR Stage 1 fine-tuning on CholecTrack20.

Usage:
    # Single GPU (simplest, works out of the box)
    python scripts/got_jepa/train_rfdetr_stage1.py

    # Multi-GPU DDP
    torchrun --standalone --nproc_per_node=3 scripts/got_jepa/train_rfdetr_stage1.py --ddp
"""

import os
import sys
import argparse

# Set NCCL env vars for L40S before importing torch
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_NET", "Socket")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

# Monkey-patch Lightning CSV logger to fix fieldnames crash
import csv as _csv
import lightning_fabric.loggers.csv_logs as _csv_logs

def _patched_rewrite(self, fieldnames):
    with self._fs.open(self.metrics_file_path, 'r', newline='') as f:
        metrics = list(_csv.DictReader(f))
    with self._fs.open(self.metrics_file_path, 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames, restval='', extrasaction='ignore')
        w.writeheader()
        w.writerows(metrics)

_csv_logs._ExperimentWriter._rewrite_with_new_header = _patched_rewrite

import torch
from rfdetr import RFDETRLarge
from rfdetr.datasets.aug_config import AUG_AGGRESSIVE

DATASET_DIR = "/scratch/kcwp264/data/surgi_world_track/cholec20_coco_augmented"
OUTPUT_DIR = "/scratch/kcwp264/Cholec_Vjepa-2/outputs/mot/rfdetr-large-v2"

# Surgical-specific augmentation config: aggressive + custom transforms
# Key: Copy-paste and heavy augmentation to combat 70% bipolar class imbalance
SURGICAL_AUG_CONFIG = {
    **AUG_AGGRESSIVE,
    # Surgical-specific: simulate smoke, bleeding, reflections
    "GaussianBlur": {"p": 0.15, "blur_limit": (3, 7)},
    "CLAHE": {"p": 0.2, "clip_limit": 4.0},
    "RandomBrightnessContrast": {"p": 0.3, "brightness_limit": 0.2, "contrast_limit": 0.2},
    "GaussNoise": {"p": 0.1, "var_limit": (10, 50)},
    # Geometric: simulate camera movement
    "Rotate": {"limit": 10, "p": 0.3, "border_mode": 0},
    "HorizontalFlip": {"p": 0.5},
    "RandomScale": {"scale_limit": (0.8, 1.2), "p": 0.3},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", action="store_true", help="Enable DDP multi-GPU")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)  # Lower for 704px on L40S 48GB
    parser.add_argument("--grad_accum_steps", type=int, default=4)  # Effective BS=4*4*3=48
    parser.add_argument("--lr", type=float, default=5e-5)  # Lower LR for small dataset
    parser.add_argument("--resolution", type=int, default=704)  # RFDETRLarge native resolution
    parser.add_argument("--lr_encoder", type=float, default=1e-4)
    parser.add_argument("--early_stopping", action="store_true", default=True,
                        help="Enable early stopping with patience=20")
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. last.ckpt or checkpoint_best_total.pth)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    effective_bs = args.batch_size * args.grad_accum_steps * max(num_gpus, 1)
    print(f"Dataset:    {DATASET_DIR}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Epochs:     {args.epochs}")
    print(f"GPUs:       {num_gpus}")
    print(f"Batch:      {args.batch_size} x {args.grad_accum_steps} x {num_gpus} = {effective_bs}")
    print(f"LR:         {args.lr}")
    print(f"Resolution: {args.resolution}")
    print()

    model = RFDETRLarge()

    train_kwargs = dict(
        dataset_dir=DATASET_DIR,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        resolution=args.resolution,
        output_dir=OUTPUT_DIR,
        device="cuda",
        wandb=True,
        checkpoint_interval=5,
        num_workers=4,
        aug_config=SURGICAL_AUG_CONFIG,
        warmup_epochs=5.0,
        log_per_class_metrics=True,
        lr_scheduler="cosine",
        use_ema=True,
        eval_interval=1,
    )

    # Early stopping (RF-DETR monitors validation/mAP internally)
    if args.early_stopping:
        train_kwargs["early_stopping"] = True
        train_kwargs["early_stopping_patience"] = args.early_stopping_patience

    if args.resume:
        train_kwargs["resume"] = args.resume
        print(f"Resume:     {args.resume}")

    if args.ddp or num_gpus > 1:
        # CRITICAL: devices="auto" so DDP uses all visible GPUs
        # CRITICAL: find_unused_parameters=True because RF-DETR's EMA model
        # has parameters not used in the training forward pass
        train_kwargs["devices"] = "auto"
        train_kwargs["strategy"] = "ddp_find_unused_parameters_true"

    model.train(**train_kwargs)

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
