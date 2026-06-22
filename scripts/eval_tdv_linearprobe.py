#!/usr/bin/env python3
"""Linear probe evaluation of TDV-pretrained frame encoder on Cholec80 phase recognition.

Extracts CLS token features from the TDV frame encoder checkpoint and trains
a linear classifier (logistic regression) to predict surgical phase.
Reports train/val accuracy and confusion matrix.

Usage:
    python scripts/eval_tdv_linearprobe.py \
        --checkpoint outputs/tdv_pretrain/final.pth.tar \
        --frames-root /scratch/kcwp264/datasets_cholec/cholec80/cholec80/frames \
        --phase-root /scratch/kcwp264/datasets_cholec/cholec80/cholec80/phase_annotations \
        --img-size 224
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Phase names in Cholec80
PHASE_NAMES = [
    "Preparation", "CalotTriangleDissection", "ClippingCutting",
    "GallbladderDissection", "GallbladderPackaging",
    "CleaningCoagulation", "GallbladderRetraction",
]
PHASE_TO_IDX = {name: i for i, name in enumerate(PHASE_NAMES)}


class Cholec80PhaseDataset(Dataset):
    """Loads Cholec80 frames with phase labels."""

    def __init__(self, frames_root: str, phase_root: str, video_names: List[str],
                 img_size: int = 224, max_frames_per_video: int = 0):
        self.frames_root = Path(frames_root)
        self.phase_root = Path(phase_root)
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.samples: List[Tuple[str, int]] = []
        for video in video_names:
            phase_file = self.phase_root / f"{video}-phase.txt"
            video_dir = self.frames_root / video
            if not phase_file.exists() or not video_dir.exists():
                continue

            phases = self._read_phases(phase_file)
            frame_files = sorted(video_dir.glob("*.png"))
            n_frames = len(frame_files)
            n_phases = len(phases)

            # Sample uniformly across the entire video to cover all phases
            if max_frames_per_video > 0 and n_frames > max_frames_per_video:
                indices = np.linspace(0, n_frames - 1, max_frames_per_video, dtype=int)
            else:
                indices = range(n_frames)

            for i in indices:
                # Annotations are at higher fps than frames (e.g. 25fps vs 1fps).
                # Map frame index to corresponding annotation row.
                phase_idx = int(i * n_phases / n_frames)
                phase_idx = min(phase_idx, n_phases - 1)
                phase_name = phases[phase_idx]
                if phase_name in PHASE_TO_IDX:
                    self.samples.append((str(frame_files[i]), PHASE_TO_IDX[phase_name]))

    @staticmethod
    def _read_phases(phase_file: Path) -> List[str]:
        phases = []
        with open(phase_file) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    phases.append(parts[1])
        return phases

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


@torch.no_grad()
def extract_features(encoder, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Extract CLS token features from encoder."""
    encoder.eval()
    all_feats = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        out = encoder(images)  # (B, 1+N, D)
        cls_token = out[:, 0, :]  # (B, D) — CLS token
        all_feats.append(cls_token.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


def load_tdv_encoder(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """Load TDV frame encoder from checkpoint."""
    from core_app.models.tdv_model import TDVFrameEncoder

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Extract only the frame encoder weights
    encoder_sd = {}
    for k, v in state_dict.items():
        if k.startswith("frame_encoder."):
            encoder_sd[k[len("frame_encoder."):]] = v

    encoder = TDVFrameEncoder(
        model_name="dinov2_vitb14",
        img_size=224,
        freeze=True,
        pretrained=True,
    )
    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

    encoder = encoder.to(device)
    return encoder


def main():
    parser = argparse.ArgumentParser(description="Linear probe eval of TDV encoder")
    parser.add_argument("--checkpoint", required=True, help="Path to TDV checkpoint")
    parser.add_argument("--frames-root", default="/scratch/kcwp264/datasets_cholec/cholec80/cholec80/frames")
    parser.add_argument("--phase-root", default="/scratch/kcwp264/datasets_cholec/cholec80/cholec80/phase_annotations")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-frames-per-video", type=int, default=100,
                        help="Sample N frames per video (0=all). Default 100 for speed.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--C", type=float, default=1.0, help="Logistic regression regularization")
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    from core_app.models.tdv_model import TDVFrameEncoder

    # SSL-excluded videos (CT20 val/test) = our eval set
    eval_videos = ["video01", "video06", "video07", "video12", "video25", "video30", "video39"]
    # Use a subset of SSL videos for train (avoid using all 73 for speed)
    train_videos = [f"video{i:02d}" for i in range(2, 81) if f"video{i:02d}" not in eval_videos]
    # Subsample train videos for speed
    train_videos = train_videos[:20]  # 20 videos for train

    print(f"=== TDV Linear Probe Evaluation ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Train videos: {len(train_videos)} ({train_videos[:5]}...)")
    print(f"Eval videos:  {len(eval_videos)} ({eval_videos})")
    print(f"Max frames/video: {args.max_frames_per_video}")

    # Load encoder
    print(f"\nLoading TDV encoder from checkpoint...")
    encoder = load_tdv_encoder(args.checkpoint, args.device)
    print(f"  Encoder loaded. Embed dim: {encoder.embed_dim}")

    # Build datasets
    print(f"\nBuilding datasets...")
    train_ds = Cholec80PhaseDataset(
        args.frames_root, args.phase_root, train_videos,
        img_size=args.img_size, max_frames_per_video=args.max_frames_per_video,
    )
    eval_ds = Cholec80PhaseDataset(
        args.frames_root, args.phase_root, eval_videos,
        img_size=args.img_size, max_frames_per_video=args.max_frames_per_video,
    )
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Eval samples:  {len(eval_ds)}")

    if len(train_ds) == 0 or len(eval_ds) == 0:
        print("ERROR: No samples found. Check paths.")
        return

    # Extract features
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"\nExtracting train features...")
    X_train, y_train = extract_features(encoder, train_loader, args.device)
    print(f"  Train features: {X_train.shape}, labels: {y_train.shape}")

    print(f"Extracting eval features...")
    X_eval, y_eval = extract_features(encoder, eval_loader, args.device)
    print(f"  Eval features: {X_eval.shape}, labels: {y_eval.shape}")

    # Linear probe
    print(f"\nTraining logistic regression (C={args.C}, max_iter={args.max_iter})...")
    clf = LogisticRegression(
        C=args.C, max_iter=args.max_iter, random_state=42,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_train_pred = clf.predict(X_train)
    y_eval_pred = clf.predict(X_eval)

    train_acc = accuracy_score(y_train, y_train_pred)
    eval_acc = accuracy_score(y_eval, y_eval_pred)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Eval accuracy:  {eval_acc:.4f}")
    print(f"{'='*60}")

    all_labels = list(range(len(PHASE_NAMES)))
    present_names = [PHASE_NAMES[i] for i in sorted(set(y_eval.tolist()) | set(y_eval_pred.tolist()))]

    print(f"\nClassification Report (eval):")
    print(classification_report(y_eval, y_eval_pred, labels=all_labels, target_names=PHASE_NAMES, digits=4, zero_division=0))

    print(f"\nConfusion Matrix (eval):")
    cm = confusion_matrix(y_eval, y_eval_pred, labels=all_labels)
    # Print compact confusion matrix
    header = "  " + "  ".join(f"{n[:4]:>4s}" for n in PHASE_NAMES)
    print(header)
    for i, name in enumerate(PHASE_NAMES):
        row = "  ".join(f"{cm[i,j]:4d}" for j in range(len(PHASE_NAMES)))
        print(f"  {name[:8]:>8s}  {row}")

    # Also evaluate with raw DINOv2 (no TDV) for comparison
    print(f"\n{'='*60}")
    print(f"BASELINE: Raw DINOv2 (no TDV pretraining)")
    print(f"{'='*60}")
    raw_encoder = TDVFrameEncoder(
        model_name="dinov2_vitb14", img_size=224, freeze=True, pretrained=True,
    ).to(args.device)

    print(f"Extracting raw DINOv2 train features...")
    X_train_raw, _ = extract_features(raw_encoder, train_loader, args.device)
    print(f"Extracting raw DINOv2 eval features...")
    X_eval_raw, _ = extract_features(raw_encoder, eval_loader, args.device)

    clf_raw = LogisticRegression(
        C=args.C, max_iter=args.max_iter, random_state=42,
        solver="lbfgs",
    )
    clf_raw.fit(X_train_raw, y_train)
    y_eval_pred_raw = clf_raw.predict(X_eval_raw)
    eval_acc_raw = accuracy_score(y_eval, y_eval_pred_raw)

    print(f"\n  Raw DINOv2 eval accuracy:  {eval_acc_raw:.4f}")
    print(f"  TDV pretrain eval accuracy: {eval_acc:.4f}")
    print(f"  Delta: {eval_acc - eval_acc_raw:+.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
