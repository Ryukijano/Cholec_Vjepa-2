#!/usr/bin/env python3
"""
Demo: Video preprocessing for surgical tool tracking.
Inspired by CoTracker3's video loading pipeline.

Run with:
    module load miniforge/24.7.1
    conda activate surgi_world_track_cuda
    python scripts/demo_video_preprocessing.py
"""

from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import time

VIDEO_DIR = Path("/scratch/kcwp264/data/surgi_world_track/cholectrack20/Training/VID02/Frames")


def load_video_eager(video_dir: Path, img_size: int = 256, max_frames: int | None = None):
    frame_files = sorted(video_dir.glob("*.png"))
    if max_frames:
        frame_files = frame_files[:max_frames]
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    frames = [transform(Image.open(f).convert("RGB")) for f in frame_files]
    return torch.stack(frames, dim=0)  # (T, C, H, W)


def random_temporal_crop(video: torch.Tensor, clip_length: int) -> torch.Tensor:
    T_total = video.shape[0]
    if T_total <= clip_length:
        return video
    start = np.random.randint(0, T_total - clip_length + 1)
    return video[start : start + clip_length]


def normalize_for_model(video: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    normalized = (video - mean) / std  # (T, C, H, W)
    return normalized.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)


def main():
    print("=" * 60)
    print("Demo: Video Preprocessing")
    print("=" * 60)

    # Eager load
    t0 = time.time()
    video = load_video_eager(VIDEO_DIR, img_size=256, max_frames=100)
    t1 = time.time()
    mem_mb = video.numel() * 4 / 1e6
    print(f"\nEager load 100 frames @ 256x256:")
    print(f"  Shape: {video.shape} (T, C, H, W)")
    print(f"  Time: {t1-t0:.3f}s | Memory: {mem_mb:.1f} MB")

    # Temporal crop
    clip = random_temporal_crop(video, clip_length=8)
    print(f"\nRandom 8-frame clip: {clip.shape}")

    # Normalize for model
    model_input = normalize_for_model(clip)
    print(f"Model input: {model_input.shape} (B, C, T, H, W)")
    print(f"Value range: [{model_input.min():.3f}, {model_input.max():.3f}]")

    # GPU test
    if torch.cuda.is_available():
        gpu_input = model_input.cuda()
        print(f"\nGPU tensor: {gpu_input.device}, shape: {gpu_input.shape}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
