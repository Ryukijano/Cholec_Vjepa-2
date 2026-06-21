#!/usr/bin/env python3
"""
Demo: Load and preprocess surgical videos from scratch storage.
Shows how to:
  1. Load frame sequences from Cholec80 and CholecTrack20
  2. Build video clips (B, C, T, H, W) tensors for model input
  3. Apply basic preprocessing (resize, normalize)
  4. Visualize frame statistics

Run with:
    module load miniforge/24.7.1
    conda activate surgi_world_track_cuda
    python scripts/demo_load_videos.py
"""

from pathlib import Path
import sys

# Adjust if running from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# Paths (adjust as needed)
# ---------------------------------------------------------------------------
from core_app.data.paths import resolve_cholec80_frames_root, resolve_cholectrack20_root

CHOLEC80_ROOT = resolve_cholec80_frames_root()
CT20_ROOT = resolve_cholectrack20_root() / 'Training'


def load_cholec80_frames(video_name: str, max_frames: int = 50) -> list[Image.Image]:
    """Load frames from Cholec80 (videoNN_XXXXXX.png format)."""
    video_dir = CHOLEC80_ROOT / video_name
    if not video_dir.exists():
        raise FileNotFoundError(f"{video_dir} not found")

    # Cholec80 frames: video01_000001.png, video01_000025.png, etc.
    frame_files = sorted(video_dir.glob(f"{video_name}_*.png"))
    if not frame_files:
        frame_files = sorted(video_dir.glob("*.png"))

    print(f"[Cholec80/{video_name}] Found {len(frame_files)} frames, loading first {max_frames}...")
    frames = []
    for f in frame_files[:max_frames]:
        img = Image.open(f).convert("RGB")
        frames.append(img)
    return frames


def load_ct20_frames(video_name: str, max_frames: int = 50) -> list[Image.Image]:
    """Load frames from CholecTrack20 (XXXXXX.png format in Frames/ subdir)."""
    frames_dir = CT20_ROOT / video_name / "Frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"{frames_dir} not found")

    frame_files = sorted(frames_dir.glob("*.png"))
    print(f"[CT20/{video_name}] Found {len(frame_files)} frames, loading first {max_frames}...")
    frames = []
    for f in frame_files[:max_frames]:
        img = Image.open(f).convert("RGB")
        frames.append(img)
    return frames


def frames_to_clip_tensor(
    frames: list[Image.Image],
    img_size: int = 392,
    clip_length: int = 3,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Convert a list of PIL images to a model-ready clip tensor.
    Output shape: (1, C, T, H, W) where T = clip_length
    """
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Take first clip_length frames
    selected = frames[:clip_length]
    if len(selected) < clip_length:
        # Pad by repeating last frame
        selected = selected + [selected[-1]] * (clip_length - len(selected))

    tensors = [transform(f) for f in selected]  # list of (C, H, W)
    clip = torch.stack(tensors, dim=1)  # (C, T, H, W)
    clip = clip.unsqueeze(0)  # (1, C, T, H, W)
    return clip.to(device)


def inspect_frames(frames: list[Image.Image], label: str):
    """Print basic statistics about a frame sequence."""
    print(f"\n=== {label} ===")
    print(f"  Num frames: {len(frames)}")
    print(f"  Frame size: {frames[0].size}")
    print(f"  Mode: {frames[0].mode}")

    # Compute mean brightness
    arr = np.array(frames[0])
    print(f"  Mean pixel value (frame 0): {arr.mean():.1f} (0-255)")
    print(f"  Std pixel value (frame 0): {arr.std():.1f}")


def main():
    print("=" * 60)
    print("Demo: Loading Surgical Videos from Scratch")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Cholec80
    # ------------------------------------------------------------------
    cholec_frames = load_cholec80_frames("video01", max_frames=10)
    inspect_frames(cholec_frames, "Cholec80 / video01")

    # Build a 3-frame clip tensor
    cholec_clip = frames_to_clip_tensor(cholec_frames, img_size=392, clip_length=3)
    print(f"  Clip tensor shape: {cholec_clip.shape}")
    print(f"  Clip tensor dtype: {cholec_clip.dtype}")
    print(f"  Value range: [{cholec_clip.min():.3f}, {cholec_clip.max():.3f}]")

    # ------------------------------------------------------------------
    # 2. CholecTrack20
    # ------------------------------------------------------------------
    ct20_frames = load_ct20_frames("VID02", max_frames=10)
    inspect_frames(ct20_frames, "CholecTrack20 / VID02")

    ct20_clip = frames_to_clip_tensor(ct20_frames, img_size=392, clip_length=3)
    print(f"  Clip tensor shape: {ct20_clip.shape}")
    print(f"  Value range: [{ct20_clip.min():.3f}, {ct20_clip.max():.3f}]")

    # ------------------------------------------------------------------
    # 3. Simulate sliding window (like SSL corpus build)
    # ------------------------------------------------------------------
    print("\n=== Sliding Window Simulation ===")
    window_size = 3
    step = 1
    num_windows = min(5, len(cholec_frames) - window_size + 1)
    for i in range(num_windows):
        window = cholec_frames[i : i + window_size]
        w_tensor = frames_to_clip_tensor(window, img_size=392, clip_length=window_size)
        print(f"  Window {i}: frames [{i}:{i+window_size}] -> tensor {w_tensor.shape}")

    # ------------------------------------------------------------------
    # 4. GPU test (if available)
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        print("\n=== GPU Test ===")
        device = "cuda:0"
        gpu_clip = frames_to_clip_tensor(cholec_frames[:3], img_size=392, clip_length=3, device=device)
        print(f"  GPU clip device: {gpu_clip.device}")
        print(f"  GPU clip shape: {gpu_clip.shape}")
        print("  GPU transfer: OK")
    else:
        print("\n=== No CUDA available, skipping GPU test ===")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
