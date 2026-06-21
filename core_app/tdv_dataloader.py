"""
Cholec80 video dataloader for TDV pretraining.

Samples consecutive frame sequences from the leak-free SSL corpus
(73 Cholec80 videos + 2 Cholec120-only videos from CT20 training).

The frames are stored as PNGs at 1fps sampling. For TDV, we sample
`num_frames` consecutive PNGs from a random video, which provides
true temporal ordering for the temporal difference objective.
"""
from __future__ import annotations

import os
import glob
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


def parse_ssl_video_list(split_yaml_path: str) -> Tuple[List[str], List[str]]:
    """Parse the SSL corpus video list from the splits YAML.
    Returns a list of video names like ['video02', 'video03', ...].
    """
    import yaml
    with open(split_yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    videos_str = cfg['ssl_corpus']['cholec80_videos']
    videos = [v.strip() for v in videos_str.split(',') if v.strip()]

    extras = cfg['ssl_corpus'].get('cholectrack20_extras', [])

    return videos, extras


class Cholec80TDVDataset(Dataset):
    """Dataset for TDV pretraining on Cholec80 frames.

    Each sample is a sequence of `num_frames` consecutive PNG frames
    from a randomly chosen video in the SSL corpus.

    Args:
        frames_root: Path to Cholec80 frames directory (e.g. .../cholec80/frames)
        video_names: List of video folder names to include (e.g. ['video02', ...])
        num_frames: Number of consecutive frames per sample (T in TDV)
        img_size: Target image size (square)
        stride: Frame stride (1 = consecutive 1fps frames, 2 = skip one, etc.)
        transform: Optional torchvision transform (if None, uses default)
        return_video_name: If True, returns (frames, video_name, start_idx)
    """

    def __init__(
        self,
        frames_root: str,
        video_names: List[str],
        num_frames: int = 4,
        img_size: int = 224,
        stride: int = 1,
        transform: Optional[T.Compose] = None,
        return_video_name: bool = False,
    ):
        self.frames_root = Path(frames_root)
        self.num_frames = num_frames
        self.img_size = img_size
        self.stride = stride
        self.return_video_name = return_video_name

        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Build index: (video_name, [sorted frame paths])
        self.video_frames: List[Tuple[str, List[str]]] = []
        for vname in video_names:
            vdir = self.frames_root / vname
            if not vdir.exists():
                continue
            frames = sorted(glob.glob(str(vdir / "*.png")))
            if len(frames) >= num_frames * stride:
                self.video_frames.append((vname, frames))

        if not self.video_frames:
            raise RuntimeError(
                f"No valid videos found in {frames_root} with {num_frames * stride}+ frames"
            )

        # Build flat index: (video_idx, start_frame_idx) for every valid starting position
        self.samples: List[Tuple[int, int]] = []
        for vi, (vname, frames) in enumerate(self.video_frames):
            max_start = len(frames) - (num_frames - 1) * stride - 1
            for si in range(0, max_start + 1, stride):
                self.samples.append((vi, si))

        print(f"Cholec80TDVDataset: {len(self.video_frames)} videos, "
              f"{len(self.samples)} sample starting positions, "
              f"num_frames={num_frames}, stride={stride}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        vi, si = self.samples[idx]
        vname, frames = self.video_frames[vi]

        # Load consecutive frames
        frame_tensors = []
        for t in range(self.num_frames):
            fpath = frames[si + t * self.stride]
            img = Image.open(fpath).convert('RGB')
            frame_tensors.append(self.transform(img))

        clip = torch.stack(frame_tensors)  # (T, C, H, W)

        if self.return_video_name:
            return clip, vname, si
        return clip


def build_tdv_dataloader(
    frames_root: str,
    video_names: List[str],
    batch_size: int = 4,
    num_frames: int = 4,
    img_size: int = 224,
    stride: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    distributed: bool = False,
) -> DataLoader:
    """Build a DataLoader for TDV pretraining on Cholec80."""
    dataset = Cholec80TDVDataset(
        frames_root=frames_root,
        video_names=video_names,
        num_frames=num_frames,
        img_size=img_size,
        stride=stride,
    )

    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
