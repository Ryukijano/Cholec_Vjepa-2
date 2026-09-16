#!/usr/bin/env python3
"""
Build the leak-free SSL corpus for GOT-JEPA Stage 2 using RF-DETR pseudo-labels.

Replaces the old SurgeNetDINO-based build_ssl_corpus.py. Uses the fine-tuned
RF-DETR model to generate bounding box pseudo-annotations on Cholec80 videos
(excluding CT20 val/test overlap), then lays out a unified SSL corpus directory
that mirrors CT20 layout for MOTCholecDataset.

Usage (single GPU):
    python scripts/got_jepa/build_ssl_corpus_rfdetr.py --device cuda:0

Usage (3-GPU parallel):
    for RANK in 0 1 2; do
        python scripts/got_jepa/build_ssl_corpus_rfdetr.py \
            --device cuda:$RANK --rank $RANK --world_size 3 &
    done
    wait
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# RF-DETR class names (0-indexed in model, 1-indexed in CT20 JSON)
CLASS_NAMES = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator", "specimen_bag"]

# CT20 val/test overlap videos to EXCLUDE from SSL
EXCLUDED_VIDEOS = {"video01", "video06", "video07", "video12", "video25", "video30", "video39"}

# CT20 Training videos (VID96, VID103 are Cholec120-only extras)
CT20_TRAIN_VIDS = {"VID02", "VID04", "VID11", "VID13", "VID17", "VID23", "VID31", "VID37", "VID96", "VID103"}

_LOG = logging.getLogger("build_ssl_rfdetr")

# Default paths
DEFAULT_CKPT = "/scratch/kcwp264/Cholec_Vjepa-2/outputs/rfdetr_stage1/checkpoint_best_ema.pth"
DEFAULT_C80_ROOT = "/scratch/kcwp264/datasets_cholec/cholec80/cholec80/frames"
DEFAULT_CT20_ROOT = "/scratch/kcwp264/data/surgi_world_track/cholectrack20"
DEFAULT_OUT_ROOT = "/scratch/kcwp264/data/surgi_world_track/ssl_corpus"


def parse_frame_number(filename: str) -> int:
    """Extract trailing integer from filename like video02_000005.png or 006825.png."""
    stem = Path(filename).stem
    digits = []
    for c in reversed(stem):
        if c.isdigit():
            digits.append(c)
        else:
            break
    if not digits:
        raise ValueError(f"Cannot parse frame number from {filename}")
    return int("".join(reversed(digits)))


def collect_video_frames(video_dir: Path) -> List[Path]:
    """Return frame files sorted by frame number."""
    exts = (".png", ".jpg", ".jpeg")
    frames = [f for f in video_dir.iterdir() if f.suffix.lower() in exts]
    frames.sort(key=lambda p: parse_frame_number(p.name))
    return frames


def load_rfdetr_model(checkpoint_path: str, device: str) -> "RFDETRBase":
    """Load RF-DETR from fine-tuned checkpoint."""
    from rfdetr import RFDETRBase

    _LOG.info(f"Loading RF-DETR from {checkpoint_path}")
    model = RFDETRBase(pretrain_weights=checkpoint_path, num_classes=7)
    model.model.to(device)
    model.model.eval()
    _LOG.info(f"Model loaded on {device}")
    return model


@torch.no_grad()
def pseudo_label_video(
    model,
    frames: List[Path],
    device: str,
    score_threshold: float,
    batch_size: int = 8,
) -> Dict[str, List[Dict]]:
    """
    Run RF-DETR inference over a video and return CT20-style pseudo annotations.

    Returns dict: {frame_number_str: [{instrument, tool_bbox, intraoperative_track_id, pseudo, score}]}
    """
    annotations: Dict[str, List[Dict]] = {}
    track_id_counter = 0  # Simple per-frame tracking (no temporal tracking needed for SSL)

    for batch_start in range(0, len(frames), batch_size):
        batch_frames = frames[batch_start : batch_start + batch_size]
        batch_images = []

        for f in batch_frames:
            try:
                img = Image.open(f).convert("RGB")
            except (OSError, Exception) as e:
                _LOG.warning(f"Corrupt frame skipped: {f.name} ({e})")
                img = Image.new("RGB", (224, 224))
            batch_images.append(img)

        # RF-DETR predict accepts list of PIL images
        detections_list = model.predict(batch_images, threshold=score_threshold)

        if not isinstance(detections_list, list):
            detections_list = [detections_list]

        for frame_path, img, detections in zip(batch_frames, batch_images, detections_list):
            frame_num = parse_frame_number(frame_path.name)
            w_img, h_img = img.size
            tools_in_frame = []

            if len(detections.xyxy) > 0:
                for i in range(len(detections.xyxy)):
                    x1, y1, x2, y2 = detections.xyxy[i].tolist()
                    cls_id = int(detections.class_id[i])
                    score = float(detections.confidence[i])

                    # Convert pixel xyxy to normalized xywh (CT20 convention)
                    x = max(0.0, x1 / w_img)
                    y = max(0.0, y1 / h_img)
                    w = max(1e-4, (x2 - x1) / w_img)
                    h = max(1e-4, (y2 - y1) / h_img)

                    # Clamp to [0, 1]
                    x = min(1.0, x)
                    y = min(1.0, y)
                    w = min(1.0 - x, w)
                    h = min(1.0 - y, h)

                    track_id_counter += 1
                    tools_in_frame.append({
                        "instrument": cls_id + 1,  # 1-indexed (CT20 convention)
                        "tool_bbox": [x, y, w, h],
                        "intraoperative_track_id": track_id_counter,
                        "pseudo": True,
                        "score": score,
                    })

            if tools_in_frame:
                annotations[str(frame_num)] = tools_in_frame

    return annotations


def layout_cholec80_video(
    video_name: str,
    source_video_dir: Path,
    out_training: Path,
    annotations: Dict[str, List[Dict]],
) -> None:
    """Symlink Cholec80 frames into CT20-style layout and write pseudo JSON."""
    out_video_dir = out_training / video_name
    out_frames_dir = out_video_dir / "Frames"
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    for f in collect_video_frames(source_video_dir):
        frame_num = parse_frame_number(f.name)
        dst = out_frames_dir / f"{frame_num:06d}{f.suffix.lower()}"
        if not dst.exists():
            os.symlink(f, dst)

    json_path = out_video_dir / f"{video_name}.json"
    with open(json_path, "w") as f:
        json.dump({"annotations": annotations}, f)


def symlink_ct20_training(ct20_root: Path, out_training: Path) -> None:
    """Symlink CT20 Training videos (with real annotations) into SSL corpus."""
    ct20_train_root = ct20_root / "Training"
    for vid_name in CT20_TRAIN_VIDS:
        src = ct20_train_root / vid_name
        if not src.is_dir():
            _LOG.warning(f"CT20 training video missing: {src}")
            continue
        dst = out_training / vid_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        _LOG.info(f"CT20 train symlink: {vid_name}")


def symlink_ct20_validation(ct20_root: Path, out_root: Path) -> None:
    """Mirror CT20 Validation folder into SSL corpus."""
    src = ct20_root / "Validation"
    dst = out_root / "Validation"
    if src.exists() and not dst.exists():
        os.symlink(src, dst)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--cholec80_root", type=Path, default=Path(DEFAULT_C80_ROOT))
    parser.add_argument("--cholectrack20_root", type=Path, default=Path(DEFAULT_CT20_ROOT))
    parser.add_argument("--out_root", type=Path, default=Path(DEFAULT_OUT_ROOT))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--score_threshold", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--skip_ct20", action="store_true", help="Skip CT20 symlink setup (rank>0)")
    args = parser.parse_args()

    device = args.device
    rank = args.rank
    world_size = args.world_size

    # 1. Get list of Cholec80 videos (excluding CT20 val/test overlap)
    all_c80 = sorted(p.name for p in args.cholec80_root.iterdir() if p.is_dir())
    ssl_videos = [v for v in all_c80 if v not in EXCLUDED_VIDEOS]
    _LOG.info(f"Cholec80 SSL videos: {len(ssl_videos)} (excluded {len(EXCLUDED_VIDEOS)})")

    # Shard across ranks
    shard = [v for i, v in enumerate(ssl_videos) if i % world_size == rank]
    _LOG.info(f"[rank {rank}/{world_size}] Processing {len(shard)} videos")

    # 2. Setup output directory (only rank 0 does CT20 symlinks)
    out_training = args.out_root / "Training"
    out_training.mkdir(parents=True, exist_ok=True)

    if rank == 0 and not args.skip_ct20:
        _LOG.info("Symlinking CT20 Training videos...")
        symlink_ct20_training(args.cholectrack20_root, out_training)
        symlink_ct20_validation(args.cholectrack20_root, args.out_root)

    # 3. Load RF-DETR model
    model = load_rfdetr_model(args.checkpoint, device)

    # 4. Pseudo-label each video in shard
    for i, video_name in enumerate(shard, 1):
        src_dir = args.cholec80_root / video_name
        frames = collect_video_frames(src_dir)
        _LOG.info(f"[rank {rank}][{i}/{len(shard)}] {video_name} ({len(frames)} frames)")

        annotations = pseudo_label_video(
            model=model,
            frames=frames,
            device=device,
            score_threshold=args.score_threshold,
            batch_size=args.batch_size,
        )

        n_boxes = sum(len(v) for v in annotations.values())
        _LOG.info(f"  -> {len(annotations)} annotated frames, {n_boxes} boxes")

        layout_cholec80_video(
            video_name=video_name,
            source_video_dir=src_dir,
            out_training=out_training,
            annotations=annotations,
        )

    _LOG.info(f"[rank {rank}] Done. SSL corpus at {args.out_root}")


if __name__ == "__main__":
    main()
