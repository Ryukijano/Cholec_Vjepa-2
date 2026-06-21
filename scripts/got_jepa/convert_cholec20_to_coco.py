#!/usr/bin/env python3
"""
Convert CholecTrack20 annotations to COCO format for RF-DETR training.

CholecTrack20 format (per-video JSON):
  {
    "frame_number": [
      {
        "instrument": 3,  # 1-7
        "tool_bbox": [x, y, w, h],  # normalized top-left x,y,w,h
        "intraoperative_track_id": 1,
        ...
      }
    ]
  }

COCO format:
  {
    "images": [{"id": int, "file_name": str, "height": int, "width": int}],
    "annotations": [{"id": int, "image_id": int, "category_id": int,
                     "bbox": [x, y, w, h], "area": float, "iscrowd": 0}],
    "categories": [{"id": int, "name": str}]
  }

Usage:
    python scripts/convert_cholec20_to_coco.py \
        --cholec20_root /scratch/kcwp264/data/surgi_world_track/cholectrack20 \
        --output_dir /scratch/kcwp264/data/surgi_world_track/cholec20_coco \
        --img_size 854 480

Note: CholecTrack20 frames are 854x480 (original resolution).
      RF-DETR will resize them during training.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from PIL import Image

# CholecTrack20 tool classes (1-indexed in source, convert to 0-indexed for COCO)
TOOL_CLASSES = {
    1: "grasper",
    2: "bipolar",
    3: "hook",
    4: "scissors",
    5: "clipper",
    6: "irrigator",
    7: "specimen_bag",
}


def get_image_size(frame_path: Path) -> Tuple[int, int]:
    """Get (width, height) from image file."""
    with Image.open(frame_path) as img:
        return img.size  # (width, height)


def parse_cholec20_annotation(
    annot: Dict[str, Any],
    img_id: int,
    annot_id: int,
    img_width: int,
    img_height: int,
) -> Dict[str, Any]:
    """Convert a single CholecTrack20 annotation to COCO format."""
    instrument = annot["instrument"]
    category_id = instrument - 1  # 0-indexed

    # Normalized bbox -> pixel bbox
    nx, ny, nw, nh = annot["tool_bbox"]
    x = nx * img_width
    y = ny * img_height
    w = nw * img_width
    h = nh * img_height

    area = w * h

    return {
        "id": annot_id,
        "image_id": img_id,
        "category_id": category_id,
        "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
        "area": round(area, 2),
        "iscrowd": 0,
    }


def convert_video_to_coco(
    video_dir: Path,
    global_img_id_start: int,
    global_annot_id_start: int,
    known_img_size: Tuple[int, int] | None = None,
) -> Tuple[List[Dict], List[Dict], int, int]:
    """
    Convert one CholecTrack20 video to COCO images and annotations.

    Returns:
        images, annotations, next_img_id, next_annot_id
    """
    video_name = video_dir.name
    json_file = video_dir / f"{video_name}.json"

    if not json_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {json_file}")

    with open(json_file) as f:
        video_data = json.load(f)

    frames_dir = video_dir / "Frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    # Determine image size once per video (all frames same resolution)
    if known_img_size is not None:
        img_w, img_h = known_img_size
    else:
        sample_frame = next(frames_dir.glob("*.png"), None)
        if sample_frame is None:
            raise FileNotFoundError(f"No frames found in {frames_dir}")
        img_w, img_h = get_image_size(sample_frame)

    images = []
    annotations = []
    img_id = global_img_id_start
    annot_id = global_annot_id_start

    # Sort frame numbers numerically
    frame_numbers = sorted(video_data.get("annotations", {}).keys(), key=lambda x: int(x))

    for frame_num in frame_numbers:
        frame_file = frames_dir / f"{int(frame_num):06d}.png"
        if not frame_file.exists():
            # Try alternative naming
            alt_files = list(frames_dir.glob(f"*{frame_num}*.png"))
            if alt_files:
                frame_file = alt_files[0]
            else:
                print(f"  Warning: frame {frame_num} not found in {frames_dir}, skipping")
                continue

        images.append({
            "id": img_id,
            "file_name": str(frame_file.relative_to(video_dir.parent.parent)),
            "height": img_h,
            "width": img_w,
        })

        frame_annots = video_data["annotations"][frame_num]
        for annot in frame_annots:
            if "tool_bbox" not in annot or "instrument" not in annot:
                continue

            coco_annot = parse_cholec20_annotation(
                annot, img_id, annot_id, img_w, img_h
            )
            annotations.append(coco_annot)
            annot_id += 1

        img_id += 1

    return images, annotations, img_id, annot_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cholec20_root", type=Path, required=True,
                        help="Root of CholecTrack20 dataset")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Output directory for COCO JSON files")
    parser.add_argument("--train_videos", nargs="+", default=None,
                        help="List of training video names (default: CT20 train split)")
    parser.add_argument("--val_videos", nargs="+", default=None,
                        help="List of validation video names (default: CT20 val split)")
    args = parser.parse_args()

    # Default splits from splits.py
    if args.train_videos is None:
        args.train_videos = [
            'VID02', 'VID04', 'VID11', 'VID13', 'VID17',
            'VID23', 'VID31', 'VID37', 'VID96', 'VID103',
        ]
    if args.val_videos is None:
        args.val_videos = ['VID30', 'VID110']

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create categories
    categories = [
        {"id": i, "name": name}
        for i, name in TOOL_CLASSES.items()
    ]
    # Re-index to 0-based
    categories = [{"id": i, "name": name} for i, (_, name) in enumerate(TOOL_CLASSES.items())]

    # --- Training set ---
    print(f"Converting training videos: {args.train_videos}")
    train_root = args.cholec20_root / "Training"
    train_images = []
    train_annots = []
    img_id = 1
    annot_id = 1

    # CT20 frames are all 854x480 - pass known size to avoid PIL per video
    CT20_IMG_SIZE = (854, 480)

    for vid in args.train_videos:
        video_dir = train_root / vid
        if not video_dir.exists():
            print(f"  Warning: {video_dir} not found, skipping")
            continue
        print(f"  Processing {vid}...")
        imgs, anns, img_id, annot_id = convert_video_to_coco(
            video_dir, img_id, annot_id, known_img_size=CT20_IMG_SIZE
        )
        train_images.extend(imgs)
        train_annots.extend(anns)
        print(f"    -> {len(imgs)} images, {len(anns)} annotations")

    train_coco = {
        "images": train_images,
        "annotations": train_annots,
        "categories": categories,
    }

    train_path = args.output_dir / "instances_train.json"
    with open(train_path, "w") as f:
        json.dump(train_coco, f)
    print(f"\nTraining COCO: {train_path}")
    print(f"  Images: {len(train_images)}")
    print(f"  Annotations: {len(train_annots)}")

    # --- Validation set ---
    print(f"\nConverting validation videos: {args.val_videos}")
    val_root = args.cholec20_root / "Validation"
    val_images = []
    val_annots = []

    for vid in args.val_videos:
        video_dir = val_root / vid
        if not video_dir.exists():
            print(f"  Warning: {video_dir} not found, skipping")
            continue
        print(f"  Processing {vid}...")
        imgs, anns, img_id, annot_id = convert_video_to_coco(
            video_dir, img_id, annot_id, known_img_size=CT20_IMG_SIZE
        )
        val_images.extend(imgs)
        val_annots.extend(anns)
        print(f"    -> {len(imgs)} images, {len(anns)} annotations")

    val_coco = {
        "images": val_images,
        "annotations": val_annots,
        "categories": categories,
    }

    val_path = args.output_dir / "instances_val.json"
    with open(val_path, "w") as f:
        json.dump(val_coco, f)
    print(f"\nValidation COCO: {val_path}")
    print(f"  Images: {len(val_images)}")
    print(f"  Annotations: {len(val_annots)}")

    print("\nDone! You can now train RF-DETR with:")
    print(f"  --train_dataset_path {train_path}")
    print(f"  --val_dataset_path {val_path}")


if __name__ == "__main__":
    main()
