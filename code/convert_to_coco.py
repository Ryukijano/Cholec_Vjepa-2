#!/usr/bin/env python3
"""
Convert CholecTrack-style annotations to COCO format for RF-DETR training.

Expected input tree:
  <dataset_root>/
    Training/VIDxx/{Frames/*.png, *.json}
    Validation/VIDxx/{Frames/*.png, *.json}
    Test/VIDxx/{Frames/*.png, *.json}

Output tree:
  <output_root>/
    train/{images..., _annotations.coco.json}
    valid/{images..., _annotations.coco.json}
    test/{images..., _annotations.coco.json}
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TOOL_NAMES = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag",
]


logging.basicConfig(level=logging.INFO, format="[%(levelname)-8s] %(message)s")
logger = logging.getLogger("convert_to_coco")


@dataclass
class COCOWriterState:
    images: List[Dict]
    annotations: List[Dict]
    image_id: int = 1
    annotation_id: int = 1


def _safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            shutil.copy2(src, dst)
            return

    # mode == "auto"
    try:
        dst.hardlink_to(src)
        return
    except OSError:
        shutil.copy2(src, dst)


def _clip_bbox_to_unit_xywh(x: float, y: float, w: float, h: float) -> Optional[Tuple[float, float, float, float]]:
    x1 = max(0.0, min(1.0, x))
    y1 = max(0.0, min(1.0, y))
    x2 = max(0.0, min(1.0, x + w))
    y2 = max(0.0, min(1.0, y + h))
    ww = x2 - x1
    hh = y2 - y1
    if ww <= 1e-6 or hh <= 1e-6:
        return None
    return x1, y1, ww, hh


def _read_video_json(video_dir: Path) -> Optional[Dict]:
    json_files = sorted(video_dir.glob("*.json"))
    if not json_files:
        return None
    with open(json_files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_split(split_dir: Path, output_split_dir: Path, link_mode: str, frame_size: int) -> Dict:
    state = COCOWriterState(images=[], annotations=[])
    videos = sorted([d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("VID")])
    missing_frames = 0

    for video_dir in videos:
        data = _read_video_json(video_dir)
        if data is None:
            logger.warning("Skipping %s (no json file)", video_dir.name)
            continue

        anns = data.get("annotations", {})
        frames_dir = video_dir / "Frames"
        if not frames_dir.exists():
            logger.warning("Skipping %s (no Frames dir)", video_dir.name)
            continue

        for fid_str in sorted(anns.keys(), key=lambda k: int(k)):
            fid = int(fid_str)
            src_img = frames_dir / f"{fid:06d}.png"
            if not src_img.exists():
                missing_frames += 1
                continue

            dst_name = f"{video_dir.name}_{fid:06d}.png"
            dst_img = output_split_dir / dst_name
            _safe_link_or_copy(src_img, dst_img, link_mode)

            image_id = state.image_id
            state.image_id += 1

            state.images.append(
                {
                    "id": image_id,
                    "file_name": dst_name,
                    "width": frame_size,
                    "height": frame_size,
                }
            )

            frame_anns = anns[fid_str]
            for ann in frame_anns:
                cls = ann.get("instrument", -1)
                bbox = ann.get("tool_bbox")
                if cls is None or cls < 0 or cls >= len(TOOL_NAMES) or bbox is None or len(bbox) != 4:
                    continue

                clipped = _clip_bbox_to_unit_xywh(
                    float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                )
                if clipped is None:
                    continue
                x, y, w, h = clipped

                # COCO uses absolute pixels and [x, y, w, h]
                x *= frame_size
                y *= frame_size
                w *= frame_size
                h *= frame_size

                state.annotations.append(
                    {
                        "id": state.annotation_id,
                        "image_id": image_id,
                        "category_id": int(cls) + 1,  # COCO category IDs start at 1
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                state.annotation_id += 1

    coco = {
        "info": {"description": "CholecTrack20 converted to COCO"},
        "licenses": [],
        "images": state.images,
        "annotations": state.annotations,
        "categories": [{"id": i + 1, "name": name, "supercategory": "tool"} for i, name in enumerate(TOOL_NAMES)],
    }

    if missing_frames > 0:
        logger.warning("Skipped %d annotations due to missing frame files in %s", missing_frames, split_dir.name)
    return coco


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="../cholec_dataset",
        help="Root containing Training/Validation/Test directories.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="../data/cholec_coco",
        help="Output COCO directory.",
    )
    parser.add_argument(
        "--frame_size",
        type=int,
        default=224,
        help="Output width/height metadata (boxes are scaled to this size).",
    )
    parser.add_argument(
        "--link_mode",
        type=str,
        choices=["auto", "hardlink", "copy"],
        default="auto",
        help="How to materialize images in output directories.",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    split_map = {
        "Training": "train",
        "Validation": "valid",
        "Test": "test",
    }

    for in_name, out_name in split_map.items():
        split_dir = root / in_name
        if not split_dir.exists():
            logger.warning("Split %s not found, skipping.", split_dir)
            continue
        out_split_dir = out_root / out_name
        out_split_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Converting %s -> %s", split_dir, out_split_dir)
        coco = _parse_split(split_dir, out_split_dir, args.link_mode, args.frame_size)
        out_json = out_split_dir / "_annotations.coco.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(coco, f)
        logger.info(
            "Wrote %s (images=%d, annotations=%d)",
            out_json,
            len(coco["images"]),
            len(coco["annotations"]),
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
