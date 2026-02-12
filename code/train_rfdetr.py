#!/usr/bin/env python3
"""
Train and evaluate RF-DETR on a COCO-formatted CholecTrack dataset.

This script is intentionally standalone so detection quality can be improved
independently from joint V-JEPA2 + Re-ID training.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="[%(levelname)-8s] %(message)s")
logger = logging.getLogger("train_rfdetr")


MODEL_BY_SIZE = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "base": "RFDETRBase",
}


def _build_model(size: str):
    try:
        from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRBase
    except ImportError as exc:
        raise RuntimeError("rfdetr is not installed in the current environment.") from exc

    ctor_map = {
        "RFDETRNano": RFDETRNano,
        "RFDETRSmall": RFDETRSmall,
        "RFDETRMedium": RFDETRMedium,
        "RFDETRLarge": RFDETRLarge,
        "RFDETRBase": RFDETRBase,
    }
    return ctor_map[MODEL_BY_SIZE[size]]()


def _load_coco_annotations(coco_json: Path) -> Tuple[Dict[int, Dict], Dict[int, List[Dict]]]:
    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = {im["id"]: im for im in data.get("images", [])}
    per_image = {}
    for ann in data.get("annotations", []):
        per_image.setdefault(ann["image_id"], []).append(ann)
    return images, per_image


def _xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter + 1e-8
    return inter / denom


def evaluate_recall_at_05(model, valid_dir: Path, threshold: float = 0.3) -> Dict[str, float]:
    """Compute class-wise and overall Recall@0.5 on COCO valid split."""
    images, anns = _load_coco_annotations(valid_dir / "_annotations.coco.json")
    cat_to_tp: Dict[int, int] = {}
    cat_to_gt: Dict[int, int] = {}

    for image_id, info in tqdm(images.items(), desc="RF-DETR val recall"):
        gt = anns.get(image_id, [])
        for g in gt:
            cat = int(g["category_id"])
            cat_to_gt[cat] = cat_to_gt.get(cat, 0) + 1
            cat_to_tp.setdefault(cat, 0)

        image_path = valid_dir / info["file_name"]
        if not image_path.exists() or not gt:
            continue

        image = Image.open(image_path).convert("RGB")
        detections = model.predict(image, threshold=threshold)

        pred_boxes = np.asarray(detections.xyxy) if len(detections.xyxy) > 0 else np.zeros((0, 4), dtype=np.float32)
        pred_cls = np.asarray(detections.class_id) if len(detections.class_id) > 0 else np.zeros((0,), dtype=np.int32)
        pred_conf = np.asarray(detections.confidence) if len(detections.confidence) > 0 else np.zeros((0,), dtype=np.float32)
        order = np.argsort(-pred_conf)

        matched = set()
        for pi in order:
            pbox = pred_boxes[pi]
            pcls = int(pred_cls[pi]) + 1  # supervision class_id is zero-based
            best_iou = 0.0
            best_gi = -1
            for gi, g in enumerate(gt):
                if gi in matched or int(g["category_id"]) != pcls:
                    continue
                iou = _iou_xyxy(pbox, _xywh_to_xyxy(g["bbox"]))
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_gi >= 0 and best_iou >= 0.5:
                matched.add(best_gi)
                cat_to_tp[pcls] = cat_to_tp.get(pcls, 0) + 1

    metrics: Dict[str, float] = {}
    total_tp = 0
    total_gt = 0
    for cat in sorted(cat_to_gt.keys()):
        tp = cat_to_tp.get(cat, 0)
        gt = cat_to_gt.get(cat, 0)
        rec = tp / max(gt, 1)
        metrics[f"class_{cat}_recall@0.5"] = rec
        total_tp += tp
        total_gt += gt
    metrics["overall_recall@0.5"] = total_tp / max(total_gt, 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../data/cholec_coco")
    parser.add_argument("--output_dir", type=str, default="../outputs/rfdetr_cholec")
    parser.add_argument("--model_size", choices=sorted(MODEL_BY_SIZE.keys()), default="medium")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--skip_train", action="store_true", help="Skip fitting and only run validation.")
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging inside RF-DETR.")
    parser.add_argument("--wandb_project", type=str, default="vjepa2-rfdetr", help="W&B project name.")
    parser.add_argument("--wandb_run", type=str, default=None, help="Optional W&B run name.")
    parser.add_argument("--early_stopping", action="store_true", help="Use RF-DETR early stopping on val mAP.")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Early stopping patience in epochs (RF-DETR).")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    valid_dir = dataset_dir / "valid"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(args.model_size)

    if not args.skip_train:
        logger.info("Starting RF-DETR training with %s model", args.model_size)
        # run_test=False avoids requiring a separate test/_annotations.coco.json split
        train_kwargs = dict(
            dataset_dir=str(dataset_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            lr=args.lr,
            output_dir=str(output_dir),
            num_classes=args.num_classes,
            run_test=False,
        )
        # Optional W&B and early stopping knobs (RF-DETR exposes these in its config)
        if args.use_wandb:
            train_kwargs.update({"wandb": True, "project": args.wandb_project})
            if args.wandb_run:
                train_kwargs["run"] = args.wandb_run
        if args.early_stopping:
            train_kwargs.update(
                {
                    "early_stopping": True,
                    "early_stopping_patience": args.early_stopping_patience,
                }
            )
        model.train(**train_kwargs)
    else:
        logger.info("Skipping training as requested.")

    if not valid_dir.exists():
        logger.warning("No valid split found at %s, skipping recall evaluation.", valid_dir)
        return

    metrics = evaluate_recall_at_05(model, valid_dir, threshold=args.score_thresh)
    logger.info("Validation Recall@0.5: %.4f", metrics["overall_recall@0.5"])

    out_path = output_dir / "val_recall_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", out_path)


if __name__ == "__main__":
    main()
