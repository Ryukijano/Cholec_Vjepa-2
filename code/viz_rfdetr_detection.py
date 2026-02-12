#!/usr/bin/env python3
"""
Visualize RF-DETR detection performance on the COCO-formatted CholecTrack dataset.

Outputs:
  - Per-image GT vs prediction overlays (PNG)
  - A single per-class Recall@0.5 bar chart (PNG)

Assumes you trained RF-DETR with train_rfdetr.py and have:
  dataset_dir/
    train/_annotations.coco.json
    valid/_annotations.coco.json
  output_dir/
    checkpoint_best_total.pth  (optional)

If checkpoint_best_total.pth exists, RF-DETR will use it automatically when
you instantiate the same model size and specify pretrain_weights. Otherwise,
we just use the model as currently initialized after training.
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

from train_rfdetr import _load_coco_annotations, _iou_xyxy, _xywh_to_xyxy, MODEL_BY_SIZE, _build_model
from detection_viz import make_bbox_overlay, make_recall_bar_chart, TOOL_NAMES


logging.basicConfig(level=logging.INFO, format="[%(levelname)-8s] %(message)s")
logger = logging.getLogger("viz_rfdetr_detection")


def _load_gt_for_image(image_id: int, images: Dict[int, Dict], anns: Dict[int, List[Dict]]) -> Tuple[Dict, List[Dict]]:
    return images[image_id], anns.get(image_id, [])


def _compute_recall_and_collect_samples(
    model,
    valid_dir: Path,
    max_images: int,
    score_thresh: float,
) -> Tuple[Dict[int, int], Dict[int, int], List[Tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Returns:
      cat_to_gt, cat_to_tp, list of (image, gt_boxes_cxcywh, gt_labels, pred_boxes_cxcywh, pred_labels, pred_scores)
    """
    images, anns = _load_coco_annotations(valid_dir / "_annotations.coco.json")
    cat_to_gt: Dict[int, int] = {}
    cat_to_tp: Dict[int, int] = {}
    samples = []

    # Map COCO cat ids (1..7) to 0..6 internal indices
    for image_id, info in tqdm(images.items(), desc="RF-DETR viz", total=len(images)):
        gt = anns.get(image_id, [])
        if not gt:
            continue

        for g in gt:
            cat = int(g["category_id"])
            cat_to_gt[cat] = cat_to_gt.get(cat, 0) + 1
            cat_to_tp.setdefault(cat, 0)

        image_path = valid_dir / info["file_name"]
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        detections = model.predict(image, threshold=score_thresh)

        if len(detections.xyxy) == 0:
            if len(samples) < max_images:
                # even if empty preds, keep sample for overlays
                gt_boxes_cxcywh = np.asarray([g["bbox"] for g in gt], dtype=np.float32)
                gt_boxes_cxcywh[:, 0] += gt_boxes_cxcywh[:, 2] * 0.5
                gt_boxes_cxcywh[:, 1] += gt_boxes_cxcywh[:, 3] * 0.5
                gt_labels = np.asarray([int(g["category_id"]) - 1 for g in gt], dtype=np.int64)
                samples.append(
                    (
                        image.copy(),
                        gt_boxes_cxcywh,
                        gt_labels,
                        np.zeros((0, 4), dtype=np.float32),
                        np.zeros((0,), dtype=np.int64),
                        np.zeros((0,), dtype=np.float32),
                    )
                )
            continue

        pred_boxes_xyxy = np.asarray(detections.xyxy, dtype=np.float32)
        pred_cls = np.asarray(detections.class_id, dtype=np.int64)
        pred_conf = np.asarray(detections.confidence, dtype=np.float32)

        # Convert GT xywh (pixels) to xyxy
        gt_boxes_xywh = np.asarray([g["bbox"] for g in gt], dtype=np.float32)
        gt_boxes_xyxy = np.stack([_xywh_to_xyxy(b) for b in gt_boxes_xywh], axis=0)
        gt_cats = np.asarray([int(g["category_id"]) for g in gt], dtype=np.int64)

        # Compute TP for recall
        order = np.argsort(-pred_conf)
        matched = set()
        for pi in order:
            pbox = pred_boxes_xyxy[pi]
            pcls_cat = int(pred_cls[pi]) + 1  # supervision detections are 0-based classes
            best_iou = 0.0
            best_gi = -1
            for gi, gbox in enumerate(gt_boxes_xyxy):
                if gi in matched or gt_cats[gi] != pcls_cat:
                    continue
                iou = _iou_xyxy(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_gi >= 0 and best_iou >= 0.5:
                matched.add(best_gi)
                cat_to_tp[pcls_cat] = cat_to_tp.get(pcls_cat, 0) + 1

        # Save a subset of images with overlays
        if len(samples) < max_images:
            # Convert GT and preds to normalized cxcywh in [0,1] for make_bbox_overlay
            W, H = image.size
            gt_boxes_cxcywh = gt_boxes_xywh.copy()
            gt_boxes_cxcywh[:, 0] = (gt_boxes_xywh[:, 0] + 0.5 * gt_boxes_xywh[:, 2]) / W
            gt_boxes_cxcywh[:, 1] = (gt_boxes_xywh[:, 1] + 0.5 * gt_boxes_xywh[:, 3]) / H
            gt_boxes_cxcywh[:, 2] = gt_boxes_xywh[:, 2] / W
            gt_boxes_cxcywh[:, 3] = gt_boxes_xywh[:, 3] / H
            gt_labels = gt_cats - 1  # back to 0..6

            pred_boxes_cxcywh = np.zeros_like(pred_boxes_xyxy)
            pred_boxes_cxcywh[:, 0] = (pred_boxes_xyxy[:, 0] + pred_boxes_xyxy[:, 2]) * 0.5 / W
            pred_boxes_cxcywh[:, 1] = (pred_boxes_xyxy[:, 1] + pred_boxes_xyxy[:, 3]) * 0.5 / H
            pred_boxes_cxcywh[:, 2] = (pred_boxes_xyxy[:, 2] - pred_boxes_xyxy[:, 0]) / W
            pred_boxes_cxcywh[:, 3] = (pred_boxes_xyxy[:, 3] - pred_boxes_xyxy[:, 1]) / H
            pred_labels = pred_cls

            samples.append(
                (
                    image.copy(),
                    gt_boxes_cxcywh,
                    gt_labels,
                    pred_boxes_cxcywh,
                    pred_labels,
                    pred_conf,
                )
            )

    return cat_to_gt, cat_to_tp, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../data/cholec_coco")
    parser.add_argument("--output_dir", type=str, default="../outputs/rfd_eval")
    parser.add_argument("--model_size", choices=sorted(MODEL_BY_SIZE.keys()), default="small")
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--max_images", type=int, default=32)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    valid_dir = dataset_dir / "valid"
    out_dir = Path(args.output_dir)
    viz_dir = out_dir / "viz_rfdetr"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Build model (uses pre-trained RF-DETR weights; if you ran train_rfdetr.py already
    # in this output_dir, RF-DETR has been fine-tuned in-place in that run.)
    model = _build_model(args.model_size)

    if not valid_dir.exists():
        logger.error("Validation directory %s does not exist.", valid_dir)
        return

    logger.info("Collecting RF-DETR predictions and recall stats...")
    cat_to_gt, cat_to_tp, samples = _compute_recall_and_collect_samples(
        model, valid_dir, max_images=args.max_images, score_thresh=args.score_thresh
    )

    # Per-class recall chart
    num_classes = len(TOOL_NAMES)
    per_class_recall = np.zeros(num_classes, dtype=np.float32)
    per_class_gt = np.zeros(num_classes, dtype=np.int32)
    total_tp, total_gt = 0, 0
    for cat in range(1, num_classes + 1):
        tp = cat_to_tp.get(cat, 0)
        gt = cat_to_gt.get(cat, 0)
        total_tp += tp
        total_gt += gt
        if gt > 0:
            per_class_recall[cat - 1] = tp / gt
            per_class_gt[cat - 1] = gt
    overall = total_tp / max(total_gt, 1)

    logger.info("Overall Recall@0.5 (RF-DETR): %.4f", overall)

    bar_img = make_recall_bar_chart(
        per_class_recall=per_class_recall,
        per_class_gt=per_class_gt,
        overall_recall=overall,
        epoch=0,
        num_classes=num_classes,
    )
    Image.fromarray(bar_img).save(viz_dir / "per_class_recall_rfdetr.png")

    # Per-image overlays
    for idx, (img, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores) in enumerate(samples):
        frame = np.asarray(img, dtype=np.float32) / 255.0
        frame_t = torch.from_numpy(frame).permute(2, 0, 1)  # [C,H,W]
        overlay = make_bbox_overlay(
            frame_t,
            torch.from_numpy(gt_boxes),
            torch.from_numpy(gt_labels),
            torch.from_numpy(pred_boxes),
            torch.from_numpy(pred_labels),
            torch.from_numpy(pred_scores),
            num_classes=num_classes,
            score_thresh=args.score_thresh,
        )
        Image.fromarray(overlay).save(viz_dir / f"overlay_{idx:03d}.png")

    logger.info("Saved RF-DETR visualizations to %s", viz_dir)


if __name__ == "__main__":
    import torch  # needed for make_bbox_overlay inputs

    main()

