"""
COCO-style detection metrics (mAP) for surgical tool detection.

Lightweight implementation that doesn't require pycocotools.
Computes mAP @ IoU=0.50:0.95 following COCO convention.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from cxcywh to xyxy format."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes (both xyxy)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def compute_coco_map(
    pred_boxes_list: List[torch.Tensor],
    pred_scores_list: List[torch.Tensor],
    pred_labels_list: List[torch.Tensor],
    gt_boxes_list: List[torch.Tensor],
    gt_labels_list: List[torch.Tensor],
    iou_thresholds: Optional[List[float]] = None,
    num_classes: int = 7,
) -> Dict[str, float]:
    """
    Compute COCO-style mAP with **per-image** matching and global TP/FP
    accumulation (proper COCO protocol).

    Args:
        pred_boxes_list:  list of (N_i, 4) predicted boxes per image (cxcywh [0,1])
        pred_scores_list: list of (N_i,) confidence scores per image
        pred_labels_list: list of (N_i,) predicted class labels per image
        gt_boxes_list:    list of (M_i, 4) ground-truth boxes per image
        gt_labels_list:   list of (M_i,) ground-truth class labels per image
        iou_thresholds:   list of IoU thresholds (default: 0.50:0.05:0.95)
        num_classes:      number of tool classes

    Returns:
        dict with 'mAP', 'mAP50', 'mAP75'
    """
    if iou_thresholds is None:
        iou_thresholds = [0.50 + 0.05 * i for i in range(10)]  # 0.50:0.05:0.95

    total_gt = sum(t.numel() for t in gt_boxes_list) // 4
    total_pred = sum(t.numel() for t in pred_boxes_list) // 4
    if total_pred == 0 or total_gt == 0:
        return {'mAP': 0.0, 'mAP50': 0.0, 'mAP75': 0.0}

    # Pre-convert all images to xyxy.
    pred_xyxy_list = [_box_cxcywh_to_xyxy(b) if b.numel() > 0 else b for b in pred_boxes_list]
    gt_xyxy_list = [_box_cxcywh_to_xyxy(b) if b.numel() > 0 else b for b in gt_boxes_list]

    aps: List[float] = []
    for iou_thr in iou_thresholds:
        ap = _compute_ap_per_image(
            pred_xyxy_list, pred_scores_list, pred_labels_list,
            gt_xyxy_list, gt_labels_list, iou_thr, num_classes,
        )
        aps.append(ap)

    mAP = float(sum(aps) / len(aps))
    return {
        'mAP': mAP,
        'mAP50': aps[0] if len(aps) > 0 else 0.0,
        'mAP75': aps[5] if len(aps) > 5 else 0.0,
    }


def _compute_ap_per_image(
    pred_xyxy_list: List[torch.Tensor],
    pred_scores_list: List[torch.Tensor],
    pred_labels_list: List[torch.Tensor],
    gt_xyxy_list: List[torch.Tensor],
    gt_labels_list: List[torch.Tensor],
    iou_threshold: float,
    num_classes: int,
) -> float:
    """
    Compute AP at a single IoU threshold with per-image matching and
    global TP/FP accumulation (COCO protocol).

    Predictions are sorted globally by score; each prediction is matched
    only against GTs in the same image.
    """
    # Build a flat global list of predictions with image index, sorted by score.
    all_preds: List[Tuple[float, int, int]] = []  # (score, img_idx, local_idx)
    for img_idx, scores in enumerate(pred_scores_list):
        if scores.numel() == 0:
            continue
        for local_idx in range(scores.size(0)):
            all_preds.append((float(scores[local_idx].item()), img_idx, local_idx))
    all_preds.sort(key=lambda x: x[0], reverse=True)

    aps: List[float] = []
    for c in range(num_classes):
        # Count total GT for this class across all images.
        total_gt_c = 0
        for gt_labels in gt_labels_list:
            if gt_labels.numel() == 0:
                continue
            total_gt_c += int((gt_labels == c).sum().item())
        if total_gt_c == 0:
            continue

        # Per-image matched-GT tracking for this class.
        matched_gt_per_img: List[torch.Tensor] = []
        for gt_labels in gt_labels_list:
            if gt_labels.numel() == 0:
                matched_gt_per_img.append(torch.zeros(0, dtype=torch.bool))
            else:
                matched_gt_per_img.append(torch.zeros(gt_labels.size(0), dtype=torch.bool))

        tp = torch.zeros(len(all_preds))
        fp = torch.zeros(len(all_preds))

        for global_p_idx, (score, img_idx, local_idx) in enumerate(all_preds):
            pred_label = int(pred_labels_list[img_idx][local_idx].item())
            if pred_label != c:
                continue
            pred_box = pred_xyxy_list[img_idx][local_idx:local_idx + 1]

            gt_labels_img = gt_labels_list[img_idx]
            if gt_labels_img.numel() == 0:
                fp[global_p_idx] = 1
                continue

            c_gt_mask = gt_labels_img == c
            c_gt_indices = torch.where(c_gt_mask)[0]
            if len(c_gt_indices) == 0:
                fp[global_p_idx] = 1
                continue

            c_gt_xyxy = gt_xyxy_list[img_idx][c_gt_indices]
            ious = _box_iou(pred_box, c_gt_xyxy)[0]  # (G_c,)
            best_iou, best_local = ious.max(dim=0)
            best_gt_idx = int(c_gt_indices[best_local].item())

            if best_iou.item() >= iou_threshold and not matched_gt_per_img[img_idx][best_gt_idx]:
                tp[global_p_idx] = 1
                matched_gt_per_img[img_idx][best_gt_idx] = True
            else:
                fp[global_p_idx] = 1

        tp_cum = tp.cumsum(dim=0)
        fp_cum = fp.cumsum(dim=0)
        recalls = tp_cum / total_gt_c
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

        ap = _interpolate_ap(recalls, precisions)
        aps.append(ap)

    if not aps:
        return 0.0
    return float(sum(aps) / len(aps))


def _interpolate_ap(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    """101-point interpolated average precision."""
    ap = 0.0
    for t in torch.linspace(0, 1, 101):
        mask = recalls >= t
        if mask.any():
            ap += float(precisions[mask].max().item())
    return ap / 101.0


def compute_map_from_detr_outputs(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    targets: List[Dict],
    score_threshold: float = 0.01,
    num_classes: int = 7,
) -> Dict[str, float]:
    """
    Compute mAP from raw DETR outputs across a batch.

    Matching is done **per-image** (COCO protocol) — a prediction in image *i*
    can only match a GT in image *i*, preventing cross-frame false matches.

    Args:
        pred_logits: (B, Q, C) class logits
        pred_boxes: (B, Q, 4) predicted boxes in cxcywh [0,1]
        targets: list of dicts with 'labels' and 'boxes'
        score_threshold: minimum score for a prediction to be considered
        num_classes: number of tool classes

    Returns:
        dict with mAP metrics
    """
    pred_boxes_list: List[torch.Tensor] = []
    pred_scores_list: List[torch.Tensor] = []
    pred_labels_list: List[torch.Tensor] = []
    gt_boxes_list: List[torch.Tensor] = []
    gt_labels_list: List[torch.Tensor] = []

    B = pred_logits.size(0)
    for i in range(B):
        scores = pred_logits[i].sigmoid()
        max_scores, labels = scores.max(dim=-1)
        keep = max_scores > score_threshold

        if keep.any():
            pred_boxes_list.append(pred_boxes[i][keep])
            pred_scores_list.append(max_scores[keep])
            pred_labels_list.append(labels[keep])
        else:
            pred_boxes_list.append(torch.zeros(0, 4))
            pred_scores_list.append(torch.zeros(0))
            pred_labels_list.append(torch.zeros(0, dtype=torch.long))

        tgt = targets[i]
        if 'boxes' in tgt and 'labels' in tgt and len(tgt['boxes']) > 0:
            gt_boxes_list.append(tgt['boxes'])
            gt_labels_list.append(tgt['labels'])
        else:
            gt_boxes_list.append(torch.zeros(0, 4))
            gt_labels_list.append(torch.zeros(0, dtype=torch.long))

    if not pred_boxes_list or not gt_boxes_list:
        return {'mAP': 0.0, 'mAP50': 0.0, 'mAP75': 0.0}

    return compute_coco_map(
        pred_boxes_list=pred_boxes_list,
        pred_scores_list=pred_scores_list,
        pred_labels_list=pred_labels_list,
        gt_boxes_list=gt_boxes_list,
        gt_labels_list=gt_labels_list,
        num_classes=num_classes,
    )
