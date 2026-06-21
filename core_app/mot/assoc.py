"""
Hungarian association for MOT.

Computes the 4-term cost matrix from the research doc:

    cost[i, j] = w_iou  * (1 - IoU(det_i, track_j))
               + w_reid * (1 - cos(emb_i, mem_j))
               + w_cls  * (1 - one_hot_match(cls_i, cls_j))
               + w_vis  * (1 - vis_j)

Default weights follow docs/multi_object_tracking_research.md:
  (0.30, 0.45, 0.15, 0.10)

and uses ``scipy.optimize.linear_sum_assignment`` for the matching.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .track import Track


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Pairwise IoU between two sets of cxcywh boxes.

    Args:
        boxes_a: (N, 4)
        boxes_b: (M, 4)

    Returns:
        (N, M) IoU matrix.
    """
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.zeros(boxes_a.size(0), boxes_b.size(0), device=boxes_a.device)

    a = box_cxcywh_to_xyxy(boxes_a)
    b = box_cxcywh_to_xyxy(boxes_b)

    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    lt = torch.max(a[:, None, :2], b[None, :, :2])  # (N, M, 2)
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)                      # (N, M, 2)
    inter = wh[..., 0] * wh[..., 1]                  # (N, M)

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def box_giou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Pairwise GIoU between two sets of cxcywh boxes.

    GIoU(A, B) = IoU(A, B) - (|C \\ (A ∪ B)| / |C|)

    where C is the smallest enclosing box.
    """
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.zeros(boxes_a.size(0), boxes_b.size(0), device=boxes_a.device)

    a = box_cxcywh_to_xyxy(boxes_a)
    b = box_cxcywh_to_xyxy(boxes_b)

    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    # Enclosing box.
    lt_e = torch.min(a[:, None, :2], b[None, :, :2])
    rb_e = torch.max(a[:, None, 2:], b[None, :, 2:])
    wh_e = (rb_e - lt_e).clamp(min=0)
    area_e = wh_e[..., 0] * wh_e[..., 1]

    return iou - (area_e - union) / area_e.clamp(min=1e-6)


def compute_cost_matrix(
    det_boxes: torch.Tensor,
    det_embeddings: torch.Tensor,
    det_classes: torch.Tensor,
    tracks: List[Track],
    w_iou: float = 0.30,
    w_reid: float = 0.45,
    w_cls: float = 0.15,
    w_vis: float = 0.10,
    large_cost: float = 1.0,
) -> torch.Tensor:
    """
    Build the 4-term cost matrix between detections and active tracks.

    Args:
        det_boxes:      (N, 4) cxcywh normalised detection boxes.
        det_embeddings: (N, D) L2-normalised detection embeddings.
        det_classes:    (N,) predicted detection classes.
        tracks:         list of ``Track`` objects (M of them).
        w_iou, w_reid, w_cls, w_vis: cost weights (sum to 1).
        large_cost:     cost assigned when a component is undefined
                        (e.g. track has no embedding yet).

    Returns:
        (N, M) cost matrix in [0, 1].
    """
    N = det_boxes.size(0)
    M = len(tracks)
    device = det_boxes.device

    if N == 0 or M == 0:
        return torch.zeros(N, M, device=device)

    # Track boxes / classes / embeddings (may have None for new tracks).
    track_boxes = torch.stack([t.bbox for t in tracks]).to(device)
    track_cls = torch.tensor([t.cls for t in tracks], device=device)
    track_vis = torch.tensor([t.visibility for t in tracks], device=device)

    iou = box_iou(det_boxes, track_boxes)                        # (N, M)
    iou_cost = 1.0 - iou

    cls_match = (det_classes.view(-1, 1) == track_cls.view(1, -1)).float()
    cls_cost = 1.0 - cls_match

    vis_cost = (1.0 - track_vis).unsqueeze(0).expand(N, M)       # (N, M)

    # ReID cost: 1 - cosine. Any track without memory gets ``large_cost``.
    reid_cost = torch.full((N, M), large_cost, device=device)
    has_mem = [t.mem_embedding is not None for t in tracks]
    if any(has_mem) and det_embeddings.numel() > 0:
        mem_stack = torch.stack(
            [t.mem_embedding if t.mem_embedding is not None
             else torch.zeros_like(det_embeddings[0])
             for t in tracks]
        ).to(device)
        cos_sim = det_embeddings @ mem_stack.t()                  # (N, M)
        r = 1.0 - cos_sim
        mask = torch.tensor(has_mem, device=device).view(1, M).expand(N, M)
        reid_cost = torch.where(mask, r, reid_cost)

    cost = (
        w_iou * iou_cost
        + w_reid * reid_cost
        + w_cls * cls_cost
        + w_vis * vis_cost
    )
    return cost


def hungarian_match(
    cost: torch.Tensor,
    threshold: float = 0.7,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Hungarian assignment with a rejection threshold on cost.

    Args:
        cost: (N, M) cost matrix.
        threshold: pairs with cost > threshold are rejected.

    Returns:
        matches, unmatched_dets, unmatched_tracks.
    """
    N, M = cost.shape
    if N == 0 or M == 0:
        return [], list(range(N)), list(range(M))

    # scipy is the right tool for Hungarian; optional but standard.
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:
        raise RuntimeError(
            "scipy is required for Hungarian matching. Install via `pip install scipy`."
        ) from e

    cost_cpu = cost.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_cpu)

    matches: List[Tuple[int, int]] = []
    unmatched_dets = set(range(N))
    unmatched_tracks = set(range(M))

    for r, c in zip(row_ind, col_ind):
        if cost_cpu[r, c] < threshold:
            matches.append((int(r), int(c)))
            unmatched_dets.discard(int(r))
            unmatched_tracks.discard(int(c))

    return matches, sorted(unmatched_dets), sorted(unmatched_tracks)
