"""
Classification / regression decoders and per-track localisation loss.

Given a per-track score map produced by ``apply_filter(omega, z)`` and
the current feature map, these modules predict (a) a refined target
classification score map and (b) an ltrb bounding-box regression at the
peak coordinate. Matches the output of the GOT-Edit / ToMP localisation
head.

Loss terms (from GOT-Edit §3.2 and Bhat et al., 2019):
  - ``L_cls``: compound hinge loss on the score map against a Gaussian
    target centred at the ground-truth bbox.
  - ``L_giou``: GIoU loss on the regressed bbox.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assoc import box_giou
from .predictor import gaussian_label_encoding


class ClsDec(nn.Module):
    """
    Classification decoder — refines the raw filter response into a
    per-pixel target score map.
    """

    def __init__(self, in_channels: int = 1, mid_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    def forward(self, score_map: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) → (B, 1, H, W)."""
        return self.net(score_map)


class RegDec(nn.Module):
    """
    Regression decoder — emits a 4-channel ltrb (left, top, right,
    bottom) offset map conditioned on the current feature map and the
    classification score.

    Matches GOT-Edit §3.2 box regression branch.
    """

    def __init__(self, feature_dim: int, mid_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feature_dim + 1, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 4, kernel_size=1),
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        cls_score: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feature_map: (B, C, H, W)
            cls_score:   (B, 1, H, W)

        Returns:
            (B, 4, H, W) ltrb offsets (in feature-map grid units).
        """
        x = torch.cat([feature_map, cls_score], dim=1)
        return F.relu(self.net(x))


def ltrb_to_bbox(
    ltrb_map: torch.Tensor,
    peak_idx: torch.Tensor,
    spatial_h: int,
    spatial_w: int,
) -> torch.Tensor:
    """
    Convert ltrb offsets at the peak coordinate into a normalised
    cxcywh bbox.

    Args:
        ltrb_map: (B, 4, H, W) predicted ltrb offsets.
        peak_idx: (B, 2) peak coordinates as (y, x) in grid units.
        spatial_h, spatial_w: feature-map spatial size.

    Returns:
        (B, 4) bboxes in normalised cxcywh.
    """
    B = ltrb_map.size(0)
    device = ltrb_map.device
    offsets = ltrb_map[torch.arange(B, device=device), :, peak_idx[:, 0], peak_idx[:, 1]]
    l, t, r, b = offsets.unbind(-1)
    x1 = (peak_idx[:, 1].float() - l) / spatial_w
    y1 = (peak_idx[:, 0].float() - t) / spatial_h
    x2 = (peak_idx[:, 1].float() + r) / spatial_w
    y2 = (peak_idx[:, 0].float() + b) / spatial_h
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    return torch.stack([cx, cy, w, h], dim=-1)


class TrackLocalizationLoss(nn.Module):
    """
    Combined classification + regression loss for per-track localisation.

    ``L_track = lambda_cls * L_cls_hinge + lambda_giou * L_giou``

    Classification uses a compound hinge loss on the raw score map
    against a Gaussian target (ToMP / DiMP convention). Regression uses
    GIoU on the predicted cxcywh bbox.
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_giou: float = 2.0,
        pos_threshold: float = 0.1,
        neg_threshold: float = 0.05,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_giou = lambda_giou
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def hinge_cls_loss(
        self,
        pred_score: torch.Tensor,
        gt_score: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compound hinge loss — penalises (a) positive region scoring
        below 1 and (b) negative region scoring above 0. Matches the
        DiMP / ToMP formulation used in GOT-Edit.

        Args:
            pred_score: (B, 1, H, W) predicted score map.
            gt_score:   (B, 1, H, W) Gaussian target in [0, 1].
        """
        pos_mask = (gt_score > self.pos_threshold).float()
        neg_mask = (gt_score <= self.neg_threshold).float()
        valid = pos_mask + neg_mask

        pos_loss = (F.relu(gt_score - pred_score) * pos_mask).pow(2)
        neg_loss = (F.relu(pred_score - gt_score) * neg_mask).pow(2)

        denom = valid.sum().clamp(min=1.0)
        return ((pos_loss + neg_loss).sum()) / denom

    @staticmethod
    def _sanitize_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
        """Clamp cxcywh to valid range so GIoU stays finite under AMP."""
        b = boxes.float()
        cx, cy, w, h = b.unbind(dim=-1)
        return torch.stack([
            cx.clamp(0.0, 1.0),
            cy.clamp(0.0, 1.0),
            w.clamp(min=1e-4, max=1.0),
            h.clamp(min=1e-4, max=1.0),
        ], dim=-1)

    def forward(
        self,
        pred_score: torch.Tensor,
        pred_bbox: torch.Tensor,
        gt_bbox: torch.Tensor,
        spatial_h: int,
        spatial_w: int,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """
        Args:
            pred_score: (B, 1, H, W) predicted score map.
            pred_bbox:  (B, 4) predicted cxcywh bbox.
            gt_bbox:    (B, 4) GT cxcywh bbox.
            spatial_h, spatial_w: feature-map size.

        Returns:
            total loss, loss_dict.
        """
        device = pred_score.device
        B = pred_score.size(0)
        if B == 0:
            zero = torch.zeros(1, device=device, requires_grad=True).squeeze()
            return zero, {'cls': 0.0, 'giou': 0.0}

        pred_score = pred_score.float()
        pred_bbox = self._sanitize_cxcywh(pred_bbox)
        gt_bbox = self._sanitize_cxcywh(gt_bbox)
        if not (
            torch.isfinite(pred_score).all()
            and torch.isfinite(pred_bbox).all()
            and torch.isfinite(gt_bbox).all()
        ):
            return None, {}

        # Build GT Gaussian heatmap.
        gt_heatmap = gaussian_label_encoding(gt_bbox, spatial_h, spatial_w).unsqueeze(1)  # (B, 1, H, W)
        cls_loss = self.hinge_cls_loss(pred_score, gt_heatmap)

        # GIoU on bbox (per-pair diagonal).
        giou_matrix = box_giou(pred_bbox, gt_bbox)                                # (B, B)
        giou_diag = giou_matrix.diagonal()
        giou_loss = (1.0 - giou_diag).mean()

        total = self.lambda_cls * cls_loss + self.lambda_giou * giou_loss
        if not torch.isfinite(total):
            return None, {}
        return total, {
            'cls': cls_loss.item(),
            'giou': giou_loss.item(),
        }
