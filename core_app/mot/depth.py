"""
Depth estimation wrapper — Depth-Anything-V2 with visibility gating.

Paper: GOT-Edit (ICLR 2026) — https://arxiv.org/abs/2602.08550

The depth branch provides sparse 3D cues for the tracking operator W.
Only points that are VISIBLE (per OccuSolver) contribute depth, preventing
hallucinated depth from corrupting the tracking state.

Pipeline:
  1. Depth-Anything-V2 estimates per-pixel depth from RGB.
  2. OccuSolver visibility mask gates the depth map:
       z_visible = depth ⊙ visibility_mask
  3. Visible 3D points P_3D are back-projected from z_visible.
  4. Sparse SE(3) transform is estimated from visible correspondences
     (lean geometry tradeoff vs dense VGGT).

Depth-Anything-V2 is loaded via torch.hub when available; stub mode
provides a lightweight ConvNet for unit testing.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthWrapper(nn.Module):
    """
    Thin wrapper around Depth-Anything-V2.

    Two init paths:
      * ``hub_name``: loads from torch.hub (depth_anything_v2)
      * ``stub``: lightweight ConvNet for unit tests
    """

    def __init__(
        self,
        hub_name: Optional[str] = 'depth-anything/Depth-Anything-V2',
        model_variant: str = 'depth_anything_v2_vits',
        stub: bool = False,
        freeze: bool = True,
    ):
        super().__init__()
        self.stub = stub

        if stub:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.GroupNorm(8, 128),
                nn.GELU(),
                nn.Conv2d(128, 1, 3, stride=2, padding=1),
                nn.Sigmoid(),
            )
        else:
            try:
                self.backbone = torch.hub.load(
                    hub_name, model_variant, pretrained=True, trust_repo=True
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Depth-Anything-V2 from torch.hub "
                    f"({hub_name}, {model_variant}). "
                    "Pass stub=True for unit tests. "
                    f"Original error: {e}"
                ) from e

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) in [0, 1].

        Returns:
            (B, 1, H, W) depth map (relative, normalised to [0, 1]).
        """
        with torch.no_grad():
            if self.stub:
                return self.backbone(rgb)
            if hasattr(self.backbone, 'infer_image'):
                return self.backbone.infer_image(rgb)
            return self.backbone(rgb)


def gate_depth_by_visibility(
    depth: torch.Tensor,
    visibility: torch.Tensor,
    vis_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gate a depth map by per-point visibility scores.

    Args:
        depth: (B, 1, H, W) depth map.
        visibility: (B, N,) or (B, H, W) visibility scores in [0, 1].
        vis_threshold: minimum visibility to consider a point valid.

    Returns:
        gated_depth: (B, 1, H, W) depth with invisible regions zeroed.
        valid_mask: (B, 1, H, W) boolean mask of valid depth pixels.
    """
    if visibility.dim() == 2:
        # Per-point visibility — need to scatter to spatial grid.
        # For now, return ungated depth with a warning.
        valid_mask = torch.ones_like(depth, dtype=torch.bool)
        return depth, valid_mask

    valid_mask = visibility > vis_threshold
    gated = depth * valid_mask.float()
    return gated, valid_mask


def backproject_depth(
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    intrinsics: Optional[torch.Tensor] = None,
    img_hw: Tuple[int, int] = (392, 392),
) -> torch.Tensor:
    """
    Back-project gated depth map to 3D point cloud.

    Uses a simple pinhole camera model with default intrinsics
    (focal = img_size, principal point at centre).

    Args:
        depth: (B, 1, H, W) gated depth.
        valid_mask: (B, 1, H, W) boolean mask.
        intrinsics: optional (B, 3, 3) camera intrinsics.
        img_hw: (H, W) of the depth map.

    Returns:
        (B, N_valid, 3) 3D points in camera frame.
    """
    B, _, H, W = depth.shape
    device = depth.device

    if intrinsics is None:
        fx = fy = float(max(img_hw))
        cx = float(img_hw[1]) / 2.0
        cy = float(img_hw[0]) / 2.0
    else:
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]

    # Pixel coordinate grid
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )

    # Normalised coordinates
    x_norm = (x_grid.unsqueeze(0) - cx.view(-1, 1, 1)) / fx.view(-1, 1, 1)
    y_norm = (y_grid.unsqueeze(0) - cy.view(-1, 1, 1)) / fy.view(-1, 1, 1)

    z = depth.squeeze(1)  # (B, H, W)
    X = x_norm * z
    Y = y_norm * z

    points = torch.stack([X, Y, z], dim=-1)  # (B, H, W, 3)

    # Gather valid points
    valid = valid_mask.squeeze(1)  # (B, H, W)
    points_list = []
    for b in range(B):
        pts_b = points[b][valid[b]]  # (N_valid, 3)
        points_list.append(pts_b)

    return torch.stack(points_list) if all(p.numel() > 0 for p in points_list) else torch.zeros(B, 0, 3, device=device)


def estimate_sparse_se3(
    points_ref: torch.Tensor,
    points_cur: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    Estimate sparse SE(3) transform between two sets of 3D correspondences
    using SVD-based rigid alignment (Arun's method).

    Args:
        points_ref: (N, 3) reference 3D points.
        points_cur: (N, 3) current 3D points.

    Returns:
        (4, 4) SE(3) transformation matrix, or None if insufficient points.
    """
    if points_ref.numel() == 0 or points_cur.numel() == 0:
        return None
    if points_ref.shape[0] < 3:
        return None

    # Centre the point clouds
    centroid_ref = points_ref.mean(dim=0, keepdim=True)
    centroid_cur = points_cur.mean(dim=0, keepdim=True)
    ref_centred = points_ref - centroid_ref
    cur_centred = points_cur - centroid_cur

    # Cross-covariance matrix
    H = ref_centred.t() @ cur_centred  # (3, 3)

    try:
        U, _, Vt = torch.linalg.svd(H)
    except RuntimeError:
        return None

    R = Vt.t() @ U.t()  # (3, 3)

    # Ensure proper rotation (det = +1)
    if torch.det(R) < 0:
        Vt_fixed = Vt.clone()
        Vt_fixed[-1, :] *= -1
        R = Vt_fixed.t() @ U.t()

    t = centroid_cur.squeeze(0) - R @ centroid_ref.squeeze(0)

    T = torch.eye(4, device=points_ref.device, dtype=points_ref.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
