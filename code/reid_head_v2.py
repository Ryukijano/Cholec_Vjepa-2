#!/usr/bin/env python3
"""
Direction-aware Re-ID head (SurgiTrack++ design).

Branches:
  - Direction branch   (temporal ROI -> trocar/operator proxy)
  - Motion branch      (temporal differences)
  - Appearance branch  (RF-DETR query features)
  - Box/Class branch   (geometry/context prior)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, 4),
        )
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          fused: [N, dim]
          gates: [N, 4] softmax weights for (dir, motion, app, box)
        """
        stacked = torch.cat([a, b, c, d], dim=-1)
        g = torch.softmax(self.gate(stacked), dim=-1)
        mix = g[:, 0:1] * a + g[:, 1:2] * b + g[:, 2:3] * c + g[:, 3:4] * d
        return self.out(mix), g


class ReIDHeadV2(nn.Module):
    """
    Direction-aware embedding head.

    Args:
      embed_dim: token dim from V-JEPA2 (1024)
      reid_dim: output embedding dim
      num_direction_bins: proxy-direction classes (e.g. left/center/right -> 3)
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        reid_dim: int = 128,
        hidden_dim: int = 512,
        grid_size: int = 14,
        num_direction_bins: int = 3,
        num_classes: int = 7,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.reid_dim = reid_dim
        self.num_direction_bins = num_direction_bins
        self.num_classes = num_classes
        # Branch curriculum / toggles
        self.enable_dir_in_fusion = True
        self.enable_box_in_fusion = False  # start conservative; Box+Class can overfit
        self.enable_app_in_fusion = False  # Disabled during training (RF-DETR features not used)
        self._last_gates: Optional[torch.Tensor] = None

        # Direction branch: temporal token sequence per ROI -> direction logits
        self.direction_temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                batch_first=True,
                dropout=0.1,
            ),
            num_layers=1,
        )
        self.direction_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, reid_dim),
        )
        self.direction_cls = nn.Linear(embed_dim, num_direction_bins)

        # Motion branch
        self.motion_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, reid_dim),
        )

        # Appearance branch (RF-DETR decoder query features, default dim 256)
        self.appearance_in = nn.Linear(256, hidden_dim)
        self.appearance_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, reid_dim),
        )

        # Box + class branch
        self.box_class_mlp = nn.Sequential(
            nn.Linear(4 + num_classes, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, reid_dim),
        )

        self.fusion = GatedFusion(reid_dim)
        self.final = nn.Sequential(
            nn.Linear(reid_dim, reid_dim),
            nn.LayerNorm(reid_dim),
        )

    def _roi_pool(self, spatial_tokens: torch.Tensor, box_cxcywh: torch.Tensor) -> torch.Tensor:
        """Pool tokens in a bbox from [N, D] spatial tokens."""
        G = self.grid_size
        cx, cy, w, h = box_cxcywh.tolist()
        x1 = int(max(0, (cx - w / 2) * G))
        y1 = int(max(0, (cy - h / 2) * G))
        x2 = int(min(G, (cx + w / 2) * G))
        y2 = int(min(G, (cy + h / 2) * G))
        if x2 <= x1:
            x2 = min(x1 + 1, G)
        if y2 <= y1:
            y2 = min(y1 + 1, G)

        indices = [gy * G + gx for gy in range(y1, y2) for gx in range(x1, x2)]
        if not indices:
            return spatial_tokens.mean(dim=0)
        idx = torch.tensor(indices, dtype=torch.long, device=spatial_tokens.device)
        return spatial_tokens[idx].mean(dim=0)

    def _to_time_spatial(self, sample_tokens: torch.Tensor) -> torch.Tensor:
        """Convert [T*N, D] or [T, N, D] into [T, N, D]."""
        N_spatial = self.grid_size * self.grid_size
        if sample_tokens.dim() == 2:
            T = sample_tokens.shape[0] // N_spatial
            D = sample_tokens.shape[1]
            return sample_tokens.view(T, N_spatial, D)
        if sample_tokens.dim() == 3:
            return sample_tokens
        raise ValueError(f"Unexpected token shape: {sample_tokens.shape}")

    def forward(
        self,
        full_tokens: torch.Tensor,
        boxes: List[torch.Tensor],
        query_features: Optional[List[torch.Tensor]] = None,
        classes: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
          full_tokens: [B, T*N, D] or [B, T, N, D]
          boxes: list of [M_i, 4] (cxcywh)
          query_features: list of [M_i, 256] aligned to boxes
          classes: list of [M_i] tool classes [0..num_classes-1]

        Returns:
          embeddings_per_image: list([M_i, reid_dim])
          direction_logits_per_image: list([M_i, num_direction_bins])
        """
        B = full_tokens.shape[0]
        device = full_tokens.device

        if query_features is None:
            query_features = [None] * B
        if classes is None:
            classes = [None] * B

        out_embs: List[torch.Tensor] = []
        out_dir_logits: List[torch.Tensor] = []

        for b in range(B):
            b_boxes = boxes[b]
            if b_boxes.numel() == 0:
                out_embs.append(torch.empty(0, self.reid_dim, device=device))
                out_dir_logits.append(torch.empty(0, self.num_direction_bins, device=device))
                continue

            sample = full_tokens[b]
            ts = self._to_time_spatial(sample)  # [T, N, D]
            T = ts.shape[0]
            M = b_boxes.shape[0]

            # Build per-detection temporal ROI sequence [M, T, D]
            roi_seq = []
            for i in range(M):
                pooled_t = [self._roi_pool(ts[t], b_boxes[i]) for t in range(T)]
                roi_seq.append(torch.stack(pooled_t, dim=0))
            roi_seq = torch.stack(roi_seq, dim=0)  # [M, T, D]

            # Direction branch
            dir_encoded = self.direction_temporal(roi_seq)  # [M, T, D]
            dir_token = dir_encoded[:, -1]  # [M, D]
            dir_feat = self.direction_proj(dir_token)
            dir_logits = self.direction_cls(dir_token)

            # Motion branch from temporal differences
            if T > 1:
                motion_tok = (roi_seq[:, -1] - roi_seq[:, -2])  # [M, D]
            else:
                motion_tok = torch.zeros_like(dir_token)
            mot_feat = self.motion_mlp(motion_tok)

            # Appearance branch from detector queries
            qf = query_features[b]
            if qf is None or qf.numel() == 0:
                qf = torch.zeros(M, 256, device=device)
            app_feat = self.appearance_mlp(self.appearance_in(qf))

            # Box + class branch
            cls_ids = classes[b]
            if cls_ids is None or cls_ids.numel() == 0:
                cls_onehot = torch.zeros(M, self.num_classes, device=device)
            else:
                cls_onehot = F.one_hot(cls_ids.long().clamp(0, self.num_classes - 1), num_classes=self.num_classes).float()
            box_cls = torch.cat([b_boxes, cls_onehot], dim=-1)
            box_feat = self.box_class_mlp(box_cls)

            # Curriculum toggles: allow training with reduced branches
            dir_for_fusion = dir_feat if self.enable_dir_in_fusion else torch.zeros_like(dir_feat)
            mot_for_fusion = mot_feat  # Always enabled (core temporal signal)
            app_for_fusion = app_feat if self.enable_app_in_fusion else torch.zeros_like(app_feat)
            box_for_fusion = box_feat if self.enable_box_in_fusion else torch.zeros_like(box_feat)

            # Get raw gates from fusion
            stacked = torch.cat([dir_for_fusion, mot_for_fusion, app_for_fusion, box_for_fusion], dim=-1)
            gate_logits = self.fusion.gate(stacked)
            gates_raw = torch.softmax(gate_logits, dim=-1)
            
            # Mask and renormalize gates based on enabled branches (prevent collapse)
            gate_mask = torch.tensor([
                float(self.enable_dir_in_fusion),
                1.0,  # motion always enabled
                float(self.enable_app_in_fusion),
                float(self.enable_box_in_fusion)
            ], device=gates_raw.device, dtype=gates_raw.dtype)
            gates_masked = gates_raw * gate_mask.unsqueeze(0)
            gates_renorm = gates_masked / (gates_masked.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Compute fusion with masked gates
            fused_mix = (
                gates_renorm[:, 0:1] * dir_for_fusion +
                gates_renorm[:, 1:2] * mot_for_fusion +
                gates_renorm[:, 2:3] * app_for_fusion +
                gates_renorm[:, 3:4] * box_for_fusion
            )
            fused = self.fusion.out(fused_mix)
            
            # Cache gates for diagnostics (detach to avoid autograd refs)
            self._last_gates = gates_renorm.detach()
            emb = F.normalize(self.final(fused), p=2, dim=-1)

            out_embs.append(emb)
            out_dir_logits.append(dir_logits)

        return out_embs, out_dir_logits


def direction_proxy_targets(boxes: List[torch.Tensor], bins: int = 3) -> List[torch.Tensor]:
    """
    Weak direction proxy from bbox center-x:
      left / center / right (or more bins).
    """
    out = []
    for b in boxes:
        if b.numel() == 0:
            out.append(torch.empty(0, dtype=torch.long, device=b.device))
            continue
        cx = b[:, 0].clamp(0.0, 1.0)
        tgt = torch.clamp((cx * bins).long(), min=0, max=bins - 1)
        out.append(tgt)
    return out
