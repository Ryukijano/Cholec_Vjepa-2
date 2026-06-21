"""
2D Rotary Position Embeddings (RoPE) for spatial transformer attention.

Based on the RoPE formulation from Su et al. (2021) and adapted for 2D
spatial grids as used in jepa-wms and DINOv3.

Usage:
    rope = RoPE2D(dim=64, max_hw=64)
    q_rope = rope(q, h_indices, w_indices)  # apply to queries
    k_rope = rope(k, h_indices, w_indices)  # apply to keys
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RoPE2D(nn.Module):
    """
    2D Rotary Position Embedding.

    Splits the head dimension in half: first half encodes row (y) position,
    second half encodes column (x) position.  This preserves the 2D spatial
    structure in the attention computation.
    """

    def __init__(self, dim: int, max_hw: int = 64, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_hw = max_hw
        half = dim // 4  # split into y and x, each using half of half
        if half < 1:
            raise ValueError(f"RoPE2D dim must be >= 4, got {dim}")

        # Precompute frequencies for y and x dimensions
        theta = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
        positions = torch.arange(max_hw, dtype=torch.float32)
        freqs = torch.outer(positions, theta)  # (max_hw, half)
        self.register_buffer("freqs_cos", freqs.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        h_idx: torch.Tensor,
        w_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply 2D RoPE to input tensor.

        Args:
            x: (B, N, dim) or (B, H, N_q, dim) — queries or keys.
            h_idx: (N,) or (B, N) row indices in [0, max_hw).
            w_idx: (N,) or (B, N) column indices in [0, max_hw).

        Returns:
            Tensor of same shape as x with RoPE applied.
        """
        half = self.dim // 4
        if half < 1:
            return x

        # Reshape: split dim into (y_half, x_half, y_half, x_half)
        # First half of dim → y, second half → x
        x_reshaped = x.reshape(*x.shape[:-1], 4, half)

        # y component (first and third quarters)
        x_y = torch.cat([x_reshaped[..., 0, :], x_reshaped[..., 2, :]], dim=-1)
        # x component (second and fourth quarters)
        x_x = torch.cat([x_reshaped[..., 1, :], x_reshaped[..., 3, :]], dim=-1)

        # Lookup frequencies
        cos_y = self.freqs_cos[h_idx.clamp(0, self.max_hw - 1)]  # (N, half)
        sin_y = self.freqs_sin[h_idx.clamp(0, self.max_hw - 1)]
        cos_x = self.freqs_cos[w_idx.clamp(0, self.max_hw - 1)]
        sin_x = self.freqs_sin[w_idx.clamp(0, self.max_hw - 1)]

        # Broadcast to match x shape
        while cos_y.dim() < x_y.dim():
            cos_y = cos_y.unsqueeze(0)
            sin_y = sin_y.unsqueeze(0)
            cos_x = cos_x.unsqueeze(0)
            sin_x = sin_x.unsqueeze(0)

        # Apply rotation: x' = x*cos + rotate(x)*sin
        x_y_rot = _rotate_half(x_y)
        x_x_rot = _rotate_half(x_x)

        x_y_out = x_y * cos_y + x_y_rot * sin_y
        x_x_out = x_x * cos_x + x_x_rot * sin_x

        # Reassemble
        half_dim = x_y_out.shape[-1] // 2
        out = torch.stack([
            x_y_out[..., :half_dim],
            x_x_out[..., :half_dim],
            x_y_out[..., half_dim:],
            x_x_out[..., half_dim:],
        ], dim=-2).reshape(x.shape)

        return out


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input: (x1, x2) → (-x2, x1)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def build_2d_position_indices(
    hw: int, batch_size: int = 1, device: torch.device = torch.device("cpu")
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build row and column indices for a square spatial grid.

    Returns:
        h_idx: (B, hw*hw) row indices
        w_idx: (B, hw*hw) column indices
    """
    h = torch.arange(hw, device=device)
    w = torch.arange(hw, device=device)
    h_grid, w_grid = torch.meshgrid(h, w, indexing="ij")
    h_idx = h_grid.flatten().unsqueeze(0).expand(batch_size, -1)
    w_idx = w_grid.flatten().unsqueeze(0).expand(batch_size, -1)
    return h_idx, w_idx
