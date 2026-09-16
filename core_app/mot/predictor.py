"""
Per-track model predictor (hypernetwork) — ToMP-style encoder-decoder.

Given reference features, per-track label encodings, and the current
frame features, this module predicts a per-track filter weight vector
``omega ∈ R^C`` that is convolved over the current feature map to
produce a target score map for localisation.

The network is a transformer encoder-decoder mirroring ToMP (Mayer et
al., CVPR 2022) and the GOT-Edit / GOT-JEPA "Model Predictor" module:

  * Encoder: self-attention over (reference features + label encodings
    + current features) for cross-frame feature interaction.
  * Decoder: a learned foreground-embedding query attends to the
    encoded tokens and emits ``omega``.

The same shared weights serve all active tracks. The per-track state
lives only in the input (reference + label encoding). This keeps the
memory footprint flat regardless of track count — a key property for
multi-object adaptation.

References:
  * ToMP: https://arxiv.org/abs/2203.11192
  * GOT-Edit §3.2 Eq. 6–8: https://arxiv.org/abs/2602.08550
  * GOT-JEPA §III-B: https://arxiv.org/abs/2602.14771
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_label_encoding(
    boxes_cxcywh: torch.Tensor,
    spatial_h: int,
    spatial_w: int,
    sigma_scale: float = 0.25,
) -> torch.Tensor:
    """
    Build a 2-D Gaussian heatmap for each box (ToMP-style label encoding).

    Args:
        boxes_cxcywh: (K, 4) normalised boxes.
        spatial_h, spatial_w: feature-map spatial size.
        sigma_scale: Gaussian std scaled by box size
                     (sigma = sigma_scale * min(w, h)).

    Returns:
        (K, spatial_h, spatial_w) heatmaps in [0, 1].
    """
    K = boxes_cxcywh.size(0)
    device = boxes_cxcywh.device
    if K == 0:
        return torch.zeros(0, spatial_h, spatial_w, device=device)

    cx = boxes_cxcywh[:, 0] * spatial_w
    cy = boxes_cxcywh[:, 1] * spatial_h
    w = boxes_cxcywh[:, 2] * spatial_w
    h = boxes_cxcywh[:, 3] * spatial_h

    sigma = (sigma_scale * torch.minimum(w, h)).clamp(min=1.0)  # (K,)

    yy, xx = torch.meshgrid(
        torch.arange(spatial_h, device=device, dtype=torch.float32),
        torch.arange(spatial_w, device=device, dtype=torch.float32),
        indexing='ij',
    )
    # (K, H, W)
    dx = xx.unsqueeze(0) - cx.view(K, 1, 1)
    dy = yy.unsqueeze(0) - cy.view(K, 1, 1)
    sigma_k = sigma.view(K, 1, 1)
    heatmap = torch.exp(-(dx.pow(2) + dy.pow(2)) / (2.0 * sigma_k.pow(2) + 1e-6))
    return heatmap  # (K, H, W)


class _MLP(nn.Module):
    """Simple 2-layer MLP."""

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PosEmbedder(nn.Module):
    """Learned 2-D positional embedding for flat (H*W, C) feature maps."""

    def __init__(self, max_hw: int = 256, dim: int = 256):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_hw * max_hw, dim) * 0.02)
        self.max_hw = max_hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            # Interpolate along the token axis as a fallback.
            pe = F.interpolate(
                self.pe.transpose(1, 2),
                size=x.size(1),
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)
        else:
            pe = self.pe[:, : x.size(1)]
        return x + pe


class PerTrackModelPredictor(nn.Module):
    """
    Transformer encoder-decoder that emits per-track filter weights.

    Forward signature (per track):
      * ``reference_features``: (B, N_ref, C) — encoder features at the
        two reference frames, concatenated along token dim.
      * ``label_encoding``:     (B, N_ref, 1) — Gaussian heatmap flattened
        and broadcast across reference tokens.
      * ``current_features``:   (B, N_cur, C) — encoder features at the
        current frame.

    Returns:
      * ``omega``: (B, C) filter-weight vector.
      * ``encoded_tokens``: (B, N_ref + N_cur, C) — handed off to
        downstream losses / regularisers (e.g. Expander in GOT-JEPA).
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_hw: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Label-encoding embedding: scalar heatmap value → dim.
        self.label_embed = _MLP(1, dim // 2, dim, dropout=dropout)

        # Learned foreground embedding (the "decoder query").
        self.fg_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Reference / current position embeddings.
        self.pos_embed = _PosEmbedder(max_hw=max_hw, dim=dim)

        # Frame-type embedding (ref vs cur vs fg-query).
        self.frame_type = nn.Embedding(3, dim)

        # Encoder: self-attention over all tokens.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder: cross-attention from the fg-query into encoded tokens.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.out_proj = nn.Linear(dim, dim)
        self._init_weights()
        # Zero-init output projection (LeWM stability pattern):
        # prevents early training instability by starting from zero filter.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        reference_features: torch.Tensor,
        label_encoding: torch.Tensor,
        current_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            reference_features: (B, N_ref, C)
            label_encoding:     (B, N_ref, 1) Gaussian heatmap flattened.
            current_features:   (B, N_cur, C)

        Returns:
            omega: (B, C)
            encoded: (B, N_ref + N_cur, C) encoded tokens
                     (used by GOT-JEPA Expander in Stage 2).
        """
        B = reference_features.size(0)

        ref_lbl = self.label_embed(label_encoding)
        ref = reference_features + ref_lbl
        cur = current_features

        ref = self.pos_embed(ref)
        cur = self.pos_embed(cur)

        ref = ref + self.frame_type(torch.zeros(1, 1, dtype=torch.long, device=ref.device))
        cur = cur + self.frame_type(torch.ones(1, 1, dtype=torch.long, device=cur.device))

        tokens = torch.cat([ref, cur], dim=1)  # (B, N_ref + N_cur, C)
        encoded = self.encoder(tokens)

        query = self.fg_embed.expand(B, -1, -1)
        query = query + self.frame_type(
            2 * torch.ones(1, 1, dtype=torch.long, device=query.device)
        )
        decoded = self.decoder(query, encoded)   # (B, 1, C)

        omega = self.out_proj(decoded.squeeze(1))  # (B, C)
        return omega, encoded


def apply_filter(
    omega: torch.Tensor,
    feature_map: torch.Tensor,
) -> torch.Tensor:
    """
    Convolve the per-track filter weights over the feature map to produce
    a target score map.

    Args:
        omega: (B, C) filter weights.
        feature_map: (B, C, H, W) current-frame features.

    Returns:
        (B, 1, H, W) score map (per-track target classification response).
    """
    # (B, C, 1, 1) @ broadcast → channel-wise inner product.
    return (feature_map * omega.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
