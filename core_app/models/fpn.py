"""
Encoder-Aware Feature Necks for Surgical Tool Detection.

- SimpleFPN: ViTDet-style 4-scale pyramid for DINOv2 (spatial-only encoder).
- VJEPANeck: Lightweight temporal-collapse neck for V-JEPA 2.1 (spatiotemporal encoder).
- EncoderNeck: Unified interface that dispatches to the correct neck based on encoder type.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class SimpleFPN(nn.Module):
    """
    ViTDet-style Simple Feature Pyramid Network for DINOv2.

    DINOv2 produces single-scale features (B, N, C) where N = H*W (e.g. 28*28=784
    for 392-px input with patch size 14). This module builds a 4-level pyramid
    from the single-scale map via transposed convolutions (upsampling) and strided
    convolutions (downsampling).

    Output levels (relative to input spatial resolution):
        P2: stride 4  — 2× upsampled  (H*2, W*2)   e.g. 56×56
        P3: stride 8  — 1× (identity) (H,   W)     e.g. 28×28
        P4: stride 16 — 2× downsampled (H/2, W/2)  e.g. 14×14
        P5: stride 32 — 4× downsampled (H/4, W/4)  e.g.  7×7

    All levels projected to `neck_dim` channels.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        neck_dim: int = 256,
        spatial_h: int = 28,
        spatial_w: int = 28,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.neck_dim = neck_dim
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w

        # Lateral projections — bring all to neck_dim
        self.lateral = nn.Conv2d(embed_dim, neck_dim, kernel_size=1, bias=False)

        # P2: upsample by 2
        self.p2_up = nn.Sequential(
            nn.ConvTranspose2d(neck_dim, neck_dim, kernel_size=2, stride=2),
            nn.GroupNorm(32, neck_dim),
            nn.GELU(),
            nn.Conv2d(neck_dim, neck_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, neck_dim),
        )

        # P3: identity (just refine)
        self.p3_refine = nn.Sequential(
            nn.Conv2d(neck_dim, neck_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, neck_dim),
        )

        # P4: downsample by 2
        self.p4_down = nn.Sequential(
            nn.Conv2d(neck_dim, neck_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, neck_dim),
        )

        # P5: downsample by 4 (two stride-2 convs)
        self.p5_down = nn.Sequential(
            nn.Conv2d(neck_dim, neck_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, neck_dim),
            nn.GELU(),
            nn.Conv2d(neck_dim, neck_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, neck_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, N, C) — flat token sequence from DINOv2.
                      N must equal spatial_h * spatial_w.

        Returns:
            dict with keys 'P2', 'P3', 'P4', 'P5' — each (B, neck_dim, H_l, W_l)
            and 'spatial_map' — (B, C, H, W) raw reshaped features for RoIAlign.
        """
        B, N, C = features.shape
        H, W = self.spatial_h, self.spatial_w
        assert N == H * W, f"Expected N={H*W}, got N={N}"

        # Reshape to spatial grid
        x = features.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        spatial_map = x  # raw spatial features for RoIAlign

        # Lateral project to neck_dim
        lat = self.lateral(x)  # (B, neck_dim, H, W)

        p3 = self.p3_refine(lat)
        p2 = self.p2_up(lat)
        p4 = self.p4_down(lat)
        p5 = self.p5_down(lat)

        return {
            'P2': p2,
            'P3': p3,
            'P4': p4,
            'P5': p5,
            'spatial_map': spatial_map,
            'flat': features,  # (B, N, C) kept for DETR attention input
        }


class VJEPANeck(nn.Module):
    """
    Lightweight neck for V-JEPA 2.1.

    V-JEPA 2.1 produces dense spatiotemporal features (B, T, N_spatial, C) where
    T=8 temporal tokens and N_spatial = 24*24 = 576 per frame. Thanks to deep
    self-supervision, last-layer features alone are sufficient for dense tasks —
    no multi-scale FPN required.

    This neck:
      1. Collapses the temporal dimension (weighted mean or select last frame).
      2. Reshapes to (B, C, H, W) spatial grid.
      3. Projects to neck_dim via a 1×1 conv.
      4. Optionally produces a light 2-scale output (P3 + P4) if multi_scale=True.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        neck_dim: int = 256,
        spatial_h: int = 24,
        spatial_w: int = 24,
        temporal_reduction: str = 'mean',  # 'mean' | 'last' | 'weighted'
        multi_scale: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.neck_dim = neck_dim
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.temporal_reduction = temporal_reduction
        self.multi_scale = multi_scale

        # Learnable temporal weights (used when temporal_reduction == 'weighted')
        self.temporal_weights = nn.Parameter(torch.ones(1))  # broadcast over T

        # 1×1 projection
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, neck_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, neck_dim),
            nn.GELU(),
        )

        # Optional refinement conv
        self.refine = nn.Conv2d(neck_dim, neck_dim, kernel_size=3, padding=1, bias=False)

        # Optional second scale
        if multi_scale:
            self.p4_down = nn.Sequential(
                nn.Conv2d(neck_dim, neck_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(32, neck_dim),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _collapse_temporal(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, N, C)

        Returns:
            (B, N, C)
        """
        if self.temporal_reduction == 'last':
            return features[:, -1]
        elif self.temporal_reduction == 'weighted':
            # Softmax over T dimension, learned weights broadcast over (B, N, C)
            T = features.size(1)
            w = torch.softmax(self.temporal_weights.expand(T), dim=0)
            return (features * w.view(1, T, 1, 1)).sum(dim=1)
        else:  # 'mean'
            return features.mean(dim=1)

    def forward(
        self,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, T, N, C) — spatiotemporal tokens from V-JEPA 2.1.
                      OR (B, N, C) — if already collapsed externally.

        Returns:
            dict with 'P3' — (B, neck_dim, H, W)
            and optionally 'P4' if multi_scale=True.
            Also 'spatial_map' (B, C, H, W) and 'flat' (B, N, C).
        """
        # Collapse temporal dim if needed
        if features.dim() == 4:
            flat = self._collapse_temporal(features)  # (B, N, C)
        else:
            flat = features  # already (B, N, C)

        B, N, C = flat.shape
        # Dynamically compute spatial dimensions from N
        H = int(N ** 0.5)
        W = N // H
        if H * W != N:
            # Fallback to configured dimensions if N is not a perfect square
            H, W = self.spatial_h, self.spatial_w

        # Reshape to spatial
        x = flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        spatial_map = x  # raw for RoIAlign

        # Project + refine
        p3 = self.proj(x)
        p3 = self.refine(p3)

        out = {
            'P3': p3,
            'spatial_map': spatial_map,
            'flat': flat,
        }

        if self.multi_scale:
            out['P4'] = self.p4_down(p3)

        return out


class EncoderNeck(nn.Module):
    """
    Unified encoder-aware neck that dispatches to SimpleFPN or VJEPANeck
    based on the encoder type.

    Usage:
        neck = EncoderNeck(encoder_type='dinov2')
        out = neck(features)  # features: (B, N, C)

        neck = EncoderNeck(encoder_type='vjepa')
        out = neck(features)  # features: (B, T, N, C)

    The returned dict always contains:
        - 'spatial_map': (B, embed_dim, H, W) — raw spatial features for RoIAlign
        - 'flat': (B, N, C) — flat tokens for DETR cross-attention
        - 'P3': (B, neck_dim, H, W) — primary detection feature map
        - 'detection_scales': list of (B, neck_dim, H_l, W_l) — all scales for DETR
    """

    ENCODER_CONFIGS = {
        'dinov2': {
            'spatial_h': 28,
            'spatial_w': 28,
            'embed_dim': 768,
        },
        'dinov2_large': {
            'spatial_h': 28,
            'spatial_w': 28,
            'embed_dim': 1024,
        },
        'vjepa': {
            'spatial_h': 24,
            'spatial_w': 24,
            'embed_dim': 768,
        },
        'vjepa_large': {
            'spatial_h': 24,
            'spatial_w': 24,
            'embed_dim': 1024,
        },
    }

    def __init__(
        self,
        encoder_type: str = 'vjepa',
        neck_dim: int = 256,
        vjepa_temporal_reduction: str = 'mean',
        vjepa_multi_scale: bool = False,
        override_spatial_h: Optional[int] = None,
        override_spatial_w: Optional[int] = None,
        override_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.encoder_type = encoder_type

        cfg = self.ENCODER_CONFIGS.get(encoder_type)
        if cfg is None:
            raise ValueError(
                f"Unknown encoder_type '{encoder_type}'. "
                f"Choose from: {list(self.ENCODER_CONFIGS.keys())}"
            )

        embed_dim = override_embed_dim or cfg['embed_dim']
        spatial_h = override_spatial_h or cfg['spatial_h']
        spatial_w = override_spatial_w or cfg['spatial_w']
        self.embed_dim = embed_dim
        self.neck_dim = neck_dim

        if encoder_type.startswith('dinov2'):
            self.neck = SimpleFPN(
                embed_dim=embed_dim,
                neck_dim=neck_dim,
                spatial_h=spatial_h,
                spatial_w=spatial_w,
            )
            self._detection_scales_keys = ['P2', 'P3', 'P4', 'P5']
        else:  # vjepa / vjepa_large
            self.neck = VJEPANeck(
                embed_dim=embed_dim,
                neck_dim=neck_dim,
                spatial_h=spatial_h,
                spatial_w=spatial_w,
                temporal_reduction=vjepa_temporal_reduction,
                multi_scale=vjepa_multi_scale,
            )
            self._detection_scales_keys = ['P3'] + (['P4'] if vjepa_multi_scale else [])

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: DINOv2 → (B, N, C)
                      V-JEPA  → (B, T, N, C)

        Returns:
            Unified output dict with:
                'spatial_map': (B, embed_dim, H, W)
                'flat': (B, N_spatial, embed_dim)
                'P3', 'P4', 'P5', ... : (B, neck_dim, H_l, W_l)
                'detection_scales': list of tensors for DETR
        """
        out = self.neck(features)
        out['detection_scales'] = [out[k] for k in self._detection_scales_keys if k in out]
        return out

    @property
    def output_dim(self) -> int:
        return self.neck_dim

    @property
    def is_multiscale(self) -> bool:
        return len(self._detection_scales_keys) > 1

    def get_flat_for_detr(self, neck_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Flatten all detection-scale feature maps and concatenate for DETR cross-attention.

        Returns:
            (B, sum(H_l * W_l), neck_dim) — flattened multi-scale tokens
        """
        scales = neck_out['detection_scales']
        B = scales[0].shape[0]
        flat_scales = [s.flatten(2).permute(0, 2, 1) for s in scales]  # each (B, H*W, C)
        return torch.cat(flat_scales, dim=1)  # (B, Σ H_l*W_l, neck_dim)
