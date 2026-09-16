"""
Geometry branch — VGGT + GatedFusion + Null-Space Editor (GOT-Edit).

Paper: GOT-Edit (ICLR 2026) — https://arxiv.org/abs/2602.08550

This module wires the geometry pathway described in §3.2 of GOT-Edit:

  (1) Extract semantic features v_sem via DINOv2 (handled outside, by
      ``SurgicalMOTSystem.encoder``).

  (2) Extract geometric features v_geo via VGGT (handled by
      ``VGGTWrapper`` below).

  (3) Align geometry to the semantic resolution / channel count with a
      small conv and fuse via a per-pixel gating mask:

          z = m ⊙ v_sem + (1 − m) ⊙ Align(v_geo)

  (4) Run the semantic predictor on v_sem alone to get ``ω_sem``.

  (5) Run the geometry predictor on the fused ``z`` to get a
      perturbation ``Δ``.

  (6) Project ``Δ`` into the null space of ``v_sem`` to preserve
      semantic discriminability:

          Δ' = P_null · Δ       where    P_null · z_sem = 0
          ω_final = ω_sem + Δ'

The null-space projector is computed on-the-fly per batch via
whitening → ridge-regularised correlation matrix → SVD → symmetric
projection onto low-energy eigenvectors (Eqs. 9–11 in the paper).

The full geometry branch is OPTIONAL and gated by
``SurgicalMOTSystem(use_geometry=True)``. Stage 1 / 2 default to off.
Stage 4 enables it and fine-tunes jointly.

VGGT can be heavy (~300M params). We load it lazily via torch.hub
(``facebookresearch/vggt``) and allow the caller to supply a dummy
stub for unit testing.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictor import PerTrackModelPredictor


def geometry_branch_kwargs_from_config(
    geom_cfg: Optional[Dict] = None,
) -> Tuple[Dict, Dict, Dict]:
    """
    Split YAML ``geometry:`` into kwargs for ``VGGTWrapper``, ``NullSpaceEditor``,
    and ``PerTrackModelPredictor`` (geometry path).

    Keys such as ``freeze_vggt``, ``ridge_lambda``, and ``energy_threshold`` must
    not be passed through to ``torch.hub.load`` / ``VGGTWrapper`` wholesale.
    """
    cfg = dict(geom_cfg or {})

    vggt: Dict = {}
    for key in (
        'hub_name',
        'model_variant',
        'stub',
        'freeze',
        'stub_out_channels',
        'hf_model_id',
        'repo_dir',
    ):
        if key in cfg:
            vggt[key] = cfg.pop(key)
    if 'vggt_variant' in cfg:
        vggt.setdefault('model_variant', cfg.pop('vggt_variant'))
    if 'freeze_vggt' in cfg:
        vggt['freeze'] = cfg.pop('freeze_vggt')
    elif 'freeze' in cfg and 'freeze' not in vggt:
        vggt['freeze'] = cfg.pop('freeze')

    null_space: Dict = {}
    for key in ('ridge_lambda', 'energy_threshold', 'min_null_rank'):
        if key in cfg:
            null_space[key] = cfg.pop(key)

    geom_predictor = cfg
    return vggt, null_space, geom_predictor


def _default_vggt_repo_dir() -> str:
    return os.path.expanduser('~/.cache/torch/hub/facebookresearch_vggt_main')


def _load_vggt_backbone(
    model_variant: str,
    hub_name: str,
    hf_model_id: Optional[str] = None,
    repo_dir: Optional[str] = None,
) -> nn.Module:
    """
    Load VGGT weights. The upstream repo has no ``hubconf.py``, so ``torch.hub``
    cannot load ``facebookresearch/vggt``; we use Hugging Face or a local clone.
    """
    del model_variant  # single public checkpoint; kept for YAML compatibility.
    hf_id = hf_model_id or 'facebook/VGGT-1B'
    repo = os.path.abspath(repo_dir or _default_vggt_repo_dir())

    if os.path.isdir(os.path.join(repo, 'vggt')) and repo not in sys.path:
        sys.path.insert(0, repo)

    try:
        from vggt.models.vggt import VGGT  # type: ignore[import-untyped]
    except ImportError as imp_err:
        raise RuntimeError(
            'VGGT Python package not found. Clone '
            'https://github.com/facebookresearch/vggt into '
            f'{repo} (or set geometry.repo_dir), or pass geometry.stub: true. '
            f'Import error: {imp_err}'
        ) from imp_err

    try:
        return VGGT.from_pretrained(hf_id)
    except Exception as hf_err:
        try:
            return VGGT()
        except Exception as init_err:
            raise RuntimeError(
                f'Failed to load VGGT from Hugging Face ({hf_id}) or init locally. '
                f'HF error: {hf_err}; init error: {init_err}'
            ) from init_err


# ---------------------------------------------------------------------- #
# 1. VGGT wrapper                                                         #
# ---------------------------------------------------------------------- #


class VGGTWrapper(nn.Module):
    """
    Thin wrapper around the VGGT model.

    VGGT (CVPR 2025 Best Paper) infers 3D attributes from single / multi
    view images. We only need its per-pixel geometric feature map
    (typically ``C' = 768``, resolution ≤ input / 14).

    Init paths:
      * ``stub=True``: lightweight ConvNet for tests / machines without VGGT weights.
      * ``stub=False``: ``VGGT.from_pretrained`` via local clone + Hugging Face
        (``facebook/VGGT-1B``). ``torch.hub`` is not used — upstream has no
        ``hubconf.py``.
    """

    def __init__(
        self,
        hub_name: Optional[str] = 'facebookresearch/vggt',
        model_variant: str = 'vggt_base',
        stub: bool = False,
        freeze: bool = True,
        stub_out_channels: int = 256,
        hf_model_id: Optional[str] = None,
        repo_dir: Optional[str] = None,
    ):
        super().__init__()
        self.stub = stub
        self.output_dim = stub_out_channels if stub else 2

        if stub:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.GroupNorm(8, 128),
                nn.GELU(),
                nn.Conv2d(128, stub_out_channels, 3, stride=2, padding=1),
            )
        else:
            self.backbone = _load_vggt_backbone(
                model_variant=model_variant,
                hub_name=hub_name or 'facebookresearch/vggt',
                hf_model_id=hf_model_id,
                repo_dir=repo_dir,
            )
            with torch.no_grad():
                probe = torch.zeros(1, 3, 224, 224)
                self.output_dim = int(self._vggt_feature_map(probe).shape[1])

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.backbone.eval()

    @staticmethod
    def _to_bchw(feat: torch.Tensor) -> torch.Tensor:
        """Normalize VGGT head output to (B, C, H, W)."""
        if feat.dim() != 4:
            return feat
        b, d1, d2, d3 = feat.shape
        # (B, H, W, C) when H,W are spatial and C is small (depth channels).
        if d3 <= 8 and d1 == d2 and d1 > d3:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def _vggt_feature_map(self, rgb: torch.Tensor) -> torch.Tensor:
        """Depth-head feature map for a single-frame batch (B, 3, H, W)."""
        images = rgb.unsqueeze(1)
        aggregated_tokens_list, patch_start_idx = self.backbone.aggregator(images)
        depth_head = self.backbone.depth_head
        if depth_head is None:
            raise RuntimeError('Loaded VGGT has no depth_head; cannot build geometry features.')
        depth, _conf = depth_head(
            aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
        )
        # depth: (B, S, ...) with S=1; layout may be (B, C, H, W) or (B, H, W, C).
        return self._to_bchw(depth[:, 0])

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) in [0, 1].

        Returns:
            (B, C', H', W') geometric feature map.
        """
        frozen = all(not p.requires_grad for p in self.parameters())
        ctx = torch.no_grad() if frozen else torch.enable_grad()
        with ctx:
            if self.stub:
                return self.backbone(rgb)
            if hasattr(self.backbone, 'extract_features'):
                return self.backbone.extract_features(rgb)
            if hasattr(self.backbone, 'aggregator') and getattr(
                self.backbone, 'depth_head', None
            ) is not None:
                return self._vggt_feature_map(rgb)
            out = self.backbone(rgb)
            if isinstance(out, dict):
                if 'depth' in out:
                    d = out['depth']
                    return d[:, 0] if d.dim() == 5 else d
                raise RuntimeError(
                    f'VGGT forward returned dict without depth: {list(out.keys())}'
                )
            return out


# ---------------------------------------------------------------------- #
# 2. Align + gated fusion                                                 #
# ---------------------------------------------------------------------- #


class Align(nn.Module):
    """1×1 conv to match semantic (C, H, W) shape."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        y = self.proj(x)
        if y.shape[-2:] != target_hw:
            y = F.interpolate(y, size=target_hw, mode='bilinear', align_corners=False)
        return y


class GatedFusion(nn.Module):
    """
    Predict a per-pixel gating mask ``m`` from the concatenation of
    ``[v_sem, Align(v_geo)]`` and blend:

        z = m ⊙ v_sem + (1 − m) ⊙ Align(v_geo)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, v_sem: torch.Tensor, v_geo_aligned: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([v_sem, v_geo_aligned], dim=1)
        m = self.gate(cat)
        return m * v_sem + (1.0 - m) * v_geo_aligned


# ---------------------------------------------------------------------- #
# 3. Null-space editor                                                    #
# ---------------------------------------------------------------------- #


class NullSpaceEditor(nn.Module):
    """
    Compute the null-space projection ``P_null ∈ R^{C x C}`` from a
    batch of semantic feature vectors and use it to project a
    perturbation ``Δ`` so it lives in the null space of the semantic
    subspace.

    Procedure (GOT-Edit Eqs. 9–11):
      1. Whiten Z (centre + unit-variance).
      2. ``M = Z Zᵀ + λI``
      3. ``M = UΣVᵀ`` via SVD.
      4. Keep the eigenvectors corresponding to low-energy singular
         values → ``U_null``.
      5. ``P̂ = U_null U_nullᵀ``.
      6. Symmetrise: ``P_null = 0.5 (P̂ + P̂ᵀ)``.
    """

    def __init__(
        self,
        ridge_lambda: float = 1e-3,
        energy_threshold: float = 0.5,
        min_null_rank: int = 8,
    ):
        super().__init__()
        self.ridge_lambda = ridge_lambda
        self.energy_threshold = energy_threshold
        self.min_null_rank = min_null_rank

    def build_projector(self, semantic_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            semantic_weights: (N, C) matrix of semantic feature samples
                              (typically ``ω_sem`` across the batch, or
                              pooled semantic tokens from the reference).

        Returns:
            (C, C) null-space projection matrix.
        """
        N, C = semantic_weights.shape
        device = semantic_weights.device

        # Whiten per-dim (centre + scale).
        mean = semantic_weights.mean(dim=0, keepdim=True)
        std = semantic_weights.std(dim=0, keepdim=True).clamp(min=1e-4)
        Z = (semantic_weights - mean) / std                          # (N, C)

        # Regularised correlation matrix.
        M = Z.t() @ Z + self.ridge_lambda * torch.eye(C, device=device)  # (C, C)

        try:
            U, S, _ = torch.linalg.svd(M, full_matrices=False)
        except RuntimeError:
            # Fallback: return identity (skip null-space editing this step).
            return torch.eye(C, device=device)

        # Determine how many dims belong to the null subspace
        # (low-energy tail of the singular-value spectrum).
        total = S.sum().clamp(min=1e-6)
        cum = torch.cumsum(S, dim=0)
        # Keep the smallest indices whose cumulative energy is below the threshold.
        null_mask = (cum / total) > self.energy_threshold
        null_indices = torch.nonzero(null_mask, as_tuple=False).flatten()
        if null_indices.numel() < self.min_null_rank:
            null_indices = torch.arange(
                C - self.min_null_rank, C, device=device
            )
        U_null = U[:, null_indices]                                  # (C, k)

        P_raw = U_null @ U_null.t()                                  # (C, C)
        P_null = 0.5 * (P_raw + P_raw.t())                           # symmetrise
        return P_null

    def apply(
        self, delta: torch.Tensor, semantic_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Project ``delta`` into the null space of ``semantic_weights``.

        Args:
            delta: (N, C) perturbation to project.
            semantic_weights: (N, C) or (M, C) reference semantic weights.
        """
        P = self.build_projector(semantic_weights)  # (C, C)
        return delta @ P.t()

    def forward(
        self, delta: torch.Tensor, semantic_weights: torch.Tensor
    ) -> torch.Tensor:  # convenience alias
        return self.apply(delta, semantic_weights)


# ---------------------------------------------------------------------- #
# 4. Top-level GeometryBranch                                             #
# ---------------------------------------------------------------------- #


class GeometryBranch(nn.Module):
    """
    Full geometry branch: VGGT → Align → GatedFusion → GeomPredictor
    → NullSpaceEditor.

    Returns the edited filter ``ω_final = ω_sem + P_null · Δ`` given
    (a) the semantic filter ``ω_sem`` from the main per-track predictor
    and (b) the raw RGB images used for VGGT.

    Stage 4 wiring plan: ``SurgicalMOTSystem`` runs the semantic
    predictor on semantic features alone (getting ``ω_sem``) and then
    hands the fused features ``z`` to this branch to emit the edited
    filter.
    """

    def __init__(
        self,
        semantic_dim: int = 768,
        pred_dim: int = 256,
        vggt_kwargs: Optional[Dict] = None,
        geom_predictor_kwargs: Optional[Dict] = None,
        null_space_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        vggt_kwargs = vggt_kwargs or {}
        geom_predictor_kwargs = geom_predictor_kwargs or {}
        null_space_kwargs = null_space_kwargs or {}

        self.vggt = VGGTWrapper(**vggt_kwargs)
        self.align = Align(in_channels=self.vggt.output_dim, out_channels=semantic_dim)
        self.fuse = GatedFusion(channels=semantic_dim)
        self.post_proj = nn.Conv2d(semantic_dim, pred_dim, kernel_size=1)

        # Geometry-only predictor emitting ``Δ`` from fused features.
        self.geom_predictor = PerTrackModelPredictor(
            dim=pred_dim, **geom_predictor_kwargs
        )
        self.null_space = NullSpaceEditor(**null_space_kwargs)

    def forward(
        self,
        rgb: torch.Tensor,
        v_sem: torch.Tensor,
        ref_features: torch.Tensor,
        ref_labels: torch.Tensor,
        cur_features: torch.Tensor,
        omega_sem: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            rgb: (B, 3, H, W) raw input (current frame).
            v_sem: (B, C_sem, H_s, W_s) semantic feature map (same as
                   ``neck_out['spatial_map']`` before per-track projection).
            ref_features, ref_labels, cur_features: see
                ``PerTrackModelPredictor.forward``.
            omega_sem: (B, C_pred) semantic filter from the main predictor.

        Returns:
            ``{'omega_final', 'delta', 'z_fused'}``.
        """
        v_geo = self.vggt(rgb)                               # (B, C_geo, H_g, W_g)
        v_geo_aligned = self.align(v_geo, v_sem.shape[-2:])  # (B, C_sem, H_s, W_s)
        z = self.fuse(v_sem, v_geo_aligned)                  # (B, C_sem, H_s, W_s)
        z_pred = self.post_proj(z)                           # (B, C_pred, H_s, W_s)

        # Geometric perturbation Δ — predictor takes fused tokens in pred_dim.
        B, _, H, W = z_pred.shape
        z_tokens = z_pred.flatten(2).transpose(1, 2)          # (B, H*W, C_pred)
        delta, _ = self.geom_predictor(ref_features, ref_labels, z_tokens)

        # Null-space projection using current frame's semantic features as the anchor
        # (avoiding degenerate N=1 sample std() warnings and SVD convergence issues).
        delta_prime = self.null_space(delta, cur_features.squeeze(0))
        omega_final = omega_sem + delta_prime

        return {
            'omega_final': omega_final,
            'delta': delta_prime,
            'z_fused': z_pred,
        }
