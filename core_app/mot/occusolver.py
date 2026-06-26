"""
OccuSolver — occlusion-aware visibility gating (GOT-JEPA §III-C).

Paper: GOT-JEPA (TCSVT 2026) — https://arxiv.org/abs/2602.14771

OccuSolver refines a frozen Point Tracker (CoTracker) with object
priors so it becomes *object-aware*, and uses the resulting per-point
visibility to gate the current-frame feature map before localisation.

Pipeline:
  1. Point tracker (CoTracker, frozen) returns per-point appearance
     features Q ∈ R^F and 2D coordinates PT ∈ R^2.
  2. PriorEncoder takes the per-track reference labels (Gaussian
     heatmaps) and emits feature-shaped embeddings (p_a, p_b) added
     element-wise to Q at the first and middle frames.
  3. A small 4-head 2-layer ``light-Trans`` transformer refines Q,
     fine-tuned via LadderSide (trainable).
  4. ScaleNet + VisHead predict per-point visibility.
  5. Per-track Gaussian kernel maps the points back to a (H, W)
     visibility map which gates the current-frame features before the
     localiser.

This implementation provides the structural wiring and stub heads.
The actual CoTracker is loaded via ``torch.hub`` when available;
otherwise the caller can enable ``stub=True`` for unit testing.

Stage 4 wiring plan: ``SurgicalMOTSystem`` receives a visibility score
per active track and passes it to ``TrackManager.step`` — which in
turn gates the per-track filter update and Hungarian cost.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_cotracker_hub_entrypoint(
    cotracker_version: int,
    cotracker_mode: str,
) -> str:
    """
    Map config flags to ``torch.hub.load(..., entrypoint)`` names
    (see facebookresearch/co-tracker ``hubconf.py``).

    CoTracker2 offline hub entry is ``cotracker2`` (not ``cotracker2_offline``).
    """
    mode = (cotracker_mode or 'online').lower()
    if mode not in ('offline', 'online'):
        raise ValueError(f"cotracker_mode must be 'offline' or 'online', got {cotracker_mode!r}")
    if isinstance(cotracker_version, str):
        norm = cotracker_version.strip().lower().replace('-', '').replace('_', '')
        if norm in {'2', 'v2', 'cotracker2', 'cotracker2online'}:
            cotracker_version = 2
        elif norm in {'3', 'v3', 'cotracker3', 'cotracker3online'}:
            cotracker_version = 3
    else:
        cotracker_version = int(cotracker_version)

    if cotracker_version == 3:
        return f'cotracker3_{mode}'
    if cotracker_version == 2:
        return 'cotracker2_online' if mode == 'online' else 'cotracker2'
    raise ValueError(f"cotracker_version must be 2 or 3, got {cotracker_version}")


def cotracker_kwargs_from_occusolver_cfg(occ_cfg: Optional[Dict]) -> Dict:
    """
    Build ``CoTrackerWrapper`` keyword args from YAML ``occusolver:`` block.

    ``occ_cfg is None`` keeps the previous default (stub tracker for tests
    and environments without hub weights).
    """
    if occ_cfg is None:
        return {'stub': True}
    out: Dict = {'stub': occ_cfg.get('stub', True)}
    if occ_cfg.get('hub_name') is not None:
        out['hub_name'] = occ_cfg['hub_name']
    if occ_cfg.get('model_variant') is not None:
        out['model_variant'] = occ_cfg['model_variant']
    else:
        out['cotracker_version'] = occ_cfg.get('cotracker_version', 2)
        out['cotracker_mode'] = str(occ_cfg.get('cotracker_mode', 'online'))
    if 'freeze' in occ_cfg:
        out['freeze'] = bool(occ_cfg['freeze'])
    return out


# ---------------------------------------------------------------------- #
# 1. CoTracker wrapper                                                    #
# ---------------------------------------------------------------------- #


class CoTrackerWrapper(nn.Module):
    """
    Thin wrapper around the CoTracker point tracker.

    Two init paths:
      * ``hub_name`` + hub entry (via ``model_variant`` or
        ``cotracker_version`` / ``cotracker_mode``): loads from
        ``torch.hub.load('facebookresearch/co-tracker', ...)``.
      * ``stub=True``: lightweight ConvNet for unit tests.
    """

    def __init__(
        self,
        hub_name: Optional[str] = 'facebookresearch/co-tracker',
        model_variant: Optional[str] = None,
        cotracker_version: int = 3,
        cotracker_mode: str = 'online',
        feature_dim: int = 128,
        stub: bool = False,
        freeze: bool = True,
    ):
        super().__init__()
        self.stub = stub
        self.feature_dim = feature_dim
        self._online_predictor = False

        if stub:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, feature_dim, 3, stride=2, padding=1),
            )
        else:  # pragma: no cover — environment-specific.
            resolved = model_variant or resolve_cotracker_hub_entrypoint(
                cotracker_version, cotracker_mode
            )
            try:
                self.backbone = torch.hub.load(
                    hub_name,
                    resolved,
                    pretrained=True,
                    trust_repo=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load CoTracker from torch.hub ({hub_name}, {resolved}). "
                    "Pass stub=True for unit tests. "
                    f"Original error: {e}"
                ) from e
            self._online_predictor = 'Online' in type(self.backbone).__name__

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    @staticmethod
    def _queries_from_norm_xy(
        query_points: torch.Tensor,
        height: int,
        width: int,
        query_frame: int = 0,
    ) -> torch.Tensor:
        """Normalised (B, N, 2) xy → CoTracker ``queries`` (B, N, 3) as (t, x, y) pixels."""
        B, N, _ = query_points.shape
        device, dtype = query_points.device, query_points.dtype
        tcol = torch.full((B, N, 1), float(query_frame), device=device, dtype=dtype)
        xs = query_points[..., 0].clamp(0.0, 1.0) * (width - 1)
        ys = query_points[..., 1].clamp(0.0, 1.0) * (height - 1)
        xy = torch.stack([xs, ys], dim=-1)
        return torch.cat([tcol, xy], dim=-1)

    def forward(
        self,
        video: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            video: (B, T, 3, H, W) clip.
            query_points: optional (B, N, 2) initial query points in
                          normalised [0, 1] coordinates.

        Returns:
            ``{'appearance': (B, N, F), 'coords': (B, T, N, 2),
              'visibility': (B, T, N)}``
        """
        if self.stub:
            B, T, C, H, W = video.shape
            # For the stub we just grab patch features at the query positions.
            if query_points is None:
                N = 128
                query_points = torch.rand(B, N, 2, device=video.device)
            else:
                N = query_points.size(1)

            # Feature map for the first frame only (cheap).
            feat = self.backbone(video[:, 0])                       # (B, F, H', W')
            H_f, W_f = feat.shape[-2:]
            xs = (query_points[..., 0] * (W_f - 1)).long().clamp(0, W_f - 1)
            ys = (query_points[..., 1] * (H_f - 1)).long().clamp(0, H_f - 1)
            appearance = feat[
                torch.arange(B, device=video.device).view(B, 1).expand(B, N),
                :,
                ys,
                xs,
            ]                                                        # (B, N, F)
            coords = query_points.unsqueeze(1).expand(B, T, N, 2)
            visibility = torch.ones(B, T, N, device=video.device)
            return {
                'appearance': appearance,
                'coords': coords,
                'visibility': visibility,
            }

        # torch.hub CoTracker predictors: (video, queries=...) → tracks, visibilities.
        # They do not expose per-point appearance embeddings; OccuSolver relies on
        # PriorEncoder + light_trans for that path (appearance placeholder below).
        B, T, _, H, W = video.shape
        if query_points is None:
            N = 128
            query_points = torch.rand(B, N, 2, device=video.device, dtype=video.dtype)
        queries = self._queries_from_norm_xy(query_points, H, W, query_frame=0)

        if self._online_predictor:
            self.backbone(video, is_first_step=True, queries=queries)
            tracks, visibilities = self.backbone(video, is_first_step=False)
        else:
            tracks, visibilities = self.backbone(video, queries=queries)

        if visibilities.dtype == torch.bool:
            vis_f = visibilities.to(dtype=tracks.dtype)
        else:
            vis_f = visibilities

        appearance = torch.zeros(
            B,
            query_points.size(1),
            self.feature_dim,
            device=video.device,
            dtype=video.dtype,
        )
        return {
            'appearance': appearance,
            'coords': tracks,
            'visibility': vis_f,
        }


# ---------------------------------------------------------------------- #
# 3. Prior encoder                                                        #
# ---------------------------------------------------------------------- #


class DepthAnythingWrapper(nn.Module):
    def __init__(self, stub=False, freeze=True, stub_out_channels=256):
        super().__init__()
        self.stub = stub
        self.output_dim = stub_out_channels if stub else 384
        if stub:
            self.backbone = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3), nn.GroupNorm(8, 64), nn.GELU(), nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.GELU(), nn.Conv2d(128, stub_out_channels, 3, 2, 1))
        else:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self.processor = AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Small')
            self.backbone = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small')
        if freeze:
            for p in self.parameters(): p.requires_grad = False
            self.backbone.eval()

    def forward(self, rgb):
        if self.stub:
            d = self.backbone(rgb)
            return d.mean(dim=1, keepdim=True) if d.size(1) != 1 else d
        with torch.no_grad():
            inp = {k: v.to(rgb.device) for k, v in self.processor(images=rgb, return_tensors='pt').items()}
            return self.backbone(**inp).predicted_depth.unsqueeze(1)

    @staticmethod
    def sample_depth_at_points(depth_map, coords_xy_norm):
        B, N = coords_xy_norm.shape[:2]
        if N == 0:
            return torch.empty(B, 0, 1, device=coords_xy_norm.device, dtype=coords_xy_norm.dtype)
        grid = (coords_xy_norm * 2 - 1).view(B, N, 1, 2)
        return F.grid_sample(depth_map, grid, mode='bilinear', align_corners=False, padding_mode='border').permute(0, 2, 3, 1).reshape(B, N, 1)


class PriorEncoder(nn.Module):
    """
    ToMP-style label encoder — turns the per-track reference Gaussian
    heatmap into a feature embedding that can be element-wise added to
    the point-tracker's per-point appearance features.
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, feature_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, feature_dim),
        )

    def forward(
        self,
        heatmap: torch.Tensor,
        query_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            heatmap: (B, 1, H, W) Gaussian heatmap from the reference frame.
            query_points: (B, N, 2) in normalised [0, 1] coords.

        Returns:
            (B, N, feature_dim) embeddings sampled at the query locations.
        """
        feat = self.net(heatmap)                                  # (B, C, H, W)
        B, _, H, W = feat.shape
        N = query_points.size(1)
        xs = (query_points[..., 0] * (W - 1)).long().clamp(0, W - 1)
        ys = (query_points[..., 1] * (H - 1)).long().clamp(0, H - 1)
        embs = feat[
            torch.arange(B, device=heatmap.device).view(B, 1).expand(B, N),
            :,
            ys,
            xs,
        ]
        return embs


# ---------------------------------------------------------------------- #
# 4. Vis / scale heads                                                    #
# ---------------------------------------------------------------------- #


class VisHead(nn.Module):
    """Predict per-point visibility in [0, 1]."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, N, F) → (B, N, 1)."""
        return self.net(features)


class ScaleNet(nn.Module):
    """
    MLP-Mixer-style conditioner that blends the iterative refined
    features with a global context representation. Lightweight.
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, N, F) → (B, N, F)."""
        mixed = self.token_mix(features.mean(dim=1, keepdim=True))
        return self.channel_mix(features + mixed)


# ---------------------------------------------------------------------- #
# 4b. Visibility map (Gaussian splat)                                     #
# ---------------------------------------------------------------------- #


def visibility_map_from_points(
    coords: torch.Tensor,
    visibility: torch.Tensor,
    height: int,
    width: int,
    sigma_frac: float = 0.06,
) -> torch.Tensor:
    """
    Splat per-point visibility onto a spatial map (GOT-JEPA OccuSolver §III-C).

    Args:
        coords: (B, N, 2) normalised (x, y) in [0, 1].
        visibility: (B, N) per-point scores in [0, 1].
        height, width: output grid size.
        sigma_frac: Gaussian width as a fraction of max(H, W).

    Returns:
        (B, 1, H, W) visibility map in [0, 1].
    """
    B, N, _ = coords.shape
    if N == 0:
        return torch.zeros(B, 1, height, width, device=coords.device, dtype=coords.dtype)

    device, dtype = coords.device, coords.dtype
    sigma_px = max(float(height), float(width)) * sigma_frac
    sigma_sq = max(sigma_px * sigma_px, 1e-6)

    yy = torch.arange(height, device=device, dtype=dtype)
    xx = torch.arange(width, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(yy, xx, indexing='ij')  # (H, W)

    px = coords[..., 0].clamp(0.0, 1.0) * (width - 1)
    py = coords[..., 1].clamp(0.0, 1.0) * (height - 1)

    dist2 = (
        (gx[None, None, :, :] - px.view(B, N, 1, 1)).pow(2)
        + (gy[None, None, :, :] - py.view(B, N, 1, 1)).pow(2)
    )
    kernels = torch.exp(-dist2 / (2.0 * sigma_sq))
    vis = visibility.clamp(0.0, 1.0).view(B, N, 1, 1)
    vis_map = (kernels * vis).amax(dim=1, keepdim=True)
    return vis_map.clamp(0.0, 1.0)


# ---------------------------------------------------------------------- #
# 5. Main OccuSolver                                                      #
# ---------------------------------------------------------------------- #


class OccuSolver(nn.Module):
    """
    Top-level OccuSolver module.

    Given a video clip and per-track object priors, produces a per-track
    visibility score in [0, 1] that is used by ``TrackManager`` to gate
    filter updates under occlusion.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        cotracker_kwargs: Optional[Dict] = None,
        num_refine_steps: int = 0,
        use_depth: bool = False,
        depth_stub: bool = False,
    ):
        super().__init__()
        if cotracker_kwargs is None:
            cotracker_kwargs = {'stub': True}
        self.point_tracker = CoTrackerWrapper(
            feature_dim=feature_dim, **cotracker_kwargs
        )
        self.prior_encoder = PriorEncoder(feature_dim=feature_dim)
        self.num_refine_steps = max(int(num_refine_steps), 0)
        self.use_depth = use_depth
        if use_depth:
            self.depth_estimator = DepthAnythingWrapper(stub=depth_stub, freeze=True)
        else:
            self.depth_estimator = None
        # Light-Trans: 4-head 2-layer transformer for Ladder-Side fine-tuning.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=4,
            dim_feedforward=feature_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.light_trans = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.scale_net = ScaleNet(feature_dim=feature_dim)
        self.vis_head = VisHead(feature_dim=feature_dim)
        self._refine_scales: Tuple[float, ...] = (1.0, 1.75)
        # Conservative fallback for local refinement:
        #  - local_rgb_dim: (mean + std) over two scales * 3 channels
        local_rgb_dim = 2 * len(self._refine_scales) * 3
        # Two local contexts (previous + current) + prev vis + prev/cur coords.
        self._refine_context_dim = feature_dim + (2 * local_rgb_dim) + 5
        self._refine_input_proj = nn.Linear(self._refine_context_dim, feature_dim)
        self._refine_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 3),
        )
        self._refine_step_scale = 0.02
        self._refine_vis_scale = 0.06

    def forward(
        self,
        video: torch.Tensor,
        ref_heatmap: torch.Tensor,
        query_points: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            video: (B, T, 3, H, W).
            ref_heatmap: (B, 1, H, W) Gaussian prior for one track.
            query_points: (B, N, 2) initial query points in [0, 1].

        Returns:
            ``{'visibility': (B, N), 'coords': (B, T, N, 2),
              'per_track_vis': (B,)}``
        """
        pt_out = self.point_tracker(video, query_points=query_points)
        app = pt_out['appearance']                                 # (B, N, F)
        prior = self.prior_encoder(ref_heatmap, query_points)      # (B, N, F)
        refined = self.light_trans(app + prior)                    # (B, N, F)
        refined = self.scale_net(refined)
        per_point_vis = self.vis_head(refined).squeeze(-1)         # (B, N)
        refined_coords = pt_out['coords']
        if self.num_refine_steps > 1 and app.size(1) > 0:
            refined_coords, per_point_vis = self._iterative_refine(
                video=video,
                features=refined,
                coords=refined_coords,
                vis=per_point_vis,
            )
        per_track_vis = per_point_vis.mean(dim=-1)                 # (B,)
        depth_norm, depth_valid = self._estimate_sparse_depth(
            video=video, coords=refined_coords, visibility=per_point_vis,
        )
        return {
            'visibility': per_point_vis,
            'coords': refined_coords,
            'per_track_vis': per_track_vis,
            'point_depth': depth_norm,
            'point_depth_valid': depth_valid,
        }

    def _sample_local_rgb_features(
        self,
        frame: torch.Tensor,
        points: torch.Tensor,
        scales: Sequence[float] | None = None,
    ) -> torch.Tensor:
        """
        Sample local RGB context around each point at a few scales.

        Args:
            frame: (B, 3, H, W) frame for feature lookup.
            points: (B, N, 2) normalised [0,1] coords in (x, y).
            scales: optional per-scale multipliers.

        Returns:
            (B, N, 2 * len(scales) * C) features (mean + std per scale).
        """
        if scales is None:
            scales = self._refine_scales

        B, N, _ = points.shape
        if N == 0:
            return torch.empty((B, 0, 0), device=points.device, dtype=points.dtype)

        device = frame.device
        dtype = frame.dtype
        H, W = frame.shape[-2:]
        base_step_x = 1.0 / max(W - 1, 1)
        base_step_y = 1.0 / max(H - 1, 1)
        # 5-point cross (center + left/right/up/down).
        cross = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            dtype=dtype,
            device=device,
        )

        repeat_frame = frame[:, None].expand(B, N, *frame.shape[1:]).reshape(B * N, *frame.shape[1:])
        per_scale_ctx = []
        num_probes = cross.size(0)
        step_vec = torch.tensor(
            [base_step_x, base_step_y], dtype=dtype, device=device
        )
        for scale in scales:
            # (5, 2) offsets: center + cross neighbours at this scale
            scaled = cross * (step_vec * float(scale))
            probe = (points[:, :, None, :] + scaled.view(1, 1, num_probes, 2)).clamp(
                0.0, 1.0
            )
            probe = (probe * 2.0 - 1.0).reshape(B * N, num_probes, 1, 2)
            sampled = F.grid_sample(
                repeat_frame,
                probe,
                mode='bilinear',
                align_corners=False,
                padding_mode='border',
            )[:, :, :, 0]                         # (B*N, C, K)
            sampled = sampled.permute(0, 2, 1).reshape(B, N, -1, frame.size(1))
            mean = sampled.mean(dim=2)
            std = sampled.std(dim=2, unbiased=False)
            per_scale_ctx.append(mean)
            per_scale_ctx.append(std)
        return torch.cat(per_scale_ctx, dim=-1)

    def _iterative_refine(
        self,
        video: torch.Tensor,
        features: torch.Tensor,
        coords: torch.Tensor,
        vis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Conservative opt-in iterative refinement for the final-frame query points.

        This path is intentionally light-weight and guarded:
        - it is disabled for num_refine_steps <= 1;
        - it updates only the current-frame coordinates and per-point visibility.
        """
        if self.num_refine_steps <= 1:
            return coords, vis

        B, T, N, _ = coords.shape
        if N == 0:
            return coords, vis

        current_frame = video[:, -1]
        working_coords = coords[:, -1].clone()
        prev_coords = coords[:, -2].clone() if T > 1 else working_coords.clone()
        working_vis = vis

        for _ in range(self.num_refine_steps):
            cur_rgb = self._sample_local_rgb_features(current_frame, working_coords)
            prev_rgb = self._sample_local_rgb_features(current_frame, prev_coords)

            refine_in = torch.cat(
                [features, prev_rgb, cur_rgb, prev_coords, working_coords, working_vis[:, :, None]],
                dim=-1,
            )
            refine_delta = self._refine_head(self._refine_input_proj(refine_in))
            delta_xy = torch.tanh(refine_delta[:, :, :2]) * self._refine_step_scale
            delta_vis = torch.tanh(refine_delta[:, :, 2:]) * self._refine_vis_scale

            next_coords = (working_coords + delta_xy).clamp(0.0, 1.0)
            working_vis = (working_vis + delta_vis[:, :, 0]).clamp(0.0, 1.0)
            prev_coords = working_coords
            working_coords = next_coords

        refined = coords.clone()
        refined[:, -1] = working_coords
        return refined, working_vis

    def _estimate_sparse_depth(
        self,
        video: torch.Tensor,
        coords: torch.Tensor,
        visibility: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample depth at visible points only. Returns (depth_norm, valid_mask)."""
        B, T, N, _ = coords.shape
        if N == 0 or not self.use_depth or self.depth_estimator is None:
            return torch.empty(B, N, 1, device=coords.device, dtype=coords.dtype), torch.zeros(B, N, device=coords.device)
        current_frame = video[:, -1]
        final_coords = coords[:, -1]
        final_vis = visibility
        depth_map = self.depth_estimator(current_frame)
        raw_depth = DepthAnythingWrapper.sample_depth_at_points(depth_map, final_coords)
        mask = (final_vis > 0.5).float().unsqueeze(-1)
        gated = raw_depth * mask
        valid = mask.squeeze(-1)
        if valid.sum() > 0:
            mean_d = (gated.sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp(min=1)).detach()
            var_d = (((gated - mean_d) * mask).pow(2).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp(min=1)).detach()
            depth_norm = (gated - mean_d) / (var_d.sqrt() + 1e-6) * mask
        else:
            depth_norm = gated
        return depth_norm, valid
