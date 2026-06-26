"""
Surgical MOT System — top-level assembly.

Pipeline (per frame / clip):

  Clip (B, 3, T, H, W) — T >= 3, first two frames are references,
                          last frame is the current frame to localise.
          ↓
  Shared encoder (frozen DINOv2 / V-JEPA) → (B, T, N, C)
          ↓
  SimpleFPN / VJEPANeck on current frame → neck_out dict with
          { 'P3', 'spatial_map', 'flat', 'detection_scales', ... }
          ↓
  ┌─────────────────────────┐      ┌──────────────────────────────┐
  │ DETR head (birth-only)  │      │ Per-track filter predictor   │
  │ → (boxes, scores, cls)  │      │ → omega_k per active track   │
  └─────────────────────────┘      └──────────────────────────────┘
          ↓                                    ↓
          └── Hungarian association (TrackManager) ──┘
                         ↓
          RoIAlign → ReID embeddings → memory EMA
                         ↓
          Active tracks with persistent IDs

Training modes
--------------
  * ``train`` — returns per-loss tensor plus a ``total_loss`` ready for
    ``.backward()``. Requires ``per_track_targets`` and ``detr_targets``.
  * ``infer`` — returns tracked output (bboxes + ids) with the internal
    ``TrackManager`` updated in place.

Stage 4 hooks are toggled via ``use_geometry`` / ``use_occusolver``
flags. When disabled (default for Stage 1), geometry / occlusion
branches are skipped and identity placeholders are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.detr_head import SurgicalToolDetector
from ..models.deformable_detr_head import DeformableSurgicalToolDetector
from ..models.reid_head import ReidHead, RoIAlignExtractor
from ..models.vjepa_world_model import Dinov2EncoderWrapper, VJEPAEncoderWrapper
from ..models.fpn import EncoderNeck

from .localizer import ClsDec, RegDec, TrackLocalizationLoss, ltrb_to_bbox
from .manager import TrackManager
from .predictor import (
    PerTrackModelPredictor,
    apply_filter,
    gaussian_label_encoding,
)
from .track import Track


TRACKER_INFER_SCORE_THRESHOLD_DEFAULT = 0.3


@dataclass
class PerTrackSample:
    """One (ref_0, ref_1, cur) annotation for a single track in one batch element."""

    batch_idx: int          # which element of the batch this sample belongs to
    ref_bbox_0: torch.Tensor  # (4,) cxcywh normalised
    ref_bbox_1: torch.Tensor  # (4,) cxcywh normalised
    cur_bbox: torch.Tensor    # (4,) cxcywh normalised (GT for L_track)
    cls: int                  # tool class (0..num_tools-1)
    track_id: int             # CholecTrack20 intraoperative track id
    # M1: persistence supervision fields
    operator: int = -1        # 0-3 surgeon side, -1 = unknown
    phase: int = -1           # surgical phase, -1 = unknown
    occluded: int = 0         # 1 = partial occlusion (hard positive)
    visible: int = 1          # 1 = track present in this frame, 0 = absent (re-entry endpoint)


class SurgicalMOTSystem(nn.Module):
    """Top-level Surgical MOT system."""

    def __init__(
        self,
        # Encoder / neck
        encoder_type: str = 'dinov2',
        encoder_checkpoint: Optional[str] = None,
        model_name: str = 'dinov2_vitb14',
        encoder_dim: int = 768,
        neck_dim: int = 256,
        pred_dim: int = 256,
        img_size: int = 392,
        num_frames: int = 4,
        layer_indices: Optional[List[int]] = None,
        vjepa_temporal_reduction: str = 'mean',
        vjepa_multi_scale: bool = False,
        use_torch_hub: bool = True,
        encoder_lora: Optional[Dict[str, Any]] = None,
        # DETR
        num_tools: int = 7,
        num_queries: int = 16,
        num_decoder_layers: int = 5,
        detr_nheads: int = 8,
        detr_dropout: float = 0.15,
        detr_dim_feedforward: int = 2048,
        detr_class_weight: float = 1.0,
        detr_bbox_weight: float = 5.0,
        detr_giou_weight: float = 2.0,
        detr_focal_alpha: float = 0.25,
        detr_focal_gamma: float = 2.0,
        detr_max_pos_tokens: Optional[int] = None,
        detr_use_denoising: bool = False,
        detr_num_denoising_groups: int = 5,
        detr_num_noise_per_group: int = 4,
        detr_label_noise_prob: float = 0.2,
        detr_box_noise_scale: float = 0.4,
        detr_denoising_weight: float = 1.0,
        # Per-track predictor
        pred_num_heads: int = 8,
        pred_num_encoder_layers: int = 4,
        pred_num_decoder_layers: int = 2,
        pred_dim_feedforward: int = 1024,
        pred_dropout: float = 0.1,
        # ReID
        reid_embedding_dim: int = 256,
        reid_supcon_weight: float = 1.0,
        reid_supcon_temperature: float = 0.07,
        reid_cross_consistency_weight: float = 0.0,
        reid_dropout: float = 0.15,
        roi_output_size: int = 7,
        # Loss weights
        track_loss_weight: float = 1.0,
        track_cls_weight: float = 1.0,
        track_giou_weight: float = 2.0,
        reid_loss_weight: float = 0.5,
        occu_loss_weight: float = 0.0,
        consist_loss_weight: float = 0.0,
        occu_gate_features: bool = True,
        occu_max_tracks_per_batch: Optional[int] = None,
        # Track manager (used in inference mode)
        tracker_birth_score: float = 0.6,
        tracker_min_hits: int = 3,
        tracker_max_age: int = 30,
        tracker_reentry_ttl: int = 300,
        tracker_cost_threshold: float = 0.7,
        tracker_infer_score_threshold: float = TRACKER_INFER_SCORE_THRESHOLD_DEFAULT,
        # Stage toggles
        use_geometry: bool = False,
        use_occusolver: bool = False,
        occusolver_kwargs: Optional[Dict] = None,
        use_deformable_detr: bool = False,
        use_depth: bool = False,
        geometry_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.encoder_dim = encoder_dim
        self.neck_dim = neck_dim
        self.pred_dim = pred_dim
        self.num_frames = num_frames
        self.num_tools = num_tools
        self.use_geometry = use_geometry
        self.use_occusolver = use_occusolver
        self.use_depth = use_depth
        self.geometry_kwargs = geometry_kwargs or {}
        self.track_loss_weight = track_loss_weight
        self.reid_loss_weight = reid_loss_weight
        self.occu_loss_weight = occu_loss_weight
        self.consist_loss_weight = consist_loss_weight
        self.occu_gate_features = occu_gate_features
        self.occu_max_tracks_per_batch = occu_max_tracks_per_batch

        if tracker_infer_score_threshold is None:
            tracker_infer_score_threshold = TRACKER_INFER_SCORE_THRESHOLD_DEFAULT

        if layer_indices is None:
            layer_indices = [-1]

        # --- 1. Encoder (frozen base + optional LoRA) ----------- #
        if encoder_type == 'dinov2':
            self.encoder = Dinov2EncoderWrapper(
                model_name=model_name,
                img_size=img_size,
                freeze=True,
                layer_indices=layer_indices,
                lora=encoder_lora,
                encoder_checkpoint=encoder_checkpoint,
            )
            neck_encoder_type = 'dinov2' if encoder_dim == 768 else 'dinov2_large'
        else:
            self.encoder = VJEPAEncoderWrapper(
                checkpoint_path=encoder_checkpoint,
                hf_model_name=model_name if not use_torch_hub and 'facebook/' in model_name else None,
                use_torch_hub=use_torch_hub,
                model_name=model_name,
                embed_dim=encoder_dim,
                num_frames=num_frames,
                img_size=img_size,
                freeze=True,
                layer_indices=layer_indices,
            )
            neck_encoder_type = 'vjepa' if encoder_dim == 768 else 'vjepa_large'

        # --- 2. Encoder-aware neck (SimpleFPN / VJEPANeck) -------- #
        self.encoder_neck = EncoderNeck(
            encoder_type=neck_encoder_type,
            neck_dim=neck_dim,
            override_embed_dim=encoder_dim,
            vjepa_temporal_reduction=vjepa_temporal_reduction,
            vjepa_multi_scale=vjepa_multi_scale,
        )

        # --- 3. Per-track feature projection (encoder_dim → pred_dim) #
        # A tiny 1×1 projection used by the per-track predictor branch.
        self.pred_feature_proj = nn.Conv2d(encoder_dim, pred_dim, kernel_size=1)
        self.pred_token_proj = nn.Linear(encoder_dim, pred_dim)

        # --- 4. DETR detection head ------------------------------- #
        if detr_max_pos_tokens is None:
            detr_max_pos_tokens = 4096 if encoder_type == 'dinov2' else 1024

        if use_deformable_detr:
            self.detr = DeformableSurgicalToolDetector(
                neck_dim=neck_dim,
                num_tools=num_tools,
                num_queries=num_queries,
                num_decoder_layers=num_decoder_layers,
                nheads=detr_nheads,
                dropout=detr_dropout,
                dim_feedforward=detr_dim_feedforward,
                class_weight=detr_class_weight,
                bbox_weight=detr_bbox_weight,
                giou_weight=detr_giou_weight,
                focal_alpha=detr_focal_alpha,
                focal_gamma=detr_focal_gamma,
                use_denoising=detr_use_denoising,
                num_denoising_groups=detr_num_denoising_groups,
                num_noise_per_group=detr_num_noise_per_group,
                label_noise_prob=detr_label_noise_prob,
                box_noise_scale=detr_box_noise_scale,
                denoising_weight=detr_denoising_weight,
            )
        else:
            self.detr = SurgicalToolDetector(
                encoder_dim=encoder_dim,
                num_tools=num_tools,
                num_queries=num_queries,
                num_decoder_layers=num_decoder_layers,
                nheads=detr_nheads,
                dropout=detr_dropout,
                dim_feedforward=detr_dim_feedforward,
                num_pos_tokens=detr_max_pos_tokens,
                class_weight=detr_class_weight,
                bbox_weight=detr_bbox_weight,
                giou_weight=detr_giou_weight,
                focal_alpha=detr_focal_alpha,
                focal_gamma=detr_focal_gamma,
                neck_dim=neck_dim,
            )

        # --- 5. Per-track model predictor ------------------------- #
        self.per_track_predictor = PerTrackModelPredictor(
            dim=pred_dim,
            num_heads=pred_num_heads,
            num_encoder_layers=pred_num_encoder_layers,
            num_decoder_layers=pred_num_decoder_layers,
            dim_feedforward=pred_dim_feedforward,
            dropout=pred_dropout,
            max_hw=max(int(img_size ** 0.5) + 32, 64),
        )
        self.cls_dec = ClsDec(in_channels=1, mid_channels=64)
        self.reg_dec = RegDec(feature_dim=pred_dim, mid_channels=128)
        self.track_loss = TrackLocalizationLoss(
            lambda_cls=track_cls_weight,
            lambda_giou=track_giou_weight,
        )

        # --- 6. RoIAlign + ReID head ------------------------------ #
        self.roi_extractor = RoIAlignExtractor(
            output_size=roi_output_size,
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        self.reid = ReidHead(
            input_dim=neck_dim,
            embedding_dim=reid_embedding_dim,
            num_classes=num_tools,
            pooling='none',
            normalize_embeddings=True,
            dropout=reid_dropout,
            ce_loss_weight=0.0,
            supcon_weight=reid_supcon_weight,
            supcon_temperature=reid_supcon_temperature,
            cross_consistency_weight=reid_cross_consistency_weight,
        )

        # --- 7. Track manager (lives only for inference) ---------- #
        self._tracker_cfg = dict(
            birth_score=tracker_birth_score,
            min_hits=tracker_min_hits,
            max_age=tracker_max_age,
            reentry_ttl=tracker_reentry_ttl,
            cost_threshold=tracker_cost_threshold,
        )
        self.track_manager = TrackManager(**self._tracker_cfg)
        self.tracker_infer_score_threshold = tracker_infer_score_threshold

        # --- 8. Optional Stage 4 branches (built lazily) ---------- #
        if use_geometry:
            from .geometry import GeometryBranch, geometry_branch_kwargs_from_config
            vggt_kw, null_kw, geom_pred_kw = geometry_branch_kwargs_from_config(
                self.geometry_kwargs
            )
            self.geometry = GeometryBranch(
                semantic_dim=encoder_dim,
                pred_dim=pred_dim,
                vggt_kwargs=vggt_kw,
                null_space_kwargs=null_kw,
                geom_predictor_kwargs=geom_pred_kw,
            )
        else:
            self.geometry = None

        if use_occusolver:
            from .occusolver import OccuSolver, cotracker_kwargs_from_occusolver_cfg
            occ_cfg = dict(occusolver_kwargs or {})
            occ_cfg.setdefault('cotracker_version', 3)
            occ_cfg.setdefault('num_refine_steps', 0)
            self.occusolver_num_query_points = int(
                occ_cfg.pop(
                    'num_query_points',
                    occ_cfg.pop('cotracker_num_query_points', 1),
                )
            )
            if self.occusolver_num_query_points < 1:
                self.occusolver_num_query_points = 1
            self.occusolver_num_refine_steps = int(occ_cfg.pop('num_refine_steps', 0))
            if self.occusolver_num_refine_steps < 0:
                self.occusolver_num_refine_steps = 0
            if 'model_variant' not in occ_cfg and 'cotracker_model_variant' in occ_cfg:
                occ_cfg['model_variant'] = occ_cfg.pop('cotracker_model_variant')
            ct_kw = cotracker_kwargs_from_occusolver_cfg(occ_cfg if occ_cfg else None)
            self.occusolver_use_depth = bool(occ_cfg.pop('use_depth', False))
            self.occusolver_depth_stub = bool(occ_cfg.pop('depth_stub', False))
            self.occusolver = OccuSolver(
                feature_dim=pred_dim,
                cotracker_kwargs=ct_kw,
                num_refine_steps=self.occusolver_num_refine_steps,
                use_depth=self.occusolver_use_depth,
                depth_stub=self.occusolver_depth_stub,
            )
        else:
            self.occusolver = None

        # --- 9. Optional depth branch ------------------------------ #
        if use_depth:
            from .depth import DepthWrapper
            self.depth_estimator = DepthWrapper(stub=False, freeze=True)
        else:
            self.depth_estimator = None

    # ------------------------------------------------------------------ #
    # Frame encoding                                                      #
    # ------------------------------------------------------------------ #

    def encode_frames(
        self, clip: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode a (B, 3, T, H, W) video clip.

        Returns:
            reality_seq: (B, T, N, C) per-frame tokens.
            neck_out:    EncoderNeck dict for the *last* frame of the clip.
        """
        feats = self.encoder(clip)
        if isinstance(feats, list):
            feats = feats[-1]  # last layer if multi-layer list
        reality_seq = feats  # (B, T, N, C)

        last_frame_feat = reality_seq[:, -1]  # (B, N, C)

        if self.encoder_type == 'dinov2':
            neck_out = self.encoder_neck(last_frame_feat)
        else:
            neck_out = self.encoder_neck(reality_seq)

        return reality_seq, neck_out

    @staticmethod
    def _clip_bcthw_to_btchw(clip: torch.Tensor) -> torch.Tensor:
        """(B, 3, T, H, W) → (B, T, 3, H, W) for CoTracker / OccuSolver."""
        return clip.permute(0, 2, 1, 3, 4).contiguous()

    @torch.no_grad()
    def _cotracker_teacher_visibility(
        self,
        video_btchw: torch.Tensor,
        query_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Frozen CoTracker visibility on the last frame (pseudo-label for L_occu).

        Returns:
            (B, N) scores in [0, 1].
        """
        pt_out = self.occusolver.point_tracker(video_btchw, query_points=query_points)
        vis = pt_out['visibility']
        if vis.ndim == 3:
            vis = vis[:, -1]
        if vis.dtype == torch.bool:
            vis = vis.float()
        return vis.clamp(0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Per-track forward (shared between train and infer)                  #
    # ------------------------------------------------------------------ #

    def _accumulate_per_track_losses(
        self,
        per_track_result: Dict[str, object],
        per_track_targets: List,
        pred_tokens_seq: torch.Tensor,
        pred_spatial: torch.Tensor,
        spatial_map: Optional[torch.Tensor],
        current_video: torch.Tensor,
    ) -> None:
        """Run per-track predictor losses in fp32 (outside AMP)."""
        ref_tokens = pred_tokens_seq[:, 0]
        ref1_tokens = pred_tokens_seq[:, 1]
        cur_tokens = pred_tokens_seq[:, -1]

        N_tokens = ref_tokens.size(1)
        hw = int(N_tokens ** 0.5)
        assert hw * hw == N_tokens, f"Expected square feature grid, got N={N_tokens}"
        H_pred = W_pred = hw

        if pred_spatial.shape[-1] != W_pred or pred_spatial.shape[-2] != H_pred:
            pred_spatial_matched = F.interpolate(
                pred_spatial, size=(H_pred, W_pred), mode='bilinear', align_corners=False
            )
        else:
            pred_spatial_matched = pred_spatial

        track_losses: List[torch.Tensor] = []
        occu_losses: List[torch.Tensor] = []
        consist_losses: List[torch.Tensor] = []
        track_loss_dict: Dict[str, List[float]] = {}
        total_tracks = 0
        skipped_tracks = 0
        occu_trained = 0

        for b, track_list in enumerate(per_track_targets):
            if not track_list:
                continue
            ref0_b = ref_tokens[b : b + 1]
            ref1_b = ref1_tokens[b : b + 1]
            cur_b = cur_tokens[b : b + 1]
            spatial_b = pred_spatial_matched[b : b + 1]
            if not (
                torch.isfinite(ref0_b).all()
                and torch.isfinite(ref1_b).all()
                and torch.isfinite(cur_b).all()
                and torch.isfinite(spatial_b).all()
            ):
                skipped_tracks += len(track_list)
                continue

            if self.occu_max_tracks_per_batch is not None and len(track_list) > self.occu_max_tracks_per_batch:
                track_list = track_list[: self.occu_max_tracks_per_batch]

            video_occ = None
            if self.occusolver is not None:
                video_occ = self._clip_bcthw_to_btchw(current_video[b : b + 1])

            for sample in track_list:
                total_tracks += 1
                ref_bbox_0 = sample.ref_bbox_0.to(ref0_b.device).view(1, 4)
                ref_bbox_1 = sample.ref_bbox_1.to(ref1_b.device).view(1, 4)
                cur_bbox = sample.cur_bbox.to(cur_b.device).view(1, 4)

                heat0 = gaussian_label_encoding(ref_bbox_0, H_pred, W_pred).view(1, -1, 1)
                heat1 = gaussian_label_encoding(ref_bbox_1, H_pred, W_pred).view(1, -1, 1)
                ref_feat = torch.cat([ref0_b, ref1_b], dim=1)
                ref_labels = torch.cat([heat0, heat1], dim=1)

                visibility_map = None
                if self.occusolver is not None and video_occ is not None:
                    from .occusolver import visibility_map_from_points

                    ref_heat = gaussian_label_encoding(
                        ref_bbox_1, H_pred, W_pred
                    ).view(1, 1, H_pred, W_pred)
                    query_pts = self._bbox_query_points(
                        cur_bbox, self.occusolver_num_query_points
                    )
                    occ_out = self.occusolver(
                        video_occ,
                        ref_heatmap=ref_heat,
                        query_points=query_pts,
                    )
                    if self.occu_loss_weight > 0:
                        teacher_vis = self._cotracker_teacher_visibility(
                            video_occ, query_pts
                        )
                        occ_loss = F.binary_cross_entropy(
                            occ_out['visibility'].clamp(1e-4, 1 - 1e-4),
                            teacher_vis,
                        )
                        if torch.isfinite(occ_loss):
                            occu_losses.append(occ_loss)
                            occu_trained += 1
                    if self.occu_gate_features:
                        visibility_map = visibility_map_from_points(
                            occ_out['coords'][:, -1],
                            occ_out['visibility'],
                            H_pred,
                            W_pred,
                        )

                pt_out = self._per_track_forward(
                    ref_feat_tokens=ref_feat,
                    ref_label_encoding=ref_labels,
                    cur_feat_tokens=cur_b,
                    cur_spatial=spatial_b,
                    rgb=current_video[b:b+1, :, -1] if self.geometry is not None else None,
                    v_sem=spatial_map[b:b+1] if self.geometry is not None else None,
                    visibility_map=visibility_map,
                )
                if (
                    self.consist_loss_weight > 0
                    and self.geometry is not None
                    and 'omega_sem' in pt_out
                    and 'omega_final' in pt_out
                ):
                    c_loss = 1.0 - F.cosine_similarity(
                        pt_out['omega_sem'], pt_out['omega_final'], dim=-1
                    ).mean()
                    if torch.isfinite(c_loss):
                        consist_losses.append(c_loss)
                if not (
                    torch.isfinite(pt_out['score_map']).all()
                    and torch.isfinite(pt_out['bbox_pred']).all()
                ):
                    skipped_tracks += 1
                    continue

                loss_t, ld = self.track_loss(
                    pred_score=pt_out['score_map'],
                    pred_bbox=pt_out['bbox_pred'],
                    gt_bbox=cur_bbox,
                    spatial_h=H_pred,
                    spatial_w=W_pred,
                )
                if loss_t is None:
                    skipped_tracks += 1
                    continue
                track_losses.append(loss_t)
                for k, v in ld.items():
                    track_loss_dict.setdefault(k, []).append(v)

        if track_losses:
            per_track_result['track_loss'] = torch.stack(track_losses).mean()
            per_track_result['loss_dict'] = {
                f'track_{k}': float(sum(v) / len(v)) for k, v in track_loss_dict.items()
            }
            per_track_result['num_tracks'] = total_tracks
        if occu_losses:
            per_track_result['occu_loss'] = torch.stack(occu_losses).mean()
            per_track_result['loss_dict']['occu_bce'] = float(per_track_result['occu_loss'].item())
            per_track_result['occu_tracks'] = occu_trained
        if consist_losses:
            per_track_result['consist_loss'] = torch.stack(consist_losses).mean()
            per_track_result['loss_dict']['consist_cos'] = float(
                per_track_result['consist_loss'].item()
            )
        per_track_result['skipped_tracks'] = skipped_tracks

    def _per_track_forward(
        self,
        ref_feat_tokens: torch.Tensor,    # (B, 2*N, C_pred)
        ref_label_encoding: torch.Tensor,  # (B, 2*N, 1)
        cur_feat_tokens: torch.Tensor,     # (B, N, C_pred)
        cur_spatial: torch.Tensor,         # (B, C_pred, H, W)
        rgb: Optional[torch.Tensor] = None,          # (B, 3, H_img, W_img) for VGGT
        v_sem: Optional[torch.Tensor] = None,        # (B, C_sem, H_s, W_s) semantic map
        visibility_map: Optional[torch.Tensor] = None,  # (B, 1, H, W) OccuSolver gate
    ) -> Dict[str, torch.Tensor]:
        """
        Run the per-track predictor + localiser for one batch of tracks.

        Returns a dict with:
          * ``omega``      (B, C_pred) filter weights
          * ``score_map``  (B, 1, H, W) refined classification score
          * ``ltrb_map``   (B, 4, H, W) regression offsets
          * ``bbox_pred``  (B, 4) predicted cxcywh bbox
          * ``peak_idx``   (B, 2) peak coordinates (y, x) in grid units
        """
        omega, _ = self.per_track_predictor(
            ref_feat_tokens, ref_label_encoding, cur_feat_tokens
        )
        omega_sem = omega

        # Stage 4: geometry-aware editing via null-space projection
        omega_final = omega_sem
        if self.geometry is not None and rgb is not None and v_sem is not None:
            geom_out = self.geometry(
                rgb=rgb,
                v_sem=v_sem,
                ref_features=ref_feat_tokens,
                ref_labels=ref_label_encoding,
                cur_features=cur_feat_tokens,
                omega_sem=omega_sem,
            )
            omega_final = geom_out['omega_final']

        spatial_for_loc = cur_spatial
        if visibility_map is not None:
            if visibility_map.shape[-2:] != cur_spatial.shape[-2:]:
                visibility_map = F.interpolate(
                    visibility_map,
                    size=cur_spatial.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            spatial_for_loc = cur_spatial * visibility_map

        score_raw = apply_filter(omega_final, spatial_for_loc)           # (B, 1, H, W)
        score_map = self.cls_dec(score_raw)                     # (B, 1, H, W)
        ltrb_map = self.reg_dec(spatial_for_loc, score_map)         # (B, 4, H, W)

        B, _, H, W = score_map.shape
        flat = score_map.view(B, -1)
        peak_flat = flat.argmax(dim=1)
        peak_y = peak_flat // W
        peak_x = peak_flat % W
        peak_idx = torch.stack([peak_y, peak_x], dim=-1)        # (B, 2)
        bbox_pred = ltrb_to_bbox(ltrb_map, peak_idx, H, W)

        return {
            'omega': omega_final,
            'omega_sem': omega_sem,
            'omega_final': omega_final,
            'score_map': score_map,
            'ltrb_map': ltrb_map,
            'bbox_pred': bbox_pred,
            'peak_idx': peak_idx,
            'spatial_hw': (H, W),
        }

    def _bbox_query_points(
        self,
        boxes: torch.Tensor,
        num_points: int,
    ) -> torch.Tensor:
        """
        Build deterministic query points inside each box for CoTracker.

        Returns:
            (N, P, 2) query points in normalised [0, 1].
        """
        if boxes.numel() == 0 or num_points <= 1:
            cx, cy = boxes[:, 0], boxes[:, 1]
            return torch.stack([cx, cy], dim=-1).unsqueeze(1)

        device = boxes.device
        num_points = int(num_points)
        grid = int(torch.ceil(torch.sqrt(torch.tensor(float(num_points), device=device))).item())
        grid = max(grid, 1)

        # Square-ish grid of offsets around the box center.
        axis = torch.linspace(-0.5, 0.5, grid, device=device)
        gy, gx = torch.meshgrid(axis, axis, indexing='ij')
        gx = gx.reshape(-1)[:num_points]
        gy = gy.reshape(-1)[:num_points]

        cx = boxes[:, 0].unsqueeze(1)
        cy = boxes[:, 1].unsqueeze(1)
        w = boxes[:, 2].unsqueeze(1).clamp(min=1e-6)
        h = boxes[:, 3].unsqueeze(1).clamp(min=1e-6)

        pts_x = (cx + gx.view(1, -1) * w).clamp(0.0, 1.0)
        pts_y = (cy + gy.view(1, -1) * h).clamp(0.0, 1.0)
        return torch.stack([pts_x, pts_y], dim=-1)

    def _estimate_occ_visibilities(
        self,
        current_video: torch.Tensor,
        boxes: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        """
        Evaluate visibility scores for each detection box through OccuSolver.

        Returns:
            (N,) visibility for each box, or None when disabled/no boxes.
        """
        if self.occusolver is None or boxes.numel() == 0:
            return None

        H, W = grid_hw
        vis_scores: List[torch.Tensor] = []
        query_points = self._bbox_query_points(boxes, self.occusolver_num_query_points)
        for i, box in enumerate(boxes):
            heat = gaussian_label_encoding(box.view(1, 4), H, W).view(1, 1, H, W)
            vis_out = self.occusolver(
                current_video,
                ref_heatmap=heat,
                query_points=query_points[i : i + 1],
            )
            vis_scores.append(vis_out['per_track_vis'])
        return torch.stack(vis_scores, dim=0).squeeze(1).to(current_video.device)

    # ------------------------------------------------------------------ #
    # Top-level forward                                                   #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        current_video: torch.Tensor,
        per_track_targets: Optional[List[List[PerTrackSample]]] = None,
        detr_targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        reid_labels: Optional[torch.Tensor] = None,
        mode: str = 'train',
    ) -> Dict[str, object]:
        """
        Args:
            current_video: (B, 3, T, H, W) with T >= 3.
            per_track_targets: list of length B; each item is a list of
                ``PerTrackSample`` for that batch element. Required in
                'train' mode, ignored in 'infer'.
            detr_targets: list-of-dicts — existing DETR target format.
            reid_labels: (N_tracks,) concatenated track ids for ReID
                supervised contrastive loss.
            mode: 'train' | 'infer' | 'infer_fast'.
                ``infer_fast`` skips ReID + per-track predictor (DETR + tracker only).

        Returns:
            dict with (depending on mode):
              * 'detr', 'reid', 'per_track', 'total_loss', 'loss_dict'
              * 'active_tracks' (inference only)
        """
        fast_infer = mode == 'infer_fast'
        if fast_infer:
            mode = 'infer'
        B, C, T, H, W = current_video.shape
        assert T >= 3, "SurgicalMOTSystem requires clips of >= 3 frames (2 refs + current)."

        # --- 1. Encode frames ------------------------------------- #
        reality_seq, neck_out = self.encode_frames(current_video)
        # reality_seq: (B, T, N, C_enc)

        # Project encoder tokens → pred_dim for the per-track branch.
        pred_tokens_seq = self.pred_token_proj(reality_seq)           # (B, T, N, C_pred)

        # Build per-track pred-dim spatial map for the *current* (last) frame.
        spatial_map = neck_out['spatial_map']                         # (B, C_enc, H_s, W_s)
        pred_spatial = self.pred_feature_proj(spatial_map)            # (B, C_pred, H_s, W_s)

        # --- 2. DETR detection on current frame ------------------- #
        if isinstance(self.detr, DeformableSurgicalToolDetector):
            # Deformable DETR needs the multi-scale neck dict
            detr_outputs = self.detr(
                neck_out=neck_out,
                targets=detr_targets,
            )
        else:
            detr_tokens = self.encoder_neck.get_flat_for_detr(neck_out)   # (B, Σ H·W, neck_dim)
            detr_outputs = self.detr(
                encoder_features=detr_tokens,
                phantom_features=None,
                targets=detr_targets,
            )
        pred_logits = detr_outputs['pred']['class_logits']            # (B, Q, num_tools)
        pred_boxes = detr_outputs['pred']['pred_boxes']               # (B, Q, 4) cxcywh

        # --- 3. ReID via RoIAlign over neck P3 features ---------- #
        p3 = neck_out.get('P3')  # (B, neck_dim, H_p3, W_p3)
        reid_result = {'embeddings': None, 'loss': None, 'loss_dict': {}}
        if p3 is not None and not fast_infer:
            roi_feats = self.roi_extractor(p3, pred_boxes)            # (B*Q, neck_dim)

            # Build per-query ReID class labels from Hungarian matching.
            # For Stage 1 (stateless learnable queries), Hungarian is the identity contract.
            # ReID labels come from Hungarian assignment indices.
            # For Stage 2 (template-conditioned queries), this will change to track_id directly.
            reid_labels_expanded: Optional[torch.Tensor] = None
            det_scores_flat: Optional[torch.Tensor] = None
            roi_feats_filtered: Optional[torch.Tensor] = None

            if mode == 'train':
                matching = detr_outputs.get('matching_indices')
                if matching is not None:
                    Q = pred_boxes.size(1)
                    per_query_cls = torch.full((B, Q), -1, dtype=torch.long, device=pred_boxes.device)
                    for i, (pred_idx, tgt_idx) in enumerate(matching):
                        if len(pred_idx) > 0 and 'labels' in (detr_targets[i] if detr_targets else {}):
                            tgt_labels = detr_targets[i]['labels']
                            if len(tgt_labels) > 0:
                                tgt_idx_clamped = torch.clamp(tgt_idx, 0, len(tgt_labels) - 1)
                                per_query_cls[i, pred_idx] = tgt_labels[tgt_idx_clamped].to(per_query_cls.device)
                    # Only pass matched queries (cls != -1) to ReID head.
                    valid = per_query_cls >= 0
                    if valid.any():
                        reid_labels_expanded = per_query_cls[valid].flatten()
                        det_scores_flat = (
                            pred_logits.sigmoid().max(dim=-1).values[valid].flatten().detach()
                        )
                        # Filter roi_feats to match valid queries
                        roi_feats_filtered = roi_feats.view(B, Q, -1)[valid]

            reid_out = self.reid(
                roi_feats_filtered if roi_feats_filtered is not None else roi_feats,
                labels=reid_labels_expanded,
                detection_scores=det_scores_flat,
            )
            # Only reshape embeddings if they weren't filtered (inference or no valid matches)
            if roi_feats_filtered is None:
                reid_result['embeddings'] = reid_out['embeddings'].view(B, pred_boxes.size(1), -1)
            else:
                # For filtered features, create placeholder embeddings with zeros for unmatched queries
                embeddings_full = torch.zeros(B, pred_boxes.size(1), reid_out['embeddings'].size(-1),
                                              device=reid_out['embeddings'].device, dtype=reid_out['embeddings'].dtype)
                embeddings_full[valid] = reid_out['embeddings']
                reid_result['embeddings'] = embeddings_full
            reid_result['loss'] = reid_out.get('loss')
            reid_result['loss_dict'] = reid_out.get('loss_dict', {})

        # --- 4. Per-track branch (training) ---------------------- #
        per_track_result: Dict[str, object] = {
            'track_loss': None,
            'loss_dict': {},
            'num_tracks': 0,
        }

        if (
            mode == 'train'
            and per_track_targets is not None
            and self.track_loss_weight > 0
            and pred_tokens_seq.size(1) >= 3
        ):
            amp_device = 'cuda' if current_video.is_cuda else 'cpu'
            with torch.autocast(device_type=amp_device, enabled=False):
                self._accumulate_per_track_losses(
                    per_track_result=per_track_result,
                    per_track_targets=per_track_targets,
                    pred_tokens_seq=pred_tokens_seq.float(),
                    pred_spatial=pred_spatial.float(),
                    spatial_map=spatial_map.float() if spatial_map is not None else None,
                    current_video=current_video,
                )

        # --- 5. Inference-time tracking loop ---------------------- #
        if mode == 'infer':
            active = self.track_manager.active_tracks()
            per_track_predictions: List[Dict[str, torch.Tensor]] = []

            if active and not fast_infer:
                ref0_tokens = pred_tokens_seq[:, 0]
                ref1_tokens = pred_tokens_seq[:, 1]
                cur_tokens = pred_tokens_seq[:, -1]
                N_tokens = ref0_tokens.size(1)
                hw = int(N_tokens ** 0.5)
                H_pred = W_pred = hw
                if pred_spatial.shape[-1] != W_pred:
                    pred_spatial_matched = F.interpolate(
                        pred_spatial, size=(H_pred, W_pred), mode='bilinear', align_corners=False
                    )
                else:
                    pred_spatial_matched = pred_spatial

                for track in active:
                    # Re-use the track's most recent box as reference prior.
                    bbox_ref = track.bbox.to(pred_spatial.device).view(1, 4)
                    heat = gaussian_label_encoding(bbox_ref, H_pred, W_pred).view(1, -1, 1)
                    ref_feat = torch.cat([ref0_tokens[:1], ref1_tokens[:1]], dim=1)
                    ref_labels = torch.cat([heat, heat], dim=1)
                    pt_out = self._per_track_forward(
                        ref_feat_tokens=ref_feat,
                        ref_label_encoding=ref_labels,
                        cur_feat_tokens=cur_tokens[:1],
                        cur_spatial=pred_spatial_matched[:1],
                        rgb=current_video[:1, :, -1] if self.geometry is not None else None,
                        v_sem=spatial_map[:1] if self.geometry is not None else None,
                    )
                    per_track_predictions.append({
                        'track_id': track.id,
                        'bbox_pred': pt_out['bbox_pred'][0],
                        'score_map': pt_out['score_map'][0],
                    })

            # TrackManager handles assoc + update using DETR candidates.
            #
            # We pick up the highest-scoring DETR detections in each batch
            # element for the first element only (inference assumes B=1 for
            # the stateful tracker).
            batch_elem = 0
            logits = pred_logits[batch_elem]                   # (Q, num_tools)
            boxes = pred_boxes[batch_elem]                     # (Q, 4)
            scores, classes = logits.sigmoid().max(dim=-1)     # (Q,), (Q,)
            keep = scores > self.tracker_infer_score_threshold
            det_boxes = boxes[keep]
            det_scores = scores[keep]
            det_classes = classes[keep]
            det_embs = None
            if reid_result['embeddings'] is not None:
                det_embs = reid_result['embeddings'][batch_elem][keep]

            det_visibilities = None
            if self.occusolver is not None:
                det_visibilities = self._estimate_occ_visibilities(
                    current_video=current_video[batch_elem : batch_elem + 1],
                    boxes=det_boxes,
                    grid_hw=(pred_spatial_matched[batch_elem:batch_elem + 1].shape[-2:])
                )
                if det_visibilities is not None:
                    det_visibilities = det_visibilities.to(current_video.device)

            step_result = self.track_manager.step(
                det_boxes=det_boxes,
                det_scores=det_scores,
                det_classes=det_classes,
                det_embeddings=det_embs,
                det_visibilities=det_visibilities,
            )

            return {
                'detr': detr_outputs,
                'reid': reid_result,
                'per_track_predictions': per_track_predictions,
                'active_tracks': step_result['active_tracks'],
                'matches': step_result['matches'],
                'new_track_ids': step_result['new_track_ids'],
            }

        # --- 6. Combine training losses --------------------------- #
        total_loss = None
        loss_dict: Dict[str, float] = {}

        if mode == 'train':
            device = current_video.device
            total_loss = torch.zeros(1, device=device).squeeze()

            if 'loss' in detr_outputs:
                total_loss = total_loss + detr_outputs['loss']
                loss_dict.update(detr_outputs.get('loss_dict', {}))

            if reid_result.get('loss') is not None:
                total_loss = total_loss + self.reid_loss_weight * reid_result['loss']
                for k, v in reid_result.get('loss_dict', {}).items():
                    loss_dict[k] = v * self.reid_loss_weight

            if per_track_result['track_loss'] is not None:
                total_loss = total_loss + self.track_loss_weight * per_track_result['track_loss']
                for k, v in per_track_result['loss_dict'].items():
                    if k.startswith('track_'):
                        loss_dict[k] = v * self.track_loss_weight
                loss_dict['track_loss'] = float(per_track_result['track_loss'].item())

            if per_track_result.get('occu_loss') is not None and self.occu_loss_weight > 0:
                total_loss = total_loss + self.occu_loss_weight * per_track_result['occu_loss']
                loss_dict['occu_loss'] = float(per_track_result['occu_loss'].item())

            if per_track_result.get('consist_loss') is not None and self.consist_loss_weight > 0:
                total_loss = total_loss + self.consist_loss_weight * per_track_result['consist_loss']
                loss_dict['consist_loss'] = float(per_track_result['consist_loss'].item())

            if not torch.isfinite(total_loss):
                p = next(p for p in self.parameters() if p.requires_grad)
                total_loss = p.reshape(-1)[0] * 0.0
                loss_dict['total'] = 0.0
            else:
                loss_dict['total'] = float(total_loss.item())

        return {
            'detr': detr_outputs,
            'reid': reid_result,
            'per_track': per_track_result,
            'total_loss': total_loss,
            'loss_dict': loss_dict,
        }

    # ------------------------------------------------------------------ #
    # Inference helpers                                                   #
    # ------------------------------------------------------------------ #

    def reset_tracker(self) -> None:
        """Reset the internal ``TrackManager`` — call between videos."""
        self.track_manager.reset()

    def configure_tracker_for_eval(
        self,
        min_hits: int = 1,
        birth_score: float = 0.3,
        max_age: Optional[int] = None,
        reentry_ttl: Optional[int] = None,
    ) -> None:
        """
        Reconfigure the ``TrackManager`` with eval-friendly settings.

        During evaluation we want tracks to be born immediately (``min_hits=1``)
        and with a lower birth threshold (``birth_score=0.3``) so that
        weak-but-real detections contribute to HOTA/MOTA.  Training-time
        conservative settings (``min_hits=3``, ``birth_score=0.6``) suppress
        false tracks but also suppress true tracks when detection is weak.

        Call this once before ``mot_rollout`` on each checkpoint.
        """
        cfg = dict(self._tracker_cfg)
        cfg['min_hits'] = min_hits
        cfg['birth_score'] = birth_score
        if max_age is not None:
            cfg['max_age'] = max_age
        if reentry_ttl is not None:
            cfg['reentry_ttl'] = reentry_ttl
        self._tracker_cfg = cfg
        self.track_manager = TrackManager(**cfg)
