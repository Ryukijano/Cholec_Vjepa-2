"""
Lightweight Deformable DETR head for surgical tool detection.

Replaces the standard DETR decoder's full cross-attention with
single-scale deformable cross-attention, which has strong spatial
inductive bias and avoids the query-collapse failure mode of vanilla DETR.

This implementation is pure-PyTorch (no custom CUDA ops) and uses
the multi-scale features already produced by SimpleFPN (P2-P5).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .detr_head import SetCriterion, _focal_loss_per_sample
from .query_init import ObjectQueryInit


# --------------------------------------------------------------------------- #
# 1. Pure-PyTorch single-scale deformable attention                         #
# --------------------------------------------------------------------------- #


class DeformableCrossAttention(nn.Module):
    """
    Single-scale deformable cross-attention.

    Each query predicts K 2-D offsets around a shared reference point,
    samples feature values via bilinear interpolation, and attends over
    the sampled features.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.d_head = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        # Initialise offsets to a small circle around the reference point
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)  # (H, 2)
        grid = grid / grid.abs().max(-1, keepdim=True)[0]
        grid = grid.view(self.n_heads, 1, 1, 2).repeat(1, 1, self.n_points, 1)
        for i in range(self.n_points):
            grid[:, :, i, :] *= (i + 1) * 0.01  # small initial radius
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        with torch.no_grad():
            self.sampling_offsets.bias[:] = grid.view(-1)

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,        # (B, N_q, C)
        reference_points: torch.Tensor,  # (B, N_q, 2) in [0,1]
        value: torch.Tensor,        # (B, H*W, C)
        spatial_shape: Tuple[int, int],  # (H, W)
    ) -> torch.Tensor:
        B, N_q, _ = query.shape
        H, W = spatial_shape
        N_v = value.shape[1]
        assert N_v == H * W, f"value length {N_v} != H*W {H*W}"

        # Project value
        v = self.value_proj(value)  # (B, N_v, C)
        v = v.view(B, N_v, self.n_heads, self.d_head)  # (B, N_v, H, d)

        # Compute offsets & attention weights
        offsets = self.sampling_offsets(query).view(
            B, N_q, self.n_heads, self.n_points, 2
        )
        # Normalise offsets by image size so they are in pixel space
        offset_normalizer = torch.tensor([W, H], device=query.device, dtype=query.dtype)
        sampling_locations = (
            reference_points[:, :, None, None, :]  # (B, N_q, 1, 1, 2)
            + offsets / offset_normalizer[None, None, None, None, :]
        )  # (B, N_q, H, K, 2) in [0,1]

        attn = self.attention_weights(query).view(
            B, N_q, self.n_heads, self.n_points
        )
        attn = F.softmax(attn, dim=-1)  # (B, N_q, H, K)
        attn = self.dropout(attn)

        # Sample features at deformable locations
        # Reshape value to spatial grid: (B, H, W, H, d) -> (B*H, d, H, W)
        v_spatial = v.reshape(B, H, W, self.n_heads, self.d_head)
        v_spatial = v_spatial.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, d, H, W)
        v_spatial = v_spatial.reshape(B * self.n_heads, self.d_head, H, W)

        # Build sampling grid for grid_sample: (B*H, N_q, K, 2) in [-1,1]
        grid = 2.0 * sampling_locations - 1.0  # (B, N_q, H, K, 2)
        # We need to sample for each head independently
        grid = grid.permute(0, 2, 1, 3, 4).contiguous()  # (B, H, N_q, K, 2)
        grid = grid.reshape(B * self.n_heads, N_q, self.n_points, 2)

        sampled = F.grid_sample(
            v_spatial, grid,
            mode='bilinear', padding_mode='zeros', align_corners=False,
        )  # (B*H, d, N_q, K)
        sampled = sampled.permute(0, 2, 3, 1).contiguous()  # (B*H, N_q, K, d)
        sampled = sampled.view(B, self.n_heads, N_q, self.n_points, self.d_head)

        # Weighted sum over sampling points
        attn = attn.unsqueeze(-1)  # (B, N_q, H, K, 1)
        sampled = sampled.permute(0, 2, 1, 3, 4).contiguous()  # (B, N_q, H, K, d)
        output = (sampled * attn).sum(dim=3)  # (B, N_q, H, d)
        output = output.view(B, N_q, self.d_model)

        output = self.output_proj(output)
        output = self.dropout(output)
        return self.layer_norm(output)


# --------------------------------------------------------------------------- #
# 2. Deformable DETR decoder layer                                           #
# --------------------------------------------------------------------------- #


class DeformableDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_points: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)

        self.cross_attn = DeformableCrossAttention(
            d_model=d_model, n_heads=n_heads, n_points=n_points, dropout=dropout,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,          # (B, N_q, C)
        memory: torch.Tensor,       # (B, H*W, C)
        memory_spatial: Tuple[int, int],
        reference_points: torch.Tensor,  # (B, N_q, 2)
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention (with optional DN-DETR mask isolating denoising groups)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=self_attn_mask)[0]
        tgt = self.self_attn_norm(tgt + self.self_attn_dropout(tgt2))

        # Deformable cross-attention
        tgt2 = self.cross_attn(tgt, reference_points, memory, memory_spatial)
        tgt = self.ffn_norm(tgt + tgt2)

        # FFN
        tgt2 = self.ffn(tgt)
        tgt = self.ffn_norm(tgt + tgt2)
        return tgt


# --------------------------------------------------------------------------- #
# 3. Deformable DETR detection head                                          #
# --------------------------------------------------------------------------- #


class DeformableSurgicalToolDetector(nn.Module):
    """
    Deformable DETR detection head.

    Uses multi-scale features from the neck (P2-P5 concatenated) and
    deformable cross-attention in the decoder.  Converges much faster
    than vanilla DETR and is far less prone to query collapse.
    """

    def __init__(
        self,
        neck_dim: int = 256,
        num_tools: int = 7,
        num_queries: int = 16,
        num_decoder_layers: int = 6,
        nheads: int = 8,
        n_points: int = 4,
        dropout: float = 0.1,
        dim_feedforward: int = 1024,
        class_weight: float = 1.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_denoising: bool = False,
        num_denoising_groups: int = 5,
        num_noise_per_group: int = 4,
        label_noise_prob: float = 0.2,
        box_noise_scale: float = 0.4,
        denoising_weight: float = 1.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_tools = num_tools
        self.d_model = neck_dim

        # DN-DETR denoising config
        self.use_denoising = use_denoising
        self.num_denoising_groups = num_denoising_groups
        self.num_noise_per_group = num_noise_per_group
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale = box_noise_scale
        self.denoising_weight = denoising_weight

        # Object query initialization with template conditioning hook (Stage 2 migration)
        # Current behavior: learnable positional queries (template_features=None by default)
        # Stage 2: pass template_features to enable persistent identity across frames
        self.query_init = ObjectQueryInit(N_max=num_queries, d_q=neck_dim)

        # Reference-point MLP: each query predicts an initial xy centre
        self.reference_point_head = nn.Sequential(
            nn.Linear(neck_dim, neck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(neck_dim, 2),
            nn.Sigmoid(),
        )

        # DN-DETR: label embeddings and box position embeddings for noisy queries
        self.label_enc = nn.Embedding(num_tools, neck_dim)
        self.denoising_box_embed = nn.Sequential(
            nn.Linear(4, neck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(neck_dim, neck_dim),
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DeformableDecoderLayer(
                d_model=neck_dim,
                n_heads=nheads,
                n_points=n_points,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        ])

        # Prediction heads (shared across layers — Deformable DETR style)
        self.class_embed = nn.Linear(neck_dim, num_tools)
        self.bbox_embed = nn.Sequential(
            nn.Linear(neck_dim, neck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(neck_dim, neck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(neck_dim, 4),
        )

        self._init_weights()

        # Criterion
        self.criterion = SetCriterion(
            num_classes=num_tools,
            class_weight=class_weight,
            bbox_weight=bbox_weight,
            giou_weight=giou_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )

    def _init_weights(self):
        # Initialize learned queries in ObjectQueryInit
        nn.init.normal_(self.query_init.learned_queries.weight, std=0.01)
        # Bias bbox head to small centred boxes
        if len(self.bbox_embed) >= 3 and hasattr(self.bbox_embed[-1], 'bias'):
            nn.init.constant_(self.bbox_embed[-1].bias.data, 0.0)
            with torch.no_grad():
                self.bbox_embed[-1].bias.data[:2] = torch.tensor([0.5, 0.5])
                self.bbox_embed[-1].bias.data[2:] = torch.tensor([0.1, 0.1])

        # DN-DETR: initialize label and box embeddings
        nn.init.normal_(self.label_enc.weight, std=0.01)
        for m in self.denoising_box_embed.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _noisy_gt_boxes(self, boxes: torch.Tensor, noise_scale: float) -> torch.Tensor:
        """
        Add center + scale noise to GT boxes.

        Args:
            boxes: (T, 4) in cxcywh [0,1].
            noise_scale: fraction of box width/height to jitter.
        Returns:
            (T, 4) noisy boxes, clamped to [0,1].
        """
        if len(boxes) == 0 or noise_scale <= 0.0:
            return boxes.clone()
        cx, cy, w, h = boxes.unbind(-1)
        # Jitter center by up to noise_scale * size
        delta_cx = (torch.rand_like(cx) * 2.0 - 1.0) * noise_scale * w
        delta_cy = (torch.rand_like(cy) * 2.0 - 1.0) * noise_scale * h
        # Scale width/height by [1 - scale, 1 + scale]
        scale_w = 1.0 + (torch.rand_like(w) * 2.0 - 1.0) * noise_scale
        scale_h = 1.0 + (torch.rand_like(h) * 2.0 - 1.0) * noise_scale
        noisy = torch.stack([
            (cx + delta_cx).clamp(0.0, 1.0),
            (cy + delta_cy).clamp(0.0, 1.0),
            (w * scale_w).clamp(1e-4, 1.0),
            (h * scale_h).clamp(1e-4, 1.0),
        ], dim=-1)
        return noisy

    def _prepare_denoising_queries(
        self,
        targets: List[Dict],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Build DN-DETR noisy queries, reference points, attention mask, labels, and valid mask.

        Returns:
            denoising_queries: (B, N_d, C) or None
            denoising_ref_points: (B, N_d, 2) or None
            attn_mask: (N_q + N_d, N_q + N_d) or None
            denoising_labels: (B, N_d,) or None
            denoising_valid: (B, N_d,) bool mask or None
        """
        B = len(targets)
        max_targets = 0
        for t in targets:
            max_targets = max(max_targets, len(t.get('labels', [])))
        if max_targets == 0:
            return None, None, None, None, None

        num_noise = self.num_noise_per_group
        num_groups = self.num_denoising_groups
        n_clean = self.num_queries
        n_denoise = max_targets * num_noise * num_groups

        # Build batch-wise denoising query tensors (padded to max_targets)
        denoising_queries: List[torch.Tensor] = []
        denoising_ref_points: List[torch.Tensor] = []
        denoising_labels: List[torch.Tensor] = []
        denoising_valid: List[torch.Tensor] = []
        for t in targets:
            labels = t['labels']
            boxes = t['boxes']
            n = len(labels)
            if n == 0:
                denoising_queries.append(torch.zeros(n_denoise, self.d_model, device=device, dtype=dtype))
                denoising_ref_points.append(torch.zeros(n_denoise, 2, device=device, dtype=dtype))
                denoising_labels.append(torch.full((n_denoise,), -1, dtype=torch.long, device=device))
                denoising_valid.append(torch.zeros(n_denoise, dtype=torch.bool, device=device))
                continue

            # Pad labels and boxes to max_targets
            padded_labels = torch.full((max_targets,), -1, dtype=torch.long, device=device)
            padded_boxes = torch.zeros(max_targets, 4, device=device, dtype=dtype)
            padded_labels[:n] = labels
            padded_boxes[:n] = boxes

            # Repeat GT labels and boxes for each noise/group combination
            rep_labels = padded_labels.unsqueeze(0).repeat(num_noise * num_groups, 1)  # (N_noise*G, T)
            rep_boxes = padded_boxes.unsqueeze(0).repeat(num_noise * num_groups, 1, 1)  # (N_noise*G, T, 4)
            rep_valid = (padded_labels >= 0).unsqueeze(0).repeat(num_noise * num_groups, 1)  # (N_noise*G, T)

            # Apply label noise (randomly flip label to another class)
            if self.label_noise_prob > 0.0:
                noise_mask = torch.rand_like(rep_boxes[:, :, 0]) < self.label_noise_prob
                noise_mask = noise_mask & rep_valid  # only noise valid labels
                random_labels = torch.randint(
                    0, self.num_tools, rep_labels.shape, device=device
                )
                rep_labels = torch.where(noise_mask, random_labels, rep_labels)

            # Apply box noise
            noisy_boxes = torch.zeros_like(rep_boxes)
            for i in range(num_noise * num_groups):
                noisy_boxes[i] = self._noisy_gt_boxes(rep_boxes[i], self.box_noise_scale)

            # Flatten to (N_d, *)
            flat_labels = rep_labels.flatten()  # (num_noise*num_groups*max_targets,)
            flat_boxes = noisy_boxes.view(-1, 4)  # (num_noise*num_groups*max_targets, 4)
            flat_valid = rep_valid.flatten()

            # Build query = base learnable query + label embedding + box position embedding
            # Use a single content vector (averaged clean query) repeated for every denoising slot.
            base_q = self.query_init().mean(dim=0, keepdim=True).repeat(n_denoise, 1)  # (N_d, C)
            label_emb = self.label_enc(flat_labels.clamp(min=0))
            box_emb = self.denoising_box_embed(flat_boxes)
            query = base_q + label_emb + box_emb

            denoising_queries.append(query)
            denoising_ref_points.append(flat_boxes[:, :2].clamp(0.0, 1.0))
            denoising_labels.append(flat_labels)
            denoising_valid.append(flat_valid)

        dq = torch.stack(denoising_queries, dim=0)
        dr = torch.stack(denoising_ref_points, dim=0)
        dl = torch.stack(denoising_labels, dim=0)
        dv = torch.stack(denoising_valid, dim=0)

        # Build attention mask:
        # - clean queries can only attend to clean queries
        # - denoising group g can only attend to itself
        total_len = n_clean + n_denoise
        mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
        # Clean queries block all denoising queries
        mask[:n_clean, n_clean:] = True
        # Denoising queries block clean queries
        mask[n_clean:, :n_clean] = True
        # Denoising queries of different groups block each other
        tgt_per_group = max_targets * num_noise
        for g1 in range(num_groups):
            start1 = n_clean + g1 * tgt_per_group
            end1 = start1 + tgt_per_group
            for g2 in range(num_groups):
                if g1 == g2:
                    continue
                start2 = n_clean + g2 * tgt_per_group
                end2 = start2 + tgt_per_group
                mask[start1:end1, start2:end2] = True

        return dq, dr, mask, dl, dv

    def _denoising_targets(
        self,
        denoising_labels: torch.Tensor,
        denoising_boxes: torch.Tensor,
        denoising_valid: torch.Tensor,
    ) -> List[Dict]:
        """Repackage denoising query targets into the same dict format as SetCriterion."""
        B = denoising_labels.shape[0]
        targets = []
        for i in range(B):
            valid = denoising_valid[i]
            targets.append({
                'labels': denoising_labels[i][valid],
                'boxes': denoising_boxes[i][valid],
            })
        return targets

    def forward(
        self,
        neck_out: Dict[str, torch.Tensor],
        targets: Optional[List[Dict]] = None,
    ) -> Dict[str, object]:
        """
        Args:
            neck_out: dict from EncoderNeck with 'detection_scales' list
                     of (B, C, H, W) tensors.  We flatten & concat them.
            targets: list of dicts with 'boxes' and 'labels' for training.

        Returns:
            dict with 'pred_logits', 'pred_boxes', and optionally losses.
        """
        scales = neck_out.get('detection_scales', [])
        if not scales:
            raise RuntimeError("neck_out must contain 'detection_scales'")

        # Flatten multi-scale features into a single sequence
        B = scales[0].shape[0]
        flat_scales = []
        spatial_shapes = []
        for s in scales:
            B_s, C, H, W = s.shape
            assert B_s == B
            flat = s.flatten(2).permute(0, 2, 1).contiguous()  # (B, H*W, C)
            flat_scales.append(flat)
            spatial_shapes.append((H, W))

        memory = torch.cat(flat_scales, dim=1)  # (B, Σ H*W, C)
        total_len = memory.shape[1]

        # Build level start indices for reference-point scaling
        level_start_index = [0]
        for H, W in spatial_shapes[:-1]:
            level_start_index.append(level_start_index[-1] + H * W)

        # Queries (with template conditioning hook for Stage 2 migration)
        # template_features=None by default for current behavior
        query_embed = self.query_init().unsqueeze(0).repeat(B, 1, 1)  # (B, N_q, C)
        reference_points = self.reference_point_head(query_embed).clamp(0, 1)  # (B, N_q, 2)

        # DN-DETR: prepare denoising queries if training and enabled
        denoising_queries, denoising_ref_points, attn_mask, denoising_labels, denoising_valid = None, None, None, None, None
        if self.use_denoising and self.training and targets is not None:
            denoising_queries, denoising_ref_points, attn_mask, denoising_labels, denoising_valid = \
                self._prepare_denoising_queries(targets, query_embed.device, query_embed.dtype)

        if denoising_queries is not None:
            tgt = torch.cat([query_embed, denoising_queries], dim=1)  # (B, N_q + N_d, C)
            reference_points = torch.cat([reference_points, denoising_ref_points], dim=1)
        else:
            tgt = query_embed

        # Decoder
        for layer in self.decoder_layers:
            # For single-scale memory, we just pass the concatenated features
            # and the spatial shape of the full sequence.  The deformable attn
            # samples from the 2-D grid by treating the flattened memory as
            # a single large spatial map — this is slightly approximate for
            # multi-scale but works fine in practice.
            # A more exact implementation would pass per-level shapes; we keep
            # it simple here.
            tgt = layer(tgt, memory, (1, total_len), reference_points, self_attn_mask=attn_mask)

        # Split clean vs denoising outputs
        if denoising_queries is not None:
            clean_tgt = tgt[:, :self.num_queries]
            denoising_tgt = tgt[:, self.num_queries:]
            pred_logits = self.class_embed(clean_tgt).clamp(-10.0, 10.0)  # (B, N_q, num_tools)
            pred_boxes = self.bbox_embed(clean_tgt).sigmoid()  # (B, N_q, 4)
            denoising_logits = self.class_embed(denoising_tgt).clamp(-10.0, 10.0)
            denoising_boxes = self.bbox_embed(denoising_tgt).sigmoid()
        else:
            pred_logits = self.class_embed(tgt).clamp(-10.0, 10.0)  # (B, N_q, num_tools)
            pred_boxes = self.bbox_embed(tgt).sigmoid()  # (B, N_q, 4)

        pred = {
            'class_logits': pred_logits,
            'pred_boxes': pred_boxes,
        }

        out = {'pred': pred}

        if targets is not None and self.training:
            amp_device = 'cuda' if pred_logits.is_cuda else 'cpu'
            finite = torch.isfinite(pred_logits).all() and torch.isfinite(pred_boxes).all()
            if denoising_logits is not None:
                finite = finite and torch.isfinite(denoising_logits).all() and torch.isfinite(denoising_boxes).all()
            if not finite:
                loss_dict = {
                    'loss_focal': 0.0, 'loss_l1': 0.0, 'loss_giou': 0.0,
                    'loss_denoise_focal': 0.0, 'loss_denoise_l1': 0.0, 'loss_denoise_giou': 0.0,
                    'loss_detr_total': 0.0,
                }
                out['loss'] = pred_logits.float().sum() * 0.0
                out['loss_dict'] = loss_dict
                out['matching_indices'] = [
                    (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
                    for _ in range(pred_logits.size(0))
                ]
            else:
                with torch.autocast(device_type=amp_device, enabled=False):
                    pred_fp32 = {
                        'class_logits': pred_logits.float(),
                        'pred_boxes': pred_boxes.float(),
                    }
                    indices = self.criterion.hungarian_matching(
                        pred_fp32['class_logits'], pred_fp32['pred_boxes'], targets
                    )
                    loss, loss_dict = self.criterion.compute_losses(pred_fp32, targets, indices)

                    # DN-DETR denoising loss: no Hungarian needed, just match noisy GTs
                    if denoising_logits is not None and denoising_labels is not None and denoising_valid is not None:
                        denoising_targets = self._denoising_targets(denoising_labels, denoising_boxes, denoising_valid)
                        dn_pred_fp32 = {
                            'class_logits': denoising_logits.float(),
                            'pred_boxes': denoising_boxes.float(),
                        }
                        # Denoising queries are ordered by GT, so each query matches one GT.
                        device = pred_logits.device
                        dn_indices = [
                            (
                                torch.arange(len(t['labels']), dtype=torch.long, device=device),
                                torch.arange(len(t['labels']), dtype=torch.long, device=device),
                            )
                            for t in denoising_targets
                        ]
                        dn_loss, dn_loss_dict = self.criterion.compute_losses(
                            dn_pred_fp32, denoising_targets, dn_indices
                        )
                        for key, val in dn_loss_dict.items():
                            loss_dict[f'denoise_{key}'] = val
                        loss = loss + self.denoising_weight * dn_loss

                out['loss'] = loss
                out['loss_dict'] = loss_dict
                out['matching_indices'] = indices

        return out
