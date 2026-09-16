"""
ReID (Re-Identification) Head for Surgical Tool Tracking
Generates discriminative embeddings for tool re-identification across frames.

Components:
  - SupConLoss: Supervised Contrastive loss for training discriminative embeddings.
  - RoIAlignExtractor: Extracts per-tool features from spatial feature maps.
  - ReidHead: Generates embeddings; supports DETR-query mode and RoIAlign mode.
  - TemporalReIDHead, ToolTracker: Unchanged tracking utilities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).

    For a batch of N embeddings with labels, each sample i is pulled toward
    all other samples with the same label (positives) and pushed away from
    all samples with different labels (negatives).

    L_SupCon = Σ_i (-1/|P(i)|) Σ_{p∈P(i)} log[
        exp(z_i · z_p / τ) / Σ_{a≠i} exp(z_i · z_a / τ)
    ]

    Inputs must be L2-normalised embeddings.
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (N, D) L2-normalised feature vectors.
            labels:     (N,) integer class labels (same label = positive pair).
            weights:    (N,) optional per-sample weights (e.g. detection scores
                        for Discriminative Focal Loss weighting). Values in [0,1].

        Returns:
            Scalar loss tensor.
        """
        device = embeddings.device
        N = embeddings.shape[0]

        if N < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Similarity matrix (N, N) — embeddings should already be L2-normed
        sim = torch.mm(embeddings, embeddings.t()) / self.temperature  # (N, N)

        # Mask diagonal (self)
        eye = torch.eye(N, device=device, dtype=torch.bool)
        sim_no_self = sim.masked_fill(eye, float('-inf'))

        # Log-softmax denominator
        log_prob = sim_no_self - torch.logsumexp(sim_no_self, dim=1, keepdim=True)

        # Positive mask: same label, different index
        labels = labels.contiguous().view(-1, 1).to(device)  # (N, 1) - ensure on correct device
        pos_mask = torch.eq(labels, labels.t()).float()  # (N, N)
        pos_mask = pos_mask * (~eye).float()

        # Number of positives per anchor
        num_pos = pos_mask.sum(dim=1)  # (N,)
        valid = num_pos > 0  # anchors that have at least one positive

        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Mean log-prob over positives per anchor
        # Use torch.where to avoid 0 * -inf = nan on diagonal
        log_prob_pos = torch.where(pos_mask > 0, log_prob, torch.zeros_like(log_prob))
        mean_log_prob_pos = log_prob_pos.sum(dim=1) / num_pos.clamp(min=1)
        loss_per_sample = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss_per_sample = loss_per_sample[valid]

        # Apply DFL weights if provided
        if weights is not None:
            w = weights[valid]
            loss = (loss_per_sample * w).sum() / w.sum().clamp(min=1e-6)
        else:
            loss = loss_per_sample.mean()

        return loss


class RoIAlignExtractor(nn.Module):
    """
    Extracts per-tool features from a spatial feature map using RoIAlign.

    Given a (B, C, H, W) feature map and predicted bounding boxes in
    normalised [cx, cy, w, h] format, this module:
      1. Converts boxes to (x1, y1, x2, y2) pixel coordinates.
      2. Runs torchvision RoIAlign to extract output_size×output_size crops.
      3. Pools the crops to a (N_tools, C) vector.

    Works identically for both DINOv2 and V-JEPA feature maps.
    """

    def __init__(
        self,
        output_size: int = 7,
        spatial_scale: float = 1.0,
        sampling_ratio: int = 2,
        aligned: bool = True,
    ):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(
        self,
        feature_map: torch.Tensor,
        boxes_cxcywh: torch.Tensor,
        box_batch_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            feature_map: (B, C, H, W) spatial feature map.
            boxes_cxcywh: (N, 4) or (B, Q, 4) boxes in normalised [cx,cy,w,h].
            box_batch_idx: (N,) int batch indices. If None and boxes is 3-D,
                           we derive it from the batch dimension.

        Returns:
            (N, C) pooled tool features.
        """
        from torchvision.ops import roi_align as tv_roi_align

        B, C, H, W = feature_map.shape

        # Flatten (B, Q, 4) → (N, 4) and build batch index
        if boxes_cxcywh.dim() == 3:
            batch_size, Q, _ = boxes_cxcywh.shape
            boxes_flat = boxes_cxcywh.reshape(-1, 4)  # (B*Q, 4)
            batch_idx = torch.arange(batch_size, device=feature_map.device)
            batch_idx = batch_idx.unsqueeze(1).expand(-1, Q).reshape(-1)  # (B*Q,)
        else:
            boxes_flat = boxes_cxcywh  # (N, 4)
            if box_batch_idx is None:
                raise ValueError("box_batch_idx required when boxes_cxcywh is 2-D")
            batch_idx = box_batch_idx

        # Convert cx,cy,w,h → x1,y1,x2,y2 in pixel coordinates
        cx, cy, w, h = boxes_flat.unbind(-1)
        x1 = (cx - 0.5 * w) * W
        y1 = (cy - 0.5 * h) * H
        x2 = (cx + 0.5 * w) * W
        y2 = (cy + 0.5 * h) * H
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)  # (N, 4)

        # Build (N, 5) rois: [batch_idx, x1, y1, x2, y2]
        rois = torch.cat([batch_idx.float().unsqueeze(1), boxes_xyxy], dim=1)

        # RoIAlign — output: (N, C, output_size, output_size)
        roi_feats = tv_roi_align(
            feature_map,
            rois,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned,
        )

        # Pool to (N, C)
        pooled = roi_feats.mean(dim=[-2, -1])  # adaptive avg pool over H×W
        return pooled  # (N, C)


class ReidHead(nn.Module):
    """
    ReID head that generates discriminative embeddings for each detected tool.
    Used for tracking tools across frames by matching embeddings.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        embedding_dim: int = 256,
        num_classes: int = 7,
        pooling: str = 'mean',
        token_index: int = 0,
        normalize_embeddings: bool = True,
        dropout: float = 0.15,
        ce_loss_weight: float = 0.0,
        embedding_l2_weight: float = 0.0,
        supcon_temperature: float = 0.07,
        supcon_weight: float = 1.0,
        cross_consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.pooling = pooling
        self.token_index = token_index
        self.normalize_embeddings = normalize_embeddings
        self.ce_loss_weight = ce_loss_weight
        self.embedding_l2_weight = embedding_l2_weight
        self.supcon_weight = supcon_weight
        self.cross_consistency_weight = cross_consistency_weight

        # Global average pooling or token selection
        if pooling == 'token':
            self.pool = lambda x: x[:, token_index]
            pool_dim = input_dim
        elif pooling == 'mean':
            self.pool = lambda x: x.mean(dim=1)
            pool_dim = input_dim
        elif pooling == 'max':
            self.pool = lambda x: x.max(dim=1)[0]
            pool_dim = input_dim
        elif pooling == 'none':
            # Input is already pooled (e.g., from RoIAlign)
            pool_dim = input_dim
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        # Bottleneck layers
        self.embedding = nn.Linear(pool_dim, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Optional CE head (kept for backward compat; disabled by default in favour of SupCon)
        if num_classes is not None and ce_loss_weight > 0:
            self.classifier = nn.Linear(embedding_dim, num_classes, bias=True)
        else:
            self.classifier = None

        # Supervised contrastive loss
        if supcon_weight > 0:
            self.supcon_loss = SupConLoss(temperature=supcon_temperature)
        else:
            self.supcon_loss = None

        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        phantom_features: Optional[torch.Tensor] = None,
        detection_scores: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            features: (B, N, C) spatial features or (N, C) already pooled (RoIAlign output).
            labels: Optional (N,) or (B,) tool class labels for training.
            phantom_features: Optional (N, C) features from phantom feature map
                              (same shape as features after pooling). When provided,
                              joint SupCon loss is computed over real + phantom.
            detection_scores: Optional (N,) detection confidence scores for DFL weighting.

        Returns:
            dict with 'embeddings' and optionally 'logits', 'loss', 'loss_dict'.
        """
        # Pool features if sequence input
        if features.dim() == 3 and self.pooling != 'none':
            pooled = self.pool(features)  # (B, C)
        else:
            pooled = features  # (N, C) or (B, C)

        B = pooled.size(0)
        x = self.embedding(pooled.view(B, -1))

        # BN only works if B > 1
        if B > 1:
            x = self.bn(x)

        embeddings = self.dropout(x)

        # L2 normalize for metric learning
        if self.normalize_embeddings:
            embeddings_normed = F.normalize(embeddings, p=2, dim=1)
        else:
            embeddings_normed = embeddings

        result = {
            'embeddings': embeddings_normed,
            'embeddings_unnormed': embeddings,
        }

        # CE head (optional, legacy)
        if self.classifier is not None:
            logits = self.classifier(embeddings_normed)
            result['logits'] = logits

        # Compute losses in training mode
        if self.training and labels is not None:
            loss, loss_dict = self.compute_loss(
                embeddings_normed,
                embeddings,
                labels,
                phantom_features=phantom_features,
                detection_scores=detection_scores,
                logits=result.get('logits'),
            )
            result['loss'] = loss
            result['loss_dict'] = loss_dict

        return result

    def compute_loss(
        self,
        embeddings_normed: torch.Tensor,
        embeddings_unnormed: torch.Tensor,
        labels: torch.Tensor,
        phantom_features: Optional[torch.Tensor] = None,
        detection_scores: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the joint ReID loss:
          L_reid = λ_supcon * L_SupCon(real [+ phantom]) +
                   λ_ce     * L_CE(logits) +
                   λ_cross  * L_cross_consistency(real, phantom) +
                   λ_l2     * L_L2(unnormed)

        Args:
            embeddings_normed:   (N, D) L2-normalised real embeddings.
            embeddings_unnormed: (N, D) un-normalised real embeddings.
            labels:              (N,) tool class labels.
            phantom_features:    (N, D) or None — raw phantom features (will be
                                 projected and normalised inside this method).
            detection_scores:    (N,) or None — DFL weights.
            logits:              (N, num_classes) or None — CE logits.

        Returns:
            total_loss (scalar Tensor), loss_dict
        """
        device = embeddings_normed.device
        loss_dict = {}
        total_loss = torch.zeros(1, device=device).squeeze()

        # ------------------------------------------------------------------ #
        # SupCon Loss — joint real + phantom
        # ------------------------------------------------------------------ #
        # Project phantom features through the same head (once, reuse below)
        ph_emb_normed: Optional[torch.Tensor] = None
        if phantom_features is not None:
            B_ph = phantom_features.size(0)
            ph_x = self.embedding(phantom_features.view(B_ph, -1))
            if B_ph > 1:
                ph_x = self.bn(ph_x)
            ph_emb_normed = F.normalize(self.dropout(ph_x), p=2, dim=1)

        if self.supcon_loss is not None and self.supcon_weight > 0:
            # Build embedding set and corresponding labels
            all_embs = [embeddings_normed]  # real
            all_labels = [labels]
            all_weights: List[Optional[torch.Tensor]] = (
                [detection_scores] if detection_scores is not None else [None]
            )

            if ph_emb_normed is not None:
                all_embs.append(ph_emb_normed)
                all_labels.append(labels)  # same tool IDs
                all_weights.append(detection_scores)  # None if not provided

            joint_embs = torch.cat(all_embs, dim=0)       # (N or 2N, D)
            joint_labels = torch.cat(all_labels, dim=0)   # (N or 2N,)
            joint_weights: Optional[torch.Tensor] = None
            if all_weights[0] is not None:
                joint_weights = torch.cat(
                    [w for w in all_weights if w is not None], dim=0
                )

            supcon = self.supcon_loss(joint_embs, joint_labels, joint_weights)
            if torch.isnan(supcon):
                # Robustness: don't let NaN ReID kill the whole batch (and thus all learning)
                loss_dict['reid_supcon'] = 0.0
            else:
                loss_dict['reid_supcon'] = supcon.item()
                total_loss = total_loss + self.supcon_weight * supcon

        # ------------------------------------------------------------------ #
        # Cross-consistency: cosine distance between real & phantom embeddings
        # ------------------------------------------------------------------ #
        if ph_emb_normed is not None and self.cross_consistency_weight > 0:
            cross = (1 - (embeddings_normed * ph_emb_normed).sum(dim=1)).mean()
            loss_dict['reid_cross_consistency'] = cross.item()
            total_loss = total_loss + self.cross_consistency_weight * cross

        # ------------------------------------------------------------------ #
        # CE Loss (legacy / auxiliary)
        # ------------------------------------------------------------------ #
        if self.ce_loss_weight > 0 and logits is not None:
            ce_loss = F.cross_entropy(logits, labels)
            loss_dict['reid_ce_loss'] = ce_loss.item()
            total_loss = total_loss + self.ce_loss_weight * ce_loss

        # ------------------------------------------------------------------ #
        # L2 regularisation on embeddings
        # ------------------------------------------------------------------ #
        if self.embedding_l2_weight > 0:
            l2_loss = torch.norm(embeddings_unnormed, p=2, dim=1).mean()
            loss_dict['reid_l2_loss'] = l2_loss.item()
            total_loss = total_loss + self.embedding_l2_weight * l2_loss

        loss_dict['reid_total_loss'] = total_loss.item()
        return total_loss, loss_dict


class TemporalReIDHead(nn.Module):
    """
    ReID head with temporal aggregation for robust tracking.
    Aggregates embeddings across multiple frames for stability.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        embedding_dim: int = 256,
        num_classes: int = 7,
        temporal_window: int = 4,
        aggregation: str = 'avg',
        normalize_embeddings: bool = True,
        dropout: float = 0.15
    ):
        super().__init__()
        self.temporal_window = temporal_window
        self.aggregation = aggregation
        
        # Base ReID head
        self.reid = ReidHead(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            pooling='mean',
            normalize_embeddings=normalize_embeddings,
            dropout=dropout
        )
        
        # Temporal attention for weighted aggregation
        if aggregation == 'attention':
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=8,
                batch_first=True,
                dropout=dropout
            )
        
        # Temporal smoothing
        self.temporal_smooth = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self,
        current_features: torch.Tensor,
        temporal_buffer: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            current_features: (B, N, C) current frame features
            temporal_buffer: (B, T, C) previous T frame embeddings
            labels: Optional (B,) for training
        
        Returns:
            dict with aggregated embeddings
        """
        # Get current embedding
        current_out = self.reid(current_features, labels)
        current_emb = current_out['embeddings']  # (B, D)
        
        # Temporal aggregation
        if temporal_buffer is not None and temporal_buffer.size(1) > 0:
            if self.aggregation == 'avg':
                # Exponential moving average style
                alpha = torch.sigmoid(self.temporal_smooth)
                aggregated = alpha * current_emb + (1 - alpha) * temporal_buffer.mean(dim=1)
            elif self.aggregation == 'attention':
                # Attention-weighted aggregation
                all_embs = torch.cat([temporal_buffer, current_emb.unsqueeze(1)], dim=1)
                attn_out, _ = self.temporal_attn(
                    current_emb.unsqueeze(1),
                    all_embs,
                    all_embs
                )
                aggregated = attn_out.squeeze(1)
            else:
                aggregated = current_emb
        else:
            aggregated = current_emb
        
        result = {
            'embeddings': aggregated,
            'current_embedding': current_emb,
            'temporal_buffer': temporal_buffer
        }
        
        # Pass through losses from base reid
        if 'loss' in current_out:
            result['loss'] = current_out['loss']
            result['loss_dict'] = current_out['loss_dict']
        
        return result


class ToolTracker:
    """
    Simple tracker using ReID embeddings for tool tracking.
    Matches detections across frames using embedding similarity.
    """
    
    def __init__(self, max_age: int = 5, min_hits: int = 1, sim_threshold: float = 0.6):
        self.max_age = max_age
        self.min_hits = min_hits
        self.sim_threshold = sim_threshold
        self.next_id = 0
        self.tracks = {}  # track_id -> {embeddings, class, age, hits}
    
    def update(
        self,
        embeddings: torch.Tensor,
        class_logits: torch.Tensor,
        scores: torch.Tensor
    ) -> dict:
        """
        Update tracks with new detections.
        
        Args:
            embeddings: (N, D) detected tool embeddings
            class_logits: (N, num_classes) class predictions
            scores: (N,) detection confidence scores
        
        Returns:
            dict with track IDs and matched detections
        """
        if embeddings.size(0) == 0:
            # No detections - just age existing tracks
            self._age_tracks()
            return {'track_ids': [], 'matched_indices': []}
        
        # Get predicted classes
        pred_classes = class_logits.argmax(dim=1)
        
        # Compute similarity with existing tracks
        if len(self.tracks) > 0:
            track_ids = list(self.tracks.keys())
            track_embs = torch.stack([self.tracks[tid]['embedding'] for tid in track_ids])
            
            # Cosine similarity
            sim_matrix = torch.mm(embeddings, track_embs.t())  # (N, M)
            
            # Hungarian matching considering class consistency
            matches = []
            unmatched_dets = list(range(embeddings.size(0)))
            unmatched_tracks = list(range(len(track_ids)))
            
            for det_idx in range(embeddings.size(0)):
                if len(unmatched_tracks) == 0:
                    break
                
                best_track = None
                best_sim = self.sim_threshold
                
                for track_idx in unmatched_tracks:
                    tid = track_ids[track_idx]
                    sim = sim_matrix[det_idx, track_idx].item()
                    
                    # Check class consistency
                    if sim > best_sim and pred_classes[det_idx] == self.tracks[tid]['class']:
                        best_sim = sim
                        best_track = track_idx
                
                if best_track is not None:
                    matches.append((det_idx, track_ids[best_track]))
                    unmatched_dets.remove(det_idx)
                    unmatched_tracks.remove(best_track)
        else:
            matches = []
            unmatched_dets = list(range(embeddings.size(0)))
        
        # Update matched tracks
        for det_idx, track_id in matches:
            self.tracks[track_id].update({
                'embedding': embeddings[det_idx],
                'age': 0,
                'hits': self.tracks[track_id]['hits'] + 1,
                'class': pred_classes[det_idx].item(),
                'score': scores[det_idx].item()
            })
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            if scores[det_idx] > 0.5:  # Confidence threshold
                self.tracks[self.next_id] = {
                    'embedding': embeddings[det_idx],
                    'class': pred_classes[det_idx].item(),
                    'age': 0,
                    'hits': 1,
                    'score': scores[det_idx].item()
                }
                matches.append((det_idx, self.next_id))
                self.next_id += 1
        
        # Age unmatched tracks
        self._age_tracks()
        
        # Remove old tracks
        to_remove = [tid for tid, track in self.tracks.items() if track['age'] > self.max_age]
        for tid in to_remove:
            del self.tracks[tid]
        
        return {
            'track_ids': [m[1] for m in matches],
            'matched_indices': [m[0] for m in matches]
        }
    
    def _age_tracks(self):
        """Increment age of all tracks."""
        for track in self.tracks.values():
            track['age'] += 1
    
    def reset(self):
        """Reset all tracks."""
        self.next_id = 0
        self.tracks.clear()
