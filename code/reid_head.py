"""
Re-ID Embedding Head for surgical tool tracking.

Produces 128-D embeddings per detection for track association.
Trained with triplet/contrastive loss using CholecTrack20 track IDs.

Major anti-collapse additions:
  1. EmbeddingMemoryBank (default 8192): FIFO of (embedding, track_id), detached.
     Enqueue each step; both losses use bank for extra negatives (~8000+ vs ~56).
  2. Batch-Hard Triplet + Semi-Hard Mining: hardest positive, semi-hard negative
     (closest neg still farther than hardest pos); fallback to hardest neg.
     Margin 0.7 (larger helps avoid collapse). Searches live batch + memory bank.
  3. Temperature-Warmed Contrastive: temp 0.15 (epoch 0) → 0.07 over 5 epochs.
     Bank entries in softmax denominator.
  4. Deeper ReIDHead: 1024→512→256→128 with BN+GELU; final BN then L2.
     Batched MLPs for all detections.

Inspired by: SurgiTrack, FairMOT, MoCo, ByteTrack.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------------------------------------------------------------------------
# Embedding Memory Bank
# ---------------------------------------------------------------------------
class EmbeddingMemoryBank:
    """Fixed-size FIFO queue of past embeddings + track IDs.

    At each training step the current batch embeddings are enqueued
    (detached, so no gradient flows through the bank).  When computing
    contrastive / triplet losses the bank entries are concatenated with
    the live batch, providing many more negatives without extra GPU
    forward passes.

    Typical size: 4096-16384 entries (tiny memory footprint for 128-D).
    """

    def __init__(self, capacity: int = 8192, embed_dim: int = 128, device: str = "cuda"):
        self.capacity = capacity
        self.embed_dim = embed_dim
        self.device = device
        self.embeddings = torch.zeros(capacity, embed_dim, device=device)
        self.track_ids = torch.full((capacity,), -1, dtype=torch.long, device=device)
        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def enqueue(self, embs: torch.Tensor, tids: torch.Tensor):
        """Push a batch of embeddings into the bank (FIFO)."""
        embs = embs.detach()
        tids = tids.detach().to(self.device)
        n = embs.shape[0]
        if n == 0:
            return
        if n >= self.capacity:
            # just keep the last `capacity` entries
            embs = embs[-self.capacity:]
            tids = tids[-self.capacity:]
            n = self.capacity
        end = self.ptr + n
        if end <= self.capacity:
            self.embeddings[self.ptr:end] = embs
            self.track_ids[self.ptr:end] = tids
        else:
            first = self.capacity - self.ptr
            self.embeddings[self.ptr:] = embs[:first]
            self.track_ids[self.ptr:] = tids[:first]
            rest = n - first
            self.embeddings[:rest] = embs[first:]
            self.track_ids[:rest] = tids[first:]
            self.full = True
        self.ptr = (self.ptr + n) % self.capacity
        if end >= self.capacity:
            self.full = True

    def get(self):
        """Return (embeddings, track_ids) currently stored."""
        if self.full:
            return self.embeddings.clone(), self.track_ids.clone()
        if self.ptr == 0:
            return torch.empty(0, self.embed_dim, device=self.device), \
                   torch.empty(0, dtype=torch.long, device=self.device)
        return self.embeddings[:self.ptr].clone(), self.track_ids[:self.ptr].clone()

    def size(self):
        return self.capacity if self.full else self.ptr


# ---------------------------------------------------------------------------
# Re-ID Head  (deeper projector + BN)
# ---------------------------------------------------------------------------
class ReIDHead(nn.Module):
    """Re-identification embedding head for tool tracking.

    Deeper projector path with BatchNorm to prevent collapse:
        encoder_features [1024-D]
          -> MLP 1024->512->256->128 with BN+GELU
          -> BatchNorm1d -> L2-normalized embedding [128-D]
    """

    def __init__(self, embed_dim: int = 1024, hidden_dim: int = 512,
                 reid_dim: int = 128, grid_size: int = 14, dropout: float = 0.1):
        super().__init__()
        self.grid_size = grid_size
        self.reid_dim = reid_dim

        mid_dim = hidden_dim // 2  # 256

        # Appearance branch: deeper MLP with BN
        self.appearance_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, reid_dim),
        )

        # Motion branch: deeper MLP with BN
        self.motion_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, reid_dim),
        )

        # Fusion: combine appearance + motion
        self.fusion = nn.Sequential(
            nn.Linear(reid_dim * 2, reid_dim),
            nn.BatchNorm1d(reid_dim),
            nn.GELU(),
        )

        # Box position encoding (normalized cx, cy, w, h -> reid_dim)
        self.box_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, reid_dim),
        )

        # Final projection with position info + BN before L2 norm
        self.final_proj = nn.Sequential(
            nn.Linear(reid_dim * 2, reid_dim),
            nn.BatchNorm1d(reid_dim),
        )

    def _roi_pool(self, spatial_tokens: torch.Tensor, box_cxcywh: torch.Tensor) -> torch.Tensor:
        """Pool encoder tokens inside a bounding box."""
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

        idx = torch.tensor(indices, device=spatial_tokens.device, dtype=torch.long)
        return spatial_tokens[idx].mean(dim=0)

    def forward(self, full_tokens: torch.Tensor, boxes: List[torch.Tensor],
                return_per_detection: bool = True) -> List[torch.Tensor]:
        """Compute Re-ID embeddings for all detections.

        Args:
            full_tokens: [B, T*N, D] or [B, T, N, D] encoder output
            boxes: list of [M_i, 4] detection boxes per image (cxcywh)

        Returns:
            embeddings: list of [M_i, reid_dim] L2-normalized embeddings per image
        """
        device = full_tokens.device if not isinstance(full_tokens, list) else full_tokens[0].device
        if isinstance(full_tokens, list):
            full_tokens = torch.stack(full_tokens) if all(
                t.shape == full_tokens[0].shape for t in full_tokens) else full_tokens

        if isinstance(full_tokens, list):
            B = len(full_tokens)
        else:
            B = full_tokens.shape[0]

        N_spatial = self.grid_size * self.grid_size  # 196

        # --- Collect per-detection raw features in flat lists, then batch MLP ---
        app_feats_list: List[torch.Tensor] = []
        mot_feats_list: List[torch.Tensor] = []
        box_feats_list: List[torch.Tensor] = []
        counts: List[int] = []  # detections per image

        for b in range(B):
            b_boxes = boxes[b]
            if b_boxes.numel() == 0:
                counts.append(0)
                continue

            sample_tokens = full_tokens[b] if not isinstance(full_tokens, list) else full_tokens[b]

            # Reshape to [T, N, D]
            if sample_tokens.dim() == 2:
                T = sample_tokens.shape[0] // N_spatial
                D = sample_tokens.shape[1]
                sample_3d = sample_tokens.view(T, N_spatial, D)
            elif sample_tokens.dim() == 3:
                sample_3d = sample_tokens
                T = sample_3d.shape[0]
            else:
                sample_3d = sample_tokens.unsqueeze(0)
                T = 1

            last_frame = sample_3d[-1]  # [196, D]
            motion_feats = (sample_3d[-1] - sample_3d[-2]) if T > 1 else torch.zeros_like(last_frame)

            M = b_boxes.shape[0]
            for i in range(M):
                app_feats_list.append(self._roi_pool(last_frame, b_boxes[i]))
                mot_feats_list.append(self._roi_pool(motion_feats, b_boxes[i]))
                box_feats_list.append(b_boxes[i])
            counts.append(M)

        total_dets = sum(counts)
        if total_dets == 0:
            return [torch.empty(0, self.reid_dim, device=device) for _ in range(B)]

        # Stack and run through batched MLPs (BN needs batch dim)
        app_batch = torch.stack(app_feats_list)   # [N_total, D]
        mot_batch = torch.stack(mot_feats_list)   # [N_total, D]
        box_batch = torch.stack(box_feats_list)   # [N_total, 4]

        app_emb = self.appearance_mlp(app_batch)  # [N_total, reid_dim]
        mot_emb = self.motion_mlp(mot_batch)      # [N_total, reid_dim]
        fused = self.fusion(torch.cat([app_emb, mot_emb], dim=-1))  # [N_total, reid_dim]
        box_emb = self.box_encoder(box_batch)     # [N_total, reid_dim]
        final = self.final_proj(torch.cat([fused, box_emb], dim=-1))  # [N_total, reid_dim]
        final = F.normalize(final, p=2, dim=-1)

        # Split back per image
        all_embeddings = []
        offset = 0
        for c in counts:
            if c == 0:
                all_embeddings.append(torch.empty(0, self.reid_dim, device=device))
            else:
                all_embeddings.append(final[offset:offset + c])
                offset += c

        return all_embeddings


# ---------------------------------------------------------------------------
# Batch-Hard Triplet Loss with Semi-Hard Negative Mining
# ---------------------------------------------------------------------------
class TripletLoss(nn.Module):
    """Batch-hard triplet loss with optional semi-hard fallback.

    For each anchor:
      - Hardest positive: same track, max distance
      - Semi-hard negative: different track, closest negative that is
        still farther than the hardest positive (if none, fall back to
        hardest negative overall)

    Supports an optional memory bank: pass (bank_embs, bank_tids) to
    expand the negative pool without extra forward passes.
    """

    def __init__(self, margin: float = 0.7):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, track_ids: torch.Tensor,
                bank_embs: Optional[torch.Tensor] = None,
                bank_tids: Optional[torch.Tensor] = None) -> torch.Tensor:
        N = embeddings.shape[0]
        if N < 2:
            return torch.tensor(0., device=embeddings.device, requires_grad=True)

        track_ids = track_ids.to(embeddings.device)

        # Optionally augment with memory bank entries (no grad)
        if bank_embs is not None and bank_embs.shape[0] > 0:
            aug_embs = torch.cat([embeddings, bank_embs.to(embeddings.device)], dim=0)
            aug_tids = torch.cat([track_ids, bank_tids.to(embeddings.device)], dim=0)
        else:
            aug_embs = embeddings
            aug_tids = track_ids

        M = aug_embs.shape[0]

        # Pairwise distances: anchors (N) vs all (M)
        dist = torch.cdist(embeddings, aug_embs, p=2)  # [N, M]

        # Masks
        anchor_ids = track_ids.unsqueeze(1)      # [N, 1]
        all_ids = aug_tids.unsqueeze(0)           # [1, M]
        pos_mask = (anchor_ids == all_ids)        # [N, M]
        neg_mask = (anchor_ids != all_ids)        # [N, M]

        # Remove self-pairs (first N columns correspond to live batch)
        eye_pad = torch.zeros(N, M, dtype=torch.bool, device=embeddings.device)
        eye_pad[:, :N] = torch.eye(N, dtype=torch.bool, device=embeddings.device)
        pos_mask = pos_mask & ~eye_pad

        loss = torch.tensor(0., device=embeddings.device)
        count = 0

        for i in range(N):
            if pos_mask[i].sum() == 0:
                continue

            # Hardest positive
            hp_dist = dist[i][pos_mask[i]].max()

            # Semi-hard negatives: neg dist > hp_dist but as close as possible
            neg_dists = dist[i][neg_mask[i]]
            semi_hard = neg_dists[neg_dists > hp_dist]
            if semi_hard.numel() > 0:
                hn_dist = semi_hard.min()
            else:
                # Fall back to hardest negative overall
                hn_dist = neg_dists.min()

            triplet = F.relu(hp_dist - hn_dist + self.margin)
            loss = loss + triplet
            count += 1

        return loss / max(count, 1)


# ---------------------------------------------------------------------------
# Contrastive Loss with Temperature Warmup + Memory Bank
# ---------------------------------------------------------------------------
class TrackContrastiveLoss(nn.Module):
    """InfoNCE-style contrastive loss with temperature warmup.

    Temperature schedule (linear warmup then hold):
      epoch 0:   temp_start  (warm / soft -> prevents early collapse)
      epoch W:   temp_final  (sharp -> discriminative)

    Supports memory bank: bank entries are included as extra negatives
    in the denominator of the softmax (no positive matching against them).
    """

    def __init__(self, temp_start: float = 0.15, temp_final: float = 0.07,
                 warmup_epochs: int = 5):
        super().__init__()
        self.temp_start = temp_start
        self.temp_final = temp_final
        self.warmup_epochs = max(warmup_epochs, 1)
        self._current_temp = temp_start

    def set_epoch(self, epoch: int):
        """Call at the start of each epoch to update temperature."""
        if epoch >= self.warmup_epochs:
            self._current_temp = self.temp_final
        else:
            frac = epoch / self.warmup_epochs
            self._current_temp = self.temp_start + (self.temp_final - self.temp_start) * frac

    @property
    def temperature(self):
        return self._current_temp

    def forward(self, embeddings: torch.Tensor, track_ids: torch.Tensor,
                bank_embs: Optional[torch.Tensor] = None,
                bank_tids: Optional[torch.Tensor] = None) -> torch.Tensor:
        N = embeddings.shape[0]
        if N < 2:
            return torch.tensor(0., device=embeddings.device, requires_grad=True)

        track_ids = track_ids.to(embeddings.device)
        temp = self._current_temp

        # Positive mask (within live batch only)
        ids = track_ids.unsqueeze(0)
        pos_mask = (ids == ids.T).float()
        pos_mask.fill_diagonal_(0)

        if pos_mask.sum() == 0:
            return torch.tensor(0., device=embeddings.device, requires_grad=True)

        # Similarity: live-batch anchors vs live-batch
        sim_batch = torch.mm(embeddings, embeddings.T) / temp  # [N, N]

        # If memory bank available, add bank similarities as extra negatives
        if bank_embs is not None and bank_embs.shape[0] > 0:
            sim_bank = torch.mm(embeddings, bank_embs.T.to(embeddings.device)) / temp  # [N, K]
            # All bank entries are treated as negatives
            all_sim = torch.cat([sim_batch, sim_bank], dim=1)  # [N, N+K]
            logits_mask = torch.cat([
                1 - torch.eye(N, device=embeddings.device),
                torch.ones(N, bank_embs.shape[0], device=embeddings.device)
            ], dim=1)
        else:
            all_sim = sim_batch
            logits_mask = 1 - torch.eye(N, device=embeddings.device)

        # Numerically stable log-softmax
        max_sim = all_sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(all_sim - max_sim) * logits_mask
        log_prob = (sim_batch - max_sim) - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-prob over positive pairs
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)

        has_pos = (pos_mask.sum(dim=1) > 0).float()
        loss = -(mean_log_prob * has_pos).sum() / (has_pos.sum() + 1e-8)

        return loss
