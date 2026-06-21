"""
Track state + long-term memory bank for MOT.

A ``Track`` holds the persistent state for one tool across frames: the
filter weights ``omega`` produced by the per-track hypernetwork, the
appearance memory (EMA of ReID embeddings), bounding-box history,
visibility score, and book-keeping counters for birth/death policy.

The ``LongTermMemoryBank`` keeps recently-dead tracks around for
re-identification when a tool re-enters the field of view. Surgery
videos have an average 8.4x re-entry rate for graspers, so this is
critical for identity continuity.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import torch
import torch.nn.functional as F


# Track status enum (str for easy serialization).
STATUS_TENTATIVE = 'tentative'   # newborn, awaiting confirmation
STATUS_CONFIRMED = 'confirmed'   # actively tracking
STATUS_OCCLUDED = 'occluded'     # active but invisible
STATUS_DEAD = 'dead'             # removed from active set


@dataclass
class Track:
    """Persistent per-track state carried across frames."""

    id: int
    cls: int                                   # predicted tool class
    bbox: torch.Tensor                         # (4,) in cxcywh normalised
    omega: Optional[torch.Tensor] = None       # (C,) per-track filter weights
    mem_embedding: Optional[torch.Tensor] = None  # (D,) EMA of ReID embedding
    visibility: float = 1.0                    # [0, 1] from OccuSolver
    score: float = 0.0                         # detection confidence

    # Motion model: constant-velocity (cx, cy, w, h deltas per frame)
    velocity: Optional[torch.Tensor] = None    # (4,) cxcywh delta per frame

    # Context cues (populated from CT20 annotations or defaults)
    operator: int = -1                         # 0-3 surgeon side, -1 = unknown
    phase: int = -1                            # surgical phase, -1 = unknown

    # Book-keeping
    status: str = STATUS_TENTATIVE
    age: int = 0                               # frames since last match
    hits: int = 1                              # total matched frames
    time_since_update: int = 0                 # frames since last seen

    # Ring-buffer history (bounded to avoid unbounded memory)
    bbox_history: Deque[torch.Tensor] = field(default_factory=lambda: deque(maxlen=60))
    score_history: Deque[float] = field(default_factory=lambda: deque(maxlen=60))

    def update(
        self,
        bbox: torch.Tensor,
        score: float,
        cls: int,
        embedding: Optional[torch.Tensor] = None,
        visibility: Optional[float] = None,
        ema_alpha: float = 0.1,
        operator: Optional[int] = None,
        phase: Optional[int] = None,
    ) -> None:
        """Update state with a matched detection."""
        prev_bbox = self.bbox
        self.bbox = bbox.detach()
        self.score = float(score)
        self.cls = int(cls)
        self.age = 0
        self.hits += 1
        self.time_since_update = 0

        # Constant-velocity estimation: delta = current - previous.
        # Use EMA smoothing to avoid noise from single-frame jumps.
        if prev_bbox is not None and prev_bbox.shape == self.bbox.shape:
            instant_vel = self.bbox - prev_bbox
            if self.velocity is None:
                self.velocity = instant_vel
            else:
                self.velocity = (1.0 - ema_alpha) * self.velocity + ema_alpha * instant_vel

        if embedding is not None:
            if self.mem_embedding is None:
                self.mem_embedding = embedding.detach()
            else:
                # EMA update — keep it on same device/dtype as current memory.
                new_mem = (1.0 - ema_alpha) * self.mem_embedding + ema_alpha * embedding.detach()
                self.mem_embedding = F.normalize(new_mem, p=2, dim=-1)

        if visibility is not None:
            self.visibility = float(visibility)

        if operator is not None:
            self.operator = int(operator)
        if phase is not None:
            self.phase = int(phase)

        self.bbox_history.append(self.bbox.clone())
        self.score_history.append(self.score)

    def predict(self) -> None:
        """
        Advance the bbox by one constant-velocity step.

        Called when the track is unmatched (coasting).  The velocity is
        damped by 0.9 each frame to decay towards zero when the track
        remains unseen, preventing unbounded drift.
        """
        if self.velocity is not None:
            damping = 0.9 ** self.time_since_update
            self.bbox = self.bbox + self.velocity * damping
            # Clamp to valid range [0, 1] for normalised coordinates.
            self.bbox = self.bbox.clamp(0.0, 1.0)

    def mark_missed(self) -> None:
        """Track was unmatched this frame — predict next position."""
        self.predict()
        self.age += 1
        self.time_since_update += 1
        if self.visibility < 0.3 or self.age > 5:
            self.status = STATUS_OCCLUDED


class LongTermMemoryBank:
    """
    Memory bank for recently-dead tracks, used for re-identification when
    a tool re-enters the field of view after a long occlusion.

    Stores (track_id, embedding, cls, ttl) entries. When a new detection
    cannot be matched to an active track, the TrackManager can query this
    bank to check if it matches a recently-dead track — in which case the
    original identity is recovered.
    """

    def __init__(self, max_ttl: int = 300, max_entries: int = 64):
        self.max_ttl = max_ttl
        self.max_entries = max_entries
        self.entries: Dict[int, Dict] = {}

    def add(self, track: Track) -> None:
        """Add a dying track to the bank (only if it has a valid embedding)."""
        if track.mem_embedding is None:
            return
        self.entries[track.id] = {
            'embedding': track.mem_embedding.detach().clone(),
            'cls': int(track.cls),
            'last_bbox': track.bbox.detach().clone(),
            'velocity': track.velocity.detach().clone() if track.velocity is not None else None,
            'operator': int(track.operator),
            'phase': int(track.phase),
            'ttl': self.max_ttl,
        }
        self._prune()

    def _prune(self) -> None:
        if len(self.entries) > self.max_entries:
            # Drop entries with lowest TTL first.
            sorted_ids = sorted(self.entries.keys(), key=lambda k: self.entries[k]['ttl'])
            for k in sorted_ids[:len(self.entries) - self.max_entries]:
                del self.entries[k]

    def tick(self) -> None:
        """Age all entries and prune expired ones."""
        expired = []
        for k, entry in self.entries.items():
            entry['ttl'] -= 1
            if entry['ttl'] <= 0:
                expired.append(k)
        for k in expired:
            del self.entries[k]

    def match(
        self,
        embeddings: torch.Tensor,
        classes: torch.Tensor,
        det_boxes: Optional[torch.Tensor] = None,
        det_operators: Optional[torch.Tensor] = None,
        sim_threshold: float = 0.7,
        direction_weight: float = 0.15,
        operator_weight: float = 0.10,
    ) -> Dict[int, int]:
        """
        Match new detection embeddings against dead-track memory.

        Uses appearance (cosine sim) as the primary cue, boosted by:
          - **Direction consistency**: if the dead track had a velocity,
            predict where it would be now and give a bonus to detections
            close to that predicted location.
          - **Operator consistency**: same surgeon side gives a small bonus.

        Args:
            embeddings: (N, D) L2-normalised detection embeddings.
            classes:    (N,) predicted tool classes.
            det_boxes:  (N, 4) detection boxes (cxcywh) for direction cue.
            det_operators: (N,) operator IDs for operator cue.
            sim_threshold: minimum combined score for a match.
            direction_weight: bonus weight for predicted-location proximity.
            operator_weight: bonus weight for operator match.

        Returns:
            ``{det_idx: reused_track_id}`` mapping.
        """
        if not self.entries or embeddings.numel() == 0:
            return {}

        bank_ids = list(self.entries.keys())
        bank_embs = torch.stack([self.entries[tid]['embedding'] for tid in bank_ids]).to(embeddings.device)
        bank_cls = torch.tensor([self.entries[tid]['cls'] for tid in bank_ids], device=embeddings.device)

        # (N, M) cosine similarity (embeddings are already unit-normed).
        sim = embeddings @ bank_embs.t()

        # Zero out class-mismatched pairs.
        cls_match = (classes.view(-1, 1) == bank_cls.view(1, -1)).float()
        sim = sim * cls_match

        # Direction cue: predict where the dead track would be now and
        # give a proximity bonus to nearby detections.
        if det_boxes is not None:
            from .assoc import box_iou
            bank_bboxes = []
            for tid in bank_ids:
                entry = self.entries[tid]
                pred_bbox = entry['last_bbox']
                vel = entry.get('velocity')
                if vel is not None:
                    # Predict forward by TTL-elapsed frames (approximate).
                    elapsed = self.max_ttl - entry['ttl']
                    damping = 0.9 ** elapsed
                    pred_bbox = pred_bbox + vel * damping
                    pred_bbox = pred_bbox.clamp(0.0, 1.0)
                bank_bboxes.append(pred_bbox)
            bank_bbox_stack = torch.stack(bank_bboxes).to(embeddings.device)
            iou = box_iou(det_boxes, bank_bbox_stack)  # (N, M)
            sim = sim + direction_weight * iou

        # Operator cue: small bonus for same operator.
        if det_operators is not None:
            bank_ops = torch.tensor(
                [self.entries[tid].get('operator', -1) for tid in bank_ids],
                device=embeddings.device,
            )
            op_match = (det_operators.view(-1, 1) == bank_ops.view(1, -1)).float()
            # Only bonus when operator is known (not -1).
            op_known = (bank_ops >= 0).float().view(1, -1)
            sim = sim + operator_weight * op_match * op_known

        matched: Dict[int, int] = {}
        used_bank: set = set()
        # Greedy best-match over combined score.
        while True:
            flat = sim.flatten()
            if flat.numel() == 0:
                break
            max_val, max_idx = flat.max(dim=0)
            if max_val.item() < sim_threshold:
                break
            det_i = int(max_idx.item() // sim.size(1))
            bank_j = int(max_idx.item() % sim.size(1))
            matched[det_i] = bank_ids[bank_j]
            used_bank.add(bank_j)
            sim[det_i, :] = -1.0
            sim[:, bank_j] = -1.0

        # Remove reused entries from the bank so they don't double-match.
        for bj in used_bank:
            del self.entries[bank_ids[bj]]

        return matched
