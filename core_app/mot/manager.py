"""
Track manager — orchestrates birth/death/update for the MOT pipeline.

Policy (tuned for CholecTrack20 per docs/multi_object_tracking_research.md):
  * Birth: tentative track is born for every unmatched detection with
    score > ``birth_score``; promoted to confirmed after
    ``min_hits`` consecutive matches.
  * Death: confirmed track is killed after ``max_age`` consecutive
    missed frames; tentative tracks die after ``max_tentative_age``.
  * Re-entry: dying confirmed tracks are moved into the long-term
    memory bank for up to ``reentry_ttl`` frames.
  * Association uses the 4-term Hungarian cost from ``assoc.py``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .assoc import compute_cost_matrix, hungarian_match
from .track import (
    LongTermMemoryBank,
    STATUS_CONFIRMED,
    STATUS_DEAD,
    STATUS_OCCLUDED,
    STATUS_TENTATIVE,
    Track,
)


class TrackManager:
    """Orchestrator for the active track set."""

    def __init__(
        self,
        birth_score: float = 0.6,
        min_hits: int = 3,
        max_age: int = 30,
        max_tentative_age: int = 5,
        reentry_ttl: int = 300,
        reentry_max_entries: int = 64,
        reid_reentry_threshold: float = 0.7,
        cost_threshold: float = 0.7,
        cost_weights: Tuple[float, float, float, float] = (0.30, 0.45, 0.15, 0.10),
        ema_alpha: float = 0.1,
    ):
        self.birth_score = birth_score
        self.min_hits = min_hits
        self.max_age = max_age
        self.max_tentative_age = max_tentative_age
        self.reentry_ttl = reentry_ttl
        self.reid_reentry_threshold = reid_reentry_threshold
        self.cost_threshold = cost_threshold
        self.w_iou, self.w_reid, self.w_cls, self.w_vis = cost_weights
        self.ema_alpha = ema_alpha

        self.tracks: Dict[int, Track] = {}
        self.dead_bank = LongTermMemoryBank(
            max_ttl=reentry_ttl, max_entries=reentry_max_entries
        )
        self._next_id = 0

    # --- public API ---------------------------------------------------- #

    def reset(self) -> None:
        """Reset all tracks — call at the start of each video."""
        self.tracks.clear()
        self.dead_bank = LongTermMemoryBank(
            max_ttl=self.reentry_ttl,
            max_entries=self.dead_bank.max_entries,
        )
        self._next_id = 0

    def active_tracks(self) -> List[Track]:
        """Return all tracks that are still alive (any non-dead status)."""
        return [t for t in self.tracks.values() if t.status != STATUS_DEAD]

    def step(
        self,
        det_boxes: torch.Tensor,
        det_scores: torch.Tensor,
        det_classes: torch.Tensor,
        det_embeddings: Optional[torch.Tensor] = None,
        det_visibilities: Optional[torch.Tensor] = None,
        det_operators: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """
        Run one association step.

        Args:
            det_boxes:        (N, 4) normalised cxcywh detection boxes.
            det_scores:       (N,) detection confidence scores in [0, 1].
            det_classes:      (N,) predicted detection classes.
            det_embeddings:   (N, D) L2-normalised ReID embeddings (optional).
            det_visibilities: (N,) OccuSolver visibility scores (optional).

        Returns:
            ``{'active_tracks': [...], 'matches': [(det_idx, track_id)],
              'new_track_ids': [...]}``
        """
        self.dead_bank.tick()

        active = self.active_tracks()
        N = det_boxes.size(0)
        device = det_boxes.device

        # Handle empty cases.
        if N == 0:
            for t in active:
                t.mark_missed()
            self._prune_dead()
            return {'active_tracks': self.active_tracks(), 'matches': [], 'new_track_ids': []}

        # Build embedding placeholder if the caller didn't give us one.
        if det_embeddings is None:
            det_embeddings = torch.zeros(N, 1, device=device)

        # --- 1. Hungarian match against active tracks. -------------- #
        if active:
            cost = compute_cost_matrix(
                det_boxes=det_boxes,
                det_embeddings=det_embeddings,
                det_classes=det_classes,
                tracks=active,
                w_iou=self.w_iou,
                w_reid=self.w_reid,
                w_cls=self.w_cls,
                w_vis=self.w_vis,
            )
            matches, unmatched_dets, unmatched_track_idx = hungarian_match(
                cost, threshold=self.cost_threshold
            )
        else:
            matches = []
            unmatched_dets = list(range(N))
            unmatched_track_idx = []

        # --- 2. Update matched tracks. ----------------------------- #
        match_pairs: List[Tuple[int, int]] = []
        for det_i, trk_i in matches:
            track = active[trk_i]
            track.update(
                bbox=det_boxes[det_i],
                score=float(det_scores[det_i].item()),
                cls=int(det_classes[det_i].item()),
                embedding=det_embeddings[det_i] if det_embeddings.size(-1) > 1 else None,
                visibility=(
                    float(det_visibilities[det_i].item())
                    if det_visibilities is not None
                    else None
                ),
                ema_alpha=self.ema_alpha,
            )
            # Promote tentative → confirmed if hits ≥ min_hits.
            if track.status == STATUS_TENTATIVE and track.hits >= self.min_hits:
                track.status = STATUS_CONFIRMED
            elif track.status == STATUS_OCCLUDED:
                track.status = STATUS_CONFIRMED
            match_pairs.append((det_i, track.id))

        # --- 3. Age unmatched tracks. ------------------------------ #
        for idx in unmatched_track_idx:
            active[idx].mark_missed()

        # --- 4. Try re-entry match from dead-bank for unmatched dets. #
        unmatched_remaining: List[int] = []
        reused_ids: List[int] = []
        if unmatched_dets:
            unmatched_embs = det_embeddings[unmatched_dets]
            unmatched_cls = det_classes[unmatched_dets]
            # Only attempt re-id if embeddings are real (D > 1).
            if unmatched_embs.size(-1) > 1:
                unmatched_boxes = det_boxes[unmatched_dets]
                unmatched_ops = (
                    det_operators[unmatched_dets] if det_operators is not None else None
                )
                reentry = self.dead_bank.match(
                    unmatched_embs, unmatched_cls,
                    det_boxes=unmatched_boxes,
                    det_operators=unmatched_ops,
                    sim_threshold=self.reid_reentry_threshold,
                )
            else:
                reentry = {}

            for local_idx, orig_idx in enumerate(unmatched_dets):
                if local_idx in reentry:
                    tid = reentry[local_idx]
                    # Resurrect the track from the bank.
                    resurrected = Track(
                        id=tid,
                        cls=int(det_classes[orig_idx].item()),
                        bbox=det_boxes[orig_idx].detach(),
                        status=STATUS_CONFIRMED,
                        hits=self.min_hits,
                    )
                    resurrected.update(
                        bbox=det_boxes[orig_idx],
                        score=float(det_scores[orig_idx].item()),
                        cls=int(det_classes[orig_idx].item()),
                        embedding=det_embeddings[orig_idx],
                        visibility=(
                            float(det_visibilities[orig_idx].item())
                            if det_visibilities is not None
                            else None
                        ),
                        ema_alpha=self.ema_alpha,
                    )
                    self.tracks[tid] = resurrected
                    match_pairs.append((orig_idx, tid))
                    reused_ids.append(tid)
                else:
                    unmatched_remaining.append(orig_idx)

        # --- 5. Birth new tracks for remaining high-score detections. # 
        new_track_ids: List[int] = []
        for det_i in unmatched_remaining:
            if float(det_scores[det_i].item()) < self.birth_score:
                continue
            tid = self._next_id
            self._next_id += 1
            new = Track(
                id=tid,
                cls=int(det_classes[det_i].item()),
                bbox=det_boxes[det_i].detach(),
                score=float(det_scores[det_i].item()),
                status=STATUS_TENTATIVE,
                hits=1,
            )
            if det_embeddings.size(-1) > 1:
                new.mem_embedding = det_embeddings[det_i].detach().clone()
            if det_visibilities is not None:
                new.visibility = float(det_visibilities[det_i].item())
            new.bbox_history.append(new.bbox.clone())
            new.score_history.append(new.score)
            self.tracks[tid] = new
            match_pairs.append((det_i, tid))
            new_track_ids.append(tid)

        # --- 6. Prune dead tracks & move them to the memory bank. -- #
        self._prune_dead()

        return {
            'active_tracks': self.active_tracks(),
            'matches': match_pairs,
            'new_track_ids': new_track_ids,
            'reused_track_ids': reused_ids,
        }

    # --- helpers ------------------------------------------------------- #

    def _prune_dead(self) -> None:
        """Move tracks that exceed age limits into the dead-bank."""
        to_kill: List[int] = []
        for tid, t in self.tracks.items():
            if t.status == STATUS_TENTATIVE and t.age > self.max_tentative_age:
                to_kill.append(tid)
            elif t.age > self.max_age:
                to_kill.append(tid)
        for tid in to_kill:
            t = self.tracks[tid]
            if t.status == STATUS_CONFIRMED or t.status == STATUS_OCCLUDED:
                self.dead_bank.add(t)
            t.status = STATUS_DEAD
            del self.tracks[tid]
