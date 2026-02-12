#!/usr/bin/env python3
"""
SurgiTrack++ tracker:
  - Direction-aware association cost
  - 3-stage matching (high/low/recovery)
  - Multi-perspective state updates (visibility/intracorporeal/intraoperative)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

from tracker import KalmanFilter, box_iou_np


@dataclass
class PerspectiveState:
    visibility: str = "new"       # new/active/lost/removed
    intracorporeal: str = "new"   # new/active/oob/reentered/removed
    intraoperative: str = "new"   # new/active/oocv/oob/reidentified/removed


@dataclass
class TrackV2:
    track_id: int
    box: np.ndarray
    cls: int
    score: float
    reid_emb: Optional[np.ndarray] = None
    direction_bin: Optional[int] = None
    kf: KalmanFilter = field(default_factory=lambda: KalmanFilter(process_noise_scale=3.0))
    state: np.ndarray = field(default_factory=lambda: np.zeros(8))
    P: np.ndarray = field(default_factory=lambda: np.eye(8))
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    status: str = "new"
    perspective: PerspectiveState = field(default_factory=PerspectiveState)

    def __post_init__(self):
        self.state, self.P = self.kf.init(self.box)

    def predict(self):
        self.state, self.P = self.kf.predict(self.state, self.P)
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Dict):
        prev_cls = self.cls
        prev_dir = self.direction_bin

        self.state, self.P = self.kf.update(self.state, self.P, det["box"])
        self.cls = int(det["cls"])
        self.score = float(det["score"])
        self.reid_emb = det.get("reid_emb", self.reid_emb)
        self.direction_bin = det.get("direction_bin", self.direction_bin)
        self.time_since_update = 0
        self.hits += 1
        self.status = "active"

        # If direction is stable but class changed, interpret this as a new
        # intracorporeal episode through the same trocar/operator, but with
        # the same longer-term intraoperative identity (SurgiTrack idea).
        if (
            prev_dir is not None
            and self.direction_bin is not None
            and int(prev_dir) == int(self.direction_bin)
            and prev_cls is not None
            and int(prev_cls) != self.cls
        ):
            self.perspective.intracorporeal = "reentered"

    @property
    def pred_box(self) -> np.ndarray:
        return self.state[:4]


def _cosine_cost(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 1.0
    aa = a / (np.linalg.norm(a) + 1e-8)
    bb = b / (np.linalg.norm(b) + 1e-8)
    return 1.0 - float(np.dot(aa, bb))


def compute_cost_matrix_v2(
    tracks: List[TrackV2],
    detections: List[Dict],
    w_iou: float = 0.45,
    w_reid: float = 0.35,
    w_dir: float = 0.15,
    w_cls: float = 0.05,
) -> np.ndarray:
    T = len(tracks)
    D = len(detections)
    cost = np.full((T, D), 1e6, dtype=np.float32)
    for ti, t in enumerate(tracks):
        for di, d in enumerate(detections):
            det_cls = int(d["cls"])
            det_dir = d.get("direction_bin", None)

            same_dir = (
                t.direction_bin is not None
                and det_dir is not None
                and int(t.direction_bin) == int(det_dir)
            )
            # Hard reject only if both class AND direction disagree.
            # If direction is stable but class changed, we still allow
            # association (new intracorporeal track, same intraoperative).
            if t.cls != det_cls and not same_dir:
                cost[ti, di] = 1e6
                continue

            cls_mismatch = float(t.cls != det_cls)
            iou_cost = 1.0 - box_iou_np(t.pred_box, d["box"])
            reid_cost = _cosine_cost(t.reid_emb, d.get("reid_emb"))
            dir_track = t.direction_bin
            if dir_track is None or det_dir is None:
                dir_cost = 0.5
            else:
                dir_cost = 0.0 if int(dir_track) == int(det_dir) else 1.0
            cost[ti, di] = w_iou * iou_cost + w_reid * reid_cost + w_dir * dir_cost + w_cls * cls_mismatch
    return cost


class SurgicalTrackerV2:
    def __init__(
        self,
        high_thresh: float = 0.65,
        low_thresh: float = 0.1,
        match_thresh: float = 0.7,
        recovery_match_thresh: float = 0.9,
        max_lost: int = 30,
        min_hits: int = 3,
        process_noise_scale: float = 3.0,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.recovery_match_thresh = recovery_match_thresh
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.process_noise_scale = process_noise_scale
        self.tracks: List[TrackV2] = []
        self.frame_count = 0
        self._next_id = 1

    def _new_track(self, det: Dict) -> None:
        tr = TrackV2(
            track_id=self._next_id,
            box=np.asarray(det["box"], dtype=np.float32),
            cls=int(det["cls"]),
            score=float(det["score"]),
            reid_emb=det.get("reid_emb"),
            direction_bin=det.get("direction_bin"),
            kf=KalmanFilter(process_noise_scale=self.process_noise_scale),
        )
        self._next_id += 1
        self.tracks.append(tr)

    def _match(self, tracks: List[TrackV2], detections: List[Dict], thresh: float) -> Tuple[set, set]:
        if not tracks or not detections:
            return set(), set()
        c = compute_cost_matrix_v2(tracks, detections)
        if linear_sum_assignment is not None:
            rr, cc = linear_sum_assignment(c)
        else:
            rr, cc = self._greedy_match(c)
        matched_t = set()
        matched_d = set()
        for r, d in zip(rr, cc):
            if c[r, d] < thresh:
                tracks[r].update(detections[d])
                matched_t.add(r)
                matched_d.add(d)
        return matched_t, matched_d

    @staticmethod
    def _greedy_match(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = [], []
        used_rows, used_cols = set(), set()
        idx = [(cost[r, c], r, c) for r in range(cost.shape[0]) for c in range(cost.shape[1])]
        idx.sort(key=lambda x: x[0])
        for _, r, c in idx:
            if r in used_rows or c in used_cols:
                continue
            rows.append(r)
            cols.append(c)
            used_rows.add(r)
            used_cols.add(c)
        return np.array(rows), np.array(cols)

    def _update_perspectives(self, tr: TrackV2):
        """
        HBGM-inspired state transitions.

        Visibility  : new -> active -> lost -> removed
        Intracorp.  : new -> active -> oob -> reentered/removed
        Intraop.    : new -> active -> oocv -> oob -> reidentified/removed

        We treat intraoperative as the most persistent notion of identity:
        it only fully dies when the track has been missing for a long time
        (time_since_update > max_lost), i.e. the tool has almost certainly
        left both the field of view and the body.
        """
        if tr.time_since_update == 0:
            tr.perspective.visibility = "active"
            if tr.perspective.intracorporeal in ("oob", "reentered"):
                tr.perspective.intracorporeal = "reentered"
            else:
                tr.perspective.intracorporeal = "active"
            if tr.perspective.intraoperative in ("oocv", "oob", "reidentified"):
                tr.perspective.intraoperative = "reidentified"
            else:
                tr.perspective.intraoperative = "active"
        else:
            # visibility "lost" may still be active in intra perspective
            tr.perspective.visibility = "lost"
            # Cross-perspective validation:
            # - If intracorporeal is still active, treat intraoperative as OOCV (occlusion).
            # - If intracorporeal is already OOB, intraoperative moves to OOB as well.
            if tr.perspective.intracorporeal == "active":
                tr.perspective.intraoperative = "oocv"
            elif tr.perspective.intracorporeal == "oob":
                tr.perspective.intraoperative = "oob"
            else:
                tr.perspective.intraoperative = "oocv"
            if tr.time_since_update > max(2, self.max_lost // 3):
                tr.perspective.intracorporeal = "oob"
                tr.perspective.intraoperative = "oob"
            if tr.time_since_update > self.max_lost:
                tr.perspective.visibility = "removed"
                tr.perspective.intracorporeal = "removed"
                tr.perspective.intraoperative = "removed"
                tr.status = "removed"

    def update(self, detections: List[Dict]) -> List[Dict]:
        self.frame_count += 1
        for tr in self.tracks:
            tr.predict()

        high = [d for d in detections if d["score"] >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d["score"] < self.high_thresh]

        active = [t for t in self.tracks if t.status in ("new", "active")]
        lost = [t for t in self.tracks if t.status == "lost"]

        # Stage1 high conf
        mt1, md1 = self._match(active, high, self.match_thresh)
        rem_active = [active[i] for i in range(len(active)) if i not in mt1]
        rem_high = [high[i] for i in range(len(high)) if i not in md1]

        # Stage2 low conf on remaining active
        self._match(rem_active, low, self.match_thresh)

        # Stage3 recovery: remaining high vs lost (direction + reid weighted cost)
        if lost and rem_high:
            mt3, md3 = self._match(lost, rem_high, self.recovery_match_thresh)
            for i, d in enumerate(rem_high):
                if i not in md3:
                    self._new_track(d)
        else:
            for d in rem_high:
                self._new_track(d)

        # status transitions and perspective updates
        for tr in self.tracks:
            if tr.time_since_update > 0 and tr.status in ("new", "active"):
                tr.status = "lost"
            self._update_perspectives(tr)

        self.tracks = [t for t in self.tracks if t.status != "removed"]

        outputs = []
        for tr in self.tracks:
            if tr.status in ("new", "active") and (tr.hits >= self.min_hits or self.frame_count <= self.min_hits):
                outputs.append(
                    {
                        "track_id": tr.track_id,
                        "box": tr.pred_box.copy(),
                        "cls": tr.cls,
                        "score": tr.score,
                        "reid_emb": tr.reid_emb,
                        "direction_bin": tr.direction_bin,
                        "visibility_state": tr.perspective.visibility,
                        "intracorporeal_state": tr.perspective.intracorporeal,
                        "intraoperative_state": tr.perspective.intraoperative,
                    }
                )
        return outputs

    def reset(self) -> None:
        self.tracks = []
        self.frame_count = 0
        self._next_id = 1
