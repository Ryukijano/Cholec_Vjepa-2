"""
ByteTrack-style multi-object tracker for surgical tools.

Combines:
- Kalman filter for motion prediction
- IoU-based association (primary)
- Re-ID embedding cosine similarity (secondary)
- Multi-class awareness (same class required for match)

Adapted from ByteTrack (Zhang et al., ECCV 2022) and
SurgiTrack's HBGM association algorithm.

Track states follow SurgiTrack's multi-perspective formalization:
- New → Active → Lost → OOCV/Removed
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


class KalmanFilter:
    """Simple 2D Kalman filter for bounding box tracking.
    
    State: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]
    
    Higher process_noise_scale makes the filter trust new measurements more,
    useful for shaky laparoscopic video where motion is less predictable.
    """
    
    def __init__(self, process_noise_scale: float = 1.0):
        # State transition (constant velocity model)
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)  # velocity integration
        
        # Measurement matrix
        self.H = np.eye(4, 8)
        
        # Process noise (higher scale = trust observations more, smoother in shaky video)
        base_q = 0.01 * process_noise_scale
        self.Q = np.eye(8) * base_q
        self.Q[4:, 4:] *= 10  # velocity uncertainty ~10x position
        
        # Measurement noise
        self.R = np.eye(4) * 0.01
        
    def init(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize state from first measurement."""
        state = np.zeros(8)
        state[:4] = measurement
        P = np.eye(8)
        P[4:, 4:] *= 100  # high initial velocity uncertainty
        return state, P
    
    def predict(self, state: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state."""
        state = self.F @ state
        P = self.F @ P @ self.F.T + self.Q
        return state, P
    
    def update(self, state: np.ndarray, P: np.ndarray, 
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update state with measurement."""
        y = measurement - self.H @ state
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        state = state + K @ y
        P = (np.eye(8) - K @ self.H) @ P
        return state, P


class Track:
    """Single object track."""
    
    _next_id = 1
    
    def __init__(self, box: np.ndarray, cls: int, score: float,
                 reid_emb: Optional[np.ndarray] = None,
                 process_noise_scale: float = 1.0):
        self.track_id = Track._next_id
        Track._next_id += 1
        
        self.cls = cls
        self.score = score
        self.reid_emb = reid_emb
        self.reid_history = [reid_emb] if reid_emb is not None else []
        
        self.kf = KalmanFilter(process_noise_scale=process_noise_scale)
        self.state, self.P = self.kf.init(box)
        
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.status = 'new'  # new, active, lost, removed
        
        # History for evaluation
        self.box_history = [box.copy()]
        self.frame_history = []
    
    @property
    def box(self) -> np.ndarray:
        """Current predicted box [cx, cy, w, h]."""
        return self.state[:4]
    
    @property
    def smooth_reid_emb(self) -> Optional[np.ndarray]:
        """Exponentially smoothed Re-ID embedding."""
        if not self.reid_history:
            return None
        # EMA with alpha=0.7 (weight recent more)
        alpha = 0.7
        emb = self.reid_history[-1].copy()
        for i in range(len(self.reid_history) - 2, -1, -1):
            emb = alpha * emb + (1 - alpha) * self.reid_history[i]
        # L2 normalize
        norm = np.linalg.norm(emb) + 1e-8
        return emb / norm
    
    def predict(self):
        """Predict next state using Kalman filter."""
        self.state, self.P = self.kf.predict(self.state, self.P)
        self.age += 1
        self.time_since_update += 1
    
    def update(self, box: np.ndarray, cls: int, score: float,
               reid_emb: Optional[np.ndarray] = None):
        """Update track with new detection."""
        self.state, self.P = self.kf.update(self.state, self.P, box)
        self.cls = cls
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        self.status = 'active'
        self.box_history.append(box.copy())
        
        if reid_emb is not None:
            self.reid_history.append(reid_emb)
            # Keep last 30 embeddings
            if len(self.reid_history) > 30:
                self.reid_history = self.reid_history[-30:]
            self.reid_emb = self.smooth_reid_emb


def box_iou_np(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two cxcywh boxes."""
    # Convert to xyxy
    x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter + 1e-8
    
    return inter / union


def compute_cost_matrix(tracks: List[Track], detections: List[dict],
                        iou_weight: float = 0.5, reid_weight: float = 0.3,
                        cls_weight: float = 0.2) -> np.ndarray:
    """Compute association cost matrix.
    
    Cost = weighted combination of:
    - (1 - IoU): spatial proximity
    - (1 - cosine_sim): appearance similarity via Re-ID
    - class_mismatch: 1.0 if different class, 0.0 if same
    
    Args:
        tracks: list of active Track objects
        detections: list of dicts with 'box', 'cls', 'score', 'reid_emb'
        
    Returns:
        cost: [num_tracks, num_detections] cost matrix
    """
    T = len(tracks)
    D = len(detections)
    cost = np.full((T, D), 1e6)
    
    for t in range(T):
        for d in range(D):
            track = tracks[t]
            det = detections[d]
            
            # Class mismatch penalty (hard constraint)
            if track.cls != det['cls']:
                cost[t, d] = 1e6
                continue
            
            # IoU cost
            iou = box_iou_np(track.box, det['box'])
            iou_cost = 1 - iou
            
            # Re-ID cost
            reid_cost = 1.0  # default if no embeddings
            if track.reid_emb is not None and det.get('reid_emb') is not None:
                cos_sim = np.dot(track.reid_emb, det['reid_emb'])
                reid_cost = 1 - cos_sim
            
            cost[t, d] = iou_weight * iou_cost + reid_weight * reid_cost
    
    return cost


class SurgicalTracker:
    """ByteTrack-style tracker for surgical tools.
    
    Two-stage association:
    1. High-confidence detections matched to tracks (IoU + Re-ID)
    2. Low-confidence detections matched to remaining tracks (IoU only)
    
    Multi-perspective track states from SurgiTrack:
    - new → active → lost → removed
    - Lost tracks kept for max_lost frames for re-identification
    """
    
    def __init__(self, 
                 high_thresh: float = 0.65,
                 low_thresh: float = 0.1,
                 match_thresh: float = 0.7,
                 recovery_match_thresh: Optional[float] = None,
                 max_lost: int = 30,
                 min_hits: int = 3,
                 iou_weight: float = 0.5,
                 reid_weight: float = 0.3,
                 process_noise_scale: float = 3.0):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        # More lenient threshold for re-identifying lost tracks (lower similarity required)
        self.recovery_match_thresh = recovery_match_thresh if recovery_match_thresh is not None else 0.85
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.iou_weight = iou_weight
        self.reid_weight = reid_weight
        self.process_noise_scale = process_noise_scale
        
        self.tracks: List[Track] = []
        self.frame_count = 0
        
        # Reset track ID counter
        Track._next_id = 1
    
    def update(self, detections: List[dict]) -> List[dict]:
        """Process one frame of detections.
        
        Args:
            detections: list of dicts with keys:
                'box': np.ndarray [4] (cxcywh normalized)
                'cls': int (tool class)
                'score': float (confidence)
                'reid_emb': np.ndarray [128] (optional Re-ID embedding)
                
        Returns:
            results: list of dicts with 'track_id', 'box', 'cls', 'score'
        """
        self.frame_count += 1
        
        # Predict all tracks forward
        for track in self.tracks:
            track.predict()
        
        # Split detections by confidence
        high_dets = [d for d in detections if d['score'] >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d['score'] < self.high_thresh]
        
        # === Stage 1: Match high-confidence detections to active tracks ===
        active_tracks = [t for t in self.tracks if t.status in ('active', 'new')]
        
        if active_tracks and high_dets:
            cost = compute_cost_matrix(
                active_tracks, high_dets,
                iou_weight=self.iou_weight, reid_weight=self.reid_weight)
            
            if linear_sum_assignment is not None:
                row_ind, col_ind = linear_sum_assignment(cost)
            else:
                # Greedy fallback
                row_ind, col_ind = self._greedy_match(cost)
            
            matched_tracks = set()
            matched_dets = set()
            
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < self.match_thresh:
                    active_tracks[r].update(
                        high_dets[c]['box'], high_dets[c]['cls'],
                        high_dets[c]['score'], high_dets[c].get('reid_emb'))
                    matched_tracks.add(r)
                    matched_dets.add(c)
            
            unmatched_tracks_1 = [active_tracks[i] for i in range(len(active_tracks)) 
                                  if i not in matched_tracks]
            unmatched_dets_1 = [high_dets[i] for i in range(len(high_dets)) 
                                if i not in matched_dets]
        else:
            unmatched_tracks_1 = list(active_tracks)
            unmatched_dets_1 = list(high_dets)
        
        # === Stage 2: Match low-confidence detections to remaining tracks (IoU only) ===
        if unmatched_tracks_1 and low_dets:
            cost = compute_cost_matrix(
                unmatched_tracks_1, low_dets,
                iou_weight=1.0, reid_weight=0.0)  # IoU only for low-conf
            
            if linear_sum_assignment is not None:
                row_ind, col_ind = linear_sum_assignment(cost)
            else:
                row_ind, col_ind = self._greedy_match(cost)
            
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < self.match_thresh:
                    unmatched_tracks_1[r].update(
                        low_dets[c]['box'], low_dets[c]['cls'],
                        low_dets[c]['score'], low_dets[c].get('reid_emb'))
        
        # === Stage 3: Match unmatched high-conf dets to lost tracks (Re-ID recovery) ===
        lost_tracks = [t for t in self.tracks if t.status == 'lost']
        if lost_tracks and unmatched_dets_1:
            cost = compute_cost_matrix(
                lost_tracks, unmatched_dets_1,
                iou_weight=0.2, reid_weight=0.8)  # Re-ID dominant for recovery
            
            if linear_sum_assignment is not None:
                row_ind, col_ind = linear_sum_assignment(cost)
            else:
                row_ind, col_ind = self._greedy_match(cost)
            
            recovered_dets = set()
            thresh = self.recovery_match_thresh  # more lenient for recovery
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < thresh:
                    lost_tracks[r].update(
                        unmatched_dets_1[c]['box'], unmatched_dets_1[c]['cls'],
                        unmatched_dets_1[c]['score'], unmatched_dets_1[c].get('reid_emb'))
                    recovered_dets.add(c)
            
            # Remaining unmatched detections → new tracks
            for i, det in enumerate(unmatched_dets_1):
                if i not in recovered_dets:
                    self._init_track(det)
        else:
            # All unmatched high-conf detections → new tracks
            for det in unmatched_dets_1:
                self._init_track(det)
        
        # === Update track states ===
        for track in self.tracks:
            if track.time_since_update > 0:
                if track.status in ('active', 'new'):
                    track.status = 'lost'
        
        # Remove old lost tracks
        self.tracks = [t for t in self.tracks 
                       if not (t.status == 'lost' and t.time_since_update > self.max_lost)]
        
        # === Output active/new tracks ===
        results = []
        for track in self.tracks:
            if track.status in ('active', 'new') and (track.hits >= self.min_hits or self.frame_count <= self.min_hits):
                results.append({
                    'track_id': track.track_id,
                    'box': track.box.copy(),
                    'cls': track.cls,
                    'score': track.score,
                    'reid_emb': track.reid_emb,
                })
        
        return results
    
    def _init_track(self, det: dict):
        """Initialize a new track from detection."""
        track = Track(
            box=det['box'],
            cls=det['cls'],
            score=det['score'],
            reid_emb=det.get('reid_emb'),
            process_noise_scale=self.process_noise_scale,
        )
        self.tracks.append(track)
    
    def _greedy_match(self, cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Greedy matching fallback when scipy not available."""
        rows, cols = [], []
        used_rows, used_cols = set(), set()
        
        # Flatten and sort by cost
        T, D = cost.shape
        indices = [(cost[t, d], t, d) for t in range(T) for d in range(D)]
        indices.sort()
        
        for _, t, d in indices:
            if t not in used_rows and d not in used_cols:
                rows.append(t)
                cols.append(d)
                used_rows.add(t)
                used_cols.add(d)
        
        return np.array(rows), np.array(cols)
    
    def reset(self):
        """Reset tracker state for new video."""
        self.tracks = []
        self.frame_count = 0
        Track._next_id = 1
