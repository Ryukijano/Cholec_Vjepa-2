"""
Data adapter — CholecTrack20 clip samples for the MOT pipeline.

The existing ``core_app.data.video_dataset.CholecDataset`` returns a
single clip with DETR targets at the middle frame. For the per-track
localisation loss we additionally need, for each annotated track in the
clip, a tuple ``(ref_bbox_0, ref_bbox_1, cur_bbox, cls, track_id)``
where all three frames have the track visible.

``MOTCholecDataset`` subclasses ``CholecDataset`` and builds these
per-track tuples alongside the detection / ReID targets. It uses the
``intraoperative_track_id`` key already parsed by the base class.

Clip layout (T frames):
  * frame 0                 → reference frame 0
  * frame 1                 → reference frame 1
  * frame T-1 (last)        → current frame (target of L_track)

For training we sample shorter clips (T=3 or T=4) to keep the per-track
predictor batch size manageable. For inference the stateful tracker
doesn't need ``per_track_targets`` — it uses the active-track state
internally.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..data.video_dataset import CholecDataset
from .system import PerTrackSample


def _annot_to_tracks(
    annotations: List[Dict[str, Any]], img_size: int
) -> Dict[int, Dict[str, Any]]:
    """
    Parse a frame's annotation list into a ``{track_id: info}`` dict.

    CholecTrack20 JSON format:
      - ``intraoperative_track``: int track id
      - ``tool_bbox``: [x, y, w, h] in normalised [0, 1] coords
      - ``instrument``: int class id in 1..7
      - ``occluded``: 1 if partially occluded (still visible, has bbox)
      - ``operator``: 0-3 surgeon side
      - ``phase``: surgical phase index
    """
    out: Dict[int, Dict[str, Any]] = {}
    for tool in annotations:
        # CT20 uses 'intraoperative_track'; older formats used '_id' suffix.
        tid = tool.get('intraoperative_track')
        if tid is None:
            tid = tool.get('intraoperative_track_id')
        bbox = tool.get('tool_bbox') or tool.get('bbox')
        label = tool.get('instrument') or tool.get('tool_id')
        if tid is None or bbox is None or label is None:
            continue
        # tool_bbox is already normalised [0,1]; just convert xywh -> cxcywh.
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        nw = w
        nh = h
        if nw <= 0 or nh <= 0:
            continue
        out[int(tid)] = {
            'bbox': torch.tensor([cx, cy, nw, nh], dtype=torch.float32),
            'cls': int(label) - 1,  # CholecTrack20 uses 1..7; shift to 0..6
            'occluded': int(tool.get('occluded', 0) or 0),
            'operator': int(tool.get('operator', -1) or -1),
            'phase': int(tool.get('phase', -1) or -1),
        }
    return out


class MOTCholecDataset(CholecDataset):
    """
    CholecTrack20 dataset that emits per-track reference/current tuples.

    The clip length ``clip_length`` controls the number of frames
    sampled. Frames 0 and 1 are used as references; the last frame is
    the current frame.

    **M1 persistence supervision:** When ``include_occluded=True``, tracks
    that are partially occluded (``occluded=1``) are included as hard
    positives.  When ``mine_reentry_pairs=True``, the dataset also mines
    re-entry events from ``intraoperative_track`` gaps and emits
    ``reid_pairs`` for re-association supervision.
    """

    def __init__(
        self,
        data_root,
        split: str = 'train',
        clip_length: int = 3,
        img_size: int = 336,
        training: bool = True,
        per_track_min_visible_frames: int = 3,
        include_occluded: bool = True,
        mine_reentry_pairs: bool = False,
    ):
        super().__init__(
            data_root=data_root,
            split=split,
            clip_length=clip_length,
            img_size=img_size,
            training=training,
            prediction_horizons=[1],
        )
        # Stage 1: require track in all 3 clip frames. Stage 2 / sparse pseudo-labels:
        # only require current frame; fill missing ref boxes from current (for JEPA).
        self.per_track_min_visible_frames = max(1, min(int(per_track_min_visible_frames), 3))
        self.include_occluded = include_occluded
        self.mine_reentry_pairs = mine_reentry_pairs

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video, start_idx, frame_numbers = self.clips[idx]

        current_video = self._load_frames(video, start_idx, self.clip_length)

        # Frame numbers for ref_0, ref_1, current.
        if len(frame_numbers) < 3:
            # Pad by repeating — rare edge case.
            frame_numbers = list(frame_numbers) + [frame_numbers[-1]] * (3 - len(frame_numbers))
        ref0_fn = frame_numbers[0]
        ref1_fn = frame_numbers[1]
        cur_fn = frame_numbers[-1]

        # Parse annotations per frame → track-id-indexed dicts.
        ref0_tracks = _annot_to_tracks(
            self.annotations.get(video, {}).get(str(ref0_fn), []), self.img_size
        )
        ref1_tracks = _annot_to_tracks(
            self.annotations.get(video, {}).get(str(ref1_fn), []), self.img_size
        )
        cur_tracks = _annot_to_tracks(
            self.annotations.get(video, {}).get(str(cur_fn), []), self.img_size
        )

        # DETR targets at the current (last) frame.
        cur_labels, cur_boxes = [], []
        for info in cur_tracks.values():
            cur_labels.append(info['cls'])
            cur_boxes.append(info['bbox'])

        # Per-track targets — Stage 1: all 3 frames; Stage 2 / pseudo: current + filled refs.
        per_track_samples: List[PerTrackSample] = []
        track_ids_for_reid: List[int] = []
        cls_for_reid: List[int] = []
        if self.per_track_min_visible_frames >= 3:
            track_ids = (
                set(ref0_tracks.keys()) & set(ref1_tracks.keys()) & set(cur_tracks.keys())
            )
        else:
            track_ids = set(cur_tracks.keys())

        for tid in sorted(track_ids):
            cur_info = cur_tracks[tid]
            ref0_info = ref0_tracks.get(tid, cur_info)
            ref1_info = ref1_tracks.get(tid, cur_info)
            per_track_samples.append(
                PerTrackSample(
                    batch_idx=0,  # filled in by collate_fn
                    ref_bbox_0=ref0_info['bbox'],
                    ref_bbox_1=ref1_info['bbox'],
                    cur_bbox=cur_info['bbox'],
                    cls=cur_info['cls'],
                    track_id=tid,
                    operator=cur_info.get('operator', -1),
                    phase=cur_info.get('phase', -1),
                    occluded=cur_info.get('occluded', 0),
                    visible=1,
                )
            )
            track_ids_for_reid.append(tid)
            cls_for_reid.append(cur_tracks[tid]['cls'])

        # M1: Mine re-entry pairs from intraoperative_track gaps.
        # A re-entry event = track present in cur but absent in ref0/ref1,
        # yet exists in the video's annotation history with same intraoperative_track.
        reid_pairs: List[Tuple[int, int, int]] = []  # (track_id, gap_frames, label)
        if self.mine_reentry_pairs:
            cur_tids = set(cur_tracks.keys())
            ref_tids = set(ref0_tracks.keys()) | set(ref1_tracks.keys())
            reentry_tids = cur_tids - ref_tids  # present now, absent in refs
            for tid in reentry_tids:
                # Check if this tid appeared in earlier frames of this video.
                video_annots = self.annotations.get(video, {})
                earlier_frames = [
                    fn for fn in video_annots
                    if int(fn) < int(ref0_fn)
                    and any(
                        (t.get('intraoperative_track') or t.get('intraoperative_track_id')) == tid
                        for t in video_annots[fn]
                    )
                ]
                if earlier_frames:
                    gap = int(ref0_fn) - int(earlier_frames[-1])
                    reid_pairs.append((tid, gap, cur_tracks[tid]['cls']))

        return {
            'current_video': current_video,
            'detr_targets': {
                'labels': torch.tensor(cur_labels, dtype=torch.long)
                if cur_labels
                else torch.zeros(0, dtype=torch.long),
                'boxes': torch.stack(cur_boxes)
                if cur_boxes
                else torch.zeros(0, 4, dtype=torch.float32),
            },
            'per_track_samples': per_track_samples,
            'reid_track_ids': torch.tensor(track_ids_for_reid, dtype=torch.long)
            if track_ids_for_reid
            else torch.zeros(0, dtype=torch.long),
            'reid_classes': torch.tensor(cls_for_reid, dtype=torch.long)
            if cls_for_reid
            else torch.zeros(0, dtype=torch.long),
            'video_name': video,
            'frame_idx': start_idx,
            'frame_numbers': frame_numbers,
            'reid_pairs': reid_pairs,
        }


def mot_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for ``MOTCholecDataset``.

    * ``current_video`` — stacked into (B, C, T, H, W).
    * ``detr_targets`` — kept as list-of-dicts (existing convention).
    * ``per_track_targets`` — list of length B; each is a list of
      ``PerTrackSample``. The ``batch_idx`` field is filled in here.
    * ``reid_labels`` — list of B tensors of track ids.
    """
    current_videos = torch.stack([b['current_video'] for b in batch])

    detr_targets = [b['detr_targets'] for b in batch]

    per_track_targets: List[List[PerTrackSample]] = []
    for b_idx, b in enumerate(batch):
        samples = []
        for s in b['per_track_samples']:
            # Re-emit PerTrackSample with correct batch_idx and M1 fields.
            samples.append(
                PerTrackSample(
                    batch_idx=b_idx,
                    ref_bbox_0=s.ref_bbox_0,
                    ref_bbox_1=s.ref_bbox_1,
                    cur_bbox=s.cur_bbox,
                    cls=s.cls,
                    track_id=s.track_id,
                    operator=getattr(s, 'operator', -1),
                    phase=getattr(s, 'phase', -1),
                    occluded=getattr(s, 'occluded', 0),
                    visible=getattr(s, 'visible', 1),
                )
            )
        per_track_targets.append(samples)

    reid_labels = [b['reid_classes'] for b in batch]

    return {
        'current_video': current_videos,
        'detr_targets': detr_targets,
        'per_track_targets': per_track_targets,
        'reid_labels': reid_labels,
        'video_names': [b['video_name'] for b in batch],
        'frame_indices': [b['frame_idx'] for b in batch],
        'reid_pairs': [b.get('reid_pairs', []) for b in batch],
    }
