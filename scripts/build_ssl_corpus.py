"""
Build the leak-free SSL corpus for GOT-JEPA Stage 2 pretraining.

This script performs three things:

  1. Runs the Stage-1 supervised Surgical MOT model on every Cholec80 video
     (minus the 7 that overlap CT20 val/test) and writes per-video pseudo
     annotations in the CholecTrack20 JSON format.

  2. Lays out a unified directory that MIRRORS the CT20 layout, so the
     existing ``MOTCholecDataset`` can consume both real and pseudo
     annotations without changes:

         <out_root>/Training/<video>/Frames/<frame_number>.png    (symlinks)
         <out_root>/Training/<video>/<video>.json                 (annotations)
         <out_root>/Validation/<VID>/...                          (CT20 val, symlinked)

  3. Verifies that no SSL video overlaps CT20 val/test via
     ``core_app.data.splits.verify_no_leak``.

The output corpus is consumed by the Stage-2 SSL config:

    configs/train_mot/dinov2/cholec80-ct20-stage2-jepa-pretrain.yaml

Usage:

    python -m scripts.build_ssl_corpus \\
        --stage1_checkpoint outputs/mot/cholec20-stage1-supervised/best.pth.tar \\
        --stage1_config configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml \\
        --out_root data/ssl_corpus \\
        --device cuda:0 \\
        --score_threshold 0.5

Notes
-----
- Pseudo-track-IDs are assigned *per video* by the TrackManager. They are
  globally unique inside a video but re-used across videos. That is fine
  for Stage 2 SSL because the per-track loss only associates a track with
  its own reference frames, never cross-video.
- Cholec80 frames are stored as ``videoNN_XXXXXX.png``. We symlink them
  into ``Frames/XXXXXX.png`` so ``MOTCholecDataset._build_clip_index`` can
  parse integer frame numbers without changes.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate corrupted PNGs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_app.data.paths import default_ssl_corpus_root  # noqa: E402
from core_app.data.splits import (  # noqa: E402
    CT20_TRAIN, build_ssl_split, verify_no_leak,
)
from core_app.mot.trainer import build_model_from_config  # noqa: E402


_LOG = logging.getLogger('build_ssl_corpus')


# --------------------------------------------------------------------------- #
# Inference helpers                                                            #
# --------------------------------------------------------------------------- #


def _log_detector_stack(cfg: dict, checkpoint: Path, model) -> None:
    """Log which Stage-1 stack is used for pseudo-labeling (not a separate detector)."""
    m = cfg.get('model', {})
    d = cfg.get('detr', {})
    t = cfg.get('tracker', {})
    detr_cls = type(model.detr).__name__
    head = 'Deformable DETR' if d.get('use_deformable_detr') or 'Deformable' in detr_cls else 'vanilla DETR'
    _LOG.info(
        'Pseudo-label stack (full SurgicalMOTSystem, mode=infer):\n'
        f'  checkpoint     : {checkpoint}\n'
        f'  encoder        : {m.get("encoder_type", "?")} {m.get("model_name", "?")} (frozen)\n'
        f'  detector head  : {head} ({detr_cls})\n'
        f'    num_queries  : {d.get("num_queries", "?")} | num_tools: {d.get("num_tools", "?")} | '
        f'decoder_layers: {d.get("num_decoder_layers", "?")}\n'
        f'  DETR keep thr  : {model.tracker_infer_score_threshold} (det scores above this)\n'
        f'  TrackManager   : birth_score={model.track_manager.birth_score}, '
        f'min_hits={model.track_manager.min_hits}, max_age={model.track_manager.max_age}'
    )


def _apply_pseudo_label_tracker_overrides(
    model,
    infer_score_threshold: float,
    birth_score: float | None,
    tracker_min_hits: int,
) -> None:
    """Relax tracker gates so weak Stage-1 DETR can still birth tracks on Cholec80."""
    model.tracker_infer_score_threshold = float(infer_score_threshold)
    model.track_manager.birth_score = float(
        infer_score_threshold if birth_score is None else birth_score
    )
    model.track_manager.min_hits = max(1, int(tracker_min_hits))


def _load_stage1_model(
    stage1_config: Path,
    stage1_checkpoint: Path,
    device: torch.device,
    infer_score_threshold: float,
    birth_score: float | None,
    tracker_min_hits: int,
):
    with open(stage1_config, 'r') as f:
        cfg = yaml.safe_load(f)
    model = build_model_from_config(cfg)
    ckpt = torch.load(stage1_checkpoint, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        _LOG.warning(f'Missing keys when loading Stage 1: {len(missing)}')
    if unexpected:
        _LOG.warning(f'Unexpected keys when loading Stage 1: {len(unexpected)}')
    _apply_pseudo_label_tracker_overrides(
        model, infer_score_threshold, birth_score, tracker_min_hits,
    )
    model.eval().to(device)
    _log_detector_stack(cfg, stage1_checkpoint, model)
    return model, cfg


def _build_transform(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _collect_video_frames(video_dir: Path) -> List[Path]:
    """Return frame files sorted by frame number parsed from the filename."""
    exts = ('.png', '.jpg', '.jpeg')
    frames = [f for f in video_dir.iterdir() if f.suffix.lower() in exts]
    frames.sort(key=lambda p: _parse_frame_number(p.name))
    return frames


def _parse_frame_number(filename: str) -> int:
    """Extract the trailing integer from a filename like video02_000005.png
    or 006825.png."""
    stem = Path(filename).stem
    # Take the trailing run of digits.
    digits = []
    for c in reversed(stem):
        if c.isdigit():
            digits.append(c)
        else:
            break
    if not digits:
        raise ValueError(f'Cannot parse frame number from {filename}')
    return int(''.join(reversed(digits)))


def _parse_class_streak_map(raw: str | None) -> Dict[int, int]:
    """Parse optional JSON map from tool class to streak threshold."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON for --min_track_streak_by_class: {e}') from e
    if not isinstance(parsed, dict):
        raise ValueError(
            'Expected a JSON object for --min_track_streak_by_class, for example {"0": 2, "1": 3}'
        )

    out: Dict[int, int] = {}
    for raw_cls, raw_thresh in parsed.items():
        cls_key = int(raw_cls)
        thr = int(raw_thresh)
        if thr <= 0:
            raise ValueError('Track streak thresholds must be positive integers.')
        out[cls_key] = thr
    return out


def _assign_frame_annotations(
    annotations: Dict[str, List[Dict]],
    frames: List[Path],
    start_idx: int,
    end_idx: int,
    tools: List[Dict],
) -> None:
    """Write the same tool list for every frame index in [start_idx, end_idx]."""
    if not tools or start_idx > end_idx:
        return
    for idx in range(start_idx, end_idx + 1):
        fn = str(_parse_frame_number(frames[idx].name))
        annotations[fn] = tools


def _load_transform_frame(
    path: Path,
    transform: T.Compose,
    img_size: int,
    fallback: Image.Image | None,
) -> Tuple[Image.Image | None, torch.Tensor]:
    try:
        img = Image.open(path).convert('RGB')
    except (OSError, struct.error):
        img = fallback
    if img is None:
        img = fallback if fallback is not None else Image.new('RGB', (img_size, img_size))
    return img, transform(img)


@torch.no_grad()
def _pseudo_label_video(
    model,
    frames: List[Path],
    img_size: int,
    clip_length: int,
    device: torch.device,
    score_threshold: float,
    min_track_streak: int,
    min_track_total_hits: int,
    min_track_streak_by_class: Optional[Dict[int, int]] = None,
    window_stride: int = 2,
    use_amp: bool = True,
    infer_mode: str = 'infer_fast',
    prefetch_workers: int = 4,
) -> Dict[str, List[Dict]]:
    """
    Run Stage-1 inference over a video and return CT20-style pseudo annotations.

    Frames are processed in a sliding window of ``clip_length``. The
    ``TrackManager`` persists state across calls (it lives on the model) so
    track IDs are coherent across the whole video.
    """
    transform = _build_transform(img_size)

    # Reset the track manager for this video so pseudo-track-IDs restart at 1.
    model.track_manager.tracks.clear()
    if hasattr(model.track_manager, 'dead_bank'):
        model.track_manager.dead_bank.entries.clear()
    if hasattr(model.track_manager, 'next_id'):
        model.track_manager.next_id = 1

    annotations: Dict[str, List[Dict]] = {}
    track_streak: Dict[int, int] = {}
    track_total_hits: Dict[int, int] = {}
    prev_active_ids: set[int] = set()
    class_streak_map = min_track_streak_by_class or {}
    min_track_streak = max(1, int(min_track_streak))
    min_track_total_hits = max(1, int(min_track_total_hits))

    # We need at least `clip_length` frames for the MOT system. Use a
    # sliding window of size clip_length with step 1, and only emit
    # annotations for the *last* (current) frame of each window.
    if len(frames) < clip_length:
        _LOG.warning(f'Video has only {len(frames)} frames (< {clip_length}); skipping')
        return annotations

    def _safe_open(path: Path, fallback: Image.Image | None = None) -> Image.Image | None:
        """Open image, returning fallback if it is truncated/empty."""
        try:
            return Image.open(path).convert('RGB')
        except (OSError, struct.error) as e:
            _LOG.warning(f'Corrupt frame skipped: {path.name} ({e})')
            return fallback

    # Prime a rolling buffer of transformed frames.
    buffer: List[torch.Tensor] = []
    last_valid: Image.Image | None = None
    for f in frames[:clip_length]:
        img = _safe_open(f)
        if img is None:
            # If first frames are corrupt, use a black placeholder.
            img = last_valid if last_valid is not None else Image.new('RGB', (img_size, img_size))
        last_valid = img
        buffer.append(transform(img))

    window_stride = max(1, int(window_stride))
    amp_enabled = use_amp and device.type == 'cuda'
    n_windows = len(frames) - clip_length + 1
    window_starts = range(0, n_windows, window_stride)
    prefetch_workers = max(0, int(prefetch_workers))

    def _build_buffer_at_start(
        start: int, lv_in: Image.Image | None,
    ) -> Tuple[List[torch.Tensor], Image.Image | None]:
        buf: List[torch.Tensor] = []
        lv = lv_in
        for fi in range(start, start + clip_length):
            img, lv = _load_transform_frame(frames[fi], transform, img_size, lv)
            buf.append(img)
        return buf, lv

    executor: ThreadPoolExecutor | None = None
    if prefetch_workers > 0 and window_stride > 1:
        executor = ThreadPoolExecutor(max_workers=prefetch_workers)

    last_tools: List[Dict] = []
    prev_labeled_end: int | None = None
    pending_buffer: object | None = None

    for window_start in window_starts:
        if window_start == 0:
            pass
        elif window_stride > 1:
            if pending_buffer is not None:
                buffer, last_valid = pending_buffer.result()  # type: ignore[union-attr]
            else:
                buffer, last_valid = _build_buffer_at_start(window_start, last_valid)
        else:
            fi = window_start + clip_length - 1
            next_img = _safe_open(frames[fi], fallback=last_valid)
            if next_img is None:
                next_img = last_valid if last_valid is not None else Image.new('RGB', (img_size, img_size))
            else:
                last_valid = next_img
            buffer = buffer[1:] + [transform(next_img)]

        clip = torch.stack(buffer).permute(1, 0, 2, 3).unsqueeze(0).to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
            out = model(current_video=clip, mode=infer_mode)

        tools_in_frame: List[Dict] = []
        current_active_ids: set[int] = set()
        for track in out['active_tracks']:
            track_id = int(track.id)
            track_class = int(track.cls)
            track_score = float(getattr(track, 'score', 1.0))

            current_active_ids.add(track_id)
            track_total_hits[track_id] = track_total_hits.get(track_id, 0) + 1
            if track_id in prev_active_ids:
                track_streak[track_id] = track_streak.get(track_id, 0) + 1
            else:
                track_streak[track_id] = 1

            required_streak = int(
                class_streak_map.get(track_class, min_track_streak)
            )
            if track_streak[track_id] < required_streak:
                continue
            if track_total_hits[track_id] < min_track_total_hits:
                continue
            if track_score < score_threshold:
                continue

            # ``track.bbox`` is normalised cxcywh in [0, 1] — store as xywh for CT20 JSON.
            cx, cy, w, h = track.bbox.tolist()
            x = max(0.0, min(1.0, cx - w / 2.0))
            y = max(0.0, min(1.0, cy - h / 2.0))
            w = max(1e-4, min(1.0, w))
            h = max(1e-4, min(1.0, h))
            tools_in_frame.append({
                'instrument': int(track.cls) + 1,  # 1..7 (CT20 convention)
                'tool_bbox': [x, y, w, h],
                'intraoperative_track_id': int(track.id),
                'pseudo': True,
                'score': track_score,
            })

        prev_active_ids = current_active_ids

        if tools_in_frame:
            last_tools = tools_in_frame

        end_idx = window_start + clip_length - 1
        if last_tools:
            gap_start = (prev_labeled_end + 1) if prev_labeled_end is not None else 0
            _assign_frame_annotations(annotations, frames, gap_start, end_idx, last_tools)
            prev_labeled_end = end_idx

        if executor is not None:
            next_ws = window_start + window_stride
            if next_ws < n_windows:
                lv_snapshot = last_valid
                pending_buffer = executor.submit(_build_buffer_at_start, next_ws, lv_snapshot)

    if executor is not None:
        executor.shutdown(wait=True)

    # Propagate last boxes through tail frames skipped by stride.
    if last_tools and prev_labeled_end is not None and prev_labeled_end < len(frames) - 1:
        _assign_frame_annotations(
            annotations, frames, prev_labeled_end + 1, len(frames) - 1, last_tools,
        )

    return annotations


# --------------------------------------------------------------------------- #
# Corpus layout builder                                                        #
# --------------------------------------------------------------------------- #


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def _layout_cholec80_video(
    video_name: str,
    source_video_dir: Path,
    out_training: Path,
    annotations: Dict[str, List[Dict]],
) -> None:
    """Symlink Cholec80 frames into CT20-style layout and write pseudo JSON.

    source_video_dir: data/cholec80/frames/videoNN/
    out_training:     data/ssl_corpus/Training/
    """
    out_video_dir = out_training / video_name
    out_frames_dir = out_video_dir / 'Frames'
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    for f in _collect_video_frames(source_video_dir):
        frame_num = _parse_frame_number(f.name)
        dst = out_frames_dir / f'{frame_num:06d}{f.suffix.lower()}'
        if not dst.exists():
            os.symlink(f, dst)

    json_path = out_video_dir / f'{video_name}.json'
    with open(json_path, 'w') as f:
        json.dump({'annotations': annotations}, f)


def _layout_ct20_video(
    ct20_src_video_dir: Path,
    out_training: Path,
) -> None:
    """Symlink an entire CT20 Training video directory into the SSL corpus."""
    dst = out_training / ct20_src_video_dir.name
    _symlink(ct20_src_video_dir, dst)


def _layout_ct20_validation(
    ct20_root: Path,
    out_root: Path,
) -> None:
    """Mirror the CT20 Validation folder into the SSL corpus (read-only reuse).

    We only use it for sanity; SSL itself does not train on Validation.
    """
    src = ct20_root / 'Validation'
    dst = out_root / 'Validation'
    if src.exists() and not dst.exists():
        os.symlink(src, dst)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage1_config', type=Path, required=True,
                   help='Stage 1 YAML config (used to rebuild the model).')
    p.add_argument('--stage1_checkpoint', type=Path, required=True,
                   help='Path to best Stage 1 checkpoint.')
    p.add_argument('--out_root', type=Path, default=default_ssl_corpus_root(),
                   help='Where to lay out the unified SSL corpus (default: data/ssl_corpus).')
    p.add_argument(
        '--cholec80_root', type=Path, default=None,
        help='Cholec80 frames root (video01..video80 subdirs). '
             'Default: MOT_CHOLEC80_FRAMES_ROOT or auto-discover.',
    )
    p.add_argument(
        '--cholectrack20_root', type=Path, default=None,
        help='CholecTrack20 dataset root. Default: MOT_CHOLECTRACK20_ROOT or auto-discover.',
    )
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument(
        '--infer_score_threshold', type=float, default=0.25,
        help='DETR detection score cutoff (model.tracker_infer_score_threshold).',
    )
    p.add_argument(
        '--birth_score', type=float, default=None,
        help='TrackManager birth threshold (default: same as --infer_score_threshold). '
             'YAML default 0.6 is too high for weak pseudo-label recall.',
    )
    p.add_argument(
        '--tracker_min_hits', type=int, default=1,
        help='TrackManager min_hits before confirmed (YAML default 3 kills pseudo labels).',
    )
    p.add_argument('--score_threshold', type=float, default=0.25,
                   help='Drop pseudo boxes with track score below this (after tracking).')
    p.add_argument('--min_track_streak', type=int, default=1,
                   help='Drop boxes unless same track is stable for this many consecutive frames.')
    p.add_argument('--min_track_total_hits', type=int, default=1,
                   help='Drop boxes unless track has at least this many total frame hits.')
    p.add_argument(
        '--min_track_streak_by_class',
        type=_parse_class_streak_map,
        default=None,
        help='Optional JSON map of track-class -> minimum consecutive-streak threshold, e.g. {"0": 2, "1": 3}.',
    )
    p.add_argument('--clip_length', type=int, default=3)
    p.add_argument(
        '--window_stride', type=int, default=2,
        help='Run detector every N frames (2≈2× faster; fills gaps with last boxes). Use 1 for max quality.',
    )
    p.add_argument(
        '--prefetch_workers', type=int, default=4,
        help='CPU threads to preload frame windows when window_stride>1 (Grace CPUs).',
    )
    p.add_argument(
        '--amp', dest='use_amp', action='store_true', default=True,
        help='bf16 autocast on CUDA (GB10).',
    )
    p.add_argument('--no-amp', dest='use_amp', action='store_false')
    p.add_argument(
        '--full_infer', action='store_true',
        help='Use full infer (ReID + per-track predictor). Slower; default is infer_fast.',
    )
    p.add_argument('--skip_pseudo_label', action='store_true',
                   help='Skip Stage 1 inference, only rebuild symlink layout.')
    p.add_argument('--dry_run', action='store_true',
                   help='Print what would be done but do not write anything.')
    p.add_argument('--rank', type=int, default=0,
                   help='Process rank for video sharding (0..world_size-1).')
    p.add_argument('--world_size', type=int, default=1,
                   help='Total number of parallel processes.')
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    )
    args = parse_args()
    rank = args.rank
    world_size = args.world_size
    device = torch.device(f'cuda:{rank}' if args.device.startswith('cuda') else args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 1. Build and verify the leak-free split.
    split = build_ssl_split(
        cholec80_root=args.cholec80_root,
        cholectrack20_root=args.cholectrack20_root,
    )
    _LOG.info(f'Cholec80 frames: {split.cholec80_root}')
    _LOG.info(f'CholecTrack20: {split.cholectrack20_root}')
    verify_no_leak(split)
    _LOG.info(split.summary())

    out_training = args.out_root / 'Training'
    out_training.mkdir(parents=True, exist_ok=True)

    # 2. Mirror CT20 Training + Validation (only rank 0 to avoid races).
    ct20_train_root = split.cholectrack20_root / 'Training'
    if rank == 0:
        _LOG.info(f'Mirroring CT20 Training → {out_training}')
        for vid_name in CT20_TRAIN:
            src = ct20_train_root / vid_name
            if not src.is_dir():
                _LOG.warning(f'CT20 training video missing on disk: {src}')
                continue
            if args.dry_run:
                _LOG.info(f'[dry] symlink {src} → {out_training / vid_name}')
            else:
                _layout_ct20_video(src, out_training)

        if not args.dry_run:
            _layout_ct20_validation(split.cholectrack20_root, args.out_root)

    # 3. Pseudo-label every Cholec80 SSL video (sharded by rank).
    if args.skip_pseudo_label:
        _LOG.info('Skipping pseudo-label generation as requested.')
        return

    with open(args.stage1_config, 'r') as f:
        cfg = yaml.safe_load(f)
    img_size = cfg.get('data', {}).get('img_size', 392)

    _LOG.info(f'[rank {rank}/{world_size}] Loading Stage 1 model from {args.stage1_checkpoint} → {device}')
    model, _ = _load_stage1_model(
        args.stage1_config,
        args.stage1_checkpoint,
        device,
        infer_score_threshold=args.infer_score_threshold,
        birth_score=args.birth_score,
        tracker_min_hits=args.tracker_min_hits,
    )
    infer_mode = 'infer' if args.full_infer else 'infer_fast'
    _LOG.info(
        f'Speed: mode={infer_mode}, window_stride={args.window_stride}, '
        f'amp={args.use_amp and device.type == "cuda"}, prefetch_workers={args.prefetch_workers}'
    )
    _LOG.info(
        f'Pseudo post-filters: score>={args.score_threshold}, '
        f'streak>={args.min_track_streak}, total_hits>={args.min_track_total_hits}'
    )

    c80_root = split.cholec80_root
    shard = [v for i, v in enumerate(split.cholec80_videos, 1) if (i - 1) % world_size == rank]
    _LOG.info(f'[rank {rank}/{world_size}] Processing {len(shard)} videos')

    for i, video in enumerate(shard, 1):
        src_dir = c80_root / video
        frames = _collect_video_frames(src_dir)
        _LOG.info(f'[rank {rank}][{i}/{len(shard)}] {video} ({len(frames)} frames)')

        if args.dry_run:
            continue

        annotations = _pseudo_label_video(
            model=model,
            frames=frames,
            img_size=img_size,
            clip_length=args.clip_length,
            device=device,
            score_threshold=args.score_threshold,
            min_track_streak=args.min_track_streak,
            min_track_total_hits=args.min_track_total_hits,
            min_track_streak_by_class=args.min_track_streak_by_class,
            window_stride=args.window_stride,
            use_amp=args.use_amp,
            infer_mode=infer_mode,
            prefetch_workers=args.prefetch_workers,
        )
        _LOG.info(
            f'  → {len(annotations)} annotated frames, '
            f'{sum(len(v) for v in annotations.values())} boxes'
        )

        _layout_cholec80_video(
            video_name=video,
            source_video_dir=src_dir,
            out_training=out_training,
            annotations=annotations,
        )

    _LOG.info(f'SSL corpus built at {args.out_root}')


if __name__ == '__main__':
    main()
