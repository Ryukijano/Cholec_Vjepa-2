"""
MOT evaluation harness — HOTA / IDF1 / MOTA via the TrackEval library.

This module provides:

  * ``mot_rollout``: run ``SurgicalMOTSystem`` in inference mode over a
    CholecTrack20 validation video, emitting MOTChallenge-format rows
    (``<frame>, <id>, <x>, <y>, <w>, <h>, <conf>, -1, -1, -1``).

  * ``write_mot_txt`` / ``load_mot_txt`` — simple IO helpers for the
    text format expected by ``TrackEval``.

  * ``evaluate_trackeval``: optional wrapper around ``trackeval`` that
    computes the standard HOTA / IDF1 / MOTA metrics. ``trackeval`` is
    imported lazily so it doesn't become a hard requirement for users
    who just want to run inference.

  * ``evaluate_stratified_mot``: smoke / bleeding / occlusion breakdown
    using CholecTrack20 per-tool challenge flags aggregated per frame.

Per-class breakdown is handled by running TrackEval separately on each
tool class (graspers, bipolar, hook, scissors, clipper, irrigator,
specimen bag). The caller is responsible for providing the ground-truth
MOTChallenge files built from CholecTrack20 annotations.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from .system import SurgicalMOTSystem

MotRow = Tuple[int, int, float, float, float, float, float]

# CT20 surgical tool classes (1-indexed in annotation files).
CT20_TOOL_CLASSES: Tuple[str, ...] = (
    'grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator', 'specimen_bag',
)
CT20_CLASS_TO_ID: Dict[str, int] = {name: i + 1 for i, name in enumerate(CT20_TOOL_CLASSES)}

CHALLENGE_FLAGS: Tuple[str, ...] = ('smoke', 'bleeding', 'occluded')

STRATUM_NAMES: Tuple[str, ...] = (
    'overall',
    'smoke',
    'no_smoke',
    'bleeding',
    'no_bleeding',
    'occluded',
    'not_occluded',
)


# ---------------------------------------------------------------------- #
# 1. IO helpers                                                           #
# ---------------------------------------------------------------------- #


def write_mot_txt(rows: Sequence[MotRow], path: Path) -> None:
    """
    Write rows in MOTChallenge format.

    Each row: ``(frame_idx, track_id, x, y, w, h, confidence)``.
    Pixel coordinates (caller must de-normalise before writing).
    Frame indices are 1-based (MOTChallenge convention).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for row in rows:
            frame, tid, x, y, w, h, conf = row[:7]
            cls = int(row[7]) if len(row) > 7 else 1
            f.write(
                f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},"
                f"{cls},-1,-1\n"
            )


def load_mot_txt(path: Path) -> List[MotRow]:
    rows: List[MotRow] = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            row_vals: tuple = (
                int(parts[0]), int(parts[1]),
                float(parts[2]), float(parts[3]),
                float(parts[4]), float(parts[5]),
                float(parts[6]),
            )
            if len(parts) >= 8:
                row_vals = row_vals + (int(float(parts[7])),)
            rows.append(row_vals)
    return rows


# ---------------------------------------------------------------------- #
# 2. CholecTrack20 → MOT helpers                                          #
# ---------------------------------------------------------------------- #


def _ct20_json_path(video_dir: Path) -> Path:
    name = video_dir.name
    for candidate in (video_dir / f'{name}.json', video_dir / f'{name.lower()}.json'):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'No CT20 JSON found under {video_dir}')


def load_ct20_video_meta(json_path: Path) -> Tuple[int, int, int]:
    """Return ``(width, height, num_frames)`` from CT20 video metadata."""
    with open(json_path) as f:
        data = json.load(f)
    video = data.get('video', {})
    return (
        int(video.get('width', 1920)),
        int(video.get('height', 1080)),
        int(video.get('num_frames', 0)),
    )


def list_ct20_frame_numbers(frames_dir: Path) -> List[int]:
    """Sorted CT20 frame indices extracted from ``Frames/*.png`` stems."""
    nums: List[int] = []
    for path in frames_dir.iterdir():
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        nums.append(int(path.stem))
    return sorted(nums)


def list_ct20_frame_numbers_from_json(json_path: Path) -> List[int]:
    """Sorted annotated frame indices when ``Frames/`` is not extracted."""
    with open(json_path) as f:
        data = json.load(f)
    return sorted(int(k) for k in data.get('annotations', {}).keys())


def resolve_ct20_frame_source(video_dir: Path) -> Tuple[List[int], Optional[Path], Optional[Path]]:
    """
    Resolve frame indices and on-disk source for a CT20 video directory.

    Returns ``(frame_numbers, frames_dir_or_none, mp4_path_or_none)``.
    Prefers pre-extracted ``Frames/``; falls back to ``*.mp4`` + JSON keys.
    """
    video_dir = Path(video_dir)
    json_path = _ct20_json_path(video_dir)
    frames_dir = video_dir / 'Frames'
    if frames_dir.is_dir():
        pngs = [p for p in frames_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        if pngs:
            return list_ct20_frame_numbers(frames_dir), frames_dir, None

    mp4 = video_dir / f'{video_dir.name}.mp4'
    if not mp4.exists():
        mp4s = sorted(video_dir.glob('*.mp4'))
        mp4 = mp4s[0] if mp4s else None
    if mp4 is None or not mp4.exists():
        return [], None, None
    return list_ct20_frame_numbers_from_json(json_path), None, mp4


def build_frame_challenges(
    annotations: Mapping[str, Sequence[Mapping[str, object]]],
    frame_numbers: Sequence[int],
) -> Dict[int, Dict[str, bool]]:
    """
    Aggregate per-tool challenge flags into per-frame booleans.

    A frame is positive for ``smoke`` / ``bleeding`` / ``occluded`` when
    *any* annotated tool on that frame has the flag set to 1.
    """
    # MOT frame index (1-based position in sorted frame list) → flags.
    out: Dict[int, Dict[str, bool]] = {}
    for mot_idx, frame_num in enumerate(frame_numbers, start=1):
        tools = annotations.get(str(int(frame_num)), [])
        flags = {flag: False for flag in CHALLENGE_FLAGS}
        for tool in tools:
            for flag in CHALLENGE_FLAGS:
                if int(tool.get(flag, 0) or 0) == 1:
                    flags[flag] = True
        out[mot_idx] = flags
    return out


def build_ct20_gt_mot_rows(
    json_path: Path,
    frame_numbers: Sequence[int],
    img_width: int,
    img_height: int,
) -> List[MotRow]:
    """
    Convert CT20 intraoperative-track annotations to MOTChallenge GT rows.

    ``tool_bbox`` is normalised tlwh in [0, 1]; converted to pixel xywh.
    """
    with open(json_path) as f:
        data = json.load(f)
    annotations = data.get('annotations', {})
    rows: List[MotRow] = []

    for mot_idx, frame_num in enumerate(frame_numbers, start=1):
        for tool in annotations.get(str(int(frame_num)), []):
            tid = tool.get('intraoperative_track')
            if tid is None:
                tid = tool.get('intraoperative_track_id')
            bbox = tool.get('tool_bbox') or tool.get('bbox')
            if tid is None or bbox is None:
                continue
            x, y, w, h = (float(v) for v in bbox)
            px = x * img_width
            py = y * img_height
            pw = w * img_width
            ph = h * img_height
            if pw <= 0 or ph <= 0:
                continue
            label = tool.get('instrument') or tool.get('tool_id') or 1
            cls_id = int(label) if label is not None else 1
            rows.append((mot_idx, int(tid), px, py, pw, ph, 1.0, cls_id))
    return rows


def filter_mot_rows_by_frames(
    rows: Sequence[MotRow],
    allowed_frames: Set[int],
) -> List[MotRow]:
    """Keep MOT rows whose 1-based frame index is in ``allowed_frames``."""
    return [row for row in rows if row[0] in allowed_frames]


def reindex_mot_rows_contiguous(
    rows: Sequence[MotRow],
    frame_order: Optional[Sequence[int]] = None,
) -> Tuple[List[MotRow], int]:
    """
    Remap sparse 1-based frame indices to contiguous ``1..T``.

    TrackEval requires timesteps to match ``seqLength`` in ``seqinfo.ini``.
    When ``frame_order`` is supplied, only those source frames are mapped
    (shared mapping for GT and predictions).
    """
    if frame_order is None:
        frames = sorted({row[0] for row in rows})
    else:
        frames = list(frame_order)
    if not frames:
        return [], 0
    mapping = {old: new for new, old in enumerate(frames, start=1)}
    reindexed = [
        (mapping[row[0]],) + tuple(row[1:])
        for row in rows
        if row[0] in mapping
    ]
    return reindexed, len(frames)


def frames_for_stratum(
    frame_challenges: Mapping[int, Mapping[str, bool]],
    stratum: str,
) -> Set[int]:
    """Return 1-based MOT frame indices belonging to ``stratum``."""
    if stratum == 'overall':
        return set(frame_challenges.keys())

    frames: Set[int] = set()
    for mot_idx, flags in frame_challenges.items():
        if stratum == 'smoke' and flags.get('smoke'):
            frames.add(mot_idx)
        elif stratum == 'no_smoke' and not flags.get('smoke'):
            frames.add(mot_idx)
        elif stratum == 'bleeding' and flags.get('bleeding'):
            frames.add(mot_idx)
        elif stratum == 'no_bleeding' and not flags.get('bleeding'):
            frames.add(mot_idx)
        elif stratum == 'occluded' and flags.get('occluded'):
            frames.add(mot_idx)
        elif stratum == 'not_occluded' and not flags.get('occluded'):
            frames.add(mot_idx)
    return frames


def load_ct20_video_tensor(
    frames_dir: Path,
    frame_numbers: Sequence[int],
    img_size: int,
    max_frames: int = 0,
    mp4_path: Optional[Path] = None,
) -> torch.Tensor:
    """
    Load CT20 frames as ``(T, 3, H, W)`` with ImageNet normalisation.

    Args:
        frames_dir: ``.../VIDxx/Frames`` (may be unused when ``mp4_path`` set).
        frame_numbers: sorted CT20 frame indices.
        img_size: square resize side (must match training).
        max_frames: truncate to first N frames when > 0.
        mp4_path: optional MP4 fallback when ``Frames/`` is missing.
    """
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    nums = list(frame_numbers)
    if max_frames > 0:
        nums = nums[:max_frames]

    if mp4_path is not None:
        return _load_ct20_video_tensor_from_mp4(Path(mp4_path), nums, transform)

    stem_to_path = {
        int(p.stem): p
        for p in Path(frames_dir).iterdir()
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
    }
    frames: List[torch.Tensor] = []
    for num in nums:
        path = stem_to_path[int(num)]
        img = Image.open(path).convert('RGB')
        frames.append(transform(img))
    if not frames:
        raise ValueError(f'No frames loaded from {frames_dir}')
    return torch.stack(frames, dim=0)


def _load_ct20_video_tensor_from_mp4(
    mp4_path: Path,
    frame_numbers: Sequence[int],
    transform: T.Compose,
) -> torch.Tensor:
    """Decode selected frame indices from a CT20 MP4 (1-fps annotation indices)."""
    import cv2

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {mp4_path}')

    frames: List[torch.Tensor] = []
    try:
        for frame_idx in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(transform(Image.fromarray(rgb)))
    finally:
        cap.release()

    if not frames:
        raise ValueError(f'No frames decoded from {mp4_path}')
    return torch.stack(frames, dim=0)


# ---------------------------------------------------------------------- #
# 3. Inference rollout                                                    #
# ---------------------------------------------------------------------- #


@torch.no_grad()
def mot_rollout(
    model: SurgicalMOTSystem,
    frames: torch.Tensor,
    clip_len: int = 3,
    img_size: int = 392,
    device: torch.device = torch.device('cuda'),
) -> List[MotRow]:
    """
    Run the model across a sliding window of ``clip_len`` frames and
    collect per-frame active-track snapshots.

    Args:
        model: trained ``SurgicalMOTSystem`` (will be set to eval mode).
        frames: (T, 3, H, W) pre-normalised frame tensor for one video.
        clip_len: length of the sliding window.
        img_size: image height / width (assumed square, pixels) — used
                  to convert normalised cxcywh → pixel (x, y, w, h).
        device: inference device.

    Returns:
        MOTChallenge rows with **1-based** frame indices.
    """
    model.eval()
    model.to(device)
    model.reset_tracker()
    frames = frames.to(device)

    t_total = frames.size(0)
    rows: List[MotRow] = []

    for t in range(t_total):
        start = max(0, t - (clip_len - 1))
        clip_frames = frames[start : t + 1]
        if clip_frames.size(0) < clip_len:
            pad = clip_frames[:1].expand(clip_len - clip_frames.size(0), -1, -1, -1)
            clip_frames = torch.cat([pad, clip_frames], dim=0)

        clip = clip_frames.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)

        out = model(current_video=clip, mode='infer')
        mot_frame = t + 1  # MOTChallenge uses 1-based indexing.
        for track in out['active_tracks']:
            if track.status == 'tentative':
                continue
            cx, cy, w, h = track.bbox.cpu().tolist()
            x = (cx - 0.5 * w) * img_size
            y = (cy - 0.5 * h) * img_size
            rows.append(
                (mot_frame, int(track.id), float(x), float(y),
                 float(w * img_size), float(h * img_size), float(track.score),
                 int(track.cls) + 1)
            )

    return rows


# ---------------------------------------------------------------------- #
# 4. TrackEval wrapper (optional)                                         #
# ---------------------------------------------------------------------- #


def _write_seqinfo(path: Path, seq_name: str, seq_length: int,
                   width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        '[Sequence]\n'
        f'name={seq_name}\n'
        'imDir=img1\n'
        'frameRate=1\n'
        f'seqLength={seq_length}\n'
        f'imWidth={width}\n'
        f'imHeight={height}\n'
        'imExt=.png\n'
    )
    path.write_text(content)


def _prepare_trackeval_layout(
    root: Path,
    seq_name: str,
    gt_rows: Sequence[MotRow],
    pred_rows: Sequence[MotRow],
    seq_length: int,
    width: int,
    height: int,
    tracker_name: str = 'pred',
) -> Tuple[Path, Path]:
    """Write MOTChallenge-style GT + tracker dirs under ``root``."""
    gt_seq = root / seq_name
    _write_seqinfo(gt_seq / 'seqinfo.ini', seq_name, seq_length, width, height)
    write_mot_txt(gt_rows, gt_seq / 'gt' / 'gt.txt')

    pred_root = root / 'trackers' / tracker_name / 'data'
    pred_root.mkdir(parents=True, exist_ok=True)
    write_mot_txt(pred_rows, pred_root / f'{seq_name}.txt')

    seqmaps = root / 'seqmaps'
    seqmaps.mkdir(parents=True, exist_ok=True)
    (seqmaps / 'MOT17-train.txt').write_text(f'{seq_name}\n')

    return root, root / 'trackers'


def _as_float(value: object) -> float:
    """Coerce TrackEval scalars / numpy arrays to ``float``."""
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        return float(np.mean(value))
    if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
        seq = list(value)  # type: ignore[arg-type]
        if not seq:
            return 0.0
        return float(sum(float(v) for v in seq) / len(seq))
    return float(value)  # type: ignore[arg-type]


def _extract_trackeval_scalars(class_results: Mapping[str, object]) -> Dict[str, float]:
    """Pull scalar HOTA / MOTA / IDF1 from TrackEval per-class output."""
    out: Dict[str, float] = {}
    hota = class_results.get('HOTA')
    if isinstance(hota, dict):
        if 'HOTA' in hota:
            out['HOTA'] = _as_float(hota['HOTA'])
        elif 'HOTA(0)' in hota:
            out['HOTA'] = _as_float(hota['HOTA(0)'])
    clear = class_results.get('CLEAR')
    if isinstance(clear, dict) and 'MOTA' in clear:
        out['MOTA'] = _as_float(clear['MOTA'])
    identity = class_results.get('Identity')
    if isinstance(identity, dict) and 'IDF1' in identity:
        out['IDF1'] = _as_float(identity['IDF1'])
    return out


def evaluate_trackeval(
    gt_dir: Path,
    pred_dir: Path,
    seq_names: Iterable[str],
    metrics: Sequence[str] = ('HOTA', 'IDF1', 'MOTA'),
    classes_to_eval: Optional[Sequence[str]] = None,
    class_to_id: Optional[Mapping[str, int]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Thin wrapper around ``trackeval`` computing HOTA / IDF1 / MOTA.

    By default evaluates CT20's **7-class multi-class protocol**
    (grasper, bipolar, hook, scissors, clipper, irrigator, specimen_bag)
    instead of collapsing all tools into a single 'pedestrian' class.

    Requires ``pip install trackeval`` (JonathonLuiten). Raises
    ``RuntimeError`` if not installed.

    Returns:
        ``{sequence: {metric: value}}``.
    """
    try:
        import trackeval  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "trackeval is required for evaluation. Install via: pip install trackeval"
        ) from e

    if classes_to_eval is None:
        classes_to_eval = list(CT20_TOOL_CLASSES)
    if class_to_id is None:
        class_to_id = CT20_CLASS_TO_ID

    dataset_cfg = {
        'GT_FOLDER': str(gt_dir),
        'TRACKERS_FOLDER': str(pred_dir),
        'OUTPUT_FOLDER': str(pred_dir / 'eval_out'),
        'SEQ_INFO': {name: None for name in seq_names},
        'CLASSES_TO_EVAL': list(classes_to_eval),
        'CLASS_TO_TP_CLASS': dict(class_to_id),
        'BENCHMARK': 'MOT17',
        'SPLIT_TO_EVAL': 'train',
        'INPUT_AS_ZIP': False,
        'TRACKERS_TO_EVAL': [p.name for p in Path(pred_dir).iterdir() if p.is_dir()],
        'SKIP_SPLIT_FOL': True,
    }
    evaluator = trackeval.Evaluator({
        'LOG_ON_ERROR': None,
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': False,
        'PRINT_ONLY_COMBINED': True,
        'DISPLAY_LESS_PROGRESS': True,
    })
    datasets = [trackeval.datasets.MotChallenge2DBox(dataset_cfg)]
    metric_classes = []
    if 'HOTA' in metrics:
        metric_classes.append(trackeval.metrics.HOTA())
    if 'IDF1' in metrics:
        metric_classes.append(trackeval.metrics.Identity())
    if 'MOTA' in metrics:
        metric_classes.append(trackeval.metrics.CLEAR())

    results, _ = evaluator.evaluate(datasets, metric_classes)

    out: Dict[str, Dict[str, float]] = {}
    for _dataset_name, dataset_results in results.items():
        for tracker_name, tracker_results in dataset_results.items():
            for seq_name, seq_results in tracker_results.items():
                if seq_name in out:
                    continue
                combined: Dict[str, float] = {}
                for _class_name, class_results in seq_results.items():
                    combined.update(_extract_trackeval_scalars(class_results))
                out[seq_name] = combined
    return out


def evaluate_trackeval_rows(
    seq_name: str,
    gt_rows: Sequence[MotRow],
    pred_rows: Sequence[MotRow],
    seq_length: int,
    width: int,
    height: int,
    metrics: Sequence[str] = ('HOTA', 'IDF1', 'MOTA'),
) -> Dict[str, float]:
    """Evaluate one sequence from in-memory MOT rows via a temp TrackEval tree."""
    if not gt_rows:
        return {}
    tmp = Path(tempfile.mkdtemp(prefix='mot_eval_'))
    try:
        gt_dir, pred_dir = _prepare_trackeval_layout(
            tmp, seq_name, gt_rows, pred_rows, seq_length, width, height,
        )
        per_seq = evaluate_trackeval(gt_dir, pred_dir, [seq_name], metrics=metrics)
        return per_seq.get(seq_name, {})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def evaluate_stratified_mot(
    gt_rows: Sequence[MotRow],
    pred_rows: Sequence[MotRow],
    frame_challenges: Mapping[int, Mapping[str, bool]],
    seq_length: int,
    width: int,
    height: int,
    strata: Sequence[str] = STRATUM_NAMES,
    metrics: Sequence[str] = ('HOTA', 'MOTA', 'IDF1'),
) -> Dict[str, Dict[str, float]]:
    """
    Compute HOTA / MOTA / IDF1 per visual-challenge stratum.

    Returns ``{stratum_name: {metric: value}}``. Empty strata are omitted.
    """
    results: Dict[str, Dict[str, float]] = {}
    for stratum in strata:
        allowed = frames_for_stratum(frame_challenges, stratum)
        if not allowed:
            continue
        gt_f = filter_mot_rows_by_frames(gt_rows, allowed)
        pred_f = filter_mot_rows_by_frames(pred_rows, allowed)
        if not gt_f:
            continue
        frame_order = sorted(allowed)
        gt_f, seq_len = reindex_mot_rows_contiguous(gt_f, frame_order=frame_order)
        pred_f, _ = reindex_mot_rows_contiguous(pred_f, frame_order=frame_order)
        metrics_out = evaluate_trackeval_rows(
            seq_name=f'CT20_{stratum}',
            gt_rows=gt_f,
            pred_rows=pred_f,
            seq_length=seq_len,
            width=width,
            height=height,
            metrics=metrics,
        )
        if metrics_out:
            metrics_out['num_frames'] = float(len(allowed))
            metrics_out['num_gt_dets'] = float(len(gt_f))
            metrics_out['num_pred_dets'] = float(len(pred_f))
            results[stratum] = metrics_out
    return results


def ct20_split_video_dirs(data_root: Path, split: str) -> List[Path]:
    """List CT20 video directories for ``train`` / ``val`` / ``test``."""
    split_map = {
        'train': 'Training',
        'val': 'Validation',
        'test': 'Testing',
    }
    if split not in split_map:
        raise ValueError(f'Unknown split: {split}')
    root = data_root / split_map[split]
    out: List[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        frame_nums, frames_dir, mp4_path = resolve_ct20_frame_source(p)
        if frame_nums and (frames_dir is not None or mp4_path is not None):
            out.append(p)
    return out
