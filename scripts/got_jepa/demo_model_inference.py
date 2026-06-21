#!/usr/bin/env python3
"""
Demo: Load a video clip and run inference through the Stage-1 model.
Shows how to:
  1. Build model from config
  2. Load checkpoint
  3. Process a real video frame-by-frame
  4. Extract detections and visualize predictions

Run with:
    module load miniforge/24.7.1
    conda activate surgi_world_track_cuda
    python scripts/demo_model_inference.py
"""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torchvision.transforms as T
import yaml
from PIL import Image, ImageDraw

from core_app.mot.trainer import build_model_from_config


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = REPO_ROOT / "configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml"
CKPT_PATH = REPO_ROOT / "outputs/mot/cholec20-stage1-supervised/best.pth.tar"
VIDEO_DIR = Path("/scratch/kcwp264/data/surgi_world_track/cholectrack20/Training/VID02/Frames")


def parse_args():
    parser = argparse.ArgumentParser(description="Run MOT inference on a local folder of frames.")
    parser.add_argument('--config', type=Path, default=CONFIG_PATH, help='Model yaml config')
    parser.add_argument('--ckpt', type=Path, default=CKPT_PATH, help='Checkpoint path')
    parser.add_argument('--video-dir', type=Path, default=VIDEO_DIR, help='Folder of frame PNGs')
    parser.add_argument('--max-frames', type=int, default=6, help='Limit the number of input frames')
    parser.add_argument('--debug', action='store_true', help='Enable debug reporting')
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.1,
        help='Detection score threshold in debug mode',
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=None,
        help='Optional alternate debug threshold for detection print and visualization',
    )
    parser.add_argument(
        '--input-frame',
        type=int,
        default=None,
        help='Force a single window start index (debug-friendly)',
    )
    parser.add_argument(
        '--save-vis',
        type=Path,
        default=None,
        help='Optional output path for one debug frame with boxes',
    )
    parser.add_argument(
        '--n-support-points',
        type=int,
        default=None,
        help='Reserved compatibility flag (currently unused in demo path)',
    )
    return parser.parse_args()


def _track_to_dict(track) -> dict:
    """Normalize Track object / dict output into a common dictionary layout."""
    if isinstance(track, dict):
        bbox = track.get('bbox', [])
        score = float(track.get('score', 0.0))
        cls = track.get('class', track.get('cls', None))
        tid = track.get('id', None)
    else:
        bbox = getattr(track, 'bbox', [])
        score = float(getattr(track, 'score', 0.0))
        cls = getattr(track, 'cls', None)
        tid = getattr(track, 'id', None)

    if bbox is None:
        return {}
    bbox = bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
    if len(bbox) >= 5:
        # Some outputs store score in bbox[4]; keep both for compatibility.
        score = float(bbox[4])
        bbox = bbox[:4]
    return {'id': tid, 'bbox': bbox, 'score': score, 'class': cls}


def _cxcywh_to_xyxy(box, width: int, height: int):
    cx, cy, w, h = box
    x1 = (cx - w / 2.0) * width
    y1 = (cy - h / 2.0) * height
    x2 = (cx + w / 2.0) * width
    y2 = (cy + h / 2.0) * height
    return (
        max(0.0, min(width - 1, x1)),
        max(0.0, min(height - 1, y1)),
        max(0.0, min(width - 1, x2)),
        max(0.0, min(height - 1, y2)),
    )


def _save_debug_vis(
    image: Image.Image,
    detections: list,
    base_path: Path,
    frame_idx: int,
    score_threshold: float,
):
    """Write a single debug frame with detections."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    palette = [
        (255, 90, 90),
        (90, 140, 255),
        (90, 220, 130),
        (230, 170, 60),
        (170, 90, 220),
    ]

    visible = 0
    for i, det in enumerate(detections):
        score = float(det.get('score', 0.0))
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = _cxcywh_to_xyxy(det.get('bbox', []), width, height)
        color = palette[i % len(palette)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"id={det.get('id', '?')} cls={det.get('class', 'n/a')} s={score:.2f}"
        draw.text((x1 + 2, max(0.0, y1 - 12)), text, fill=color)
        visible += 1

    if base_path.suffix:
        out_path = base_path.with_name(f"{base_path.stem}_{frame_idx:04d}{base_path.suffix}")
    else:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        out_path = base_path / f"frame_{frame_idx:04d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return out_path, visible


def _debug_threshold(args):
    if args.confidence_threshold is not None:
        return args.confidence_threshold
    if args.debug:
        return args.score_threshold
    return 0.3


def load_frames(video_dir: Path, max_frames: int = 10):
    """Load and sort PNG frames from a video directory."""
    frames = sorted(video_dir.glob("*.png"))
    print(f"Found {len(frames)} frames, loading {max_frames}...")
    return [Image.open(f).convert("RGB") for f in frames[:max_frames]]


def preprocess_clip(frames, img_size: int = 392, clip_length: int = 3):
    """Convert list of PIL frames to model-ready (B, C, T, H, W) tensor."""
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensors = [transform(f) for f in frames]
    clip = torch.stack(tensors, dim=1).unsqueeze(0)  # (1, C, T, H, W)
    return clip


def main():
    args = parse_args()
    print("=" * 60)
    print("Demo: Stage-1 Model Inference on Real Video")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load config and build model
    # ------------------------------------------------------------------
    print("\n[1/4] Loading config and building model...")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = build_model_from_config(cfg)

    # Load checkpoint
    if args.ckpt.exists():
        ckpt = torch.load(args.ckpt, map_location="cpu")
        sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(sd, strict=False)
        print(f"  Checkpoint loaded: {args.ckpt}")
    else:
        print(f"  WARNING: No checkpoint found at {args.ckpt}, using random init")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"  Model on: {device}")
    print(f"  Backbone: {cfg['model']['encoder_type']}")
    print(f"  DETR: {'Deformable' if cfg.get('detr', {}).get('use_deformable_detr') else 'Vanilla'}")

    # ------------------------------------------------------------------
    # 2. Load frames
    # ------------------------------------------------------------------
    print(f"\n[2/4] Loading frames from {args.video_dir}...")
    frames = load_frames(args.video_dir, max_frames=args.max_frames)
    if not frames:
        print("No frames found in dataset path.")
        return

    # ------------------------------------------------------------------
    # 3. Sliding-window inference
    # ------------------------------------------------------------------
    print("\n[3/4] Running sliding-window inference...")
    clip_length = cfg["data"].get("clip_length", 3)
    all_detections = []
    max_windows = len(frames) - clip_length + 1
    if max_windows < 1:
        print(f"Need at least {clip_length} frames; got {len(frames)}.")
        return

    if args.input_frame is not None:
        if args.input_frame < 0 or args.input_frame >= max_windows:
            print(f"--input-frame {args.input_frame} is out of range [0, {max_windows - 1}]")
            return
        window_indices = [args.input_frame]
    else:
        window_indices = list(range(max_windows))

    debug_threshold = _debug_threshold(args)
    tracks_visible_summary = []

    with torch.no_grad():
        for i in window_indices:
            window = frames[i : i + clip_length]
            clip = preprocess_clip(window, img_size=cfg["data"].get("img_size", 392))
            clip = clip.to(device)

            outputs = model(current_video=clip, mode="infer")
            active = outputs.get("active_tracks", [])
            all_detections.append(len(active))

            normalized_tracks = [_track_to_dict(t) for t in active]
            normalized_tracks = [t for t in normalized_tracks if t]
            visible = [t for t in normalized_tracks if t['score'] >= debug_threshold]
            tracks_visible_summary.append(len(visible))

            # --- Raw DETR debug path (bypass TrackManager birth gating) --- #
            raw_detr_dets: list = []
            if args.debug and len(visible) == 0:
                try:
                    detr = outputs.get('detr', {})
                    if 'pred' in detr:
                        logits = detr['pred']['class_logits'][0]   # (Q, num_tools)
                        boxes = detr['pred']['pred_boxes'][0]      # (Q, 4)
                        scores, classes = logits.sigmoid().max(dim=-1)
                        keep = scores > debug_threshold
                        for q in keep.nonzero(as_tuple=True)[0]:
                            raw_detr_dets.append({
                                'id': None,
                                'bbox': boxes[q].cpu().tolist(),
                                'score': float(scores[q].item()),
                                'class': int(classes[q].item()),
                            })
                except Exception as e:
                    print(f"    [raw-DETR extraction error: {e}]")

            if args.debug:
                print(f"  Window [{i}:{i+clip_length}]: tracks={len(active)} visible={len(visible)}")
                print(f"    Detections above threshold: {len(visible) > 0}")
                if raw_detr_dets:
                    print(f"    Raw DETR detections (birth gate bypassed): {len(raw_detr_dets)}")
                    for j, t in enumerate(raw_detr_dets[:3]):
                        print(
                            f"    RawDet {j}: cls={t['class']}, "
                            f"box={t['bbox']}, score={t['score']:.3f}"
                        )
                    if len(raw_detr_dets) > 3:
                        print(f"    ... +{len(raw_detr_dets) - 3} more raw dets")
                if visible:
                    for j, t in enumerate(visible[:3]):
                        print(
                            f"    Track {j}: id={t.get('id', '?')}, "
                            f"box={t['bbox']}, score={t.get('score', 0.0):.3f}, "
                            f"class={t.get('class', 'N/A')}"
                        )
                    if len(visible) > 3:
                        print(f"    ... +{len(visible) - 3} more tracks")
                if not visible and not raw_detr_dets:
                    print("    No detections above threshold.")

                if args.save_vis is not None:
                    vis_dets = visible if visible else raw_detr_dets
                    out_path, n_vis = _save_debug_vis(
                        image=window[-1],
                        detections=vis_dets,
                        base_path=args.save_vis,
                        frame_idx=i,
                        score_threshold=debug_threshold,
                    )
                    print(f"    Frame saved: {out_path} ({n_vis} boxes)")
            elif i < 3:  # Print details for first few windows (legacy baseline path)
                print(f"  Window [{i}:{i+clip_length}]: {len(active)} tracks")
                if normalized_tracks:
                    t = normalized_tracks[0]
                    print(
                        f"    Track 0: box={t['bbox']}, "
                        f"score={t.get('score', 0.0):.3f}, class={t.get('class', 'N/A')}"
                    )

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print("\n[4/4] Summary")
    print(f"  Total windows: {len(all_detections)}")
    print(f"  Avg detections per window: {sum(all_detections)/len(all_detections):.1f}")
    print(f"  Max detections in a window: {max(all_detections)}")
    print(f"  Min detections in a window: {min(all_detections)}")
    if args.debug:
        print(f"  Debug threshold: {debug_threshold:.3f}")
        if tracks_visible_summary:
            print(f"  Avg tracks above threshold: {sum(tracks_visible_summary)/len(tracks_visible_summary):.1f}")
        if args.save_vis is not None:
            print(f"  Visualization base path: {args.save_vis}")
        print(f"  n-support-points: {args.n_support_points if args.n_support_points is not None else 'default'}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
