#!/usr/bin/env python3
"""
Visualize V-JEPA tracking on validation frames.

Unlike single-frame detectors, V-JEPA consumes an 8-frame clip and predicts
boxes in the shared temporal representation (temporal pooling = last frame).
We draw tracks on the *target frame* (last frame of the clip) so boxes
align with what the model is predicting for.
"""

import argparse
import json
import sys
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent
vjepa2_path = project_root / "vjepa2"
src_path = project_root / "vjepa2" / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(vjepa2_path))
sys.path.insert(0, str(current_dir))

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from eval_hota import load_models
from models import detection_postprocess
from tracker import SurgicalTracker
from detection_viz import TOOL_NAMES, TOOL_COLORS, box_cxcywh_to_xyxy_np


TRACKER_CONFIG = {
    "high_thresh": 0.65,
    "low_thresh": 0.1,
    "match_thresh": 0.7,
    "recovery_match_thresh": 0.85,
    "max_lost": 30,
    "min_hits": 3,
    "iou_weight": 0.5,
    "reid_weight": 0.3,
    "process_noise_scale": 3.0,
}

# Distinct colors for track IDs (cycle if more tracks)
TRACK_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (23, 190, 207),
    (255, 187, 120), (174, 199, 232),
]


def run_tracking_and_collect_frames(
    video_dir,
    encoder,
    det_head,
    mid_hook,
    reid_head,
    device,
    max_frames=None,
    score_thresh=0.3,
):
    """
    Run the full pipeline (8-frame clip -> detection -> Re-ID -> tracker)
    and yield (target_fid, frame_img_224, tracks) for each step.

    frame_img_224 is the *target frame* (last frame of the clip) at 224x224,
    i.e. the frame the model's predictions refer to.
    """
    tracker = SurgicalTracker(**TRACKER_CONFIG)
    json_path = next(video_dir.glob("*.json"), None)
    if not json_path:
        return
    with open(json_path, "r") as f:
        data = json.load(f)
    anns = data.get("annotations", {})
    frames_dir = video_dir / "Frames"
    if not frames_dir.exists():
        return
    all_fids = sorted([int(k) for k in anns.keys()])
    existing = {int(p.stem) for p in frames_dir.glob("*.png")}
    fids = [fid for fid in all_fids if fid in existing]
    if not fids:
        return
    if max_frames is not None:
        fids = fids[: max_frames]

    for i in tqdm(range(len(fids)), desc=video_dir.name, leave=False):
        target_fid = fids[i]
        clip_fids = [fids[max(0, j)] for j in range(i - 7, i + 1)]
        imgs = []
        for fid in clip_fids:
            path = video_dir / "Frames" / f"{fid:06d}.png"
            img = Image.open(path).convert("RGB")
            img = img.resize((224, 224), Image.BILINEAR)
            imgs.append(np.array(img, dtype=np.float32) / 255.0)
        # [1, C, T, H, W]
        clip = torch.from_numpy(np.stack(imgs, axis=0)).permute(3, 0, 1, 2).unsqueeze(0).to(device)
        target_frame = imgs[-1]  # last frame = target frame (224, 224, 3)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                full_tokens = encoder([clip])[0]
                mid_features = mid_hook.get_features()
                logits, pred_boxes, _, _ = det_head(full_tokens, mid_features)
                probs = logits.softmax(-1)[0]
                scores, pred_cls = probs[..., :-1].max(-1)
                det_boxes, det_scores, det_cls = detection_postprocess(
                    pred_boxes[0], scores, pred_cls,
                    score_thresh=max(score_thresh, 0.8), nms_thresh=0.2, num_classes=7
                )
                if det_boxes.shape[0] == 0:
                    online_tracks = tracker.update([])
                else:
                    embs = reid_head(full_tokens, [det_boxes])[0]
                    detections = [
                        {
                            "box": det_boxes[b].cpu().numpy(),
                            "score": det_scores[b].item(),
                            "cls": det_cls[b].item(),
                            "reid_emb": embs[b].cpu().numpy(),
                        }
                        for b in range(len(det_boxes))
                    ]
                    online_tracks = tracker.update(detections)

        yield target_fid, target_frame, online_tracks


def draw_tracks_on_frame(frame_uint8, tracks, frame_size=224):
    """
    Draw track boxes and labels on a frame. frame in [0,255] or [0,1];
    tracks = list of {box (cxcywh norm), cls, score, track_id}.
    Returns [H,W,3] uint8.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if frame_uint8.max() <= 1.0:
        frame_uint8 = (np.clip(frame_uint8, 0, 1) * 255).astype(np.uint8)
    H, W = frame_uint8.shape[:2]
    scale = np.array([W, H, W, H])

    fig, ax = plt.subplots(1, 1, figsize=(W / 80, H / 80), dpi=80)
    ax.imshow(frame_uint8)
    ax.set_title("V-JEPA tracks (clip→det→Re-ID→tracker)", fontsize=10)
    ax.axis("off")

    for tr in tracks:
        box = np.array(tr["box"]).reshape(1, 4)
        xyxy = box_cxcywh_to_xyxy_np(box)[0] * scale
        x1, y1, x2, y2 = xyxy
        tid = tr.get("track_id", tr.get("id", -1))
        cls_id = tr.get("cls", 0)
        score = tr.get("score", 0.0)
        color = TRACK_COLORS[int(tid) % len(TRACK_COLORS)]
        color_norm = tuple(v / 255 for v in color)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color_norm, facecolor="none", linestyle="-",
        )
        ax.add_patch(rect)
        label = TOOL_NAMES[cls_id] if cls_id < len(TOOL_NAMES) else f"cls{cls_id}"
        ax.text(
            x1, max(0, y1 - 4), f"T{tid} {label} {score:.2f}",
            fontsize=7, color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color_norm, alpha=0.9),
        )
    plt.tight_layout()
    buf = __import__("io").BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    out = np.array(Image.open(buf).convert("RGB"))
    plt.close(fig)
    buf.close()
    return out


def main():
    parser = argparse.ArgumentParser(description="Visualize V-JEPA tracking on validation frames")
    parser.add_argument("--detection_checkpoint", type=str, required=True)
    parser.add_argument("--reid_checkpoint", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default="../cholec_dataset/Validation")
    parser.add_argument("--output_dir", type=str, default="../outputs/track_viz")
    parser.add_argument("--max_videos", type=int, default=2, help="Number of videos to process")
    parser.add_argument("--max_frames_per_video", type=int, default=40, help="Max frames per video to run (for speed)")
    parser.add_argument("--num_viz", type=int, default=6, help="Number of frames to save as images (total across videos)")
    parser.add_argument("--score_thresh", type=float, default=0.65, help="Detection score threshold (higher = fewer, cleaner detections)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_dir = Path(args.val_dir)
    video_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir() and d.name.startswith("VID")])
    video_dirs = video_dirs[: args.max_videos]
    if not video_dirs:
        print("No video dirs found in", val_dir)
        return

    print("Loading models (same as eval_hota)...")
    encoder, det_head, mid_hook, reid_head = load_models(args, device)

    collected = []
    for video_dir in video_dirs:
        for target_fid, frame_img, tracks in run_tracking_and_collect_frames(
            video_dir,
            encoder,
            det_head,
            mid_hook,
            reid_head,
            device,
            max_frames=args.max_frames_per_video,
            score_thresh=args.score_thresh,
        ):
            tracks_for_viz = [
                {"box": t["box"], "cls": t["cls"], "score": t["score"], "track_id": t["track_id"]}
                for t in tracks
            ]
            collected.append((video_dir.name, target_fid, frame_img, tracks_for_viz))

    if not collected:
        print("No frames collected.")
        return
    n = min(args.num_viz, len(collected))
    step = max(1, (len(collected) - 1) // n) if n > 1 else 1
    to_viz = [collected[i] for i in range(0, len(collected), step)][:n]

    print(f"Saving {len(to_viz)} track visualizations to {out_dir}")
    for video_name, target_fid, frame_img, tracks in to_viz:
        out_img = draw_tracks_on_frame(frame_img.copy(), tracks)
        fname = f"{video_name}_frame{target_fid:06d}.png"
        Image.fromarray(out_img).save(out_dir / fname)
        print(f"  {fname} ({len(tracks)} tracks)")

    print("Done.")


if __name__ == "__main__":
    main()
