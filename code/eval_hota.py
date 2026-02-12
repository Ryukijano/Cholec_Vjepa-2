#!/usr/bin/env python3
"""
Evaluate HOTA metric on CholecTrack20 validation set using detection + Re-ID tracking.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from collections import defaultdict

# === WINDOWS PATH FIXES ===
current_dir = Path(__file__).parent  # code/
project_root = current_dir.parent
vjepa2_path = project_root / "vjepa2"
src_path = project_root / "vjepa2" / "src"

# Prefer local code/ modules (models, reid_head, tracker) over vjepa2/src
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(vjepa2_path))
sys.path.insert(0, str(current_dir))

import numpy as np

# NumPy 2.0 compatibility: motmetrics uses np.asfarray which was removed
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from vjepa2.app.vjepa.utils import init_video_model
from vjepa2.app.vjepa_cholec80.lora import apply_lora_to_encoder, load_checkpoint_into_lora_model

from reid_head import ReIDHead
from tracker import SurgicalTracker
from models import LightweightQueryDecoder, MidBackboneHook, detection_postprocess

try:
    import motmetrics as mm
except ImportError:
    print("Install motmetrics: pip install motmetrics")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='[%(levelname)-8s][%(name)-20s] %(message)s')
logger = logging.getLogger(__name__)

def load_models(args, device):
    """Load detection + Re-ID models."""
    # Encoder
    encoder, _ = init_video_model(
        device=device, model_name='vit_large', patch_size=16,
        max_num_frames=8, tubelet_size=2, crop_size=224,
        pred_depth=12, pred_embed_dim=384,
        use_mask_tokens=True, use_sdpa=True)

    # LoRA
    apply_lora_to_encoder(encoder, rank=16, alpha=16.0, start_layer=12)

    # Load detection checkpoint
    logger.info(f"Loading detection checkpoint from {args.detection_checkpoint}")
    ckpt = torch.load(args.detection_checkpoint, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'encoder_lora' in ckpt:
        load_checkpoint_into_lora_model(encoder, ckpt['encoder_lora'])
    elif 'encoder' in ckpt:
        load_checkpoint_into_lora_model(encoder, ckpt['encoder'])
    
    encoder.to(device).eval()

    # Detection Head
    num_classes = 7 # CholecTrack20 default
    det_head = LightweightQueryDecoder(
        embed_dim=1024, num_classes=num_classes,
        num_queries=100, num_decoder_layers=2,
        nheads=8, dim_feedforward=512, dropout=0.1, grid_size=14,
        internal_dim=256, temporal_pooling='last'
    ).to(device)
    
    if 'head' in ckpt:
        det_head.load_state_dict(ckpt['head'])
    elif 'det_head' in ckpt:
        det_head.load_state_dict(ckpt['det_head'])
    det_head.eval()

    # Mid-backbone hook
    mid_hook = MidBackboneHook(layer_idx=12)
    mid_hook.register(encoder.backbone.blocks)

    # Re-ID head
    reid_head = ReIDHead(embed_dim=1024, grid_size=14, hidden_dim=512, reid_dim=128)
    if args.reid_checkpoint:
        logger.info(f"Loading Re-ID checkpoint from {args.reid_checkpoint}")
        ckpt_reid = torch.load(args.reid_checkpoint, map_location='cpu')
        reid_raw = ckpt_reid.get('reid_head', ckpt_reid)
        if not isinstance(reid_raw, dict):
            logger.warning("Re-ID checkpoint has no 'reid_head' state_dict; skipping Re-ID load.")
        else:
            reid_sd = {k.replace("_orig_mod.", ""): v for k, v in reid_raw.items()}
            current_sd = reid_head.state_dict()
            compatible_sd = {k: v for k, v in reid_sd.items() if k in current_sd and current_sd[k].shape == v.shape}
            reid_head.load_state_dict(compatible_sd, strict=False)
    reid_head.to(device).eval()

    return encoder, det_head, mid_hook, reid_head

def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5*w, cy - 0.5*h, cx + 0.5*w, cy + 0.5*h], dim=-1)

def run_tracking(video_dir, encoder, det_head, mid_hook, reid_head, device, tracker_config, track_perspective):
    """Run tracking on a single video."""
    tracker = SurgicalTracker(**tracker_config)
    
    json_path = next(video_dir.glob('*.json'), None)
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    anns = data.get('annotations', {})
    
    # Get existing frame files
    frames_dir = video_dir / 'Frames'
    existing_frames = set()
    if frames_dir.exists():
        for p in frames_dir.glob('*.png'):
            existing_frames.add(int(p.stem))
    
    # Filter fids to only those with both annotation and image
    all_fids = sorted([int(k) for k in anns.keys()])
    fids = [fid for fid in all_fids if fid in existing_frames]
    
    if not fids:
        logger.warning(f"No matching frames found for {video_dir.name}")
        return []
    
    track_key = f'{track_perspective}_track'
    
    results = []
    
    for i in tqdm(range(len(fids)), desc=f"Video {video_dir.name}", leave=False):
        target_fid = fids[i]
        
        # Load clip (8 frames)
        clip_fids = []
        for j in range(i - 7, i + 1):
            idx = max(0, j)
            clip_fids.append(fids[idx])
            
        imgs = []
        for fid in clip_fids:
            path = video_dir / 'Frames' / f"{fid:06d}.png"
            img = Image.open(path).convert('RGB')
            img = img.resize((224, 224), Image.BILINEAR)
            imgs.append(np.array(img, dtype=np.float32) / 255.0)
            
        clip = torch.from_numpy(np.stack(imgs, axis=0)).permute(3, 0, 1, 2).unsqueeze(0).to(device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=True):
                full_tokens = encoder([clip])[0]
                mid_features = mid_hook.get_features()
                
                logits, pred_boxes, _, _ = det_head(full_tokens, mid_features)
                probs = logits.softmax(-1)[0]
                scores, pred_cls = probs[..., :-1].max(-1)
                # Score filter + NMS to merge duplicate query boxes (conf_thresh=0.8, nms=0.2)
                det_boxes, det_scores, det_cls = detection_postprocess(
                    pred_boxes[0], scores, pred_cls,
                    score_thresh=0.8, nms_thresh=0.2, num_classes=7
                )
                if det_boxes.shape[0] == 0:
                    online_tracks = tracker.update([])
                else:
                    
                    # Get Re-ID embeddings
                    # ReIDHead expects list of boxes
                    embs = reid_head(full_tokens, [det_boxes])[0]
                    
                    detections = []
                    for b in range(len(det_boxes)):
                        detections.append({
                            'box': det_boxes[b].cpu().numpy(),
                            'score': det_scores[b].item(),
                            'cls': det_cls[b].item(),
                            'reid_emb': embs[b].cpu().numpy()
                        })
                    
                    online_tracks = tracker.update(detections)
        
        for track in online_tracks:
            results.append({
                'frame': target_fid,
                'id': track['track_id'],
                'box': track['box'], # cxcywh
                'cls': track['cls'],
                'score': track['score']
            })
            
    return results, fids


def _load_mot_predictions(mot_file: Path):
    preds = defaultdict(list)
    if not mot_file.exists():
        return preds
    with open(mot_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            fid = int(float(parts[0]))
            tid = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            score = float(parts[6])
            preds[fid].append((tid, [x, y, w, h], score))
    return preds


def evaluate_from_mot_directory(val_dir: Path, mot_dir: Path, perspective: str):
    """
    Evaluate one perspective from MOT txt files generated by run_tracking.py.
    """
    metrics_accumulator = mm.MOTAccumulator(auto_id=False)
    ID_OFFSET = 1_000_000
    videos = sorted([d for d in val_dir.iterdir() if d.is_dir() and d.name.startswith("VID")])
    track_key = f"{perspective}_track"

    for video_idx, video_dir in enumerate(videos):
        pred_path = mot_dir / f"{video_dir.name}.txt"
        pred_by_frame = _load_mot_predictions(pred_path)
        json_path = next(video_dir.glob("*.json"), None)
        if json_path is None:
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        anns = data.get("annotations", {})

        fids = sorted(set(pred_by_frame.keys()) | set(int(k) for k in anns.keys()))
        for fid in fids:
            frame_anns = anns.get(str(fid), [])
            gt_boxes = []
            gt_ids = []
            for ann in frame_anns:
                bbox = ann.get("tool_bbox")
                tid = ann.get(track_key, -1)
                if bbox is None or tid is None or tid < 0:
                    continue
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                gt_boxes.append(bbox)
                gt_ids.append(video_idx * ID_OFFSET + int(tid))

            pred_boxes = []
            pred_ids = []
            for tid, box, _score in pred_by_frame.get(fid, []):
                pred_boxes.append(box)
                pred_ids.append(video_idx * ID_OFFSET + int(tid))

            if not gt_boxes and not pred_boxes:
                distances = np.empty((0, 0))
            elif not gt_boxes:
                distances = np.empty((0, len(pred_boxes)))
            elif not pred_boxes:
                distances = np.empty((len(gt_boxes), 0))
            else:
                distances = mm.distances.iou_matrix(
                    np.asarray(gt_boxes), np.asarray(pred_boxes), max_iou=0.5
                )
            frameid = video_idx * ID_OFFSET + fid
            metrics_accumulator.update(gt_ids, pred_ids, distances, frameid=frameid)

    mh = mm.metrics.create()
    summary = mh.compute(metrics_accumulator, metrics=["mota", "idf1"], name=perspective)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_checkpoint', type=str, required=True)
    parser.add_argument('--reid_checkpoint', type=str, default=None)
    parser.add_argument('--val_dir', type=str, default='../cholec_dataset/Validation')
    parser.add_argument('--output_dir', type=str, default='../outputs/hota_eval')
    parser.add_argument('--track_perspective', type=str, default='intraoperative')
    parser.add_argument('--all_perspectives', action='store_true',
                        help='Evaluate visibility/intracorporeal/intraoperative together.')
    parser.add_argument('--mot_pred_dir', type=str, default='',
                        help='Directory containing MOT txt files (e.g. outputs/surgitrackpp/mot). '
                             'If set, skips model inference and only evaluates predictions.')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_dir = Path(args.val_dir)

    if args.mot_pred_dir:
        mot_pred_dir = Path(args.mot_pred_dir)
        perspectives = ["visibility", "intracorporeal", "intraoperative"] if args.all_perspectives else [args.track_perspective]
        all_text = []
        for p in perspectives:
            summary = evaluate_from_mot_directory(val_dir, mot_pred_dir, p)
            print(f"\nTracking Results ({p}):")
            print(summary)
            all_text.append(f"[{p}]\n{summary.to_string()}\n")
        results_path = output_dir / 'hota_results.txt'
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(all_text))
        logger.info("Results saved to %s", results_path)
        return

    # Load models for legacy direct-eval mode
    encoder, det_head, mid_hook, reid_head = load_models(args, device)

    # Tracker config: higher high_thresh to cut clutter; lenient recovery; more Kalman process noise for shaky video
    tracker_config = {
        'high_thresh': 0.65,
        'low_thresh': 0.1,
        'match_thresh': 0.7,
        'recovery_match_thresh': 0.85,
        'max_lost': 30,
        'min_hits': 3,
        'iou_weight': 0.5,
        'reid_weight': 0.3,
        'process_noise_scale': 3.0,
    }

    # Load validation videos
    video_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir() and d.name.startswith('VID')])

    # Single accumulator with namespaced IDs per video so tracks don't mix across videos.
    metrics_accumulator = mm.MOTAccumulator(auto_id=False)
    ID_OFFSET = 1_000_000

    for video_idx, video_dir in enumerate(video_dirs):
        logger.info(f"Processing {video_dir.name}...")
        
        # Run tracking
        results, fids = run_tracking(video_dir, encoder, det_head, mid_hook, reid_head, device, tracker_config, args.track_perspective)
        
        # Save tracking results in MOT format
        track_path = output_dir / 'data' / 'trackers' / 'mot_challenge' / 'test' / 'vjepa_tracker' / 'data'
        track_path.mkdir(parents=True, exist_ok=True)
        with open(track_path / f'{video_dir.name}.txt', 'w') as f:
            for track in results:
                fid = track['frame']
                cx, cy, w, h = track['box']
                x = cx - w / 2
                y = cy - h / 2
                score = track['score']
                track_id = track['id']
                f.write(f"{fid},{track_id},{x},{y},{w},{h},{score},1,1\n")
        
        # Load ground truth for this video
        json_path = next(video_dir.glob('*.json'), None)
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        anns = data.get('annotations', {})
        track_key = f'{args.track_perspective}_track'
        
        fids_in_results = set(r['frame'] for r in results)
        fids_in_anns = set(int(k) for k in anns.keys())
        fids = sorted(fids_in_results & fids_in_anns)

        for fid in fids:
            frame_anns = anns.get(str(fid), [])
            gt_boxes = []
            gt_ids = []
            for ann in frame_anns:
                bbox = ann.get('tool_bbox')
                tid = ann.get(track_key, -1)
                if bbox is not None and tid is not None and tid >= 0 and (bbox[2] > 0 and bbox[3] > 0):
                    gt_boxes.append(bbox)
                    gt_ids.append(video_idx * ID_OFFSET + int(tid))

            pred_boxes = []
            pred_ids = []
            for res in results:
                if res['frame'] == fid:
                    cx, cy, w, h = res['box']
                    x, y = cx - w / 2, cy - h / 2
                    pred_boxes.append([x, y, w, h])
                    pred_ids.append(video_idx * ID_OFFSET + res['id'])

            if not gt_boxes and not pred_boxes:
                distances = np.empty((0, 0))
            elif not gt_boxes:
                distances = np.empty((0, len(pred_boxes)))
            elif not pred_boxes:
                distances = np.empty((len(gt_boxes), 0))
            else:
                distances = mm.distances.iou_matrix(
                    np.asarray(gt_boxes), np.asarray(pred_boxes), max_iou=0.5
                )
            frameid = video_idx * ID_OFFSET + fid
            metrics_accumulator.update(gt_ids, pred_ids, distances, frameid=frameid)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(metrics_accumulator, metrics=['mota', 'idf1'], name='vjepa2_tracker')
    
    print("\nTracking Results:")
    print(summary)
    
    # SurgiTrack Benchmarks (from Re-ID Head Implementation.md)
    print("\nSurgiTrack Benchmarks (CholecTrack20):")
    print("Intraoperative: 67.0% HOTA")
    print("Intracorporeal: 55.0% HOTA")
    print("Visibility:     62.0% HOTA")

    # Save results
    results_path = output_dir / 'hota_results.txt'
    with open(results_path, 'w') as f:
        f.write(summary.to_string())
        f.write("\n\nSurgiTrack Benchmarks:\n")
        f.write("Intraoperative: 67.0% HOTA\n")
        f.write("Intracorporeal: 55.0% HOTA\n")
        f.write("Visibility:     62.0% HOTA\n")
    
    logger.info(f"Results saved to {results_path}")

if __name__ == '__main__':
    main()
