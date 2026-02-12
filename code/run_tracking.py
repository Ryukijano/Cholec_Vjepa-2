#!/usr/bin/env python3
"""
SurgiTrack++ inference pipeline:
  RF-DETR (detection) + V-JEPA2 (temporal features) + ReIDHeadV2 + SurgicalTrackerV2.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
vjepa2_path = project_root / "vjepa2"
src_path = project_root / "vjepa2" / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(vjepa2_path))
sys.path.insert(0, str(current_dir))

from vjepa2.app.vjepa.utils import init_video_model
from vjepa2.app.vjepa_cholec80.lora import apply_lora_to_encoder, load_checkpoint_into_lora_model

from reid_head_v2 import ReIDHeadV2, direction_proxy_targets
from rfdetr_wrapper import RFDETRFeatureWrapper
from tracker_v2 import SurgicalTrackerV2


logging.basicConfig(level=logging.INFO, format="[%(levelname)-8s][%(name)-20s] %(message)s")
logger = logging.getLogger("run_tracking")


def _build_rfdetr(size: str, checkpoint: str = "", device: str = "cpu"):
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRBase

    mapping = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
        "base": RFDETRBase,
    }
    # Initial build
    model = mapping[size]()
    
    if checkpoint:
        logger.info("Loading RF-DETR checkpoint: %s", checkpoint)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        
        # Extract state_dict
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "ema_model" in ckpt:
                state_dict = ckpt["ema_model"]
            else:
                state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        else:
            state_dict = ckpt
            
        # Infer num_classes
        num_classes = 8 # Default for CholecTrack20
        for key in state_dict.keys():
            if "class_embed" in key and "weight" in key:
                num_classes = state_dict[key].shape[0]
                break
        
        # Find inner model for resizing
        inner_model = None
        curr = model
        for _ in range(3):
            if isinstance(curr, torch.nn.Module) and (hasattr(curr, "class_embed") or hasattr(curr, "transformer")):
                inner_model = curr
                break
            if hasattr(curr, "model"):
                curr = curr.model
            else:
                break
        
        if inner_model is not None:
            # Resize if needed
            curr_classes = 0
            if hasattr(inner_model, "class_embed"):
                curr_classes = inner_model.class_embed.weight.shape[0]
            
            if curr_classes != num_classes:
                logger.info("Resizing RF-DETR class embeddings from %d to %d", curr_classes, num_classes)
                if hasattr(inner_model, "class_embed"):
                    old = inner_model.class_embed
                    new = torch.nn.Linear(old.in_features, num_classes)
                    inner_model.class_embed = new
                
                if hasattr(inner_model, "transformer") and hasattr(inner_model.transformer, "enc_out_class_embed"):
                    enc = inner_model.transformer.enc_out_class_embed
                    if isinstance(enc, torch.nn.ModuleList):
                        for i in range(len(enc)):
                            if isinstance(enc[i], torch.nn.Linear):
                                enc[i] = torch.nn.Linear(enc[i].in_features, num_classes)
            
            # Load
            inner_model.load_state_dict(state_dict, strict=False)
            
        # Move to device
        if hasattr(model, "model"):
            curr = model
            for _ in range(3):
                if hasattr(curr, "model"):
                    curr = curr.model
                    if hasattr(curr, "to") and isinstance(curr, torch.nn.Module):
                        curr.to(device)
                        break
                else:
                    break
        elif hasattr(model, "to") and isinstance(model, torch.nn.Module):
            model.to(device)

    return RFDETRFeatureWrapper(model)


def _load_vjepa_encoder(device: str, checkpoint: str = ""):
    encoder, _ = init_video_model(
        device=device,
        model_name="vit_large",
        patch_size=16,
        max_num_frames=16,
        tubelet_size=2,
        crop_size=224,
        pred_depth=12,
        pred_embed_dim=384,
        use_mask_tokens=True,
        use_sdpa=True,
    )
    apply_lora_to_encoder(encoder, rank=16, alpha=16.0, start_layer=12)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        if "encoder_lora" in ckpt:
            load_checkpoint_into_lora_model(encoder, ckpt["encoder_lora"])
        elif "encoder" in ckpt:
            load_checkpoint_into_lora_model(encoder, ckpt["encoder"])
    encoder.to(device).eval()
    return encoder


def _load_reid_head(device: str, checkpoint: str = "", reid_dim: int = 128, num_dir_bins: int = 3):
    model = ReIDHeadV2(reid_dim=reid_dim, num_direction_bins=num_dir_bins).to(device).eval()
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        sd = ckpt.get("reid_head_v2", ckpt.get("reid_head", ckpt))
        if isinstance(sd, dict):
            model.load_state_dict(sd, strict=False)
    return model


def _load_clip(video_dir: Path, frame_ids: List[int], frame_size: int = 224) -> torch.Tensor:
    imgs = []
    for fid in frame_ids:
        path = video_dir / "Frames" / f"{fid:06d}.png"
        img = Image.open(path).convert("RGB")
        img = img.resize((frame_size, frame_size), Image.BILINEAR)
        imgs.append(np.asarray(img, dtype=np.float32) / 255.0)
    clip = torch.from_numpy(np.stack(imgs, axis=0)).permute(3, 0, 1, 2)  # [C,T,H,W]
    return clip


def _collect_video_frames(video_dir: Path) -> List[int]:
    frames_dir = video_dir / "Frames"
    if not frames_dir.exists():
        return []
    return sorted(int(p.stem) for p in frames_dir.glob("*.png"))


def _det_to_cxcywh_norm(det_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = det_xyxy
    cx = ((x1 + x2) * 0.5) / width
    cy = ((y1 + y2) * 0.5) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return np.array([cx, cy, w, h], dtype=np.float32)


def run_video(
    video_dir: Path,
    encoder,
    rf_wrap: RFDETRFeatureWrapper,
    reid_head: ReIDHeadV2,
    tracker: SurgicalTrackerV2,
    device: str,
    score_thresh: float,
) -> List[Dict]:
    frame_ids = _collect_video_frames(video_dir)
    if not frame_ids:
        logger.warning("No frames in %s", video_dir)
        return []

    outputs = []
    for i, fid in enumerate(tqdm(frame_ids, desc=video_dir.name, leave=False)):
        # Build 16-frame clip ending at current frame (pad at start)
        idxs = [frame_ids[max(0, j)] for j in range(i - 15, i + 1)]
        clip = _load_clip(video_dir, idxs).unsqueeze(0).to(device)

        last_img_path = video_dir / "Frames" / f"{fid:06d}.png"
        last_img = Image.open(last_img_path).convert("RGB")
        W, H = last_img.size

        with torch.no_grad():
            # Detector on current frame
            pred = rf_wrap.predict_with_features(last_img, threshold=score_thresh)
            det = pred.detections
            if len(det.xyxy) == 0:
                online = tracker.update([])
            else:
                boxes_xyxy = np.asarray(det.xyxy)
                cls = np.asarray(det.class_id).astype(np.int64)
                conf = np.asarray(det.confidence).astype(np.float32)
                boxes_cxcywh = np.stack([_det_to_cxcywh_norm(b, W, H) for b in boxes_xyxy], axis=0)

                # Temporal tokens from V-JEPA2
                full_tokens = encoder([clip])[0]
                boxes_t = [torch.tensor(boxes_cxcywh, dtype=torch.float32, device=device)]
                classes_t = [torch.tensor(cls, dtype=torch.long, device=device)]
                qf = None
                if pred.matched_query_features is not None:
                    qf = [pred.matched_query_features.to(device)]
                else:
                    qf = [None]

                reid_embs, _ = reid_head(full_tokens, boxes_t, query_features=qf, classes=classes_t)
                dir_bins = direction_proxy_targets(boxes_t, bins=reid_head.num_direction_bins)[0].cpu().numpy()

                detections = []
                emb_np = reid_embs[0].detach().cpu().numpy()
                for di in range(len(boxes_cxcywh)):
                    detections.append(
                        {
                            "box": boxes_cxcywh[di].tolist(),  # Convert numpy array to list
                            "cls": int(cls[di]),
                            "score": float(conf[di]),
                            "reid_emb": emb_np[di].tolist() if di < emb_np.shape[0] else None,  # Convert numpy array to list
                            "direction_bin": int(dir_bins[di]) if di < len(dir_bins) else None,
                        }
                    )
                online = tracker.update(detections)

        for tr in online:
            # Convert numpy arrays to lists for JSON serialization
            box = tr["box"]
            if isinstance(box, np.ndarray):
                box = box.tolist()
            elif hasattr(box, "copy"):  # Handle numpy scalar arrays
                box = float(box) if box.ndim == 0 else box.tolist()
            
            outputs.append(
                {
                    "frame": int(fid),
                    "track_id": int(tr["track_id"]),
                    "box": box,  # cxcywh normalized, now a list
                    "cls": int(tr["cls"]),
                    "score": float(tr["score"]),
                    "visibility_state": str(tr.get("visibility_state", "active")),
                    "intracorporeal_state": str(tr.get("intracorporeal_state", "active")),
                    "intraoperative_state": str(tr.get("intraoperative_state", "active")),
                }
            )
    return outputs


def save_mot_results(results: List[Dict], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            fid = int(r["frame"])
            tid = int(r["track_id"])
            cx, cy, w, h = r["box"]
            x = cx - w * 0.5
            y = cy - h * 0.5
            score = float(r["score"])
            f.write(f"{fid},{tid},{x},{y},{w},{h},{score},1,1\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../cholec_dataset/Validation")
    parser.add_argument("--output_dir", type=str, default="../outputs/surgitrackpp")
    parser.add_argument("--vjepa_checkpoint", type=str, default="")
    parser.add_argument("--reid_checkpoint", type=str, default="")
    parser.add_argument("--rfdetr_model_size", choices=["nano", "small", "medium", "large", "base"], default="medium")
    parser.add_argument("--rfdetr_checkpoint", type=str, default="")
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--reid_dim", type=int, default=128)
    parser.add_argument("--direction_bins", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = _load_vjepa_encoder(device=device, checkpoint=args.vjepa_checkpoint)
    reid_head = _load_reid_head(
        device=device,
        checkpoint=args.reid_checkpoint,
        reid_dim=args.reid_dim,
        num_dir_bins=args.direction_bins,
    )
    rf_wrap = _build_rfdetr(args.rfdetr_model_size, checkpoint=args.rfdetr_checkpoint, device=device)
    tracker = SurgicalTrackerV2()

    video_dirs = sorted([d for d in Path(args.data_dir).iterdir() if d.is_dir() and d.name.startswith("VID")])
    for v in video_dirs:
        logger.info("Processing %s", v.name)
        tracker.reset()
        results = run_video(v, encoder, rf_wrap, reid_head, tracker, device, args.score_thresh)
        save_mot_results(results, out_dir / "mot" / f"{v.name}.txt")

        # Optional JSON export with perspective states
        (out_dir / "json").mkdir(parents=True, exist_ok=True)
        
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return super().default(obj)
        
        with open(out_dir / "json" / f"{v.name}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, cls=NumpyEncoder, indent=2)

    rf_wrap.close()
    logger.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()
