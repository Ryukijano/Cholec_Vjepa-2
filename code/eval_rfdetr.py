#!/usr/bin/env python3
"""
Evaluate RF-DETR model performance on CholecTrack20 validation set.
Computes Recall@0.5, mAP, and per-class metrics.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRBase
except ImportError:
    print("Error: rf-detr package not installed. Install with: pip install rf-detr")
    exit(1)


TOOL_NAMES = [
    "Background",  # 0 (not used)
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag",
]


def _load_coco_annotations(json_path: Path) -> tuple[Dict, Dict]:
    """Load COCO format annotations."""
    with open(json_path, "r") as f:
        data = json.load(f)
    
    images = {img["id"]: img for img in data["images"]}
    annotations = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(ann)
    
    return images, annotations


def _xywh_to_xyxy(box: List[float]) -> np.ndarray:
    """Convert COCO [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def _iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / max(union_area, 1e-8)


def compute_recall_at_iou(
    model, valid_dir: Path, iou_threshold: float = 0.5, score_threshold: float = 0.3
) -> Dict[str, float]:
    """Compute class-wise and overall Recall@IoU on COCO valid split."""
    images, anns = _load_coco_annotations(valid_dir / "_annotations.coco.json")
    cat_to_tp: Dict[int, int] = {}
    cat_to_gt: Dict[int, int] = {}
    
    print(f"Evaluating on {len(images)} images...")
    
    for image_id, info in tqdm(images.items(), desc="RF-DETR eval"):
        gt = anns.get(image_id, [])
        for g in gt:
            cat = int(g["category_id"])
            cat_to_gt[cat] = cat_to_gt.get(cat, 0) + 1
            cat_to_tp.setdefault(cat, 0)
        
        image_path = valid_dir / info["file_name"]
        if not image_path.exists() or not gt:
            continue
        
        image = Image.open(image_path).convert("RGB")
        detections = model.predict(image, threshold=score_threshold)
        
        pred_boxes = (
            np.asarray(detections.xyxy) if len(detections.xyxy) > 0 
            else np.zeros((0, 4), dtype=np.float32)
        )
        pred_cls = (
            np.asarray(detections.class_id) if len(detections.class_id) > 0 
            else np.zeros((0,), dtype=np.int32)
        )
        pred_conf = (
            np.asarray(detections.confidence) if len(detections.confidence) > 0 
            else np.zeros((0,), dtype=np.float32)
        )
        order = np.argsort(-pred_conf)
        
        matched = set()
        for pi in order:
            pbox = pred_boxes[pi]
            pcls = int(pred_cls[pi]) + 1  # supervision class_id is zero-based
            best_iou = 0.0
            best_gi = -1
            for gi, g in enumerate(gt):
                if gi in matched or int(g["category_id"]) != pcls:
                    continue
                iou = _iou_xyxy(pbox, _xywh_to_xyxy(g["bbox"]))
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_gi >= 0 and best_iou >= iou_threshold:
                matched.add(best_gi)
                cat_to_tp[pcls] = cat_to_tp.get(pcls, 0) + 1
    
    metrics: Dict[str, float] = {}
    total_tp = 0
    total_gt = 0
    
    print("\n" + "="*70)
    print(f"Recall@{iou_threshold} Results (score threshold: {score_threshold})")
    print("="*70)
    
    for cat in sorted(cat_to_gt.keys()):
        tp = cat_to_tp.get(cat, 0)
        gt_count = cat_to_gt.get(cat, 0)
        rec = tp / max(gt_count, 1)
        metrics[f"class_{cat}_recall@{iou_threshold}"] = rec
        total_tp += tp
        total_gt += gt_count
        
        if cat < len(TOOL_NAMES):
            tool_name = TOOL_NAMES[cat]
            print(f"  {tool_name:15s}: {rec:6.2%} ({tp:4d}/{gt_count:4d} TP/GT)")
        else:
            print(f"  Class {cat:2d}        : {rec:6.2%} ({tp:4d}/{gt_count:4d} TP/GT)")
    
    metrics[f"overall_recall@{iou_threshold}"] = total_tp / max(total_gt, 1)
    print("-"*70)
    print(f"  {'Overall':15s}: {metrics[f'overall_recall@{iou_threshold}']:6.2%} ({total_tp:4d}/{total_gt:4d} TP/GT)")
    print("="*70)
    
    return metrics


def compute_map(
    model, valid_dir: Path, iou_thresholds: List[float] = None, score_threshold: float = 0.3
) -> Dict[str, float]:
    """Compute mAP@IoU thresholds (simplified version)."""
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]
    
    images, anns = _load_coco_annotations(valid_dir / "_annotations.coco.json")
    
    # Collect all predictions and GT
    all_preds: Dict[int, List] = {}  # class_id -> [(score, iou, matched), ...]
    all_gt_counts: Dict[int, int] = {}
    
    print(f"\nComputing mAP on {len(images)} images...")
    
    for image_id, info in tqdm(images.items(), desc="mAP eval"):
        gt = anns.get(image_id, [])
        for g in gt:
            cat = int(g["category_id"])
            all_gt_counts[cat] = all_gt_counts.get(cat, 0) + 1
        
        image_path = valid_dir / info["file_name"]
        if not image_path.exists():
            continue
        
        image = Image.open(image_path).convert("RGB")
        detections = model.predict(image, threshold=score_threshold)
        
        pred_boxes = (
            np.asarray(detections.xyxy) if len(detections.xyxy) > 0 
            else np.zeros((0, 4), dtype=np.float32)
        )
        pred_cls = (
            np.asarray(detections.class_id) if len(detections.class_id) > 0 
            else np.zeros((0,), dtype=np.int32)
        )
        pred_conf = (
            np.asarray(detections.confidence) if len(detections.confidence) > 0 
            else np.zeros((0,), dtype=np.float32)
        )
        
        # Match predictions to GT
        matched_gt = set()
        for pi in range(len(pred_boxes)):
            pbox = pred_boxes[pi]
            pcls = int(pred_cls[pi]) + 1
            score = float(pred_conf[pi])
            
            best_iou = 0.0
            best_gi = -1
            for gi, g in enumerate(gt):
                if gi in matched_gt or int(g["category_id"]) != pcls:
                    continue
                iou = _iou_xyxy(pbox, _xywh_to_xyxy(g["bbox"]))
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            
            if pcls not in all_preds:
                all_preds[pcls] = []
            
            is_matched = best_gi >= 0
            all_preds[pcls].append((score, best_iou, is_matched))
            if is_matched:
                matched_gt.add(best_gi)
    
    # Compute AP for each IoU threshold
    metrics = {}
    
    for iou_thresh in iou_thresholds:
        print(f"\n{'='*70}")
        print(f"mAP@{iou_thresh} Results")
        print("="*70)
        
        total_ap = 0.0
        num_classes = 0
        
        for cat in sorted(all_gt_counts.keys()):
            if cat not in all_preds or len(all_preds[cat]) == 0:
                continue
            
            # Sort by score descending
            preds = sorted(all_preds[cat], key=lambda x: x[0], reverse=True)
            
            tp = 0
            fp = 0
            gt_count = all_gt_counts.get(cat, 0)
            if gt_count == 0:
                continue
            
            precisions = []
            recalls = []
            
            for score, iou, is_matched in preds:
                if iou >= iou_thresh and is_matched:
                    tp += 1
                else:
                    fp += 1
                
                precision = tp / max(tp + fp, 1)
                recall = tp / max(gt_count, 1)
                precisions.append(precision)
                recalls.append(recall)
            
            # Compute AP (area under precision-recall curve)
            if len(precisions) == 0:
                ap = 0.0
            else:
                # Simple AP: average precision at 11 recall points
                ap = 0.0
                for r in np.linspace(0, 1, 11):
                    # Find max precision at recall >= r
                    max_prec = 0.0
                    for rec, prec in zip(recalls, precisions):
                        if rec >= r:
                            max_prec = max(max_prec, prec)
                    ap += max_prec
                ap /= 11.0
            
            metrics[f"class_{cat}_ap@{iou_thresh}"] = ap
            total_ap += ap
            num_classes += 1
            
            if cat < len(TOOL_NAMES):
                tool_name = TOOL_NAMES[cat]
                print(f"  {tool_name:15s}: AP = {ap:6.2%}")
            else:
                print(f"  Class {cat:2d}        : AP = {ap:6.2%}")
        
        map_score = total_ap / max(num_classes, 1)
        metrics[f"map@{iou_thresh}"] = map_score
        print("-"*70)
        print(f"  {'mAP':15s}: {map_score:6.2%}")
        print("="*70)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate RF-DETR model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="H:/vjepa2_complete_windows_20260210_200325/outputs/rfd_eval/checkpoint_best_ema.pth",
        help="Path to RF-DETR checkpoint",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="H:/vjepa2_complete_windows_20260210_200325/data/cholec_coco",
        help="COCO dataset directory",
    )
    parser.add_argument(
        "--model_size",
        choices=["nano", "small", "medium", "large", "base"],
        default="small",
        help="RF-DETR model size",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=8,
        help="Number of classes (7 tool classes + 1 background = 8)",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.3,
        help="Detection score threshold",
    )
    parser.add_argument(
        "--compute_map",
        action="store_true",
        help="Also compute mAP (slower)",
    )
    args = parser.parse_args()
    
    # Build model - try with num_classes parameter first
    model_classes = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
        "base": RFDETRBase,
    }
    try:
        # Try initializing with num_classes (may not be supported in all versions)
        model = model_classes[args.model_size](num_classes=args.num_classes)
        print(f"Initialized RF-DETR {args.model_size} with num_classes={args.num_classes}")
    except TypeError:
        # Fallback: initialize without num_classes, will resize after loading checkpoint
        print(f"RF-DETR {args.model_size} doesn't support num_classes in constructor, will resize after loading")
        model = model_classes[args.model_size]()
    
    # Move model to CUDA if available (needed for device detection during resizing)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # RF-DETR is a wrapper; move the internal model
        # Traverse model.model.model... until we find a torch.nn.Module or something with .to()
        curr = model
        moved = False
        for _ in range(3):
            if hasattr(curr, "model"):
                curr = curr.model
                if hasattr(curr, "to") and isinstance(curr, torch.nn.Module):
                    curr.to(device)
                    moved = True
                    break
            else:
                break
        if not moved and hasattr(model, "to") and isinstance(model, torch.nn.Module):
            model.to(device)
            moved = True
            
        if moved:
            print(f"Model moved to {device}")
        else:
            print("Warning: Could not find internal torch module to move to device")
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            import torch
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
            
            # Debug: print checkpoint keys
            if isinstance(checkpoint, dict):
                print(f"Checkpoint keys: {list(checkpoint.keys())[:10]}...")
            
            # RF-DETR checkpoints may be stored in different formats
            # Try common keys: 'model', 'state_dict', or direct state_dict
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "ema_model" in checkpoint:
                    # EMA checkpoint
                    state_dict = checkpoint["ema_model"]
                else:
                    # Assume it's a state_dict directly (filter out non-tensor keys)
                    state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
            else:
                state_dict = checkpoint
            
            # Infer num_classes from checkpoint (check class_embed weight shape)
            checkpoint_num_classes = None
            for key in state_dict.keys():
                if "class_embed" in key and "weight" in key:
                    checkpoint_num_classes = state_dict[key].shape[0]
                    print(f"Inferred num_classes={checkpoint_num_classes} from checkpoint ({key})")
                    break
            
            if checkpoint_num_classes is None:
                print("Warning: Could not infer num_classes from checkpoint, using default")
                checkpoint_num_classes = args.num_classes
            
            # Get inner model for potential resizing
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
            if inner_model is None:
                inner_model = model
            
            print(f"Using inner_model: {type(inner_model).__name__}")
            
            # Check if model needs resizing (if num_classes wasn't supported in constructor)
            model_num_classes = None
            if inner_model is not None:
                if hasattr(inner_model, "class_embed"):
                    model_num_classes = inner_model.class_embed.weight.shape[0]
                elif hasattr(inner_model, "transformer") and hasattr(inner_model.transformer, "enc_out_class_embed"):
                    enc_embed = inner_model.transformer.enc_out_class_embed
                    if isinstance(enc_embed, torch.nn.ModuleList) and len(enc_embed) > 0:
                        model_num_classes = enc_embed[0].weight.shape[0]
            
            # Get device from model (needed for resized embeddings)
            try:
                device = next(inner_model.parameters()).device
            except (StopIteration, AttributeError):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Resize model class embeddings if needed
            if model_num_classes is not None and model_num_classes != checkpoint_num_classes:
                print(f"Resizing model class embeddings from {model_num_classes} to {checkpoint_num_classes} classes")
                # Resize class_embed
                if hasattr(inner_model, "class_embed"):
                    old_embed = inner_model.class_embed
                    # Create new embedding directly on the correct device
                    new_embed = torch.nn.Linear(old_embed.in_features, checkpoint_num_classes).to(device)
                    # Copy matching weights
                    min_classes = min(model_num_classes, checkpoint_num_classes)
                    new_embed.weight.data[:min_classes] = old_embed.weight.data[:min_classes]
                    if old_embed.bias is not None:
                        new_embed.bias.data[:min_classes] = old_embed.bias.data[:min_classes]
                    inner_model.class_embed = new_embed
                
                # Resize transformer.enc_out_class_embed
                if hasattr(inner_model, "transformer") and hasattr(inner_model.transformer, "enc_out_class_embed"):
                    enc_embed = inner_model.transformer.enc_out_class_embed
                    if isinstance(enc_embed, torch.nn.ModuleList):
                        for i, layer in enumerate(enc_embed):
                            if isinstance(layer, torch.nn.Linear):
                                old_embed = layer
                                # Create new embedding directly on the correct device
                                new_embed = torch.nn.Linear(old_embed.in_features, checkpoint_num_classes).to(device)
                                min_classes = min(model_num_classes, checkpoint_num_classes)
                                new_embed.weight.data[:min_classes] = old_embed.weight.data[:min_classes]
                                if old_embed.bias is not None:
                                    new_embed.bias.data[:min_classes] = old_embed.bias.data[:min_classes]
                                enc_embed[i] = new_embed
            
            # Use full state_dict (resizing should have fixed size mismatches)
            filtered_state_dict = state_dict
            
            # Load into model (may need to access internal model)
            # RF-DETR structure: model.model.model is the actual PyTorch module
            if hasattr(inner_model, "load_state_dict"):
                missing_keys, unexpected_keys = inner_model.load_state_dict(filtered_state_dict, strict=False)
            else:
                # Fallback to model.model.model chain
                if hasattr(model, "model") and hasattr(model.model, "model"):
                    # RF-DETR wraps: model.model.model is the actual PyTorch module
                    missing_keys, unexpected_keys = model.model.model.load_state_dict(filtered_state_dict, strict=False)
                elif hasattr(model, "model"):
                    missing_keys, unexpected_keys = model.model.load_state_dict(filtered_state_dict, strict=False)
                else:
                    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            
            if missing_keys:
                print(f"Note: {len(missing_keys)} keys missing (expected with strict=False)")
            if unexpected_keys:
                print(f"Note: {len(unexpected_keys)} unexpected keys (expected with strict=False)")
            
            print("Checkpoint loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            print("Using pretrained weights only.")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using pretrained weights only.")
    
    # Evaluate
    valid_dir = Path(args.dataset_dir) / "valid"
    if not valid_dir.exists():
        print(f"Error: Validation directory not found: {valid_dir}")
        return
    
    # Recall@0.5
    recall_metrics = compute_recall_at_iou(
        model, valid_dir, iou_threshold=0.5, score_threshold=args.score_threshold
    )
    
    # Optional: mAP
    if args.compute_map:
        map_metrics = compute_map(model, valid_dir, score_threshold=args.score_threshold)
        recall_metrics.update(map_metrics)
    
    # Save results
    output_file = checkpoint_path.parent / "eval_results.json"
    with open(output_file, "w") as f:
        json.dump(recall_metrics, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
