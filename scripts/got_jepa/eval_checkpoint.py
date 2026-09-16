#!/usr/bin/env python3
"""
Evaluate a MOT checkpoint — detection mAP, optional HOTA/MOTA rollout.

Usage:
    # Detection mAP on val split
    python scripts/eval_checkpoint.py \\
        --config configs/train_mot/dinov2/cholec20-mot-stage3-joint-finetune-vits.yaml \\
        --checkpoint outputs/mot/cholec20-stage3-joint-finetune-vits/best.pth.tar \\
        --split val

    # MOT metrics with smoke/bleeding/occlusion breakdown on CT20 test
    python scripts/eval_checkpoint.py \\
        --config configs/train_mot/dinov2/cholec20-mot-stage3-joint-finetune-vits.yaml \\
        --checkpoint outputs/mot/cholec20-stage3-joint-finetune-vits/best.pth.tar \\
        --split test --mot-eval --stratify-smoke
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from tqdm import tqdm

from core_app.mot.data import MOTCholecDataset, mot_collate_fn
from core_app.mot.det_metrics import compute_map_from_detr_outputs
from core_app.mot.eval import (
    STRATUM_NAMES,
    build_ct20_gt_mot_rows,
    build_frame_challenges,
    ct20_split_video_dirs,
    evaluate_stratified_mot,
    list_ct20_frame_numbers,
    load_ct20_video_tensor,
    mot_rollout,
    resolve_ct20_frame_source,
    _ct20_json_path,
)
from core_app.utils.checkpoint import load_checkpoint

_LOG = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MOT checkpoint")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--checkpoint", required=True, help="Checkpoint .pth.tar")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--data-root", default=None, help="Override dataset root")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=0, help="Limit batches (0=all)")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--mot-eval",
        action="store_true",
        help="Run tracker rollout + HOTA/MOTA (requires CT20 frames on disk).",
    )
    p.add_argument(
        "--stratify-smoke",
        action="store_true",
        help="Report smoke/bleeding/occlusion-stratified HOTA/MOTA (implies --mot-eval).",
    )
    p.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Limit MOT rollout to first N videos (0=all).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit MOT rollout to first N frames per video (0=all).",
    )
    p.add_argument(
        "--mot-output",
        default=None,
        help="Optional directory to write stratified MOT metrics JSON.",
    )
    return p.parse_args()


@torch.no_grad()
def _eval_detection_map(args, config, model) -> None:
    data_cfg = config.get("data", {})
    img_size = data_cfg.get("img_size", 392)
    clip_length = data_cfg.get("clip_length", 3)
    data_root = args.data_root or data_cfg["datasets"][0]

    ds = MOTCholecDataset(
        data_root=data_root,
        split=args.split,
        clip_length=clip_length,
        img_size=img_size,
        training=False,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=mot_collate_fn,
        drop_last=False,
    )
    _LOG.info(f"Dataset: {len(ds)} clips, {len(loader)} batches")

    all_pred_logits: List[torch.Tensor] = []
    all_pred_boxes: List[torch.Tensor] = []
    all_targets: List[Dict] = []
    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Eval")):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break

        current_video = batch["current_video"].to(args.device)
        out = model(
            current_video=current_video,
            per_track_targets=batch["per_track_targets"],
            detr_targets=batch["detr_targets"],
            reid_labels=batch["reid_labels"],
            mode="train",
        )

        if out.get("total_loss") is not None:
            total_loss += float(out["total_loss"].item())
            total_batches += 1

        if "detr" in out:
            detr = out["detr"]
            pred = detr.get("pred", detr)
            if "class_logits" in pred and "pred_boxes" in pred:
                all_pred_logits.append(pred["class_logits"].cpu())
                all_pred_boxes.append(pred["pred_boxes"].cpu())
                all_targets.extend(batch["detr_targets"])

    avg_loss = total_loss / max(total_batches, 1)
    _LOG.info(f"Avg loss: {avg_loss:.4f}")

    if all_pred_logits:
        num_classes = config.get("detr", {}).get("num_tools", 7)
        map_metrics = compute_map_from_detr_outputs(
            pred_logits=torch.cat(all_pred_logits),
            pred_boxes=torch.cat(all_pred_boxes),
            targets=all_targets,
            num_classes=num_classes,
        )
        for k, v in map_metrics.items():
            _LOG.info(f"  {k}: {v:.4f}")
    else:
        _LOG.warning("No predictions accumulated — mAP unavailable")


@torch.no_grad()
def _eval_mot_rollout(args, config, model) -> Dict[str, Dict[str, float]]:
    data_cfg = config.get("data", {})
    img_size = data_cfg.get("img_size", 392)
    clip_length = data_cfg.get("clip_length", 3)
    data_root = Path(args.data_root or data_cfg["datasets"][0])
    device = torch.device(args.device)

    model.configure_tracker_for_eval()

    video_dirs = ct20_split_video_dirs(data_root, args.split)
    if args.max_videos > 0:
        video_dirs = video_dirs[: args.max_videos]

    aggregated: Dict[str, List[float]] = {}
    per_video: Dict[str, Dict[str, Dict[str, float]]] = {}

    for video_dir in tqdm(video_dirs, desc="MOT rollout"):
        json_path = _ct20_json_path(video_dir)
        frame_numbers, frames_dir, mp4_path = resolve_ct20_frame_source(video_dir)
        if args.max_frames > 0:
            frame_numbers = frame_numbers[: args.max_frames]
        if not frame_numbers:
            _LOG.warning("Skipping %s — no frames or MP4", video_dir.name)
            continue

        with open(json_path) as f:
            annot_data = json.load(f)

        gt_rows = build_ct20_gt_mot_rows(json_path, frame_numbers, img_size, img_size)
        frame_challenges = build_frame_challenges(
            annot_data.get("annotations", {}), frame_numbers,
        )

        frames = load_ct20_video_tensor(
            frames_dir or (video_dir / "Frames"),
            frame_numbers,
            img_size,
            max_frames=0,
            mp4_path=mp4_path,
        )
        pred_rows = mot_rollout(
            model, frames, clip_len=clip_length, img_size=img_size, device=device,
        )

        strata = STRATUM_NAMES if args.stratify_smoke else ("overall",)
        video_metrics = evaluate_stratified_mot(
            gt_rows=gt_rows,
            pred_rows=pred_rows,
            frame_challenges=frame_challenges,
            seq_length=len(frame_numbers),
            width=img_size,
            height=img_size,
            strata=strata,
        )
        per_video[video_dir.name] = video_metrics

        for stratum, metrics in video_metrics.items():
            for key, val in metrics.items():
                if key.startswith("num_"):
                    continue
                aggregated.setdefault(stratum, {}).setdefault(key, []).append(val)

    # Mean across videos per stratum.
    summary: Dict[str, Dict[str, float]] = {}
    for stratum, metric_lists in aggregated.items():
        summary[stratum] = {
            key: sum(vals) / len(vals) for key, vals in metric_lists.items() if vals
        }

    _LOG.info("=== MOT metrics (%s split, %d videos) ===", args.split, len(video_dirs))
    for stratum in sorted(summary.keys()):
        m = summary[stratum]
        hota = m.get("HOTA", m.get("HOTA.HOTA", float("nan")))
        mota = m.get("MOTA", m.get("CLEAR.MOTA", float("nan")))
        idf1 = m.get("IDF1", m.get("Identity.IDF1", float("nan")))
        _LOG.info(
            "  %-14s  HOTA=%.2f  MOTA=%.2f  IDF1=%.2f",
            stratum, hota * 100, mota * 100, idf1 * 100,
        )

    if args.mot_output:
        out_path = Path(args.mot_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "per_video": per_video}
        out_path.write_text(json.dumps(payload, indent=2))
        _LOG.info("Wrote MOT metrics to %s", out_path)

    return summary


@torch.no_grad()
def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.stratify_smoke:
        args.mot_eval = True

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from core_app.mot.trainer import build_model_from_config
    model = build_model_from_config(config)
    model.eval()
    model.to(args.device)

    ckpt = load_checkpoint(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model"], strict=False)
    _LOG.info("Loaded checkpoint from epoch %s", ckpt.get("epoch", "?"))

    if args.mot_eval:
        _eval_mot_rollout(args, config, model)
    else:
        _eval_detection_map(args, config, model)


if __name__ == "__main__":
    main()
