"""
Wandb logging utilities for Surgical MOT training.
"""
from __future__ import annotations
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

_LOG = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    wandb = None

_TOOL_NAMES = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator", "specimen_bag"]


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _denorm_for_vis(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.clone()
    if t.dim() == 4:
        t = t[0]
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(3, 1, 1)
    t = t * std + mean
    t = torch.clamp(t, 0.0, 1.0)
    return _to_numpy(t).transpose(1, 2, 0)


def _cxcywh_to_xyxy(boxes: torch.Tensor, img_size: int = 1) -> np.ndarray:
    b = _to_numpy(boxes)
    if b.ndim == 1:
        b = b[np.newaxis, :]
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x1 = (cx - 0.5 * w) * img_size
    y1 = (cy - 0.5 * h) * img_size
    x2 = (cx + 0.5 * w) * img_size
    y2 = (cy + 0.5 * h) * img_size
    return np.stack([x1, y1, x2, y2], axis=1)


class WandbLogger:
    """Wrapper around wandb that degrades gracefully when unavailable."""

    def __init__(
        self,
        config: Dict[str, Any],
        enabled: bool = True,
        project: str = "surgical-mot",
        entity: Optional[str] = None,
        group: Optional[str] = None,
        job_type: str = "train",
        tags: Optional[List[str]] = None,
        name: Optional[str] = None,
    ):
        self._enabled = enabled and _WANDB_AVAILABLE
        self._run = None
        self._global_step = 0
        self._alert_sent = set()

        if not _WANDB_AVAILABLE and enabled:
            warnings.warn("wandb not installed. Install: pip install wandb", stacklevel=2)
            self._enabled = False

        if self._enabled:
            run_name = name or config.get("meta", {}).get("run_name")
            wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                group=group,
                job_type=job_type,
                tags=tags or [],
                config=config,
                reinit=True,
            )
            self._run = wandb.run
            _LOG.info(f"wandb run: {self._run.url}")

    def watch_model(self, model: nn.Module, log: str = "gradients", log_freq: int = 100):
        if not self._enabled:
            return
        wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self):
        if self._enabled and self._run is not None:
            wandb.finish()
            _LOG.info("wandb run finished.")

    def log_scalars(self, scalars: Dict[str, Any], step: Optional[int] = None):
        if not self._enabled:
            return
        s = step if step is not None else self._global_step
        wandb.log(scalars, step=s)

    def log_lr(self, lr: float, step: Optional[int] = None):
        self.log_scalars({"train/lr": lr}, step=step)

    def log_grad_norm(self, model: nn.Module, step: Optional[int] = None):
        if not self._enabled:
            return
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += float(p.grad.data.norm(2).item() ** 2)
        total_norm = total_norm ** 0.5
        self.log_scalars({"train/grad_norm": total_norm}, step=step)

    def log_images_with_boxes(
        self,
        images: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_labels: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        gt_labels: Optional[torch.Tensor] = None,
        img_size: int = 392,
        step: Optional[int] = None,
        max_images: int = 4,
        key: str = "val/predictions",
    ):
        if not self._enabled:
            return
        if images.dim() == 3:
            images = images.unsqueeze(0)
        n = min(images.size(0), max_images)
        wandb_images = []
        for i in range(n):
            img_np = _denorm_for_vis(images[i])
            h, w = img_np.shape[:2]
            pboxes = _cxcywh_to_xyxy(pred_boxes, img_size=w)
            pboxes = np.clip(pboxes, 0, max(h, w))
            pred_box_data = []
            for b, lbl, scr in zip(pboxes, _to_numpy(pred_labels), _to_numpy(pred_scores)):
                cls_name = _TOOL_NAMES[int(lbl)] if int(lbl) < len(_TOOL_NAMES) else str(int(lbl))
                pred_box_data.append({
                    "position": {"minX": float(b[0]), "minY": float(b[1]), "maxX": float(b[2]), "maxY": float(b[3])},
                    "class_id": int(lbl),
                    "box_caption": f"{cls_name} {scr:.2f}",
                    "scores": {"confidence": float(scr)},
                })
            boxes_dict = {
                "predictions": {"box_data": pred_box_data, "class_labels": {i: n for i, n in enumerate(_TOOL_NAMES)}}
            }
            if gt_boxes is not None and gt_labels is not None:
                gboxes = _cxcywh_to_xyxy(gt_boxes, img_size=w)
                gboxes = np.clip(gboxes, 0, max(h, w))
                gt_box_data = []
                for b, lbl in zip(gboxes, _to_numpy(gt_labels)):
                    cls_name = _TOOL_NAMES[int(lbl)] if int(lbl) < len(_TOOL_NAMES) else str(int(lbl))
                    gt_box_data.append({
                        "position": {"minX": float(b[0]), "minY": float(b[1]), "maxX": float(b[2]), "maxY": float(b[3])},
                        "class_id": int(lbl),
                        "box_caption": f"GT {cls_name}",
                    })
                boxes_dict["ground_truth"] = {"box_data": gt_box_data, "class_labels": {i: n for i, n in enumerate(_TOOL_NAMES)}}
            wandb_images.append(wandb.Image(img_np, boxes=boxes_dict))
        self.log_scalars({key: wandb_images}, step=step)

    def log_clean_vs_corrupted(
        self,
        clean: torch.Tensor,
        corrupted: torch.Tensor,
        step: Optional[int] = None,
        max_images: int = 2,
    ):
        if not self._enabled:
            return
        if clean.dim() == 3:
            clean = clean.unsqueeze(0)
            corrupted = corrupted.unsqueeze(0)
        n = min(clean.size(0), max_images)
        cols = []
        for i in range(n):
            cols.append(wandb.Image(_denorm_for_vis(clean[i]), caption="clean"))
            cols.append(wandb.Image(_denorm_for_vis(corrupted[i]), caption="corrupted"))
        self.log_scalars({"stage2/clean_vs_corrupted": cols}, step=step)

    def log_track_trajectories(
        self,
        frame_sequence: torch.Tensor,
        track_rows: List[Tuple[int, int, float, float, float, float, float]],
        img_size: int = 392,
        step: Optional[int] = None,
        max_frames: int = 16,
    ):
        if not self._enabled:
            return
        T = frame_sequence.size(0)
        stride = max(1, T // max_frames)
        frames_to_log = list(range(0, T, stride))[:max_frames]
        wandb_images = []
        for t in frames_to_log:
            img_np = _denorm_for_vis(frame_sequence[t])
            h, w = img_np.shape[:2]
            frame_boxes = [r for r in track_rows if int(r[0]) == t]
            box_data = []
            for _, tid, x, y, bw, bh, conf in frame_boxes:
                box_data.append({
                    "position": {"minX": float(x), "minY": float(y), "maxX": float(x + bw), "maxY": float(y + bh)},
                    "class_id": int(tid) % 20,
                    "box_caption": f"ID:{tid}",
                    "scores": {"confidence": float(conf)},
                })
            boxes_dict = {
                "predictions": {
                    "box_data": box_data,
                    "class_labels": {i: f"ID_{i}" for i in range(20)},
                }
            }
            wandb_images.append(wandb.Image(img_np, boxes=boxes_dict, caption=f"frame_{t:04d}"))
        self.log_scalars({"mot/trajectory_frames": wandb_images}, step=step)

    def log_mot_metrics_table(
        self,
        metrics: Dict[str, Dict[str, float]],
        step: Optional[int] = None,
    ):
        if not self._enabled:
            return
        columns = ["sequence"] + sorted({k for seq in metrics.values() for k in seq.keys()})
        data = []
        for seq_name, seq_metrics in metrics.items():
            row = [seq_name] + [seq_metrics.get(c, 0.0) for c in columns[1:]]
            data.append(row)
        table = wandb.Table(columns=columns, data=data)
        self.log_scalars({"eval/mot_metrics_table": table}, step=step)

    def log_histogram(self, values: torch.Tensor, key: str, step: Optional[int] = None):
        if not self._enabled:
            return
        wandb.log({key: wandb.Histogram(_to_numpy(values).flatten())}, step=step)

    def log_bbox_distribution(self, bboxes: torch.Tensor, step: Optional[int] = None):
        if not self._enabled or bboxes.numel() == 0:
            return
        b = _to_numpy(bboxes)
        w, h = b[:, 2], b[:, 3]
        ar = w / (h + 1e-6)
        self.log_histogram(torch.from_numpy(w), "bbox/widths", step=step)
        self.log_histogram(torch.from_numpy(h), "bbox/heights", step=step)
        self.log_histogram(torch.from_numpy(ar), "bbox/aspect_ratios", step=step)

    def log_jepa_collapse_monitor(
        self,
        inv_loss: float,
        cov_loss: float,
        step: Optional[int] = None,
    ):
        if not self._enabled:
            return
        self.log_scalars(
            {
                "jepa/inv_loss": inv_loss,
                "jepa/cov_loss": cov_loss,
                "jepa/inv_cov_ratio": inv_loss / (abs(cov_loss) + 1e-8),
            },
            step=step,
        )
        if inv_loss < 0.001 and abs(cov_loss) < 0.001:
            alert_key = "collapse_detected"
            if alert_key not in self._alert_sent:
                wandb.alert(
                    title="JEPA Collapse Detected",
                    text=(
                        f"jepa_inv={inv_loss:.6f} and jepa_cov={cov_loss:.6f} near zero. "
                        "Predictor may have collapsed. Lower corruption strength or increase jepa_inv_weight."
                    ),
                    level=wandb.AlertLevel.WARN,
                )
                self._alert_sent.add(alert_key)

    def log_epoch_summary(self, train_stats: Dict[str, float], val_stats: Dict[str, float], epoch: int, step: int):
        if not self._enabled:
            return
        scalars = {}
        for k, v in train_stats.items():
            scalars[f"epoch/train_{k}"] = v
        for k, v in val_stats.items():
            scalars[f"epoch/val_{k}"] = v
        scalars["epoch/number"] = epoch
        self.log_scalars(scalars, step=step)
