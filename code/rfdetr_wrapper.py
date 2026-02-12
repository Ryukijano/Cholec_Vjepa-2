#!/usr/bin/env python3
"""
RF-DETR wrapper that exposes detections + intermediate features.

The public RF-DETR API returns supervision.Detections. For joint detection + Re-ID
we also need:
  - multi-scale backbone features
  - decoder query features
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image


@dataclass
class RFDETRPrediction:
    detections: Any  # supervision.Detections
    backbone_features: Optional[List[torch.Tensor]]
    decoder_queries: Optional[torch.Tensor]  # [Q, D]
    matched_query_features: Optional[torch.Tensor]  # [N_det, D]


class RFDETRFeatureWrapper:
    """
    Thin wrapper around rfdetr models.

    Notes:
      - Uses forward hooks to capture internal outputs when possible.
      - Falls back gracefully if internals differ across RF-DETR versions.
    """

    def __init__(self, model: Any):
        self.model = model
        self._backbone_cache = None
        self._decoder_cache = None
        self._hooks = []
        self._install_hooks()

    def _get_internal_model(self) -> Optional[torch.nn.Module]:
        # Current package usually nests as model.model.model (RFDETR -> Model -> LWDETR)
        current = self.model
        for attr in ("model", "model"):
            current = getattr(current, attr, None)
            if current is None:
                return None
        return current

    def _install_hooks(self) -> None:
        inner = self._get_internal_model()
        if inner is None:
            return

        def backbone_hook(_mod, _inp, out):
            self._backbone_cache = out

        def transformer_hook(_mod, _inp, out):
            self._decoder_cache = out

        backbone = getattr(inner, "backbone", None)
        transformer = getattr(inner, "transformer", None)
        if backbone is not None:
            self._hooks.append(backbone.register_forward_hook(backbone_hook))
        if transformer is not None:
            self._hooks.append(transformer.register_forward_hook(transformer_hook))

    @staticmethod
    def _to_tensor_image(image: Any, device: torch.device) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            x = image
            if x.dim() == 3:
                x = x.unsqueeze(0)
            return x.to(device)
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 3:
                arr = arr[None, ...]
            x = torch.from_numpy(arr).float()
            if x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            return (x / 255.0).to(device)
        if isinstance(image, Image.Image):
            arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            return x.to(device)
        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _extract_backbone_tensors(backbone_out: Any) -> Optional[List[torch.Tensor]]:
        if backbone_out is None:
            return None
        # Common format: (features, pos)
        features = backbone_out[0] if isinstance(backbone_out, (tuple, list)) else backbone_out
        tensors: List[torch.Tensor] = []
        if isinstance(features, (list, tuple)):
            for f in features:
                if hasattr(f, "tensors"):
                    tensors.append(f.tensors.detach().cpu())
                elif isinstance(f, torch.Tensor):
                    tensors.append(f.detach().cpu())
        elif isinstance(features, torch.Tensor):
            tensors.append(features.detach().cpu())
        return tensors if tensors else None

    @staticmethod
    def _extract_decoder_queries(decoder_out: Any) -> Optional[torch.Tensor]:
        """
        Transformer forward commonly returns:
          hs, ref_unsigmoid, hs_enc, ref_enc
        where hs shape is [L, B, Q, D].
        """
        if decoder_out is None:
            return None
        hs = None
        if isinstance(decoder_out, (tuple, list)) and len(decoder_out) > 0:
            hs = decoder_out[0]
        elif isinstance(decoder_out, torch.Tensor):
            hs = decoder_out
        if hs is None or not isinstance(hs, torch.Tensor):
            return None
        if hs.dim() == 4:
            return hs[-1, 0].detach().cpu()  # [Q, D]
        if hs.dim() == 3:
            return hs[0].detach().cpu()
        return None

    @staticmethod
    def _match_detections_to_queries(
        detections: Any,
        pred_boxes_q: Optional[torch.Tensor],
        query_features: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Match postprocessed detections to nearest query by L1 center-distance.
        pred_boxes_q must be [Q, 4] in xyxy format.
        """
        if query_features is None or pred_boxes_q is None:
            return None
        if not hasattr(detections, "xyxy"):
            return None
        det_xyxy = np.asarray(detections.xyxy)
        if det_xyxy.size == 0:
            return torch.empty((0, query_features.shape[-1]))
        det_t = torch.from_numpy(det_xyxy.astype(np.float32))

        q_centers = 0.5 * (pred_boxes_q[:, :2] + pred_boxes_q[:, 2:])
        d_centers = 0.5 * (det_t[:, :2] + det_t[:, 2:])
        # [N_det, Q]
        dist = (d_centers[:, None, :] - q_centers[None, :, :]).abs().sum(-1)
        q_idx = dist.argmin(dim=1)
        return query_features[q_idx]

    def predict_with_features(
        self,
        image: Any,
        threshold: float = 0.5,
    ) -> RFDETRPrediction:
        # reset caches
        self._backbone_cache = None
        self._decoder_cache = None

        detections = self.model.predict(image, threshold=threshold)

        backbone_features = self._extract_backbone_tensors(self._backbone_cache)
        decoder_queries = self._extract_decoder_queries(self._decoder_cache)

        matched_query_features = None
        inner = self._get_internal_model()
        if inner is not None and decoder_queries is not None:
            # Attempt to run a lightweight raw forward to get per-query boxes.
            # If this fails due to version differences, we still return features.
            try:
                device = next(inner.parameters()).device
                x = self._to_tensor_image(image, device)
                with torch.no_grad():
                    outputs = inner(x)
                pred_boxes = outputs.get("pred_boxes", None) if isinstance(outputs, dict) else None
                if isinstance(pred_boxes, torch.Tensor) and pred_boxes.dim() == 3:
                    # Convert cxcywh -> xyxy in absolute pixel space of input
                    q = pred_boxes[0].detach().cpu()
                    cx, cy, w, h = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                    xyxy = torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)
                    H, W = x.shape[-2], x.shape[-1]
                    scale = torch.tensor([W, H, W, H], dtype=xyxy.dtype)
                    xyxy = xyxy * scale
                    matched_query_features = self._match_detections_to_queries(detections, xyxy, decoder_queries)
            except Exception:
                matched_query_features = None

        return RFDETRPrediction(
            detections=detections,
            backbone_features=backbone_features,
            decoder_queries=decoder_queries,
            matched_query_features=matched_query_features,
        )

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
