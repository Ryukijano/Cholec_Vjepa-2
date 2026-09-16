"""
Transfer RF-DETR COCO-pretrained decoder weights into our DeformableSurgicalToolDetector.

RF-DETR (LWDETR) and our DeformableSurgicalToolDetector share the same core architecture:
  - DINOv2 backbone (different sizes, we keep our own)
  - Deformable cross-attention decoder layers
  - Self-attention (nn.MultiheadAttention)
  - FFN (linear1/linear2)
  - bbox_embed (3-layer MLP)
  - class_embed (Linear)

This script maps the COCO-pretrained decoder weights from RF-DETR into our model,
skipping incompatible shapes (class_embed num_classes, query count, ref_point_head).

Usage:
    python scripts/got_jepa/transfer_rfdetr_weights.py \
        --rfdetr-ckpt outputs/mot/rfdetr-baseline/checkpoint_best_total.pth \
        --config configs/train_mot/dinov2/ablation-small-detr.yaml \
        --output outputs/mot/ablation-small-detr/rfdetr_init.pth
"""
import argparse
import os
import sys
import re
from typing import Dict, Optional

import torch
import yaml


def build_model_from_config_path(config_path: str):
    """Load config and build model."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from core_app.mot.trainer import build_model_from_config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return build_model_from_config(config), config


def map_rfdetr_to_ours(rfdetr_state: Dict[str, torch.Tensor], our_state: Dict[str, torch.Tensor],
                       num_decoder_layers: int) -> Dict[str, torch.Tensor]:
    """
    Map RF-DETR checkpoint keys to our model keys.

    Returns a dict of {our_key: tensor} for all transferable weights.
    """
    transferred = {}
    skipped = []

    def try_transfer(rfdetr_key, our_key, name=""):
        if rfdetr_key in rfdetr_state and our_key in our_state:
            rfdetr_shape = rfdetr_state[rfdetr_key].shape
            our_shape = our_state[our_key].shape
            if rfdetr_shape == our_shape:
                transferred[our_key] = rfdetr_state[rfdetr_key].clone()
                return True
            else:
                # Try slicing for class_embed (8 classes -> 7)
                if rfdetr_key.endswith('weight'):
                    # Try slicing dim 0 (output) or dim 1 (input)
                    if rfdetr_shape[0] == our_shape[0] and rfdetr_shape[1] > our_shape[1]:
                        transferred[our_key] = rfdetr_state[rfdetr_key][:, :our_shape[1]].clone()
                        print(f"  Sliced {name}: {rfdetr_shape} -> {our_shape} (dim=1)")
                        return True
                    elif rfdetr_shape[0] > our_shape[0] and rfdetr_shape[1] == our_shape[1]:
                        transferred[our_key] = rfdetr_state[rfdetr_key][:our_shape[0]].clone()
                        print(f"  Sliced {name}: {rfdetr_shape} -> {our_shape} (dim=0)")
                        return True
                elif rfdetr_key.endswith('bias') and rfdetr_shape[0] > our_shape[0]:
                    transferred[our_key] = rfdetr_state[rfdetr_key][:our_shape[0]].clone()
                    print(f"  Sliced {name}: {rfdetr_shape} -> {our_shape} (dim=0)")
                    return True
                else:
                    skipped.append(f"Shape mismatch: {name} | rfdetr {rfdetr_key} {rfdetr_shape} vs our {our_key} {our_shape}")
                    return False
        return False

    # --- Decoder layers ---
    rfdetr_layer_count = 3  # RF-DETR has 3 decoder layers
    transfer_layers = min(num_decoder_layers, rfdetr_layer_count)

    print(f"\nTransferring {transfer_layers}/{num_decoder_layers} decoder layers from RF-DETR")

    for i in range(transfer_layers):
        prefix_rfdetr = f"transformer.decoder.layers.{i}"
        prefix_ours = f"detr.decoder_layers.{i}"

        # Self-attention (nn.MultiheadAttention — same structure)
        try_transfer(f"{prefix_rfdetr}.self_attn.in_proj_weight", f"{prefix_ours}.self_attn.in_proj_weight", f"layer{i}.self_attn.in_proj_weight")
        try_transfer(f"{prefix_rfdetr}.self_attn.in_proj_bias", f"{prefix_ours}.self_attn.in_proj_bias", f"layer{i}.self_attn.in_proj_bias")
        try_transfer(f"{prefix_rfdetr}.self_attn.out_proj.weight", f"{prefix_ours}.self_attn.out_proj.weight", f"layer{i}.self_attn.out_proj.weight")
        try_transfer(f"{prefix_rfdetr}.self_attn.out_proj.bias", f"{prefix_ours}.self_attn.out_proj.bias", f"layer{i}.self_attn.out_proj.bias")

        # norm1 -> self_attn_norm
        try_transfer(f"{prefix_rfdetr}.norm1.weight", f"{prefix_ours}.self_attn_norm.weight", f"layer{i}.self_attn_norm.weight")
        try_transfer(f"{prefix_rfdetr}.norm1.bias", f"{prefix_ours}.self_attn_norm.bias", f"layer{i}.self_attn_norm.bias")

        # Cross-attention (deformable — same module structure)
        try_transfer(f"{prefix_rfdetr}.cross_attn.sampling_offsets.weight", f"{prefix_ours}.cross_attn.sampling_offsets.weight", f"layer{i}.cross_attn.sampling_offsets.weight")
        try_transfer(f"{prefix_rfdetr}.cross_attn.sampling_offsets.bias", f"{prefix_ours}.cross_attn.sampling_offsets.bias", f"layer{i}.cross_attn.sampling_offsets.bias")
        try_transfer(f"{prefix_rfdetr}.cross_attn.attention_weights.weight", f"{prefix_ours}.cross_attn.attention_weights.weight", f"layer{i}.cross_attn.attention_weights.weight")
        try_transfer(f"{prefix_rfdetr}.cross_attn.attention_weights.bias", f"{prefix_ours}.cross_attn.attention_weights.bias", f"layer{i}.cross_attn.attention_weights.bias")
        try_transfer(f"{prefix_rfdetr}.cross_attn.value_proj.weight", f"{prefix_ours}.cross_attn.value_proj.weight", f"layer{i}.cross_attn.value_proj.weight")
        try_transfer(f"{prefix_rfdetr}.cross_attn.value_proj.bias", f"{prefix_ours}.cross_attn.value_proj.bias", f"layer{i}.cross_attn.value_proj.bias")
        try_transfer(f"{prefix_rfdetr}.cross_attn.output_proj.weight", f"{prefix_ours}.cross_attn.output_proj.weight", f"layer{i}.cross_attn.output_proj.weight")
        try_transfer(f"{prefix_rfdetr}.cross_attn.output_proj.bias", f"{prefix_ours}.cross_attn.output_proj.bias", f"layer{i}.cross_attn.output_proj.bias")

        # norm2 -> cross_attn.layer_norm (inside DeformableCrossAttention)
        try_transfer(f"{prefix_rfdetr}.norm2.weight", f"{prefix_ours}.cross_attn.layer_norm.weight", f"layer{i}.cross_attn.layer_norm.weight")
        try_transfer(f"{prefix_rfdetr}.norm2.bias", f"{prefix_ours}.cross_attn.layer_norm.bias", f"layer{i}.cross_attn.layer_norm.bias")

        # FFN: linear1 -> ffn.0, linear2 -> ffn.3
        try_transfer(f"{prefix_rfdetr}.linear1.weight", f"{prefix_ours}.ffn.0.weight", f"layer{i}.ffn.0.weight")
        try_transfer(f"{prefix_rfdetr}.linear1.bias", f"{prefix_ours}.ffn.0.bias", f"layer{i}.ffn.0.bias")
        try_transfer(f"{prefix_rfdetr}.linear2.weight", f"{prefix_ours}.ffn.3.weight", f"layer{i}.ffn.3.weight")
        try_transfer(f"{prefix_rfdetr}.linear2.bias", f"{prefix_ours}.ffn.3.bias", f"layer{i}.ffn.3.bias")

        # norm3 -> ffn_norm
        try_transfer(f"{prefix_rfdetr}.norm3.weight", f"{prefix_ours}.ffn_norm.weight", f"layer{i}.ffn_norm.weight")
        try_transfer(f"{prefix_rfdetr}.norm3.bias", f"{prefix_ours}.ffn_norm.bias", f"layer{i}.ffn_norm.bias")

    # --- Bbox embed (3-layer MLP) ---
    # RF-DETR: bbox_embed.layers.{0,1,2} -> ours: bbox_embed.{0,2,4} (Sequential with ReLU at 1,3)
    try_transfer("bbox_embed.layers.0.weight", "detr.bbox_embed.0.weight", "bbox_embed.0.weight")
    try_transfer("bbox_embed.layers.0.bias", "detr.bbox_embed.0.bias", "bbox_embed.0.bias")
    try_transfer("bbox_embed.layers.1.weight", "detr.bbox_embed.2.weight", "bbox_embed.2.weight")
    try_transfer("bbox_embed.layers.1.bias", "detr.bbox_embed.2.bias", "bbox_embed.2.bias")
    try_transfer("bbox_embed.layers.2.weight", "detr.bbox_embed.4.weight", "bbox_embed.4.weight")
    try_transfer("bbox_embed.layers.2.bias", "detr.bbox_embed.4.bias", "bbox_embed.4.bias")

    # --- Class embed (slice 7 from 8 COCO classes — drop "specimen_bag") ---
    try_transfer("class_embed.weight", "detr.class_embed.weight", "class_embed.weight")
    try_transfer("class_embed.bias", "detr.class_embed.bias", "class_embed.bias")

    # --- Query init: select top-k from RF-DETR's 3900 queries ---
    if "query_feat.weight" in rfdetr_state and "detr.query_init.learned_queries.weight" in our_state:
        rfdetr_queries = rfdetr_state["query_feat.weight"]  # (3900, 256)
        our_queries_shape = our_state["detr.query_init.learned_queries.weight"].shape  # (num_queries, 256)
        num_queries = our_queries_shape[0]
        # Take first N queries (they're ordered by importance in RF-DETR)
        transferred["detr.query_init.learned_queries.weight"] = rfdetr_queries[:num_queries].clone()
        print(f"  Transferred query_init: {rfdetr_queries.shape} -> {our_queries_shape} (first {num_queries})")

    # Report
    print(f"\nTransferred: {len(transferred)}/{len(our_state)} parameters")
    if skipped:
        print(f"\nSkipped ({len(skipped)}):")
        for s in skipped:
            print(f"  {s}")

    # List untransferred DETR params
    untransferred_detr = [k for k in our_state if k.startswith("detr.") and k not in transferred]
    if untransferred_detr:
        print(f"\nUntransferred DETR params ({len(untransferred_detr)}):")
        for k in sorted(untransferred_detr):
            print(f"  {k}: {our_state[k].shape}")

    return transferred


def main():
    parser = argparse.ArgumentParser(description="Transfer RF-DETR weights to our DeformableSurgicalToolDetector")
    parser.add_argument("--rfdetr-ckpt", type=str, default="outputs/mot/rfdetr-baseline/checkpoint_best_total.pth")
    parser.add_argument("--config", type=str, required=True, help="Path to ablation config YAML")
    parser.add_argument("--output", type=str, default=None, help="Output checkpoint path")
    args = parser.parse_args()

    # Load RF-DETR checkpoint
    print(f"Loading RF-DETR checkpoint: {args.rfdetr_ckpt}")
    rfdetr_ckpt = torch.load(args.rfdetr_ckpt, map_location="cpu", weights_only=False)
    rfdetr_state = rfdetr_ckpt.get("model", rfdetr_ckpt.get("state_dict", rfdetr_ckpt))
    print(f"  RF-DETR state dict: {len(rfdetr_state)} keys")

    # Build our model
    print(f"\nBuilding model from config: {args.config}")
    model, config = build_model_from_config_path(args.config)
    our_state = model.state_dict()
    print(f"  Our model state dict: {len(our_state)} keys")

    # Get decoder layer count
    detr_cfg = config.get("detr", {})
    num_decoder_layers = detr_cfg.get("num_decoder_layers", 5)
    print(f"  Decoder layers: {num_decoder_layers}")

    # Map weights
    transferred = map_rfdetr_to_ours(rfdetr_state, our_state, num_decoder_layers)

    # Load transferred weights into model
    print(f"\nLoading transferred weights into model...")
    missing, unexpected = model.load_state_dict(transferred, strict=False)
    print(f"  Missing: {len(missing)} keys (expected — encoder, neck, etc.)")
    print(f"  Unexpected: {len(unexpected)} keys")

    # Save
    output_path = args.output or f"outputs/mot/rfdetr_init_{os.path.basename(args.config).replace('.yaml', '')}.pth"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "rfdetr_source": args.rfdetr_ckpt,
        "transferred_keys": list(transferred.keys()),
    }, output_path)
    print(f"\nSaved initialized model to: {output_path}")
    print(f"  Transferred {len(transferred)} decoder weights from RF-DETR COCO pretraining")


if __name__ == "__main__":
    main()
