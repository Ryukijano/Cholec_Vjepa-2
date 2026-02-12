#!/usr/bin/env python3
"""
Debug script to verify the full V-JEPA pipeline: Encoder -> Decoder -> Re-ID.
Checks:
1. Encoder Feature Map (spatial localization).
2. Decoder Attention & Detections (with NMS).
3. Re-ID Embeddings (similarity matrix).
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
vjepa2_path = project_root / "vjepa2"
src_path = project_root / "vjepa2" / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(vjepa2_path))
sys.path.insert(0, str(current_dir))

from eval_hota import load_models
from models import detection_postprocess
from detection_viz import TOOL_NAMES, make_bbox_overlay

def extract_attention_weights(head, full_tokens, mid_tokens):
    """
    Manually run the first decoder layer to extract cross-attention weights.
    Adapts to LightweightQueryDecoder structure.
    """
    B = full_tokens.shape[0]
    
    # 1. Prepare features (same as head.forward)
    # Temporal pooling (last frame)
    if full_tokens.dim() == 3 and full_tokens.shape[1] > 196:
        D = full_tokens.shape[-1]
        T = full_tokens.shape[1] // 196
        full_tokens = full_tokens.view(B, T, 196, D)
    if full_tokens.dim() == 4:
        full_tokens = full_tokens[:, -1] # [B, 196, D]
        
    if mid_tokens is not None:
        if mid_tokens.dim() == 3 and mid_tokens.shape[1] > 196:
            D_m = mid_tokens.shape[-1]
            T_m = mid_tokens.shape[1] // 196
            mid_tokens = mid_tokens.view(B, T_m, 196, D_m)
        if mid_tokens.dim() == 4:
            mid_tokens = mid_tokens[:, -1]

    # Project and fuse
    tokens_proj = head.input_proj(full_tokens)
    if mid_tokens is not None:
        fused = head.fusion(torch.cat([
            head.mid_projector(mid_tokens), tokens_proj], dim=-1))
    else:
        fused = tokens_proj
        
    fused = fused + head.spatial_pos_embed
    
    # Initialize queries
    global_ctx = head.global_context_proj(fused.mean(dim=1, keepdim=True))
    prior_pos_emb = head.prior_pos_embed(head.prior_boxes)
    queries = (head.query_content.weight + prior_pos_emb).unsqueeze(0).expand(B, -1, -1)
    queries = queries + global_ctx
    
    # 2. Run Layer 0 logic manually to get attention
    layer = head.layers[0]
    
    # Self-attention
    q2 = layer.norm1(queries)
    q2, _ = layer.self_attn(q2, q2, q2)
    queries = queries + q2
    
    # Cross-attention
    q_in = layer.norm2(queries)
    memory = fused
    
    # Inside SpatialCrossAttention
    cross_attn = layer.cross_attn
    B, Q, D = q_in.shape
    N = memory.shape[1]
    
    q = cross_attn.q_proj(q_in).view(B, Q, cross_attn.nheads, cross_attn.head_dim).transpose(1, 2)
    k = cross_attn.k_proj(memory).view(B, N, cross_attn.nheads, cross_attn.head_dim).transpose(1, 2)
    
    # Raw attention scores
    attn = torch.matmul(q, k.transpose(-2, -1)) / (cross_attn.head_dim ** 0.5)
    
    # Add spatial bias
    query_pos = head.query_positions
    token_pos = head.token_positions
    dist2 = ((query_pos.unsqueeze(1) - token_pos.unsqueeze(0)) ** 2).sum(-1) # [Q, N]
    spatial_bias = -cross_attn.spatial_bias_scale * dist2
    attn = attn + spatial_bias.unsqueeze(0).unsqueeze(1)
    
    # Softmax
    attn_weights = attn.softmax(-1) # [B, nheads, Q, N]
    
    return attn_weights, spatial_bias, fused

def plot_attention(frame_img, attn_weights, spatial_bias, pred_boxes, pred_scores, save_path):
    """
    Plot top-3 queries' attention maps.
    attn_weights: [1, nheads, Q, 196]
    """
    # Average over heads
    attn_avg = attn_weights[0].mean(dim=0) # [Q, 196]
    
    # Pick top 3 queries
    top_indices = torch.argsort(pred_scores[0], descending=True)[:3]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original frame
    axes[0, 0].imshow(frame_img)
    axes[0, 0].set_title("Input Frame")
    axes[0, 0].axis('off')
    
    # Spatial Bias (for first query)
    bias_map = spatial_bias[0].reshape(14, 14).cpu().numpy()
    axes[1, 0].imshow(bias_map, cmap='viridis')
    axes[1, 0].set_title("Spatial Bias (Q0)")
    axes[1, 0].axis('off')
    
    for i, idx in enumerate(top_indices):
        idx = idx.item()
        score = pred_scores[0, idx].item()
        
        # Attention map
        att = attn_avg[idx].reshape(14, 14).detach().cpu().numpy()
        att = (att - att.min()) / (att.max() - att.min() + 1e-8)
        
        # Upsample to frame size
        att_img = Image.fromarray((att * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        
        # Overlay
        axes[0, i+1].imshow(frame_img)
        axes[0, i+1].imshow(att_img, cmap='jet', alpha=0.5)
        
        # Draw predicted box
        box = pred_boxes[0, idx].detach().cpu().numpy() # cxcywh
        x, y, w, h = box
        x1, y1 = (x - w/2)*224, (y - h/2)*224
        w, h = w*224, h*224
        rect = plt.Rectangle((x1, y1), w, h, fill=False, color='white', linewidth=2)
        axes[0, i+1].add_patch(rect)
        
        axes[0, i+1].set_title(f"Q{idx} (Score {score:.2f})")
        axes[0, i+1].axis('off')
        
        # Raw attention
        axes[1, i+1].imshow(att, cmap='viridis')
        axes[1, i+1].set_title(f"Raw Attn {idx}")
        axes[1, i+1].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved attention visualization to {save_path}")

def plot_encoder_features(frame_img, fused_features, save_path):
    """
    Visualize the norm of the encoder features (fused with mid-level).
    fused_features: [1, 196, D]
    """
    # Compute norm per spatial token
    feats = fused_features[0] # [196, D]
    norm = feats.norm(dim=-1).reshape(14, 14).detach().cpu().numpy()
    
    # Normalize for visualization
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-8)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(frame_img)
    axes[0].set_title("Input Frame")
    axes[0].axis('off')
    
    # Upsample heatmap
    heatmap = Image.fromarray((norm * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    
    axes[1].imshow(frame_img)
    axes[1].imshow(heatmap, cmap='jet', alpha=0.6)
    axes[1].set_title("Encoder Feature Norm")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved encoder feature visualization to {save_path}")

def plot_reid_similarity(embeddings, det_cls, save_path):
    """
    Plot similarity matrix of Re-ID embeddings for detected objects.
    """
    if embeddings.shape[0] < 2:
        print("Not enough detections for Re-ID similarity matrix.")
        return

    # Cosine similarity
    sim_matrix = torch.mm(embeddings, embeddings.t()).detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im)
    
    # Labels
    labels = [TOOL_NAMES[c.item()] if c < len(TOOL_NAMES) else str(c.item()) for c in det_cls]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Re-ID Embedding Similarity")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Re-ID similarity matrix to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_checkpoint', type=str, required=True)
    parser.add_argument('--reid_checkpoint', type=str, default=None, help="Path to Re-ID checkpoint")
    parser.add_argument('--val_dir', type=str, default='../cholec_dataset/Validation')
    parser.add_argument('--output_dir', type=str, default='../outputs/debug')
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading models...")
    encoder, det_head, mid_hook, reid_head = load_models(args, device)
    
    # Find a video
    val_dir = Path(args.val_dir)
    video_dir = next(val_dir.glob("VID*"))
    print(f"Using video: {video_dir}")
    
    # Load one clip (8 frames)
    frames_dir = video_dir / "Frames"
    frames = sorted(list(frames_dir.glob("*.png")))
    if len(frames) < 8:
        print("Not enough frames")
        return
        
    # Pick a middle clip
    start_idx = len(frames) // 2
    clip_paths = frames[start_idx:start_idx+8]
    
    imgs = []
    for p in clip_paths:
        img = Image.open(p).convert('RGB').resize((224, 224))
        imgs.append(np.array(img, dtype=np.float32) / 255.0)
        
    clip = torch.from_numpy(np.stack(imgs)).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    last_frame_img = (imgs[-1] * 255).astype(np.uint8)
    
    print("Running forward pass...")
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', enabled=True):
            full_tokens = encoder([clip])[0]
            mid_features = mid_hook.get_features()
            
            # 1. Encoder Features
            # Extract attention also returns the fused features used by decoder
            attn_weights, spatial_bias, fused_features = extract_attention_weights(det_head, full_tokens, mid_features)
            
            plot_encoder_features(
                last_frame_img,
                fused_features,
                Path(args.output_dir) / "debug_encoder_features.png"
            )
            
            # 2. Detection Head
            logits, pred_boxes, _, _ = det_head(full_tokens, mid_features)
            
            probs = logits.softmax(-1)[0]
            scores, pred_cls = probs[..., :-1].max(-1)
            
            # Stats
            print("\n--- Detection Stats ---")
            print(f"Max score: {scores.max().item():.4f}")
            print(f"Min score: {scores.min().item():.4f}")
            print(f"Mean score: {scores.mean().item():.4f}")
            print(f"Detections > 0.5: {(scores > 0.5).sum().item()}")
            print(f"Detections > 0.3: {(scores > 0.3).sum().item()}")
            print(f"Detections > 0.1: {(scores > 0.1).sum().item()}")
            
            # Visualize Attention
            plot_attention(
                last_frame_img, 
                attn_weights, 
                spatial_bias, 
                pred_boxes, 
                scores.unsqueeze(0), 
                Path(args.output_dir) / "debug_attention.png"
            )
            
            # 3. NMS & Re-ID
            # Visualize boxes (with NMS so overlay matches eval/viz pipeline)
            det_boxes_nms, det_scores_nms, det_cls_nms = detection_postprocess(
                pred_boxes[0], scores, pred_cls,
                score_thresh=0.8, nms_thresh=0.2, num_classes=7
            )
            
            if det_boxes_nms.shape[0] > 0:
                # Box overlay
                gt_boxes = torch.empty(0, 4)
                gt_labels = torch.empty(0)
                overlay = make_bbox_overlay(
                    torch.from_numpy(imgs[-1]).permute(2, 0, 1),
                    gt_boxes, gt_labels,
                    det_boxes_nms, det_cls_nms, det_scores_nms,
                    score_thresh=0.5
                )
                Image.fromarray(overlay).save(Path(args.output_dir) / "debug_overlay.png")
                print(f"Saved box overlay (after NMS, conf>=0.8) to {Path(args.output_dir) / 'debug_overlay.png'}")
                
                # Re-ID Embeddings
                if args.reid_checkpoint:
                    print("Running Re-ID head on detections...")
                    # ReID head expects list of boxes
                    embs = reid_head(full_tokens, [det_boxes_nms])[0] # [N_det, 128]
                    
                    plot_reid_similarity(
                        embs,
                        det_cls_nms,
                        Path(args.output_dir) / "debug_reid_similarity.png"
                    )
                else:
                    print("No Re-ID checkpoint provided, skipping Re-ID visualization.")
            else:
                print("No detections after NMS (conf>=0.8, nms=0.2) to visualize.")

if __name__ == "__main__":
    main()
