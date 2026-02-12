"""
Detection training visualizations for wandb/tensorboard.

1. Bounding box overlays (GT vs Predicted) on input frames
2. Per-class recall bar chart
3. Confusion matrix (predicted vs GT after Hungarian matching)
4. Cross-attention heatmaps (where each query attends in the image)
5. Query-to-tool assignment visualization
"""

import io
import math
from collections import defaultdict
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F

logger = getLogger(__name__)

TOOL_NAMES = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]
TOOL_COLORS = [
    (31, 119, 180),   # Grasper - blue
    (255, 127, 14),   # Bipolar - orange
    (44, 160, 44),    # Hook - green
    (214, 39, 40),    # Scissors - red
    (148, 103, 189),  # Clipper - purple
    (140, 86, 75),    # Irrigator - brown
    (227, 119, 194),  # SpecimenBag - pink
]
NO_OBJ_COLOR = (180, 180, 180)


def _fig_to_array(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    from PIL import Image
    img = np.array(Image.open(buf).convert("RGB"))
    buf.close()
    return img


def box_cxcywh_to_xyxy_np(boxes):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2] numpy."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bounding Box Overlay Visualization
# ─────────────────────────────────────────────────────────────────────────────

def make_bbox_overlay(
    frame,
    gt_boxes_cxcywh,
    gt_labels,
    pred_boxes_cxcywh,
    pred_labels,
    pred_scores,
    num_classes=7,
    score_thresh=0.3,
    frame_size=224,
):
    """Draw GT and predicted bounding boxes on a frame.
    
    Args:
        frame: [C, H, W] tensor or [H, W, C] numpy, normalized [0,1]
        gt_boxes_cxcywh: [M, 4] GT boxes
        gt_labels: [M] GT class indices
        pred_boxes_cxcywh: [Q, 4] predicted boxes
        pred_labels: [Q] predicted class indices
        pred_scores: [Q] confidence scores
        
    Returns:
        [H, W, 3] numpy uint8 image with overlaid boxes
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    if isinstance(frame, torch.Tensor):
        if frame.dim() == 3 and frame.shape[0] in (1, 3):
            frame = frame.permute(1, 2, 0).cpu().numpy()
        else:
            frame = frame.cpu().numpy()
    
    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
    H, W = frame.shape[:2]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: GT boxes
    axes[0].imshow(frame)
    axes[0].set_title("Ground Truth", fontsize=11)
    gt_np = gt_boxes_cxcywh.cpu().numpy() if isinstance(gt_boxes_cxcywh, torch.Tensor) else gt_boxes_cxcywh
    gt_xyxy = box_cxcywh_to_xyxy_np(gt_np) * np.array([W, H, W, H])
    gt_l = gt_labels.cpu().numpy() if isinstance(gt_labels, torch.Tensor) else gt_labels
    
    for i, (box, cls) in enumerate(zip(gt_xyxy, gt_l)):
        x1, y1, x2, y2 = box
        c = TOOL_COLORS[cls % len(TOOL_COLORS)]
        color = tuple(v/255 for v in c)
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                  linewidth=2, edgecolor=color, facecolor='none')
        axes[0].add_patch(rect)
        name = TOOL_NAMES[cls] if cls < len(TOOL_NAMES) else f"cls{cls}"
        axes[0].text(x1, y1-3, name, fontsize=8, color='white',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))
    axes[0].axis("off")
    
    # Right: Predicted boxes (filtered by score)
    axes[1].imshow(frame)
    axes[1].set_title("Predictions", fontsize=11)
    pred_np = pred_boxes_cxcywh.cpu().numpy() if isinstance(pred_boxes_cxcywh, torch.Tensor) else pred_boxes_cxcywh
    pred_xyxy = box_cxcywh_to_xyxy_np(pred_np) * np.array([W, H, W, H])
    pred_l = pred_labels.cpu().numpy() if isinstance(pred_labels, torch.Tensor) else pred_labels
    pred_s = pred_scores.cpu().numpy() if isinstance(pred_scores, torch.Tensor) else pred_scores
    
    for i, (box, cls, score) in enumerate(zip(pred_xyxy, pred_l, pred_s)):
        if score < score_thresh or cls >= num_classes:
            continue
        x1, y1, x2, y2 = box
        c = TOOL_COLORS[cls % len(TOOL_COLORS)]
        color = tuple(v/255 for v in c)
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor=color, facecolor='none',
                                  linestyle='--')
        axes[1].add_patch(rect)
        name = TOOL_NAMES[cls] if cls < len(TOOL_NAMES) else f"cls{cls}"
        axes[1].text(x1, y1-3, f"{name} {score:.2f}", fontsize=8, color='white',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    axes[1].axis("off")
    
    plt.tight_layout()
    arr = _fig_to_array(fig)
    plt.close(fig)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-Class Recall Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def make_recall_bar_chart(per_class_recall, per_class_gt, overall_recall, epoch, num_classes=7):
    """Create a bar chart of per-class recall with GT counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    names = [TOOL_NAMES[c] if c < len(TOOL_NAMES) else f"cls{c}" for c in range(num_classes)]
    recalls = [per_class_recall[c].item() if isinstance(per_class_recall[c], torch.Tensor) 
               else per_class_recall[c] for c in range(num_classes)]
    gt_counts = [int(per_class_gt[c].item()) if isinstance(per_class_gt[c], torch.Tensor)
                 else int(per_class_gt[c]) for c in range(num_classes)]
    colors = [tuple(v/255 for v in TOOL_COLORS[c % len(TOOL_COLORS)]) for c in range(num_classes)]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, recalls, color=colors, edgecolor='black', linewidth=0.5)
    
    for bar, r, gt in zip(bars, recalls, gt_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{r:.2f}\n(n={gt})", ha='center', va='bottom', fontsize=9)
    
    ax.axhline(y=overall_recall, color='red', linestyle='--', linewidth=1.5, 
               label=f'Overall: {overall_recall:.3f}')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall@0.5")
    ax.set_title(f"Epoch {epoch} — Per-Class Recall@0.5 IoU")
    ax.legend(loc='upper right')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    
    arr = _fig_to_array(fig)
    plt.close(fig)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 3. Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def make_confusion_matrix(y_true, y_pred, num_classes=7, epoch=0):
    """Build and plot confusion matrix from matched predictions.
    
    Args:
        y_true: list of GT class indices (from Hungarian-matched pairs)
        y_pred: list of predicted class indices
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    names = [TOOL_NAMES[c] if c < len(TOOL_NAMES) else f"cls{c}" for c in range(num_classes)]
    names.append("No-obj")
    
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)
    for gt, pred in zip(y_true, y_pred):
        gt_idx = min(gt, num_classes)
        pred_idx = min(pred, num_classes)
        cm[gt_idx, pred_idx] += 1
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks(range(num_classes + 1))
    ax.set_yticks(range(num_classes + 1))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Epoch {epoch} — Confusion Matrix (Hungarian Matched)")
    
    # Annotate cells
    for i in range(num_classes + 1):
        for j in range(num_classes + 1):
            val = cm[i, j]
            if val > 0:
                text_color = 'white' if val > cm.max() * 0.5 else 'black'
                ax.text(j, i, str(val), ha='center', va='center', 
                        fontsize=8, color=text_color)
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    arr = _fig_to_array(fig)
    plt.close(fig)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-Attention Heatmaps
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_cross_attention(head, final_tokens, mid_tokens, grid_size=14):
    """Extract cross-attention weights from the decoder's first layer.
    
    Returns:
        attn_weights: [B, Q, H*W] attention from queries to spatial tokens
    """
    B = final_tokens.shape[0]
    
    # Feature fusion
    if mid_tokens is not None:
        fused = head.fusion(torch.cat([
            head.mid_projector(mid_tokens), final_tokens], dim=-1))
    else:
        fused = final_tokens
    fused = fused + head.spatial_pos_embed
    
    global_ctx = head.global_context_proj(fused.mean(dim=1, keepdim=True))
    prior_pos_emb = head.prior_pos_embed(head.prior_boxes)
    queries = (head.query_content.weight + prior_pos_emb).unsqueeze(0).expand(B, -1, -1)
    queries = queries + global_ctx
    
    # Self-attention in layer1
    q2 = head.layer1.norm1(queries)
    q2, _ = head.layer1.self_attn(q2, q2, q2)
    queries = queries + q2
    
    # Cross-attention — extract weights manually
    q2 = head.layer1.norm2(queries)
    Q_dim = queries.shape[1]
    N = fused.shape[1]
    nheads = head.layer1.cross_attn.nheads
    head_dim = head.layer1.cross_attn.head_dim
    
    q = head.layer1.cross_attn.q_proj(q2).view(B, Q_dim, nheads, head_dim).transpose(1, 2)
    k = head.layer1.cross_attn.k_proj(fused).view(B, N, nheads, head_dim).transpose(1, 2)
    
    attn = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    
    # Add spatial bias
    dist2 = ((head.query_positions.unsqueeze(1) - head.token_positions.unsqueeze(0)) ** 2).sum(-1)
    spatial_bias = -head.layer1.cross_attn.spatial_bias_scale * dist2
    attn = attn + spatial_bias.unsqueeze(0).unsqueeze(1)
    
    attn = attn.softmax(-1)  # [B, nheads, Q, N]
    attn_avg = attn.mean(dim=1)  # [B, Q, N] average over heads
    
    return attn_avg


def make_attention_heatmaps(frame, attn_weights, pred_boxes, pred_labels, pred_scores,
                            num_classes=7, grid_size=14, top_k=5, score_thresh=0.3):
    """Visualize where top-K queries attend in the image.
    
    Args:
        frame: [C, H, W] or [H, W, C] input frame
        attn_weights: [Q, H*W] cross-attention weights for one image
        pred_boxes: [Q, 4] predicted boxes cxcywh
        pred_labels: [Q] predicted classes
        pred_scores: [Q] confidence scores
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    if isinstance(frame, torch.Tensor):
        if frame.dim() == 3 and frame.shape[0] in (1, 3):
            frame = frame.permute(1, 2, 0).cpu().numpy()
        else:
            frame = frame.cpu().numpy()
    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
    
    attn_np = attn_weights.cpu().numpy() if isinstance(attn_weights, torch.Tensor) else attn_weights
    scores_np = pred_scores.cpu().numpy() if isinstance(pred_scores, torch.Tensor) else pred_scores
    labels_np = pred_labels.cpu().numpy() if isinstance(pred_labels, torch.Tensor) else pred_labels
    
    # Select top-K queries by score (excluding no-object)
    valid = (labels_np < num_classes) & (scores_np >= score_thresh)
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        valid_indices = np.argsort(-scores_np)[:top_k]
    else:
        order = np.argsort(-scores_np[valid_indices])
        valid_indices = valid_indices[order[:top_k]]
    
    n_show = min(len(valid_indices), top_k)
    if n_show == 0:
        return None
    
    fig, axes = plt.subplots(1, n_show + 1, figsize=(4 * (n_show + 1), 4))
    if n_show + 1 == 1:
        axes = [axes]
    
    # First: input frame
    axes[0].imshow(frame)
    axes[0].set_title("Input", fontsize=10)
    axes[0].axis("off")
    
    H, W = frame.shape[:2]
    
    for i, qi in enumerate(valid_indices[:n_show]):
        attn_map = attn_np[qi].reshape(grid_size, grid_size)
        # Upsample to frame size
        attn_up = np.array(
            __import__('PIL').Image.fromarray(
                (attn_map * 255).astype(np.uint8)
            ).resize((W, H), __import__('PIL').Image.BILINEAR)
        ).astype(np.float32) / 255.0
        
        axes[i+1].imshow(frame)
        axes[i+1].imshow(attn_up, cmap='hot', alpha=0.6, vmin=0)
        
        cls = labels_np[qi]
        score = scores_np[qi]
        name = TOOL_NAMES[cls] if cls < len(TOOL_NAMES) else f"cls{cls}"
        c = TOOL_COLORS[cls % len(TOOL_COLORS)]
        
        # Draw predicted box
        boxes_np = pred_boxes.cpu().numpy() if isinstance(pred_boxes, torch.Tensor) else pred_boxes
        box_xyxy = box_cxcywh_to_xyxy_np(boxes_np[qi:qi+1])[0] * np.array([W, H, W, H])
        import matplotlib.patches as patches
        rect = patches.Rectangle(
            (box_xyxy[0], box_xyxy[1]), box_xyxy[2]-box_xyxy[0], box_xyxy[3]-box_xyxy[1],
            linewidth=2, edgecolor=tuple(v/255 for v in c), facecolor='none')
        axes[i+1].add_patch(rect)
        
        axes[i+1].set_title(f"Q{qi}: {name} ({score:.2f})", fontsize=10)
        axes[i+1].axis("off")
    
    plt.suptitle("Cross-Attention Heatmaps (Query → Spatial Tokens)", fontsize=12)
    plt.tight_layout()
    arr = _fig_to_array(fig)
    plt.close(fig)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 5. Query Assignment Visualization
# ─────────────────────────────────────────────────────────────────────────────

def make_query_assignment_viz(pred_boxes, pred_labels, pred_scores, 
                               gt_boxes, gt_labels, match_q, match_t,
                               num_classes=7, frame_size=224):
    """Visualize Hungarian matching: which query matched which GT box.
    
    Shows query prior positions, predicted boxes, GT boxes, and match lines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Invert y for image coords
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f0f0')
    
    pred_np = pred_boxes.cpu().numpy() if isinstance(pred_boxes, torch.Tensor) else pred_boxes
    gt_np = gt_boxes.cpu().numpy() if isinstance(gt_boxes, torch.Tensor) else gt_boxes
    pred_l = pred_labels.cpu().numpy() if isinstance(pred_labels, torch.Tensor) else pred_labels
    gt_l = gt_labels.cpu().numpy() if isinstance(gt_labels, torch.Tensor) else gt_labels
    pred_s = pred_scores.cpu().numpy() if isinstance(pred_scores, torch.Tensor) else pred_scores
    
    # Draw GT boxes (solid)
    gt_xyxy = box_cxcywh_to_xyxy_np(gt_np)
    for i, (box, cls) in enumerate(zip(gt_xyxy, gt_l)):
        c = tuple(v/255 for v in TOOL_COLORS[cls % len(TOOL_COLORS)])
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                  linewidth=2.5, edgecolor=c, facecolor=c + (0.15,))
        ax.add_patch(rect)
        name = TOOL_NAMES[cls] if cls < len(TOOL_NAMES) else f"cls{cls}"
        ax.text(gt_np[i, 0], gt_np[i, 1], f"GT:{name}", ha='center', va='center',
                fontsize=7, fontweight='bold', color='black',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Draw predicted boxes (dashed)
    pred_xyxy = box_cxcywh_to_xyxy_np(pred_np)
    for i, (box, cls, score) in enumerate(zip(pred_xyxy, pred_l, pred_s)):
        if cls >= num_classes or score < 0.2:
            continue
        c = tuple(v/255 for v in TOOL_COLORS[cls % len(TOOL_COLORS)])
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                  linewidth=1.5, edgecolor=c, facecolor='none',
                                  linestyle='--')
        ax.add_patch(rect)
    
    # Draw match lines
    mq = match_q.cpu().numpy() if isinstance(match_q, torch.Tensor) else match_q
    mt = match_t.cpu().numpy() if isinstance(match_t, torch.Tensor) else match_t
    for qi, ti in zip(mq, mt):
        pred_center = pred_np[qi, :2]
        gt_center = gt_np[ti, :2]
        ax.plot([pred_center[0], gt_center[0]], [pred_center[1], gt_center[1]],
                'g-', linewidth=1.5, alpha=0.7)
        ax.plot(*pred_center, 'go', markersize=6)
        ax.plot(*gt_center, 'rs', markersize=6)
    
    ax.set_title("Hungarian Matching: Query → GT Assignment")
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (normalized)")
    plt.tight_layout()
    
    arr = _fig_to_array(fig)
    plt.close(fig)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 6. Query Box Evolution — where each query predicts across samples
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def make_query_box_evolution(head, dataset, device, num_samples=50, num_classes=7, epoch=0):
    """Show where each query's predicted box lands across many samples.
    
    Reveals query specialization: do queries learn to cover specific image regions?
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    head.eval()
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    Q = head.num_queries
    query_centers = [[] for _ in range(Q)]  # list of (cx, cy) per query
    query_classes = [[] for _ in range(Q)]  # predicted class per query
    
    for idx in indices:
        final_tokens, mid_tokens, boxes, labels = dataset[int(idx)]
        ft = final_tokens.unsqueeze(0).to(device)
        mt = mid_tokens.unsqueeze(0).to(device)
        
        with torch.amp.autocast(device_type='cuda', enabled=True):
            logits, pred_boxes, _ = head(ft.float(), mt.float())
        
        probs = logits[0].softmax(-1).cpu()
        scores, pred_cls = probs[:, :-1].max(-1)
        pb = pred_boxes[0].cpu()
        
        for qi in range(Q):
            query_centers[qi].append((pb[qi, 0].item(), pb[qi, 1].item()))
            query_classes[qi].append(pred_cls[qi].item() if scores[qi] > 0.2 else -1)
    
    # Plot: one subplot per query showing scatter of predicted centers
    cols = min(Q, 5)
    rows = (Q + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if Q == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)
    
    for qi in range(Q):
        r, c = qi // cols, qi % cols
        ax = axes[r, c]
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect('equal')
        ax.set_facecolor('#f5f5f5')
        
        centers = np.array(query_centers[qi])
        classes = np.array(query_classes[qi])
        
        for cls_id in range(-1, num_classes):
            mask = classes == cls_id
            if mask.sum() == 0:
                continue
            if cls_id == -1:
                color = (0.7, 0.7, 0.7)
                label = "no-obj"
            else:
                color = tuple(v/255 for v in TOOL_COLORS[cls_id % len(TOOL_COLORS)])
                label = TOOL_NAMES[cls_id] if cls_id < len(TOOL_NAMES) else f"cls{cls_id}"
            ax.scatter(centers[mask, 0], centers[mask, 1], c=[color], s=12, alpha=0.5, label=label)
        
        # Mark prior position
        prior = head.prior_boxes[qi].detach().cpu().numpy()
        ax.plot(prior[0], prior[1], 'k*', markersize=12, markeredgewidth=1.5)
        
        ax.set_title(f"Q{qi}", fontsize=9)
        ax.grid(True, alpha=0.2)
    
    # Hide unused axes
    for qi in range(Q, rows * cols):
        r, c = qi // cols, qi % cols
        axes[r, c].set_visible(False)
    
    # Single legend
    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center', ncol=min(8, len(handles)), fontsize=8)
    plt.suptitle(f"Epoch {epoch} — Query Box Centers (★ = prior position)", fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    arr = _fig_to_array(fig)
    plt.close(fig)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 7. Query Confidence Distribution — per-class score histograms
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def make_confidence_histogram(head, dataset, device, num_samples=200, num_classes=7, epoch=0):
    """Histogram of per-class confidence scores across queries.
    
    Shows whether the decoder is learning to be confident on the right classes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    head.eval()
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    matched_scores = defaultdict(list)   # class → list of matched query scores
    unmatched_scores = []                # scores of unmatched queries
    
    from losses import hungarian_match
    
    for idx in indices:
        final_tokens, mid_tokens, boxes, labels = dataset[int(idx)]
        ft = final_tokens.unsqueeze(0).to(device)
        mt = mid_tokens.unsqueeze(0).to(device)
        
        with torch.amp.autocast(device_type='cuda', enabled=True):
            logits, pred_boxes, _ = head(ft.float(), mt.float())
        
        probs = logits[0].softmax(-1).cpu()
        scores_all, pred_cls = probs[:, :-1].max(-1)
        no_obj_score = probs[:, -1]
        
        if labels.numel() > 0:
            mq, mt_idx = hungarian_match(logits[0], pred_boxes[0], 
                                          labels.to(device), boxes.to(device))
            matched_set = set(mq.tolist()) if len(mq) > 0 else set()
            
            for qi, ti in zip(mq.tolist(), mt_idx.tolist()):
                cls = labels[ti].item()
                matched_scores[cls].append(scores_all[qi].item())
            
            for qi in range(logits.shape[1]):
                if qi not in matched_set:
                    unmatched_scores.append(scores_all[qi].item())
        else:
            unmatched_scores.extend(scores_all.tolist())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: matched query scores per class
    ax = axes[0]
    for cls in range(num_classes):
        if cls not in matched_scores or len(matched_scores[cls]) == 0:
            continue
        color = tuple(v/255 for v in TOOL_COLORS[cls % len(TOOL_COLORS)])
        name = TOOL_NAMES[cls] if cls < len(TOOL_NAMES) else f"cls{cls}"
        ax.hist(matched_scores[cls], bins=20, range=(0, 1), alpha=0.6, 
                color=color, label=f"{name} (n={len(matched_scores[cls])})")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title("Matched Query Scores (per class)")
    ax.legend(fontsize=8)
    ax.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='thresh=0.3')
    
    # Right: unmatched (should be low confidence)
    ax = axes[1]
    ax.hist(unmatched_scores, bins=30, range=(0, 1), alpha=0.7, color='gray')
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Unmatched Query Scores (n={len(unmatched_scores)})")
    ax.axvline(x=0.3, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle(f"Epoch {epoch} — Query Confidence Distribution", fontsize=12)
    plt.tight_layout()
    
    arr = _fig_to_array(fig)
    plt.close(fig)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 8. Decoder Cross-Attention Heatmaps (works with pre-computed features)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def make_decoder_attention_grid(head, dataset, device, num_samples=4, 
                                 grid_size=14, num_classes=7, epoch=0):
    """Visualize decoder cross-attention: where each query attends on 14×14 grid.
    
    Works with pre-computed features (no encoder needed).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    head.eval()
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    n_show = min(num_samples, 4)
    top_k = 4  # top queries per sample
    
    fig, axes = plt.subplots(n_show, top_k + 1, figsize=(4 * (top_k + 1), 4 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    
    for si in range(n_show):
        final_tokens, mid_tokens, boxes, labels = dataset[int(indices[si])]
        ft = final_tokens.unsqueeze(0).to(device)
        mt = mid_tokens.unsqueeze(0).to(device)
        
        with torch.amp.autocast(device_type='cuda', enabled=True):
            logits, pred_boxes, _ = head(ft.float(), mt.float())
        
        # Extract attention weights
        try:
            attn_weights = extract_cross_attention(head, ft.float(), mt.float())
            attn = attn_weights[0].cpu().numpy()  # [Q, 196]
        except Exception:
            continue
        
        probs = logits[0].softmax(-1).cpu()
        scores, pred_cls = probs[:, :-1].max(-1)
        
        # First column: GT info
        tool_names = [TOOL_NAMES[int(l)] if int(l) < len(TOOL_NAMES) else f"cls{int(l)}" 
                      for l in labels]
        gt_str = "\n".join(tool_names) if tool_names else "none"
        axes[si, 0].text(0.5, 0.5, f"GT:\n{gt_str}", ha='center', va='center', fontsize=10,
                         transform=axes[si, 0].transAxes)
        axes[si, 0].set_facecolor('#f0f0f0')
        axes[si, 0].axis("off")
        
        # Select top-K queries by score
        valid = (pred_cls < num_classes) & (scores >= 0.15)
        valid_idx = torch.where(valid)[0]
        if len(valid_idx) == 0:
            valid_idx = scores.argsort(descending=True)[:top_k]
        else:
            order = scores[valid_idx].argsort(descending=True)
            valid_idx = valid_idx[order[:top_k]]
        
        for ki, qi in enumerate(valid_idx[:top_k]):
            qi = qi.item()
            attn_map = attn[qi].reshape(grid_size, grid_size)
            
            axes[si, ki + 1].imshow(attn_map, cmap='hot', vmin=0)
            
            cls = pred_cls[qi].item()
            score = scores[qi].item()
            name = TOOL_NAMES[cls] if cls < len(TOOL_NAMES) else f"cls{cls}"
            color = tuple(v/255 for v in TOOL_COLORS[cls % len(TOOL_COLORS)])
            
            # Draw predicted box on attention map
            pb = pred_boxes[0, qi].cpu().numpy()
            box_xyxy = box_cxcywh_to_xyxy_np(pb.reshape(1, 4))[0] * grid_size
            import matplotlib.patches as mpatches
            rect = mpatches.Rectangle(
                (box_xyxy[0], box_xyxy[1]), box_xyxy[2]-box_xyxy[0], box_xyxy[3]-box_xyxy[1],
                linewidth=2, edgecolor=color, facecolor='none')
            axes[si, ki + 1].add_patch(rect)
            axes[si, ki + 1].set_title(f"Q{qi}: {name} ({score:.2f})", fontsize=9)
            axes[si, ki + 1].axis("off")
        
        # Hide unused
        for ki in range(len(valid_idx), top_k):
            axes[si, ki + 1].axis("off")
    
    plt.suptitle(f"Epoch {epoch} — Decoder Cross-Attention (14×14 grid)", fontsize=12)
    plt.tight_layout()
    
    arr = _fig_to_array(fig)
    plt.close(fig)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 9. Loss Component Breakdown (stacked area over training)
# ─────────────────────────────────────────────────────────────────────────────

def make_loss_breakdown(loss_history, epoch=0):
    """Stacked area chart of loss components over epochs.
    
    Args:
        loss_history: dict of lists, keys = 'ce', 'l1', 'giou', 'sc'
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    epochs = list(range(len(loss_history.get('ce', []))))
    if len(epochs) < 2:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: stacked area
    ax = axes[0]
    components = ['ce', 'l1', 'giou', 'sc']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    labels = ['CE (class)', 'L1 (bbox)', 'GIoU', 'SupCon']
    
    bottom = np.zeros(len(epochs))
    for comp, color, label in zip(components, colors, labels):
        vals = np.array(loss_history.get(comp, [0]*len(epochs)))
        ax.fill_between(epochs, bottom, bottom + vals, alpha=0.7, color=color, label=label)
        bottom += vals
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Component Breakdown (stacked)")
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, len(epochs) - 1)
    
    # Right: individual lines (log scale)
    ax = axes[1]
    for comp, color, label in zip(components, colors, labels):
        vals = loss_history.get(comp, [])
        if vals:
            ax.plot(epochs, vals, color=color, label=label, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log)")
    ax.set_yscale('log')
    ax.set_title("Loss Components (log scale)")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Epoch {epoch} — Training Loss Breakdown", fontsize=12)
    plt.tight_layout()
    
    arr = _fig_to_array(fig)
    plt.close(fig)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Main visualization runner
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_detection_visualization(
    encoder, head, mid_hook, dataset, device, epoch,
    wandb_run=None, tb_writer=None, num_classes=7,
    num_samples=8, score_thresh=0.3,
):
    """Run all detection visualizations and log to wandb/tensorboard.
    
    Args:
        encoder: V-JEPA2 encoder (eval mode)
        head: LightweightQueryDecoder
        mid_hook: MidBackboneHook
        dataset: CholecDetectDataset
        device: torch device
        epoch: current epoch
        wandb_run: wandb module or None
        tb_writer: tensorboard SummaryWriter or None
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    head.eval()
    
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    all_y_true, all_y_pred = [], []
    bbox_images = []
    attn_images = []
    assign_images = []
    
    from losses import hungarian_match
    
    for idx in indices:
        try:
            out = dataset[int(idx)]
            clip = out[0]
            gt_boxes = out[1]
            gt_labels = out[2]
            # out[3] is track_ids when load_track_ids=True; ignore for viz
            clip = clip.unsqueeze(0).to(device)  # [1, C, T, H, W]
            
            if clip.size(2) < 2:
                clip = torch.cat([clip, clip], dim=2)
            
            with torch.amp.autocast(device_type='cuda', enabled=True):
                full_tokens = encoder([clip])[0]
                mid_features = mid_hook.get_features()
                
                B, N, D = full_tokens.shape
                num_spatial = 196
                num_temporal = N // num_spatial
                
                if num_temporal > 1:
                    final_tokens = full_tokens.view(B, num_temporal, num_spatial, D)[:, -1]
                    mid_tokens = mid_features.view(B, num_temporal, num_spatial, D)[:, -1]
                else:
                    final_tokens = full_tokens
                    mid_tokens = mid_features
                
                logits, pred_boxes, query_features = head(
                    final_tokens.float(), mid_tokens.float())
            
            probs = logits[0].softmax(-1)
            scores, pred_cls = probs[:, :-1].max(-1)
            
            # Get last frame for visualization
            frame = clip[0, :, -1]  # [C, H, W]
            
            # 1. Bbox overlay
            bbox_img = make_bbox_overlay(
                frame, gt_boxes, gt_labels,
                pred_boxes[0], pred_cls, scores,
                num_classes=num_classes, score_thresh=score_thresh)
            bbox_images.append(bbox_img)
            
            # 2. Cross-attention heatmaps
            try:
                attn_weights = extract_cross_attention(
                    head, final_tokens.float(), mid_tokens.float())
                attn_img = make_attention_heatmaps(
                    frame, attn_weights[0], pred_boxes[0], pred_cls, scores,
                    num_classes=num_classes, top_k=4, score_thresh=score_thresh)
                if attn_img is not None:
                    attn_images.append(attn_img)
            except Exception as e:
                logger.warning(f"Attention viz failed: {e}")
            
            # 3. Hungarian matching for confusion matrix
            gt_boxes_dev = gt_boxes.to(device)
            gt_labels_dev = gt_labels.to(device)
            mq, mt = hungarian_match(
                logits[0], pred_boxes[0], gt_labels_dev, gt_boxes_dev)
            
            # Collect matched pairs for confusion matrix
            target_cls = torch.full((logits.shape[1],), num_classes, dtype=torch.long)
            if len(mq) > 0:
                target_cls[mq] = gt_labels_dev[mt].cpu()
            for qi in range(logits.shape[1]):
                all_y_true.append(target_cls[qi].item())
                all_y_pred.append(pred_cls[qi].item() if scores[qi] >= score_thresh else num_classes)
            
            # 4. Query assignment (first sample only)
            if len(assign_images) == 0 and len(mq) > 0:
                assign_img = make_query_assignment_viz(
                    pred_boxes[0], pred_cls, scores,
                    gt_boxes, gt_labels, mq, mt,
                    num_classes=num_classes)
                assign_images.append(assign_img)
                
        except Exception as e:
            logger.warning(f"Detection viz failed for idx {idx}: {e}")
            continue
    
    head.train()
    
    # Log to wandb/tensorboard
    try:
        # Bbox overlays
        if bbox_images:
            for i, img in enumerate(bbox_images[:4]):
                if wandb_run is not None:
                    wandb_run.log({f"viz/bbox_overlay_{i}": wandb_run.Image(img)}, step=epoch)
                if tb_writer is not None:
                    tb_writer.add_image(f"viz/bbox_overlay_{i}", img, global_step=epoch, 
                                       dataformats='HWC')
        
        # Attention heatmaps
        if attn_images:
            for i, img in enumerate(attn_images[:2]):
                if wandb_run is not None:
                    wandb_run.log({f"viz/attention_{i}": wandb_run.Image(img)}, step=epoch)
                if tb_writer is not None:
                    tb_writer.add_image(f"viz/attention_{i}", img, global_step=epoch,
                                       dataformats='HWC')
        
        # Confusion matrix
        if all_y_true:
            cm_img = make_confusion_matrix(all_y_true, all_y_pred, num_classes, epoch)
            if wandb_run is not None:
                wandb_run.log({"viz/confusion_matrix": wandb_run.Image(cm_img)}, step=epoch)
            if tb_writer is not None:
                tb_writer.add_image("viz/confusion_matrix", cm_img, global_step=epoch,
                                   dataformats='HWC')
        
        # Query assignment
        if assign_images:
            if wandb_run is not None:
                wandb_run.log({"viz/query_assignment": wandb_run.Image(assign_images[0])}, step=epoch)
            if tb_writer is not None:
                tb_writer.add_image("viz/query_assignment", assign_images[0], global_step=epoch,
                                   dataformats='HWC')
        
        logger.info(f"[Epoch {epoch}] Detection visualizations logged "
                    f"({len(bbox_images)} bbox, {len(attn_images)} attn, "
                    f"{'1' if all_y_true else '0'} cm, {len(assign_images)} assign)")
    
    except Exception as e:
        logger.warning(f"Failed to log visualizations: {e}")
