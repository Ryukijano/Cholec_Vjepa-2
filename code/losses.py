"""
Loss functions for detection and supervised contrastive learning.
"""

import torch
import torch.nn as nn
from typing import List
from scipy.optimize import linear_sum_assignment

try:
    from torchvision.ops import generalized_box_iou, box_iou
except ImportError:
    generalized_box_iou = None
    box_iou = None


def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [..., 4] cxcywh to xyxy."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


class SupConProjectionHead(nn.Module):
    """2-layer MLP projection head for SupCon loss."""
    
    def __init__(self, embed_dim: int = 1024, hidden_dim: int = 512, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
    
    def forward(self, x):
        """x: [N, D] -> [N, proj_dim] L2-normalized."""
        z = self.proj(x)
        return nn.functional.normalize(z, dim=-1)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            features: [N, D] L2-normalized embeddings
            labels: [N] integer class labels
        Returns:
            scalar loss
        """
        device = features.device
        N = features.shape[0]
        
        if N <= 1:
            return torch.tensor(0., device=device, requires_grad=True)
        
        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature
        
        # Mask self-similarity
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(self_mask, -1e4)
        
        # Positive mask
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = labels_eq & ~self_mask
        
        num_positives = pos_mask.sum(dim=1)
        has_positives = num_positives > 0
        
        if not has_positives.any():
            return torch.tensor(0., device=device, requires_grad=True)
        
        # Log-sum-exp for stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        sim_stable = sim - logits_max.detach()
        
        exp_sim = torch.exp(sim_stable)
        exp_sim = exp_sim.masked_fill(self_mask, 0.0)
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        log_prob = sim_stable - log_denom
        pos_log_prob = log_prob * pos_mask.float()
        loss_per_anchor = -pos_log_prob.sum(dim=1) / (num_positives.float() + 1e-8)
        
        return loss_per_anchor[has_positives].mean()


def focal_loss(pred_logits, target, alpha=0.25, gamma=2.0, weight=None):
    """Focal loss for classification - down-weights easy examples."""
    ce = nn.functional.cross_entropy(pred_logits, target, weight=weight, reduction='none')
    pt = torch.exp(-ce)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * ce).mean()


# Inverse-frequency class weights from CholecTrack20 training distribution
# grasper:46%, bipolar:3.8%, hook:35.7%, scissors:1.4%, clipper:2.8%, irrigator:2.8%, specimen_bag:7.5%
# Aggressive weighting for rare classes to break the plateau
DEFAULT_CLASS_WEIGHTS = torch.tensor([1.0, 15.0, 1.5, 40.0, 20.0, 20.0, 8.0])


def hungarian_match(pred_logits, pred_boxes, tgt_labels, tgt_boxes,
                   class_cost=1.0, bbox_cost=5.0, giou_cost=2.0,
                   class_weights=None):
    """Perform cost-sensitive Hungarian matching between predictions and targets.
    
    Args:
        pred_logits: [Q, num_classes+1]
        pred_boxes: [Q, 4] cxcywh
        tgt_labels: [M]
        tgt_boxes: [M, 4] cxcywh
        class_weights: [num_classes] per-class cost multiplier (higher = prioritize matching)
    
    Returns:
        row_ind, col_ind: matched query and target indices
    """
    Q = pred_logits.shape[0]
    M = tgt_labels.shape[0]
    
    if M == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
    
    # Class cost with per-class weighting
    prob = pred_logits.softmax(-1)[:, tgt_labels]  # [Q, M]
    class_c = -prob
    
    # Apply per-class cost multiplier so rare classes get priority in matching
    if class_weights is not None:
        cw = class_weights.to(pred_logits.device)
        per_target_weight = cw[tgt_labels]  # [M]
        class_c = class_c * per_target_weight.unsqueeze(0)  # [Q, M]
    
    # Bbox L1 cost
    bbox_c = torch.cdist(pred_boxes, tgt_boxes, p=1)
    
    # GIoU cost
    giou_c = 0
    if generalized_box_iou is not None:
        p_xyxy = _box_cxcywh_to_xyxy(pred_boxes)
        t_xyxy = _box_cxcywh_to_xyxy(tgt_boxes)
        giou_c = -generalized_box_iou(p_xyxy, t_xyxy)
    
    C = class_cost * class_c + bbox_cost * bbox_c + giou_cost * giou_c
    C = C.detach().cpu().numpy()
    # Sanitize: replace NaN/Inf with large finite value to avoid linear_sum_assignment crash
    import numpy as np
    C = np.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)
    row_ind, col_ind = linear_sum_assignment(C)
    
    return torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)


def detection_loss(pred_logits, pred_boxes, targets_labels, targets_boxes,
                    num_classes=7, giou_weight=5.0, class_weights=None):
    """Compute detection loss with cost-sensitive Hungarian matching.
    
    Args:
        class_weights: [num_classes] inverse-frequency weights for rare classes
    
    Returns:
        total_ce, total_bbox, total_giou (scaled losses)
    """
    device = pred_logits.device
    num_queries = pred_logits.shape[1]
    
    total_ce = torch.tensor(0., device=device)
    total_bbox = torch.tensor(0., device=device)
    total_giou = torch.tensor(0., device=device)
    total = 0
    
    if class_weights is None:
        class_weights = DEFAULT_CLASS_WEIGHTS
    
    # CE weight: class_weights for tool classes, low weight for no-object
    ce_weight = torch.ones(num_classes + 1, device=device)
    ce_weight[:num_classes] = class_weights[:num_classes].to(device)
    ce_weight[-1] = 0.1  # no-object class (0.1 is standard DETR practice; 0.001 was too low causing FP collapse)
    
    for b in range(pred_logits.shape[0]):
        logits = pred_logits[b]
        boxes = pred_boxes[b]
        tlabels = targets_labels[b].to(device)
        tboxes = targets_boxes[b].to(device)
        
        if tlabels.numel() == 0:
            ce = nn.functional.cross_entropy(
                logits,
                torch.full((num_queries,), num_classes, device=device, dtype=torch.long)
            )
            total_ce += ce
            total += 1
            continue
        
        # Cost-sensitive Hungarian matching
        match_q, match_t = hungarian_match(
            logits, boxes, tlabels, tboxes,
            class_weights=class_weights
        )
        
        # Target classes
        target_classes = torch.full((num_queries,), num_classes, device=device, dtype=torch.long)
        target_classes[match_q] = tlabels[match_t]
        
        # Class-weighted CE loss
        ce = nn.functional.cross_entropy(logits, target_classes, weight=ce_weight)
        total_ce += ce
        
        if len(match_q) > 0:
            # L1 loss
            l1 = nn.functional.l1_loss(boxes[match_q], tboxes[match_t])
            total_bbox += l1
            
            # GIoU loss
            if generalized_box_iou is not None:
                p_xyxy = _box_cxcywh_to_xyxy(boxes[match_q])
                t_xyxy = _box_cxcywh_to_xyxy(tboxes[match_t])
                giou = generalized_box_iou(p_xyxy, t_xyxy)
                total_giou += (1 - giou.diag()).mean()
        
        total += 1
    
    return total_ce / total, total_bbox / total, total_giou / total


def detection_loss_focal(pred_logits, pred_boxes, targets_labels, targets_boxes,
                          num_classes=7, giou_weight=5.0, class_weights=None,
                          focal_alpha=0.25, focal_gamma=2.0):
    """Compute detection loss with focal loss for classification.
    
    Focal loss down-weights easy examples and focuses on hard negatives.
    Better for imbalanced datasets with rare classes.
    
    Args:
        class_weights: [num_classes] inverse-frequency weights for rare classes
        focal_alpha: focal loss alpha (balance factor)
        focal_gamma: focal loss gamma (focusing parameter, higher = more focus on hard examples)
    
    Returns:
        total_ce, total_bbox, total_giou (scaled losses)
    """
    device = pred_logits.device
    num_queries = pred_logits.shape[1]
    
    total_ce = torch.tensor(0., device=device)
    total_bbox = torch.tensor(0., device=device)
    total_giou = torch.tensor(0., device=device)
    total = 0
    
    if class_weights is None:
        class_weights = DEFAULT_CLASS_WEIGHTS
    
    # CE weight: class_weights for tool classes, low weight for no-object
    ce_weight = torch.ones(num_classes + 1, device=device)
    ce_weight[:num_classes] = class_weights[:num_classes].to(device)
    ce_weight[-1] = 0.1  # no-object class (0.1 is standard DETR practice; 0.001 was too low causing FP collapse)
    
    for b in range(pred_logits.shape[0]):
        logits = pred_logits[b]
        boxes = pred_boxes[b]
        tlabels = targets_labels[b].to(device)
        tboxes = targets_boxes[b].to(device)
        
        if tlabels.numel() == 0:
            # All queries should predict no-object
            target_classes = torch.full((num_queries,), num_classes, device=device, dtype=torch.long)
            ce = focal_loss(logits, target_classes, alpha=focal_alpha, gamma=focal_gamma, weight=ce_weight)
            total_ce += ce
            total += 1
            continue
        
        # Cost-sensitive Hungarian matching
        match_q, match_t = hungarian_match(
            logits, boxes, tlabels, tboxes,
            class_weights=class_weights
        )
        
        # Target classes
        target_classes = torch.full((num_queries,), num_classes, device=device, dtype=torch.long)
        target_classes[match_q] = tlabels[match_t]
        
        # Focal loss for classification
        ce = focal_loss(logits, target_classes, alpha=focal_alpha, gamma=focal_gamma, weight=ce_weight)
        total_ce += ce
        
        if len(match_q) > 0:
            # L1 loss
            l1 = nn.functional.l1_loss(boxes[match_q], tboxes[match_t])
            total_bbox += l1
            
            # GIoU loss
            if generalized_box_iou is not None:
                p_xyxy = _box_cxcywh_to_xyxy(boxes[match_q])
                t_xyxy = _box_cxcywh_to_xyxy(tboxes[match_t])
                giou = generalized_box_iou(p_xyxy, t_xyxy)
                total_giou += (1 - giou.diag()).mean()
        
        total += 1
    
    return total_ce / total, total_bbox / total, total_giou / total
