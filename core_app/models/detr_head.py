"""
DETR (DEtection TRansformer) Head for Surgical Tool Detection
Based on "End-to-End Object Detection with Transformers" and Deformable DETR.
Loss functions: Focal Loss + L1 + GIoU with improved Hungarian matching cost.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


def _get_activation_fn(activation: str):
    """Return activation function by name."""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "sigmoid":
        return torch.sigmoid
    else:
        raise ValueError(f"Unknown activation: {activation}")


class TriAttention(nn.Module):
    """
    Tri-Attention module for multi-modal / multi-source fusion.

    In the context of DINO-WM, it explicitly fuses three streams:
      1. Query (e.g., Object Queries in DETR)
      2. Source A (e.g., Real Neck Features)
      3. Source B (e.g., Phantom/Predicted Features)

    Instead of standard cross-attention (Query attending to one Source),
    Tri-Attention allows the Query to attend to *both* sources dynamically,
    learning which source (real vs. phantom) is more reliable for a given
    spatial location or tool.

    Formula:
        Attn(Q, K_A, V_A, K_B, V_B) = softmax( (Q·K_A^T + Q·K_B^T) / sqrt(d) ) · (V_A + V_B)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.ka_proj = nn.Linear(embed_dim, embed_dim)
        self.va_proj = nn.Linear(embed_dim, embed_dim)
        self.kb_proj = nn.Linear(embed_dim, embed_dim)
        self.vb_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.ka_proj.weight)
        nn.init.xavier_uniform_(self.va_proj.weight)
        nn.init.xavier_uniform_(self.kb_proj.weight)
        nn.init.xavier_uniform_(self.vb_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        for proj in [self.q_proj, self.ka_proj, self.va_proj, self.kb_proj, self.vb_proj, self.out_proj]:
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0.)

    def forward(
        self,
        query: torch.Tensor,
        source_a: torch.Tensor,
        source_b: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, T_q, C)
            source_a: (B, T_kv, C) — e.g., Real features
            source_b: (B, T_kv, C) — e.g., Phantom features. If None, falls back to standard cross-attention.
            key_padding_mask: (B, T_kv)

        Returns:
            (B, T_q, C) fused features
        """
        B, T_q, _ = query.shape
        B_k, T_kv, _ = source_a.shape

        q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T_q, d)
        k_a = self.ka_proj(source_a).view(B_k, T_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T_kv, d)
        v_a = self.va_proj(source_a).view(B_k, T_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Q · K_A^T
        scores = torch.matmul(q, k_a.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T_q, T_kv)

        if source_b is not None:
            # Tri-Attention fusion
            k_b = self.kb_proj(source_b).view(B_k, T_kv, self.num_heads, self.head_dim).transpose(1, 2)
            v_b = self.vb_proj(source_b).view(B_k, T_kv, self.num_heads, self.head_dim).transpose(1, 2)

            # Add Q · K_B^T to the scores
            scores = scores + (torch.matmul(q, k_b.transpose(-2, -1)) / math.sqrt(self.head_dim))
            # Values are summed
            v = v_a + v_b
        else:
            # Standard cross-attention fallback
            v = v_a

        if key_padding_mask is not None:
            # key_padding_mask: (B, T_kv). True means ignore.
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, T_q, d)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        return self.out_proj(out)


class DetrTransformerDecoderLayer(nn.Module):
    """DETR-style transformer decoder with learned object queries."""
    
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.tri_attn = TriAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        phantom_memory: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: Object queries (B, Q, C)
            memory: Real encoder/neck features (B, N, C)
            phantom_memory: Phantom/predicted features (B, N, C) for Tri-Attention
            pos: Positional encoding for memory
            query_pos: Positional encoding for queries
        """
        # Self attention (Queries attend to queries)
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Tri-Attention (Queries attend to Real Memory + Phantom Memory)
        q = self.with_pos_embed(tgt, query_pos)
        k_a = self.with_pos_embed(memory, pos)
        k_b = self.with_pos_embed(phantom_memory, pos) if phantom_memory is not None else None

        tgt2 = self.tri_attn(
            query=q,
            source_a=k_a,
            source_b=k_b,
            key_padding_mask=memory_key_padding_mask
        )

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        phantom_memory: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: Object queries (B, Q, C)
            memory: Real encoder/neck features (B, N, C)
            phantom_memory: Phantom/predicted features (B, N, C) for Tri-Attention
            pos: Positional encoding for memory
            query_pos: Positional encoding for queries
        """
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        q = self.with_pos_embed(tgt2, query_pos)
        k_a = self.with_pos_embed(memory, pos)
        k_b = self.with_pos_embed(phantom_memory, pos) if phantom_memory is not None else None

        tgt2 = self.tri_attn(
            query=q,
            source_a=k_a,
            source_b=k_b,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        phantom_memory: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.normalize_before:
            return self.forward_pre(tgt, memory, phantom_memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, phantom_memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class DetrTransformerDecoder(nn.Module):
    """DETR-style transformer decoder with learned object queries."""
    
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        num_layers: int = 5,
        num_queries: int = 16,
        dropout: float = 0.15,
        dim_feedforward: int = 2048
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        
        # Learnable object queries - each query will detect one potential tool
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Track query embeddings (matches checkpoint shape [16, 768] and bias [16])
        self.track_embed = nn.Linear(d_model, num_queries)
        
        # Transformer decoder layers - manually create list for custom phantom handling
        self.decoder_layers = nn.ModuleList([
            DetrTransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.query_embed.weight, std=0.01)
    
    def forward(self, features: torch.Tensor, phantom_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: (B, N, C) encoder or neck features
            phantom_features: (B, N, C) predicted features for Tri-Attention
        """
        bs, seq_len, _ = features.shape

        # Prepare queries (B, num_queries, C)
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        # --- Query diversity regularisation (train only) ---
        # Dropout + Gaussian noise prevent all queries from collapsing to the
        # same mode during training, a well-known failure mode in DETR decoders.
        if self.training:
            query_embed = F.dropout(query_embed, p=0.1, training=True)
            query_embed = query_embed + torch.randn_like(query_embed) * 0.02

        tgt = torch.zeros_like(query_embed)

        # Prepare positional encoding for memory
        pos = torch.zeros_like(features)
        query_pos = query_embed

        # Manual decoder loop to pass phantom_memory through each layer
        output = tgt
        for layer in self.decoder_layers:
            output = layer(
                output,
                features,
                phantom_memory=phantom_features,
                pos=pos,
                query_pos=query_pos
            )

        return output


class DetrDetectionHead(nn.Module):
    """
    DETR-style detection head for surgical tools.
    Predicts class logits and bounding boxes for each object query.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        num_tools: int = 7,
        num_queries: int = 16,
        num_decoder_layers: int = 5,
        nheads: int = 8,
        dropout: float = 0.15,
        dim_feedforward: int = 2048,
        num_pos_tokens: int = 4608,
        class_weight: float = 1.0,
        bbox_weight: float = 0.0,
        aux_loss: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.num_tools = num_tools
        self.num_queries = num_queries
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.aux_loss = aux_loss
        
        # DETR decoder
        self.decoder = DetrTransformerDecoder(
            d_model=d_model,
            num_heads=nheads,
            num_layers=num_decoder_layers,
            num_queries=num_queries,
            dropout=dropout,
            dim_feedforward=dim_feedforward
        )
        
        # Spatial positional encoding for encoder features
        # V-JEPA 2.1 (16 frames, 384px): 8 temporal steps * (24*24) spatial patches = 4608 tokens
        self.pos_embed = nn.Embedding(num_pos_tokens, d_model) # Match checkpoint 'pos_embed.weight'
        
        # Classification head (checkpoint expects exactly num_tools, no +1 for empty class)
        self.class_embed = nn.Linear(d_model, num_tools)
        
        # Bounding box head (2-layer MLP as per checkpoint)
        if bbox_weight > 0:
            self.bbox_embed = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, 4)
            )
        else:
            self.bbox_embed = None
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize classification head
        nn.init.xavier_uniform_(self.class_embed.weight)
        nn.init.constant_(self.class_embed.bias, 0)
        
        # Initialize bbox head if present
        if self.bbox_embed is not None:
            for m in self.bbox_embed:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
            # Initialize to predict center of image with small size
            with torch.no_grad():
                self.bbox_embed[-1].bias.data[:] = torch.tensor([0.5, 0.5, 0.1, 0.1])
    
    def forward(self, encoder_features: torch.Tensor, phantom_features: Optional[torch.Tensor] = None) -> dict:
        """
        Args:
            encoder_features: (B, N, C) from V-JEPA encoder or fused features
            phantom_features: (B, N, C) predicted features for Tri-Attention
        """
        B, N, C = encoder_features.shape
        
        # Add spatial positional encoding
        if self.pos_embed.num_embeddings >= N:
            pos_indices = torch.arange(N, device=encoder_features.device)
            encoder_features = encoder_features + self.pos_embed(pos_indices).unsqueeze(0)
            
        # Decode
        query_features = self.decoder(encoder_features, phantom_features)  # (B, num_queries, C)
        
        # Class predictions
        class_logits = self.class_embed(query_features)  # (B, num_queries, num_tools)
        
        outputs = {
            'class_logits': class_logits,
            'features': query_features
        }
        
        # Bounding box predictions if enabled
        if self.bbox_embed is not None:
            pred_boxes = self.bbox_embed(query_features).sigmoid()  # (B, num_queries, 4)
            outputs['pred_boxes'] = pred_boxes
        
        return outputs


def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2) format."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h,
                        cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def _generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU between two sets of boxes.
    Both inputs in (x1, y1, x2, y2) format, values in [0, 1].

    Args:
        boxes1: (N, 4)
        boxes2: (M, 4)

    Returns:
        (N, M) GIoU matrix
    """
    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)

    # Intersection
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    union_area = area1[:, None] + area2[None, :] - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)

    # Enclosing box
    enc_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    enc_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    enc_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    enc_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
    enc_area = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0)

    giou = iou - (enc_area - union_area) / enc_area.clamp(min=1e-6)
    return giou


def _sanitize_cost_matrix(cost_matrix: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf so scipy linear_sum_assignment does not crash."""
    cost = cost_matrix.float()
    finite = torch.isfinite(cost)
    if finite.all():
        return cost
    fill = (cost[finite].max() + 1e4) if finite.any() else cost.new_tensor(1e4)
    return torch.where(finite, cost, fill)


def _normalize_cxcywh_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """Clamp cxcywh to [0,1] with positive w/h (avoids degenerate GIoU)."""
    boxes = boxes.float()
    cx, cy, w, h = boxes.unbind(-1)
    w = w.clamp(min=1e-4, max=1.0)
    h = h.clamp(min=1e-4, max=1.0)
    cx = cx.clamp(0.0, 1.0)
    cy = cy.clamp(0.0, 1.0)
    return torch.stack([cx, cy, w, h], dim=-1)


def _focal_loss_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Focal Loss for dense binary predictions.

    Args:
        logits: (...) raw logits
        targets: (...) same shape, values in {0, 1}
        alpha: foreground weight
        gamma: focusing parameter
        reduction: 'mean' | 'sum' | 'none'

    Returns:
        scalar or per-element loss
    """
    # Clamp logits to prevent BCE(+inf, 1) -> NaN on randomly-init heads.
    logits = logits.clamp(min=-10.0, max=10.0)
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


class SetCriterion(nn.Module):
    """
    Hungarian matching + loss computation for DETR.

    Loss functions:
      - Classification: Focal Loss (α=0.25, γ=2.0) — replaces BCE/CE
      - Bounding boxes: L1 + GIoU (Deformable DETR defaults)

    Hungarian matching cost:
      C = λ_cls * C_focal + λ_L1 * C_L1 + λ_giou * C_giou
    """

    def __init__(
        self,
        num_classes: int,
        class_weight: float = 1.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def hungarian_matching(
        self,
        class_logits: torch.Tensor,
        pred_boxes: Optional[torch.Tensor],
        targets: List[dict],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Hungarian matching using focal cost + L1 cost + GIoU cost.

        Args:
            class_logits: (B, num_queries, num_classes)
            pred_boxes: (B, num_queries, 4) in cxcywh [0,1], or None
            targets: list of dicts with 'labels' (LongTensor) and 'boxes' (cxcywh)

        Returns:
            List of (pred_idx, tgt_idx) tuples per batch element
        """
        from scipy.optimize import linear_sum_assignment

        amp_device = 'cuda' if class_logits.is_cuda else 'cpu'
        with torch.autocast(device_type=amp_device, enabled=False):
            return self._hungarian_matching_impl(
                class_logits, pred_boxes, targets, linear_sum_assignment
            )

    def _hungarian_matching_impl(
        self,
        class_logits: torch.Tensor,
        pred_boxes: Optional[torch.Tensor],
        targets: List[dict],
        linear_sum_assignment,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        B, Q = class_logits.shape[:2]
        indices = []
        # fp32 matching: bf16 autocast can make focal/GIoU costs NaN (scipy then fails).
        class_logits = class_logits.float().clamp(min=-10.0, max=10.0)
        if pred_boxes is not None:
            pred_boxes = _normalize_cxcywh_boxes(pred_boxes)

        for i in range(B):
            tgt_ids = targets[i]['labels']  # (T,)
            tgt_boxes = targets[i].get('boxes')  # (T, 4) or None
            num_targets = len(tgt_ids)

            if num_targets == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long)
                ))
                continue

            T = len(tgt_ids)
            logits_i = class_logits[i]  # (Q, num_classes)
            prob = torch.sigmoid(logits_i)  # (Q, num_classes)

            # --- Classification cost (Focal) ---
            # For each query q and target t: focal cost = -α(1-p)^γ log(p)
            # Index prob by target class labels to get per-target-class probabilities
            # tgt_ids are class label values (0-6), use them to index prob columns
            p_t = prob[:, tgt_ids].clamp(min=1e-6, max=1.0 - 1e-6)  # (Q, T)
            alpha_t = self.focal_alpha
            cost_cls = -(alpha_t * (1 - p_t) ** self.focal_gamma * torch.log(p_t) +
                        (1 - alpha_t) * p_t ** self.focal_gamma * torch.log(1 - p_t))
            # cost_cls: (Q, T)

            cost_matrix = self.cost_class * cost_cls

            # --- Bounding box costs ---
            if pred_boxes is not None and tgt_boxes is not None and len(tgt_boxes) > 0:
                src_boxes = pred_boxes[i]  # (Q, 4) cxcywh
                tgt_boxes_n = _normalize_cxcywh_boxes(tgt_boxes.to(src_boxes.device))
                if self.cost_bbox > 0:
                    # L1 cost: (Q, T)
                    cost_l1 = torch.cdist(src_boxes, tgt_boxes_n, p=1)
                    cost_matrix = cost_matrix + self.cost_bbox * cost_l1

                if self.cost_giou > 0:
                    # GIoU cost: (Q, T) — negative GIoU as cost
                    src_xyxy = _box_cxcywh_to_xyxy(src_boxes)
                    tgt_xyxy = _box_cxcywh_to_xyxy(tgt_boxes_n)
                    giou = _generalized_box_iou(src_xyxy, tgt_xyxy)  # (Q, T)
                    cost_matrix = cost_matrix - self.cost_giou * giou

            cost_matrix = _sanitize_cost_matrix(cost_matrix)
            pred_idx, tgt_idx = linear_sum_assignment(cost_matrix.cpu().numpy())
            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(tgt_idx, dtype=torch.long)
            ))

        return indices

    def forward(
        self,
        outputs: dict,
        targets: List[dict],
    ) -> Tuple[torch.Tensor, dict]:
        class_logits = outputs['class_logits']
        pred_boxes = outputs.get('pred_boxes')
        indices = self.hungarian_matching(class_logits, pred_boxes, targets)
        return self.compute_losses(outputs, targets, indices)

    def compute_losses(
        self,
        outputs: dict,
        targets: List[dict],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute detection losses given pre-computed Hungarian matching indices.

        Args:
            outputs: dict with 'class_logits' (B, Q, C) and optionally 'pred_boxes' (B, Q, 4)
            targets: list of dicts per batch element with 'labels' and 'boxes'
            indices: list of (pred_idx, tgt_idx) tuples from hungarian_matching

        Returns:
            total_loss (scalar), loss_dict
        """
        class_logits = outputs['class_logits']  # (B, Q, num_classes)
        pred_boxes = outputs.get('pred_boxes')  # (B, Q, 4) or None

        loss_dict = {}
        total_loss = torch.zeros(1, device=class_logits.device, dtype=class_logits.dtype).squeeze()

        # --- Focal classification loss ---
        if self.class_weight > 0:
            tgt_cls = torch.zeros_like(class_logits)
            for i, (pred_idx, tgt_idx) in enumerate(indices):
                if len(pred_idx) > 0:
                    num_targets = len(targets[i]['labels'])
                    if num_targets > 0:
                        tgt_idx_clamped = torch.clamp(tgt_idx, 0, num_targets - 1)
                        matched_labels = targets[i]['labels'][tgt_idx_clamped]
                        tgt_cls[i, pred_idx, matched_labels] = 1.0

            loss_focal = _focal_loss_per_sample(
                class_logits, tgt_cls,
                alpha=self.focal_alpha, gamma=self.focal_gamma, reduction='mean'
            )
            loss_dict['loss_focal'] = loss_focal.item()
            total_loss = total_loss + self.class_weight * loss_focal

        # --- Bounding box losses ---
        if pred_boxes is not None and self.bbox_weight > 0:
            l1_losses, giou_losses = [], []
            num_boxes = 0

            for i, (pred_idx, tgt_idx) in enumerate(indices):
                if len(pred_idx) == 0:
                    continue
                num_targets_i = len(targets[i]['boxes'])
                if num_targets_i > 0:
                    tgt_idx_clamped = torch.clamp(tgt_idx, 0, num_targets_i - 1)
                else:
                    continue
                src = pred_boxes[i, pred_idx]
                tgt = targets[i]['boxes'][tgt_idx_clamped].to(src)
                l1_losses.append(F.l1_loss(src, tgt, reduction='sum'))

                if self.giou_weight > 0:
                    src_xyxy = _box_cxcywh_to_xyxy(src)
                    tgt_xyxy = _box_cxcywh_to_xyxy(tgt)
                    giou_diag = torch.diag(_generalized_box_iou(src_xyxy, tgt_xyxy))
                    giou_losses.append((1 - giou_diag).sum())

                num_boxes += len(pred_idx)

            if num_boxes > 0:
                loss_l1 = torch.stack(l1_losses).sum() / num_boxes
                loss_dict['loss_l1'] = loss_l1.item()
                total_loss = total_loss + self.bbox_weight * loss_l1

                if giou_losses:
                    loss_giou = torch.stack(giou_losses).sum() / num_boxes
                    loss_dict['loss_giou'] = loss_giou.item()
                    total_loss = total_loss + self.giou_weight * loss_giou

        loss_dict['loss_detr_total'] = total_loss.item()
        return total_loss, loss_dict


class SurgicalToolDetector(nn.Module):
    """
    Complete surgical tool detection module combining encoder features
    with DETR head for end-to-end detection.
    """
    
    def __init__(
        self,
        encoder_dim: int = 768,
        num_tools: int = 7,
        num_queries: int = 16,
        num_decoder_layers: int = 5,
        nheads: int = 8,
        dropout: float = 0.15,
        dim_feedforward: int = 2048,
        num_pos_tokens: int = 4608,
        class_weight: float = 1.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        neck_dim: int = 256,
    ):
        super().__init__()
        self.num_classes = num_tools
        self.neck_dim = neck_dim

        # Input projection — maps neck_dim → encoder_dim for DETR decoder
        # (neck outputs neck_dim; decoder expects encoder_dim)
        self.input_proj = nn.Linear(neck_dim, encoder_dim)

        # DETR head
        self.detr_head = DetrDetectionHead(
            d_model=encoder_dim,
            num_tools=num_tools,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            nheads=nheads,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            num_pos_tokens=num_pos_tokens,
            class_weight=class_weight,
            bbox_weight=bbox_weight
        )

        # Loss criterion with Focal Loss + GIoU
        self.criterion = SetCriterion(
            num_classes=num_tools,
            class_weight=class_weight,
            bbox_weight=bbox_weight,
            giou_weight=giou_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
    
    def forward(
        self,
        encoder_features: torch.Tensor,
        phantom_features: Optional[torch.Tensor] = None,
        targets: Optional[List[dict]] = None
    ) -> dict:
        """
        Args:
            encoder_features: (B, N, C) from encoder/fused features.
                              C can be neck_dim (from EncoderNeck) or encoder_dim.
            phantom_features: (B, N, C) predicted features for Tri-Attention.
            targets: Optional ground truth for training

        Returns:
            dict with predictions and optionally losses
        """
        # Project from neck_dim → encoder_dim for DETR decoder
        features = self.input_proj(encoder_features)
        
        if phantom_features is not None:
            # Phantom comes in encoder_dim space from predictor (768)
            # Only project if it's neck_dim (256), otherwise use as-is
            if phantom_features.size(-1) == self.neck_dim:
                phantom = self.input_proj(phantom_features)
            else:
                phantom = phantom_features  # Already encoder_dim
        else:
            phantom = None

        # DETR forward with Tri-Attention
        outputs = self.detr_head(features, phantom_features=phantom)

        result = {'pred': outputs}

        # Compute loss if targets provided
        if targets is not None and self.training:
            class_logits = outputs['class_logits']
            pred_boxes_out = outputs.get('pred_boxes')
            indices = self.criterion.hungarian_matching(class_logits, pred_boxes_out, targets)
            loss, loss_dict = self.criterion.compute_losses(outputs, targets, indices)
            result['loss'] = loss
            result['loss_dict'] = loss_dict
            result['matching_indices'] = indices

        return result
    
    def post_process(
        self,
        outputs: dict,
        threshold: float = 0.5
    ) -> List[dict]:
        """
        Post-process predictions to get final detections.
        
        Args:
            outputs: dict with 'class_logits' and optionally 'pred_boxes'
            threshold: Confidence threshold for valid detections
        
        Returns:
            List of dicts with 'labels', 'scores', and optionally 'boxes'
        """
        class_logits = outputs['class_logits']
        pred_boxes = outputs.get('pred_boxes')
        
        prob = class_logits.sigmoid() if class_logits.shape[-1] == self.num_classes else class_logits.softmax(-1)[..., :-1]
        scores, labels = prob.max(-1)
        
        results = []
        for i in range(class_logits.size(0)):
            # Filter by threshold
            mask = scores[i] > threshold
            
            result = {
                'labels': labels[i][mask],
                'scores': scores[i][mask]
            }
            
            if pred_boxes is not None:
                result['boxes'] = pred_boxes[i][mask]
            
            results.append(result)
        
        return results
