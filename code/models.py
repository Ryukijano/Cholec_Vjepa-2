"""
Detection head models for Phase 2 training.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class MidBackboneHook:
    """Hook to extract intermediate features from ViT encoder."""
    
    def __init__(self, layer_idx: int = 12):
        self.layer_idx = layer_idx
        self.features = None
        self.handle = None
    
    def register(self, blocks: nn.ModuleList):
        """Register hook on specified layer."""
        def hook_fn(module, input, output):
            self.features = output
        
        self.handle = blocks[self.layer_idx].register_forward_hook(hook_fn)
    
    def remove(self):
        if self.handle:
            self.handle.remove()
    
    def get_features(self):
        return self.features


class LoRAAdapter(nn.Module):
    """Simple LoRA adapter for feature adaptation."""
    
    def __init__(self, dim: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank if rank > 0 else 0.0
        
        if rank > 0:
            self.down = nn.Linear(dim, rank, bias=False)
            self.up = nn.Linear(rank, dim, bias=False)
        else:
            self.register_parameter('down', None)
            self.register_parameter('up', None)
    
    def forward(self, x):
        if self.rank == 0:
            return x
        return x + self.up(self.down(x)) * self.scaling


class TokenSpatialClassifier(nn.Module):
    """Direct spatial supervision head on encoder tokens.
    
    For each GT box, pools encoder patches inside the box and classifies them.
    This provides DIRECT gradient signal to LoRA adapters, bypassing the decoder.
    Inspired by iBOT (patch-level supervision) and ViTDet (dense prediction on ViT tokens).
    """
    
    def __init__(self, embed_dim: int = 1024, num_classes: int = 7, hidden_dim: int = 256, grid_size: int = 14):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        # Lightweight 2-layer classifier on pooled encoder tokens
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )
        # Background vs foreground binary head (helps encoder learn object boundaries)
        self.fg_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def _get_roi_indices(self, box_cxcywh: torch.Tensor) -> List[int]:
        """Convert a cxcywh box to list of grid cell indices inside it."""
        G = self.grid_size
        cx, cy, w, h = box_cxcywh.tolist()
        x1 = int(max(0, (cx - w/2) * G))
        y1 = int(max(0, (cy - h/2) * G))
        x2 = int(min(G, (cx + w/2) * G))
        y2 = int(min(G, (cy + h/2) * G))
        if x2 <= x1: x2 = min(x1 + 1, G)
        if y2 <= y1: y2 = min(y1 + 1, G)
        return [gy * G + gx for gy in range(y1, y2) for gx in range(x1, x2)]
    
    def forward(self, spatial_tokens: torch.Tensor, 
                targets_boxes: List[torch.Tensor],
                targets_labels: List[torch.Tensor]) -> tuple:
        """Compute direct spatial classification + foreground/background loss.
        
        Args:
            spatial_tokens: [B, 196, D] encoder output (after temporal pooling)
            targets_boxes: list of [N_i, 4] cxcywh boxes per image
            targets_labels: list of [N_i] class labels per image
            
        Returns:
            cls_loss: token-level classification loss
            fg_loss: foreground/background binary loss
        """
        device = spatial_tokens.device
        B, N, D = spatial_tokens.shape
        G = self.grid_size
        
        all_fg_tokens = []
        all_fg_labels = []
        all_bg_tokens = []
        
        for b in range(B):
            boxes = targets_boxes[b]
            labels = targets_labels[b]
            
            if boxes.numel() == 0:
                # All tokens are background
                all_bg_tokens.append(spatial_tokens[b])  # [196, D]
                continue
            
            # Mark which grid cells are inside any GT box
            fg_mask = torch.zeros(N, dtype=torch.bool, device=device)
            
            for i in range(boxes.shape[0]):
                indices = self._get_roi_indices(boxes[i])
                if not indices:
                    continue
                idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
                fg_mask[idx_tensor] = True
                
                # Pool tokens inside this box for classification
                roi_tokens = spatial_tokens[b, idx_tensor]  # [K, D]
                roi_pooled = roi_tokens.mean(dim=0, keepdim=True)  # [1, D]
                all_fg_tokens.append(roi_pooled)
                all_fg_labels.append(labels[i].unsqueeze(0))
            
            # Background tokens (not inside any GT box)
            bg_indices = (~fg_mask).nonzero(as_tuple=True)[0]
            if bg_indices.numel() > 0:
                # Subsample background tokens (max 16 per image to balance)
                if bg_indices.numel() > 16:
                    perm = torch.randperm(bg_indices.numel(), device=device)[:16]
                    bg_indices = bg_indices[perm]
                all_bg_tokens.append(spatial_tokens[b, bg_indices])
        
        # === Classification loss on ROI-pooled tokens ===
        cls_loss = torch.tensor(0., device=device)
        if all_fg_tokens:
            fg_tokens = torch.cat(all_fg_tokens, dim=0)  # [M, D]
            fg_labels = torch.cat(all_fg_labels, dim=0)   # [M]
            fg_logits = self.classifier(fg_tokens)         # [M, num_classes]
            cls_loss = nn.functional.cross_entropy(fg_logits, fg_labels)
        
        # === Foreground/Background binary loss ===
        fg_loss = torch.tensor(0., device=device)
        if all_fg_tokens and all_bg_tokens:
            fg_tokens_flat = torch.cat(all_fg_tokens, dim=0)  # [M, D]
            bg_tokens_flat = torch.cat(all_bg_tokens, dim=0)  # [K, D]
            
            fg_scores = self.fg_head(fg_tokens_flat).squeeze(-1)  # [M]
            bg_scores = self.fg_head(bg_tokens_flat).squeeze(-1)  # [K]
            
            fg_targets = torch.ones_like(fg_scores)
            bg_targets = torch.zeros_like(bg_scores)
            
            all_scores = torch.cat([fg_scores, bg_scores])
            all_targets = torch.cat([fg_targets, bg_targets])
            fg_loss = nn.functional.binary_cross_entropy_with_logits(all_scores, all_targets)
        
        return cls_loss, fg_loss


class SimpleROIHead(nn.Module):
    """Lightweight detection head using ROI-pooled features."""
    
    def __init__(self, embed_dim: int = 1024, num_classes: int = 7, hidden_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.grid_size = 14
        
        # Mid-backbone projector and fusion
        self.mid_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim))
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim))
        
        # Classifier and bbox heads
        self.class_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes + 1))
        self.box_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4))
    
    def roi_pool(self, spatial_tokens: torch.Tensor, boxes_xywh: torch.Tensor) -> torch.Tensor:
        """Pool spatial tokens within each bbox region."""
        roi_feats = []
        for box in boxes_xywh:
            x, y, w, h = box.tolist()
            x1 = int(max(0, x * self.grid_size))
            y1 = int(max(0, y * self.grid_size))
            x2 = int(min(self.grid_size, (x + w) * self.grid_size))
            y2 = int(min(self.grid_size, (y + h) * self.grid_size))
            if x2 <= x1: x2 = min(x1 + 1, self.grid_size)
            if y2 <= y1: y2 = min(y1 + 1, self.grid_size)
            indices = [gy * self.grid_size + gx for gy in range(y1, y2) for gx in range(x1, x2)]
            roi_feats.append(spatial_tokens[indices].mean(dim=0))
        return torch.stack(roi_feats) if roi_feats else torch.empty(
            (0, spatial_tokens.shape[-1]), device=spatial_tokens.device)
    
    def forward(self, tokens: torch.Tensor, mid_tokens: torch.Tensor, boxes_xywh: List[torch.Tensor]):
        """Training forward with GT boxes."""
        if mid_tokens is not None:
            mid_proj = self.mid_projector(mid_tokens.float())
            fused = self.fusion(torch.cat([mid_proj, tokens.float()], dim=-1))
        else:
            fused = tokens.float()
        
        all_logits, all_deltas = [], []
        for b, boxes in enumerate(boxes_xywh):
            if boxes.shape[0] == 0:
                continue
            roi_feats = self.roi_pool(fused[b], boxes)
            all_logits.append(self.class_head(roi_feats))
            all_deltas.append(self.box_head(roi_feats))
        
        if not all_logits:
            device = tokens.device
            return (torch.zeros((0, self.num_classes + 1), device=device),
                    torch.zeros((0, 4), device=device))
        return torch.cat(all_logits), torch.cat(all_deltas)


class SpatialCrossAttention(nn.Module):
    """Cross-attention with spatial bias toward each query's prior position.
    
    Queries at position (qx, qy) preferentially attend to nearby spatial tokens,
    preventing queries from fighting over the same central region.
    """
    
    def __init__(self, embed_dim: int, nheads: int, dropout: float = 0.1,
                 spatial_bias_scale: float = 4.0):
        super().__init__()
        self.nheads = nheads
        self.head_dim = embed_dim // nheads
        self.spatial_bias_scale = spatial_bias_scale
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries: torch.Tensor, memory: torch.Tensor,
                query_positions: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: [B, Q, D]
            memory: [B, N, D] spatial tokens
            query_positions: [Q, 2] normalized (x, y) of each query's prior
            token_positions: [N, 2] normalized (x, y) of each spatial token
        """
        B, Q, D = queries.shape
        N = memory.shape[1]
        
        q = self.q_proj(queries).view(B, Q, self.nheads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(B, N, self.nheads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(B, N, self.nheads, self.head_dim).transpose(1, 2)
        
        # Standard dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Spatial bias: Gaussian centered on query prior position
        # dist2[q, n] = ||query_pos[q] - token_pos[n]||^2
        dist2 = ((query_positions.unsqueeze(1) - token_positions.unsqueeze(0)) ** 2).sum(-1)  # [Q, N]
        spatial_bias = -self.spatial_bias_scale * dist2  # [Q, N]
        attn = attn + spatial_bias.unsqueeze(0).unsqueeze(1)  # broadcast over B and heads
        
        attn = attn.softmax(-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Q, D)
        return self.out_proj(out)


class RefinementDecoderLayer(nn.Module):
    """Single decoder layer with spatial-biased cross-attention + FFN."""
    
    def __init__(self, embed_dim: int, nheads: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, spatial_bias_scale: float = 4.0):
        super().__init__()
        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(embed_dim, nheads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Spatial-biased cross-attention
        self.cross_attn = SpatialCrossAttention(embed_dim, nheads, dropout, spatial_bias_scale)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim), nn.Dropout(dropout))
        self.norm3 = nn.LayerNorm(embed_dim)
    
    def forward(self, queries: torch.Tensor, memory: torch.Tensor,
                query_positions: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Self-attention
        q2 = self.norm1(queries)
        q2, _ = self.self_attn(q2, q2, q2)
        queries = queries + q2
        
        # Spatial-biased cross-attention
        q2 = self.norm2(queries)
        q2 = self.cross_attn(q2, memory, query_positions, token_positions)
        queries = queries + q2
        
        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        return queries


class LightweightQueryDecoder(nn.Module):
    """Two-step refinement decoder with spatial priors and relative delta prediction.
    
    Architecture:
      1. Input projection: embed_dim (1024) → internal_dim (e.g. 256)
      2. Periphery-prior query initialization (tools enter from frame edges)
      3. N decoder layers: spatial-biased cross-attention → predict box deltas
      4. ROI re-sample after first layer, refine in subsequent layers
      5. Class-weighted prediction on refined query features
    
    The internal_dim bottleneck prevents overfitting on small datasets
    (37M→~3M params with internal_dim=256).
    """
    
    def __init__(self, embed_dim: int = 1024, num_classes: int = 7, 
                 num_queries: int = 10, num_decoder_layers: int = 2,
                 nheads: int = 8, dropout: float = 0.1,
                 dim_feedforward: int = 2048, grid_size: int = 14,
                 internal_dim: int = 0, temporal_pooling: str = 'last'):
        super().__init__()
        # If internal_dim=0, use embed_dim (backward compat)
        d = internal_dim if internal_dim > 0 else embed_dim
        self.embed_dim = embed_dim
        self.internal_dim = d
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_decoder_layers = max(num_decoder_layers, 1)
        self.grid_size = grid_size
        self.temporal_pooling = temporal_pooling
        
        # === Input projection: embed_dim → internal_dim ===
        if d != embed_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(embed_dim, d), nn.LayerNorm(d), nn.GELU())
            self.mid_projector = nn.Sequential(
                nn.Linear(embed_dim, d), nn.LayerNorm(d), nn.GELU())
        else:
            self.input_proj = nn.Identity()
            self.mid_projector = nn.Sequential(
                nn.Linear(embed_dim, d), nn.LayerNorm(d), nn.GELU(),
                nn.Linear(d, d))
        
        # Fusion (mid + final) → internal_dim
        self.fusion = nn.Sequential(
            nn.Linear(d * 2, d), nn.LayerNorm(d), nn.GELU())
        
        # Periphery spatial priors — tools enter from frame edges in surgery
        base_priors = torch.tensor([
            [0.10, 0.30], [0.10, 0.50], [0.10, 0.70],  # left edge
            [0.90, 0.30], [0.90, 0.50], [0.90, 0.70],  # right edge
            [0.50, 0.10], [0.50, 0.90],                  # top/bottom
            [0.30, 0.50], [0.70, 0.50],                  # mid-left/mid-right
        ], dtype=torch.float32)
        # Tile base priors to cover num_queries, with small jitter for diversity
        repeats = (num_queries + len(base_priors) - 1) // len(base_priors)
        spatial_priors = base_priors.repeat(repeats, 1)[:num_queries]
        if num_queries > len(base_priors):
            jitter = torch.randn_like(spatial_priors) * 0.05
            spatial_priors = (spatial_priors + jitter).clamp(0.01, 0.99)
            # Keep first 10 exact (no jitter on base priors)
            spatial_priors[:len(base_priors)] = base_priors
        # Default prior box size (typical tool dimensions)
        prior_sizes = torch.tensor([0.18, 0.20]).unsqueeze(0).expand(num_queries, -1)
        # [Q, 4] = [cx, cy, w, h]
        self.register_buffer('prior_boxes', 
                             torch.cat([spatial_priors, prior_sizes], dim=-1))
        # [Q, 2] = just the center positions for spatial bias
        self.register_buffer('query_positions', spatial_priors)
        
        # Token grid positions [196, 2] — normalized (x, y) for each 14x14 token
        gy, gx = torch.meshgrid(
            torch.linspace(0.5/grid_size, 1 - 0.5/grid_size, grid_size),
            torch.linspace(0.5/grid_size, 1 - 0.5/grid_size, grid_size),
            indexing='ij')
        self.register_buffer('token_positions', torch.stack([gx.flatten(), gy.flatten()], dim=-1))
        
        # Learnable query content embeddings at internal_dim
        self.query_content = nn.Embedding(num_queries, d)
        # Spatial prior → query position encoding
        self.prior_pos_embed = nn.Sequential(
            nn.Linear(4, d // 2), nn.GELU(),
            nn.Linear(d // 2, d))
        
        # 2D sinusoidal positional embeddings for spatial tokens
        self.register_buffer('spatial_pos_embed', 
                             self._build_2d_sincos_pos_embed(d, grid_size))
        
        # Global context projector
        self.global_context_proj = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())
        
        # Decoder layers (ModuleList — num_decoder_layers actually works now)
        self.layers = nn.ModuleList([
            RefinementDecoderLayer(d, nheads, dim_feedforward, dropout)
            for _ in range(self.num_decoder_layers)
        ])
        
        # Box heads — predict DELTAS from prior, not absolute coords
        # After first layer: coarse box delta
        self.box_head_coarse = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, 4))
        # After last layer: refined box delta (from coarse box)
        self.box_head_refine = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, 4))
        
        # Class head (applied to final refined features)
        self.class_head = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(d, num_classes + 1))
        
        # ROI feature aggregator for refinement step
        self.roi_proj = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())
        
        # Temporal propagation
        self.query_init_proj = nn.Linear(d, d)
        self.temporal_alpha = nn.Parameter(torch.tensor(0.5))
        
        # Init biases
        nn.init.constant_(self.class_head[-1].bias, 0)
        for bh in [self.box_head_coarse, self.box_head_refine]:
            nn.init.xavier_uniform_(bh[-1].weight, gain=0.01)
            nn.init.constant_(bh[-1].bias, 0)
    
    @staticmethod
    def _build_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
        """Build 2D sinusoidal positional embedding."""
        assert embed_dim % 4 == 0
        half_dim = embed_dim // 4
        
        gy = torch.arange(grid_size, dtype=torch.float32)
        gx = torch.arange(grid_size, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        grid_y = grid_y.reshape(-1)
        grid_x = grid_x.reshape(-1)
        
        omega = 1.0 / (10000 ** (torch.arange(half_dim, dtype=torch.float32) / half_dim))
        
        pos_x = grid_x.unsqueeze(1) * omega.unsqueeze(0)
        pos_y = grid_y.unsqueeze(1) * omega.unsqueeze(0)
        
        pos_embed = torch.cat([
            torch.sin(pos_x), torch.cos(pos_x),
            torch.sin(pos_y), torch.cos(pos_y)
        ], dim=-1)
        
        return pos_embed.unsqueeze(0)
    
    def _apply_box_delta(self, prior: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Apply predicted delta to prior box to get final box.
        
        Args:
            prior: [Q, 4] or [B, Q, 4] in cxcywh
            delta: [B, Q, 4] raw delta predictions
        Returns:
            boxes: [B, Q, 4] in cxcywh, clamped to [0,1]
        """
        # delta[:,:,0:2] shifts center, delta[:,:,2:4] scales w/h
        cx = prior[..., 0] + delta[..., 0] * 0.2  # small center shift
        cy = prior[..., 1] + delta[..., 1] * 0.2
        w = prior[..., 2] * torch.exp(delta[..., 2].clamp(-2, 2))  # log-space scale
        h = prior[..., 3] * torch.exp(delta[..., 3].clamp(-2, 2))
        
        boxes = torch.stack([cx, cy, w, h], dim=-1)
        return boxes.clamp(0, 1)
    
    def _roi_pool_from_grid(self, spatial_tokens: torch.Tensor, 
                             boxes_cxcywh: torch.Tensor) -> torch.Tensor:
        """Pool spatial tokens within predicted box regions on the 14x14 grid.
        
        Args:
            spatial_tokens: [B, 196, D]
            boxes_cxcywh: [B, Q, 4] normalized
        Returns:
            roi_features: [B, Q, D]
        """
        B, Q, _ = boxes_cxcywh.shape
        G = self.grid_size
        D = spatial_tokens.shape[-1]
        
        roi_feats = torch.zeros(B, Q, D, device=spatial_tokens.device, dtype=spatial_tokens.dtype)
        
        for b in range(B):
            for q in range(Q):
                cx, cy, w, h = boxes_cxcywh[b, q].tolist()
                x1 = int(max(0, (cx - w/2) * G))
                y1 = int(max(0, (cy - h/2) * G))
                x2 = int(min(G, (cx + w/2) * G))
                y2 = int(min(G, (cy + h/2) * G))
                if x2 <= x1: x2 = min(x1 + 1, G)
                if y2 <= y1: y2 = min(y1 + 1, G)
                
                indices = [gy * G + gx for gy in range(y1, y2) for gx in range(x1, x2)]
                if indices:
                    roi_feats[b, q] = spatial_tokens[b, indices].mean(dim=0)
        
        return roi_feats
    
    def forward(self, tokens: torch.Tensor, mid_tokens: torch.Tensor,
                prev_query_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
        """Forward pass with two-step refinement.
        
        Returns:
            logits: [B, num_queries, num_classes+1]
            boxes: [B, num_queries, 4] refined cxcywh boxes
            query_features: [B, num_queries, internal_dim]
            aux_outputs: list of {logits, boxes} for intermediate layers
        """
        B = tokens.shape[0]
        N_spatial = self.grid_size * self.grid_size  # 196
        
        # Temporal pooling: encoder outputs [B, T*N_spatial, D] flattened
        # or [B, T, N_spatial, D] pre-reshaped, or [B, N_spatial, D] already pooled
        if tokens.dim() == 3 and tokens.shape[1] > N_spatial:
            # Flattened temporal: [B, T*196, D] → reshape to [B, T, 196, D]
            D = tokens.shape[-1]
            T = tokens.shape[1] // N_spatial
            tokens = tokens.view(B, T, N_spatial, D)
        if tokens.dim() == 4:
            if self.temporal_pooling == 'mean':
                tokens = tokens.mean(dim=1)  # [B, N_spatial, D]
            else:
                tokens = tokens[:, -1]       # [B, N_spatial, D] last frame
        
        if mid_tokens is not None:
            if mid_tokens.dim() == 3 and mid_tokens.shape[1] > N_spatial:
                D_m = mid_tokens.shape[-1]
                T_m = mid_tokens.shape[1] // N_spatial
                mid_tokens = mid_tokens.view(B, T_m, N_spatial, D_m)
            if mid_tokens.dim() == 4:
                if self.temporal_pooling == 'mean':
                    mid_tokens = mid_tokens.mean(dim=1)
                else:
                    mid_tokens = mid_tokens[:, -1]
        
        # Project to internal_dim (no-op if internal_dim == embed_dim)
        tokens_proj = self.input_proj(tokens)
        
        # Feature fusion (mid + final backbone features)
        if mid_tokens is not None:
            fused = self.fusion(torch.cat([
                self.mid_projector(mid_tokens), tokens_proj], dim=-1))
        else:
            fused = tokens_proj
        
        # Add spatial positional embeddings to memory tokens
        fused = fused + self.spatial_pos_embed
        
        # Global context
        global_ctx = self.global_context_proj(fused.mean(dim=1, keepdim=True))
        
        # Initialize queries: content + positional prior embedding
        prior_pos_emb = self.prior_pos_embed(self.prior_boxes)  # [Q, d]
        
        if prev_query_features is not None:
            static_queries = self.query_content.weight + prior_pos_emb
            static_queries = static_queries.unsqueeze(0).expand(B, -1, -1)
            temporal_queries = self.query_init_proj(prev_query_features)
            alpha = torch.sigmoid(self.temporal_alpha)
            queries = alpha * temporal_queries + (1.0 - alpha) * static_queries
        else:
            queries = (self.query_content.weight + prior_pos_emb).unsqueeze(0).expand(B, -1, -1)
        
        queries = queries + global_ctx
        
        # Inject noise during training to force query diversity
        if self.training:
            query_noise = torch.randn_like(queries) * 0.05
            queries = queries + query_noise
        
        aux_outputs = []
        
        # === LAYER 1: Coarse prediction ===
        queries = self.layers[0](queries, fused, self.query_positions, self.token_positions)
        coarse_delta = self.box_head_coarse(queries)
        coarse_boxes = self._apply_box_delta(
            self.prior_boxes.unsqueeze(0).expand(B, -1, -1), coarse_delta)
        
        # For auxiliary loss
        l1_logits = self.class_head(queries)
        aux_outputs.append({'logits': l1_logits, 'boxes': coarse_boxes})
        
        # === ROI re-sample + remaining layers for refinement ===
        roi_features = self._roi_pool_from_grid(fused, coarse_boxes.detach())
        # Clone to avoid CUDA graph overwrite when torch.compile(reduce-overhead) is used
        roi_features = self.roi_proj(roi_features.clone())
        queries_refined = queries + roi_features
        
        # Update query positions to coarse box centers for spatial bias
        coarse_centers = coarse_boxes[:, :, :2].mean(dim=0).detach()  # [Q, 2]
        
        for i, layer in enumerate(self.layers[1:]):
            queries_refined = layer(queries_refined, fused, coarse_centers, self.token_positions)
            
            # Add intermediate outputs if not the last layer
            if i < len(self.layers[1:]) - 1:
                cur_logits = self.class_head(queries_refined)
                cur_delta = self.box_head_refine(queries_refined)
                cur_boxes = self._apply_box_delta(coarse_boxes, cur_delta)
                aux_outputs.append({'logits': cur_logits, 'boxes': cur_boxes})
        
        refine_delta = self.box_head_refine(queries_refined)
        refined_boxes = self._apply_box_delta(coarse_boxes, refine_delta)
        
        # Final classification on refined features
        logits = self.class_head(queries_refined)
        
        return logits, refined_boxes, queries_refined, aux_outputs


class DetectionHeadWithMidInjection(nn.Module):
    """Full DETR-style decoder with mid-backbone feature injection."""
    
    def __init__(self, embed_dim: int = 1024, num_queries: int = 20,
                 num_classes: int = 10, num_heads: int = 16, depth: int = 3,
                 mid_layer_idx: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        
        self.mid_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim))
        
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim))
        
        # Spatially-aware query initialization
        spatial_positions = torch.tensor([
            [0.1, 0.1], [0.1, 0.5], [0.1, 0.9],
            [0.9, 0.1], [0.9, 0.5], [0.9, 0.9],
            [0.5, 0.1], [0.5, 0.9],
            [0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8],
            [0.3, 0.3], [0.3, 0.7], [0.7, 0.3], [0.7, 0.7],
            [0.5, 0.5], [0.4, 0.4], [0.6, 0.4], [0.6, 0.6],
        ], dtype=torch.float32)
        
        spatial_embed = nn.Linear(2, embed_dim, bias=False)
        spatial_bias = spatial_embed(spatial_positions[:num_queries])
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, embed_dim) + spatial_bias.unsqueeze(0))
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        
        # Heads
        self.class_head = nn.Linear(embed_dim, num_classes + 1)
        self.box_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 4), nn.Sigmoid())
    
    def forward(self, tokens: torch.Tensor, mid_tokens: torch.Tensor = None):
        """Forward pass.
        
        Returns:
            logits: [B, Q, num_classes+1]
            boxes: [B, Q, 4] in cxcywh format
        """
        B = tokens.size(0)
        
        if mid_tokens is not None:
            mid_proj = self.mid_projector(mid_tokens)
            combined = torch.cat([mid_proj, tokens], dim=-1)
            tokens = self.fusion(combined)
        
        queries = self.query_tokens.expand(B, -1, -1)
        queries_out = self.decoder(tgt=queries, memory=tokens)
        
        logits = self.class_head(queries_out)
        boxes = self.box_head(queries_out)
        
        return logits, boxes


# ---------------------------------------------------------------------------
# Inference post-processing: score filter + NMS to merge duplicate query boxes
# ---------------------------------------------------------------------------
def box_cxcywh_to_xyxy_torch(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [..., 4] cxcywh (normalized) to xyxy."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def detection_postprocess(
    pred_boxes: torch.Tensor,
    scores: torch.Tensor,
    pred_cls: torch.Tensor,
    score_thresh: float = 0.8,
    nms_thresh: float = 0.2,
    num_classes: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter by score and run per-class NMS to merge duplicate decoder queries.
    Returns (boxes, scores, pred_cls) for kept detections only.
    """
    # Score filter (and drop no-object / invalid class)
    keep_mask = (scores >= score_thresh) & (pred_cls < num_classes)
    if not keep_mask.any():
        return pred_boxes[keep_mask], scores[keep_mask], pred_cls[keep_mask]
    boxes = pred_boxes[keep_mask]
    sc = scores[keep_mask]
    cl = pred_cls[keep_mask]
    # cxcywh -> xyxy for NMS
    xyxy = box_cxcywh_to_xyxy_torch(boxes)
    try:
        from torchvision.ops import batched_nms
        keep_idx = batched_nms(xyxy, sc, cl, nms_thresh)
    except ImportError:
        keep_idx = torch.arange(boxes.shape[0], device=boxes.device)
    return boxes[keep_idx], sc[keep_idx], cl[keep_idx]
