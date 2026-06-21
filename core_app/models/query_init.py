"""
Object query initialization with template conditioning support.

This module provides ObjectQueryInit, which supports both:
- Current behavior: learnable positional queries (reset every frame)
- Future Stage 2: template-conditioned queries (seeded from first-frame bbox crops)

The template conditioning hook allows persistent identity across frames without
re-running detection from scratch every frame (TrackFormer / MOTR pattern).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ObjectQueryInit(nn.Module):
    """
    Object query initialization with optional template conditioning.
    
    This module provides a clean migration path from standard DETR learnable queries
    to template-conditioned queries for persistent tracking (Stage 2).
    
    Current behavior (template_features=None):
        - Uses learnable positional queries
        - Queries reset every frame (standard DETR)
    
    Stage 2 behavior (template_features provided):
        - Adds template features to learnable queries
        - Queries carry identity across frames (persistent tracking)
    
    Args:
        N_max: Maximum number of queries (e.g., 16 for surgical tools)
        d_q: Query dimension (e.g., 256 for neck_dim)
        template_dim: Dimension of template features (if different from d_q)
    
    Usage:
        # Current behavior (Stage 1)
        query_init = ObjectQueryInit(N_max=16, d_q=256)
        queries = query_init()  # (N_max, d_q)
        
        # Stage 2 with template conditioning
        queries = query_init(template_features)  # (N_max, d_q)
    """
    
    def __init__(self, N_max: int = 16, d_q: int = 256, template_dim: Optional[int] = None):
        super().__init__()
        self.N_max = N_max
        self.d_q = d_q
        
        # Current behavior: learnable positional queries
        self.learned_queries = nn.Embedding(N_max, d_q)
        
        # Future hook: template projection for Stage 2
        # Zero-initialized to start from current behavior when enabled
        if template_dim is not None and template_dim != d_q:
            self.template_proj = nn.Linear(template_dim, d_q)
            nn.init.zeros_(self.template_proj.weight)
            nn.init.zeros_(self.template_proj.bias)
        else:
            # Identity projection if dimensions match
            self.template_proj = None
    
    def forward(self, template_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Initialize object queries.
        
        Args:
            template_features: Optional (N_max, template_dim) template features
                               from first-frame bbox crops. If None, uses current
                               learnable queries only.
        
        Returns:
            (N_max, d_q) query embeddings
        """
        # Base queries from learned positional embeddings
        q = self.learned_queries.weight  # (N_max, d_q)
        
        # Add template conditioning if provided (Stage 2 path)
        if template_features is not None:
            if self.template_proj is not None:
                q = q + self.template_proj(template_features)
            else:
                q = q + template_features  # dimensions match, direct addition
        
        return q
