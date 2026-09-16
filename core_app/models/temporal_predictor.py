"""
Multi-Scale Temporal Predictor for V-JEPA 2.1 World Model
Predicts future frame representations at 1, 4, and 16 frame horizons.
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List


class PositionalEncoding3D(nn.Module):
    """3D positional encoding for spatiotemporal features."""
    
    def __init__(self, d_model: int, max_t: int = 32, max_h: int = 24, max_w: int = 24):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_t, max_h, max_w, d_model)
        t_pos = torch.arange(0, max_t, dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        h_pos = torch.arange(0, max_h, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        w_pos = torch.arange(0, max_w, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, :, :, 0::2] = torch.sin(t_pos.unsqueeze(-1) * div_term)
        pe[:, :, :, 1::2] = torch.cos(t_pos.unsqueeze(-1) * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor, t_idx: int = 0) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W, C) or (B, N, C) where N = T*H*W
            t_idx: Starting temporal index for encoding
        """
        if x.dim() == 5:
            B, T, H, W, C = x.shape
            return x + self.pe[t_idx:t_idx+T, :H, :W, :]
        return x


class TemporalAttention(nn.Module):
    """Temporal attention with causal masking for future prediction."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class PredictorBlock(nn.Module):
    """Transformer block for temporal prediction."""
    
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TemporalAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class MultiScaleTemporalPredictor(nn.Module):
    """
    Multi-scale temporal predictor that forecasts future V-JEPA representations
    at 1, 4, and 16 frame horizons.
    
    Architecture:
    - Shared transformer trunk (6-12 layers)
    - Separate prediction heads for each horizon
    - Causal masking to prevent future information leakage
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        prediction_horizons: List[int] = [1, 4, 16],
        dropout: float = 0.2,
        max_seq_len: int = 4608,
        mlp_ratio: float = 2.0,
        use_tiny_mode: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prediction_horizons = prediction_horizons
        
        # Input projection (small if input_dim != hidden_dim, identity if equal)
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        
        # Transformer trunk - tiny for V-JEPA features which are already excellent
        effective_mlp_ratio = 2.0 if use_tiny_mode else mlp_ratio
        self.blocks = nn.ModuleList([
            PredictorBlock(hidden_dim, num_heads, mlp_ratio=effective_mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Separate prediction heads for each horizon - tiny MLP
        head_hidden = hidden_dim // 2 if use_tiny_mode else hidden_dim
        self.prediction_heads = nn.ModuleDict({
            f'horizon_{h}': nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, input_dim)
            )
            for h in prediction_horizons
        })
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'prediction_heads' in name and isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
    
    def create_causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Create causal mask for autoregressive prediction."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_horizons: bool = True,
        return_all_steps: bool = False
    ) -> dict:
        """
        Args:
            x: (B, T, N, C) where T=temporal, N=spatial tokens, C=features
               or (B, T*N, C) flattened
            return_all_horizons: If True, return predictions for all horizons
            return_all_steps: If True, return predictions for all input time steps (T)
                              of shape (B, T, N, C). Otherwise, only for the last step.
        
        Returns:
            dict with keys like 'horizon_1', 'horizon_4', 'horizon_16'
            If return_all_steps=True: Each value is (B, T, N, C)
            If return_all_steps=False: Each value is (B, N, C)
        """
        B = x.shape[0]
        
        # Handle input shape
        if x.dim() == 4:
            B, T, N, C = x.shape
            x_flat = x.reshape(B, T * N, C)
        else:
            B, TN, C = x.shape
            x_flat = x
            T = 1 # We don't know T/N split if flattened, assume 1 temporal step
            N = TN
        
        # Project to hidden dim
        h = self.input_proj(x_flat)
        
        # Add positional encoding
        if h.size(1) <= self.pos_embed.size(1):
            h = h + self.pos_embed[:, :h.size(1), :]
        else:
            # Linear interpolation or repeat if sequence is longer (fallback)
            repeat = (h.size(1) // self.pos_embed.size(1)) + 1
            pe = self.pos_embed.repeat(1, repeat, 1)
            h = h + pe[:, :h.size(1), :]
        
        # Create causal mask
        mask = self.create_causal_mask(h.size(1), h.device, dtype=h.dtype)
        
        # Transformer forward
        for block in self.blocks:
            h = block(h, mask)
        
        h = self.norm(h)
        
        # Generate predictions for each horizon
        predictions = {}
        
        if return_all_steps and T > 1:
            # Reshape h back to (B, T, N, hidden_dim) to predict from each step
            h_seq = h.reshape(B, T, N, -1)
            
            for hor in self.prediction_horizons:
                pred = self.prediction_heads[f'horizon_{hor}'](h_seq) # (B, T, N, C)
                predictions[f'horizon_{hor}'] = pred
        else:
            # We use the last N tokens (representing the last temporal step) to predict the future.
            last_frame_h = h[:, -N:] # (B, N, hidden_dim)
            
            for hor in self.prediction_horizons:
                pred = self.prediction_heads[f'horizon_{hor}'](last_frame_h)
                predictions[f'horizon_{hor}'] = pred
        
        # Handle case where return_all_horizons is False (only return first horizon)
        if not return_all_horizons:
            first_hor = self.prediction_horizons[0]
            return {f'horizon_{first_hor}': predictions[f'horizon_{first_hor}']}
            
        return predictions

    def rollout(
        self,
        initial_features: torch.Tensor,
        num_steps: int,
        step_size: int = 1
    ) -> List[torch.Tensor]:
        """
        Autoregressive rollout (DINO-WM style).
        Predicts future features by feeding predictions back as input.
        
        Args:
            initial_features: (B, T, N, C) starting sequence
            num_steps: How many future steps to predict
            step_size: The temporal distance between steps (usually matches horizon_1)
            
        Returns:
            List of (B, N, C) predicted features for each step
        """
        B, T, N, C = initial_features.shape
        current_seq = initial_features.clone()
        rollout_predictions = []
        
        # We'll use the 'horizon_1' head (or the smallest available horizon)
        # as the one-step-forward predictor for autoregressive rollout.
        horizon_key = f'horizon_{step_size}'
        if horizon_key not in self.prediction_heads:
            # Fallback to the first available horizon
            horizon_key = f'horizon_{self.prediction_horizons[0]}'
            
        for _ in range(num_steps):
            # Forward pass with current sequence
            preds = self.forward(current_seq, return_all_horizons=False)
            next_feat = preds[horizon_key] # (B, N, C)
            
            rollout_predictions.append(next_feat)
            
            # Update sequence for next step (append new prediction, remove oldest if needed)
            # DINO-WM usually keeps a fixed history window or grows it
            next_feat_unsqueezed = next_feat.unsqueeze(1) # (B, 1, N, C)
            current_seq = torch.cat([current_seq, next_feat_unsqueezed], dim=1)
            
            # Optional: Limit history to max_seq_len / N
            max_frames = self.pos_embed.size(1) // N
            if current_seq.size(1) > max_frames:
                current_seq = current_seq[:, -max_frames:]
                
        return rollout_predictions
    
    def predict_future(
        self,
        current_features: torch.Tensor,
        horizon: int
    ) -> torch.Tensor:
        """
        Convenience method for single-horizon inference.
        
        Args:
            current_features: (B, N, C) current frame features
            horizon: Which horizon to predict (must be in prediction_horizons)
        
        Returns:
            (B, N, C) predicted future features
        """
        if horizon not in self.prediction_horizons:
            raise ValueError(f"Horizon {horizon} not in {self.prediction_horizons}")
        
        # Add temporal dim if needed
        if current_features.dim() == 3:
            current_features = current_features.unsqueeze(1)  # (B, 1, N, C)
        
        predictions = self.forward(current_features, return_all_horizons=False)
        return predictions[f'horizon_{horizon}']


class FeatureFusion(nn.Module):
    """
    Fuses current frame features with predicted future features
    for robust tracking under occlusion.
    """
    
    def __init__(self, dim: int, num_horizons: int = 3):
        super().__init__()
        self.dim = dim
        self.num_horizons = num_horizons
        
        # Learnable weights for combining features
        self.fusion_weights = nn.Parameter(torch.ones(num_horizons + 1))  # +1 for current
        
        # Optional: Cross-attention fusion
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(
        self,
        current: torch.Tensor,
        predictions: dict
    ) -> torch.Tensor:
        """
        Args:
            current: (B, N, C) current frame features
            predictions: dict of predicted features for each horizon
        
        Returns:
            (B, N, C) fused features
        """
        # Stack all features
        all_features = [current]
        for h in sorted([int(k.split('_')[1]) for k in predictions.keys()]):
            all_features.append(predictions[f'horizon_{h}'])
        
        stacked = torch.stack(all_features, dim=1)  # (B, num_features, N, C)
        B, K, N, C = stacked.shape
        
        # Apply softmax weights
        weights = torch.softmax(self.fusion_weights[:K], dim=0)
        weights = weights.view(1, K, 1, 1)
        
        # Weighted combination
        fused = (stacked * weights).sum(dim=1)  # (B, N, C)
        
        # Optional cross-attention refinement
        attn_out, _ = self.cross_attn(fused, fused, fused)
        fused = self.norm(fused + attn_out)
        
        return fused


class WorldModelLoss(nn.Module):
    """
    Combined loss for world model (predictor) training.

    Loss terms:
      1. Latent Consistency (MSE)  — per DINO-WM paper: MSE between predicted and
         real patch embeddings in feature space. Stop-gradient on targets is implicit
         because the encoder is frozen.
      2. Rollout Loss (optional)  — short 2-step autoregressive unroll without
         teacher forcing, weighted by rollout_weight. Penalises error accumulation.
      3. Temporal Consistency     — encourages smooth transitions across horizons
         (lightweight, can be disabled).
    """

    def __init__(
        self,
        prediction_horizons: List[int] = [1, 4, 16],
        horizon_weights: Optional[List[float]] = None,
        feature_weight: float = 1.0,
        rollout_weight: float = 0.3,
        temporal_consistency_weight: float = 0.0,
        normalize_targets: bool = True,
    ):
        super().__init__()
        self.prediction_horizons = prediction_horizons
        self.feature_weight = feature_weight
        self.rollout_weight = rollout_weight
        self.temporal_consistency_weight = temporal_consistency_weight
        self.normalize_targets = normalize_targets

        if horizon_weights is None:
            self.horizon_weights = {h: 1.0 for h in prediction_horizons}
        else:
            self.horizon_weights = {h: w for h, w in zip(prediction_horizons, horizon_weights)}

    def _mse_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Per-patch MSE loss in latent space.
        Optionally L2-normalises both pred and target before computing MSE
        (cosine-MSE variant, removes magnitude differences).
        """
        if self.normalize_targets:
            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target.detach(), p=2, dim=-1)  # explicit sg
        else:
            target = target.detach()  # explicit stop-gradient
        return F.mse_loss(pred, target)

    def forward(
        self,
        predictions: dict,
        targets: dict,
        rollout_predictions: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            predictions: dict[f'horizon_{h}'] = (B, N, C) or (B, T, N, C)
                         teacher-forced predictions from the predictor.
            targets: dict[f'horizon_{h}'] = (B, N, C) real encoder outputs.
            rollout_predictions: (optional) dict with same keys but from
                                 autoregressive rollout (no teacher forcing).

        Returns:
            total_loss (scalar Tensor), loss_dict
        """
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device,
                                  dtype=next(iter(predictions.values())).dtype)
        loss_dict = {}

        # ------------------------------------------------------------------ #
        # 1. Latent Consistency Loss (MSE, teacher-forcing)
        # ------------------------------------------------------------------ #
        for h in self.prediction_horizons:
            key = f'horizon_{h}'
            if key not in predictions or key not in targets:
                continue

            pred = predictions[key]
            tgt = targets[key]

            # Flatten temporal dim if present (B, T, N, C) → (B*T, N, C)
            if pred.dim() == 4:
                pred = pred.flatten(0, 1)
                tgt = tgt.flatten(0, 1)

            mse = self._mse_loss(pred, tgt)
            weighted = self.horizon_weights[h] * self.feature_weight * mse
            total_loss = total_loss + weighted
            loss_dict[f'mse_h{h}'] = mse.item()

        # ------------------------------------------------------------------ #
        # 2. Rollout Loss (optional, 2-step autoregressive without teacher)
        # ------------------------------------------------------------------ #
        if rollout_predictions is not None and self.rollout_weight > 0:
            for h in self.prediction_horizons:
                key = f'horizon_{h}'
                if key not in rollout_predictions or key not in targets:
                    continue

                pred_ro = rollout_predictions[key]
                tgt_ro = targets[key]

                if pred_ro.dim() == 4:
                    pred_ro = pred_ro.flatten(0, 1)
                    tgt_ro = tgt_ro.flatten(0, 1)

                ro_loss = self._mse_loss(pred_ro, tgt_ro)
                total_loss = total_loss + self.rollout_weight * ro_loss
                loss_dict[f'rollout_mse_h{h}'] = ro_loss.item()

        # ------------------------------------------------------------------ #
        # 3. Temporal Consistency (optional)
        # ------------------------------------------------------------------ #
        if self.temporal_consistency_weight > 0 and len(self.prediction_horizons) > 1:
            sorted_h = sorted(self.prediction_horizons)
            for i in range(len(sorted_h) - 1):
                h1, h2 = sorted_h[i], sorted_h[i + 1]
                k1, k2 = f'horizon_{h1}', f'horizon_{h2}'
                if k1 not in predictions or k2 not in predictions:
                    continue
                p1 = predictions[k1].detach() if predictions[k1].dim() == 3 else predictions[k1].flatten(0, 1).detach()
                p2 = predictions[k2] if predictions[k2].dim() == 3 else predictions[k2].flatten(0, 1)
                cons = F.mse_loss(p2, p1)
                total_loss = total_loss + self.temporal_consistency_weight * cons
                loss_dict[f'consistency_{h1}_{h2}'] = cons.item()

        loss_dict['total_world_model_loss'] = total_loss.item()
        return total_loss, loss_dict
