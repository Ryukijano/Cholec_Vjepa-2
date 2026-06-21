"""
V-JEPA 2.1 World Model with Multi-Scale Prediction + DETR/ReID Tracking
Main architecture integrating all components.
"""
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path

from .temporal_predictor import MultiScaleTemporalPredictor, FeatureFusion, WorldModelLoss
from .detr_head import SurgicalToolDetector
from .reid_head import ReidHead, RoIAlignExtractor, ToolTracker
from .fpn import EncoderNeck


class VJEPAEncoderWrapper(nn.Module):
    """
    Wrapper for frozen V-JEPA 2 or V-JEPA 2.1 encoder from Meta AI.
    Supports loading from pretrained checkpoints or torch.hub.
    
    V-JEPA 2: Pretrained on endoscopy videos (human phase recognition)
    V-JEPA 2.1: Meta's public model trained on internet-scale video
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        hf_model_name: Optional[str] = None,
        use_torch_hub: bool = False,
        model_name: str = 'vit_base',
        img_size: int = 384,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
        embed_dim: int = 768,
        freeze: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        layer_indices: Optional[List[int]] = None
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim
        self.dtype = dtype
        self.layer_indices = layer_indices or [-1] # Default to last layer
        
        # Use Hugging Face V-JEPA 2.1 if hf_model_name provided
        if use_torch_hub:
            self.encoder = self._build_hub_encoder(model_name, dtype)
        elif hf_model_name is not None:
            self.encoder = self._build_hf_encoder(hf_model_name, dtype, attn_implementation)
        elif checkpoint_path is not None and Path(checkpoint_path).exists():
            # Load from local checkpoint
            self.encoder = self._build_encoder()
            self.load_pretrained(checkpoint_path)
        else:
            # Build from scratch (for training from random init)
            self.encoder = self._build_encoder()
        
        # Freeze encoder
        if freeze:
            self._freeze_encoder()
    
    def _build_hub_encoder(self, model_name: str, dtype: torch.dtype):
        """Build V-JEPA 2 or 2.1 encoder from Facebook Research GitHub via torch.hub.
        
        The hub entry points return (encoder, predictor). We keep only the encoder.
        
        V-JEPA 2 hub names: 'vjepa2_vit_base_384', 'vjepa2_vit_large_384'
        V-JEPA 2.1 hub names: 'vjepa2_1_vit_base_384', 'vjepa2_1_vit_large_384'
        """
        import sys as _sys
        
        # Map model_name to hub name - support both V-JEPA 2 and 2.1
        if 'vjepa2_1' in model_name:
            # User explicitly wants V-JEPA 2.1
            hub_name = model_name if 'vjepa2_1' in model_name else 'vjepa2_1_vit_base_384'
            print(f"Loading V-JEPA 2.1 from torch.hub: facebookresearch/vjepa2:{hub_name}")
        elif 'vjepa2' in model_name and 'vjepa2_1' not in model_name:
            # User explicitly wants V-JEPA 2
            hub_name = model_name if 'vjepa2' in model_name else 'vjepa2_vit_base_384'
            print(f"Loading V-JEPA 2 from torch.hub: facebookresearch/vjepa2:{hub_name}")
        else:
            # Default to V-JEPA 2 (not 2.1) for endoscopy models
            hub_name = 'vjepa2_vit_base_384' if 'base' in model_name else 'vjepa2_vit_large_384'
            print(f"Loading V-JEPA 2 from torch.hub: facebookresearch/vjepa2:{hub_name}")

        # Temporarily remove project-root from sys.path to avoid namespace
        # collision: the hub repo has its own 'app' package that would be
        # shadowed by a local 'app' or 'core_app' directory on the path.
        saved_path = list(_sys.path)
        _sys.path = [p for p in _sys.path if 'surgi_world_track' not in p.replace('\\', '/')]
        # Also purge any stale 'app' module entries
        for key in list(_sys.modules.keys()):
            if key == 'app' or key.startswith('app.'):
                del _sys.modules[key]

        try:
            encoder, _predictor = torch.hub.load('facebookresearch/vjepa2', hub_name)
            encoder = encoder.to(dtype)
        finally:
            # Restore original sys.path
            _sys.path = saved_path

        return encoder
    
    def _build_hf_encoder(self, model_name: str, dtype: torch.dtype, attn_impl: str):
        """Build V-JEPA 2/2.1 encoder from Hugging Face."""
        try:
            from transformers import VJEPA2Model, VJEPA2Config
            
            print(f"Loading V-JEPA from Hugging Face: {model_name}")
            
            model = VJEPA2Model.from_pretrained(
                model_name,
                torch_dtype=dtype,
                attn_implementation=attn_impl
            )
            
            print(f"Loaded V-JEPA: {model_name}")
            return model
            
        except ImportError:
            raise ImportError(
                "transformers library required for Hugging Face models. "
                "Install with: pip install transformers"
            )
        except Exception as e:
            print(f"Failed to load from Hugging Face: {e}")
            print("Falling back to local implementation...")
            return self._build_encoder()
    
    def _build_encoder(self):
        """Build local V-JEPA ViT encoder (fallback)."""
        from .vision_transformer import VisionTransformer
        
        return VisionTransformer(
            img_size=self.img_size,
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
            embed_dim=self.embed_dim,
            depth=12,
            num_heads=12,
            use_rope=True
        )
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained V-JEPA weights from local checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'encoder' in checkpoint:
            state_dict = checkpoint['encoder']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        if state_dict is None:
            print(f"Warning: encoder weights in {checkpoint_path} are None. Skipping.")
            return

        # Strip 'backbone.' prefix if present (common in V-JEPA checkpoints)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('backbone.', '')
            new_state_dict[new_k] = v
        
        # Load with strict=False to handle potential mismatches
        self.encoder.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded V-JEPA encoder from {checkpoint_path}")
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        print("V-JEPA encoder frozen")
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video using V-JEPA 2 or 2.1.
        
        Args:
            video: (B, C, T, H, W) video tensor
        
        Returns:
            (B, T', N, C) if single layer, or List[(B, T', N, C)] if multiple layers
            T' = num_frames // tubelet_size
            N = (img_size // patch_size) ** 2
        """
        T_tokens = self.num_frames // self.tubelet_size
        N_tokens = (self.img_size // self.patch_size) ** 2

        with torch.no_grad() if not any(p.requires_grad for p in self.encoder.parameters()) else torch.enable_grad():
            # Handle Hugging Face model format: expects (B, T, C, H, W)
            if hasattr(self.encoder, 'config') and hasattr(self.encoder.config, 'model_type'):
                # Hugging Face model - input is (B, T, C, H, W)
                if video.dim() == 5 and video.shape[1] == 3:
                    # Convert (B, C, T, H, W) to (B, T, C, H, W)
                    video = video.permute(0, 2, 1, 3, 4)
                
                # Request hidden states if multiple layers are needed
                output_hidden_states = len(self.layer_indices) > 1 or self.layer_indices[0] != -1
                
                outputs = self.encoder(
                    pixel_values_videos=video, 
                    skip_predictor=True,
                    output_hidden_states=output_hidden_states
                )
                
                if output_hidden_states:
                    all_hidden_states = outputs.hidden_states # Tuple of (embed, layer1, ..., layerN)
                    selected_features = []
                    for idx in self.layer_indices:
                        if idx == -1:
                            feat = outputs.last_hidden_state
                        else:
                            # HF hidden_states[0] is embedding, [1] is layer 1
                            feat = all_hidden_states[idx + 1] if idx >= 0 else all_hidden_states[idx]
                        
                        # Reshape to (B, T, N, C)
                        if feat.dim() == 4: # (B, T, H_W, C)
                            B, T, HW, C = feat.shape
                            feat = feat.reshape(B, T, HW, C)
                        elif feat.dim() == 3: # (B, TN, C)
                            B, TN, C = feat.shape
                            feat = feat.reshape(B, T_tokens, N_tokens, C)
                        selected_features.append(feat)
                    
                    return selected_features if len(selected_features) > 1 else selected_features[0]
                else:
                    features = outputs.last_hidden_state
                    if features.dim() == 4:
                        B, T, HW, C = features.shape
                        features = features.reshape(B, T, HW, C)
                    elif features.dim() == 3:
                        B, TN, C = features.shape
                        features = features.reshape(B, T_tokens, N_tokens, C)
                    return features
            else:
                # Official/Hub V-JEPA 2/2.1 VisionTransformer - expects (B, C, T, H, W)
                if video.dim() == 5 and video.shape[2] == 3:
                    # Convert (B, T, C, H, W) to (B, C, T, H, W)
                    video = video.permute(0, 2, 1, 3, 4)

                # Use CUDA with cuDNN enabled
                with torch.no_grad():
                    enc_dtype = self.encoder.dtype if hasattr(self.encoder, 'dtype') else torch.bfloat16
                    with torch.autocast(video.device.type, dtype=enc_dtype):
                        final_output = self.encoder(video)
                    if isinstance(final_output, tuple):
                        final_output = final_output[0]
                outputs = [final_output]

                processed_feats = []
                for feat in outputs:
                    # Handle tuple from hook output
                    if isinstance(feat, tuple):
                        feat = feat[0]
                    if feat.dim() == 5:  # (B, C, T, H, W)
                        B, C, T, H, W = feat.shape
                        feat = feat.permute(0, 2, 3, 4, 1) # (B, T, H, W, C)
                        feat = feat.reshape(B, T, H * W, C)
                    elif feat.dim() == 4:  # (B, T, H_W, C)
                        B, T, HW, C = feat.shape
                        # Already in good shape, but let's be explicit
                        feat = feat.reshape(B, T, HW, C)
                    elif feat.dim() == 3:  # (B, TN, C)
                        B, TN, C = feat.shape
                        feat = feat.reshape(B, T_tokens, N_tokens, C)
                    else:
                        raise RuntimeError(f"Unexpected feature dimension: {feat.dim()}, shape: {feat.shape}. Expected 3, 4, or 5 dims.")

                    processed_feats.append(feat)

                if len(processed_feats) == 0:
                    raise RuntimeError("No features were processed. Check encoder outputs and layer_indices configuration.")

                return processed_feats if len(processed_feats) > 1 else processed_feats[0]


class HierarchicalTransformerFusion(nn.Module):
    """
    Fuses multiple intermediate layers from V-JEPA encoder using a transformer neck.
    This helps recover dense spatial details for tracking.
    """
    def __init__(
        self,
        embed_dim: int,
        num_layers_to_fuse: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Learnable layer embeddings to distinguish between different layers
        self.layer_embed = nn.Parameter(torch.randn(1, num_layers_to_fuse, 1, embed_dim) * 0.02)
        
        # Small transformer to fuse layers across the same spatial/temporal position
        # We treat layers as a new "dimension" to attend over
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Final projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x_list: List of (B, T, N, C) tensors from different encoder layers
        Returns:
            (B, T, N, C) fused features
        """
        # x_list elements are (B, T, N, C)
        B, T, N, C = x_list[0].shape
        num_layers = len(x_list)
        
        # Stack layers: (B, num_layers, T, N, C)
        x = torch.stack(x_list, dim=1)
        
        # Add layer embeddings - dynamically adjust to actual number of layers
        # self.layer_embed is (1, num_layers_to_fuse, 1, embed_dim)
        layer_embed = self.layer_embed[:, :num_layers, :, :]
        x = x + layer_embed.unsqueeze(2)
        
        # Reshape for transformer: (B*T*N, num_layers, C)
        # We attend across layers for each spatiotemporal token position
        x = x.permute(0, 2, 3, 1, 4).reshape(B * T * N, num_layers, C)
        
        # Transformer fusion
        x = self.fusion_transformer(x)
        
        # Pool across layers
        x = x.mean(dim=1) # (B*T*N, C)
        
        # Reshape back: (B, T, N, C)
        
        x = x.reshape(B, T, N, C)
        
        return self.norm(self.output_proj(x))


class Dinov2EncoderWrapper(nn.Module):
    """
    Wrapper for frozen DINOv2 encoder from torch.hub.
    Extracts high-quality spatial features for surgical tool tracking and world modeling.
    """
    
    def __init__(
        self,
        model_name: str = 'dinov2_vitb14',
        img_size: int = 392, # DINOv2 usually uses multiples of 14
        freeze: bool = True,
        layer_indices: Optional[List[int]] = None,
        lora: Optional[Dict[str, Any]] = None,
        encoder_checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.layer_indices = layer_indices or [-1]
        self.lora_config = lora or {}

        print(f"Loading DINOv2 from torch.hub: facebookresearch/dinov2:{model_name}")
        self.encoder = torch.hub.load('facebookresearch/dinov2', model_name)

        # Load TDV-pretrained or custom encoder checkpoint
        if encoder_checkpoint is not None and Path(encoder_checkpoint).exists():
            print(f"Loading encoder checkpoint: {encoder_checkpoint}")
            ckpt = torch.load(encoder_checkpoint, map_location='cpu', weights_only=True)
            # The checkpoint may be a raw state dict or a dict with 'model_state_dict'
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                # TDV checkpoint — extract frame encoder weights
                sd = ckpt['model_state_dict']
                # Keys may be prefixed with 'frame_encoder.encoder.'
                encoder_sd = {}
                for k, v in sd.items():
                    if k.startswith('frame_encoder.encoder.'):
                        encoder_sd[k.replace('frame_encoder.encoder.', '')] = v
                if encoder_sd:
                    msg = self.encoder.load_state_dict(encoder_sd, strict=False)
                    print(f"  Loaded TDV frame encoder: {len(encoder_sd)} keys")
                else:
                    # Try loading as raw encoder state dict
                    msg = self.encoder.load_state_dict(sd, strict=False)
                    print(f"  Loaded encoder checkpoint: {len(sd)} keys")
            else:
                # Raw state dict
                msg = self.encoder.load_state_dict(ckpt, strict=False)
                print(f"  Loaded raw encoder checkpoint: {len(ckpt)} keys")
            if msg.missing_keys:
                print(f"  Missing keys: {len(msg.missing_keys)}")
            if msg.unexpected_keys:
                print(f"  Unexpected keys: {len(msg.unexpected_keys)}")

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            print("DINOv2 encoder frozen")

        # A1: Inject LoRA adapters while keeping base weights frozen.
        if self.lora_config.get('enable', False):
            from .lora import inject_lora_into_dinov2, get_lora_params
            inject_lora_into_dinov2(
                self.encoder,
                r=self.lora_config.get('rank', 8),
                alpha=self.lora_config.get('alpha', 16),
                dropout=self.lora_config.get('dropout', 0.05),
                target_modules=self.lora_config.get('target_modules', ['qkv', 'proj', 'fc1', 'fc2']),
                start_block=self.lora_config.get('start_block', 0),
                end_block=self.lora_config.get('end_block', None),
            )
            # Keep encoder in train mode so LoRA path is active during training,
            # but base layers remain frozen via LoRALinear wrappers.
            self.encoder.train()
            lora_params = len(get_lora_params(self.encoder))
            print(f"DINOv2 encoder ready with {lora_params} LoRA parameters trainable")

        # DINOv2 ViT-B/14 has 768 dim, patch size 14
        self.embed_dim = self.encoder.embed_dim
        self.patch_size = self.encoder.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) single frame or (B, C, T, H, W) video
        Returns:
            (B, T, N, C) where N is number of patches
        """
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            # Reshape to process all frames as a batch: (B*T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        else:
            B, C, H, W = x.shape
            T = 1

        # DINOv2 expects image size to be multiple of patch_size (14)
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            H_new = (H // self.patch_size) * self.patch_size
            W_new = (W // self.patch_size) * self.patch_size
            x = torch.nn.functional.interpolate(x, size=(H_new, W_new), mode='bicubic', align_corners=False)

        # Extract features
        # Note: DINOv2 forward_features returns dict with 'x_norm_patchtokens', etc.
        outputs = self.encoder.forward_features(x)
        features = outputs['x_norm_patchtokens'] # (B*T, N, C)
        
        # Reshape back to include temporal dimension
        _, N, C = features.shape
        features = features.reshape(B, T, N, C)
        
        return features


class WorldModel(nn.Module):
    """
    The core World Model:
    - Frozen Vision Encoder (DINOv2 or V-JEPA) extracts reality features.
    - Hierarchical fusion neck (optional) for multi-layer encoder features.
    - EncoderNeck: encoder-aware FPN or VJEPANeck for detection/RoIAlign features.
    - Transformer Predictor "imagines" future states (phantom features).
    """

    def __init__(
        self,
        encoder_type: str = 'dinov2',
        encoder_checkpoint: Optional[str] = None,
        model_name: str = 'dinov2_vitb14',
        encoder_dim: int = 768,
        neck_dim: int = 256,
        num_frames: int = 16,
        img_size: int = 392,
        layer_indices: List[int] = [-1],
        predictor_hidden_dim: int = 768,
        predictor_num_layers: int = 8,
        predictor_num_heads: int = 12,
        prediction_horizons: List[int] = [1, 4, 16],
        horizon_weights: Optional[List[float]] = None,
        feature_weight: float = 1.0,
        rollout_weight: float = 0.3,
        temporal_consistency_weight: float = 0.0,
        vjepa_temporal_reduction: str = 'mean',
        vjepa_multi_scale: bool = False,
        use_torch_hub: bool = True,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.encoder_dim = encoder_dim
        self.neck_dim = neck_dim
        self.num_frames = num_frames
        self.prediction_horizons = prediction_horizons
        self.layer_indices = layer_indices

        # 1. Vision Encoder (Frozen)
        if encoder_type == 'dinov2':
            self.encoder = Dinov2EncoderWrapper(
                model_name=model_name,
                img_size=img_size,
                freeze=True,
                layer_indices=layer_indices,
                encoder_checkpoint=encoder_checkpoint,
            )
            _neck_type = 'dinov2'
        else:
            self.encoder = VJEPAEncoderWrapper(
                checkpoint_path=encoder_checkpoint,
                hf_model_name=model_name if not use_torch_hub and 'facebook/' in model_name else None,
                use_torch_hub=use_torch_hub,
                model_name=model_name,
                embed_dim=encoder_dim,
                num_frames=num_frames,
                img_size=img_size,
                freeze=True,
                layer_indices=layer_indices
            )
            _neck_type = 'vjepa'

        # 2. Optional hierarchical fusion neck (multi-layer encoders)
        if len(layer_indices) > 1:
            self.fusion_neck = HierarchicalTransformerFusion(
                embed_dim=encoder_dim,
                num_layers_to_fuse=len(layer_indices)
            )
        else:
            self.fusion_neck = None

        # 3. Encoder-Aware Feature Neck (SimpleFPN for DINOv2, VJEPANeck for VJEPA)
        self.encoder_neck = EncoderNeck(
            encoder_type=_neck_type,
            neck_dim=neck_dim,
            override_embed_dim=encoder_dim,
            vjepa_temporal_reduction=vjepa_temporal_reduction,
            vjepa_multi_scale=vjepa_multi_scale,
        )

        # 4. Transformer Predictor (The "Imagination" Engine)
        self.predictor = MultiScaleTemporalPredictor(
            input_dim=encoder_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_num_layers,
            num_heads=predictor_num_heads,
            prediction_horizons=prediction_horizons,
            dropout=0.1,
            max_seq_len=16384
        )

        # 5. World Model Loss (MSE + rollout)
        self.loss_fn = WorldModelLoss(
            prediction_horizons=prediction_horizons,
            horizon_weights=horizon_weights,
            feature_weight=feature_weight,
            rollout_weight=rollout_weight,
            temporal_consistency_weight=temporal_consistency_weight,
        )

    def _get_flat_features(self, feats) -> torch.Tensor:
        """
        Normalise encoder output to (B, T, N, C) regardless of encoder type.
        DINOv2 returns (B, T, N, C); V-JEPA also returns (B, T, N, C).
        """
        if isinstance(feats, list) and self.fusion_neck is not None:
            return self.fusion_neck(feats)  # (B, T, N, C)
        if isinstance(feats, list):
            return feats[-1]  # last layer
        return feats  # (B, T, N, C)

    def encode(self, video: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Encode a video clip.

        Returns:
            reality_seq: (B, T, N, C) flat spatiotemporal tokens
            neck_out: dict from EncoderNeck containing spatial_map, detection_scales, etc.
        """
        feats = self.encoder(video)  # (B, T, N, C) or List
        reality_seq = self._get_flat_features(feats)  # (B, T, N, C)

        # For the neck, collapse temporal dim to get per-frame spatial features
        # Use the last (most recent) frame for detection
        # reality_seq: (B, T, N, C) -> last frame: (B, N, C)
        last_frame_feat = reality_seq[:, -1]  # (B, N, C)

        # For V-JEPA the neck receives (B, T, N, C) directly and collapses internally
        if self.encoder_type == 'dinov2':
            neck_out = self.encoder_neck(last_frame_feat)  # (B, N, C) -> FPN
        else:
            neck_out = self.encoder_neck(reality_seq)  # (B, T, N, C) -> VJEPANeck

        return reality_seq, neck_out

    def imagine(self, reality_seq: torch.Tensor) -> dict:
        """
        Run the predictor over reality_seq to produce phantom / future features.

        Returns:
            predictions: dict[f'horizon_{h}'] = (B, N, C) predicted features
        """
        return self.predictor(reality_seq, return_all_steps=False)

    def forward(self, video: torch.Tensor, mode: str = 'train') -> dict:
        """Full encode + imagine pass."""
        reality_seq, neck_out = self.encode(video)
        predictions = self.imagine(reality_seq)
        return {
            'reality_seq': reality_seq,
            'neck_out': neck_out,
            'predictions': predictions,
        }


class SurgicalTrackingSystem(nn.Module):
    """
    Main System integrating the World Model with downstream Tracking Heads.

    Key changes vs. original:
      - EncoderNeck (SimpleFPN / VJEPANeck) produces detection feature pyramid.
      - DETR operates on neck features (neck_dim) instead of raw encoder_dim.
      - Phantom features from the predictor are used in joint ReID (SupCon).
      - RoIAlignExtractor extracts per-tool embeddings from real + phantom maps.
    """

    def __init__(
        self,
        # World Model Config
        encoder_type: str = 'dinov2',
        encoder_checkpoint: Optional[str] = None,
        model_name: str = 'dinov2_vitb14',
        encoder_dim: int = 768,
        neck_dim: int = 256,
        num_frames: int = 16,
        img_size: int = 392,
        layer_indices: List[int] = [-1],
        predictor_hidden_dim: int = 768,
        predictor_num_layers: int = 8,
        predictor_num_heads: int = 12,
        prediction_horizons: List[int] = [1, 4, 16],
        vjepa_temporal_reduction: str = 'mean',
        vjepa_multi_scale: bool = False,
        use_torch_hub: bool = True,
        # Tracking Heads Config
        num_tools: int = 7,
        num_queries: int = 16,
        num_decoder_layers: int = 5,
        detr_nheads: int = 8,
        detr_dropout: float = 0.15,
        detr_dim_feedforward: int = 2048,
        detr_class_weight: float = 1.0,
        detr_bbox_weight: float = 5.0,
        detr_giou_weight: float = 2.0,
        detr_focal_alpha: float = 0.25,
        detr_focal_gamma: float = 2.0,
        detr_max_pos_tokens: Optional[int] = None,
        reid_input_dim: Optional[int] = None,  # If None, use neck_dim
        reid_embedding_dim: int = 256,
        reid_normalize: bool = True,
        reid_dropout: float = 0.15,
        reid_supcon_temperature: float = 0.07,
        reid_supcon_weight: float = 1.0,
        reid_cross_consistency_weight: float = 0.1,
        roi_output_size: int = 7,
        # Loss config
        horizon_weights: Optional[List[float]] = None,
        feature_weight: float = 1.0,
        rollout_weight: float = 0.3,
        temporal_consistency_weight: float = 0.0,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.neck_dim = neck_dim

        # 1. The World Model (Encoder + EncoderNeck + Predictor)
        self.world_model = WorldModel(
            encoder_type=encoder_type,
            encoder_checkpoint=encoder_checkpoint,
            model_name=model_name,
            encoder_dim=encoder_dim,
            neck_dim=neck_dim,
            num_frames=num_frames,
            img_size=img_size,
            layer_indices=layer_indices,
            predictor_hidden_dim=predictor_hidden_dim,
            predictor_num_layers=predictor_num_layers,
            predictor_num_heads=predictor_num_heads,
            prediction_horizons=prediction_horizons,
            horizon_weights=horizon_weights,
            feature_weight=feature_weight,
            rollout_weight=rollout_weight,
            temporal_consistency_weight=temporal_consistency_weight,
            vjepa_temporal_reduction=vjepa_temporal_reduction,
            vjepa_multi_scale=vjepa_multi_scale,
            use_torch_hub=use_torch_hub,
        )

        # 2. DETR Detection Head — operates on neck_dim tokens
        if detr_max_pos_tokens is None:
            # DINOv2: sum of 4 FPN levels (56^2 + 28^2 + 14^2 + 7^2 = 3969)
            # V-JEPA: single level 24^2=576 (or +12^2=720 if multi_scale)
            detr_max_pos_tokens = 4096 if encoder_type == 'dinov2' else 1024

        self.detr = SurgicalToolDetector(
            encoder_dim=encoder_dim,   # DETR decoder hidden dim
            num_tools=num_tools,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            nheads=detr_nheads,
            dropout=detr_dropout,
            dim_feedforward=detr_dim_feedforward,
            num_pos_tokens=detr_max_pos_tokens,
            class_weight=detr_class_weight,
            bbox_weight=detr_bbox_weight,
            giou_weight=detr_giou_weight,
            focal_alpha=detr_focal_alpha,
            focal_gamma=detr_focal_gamma,
            neck_dim=neck_dim,
        )

        # 3. RoIAlign Extractor — per-tool features from real + phantom maps
        self.roi_extractor = RoIAlignExtractor(
            output_size=roi_output_size,
            spatial_scale=1.0,   # feature map is in [0,1] normalised space
            sampling_ratio=2,
            aligned=True,
        )

        # 4. ReID Head — SupCon + RoIAlign-based, pooling='none'
        reid_in_dim = reid_input_dim if reid_input_dim is not None else neck_dim
        self.reid = ReidHead(
            input_dim=reid_in_dim,   # RoIAlign extracts features (neck_dim or encoder_dim)
            embedding_dim=reid_embedding_dim,
            num_classes=num_tools,
            pooling='none',           # already pooled by RoIAlign
            normalize_embeddings=reid_normalize,
            dropout=reid_dropout,
            ce_loss_weight=0.0,       # SupCon replaces CE
            supcon_temperature=reid_supcon_temperature,
            supcon_weight=reid_supcon_weight,
            cross_consistency_weight=reid_cross_consistency_weight,
        )

        # Phantom → neck projection: maps encoder_dim back to neck_dim spatial map
        # The predictor outputs (B, N, C) in encoder_dim; we need (B, neck_dim, H, W)
        # Use the world_model's encoder_neck for this (shared weights).
        # A small adapter handles the dim difference if needed.
        _enc_neck = self.world_model.encoder_neck
        _enc_spatial_h = _enc_neck.neck.spatial_h
        _enc_spatial_w = _enc_neck.neck.spatial_w
        self._phantom_spatial_h = _enc_spatial_h
        self._phantom_spatial_w = _enc_spatial_w

        self._num_queries = num_queries
        self.register_buffer('reid_weight', torch.tensor(0.2))
        self.register_buffer('epoch', torch.tensor(0))

    def forward(
        self,
        current_video: torch.Tensor,
        future_videos: Optional[Dict[int, torch.Tensor]] = None,
        detr_targets: Optional[List[dict]] = None,
        reid_labels: Optional[torch.Tensor] = None,
        mode: str = 'train'
    ) -> dict:
        device_type = current_video.device.type
        # Determine dtype for autocast
        _first_param = next(self.parameters(), None)
        model_dtype = _first_param.dtype if _first_param is not None else torch.float32
        
        with torch.no_grad() if mode != 'train' else torch.enable_grad():
            # Use float32 for CPU, model_dtype for CUDA
            effective_dtype = model_dtype if device_type == 'cuda' else torch.float32
            with torch.autocast(device_type, dtype=effective_dtype):
                return self._forward_impl(
                    current_video, 
                    future_videos=future_videos, 
                    detr_targets=detr_targets, 
                    reid_labels=reid_labels, 
                    mode=mode
                )

    def _build_phantom_spatial_map(self, phantom_feat: torch.Tensor) -> torch.Tensor:
        """
        Reshape flat phantom tokens to spatial map for DETR.

        Args:
            phantom_feat: (B, N, C) or (B*T, N, C) predicted tokens

        Returns:
            (B, neck_dim, H, W) spatial feature map
        """
        neck = self.world_model.encoder_neck.neck

        B, N, C = phantom_feat.shape
        # Dynamically compute spatial dimensions from N
        H = int(N ** 0.5)
        W = N // H
        if H * W != N:
            # Fallback to configured dimensions
            H, W = self._phantom_spatial_h, self._phantom_spatial_w

        # Reshape to (B, C, H, W)
        x = phantom_feat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        if hasattr(neck, 'lateral'):
            # SimpleFPN path
            return neck.lateral(x)  # (B, neck_dim, H, W)
        else:
            # VJEPANeck path
            return neck.proj[0](x)  # first Conv2d in proj Sequential

    def _forward_impl(
        self,
        current_video: torch.Tensor,
        future_videos: Optional[Dict[int, torch.Tensor]] = None,
        detr_targets: Optional[List[dict]] = None,
        reid_labels: Optional[torch.Tensor] = None,
        mode: str = 'train',
    ) -> dict:
        B = current_video.size(0)

        # ------------------------------------------------------------------ #
        # 1. Encode current frame(s) → reality features + neck output
        # ------------------------------------------------------------------ #
        reality_seq, neck_out = self.world_model.encode(current_video)
        # reality_seq: (B, T, N, C)
        # neck_out: {'P3', ..., 'spatial_map', 'flat', 'detection_scales'}

        # ------------------------------------------------------------------ #
        # 2. Predictor → phantom features (one step ahead = horizon_1)
        # ------------------------------------------------------------------ #
        predictions = self.world_model.imagine(reality_seq)
        # predictions[f'horizon_{h}']: (B, N, C)

        # Pick horizon-1 phantom for ReID (most relevant to current frame)
        phantom_flat = predictions.get('horizon_1')  # (B, N, C) or None

        # ------------------------------------------------------------------ #
        # 3. DETR Detection — operates on flattened neck features
        # ------------------------------------------------------------------ #
        # EncoderNeck.get_flat_for_detr concatenates all scale tokens
        detr_tokens = self.world_model.encoder_neck.get_flat_for_detr(neck_out)
        # (B, Σ H_l*W_l, neck_dim)
        
        # Tri-Attention needs phantom features in flat format
        detr_outputs = self.detr(
            encoder_features=detr_tokens,
            phantom_features=phantom_flat,
            targets=detr_targets
        )
        # detr_outputs['pred']['pred_boxes']: (B, Q, 4) in cxcywh

        # ------------------------------------------------------------------ #
        # 4. RoIAlign → per-tool features from real spatial map
        # ------------------------------------------------------------------ #
        # Use P3 (main scale) for RoIAlign — shape (B, neck_dim, H, W)
        real_spatial_map = neck_out.get('P3')  # (B, neck_dim, H, W)

        pred_boxes = detr_outputs['pred'].get('pred_boxes')  # (B, Q, 4) or None

        roi_real: Optional[torch.Tensor] = None
        roi_phantom: Optional[torch.Tensor] = None

        if real_spatial_map is not None and pred_boxes is not None:
            # Real map RoIAlign → (B*Q, neck_dim)
            roi_real = self.roi_extractor(real_spatial_map, pred_boxes)

            # Phantom map RoIAlign (if phantom available)
            if phantom_flat is not None:
                phantom_map = self._build_phantom_spatial_map(phantom_flat)
                # Interpolate to match real_spatial_map spatial resolution
                if phantom_map.shape[-2:] != real_spatial_map.shape[-2:]:
                    phantom_map = torch.nn.functional.interpolate(
                        phantom_map,
                        size=real_spatial_map.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )
                roi_phantom = self.roi_extractor(phantom_map, pred_boxes)
                # roi_phantom: (B*Q, neck_dim)

        # ------------------------------------------------------------------ #
        # 5. ReID — SupCon over real [+ phantom] per-tool features
        # ------------------------------------------------------------------ #
        reid_labels_expanded: Optional[torch.Tensor] = None
        detection_scores: Optional[torch.Tensor] = None

        if reid_labels is not None and roi_real is not None:
            B_detr, Q = detr_outputs['pred']['class_logits'].shape[:2]
            if isinstance(reid_labels, list):
                # Stack if all same length, else handle per-sample
                if all(isinstance(r, torch.Tensor) for r in reid_labels):
                    reid_labels_tensor = torch.cat([r.flatten() for r in reid_labels])
                else:
                    reid_labels_tensor = torch.tensor(reid_labels, device=roi_real.device)
            else:
                reid_labels_tensor = reid_labels
            
            # Ensure correct shape (B,) for track IDs
            if reid_labels_tensor.dim() > 1:
                reid_labels_tensor = reid_labels_tensor.flatten()[:roi_real.size(0)]
            
            # Only expand if we have labels, otherwise skip
            if reid_labels_tensor.numel() > 0:
                # Ensure tensor has B_detr elements (one per batch)
                if reid_labels_tensor.numel() != B_detr:
                    # Pad or truncate to match batch size
                    if reid_labels_tensor.numel() < B_detr:
                        # Pad with zeros (background class) to match batch size
                        padding = torch.zeros(B_detr - reid_labels_tensor.numel(), 
                                            dtype=reid_labels_tensor.dtype, 
                                            device=reid_labels_tensor.device)
                        reid_labels_tensor = torch.cat([reid_labels_tensor, padding])
                    else:
                        # Truncate to batch size
                        reid_labels_tensor = reid_labels_tensor[:B_detr]
                reid_labels_expanded = (
                    reid_labels_tensor.unsqueeze(1)
                    .expand(B_detr, Q)
                    .reshape(B_detr * Q)
                )
            else:
                reid_labels_expanded = None
            # Confidence weights for DFL: sigmoid of max-class logit
            detection_scores = (
                detr_outputs['pred']['class_logits']
                .sigmoid()
                .max(dim=-1)
                .values
                .reshape(-1)
                .detach()
            )

        # Determine ReID input: RoIAlign features when boxes available, else pooled flat tokens
        if roi_real is not None:
            reid_input = roi_real  # (B*Q, neck_dim)
        else:
            # Fallback: pool flat encoder tokens per batch element
            flat = neck_out['flat']  # (B, N, C)
            reid_input = flat.mean(dim=1)  # (B, C) — one embedding per batch elem

        reid_outputs = self.reid(
            reid_input,
            labels=reid_labels_expanded,
            phantom_features=roi_phantom,
            detection_scores=detection_scores,
        )

        # Reshape embeddings back to (B, Q, D)
        B_detr = B
        Q = detr_outputs['pred']['class_logits'].shape[1]
        embeddings_flat = reid_outputs['embeddings']  # (B*Q, D)
        embeddings = embeddings_flat.reshape(B_detr, Q, -1)

        result = {
            'detr': detr_outputs,
            'reid': {
                'embeddings': embeddings,
                'loss': reid_outputs.get('loss'),
                'loss_dict': reid_outputs.get('loss_dict'),
            },
            'predictions': predictions,
            'neck_out': neck_out,
            'reality_seq': reality_seq,
        }

        # ------------------------------------------------------------------ #
        # 6. World Model Loss (MSE + rollout, self-supervised)
        # ------------------------------------------------------------------ #
        if future_videos is not None and mode == 'train':
            wm_targets: Dict[str, torch.Tensor] = {}
            for h in self.world_model.prediction_horizons:
                if h in future_videos:
                    with torch.no_grad():
                        fut_seq, _ = self.world_model.encode(future_videos[h])
                        # Use last frame features as prediction target
                        wm_targets[f'horizon_{h}'] = fut_seq[:, -1]  # (B, N, C)

            wm_loss, wm_loss_dict = self.world_model.loss_fn(predictions, wm_targets)
            result['world_model_loss'] = wm_loss
            result['world_model_loss_dict'] = wm_loss_dict

        # ------------------------------------------------------------------ #
        # 7. Combine all losses
        # ------------------------------------------------------------------ #
        if mode == 'train':
            total_loss = torch.zeros(1, device=current_video.device).squeeze()
            loss_dict: Dict[str, float] = {}

            if 'loss' in detr_outputs:
                total_loss = total_loss + detr_outputs['loss']
                loss_dict.update(detr_outputs.get('loss_dict', {}))

            if reid_outputs.get('loss') is not None:
                reid_w = self.reid_weight.item()
                total_loss = total_loss + reid_w * reid_outputs['loss']
                for k, v in reid_outputs.get('loss_dict', {}).items():
                    loss_dict[k] = v * reid_w

            if 'world_model_loss' in result:
                total_loss = total_loss + result['world_model_loss']
                for k, v in result['world_model_loss_dict'].items():
                    loss_dict[f'wm_{k}'] = v

            result['total_loss'] = total_loss
            result['loss_dict'] = loss_dict

        return result

    def update_curriculum(self, epoch: int, max_epochs: int):
        progress = epoch / max_epochs
        new_weight = 0.2 + 0.8 * progress
        self.reid_weight.fill_(new_weight)
        self.epoch.fill_(epoch)

    def predict_future(
        self,
        current_video: torch.Tensor,
        horizon: int
    ) -> dict:
        """
        Inference mode: predict future frame features and detect tools.
        """
        with torch.no_grad():
            return self.forward(
                current_video,
                future_videos=None,
                detr_targets=None,
                reid_labels=None,
                mode='predict'
            )

# Backward compatibility aliases
VJEPAWorldModel = SurgicalTrackingSystem
SurgicalWorldModel = SurgicalTrackingSystem


class VisionTransformer(nn.Module):
    """
    Simple ViT implementation for V-JEPA encoder placeholder.
    In practice, this would be replaced with the official V-JEPA encoder.
    """
    
    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        use_rope: bool = True
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )
        
        # Calculate number of patches
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        self.num_temporal_tokens = num_frames // tubelet_size
        self.num_tokens = self.num_patches_per_frame * self.num_temporal_tokens
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim) * 0.02)
        
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, C, T, H, W)
        
        Returns:
            (B, N, C) features
        """
        B, C, T, H, W = video.shape
        
        # Patch embedding
        x = self.patch_embed(video)  # (B, C, T', H', W')
        
        # Flatten to tokens
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        return x


def load_vjepa_world_model_fixed(checkpoint_path: str, use_vjepa21=False, extract_encoder_only=False):
    """
    Load SurgicalTrackingSystem from checkpoint with automatic dimension detection.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pth.tar or .pt)
        use_vjepa21: If True, use V-JEPA 2.1; else use V-JEPA 2
        extract_encoder_only: If True, skip DETR and ReID heads, return encoder+predictor only
    
    Returns:
        SurgicalTrackingSystem model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract state dicts
    encoder_sd = checkpoint.get('encoder', {})
    detr_sd = checkpoint.get('detr_head', checkpoint.get('detr', {}))
    reid_sd = checkpoint.get('reid_head', checkpoint.get('reid', {}))
    
    # Detect dimensions from checkpoint
    encoder_dim = 768  # default for ViT-B
    if detr_sd and 'query_embed.weight' in detr_sd:
        encoder_dim = detr_sd['query_embed.weight'].shape[1]
    elif encoder_sd:
        # Try to infer from patch embed or first layer
        for key in encoder_sd.keys():
            if 'patch_embed' in key and 'weight' in key:
                encoder_dim = encoder_sd[key].shape[0]
                break
    
    # reid_input_dim should match neck_dim (256) since RoIAlign extracts from neck features
    # The checkpoint's reid_head.embedding expects the neck output dimension, not encoder_dim
    reid_input_dim = 256  # neck_dim default
    
    # When extracting encoder only, set num_tools=0 to disable DETR and ReID heads
    num_tools = 0 if extract_encoder_only else 7
    
    # Determine model name based on encoder_dim
    if encoder_dim == 1024:
        model_name = 'vjepa2_1_vit_large_384' if use_vjepa21 else 'vjepa2_vit_large_384'
    else:
        model_name = 'vjepa2_1_vit_base_384' if use_vjepa21 else 'vjepa2_vit_base_384'
    
    print(f"Creating SurgicalTrackingSystem:")
    print(f"  encoder_dim={encoder_dim}, reid_input_dim={reid_input_dim}")
    print(f"  model_name={model_name}, use_vjepa21={use_vjepa21}")
    print(f"  extract_encoder_only={extract_encoder_only}, num_tools={num_tools}")
    
    # Create model
    model = SurgicalTrackingSystem(
        encoder_type='vjepa',
        model_name=model_name,
        encoder_dim=encoder_dim,
        reid_input_dim=reid_input_dim,
        num_tools=num_tools,
        num_queries=16,
        use_torch_hub=use_vjepa21,
    )
    
    # Build state dict to load
    new_state_dict = {}
    
    # Remap encoder keys: strip 'module.' and 'backbone.' prefixes
    if encoder_sd:
        for k, v in encoder_sd.items():
            new_k = k.replace('module.', '').replace('backbone.', '')
            new_state_dict[f'world_model.encoder.encoder.{new_k}'] = v
    
    # Load predictor if present
    if 'predictor' in checkpoint and checkpoint['predictor']:
        for k, v in checkpoint['predictor'].items():
            new_state_dict[f'world_model.predictor.{k}'] = v
    
    # Only load DETR and ReID heads if not extracting encoder only
    if not extract_encoder_only:
        # Load DETR head
        if detr_sd:
            for k, v in detr_sd.items():
                new_state_dict[f'detr.{k}'] = v
        
        # Load ReID head
        if reid_sd:
            for k, v in reid_sd.items():
                new_state_dict[f'reid.{k}'] = v
    else:
        print("  Skipping DETR and ReID head weights (extract_encoder_only=True)")
    
    # Load weights
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded checkpoint: {len(new_state_dict)} parameters")
    if msg.missing_keys:
        print(f"  Missing: {len(msg.missing_keys)} keys")
    if msg.unexpected_keys:
        print(f"  Unexpected: {len(msg.unexpected_keys)} keys")
    
    return model
