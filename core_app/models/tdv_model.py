"""
Temporal Difference in Vision (TDV) model for surgical video pretraining.

Adapted from the official TDV implementation:
  https://github.com/ninaddaithankar/tdv
  Paper: "You Don't Need Strong Assumptions: Visual Representation Learning
          via Temporal Differences" (Daithankar et al., 2026)

Core idea: frame_encoder(F_t) + motion_encoder(F_{t+1} - F_t) ≈ frame_encoder(F_{t+1})
The frame encoder is a DINOv2 ViT; the motion encoder is a lightweight ViT
that takes RGB difference as input and cross-attends to the frame encoder's
patch tokens. An EMA teacher provides stable targets.

This module is self-contained and does NOT depend on the upstream TDV repo's
PyTorch Lightning infrastructure. It integrates with the Cholec_Vjepa-2 config
system and can be trained with the provided `scripts/pretrain_tdv.py`.
"""
from __future__ import annotations

import copy
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .tdv_losses import CenterSharpReconstructionLoss, DinoLoss, DINOHead


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def set_trainable(module: nn.Module, trainable: bool):
    module.train(trainable)
    for p in module.parameters():
        p.requires_grad = trainable


def update_teacher_using_ema(students, teachers, ema_momentum):
    assert len(students) == len(teachers)
    with torch.no_grad():
        for student, teacher in zip(students, teachers):
            student_params = dict(student.named_parameters())
            teacher_params = dict(teacher.named_parameters())
            assert student_params.keys() == teacher_params.keys()
            for name in teacher_params:
                teacher_params[name].data.mul_(ema_momentum)
                teacher_params[name].data.add_((1.0 - ema_momentum) * student_params[name].data)


def init_dino_head(in_dim, out_dim, layers=3):
    return DINOHead(in_dim=in_dim, out_dim=out_dim, hidden_dim=2048, bottleneck_dim=256, nlayers=layers)


def get_rgb_diff(frame_sequences: torch.Tensor, use_full_frame: bool = False) -> torch.Tensor:
    """Compute RGB difference between consecutive frames.
    frame_sequences: (B, T, C, H, W)
    returns: (B, T-1, C, H, W)
    """
    if use_full_frame:
        return frame_sequences[:, 1:]
    return frame_sequences[:, 1:] - frame_sequences[:, :-1]


def calculate_static_frames_mask(rgb_diff: torch.Tensor, threshold: float = 0.0) -> Optional[torch.Tensor]:
    if threshold <= 0:
        return None
    pixel_diff = torch.abs(rgb_diff)
    mean_rgb_diff = pixel_diff.mean(dim=[1, 2, 3, 4])
    return mean_rgb_diff > threshold


def skip_this_batch(mask: Optional[torch.Tensor]) -> bool:
    if mask is None:
        return False
    local_skip = torch.tensor(
        1 if mask.sum() == 0 else 0,
        device=mask.device, dtype=torch.uint8,
    )
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.all_reduce(local_skip, op=torch.distributed.ReduceOp.MAX)
    return bool(local_skip.item())


def rollout_n_frames(previous, rgb_diff, target, n=1):
    if n <= 1:
        return previous, rgb_diff, target
    assert previous.shape == rgb_diff.shape == target.shape
    B, T = previous.shape[:2]
    csum = rgb_diff.cumsum(dim=1)
    pad = torch.zeros_like(csum[:, :1])
    csum = torch.cat([pad, csum], dim=1)
    n_rgb_diff = csum[:, n:] - csum[:, :-n]
    previous_n = previous[:, :T - n + 1]
    target_n = target[:, n - 1:]
    return previous_n, n_rgb_diff, target_n


# --------------------------------------------------------------------------- #
# DINO-style clip augmentation (spatial params shared across all T frames)
# --------------------------------------------------------------------------- #

class DINOClipAugmentation:
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, crop_scale, image_size, gaussian_blur_p, solarization_p=0.0):
        self.crop_scale = crop_scale
        self.image_size = image_size
        self.gaussian_blur_p = gaussian_blur_p
        self.solarization_p = solarization_p
        self.color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        self.gaussian_blur = T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """clip: [T, C, H, W] float tensor in [0, 1]"""
        clip = TF.resize(clip, [self.image_size, self.image_size],
                         interpolation=TF.InterpolationMode.BICUBIC)
        if torch.rand(1).item() < 0.8:
            clip = self.color_jitter(clip)
        if torch.rand(1).item() < 0.2:
            clip = TF.rgb_to_grayscale(clip, num_output_channels=3)
        if torch.rand(1).item() < self.gaussian_blur_p:
            clip = self.gaussian_blur(clip)
        if torch.rand(1).item() < self.solarization_p:
            clip = TF.solarize(clip, threshold=0.5)
        return clip


def apply_clip_augmentation(frames: torch.Tensor, aug: DINOClipAugmentation) -> torch.Tensor:
    """frames: [B, T, C, H, W] -> [B, T, C, H, W]"""
    return torch.stack([aug(frames[b]) for b in range(frames.shape[0])])


# --------------------------------------------------------------------------- #
# Motion encoder — lightweight ViT with cross-attention to frame tokens
# --------------------------------------------------------------------------- #

class MotionEncoder(nn.Module):
    """Lightweight motion encoder that processes RGB difference frames.

    Takes (B, C, H, W) RGB diff as input, patchifies it, and produces
    token-level motion features via a stack of self-attention + cross-attention
    blocks. The cross-attention condition comes from the frame encoder's
    patch tokens.

    This is a simplified version of TDV's dinoViT_xattn that avoids the
    full DINOv2 fork dependency.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 768,
        condition_dim: int = 768,
        depth: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # CLS + register tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens > 0 else None

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + num_register_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        # Blocks: alternating self-attention and cross-attention
        self.blocks = nn.ModuleList([
            MotionEncoderBlock(
                dim=embed_dim,
                condition_dim=condition_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, rgb_diff: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_diff: (B, C, H, W) — RGB difference between consecutive frames
            condition: (B, N_cond, C_cond) — frame encoder patch tokens (no CLS)

        Returns:
            (B, 1 + N_reg + N_patches, D) — motion-encoded tokens
        """
        B = rgb_diff.shape[0]
        x = self.patch_embed(rgb_diff)  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)
        if self.register_tokens is not None:
            reg = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([cls, reg, x], dim=1)
        else:
            x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x, condition)

        x = self.norm(x)
        return x


class MotionEncoderBlock(nn.Module):
    """One block of the motion encoder: self-attn → cross-attn → MLP."""

    def __init__(self, dim: int, condition_dim: int, num_heads: int = 12, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, kdim=condition_dim, vdim=condition_dim, batch_first=True)

        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, condition):
        # Self-attention
        residual = x
        x = self.norm1(x)
        x_sa, _ = self.self_attn(x, x, x)
        x = residual + x_sa

        # Cross-attention to frame encoder condition
        residual = x
        x = self.norm2(x)
        x_ca, _ = self.cross_attn(x, condition, condition)
        x = residual + x_ca

        # MLP
        residual = x
        x = self.norm3(x)
        x = residual + self.mlp(x)
        return x


# --------------------------------------------------------------------------- #
# Frame encoder wrapper — wraps DINOv2 for TDV's encode_sequences
# --------------------------------------------------------------------------- #

class TDVFrameEncoder(nn.Module):
    """Wraps a DINOv2 ViT to produce (B, N, D) patch tokens (with CLS prepended).

    This is compatible with the existing Dinov2EncoderWrapper but returns
    CLS + patch tokens in the format TDV expects.
    """

    def __init__(
        self,
        model_name: str = 'dinov2_vitb14',
        img_size: int = 224,
        freeze: bool = True,
        pretrained: bool = True,
        encoder_checkpoint: str = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.embed_dim = 768  # default for vitb14

        if pretrained:
            self.encoder = torch.hub.load('facebookresearch/dinov2', model_name)
        else:
            self.encoder = torch.hub.load('facebookresearch/dinov2', model_name)
            # Reset to random init for from-scratch training
            self.encoder.apply(self._init_weights)

        # Override with custom pretrained weights if provided
        if encoder_checkpoint:
            ckpt = torch.load(encoder_checkpoint, map_location='cpu', weights_only=False)
            # Support both raw state_dict and wrapped checkpoint
            if isinstance(ckpt, dict) and 'student' in ckpt:
                sd = ckpt['student']
            elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                sd = ckpt['state_dict']
            elif isinstance(ckpt, dict) and 'model' in ckpt:
                sd = ckpt['model']
            else:
                sd = ckpt
            # Strip common prefixes
            sd = {k.replace('encoder.', '', 1) if k.startswith('encoder.') else k: v for k, v in sd.items()}
            missing, unexpected = self.encoder.load_state_dict(sd, strict=False)
            if missing:
                print(f"[TDVFrameEncoder] Missing keys: {len(missing)} (first 5: {missing[:5]})")
            if unexpected:
                print(f"[TDVFrameEncoder] Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
            print(f"[TDVFrameEncoder] Loaded custom weights from {encoder_checkpoint}")

        self.embed_dim = self.encoder.embed_dim
        self.patch_size = self.encoder.patch_size

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W)
        Returns:
            (B, 1+N, D) — CLS token prepended to patch tokens
        """
        out = self.encoder.forward_features(images)
        cls = out['x_norm_clstoken'].unsqueeze(1)      # (B, 1, D)
        patches = out['x_norm_patchtokens']             # (B, N, D)
        return torch.cat([cls, patches], dim=1)         # (B, 1+N, D)


# --------------------------------------------------------------------------- #
# Full TDV model
# --------------------------------------------------------------------------- #

class TDVModel(nn.Module):
    """Temporal Difference in Vision model for self-supervised video pretraining.

    Training objective:
        L = λ_recon * MSE(frame_enc(F_t) + motion_enc(ΔF), teacher_enc(F_{t+1}))
          + λ_dino  * DINO_loss(student CLS, teacher CLS)
          + λ_ibot  * iBOT_loss(student patches, teacher patches)
          + λ_motion * motion_capture_loss

    The EMA teacher is a copy of the frame encoder updated with exponential
    moving average, providing stable targets for self-distillation.
    """

    def __init__(
        self,
        # Frame encoder
        backbone_type: str = 'dinov2',
        backbone_size: str = 'base',
        pretrained: bool = True,
        unfreeze_frame_encoder: bool = False,
        img_size: int = 224,
        patch_size: int = 14,
        encoder_checkpoint: str = None,
        # Motion encoder
        motion_encoder_depth: int = 4,
        motion_encoder_heads: int = 12,
        remove_motion_encoder: bool = False,
        # EMA teacher
        use_ema: bool = True,
        ema_momentum: float = 0.996,
        use_fixed_dino_teacher: bool = False,
        # DINO head
        use_dino_head: bool = True,
        dino_head_prototype_dim: int = 65536,
        use_separate_ibot_head: bool = False,
        # Losses
        use_mse_loss: bool = True,
        mse_loss_weight: float = 1.0,
        use_dino_loss: bool = True,
        dino_loss_weight: float = 1.0,
        use_ibot_loss: bool = False,
        ibot_loss_weight: float = 1.0,
        use_motion_loss: bool = True,
        motion_loss_weight: float = 0.1,
        min_embed_diff_per_pixel_diff: float = 0.0,
        # Recon loss hparams
        recon_predicted_temp: float = 1.0,
        recon_target_temp: float = 1.0,
        recon_use_centering: bool = True,
        recon_use_sharpening: bool = True,
        recon_loss_type: str = 'mse',
        # DINO loss hparams
        dino_student_temp: float = 0.1,
        dino_teacher_temp: float = 0.04,
        dino_center_update_momentum: float = 0.9,
        ibot_student_temp: float = 0.1,
        ibot_teacher_temp: float = 0.04,
        ibot_center_update_momentum: float = 0.9,
        use_centering: bool = True,
        use_sharpening: bool = True,
        # Augmentation
        use_dino_augmentation: bool = True,
        # Rollout
        rollout_n_frames: int = 1,
        # Misc
        rgb_diff_threshold: float = 0.0,
        use_only_cls_token: bool = False,
        rgb_diff_from_unaugmented: bool = True,
        use_full_frame: bool = False,
        log_var_covar: bool = True,
        log_baseline_losses: bool = True,
    ):
        super().__init__()
        self.EPS = 0.00001
        self.use_ema = use_ema
        self.ema_momentum = ema_momentum
        self.use_fixed_dino_teacher = use_fixed_dino_teacher
        self.use_dino_head = use_dino_head
        self.use_separate_ibot_head = use_separate_ibot_head
        self.remove_motion_encoder = remove_motion_encoder
        self.use_mse_loss = use_mse_loss
        self.mse_loss_weight = mse_loss_weight
        self.use_dino_loss = use_dino_loss
        self.dino_loss_weight = dino_loss_weight
        self.use_ibot_loss = use_ibot_loss
        self.ibot_loss_weight = ibot_loss_weight
        self.use_motion_loss = use_motion_loss
        self.motion_loss_weight = motion_loss_weight
        self.min_embed_diff_per_pixel_diff = min_embed_diff_per_pixel_diff
        self.recon_use_centering = recon_use_centering
        self.recon_use_sharpening = recon_use_sharpening
        self.use_centering = use_centering
        self.use_sharpening = use_sharpening
        self.rollout_n_frames = rollout_n_frames
        self.rgb_diff_threshold = rgb_diff_threshold
        self.use_only_cls_token = use_only_cls_token
        self.rgb_diff_from_unaugmented = rgb_diff_from_unaugmented
        self.use_full_frame = use_full_frame
        self.use_dino_augmentation = use_dino_augmentation
        self.log_var_covar = log_var_covar
        self.log_baseline_losses = log_baseline_losses

        vit_size_map = {
            'small': ('dinov2_vits14', 384),
            'base': ('dinov2_vitb14', 768),
            'large': ('dinov2_vitl14', 1024),
        }
        hub_name, embed_dim = vit_size_map.get(backbone_size, ('dinov2_vitb14', 768))
        self.encoder_dim = embed_dim

        # -- Frame encoder
        self.frame_encoder = TDVFrameEncoder(
            model_name=hub_name,
            img_size=img_size,
            freeze=not unfreeze_frame_encoder,
            pretrained=pretrained,
            encoder_checkpoint=encoder_checkpoint,
        )
        set_trainable(self.frame_encoder, trainable=unfreeze_frame_encoder)

        # -- Motion encoder
        if not remove_motion_encoder:
            self.motion_encoder = MotionEncoder(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                condition_dim=embed_dim,
                depth=motion_encoder_depth,
                num_heads=motion_encoder_heads,
            )
            self.linear_fc = nn.Linear(embed_dim, embed_dim)
        else:
            self.motion_encoder = None
            self.linear_fc = None

        # -- DINO head
        self.dino_head = None
        if use_dino_head:
            self.dino_head = init_dino_head(in_dim=embed_dim, out_dim=dino_head_prototype_dim, layers=3)
            if use_separate_ibot_head:
                self.ibot_head = init_dino_head(in_dim=embed_dim, out_dim=dino_head_prototype_dim, layers=3)
            else:
                self.ibot_head = None

        # -- EMA teacher
        if use_ema:
            if use_fixed_dino_teacher:
                self.teacher_frame_encoder = TDVFrameEncoder(
                    model_name=hub_name, img_size=img_size, freeze=True, pretrained=True,
                    encoder_checkpoint=encoder_checkpoint,
                )
            else:
                self.teacher_frame_encoder = copy.deepcopy(self.frame_encoder)
            set_trainable(self.teacher_frame_encoder, trainable=False)

            if use_dino_head:
                self.teacher_dino_head = init_dino_head(in_dim=embed_dim, out_dim=dino_head_prototype_dim, layers=3)
                set_trainable(self.teacher_dino_head, trainable=False)
                if use_separate_ibot_head:
                    self.teacher_ibot_head = init_dino_head(in_dim=embed_dim, out_dim=dino_head_prototype_dim, layers=3)
                    set_trainable(self.teacher_ibot_head, trainable=False)
                else:
                    self.teacher_ibot_head = None
            else:
                self.teacher_dino_head = None
                self.teacher_ibot_head = None
        else:
            self.teacher_frame_encoder = None
            self.teacher_dino_head = None
            self.teacher_ibot_head = None

        # -- Losses
        self.recon_loss = CenterSharpReconstructionLoss(
            out_dim=embed_dim,
            predicted_temp=recon_predicted_temp,
            target_temp=recon_target_temp,
            center_momentum=0.99,
            loss=recon_loss_type,
        )
        self.dino_cce_loss = DinoLoss(
            out_dim=dino_head_prototype_dim,
            student_temp=dino_student_temp,
            teacher_temp=dino_teacher_temp,
            center_momentum=dino_center_update_momentum,
        )
        self.ibot_cce_loss = DinoLoss(
            out_dim=dino_head_prototype_dim,
            student_temp=ibot_student_temp,
            teacher_temp=ibot_teacher_temp,
            center_momentum=ibot_center_update_momentum,
        )

        # -- Logging losses
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.l1_loss = nn.L1Loss()
        self.baseline_mse_loss = nn.MSELoss()

        # -- Augmentation
        self.student_aug = None
        self.teacher_aug = None
        if use_dino_augmentation:
            image_size = img_size
            global_crops_scale = (0.4, 1.0)
            self.teacher_aug = DINOClipAugmentation(
                crop_scale=global_crops_scale, image_size=image_size,
                gaussian_blur_p=1.0, solarization_p=0.0)
            self.student_aug = DINOClipAugmentation(
                crop_scale=global_crops_scale, image_size=image_size,
                gaussian_blur_p=0.1, solarization_p=0.2)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def encode_sequences(self, frame_sequences, encoder, enable_grad=True):
        """Encode a batch of frame sequences.
        Args:
            frame_sequences: (B, T, C, H, W)
            encoder: TDVFrameEncoder
        Returns:
            (B, T, 1+N, D)
        """
        B, T, C, H, W = frame_sequences.shape
        context = torch.enable_grad if enable_grad else torch.no_grad
        with context():
            encoded = encoder(frame_sequences.reshape(B * T, C, H, W))
        D = encoded.shape[-1]
        return encoded.reshape(B, T, -1, D)

    def forward(self, frame_sequences: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full TDV forward pass for training.
        Args:
            frame_sequences: (B, T, C, H, W) — consecutive video frames
        Returns:
            dict with 'loss' and individual loss/metric tensors
        """
        # -- Augmentation
        if self.use_dino_augmentation:
            unaugmented_frames = frame_sequences
            student_frames = apply_clip_augmentation(frame_sequences, self.student_aug)
            teacher_frames = apply_clip_augmentation(frame_sequences, self.teacher_aug)
        else:
            student_frames = teacher_frames = unaugmented_frames = frame_sequences

        # -- Encode student frames (all but last)
        enable_grad = True  # frame encoder may be unfrozen
        previous_frame_encodings = self.encode_sequences(
            student_frames[:, :-1], self.frame_encoder, enable_grad
        )

        # -- Encode teacher frames (all but first) with no grad
        if self.use_ema:
            next_frame_encodings = self.encode_sequences(
                teacher_frames[:, 1:], self.teacher_frame_encoder, enable_grad=False
            ).detach()
        else:
            next_frame_encodings = self.encode_sequences(
                teacher_frames[:, 1:], self.frame_encoder, enable_grad=False
            ).detach()

        # -- Optionally use only CLS token
        if self.use_only_cls_token:
            previous_frame_encodings = previous_frame_encodings[:, :, 0]
            next_frame_encodings = next_frame_encodings[:, :, 0]

        # -- RGB diff
        rgb_diff_src = unaugmented_frames if self.rgb_diff_from_unaugmented else student_frames
        rgb_diff = get_rgb_diff(rgb_diff_src, self.use_full_frame)
        static_frames_mask = calculate_static_frames_mask(rgb_diff, self.rgb_diff_threshold)

        if skip_this_batch(static_frames_mask):
            return {"loss": torch.tensor(0., device=frame_sequences.device, requires_grad=True)}

        # -- Encode motion
        encoded_rgb_diff = None
        if not self.remove_motion_encoder:
            B, Tm1, C, H, W = rgb_diff.shape
            # Prepare condition: frame encoder patch tokens (no CLS) for each frame
            cond = previous_frame_encodings[:, :, 1:]  # (B, T-1, N, D)
            cond = cond.reshape(B * Tm1, -1, cond.shape[-1])

            rgb_diff_flat = rgb_diff.reshape(B * Tm1, C, H, W)
            encoded_rgb_diff = self.motion_encoder(rgb_diff_flat, cond)
            if self.use_only_cls_token:
                encoded_rgb_diff = encoded_rgb_diff[:, :, 0]
            encoded_rgb_diff = self.linear_fc(encoded_rgb_diff)
            encoded_rgb_diff = encoded_rgb_diff.reshape(B, Tm1, -1, self.encoder_dim)

        # -- Mask static frames
        if static_frames_mask is not None:
            rgb_diff = rgb_diff[static_frames_mask]
            encoded_rgb_diff = encoded_rgb_diff[static_frames_mask]
            previous_frame_encodings = previous_frame_encodings[static_frames_mask]
            next_frame_encodings = next_frame_encodings[static_frames_mask]

        # -- Hierarchical rollout
        previous_frame_encodings, encoded_rgb_diff, next_frame_encodings = rollout_n_frames(
            previous_frame_encodings, encoded_rgb_diff, next_frame_encodings,
            n=self.rollout_n_frames,
        )

        # -- Predict next frame: F_t + ΔF ≈ F_{t+1}
        if not self.remove_motion_encoder:
            predicted_next = previous_frame_encodings + encoded_rgb_diff
        else:
            predicted_next = previous_frame_encodings

        # -- Compute losses
        total_loss, individual_losses = self._compute_losses(
            previous_frame_encodings, predicted_next, next_frame_encodings, rgb_diff,
        )

        # -- Logging metrics
        logging_metrics = self._get_logging_metrics(
            previous_frame_encodings, predicted_next, next_frame_encodings,
        )

        return {"loss": total_loss, **individual_losses, **logging_metrics}

    def _compute_losses(self, previous, predicted, target, rgb_diff):
        total_loss = torch.zeros(1, device=predicted.device).squeeze()
        individual_losses = {}

        if self.use_mse_loss:
            recon_loss = self.recon_loss(
                predicted, target,
                use_centering=self.recon_use_centering,
                use_sharpening=self.recon_use_sharpening,
            )
            total_loss = total_loss + self.mse_loss_weight * recon_loss
            individual_losses["mse_loss"] = recon_loss.detach()

        if self.use_dino_loss:
            loss, metrics = self._compute_dino_style_loss(
                "dino",
                predicted[:, :, 0, :],
                target[:, :, 0, :],
                log_center=True,
            )
            total_loss = total_loss + self.dino_loss_weight * loss
            individual_losses.update(metrics)

        if self.use_ibot_loss:
            student_patches = predicted[:, :, 1:, :]
            teacher_patches = target[:, :, 1:, :]
            loss, metrics = self._compute_dino_style_loss(
                "ibot",
                student_patches,
                teacher_patches,
                log_center=True,
            )
            total_loss = total_loss + self.ibot_loss_weight * loss
            individual_losses.update(metrics)

        if self.use_motion_loss:
            loss, metrics = self._compute_motion_loss(previous, target, rgb_diff)
            total_loss = total_loss + self.motion_loss_weight * loss
            individual_losses.update(metrics)

        return total_loss, individual_losses

    def _compute_dino_style_loss(self, prefix, student_tokens, teacher_tokens, log_center=False, token_weights=None):
        student_head = self.dino_head
        teacher_head = self.teacher_dino_head
        if prefix == "ibot" and self.use_separate_ibot_head:
            student_head = self.ibot_head
            teacher_head = self.teacher_ibot_head

        student_logits = student_head(student_tokens)
        with torch.no_grad():
            teacher_logits = teacher_head(teacher_tokens.detach())

        cce_loss = self.ibot_cce_loss if (prefix == "ibot" and self.use_separate_ibot_head) else self.dino_cce_loss
        loss, entropy, kl = cce_loss(
            student_logits, teacher_logits,
            use_centering=self.use_centering,
            use_sharpening=self.use_sharpening,
            token_weights=token_weights,
        )

        metrics = {
            f"{prefix}_loss": loss.detach(),
            f"{prefix}_entropy": entropy.detach(),
            f"{prefix}_kl_div": kl.detach(),
        }
        if log_center:
            metrics[f"{prefix}_center_mean"] = cce_loss.center.mean().detach()
            metrics[f"{prefix}_center_std"] = cce_loss.center.std().detach()
            metrics[f"{prefix}_center_norm"] = cce_loss.center.norm().detach()
        return loss, metrics

    def _compute_motion_loss(self, previous, target, rgb_diff):
        pixel_diff_mean = torch.abs(rgb_diff).mean(dim=(-3, -2, -1)) + self.EPS
        embed_diff = torch.abs(previous - target)
        embed_diff_mean = embed_diff.mean(dim=(-2, -1))
        motion_loss = F.relu(
            self.min_embed_diff_per_pixel_diff - (embed_diff_mean / pixel_diff_mean)
        ).mean()
        metrics = {
            "motion_capture_loss": motion_loss.detach(),
            "embed_diff_mean": embed_diff_mean.mean().detach(),
            "pixel_diff_mean": pixel_diff_mean.mean().detach(),
        }
        return motion_loss, metrics

    def _get_logging_metrics(self, previous, predicted, target):
        metrics = {}
        if self.log_var_covar:
            with torch.no_grad():
                var, off_diag = self._calc_var_covar(previous)
                metrics.update({
                    "variance": var, "off_diag_covariance": off_diag,
                    "predicted_variance": self._calc_var_covar(predicted)[0],
                    "teacher_variance": self._calc_var_covar(target)[0],
                })

        with torch.no_grad():
            if self.log_baseline_losses:
                metrics["baseline_mse_loss"] = self.baseline_mse_loss(previous, target).detach()
                metrics["baseline_smooth_l1_loss"] = self.smooth_l1_loss(previous, target).detach()
            metrics["smooth_l1_loss"] = self.smooth_l1_loss(predicted, target).detach()
            metrics["l1_loss"] = self.l1_loss(predicted, target).detach()
        return metrics

    @staticmethod
    @torch.no_grad()
    def _calc_var_covar(encodings):
        flattened = encodings.reshape(-1, encodings.size(-1))
        flattened_centered = flattened - flattened.mean(dim=0, keepdim=True)
        variance = flattened.var(dim=0, unbiased=False).mean()
        cov = (flattened_centered.T @ flattened_centered) / max(flattened_centered.shape[0] - 1, 1)
        off_diag = cov.flatten()[1:].view(cov.size(0) - 1, cov.size(1) + 1)[:, :-1]
        return variance.detach(), off_diag.abs().mean().detach()

    # ------------------------------------------------------------------ #
    # EMA update
    # ------------------------------------------------------------------ #

    def ema_update(self):
        if not self.use_ema:
            return
        if self.use_fixed_dino_teacher:
            students = [self.dino_head]
            teachers = [self.teacher_dino_head]
        else:
            students = [self.frame_encoder]
            teachers = [self.teacher_frame_encoder]
            if self.use_dino_head:
                students.append(self.dino_head)
                teachers.append(self.teacher_dino_head)
            if self.use_separate_ibot_head:
                students.append(self.ibot_head)
                teachers.append(self.teacher_ibot_head)
        update_teacher_using_ema(students, teachers, self.ema_momentum)

    # ------------------------------------------------------------------ #
    # Checkpoint helpers
    # ------------------------------------------------------------------ #

    def get_frame_encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        """Extract frame encoder weights for downstream detection pipeline."""
        return self.frame_encoder.encoder.state_dict()

    def load_frame_encoder_from_state_dict(self, sd: Dict[str, torch.Tensor]):
        """Load frame encoder weights (e.g. from a TDV checkpoint)."""
        self.frame_encoder.encoder.load_state_dict(sd, strict=True)
