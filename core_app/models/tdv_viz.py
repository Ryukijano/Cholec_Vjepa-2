"""Visualization utilities for TDV training — logs to Weights & Biases.

Produces:
  - Attention rollout heatmaps (frame encoder + motion encoder)
  - Feature PCA RGB maps (first 3 principal components → RGB)
  - RGB difference overlay (what the motion encoder sees)
  - Prediction error maps (where TDV fails to predict next frame)
  - Feature similarity matrix (collapse detection)
  - Weight histogram (encoder layer-wise statistics)
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def attention_rollout(attn_weights: list, num_patches: int, img_size: int = 224):
    """Compute attention rollout from list of per-layer attention maps.

    Args:
        attn_weights: list of (B, num_heads, N, N) attention matrices
        num_patches: number of spatial patches (excluding CLS)
        img_size: output heatmap size

    Returns:
        (B, img_size, img_size) attention heatmap from CLS token to patches
    """
    # Average over heads, then cumulatively multiply
    rollout = None
    for attn in attn_weights:
        # attn: (B, H, N, N) → average heads → (B, N, N)
        attn_avg = attn.mean(dim=1)
        # Add identity for residual connection
        attn_avg = attn_avg + torch.eye(attn_avg.size(-1), device=attn.device).unsqueeze(0)
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        if rollout is None:
            rollout = attn_avg
        else:
            rollout = torch.bmm(attn_avg, rollout)

    # CLS token attention to patches: rollout[:, 0, 1:]
    cls_attn = rollout[:, 0, 1:]  # (B, num_patches)
    gs = int(math.sqrt(num_patches))
    heatmap = cls_attn.reshape(-1, gs, gs)
    heatmap = F.interpolate(heatmap.unsqueeze(1), size=(img_size, img_size),
                            mode='bilinear', align_corners=False).squeeze(1)
    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0])
    heatmap = heatmap / (heatmap.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
    return heatmap


def feature_pca_rgb(features: torch.Tensor, img_size: int = 224):
    """Map first 3 principal components of patch features to RGB.

    Args:
        features: (B, N, D) patch tokens (no CLS)
        img_size: output image size

    Returns:
        (B, 3, img_size, img_size) RGB visualization
    """
    B, N, D = features.shape
    gs = int(math.sqrt(N))
    imgs = []
    for b in range(B):
        feat = features[b]  # (N, D)
        # Center
        feat_centered = feat - feat.mean(dim=0, keepdim=True)
        # SVD for PCA (fast)
        U, S, Vh = torch.linalg.svd(feat_centered, full_matrices=False)
        proj = feat_centered @ Vh[:3].T  # (N, 3)
        # Normalize to [0, 1]
        proj = (proj - proj.min(dim=0, keepdim=True)[0])
        proj = proj / (proj.max(dim=0, keepdim=True)[0] + 1e-8)
        # Reshape to image
        img = proj.reshape(gs, gs, 3).permute(2, 0, 1)  # (3, gs, gs)
        img = F.interpolate(img.unsqueeze(0), size=(img_size, img_size),
                           mode='bilinear', align_corners=False).squeeze(0)
        imgs.append(img)
    return torch.stack(imgs)  # (B, 3, H, W)


def prediction_error_map(predicted: torch.Tensor, target: torch.Tensor,
                         num_patches: int, img_size: int = 224):
    """Visualize where TDV's prediction (F_t + ΔF) differs from teacher (F_{t+1}).

    Args:
        predicted: (B, N, D) predicted next-frame patch tokens
        target: (B, N, D) teacher next-frame patch tokens
        num_patches: number of spatial patches
        img_size: output heatmap size

    Returns:
        (B, img_size, img_size) error heatmap
    """
    error = (predicted - target).pow(2).mean(dim=-1)  # (B, N)
    gs = int(math.sqrt(num_patches))
    heatmap = error.reshape(-1, gs, gs)
    heatmap = F.interpolate(heatmap.unsqueeze(1), size=(img_size, img_size),
                            mode='bilinear', align_corners=False).squeeze(1)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap


def feature_similarity_matrix(features: torch.Tensor):
    """Compute cosine similarity matrix between batch elements' CLS tokens.

    High off-diagonal similarity → potential collapse.

    Args:
        features: (B, D) CLS token features

    Returns:
        (B, B) similarity matrix
    """
    feat_norm = F.normalize(features, dim=-1)
    sim = feat_norm @ feat_norm.T  # (B, B)
    return sim


def make_grid_image(images: torch.Tensor, nrow: int = 4):
    """Make a grid from batch of images for logging.

    Args:
        images: (B, C, H, W)
        nrow: number of images per row

    Returns:
        (3, H*nrow//B+1, W*nrow) grid image
    """
    B, C, H, W = images.shape
    ncols = min(nrow, B)
    nrows = math.ceil(B / ncols)
    grid = torch.zeros(C, nrows * H, ncols * W)
    for i in range(B):
        r, c = i // ncols, i % ncols
        grid[:, r*H:(r+1)*H, c*W:(c+1)*W] = images[i]
    return grid


def denormalize(tensor: torch.Tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Denormalize a normalized image tensor for visualization."""
    mean_t = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)
    return (tensor * std_t + mean_t).clamp(0, 1)


def overlay_heatmap(image: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.5):
    """Overlay a heatmap on an image.

    Args:
        image: (B, 3, H, W) in [0, 1]
        heatmap: (B, H, W) in [0, 1]
        alpha: heatmap blending factor

    Returns:
        (B, 3, H, W) overlaid image
    """
    # Convert heatmap to jet colormap
    heatmap_colored = jet_colormap(heatmap)  # (B, 3, H, W)
    return (1 - alpha) * image + alpha * heatmap_colored


def jet_colormap(heatmap: torch.Tensor):
    """Convert grayscale heatmap to jet colormap.

    Args:
        heatmap: (B, H, W) in [0, 1]

    Returns:
        (B, 3, H, W) jet-colored heatmap
    """
    h = heatmap.unsqueeze(1)  # (B, 1, H, W)
    r = torch.clamp(1.5 - torch.abs(4 * h - 3), 0, 1)
    g = torch.clamp(1.5 - torch.abs(4 * h - 2), 0, 1)
    b = torch.clamp(1.5 - torch.abs(4 * h - 1), 0, 1)
    return torch.cat([r, g, b], dim=1)


class AttentionExtractor:
    """Extract attention weights from DINOv2 ViT blocks via forward hooks."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.attention_weights = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for block in self.encoder.blocks:
            def hook(module, input, output, block_idx=len(self.hooks)):
                # MemEffAttention returns (attn_output, attn_weights)
                # If xformers, output is just the output — we need to capture qkv
                pass
            # We can't easily get attention from MemEffAttention
            # Instead, we'll use a different approach: register on qkv

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def extract_attention_simple(encoder, images, layer_idx=-1):
    """Extract attention weights from a specific layer by temporarily
    replacing MemEffAttention with a version that returns weights.

    This is a simpler approach: compute attention manually from qkv.
    """
    block = encoder.blocks[layer_idx]
    attn = block.attn

    # Get input to attention (after norm1)
    # We need to run the block up to attention, then compute manually
    # Easier: register a forward pre-hook on attn to capture input,
    # then compute attention from qkv weights
    captured_input = []

    def capture_hook(module, inp):
        captured_input.append(inp[0])

    handle = attn.register_forward_pre_hook(capture_hook)
    with torch.no_grad():
        encoder(images)
    handle.remove()

    if not captured_input:
        return None

    x = captured_input[0]  # (B, N, D)
    B, N, D = x.shape

    # Compute Q, K, V
    qkv = attn.qkv(x)  # (B, N, 3*D)
    q, k, v = qkv.chunk(3, dim=-1)  # each (B, N, D)

    num_heads = attn.num_heads
    head_dim = D // num_heads

    q = q.reshape(B, N, num_heads, head_dim).transpose(1, 2)  # (B, H, N, hd)
    k = k.reshape(B, N, num_heads, head_dim).transpose(1, 2)

    # Attention weights
    scale = head_dim ** -0.5
    attn_weights = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N)
    attn_weights = attn_weights.softmax(dim=-1)

    return attn_weights


def log_visualizations_to_wandb(wandb_run, model, frame_sequences, step, max_images=4):
    """Generate and log visualizations to W&B.

    Args:
        wandb_run: active wandb.run object
        model: TDVModel (raw, not DDP-wrapped)
        frame_sequences: (B, T, C, H, W) sample batch
        step: training step
        max_images: max number of images to log
    """
    import wandb

    model.eval()
    with torch.no_grad():
        B, T, C, H, W = frame_sequences.shape
        n = min(max_images, B)

        # -- 1. Input frames (denormalized)
        frames = denormalize(frame_sequences[:n])  # (n, T, C, H, W)
        # Log first and last frame side by side
        frame_grid = make_grid_image(frames[:, 0], nrow=n)  # first frames
        wandb.log({"viz/input_frames_first": wandb.Image(frame_grid.cpu().numpy(),
                   caption=f"First frames (step {step})")}, step=step)

        # -- 2. RGB difference (what motion encoder sees)
        rgb_diff = frame_sequences[:n, 1:] - frame_sequences[:n, :-1]  # (n, T-1, C, H, W)
        diff_mag = rgb_diff.abs().mean(dim=2)  # (n, T-1, H, W)
        diff_grid = make_grid_image(diff_mag[:, 0:1].expand(-1, 3, -1, -1), nrow=n)
        wandb.log({"viz/rgb_diff": wandb.Image(diff_grid.cpu().numpy(),
                   caption=f"RGB difference (step {step})")}, step=step)

        # -- 3. Frame encoder attention rollout
        try:
            sample_imgs = frame_sequences[:n, 0]  # first frame of each sequence
            attn_weights = extract_attention_simple(
                model.frame_encoder.encoder, sample_imgs, layer_idx=-1
            )
            if attn_weights is not None:
                num_patches = (H // model.patch_size) ** 2
                heatmap = attention_rollout([attn_weights], num_patches, img_size=H)
                frames_denorm = denormalize(sample_imgs)
                overlaid = overlay_heatmap(frames_denorm, heatmap, alpha=0.5)
                attn_grid = make_grid_image(overlaid, nrow=n)
                wandb.log({"viz/attention_rollout": wandb.Image(attn_grid.cpu().numpy(),
                           caption=f"CLS attention rollout (step {step})")}, step=step)
        except Exception as e:
            print(f"  [viz] Attention rollout failed: {e}")

        # -- 4. Feature PCA RGB map
        try:
            encoded = model.frame_encoder(frame_sequences[:n, 0])  # (n, 1+N, D)
            patches = encoded[:, 1:, :]  # (n, N, D)
            pca_img = feature_pca_rgb(patches, img_size=H)
            pca_grid = make_grid_image(pca_img, nrow=n)
            wandb.log({"viz/feature_pca": wandb.Image(pca_grid.cpu().numpy(),
                       caption=f"Patch feature PCA→RGB (step {step})")}, step=step)
        except Exception as e:
            print(f"  [viz] Feature PCA failed: {e}")

        # -- 5. Prediction error map
        try:
            with torch.no_grad():
                student_enc = model.encode_sequences(
                    frame_sequences[:n, :-1], model.frame_encoder, enable_grad=False
                )
                if model.use_ema:
                    teacher_enc = model.encode_sequences(
                        frame_sequences[:n, 1:], model.teacher_frame_encoder, enable_grad=False
                    )
                else:
                    teacher_enc = model.encode_sequences(
                        frame_sequences[:n, 1:], model.frame_encoder, enable_grad=False
                    )

                # Compute RGB diff and encode motion
                rgb_diff_batch = frame_sequences[:n, 1:] - frame_sequences[:n, :-1]
                B2, T2, C2, H2, W2 = rgb_diff_batch.shape
                cond = student_enc[:, :, 1:].reshape(B2 * T2, -1, student_enc.shape[-1])
                rgb_diff_flat = rgb_diff_batch.reshape(B2 * T2, C2, H2, W2)
                motion_enc = model.motion_encoder(rgb_diff_flat, cond)
                motion_enc = model.linear_fc(motion_enc)
                motion_enc = motion_enc.reshape(B2, T2, -1, model.encoder_dim)

                predicted = student_enc + motion_enc
                num_patches = (H // model.patch_size) ** 2

                # Error for first timestep
                err_map = prediction_error_map(
                    predicted[:, 0], teacher_enc[:, 0], num_patches, img_size=H
                )
                frames_denorm = denormalize(frame_sequences[:n, 1])
                overlaid_err = overlay_heatmap(frames_denorm, err_map, alpha=0.6)
                err_grid = make_grid_image(overlaid_err, nrow=n)
                wandb.log({"viz/prediction_error": wandb.Image(err_grid.cpu().numpy(),
                           caption=f"Prediction error F_t+ΔF vs teacher (step {step})")}, step=step)
        except Exception as e:
            print(f"  [viz] Prediction error failed: {e}")

        # -- 6. Feature similarity matrix (collapse detection)
        try:
            cls_tokens = encoded[:, 0, :]  # (n, D)
            sim_mat = feature_similarity_matrix(cls_tokens)
            wandb.log({"viz/feature_similarity": wandb.Image(sim_mat.cpu().numpy(),
                       caption=f"CLS cosine similarity (step {step})")}, step=step)
            off_diag = sim_mat[~torch.eye(n, dtype=bool, device=sim_mat.device)]
            mean_sim = off_diag.mean().item()
            wandb.log({"viz/mean_cls_similarity": mean_sim}, step=step)
        except Exception as e:
            print(f"  [viz] Feature similarity failed: {e}")

        # -- 7. Weight histogram for encoder
        try:
            for name, param in model.frame_encoder.encoder.named_parameters():
                if param.requires_grad and 'weight' in name and 'norm' not in name:
                    wandb.log({f"weights/{name}": wandb.Histogram(param.detach().cpu().numpy())},
                             step=step)
                    break  # only log one layer per step to avoid spam
        except Exception as e:
            print(f"  [viz] Weight histogram failed: {e}")

    model.train()
