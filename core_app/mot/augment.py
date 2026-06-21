"""
Surgical corruption augmentations for GOT-JEPA student-branch training.

GOT-JEPA (TCSVT 2026, https://arxiv.org/abs/2602.14771) trains the
student model predictor to produce the *same* filter weights as a
frozen teacher, even when the student sees a corrupted current frame.
This forces the predictor to learn invariance to the corruption modes.

For surgical video the dominant corruption modes (per
docs/multi_object_tracking_research.md) are:

  * Smoke / plume — ~2000 annotated instances in CholecTrack20
  * Fouled lens  — ~2196 instances
  * Blood splatter
  * Specular reflections (metallic tool glare)
  * Defocus blur (motion / autofocus hunting)

All augmentations run on GPU tensors of shape (B, 3, H, W) in [0, 1].
Each augmentation has a ``probability`` argument controlling per-sample
application. Corruptions are intentionally agnostic of bounding boxes —
they apply to the whole frame.

This module is imported only by ``jepa.py`` during Stage 2 pretraining.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def _sample_mask(batch_size: int, p: float, device: torch.device) -> torch.Tensor:
    """Bernoulli mask deciding per-sample whether to apply an augmentation."""
    return (torch.rand(batch_size, device=device) < p).view(batch_size, 1, 1, 1)


def _perlin_noise(
    b: int, c: int, h: int, w: int, device: torch.device, octaves: int = 4
) -> torch.Tensor:
    """
    Cheap Perlin-ish noise by summing up-sampled random low-res maps.

    Not strictly Perlin, but indistinguishable at the scale we care about
    and fully differentiable-free (we never backprop through this).
    """
    noise = torch.zeros(b, c, h, w, device=device)
    freq = 1.0
    amp = 1.0
    for _ in range(octaves):
        res_h = max(1, int(h / (8.0 * freq)))
        res_w = max(1, int(w / (8.0 * freq)))
        low = torch.randn(b, c, res_h, res_w, device=device)
        up = F.interpolate(low, size=(h, w), mode='bilinear', align_corners=False)
        noise = noise + amp * up
        freq *= 2.0
        amp *= 0.5
    # Normalise roughly to [0, 1].
    noise = (noise - noise.amin(dim=(-1, -2), keepdim=True)) / (
        noise.amax(dim=(-1, -2), keepdim=True)
        - noise.amin(dim=(-1, -2), keepdim=True)
        + 1e-6
    )
    return noise


def _depth_prior(b: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """
    Generate a synthetic depth prior for surgical scenes.

    Surgical images have a radial depth gradient (centre = near, edges = far)
    plus a vertical gradient (top = far, bottom = near, due to camera angle).
    Returns values in [0, 1] where 0 = near, 1 = far.
    """
    yy = torch.linspace(0, 1, h, device=device).view(1, 1, h, 1)
    xx = torch.linspace(-0.5, 0.5, w, device=device).view(1, 1, 1, w)
    radius = torch.sqrt(xx ** 2 + ((yy - 0.4) * 1.2) ** 2)
    depth = torch.sigmoid(3.0 * (radius - 0.3))
    return depth.expand(b, 1, h, w)


def smoke(
    x: torch.Tensor,
    probability: float = 0.4,
    intensity: float = 0.5,
    density_scale: float = 2.5,
    airlight_color: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Physics-based atmospheric scattering smoke augmentation.

    Models surgical smoke as a participating medium using the atmospheric
    scattering equation:

        I_observed = T(d) * I_scene + (1 - T(d)) * A

    where:
      - T(d) = exp(-beta * d)  is the Beer-Lambert transmittance
      - d is a synthetic depth prior (radial + vertical gradient)
      - beta is the scattering coefficient (spatially varying via Perlin noise)
      - A is the airlight (scattered light colour, ~grey-white)

    This creates **depth-dependent** smoke: distant tissue is more obscured
    than near surgical tools, matching real surgical smoke behaviour.

    Args:
        x: (B, 3, H, W) input images in [0, 1].
        probability: per-sample probability of applying smoke.
        intensity: overall smoke intensity multiplier [0, 1].
        density_scale: controls scattering coefficient magnitude.
        airlight_color: (3,) tensor for airlight colour; defaults to warm grey.
    """
    B, C, H, W = x.shape
    device = x.device
    mask = _sample_mask(B, probability, device)
    if not mask.any():
        return x

    if airlight_color is None:
        airlight_color = torch.tensor([0.85, 0.83, 0.78], device=device)
    airlight_color = airlight_color.view(1, 3, 1, 1)

    # Spatially varying scattering coefficient (Perlin noise * density).
    density_noise = _perlin_noise(B, 1, H, W, device, octaves=5)
    # Concentrate smoke near centre-bottom (where cautery smoke rises).
    yy = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1)
    xx = torch.linspace(-0.5, 0.5, W, device=device).view(1, 1, 1, W)
    spatial_bias = torch.sigmoid(2.0 * (0.5 - torch.sqrt(xx ** 2 + (yy - 0.5) ** 2)))
    beta = density_noise * density_scale * intensity * (0.5 + spatial_bias)

    # Synthetic depth prior: 0 = near, 1 = far.
    depth = _depth_prior(B, H, W, device)

    # Beer-Lambert transmittance: T = exp(-beta * depth).
    # Higher beta (more smoke) and higher depth (farther) → less transmittance.
    transmittance = torch.exp(-beta * depth)  # (B, 1, H, W) in (0, 1]

    # Atmospheric scattering: observed = T * scene + (1 - T) * airlight.
    smoked = transmittance * x + (1.0 - transmittance) * airlight_color

    return _ensure_01(torch.where(mask, smoked, x))


def blood_splatter(
    x: torch.Tensor, probability: float = 0.2, intensity: float = 0.7
) -> torch.Tensor:
    """
    Overlay reddish blood splatter via a low-threshold Perlin mask.
    """
    B, C, H, W = x.shape
    device = x.device
    mask = _sample_mask(B, probability, device)
    if not mask.any():
        return x

    raw = _perlin_noise(B, 1, H, W, device, octaves=3)
    blood_mask = ((raw > 0.7).float()) * intensity
    blood_col = torch.tensor([0.55, 0.10, 0.10], device=device).view(1, 3, 1, 1)
    bloody = (1.0 - blood_mask) * x + blood_mask * blood_col
    return _ensure_01(torch.where(mask, bloody, x))


def specular_highlights(
    x: torch.Tensor, probability: float = 0.3, intensity: float = 0.8
) -> torch.Tensor:
    """
    Inject bright specular blobs (mimics metallic tool glare).
    """
    B, C, H, W = x.shape
    device = x.device
    mask = _sample_mask(B, probability, device)
    if not mask.any():
        return x

    raw = _perlin_noise(B, 1, H, W, device, octaves=2)
    spec = torch.sigmoid(8.0 * (raw - 0.85)) * intensity
    highlighted = x * (1 - spec) + spec
    return _ensure_01(torch.where(mask, highlighted, x))


def defocus_blur(
    x: torch.Tensor, probability: float = 0.3, kernel_size: int = 7
) -> torch.Tensor:
    """
    Gaussian blur applied per-sample.
    """
    B, C, H, W = x.shape
    device = x.device
    mask = _sample_mask(B, probability, device)
    if not mask.any():
        return x

    # Build a separable Gaussian kernel.
    sigma = kernel_size / 3.0
    coords = torch.arange(kernel_size, device=device) - kernel_size // 2
    g1d = torch.exp(-(coords.float() ** 2) / (2.0 * sigma ** 2))
    g1d = g1d / g1d.sum()
    kernel = g1d.view(1, 1, 1, -1) * g1d.view(1, 1, -1, 1)  # (1, 1, K, K)
    kernel = kernel.expand(C, 1, kernel_size, kernel_size)

    blurred = F.conv2d(x, kernel, padding=kernel_size // 2, groups=C)
    return _ensure_01(torch.where(mask, blurred, x))


def color_jitter(
    x: torch.Tensor,
    probability: float = 0.5,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.3,
) -> torch.Tensor:
    """
    Per-sample brightness/contrast/saturation jitter.
    """
    B, C, H, W = x.shape
    device = x.device
    mask = _sample_mask(B, probability, device)
    if not mask.any():
        return x

    b_fac = 1.0 + (torch.rand(B, device=device) * 2 - 1) * brightness
    c_fac = 1.0 + (torch.rand(B, device=device) * 2 - 1) * contrast
    s_fac = 1.0 + (torch.rand(B, device=device) * 2 - 1) * saturation

    y = x * b_fac.view(B, 1, 1, 1)
    mean = y.mean(dim=(-1, -2), keepdim=True)
    y = (y - mean) * c_fac.view(B, 1, 1, 1) + mean
    grey = y.mean(dim=1, keepdim=True)
    y = (y - grey) * s_fac.view(B, 1, 1, 1) + grey
    return _ensure_01(torch.where(mask, y, x))


def cutout(
    x: torch.Tensor,
    probability: float = 0.2,
    patch_frac: float = 0.15,
) -> torch.Tensor:
    """
    Zero-out a random square patch per sample.
    """
    B, C, H, W = x.shape
    device = x.device
    mask = _sample_mask(B, probability, device)
    if not mask.any():
        return x

    ph = int(H * patch_frac)
    pw = int(W * patch_frac)
    out = x.clone()
    for b in range(B):
        if not mask[b].any():
            continue
        cy = int(torch.randint(0, H - ph, (1,)).item())
        cx = int(torch.randint(0, W - pw, (1,)).item())
        out[b, :, cy : cy + ph, cx : cx + pw] = 0.0
    return out


class SurgicalCorruption(nn.Module):
    """
    Compose the surgical corruption bank. Applied only to the student
    branch of GOT-JEPA pretraining. Deterministic when ``eval()``'d
    (corruptions become identity).
    """

    def __init__(
        self,
        smoke_p: float = 0.4,
        blood_p: float = 0.2,
        spec_p: float = 0.3,
        blur_p: float = 0.3,
        jitter_p: float = 0.5,
        cutout_p: float = 0.2,
    ):
        super().__init__()
        self.ops: List[Tuple[Callable, dict]] = [
            (smoke, {'probability': smoke_p}),
            (blood_splatter, {'probability': blood_p}),
            (specular_highlights, {'probability': spec_p}),
            (defocus_blur, {'probability': blur_p}),
            (color_jitter, {'probability': jitter_p}),
            (cutout, {'probability': cutout_p}),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        out = x
        for op, kwargs in self.ops:
            out = op(out, **kwargs)
        return out
