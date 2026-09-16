"""
GOT-JEPA teacher-student predictor pretraining.

Paper: GOT-JEPA (TCSVT 2026) — https://arxiv.org/abs/2602.14771

Architecture (Figure 2a in the paper):

  Reference history ───┬──► clean current frame ─► t-Predictor (frozen) ─► ω̂
                       │
                       └──► corrupt current frame ─► s-Predictor ──► ProjNet ─► ω
                                                            │
                                                            └─► Expander ─► ω_exp

Losses (Eq. 2–3 in the paper):

  L_inv = (1 / n) * Σ_i || ω_i − ω̂_i ||²
  L_cov = Σ_{i ≠ j} covM(Exp(ω))[i, j]²     (off-diagonal of covariance)

  L_total = α · L_inv + β · L_cov

This module provides:

  * ``JEPAProjector`` — small linear projector (ProjNet) on the student.
  * ``JEPAExpander`` — 1×1 conv expander for the covariance-loss branch.
  * ``invariance_loss`` / ``covariance_loss`` — standalone loss helpers.
  * ``GOTJEPAWrapper`` — ties a teacher + student ``PerTrackModelPredictor``
    together with ProjNet / Expander for pretraining. The teacher is
    fully frozen. The student (+ ProjNet + Expander) is trained.

The teacher is initialised as a deep copy of the Stage-1-trained
``PerTrackModelPredictor``.
"""
from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .predictor import PerTrackModelPredictor


class JEPAProjector(nn.Module):
    """Linear ProjNet tail for the student predictor."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        return self.proj(omega)


class JEPAExpander(nn.Module):
    """
    1×1 conv expander that broadens the output channel for the
    covariance loss. Input is treated as a (N, dim, 1, 1) tensor.
    """

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * expansion, kernel_size=1)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        """
        Args:
            omega: (N, dim) filter weights.

        Returns:
            (N, dim * expansion) expanded weights.
        """
        x = omega.unsqueeze(-1).unsqueeze(-1)  # (N, dim, 1, 1)
        y = self.conv(x)                        # (N, dim*expansion, 1, 1)
        return y.squeeze(-1).squeeze(-1)         # (N, dim*expansion)


def invariance_loss(student_omega: torch.Tensor, teacher_omega: torch.Tensor) -> torch.Tensor:
    """
    L_inv = (1 / n) * Σ_i || ω_i − ω̂_i ||²

    Teacher weights are detached so no gradient flows back (teacher is
    kept frozen anyway, but this is a safety rail).
    """
    return F.mse_loss(student_omega, teacher_omega.detach())


def covariance_loss(omega_exp: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    L_cov = off-diagonal-sum-of-squares of covariance(ω_exp).

    Eq. 3 in the GOT-JEPA paper — encourages de-correlated output
    dimensions (prevents collapse, forces diverse tracking models).

    Args:
        omega_exp: (N, D) where N is batch (tracks) and D is the
                   expanded output dim.
    """
    N, D = omega_exp.shape
    if N < 2:
        return omega_exp.new_tensor(0.0)
    centred = omega_exp - omega_exp.mean(dim=0, keepdim=True)
    cov = (centred.t() @ centred) / (N - 1)                 # (D, D)
    # Off-diagonal squared sum, averaged per dim (following VICReg).
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag.pow(2).sum()) / (D + eps)


def visreg_loss(
    z: torch.Tensor,
    num_slices: int = 64,
    scale_weight: float = 1.0,
    shape_weight: float = 1.0,
    center_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    VISReg: Variance-Invariance-Sketching Regularization.

    Replaces the VICReg-style covariance loss with a decoupled
    scale + shape + center regularizer that has non-vanishing
    gradient at collapse (Wu et al., 2026).

    L_scale  = (1/D) * Σ_j (1 - σ_j(z))²       — per-dim unit variance
    L_shape  = (1/K) * Σ_k ||sort(z̃ w_k) - q_N||²  — SWD to isotropic Gaussian
    L_center = (1/D) * Σ_j μ_j²                 — zero mean per dim

    Args:
        z: (N, D) expanded student output.
        num_slices: K random projection directions for SWD.
        scale_weight: weight for L_scale.
        shape_weight: weight for L_shape.
        center_weight: weight for L_center.

    Returns:
        (total_loss, {'scale': float, 'shape': float, 'center': float})
    """
    N, D = z.shape
    if N < 2:
        zero = z.new_tensor(0.0)
        return zero, {'scale': 0.0, 'shape': 0.0, 'center': 0.0}

    # 1. Center loss — zero mean per dimension
    mu = z.mean(dim=0)                                   # (D,)
    l_center = mu.pow(2).mean()

    # 2. Scale loss — unit variance per dimension
    z_cent = z - mu.unsqueeze(0)                         # (N, D)
    std = z_cent.std(dim=0, unbiased=False)              # (D,)
    l_scale = (1.0 - std).pow(2).mean()

    # 3. Shape loss — Sliced Wasserstein Distance to isotropic Gaussian
    #    Normalize out scale, then project onto K random directions
    std_safe = std.clamp(min=1e-6)
    z_norm = z_cent / std_safe.detach()                  # (N, D), detached std for no grad through scale

    # Random projection directions (D, K) — resampled each call (stochastic)
    W = torch.randn(D, num_slices, device=z.device, dtype=z.dtype)
    W = W / W.norm(p=2, dim=0, keepdim=True)             # unit-norm columns

    # Project and sort: (N, K)
    projections = z_norm @ W
    p_sorted = torch.sort(projections, dim=0).values

    # Gaussian target quantiles
    u = torch.arange(1, N + 1, device=z.device, dtype=z.dtype) / (N + 1)
    target = Normal(0.0, 1.0).icdf(u)                    # (N,)

    l_shape = (p_sorted - target.unsqueeze(1)).pow(2).mean()

    total = scale_weight * l_scale + shape_weight * l_shape + center_weight * l_center

    return total, {
        'scale': float(l_scale.item()),
        'shape': float(l_shape.item()),
        'center': float(l_center.item()),
    }


class GOTJEPAWrapper(nn.Module):
    """
    Ties together the frozen teacher, trainable student, ProjNet, and
    Expander for GOT-JEPA pretraining.

    The caller is responsible for feeding the *clean* current-frame
    features to the teacher and the *corrupted* current-frame features
    to the student. Reference history is shared.

    Regularization modes:
      - ``'vicreg'`` (default): VICReg-style off-diagonal covariance loss
        (original GOT-JEPA paper Eq. 3).
      - ``'visreg'``: VISReg regularization (Wu et al., 2026) — decoupled
        scale + shape (SWD) + center losses with non-vanishing gradient
        at collapse. Better for small datasets and OOD generalization.
    """

    def __init__(
        self,
        student_predictor: PerTrackModelPredictor,
        teacher_predictor: Optional[PerTrackModelPredictor] = None,
        expander_dim_mul: int = 4,
        inv_weight: float = 1.0,
        cov_weight: float = 0.5,
        reg_mode: str = 'vicreg',
        visreg_num_slices: int = 64,
        visreg_scale_weight: float = 1.0,
        visreg_shape_weight: float = 1.0,
        visreg_center_weight: float = 1.0,
    ):
        super().__init__()

        self.student = student_predictor
        if teacher_predictor is None:
            # Deep-copy student as the teacher starting point.
            teacher_predictor = copy.deepcopy(student_predictor)
        self.teacher = teacher_predictor
        self._freeze_teacher()

        self.projector = JEPAProjector(dim=student_predictor.dim)
        self.expander = JEPAExpander(
            dim=student_predictor.dim, expansion=expander_dim_mul
        )

        self.inv_weight = inv_weight
        self.cov_weight = cov_weight
        self.reg_mode = reg_mode
        self.visreg_num_slices = visreg_num_slices
        self.visreg_scale_weight = visreg_scale_weight
        self.visreg_shape_weight = visreg_shape_weight
        self.visreg_center_weight = visreg_center_weight

    def _freeze_teacher(self) -> None:
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        """Keep teacher in eval mode regardless of parent train/eval."""
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(
        self,
        reference_features: torch.Tensor,     # (N, N_ref, C)
        label_encoding: torch.Tensor,         # (N, N_ref, 1)
        current_features_clean: torch.Tensor,  # (N, N_cur, C)
        current_features_dirty: torch.Tensor,  # (N, N_cur, C)
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            reference_features, label_encoding: shared history inputs.
            current_features_clean: fed to the teacher.
            current_features_dirty: fed to the student (augmented).

        Returns:
            ``{'loss', 'loss_dict', 'omega_student', 'omega_teacher'}``.
        """
        with torch.no_grad():
            omega_t, _ = self.teacher(
                reference_features, label_encoding, current_features_clean
            )

        omega_s_raw, _ = self.student(
            reference_features, label_encoding, current_features_dirty
        )
        omega_s = self.projector(omega_s_raw)

        # Invariance: student should match teacher despite corruption.
        l_inv = invariance_loss(omega_s, omega_t)

        # Regularization on expanded student output.
        omega_s_exp = self.expander(omega_s)

        if self.reg_mode == 'visreg':
            l_reg, reg_components = visreg_loss(
                omega_s_exp,
                num_slices=self.visreg_num_slices,
                scale_weight=self.visreg_scale_weight,
                shape_weight=self.visreg_shape_weight,
                center_weight=self.visreg_center_weight,
            )
            total = self.inv_weight * l_inv + self.cov_weight * l_reg
            loss_dict = {
                'jepa_inv': float(l_inv.item()),
                'jepa_reg': float(l_reg.item()),
                'jepa_reg_scale': reg_components['scale'],
                'jepa_reg_shape': reg_components['shape'],
                'jepa_reg_center': reg_components['center'],
                'jepa_total': float(total.item()),
            }
        else:
            l_cov = covariance_loss(omega_s_exp)
            total = self.inv_weight * l_inv + self.cov_weight * l_cov
            loss_dict = {
                'jepa_inv': float(l_inv.item()),
                'jepa_cov': float(l_cov.item()),
                'jepa_total': float(total.item()),
            }

        return {
            'loss': total,
            'loss_dict': loss_dict,
            'omega_student': omega_s,
            'omega_teacher': omega_t,
        }
