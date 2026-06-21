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


class GOTJEPAWrapper(nn.Module):
    """
    Ties together the frozen teacher, trainable student, ProjNet, and
    Expander for GOT-JEPA pretraining.

    The caller is responsible for feeding the *clean* current-frame
    features to the teacher and the *corrupted* current-frame features
    to the student. Reference history is shared.
    """

    def __init__(
        self,
        student_predictor: PerTrackModelPredictor,
        teacher_predictor: Optional[PerTrackModelPredictor] = None,
        expander_dim_mul: int = 4,
        inv_weight: float = 1.0,
        cov_weight: float = 0.5,
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

        # Covariance: expand student output, penalise off-diagonal covariance.
        omega_s_exp = self.expander(omega_s)
        l_cov = covariance_loss(omega_s_exp)

        total = self.inv_weight * l_inv + self.cov_weight * l_cov

        return {
            'loss': total,
            'loss_dict': {
                'jepa_inv': float(l_inv.item()),
                'jepa_cov': float(l_cov.item()),
                'jepa_total': float(total.item()),
            },
            'omega_student': omega_s,
            'omega_teacher': omega_t,
        }
