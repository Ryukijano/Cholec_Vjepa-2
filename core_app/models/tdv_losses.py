"""
TDV Losses — adapted from the official TDV repo.
https://github.com/ninaddaithankar/tdv

Center-sharpened MSE reconstruction loss and DINO-style cross-entropy loss
for self-distillation with EMA teacher.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm


class CenterSharpReconstructionLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        predicted_temp: float = 1.0,
        target_temp: float = 1.0,
        center_momentum: float = 0.99,
        loss: str = "mse",
    ):
        super().__init__()
        self.predicted_temp = predicted_temp
        self.target_temp = target_temp
        self.center_momentum = center_momentum
        self.loss_fn = nn.MSELoss() if loss == "mse" else nn.SmoothL1Loss()
        self.register_buffer("center", torch.zeros(1, 1, out_dim))

    @torch.no_grad()
    def _update_center(self, target_logits, sync_center=True):
        dims = tuple(range(target_logits.ndim - 1))
        batch_center = target_logits.mean(dim=dims, keepdim=False)
        batch_center = batch_center.reshape(1, 1, -1)
        assert batch_center.shape[-1] == self.center.shape[-1]
        if sync_center and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(batch_center, op=torch.distributed.ReduceOp.SUM)
            batch_center /= torch.distributed.get_world_size()
        self.center.mul_(self.center_momentum).add_(batch_center * (1 - self.center_momentum))

    def forward(self, predicted, target, use_centering=True, use_sharpening=True, sync_center=True):
        raw_target_logits = target
        if use_centering:
            target = target - self.center
        if use_sharpening:
            target = target / self.target_temp
            predicted = predicted / self.predicted_temp
        loss = self.loss_fn(predicted, target)
        if use_centering:
            with torch.no_grad():
                self._update_center(raw_target_logits, sync_center)
        return loss


class DinoLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.eps = 1e-5
        self.register_buffer("center", torch.zeros(1, 1, out_dim))

    @torch.no_grad()
    def _update_center(self, teacher_logits, sync_center=True):
        dims = tuple(range(teacher_logits.ndim - 1))
        batch_center = teacher_logits.mean(dim=dims, keepdim=False)
        batch_center = batch_center.view_as(self.center)
        if sync_center and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(batch_center, op=torch.distributed.ReduceOp.SUM)
            batch_center /= torch.distributed.get_world_size()
        self.center.mul_(self.center_momentum).add_(batch_center * (1 - self.center_momentum))

    def forward(self, student_logits, teacher_logits, use_centering=True, use_sharpening=True, sync_center=True, token_weights=None):
        raw_teacher_logits = teacher_logits
        if use_centering:
            teacher_logits = teacher_logits - self.center
        if use_sharpening:
            teacher_logits = teacher_logits / self.teacher_temp
            student_logits = student_logits / self.student_temp
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        per_token_loss = -torch.sum(teacher_probs * student_log_probs, dim=-1)
        weight_sum = None
        if token_weights is not None:
            weight_sum = token_weights.sum().clamp(min=1e-6)
            loss = (per_token_loss * token_weights).sum() / weight_sum
        else:
            loss = per_token_loss.mean()
        with torch.no_grad():
            per_token_entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
            per_token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
            if token_weights is not None and weight_sum is not None:
                entropy = (per_token_entropy * token_weights).sum() / weight_sum
                kl_div = (per_token_kl * token_weights).sum() / weight_sum
            else:
                entropy = per_token_entropy.mean()
                kl_div = per_token_kl.mean()
            if use_centering:
                self._update_center(raw_teacher_logits, sync_center)
        return loss, entropy.detach(), kl_div.detach()


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, nlayers=3, hidden_dim=2048, bottleneck_dim=256, mlp_bias=True):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
    if use_bn:
        layers.append(nn.BatchNorm1d(hidden_dim))
    layers.append(nn.GELU())
    for _ in range(nlayers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
    layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
    return nn.Sequential(*layers)
