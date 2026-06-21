"""
Minimal LoRA (Low-Rank Adaptation) helpers for fine-tuning frozen DINOv2.

This avoids adding a heavy dependency like ``peft``; it is intentionally
small and self-contained. LoRA is injected into selected linear layers of a
pretrained ViT; base weights stay frozen while only low-rank A/B matrices train.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
ICLR 2022.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Wrap a ``nn.Linear`` layer with a LoRA bypass.

    Forward: ``y = base(x) + (alpha / r) * B(A(x))``.

    Args:
        base_layer: the pretrained linear layer to freeze.
        r: LoRA rank; 0 means disabled (keeps base frozen).
        alpha: scaling factor; typically ``alpha == r`` or ``alpha == 2*r``.
        dropout: dropout applied to the input before the low-rank path.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0

        # Freeze base layer
        for param in self.base.parameters():
            param.requires_grad = False

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        if r > 0:
            self.lora_A = nn.Parameter(
                torch.zeros(self.in_features, r)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(r, self.out_features)
            )
            # Dropout on the low-rank path input
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            # Initialize A with Kaiming uniform, B with zeros -> zero output at start
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.lora_A is not None and self.lora_B is not None:
            # Low-rank update: x @ A @ B
            lora_x = self.dropout(x)
            out = out + (lora_x @ self.lora_A @ self.lora_B) * self.scaling
        return out

    def merge(self) -> nn.Linear:
        """
        Return a standard ``nn.Linear`` with merged weights (base + LoRA).

        Useful for inference speed-up after training. The returned layer is
        a *new* object; the wrapper remains unchanged.
        """
        merged = nn.Linear(self.in_features, self.out_features, bias=self.base.bias is not None)
        merged.weight.data = self.base.weight.data.clone()
        if self.base.bias is not None:
            merged.bias.data = self.base.bias.data.clone()
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            delta = (self.lora_A @ self.lora_B).T * self.scaling
            merged.weight.data = merged.weight.data + delta.to(merged.weight.device)
        return merged

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, " \
               f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}, " \
               f"dropout={self.dropout.p if hasattr(self.dropout, 'p') else 0.0}"


def inject_lora_into_linear(
    parent_module: nn.Module,
    attribute_name: str,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> None:
    """Replace ``parent_module.attribute_name`` (nn.Linear) with a LoRA wrapper."""
    base_layer = getattr(parent_module, attribute_name)
    if not isinstance(base_layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(base_layer)}")
    wrapped = LoRALinear(base_layer, r=r, alpha=alpha, dropout=dropout)
    setattr(parent_module, attribute_name, wrapped)


def inject_lora_into_dinov2(
    encoder: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    start_block: int = 0,
    end_block: Optional[int] = None,
) -> None:
    """
    Inject LoRA into a DINOv2 (or generic ViT) encoder.

    Targets the attention qkv/projection and optionally MLP layers inside each
    transformer block. The base weights remain frozen; only the LoRA parameters
    get ``requires_grad=True``.

    Args:
        encoder: the loaded DINOv2 model (e.g. from torch.hub).
        r: LoRA rank.
        alpha: LoRA scaling.
        dropout: dropout on the LoRA path.
        target_modules: list of submodule names to wrap; defaults to
            ``["qkv", "proj", "fc1", "fc2"]``.
        start_block: first block index to modify (default 0).
        end_block: exclusive end index (default all blocks).
    """
    if target_modules is None:
        target_modules = ["qkv", "proj", "fc1", "fc2"]

    # DINOv2 blocks live in encoder.blocks
    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        raise RuntimeError("Could not find encoder.blocks; unsupported encoder architecture")

    end_block = end_block if end_block is not None else len(blocks)
    modified = 0
    for idx in range(start_block, end_block):
        block = blocks[idx]
        for name in target_modules:
            module = getattr(block, name, None)
            if isinstance(module, nn.Linear):
                inject_lora_into_linear(block, name, r=r, alpha=alpha, dropout=dropout)
                modified += 1
            elif hasattr(block, "attn") and hasattr(block.attn, name):
                # Some DINOv2 variants nest qkv/proj under block.attn
                submodule = getattr(block.attn, name)
                if isinstance(submodule, nn.Linear):
                    inject_lora_into_linear(block.attn, name, r=r, alpha=alpha, dropout=dropout)
                    modified += 1
    print(f"Injected LoRA into {modified} linear layers (rank={r}, alpha={alpha})")


def get_lora_params(module: nn.Module) -> List[torch.nn.Parameter]:
    """Return all trainable LoRA parameters in a module."""
    params: List[torch.nn.Parameter] = []
    for m in module.modules():
        if isinstance(m, LoRALinear):
            if m.lora_A is not None:
                params.append(m.lora_A)
            if m.lora_B is not None:
                params.append(m.lora_B)
    return params


def lora_state_dict(module: nn.Module) -> Dict[str, Any]:
    """Return only the LoRA parameters and their names (for checkpointing)."""
    state: Dict[str, Any] = {}
    for name, m in module.named_modules():
        if isinstance(m, LoRALinear):
            if m.lora_A is not None:
                state[f"{name}.lora_A"] = m.lora_A.data
            if m.lora_B is not None:
                state[f"{name}.lora_B"] = m.lora_B.data
    return state
