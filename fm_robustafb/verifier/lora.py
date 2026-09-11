"""Parameter-efficient adaptation of a frozen foundation backbone.

LoRA (rank 8-16) or BitFit recover most of full fine-tuning's benefit on frozen
DINOv2 ViTs while updating under 1% of parameters - the right regime for a
dataset of ~1.4k images. Full fine-tuning is postponed to the journal track.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha or rank
        self.a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scaling = self.alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * (x @ self.a.t() @ self.b.t())


def _inject_lora(module: nn.Module, rank: int) -> int:
    count = 0
    for name, child in module.named_children():
        # nn.MultiheadAttention accesses its projection weights directly; leave
        # it intact. Real DINOv2 attention uses plain nn.Linear qkv/proj layers.
        if isinstance(child, nn.MultiheadAttention):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank))
            count += 1
        else:
            count += _inject_lora(child, rank)
    return count


def apply_adaptation(backbone: nn.Module, method: str, rank: int = 8) -> nn.Module:
    for p in backbone.parameters():
        p.requires_grad_(False)
    if method == "frozen":
        return backbone
    if method == "bitfit":
        for name, p in backbone.named_parameters():
            if name.endswith("bias"):
                p.requires_grad_(True)
        return backbone
    if method == "lora":
        _inject_lora(backbone, rank)
        return backbone
    raise ValueError(f"unknown adaptation: {method}")


def trainable_parameters(module: nn.Module):
    return [p for p in module.parameters() if p.requires_grad]
