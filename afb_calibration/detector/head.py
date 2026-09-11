"""Anchor-free (FCOS-style) detection head.

The classification-tower output is retained per location as ``z_i`` (the
pre-classification-head embedding) so that confirmed candidates carry a detector
feature into the Stage-4 fusion input, not just a box and a score.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn


class Scale(nn.Module):
    def __init__(self, init: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def _tower(channels: int, n_convs: int) -> nn.Sequential:
    layers = []
    for _ in range(n_convs):
        layers += [nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                   nn.GroupNorm(8, channels), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, levels: List[int], n_convs: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.levels = levels
        self.cls_tower = _tower(in_channels, n_convs)
        self.reg_tower = _tower(in_channels, n_convs)
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.bbox_reg = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in levels])
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        prior = 0.01
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior) / prior))

    def forward(self, features: Dict[int, torch.Tensor]) -> Dict[str, list]:
        cls_logits, centerness, bbox_reg, cls_feat = [], [], [], []
        for i, lvl in enumerate(self.levels):
            feat = features[lvl]
            ct = self.cls_tower(feat)
            rt = self.reg_tower(feat)
            cls_logits.append(self.cls_logits(ct))
            centerness.append(self.centerness(rt))
            bbox_reg.append(torch.exp(self.scales[i](self.bbox_reg(rt))))
            cls_feat.append(ct)
        return {"cls_logits": cls_logits, "centerness": centerness,
                "bbox_reg": bbox_reg, "cls_feat": cls_feat}
