"""ResNet + FPN backbone with an optional fine P2 (stride 4) level.

Standard one-stage detectors range over P3-P5 (finest stride 8) and drop
sub-8px structures. Including P2 is what preserves few-pixel bacilli; whether it
is switched on is a config decision justified by the Stage-1a box statistics.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork

_RESNET_LAYERS = {"layer1": 2, "layer2": 3, "layer3": 4, "layer4": 5}
_RESNET_CHANNELS = {
    "resnet18": {2: 64, 3: 128, 4: 256, 5: 512},
    "resnet34": {2: 64, 3: 128, 4: 256, 5: 512},
    "resnet50": {2: 256, 3: 512, 4: 1024, 5: 2048},
}


class FPNBackbone(nn.Module):
    def __init__(self, arch: str = "resnet18", pretrained: bool = False,
                 min_level: int = 2, max_level: int = 5, out_channels: int = 128):
        super().__init__()
        if arch not in _RESNET_CHANNELS:
            raise ValueError(f"unsupported backbone: {arch}")
        weights = "DEFAULT" if pretrained else None
        resnet = torchvision.models.__dict__[arch](weights=weights)
        self.min_level = min_level
        self.max_level = max_level
        self.levels = list(range(min_level, max_level + 1))
        self.strides = {lvl: 2 ** lvl for lvl in self.levels}

        return_layers = {name: f"c{lvl}" for name, lvl in _RESNET_LAYERS.items()
                         if min_level <= lvl <= max_level}
        self.body = IntermediateLayerGetter(resnet, return_layers=return_layers)
        in_channels = [_RESNET_CHANNELS[arch][lvl] for lvl in self.levels]
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels, out_channels=out_channels)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        feats = self.body(x)
        ordered = {f"c{lvl}": feats[f"c{lvl}"] for lvl in self.levels}
        pyramid = self.fpn(ordered)
        return {lvl: pyramid[f"c{lvl}"] for lvl in self.levels}

    @property
    def level_strides(self) -> List[int]:
        return [self.strides[lvl] for lvl in self.levels]
