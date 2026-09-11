"""Stage-2 DINO candidate verifier.

DINO has exactly one role here: a frozen or lightly-adapted verifier that
classifies detector-proposed crops as bacillus vs artifact. It is not
simultaneously a backbone, a distillation teacher, and a prototype engine.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import VerifierConfig
from ..utils.boxes import expand_boxes
from .dino import DinoFeatureExtractor, build_dino
from .lora import apply_adaptation, trainable_parameters
from .prototypes import build_head


def extract_crops(image: torch.Tensor, boxes: torch.Tensor, context: float, size: int) -> torch.Tensor:
    """Crop tight box + context ring per candidate and resize to ``size``.

    Never resizes the full field: at patch size 14 the bacillus must span
    multiple patches after resizing.
    """
    _, h, w = image.shape
    if boxes.numel() == 0:
        return image.new_zeros((0, 3, size, size))
    exp = expand_boxes(boxes, context, h, w)
    crops = []
    for x0, y0, x1, y1 in exp.round().long().tolist():
        # Belt-and-suspenders guard: every crop must be at least 1x1 in-bounds,
        # regardless of what the caller or expand_boxes produced. This is the
        # boundary that protects F.interpolate from a 0-size input (the ZNSM
        # LOBO crash); expand_boxes already enforces this upstream but we keep
        # the guard so a future caller cannot reintroduce the crash.
        x0 = int(min(max(x0, 0), w - 1))
        y0 = int(min(max(y0, 0), h - 1))
        x1 = int(min(max(x1, x0 + 1), w))
        y1 = int(min(max(y1, y0 + 1), h))
        crop = image[:, y0:y1, x0:x1].unsqueeze(0)
        crops.append(F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False))
    return torch.cat(crops, dim=0)


class DINOVerifier(nn.Module):
    def __init__(self, cfg: VerifierConfig):
        super().__init__()
        self.cfg = cfg
        backbone = build_dino(cfg.backbone, cfg.offline_stub)
        backbone = apply_adaptation(backbone, cfg.adaptation, cfg.lora_rank)
        self.extractor = DinoFeatureExtractor(backbone, cfg.fuse_layers)
        self.head = build_head(cfg.head, self.extractor.embed_dim, cfg.prototypes_per_class)

    def features(self, crops: torch.Tensor) -> torch.Tensor:
        return self.extractor(crops)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(crops))

    def verify_boxes(self, image: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        crops = extract_crops(image, boxes, self.cfg.crop_context, self.cfg.crop_size)
        if crops.numel() == 0:
            return boxes.new_zeros((0,))
        return self.forward(crops)

    def trainable_parameters(self) -> List[nn.Parameter]:
        params = trainable_parameters(self.extractor)
        params += list(self.head.parameters())
        return params
