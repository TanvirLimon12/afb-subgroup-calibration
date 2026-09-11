"""External-detector baseline wrappers for head-to-head comparison.

Provides an external state-of-the-art comparison alongside internal ablations. These adapters
wrap off-the-shelf detectors behind the ``AFBDetector`` interface so they share the
same tiled-inference, evaluation, and group-robust training path as the specialist
detector. A baseline that runs through ``tiled_inference`` + ``evaluate_detection``
is directly comparable to the FM-RobustAFB rows in the ablation table.

The closest published AFB competitor is Faster R-CNN ResNet-50 (Rulaningtyas et al.
2026, P43), so ``FasterRCNNBaseline`` is the primary external baseline. A YOLO-style
detector can be added behind the same interface.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .detector import DetectorOutputs


class _BaselineAdapter(nn.Module):
    """Common scaffolding: exposes forward()/loss()/predict() like AFBDetector.

    Subclasses set ``self.net`` to a torchvision detector and implement
    ``_decode`` to convert its eval output into the per-tile dict that
    ``tiled_inference`` expects (boxes, scores, logits, feats).
    """

    def __init__(self):
        super().__init__()
        self.net = None  # set by subclass
        self.num_classes = 1

    def forward(self, images: torch.Tensor) -> DetectorOutputs:
        # Cache the batched images so loss() can re-forward through the torchvision
        # detector with the targets it needs (torchvision's API is
        # net(images, targets) at train time). This keeps the train_detector loop
        # unchanged — it calls det(images) then det.loss(outputs, targets).
        self._cached_images = images
        n = images.shape[0]
        device = images.device
        return DetectorOutputs(
            cls_logits=[torch.zeros(n, 1, 1, 1, device=device)],
            centerness=[torch.zeros(n, 1, 1, 1, device=device)],
            bbox_reg=[torch.zeros(n, 1, 4, 1, 1, device=device)],
            cls_feat=[torch.zeros(n, 1, device=device)],
            shapes=[(1, 1)],
        )

    def loss(self, outputs: DetectorOutputs, targets: List[dict]) -> Dict[str, torch.Tensor]:
        # torchvision detectors compute their own loss given images+targets.
        images = self._cached_images
        tv_targets = []
        for t in targets:
            mask = t["boxes"].numel() > 0 and (t["boxes"][:, 2] > t["boxes"][:, 0]).any()
            boxes = t["boxes"] if mask else torch.zeros(0, 4, device=images.device)
            labels = t["labels"] if mask else torch.zeros(0, dtype=torch.int64, device=images.device)
            tv_targets.append({"boxes": boxes.float(), "labels": labels.long() + 1})  # bg=0, fg=1
        self.net.train()
        loss_dict = self.net(images, tv_targets)
        total = sum(loss_dict.values())
        return {"loss": total, "loss_cls": loss_dict.get("loss_classifier", total.detach()),
                "loss_box": loss_dict.get("loss_box_reg", total.detach()),
                "loss_ctr": torch.tensor(0.0, device=total.device)}

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode one tile's detections into the tiled_inference format."""
        self.eval()
        out = self.net([image])[0]
        return self._decode(out, image)

    def _decode(self, out, image):
        raise NotImplementedError


class FasterRCNNBaseline(_BaselineAdapter):
    """torchvision Faster R-CNN ResNet-50 FPN — the external SOTA baseline.

    Matches Rulaningtyas et al. 2026 (P43), the closest AFB competitor (single
    microscope, accuracy only). Run through the same eval path as FM-RobustAFB for
    a directly comparable worst-group AP row.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, min_size: int = 800):
        super().__init__()
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        weights = "DEFAULT" if pretrained else None
        # num_classes includes background (0); AFB foreground = 1
        self.net = fasterrcnn_resnet50_fpn(weights=weights, num_classes=num_classes,
                                           min_size=min_size, max_size=min_size)
        self.num_classes = num_classes

    def _decode(self, out, image):
        boxes = out["boxes"]
        scores = out["scores"]
        # torchvision returns scores already as probabilities; logits approximated
        # via logit for the fusion stage (which sanitizes them anyway).
        scores_clamped = scores.clamp(1e-6, 1 - 1e-6)
        logits = torch.log(scores_clamped / (1 - scores_clamped))
        # No feature embedding from the baseline (not used in detector-only rows).
        feats = torch.zeros(boxes.shape[0], 1, device=boxes.device) if boxes.numel() \
            else torch.zeros(0, 1, device=boxes.device)
        return {"boxes": boxes, "scores": scores, "logits": logits, "feats": feats}


def build_baseline(name: str = "faster_rcnn", **kwargs) -> _BaselineAdapter:
    """Factory for external baselines. Currently Faster R-CNN; extend for YOLO/RT-DETR."""
    if name == "faster_rcnn":
        return FasterRCNNBaseline(**kwargs)
    raise ValueError(f"unknown baseline: {name}")
