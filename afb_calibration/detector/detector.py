"""Full high-resolution specialist detector (Stage 1b)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DetectorConfig
from .assigner import SimOTAAssigner
from .backbone import FPNBackbone
from .head import DetectionHead
from .losses import centerness_target, giou_loss, sigmoid_focal_loss
from .postprocess import build_points, decode_boxes, decode_detections


@dataclass
class DetectorOutputs:
    cls_logits: List[torch.Tensor]
    centerness: List[torch.Tensor]
    bbox_reg: List[torch.Tensor]
    cls_feat: List[torch.Tensor]
    shapes: List[Tuple[int, int]]


def pad_batch(images: List[torch.Tensor], size_divisible: int = 32) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    sizes = [(im.shape[1], im.shape[2]) for im in images]
    max_h = max(s[0] for s in sizes)
    max_w = max(s[1] for s in sizes)
    max_h = int(((max_h + size_divisible - 1) // size_divisible) * size_divisible)
    max_w = int(((max_w + size_divisible - 1) // size_divisible) * size_divisible)
    batched = images[0].new_zeros((len(images), images[0].shape[0], max_h, max_w))
    for i, im in enumerate(images):
        batched[i, :, : im.shape[1], : im.shape[2]] = im
    return batched, sizes


class AFBDetector(nn.Module):
    def __init__(self, cfg: DetectorConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = FPNBackbone(cfg.backbone, cfg.pretrained, cfg.fpn_min_level,
                                    cfg.fpn_max_level, cfg.fpn_channels)
        self.levels = self.backbone.levels
        self.level_strides = self.backbone.level_strides
        self.head = DetectionHead(cfg.fpn_channels, cfg.num_classes, self.levels, cfg.head_convs)
        self.assigner = SimOTAAssigner(cfg.assigner.center_radius, cfg.assigner.candidate_topk)
        self.num_classes = cfg.num_classes

    def forward(self, images: torch.Tensor) -> DetectorOutputs:
        feats = self.backbone(images)
        out = self.head(feats)
        shapes = [tuple(x.shape[-2:]) for x in out["cls_logits"]]
        return DetectorOutputs(out["cls_logits"], out["centerness"], out["bbox_reg"],
                               out["cls_feat"], shapes)

    @staticmethod
    def _flatten(tensors: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([t.flatten(2).permute(0, 2, 1) for t in tensors], dim=1)

    def loss(self, outputs: DetectorOutputs, targets: List[dict]) -> Dict[str, torch.Tensor]:
        device = outputs.cls_logits[0].device
        points, strides = build_points(outputs.shapes, self.level_strides, device)
        cls = self._flatten(outputs.cls_logits)
        ctr = self._flatten(outputs.centerness)
        reg = self._flatten(outputs.bbox_reg)

        b = cls.shape[0]
        w = self.cfg.loss
        per_image = []
        cls_terms, box_terms, ctr_terms = [], [], []
        for i in range(b):
            gt_boxes = targets[i]["boxes"].to(device)
            gt_labels = targets[i]["labels"].to(device)
            decoded = decode_boxes(points, strides, reg[i])
            quality = torch.sigmoid(cls[i]) * torch.sigmoid(ctr[i])
            fg_mask, matched_gt, matched_labels = self.assigner(
                points, strides, quality, decoded, gt_boxes, gt_labels)
            cls_target = torch.zeros_like(cls[i])
            box_loss = cls.new_zeros(())
            ctr_loss = cls.new_zeros(())
            n_pos = 1
            if fg_mask.any():
                pos_idx = torch.nonzero(fg_mask, as_tuple=False).flatten()
                n_pos = pos_idx.numel()
                cls_target[pos_idx, matched_labels] = 1.0
                gt_matched = gt_boxes[matched_gt]
                box_loss = giou_loss(decoded[pos_idx], gt_matched).sum() / n_pos
                ltrb = self._ltrb(points[pos_idx], strides[pos_idx], gt_matched)
                ctr_tgt = centerness_target(ltrb)
                ctr_loss = F.binary_cross_entropy_with_logits(
                    ctr[i, pos_idx, 0], ctr_tgt, reduction="sum") / n_pos
            cls_loss = sigmoid_focal_loss(cls[i], cls_target, reduction="sum") / n_pos
            cls_terms.append(cls_loss)
            box_terms.append(box_loss)
            ctr_terms.append(ctr_loss)
            per_image.append(w.cls_weight * cls_loss + w.box_weight * box_loss + w.ctr_weight * ctr_loss)

        per_image = torch.stack(per_image)
        return {
            "loss": per_image.mean(),
            "per_image": per_image,
            "loss_cls": torch.stack(cls_terms).mean(),
            "loss_box": torch.stack(box_terms).mean(),
            "loss_ctr": torch.stack(ctr_terms).mean(),
        }

    @staticmethod
    def _ltrb(points, strides, boxes):
        l = points[:, 0] - boxes[:, 0]
        t = points[:, 1] - boxes[:, 1]
        r = boxes[:, 2] - points[:, 0]
        b = boxes[:, 3] - points[:, 1]
        return torch.stack([l, t, r, b], dim=1).clamp(min=0)

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode detections for a single (C, H, W) tile."""
        self.eval()
        batched, _ = pad_batch([image])
        out = self.forward(batched)
        points, strides = build_points(out.shapes, self.level_strides, image.device)
        single = [x[0] for x in out.cls_logits], [x[0] for x in out.centerness], \
                 [x[0] for x in out.bbox_reg], [x[0] for x in out.cls_feat]
        return decode_detections(*single, points, strides, self.cfg.score_thresh,
                                 self.cfg.nms_iou, self.cfg.max_det_per_tile)
