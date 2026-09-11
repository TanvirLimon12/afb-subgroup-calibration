"""Point geometry, box decoding, and per-tile detection decoding."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from ..utils.boxes import nms_boxes


def build_points(
    feature_shapes: List[Tuple[int, int]], strides: List[int], device
) -> Tuple[torch.Tensor, torch.Tensor]:
    points, point_strides = [], []
    for (h, w), s in zip(feature_shapes, strides):
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
        )
        px = (xs.reshape(-1) + 0.5) * s
        py = (ys.reshape(-1) + 0.5) * s
        points.append(torch.stack([px, py], dim=1))
        point_strides.append(torch.full((h * w,), s, device=device, dtype=torch.float32))
    return torch.cat(points, 0), torch.cat(point_strides, 0)


def decode_boxes(points: torch.Tensor, strides: torch.Tensor, reg: torch.Tensor) -> torch.Tensor:
    dist = reg * strides[:, None]
    x1 = points[:, 0] - dist[:, 0]
    y1 = points[:, 1] - dist[:, 1]
    x2 = points[:, 0] + dist[:, 2]
    y2 = points[:, 1] + dist[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=1)


def _flatten_level(t: torch.Tensor) -> torch.Tensor:
    c = t.shape[0]
    return t.reshape(c, -1).permute(1, 0)


def decode_detections(
    cls_logits: List[torch.Tensor], centerness: List[torch.Tensor], bbox_reg: List[torch.Tensor],
    cls_feat: List[torch.Tensor], points: torch.Tensor, strides: torch.Tensor,
    score_thresh: float = 0.05, nms_iou: float = 0.5, max_det: int = 300,
) -> Dict[str, torch.Tensor]:
    cls = torch.cat([_flatten_level(x) for x in cls_logits], 0)
    ctr = torch.cat([_flatten_level(x) for x in centerness], 0)
    reg = torch.cat([_flatten_level(x) for x in bbox_reg], 0)
    feat = torch.cat([_flatten_level(x) for x in cls_feat], 0)

    boxes = decode_boxes(points, strides, reg)
    cls_prob, cls_idx = torch.sigmoid(cls).max(dim=1)
    quality = cls_prob * torch.sigmoid(ctr[:, 0])
    logit = cls.gather(1, cls_idx[:, None]).squeeze(1)

    keep = quality >= score_thresh
    boxes, scores = boxes[keep], quality[keep]
    logit, feat = logit[keep], feat[keep]
    if boxes.numel() == 0:
        return {"boxes": boxes, "scores": scores, "logits": logit, "feats": feat}

    order = scores.argsort(descending=True)[: max_det * 4]
    boxes, scores, logit, feat = boxes[order], scores[order], logit[order], feat[order]
    kept = nms_boxes(boxes, scores, nms_iou)[:max_det]
    return {"boxes": boxes[kept], "scores": scores[kept], "logits": logit[kept], "feats": feat[kept]}
