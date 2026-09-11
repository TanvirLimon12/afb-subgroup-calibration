"""Detection losses: focal (cls), IoU/GIoU (box), BCE (centerness)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def giou_loss(pred, target, eps=1e-7):
    """pred/target as xyxy boxes, elementwise (N,4). Returns per-box loss."""
    x1 = torch.max(pred[:, 0], target[:, 0])
    y1 = torch.max(pred[:, 1], target[:, 1])
    x2 = torch.min(pred[:, 2], target[:, 2])
    y2 = torch.min(pred[:, 3], target[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_t = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union
    cx1 = torch.min(pred[:, 0], target[:, 0])
    cy1 = torch.min(pred[:, 1], target[:, 1])
    cx2 = torch.max(pred[:, 2], target[:, 2])
    cy2 = torch.max(pred[:, 3], target[:, 3])
    enclose = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + eps
    giou = iou - (enclose - union) / enclose
    return 1 - giou


def centerness_target(ltrb: torch.Tensor) -> torch.Tensor:
    left_right = ltrb[:, [0, 2]]
    top_bottom = ltrb[:, [1, 3]]
    ctr = (left_right.min(dim=1).values / left_right.max(dim=1).values.clamp(min=1e-6)) * \
          (top_bottom.min(dim=1).values / top_bottom.max(dim=1).values.clamp(min=1e-6))
    return torch.sqrt(ctr.clamp(min=0))
