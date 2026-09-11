"""Bounding-box operations in xyxy format."""
from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import batched_nms, nms


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU. a:(N,4) b:(M,4) -> (N,M)."""
    area_a = box_area(a)
    area_b = box_area(b)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-9)


def clip_boxes(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    return boxes


def expand_boxes(boxes: torch.Tensor, factor: float, height: int, width: int) -> torch.Tensor:
    """Expand each box about its centre by ``factor`` (context ring for FM crops).

    Degenerate / non-xyxy inputs are sanitised first: negative or zero extent is
    clamped to a 1px minimum so the expanded box is always at least 1x1 in-bounds.
    This was the root cause of the ZNSM LOBO crash — a recovered annotation box
    with x2<x1 (a flipped rectangle from cv2.boundingRect on a thin contour)
    produced a negative width, expansion kept it negative, clip_boxes clamped
    both edges to the same coordinate, and F.interpolate got a 0-height crop.
    """
    boxes = boxes.clone()
    # Enforce xyxy ordering (x2>=x1, y2>=y1) with a 1px floor on each side.
    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = torch.maximum(boxes[:, 2], boxes[:, 0] + 1.0)
    y1 = torch.maximum(boxes[:, 3], boxes[:, 1] + 1.0)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    w = (x1 - x0) * factor
    h = (y1 - y0) * factor
    out = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
    return clip_boxes(out, height, width)


def nms_boxes(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    return nms(boxes, scores, iou_thresh)


def batched_nms_boxes(
    boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_thresh: float
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    return batched_nms(boxes, scores, idxs, iou_thresh)


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.clip(union, 1e-9, None)).astype(np.float32)
