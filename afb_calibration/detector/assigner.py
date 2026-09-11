"""SimOTA dynamic label assignment (YOLOX-style).

Dynamic, task-aligned assignment handles tiny, dense, elongated objects better
than fixed IoU-threshold assignment.
"""
from __future__ import annotations

from typing import Tuple

import torch

from ..utils.boxes import box_iou


class SimOTAAssigner:
    def __init__(self, center_radius: float = 2.5, candidate_topk: int = 10,
                 cls_weight: float = 1.0, iou_weight: float = 3.0):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.cls_weight = cls_weight
        self.iou_weight = iou_weight

    @torch.no_grad()
    def __call__(
        self, points: torch.Tensor, strides: torch.Tensor, cls_quality: torch.Tensor,
        decoded_boxes: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (fg_mask[L], matched_gt_idx[num_fg], matched_labels[num_fg])."""
        num_points = points.shape[0]
        device = points.device
        if gt_boxes.numel() == 0:
            return (torch.zeros(num_points, dtype=torch.bool, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device))

        geom_mask, is_in_center = self._geometry(points, strides, gt_boxes)
        valid = geom_mask.any(dim=1)
        if valid.sum() == 0:
            return (torch.zeros(num_points, dtype=torch.bool, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device))

        valid_boxes = decoded_boxes[valid]
        ious = box_iou(valid_boxes, gt_boxes).clamp(min=1e-8)
        iou_cost = -torch.log(ious)

        q = cls_quality[valid].clamp(1e-6, 1 - 1e-6)
        gt_onehot = torch.zeros((int(valid.sum()), gt_boxes.shape[0]), device=device)
        cls_cost = torch.nn.functional.binary_cross_entropy(
            q.expand(-1, gt_boxes.shape[0]) if q.shape[1] == 1 else q[:, gt_labels],
            torch.ones_like(iou_cost), reduction="none",
        )

        center_pen = (~is_in_center[valid]).float() * 1e5
        cost = self.cls_weight * cls_cost + self.iou_weight * iou_cost + center_pen

        matched_gt, fg_local = self._dynamic_k(cost, ious)
        fg_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        fg_mask[valid_idx[fg_local]] = True
        return fg_mask, matched_gt, gt_labels[matched_gt]

    def _geometry(self, points, strides, gt_boxes):
        px = points[:, 0:1]
        py = points[:, 1:2]
        l = px - gt_boxes[:, 0][None, :]
        t = py - gt_boxes[:, 1][None, :]
        r = gt_boxes[:, 2][None, :] - px
        b = gt_boxes[:, 3][None, :] - py
        in_box = torch.stack([l, t, r, b], dim=-1).min(dim=-1).values > 0

        gcx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
        gcy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
        radius = strides[:, None] * self.center_radius
        in_center = ((px - gcx[None, :]).abs() < radius) & ((py - gcy[None, :]).abs() < radius)
        return in_box | in_center, in_box & in_center

    def _dynamic_k(self, cost, ious):
        matching = torch.zeros_like(cost, dtype=torch.bool)
        topk = min(self.candidate_topk, ious.shape[0])
        topk_ious, _ = torch.topk(ious, topk, dim=0)
        dynamic_ks = topk_ious.sum(dim=0).int().clamp(min=1)
        for g in range(cost.shape[1]):
            _, idx = torch.topk(cost[:, g], k=int(dynamic_ks[g]), largest=False)
            matching[idx, g] = True

        multi = matching.sum(dim=1) > 1
        if multi.any():
            min_cost_gt = cost[multi].argmin(dim=1)
            matching[multi] = False
            matching[torch.nonzero(multi, as_tuple=False).flatten(), min_cost_gt] = True

        fg = matching.any(dim=1)
        matched_gt = matching[fg].float().argmax(dim=1)
        return matched_gt, fg
