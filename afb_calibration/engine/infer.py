"""Tiled inference with cross-tile box fusion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from ..data.tiling import build_grid, fuse_tile_detections, tile_image
from ..detector import AFBDetector


@dataclass
class InferenceResult:
    image_id: int
    group: int
    boxes: np.ndarray
    scores: np.ndarray
    det_logits: np.ndarray
    feats: np.ndarray
    gt_boxes: np.ndarray
    meta: dict


@torch.no_grad()
def tiled_inference(
    detector: AFBDetector, dataset, device: torch.device, tile_size: int = 640,
    overlap: float = 0.18, fuse_iou: float = 0.5, corruption=None, max_images: Optional[int] = None,
) -> List[InferenceResult]:
    detector.eval()
    results = []
    n = len(dataset) if max_images is None else min(max_images, len(dataset))
    for i in range(n):
        sample = dataset[i]
        image = sample["image"]
        if corruption is not None:
            image = corruption(image)
        _, h, w = image.shape
        grid = build_grid(h, w, tile_size, overlap)
        tile_dets = []
        for crop in tile_image(image, grid):
            tile_dets.append(detector.predict(crop.to(device)))
        fused = fuse_tile_detections(tile_dets, grid, fuse_iou)
        results.append(InferenceResult(
            image_id=sample["image_id"], group=sample["group"],
            boxes=fused["boxes"].cpu().numpy(), scores=fused["scores"].cpu().numpy(),
            det_logits=fused["logits"].cpu().numpy(), feats=fused["feats"].cpu().numpy(),
            gt_boxes=sample["boxes"].numpy(), meta=sample["meta"],
        ))
    return results


def as_predictions(results: List[InferenceResult]):
    preds = [{"boxes": r.boxes, "scores": r.scores} for r in results]
    gts = [r.gt_boxes for r in results]
    groups = [r.group for r in results]
    return preds, gts, groups
