"""Overlapping high-resolution tiling and cross-tile detection fusion.

AFB can be a few pixels wide in the full field, so images are tiled at native
resolution (no pre-tiling downsampling). Because tiles overlap, NMS is applied in
original-image coordinates after inference, not per tile, so a bacillus at a tile
seam is not double-counted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from ..utils.boxes import nms_boxes


@dataclass
class TileGrid:
    tiles: List[Tuple[int, int, int, int]]
    image_h: int
    image_w: int

    def __len__(self) -> int:
        return len(self.tiles)


def build_grid(image_h: int, image_w: int, tile: int, overlap: float) -> TileGrid:
    stride = max(1, int(round(tile * (1.0 - overlap))))
    xs = _axis_offsets(image_w, tile, stride)
    ys = _axis_offsets(image_h, tile, stride)
    tiles = []
    for y0 in ys:
        for x0 in xs:
            tiles.append((x0, y0, min(x0 + tile, image_w), min(y0 + tile, image_h)))
    return TileGrid(tiles=tiles, image_h=image_h, image_w=image_w)


def _axis_offsets(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    offsets = list(range(0, length - tile + 1, stride))
    if offsets[-1] != length - tile:
        offsets.append(length - tile)
    return offsets


def tile_image(image: torch.Tensor, grid: TileGrid) -> List[torch.Tensor]:
    """Return list of tile crops (C, h, w) for a (C, H, W) image."""
    crops = []
    for x0, y0, x1, y1 in grid.tiles:
        crops.append(image[:, y0:y1, x0:x1])
    return crops


def fuse_tile_detections(
    tile_dets: List[dict], grid: TileGrid, iou_thresh: float
) -> dict:
    """Merge per-tile detections into image coordinates and apply cross-tile NMS.

    Each element of ``tile_dets`` has keys ``boxes`` (n,4 in tile coords),
    ``scores`` (n,), and optionally ``logits`` (n,) and ``feats`` (n,d).
    """
    boxes, scores, logits, feats = [], [], [], []
    has_logits = all("logits" in d for d in tile_dets) and len(tile_dets) > 0
    has_feats = all("feats" in d for d in tile_dets) and len(tile_dets) > 0
    for det, (x0, y0, _, _) in zip(tile_dets, grid.tiles):
        if det["boxes"].numel() == 0:
            continue
        shift = torch.tensor([x0, y0, x0, y0], dtype=det["boxes"].dtype, device=det["boxes"].device)
        boxes.append(det["boxes"] + shift)
        scores.append(det["scores"])
        if has_logits:
            logits.append(det["logits"])
        if has_feats:
            feats.append(det["feats"])

    if not boxes:
        return {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros((0,)),
            "logits": torch.zeros((0,)),
            "feats": torch.zeros((0, 0)),
        }

    boxes_t = torch.cat(boxes, 0)
    scores_t = torch.cat(scores, 0)
    boxes_t[:, 0::2] = boxes_t[:, 0::2].clamp(0, grid.image_w)
    boxes_t[:, 1::2] = boxes_t[:, 1::2].clamp(0, grid.image_h)
    keep = nms_boxes(boxes_t, scores_t, iou_thresh)
    out = {"boxes": boxes_t[keep], "scores": scores_t[keep]}
    out["logits"] = torch.cat(logits, 0)[keep] if has_logits else out["scores"].logit().clamp(-20, 20)
    if has_feats:
        out["feats"] = torch.cat(feats, 0)[keep]
    return out
