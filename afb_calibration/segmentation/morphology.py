"""Shape-based mask scoring for sub-cellular rods.

Box-IoU penalises a correctly segmented thin rod for not filling its rectangle,
so segmentation support is scored by containment / elongation / skeleton length /
width consistency / connected-component count / orientation alignment - not IoU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MorphologyScore:
    containment: float
    elongation: float
    skeleton_length: float
    width_consistency: float
    num_components: int
    orientation_align: float
    foreground_ratio: float


def _principal_axes(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if xs.size < 2:
        return 0.0, 0.0, 0.0
    coords = np.stack([xs - xs.mean(), ys - ys.mean()], axis=0).astype(np.float64)
    cov = coords @ coords.T / coords.shape[1]
    evals, evecs = np.linalg.eigh(cov)
    major = float(np.sqrt(max(evals[1], 1e-9)))
    minor = float(np.sqrt(max(evals[0], 1e-9)))
    angle = float(np.arctan2(evecs[1, 1], evecs[0, 1]))
    return major, minor, angle


def mask_morphology(mask: np.ndarray, box: np.ndarray) -> MorphologyScore:
    from skimage.measure import label
    from skimage.morphology import skeletonize

    mask = mask.astype(bool)
    area = mask.sum()
    box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    inside = mask[max(0, y0):max(1, y1), max(0, x0):max(1, x1)].sum()
    containment = float(inside) / float(max(1, area))

    major, minor, angle = _principal_axes(mask)
    elongation = major / max(minor, 1e-6)

    skel = skeletonize(mask) if area > 0 else np.zeros_like(mask)
    skeleton_length = float(skel.sum())
    width_consistency = float(area) / max(1.0, skeleton_length)

    labeled = label(mask)
    num_components = int(labeled.max())

    box_angle = 0.0 if (x1 - x0) >= (y1 - y0) else np.pi / 2
    orientation_align = float(abs(np.cos(angle - box_angle)))

    return MorphologyScore(
        containment=containment, elongation=elongation, skeleton_length=skeleton_length,
        width_consistency=width_consistency, num_components=num_components,
        orientation_align=orientation_align, foreground_ratio=float(area) / box_area,
    )


def filter_pseudo_mask(
    score: MorphologyScore, containment_min: float = 0.7, elongation_min: float = 2.0,
    fg_ratio_band=(0.05, 0.9), max_components: int = 2,
) -> bool:
    return (
        score.containment >= containment_min
        and score.elongation >= elongation_min
        and fg_ratio_band[0] <= score.foreground_ratio <= fg_ratio_band[1]
        and 1 <= score.num_components <= max_components
    )
