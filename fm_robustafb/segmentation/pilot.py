"""Stage-2a segmentation pilot (gated, off by default).

SAM-family models have not been validated on sub-cellular rod-shaped AFB on raw
stained backgrounds; treat the pilot's success probability as low. The core claim
stands without this branch (DINO is the verifier; SAM is optional morphology
support only). Pseudo-mask supervision is retained only if the μSAM/CellSAM
consensus clears Dice/containment/shape thresholds on a hand-segmented pilot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from ..utils.logging import get_logger
from .morphology import filter_pseudo_mask, mask_morphology

logger = get_logger(__name__)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * (a & b).sum() / denom)


def consensus_mask(masks: List[np.ndarray]) -> np.ndarray:
    stack = np.stack([m.astype(np.float32) for m in masks], axis=0)
    return (stack.mean(axis=0) >= 0.5)


class SegmenterBackend:
    """Prompt a box-conditioned segmenter. Real backends (micro_sam, CellSAM) are
    optional; a redness-threshold stub keeps the pilot logic runnable offline."""

    def __init__(self, name: str, device: str = "cpu"):
        self.name = name
        self.device = device
        self._predictor = None
        self._impl = self._load(name)

    def _load(self, name: str) -> Callable:
        if name == "micro_sam":
            try:
                from micro_sam.util import get_sam_model
                from segment_anything import SamPredictor

                model = get_sam_model(model_type="vit_b_lm", device=self.device)
                self._predictor = SamPredictor(model)
                return self._micro_sam
            except Exception as exc:
                logger.warning("micro_sam unavailable (%s); using stub segmenter", exc)
        elif name == "cellsam":
            try:
                from cellSAM import segment_cellular_image
                self._cellsam_fn = segment_cellular_image
                return self._cellsam
            except Exception as exc:
                logger.warning("cellSAM unavailable (%s); using stub segmenter", exc)
        return self._stub

    def segment(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        return self._impl(image, box)

    @staticmethod
    def _as_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image
        return np.clip(image * (255.0 if image.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)

    def _stub(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        crop = image[max(0, y0):min(h, y1), max(0, x0):min(w, x1)].astype(np.float32)
        if crop.size == 0:
            return mask
        redness = crop[..., 0] - 0.5 * (crop[..., 1] + crop[..., 2])
        thr = redness.mean() + 0.5 * redness.std()
        mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = redness > thr
        return mask

    def _micro_sam(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        self._predictor.set_image(self._as_uint8(image))
        masks, scores, _ = self._predictor.predict(
            box=np.asarray(box, dtype=np.float32), multimask_output=False)
        return masks[int(np.argmax(scores))].astype(bool)

    def _cellsam(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        labels = self._cellsam_fn(self._as_uint8(image), device=self.device, normalize=True)
        labels = labels[0] if isinstance(labels, tuple) else labels
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        region = labels[max(0, y0):max(1, y1), max(0, x0):max(1, x1)]
        ids, counts = np.unique(region[region > 0], return_counts=True)
        if ids.size == 0:
            return np.zeros(labels.shape, dtype=bool)
        return labels == ids[int(np.argmax(counts))]


@dataclass
class PilotResult:
    mean_dice_gt: Dict[str, float]
    consensus_dice_gt: float
    kept_fraction: float
    passed: bool


def run_pilot(
    samples: List[dict], backends: List[SegmenterBackend],
    consensus_dice_min: float = 0.7, containment_min: float = 0.7, elongation_min: float = 2.0,
) -> PilotResult:
    """``samples``: list of {image (H,W,3), box (4,), gt_mask (H,W)}."""
    per_backend: Dict[str, List[float]] = {b.name: [] for b in backends}
    consensus_scores, kept = [], 0
    for s in samples:
        masks = []
        for b in backends:
            m = b.segment(s["image"], s["box"])
            per_backend[b.name].append(dice(m, s["gt_mask"]))
            masks.append(m)
        cons = consensus_mask(masks)
        consensus_scores.append(dice(cons, s["gt_mask"]))
        morph = mask_morphology(cons, s["box"])
        if filter_pseudo_mask(morph, containment_min, elongation_min):
            kept += 1

    mean_dice = {k: float(np.mean(v)) if v else 0.0 for k, v in per_backend.items()}
    consensus_dice = float(np.mean(consensus_scores)) if consensus_scores else 0.0
    passed = consensus_dice >= consensus_dice_min
    logger.info("pilot consensus Dice=%.3f (threshold %.3f) -> %s",
                consensus_dice, consensus_dice_min, "PASS" if passed else "FAIL")
    return PilotResult(mean_dice, consensus_dice, kept / max(1, len(samples)), passed)
