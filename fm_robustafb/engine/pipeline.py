"""End-to-end pipeline assembly reused by every experiment."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import Config
from ..utils.logging import get_logger
from ..utils.seed import resolve_device, seed_everything
from .build import build_dataset, build_group_index
from .evaluate import evaluate_all
from .fusion_stage import apply_fusion, fit_fusion
from .infer import tiled_inference
from .train_detector import train_detector
from .train_verifier import build_candidate_bank, train_verifier

logger = get_logger(__name__)


@dataclass
class PipelineArtifacts:
    detector: object
    verifier: Optional[object]
    fusion: Optional[object]
    report: dict
    fusion_out: Optional[dict] = None


def run_pipeline(
    cfg: Config, use_verifier: bool = True, use_fusion: bool = True,
    max_images: Optional[int] = None, verifier_epochs: int = 5,
    test_cameras=None, test_backgrounds=None, train_image_fraction: float = 1.0,
) -> PipelineArtifacts:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    gi = build_group_index(cfg)

    train_ds = build_dataset(cfg, "train", gi, image_fraction=train_image_fraction,
                             fraction_seed=cfg.seed)
    val_ds = build_dataset(cfg, "val", gi)
    test_ds = build_dataset(cfg, "test", gi, cameras=test_cameras, backgrounds=test_backgrounds)

    detector = train_detector(cfg, train_ds, device, gi)
    t0 = time.perf_counter()
    test_results = tiled_inference(detector, test_ds, device, cfg.data.tile_size,
                                   cfg.data.tile_overlap, cfg.detector.cross_tile_nms_iou,
                                   max_images=max_images)
    inference_cost = (time.perf_counter() - t0) / max(1, len(test_results))

    verifier = fusion = fusion_out = None
    if use_verifier:
        val_bank = build_candidate_bank(detector, val_ds, device, cfg, max_images=max_images)
        verifier = train_verifier(cfg, val_bank, device, epochs=verifier_epochs)
        if use_fusion:
            fusion_out = _try_fusion(cfg, verifier, detector, val_bank, val_ds, test_ds, device, max_images)
            fusion = fusion_out[1] if fusion_out else None
            fusion_out = fusion_out[0] if fusion_out else None

    report = evaluate_all(test_results, fusion_out, cfg)
    report["detection"]["inference_cost_s_per_image"] = inference_cost
    return PipelineArtifacts(detector, verifier, fusion, report, fusion_out)


def _try_fusion(cfg, verifier, detector, val_bank, val_ds, test_ds, device, max_images):
    arr = val_bank.arrays()
    if len(set(arr["labels"].tolist())) < 2:
        logger.warning("fusion skipped: validation candidates have a single class "
                       "(detector produced too few true positives to calibrate)")
        return None
    fusion = fit_fusion(cfg, verifier, val_bank, device)
    test_bank = build_candidate_bank(detector, test_ds, device, cfg,
                                     max_images=max_images, add_gt_positives=False)
    if len(test_bank.arrays()["records"]) == 0:
        return None
    return apply_fusion(fusion, verifier, test_bank, device), fusion


def save_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=_default)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
