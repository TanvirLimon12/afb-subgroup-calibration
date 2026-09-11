"""Experiment 5 - calibration and risk-coverage, clean and under shift.

D-ECE (global and per group), reliability diagrams, Brier, NLL,
failure-detection AUROC/AUPRC, risk-coverage curves, and cross-camera
calibration transfer. A well-calibrated system shows accuracy degradation and
uncertainty increase moving together under corruption.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..data.augment import apply_corruption
from ..engine import run_pipeline
from ..engine.evaluate import evaluate_calibration, evaluate_selective
from ..engine.fusion_stage import apply_fusion
from ..engine.build import build_dataset, build_group_index
from ..engine.train_verifier import build_candidate_bank
from ..utils.seed import resolve_device


def run(cfg: Config, max_images=None, corruptions=("sensor_noise", "defocus_blur"),
        train_image_fraction: float = 1.0) -> dict:
    art = run_pipeline(cfg, use_verifier=True, use_fusion=True, max_images=max_images,
                       train_image_fraction=train_image_fraction)
    if art.fusion_out is None:
        return {"error": "fusion unavailable"}

    result = {
        "clean": {
            "calibration": evaluate_calibration(art.fusion_out, cfg),
            "selective": evaluate_selective(art.fusion_out, cfg),
        }
    }

    device = resolve_device(cfg.device)
    gi = build_group_index(cfg)
    test_ds = build_dataset(cfg, "test", gi)
    shifted = {}
    for name in corruptions:
        corrupt = lambda img, n=name: apply_corruption(img, n)
        bank = build_candidate_bank(art.detector, test_ds, device, cfg, max_images=max_images,
                                    add_gt_positives=False, corruption=corrupt)
        if len(bank.arrays()["records"]) == 0:
            continue
        fo = apply_fusion(art.fusion, art.verifier, bank, device)
        shifted[name] = {
            "calibration": evaluate_calibration(fo, cfg),
            "selective": evaluate_selective(fo, cfg),
        }
    result["under_shift"] = shifted
    return result
