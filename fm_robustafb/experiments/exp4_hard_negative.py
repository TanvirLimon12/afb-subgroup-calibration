"""Experiment 4 - false-positive analysis on the configured test split.

Reports hard-negative rejection accuracy and how much the FM fusion step changes
the number of false positives versus detector-confidence-only thresholding.

IMPORTANT - scope: this experiment runs on the **config's test split** (e.g. the
Raw Sputum test set), NOT on an external hard-negative corpus. Because the
detector is retrained per seed, each seed produces a different candidate set, so
the baseline false-positive COUNT varies across seeds (this is a per-detector
count, not a dataset size). Do not describe the output as a cross-dataset or
external hard-negative result. An exploratory version of this analysis is
reported in the paper (Sec. V-D) as an inconsistent, non-reproducible effect.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..engine import run_pipeline


def run(cfg: Config, max_images=None, threshold: float = 0.5, train_image_fraction: float = 1.0) -> dict:
    art = run_pipeline(cfg, use_verifier=True, use_fusion=True, max_images=max_images,
                       train_image_fraction=train_image_fraction)
    fo = art.fusion_out
    if fo is None:
        return {"error": "fusion unavailable (too few calibratable candidates)"}

    labels = np.asarray(fo["labels"])
    p_det = np.asarray(fo["p_det"])
    fused = np.asarray(fo["fused"])
    negatives = labels == 0
    n_neg = int(negatives.sum())

    det_fp = int(((p_det >= threshold) & negatives).sum())
    fused_fp = int(((fused >= threshold) & negatives).sum())
    reduction = (det_fp - fused_fp) / max(1, det_fp)

    return {
        "num_hard_negatives": n_neg,
        "detector_only_false_positives": det_fp,
        "fusion_false_positives": fused_fp,
        "fp_reduction_fraction": float(reduction),
        "hard_negative_rejection_accuracy": float(((fused < threshold) & negatives).sum() / max(1, n_neg)),
    }
