"""Experiment 3 - leave-one-background-out testing.

One fold per background profile; reports mean AP, worst-background AP, the max
performance gap, and calibration (D-ECE) for the held-out group.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..engine import run_pipeline
from .common import clone_config


def run(cfg: Config, max_images=None, use_verifier: bool = True) -> dict:
    folds = {}
    for bg in cfg.data.backgrounds:
        c = clone_config(cfg)
        art = run_pipeline(c, use_verifier=use_verifier, use_fusion=use_verifier,
                           max_images=max_images, test_backgrounds=[bg])
        det = art.report["detection"]
        cal = art.report.get("calibration", {})
        folds[bg] = {
            "AP50": det.get("mean_AP50", det.get("AP50")),
            "D-ECE": cal.get("D-ECE"),
        }
    aps = [v["AP50"] for v in folds.values() if v["AP50"] is not None and not np.isnan(v["AP50"])]
    return {
        "folds": folds,
        "mean_AP50": float(np.mean(aps)) if aps else float("nan"),
        "worst_background_AP50": float(np.min(aps)) if aps else float("nan"),
        "max_gap_AP50": float(np.max(aps) - np.min(aps)) if aps else float("nan"),
    }
