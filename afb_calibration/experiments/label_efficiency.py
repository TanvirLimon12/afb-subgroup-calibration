"""Experiment 7 - label efficiency.

Complete image subsets (10/25/50/100%), stratified by camera and background,
comparing from-scratch detection against the DINO-guided full system. The clearest
evidence for FM guidance should appear in the low-label regime; if the gap closes
entirely by 100% labels, that is a reportable limitation, not something to hide.
"""
from __future__ import annotations

from ..config import Config
from ..engine import run_pipeline
from .common import clone_config


def run(cfg: Config, fractions=(0.1, 0.25, 0.5, 1.0), max_images=None,
        verifier_epochs: int = 5) -> dict:
    curve = {"from_scratch": {}, "full_afb_calibration": {}}
    for frac in fractions:
        base = run_pipeline(clone_config(cfg), use_verifier=False, use_fusion=False,
                            max_images=max_images, train_image_fraction=frac)
        curve["from_scratch"][frac] = base.report["detection"].get(
            "worst_group_AP50", base.report["detection"].get("AP50"))

        full = run_pipeline(clone_config(cfg), use_verifier=True, use_fusion=True,
                            max_images=max_images, train_image_fraction=frac,
                            verifier_epochs=verifier_epochs)
        curve["full_afb_calibration"][frac] = full.report["detection"].get(
            "worst_group_AP50", full.report["detection"].get("AP50"))
    return {"label_efficiency_curve": curve}
