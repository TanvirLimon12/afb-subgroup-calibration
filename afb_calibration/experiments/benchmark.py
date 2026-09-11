"""Experiment 1 - standard detection benchmark.

AP50/AP75/AP50:95, F1-relevant precision/recall via AP, small-object AR, count
MAE, and inference cost. Worst-group AP50 is reported first.
"""
from __future__ import annotations

from ..config import Config
from ..engine import run_pipeline
from ..utils.logging import get_logger
from .common import clone_config

logger = get_logger(__name__)

VARIANTS = {
    "detector_erm": {"robust.method": "erm", "use_verifier": False},
    "detector_group_balanced": {"robust.method": "group_balanced", "use_verifier": False},
    "detector_group_dro": {"robust.method": "group_dro", "use_verifier": False},
    "full_afb_calibration": {"robust.method": "group_balanced", "use_verifier": True},
}


def _set(cfg: Config, dotted: str, value):
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = getattr(node, p)
    setattr(node, parts[-1], value)


def run(cfg: Config, max_images=None, verifier_epochs: int = 5, train_image_fraction: float = 1.0) -> dict:
    rows = []
    for name, opts in VARIANTS.items():
        c = clone_config(cfg)
        use_verifier = opts.pop("use_verifier", False) if "use_verifier" in opts else False
        for k, v in opts.items():
            _set(c, k, v)
        art = run_pipeline(c, use_verifier=use_verifier, use_fusion=use_verifier,
                           max_images=max_images, verifier_epochs=verifier_epochs,
                           train_image_fraction=train_image_fraction)
        det = art.report["detection"]
        rows.append({
            "variant": name,
            "worst_group_AP50": det.get("worst_group_AP50"),
            "mean_AP50": det.get("mean_AP50"),
            "AP75": det.get("AP75"),
            "AP50:95": det.get("AP50:95"),
            "AR_small": det.get("AR_small"),
            "count_MAE": det.get("count_MAE"),
            "inference_cost_s": det.get("inference_cost_s_per_image"),
        })
        logger.info("benchmark variant %s done", name)
    return {"table": rows}
