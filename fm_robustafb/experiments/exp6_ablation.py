"""Experiment 6 - central ablation table.

Worst-group AP@0.5 is the primary column for every row; mean AP is secondary.
Calibration (D-ECE, LaECE) and selective reliability (AURC) accompany the fusion rows.

The table answers four literature-grounded questions:
  - Does the FM verifier help? (dino_verifier_only vs baseline_erm; tests D02)
  - Does frozen-FM NN scoring alone work, or is adaptation needed? (nn_verifier_only
    vs dino_verifier_only; AnomalyDINO P56 / Khan P57 lineage)
  - Which group-robust method is best for tiny-object detection? (group_balanced vs
    group_dro vs last_layer_retrain; P15 vs P16/P39 vs P55)
  - Does calibrated fusion close the worst-group reliability gap? (full_fm_robustafb)
"""
from __future__ import annotations

from ..config import Config
from ..engine import run_pipeline
from .common import clone_config

# (robust method, use verifier, use fusion, verifier head override or None)
VARIANTS = {
    "baseline_erm": ("erm", False, False, None),
    "dino_verifier_only": ("erm", True, True, "linear"),       # LoRA-adapted FM verifier
    "nn_verifier_only": ("erm", True, True, "nn"),             # training-free NN (AnomalyDINO, P56) — tests D02
    "group_balanced": ("group_balanced", False, False, None),
    "group_dro": ("group_dro", False, False, None),
    "last_layer_retrain": ("last_layer_retrain", False, False, None),  # LLR (P55)
    "dino_plus_best_group": ("group_balanced", True, True, "linear"),
    "full_fm_robustafb": ("group_balanced", True, True, "linear"),
}


def run(cfg: Config, max_images=None, verifier_epochs: int = 5, train_image_fraction: float = 1.0) -> dict:
    rows = []
    for name, (method, use_v, use_f, head_override) in VARIANTS.items():
        c = clone_config(cfg)
        c.robust.method = method
        if head_override is not None:
            c.verifier.head = head_override
        # Only the full row enables the calibrated-fusion column.
        enable_cal = name in ("full_fm_robustafb", "dino_plus_best_group", "dino_verifier_only",
                              "nn_verifier_only")
        art = run_pipeline(c, use_verifier=use_v, use_fusion=use_f and enable_cal,
                           max_images=max_images, verifier_epochs=verifier_epochs,
                           train_image_fraction=train_image_fraction)
        det = art.report["detection"]
        cal = art.report.get("calibration", {})
        sel = art.report.get("selective", {})
        rows.append({
            "variant": name,
            "worst_group_AP50": det.get("worst_group_AP50"),
            "mean_AP50": det.get("mean_AP50"),
            "LaECE": det.get("LaECE"),
            "LRP50": det.get("LRP50"),
            "D-ECE": cal.get("D-ECE"),
            "AURC": sel.get("AURC"),
        })
    return {"table": rows}
