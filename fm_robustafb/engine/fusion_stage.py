"""Stage-4 calibrated fusion fitting and application.

Calibration and fusion weights are fit on validation candidates only, then
applied to test candidates.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..fusion import CalibrateThenFuse, js_divergence
from .train_verifier import verifier_logits


def _num_groups(cfg: Config) -> int:
    return len(cfg.data.cameras) * len(cfg.data.backgrounds)


def fit_fusion(cfg: Config, verifier, val_bank, device) -> CalibrateThenFuse:
    arr = val_bank.arrays()
    ver_logits = verifier_logits(verifier, val_bank, device, arr["records"])
    fusion = CalibrateThenFuse(
        calibration=cfg.fusion.calibration, model=cfg.fusion.model,
        use_group_feature=cfg.fusion.use_group_feature, num_groups=_num_groups(cfg))
    fusion.fit(arr["det_logits"], ver_logits, arr["labels"], arr["groups"])
    return fusion


def apply_fusion(fusion: CalibrateThenFuse, verifier, test_bank, device) -> dict:
    arr = test_bank.arrays()
    ver_logits = verifier_logits(verifier, test_bank, device, arr["records"])
    fused = fusion.predict_proba(arr["det_logits"], ver_logits, arr["groups"])
    p_det, p_ver = fusion.calibrated_signals(arr["det_logits"], ver_logits)
    # Carry per-detection IoU + tp flag through for localization-aware calibration
    # (LaECE) and LRP Error (Kuzucu et al., ECCV 2024 — P54), computed once in
    # build_candidate_bank and surfaced via Candidate.iou.
    import numpy as _np
    return {
        "labels": arr["labels"], "groups": arr["groups"],
        "det_logits": arr["det_logits"], "ver_logits": ver_logits,
        "p_det": p_det, "p_ver": p_ver, "fused": fused,
        "disagreement": js_divergence(p_det, p_ver),
        "ious": arr["ious"],  # best-IoU per candidate vs GT (for LaECE)
        "image_ids": _np.array([r.image_id for r in arr["records"]], dtype=_np.int64),
    }
