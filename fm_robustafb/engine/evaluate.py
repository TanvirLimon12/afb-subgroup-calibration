"""Aggregate evaluation: detection, calibration, and selective prediction."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..config import Config
from ..fusion.referral import referral_scores
from ..metrics import (aurc, brier_score, detection_ece, evaluate_detection, failure_auroc_auprc,
                       localization_aware_ece, negative_log_likelihood, risk_coverage_curve)
from .infer import InferenceResult, as_predictions


def evaluate_detection_results(results: List[InferenceResult], cfg: Config) -> dict:
    preds, gts, groups = as_predictions(results)
    return evaluate_detection(preds, gts, cfg.eval.ap_iou, groups=groups, coco_range=True)


def evaluate_calibration(fusion_out: dict, cfg: Config, confidence_key: str = "fused") -> dict:
    conf = np.asarray(fusion_out[confidence_key], dtype=np.float64)
    labels = np.asarray(fusion_out["labels"], dtype=np.float64)
    groups = np.asarray(fusion_out["groups"])
    out = {
        "D-ECE": detection_ece(conf, labels, cfg.eval.dece_bins),
        "brier": brier_score(conf, labels),
        "nll": negative_log_likelihood(conf, labels),
    }
    # Localization-aware calibration (LaECE) + LRP Error (Kuzucu et al., ECCV 2024,
    # P54). These account for localization quality, which D-ECE ignores. Requires
    # the per-candidate IoU carried through apply_fusion from build_candidate_bank.
    ious = fusion_out.get("ious")
    if ious is not None and len(ious) == len(conf):
        tp = (labels >= 0.5).astype(np.float64)
        out["LaECE"] = localization_aware_ece(conf, tp, np.asarray(ious), cfg.eval.dece_bins)
    per_group = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        if m.sum() >= 5:
            per_group[int(g)] = detection_ece(conf[m], labels[m], cfg.eval.dece_bins)
    out["per_group_D-ECE"] = per_group
    if per_group:
        out["worst_group_D-ECE"] = float(max(per_group.values()))
    return out


def evaluate_selective(fusion_out: dict, cfg: Config, use_disagreement: bool = True) -> dict:
    labels = np.asarray(fusion_out["labels"], dtype=np.int64)
    fused = np.asarray(fusion_out["fused"], dtype=np.float64)
    decision = (fused >= 0.5).astype(np.int64)
    correct = (decision == labels).astype(np.int64)
    dis = fusion_out["disagreement"] if use_disagreement else None
    conf = referral_scores(fused, dis)
    return {
        "failure_detection": failure_auroc_auprc(conf, correct),
        "risk_coverage": risk_coverage_curve(conf, correct, cfg.eval.coverage_points),
        "AURC": aurc(conf, correct),
        "accuracy": float(correct.mean()) if correct.size else float("nan"),
    }


def evaluate_all(
    results: List[InferenceResult], fusion_out: Optional[dict], cfg: Config
) -> dict:
    report = {"detection": evaluate_detection_results(results, cfg)}
    if fusion_out is not None:
        report["calibration"] = evaluate_calibration(fusion_out, cfg)
        report["selective"] = evaluate_selective(fusion_out, cfg)
    return report
