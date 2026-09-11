"""Detection calibration metrics: D-ECE, reliability curves, Brier, NLL.

D-ECE bins detections by confidence and computes the weighted absolute
difference between bin-wise precision and bin-wise mean confidence:

    D-ECE = sum_i (|B_i| / N_det) * |precision(B_i) - confidence(B_i)|

Precision replaces accuracy here because true negatives are not defined in
detection - this is why D-ECE is distinct from classification ECE.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def detection_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    conf = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = conf.size
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        precision = correct[mask].mean()
        mean_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(precision - mean_conf)
    return float(ece)


def localization_aware_ece(confidences: np.ndarray, tp: np.ndarray, ious: np.ndarray,
                           n_bins: int = 15) -> float:
    """LaECE (Kuzucu et al., ECCV 2024): like D-ECE but the per-bin target is the
    localization-aware precision (mean IoU over the bin's true positives), so
    confidence is calibrated against how well boxes are localized, not just whether
    they are correct. Lower is better.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    tp = np.asarray(tp, dtype=np.float64)
    ious = np.asarray(ious, dtype=np.float64)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, conf.size
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        loc_precision = (tp[mask] * ious[mask]).sum() / mask.sum()
        ece += (mask.sum() / n) * abs(conf[mask].mean() - loc_precision)
    return float(ece)


def reliability_curve(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    conf = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf, bin_acc, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        counts.append(int(mask.sum()))
        bin_conf.append(float(conf[mask].mean()) if mask.any() else (lo + hi) / 2)
        bin_acc.append(float(correct[mask].mean()) if mask.any() else 0.0)
    return np.array(bin_conf), np.array(bin_acc), np.array(counts)


def brier_score(prob: np.ndarray, label: np.ndarray) -> float:
    prob = np.asarray(prob, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    return float(np.mean((prob - label) ** 2)) if prob.size else float("nan")


def negative_log_likelihood(prob: np.ndarray, label: np.ndarray, eps: float = 1e-9) -> float:
    prob = np.clip(np.asarray(prob, dtype=np.float64), eps, 1 - eps)
    label = np.asarray(label, dtype=np.float64)
    if prob.size == 0:
        return float("nan")
    return float(-np.mean(label * np.log(prob) + (1 - label) * np.log(1 - prob)))
