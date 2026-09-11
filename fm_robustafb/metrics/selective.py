"""Selective-prediction metrics: failure-detection AUROC/AUPRC and risk-coverage.

Does uncertainty separate true positives from false positives, well- from
poorly-localised boxes, in-domain from cross-camera samples? Risk-coverage is
framed as graceful degradation under shift, not distribution-free coverage.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def failure_auroc_auprc(confidence: np.ndarray, correct: np.ndarray) -> Dict[str, float]:
    """Higher confidence should indicate correct predictions. Failure = incorrect."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.int64)
    if correct.min() == correct.max():
        return {"AUROC": float("nan"), "AUPRC": float("nan")}
    failure = 1 - correct
    score = -confidence
    return {
        "AUROC": float(roc_auc_score(failure, score)),
        "AUPRC": float(average_precision_score(failure, score)),
    }


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray,
                        coverages: Sequence[float] = (1.0, 0.95, 0.9, 0.8)) -> List[dict]:
    confidence = np.asarray(confidence, dtype=np.float64)
    error = 1 - np.asarray(correct, dtype=np.float64)
    order = np.argsort(-confidence)
    error_sorted = error[order]
    n = len(error_sorted)
    out = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        out.append({"coverage": float(cov), "risk": float(error_sorted[:k].mean())})
    return out


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    confidence = np.asarray(confidence, dtype=np.float64)
    error = 1 - np.asarray(correct, dtype=np.float64)
    order = np.argsort(-confidence)
    error_sorted = error[order]
    risks = np.cumsum(error_sorted) / np.arange(1, len(error_sorted) + 1)
    return float(risks.mean()) if risks.size else float("nan")
