"""Selective-prediction referral scoring.

The value proposition is referable confidence that degrades gracefully under
shift: the detector knows when to abstain in a new lab rather than failing
silently. No distribution-free coverage claim is made - camera shift breaks the
exchangeability such guarantees assume.
"""
from __future__ import annotations

import numpy as np


def confidence_from_prob(prob: np.ndarray) -> np.ndarray:
    """Decision confidence = distance of the fused probability from 0.5."""
    return np.abs(np.asarray(prob, dtype=np.float64) - 0.5) * 2.0


def referral_scores(fused_prob: np.ndarray, disagreement: np.ndarray | None = None,
                    disagreement_weight: float = 1.0) -> np.ndarray:
    """Higher score = more reliable (kept); lower = referred.

    Combines fused-probability confidence with optional FM-disagreement
    uncertainty (high disagreement between calibrated signals triggers referral).
    """
    conf = confidence_from_prob(fused_prob)
    if disagreement is None:
        return conf
    dis = np.asarray(disagreement, dtype=np.float64)
    dis = dis / (dis.max() + 1e-9)
    return conf - disagreement_weight * dis


def should_invoke_fm(det_prob: np.ndarray, low: float = 0.2, high: float = 0.8) -> np.ndarray:
    """Only run the expensive FM arbitration on medium-confidence detections;
    high/low detector confidence is accepted/rejected directly."""
    p = np.asarray(det_prob, dtype=np.float64)
    return (p > low) & (p < high)
