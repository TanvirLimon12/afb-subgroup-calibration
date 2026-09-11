"""Calibrate-then-fuse candidate scoring (Stage 4).

Each of the detector logit and the Stage-2 verifier logit is independently
calibrated to a probability first; the fusion model then combines calibrated
probabilities (plus optional morphology features and group identity), never raw
variance across differently-scaled scores. Fusion weights are fit on validation
only - never on test, never on the detector's own training data.

Input sanitisation: detector/verifier logits can be non-finite (inf/NaN) under
distribution shift — e.g. an out-of-distribution tile drives a logit to ±inf, or
a degenerate crop produces NaN from the verifier. Left untreated these propagate
into isotonic/logistic fitting as 'divide by zero' / 'overflow in matmul'
warnings (seen in the v3 suite under sensor_noise/defocus_blur) and silently
corrupt the calibration. All logits are clipped to a finite range and
non-finite values are replaced before any calibrator sees them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .calibration import build_calibrator

# Clamp logits before calibration. Wide enough to leave any real logit untouched
# (a detector emitting |z|>100 is already broken), tight enough to make every
# downstream matmul finite. Also bounds the isotonic input domain.
_LOGIT_CLIP = 50.0


def _sanitize_logits(x) -> np.ndarray:
    """Return a finite float64 array, replacing inf/NaN and clipping extremes.

    NaN -> 0 (the neutral logit), inf -> +clip, -inf -> -clip, then clip the
    rest. This is the standard pre-calibration hygiene; it must happen before
    the calibrator sees the data, not after.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return a
    a = np.where(np.isfinite(a), a, np.nan_to_num(a, nan=0.0, posinf=_LOGIT_CLIP, neginf=-_LOGIT_CLIP))
    return np.clip(a, -_LOGIT_CLIP, _LOGIT_CLIP)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Jensen-Shannon divergence between two Bernoulli signals (per element)."""
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    m = 0.5 * (p + q)

    def kl(a, b):
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


@dataclass
class CalibrateThenFuse:
    calibration: str = "temperature"
    model: str = "logistic"
    use_group_feature: bool = True
    num_groups: int = 8
    det_cal: object = field(default=None)
    ver_cal: object = field(default=None)
    fusion: object = field(default=None)

    def _features(self, p_det, p_ver, morph, groups):
        cols = [p_det.reshape(-1, 1), p_ver.reshape(-1, 1)]
        cols.append(js_divergence(p_det, p_ver).reshape(-1, 1))
        if morph is not None:
            m = np.asarray(morph, dtype=np.float64).reshape(len(p_det), -1)
            m = np.where(np.isfinite(m), m, 0.0)  # non-finite morphology -> neutral
            cols.append(m)
        if self.use_group_feature:
            onehot = np.zeros((len(p_det), self.num_groups))
            onehot[np.arange(len(p_det)), np.asarray(groups, dtype=int)] = 1.0
            cols.append(onehot)
        return np.concatenate(cols, axis=1)

    def fit(self, det_logits, ver_logits, labels, groups, morph=None) -> "CalibrateThenFuse":
        det_logits = _sanitize_logits(det_logits)
        ver_logits = _sanitize_logits(ver_logits)
        labels = np.asarray(labels, dtype=np.float64)
        self.det_cal = build_calibrator(self.calibration).fit(det_logits, labels)
        self.ver_cal = build_calibrator(self.calibration).fit(ver_logits, labels)
        p_det = self.det_cal.predict_proba(det_logits)
        p_ver = self.ver_cal.predict_proba(ver_logits)
        x = self._features(p_det, p_ver, morph, groups)
        if self.model == "logistic":
            self.fusion = LogisticRegression(max_iter=1000)
        else:
            self.fusion = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000)
        # Inputs are finite after sanitisation; any remaining FP warnings come from
        # the solver's internal matmul (spurious under some BLAS builds).
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            self.fusion.fit(x, labels.astype(int))
        return self

    def predict_proba(self, det_logits, ver_logits, groups, morph=None) -> np.ndarray:
        det_logits = _sanitize_logits(det_logits)
        ver_logits = _sanitize_logits(ver_logits)
        p_det = self.det_cal.predict_proba(det_logits)
        p_ver = self.ver_cal.predict_proba(ver_logits)
        x = self._features(p_det, p_ver, morph, groups)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return self.fusion.predict_proba(x)[:, 1]

    def calibrated_signals(self, det_logits, ver_logits):
        det_logits = _sanitize_logits(det_logits)
        ver_logits = _sanitize_logits(ver_logits)
        return self.det_cal.predict_proba(det_logits), self.ver_cal.predict_proba(ver_logits)
