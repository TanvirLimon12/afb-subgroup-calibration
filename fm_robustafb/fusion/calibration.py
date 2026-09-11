"""Per-signal calibration, fit on the validation split only, applied at test time.

Detector confidence is systematically miscalibrated and worsens under shift, so
each signal is converted to a probability independently BEFORE any disagreement
measure or fusion (calibrate-then-fuse).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression


class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> "TemperatureScaler":
        z = torch.as_tensor(logits, dtype=torch.float64)
        y = torch.as_tensor(labels, dtype=torch.float64)
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z / log_t.exp(), y)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(log_t.exp().item())
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        z = np.asarray(logits, dtype=np.float64) / self.temperature
        return 1.0 / (1.0 + np.exp(-z))


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        p = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
        self.model.fit(p, np.asarray(labels, dtype=np.float64))
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
        return self.model.predict(p)


def build_calibrator(kind: str):
    if kind == "temperature":
        return TemperatureScaler()
    if kind == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"unknown calibrator: {kind}")
