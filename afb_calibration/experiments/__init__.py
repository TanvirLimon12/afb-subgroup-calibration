from . import (benchmark, cross_camera, lobo, hard_negative,
               calibration, ablation, label_efficiency, baseline)

REGISTRY = {
    "benchmark": benchmark.run,
    "cross_camera": cross_camera.run,
    "lobo": lobo.run,
    "hard_negative": hard_negative.run,
    "calibration": calibration.run,
    "ablation": ablation.run,
    "label_efficiency": label_efficiency.run,
    "baseline": baseline.run,
}

__all__ = ["REGISTRY"]
