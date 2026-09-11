from . import (exp1_benchmark, exp2_cross_camera, exp3_lobo, exp4_hard_negative,
               exp5_calibration, exp6_ablation, exp7_label_efficiency, exp8_baseline)

REGISTRY = {
    "benchmark": exp1_benchmark.run,
    "cross_camera": exp2_cross_camera.run,
    "lobo": exp3_lobo.run,
    "hard_negative": exp4_hard_negative.run,
    "calibration": exp5_calibration.run,
    "ablation": exp6_ablation.run,
    "label_efficiency": exp7_label_efficiency.run,
    "baseline": exp8_baseline.run,
}

__all__ = ["REGISTRY"]
