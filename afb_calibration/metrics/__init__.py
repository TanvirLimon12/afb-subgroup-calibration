from .detection import (evaluate_detection, average_precision, coco_iou_range, lrp_error,
                        match_detections)
from .calibration import (detection_ece, reliability_curve, brier_score,
                          negative_log_likelihood, localization_aware_ece)
from .selective import failure_auroc_auprc, risk_coverage_curve, aurc
from .plots import (plot_reliability, plot_risk_coverage, plot_detection_overlay,
                    plot_failure_cases)

__all__ = [
    "evaluate_detection",
    "average_precision",
    "coco_iou_range",
    "lrp_error",
    "match_detections",
    "detection_ece",
    "reliability_curve",
    "brier_score",
    "negative_log_likelihood",
    "localization_aware_ece",
    "failure_auroc_auprc",
    "risk_coverage_curve",
    "aurc",
    "plot_reliability",
    "plot_risk_coverage",
    "plot_detection_overlay",
    "plot_failure_cases",
]
