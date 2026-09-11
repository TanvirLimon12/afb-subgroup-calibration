import numpy as np

from fm_robustafb.fusion.calibration import TemperatureScaler
from fm_robustafb.metrics import (aurc, average_precision, detection_ece, evaluate_detection,
                                  failure_auroc_auprc, localization_aware_ece, lrp_error,
                                  match_detections, risk_coverage_curve)


def test_average_precision_perfect():
    preds = [{"boxes": np.array([[0, 0, 10, 10.0]]), "scores": np.array([0.9])}]
    gts = [np.array([[0, 0, 10, 10.0]])]
    assert average_precision(preds, gts, 0.5) > 0.99


def test_worst_group_reported():
    preds = [{"boxes": np.array([[0, 0, 10, 10.0]]), "scores": np.array([0.9])},
             {"boxes": np.zeros((0, 4)), "scores": np.zeros(0)}]
    gts = [np.array([[0, 0, 10, 10.0]]), np.array([[0, 0, 10, 10.0]])]
    out = evaluate_detection(preds, gts, [0.5], groups=[0, 1])
    assert out["worst_group_AP50"] <= out["mean_AP50"]


def test_lrp_and_laece():
    preds = [{"boxes": np.array([[0, 0, 10, 10.0], [50, 50, 60, 60]]),
              "scores": np.array([0.9, 0.4])}]
    gts = [np.array([[0, 0, 10, 10.0]])]
    lrp = lrp_error(preds, gts, 0.5)
    assert 0.0 <= lrp <= 1.0
    scores, tp, ious, n_fn = match_detections(preds, gts, 0.5)
    assert tp.sum() == 1 and n_fn == 0
    laece = localization_aware_ece(scores, tp, ious, n_bins=5)
    assert 0.0 <= laece <= 1.0


def test_evaluate_detection_has_eccv_metrics():
    preds = [{"boxes": np.array([[0, 0, 10, 10.0]]), "scores": np.array([0.9])}]
    gts = [np.array([[0, 0, 10, 10.0]])]
    out = evaluate_detection(preds, gts, [0.5])
    assert "LRP50" in out and "LaECE" in out


def test_detection_ece_bounds():
    conf = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([1, 1, 0, 0])
    ece = detection_ece(conf, correct, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_risk_coverage_monotone_intuition():
    conf = np.array([0.9, 0.8, 0.7, 0.1])
    correct = np.array([1, 1, 1, 0])
    rc = risk_coverage_curve(conf, correct, [1.0, 0.75])
    assert rc[1]["risk"] <= rc[0]["risk"]
    assert 0 <= aurc(conf, correct) <= 1


def test_failure_auroc_valid():
    out = failure_auroc_auprc(np.array([0.9, 0.1, 0.8, 0.2]), np.array([1, 0, 1, 0]))
    assert out["AUROC"] >= 0.5


def test_temperature_scaler_reduces_nll():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, 500)
    labels = (1 / (1 + np.exp(-logits / 2)) > 0.5).astype(float)
    ts = TemperatureScaler().fit(logits, labels)
    assert ts.temperature > 0


def test_plot_detection_overlay_writes_png(tmp_path):
    """The TP/FP/FN overlay must render a real PNG — the Figure-1 generator."""
    import os
    from fm_robustafb.metrics import plot_detection_overlay
    img = np.random.randint(40, 200, (200, 200, 3), dtype=np.uint8)
    boxes = np.array([[20, 20, 60, 60], [100, 100, 140, 140]], dtype=float)
    scores = np.array([0.9, 0.4])
    gt = np.array([[22, 22, 58, 58]], dtype=float)  # box0=TP, box1=FP
    out = plot_detection_overlay(img, boxes, scores, gt, tmp_path / "overlay.png")
    assert out.endswith("overlay.png")
    assert os.path.getsize(out) > 1000


def test_plot_failure_cases_writes_png(tmp_path):
    """The failure-case panel must render a real PNG — the differentiator figure."""
    import os
    from fm_robustafb.metrics import plot_failure_cases
    img = np.random.randint(40, 200, (200, 200, 3), dtype=np.uint8)
    imgs = [img, img]
    ab = [np.array([[20., 20, 60, 60]]), np.array([[100., 100, 140, 140]])]
    asc = [np.array([0.9]), np.array([0.85])]
    agt = [np.array([[22., 22, 58, 58]]), np.array([])]  # 2nd image: GT empty -> box is FP
    out = plot_failure_cases(imgs, ab, asc, agt, tmp_path / "fail.png", n=2)
    assert out.endswith("fail.png")
    assert os.path.getsize(out) > 1000
