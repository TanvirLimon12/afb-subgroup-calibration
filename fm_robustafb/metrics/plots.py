"""Publication figures: reliability diagrams, risk-coverage curves, and
qualitative detection overlays.

Uses a non-interactive backend so figures render headless. The overlay functions
produce the TP/FP/FN detection figures and failure-case panels that medical-imaging
accompany quantitative tables in medical-imaging work.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .calibration import detection_ece, reliability_curve  # noqa: E402
from .selective import risk_coverage_curve  # noqa: E402


def plot_reliability(confidences: np.ndarray, correct: np.ndarray, out_path: str | Path,
                     n_bins: int = 15, title: Optional[str] = None) -> str:
    bin_conf, bin_acc, counts = reliability_curve(confidences, correct, n_bins)
    ece = detection_ece(confidences, correct, n_bins)
    valid = counts > 0
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.bar(bin_conf[valid], bin_acc[valid], width=0.9 / n_bins, alpha=0.8,
           edgecolor="black", label="precision")
    ax.set_xlabel("confidence")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title or f"Reliability (D-ECE={ece:.3f})")
    ax.legend(loc="upper left")
    return _save(fig, out_path)


def plot_risk_coverage(confidence: np.ndarray, correct: np.ndarray, out_path: str | Path,
                       coverages: Sequence[float] = tuple(np.linspace(0.1, 1.0, 19)),
                       title: Optional[str] = None) -> str:
    rc = risk_coverage_curve(confidence, correct, coverages)
    cov = [p["coverage"] for p in rc]
    risk = [p["risk"] for p in rc]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(cov, risk, marker="o", markersize=3)
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk")
    ax.set_title(title or "Risk-coverage")
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def _classify_predictions(boxes: np.ndarray, scores: np.ndarray, gt: np.ndarray,
                          iou_thresh: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy match predictions to GT; return (tp_idx, fp_idx, missed_gt_idx).

    tp_idx: predicted boxes that match a GT box at IoU>=thresh.
    fp_idx: predicted boxes with no GT match.
    missed_gt_idx: GT boxes not matched by any prediction (false negatives).
    """
    tp_idx, fp_idx, missed = [], [], []
    matched_gt = np.zeros(len(gt), dtype=bool) if len(gt) else np.zeros(0, dtype=bool)
    if len(boxes) == 0:
        return np.array([], int), np.array([], int), np.where(~matched_gt)[0]
    order = np.argsort(-scores)
    for pi in order:
        if len(gt) == 0:
            fp_idx.append(pi)
            continue
        from ..utils.boxes import box_iou_np
        ious = box_iou_np(boxes[pi:pi + 1], gt)[0]
        best = int(ious.argmax())
        if ious[best] >= iou_thresh and not matched_gt[best]:
            matched_gt[best] = True
            tp_idx.append(pi)
        else:
            fp_idx.append(pi)
    return np.array(tp_idx, int), np.array(fp_idx, int), np.where(~matched_gt)[0]


def plot_detection_overlay(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                           gt_boxes: np.ndarray, out_path: str | Path,
                           iou_thresh: float = 0.5, score_thresh: float = 0.3,
                           title: Optional[str] = None, max_draw: int = 60) -> str:
    """Draw a TP/FP/FN detection overlay on one image.

    Color code: green = true positive, red = false positive, yellow dashed = false
    negative (missed GT). Confidence score annotated on each prediction. This is
    a qualitative detection figure.

    ``image``: (H, W, 3) uint8 RGB. ``boxes``/``gt_boxes``: (N,4) xyxy.
    """
    keep = scores >= score_thresh
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) > max_draw:  # cap clutter on dense fields
        order = np.argsort(-scores)[:max_draw]
        boxes, scores = boxes[order], scores[order]
    tp_idx, fp_idx, miss_idx = _classify_predictions(boxes, scores, gt_boxes, iou_thresh)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    for i in tp_idx:
        x0, y0, x1, y1 = boxes[i]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="lime", linewidth=1.8))
        ax.text(x0, max(0, y0 - 3), f"{scores[i]:.2f}", color="lime", fontsize=6)
    for i in fp_idx:
        x0, y0, x1, y1 = boxes[i]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="red", linewidth=1.8))
        ax.text(x0, max(0, y0 - 3), f"{scores[i]:.2f}", color="red", fontsize=6)
    for gi in miss_idx:
        x0, y0, x1, y1 = gt_boxes[gi]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="yellow", linewidth=1.5, linestyle="--"))
    ax.set_axis_off()
    n_tp, n_fp, n_fn = len(tp_idx), len(fp_idx), len(miss_idx)
    handles = [
        plt.Line2D([0], [0], color="lime", lw=2, label=f"TP ({n_tp})"),
        plt.Line2D([0], [0], color="red", lw=2, label=f"FP ({n_fp})"),
        plt.Line2D([0], [0], color="yellow", lw=2, ls="--", label=f"FN ({n_fn})"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.8)
    if title:
        ax.set_title(title, fontsize=9)
    return _save(fig, out_path)


def plot_failure_cases(images: List[np.ndarray], all_boxes: List[np.ndarray],
                       all_scores: List[np.ndarray], all_gt: List[np.ndarray],
                       out_path: str | Path, n: int = 4, iou_thresh: float = 0.5,
                       score_thresh: float = 0.3, prefer: str = "fp",
                       titles: Optional[List[str]] = None) -> str:
    """Assemble a panel of the worst failure cases for the failure-case figure.

    ``prefer`` selects the ranking criterion:
      "fp"  — images with the most high-confidence false positives (the
              detector's over-confidence failure, which selective prediction
              must catch).
      "fn"  — images with the most missed ground-truth bacilli.

    Produces a 1xN (or 2xN) panel. This is the differentiator figure that lifts a
    medical-detection study.
    """
    # Rank images by failure severity
    ranked = []
    for i, (b, s, g) in enumerate(zip(all_boxes, all_scores, all_gt)):
        keep = s >= score_thresh
        b2, s2 = b[keep], s[keep]
        _, fp_idx, miss_idx = _classify_predictions(b2, s2, g, iou_thresh)
        score = len(fp_idx) if prefer == "fp" else len(miss_idx)
        ranked.append((score, i))
    ranked.sort(reverse=True)
    picked = [i for _, i in ranked[:n] if ranked[0][0] > 0]
    if not picked:  # no failures on any image — show the top-N anyway
        picked = [i for _, i in ranked[:n]]
    n_show = len(picked)
    ncol = min(n_show, 4)
    nrow = (n_show + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False)
    for k, idx in enumerate(picked):
        r, c = k // ncol, k % ncol
        ax = axes[r][c]
        ax.imshow(images[idx])
        b, s, g = all_boxes[idx], all_scores[idx], all_gt[idx]
        keep = s >= score_thresh
        b2, s2 = b[keep], s[keep]
        tp_idx, fp_idx, miss_idx = _classify_predictions(b2, s2, g, iou_thresh)
        for i in tp_idx:
            x0, y0, x1, y1 = b2[i]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor="lime", linewidth=1.5))
        for i in fp_idx:
            x0, y0, x1, y1 = b2[i]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor="red", linewidth=1.5))
        for gi in miss_idx:
            x0, y0, x1, y1 = g[gi]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor="yellow", linewidth=1.5, linestyle="--"))
        ax.set_axis_off()
        lbl = (titles[idx] if titles else f"img {idx}") + \
            f"  (FP={len(fp_idx)}, FN={len(miss_idx)})"
        ax.set_title(lbl, fontsize=8, color="darkred")
    # hide unused axes
    for k in range(n_show, nrow * ncol):
        axes[k // ncol][k % ncol].set_axis_off()
    fig.suptitle("Failure cases (red=FP, yellow=FN, green=TP)", fontsize=10)
    return _save(fig, out_path)


def _save(fig, out_path: str | Path) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path)


def plot_reliability(confidences: np.ndarray, correct: np.ndarray, out_path: str | Path,
                     n_bins: int = 15, title: Optional[str] = None) -> str:
    bin_conf, bin_acc, counts = reliability_curve(confidences, correct, n_bins)
    ece = detection_ece(confidences, correct, n_bins)
    valid = counts > 0
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.bar(bin_conf[valid], bin_acc[valid], width=0.9 / n_bins, alpha=0.8,
           edgecolor="black", label="precision")
    ax.set_xlabel("confidence")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title or f"Reliability (D-ECE={ece:.3f})")
    ax.legend(loc="upper left")
    return _save(fig, out_path)


def plot_risk_coverage(confidence: np.ndarray, correct: np.ndarray, out_path: str | Path,
                       coverages: Sequence[float] = tuple(np.linspace(0.1, 1.0, 19)),
                       title: Optional[str] = None) -> str:
    rc = risk_coverage_curve(confidence, correct, coverages)
    cov = [p["coverage"] for p in rc]
    risk = [p["risk"] for p in rc]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(cov, risk, marker="o", markersize=3)
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk")
    ax.set_title(title or "Risk-coverage")
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def _save(fig, out_path: str | Path) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path)
