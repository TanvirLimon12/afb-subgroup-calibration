"""Detection metrics: AP, small-object AR, count error, and worst-group AP.

Worst-group AP@0.5 is the headline number for every table; mean AP is reported
alongside but is never the headline.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..utils.boxes import box_iou_np

Prediction = Dict[str, np.ndarray]  # {"boxes": (n,4), "scores": (n,)}


def coco_iou_range() -> List[float]:
    return [round(0.5 + 0.05 * i, 2) for i in range(10)]


def _match_single(preds: List[Prediction], gts: List[np.ndarray], iou_thresh: float):
    order = []
    for img_idx, p in enumerate(preds):
        for j in range(len(p["scores"])):
            order.append((p["scores"][j], img_idx, j))
    order.sort(key=lambda t: -t[0])
    npos = sum(len(g) for g in gts)
    matched = [np.zeros(len(g), dtype=bool) for g in gts]
    tp = np.zeros(len(order))
    fp = np.zeros(len(order))
    for rank, (_, img_idx, j) in enumerate(order):
        gt = gts[img_idx]
        if len(gt) == 0:
            fp[rank] = 1
            continue
        ious = box_iou_np(preds[img_idx]["boxes"][j:j + 1], gt)[0]
        best = int(ious.argmax())
        if ious[best] >= iou_thresh and not matched[img_idx][best]:
            matched[img_idx][best] = True
            tp[rank] = 1
        else:
            fp[rank] = 1
    return tp, fp, npos


def match_detections(preds: List[Prediction], gts: List[np.ndarray], iou_thresh: float = 0.5):
    """Greedy score-ranked matching. Returns per-detection (scores, tp_flags, ious)
    plus false-negative count, for localization-aware calibration and LRP."""
    order = []
    for img_idx, p in enumerate(preds):
        for j in range(len(p["scores"])):
            order.append((p["scores"][j], img_idx, j))
    order.sort(key=lambda t: -t[0])
    matched = [np.zeros(len(g), dtype=bool) for g in gts]
    scores, tp, ious = [], [], []
    for score, img_idx, j in order:
        gt = gts[img_idx]
        scores.append(float(score))
        if len(gt) == 0:
            tp.append(0); ious.append(0.0); continue
        iou_row = box_iou_np(preds[img_idx]["boxes"][j:j + 1], gt)[0]
        best = int(iou_row.argmax())
        if iou_row[best] >= iou_thresh and not matched[img_idx][best]:
            matched[img_idx][best] = True
            tp.append(1); ious.append(float(iou_row[best]))
        else:
            tp.append(0); ious.append(0.0)
    n_fn = sum(int((~m).sum()) for m in matched)
    return np.array(scores), np.array(tp), np.array(ious), n_fn


def lrp_error(preds: List[Prediction], gts: List[np.ndarray], iou_thresh: float = 0.5) -> float:
    """Localisation-Recall-Precision error (Oksuz et al.); lower is better.

    LRP = (1/Z)[ (1/(1-tau)) * sum_TP(1-IoU) + N_FP + N_FN ], Z = N_TP+N_FP+N_FN.
    """
    _, tp, ious, n_fn = match_detections(preds, gts, iou_thresh)
    n_tp = int(tp.sum())
    n_fp = int((tp == 0).sum())
    z = n_tp + n_fp + n_fn
    if z == 0:
        return float("nan")
    loc = ((1.0 - ious[tp == 1]).sum() / (1.0 - iou_thresh)) if n_tp else 0.0
    return float((loc + n_fp + n_fn) / z)


def average_precision(preds: List[Prediction], gts: List[np.ndarray], iou_thresh: float = 0.5) -> float:
    tp, fp, npos = _match_single(preds, gts, iou_thresh)
    if npos == 0:
        return float("nan")
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    recall = tp_c / npos
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    return _ap_101(recall, precision)


def _ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recall >= t
        p = precision[mask].max() if mask.any() else 0.0
        ap += p / 101.0
    return float(ap)


def small_object_ar(preds: List[Prediction], gts: List[np.ndarray],
                    iou_thresh: float = 0.5, small_area: float = 32 * 32) -> float:
    total_small, matched_small = 0, 0
    for p, gt in zip(preds, gts):
        if len(gt) == 0:
            continue
        areas = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
        small = areas < small_area
        total_small += int(small.sum())
        if len(p["boxes"]) == 0 or small.sum() == 0:
            continue
        ious = box_iou_np(p["boxes"], gt)
        hit = (ious >= iou_thresh).any(axis=0)
        matched_small += int((hit & small).sum())
    return matched_small / max(1, total_small)


def count_errors(preds: List[Prediction], gts: List[np.ndarray], score_thresh: float = 0.3):
    pred_counts = np.array([int((p["scores"] >= score_thresh).sum()) for p in preds], dtype=np.float64)
    gt_counts = np.array([len(g) for g in gts], dtype=np.float64)
    mae = float(np.abs(pred_counts - gt_counts).mean()) if len(gt_counts) else float("nan")
    if len(gt_counts) > 1 and pred_counts.std() > 0 and gt_counts.std() > 0:
        corr = float(np.corrcoef(pred_counts, gt_counts)[0, 1])
    else:
        corr = float("nan")
    return mae, corr


def evaluate_detection(
    preds: List[Prediction], gts: List[np.ndarray], iou_list: Sequence[float] = (0.5, 0.75),
    groups: Optional[Sequence[int]] = None, coco_range: bool = False, score_thresh: float = 0.3,
) -> dict:
    result = {f"AP{int(round(t * 100))}": average_precision(preds, gts, t) for t in iou_list}
    if coco_range:
        aps = [average_precision(preds, gts, t) for t in coco_iou_range()]
        result["AP50:95"] = float(np.nanmean(aps))
    result["AR_small"] = small_object_ar(preds, gts)
    result["LRP50"] = lrp_error(preds, gts, 0.5)
    scores, tp, ious, _ = match_detections(preds, gts, 0.5)
    from .calibration import localization_aware_ece
    result["LaECE"] = localization_aware_ece(scores, tp, ious)
    mae, corr = count_errors(preds, gts, score_thresh)
    result["count_MAE"] = mae
    result["count_corr"] = corr

    if groups is not None:
        groups = np.asarray(groups)
        per_group = {}
        for g in sorted(set(groups.tolist())):
            idx = np.nonzero(groups == g)[0]
            ap = average_precision([preds[i] for i in idx], [gts[i] for i in idx], 0.5)
            per_group[int(g)] = ap
        result["per_group_AP50"] = per_group
        valid = [v for v in per_group.values() if not np.isnan(v)]
        result["worst_group_AP50"] = float(min(valid)) if valid else float("nan")
        result["mean_AP50"] = float(np.mean(valid)) if valid else float("nan")
        result["group_gap_AP50"] = (result["mean_AP50"] - result["worst_group_AP50"]
                                    if valid else float("nan"))
    return result
