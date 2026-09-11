"""Calibration analyses computed off the per-candidate dumps. CPU only, seconds to run.

Metric implementations are byte-for-byte equivalent to
``fm_robustafb.metrics.calibration`` so numbers here are comparable to the paper's.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

GROUPS = [0, 1, 2]
GNAME = {0: "style 0", 1: "style 1", 2: "style 2"}
NBINS = 15


# --------------------------------------------------------------------- metrics
def ece(conf, correct, n_bins=NBINS, edges=None):
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1) if edges is None else np.asarray(edges)
    e, n = 0.0, conf.size
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        e += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def laece(conf, tp, ious, n_bins=NBINS):
    conf = np.asarray(conf, float); tp = np.asarray(tp, float); ious = np.asarray(ious, float)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, conf.size
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        e += (m.sum() / n) * abs(conf[m].mean() - (tp[m] * ious[m]).sum() / m.sum())
    return float(e)


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def nll(p, y, eps=1e-9):
    p = np.clip(np.asarray(p, float), eps, 1 - eps); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def equal_mass_edges(conf, n_bins):
    q = np.quantile(np.asarray(conf, float), np.linspace(0, 1, n_bins + 1))
    q[0], q[-1] = 0.0, 1.0
    return np.unique(q)


# ------------------------------------------------------------------- dump access
def load_dump(artifacts, tag):
    z = np.load(f"{artifacts}/dumps/cand_{tag}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def available(artifacts, prefix=""):
    out = []
    for p in sorted(glob.glob(f"{artifacts}/dumps/cand_{prefix}*.npz")):
        out.append(os.path.basename(p)[len("cand_"):-len(".npz")])
    return out


# ------------------------------------------------------------------- 3A counts
def counts_and_occupancy(artifacts, tags, n_bins=NBINS):
    rows, occ = [], []
    for tag in tags:
        d = load_dump(artifacts, tag)
        g = d["groups"].astype(int); f = d["fused"]; y = d["labels"]; img = d["image_ids"]
        for gi in GROUPS:
            m = g == gi
            rows.append(dict(tag=tag, group=GNAME[gi], n_cand=int(m.sum()),
                             n_img_with_cand=int(len(np.unique(img[m]))),
                             n_pos=int(y[m].sum()),
                             pos_rate=float(y[m].mean()) if m.any() else float("nan"),
                             cece=ece(f[m], y[m], n_bins)))
            edges = np.linspace(0, 1, n_bins + 1)
            c = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                bm = (f[m] > lo) & (f[m] <= hi) if lo > 0 else (f[m] >= lo) & (f[m] <= hi)
                c.append(int(bm.sum()))
            occ.append(dict(tag=tag, group=GNAME[gi], counts=c,
                            occupied_bins=int(sum(x > 0 for x in c)),
                            bins_with_1=int(sum(x == 1 for x in c)),
                            median_occupied=int(np.median([x for x in c if x > 0]) if any(c) else 0)))
    return {"per_group": rows, "occupancy": occ}


# --------------------------------------------------------------- 3B bootstrap CI
def cluster_bootstrap(d, gi=None, B=2000, seed=0, n_bins=NBINS):
    """Percentile CI for C-ECE, resampling *images* (candidates are correlated within one)."""
    rng = np.random.default_rng(seed)
    g = d["groups"].astype(int); f = d["fused"]; y = d["labels"]; img = d["image_ids"].astype(int)
    sel = np.ones(len(g), bool) if gi is None else (g == gi)
    imgs = np.unique(img[sel])
    if imgs.size == 0:
        return np.array([]), 0
    by = {i: np.where(sel & (img == i))[0] for i in imgs}
    out = []
    for _ in range(B):
        pick = rng.choice(imgs, imgs.size, replace=True)
        idx = np.concatenate([by[i] for i in pick])
        if idx.size:
            out.append(ece(f[idx], y[idx], n_bins))
    return np.array(out), int(imgs.size)


def bootstrap_table(artifacts, tags, B=2000):
    res = {}
    for tag in tags:
        d = load_dump(artifacts, tag)
        g = d["groups"].astype(int)
        for gi in GROUPS + [None]:
            name = "pooled" if gi is None else GNAME[gi]
            sel = np.ones(len(g), bool) if gi is None else (g == gi)
            s, n_img = cluster_bootstrap(d, gi, B=B)
            res[f"{tag}|{name}"] = dict(
                point=ece(d["fused"][sel], d["labels"][sel]),
                lo=float(np.nanpercentile(s, 2.5)) if s.size else float("nan"),
                hi=float(np.nanpercentile(s, 97.5)) if s.size else float("nan"),
                n_img=n_img, n_cand=int(sel.sum()))
    return res


# ------------------------------------------------------------ 3C binning robustness
def binning_robustness(artifacts, tags):
    res = defaultdict(dict)
    for tag in tags:
        d = load_dump(artifacts, tag)
        g = d["groups"].astype(int); f = d["fused"]; y = d["labels"]
        for gi in GROUPS + [None]:
            name = "pooled" if gi is None else GNAME[gi]
            m = np.ones(len(g), bool) if gi is None else (g == gi)
            res[tag][name] = {"eq-width 10": ece(f[m], y[m], 10),
                              "eq-width 15": ece(f[m], y[m], 15),
                              "eq-width 20": ece(f[m], y[m], 20),
                              "eq-mass 10": ece(f[m], y[m], edges=equal_mass_edges(f[m], 10))}
    return dict(res)


# ------------------------------------------------------------ 3D stage decomposition
def stage_decomposition(artifacts, tags):
    res = defaultdict(dict)
    for tag in tags:
        d = load_dump(artifacts, tag)
        g = d["groups"].astype(int); y = d["labels"]
        for gi in GROUPS + [None]:
            name = "pooled" if gi is None else GNAME[gi]
            m = np.ones(len(g), bool) if gi is None else (g == gi)
            res[tag][name] = {"p_det": ece(d["p_det"][m], y[m]),
                              "p_ver": ece(d["p_ver"][m], y[m]),
                              "fused": ece(d["fused"][m], y[m])}
    return dict(res)


# ---------------------------------------------------------- 3E recall-matched verifier
def fp_at_matched_recall(p_ref, p_alt, y, target_tp=None):
    y = np.asarray(y).astype(bool)

    def curve(p):
        o = np.argsort(-np.asarray(p, float))
        return np.cumsum(y[o]), np.cumsum(~y[o])

    tp_r, fp_r = curve(p_ref)
    tp_a, fp_a = curve(p_alt)
    if target_tp is None:
        target_tp = int(min(tp_r[-1], tp_a[-1]))
    if target_tp <= 0:
        return None
    i = min(int(np.searchsorted(tp_r, target_tp)), len(fp_r) - 1)
    j = min(int(np.searchsorted(tp_a, target_tp)), len(fp_a) - 1)
    return dict(target_tp=int(target_tp), fp_detector=int(fp_r[i]), fp_fused=int(fp_a[j]),
                delta_fp=int(fp_a[j] - fp_r[i]),
                rel=float((fp_a[j] - fp_r[i]) / max(1, fp_r[i])))


def recall_matched_table(artifacts, tags, per_group=True):
    """Global and (new) per-group recall-matched verifier effect."""
    rows = []
    for tag in tags:
        d = load_dump(artifacts, tag)
        r = fp_at_matched_recall(d["p_det"], d["fused"], d["labels"])
        if r:
            r.update(tag=tag, group="all")
            rows.append(r)
        if per_group:
            g = d["groups"].astype(int)
            for gi in GROUPS:
                m = g == gi
                if m.sum() < 5:
                    continue
                rg = fp_at_matched_recall(d["p_det"][m], d["fused"][m], d["labels"][m])
                if rg:
                    rg.update(tag=tag, group=GNAME[gi])
                    rows.append(rg)
    return rows


# -------------------------------------------------------------- 3F per-group metrics
def per_group_metrics(artifacts, tags):
    res = defaultdict(dict)
    for tag in tags:
        d = load_dump(artifacts, tag)
        g = d["groups"].astype(int); y = d["labels"]; f = d["fused"]
        tp = (y >= 0.5).astype(float); iou = d["ious"]
        for gi in GROUPS + [None]:
            name = "pooled" if gi is None else GNAME[gi]
            m = np.ones(len(g), bool) if gi is None else (g == gi)
            res[tag][name] = {"n": int(m.sum()), "C-ECE": ece(f[m], y[m]),
                              "LaECE": laece(f[m], tp[m], iou[m]),
                              "Brier": brier(f[m], y[m]), "NLL": nll(f[m], y[m])}
    return dict(res)


# ------------------------------------------------------------------ style-2 detail
def minority_group_detail(artifacts, tags, test_json="data/afb/test.json", gi=2):
    """Which minority images actually contribute candidates, and how many.

    The paper says style 2 has seven test images. Only the images with at least one
    candidate above the 0.05 threshold enter the calibration estimate, and that is
    fewer. Report the effective number, not the nominal one.
    """
    tj = json.load(open(test_json))
    style_imgs = [im["id"] for im in tj["images"] if im["background"] == f"style{gi}"]
    out = {"nominal_test_images": len(style_imgs), "per_seed": []}
    for tag in tags:
        d = load_dump(artifacts, tag)
        g = d["groups"].astype(int); img = d["image_ids"].astype(int)
        m = g == gi
        ids, cnt = np.unique(img[m], return_counts=True)
        out["per_seed"].append({
            "tag": tag, "n_candidates": int(m.sum()),
            "n_images_contributing": int(ids.size),
            "images_with_zero_candidates": sorted(set(style_imgs) - set(ids.tolist())),
            "candidates_per_image": {int(i): int(c) for i, c in zip(ids, cnt)}})
    return out
