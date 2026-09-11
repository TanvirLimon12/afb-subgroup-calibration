"""Dataset composition and representativeness , R1 item 9.

 "The validation and test sets contain only 122 images and 129 images respectively.
       ... The authors should discuss whether these sizes provide adequate statistical
       power. Including percentages, balance analysis, and a brief discussion of the
       dataset's representativeness would significantly strengthen this section."

 "Severe imbalance exists among the visual-style groups. ... The manuscript should
       discuss: whether any balancing strategy was applied, whether class weights or
       sampling techniques were used, and whether performance differs across style groups."

Produces the numbers those two paragraphs need. CPU, seconds.
"""
from __future__ import annotations

import collections
import csv
import json
import os

import numpy as np


def split_composition(ann_dir="data/afb", metadata_csv=None):
    """Per-split image/box counts, group shares, boxes per image, and metadata crosstabs."""
    out = {"splits": {}, "totals": {}}
    md = {}
    if metadata_csv and os.path.exists(metadata_csv):
        md = {r["Image_ID"]: (r["Background_Color"], r["Camera_System"])
              for r in csv.DictReader(open(metadata_csv))}

    grand_img = grand_box = 0
    for split in ["train", "val", "test"]:
        d = json.load(open(f"{ann_dir}/{split}.json"))
        imgs, anns = d["images"], d["annotations"]
        per_img = collections.Counter(a["image_id"] for a in anns)
        boxes_per = np.array([per_img.get(im["id"], 0) for im in imgs], float)
        groups = collections.Counter(im["background"] for im in imgs)
        gbox = collections.Counter()
        gid2bg = {im["id"]: im["background"] for im in imgs}
        for a in anns:
            gbox[gid2bg[a["image_id"]]] += 1

        rec = {"images": len(imgs), "boxes": len(anns),
               "group_images": dict(groups),
               "group_image_pct": {k: round(100 * v / len(imgs), 1) for k, v in groups.items()},
               "group_boxes": dict(gbox),
               "boxes_per_image": {"mean": float(boxes_per.mean()),
                                   "median": float(np.median(boxes_per)),
                                   "min": int(boxes_per.min()), "max": int(boxes_per.max()),
                                   "images_with_zero_boxes": int((boxes_per == 0).sum())}}
        if md:
            bg = collections.Counter(); cam = collections.Counter(); miss = 0
            for im in imgs:
                b = os.path.basename(im["file_name"])
                if b in md:
                    bg[md[b][0]] += 1; cam[md[b][1]] += 1
                else:
                    miss += 1
            rec["metadata_background"] = dict(bg)
            rec["metadata_camera"] = dict(cam)
            rec["metadata_missing"] = miss
        out["splits"][split] = rec
        grand_img += len(imgs); grand_box += len(anns)

    out["totals"] = {"images": grand_img, "boxes": grand_box,
                     "split_pct": {s: round(100 * out["splits"][s]["images"] / grand_img, 1)
                                   for s in out["splits"]}}
    return out


def group_share_drift(comp):
    """Does the style mix hold across splits? A large drift undermines the test set as a
    representative sample, which is exactly the representativeness question  asks."""
    keys = sorted({k for s in comp["splits"].values() for k in s["group_images"]})
    rows = {}
    for k in keys:
        rows[k] = {s: comp["splits"][s]["group_image_pct"].get(k, 0.0) for s in comp["splits"]}
    max_drift = {k: round(max(v.values()) - min(v.values()), 1) for k, v in rows.items()}
    return {"pct_by_split": rows, "max_pct_drift": max_drift}


def minimum_detectable_gap(n_a, n_b, p_pooled=0.5, alpha=0.05, power=0.80):
    """Two-proportion MDE - the honest way to answer 'is 129 test images enough?'.

    Returns the smallest difference in per-candidate correctness rate detectable between
    two subgroups of the given sizes. Uses the normal approximation; with n as small as
    the minority subgroup here the true requirement is worse, which is the point.
    """
    from math import sqrt
    z_a, z_b = 1.959963985, 0.8416212336   # alpha=0.05 two-sided, power=0.80
    if alpha != 0.05 or power != 0.80:
        try:
            from scipy.stats import norm
            z_a = norm.ppf(1 - alpha / 2); z_b = norm.ppf(power)
        except Exception:
            pass
    se = sqrt(p_pooled * (1 - p_pooled) * (1.0 / max(n_a, 1) + 1.0 / max(n_b, 1)))
    return float((z_a + z_b) * se)


def power_note(comp, candidate_counts=None):
    """Assemble the statistical-power paragraph's raw numbers."""
    note = {"nominal": {s: comp["splits"][s]["images"] for s in comp["splits"]},
            "minority_test_images": comp["splits"]["test"]["group_images"].get("style2"),
            "minority_dev_images": comp["splits"]["val"]["group_images"].get("style2")}
    if candidate_counts:
        note["candidate_level"] = candidate_counts
        maj = candidate_counts.get("style 1"); mino = candidate_counts.get("style 2")
        if maj and mino:
            note["mde_candidate_correctness"] = round(minimum_detectable_gap(maj, mino), 3)
    return note


def balancing_spec(config_path="configs/raw_sputum.yaml"):
    """What balancing the pipeline actually applies - the factual answer to ."""
    import yaml
    cfg = yaml.safe_load(open(config_path))
    method = cfg.get("robust", {}).get("method")
    return {
        "robust.method": method,
        "loss_side_balancing": {
            "erm": "none - plain mean over the batch",
            "group_balanced": ("present-group equalization: per-example losses are averaged "
                               "within each group present in the batch, then those group "
                               "means are averaged, so an under-represented group present "
                               "in a batch gets equal weight to a common one"),
            "group_dro": ("EMA-tracked worst-group upweighting (Sagawa et al.), "
                          "exponentiated-gradient reweighting"),
        }.get(method, "unknown"),
        "sampler": ("GroupBalancedSampler (inverse group frequency) is used only when "
                    "robust.method == 'group_balanced'; ERM and Group-DRO use a plain "
                    "shuffled loader"),
        "class_weights": "none - the detector loss applies no per-class weighting",
        "oversampling": "none - no minority image duplication or synthetic augmentation",
        "batch_size": cfg.get("train", {}).get("batch_size"),
        "caveat": ("with batch_size 2 and three groups, most batches contain one or two "
                   "groups, so present-group equalization equalizes over whichever groups "
                   "co-occur rather than over all three at every step"),
    }
