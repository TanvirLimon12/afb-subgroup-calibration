"""Run one (reducer, seed) through the pipeline and persist per-candidate predictions.

This is the Phase-2 `dump_run` from the first notebook, with the defect that killed
`group_dro` seeds 0 and 2 fixed: a collapsed detector produces a validation candidate
bank with a single class, which makes isotonic calibration raise, which killed the whole
run before any artifact was written. `afb_calibration.engine.pipeline._try_fusion` guards
against this; the original dump_run did not.

Collapsed runs now emit a detection-only report with ``collapsed: true`` so the DRO row
of Table 2 (0.00 on two of three seeds) is recorded rather than missing.
"""
from __future__ import annotations

import copy
import json
import os
import pickle
import time

import numpy as np
import torch

# Which cached checkpoint belongs to which (reducer, seed) of the submitted run.
KNOWN_CHECKPOINTS = {
    ("erm", 0): "det_e39c6bd589d40abd", ("erm", 1): "det_b7a7d47ab92cccce",
    ("erm", 2): "det_fbb15647fe3d5577",
    ("group_balanced", 0): "det_a10c258a3a007254", ("group_balanced", 1): "det_f93fac1ad30baee2",
    ("group_balanced", 2): "det_fcd6620e131fdcbc",
    ("group_dro", 0): "det_38c0737d5af56bf5", ("group_dro", 1): "det_65b2065ca296536a",
    ("group_dro", 2): "det_2839eb78d1de3315",
}
METHODS = ["erm", "group_balanced", "group_dro"]
PRETTY = {"erm": "ERM", "group_balanced": "PGE", "group_dro": "DRO-style"}

# NOTE: data.root is hashed into the detector checkpoint cache key, so a cached
# detector is only reused when the dataset path matches the run that produced it.

VERIFIER_EPOCHS = 5


def base_overrides(data_root: str):
    return [f"data.root={data_root}", "data.max_side=1600",
            "detector.pretrained=True", "verifier.offline_stub=False",
            "train.batch_size=2", "train.epochs=24"]


def cfg_for(method, seed, data_root, config="configs/raw_sputum.yaml", extra=()):
    from afb_calibration.config import load_config
    c = load_config(config, base_overrides(data_root) + list(extra))
    c.seed = seed
    c.robust.method = method
    c.device = "cuda" if torch.cuda.is_available() else "cpu"
    return c


class _FakeDS:
    """The cache key only reads len(dataset); avoids building one just to verify keys."""
    def __init__(self, n): self.n = n
    def __len__(self): return self.n


def verify_checkpoints(cache_dir, data_root, config="configs/raw_sputum.yaml", link=True):
    """Recompute each cache key and report whether it lands on a file that exists.

    Returns (n_resolved, mismatches). With link=True, mismatched keys are symlinked to
    the known filename so a drifted data.root does not trigger 10 GPU-hours of retraining.
    """
    from afb_calibration.engine.train_detector import _detector_cache_key
    resolved, bad = 0, []
    for m in METHODS:
        for s in [0, 1, 2]:
            key = _detector_cache_key(cfg_for(m, s, data_root, config), _FakeDS(1081), 3)
            want = KNOWN_CHECKPOINTS[(m, s)]
            path = os.path.join(cache_dir, f"det_{key}.pt")
            if os.path.exists(path) and f"det_{key}" == want:
                resolved += 1
            else:
                bad.append((m, s, key, want))
    if link:
        for m, s, key, want in bad:
            src = os.path.join(cache_dir, want + ".pt")
            dst = os.path.join(cache_dir, f"det_{key}.pt")
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)
    return resolved, bad


def dump_run(method, seed, artifacts, data_root, tag=None, extra_overrides=(),
             force=False, config="configs/raw_sputum.yaml", detector=None):
    """Load (or train) a detector, run verifier + fusion, persist everything.

    detector: pass an already-built model to skip the cache lookup entirely (Phase 5B
    uses this for the fixed-DRO models, which live outside the standard cache).
    """
    from afb_calibration.engine.build import build_dataset, build_group_index
    from afb_calibration.engine.infer import tiled_inference
    from afb_calibration.engine.train_verifier import build_candidate_bank, train_verifier
    from afb_calibration.engine.fusion_stage import fit_fusion, apply_fusion
    from afb_calibration.engine.evaluate import evaluate_calibration, evaluate_detection_results
    from afb_calibration.engine.train_detector import train_detector
    from afb_calibration.utils.seed import seed_everything, resolve_device

    tag = tag or f"{method}_seed{seed}"
    cand_p = f"{artifacts}/predictions/cand_{tag}.npz"
    rep_p = f"{artifacts}/runs/report_{tag}.json"
    if os.path.exists(rep_p) and not force:
        print(f"[skip] {tag} already done")
        return json.load(open(rep_p))

    t0 = time.perf_counter()
    cfg = cfg_for(method, seed, data_root, config, extra_overrides)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    gi = build_group_index(cfg)

    train_ds = build_dataset(cfg, "train", gi)
    val_ds = build_dataset(cfg, "val", gi)
    test_ds = build_dataset(cfg, "test", gi)

    if detector is None:
        detector = train_detector(cfg, train_ds, device, gi)

    test_results = tiled_inference(detector, test_ds, device, cfg.data.tile_size,
                                   cfg.data.tile_overlap, cfg.detector.cross_tile_nms_iou)
    det_report = evaluate_detection_results(test_results, cfg)
    n_boxes = int(sum(len(r.scores) for r in test_results))

    with open(f"{artifacts}/predictions/det_{tag}.pkl", "wb") as f:
        pickle.dump({"preds": [{"boxes": r.boxes, "scores": r.scores} for r in test_results],
                     "gts": [r.gt_boxes for r in test_results],
                     "groups": [int(r.group) for r in test_results],
                     "image_ids": [int(r.image_id) for r in test_results],
                     "det_report": det_report}, f)

    # ---- the guard that was missing -----------------------------------------
    val_bank = build_candidate_bank(detector, val_ds, device, cfg)
    va = val_bank.arrays()
    collapsed = len(va["records"]) == 0 or len(set(va["labels"].tolist())) < 2
    if collapsed:
        report = {"detection": det_report, "calibration": None, "collapsed": True,
                  "reason": ("validation candidates are single-class or empty; the detector "
                             "produced no usable positives, so calibration and fusion are "
                             "undefined"),
                  "n_test_boxes": n_boxes,
                  "n_val_candidates": int(len(va["records"]))}
        json.dump(report, open(rep_p, "w"), indent=2, default=str)
        print(f"[COLLAPSED] {tag}  test boxes={n_boxes}  val candidates={len(va['records'])}  "
              f"wgAP50={det_report.get('worst_group_AP50')}  "
              f"({(time.perf_counter()-t0)/60:.1f} min)")
        return report

    verifier = train_verifier(cfg, val_bank, device, epochs=VERIFIER_EPOCHS)
    fusion = fit_fusion(cfg, verifier, val_bank, device)
    test_bank = build_candidate_bank(detector, test_ds, device, cfg, add_gt_positives=False)
    if len(test_bank.arrays()["records"]) == 0:
        report = {"detection": det_report, "calibration": None, "collapsed": True,
                  "reason": "no test candidates survived the 0.05 detector threshold",
                  "n_test_boxes": n_boxes}
        json.dump(report, open(rep_p, "w"), indent=2, default=str)
        print(f"[COLLAPSED] {tag} (no test candidates)")
        return report

    fo = apply_fusion(fusion, verifier, test_bank, device)
    np.savez_compressed(cand_p, **{k: np.asarray(v) for k, v in fo.items()},
                        method=np.array(method), seed=np.array(seed))
    cal = evaluate_calibration(fo, cfg)
    report = {"detection": det_report, "calibration": cal, "collapsed": False,
              "n_candidates": int(len(fo["fused"]))}
    json.dump(report, open(rep_p, "w"), indent=2, default=str)
    print(f"[ok] {tag}  {len(fo['fused'])} candidates  "
          f"wgAP50={det_report.get('worst_group_AP50'):.3f}  "
          f"C-ECE={cal.get('D-ECE'):.4f}  ({(time.perf_counter()-t0)/60:.1f} min)")
    return report
