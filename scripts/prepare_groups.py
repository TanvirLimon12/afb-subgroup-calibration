"""Check cross-split leakage and assign unsupervised staining-style groups.

Run after afb-convert-yolo when the corpus lacks camera/staining metadata. This
rewrites the background field of each split JSON with a style-cluster id so the
group-robust experiments and worst-group AP become measurable, and reports any
train/test image leak that would inflate results.
"""
from __future__ import annotations

import argparse
import json

from afb_calibration.config import load_config
from afb_calibration.data.grouping import assign_style_groups, group_sizes
from afb_calibration.data.leakage import cross_split_leakage, remove_cross_split_leaks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/raw_sputum.yaml")
    ap.add_argument("--k", type=int, default=4, help="number of style groups")
    ap.add_argument("--skip-groups", action="store_true", help="only run the leakage check")
    ap.add_argument("--keep-leaks", action="store_true", help="report leaks but do not remove them")
    ap.add_argument("opts", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.opts)
    ann = {"train": cfg.data.ann_train, "val": cfg.data.ann_val, "test": cfg.data.ann_test}

    leak = cross_split_leakage(ann, cfg.data.root)
    print("leakage:", json.dumps({k: leak[k] for k in ("num_exact_leaks", "num_near_leaks", "clean")}))
    if not leak["clean"] and not args.keep_leaks:
        removed = remove_cross_split_leaks(ann, cfg.data.root)
        print("removed leaked images:", json.dumps(removed))

    if not args.skip_groups:
        assign_style_groups(ann, cfg.data.root, k=args.k, seed=cfg.seed)
        print("style-group sizes:", json.dumps(group_sizes(ann), indent=2))
        print(f"set config data.backgrounds to: {[f'style{i}' for i in range(args.k)]}")


if __name__ == "__main__":
    main()
