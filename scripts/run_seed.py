"""Train and evaluate one (reducer, seed) pair, persisting per-candidate predictions.

Each invocation is a single unit of work: one detector, its verifier, its fusion head,
and the resulting predictions. Runs are independent, so seeds can be distributed across
machines or sessions and merged afterwards simply by collecting their output files.

Results land in ``--out``:

    runs/report_<reducer>_seed<n>.json     detection + calibration summary
    predictions/cand_<reducer>_seed<n>.npz per-candidate arrays
    predictions/det_<reducer>_seed<n>.pkl  per-image boxes, scores and ground truth

A detector that collapses to an all-background solution yields a validation candidate
bank with a single class, for which calibration is undefined. Such runs are recorded with
``collapsed: true`` and their detection report rather than being discarded, because the
collapse rate is itself a reported property of the reducers.

Examples
--------
    python -m scripts.run_seed --config configs/raw_sputum.yaml --reducer group_balanced --seed 3
    python -m scripts.run_seed --config configs/raw_sputum.yaml --reducer erm --seed 3 --out results
"""
from __future__ import annotations

import argparse
import os
import sys

REDUCERS = ["erm", "group_balanced", "group_dro"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/raw_sputum.yaml")
    ap.add_argument("--reducer", required=True, choices=REDUCERS,
                    help="erm | group_balanced (present-group equalization) | group_dro")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="results", help="output root (default: results)")
    ap.add_argument("--data-root", default=None,
                    help="dataset root; defaults to data.root in the config")
    ap.add_argument("--verifier-epochs", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="recompute even if a report exists")
    a = ap.parse_args(argv)

    from fm_robustafb.engine import runner
    from fm_robustafb.config import load_config

    data_root = a.data_root or load_config(a.config, []).data.root
    for sub in ("runs", "predictions", "analysis"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    runner.VERIFIER_EPOCHS = a.verifier_epochs
    report = runner.dump_run(a.reducer, a.seed, a.out, data_root,
                             config=a.config, force=a.force)
    return 0 if report else 1


if __name__ == "__main__":
    sys.exit(main())
