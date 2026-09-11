"""Run a named experiment (Experiments 1-7) and save its report."""
from __future__ import annotations

import argparse
from pathlib import Path

from afb_calibration.config import load_config
from afb_calibration.engine.pipeline import save_report
from afb_calibration.experiments import REGISTRY
from afb_calibration.experiments.common import aggregate_runs, clone_config, report_to_markdown


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment", choices=sorted(REGISTRY))
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="override eval.seeds; runs the experiment once per seed with CIs")
    ap.add_argument("--train-fraction", type=float, default=None,
                    help="forwarded as train_image_fraction to experiments that accept it; "
                         "bounds runtime by training on a subset of images.")
    ap.add_argument("--verifier-epochs", type=int, default=None,
                    help="forwarded as verifier_epochs to experiments that accept it.")
    ap.add_argument("opts", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.opts)
    run_fn = REGISTRY[args.experiment]
    kwargs = {}
    if args.max_images is not None:
        kwargs["max_images"] = args.max_images
    if args.train_fraction is not None:
        kwargs["train_image_fraction"] = args.train_fraction
    if args.verifier_epochs is not None:
        kwargs["verifier_epochs"] = args.verifier_epochs
    # Inspect the experiment's signature and drop kwargs it doesn't accept, so the
    # same flags work uniformly across experiments with different parameters.
    import inspect
    params = set(inspect.signature(run_fn).parameters)
    kwargs = {k: v for k, v in kwargs.items() if k in params}

    seeds = args.seeds if args.seeds is not None else cfg.eval.seeds
    reports = []
    for s in seeds:
        c = clone_config(cfg)
        c.seed = s
        reports.append(run_fn(c, **kwargs))
    report = aggregate_runs(reports) if len(reports) > 1 else reports[0]

    out = Path(cfg.output_dir) / f"exp_{args.experiment}.json"
    save_report(report, out)
    md = report_to_markdown(report, title=f"Experiment: {args.experiment}")
    out.with_suffix(".md").write_text(md)
    print(f"saved {args.experiment} report ({len(seeds)} seed(s)) to {out} and .md")


if __name__ == "__main__":
    main()
