"""Run the full pipeline on a prepared real corpus and save the report.

A convenience entry point for a single end-to-end run (detector + verifier +
calibrated fusion) with all metrics, used to produce the headline numbers.
"""
from __future__ import annotations

import argparse
import time

from afb_calibration.config import load_config
from afb_calibration.engine import run_pipeline
from afb_calibration.engine.pipeline import save_report
from afb_calibration.utils.logging import get_logger

logger = get_logger("run_main")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/raw_sputum.yaml")
    ap.add_argument("--train-fraction", type=float, default=1.0)
    ap.add_argument("--verifier-epochs", type=int, default=5)
    ap.add_argument("--no-verifier", action="store_true")
    ap.add_argument("opts", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.opts)
    t0 = time.time()
    art = run_pipeline(cfg, use_verifier=not args.no_verifier, use_fusion=not args.no_verifier,
                       train_image_fraction=args.train_fraction, verifier_epochs=args.verifier_epochs)
    save_report(art.report, f"{cfg.output_dir}/report.json")
    logger.info("done in %.1f min; report at %s/report.json", (time.time() - t0) / 60, cfg.output_dir)
    import json
    print(json.dumps(art.report, indent=2, default=str))


if __name__ == "__main__":
    main()
