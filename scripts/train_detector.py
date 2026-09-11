"""Train the Stage-1b specialist detector with the configured group-robust method."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from afb_calibration.config import load_config
from afb_calibration.engine import build_dataset, build_group_index
from afb_calibration.engine.evaluate import evaluate_detection_results
from afb_calibration.engine.infer import tiled_inference
from afb_calibration.engine.pipeline import save_report
from afb_calibration.engine.train_detector import train_detector
from afb_calibration.utils.seed import resolve_device, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("opts", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.opts)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    gi = build_group_index(cfg)

    train_ds = build_dataset(cfg, "train", gi)
    test_ds = build_dataset(cfg, "test", gi)
    detector = train_detector(cfg, train_ds, device, gi)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(detector.state_dict(), out / "detector.pt")

    results = tiled_inference(detector, test_ds, device, cfg.data.tile_size,
                              cfg.data.tile_overlap, cfg.detector.cross_tile_nms_iou)
    save_report(evaluate_detection_results(results, cfg), out / "detector_report.json")
    print(f"saved detector and report to {out}")


if __name__ == "__main__":
    main()
