"""Train the full pipeline (detector + Stage-2 verifier + Stage-4 fusion)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fm_robustafb.config import load_config
from fm_robustafb.engine import run_pipeline
from fm_robustafb.engine.pipeline import save_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--verifier-epochs", type=int, default=5)
    ap.add_argument("opts", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.opts)
    art = run_pipeline(cfg, use_verifier=True, use_fusion=True, verifier_epochs=args.verifier_epochs)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(art.detector.state_dict(), out / "detector.pt")
    if art.verifier is not None:
        torch.save(art.verifier.state_dict(), out / "verifier.pt")
    save_report(art.report, out / "report.json")
    print(f"saved pipeline artifacts and report to {out}")


if __name__ == "__main__":
    main()
