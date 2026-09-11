"""Run a curated real-data experiment suite.

Two presets:
  --preset mps    (default): bounded config for an Apple-silicon Mac (hours).
  --preset gpu    : CUDA/CPU with the real DINOv2 verifier and full training —
                    use this for any result you intend to report.

IMPORTANT: the foundation-model contribution (the "FM" in AFB Subgroup Calibration) is only
real when the verifier uses the actual DINOv2 checkpoint. A run with the offline
stub verifier carries NO pretrained signal and must NOT be reported as an
AFB Subgroup Calibration result. The build_dino loader now RAISES (rather than silently
falling back) when offline_stub=False and the hub load fails, so a stub result
can never again masquerade as a real one. The MPS preset deliberately uses the
stub (no real DINOv2 contribution) and prints a banner to that effect.

Writes per-experiment JSON/MD plus SUMMARY.md / SUMMARY.json.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from afb_calibration.config import load_config
from afb_calibration.experiments import REGISTRY
from afb_calibration.experiments.common import report_to_markdown
from afb_calibration.engine.pipeline import save_report
from afb_calibration.utils.logging import get_logger

logger = get_logger("run_all")

# Per-preset shared overrides. Each TASK may add more.
_PRESETS = {
    "mps": [
        "device=mps",
        "detector.backbone=resnet18", "detector.pretrained=True", "detector.fpn_channels=128",
        "train.epochs=6", "train.batch_size=2", "train.num_workers=0",
        "robust.consistency.enabled=False",
        # Stub by design on the quick preset: real DINOv2 hub load is slow/unreliable
        # on some MPS setups and the MPS preset is for wiring checks, not reporting.
        "verifier.offline_stub=True", "verifier.head=linear",
        "verifier.crop_size=96", "verifier.fuse_layers=[2,4]",
    ],
    "gpu": [
        "device=cuda",                 # override to cpu if no CUDA
        "detector.backbone=resnet50", "detector.pretrained=True", "detector.fpn_channels=256",
        "train.epochs=24", "train.batch_size=4", "train.num_workers=4",
        "robust.consistency.enabled=True",
        # Real verifier: this is the only preset that produces reportable results.
        "verifier.offline_stub=False", "verifier.head=linear",
        "verifier.crop_size=224", "verifier.fuse_layers=[6,9,12]",
    ],
}

# (experiment, config, extra overrides, run kwargs)
_RS = 0.4  # Raw Sputum train fraction to bound the quick preset's runtime
TASKS = [
    ("benchmark", "configs/raw_sputum.yaml", ["detector.fpn_min_level=3"],
     {"verifier_epochs": 2, "train_image_fraction": _RS}),
    ("ablation", "configs/raw_sputum.yaml", ["detector.fpn_min_level=3"],
     {"verifier_epochs": 2, "train_image_fraction": _RS}),
    ("hard_negative", "configs/raw_sputum.yaml", ["detector.fpn_min_level=3"],
     {"train_image_fraction": _RS}),
    ("calibration", "configs/raw_sputum.yaml", ["detector.fpn_min_level=3"],
     {"train_image_fraction": _RS}),
    ("cross_camera", "configs/znsm_idb.yaml", ["detector.fpn_min_level=2"], {}),
    ("lobo", "configs/znsm_idb.yaml", ["detector.fpn_min_level=2"], {}),
    # External baseline (Faster R-CNN) for state-of-the-art comparison.
    # Runs through the same eval path as AFB Subgroup Calibration for a comparable row.
    ("baseline", "configs/raw_sputum.yaml", ["detector.fpn_min_level=3"],
     {"train_image_fraction": _RS}),
]


def _headline(name: str, report: dict) -> dict:
    out = {"experiment": name}
    if "table" in report:
        out["variants"] = [r.get("variant") for r in report["table"]]
    for key in ("worst_group_AP50", "mean_AP50", "worst_background_AP50",
                "fp_reduction_fraction", "mean_in_camera_AP50"):
        if key in report:
            out[key] = report[key]
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/suite")
    ap.add_argument("--train-fraction", type=float, default=0.5)
    ap.add_argument("--preset", choices=("mps", "gpu"), default="mps",
                    help="mps = bounded quick preset with stub verifier (NOT reportable); "
                         "gpu = real DINOv2 + full training (the only reportable preset).")
    ap.add_argument("--only", default=None, help="comma-separated subset of experiment names")
    ap.add_argument("opts", nargs="*", help="extra config overrides, e.g. data.root=... data.max_side=1600")
    args = ap.parse_args()

    bounded = _PRESETS[args.preset] + list(args.opts)
    tasks = TASKS
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        tasks = [t for t in TASKS if t[0] in want]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loud banner: the single most important thing a collaborator must not miss.
    using_stub = any(o.startswith("verifier.offline_stub=True") for o in bounded)
    banner = (
        "\n" + "=" * 78 + "\n"
        + f"  SUITE PRESET: {args.preset.upper()}\n"
        + f"  Verifier: {'STUB (random init, NO pretrained signal)' if using_stub else 'REAL DINOv2 via torch.hub'}\n"
        + ("  >>> RESULTS FROM THIS RUN ARE NOT REPORTABLE AS AFB Subgroup Calibration <<<\n"
           if using_stub else
           "  Results are reportable, provided the DINOv2 hub load succeeds.\n")
        + "=" * 78 + "\n"
    )
    print(banner)
    logger.warning("suite preset=%s verifier_stub=%s", args.preset, using_stub)
    (out_dir / "PRESET.txt").write_text(banner)

    summary = []
    for name, cfg_path, extra, kwargs in tasks:
        logger.info("=== running %s on %s ===", name, cfg_path)
        t0 = time.time()
        try:
            cfg = load_config(cfg_path, bounded + extra)
            run_fn = REGISTRY[name]
            report = run_fn(cfg, **kwargs)
            save_report(report, out_dir / f"exp_{name}.json")
            (out_dir / f"exp_{name}.md").write_text(report_to_markdown(report, f"Experiment: {name}"))
            entry = _headline(name, report)
            entry["minutes"] = round((time.time() - t0) / 60, 1)
            entry["status"] = "ok"
        except Exception as exc:
            logger.error("%s FAILED: %s", name, exc)
            traceback.print_exc()
            entry = {"experiment": name, "status": "failed", "error": str(exc)[:200],
                     "minutes": round((time.time() - t0) / 60, 1)}
        summary.append(entry)
        with open(out_dir / "SUMMARY.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    lines = [f"# Real-data experiment suite (preset={args.preset}, "
             f"verifier={'stub' if using_stub else 'real-DINOv2'})", ""]
    for e in summary:
        lines.append(f"- **{e['experiment']}** [{e['status']}, {e.get('minutes','?')} min] "
                     + ", ".join(f"{k}={v}" for k, v in e.items()
                                 if k not in ("experiment", "status", "minutes", "variants", "error")))
    if using_stub:
        lines += ["", "> ⚠ STUB verifier — these numbers are NOT reportable as AFB Subgroup Calibration results. "
                    "Re-run with --preset gpu for the real verifier."]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    logger.info("suite done; summary at %s/SUMMARY.md", out_dir)


if __name__ == "__main__":
    main()
