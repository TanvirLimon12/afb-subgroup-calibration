"""Shared experiment utilities: seed aggregation and table rendering.

Every table carries uncertainty: >=3 seeds with mean +/- std, not only on label
efficiency.
"""
from __future__ import annotations

import copy
import time
from typing import Callable, Dict, List

import numpy as np
import torch

from ..config import Config


def clone_config(cfg: Config) -> Config:
    return copy.deepcopy(cfg)


def flatten_metrics(d: dict, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_metrics(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def aggregate_seeds(reports: List[dict]) -> Dict[str, dict]:
    flats = [flatten_metrics(r) for r in reports]
    keys = set().union(*flats) if flats else set()
    agg = {}
    for k in sorted(keys):
        vals = np.array([f[k] for f in flats if k in f and not np.isnan(f[k])], dtype=np.float64)
        if vals.size:
            agg[k] = {"mean": float(vals.mean()), "std": float(vals.std()), "n": int(vals.size)}
    return agg


def run_over_seeds(cfg: Config, seeds: List[int], run_fn: Callable[[Config], dict]) -> dict:
    reports = []
    for s in seeds:
        c = clone_config(cfg)
        c.seed = s
        reports.append(run_fn(c))
    return {"per_seed": reports, "aggregate": aggregate_seeds(reports)}


def _mean_std(values: List[float]) -> dict:
    v = np.array([x for x in values if isinstance(x, (int, float)) and not np.isnan(x)], dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "std": float(v.std()), "n": int(v.size)}


def aggregate_runs(reports: List[dict]) -> dict:
    """Aggregate repeated-seed experiment outputs. Handles both table-shaped
    reports (list of per-variant rows) and flat metric dicts."""
    if reports and all("table" in r for r in reports):
        by_variant: Dict[str, Dict[str, list]] = {}
        for r in reports:
            for row in r["table"]:
                name = row.get("variant", "")
                cols = by_variant.setdefault(name, {})
                for k, v in row.items():
                    if k == "variant":
                        continue
                    cols.setdefault(k, []).append(v)
        table = []
        for name, cols in by_variant.items():
            entry = {"variant": name}
            for k, vals in cols.items():
                entry[k] = _mean_std(vals)
            table.append(entry)
        return {"per_seed": reports, "aggregate": {"table": table}}
    return {"per_seed": reports, "aggregate": aggregate_seeds(reports)}


def markdown_table(rows: List[dict], columns: List[str], name_key: str = "variant") -> str:
    header = "| " + " | ".join([name_key] + columns) + " |"
    sep = "|" + "|".join(["---"] * (len(columns) + 1)) + "|"
    lines = [header, sep]
    for r in rows:
        cells = [str(r.get(name_key, ""))]
        for c in columns:
            v = r.get(c)
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fmt_cell(v) -> str:
    if isinstance(v, dict) and "mean" in v:
        if np.isnan(v["mean"]):
            return "-"
        return f"{v['mean']:.4f} ± {v['std']:.4f}"
    if isinstance(v, float):
        return "-" if np.isnan(v) else f"{v:.4f}"
    return str(v)


def report_to_markdown(report: dict, title: str = "Experiment") -> str:
    table = None
    if "table" in report:
        table = report["table"]
    elif isinstance(report.get("aggregate"), dict) and "table" in report["aggregate"]:
        table = report["aggregate"]["table"]
    if not table:
        return f"## {title}\n\n```\n{report}\n```\n"

    columns = [k for k in table[0].keys() if k != "variant"]
    header = "| variant | " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * (len(columns) + 1)) + "|"
    lines = [f"## {title}", "", header, sep]
    for row in table:
        cells = [str(row.get("variant", ""))] + [_fmt_cell(row.get(c)) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def measure_inference_cost(detector, dataset, device, tile_size, overlap, fuse_iou, n: int = 5) -> float:
    from ..engine.infer import tiled_inference

    n = min(n, len(dataset))
    start = time.perf_counter()
    tiled_inference(detector, dataset, device, tile_size, overlap, fuse_iou, max_images=n)
    return (time.perf_counter() - start) / max(1, n)
