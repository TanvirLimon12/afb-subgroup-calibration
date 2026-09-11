import json
import shutil

import numpy as np

from afb_calibration.data.grouping import assign_style_groups, group_sizes
from afb_calibration.data.leakage import cross_split_leakage, remove_cross_split_leaks
from afb_calibration.experiments.common import aggregate_runs, report_to_markdown
from afb_calibration.metrics.plots import plot_reliability, plot_risk_coverage


def _ann(ds):
    return {"train": ds["paths"]["train"], "val": ds["paths"]["val"], "test": ds["paths"]["test"]}


def test_leakage_detection_and_removal(synthetic_dataset, tmp_path):
    # Inject a leak: copy first train image record into test.
    ann = {k: str(tmp_path / f"{k}.json") for k in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        shutil.copy(synthetic_dataset["paths"][split], ann[split])
    train = json.load(open(ann["train"]))
    test = json.load(open(ann["test"]))
    leaked = dict(train["images"][0])
    test["images"].append(leaked)
    json.dump(test, open(ann["test"], "w"))

    root = synthetic_dataset["root"]
    assert cross_split_leakage(ann, root)["num_exact_leaks"] >= 1
    removed = remove_cross_split_leaks(ann, root)
    assert removed["test"] >= 1
    assert cross_split_leakage(ann, root)["num_exact_leaks"] == 0


def test_style_grouping(synthetic_dataset, tmp_path):
    ann = {k: str(tmp_path / f"{k}.json") for k in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        shutil.copy(synthetic_dataset["paths"][split], ann[split])
    assign_style_groups(ann, synthetic_dataset["root"], k=2, seed=0)
    sizes = group_sizes(ann)
    assert all(g.startswith("style") for g in sizes["train"])
    assert sum(sizes["train"].values()) > 0


def test_aggregate_runs_table():
    reports = [
        {"table": [{"variant": "a", "AP50": 0.4}, {"variant": "b", "AP50": 0.6}]},
        {"table": [{"variant": "a", "AP50": 0.5}, {"variant": "b", "AP50": 0.7}]},
    ]
    agg = aggregate_runs(reports)["aggregate"]["table"]
    a_row = next(r for r in agg if r["variant"] == "a")
    assert abs(a_row["AP50"]["mean"] - 0.45) < 1e-6
    assert a_row["AP50"]["n"] == 2


def test_report_markdown():
    report = {"aggregate": {"table": [{"variant": "full", "AP50": {"mean": 0.5, "std": 0.02, "n": 3}}]}}
    md = report_to_markdown(report, "Ablation")
    assert "| variant | AP50 |" in md
    assert "0.5000 ± 0.0200" in md


def test_plots(tmp_path):
    conf = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3])
    correct = np.array([1, 1, 0, 0, 1, 0])
    p1 = plot_reliability(conf, correct, tmp_path / "rel.png")
    p2 = plot_risk_coverage(conf, correct, tmp_path / "rc.png")
    assert (tmp_path / "rel.png").exists() and (tmp_path / "rc.png").exists()
