"""Regenerate every figure in results/figures from results/.

No GPU and no retraining: all figures are derived from the saved per-run reports and
per-candidate predictions, so adding a run and re-running this reflects it everywhere.

    python scripts/make_figures.py
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "runs")
ANALYSIS = os.path.join(ROOT, "results", "analysis")
FIGURES = os.path.join(ROOT, "results", "figures")

GROUPS = ["style 0", "style 1", "style 2", "pooled"]
LABEL = {"erm": "ERM", "group_balanced": "PGE", "group_dro": "DRO-style"}
COLOUR = {"erm": "#c44e52", "group_balanced": "#4c72b0", "group_dro": "#55a868"}


def _load(name):
    p = os.path.join(ANALYSIS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _report(tag):
    return json.load(open(os.path.join(RUNS, f"report_{tag}.json")))


def _worst_group(tag):
    return (_report(tag).get("detection") or {}).get("worst_group_AP50")


def figure_paired_comparison(path):
    """PGE vs ERM on every seed. Seeds are exchangeable, so pairs are drawn as
    connected points rather than a line, which would imply an ordering."""
    seeds, pge, erm, collapsed = [], [], [], []
    for s in range(10):
        a = os.path.join(RUNS, f"report_group_balanced_seed{s}.json")
        b = os.path.join(RUNS, f"report_erm_seed{s}.json")
        if not (os.path.exists(a) and os.path.exists(b)):
            continue
        seeds.append(s)
        pge.append(_worst_group(f"group_balanced_seed{s}") or 0.0)
        erm.append(_worst_group(f"erm_seed{s}") or 0.0)
        collapsed.append(bool(_report(f"erm_seed{s}").get("collapsed")))

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, s in enumerate(seeds):
        ax.plot([i, i], [erm[i], pge[i]], color="#bbbbbb", lw=1.4, zorder=1)
    ax.scatter(range(len(seeds)), erm, s=52, color=COLOUR["erm"], label="ERM", zorder=3)
    ax.scatter(range(len(seeds)), pge, s=52, color=COLOUR["group_balanced"], label="PGE", zorder=3)
    bad = [i for i, c in enumerate(collapsed) if c]
    if bad:
        ax.scatter(bad, [erm[i] for i in bad], marker="x", s=90, color="black",
                   zorder=4, label="ERM collapsed")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.set_xlabel("seed")
    ax.set_ylabel(r"worst-group AP$_{50}$")
    sig = _load("significance_pge_vs_erm.json") or {}
    p = sig.get("p_sign")
    ax.set_title("PGE exceeds ERM on every seed"
                 + (f"  (sign test $p={p:.3f}$)" if p else ""))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_dataset_composition(path):
    """Why the minority group is hard to measure: it is small at every split."""
    stats = _load("dataset_statistics.json")
    if not stats:
        return
    splits = ["train", "val", "test"]
    styles = ["style0", "style1", "style2"]
    counts = [[stats["composition"]["splits"][s]["group_images"].get(g, 0) for s in splits]
              for g in styles]
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.4))
    bottom = np.zeros(len(splits))
    for g, c in zip(styles, counts):
        ax[0].bar(splits, c, bottom=bottom, label=g)
        bottom += np.array(c, dtype=float)
    ax[0].set_ylabel("images")
    ax[0].set_title("(a) visual-style groups per split")
    ax[0].legend(fontsize=8)
    ax[0].grid(axis="y", alpha=0.3)

    bg = stats["composition"]["splits"]["train"].get("metadata_background", {})
    if bg:
        keys = sorted(bg, key=lambda k: -bg[k])
        ax[1].barh(keys, [bg[k] for k in keys], color="#4c72b0")
        ax[1].set_xlabel("training images")
        ax[1].set_title("(b) recorded background colour")
        ax[1].grid(axis="x", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_subgroup_calibration(path):
    """Per-group C-ECE with bootstrap intervals, and the stage decomposition."""
    boot, stage = _load("subgroup_bootstrap_ci.json"), _load("calibration_stage_decomposition.json")
    if not (boot and stage):
        return
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))

    point = [np.mean([boot[k]["point"] for k in boot if k.endswith("|" + g)]) for g in GROUPS]
    lo = [np.mean([boot[k]["lo"] for k in boot if k.endswith("|" + g)]) for g in GROUPS]
    hi = [np.mean([boot[k]["hi"] for k in boot if k.endswith("|" + g)]) for g in GROUPS]
    ax[0].bar(GROUPS, point,
              yerr=[np.array(point) - np.array(lo), np.array(hi) - np.array(point)],
              capsize=4, color=["#4c72b0"] * 3 + ["#9e9e9e"])
    ax[0].set_ylabel("fused-score C-ECE")
    ax[0].set_title("(a) subgroup calibration, bootstrap 95% CI")
    ax[0].tick_params(axis="x", rotation=15)
    ax[0].grid(axis="y", alpha=0.3)

    width, x = 0.26, np.arange(len(GROUPS))
    for i, (key, name) in enumerate([("p_det", "detector"), ("p_ver", "verifier"), ("fused", "fused")]):
        ax[1].bar(x + i * width, [np.mean([stage[t][g][key] for t in stage]) for g in GROUPS],
                  width, label=name)
    ax[1].set_xticks(x + width)
    ax[1].set_xticklabels(GROUPS, rotation=15)
    ax[1].set_ylabel("candidate ECE")
    ax[1].set_title("(b) where the miscalibration originates")
    ax[1].legend(fontsize=8)
    ax[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_cluster_selection(path):
    """Cluster-count selection: stability against degeneracy."""
    ks = _load("cluster_count_selection.json")
    if not ks:
        return
    kk = sorted(int(k) for k in ks)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(kk, [ks[str(k)]["ari_mean"] for k in kk], "o-", label="bootstrap ARI", color="#4c72b0")
    ax.plot(kk, [ks[str(k)]["silhouette"] for k in kk], "s--", label="silhouette", color="#dd8452")
    ax.axhline(0.9, color="grey", ls=":", lw=1)
    ax.axvline(3, color="#c44e52", ls=":", lw=1.5)
    ax.annotate("degenerate cluster\nappears at k=4", xy=(4, ks["4"]["ari_mean"]),
                xytext=(4.6, 0.62), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("number of clusters $k$")
    ax.set_title("Visual-style cluster count")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIGURES, exist_ok=True)
    figure_paired_comparison(os.path.join(FIGURES, "pge_vs_erm_paired"))
    figure_dataset_composition(os.path.join(FIGURES, "dataset_composition"))
    figure_subgroup_calibration(os.path.join(FIGURES, "subgroup_calibration"))
    figure_cluster_selection(os.path.join(FIGURES, "cluster_selection"))
    for f in sorted(glob.glob(os.path.join(FIGURES, "*"))):
        print("  ", os.path.relpath(f, ROOT))


if __name__ == "__main__":
    main()
