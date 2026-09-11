#!/usr/bin/env python3
"""Consolidate all run results into one report plus LaTeX tables.

Reads only `results/`. Re-run after new seeds land; it recomputes over whatever
is present and re-derives the significance tests.

    python3 build_report.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "results", "analysis")
RUNS = os.path.join(ROOT, "results", "runs")
T = os.path.join(ROOT, "results", "tables")
os.makedirs(T, exist_ok=True)
GN = ["style 0", "style 1", "style 2", "pooled"]


def load(name):
    p = os.path.join(A, name)
    return json.load(open(p)) if os.path.exists(p) else None


def reports():
    out = {}
    for p in sorted(glob.glob(f"{RUNS}/report_*.json")):
        out[os.path.basename(p)[len("report_"):-len(".json")]] = json.load(open(p))
    return out


def wg(r):
    return (r.get("detection") or {}).get("worst_group_AP50")


# ------------------------------------------------------------------ table 2
def table_reducers(R):
    rows, tab = [], {}
    label = {"erm": "ERM", "group_balanced": "PGE", "group_dro": "DRO-style (submitted)"}
    for m in ["erm", "group_balanced", "group_dro"]:
        per = []
        for s in range(10):
            r = R.get(f"{m}_seed{s}")
            if r is None:
                continue
            per.append((s, float(wg(r) or 0.0), bool(r.get("collapsed"))))
        if per:
            tab[m] = per
    for m in ["group_cycling"]:
        per = [(int(k.rsplit("seed", 1)[1]), float(wg(v) or 0.0), bool(v.get("collapsed")))
               for k, v in R.items() if k.startswith("group_cycling")]
        if per:
            tab[m] = sorted(per)
            label[m] = "DRO-style (group cycling + accumulation)"
    lines = []
    for m, per in tab.items():
        vals = np.array([v for _, v, _ in per])
        cell = ", ".join(f"{v:.2f}" + (r"$^{\dagger}$" if c else "") for _, v, c in per)
        lines.append(f"{label[m]} & {len(per)} & {cell} & "
                     f"{vals.mean():.2f} $\\pm$ {vals.std():.2f} \\\\")
    body = ("Method & $n$ & Per-seed worst-group AP$_{50}$ & Mean $\\pm$ SD \\\\\n"
            "\\midrule\n" + "\n".join(lines))
    return tab, tex(body,
        "Worst-group AP at IoU 0.5 by training reducer. $\\dagger$ marks runs that "
        "collapsed to an all-background solution; those detectors emitted 1--4 boxes over "
        "the entire test set, so calibration is undefined for them.",
        "tab:reducers", "llll")


# ------------------------------------------------------------------ significance
def significance(R):
    seeds = sorted({s for s in range(10)
                    if R.get(f"erm_seed{s}") and R.get(f"group_balanced_seed{s}")})
    pairs = [(s, wg(R[f"group_balanced_seed{s}"]), wg(R[f"erm_seed{s}"])) for s in seeds]
    pairs = [(s, a, b) for s, a, b in pairs if a is not None and b is not None]
    if not pairs:
        return None
    d = np.array([a - b for _, a, b in pairs])
    n, w = len(d), int((d > 0).sum())
    out = {"seeds": [s for s, _, _ in pairs], "n": n, "pge_wins": w,
           "pge": [a for _, a, _ in pairs], "erm": [b for _, _, b in pairs],
           "mean_diff": float(d.mean())}
    try:
        from scipy import stats
        out["p_sign"] = float(stats.binomtest(w, n, 0.5, alternative="greater").pvalue)
        if n >= 5:
            out["p_wilcoxon"] = float(stats.wilcoxon(d, alternative="greater").pvalue)
    except Exception:
        out["p_sign"] = float(0.5 ** n) if w == n else None
    rng = np.random.default_rng(0)
    bs = [np.mean(rng.choice(d, n, replace=True)) for _ in range(10000)]
    out["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


# ------------------------------------------------------------------ subgroup table
def table_subgroups():
    cnt, boot, occ = load("subgroup_candidate_counts.json"), load("subgroup_bootstrap_ci.json"), None
    if not (cnt and boot):
        return None
    occ = cnt["occupancy"]
    rows = []
    for g in GN[:3]:
        rs = [r for r in cnt["per_group"] if r["group"] == g]
        ks = [k for k in boot if k.endswith("|" + g)]
        rows.append(f"{g} & {int(np.mean([r['n_img_with_cand'] for r in rs]))} & "
                    f"{int(np.mean([r['n_cand'] for r in rs]))} & "
                    f"{int(np.mean([o['occupied_bins'] for o in occ if o['group']==g]))}/15 & "
                    f"{np.mean([boot[k]['point'] for k in ks]):.2f} & "
                    f"[{np.mean([boot[k]['lo'] for k in ks]):.2f}, "
                    f"{np.mean([boot[k]['hi'] for k in ks]):.2f}] \\\\")
    ks = [k for k in boot if k.endswith("|pooled")]
    rows.append("\\midrule\npooled & "
                f"{int(np.mean([boot[k]['n_img'] for k in ks]))} & "
                f"{int(np.mean([boot[k]['n_cand'] for k in ks]))} & -- & "
                f"{np.mean([boot[k]['point'] for k in ks]):.2f} & "
                f"[{np.mean([boot[k]['lo'] for k in ks]):.2f}, "
                f"{np.mean([boot[k]['hi'] for k in ks]):.2f}] \\\\")
    return tex("Group & Images & Candidates & Occupied bins & C-ECE & 95\\% CI \\\\\n"
               "\\midrule\n" + "\n".join(rows),
        "Subgroup calibration with the candidate counts and bin occupancy behind each "
        "estimate. \\emph{Images} counts only those contributing at least one candidate "
        "above the detector threshold, which is fewer than the nominal per-group test "
        "size. Intervals are percentile bootstrap over test images (2000 resamples), "
        "which accounts for within-image correlation between candidates.",
        "tab:subgroup_counts", "lrrrrr")


def table_stages():
    D = load("calibration_stage_decomposition.json")
    if not D:
        return None
    rows = [f"{g} & " + " & ".join(f"{np.mean([D[t][g][k] for t in D]):.3f}"
            for k in ["p_det", "p_ver", "fused"]) + " \\\\" for g in GN]
    return tex("Group & Detector & Verifier & Fused \\\\\n\\midrule\n" + "\n".join(rows),
        "Stage-wise candidate ECE after per-signal isotonic calibration. The subgroup gap "
        "is already present in the detector and is not reduced by group-conditioned "
        "fusion, which rules out the calibrator as its origin.",
        "tab:stage_decomp", "lrrr")


def table_kselect():
    K = load("cluster_count_selection.json")
    if not K:
        return None
    rows = [f"{k} & {v['inertia']:.0f} & {v['silhouette']:.3f} & {min(v['sizes'])} & "
            f"{v['ari_mean']:.2f} $\\pm$ {v['ari_sd']:.2f} \\\\"
            for k, v in sorted(K.items(), key=lambda x: int(x[0]))]
    return tex("$k$ & Inertia & Silhouette & Smallest cluster & Bootstrap ARI \\\\\n"
               "\\midrule\n" + "\n".join(rows),
        "Cluster-count selection on training-split style features. $k=3$ is the largest "
        "value that is both reproducible under resampling (ARI $\\geq 0.9$) and free of "
        "degenerate clusters.", "tab:kselect", "lrrrr")


def tex(body, caption, label, cols):
    return ("\\begin{table}[t]\n\\centering\n\\small\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{body}\n\\bottomrule\n"
            "\\end{tabular}\n\\end{table}\n")


def main():
    R = reports()
    tab, t_red = table_reducers(R)
    sig = significance(R)
    parts = [p for p in [table_subgroups(), table_stages(), t_red, table_kselect()] if p]
    repro = os.path.join(T, "T7_repro_paragraph.tex")
    if os.path.exists(repro):
        parts.append(open(repro).read())
    open(os.path.join(T, "results_tables.tex"), "w").write("\n\n".join(parts))

    print("=" * 74)
    print(f"RUNS ON DISK: {len(R)}")
    for k in sorted(R):
        r = R[k]
        v = wg(r)
        c = (r.get("calibration") or {}).get("D-ECE")
        print(f"   {k:34s} wgAP50={v if v is None else round(v,3):<7} "
              f"C-ECE={'n/a' if c is None else round(c,3):<7}"
              f"{'  COLLAPSED' if r.get('collapsed') else ''}")

    if sig:
        print("\nPGE vs ERM")
        for s, a, b in zip(sig["seeds"], sig["pge"], sig["erm"]):
            print(f"   seed {s}: PGE {a:.3f}  ERM {b:.3f}  diff {a-b:+.3f}")
        print(f"   n={sig['n']}, PGE wins {sig['pge_wins']}")
        print(f"   sign test p = {sig['p_sign']:.4f}"
              f"  ({'SIGNIFICANT' if sig['p_sign'] < 0.05 else 'not significant'} at 0.05)")
        if "p_wilcoxon" in sig:
            print(f"   Wilcoxon  p = {sig['p_wilcoxon']:.4f}")
        print(f"   mean diff {sig['mean_diff']:+.3f}  95% CI "
              f"[{sig['ci95'][0]:+.3f}, {sig['ci95'][1]:+.3f}]"
              f"{'  (excludes zero)' if sig['ci95'][0] > 0 else ''}")
        json.dump(sig, open(os.path.join(A, "significance_pge_vs_erm.json"), "w"), indent=2)

    print(f"\n-> tables/results_tables.tex ({len(parts)} blocks)")
    print("=" * 74)


if __name__ == "__main__":
    main()
