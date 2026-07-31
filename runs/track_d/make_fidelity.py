#!/usr/bin/env python
"""Track D fidelity table — derived, no re-run.  Reads each rundir's phi.parquet and
recomputes per (cell, seed, method) the agreement vs the (b) in-run oracle: rank
(Spearman, Kendall) + value-level (Pearson, affine-invariant) + the GTG distance trio
(cosine / euclidean / max).  Writes runs/track_d/fidelity.csv (gitignored, regenerable)
and prints a per-cell Spearman|Pearson summary.

Mirrors the runner's report_fidelity (truth = (b)oracle, over the clients present in
phi.parquet = the selected set; the (a) retrain oracle is scored vs (b) too = the
dual-oracle agreement).

Where the cell also carries an (a) retrain oracle, each method is ALSO scored
directly against it -> spearman_a / pearson_a, same convention as the five-domain
runs/phase2_matrix/merge_silo5_a.py.  Scoring vs (a) only through (b) is valid just
when a method reproduces (b) exactly; the direct column removes that precondition
(supplement tab:llm-retrain, Alpaca row).  Cells with no (a) column leave it blank.
Standalone:
  python runs/track_d/make_fidelity.py
"""
import csv
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))      # runs/track_d
sys.path.insert(0, ROOT)                                # sibling: make_target_stability
RUNDIRS = os.path.join(ROOT, "rundirs")
OUT = os.path.join(ROOT, "fidelity.csv")
TRUTH = "(b)oracle"
TRUTH_A = "(a)oracle"
FIELDS = ["cell", "seed", "method", "spearman", "kendall", "pearson",
          "cosine_d", "euclid_d", "max_diff", "spearman_a", "pearson_a"]


def _fidelity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return dict(
        spearman=float(spearmanr(a, b).correlation),
        kendall=float(kendalltau(a, b).correlation),
        pearson=float("nan") if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1]),
        cosine_d=float(1 - (a @ b) / (na * nb)) if na and nb else float("nan"),
        euclid_d=float(np.linalg.norm(a - b)),
        max_diff=float(np.abs(a - b).max()))


def _vs_a(piv, m):
    """Direct rank/value agreement of method m with the (a) retrain oracle, or blanks
    when the cell has no (a) column (most cells) or m IS the (a) oracle."""
    if TRUTH_A not in piv.columns or m == TRUTH_A:
        return dict(spearman_a="", pearson_a="")
    pair = piv[[m, TRUTH_A]].dropna()
    if len(pair) < 2:
        return dict(spearman_a="", pearson_a="")
    a, b = pair[m].to_numpy(float), pair[TRUTH_A].to_numpy(float)
    return dict(
        spearman_a=float(spearmanr(a, b).correlation),
        pearson_a=float("nan") if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1]))


def main():
    rows = []
    cells = sorted(glob.glob(os.path.join(RUNDIRS, "*")))
    for d in cells:
        pf = os.path.join(d, "phi.parquet")
        if not os.path.exists(pf):
            continue
        cell = os.path.basename(d)
        phi = pd.read_parquet(pf)
        for seed, g in phi.groupby("seed"):
            piv = g.pivot_table(index="client", columns="method", values="phi")
            if TRUTH not in piv.columns:
                continue
            for m in piv.columns:
                if m == TRUTH:
                    continue
                pair = piv[[m, TRUTH]].dropna()            # shared (selected) clients
                if len(pair) < 2:
                    continue
                rows.append(dict(cell=cell, seed=int(seed), method=m,
                                 **_fidelity(pair[m].to_numpy(float), pair[TRUTH].to_numpy(float)),
                                 **_vs_a(piv, m)))

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"track_d fidelity: {len(cells)} cells -> {OUT} ({len(rows)} rows)")

    df = pd.DataFrame(rows)
    for cell in sorted(df["cell"].unique()) if len(df) else []:
        sub = df[df["cell"] == cell]
        has_a = sub["spearman_a"].astype(str).str.len().gt(0).any()
        print(f"\n=== {cell} (vs {TRUTH}{f' | vs {TRUTH_A}' if has_a else ''}) ===")
        print(f"  {'method':10s} {'Spearman':>9s} {'Pearson':>8s}"
              + (f" {'Spear(a)':>9s} {'Pears(a)':>9s}" if has_a else ""))
        for m in sub["method"]:
            s = sub[sub["method"] == m].iloc[0]
            line = f"  {m:10s} {s['spearman']:+9.3f} {s['pearson']:+8.3f}"
            if has_a:
                line += (f" {s['spearman_a']:+9.3f} {s['pearson_a']:+9.3f}"
                         if s["spearman_a"] != "" else f" {'-':>9s} {'-':>9s}")
            print(line)

    # ---- Exp C: report the (b) TARGET self-stability alongside fidelity (review C-2, §4/§5.1) ----
    # fidelity's per-seed +rho is agreement WITH the (b) oracle; if that target itself reorders
    # across seeds the +1.000 is riding on an unstable ground truth -> report both, always.
    try:
        from make_target_stability import cell_stability
        ts = cell_stability(RUNDIRS, "(b)oracle")
        print(f"\n=== (b) target self-stability across seeds (Exp C / review C-2) ===")
        print(f"  {'cell':24s} {'n_seed':>6s} {'mean_xseed_rho':>14s}   pairs")
        for r in ts:
            print(f"  {r['cell']:24s} {r['n_seeds']:6d} {r['mean_xseed_spearman']:+14.3f}   {r['pairs']}")
    except Exception as e:                                  # never break the fidelity table on this add-on
        print(f"\n[target-stability] skipped ({e!r})")


if __name__ == "__main__":
    main()
