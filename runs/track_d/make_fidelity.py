#!/usr/bin/env python
"""Track D fidelity table — derived, no re-run.  Reads each rundir's phi.parquet and
recomputes per (cell, seed, method) the agreement vs the (b) in-run oracle: rank
(Spearman, Kendall) + value-level (Pearson, affine-invariant) + the GTG distance trio
(cosine / euclidean / max).  Writes runs/track_d/fidelity.csv (gitignored, regenerable)
and prints a per-cell Spearman|Pearson summary.

Mirrors the runner's report_fidelity (truth = (b)oracle, over the clients present in
phi.parquet = the selected set; the (a) retrain oracle is scored vs (b) too = the
dual-oracle agreement).  Standalone:
  python runs/track_d/make_fidelity.py
"""
import csv
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))      # runs/track_d
RUNDIRS = os.path.join(ROOT, "rundirs")
OUT = os.path.join(ROOT, "fidelity.csv")
TRUTH = "(b)oracle"
FIELDS = ["cell", "seed", "method", "spearman", "kendall", "pearson",
          "cosine_d", "euclid_d", "max_diff"]


def _fidelity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return dict(
        spearman=float(spearmanr(a, b).correlation),
        kendall=float(kendalltau(a, b).correlation),
        pearson=float("nan") if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1]),
        cosine_d=float(1 - (a @ b) / (na * nb)) if na and nb else float("nan"),
        euclid_d=float(np.linalg.norm(a - b)),
        max_diff=float(np.abs(a - b).max()))


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
                                 **_fidelity(pair[m].to_numpy(float), pair[TRUTH].to_numpy(float))))

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"track_d fidelity: {len(cells)} cells -> {OUT} ({len(rows)} rows)")

    df = pd.DataFrame(rows)
    for cell in sorted(df["cell"].unique()) if len(df) else []:
        sub = df[df["cell"] == cell]
        print(f"\n=== {cell} (vs {TRUTH}) ===")
        print(f"  {'method':10s} {'Spearman':>9s} {'Pearson':>8s}")
        for m in sub["method"]:
            s = sub[sub["method"] == m].iloc[0]
            print(f"  {m:10s} {s['spearman']:+9.3f} {s['pearson']:+8.3f}")


if __name__ == "__main__":
    main()
