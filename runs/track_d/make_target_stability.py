#!/usr/bin/env python
"""Exp C -- (b) oracle TARGET self-stability across seeds (review C-2 direct answer).

The headline fidelity "+1.000" is agreement WITH the (b) in-run oracle.  C-2 (review-claude
K-1/K-6) notes that the matching target (b) itself was never persisted as a seed-stability
number for the LLM stage -- only noted.  This DERIVES it (no re-run): read each cell's saved
(b)oracle per-client phi across seeds, pivot client x seed, and report the pairwise seed
Spearman -- how reproducible the ground truth we match against actually is.

Handles both rundir layouts transparently:
  - track_d:        one seed per dir ({scale}_{regime}_seed{0,1,2}); seed read from the 'seed'
                    column, cell = dirname with the trailing _seed{N} stripped -> 3 dirs merge.
  - phase2_matrix:  many seeds inside one dir's phi.parquet ({scale}_{setting}_{cond}); cell =
                    dirname, seeds already in the 'seed' column.

Low xseed rho => the matched target is seed-unstable, so a per-seed +1.000 rides on a target
that itself reorders across seeds -- exactly the C-2 caveat to report ALONGSIDE fidelity.
Expectation (signal-size diagnosis): IID/clean ~0 or negative; non-IID / corrupted +0.9..1.0.

Usage (no GPU; reads phi.parquet only):
  python runs/track_d/make_target_stability.py                       # track_d rundirs -> track_d/target_stability.csv
  python runs/track_d/make_target_stability.py <rundirs_root> <out.csv> [method]
"""
import csv
import os
import re
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
_SEED_SUFFIX = re.compile(r"_(?:seed|s)\d+$")   # _seed0 (track_d) and _s0 (phase2_matrix R4/aonly)


def cell_stability(root, method="(b)oracle"):
    """Return per-cell (method) target stability rows: for each cell, every seed-pair Spearman
    over the clients present in both seeds + the mean.  A cell = a rundir name with any trailing
    _seed{N} stripped (so track_d's per-seed dirs merge into one cell)."""
    frames = []
    for d in sorted(os.listdir(root)):
        pf = os.path.join(root, d, "phi.parquet")
        if not os.path.exists(pf):
            continue
        df = pd.read_parquet(pf)
        if "method" not in df.columns or method not in set(df["method"]):
            continue
        sub = df[df["method"] == method][["seed", "client", "phi"]].copy()
        sub["cell"] = _SEED_SUFFIX.sub("", d)               # merge per-seed dirs (track_d)
        frames.append(sub)
    if not frames:
        return []
    allrows = pd.concat(frames, ignore_index=True)
    out = []
    for cell, g in allrows.groupby("cell"):
        piv = g.pivot_table(index="client", columns="seed", values="phi")
        seeds = sorted(piv.columns)
        pairs = []
        for si, sj in combinations(seeds, 2):
            pair = piv[[si, sj]].dropna()                   # clients present in both seeds
            if len(pair) < 2:
                continue
            a, b = pair[si].to_numpy(float), pair[sj].to_numpy(float)
            rho = float(spearmanr(a, b).correlation) if a.std() and b.std() else float("nan")
            pairs.append((int(si), int(sj), rho, len(pair)))
        vals = [p[2] for p in pairs if not np.isnan(p[2])]
        mean_rho = float(np.mean(vals)) if vals else float("nan")
        out.append(dict(cell=cell, method=method, n_seeds=len(seeds), n_pairs=len(pairs),
                        mean_xseed_spearman=mean_rho,
                        min_xseed_spearman=(float(np.min(vals)) if vals else float("nan")),
                        pairs=";".join(f"{i}-{j}:{r:+.3f}(n{n})" for i, j, r, n in pairs)))
    return sorted(out, key=lambda r: r["cell"])


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "rundirs")
    out_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_HERE, "target_stability.csv")
    method = sys.argv[3] if len(sys.argv) > 3 else "(b)oracle"

    rows = cell_stability(root, method)
    fields = ["cell", "method", "n_seeds", "n_pairs", "mean_xseed_spearman",
              "min_xseed_spearman", "pairs"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"target-stability ({method}): {root}\n  {len(rows)} cells -> {out_csv}")
    print(f"\n  {'cell':32s} {'n_seed':>6s} {'mean_xseed_rho':>14s} {'min':>8s}   pairs")
    for r in rows:
        print(f"  {r['cell']:32s} {r['n_seeds']:6d} {r['mean_xseed_spearman']:+14.3f} "
              f"{r['min_xseed_spearman']:+8.3f}   {r['pairs']}")
    if not rows:
        print(f"  (no cells with method {method!r} found under {root})")


if __name__ == "__main__":
    main()
