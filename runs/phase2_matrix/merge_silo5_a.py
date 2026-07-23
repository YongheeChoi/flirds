"""L8/T5 silo5 (a)-leg join: fidelity vs BOTH oracles, derived, no re-run.

Reads the canonical silo5 rundirs (methods + (b) oracle on the realized trajectory)
and the `*_aonly` rundirs (the exact retrain (a) oracle from track_a_silo5.py), joins
them per (seed, client), and reports each method's agreement with:
  (b) the in-run 2^N oracle  (truth = canonical phi.parquet, method '(b)oracle')
  (a) the 2^N retrain oracle (canonical *_aonly phi.parquet, method '(a)oracle')
as Spearman (rank) AND Pearson (value-level), averaged over seeds per threat.  Both
phi vectors are suspicion-oriented (good->low), so a positive correlation = agreement.
The '(b)oracle' row's spearman_a is the headline dual-oracle agreement: does (a) rank
the same real signal (b) does (overview §5.4: silo5 clean +0.87 / noisy +0.93).

Standalone (paths repo-root relative; no PYTHONPATH):
  python runs/phase2_matrix/merge_silo5_a.py         # 1B (default)
  SCALE=3B python runs/phase2_matrix/merge_silo5_a.py
"""
import csv
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNDIRS = os.path.join(_REPO, "runs", "phase2_matrix", "rundirs")
SCALE = os.environ.get("SCALE", "1B")
THREATS = ["clean", "noisy", "frzero"]
OUT = os.path.join(_REPO, "runs", "phase2_matrix", f"silo5_a_fidelity_{SCALE}.csv")
METHODS = ["(b)oracle", "Flirds", "Flirds1st", "GTG", "FedSV", "ComFedSV", "Banzhaf",
           "ShapleyFL", "FedIF", "loss-heur"]


def _pearson(a, b):
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _vec(df, method, seed):
    """method's phi as a client-indexed np array for one seed (or None if absent)."""
    g = df[(df["method"] == method) & (df["seed"] == seed)].sort_values("client")
    return g["phi"].to_numpy(float) if len(g) else None


def main():
    rows = []                                            # long: (threat, seed, method, cols)
    agg = {}                                             # threat -> method -> col -> [..]
    for threat in THREATS:
        canon = os.path.join(RUNDIRS, f"{SCALE}_silo5_{threat}")
        cfile = os.path.join(canon, "phi.parquet")
        if not os.path.exists(cfile):
            print(f"[skip] no canonical rundir for {threat} ({cfile})")
            continue
        aonly = sorted(glob.glob(os.path.join(RUNDIRS, f"{SCALE}_silo5_{threat}_aonly*",
                                              "phi.parquet")))
        if not aonly:
            print(f"[skip] no *_aonly rundir for {threat} -> run track_a_silo5.py THREAT={threat}")
            continue
        cdf = pd.read_parquet(cfile)
        adf = pd.concat([pd.read_parquet(f) for f in aonly], ignore_index=True)
        seeds = sorted(set(cdf["seed"]).intersection(adf["seed"]))
        if not seeds:
            print(f"[skip] {threat}: canonical seeds {sorted(set(cdf['seed']))} disjoint from "
                  f"(a) seeds {sorted(set(adf['seed']))}")
            continue
        agg[threat] = {}
        for seed in seeds:
            gt_b = _vec(cdf, "(b)oracle", seed)
            gt_a = _vec(adf, "(a)oracle", seed)
            if gt_b is None or gt_a is None:
                continue
            for m in METHODS:
                mv = _vec(cdf, m, seed)
                if mv is None:
                    continue
                rec = {"threat": threat, "seed": int(seed), "method": m,
                       "spearman_b": (float(spearmanr(mv, gt_b).correlation)
                                      if m != "(b)oracle" else float("nan")),
                       "pearson_b": _pearson(mv, gt_b) if m != "(b)oracle" else float("nan"),
                       "spearman_a": float(spearmanr(mv, gt_a).correlation),
                       "pearson_a": _pearson(mv, gt_a)}
                rows.append(rec)
                d = agg[threat].setdefault(m, {c: [] for c in
                                               ("spearman_b", "pearson_b", "spearman_a", "pearson_a")})
                for c in d:
                    if not np.isnan(rec[c]):
                        d[c].append(rec[c])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["threat", "seed", "method", "spearman_b",
                                          "pearson_b", "spearman_a", "pearson_a"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nmerged {len(rows)} (threat,seed,method) rows -> {OUT}\n")

    for threat in THREATS:
        if threat not in agg:
            continue
        nseed = max((len(v.get("spearman_a", [])) for v in agg[threat].values()), default=0)
        print(f"=== silo5 (a)-leg {threat} ({nseed} seeds) — vs (b) | vs (a) retrain ===")
        print(f"  {'method':10s} {'rho_b':>7s} {'r_p_b':>7s} {'rho_a':>7s} {'r_p_a':>7s}")
        for m in METHODS:
            if m not in agg[threat]:
                continue
            c = agg[threat][m]
            cell = (f"{np.nanmean(c[col]):+.3f}" if c.get(col) else "   -  "
                    for col in ("spearman_b", "pearson_b", "spearman_a", "pearson_a"))
            print(f"  {m:10s} " + " ".join(f"{x:>7s}" for x in cell))
        print()
    print("Headline (T5 §2): the '(b)oracle' row's rho_a = the two oracles' agreement "
          "on the real signal (clean +0.87 / noisy +0.93 target; overview §5.4).")


if __name__ == "__main__":
    main()
