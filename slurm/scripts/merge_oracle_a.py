"""Post-hoc C1 fidelity vs BOTH oracles, rank AND value-level — derived, no re-run.

Reads the run-dirs only and reports, per method, agreement with:
  (b) the in-run 2^N oracle  (truth = methods['(b)oracle'].phi, stored in the traj cell)
  (a) the 2^10 retrain oracle (gt_a = -phi_a, swept separately into the *_aonly cells)
each as Spearman (rank) AND Pearson (value-level, affine-invariant), averaged over seeds
per (dataset, scenario).  Writes the long table to runs/track_c/fidelity.csv (gitignored,
regenerable).  gt_a = -phi_a reproduces the runner's gt['a'] convention (good->low).

Standalone (no PYTHONPATH / CWD assumptions — paths are repo-root relative):
  python slurm/scripts/merge_oracle_a.py
"""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAJ = os.path.join(_REPO, "runs", "track_c", "c1")
ORACLE = os.path.join(_REPO, "runs", "track_c", "c1_oracle")
OUT = os.path.join(_REPO, "runs", "track_c", "fidelity.csv")
COLS = ["spearman_b", "pearson_b", "spearman_a", "pearson_a"]
METHODS = ["Flirds", "Flirds1st", "GTG", "FedSV", "ComFedSV", "Banzhaf",
           "ShapleyFL", "FedIF", "loss-heur", "Ripple"]


def _load(path):
    with open(os.path.join(path, "metrics.json")) as f:
        return json.load(f)


def _pearson(a, b):
    if a.std() == 0 or b.std() == 0:                 # constant vector -> correlation undefined
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    rows = []                                        # long: one record per (ds,scen,seed,method)
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # (ds,scen)->method->col->[..]
    n_cells = 0
    for traj_dir in sorted(glob.glob(os.path.join(TRAJ, "*_seed*"))):
        base = os.path.basename(traj_dir)            # e.g. cifar10_label-flip_seed0
        a_dir = os.path.join(ORACLE, base.replace("_seed", "_aonly_seed"))   # ..._aonly_seed0
        tm = _load(traj_dir)
        gt_b = np.asarray(tm["methods"]["(b)oracle"]["phi"], float)
        gt_a = None
        if os.path.isdir(a_dir):
            gt_a = -np.asarray(_load(a_dir)["phi_a"], float)               # good->low (runner convention)
        else:
            print(f"[warn] no (a) oracle for {base} -> (a) columns blank")
        ds, scen, seed = tm["dataset"], tm["scenario"], tm["seed"]
        for name, m in tm["methods"].items():
            if name == "(b)oracle":
                continue
            phi = np.asarray(m["phi"], float)
            rec = {"dataset": ds, "scenario": scen, "seed": seed, "method": name,
                   "spearman_b": float(spearmanr(phi, gt_b).correlation),
                   "pearson_b": _pearson(phi, gt_b),
                   "spearman_a": float(spearmanr(phi, gt_a).correlation) if gt_a is not None else "",
                   "pearson_a": _pearson(phi, gt_a) if gt_a is not None else ""}
            rows.append(rec)
            for c in COLS:
                if rec[c] != "":
                    agg[(ds, scen)][name][c].append(rec[c])
        n_cells += 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "scenario", "seed", "method"] + COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nmerged {n_cells} cells -> {OUT}\n")
    for key in sorted(agg):
        ds, scen = key
        nseed = max((len(v.get("spearman_b", [])) for v in agg[key].values()), default=0)
        print(f"=== C1 fidelity {ds}_{scen} ({nseed} seeds) — rank | value-level ===")
        print(f"  {'method':10s} {'rho_b':>7s} {'r_p_b':>7s} {'rho_a':>7s} {'r_p_a':>7s}")
        for name in METHODS:
            if name not in agg[key]:
                continue
            c = agg[key][name]
            cell = (f"{np.nanmean(c[col]):+.3f}" if c.get(col) else "   -  " for col in COLS)
            print(f"  {name:10s} " + " ".join(f"{x:>7s}" for x in cell))
        print()


if __name__ == "__main__":
    main()
