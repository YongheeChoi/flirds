"""Post-hoc merge: fidelity of each C1 method vs the (a) 2^10 retrain oracle.

The traj runs stored method phi vectors (good->low) with C1_ORACLE_A=0; the
expensive (a) sweep ran separately into *_aonly dirs (phi = -val-loss SV,
good->high).  This reproduces the runner's gt["a"] = -phi_a convention and
reports per-method Spearman vs (a), averaged over seeds per (dataset, scenario).
Run from codes/ with PYTHONPATH=.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

TRAJ = "runs/track_c1"
ORACLE = "runs/track_c1_oracle"


def load(path):
    with open(os.path.join(path, "metrics.json")) as f:
        return json.load(f)


rows = defaultdict(lambda: defaultdict(list))   # (ds,scen) -> method -> [rho_a,...]
n_cells = 0
for traj_dir in sorted(glob.glob(f"{TRAJ}/*_seed*")):
    base = os.path.basename(traj_dir)
    a_dir = os.path.join(ORACLE, base + "_aonly")
    if not os.path.isdir(a_dir):
        print(f"[skip] no (a) oracle for {base}")
        continue
    tm, am = load(traj_dir), load(a_dir)
    phi_a = np.asarray(am["phi_a"] if "phi_a" in am else am["oracle_a"]["phi"])
    gt_a = -phi_a                                # good->low, runner convention
    ds, scen = tm["dataset"], tm["scenario"]
    for name, m in tm["methods"].items():
        if name == "(b)oracle":
            continue
        rho = spearmanr(np.asarray(m["phi"]), gt_a).correlation
        rows[(ds, scen)][name].append(float(rho))
    n_cells += 1

print(f"\nmerged {n_cells} cells\n")
METHODS = ["Flirds", "Flirds1st", "GTG", "FedSV", "ComFedSV", "Banzhaf",
           "ShapleyFL", "FedIF", "loss-heur", "Ripple"]
for key in sorted(rows):
    ds, scen = key
    print(f"=== (a)-oracle fidelity {ds}_{scen} ({len(next(iter(rows[key].values())))} seeds) ===")
    print(f"  {'method':10s} {'rho_a_mean':>10s} {'rho_a_std':>9s}")
    for name in METHODS:
        vals = rows[key].get(name, [])
        if not vals:
            continue
        print(f"  {name:10s} {np.mean(vals):10.3f} {np.std(vals):9.3f}")
    print()
