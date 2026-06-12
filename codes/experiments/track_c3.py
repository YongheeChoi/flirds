"""Track C3 -- cross-seed stability (plan §3.11; reporting axis, ~zero extra cost).

Post-hoc analysis over the multi-seed run-dirs C1/C2 already wrote -- NO new
training.  Answers the Data-Banzhaf stability question (Wang & Jia 2023: ranking
stability across SGD randomness) + the Volatility prescription (Geimer: report
stability ALONGSIDE GT-similarity), per method:

  C1 mode (phi stability): for each valuation method, collect its phi vector per
    seed (metrics['methods'][name]['phi'] + the (a) oracle), then report
      - cross-seed mean pairwise Spearman (rank stability),
      - top-k% / bottom-k% client-set Jaccard across seed pairs (the decision a
        weighting/dismissal arm actually makes).
    Banzhaf/Flirds are expected most stable; FedSV/ShapleyFL (MC / within-subset
    renorm) least -- the headline C3 table.

  C2 mode (outcome stability): for each intervention arm, cross-seed mean/std of
    final_acc + detection AUROC -- the Volatility 'aggregation-strategy variance'
    response (does the intervention's benefit survive seed variance?).

Run (from codes/):
  PYTHONPATH=. python experiments/track_c3.py c1 runs/track_c1 mnist label_flip
  PYTHONPATH=. python experiments/track_c3.py c2 runs/track_c2 cifar10 dir1 label_flip
With no positional filter it auto-groups every (dataset, scenario/partition+threat).
"""
from __future__ import annotations

import glob
import json
import os
import sys
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

TOPQ = 0.2                                                # top/bottom 20% client set


def _load(root):
    out = []
    for mj in sorted(glob.glob(os.path.join(root, "*", "metrics.json"))):
        out.append((os.path.basename(os.path.dirname(mj)), json.load(open(mj))))
    return out


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def _topset(phi, q, bottom):
    """Indices of the bottom-q (most-corrupt, highest phi good->low) or top-q."""
    k = max(1, int(q * len(phi)))
    order = np.argsort(phi)                               # ascending = best(low phi)->worst
    return order[:k] if not bottom else order[-k:]


def _phi_vectors(runs):
    """{method: [phi_seed0, phi_seed1, ...]} from C1 metrics (+ (a) oracle)."""
    by_method = {}
    for _, m in runs:
        for name, d in m["methods"].items():
            by_method.setdefault(name, []).append(np.asarray(d["phi"], dtype=float))
        if m.get("oracle_a"):
            by_method.setdefault("(a)oracle", []).append(np.asarray(m["oracle_a"]["phi"], dtype=float))
    return by_method


def c1_report(root, runs):
    print(f"\n=== C1 stability {root} ({len(runs)} seed(s)) ===")
    if len(runs) < 2:
        print("  (need >=2 seeds for cross-seed stability)")
    by_method = _phi_vectors(runs)
    print(f"  {'method':11s} {'rho_xseed':>9s} {'topJ':>6s} {'botJ':>6s}   (top/bot = "
          f"{int(TOPQ*100)}% client-set Jaccard)")
    for name, vecs in by_method.items():
        if len(vecs) < 2:
            continue
        rhos = [spearmanr(a, b).correlation for a, b in combinations(vecs, 2)]
        topj = [_jaccard(_topset(a, TOPQ, False), _topset(b, TOPQ, False))
                for a, b in combinations(vecs, 2)]
        botj = [_jaccard(_topset(a, TOPQ, True), _topset(b, TOPQ, True))
                for a, b in combinations(vecs, 2)]
        print(f"  {name:11s} {np.nanmean(rhos):9.3f} {np.mean(topj):6.3f} {np.mean(botj):6.3f}")


def c2_report(root, runs):
    print(f"\n=== C2 outcome stability {root} ({len(runs)} seed(s)) ===")
    arms = {}
    for _, m in runs:
        for arm, d in m["arms"].items():
            arms.setdefault(arm, {"acc": [], "auroc": []})
            arms[arm]["acc"].append(d["final_acc"])
            if d.get("auroc") is not None and not np.isnan(d["auroc"]):
                arms[arm]["auroc"].append(d["auroc"])
    print(f"  {'arm':14s} {'acc_mean':>8s} {'acc_std':>7s} {'auroc_mean':>10s} {'auroc_std':>9s}")
    for arm, d in arms.items():
        a = np.asarray(d["acc"]); au = np.asarray(d["auroc"])
        am = f"{au.mean():10.3f}" if au.size else f"{'--':>10s}"
        ast = f"{au.std():9.3f}" if au.size else f"{'--':>9s}"
        print(f"  {arm:14s} {a.mean():8.4f} {a.std():7.4f} {am} {ast}")


def _group_key(name, ndrop):
    """Strip the trailing _seed<k> to group seeds of the same config."""
    parts = name.split("_")
    return "_".join(parts[:-1]) if parts[-1].startswith("seed") else name


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "c1"
    root = sys.argv[2] if len(sys.argv) > 2 else f"runs/track_{mode}"
    filt = "_".join(sys.argv[3:]) if len(sys.argv) > 3 else None

    runs = _load(root)
    if not runs:
        print(f"no run-dirs under {root}")
        return
    groups = {}
    for name, m in runs:
        key = _group_key(name, 1)
        if filt and not name.startswith(filt):
            continue
        groups.setdefault(key, []).append((name, m))
    if not groups:
        print(f"no run-dirs match filter {filt!r} under {root}")
        return
    report = c1_report if mode == "c1" else c2_report
    for key, g in sorted(groups.items()):
        report(key, g)


if __name__ == "__main__":
    main()
