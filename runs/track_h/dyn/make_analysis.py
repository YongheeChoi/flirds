"""Dyn (per-round corrupt re-draw, R1 stage) -- rundir-only aggregation.

Reads runs/track_h/rundirs_cnn_dyn/*/{config.yaml,metrics.json,phi_rounds.parquet}
and emits, per RUN_DYN.md §7:
  1. absolute test-acc table: arms x threats (+ corrupt-mean), seed mean+-sd
  2. DP-1..3 checks (P5s parity band / P1-vs-vanilla / anchor orderings)
  3. DP-4 diagnostic: P1's would-be-excluded set (cum<=0, r>=burn_in) hit-rate
     on the CURRENT round's corrupt mask vs the 40% chance baseline
CSVs land in runs/track_h/dyn/analysis/.  Re-runnable from rundirs alone.

Usage: PYTHONPATH=codes python runs/track_h/dyn/make_analysis.py [--root DIR]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

from flirds.fl.intervene import make_roundwise_mask

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DEFAULT_ROOT = os.path.join(REPO, "runs", "track_h", "rundirs_cnn_dyn")

ARMS = {"vanilla": "vanilla", "oracle_excl": "oracle_excl(per-round)",
        "random_excl": "random_excl(per-round)", "flirds_gate_v2": "flirds P1(sign)",
        "flirds_pweight": "flirds P5s(pweight)"}
THREATS = ("label_flip", "free_rider", "grad_noise")
PARITY_BAND = 0.006                                   # CNN parity band (overview §3.2.6)


def load_cells(root):
    rows, dp4 = [], []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        cfgp, mp = os.path.join(d, "config.yaml"), os.path.join(d, "metrics.json")
        if not (os.path.isfile(cfgp) and os.path.isfile(mp)):
            continue
        with open(cfgp) as f:
            cfg = yaml.safe_load(f)
        if not cfg.get("dyn", {}).get("roundwise"):
            continue                                   # dyn cells only
        with open(mp) as f:
            m = json.load(f)
        threat, seed = cfg["threat"], int(cfg["seed"])
        for arm, am in m.get("arms", {}).items():
            if arm not in ARMS or am.get("final_acc") is None:
                continue
            rows.append(dict(rundir=name, threat=threat, seed=seed, arm=arm,
                             label=ARMS[arm], final_acc=float(am["final_acc"]),
                             rounds_to_target=am.get("rounds_to_target")))
        h = dp4_hit_rate(d, cfg)
        if h is not None:
            dp4.append(dict(rundir=name, threat=threat, seed=seed, **h))
    return pd.DataFrame(rows), pd.DataFrame(dp4)


def dp4_hit_rate(d, cfg, burn_in=None):
    """P1 would-be-excluded set (cum<=0 snapshot) vs the round's true mask."""
    pq = os.path.join(d, "phi_rounds.parquet")
    if not os.path.isfile(pq):
        return None
    df = pd.read_parquet(pq)
    if "arm" in df.columns:
        df = df[df["arm"] == "flirds_gate_v2"]
    if df.empty:
        return None
    n = int(cfg["cfg"]["n"])
    mask_at = make_roundwise_mask(n, int(cfg["dyn"]["n_corrupt"]),
                                  int(cfg["seed"]))
    burn = burn_in if burn_in is not None else int(cfg.get("gate", {}).get("burn_in", 10))
    hits, sizes = [], []
    for r, g in df[df["round"] >= burn].groupby("round"):
        excl = g[g["cum"] <= 0]["client"].astype(int).tolist()
        if not excl:
            continue
        mk = mask_at(int(r))
        hits.append(np.mean([c in mk for c in excl]))
        sizes.append(len(excl))
    if not hits:
        return None
    return dict(p1_excl_hit_rate=float(np.mean(hits)),
                p1_excl_size_mean=float(np.mean(sizes)),
                chance=float(cfg["dyn"]["n_corrupt"]) / n)


def _fmt(vals):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return "--"
    return f"{v.mean():.4f}" + (f"±{v.std(ddof=1):.4f}" if v.size > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()
    df, dp4 = load_cells(args.root)
    if df.empty:
        print(f"no dyn rundirs under {args.root}")
        return
    print(f"# Dyn analysis -- root={args.root}")
    ns = df.groupby("threat")["seed"].nunique().to_dict()
    print("cells: " + ", ".join(f"{t}({ns.get(t, 0)} seeds)" for t in THREATS))

    print("\n## absolute test acc (3-seed mean±sd)\n")
    cols = list(THREATS) + ["corrupt-mean"]
    print(f"| {'arm':24s} | " + " | ".join(f"{c:>15s}" for c in cols) + " |")
    print("|" + "-" * 26 + ("|" + "-" * 17) * len(cols) + "|")
    van = {t: df[(df.arm == "vanilla") & (df.threat == t)].final_acc.mean()
           for t in THREATS}
    for arm, label in ARMS.items():
        cells, means = [], []
        for t in THREATS:
            v = df[(df.arm == arm) & (df.threat == t)].final_acc
            cells.append(_fmt(v))
            means.append(v.mean())
        cm = f"{np.nanmean(means):.4f}" if not np.isnan(means).all() else "--"
        print(f"| {label:24s} | " + " | ".join(f"{c:>15s}" for c in cells) + f" | {cm:>15s} |")

    print("\n## DP checks (preregistered, RUN_DYN.md §4)\n")
    for t in THREATS:
        d5 = df[(df.arm == "flirds_pweight") & (df.threat == t)].final_acc.mean() - van[t]
        d1 = df[(df.arm == "flirds_gate_v2") & (df.threat == t)].final_acc.mean() - van[t]
        do = df[(df.arm == "oracle_excl") & (df.threat == t)].final_acc.mean() - van[t]
        dr = df[(df.arm == "random_excl") & (df.threat == t)].final_acc.mean() - van[t]
        print(f"{t:11s} dP5s={d5:+.4f} ({'in' if abs(d5) <= PARITY_BAND else 'OUT of'} "
              f"±{PARITY_BAND} band) | dP1={d1:+.4f} | dOracle={do:+.4f} | dRandom={dr:+.4f}")

    if not dp4.empty:
        print("\n## DP-4: P1 would-be-excluded (cum<=0) hit rate on the round's mask\n")
        agg = dp4.groupby("threat")[["p1_excl_hit_rate", "p1_excl_size_mean", "chance"]].mean()
        print(agg.round(3).to_string())

    outdir = os.path.join(HERE, "analysis")
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "dyn_acc.csv"), index=False)
    if not dp4.empty:
        dp4.to_csv(os.path.join(outdir, "dyn_dp4.csv"), index=False)
    print(f"\n[write] {outdir}/dyn_acc.csv" + ("" if dp4.empty else " + dyn_dp4.csv"))


if __name__ == "__main__":
    main()
