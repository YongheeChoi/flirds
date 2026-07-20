"""Scale (CNN full participation 100/100) -- rundir-only aggregation.

Reads runs/track_h/rundirs_cnn_scale/*/{config.yaml,metrics.json,phi_rounds.parquet}
(frac==1.0 cells only) and emits, per RUN_SCALE.md §7:
  1. absolute test-acc table: arms x threats (+ corrupt-mean col), seed mean+-sd
  2. detection AUROC table (gate arms, corrupt threats)
  3. gate-behavior summary (corrupt vs clean participation rate; pweight rel. weight)
CSV copies land in runs/track_h/scale/analysis/.  Re-runnable from rundirs alone.

Usage: PYTHONPATH=codes python runs/track_h/scale/make_analysis.py [--root DIR]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DEFAULT_ROOT = os.path.join(REPO, "runs", "track_h", "rundirs_cnn_scale")

ARMS = {"observer": "vanilla(observer)", "flirds_gate_v2": "flirds P1(sign)",
        "flirds_cgate": "flirds P5h(cgate)", "flirds_pweight": "flirds P5s(pweight)"}
THREATS = ("clean", "label_flip", "free_rider", "grad_noise")
CORRUPT_THREATS = ("label_flip", "free_rider", "grad_noise")


def load_cells(root):
    """One row per (threat, seed, arm) with final_acc/auroc + rundir context."""
    rows, gate_rows = [], []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        cfgp, mp = os.path.join(d, "config.yaml"), os.path.join(d, "metrics.json")
        if not (os.path.isfile(cfgp) and os.path.isfile(mp)):
            continue
        with open(cfgp) as f:
            cfg = yaml.safe_load(f)
        if float(cfg.get("cfg", {}).get("frac", 0)) != 1.0:
            continue                                   # scale cells only
        with open(mp) as f:
            m = json.load(f)
        threat, seed = cfg["threat"], int(cfg["seed"])
        corrupt = np.asarray(m.get("corrupt", []), dtype=int)
        for arm, am in m.get("arms", {}).items():
            if arm not in ARMS or am.get("final_acc") is None:
                continue
            rows.append(dict(rundir=name, threat=threat, seed=seed, arm=arm,
                             label=ARMS[arm], final_acc=float(am["final_acc"]),
                             auroc=(float(am["auroc"]) if am.get("auroc") is not None
                                    else float("nan")),
                             rounds_to_target=am.get("rounds_to_target")))
        gb = gate_behavior(d, cfg, corrupt)
        if gb is not None:
            gate_rows.append(gb)
    return (pd.DataFrame(rows),
            pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame())


def gate_behavior(d, cfg, corrupt):
    """Per gate arm: corrupt vs clean mean participation (rounds >= burn_in) and,
    for participants, mean relative weight (weight * cohort size; 1.0 == uniform)."""
    pq = os.path.join(d, "phi_rounds.parquet")
    if not os.path.isfile(pq) or corrupt.size == 0 or not corrupt.any():
        return None
    try:
        df = pd.read_parquet(pq)
    except Exception:
        return None
    if "arm" not in df.columns:
        return None
    burn = int(cfg.get("gate", {}).get("burn_in", 10))
    df = df[(df["arm"] != "observer") & (df["round"] >= burn)].copy()
    if df.empty:
        return None
    df["corrupt"] = df["client"].map(lambda c: bool(corrupt[int(c)]))
    csize = df[df["participated"]].groupby(["arm", "round"])["client"].transform("size") \
        if df["participated"].any() else None
    df["rel_w"] = df["weight"] * csize if csize is not None else float("nan")
    out = []
    for (arm, corr), g in df.groupby(["arm", "corrupt"]):
        p = g[g["participated"]]
        out.append(dict(rundir=os.path.basename(d), threat=cfg["threat"],
                        seed=int(cfg["seed"]), arm=arm,
                        group="corrupt" if corr else "clean",
                        participation=float(g["participated"].mean()),
                        rel_weight=float(p["rel_w"].mean()) if len(p) else float("nan")))
    return pd.DataFrame(out)


def _fmt(vals):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return "--"
    return f"{v.mean():.3f}" + (f"±{v.std(ddof=1):.3f}" if v.size > 1 else "")


def print_table(df, value, title, arms=ARMS):
    print(f"\n## {title}\n")
    cols = [t for t in THREATS] + ["corrupt-mean"]
    hdr = f"| {'arm':22s} | " + " | ".join(f"{c:>14s}" for c in cols) + " |"
    print(hdr)
    print("|" + "-" * 24 + ("|" + "-" * 16) * len(cols) + "|")
    for arm, label in arms.items():
        cells = []
        per_threat_means = []
        for t in THREATS:
            v = df[(df["arm"] == arm) & (df["threat"] == t)][value]
            cells.append(_fmt(v))
            if t in CORRUPT_THREATS and len(v.dropna()):
                per_threat_means.append(v.dropna().mean())
        cm = (f"{np.mean(per_threat_means):.3f}"
              if len(per_threat_means) == len(CORRUPT_THREATS) else "--")
        print(f"| {label:22s} | " + " | ".join(f"{c:>14s}" for c in cells)
              + f" | {cm:>14s} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()
    df, gate = load_cells(args.root)
    if df.empty:
        print(f"no frac=1.0 rundirs under {args.root}")
        return
    ns = df.groupby("threat")["seed"].nunique().to_dict()
    print(f"# Scale analysis -- root={args.root}")
    print(f"cells: " + ", ".join(f"{t}({ns.get(t, 0)} seeds)" for t in THREATS))

    print_table(df, "final_acc", "absolute test acc (3-seed mean±sd)")
    gate_arms = {a: l for a, l in ARMS.items() if a != "observer"}
    print_table(df[df["threat"].isin(CORRUPT_THREATS)], "auroc",
                "detection AUROC (suspicion = -cum; last-priority metric)", gate_arms)

    if not gate.empty:
        print("\n## gate behavior (rounds >= burn_in; participation / rel. weight)\n")
        agg = (gate.groupby(["arm", "threat", "group"])[["participation", "rel_weight"]]
               .mean().round(3))
        print(agg.to_string())

    outdir = os.path.join(HERE, "analysis")
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "scale_acc.csv"), index=False)
    if not gate.empty:
        gate.to_csv(os.path.join(outdir, "scale_gate_behavior.csv"), index=False)
    print(f"\n[write] {outdir}/scale_acc.csv"
          + ("" if gate.empty else f" + scale_gate_behavior.csv"))


if __name__ == "__main__":
    main()
