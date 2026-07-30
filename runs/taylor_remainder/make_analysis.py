"""Roll up the Taylor-remainder rundirs into the tables the appendix needs.

rundir-only and read-only, in the same spirit as runs/phase2_matrix/make_analysis.py:
every number here is recomputed from rundirs/*/summary.json + coalitions.parquet, so
the analysis can be regenerated without rerunning anything.

Emits (to analysis/):
  remainder_by_condition.csv   per (track, condition): resid1/resid2, floor ratio,
                               order slopes, 3-seed mean +- std
  remainder_pooled.csv         per track: the C.5-style headline row
  order_slopes.csv             coalition-spread vs scale-sweep slopes, side by side
  cost.csv                     trajectory-vs-measurement split per cell

Run:  python runs/taylor_remainder/make_analysis.py [--rundirs DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import yaml

# Table 1's column order, plus the LLM-only condition.
COND_ORDER = ["clean", "free_rider", "grad_noise", "label_flip",
              "answer_swap", "freerider_zero"]
COND_LABEL = {"clean": "clean", "free_rider": "zero-update", "grad_noise": "gradient noise",
              "label_flip": "label-flip", "answer_swap": "answer-swap",
              "freerider_zero": "zero-update"}


def _load(rundir):
    with open(os.path.join(rundir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(rundir, "summary.json")) as f:
        summ = json.load(f)
    metrics_p = os.path.join(rundir, "metrics.json")
    timing = {}
    if os.path.exists(metrics_p):
        with open(metrics_p) as f:
            timing = (json.load(f) or {}).get("timing", {}) or {}
    if cfg.get("smoke"):
        return None                                  # wiring runs are not results
    P = summ["pooled"]
    # A sweep whose ladder never approached the realized displacement measured the loss
    # surface somewhere the estimator never operates; its slope is not evidence.  Older
    # rundirs predate the flag, so recompute it from the stored sweep.
    fr = []
    for r in summ["rounds"]:
        if r.get("sweep") and r.get("norm_dW"):
            fr += [s["norm"] / r["norm_dW"] for s in r["sweep"]]
    sweep_ok = bool(fr) and min(fr) < 1.5 and max(fr) > 0.5
    return dict(
        rundir=os.path.basename(rundir),
        track=cfg.get("track"),
        # The smooth-activation run is an attribution CONTROL, not one of the paper's
        # models; keep it in its own stage so it never pools with a result row.
        stage=(cfg.get("regime")
               or f"{cfg.get('dataset')}/{cfg.get('partition')}"
                  + ("/smooth-ctl" if cfg.get("smooth_control") else "")),
        condition=cfg.get("threat"),
        seed=cfg.get("seed"),
        n_rounds=len(summ["rounds"]),
        norm_dW=float(np.mean([r["norm_dW"] for r in summ["rounds"]])),
        resid1_median=P["resid1"]["median"],
        resid2_median=P["resid2"]["median"],
        floor=P["ulp"],
        resid2_over_floor=P["resid2_median_over_ulp"],
        frac_t2_le_t1=P["frac_t2_le_t1"],
        slope1_coalition=P["loglog_slope_r1"],
        slope2_coalition=P["loglog_slope_r2"],
        sweep_ok=sweep_ok,
        sweep_frac_min=min(fr) if fr else None,
        sweep_frac_max=max(fr) if fr else None,
        # Blank the slopes rather than print a number nobody should read.
        slope1_sweep=P["sweep_slope_r1"] if sweep_ok else None,
        slope2_sweep=P["sweep_slope_r2_above_floor"] if sweep_ok else None,
        mean_abs_u_grand=P["mean_abs_u_grand"],
        closed_vs_shapley=P["max_phi_t2_vs_closed"],
        fl_train_s=timing.get("fl_train_s"),
        measure_s=timing.get("measure_s"),
        train_frac=timing.get("train_frac"),
    )


def _ms(g, col):
    """mean +- std over seeds, matching the paper's \\ms convention."""
    v = g[col].dropna().astype(float)
    if v.empty:
        return ""
    return f"{v.mean():.3g} +- {v.std(ddof=1):.3g}" if len(v) > 1 else f"{v.mean():.3g}"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundirs", default=os.path.join(here, "rundirs"))
    ap.add_argument("--out", default=os.path.join(here, "analysis"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for d in sorted(glob.glob(os.path.join(args.rundirs, "*"))):
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "summary.json")):
            continue
        try:
            r = _load(d)
        except Exception as e:                        # a broken cell must not sink the rollup
            print(f"[skip] {os.path.basename(d)}: {e!r}")
            continue
        if r is not None:
            rows.append(r)
    if not rows:
        raise SystemExit(f"no result rundirs under {args.rundirs} "
                         "(run runs/taylor_remainder/run_table1.sh first)")
    df = pd.DataFrame(rows)
    df["cond_rank"] = df["condition"].map({c: i for i, c in enumerate(COND_ORDER)})
    df = df.sort_values(["track", "stage", "cond_rank", "seed"]).drop(columns="cond_rank")
    df.to_csv(os.path.join(args.out, "cells.csv"), index=False)

    # ---- per-condition, 3-seed aggregate ----
    agg = []
    for (track, stage, cond), g in df.groupby(["track", "stage", "condition"], sort=False):
        agg.append({
            "track": track, "stage": stage,
            "condition": COND_LABEL.get(cond, cond), "seeds": len(g),
            "mean ||dW||": f"{g['norm_dW'].mean():.4g}",
            "resid1 median": _ms(g, "resid1_median"),
            "resid2 median": _ms(g, "resid2_median"),
            "resid2 / fp32 floor": _ms(g, "resid2_over_floor"),
            "frac(resid2<=resid1)": _ms(g, "frac_t2_le_t1"),
            "slope1 coal (pred 2)": _ms(g, "slope1_coalition"),
            "slope2 coal (pred 3)": _ms(g, "slope2_coalition"),
            "slope1 sweep (pred 2)": _ms(g, "slope1_sweep") or "-- out of range",
            "slope2 sweep (pred 3)": _ms(g, "slope2_sweep") or "-- out of range",
            "sweep range (x||dW||)": f"{g['sweep_frac_min'].mean():.3g}-{g['sweep_frac_max'].mean():.3g}",
        })
    by_cond = pd.DataFrame(agg)
    by_cond.to_csv(os.path.join(args.out, "remainder_by_condition.csv"), index=False)

    # ---- per-track headline (the C.5-style row) ----
    head = []
    for (track, stage), g in df.groupby(["track", "stage"], sort=False):
        head.append({
            "track": track, "stage": stage, "cells": len(g),
            "mean ||dW||": f"{g['norm_dW'].mean():.4g}",
            "resid2 median": f"{g['resid2_median'].median():.3e}",
            "fp32 floor": f"{g['floor'].mean():.3e}",
            "resid2 / floor": f"{g['resid2_over_floor'].median():,.1f}x",
            "frac(resid2<=resid1)": f"{g['frac_t2_le_t1'].mean():.3f}",
            "slope2 coal": f"{g['slope2_coalition'].mean():.2f}",
            "slope2 sweep": f"{g['slope2_sweep'].mean():.2f}"
            if g["slope2_sweep"].notna().any() else "",
            "mean |u_r(P_r)|": f"{g['mean_abs_u_grand'].mean():.3e}",
            "closed vs Shapley(u2)": f"{g['closed_vs_shapley'].max():.2e}",
        })
    pd.DataFrame(head).to_csv(os.path.join(args.out, "remainder_pooled.csv"), index=False)

    # ---- order slopes side by side (the part that speaks to A.4 / C.5) ----
    df[["track", "stage", "condition", "seed", "norm_dW",
        "slope1_coalition", "slope2_coalition",
        "slope1_sweep", "slope2_sweep"]].to_csv(
        os.path.join(args.out, "order_slopes.csv"), index=False)

    # ---- cost split ----
    df[["track", "stage", "condition", "seed", "n_rounds",
        "fl_train_s", "measure_s", "train_frac"]].to_csv(
        os.path.join(args.out, "cost.csv"), index=False)

    print(f"# {len(df)} result cells from {args.rundirs}\n")
    print(pd.DataFrame(head).to_string(index=False))
    print()
    print(by_cond.to_string(index=False))
    print(f"\nwrote {args.out}/"
          "{cells,remainder_by_condition,remainder_pooled,order_slopes,cost}.csv")


if __name__ == "__main__":
    main()
