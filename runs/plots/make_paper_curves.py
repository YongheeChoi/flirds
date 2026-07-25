#!/usr/bin/env python
"""Paper convergence figures from rundirs alone (re-runnable; NO GPU, NO re-training).

Scope = the two **본문(main-text) downstream** stages that persisted a per-ROUND
trajectory (plan §1 배치표 / [[flirds-results-downstream]]):

  [A] 2-LLM 주무대 정확도 개입 (R4)   gsm50k5 · Llama-3.2-1B · N=50 5/50 · R=200
      curve = metrics.json["val_curve"]  (val loss; curve[0]=init, curve[r]=loss
      entering round r, curve[-1]=deployed)
      arms  = 5-arm 수록 범위 (Yonghee 07-25): observer · oracle_excl · random_excl
              · Flirds · Flirds-1st.  Only the ONLINE leg has curves -- the retrain
              (T2) arms persist final_val_loss only (track_g.py caches curve[-1]),
              and the online Flirds-1st leg is ⬚ 미실행, so both are absent BY DATA,
              not by filtering.  GSM8K EM is a single end-of-run eval -> no EM curve.

  [B] 2-CNN 점수원 경쟁                cifar10/dir1 · FedSVCNN · N=100 10/100 · R=120
      curve = metrics.json["arms"][arm]["acc_curve"] = [[round, test_acc], ...]
      arms  = the 본문 표 roster: anchors + 8 score sources, both timings
              (online = `<src>_gate_v2` · retrain T2 = `t2_sign_<src>`).

Cell grouping + arm parsing are IMPORTED from runs/track_h/make_analysis.py so the
figures and the published tables can never disagree about which rundirs form a cell.

Bands are mean +/- std over seeds with **ddof=0** (project aggregation standard).
A panel whose cell has <3 seeds is titled with ◐ and its n -- §0.3 requires 3-seed
for anything that ships, so an incomplete panel must LOOK incomplete.

Usage:
    python runs/plots/make_paper_curves.py                 # both stages -> runs/plots/figs/
    python runs/plots/make_paper_curves.py --stage llm --dpi 300
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent          # runs/plots
RUNS = ROOT.parent                              # runs/


def _import_track_h():
    """Reuse Track H's cell/arm conventions verbatim (single source of truth)."""
    p = RUNS / "track_h" / "make_analysis.py"
    spec = importlib.util.spec_from_file_location("track_h_make_analysis", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


TH = _import_track_h()

# --------------------------------------------------------------------- styling
# Anchors are grayscale (they are the frame, not a competitor); score sources get
# colour, with same-game solid / renorm cross-game dashed -- the paper's grouping.
ANCHOR = {
    "vanilla":     dict(label="vanilla (floor)",     color="#8a8f94", ls="--",  lw=1.7, zorder=2),
    "observer":    dict(label="observer (floor)",    color="#8a8f94", ls="--",  lw=1.7, zorder=2),
    "oracle_excl": dict(label="oracle-excl (ceil.)", color="#101418", ls=":",   lw=2.2, zorder=4),
    "random_excl": dict(label="random-excl",         color="#bfc4c9", ls="-.",  lw=1.7, zorder=2),
}
SOURCE = {   # source -> (label, colour, linestyle)
    "flirds":    ("Flirds",     "#d62728", "-"),
    "flirds1st": ("Flirds-1st", "#ff7f0e", "-"),
    "lossheur":  ("loss-heur",  "#2ca02c", "-"),
    "fedif":     ("FedIF",      "#8c564b", "-"),
    "gtg":       ("GTG",        "#1f77b4", "--"),
    "fedsv":     ("FedSV",      "#17becf", "--"),
    "comfedsv":  ("ComFedSV",   "#9467bd", "--"),
    "shapleyfl": ("ShapleyFL",  "#e377c2", "--"),
}


def arm_style(arm):
    """Plot style for a rundir arm name; None -> not part of any 본문 roster."""
    if arm in ANCHOR:
        return dict(ANCHOR[arm])
    sp = TH.parse_arm(arm)
    if sp is None or sp[0] not in SOURCE:
        return None
    label, colour, ls = SOURCE[sp[0]]
    return dict(label=label, color=colour, ls=ls, lw=1.8, zorder=3)


# ------------------------------------------------------------------- LLM loader
LLM_ROOTS = ("track_g/rundirs", "track_h/rundirs_llm_l4", "track_h/rundirs_llm_hj",
             "track_h/rundirs_llm_jw", "track_h/rundirs_llm_jb",
             "track_h/rundirs_llm_yh", "track_h/rundirs_llm_p1w",
             "track_h/rundirs_llm_g4c", "track_h/rundirs_llm")   # canonical LAST (wins on dup)


def load_llm_curves(regime="gsm50k5"):
    """Long frame: regime, threat, nr, seed, arm, round, val_loss (one row per round)."""
    by_cell = {}
    for r in LLM_ROOTS:
        for c in TH._load(RUNS / Path(r)):
            cfg = c["cfg"]
            if cfg.get("regime") != regime:
                continue
            key = (cfg["regime"], cfg["threat"], cfg.get("noisy_rate", 1.0),
                   cfg["seed"], cfg["arm"])
            by_cell[key] = c                        # later root wins on dup
    rows = []
    for (reg, threat, nr, seed, arm), c in by_cell.items():
        curve = c["m"].get("val_curve")
        if not isinstance(curve, list) or not curve:
            continue                                # T2/V3 arms: final_val_loss only
        for r, v in enumerate(curve):
            rows.append((reg, threat, nr, seed, arm, r, float(v)))
    return pd.DataFrame(rows, columns=["regime", "threat", "nr", "seed", "arm",
                                       "round", "val_loss"])


# ------------------------------------------------------------------- CNN loader
def load_cnn_curves(dataset="cifar10", partition="dir1"):
    """Long frame: dataset, partition, threat, flip_rate, seed, arm, round, acc,
    equals_vanilla.

    A T2 arm that kept EVERY client is persisted as `skipped=equals_vanilla` with no
    curve of its own -- the retrain is vanilla by construction.  make_analysis scores
    it as delta=0 rather than dropping it; the same must hold here, otherwise the
    grad-noise panel would silently lose the 1st-order arms that never fired, which
    IS the finding.  Those series are re-emitted from the cell's vanilla curve and
    flagged `equals_vanilla`."""
    rows, alias = [], []
    for r in ("track_g/rundirs_cnn", "track_h/rundirs_cnn"):
        for c in TH._load(RUNS / Path(r)):
            m = c["m"]
            if m.get("dataset") != dataset or m.get("partition") != partition:
                continue
            fr, seed, threat = TH._flip_rate(c), m.get("seed"), m.get("threat")
            for arm, a in (m.get("arms") or {}).items():
                if not isinstance(a, dict):
                    continue
                curve = a.get("acc_curve")
                if not isinstance(curve, list) or not curve:
                    if a.get("skipped") == "equals_vanilla":
                        alias.append((threat, fr, seed, arm))
                    continue
                for rd, acc in curve:
                    rows.append((dataset, partition, threat, fr, seed, arm,
                                 int(rd), float(acc), False))
    cols = ["dataset", "partition", "threat", "flip_rate", "seed", "arm", "round",
            "acc", "equals_vanilla"]
    df = pd.DataFrame(rows, columns=cols)
    # track_h wins on dup (same convention as make_analysis): keep last occurrence
    df = df.drop_duplicates(subset=["threat", "flip_rate", "seed", "arm", "round"],
                            keep="last")
    van = df[df["arm"] == "vanilla"]
    extra = []
    for threat, fr, seed, arm in alias:
        v = van[(van["threat"] == threat) & (van["seed"] == seed)
                & (van["flip_rate"].isna() if fr is None else van["flip_rate"] == fr)]
        if v.empty:
            continue
        extra.append(v.assign(arm=arm, equals_vanilla=True))
    return pd.concat([df, *extra], ignore_index=True) if extra else df


# ------------------------------------------------------------------- plotting
def _band(ax, d, ycol, style, show_band=True, smooth=1):
    """mean +/- std (ddof=0) over seeds, plotted against `round`.

    `smooth` > 1 applies a CENTERED rolling mean to the aggregated curve only --
    it never changes the seed statistics, and the figure says it was applied."""
    g = d.groupby("round")[ycol]
    mu, sd, n = g.mean(), g.std(ddof=0), g.size()
    if smooth > 1:
        mu = mu.rolling(smooth, center=True, min_periods=1).mean()
        sd = sd.rolling(smooth, center=True, min_periods=1).mean()
    ax.plot(mu.index, mu.values, **style)
    if show_band and int(n.max()) > 1:
        ax.fill_between(mu.index, (mu - sd).values, (mu + sd).values,
                        color=style["color"], alpha=0.13, lw=0, zorder=1)
    return int(n.max())


def plot_panels(df, *, panels, arms, ycol, ylabel, suptitle, out,
                seed_col="seed", sharey=True, dpi=200, band=True, note=None,
                ncol=3, zoom_from=None, smooth=1):
    """One panel per (title, filter-dict); one line per arm in `arms` (order = legend).

    `panels`    = [(title, {col: value, ...}), ...]
    `arms`      = [rundir arm name, ...] -- anything absent from the data is skipped
                  and reported in the returned coverage table, so a ⬚ leg stays visible.
    `zoom_from` = round index; adds a SECOND row per panel, same series restricted to
                  round >= zoom_from and y-autoscaled.  Needed whenever the arms
                  separate only in the tail (R4: every arm shares one decay).
    """
    if zoom_from is not None:
        nrow, ncol = 2, len(panels)
        grid = [(i, p, False) for i, p in enumerate(panels)] + \
               [(len(panels) + i, p, True) for i, p in enumerate(panels)]
    else:
        ncol = min(ncol, len(panels))
        nrow = int(np.ceil(len(panels) / ncol))
        grid = [(i, p, False) for i, p in enumerate(panels)]
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.2 * nrow),
                             sharey=False if zoom_from is not None else sharey,
                             squeeze=False)
    axes = axes.ravel()
    cover, handles = [], {}
    for idx, (title, flt), is_zoom in grid:
        ax = axes[idx]
        d0 = df
        for k, v in flt.items():
            d0 = d0[d0[k].isna()] if v is None else d0[d0[k] == v]
        nseed = d0[seed_col].nunique()
        if is_zoom:
            d0 = d0[d0["round"] >= zoom_from]
        for ai, arm in enumerate(arms):
            st = arm_style(arm)
            if st is None:
                continue
            # Linewidth taper by draw order: arms that coincide EXACTLY (a T2 arm that
            # kept everyone lies on vanilla; a gate that reproduces the oracle set lies
            # on oracle_excl) would otherwise hide each other -- the earlier, thicker
            # line shows as a halo around the later, thinner one.
            st["lw"] = st["lw"] * (1.0 + 0.55 * (1 - ai / max(1, len(arms) - 1)))
            d = d0[d0["arm"] == arm]
            if d.empty:
                if not is_zoom:
                    cover.append((title, arm, 0))
                continue
            n = _band(ax, d, ycol, st, show_band=band, smooth=smooth)
            if not is_zoom:
                cover.append((title, arm, n))
            handles.setdefault(st["label"], ax.lines[-1])
        mark = "" if nseed >= 3 else f"  ◐ n={nseed}"
        ax.set_title(f"{title}{mark}" if not is_zoom
                     else f"↳ zoom: round ≥ {zoom_from}", fontsize=10.5,
                     color="#101418" if (nseed >= 3 or is_zoom) else "#b45309")
        ax.set_xlabel("round")
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8.5)
        if idx % ncol == 0:
            ax.set_ylabel(ylabel)
    blank = [ax for ax in axes[max(i for i, _, _ in grid) + 1:]]
    for ax in blank:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12, y=0.995)
    lg_kw = dict(fontsize=8.5, frameon=False)
    if blank:                                  # legend rides in the leftover cell
        blank[0].legend(handles.values(), handles.keys(), loc="center",
                        ncol=1, **lg_kw)
        rect = (0, 0.035 if note else 0.0, 1, 0.975)
    else:
        fig.legend(handles.values(), handles.keys(), loc="lower center",
                   ncol=min(6, max(1, len(handles))),
                   bbox_to_anchor=(0.5, 0.022 if note else -0.015), **lg_kw)
        rect = (0, 0.10 if note else 0.085, 1, 0.975)
    if note:
        fig.text(0.5, 0.005, note + (f" · curves smoothed (centered rolling mean, "
                                     f"w={smooth})" if smooth > 1 else ""),
                 ha="center", fontsize=8, color="#5f6368")
    fig.tight_layout(rect=rect)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(cover, columns=["panel", "arm", "n_seed"])


# ------------------------------------------------------------------- figures
LLM_PANELS = [("clean", dict(threat="clean")),
              ("answer-swap @0.7", dict(threat="noisy")),
              ("free-rider (zero)", dict(threat="frzero"))]
LLM_ARMS = ["observer", "oracle_excl", "random_excl", "flirds_gate_v2",
            "flirds1st_gate_v2"]                       # 1st leg ⬚ -> absent by data

CNN_PANELS = [("clean", dict(threat="clean", flip_rate=None)),
              ("free-rider (zero)", dict(threat="free_rider", flip_rate=None)),
              ("free-rider (rand)", dict(threat="frrand", flip_rate=None)),
              ("grad-noise", dict(threat="grad_noise", flip_rate=None)),
              ("label-flip @0.70", dict(threat="label_flip", flip_rate=0.70))]
CNN_SRC = ["flirds", "flirds1st", "lossheur", "fedif", "gtg", "fedsv",
           "comfedsv", "shapleyfl"]
CNN_ONLINE = ["vanilla", "oracle_excl", "random_excl"] + [f"{s}_gate_v2" for s in CNN_SRC]
CNN_RETRAIN = ["vanilla", "oracle_excl"] + [f"t2_sign_{s}" for s in CNN_SRC]


def fig_llm(out_dir, dpi, smooth=1):
    df = load_llm_curves()
    cov = plot_panels(
        df, panels=LLM_PANELS, arms=LLM_ARMS, ycol="val_loss",
        ylabel="validation loss", dpi=dpi, zoom_from=100, smooth=smooth,
        suptitle="[A] R4 main stage — online sign-gating convergence "
                 "(Llama-3.2-1B · gsm50k5 · N=50, 5/50 · R=200)",
        note="band = mean ± std over seeds (ddof=0) · retrain (T2) arms persist "
             "final val-loss only, so the online leg is shown · Flirds-1st online "
             "leg not yet run · clean has no corruption to exclude (no oracle/random)",
        out=out_dir / "fig_r4_llm_convergence")
    return df, cov


def fig_cnn(out_dir, dpi, smooth=1):
    df = load_cnn_curves()
    covs = []
    for tag, arms, what in (("online", CNN_ONLINE, "online deployment gating (gate v2)"),
                            ("retrain", CNN_RETRAIN, "retrain T2 (observer final sign → kept)")):
        covs.append(plot_panels(
            df, panels=CNN_PANELS, arms=arms, ycol="acc",
            ylabel="test accuracy", dpi=dpi, ncol=3, smooth=smooth,
            suptitle=f"[B] CNN score-source competition — {what} "
                     f"(FedSVCNN · cifar10/dir1 · N=100, 10/100 · R=120)",
            note="band = mean ± std over seeds (ddof=0) · solid = same-game "
                 "(Flirds/1st/loss-heur/FedIF) · dashed = renorm cross-game"
                 + (" · a T2 arm that kept everyone lies exactly on vanilla "
                    "(retrain == vanilla by construction)" if tag == "retrain"
                    else " · clean has no corruption to exclude (no oracle/random)"),
            out=out_dir / f"fig_cnn_dir1_{tag}").assign(timing=tag))
    ev = df[df["equals_vanilla"]].groupby(["threat", "arm"])["seed"].nunique()
    if len(ev):
        print("\n  T2 arms that kept ALL clients (curve == vanilla by construction):")
        for (threat, arm), n in ev.items():
            print(f"    {threat:12s} {arm:22s} {n} seed(s)")
    return df, pd.concat(covs, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["llm", "cnn", "all"], default="all")
    ap.add_argument("--out", type=Path, default=ROOT / "figs")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--smooth", type=int, default=1,
                    help="centered rolling-mean window on the AGGREGATED curve "
                         "(1 = raw; only for readability, stated on the figure)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    if a.stage in ("llm", "all"):
        df, cov = fig_llm(a.out, a.dpi, a.smooth)
        df.to_csv(a.out / "curves_r4_llm.csv", index=False)
        cov.to_csv(a.out / "coverage_r4_llm.csv", index=False)
        print(f"[A] LLM R4: {df['seed'].nunique()} seeds · "
              f"{df.groupby(['threat', 'arm']).ngroups} (threat,arm) series · "
              f"{len(df)} rows -> fig_r4_llm_convergence.{{png,pdf}}")
        print(cov.pivot_table(index="arm", columns="panel", values="n_seed",
                              aggfunc="max", fill_value=0).to_string())

    if a.stage in ("cnn", "all"):
        df, cov = fig_cnn(a.out, a.dpi, a.smooth)
        df.to_csv(a.out / "curves_cnn_dir1.csv", index=False)
        cov.to_csv(a.out / "coverage_cnn_dir1.csv", index=False)
        print(f"\n[B] CNN dir1: {df['seed'].nunique()} seeds · "
              f"{df.groupby(['threat', 'flip_rate', 'arm']).ngroups} series · "
              f"{len(df)} rows -> fig_cnn_dir1_{{online,retrain}}.{{png,pdf}}")
        print(cov.pivot_table(index=["timing", "arm"], columns="panel",
                              values="n_seed", aggfunc="max", fill_value=0).to_string())
    print(f"\nfigures + derived CSVs -> {a.out}")


if __name__ == "__main__":
    main()
