#!/usr/bin/env python
"""Paper fidelity figures from rundirs / analysis CSVs alone (re-runnable; NO GPU).

Three figures.  The body one first, then the two appendix ones:

  fig_fidelity_body     the SS5.2 body figure (2026-07-28; whiskers 07-29), \textwidth x 2.45in
      (a) CNN track   : ONE panel, 4 threats x {Dir(1), IID} x 2 methods as
                        THIN horizontal bars = Spearman rho + a +-1 std seed
                        whisker (ddof=1, matching the tables); Pearson tick =
                        the cell's Pearson r, in the method's DARKER ramp step
                        (07-29: the one negative r must be attributable without
                        a bar next to it).  No shading, no direct labels. (8 cells)
      (b) LLM track   : 2 settings x 6 cells (GSM8K threats + Alpaca-GPT4
                        scale), paired bars anchored at the ZOOMED axis start
                        (explicit author call, the axis-start tick label
                        discloses the baseline; start is whisker-aware);
                        colored tick = Pearson r; +-1 std whiskers.
                        Cross-device & 5-domain (all exactly 1.0) -> appendix.
      The retraining leg is a TABLE in the body (CIFAR-10 Dir(1), from
      track_c/c1); its figure form stays below as fig_fidelity_retrain.

  fig_fidelity_inrun    vs **in-run Shapley** (same-game exact decomposition)
      (a) value level : one judging cell, per-client phi, standardized
      (b) CNN sweep   : 8 dataset x partition settings x 4 threat axes  (32 cells)
      (c) LLM sweep   : 4 stages (GSM8K N=50 / 5-domain N=5 / alpaca N=20 scale
                        leg / cross-device N=100 anchor)               (11 cells)

  fig_fidelity_retrain  vs **retraining-based Shapley** (method-neutral ground truth)
      (a) diagonal    : x = agreement BETWEEN the two Shapley values,
                        y = Flirds vs retraining Shapley.  60 cells
                        (48 CNN + 9 LLM 5-domain + 3 LLM anchor).
      (b) method gap  : per-method (method - in-run) paired difference, 48 CNN cells

Every plotted point is also written to a *_points.csv next to the figure so the
figure and the paper tables can never disagree.

Design follows the `dataviz` skill: reference categorical slots 1-2 only
(blue #2a78d6 / orange #eb6834 -- the certified all-pairs-safe head of the
documented palette, used unchanged, no re-stepping), hairline solid grid, no
dual axes, surface ring on markers, legend always present for >=2 series,
selective direct labels.  Identity is carried redundantly by shape/fill so the
figures survive grayscale print and full CVD.

Aggregation: seed mean; the body figure adds +-1 std whiskers (ddof=1, the
tables' convention).  In the appendix figures a cell with n < 3 seeds is drawn
HOLLOW so an incomplete cell looks incomplete (n is in every points CSV).

Usage:
    python runs/plots/make_fig_fidelity.py                 # both -> runs/plots/figs/
    python runs/plots/make_fig_fidelity.py --which inrun --dpi 300
    python runs/plots/make_fig_fidelity.py --dose          # + off-axis lf dose ladder
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent          # runs/plots
ROOT = HERE.parent.parent                       # repo root
FIG = HERE / "figs"
PAPER_FIG = ROOT / "paper" / "AAAI" / "Figures"  # body figure lands here too

# --------------------------------------------------------------------------- #
# palette (dataviz reference instance, slots 1-2 + chrome ink)                 #
# --------------------------------------------------------------------------- #
C_FLIRDS = "#2a78d6"      # categorical slot 1
C_FIRST = "#eb6834"       # categorical slot 2
C_LLM = "#eb6834"         # slot 2 doubles as the LLM track in fig B
C_CNN = "#2a78d6"
# Pearson-tick colors: the darker step of the SAME hue (blue ramp step 650;
# the orange mirrors that 450->650 darkening) so an off-bar tick still names
# its method while staying redundant with row position -- not new series hues
T_FLIRDS = "#104281"
T_FIRST = "#a34413"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"       # paper surface (not the web #fcfcfb)


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Times-family serif to match the AAAI body text; STIX math to match;
        # fonttype 42 keeps the PDFs free of Type-3 fonts (camera-ready rule)
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman",
                       "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7.4,
        "axes.titlesize": 7.8,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 6.9,
        "legend.fontsize": 6.9,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })


def despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def save(fig, name: str, dpi: int) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=dpi, facecolor=SURFACE)
    print(f"  wrote {FIG / (name + '.pdf')}  (+ .png)")


# --------------------------------------------------------------------------- #
# labels                                                                       #
# --------------------------------------------------------------------------- #
AXIS_THREATS = ["clean", "free_rider", "grad_noise", "label_flip"]
THREAT_LABEL = {
    "clean": "clean",
    "free_rider": "zero-update",   # 07-28 terminology: match the tables ("zero-update")
    "grad_noise": "gradient\nnoise",
    "label_flip": "label-flip",
    "frrand": "random-\nupdate FR",
}
CNN_SETTINGS = [
    ("cifar10", "dir1"), ("cifar10", "iid"), ("cifar10", "qskew"), ("cifar10", "shard"),
    ("fmnist", "dir1"), ("fmnist", "iid"), ("mnist", "dir1"), ("mnist", "iid"),
]
MAIN_SETTING = ("cifar10", "dir1")
METHOD_LABEL = {
    "Flirds": "Flirds", "Flirds1st": "Flirds-1st", "loss-heur": "singleton utility",
    "GTG": "GTG-Shapley", "FedSV": "FedSV", "ComFedSV": "ComFedSV",
    "ShapleyFL": "ShapleyFL", "FedIF": "FedIF",
}
OTHER_METHODS = ["Flirds1st", "loss-heur", "GTG", "FedSV", "ComFedSV", "ShapleyFL", "FedIF"]


# --------------------------------------------------------------------------- #
# loaders -- vs in-run Shapley                                                 #
# --------------------------------------------------------------------------- #
def load_cnn_inrun(dose: bool = False) -> pd.DataFrame:
    """8 settings x 4 threat axes x 3 seeds, Spearman vs in-run Shapley."""
    df = pd.read_csv(ROOT / "runs/track_c/c2fid/analysis/fidelity.csv")
    df = df[df["method"].isin(["Flirds", "Flirds1st"]) & df["spearman_b"].notna()].copy()
    keep = AXIS_THREATS + (["frrand"] if dose else [])
    df = df[df["threat"].isin(keep)]
    # the confirmed axis uses the canonical label-flip dose only
    if not dose:
        df = df[(df["threat"] != "label_flip") | (df["scenario"].str.contains("0.70"))]
    df["setting"] = df["dataset"] + "/" + df["partition"]
    return df[["setting", "dataset", "partition", "threat", "scenario", "seed",
               "method", "spearman_b", "pearson_b"]]


def _llm_metrics(rundir: Path, threat_key: str) -> dict[int, dict]:
    """{seed: {"spearman": {...}, "pearson": {...}}} from a rundir metrics.json.

    `pearson` is absent in the 2026-06 cross-device rundirs (B.7) -- callers get
    an empty dict there and simply draw no value-level mark.
    """
    out: dict[int, dict] = {}
    m = json.loads((rundir / "metrics.json").read_text())
    for key, blob in m.items():
        if not key.startswith(threat_key + "_seed"):
            continue
        seed = int(key.rsplit("seed", 1)[1])
        out[seed] = {"spearman": blob.get("spearman", {}),
                     "pearson": blob.get("pearson", {})}
    return out


def load_llm_inrun() -> pd.DataFrame:
    """LLM stages: GSM8K main / 5-domain non-IID / alpaca scale leg / cross-device."""
    rows = []

    def push(group, label, seed, method, value, order, pearson=np.nan):
        rows.append(dict(group=group, label=label, seed=seed, method=method,
                         spearman_b=value, pearson_b=pearson, order=order))

    P2 = ROOT / "runs/phase2_matrix/rundirs"

    def drain(group, label, rundir, tkey, seed_override=None):
        if not (rundir / "metrics.json").exists():
            return
        for s, blob in _llm_metrics(rundir, tkey).items():
            for meth in ("Flirds", "Flirds1st"):
                if meth in blob["spearman"]:
                    push(group, label, seed_override if seed_override is not None else s,
                         meth, blob["spearman"][meth], len(rows),
                         blob["pearson"].get(meth, np.nan))

    # 1. GSM8K main stage -- one rundir per (threat, seed)
    g = "GSM8K|$N$=50, 5/50"
    for label, stem, tkey in [("clean", "1B_gsm50k5_clean_nr0.7_s{}", "clean"),
                              ("answer-swap", "1B_gsm50k5_noisy_nr0.7_s{}", "noisy"),
                              ("zero-update", "1B_gsm50k5_frzero_nr0.7_s{}", "freerider_zero")]:
        for seed in (0, 1, 2):
            drain(g, label, P2 / stem.format(seed), tkey, seed_override=seed)

    # 2. 5-domain non-IID stage -- one rundir holding all seeds
    g = "5-domain|$N$=5, full"
    for label, stem, tkey in [("clean", "1B_silo5_clean", "clean"),
                              ("answer-swap", "1B_silo5_noisy", "noisy"),
                              ("zero-update", "1B_silo5_frzero", "freerider_zero")]:
        drain(g, label, P2 / stem, tkey)

    # 3. Alpaca-GPT4 IID scale leg (1B / 3B / 7B) -- derived emitter CSV
    #    (07-29: label matches the body text's "Alpaca-GPT4", not "alpaca")
    g = "Alpaca-GPT4|$N$=20, 2/20, clean"
    fid = pd.read_csv(ROOT / "runs/track_d/fidelity.csv")
    for scale in ("1B", "3B", "7B"):
        sub = fid[(fid["cell"].str.startswith(f"{scale}_std20")) &
                  (fid["method"].isin(["Flirds", "Flirds1st"]))]
        for _, r in sub.iterrows():
            push(g, scale, int(r["seed"]), r["method"], float(r["spearman"]),
                 len(rows), float(r["pearson"]))

    # 4. cross-device anchor (exact per-round oracle attached to the anchor cells)
    g = "cross-dev.|$N$=100"
    for label, stem, tkey in [("answer-swap", "1B_device100-a0.5_noisy_anchor", "noisy"),
                              ("zero-update", "1B_device100-a0.5_frzero_anchor", "freerider_zero")]:
        drain(g, label, P2 / stem, tkey)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# loaders -- vs retraining-based Shapley                                       #
# --------------------------------------------------------------------------- #
def load_cnn_retrain() -> pd.DataFrame:
    """48 cells: dual-oracle agreement + every method vs retraining Shapley."""
    df = pd.read_csv(ROOT / "runs/track_c/c1/analysis/methods_long.csv")
    df = df[df["spearman_a"].notna()]
    base = (df[df["method"] == "(b)oracle"]
            .set_index("cell")["spearman_a"].astype(float).to_dict())
    out = df[df["method"] != "(a)oracle"].copy()
    out["dual"] = out["cell"].map(base)
    out["gap"] = out["spearman_a"].astype(float) - out["dual"]
    out["track"] = "CNN"
    return out[["cell", "dataset", "partition", "threat", "seed", "method",
                "spearman_a", "spearman_b", "dual", "gap", "track"]]


def load_llm_retrain() -> pd.DataFrame:
    """LLM cells that have a retraining oracle: 5-domain (9) + anchor (3)."""
    rows = []

    # 5-domain non-IID: merge emitter already carries both legs
    sil = pd.read_csv(ROOT / "runs/phase2_matrix/silo5_a_fidelity_1B.csv")
    base = (sil[sil["method"] == "(b)oracle"]
            .set_index(["threat", "seed"])["spearman_a"].astype(float).to_dict())
    for _, r in sil[sil["method"].isin(["Flirds", "Flirds1st"])].iterrows():
        key = (r["threat"], r["seed"])
        if key not in base or pd.isna(r["spearman_a"]):
            continue
        rows.append(dict(cell=f"silo5_{r['threat']}_s{r['seed']}", dataset="LLM 5-domain",
                         partition="silo5", threat={"noisy": "answer_swap",
                                                    "frzero": "free_rider"}.get(r["threat"], r["threat"]),
                         seed=int(r["seed"]), method=r["method"],
                         spearman_a=float(r["spearman_a"]), spearman_b=np.nan,
                         dual=base[key], gap=float(r["spearman_a"]) - base[key], track="LLM"))

    # anchor stage: compute both legs straight from per-client phi
    for seed in (0, 1, 2):
        p = ROOT / f"runs/track_d/rundirs/1B_anchor5_seed{seed}/phi.parquet"
        if not p.exists():
            continue
        w = pd.read_parquet(p).pivot(index="client", columns="method", values="phi")
        if "(a)oracle" not in w or "(b)oracle" not in w:
            continue
        dual = spearmanr(w["(b)oracle"], w["(a)oracle"]).statistic
        for meth in ("Flirds", "Flirds1st"):
            if meth not in w:
                continue
            val = spearmanr(w[meth], w["(a)oracle"]).statistic
            rows.append(dict(cell=f"anchor5_s{seed}", dataset="LLM alpaca", partition="anchor5",
                             threat="clean", seed=seed, method=meth, spearman_a=val,
                             spearman_b=np.nan, dual=dual, gap=val - dual, track="LLM"))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# figure A -- vs in-run Shapley                                                #
# --------------------------------------------------------------------------- #
def panel_value_level(ax) -> pd.DataFrame:
    cell = ROOT / "runs/track_c/c2fid/rundirs/cifar10_dir1_grad-noise_fid_seed0/phi.parquet"
    d = pd.read_parquet(cell)
    z = lambda v: (v - v.mean()) / v.std(ddof=0)
    x = z(d["phi_(b)oracle"])
    lo, hi = -3.45, 3.45
    ax.plot([lo, hi], [lo, hi], color=AXIS, lw=0.8, zorder=1)
    ax.text(hi - 0.12, 1.15, "$y = x$", color=MUTED, fontsize=6.6, ha="right", va="center")

    out, rs = [], {}
    for meth, color in (("Flirds1st", C_FIRST), ("Flirds", C_FLIRDS)):
        y = z(d[f"phi_{meth}"])
        rs[meth] = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, s=12, facecolor=color, edgecolor=SURFACE, linewidth=0.45,
                   alpha=0.85, zorder=3 if meth == "Flirds" else 2)
        out.append(pd.DataFrame(dict(client=d["client"], corrupt=d["corrupt"],
                                     method=meth, z_inrun=x, z_est=y)))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("in-run Shapley  (standardized)")
    ax.set_ylabel("estimate  (standardized)")
    ax.set_title("(a)  per-client values, one cell", loc="left", pad=6)
    ax.grid(True, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)

    # direct labels: a colored mark carries identity, the text stays in ink
    for meth, color, ypos in (("Flirds", C_FLIRDS, 0.955), ("Flirds1st", C_FIRST, 0.875)):
        ax.plot([0.055], [ypos], marker="o", ms=3.6, color=color, mec=SURFACE, mew=0.45,
                transform=ax.transAxes, clip_on=False, zorder=5)
        ax.text(0.105, ypos, f"{METHOD_LABEL[meth]}   $r$ = {rs[meth]:+.2f}",
                transform=ax.transAxes, va="center", fontsize=6.8, color=INK2, zorder=5)
    return pd.concat(out, ignore_index=True)


def _dumbbell(ax, xs, hi_vals, lo_vals, emph, hollow=None):
    """Paired dot plot: one connector + two dots per slot.

    emph   -> the main-text cell, ringed in ink so the reader can find it
    hollow -> fewer than 3 seeds, drawn open so an incomplete cell looks incomplete
    """
    hollow = list(hollow) if hollow is not None else [False] * len(xs)
    emph = list(emph)
    for x, a, b in zip(xs, hi_vals, lo_vals):
        if np.isnan(a) or np.isnan(b):
            continue
        ax.plot([x, x], [b, a], color=AXIS, lw=0.7, zorder=2, solid_capstyle="round")
    for vals, color in ((hi_vals, C_FLIRDS), (lo_vals, C_FIRST)):
        for x, v, e, h in zip(xs, vals, emph, hollow):
            if np.isnan(v):
                continue
            ax.scatter([x], [v], s=21 if e else 13,
                       facecolor=SURFACE if h else color,
                       edgecolor=color if h else (INK if e else SURFACE),
                       linewidth=0.9 if h else (0.7 if e else 0.45), zorder=4)


def panel_cnn_sweep(ax, dose: bool) -> pd.DataFrame:
    df = load_cnn_inrun(dose=dose)
    cellmean = (df.groupby(["setting", "threat", "scenario", "method"])
                  .agg(mean=("spearman_b", "mean"), n=("spearman_b", "size"))
                  .reset_index())
    threats = AXIS_THREATS + (["frrand"] if dose else [])
    scen_order = {}
    for t in threats:
        s = sorted(df[df["threat"] == t]["scenario"].unique())
        scen_order[t] = s

    rows, xs_ticks, labels = [], [], []
    band_w = 0.74
    for bi, t in enumerate(threats):
        scens = scen_order[t]
        slots = [(sc, st) for sc in scens for st in CNN_SETTINGS]
        n = len(slots)
        for si, (sc, st) in enumerate(slots):
            key = f"{st[0]}/{st[1]}"
            x = bi + (si - (n - 1) / 2) * (band_w / max(n - 1, 1))
            get = lambda m: cellmean[(cellmean["setting"] == key) &
                                     (cellmean["threat"] == t) &
                                     (cellmean["scenario"] == sc) &
                                     (cellmean["method"] == m)]
            a, b = get("Flirds"), get("Flirds1st")
            rows.append(dict(threat=t, scenario=sc, setting=key, x=x,
                             flirds=float(a["mean"].iloc[0]) if len(a) else np.nan,
                             flirds1st=float(b["mean"].iloc[0]) if len(b) else np.nan,
                             n_seeds=int(a["n"].iloc[0]) if len(a) else 0,
                             main=(st == MAIN_SETTING)))
        xs_ticks.append(bi)
        labels.append(THREAT_LABEL[t])
    pts = pd.DataFrame(rows)

    _dumbbell(ax, pts["x"], pts["flirds"], pts["flirds1st"], pts["main"],
              hollow=pts["n_seeds"] < 3)

    for bi in range(len(threats) - 1):
        ax.axvline(bi + 0.5, color=GRID, lw=0.6, zorder=0)
    ax.set_xticks(xs_ticks)
    ax.set_xticklabels(labels, fontsize=6.5, linespacing=1.15)
    ax.set_xlim(-0.5, len(threats) - 0.5)
    ax.set_ylabel(r"Spearman $\rho$  vs in-run Shapley")
    ax.set_title(f"(b)  CNN sweep: {len(CNN_SETTINGS)} settings $\\times$ {len(threats)} threats",
                 loc="left", pad=6)
    ax.grid(True, axis="y", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    return pts


def panel_llm_sweep(ax) -> pd.DataFrame:
    df = load_llm_inrun()
    agg = (df.groupby(["group", "label", "method"])
             .agg(mean=("spearman_b", "mean"), lo=("spearman_b", "min"),
                  hi=("spearman_b", "max"), n=("spearman_b", "size"))
             .reset_index())
    groups = list(dict.fromkeys(df["group"]))
    entries = []
    for g in groups:
        for lab in list(dict.fromkeys(df[df["group"] == g]["label"])):
            entries.append((g, lab))

    xs, rows = [], []
    x = 0.0
    bounds = []
    for gi, g in enumerate(groups):
        labs = [lab for gg, lab in entries if gg == g]
        start = x
        for lab in labs:
            get = lambda m: agg[(agg["group"] == g) & (agg["label"] == lab) & (agg["method"] == m)]
            a, b = get("Flirds"), get("Flirds1st")
            rows.append(dict(group=g, label=lab, x=x,
                             flirds=float(a["mean"].iloc[0]) if len(a) else np.nan,
                             flirds1st=float(b["mean"].iloc[0]) if len(b) else np.nan,
                             n_seeds=int(a["n"].iloc[0]) if len(a) else 0))
            xs.append(x)
            x += 1.0
        bounds.append((start, x - 1.0, g))
        x += 0.7
    pts = pd.DataFrame(rows)

    _dumbbell(ax, pts["x"], pts["flirds"], pts["flirds1st"],
              emph=[False] * len(pts), hollow=pts["n_seeds"] < 3)

    ax.set_xticks(pts["x"])
    ax.set_xticklabels([f"{r.label}{'*' if r.n_seeds < 3 else ''}" for r in pts.itertuples()],
                       rotation=45, ha="right", rotation_mode="anchor", fontsize=6.3)
    ax.set_xlim(-0.8, x - 0.9)
    for _, e, _g in bounds[:-1]:
        ax.axvline(e + 0.85, color=GRID, lw=0.6, zorder=0)
    ax.set_title("(c)  LLM sweep: 4 stages", loc="left", pad=6)
    ax.grid(True, axis="y", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)

    lo = np.nanmin([pts["flirds"].min(), pts["flirds1st"].min()])
    ax.annotate(f"every stage $\\geq$ {lo:.3f}", xy=(0.5, 0.885), xycoords="axes fraction",
                ha="center", fontsize=6.5, color=INK2)
    trans = ax.get_xaxis_transform()
    for gi, (s, e, g) in enumerate(bounds):
        # alternate depths so adjacent stage labels cannot collide
        ax.annotate(g.replace("|", "\n"), xy=((s + e) / 2, -0.36 - 0.12 * (gi % 2)),
                    xycoords=trans,
                    ha="center", va="top", fontsize=5.9, color=MUTED, linespacing=1.3)
    return pts


def fig_inrun(dpi: int, dose: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.72),
                             gridspec_kw=dict(width_ratios=[1.02, 1.50, 1.28], wspace=0.26))
    p_a = panel_value_level(axes[0])
    p_b = panel_cnn_sweep(axes[1], dose=dose)
    p_c = panel_llm_sweep(axes[2])

    ylo = min(np.nanmin(p_b[["flirds", "flirds1st"]].to_numpy()),
              np.nanmin(p_c[["flirds", "flirds1st"]].to_numpy()))
    for ax in axes[1:]:
        ax.set_ylim(max(-0.05, ylo - 0.09), 1.035)
    axes[2].tick_params(labelleft=False)

    # the shared key lives in panel (c)'s empty lower half -- no extra figure space
    handles = [
        Line2D([], [], marker="o", ls="", markersize=4.2, color=C_FLIRDS,
               markeredgecolor=SURFACE, markeredgewidth=0.45, label="Flirds"),
        Line2D([], [], marker="o", ls="", markersize=4.2, color=C_FIRST,
               markeredgecolor=SURFACE, markeredgewidth=0.45, label="Flirds-1st"),
        Line2D([], [], marker="o", ls="", markersize=4.4, color=MUTED,
               markeredgecolor=INK, markeredgewidth=0.7, label="CIFAR-10 / Dir(1) = main text"),
    ]
    if bool((p_b["n_seeds"] < 3).any() or (p_c["n_seeds"] < 3).any()):
        handles.append(
            Line2D([], [], marker="o", ls="", markersize=4.2, markerfacecolor=SURFACE,
                   markeredgecolor=MUTED, markeredgewidth=0.9, label="* fewer than 3 seeds"))
    axes[2].legend(handles=handles, loc="center left", handletextpad=0.35,
                   borderpad=0.2, labelspacing=0.42, labelcolor=INK2,
                   bbox_to_anchor=(0.02, 0.36))
    save(fig, "fig_fidelity_inrun", dpi)
    plt.close(fig)

    FIG.mkdir(parents=True, exist_ok=True)
    p_a.to_csv(FIG / "fig_fidelity_inrun_points_a.csv", index=False)
    p_b.to_csv(FIG / "fig_fidelity_inrun_points_b.csv", index=False)
    p_c.to_csv(FIG / "fig_fidelity_inrun_points_c.csv", index=False)


# --------------------------------------------------------------------------- #
# figure B -- vs retraining-based Shapley                                      #
# --------------------------------------------------------------------------- #
THREAT_MARKER = {"clean": "o", "free_rider": "^", "grad_noise": "s",
                 "label_flip": "D", "answer_swap": "v"}


def panel_diagonal(ax, cnn: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    pts = pd.concat([cnn[cnn["method"] == "Flirds"], llm[llm["method"] == "Flirds"]],
                    ignore_index=True)
    lo, hi = -0.75, 1.06
    ax.plot([lo, hi], [lo, hi], color=AXIS, lw=0.8, zorder=1)
    ax.text(0.10, 0.40, "$y = x$", color=MUTED, fontsize=6.6, ha="right", va="center")
    ax.axhline(0, color=GRID, lw=0.6, zorder=0)
    ax.axvline(0, color=GRID, lw=0.6, zorder=0)

    for _, r in pts.iterrows():
        color = C_CNN if r["track"] == "CNN" else C_LLM
        clean = r["threat"] == "clean"
        ax.scatter([r["dual"]], [r["spearman_a"]], s=17,
                   marker=THREAT_MARKER.get(r["threat"], "o"),
                   facecolor=SURFACE if clean else color, edgecolor=color,
                   linewidth=0.9 if clean else 0.5, alpha=0.95, zorder=3)

    dc = pts.loc[pts["track"] == "CNN", "gap"].astype(float)
    dl = pts.loc[pts["track"] == "LLM", "gap"].astype(float)
    ax.annotate("paired $\\Delta$ = Flirds $-$ in-run Shapley\n"
                f"CNN  {dc.mean():+.3f} $\\pm$ {dc.std(ddof=1):.3f}  "
                f"({int((dc.abs() <= 0.05).sum())}/{len(dc)} $|\\Delta| \\leq$ .05)\n"
                f"LLM  {dl.mean():+.3f} $\\pm$ {dl.std(ddof=1):.3f}  "
                f"({int((dl.abs() <= 0.05).sum())}/{len(dl)})",
                xy=(0.035, 0.968), xycoords="axes fraction", va="top", ha="left",
                fontsize=6.1, color=INK2, linespacing=1.45)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"in-run vs retraining Shapley  (Spearman $\rho$)")
    ax.set_ylabel(r"Flirds vs retraining Shapley")
    ax.set_title("(a)  per-cell agreement with retraining Shapley", loc="left", pad=6)
    ax.grid(True, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)

    handles = [
        Line2D([], [], marker="o", ls="", markersize=4.0, color=C_CNN,
               markeredgecolor=SURFACE, markeredgewidth=0.4, label="CNN  $N$=10 (48)"),
        Line2D([], [], marker="o", ls="", markersize=4.0, color=C_LLM,
               markeredgecolor=SURFACE, markeredgewidth=0.4, label="LLM  $N$=5 (12)"),
        Line2D([], [], marker="o", ls="", markersize=4.0, markerfacecolor=SURFACE,
               markeredgecolor=MUTED, markeredgewidth=0.9, label="clean (hollow)"),
        Line2D([], [], marker="^", ls="", markersize=4.0, color=MUTED, label="zero-update"),
        Line2D([], [], marker="s", ls="", markersize=4.0, color=MUTED, label="gradient noise"),
        Line2D([], [], marker="D", ls="", markersize=3.6, color=MUTED, label="label-flip"),
        Line2D([], [], marker="v", ls="", markersize=4.0, color=MUTED, label="answer-swap"),
    ]
    ax.legend(handles=handles, loc="lower right", ncol=2, columnspacing=0.9,
              handletextpad=0.3, borderpad=0.2, labelspacing=0.28, labelcolor=INK2)
    return pts


def panel_method_gap(ax, cnn: pd.DataFrame) -> pd.DataFrame:
    order = ["Flirds"] + OTHER_METHODS
    rows = []
    rng = np.random.default_rng(0)
    for i, meth in enumerate(order):
        sub = cnn[cnn["method"] == meth]
        if not len(sub):
            continue
        y = len(order) - 1 - i
        g = sub["gap"].astype(float).to_numpy()
        emph = meth == "Flirds"
        color = C_FLIRDS if emph else MUTED
        ax.scatter(g, y + rng.uniform(-0.17, 0.17, len(g)), s=8,
                   facecolor=color, edgecolor=SURFACE, linewidth=0.35,
                   alpha=0.85 if emph else 0.5, zorder=3 if emph else 2)
        med = float(np.median(g))
        ax.plot([med, med], [y - 0.34, y + 0.34], color=INK if emph else INK2,
                lw=1.4 if emph else 0.9, zorder=4, solid_capstyle="round")
        rows.append(dict(method=meth, median=med, mean=float(g.mean()),
                         std=float(g.std(ddof=1)), n=len(g)))
    ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([METHOD_LABEL[m] for m in reversed(order)])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(r"(method $-$ in-run Shapley) agreement with retraining Shapley")
    ax.set_title("(b)  the same paired difference, per method  (48 CNN cells)",
                 loc="left", pad=6)
    ax.grid(True, axis="x", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.annotate("$\\longleftarrow$ tracks retraining Shapley worse than the in-run target does",
                xy=(0.0, -0.30), xycoords="axes fraction", fontsize=6.1, color=MUTED, ha="left")
    return pd.DataFrame(rows)


def fig_retrain(dpi: int) -> None:
    cnn, llm = load_cnn_retrain(), load_llm_retrain()
    fig, axes = plt.subplots(2, 1, figsize=(3.42, 5.45),
                             gridspec_kw=dict(height_ratios=[1.34, 1.0], hspace=0.42))
    pts = panel_diagonal(axes[0], cnn, llm)
    summ = panel_method_gap(axes[1], cnn)
    save(fig, "fig_fidelity_retrain", dpi)
    plt.close(fig)

    FIG.mkdir(parents=True, exist_ok=True)
    pts.to_csv(FIG / "fig_fidelity_retrain_points.csv", index=False)
    summ.to_csv(FIG / "fig_fidelity_retrain_methodgap.csv", index=False)
    print(summ.to_string(index=False))


# --------------------------------------------------------------------------- #
# figure C -- the SS5.2 body figure: one \textwidth row                        #
#                                                                              #
#   (a) CNN in ONE panel: 4 threats (major rows) x {Dir(1), IID} (minor rows)  #
#       x {Flirds, Flirds-1st} as THIN horizontal bars from 0 + a +-1 std      #
#       seed whisker (07-29); the method-colored tick across each bar = the    #
#       same cell's Pearson r (it goes negative once: gradient-noise Dir(1)    #
#       Flirds-1st, left of the baseline; no connector -- row position         #
#       assigns it, the darker method color re-states it).  No shading, no     #
#       direct value labels.                                                   #
#   (b) LLM 2 settings x 6 cells as paired bars anchored at the ZOOMED axis    #
#       start (explicit Yonghee 07-28 call; the axis-start tick label          #
#       discloses the baseline; start is whisker-aware); colored tick =        #
#       Pearson r; +-1 std whiskers.  Cross-device & 5-domain stay in the      #
#       appendix.                                                              #
# Zoom is legitimate for position marks (dots); bars in (a) keep their zero    #
# baseline.  Every plotted value lands in the points CSVs.                     #
# --------------------------------------------------------------------------- #
CNN_XLIM = (-0.15, 1.045)     # negative margin: the -.05 Pearson tick must sit
                              # clearly INSIDE the field, not against the labels
CNN_XTICKS = (0.0, 0.25, 0.50, 0.75, 1.0)


def _pearson_tick(ax, x, y, half, lw=1.0, color=INK):
    """Pearson mark: a short vertical tick at x, crossing the bar/dot.

    Method-colored (darker ramp step) with a surface halo, so the tick reads
    on top of its own bar and an off-bar tick (the one negative r) still
    names its method (07-29).
    """
    if x is None or np.isnan(x):
        return
    ax.plot([x, x], [y - half, y + half], color=SURFACE, lw=lw + 1.1, zorder=5,
            solid_capstyle="butt")
    ax.plot([x, x], [y - half, y + half], color=color, lw=lw, zorder=5,
            solid_capstyle="butt")


def _std_whisker(ax, x, y, sd, lw=0.8):
    """Seed whisker: mean +- 1 std (ddof=1), capless so it cannot be mistaken
    for a Pearson tick.  Clipped to the rho bound: past +-1 there is no
    attainable value, so drawing there would only invite a "rho > 1?"
    misreading (07-29).  n=1 cells (sd = NaN) simply draw nothing."""
    if sd is None or np.isnan(sd) or sd == 0:
        return
    ax.plot([max(-1.0, x - sd), min(1.0, x + sd)], [y, y], color=INK2, lw=lw,
            zorder=4, solid_capstyle="butt")


def _rho_dot(ax, x, y, sd, color, min_arrow):
    """Dot-variant rho mark: a point on the seed mean plus a double-headed
    arrow spanning +-1 std (clipped like the whisker).  A +-std too narrow to
    carry two arrowheads (span < min_arrow, data units) falls back to a bare
    segment -- the dot alone then carries the cell."""
    if not (sd is None or np.isnan(sd) or sd == 0):
        lo, hi = max(-1.0, x - sd), min(1.0, x + sd)
        if hi - lo >= min_arrow:
            ax.annotate("", xy=(lo, y), xytext=(hi, y), zorder=4,
                        arrowprops=dict(arrowstyle="<|-|>", color=INK2, lw=0.8,
                                        shrinkA=0, shrinkB=0, mutation_scale=5.5))
        else:
            ax.plot([lo, hi], [y, y], color=INK2, lw=0.8, zorder=4,
                    solid_capstyle="butt")
    ax.scatter([x], [y], s=18, facecolor=color, edgecolor=SURFACE,
               linewidth=0.5, zorder=6)


def _r_band(ax, r, rsd, y, half, color):
    """Dot-variant Pearson spread: +-1 std (clipped) as a translucent band
    behind the tick -- same uncertainty, quieter than a second whisker, and
    it cannot be confused with the rho arrow."""
    if r is None or np.isnan(r) or np.isnan(rsd) or rsd == 0:
        return
    ax.fill_betweenx([y - half, y + half], max(-1.0, r - rsd),
                     min(1.0, r + rsd), color=color, alpha=0.20, lw=0,
                     zorder=2.5)


def panel_body_cnn(ax, tag: str, variant: str = "bar",
                   show_std: bool = True) -> pd.DataFrame:
    """The CNN track in one panel: threat (major) x partition (minor) rows."""
    df = load_cnn_inrun()
    df = df[(df["dataset"] == "cifar10") & df["partition"].isin(("dir1", "iid"))]
    agg = (df.groupby(["partition", "threat", "method"])
             .agg(rho=("spearman_b", "mean"), rho_sd=("spearman_b", "std"),
                  r=("pearson_b", "mean"), r_sd=("pearson_b", "std"),
                  n=("spearman_b", "size"))
             .reset_index())
    get = lambda p, t, m: agg[(agg["partition"] == p) & (agg["threat"] == t)
                              & (agg["method"] == m)].iloc[0]

    rows, yt, ytl = [], [], []
    y, h, sub = 0.0, 0.17, 0.54          # bar height / partition-row pitch
    for t in AXIS_THREATS:
        g0 = y - 0.33                    # group extent, for the threat label
        for part, plab in (("dir1", "Dir(1)"), ("iid", "IID")):
            for mi, (meth, color, tick) in enumerate((("Flirds", C_FLIRDS, T_FLIRDS),
                                                      ("Flirds1st", C_FIRST, T_FIRST))):
                rec = get(part, t, meth)
                yy = y + (mi - 0.5) * (h + 0.05)
                if variant == "bar":
                    ax.barh(yy, rec["rho"], height=h, color=color, lw=0, zorder=3)
                    if show_std:
                        _std_whisker(ax, rec["rho"], yy, rec["rho_sd"])
                else:
                    _rho_dot(ax, rec["rho"], yy,
                             rec["rho_sd"] if show_std else np.nan, color,
                             min_arrow=0.09)
                    if show_std:
                        _r_band(ax, rec["r"], rec["r_sd"], yy, half=0.145,
                                color=tick)
                # no rho->r connector (07-28): row position assigns the tick,
                # and its darker method color re-states it (07-29) -- incl.
                # the one negative r
                _pearson_tick(ax, rec["r"], yy, half=0.145, color=tick)
                rows.append(dict(partition=part, threat=t, method=meth,
                                 spearman=rec["rho"], spearman_std=rec["rho_sd"],
                                 pearson=rec["r"], pearson_std=rec["r_sd"],
                                 n_seeds=int(rec["n"])))
            yt.append(y)
            ytl.append(plab)
            y += sub
        g1 = y - sub + 0.33
        ax.annotate(THREAT_LABEL[t], xy=(-0.175, (g0 + g1) / 2),
                    xycoords=("axes fraction", "data"), ha="right", va="center",
                    fontsize=7.5, color=INK, linespacing=1.15,
                    annotation_clip=False)
        y += 0.46

    ax.axvline(0, color=AXIS, lw=0.7, zorder=4)
    ax.set_yticks(yt)
    ax.set_yticklabels(ytl, fontsize=7.0)
    ax.tick_params(axis="y", length=0, pad=2.0)
    # bottom margin mirrors the top one; the loop's trailing sub + group gap
    # used to leave a dead half-row under the last group (07-29 centering fix)
    ax.set_ylim(y - 0.46 - sub + 0.44, -0.44)
    xlo = CNN_XLIM[0]
    if variant == "dot" and show_std:
        # the Pearson band can reach further left than the -.05 tick does
        lo_r = float(np.nanmin((agg["r"] - agg["r_sd"].fillna(0)).to_numpy()))
        xlo = min(xlo, lo_r - 0.03)
    ax.set_xlim(xlo, CNN_XLIM[1])
    ax.set_xticks(CNN_XTICKS)
    # decimal ticks (07-28): match the tables' decimal correlations, not %
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax.grid(True, axis="x", lw=0.6, zorder=0.5)
    ax.set_axisbelow(True)
    ax.set_title(tag, loc="center", pad=5)
    despine(ax, keep=("bottom",))
    return pd.DataFrame(rows)


def panel_body_llm(ax, variant: str = "bar",
                   show_std: bool = True) -> pd.DataFrame:
    """The LLM sweep as paired bars, zoomed so the method gap is visible.

    The bars are anchored at the zoomed axis start (0.992), an explicit
    author decision of 07-28; the 0.992 tick label discloses the baseline.
    The axis start is chosen from the data on a .004 grid so the midpoint
    tick lands exactly.  Rows are dodged like the CNN panel's pairs.
    The "dot" variant swaps bars for mean dots with +-1 std arrows -- pure
    position marks, for which the zoom needs no baseline disclosure at all.
    """
    df = load_llm_inrun()
    # cross-device (07-28) and 5-domain (07-28: every cell exactly 1.0000, no
    # difference to show) stay in the appendix, not the body panel
    df = df[~df["group"].str.startswith(("cross-dev", "5-domain"))].copy()
    agg = (df.groupby(["group", "label", "method"])
             .agg(rho=("spearman_b", "mean"), rho_sd=("spearman_b", "std"),
                  r=("pearson_b", "mean"), r_sd=("pearson_b", "std"),
                  n=("spearman_b", "size"))
             .reset_index())
    get = lambda g, lab, m: next(iter(agg[(agg["group"] == g) & (agg["label"] == lab)
                                          & (agg["method"] == m)].itertuples()), None)
    # zoom start on the .004 grid, low enough for every whisker tip, tick,
    # and (dot variant) Pearson band edge; means only in the no-std set
    sd_or_0 = (lambda c: agg[c].fillna(0)) if show_std else (lambda c: 0.0)
    lo = float(np.nanmin(np.concatenate([
        (agg["rho"] - sd_or_0("rho_sd")).to_numpy(),
        (agg["r"] - sd_or_0("r_sd")).to_numpy()])))
    x0 = np.floor((lo - 0.0002) * 250) / 250

    rows, yt, ytl = [], [], []
    y, dy = 0.0, 0.20
    for g in list(dict.fromkeys(df["group"])):
        name, _, params = g.partition("|")
        if y > 0:
            ax.axhline(y - 0.50, color=GRID, lw=0.6, zorder=0.5)
        ax.annotate(f"{name}  ·  {params}", xy=(0.02, y),
                    xycoords=("axes fraction", "data"), ha="left", va="center",
                    fontsize=7.3, color=INK, annotation_clip=False)
        y += 1.05
        for lab in list(dict.fromkeys(df[df["group"] == g]["label"])):
            a, b = get(g, lab, "Flirds"), get(g, lab, "Flirds1st")
            if a is None:
                continue
            for rec, color, tick, off in ((b, C_FIRST, T_FIRST, +dy),
                                          (a, C_FLIRDS, T_FLIRDS, -dy)):
                if rec is None:
                    continue
                yy = y + off
                if variant == "bar":
                    # paired bars anchored at the zoomed axis start (explicit
                    # Yonghee 07-28 call -- the leaders already read as bars;
                    # the axis-start tick label discloses the zoomed baseline)
                    ax.barh(yy, rec.rho - x0, left=x0, height=0.30, color=color,
                            lw=0, zorder=3)
                    if show_std:
                        _std_whisker(ax, rec.rho, yy, rec.rho_sd)
                else:
                    _rho_dot(ax, rec.rho, yy,
                             rec.rho_sd if show_std else np.nan, color,
                             min_arrow=0.0007)
                    if show_std:
                        _r_band(ax, rec.r, rec.r_sd, yy, half=0.24, color=tick)
                _pearson_tick(ax, rec.r, yy, half=0.24, lw=0.9, color=tick)
            yt.append(y)
            ytl.append(lab)
            rows.append(dict(group=g.replace("|", " "), label=lab,
                             flirds_rho=a.rho, flirds_rho_std=a.rho_sd,
                             flirds_r=a.r, flirds_r_std=a.r_sd,
                             first_rho=b.rho if b is not None else np.nan,
                             first_rho_std=b.rho_sd if b is not None else np.nan,
                             first_r=b.r if b is not None else np.nan,
                             first_r_std=b.r_sd if b is not None else np.nan,
                             n_seeds=int(a.n)))
            y += 1.0
        y += 0.42
    pts = pd.DataFrame(rows)

    # decimal ticks (07-28): match the tables' decimal correlations, not %
    fmt = lambda v: "1" if v >= 0.9999999 else f"{v:.3f}"
    # whiskers/arrows are clipped at rho = 1, so the axis ends just past it
    ax.set_xlim(x0, 1.0 + (1.0 - x0) * 0.03)
    xticks = [x0, (x0 + 1.0) / 2, 1.0]
    ax.set_xticks(xticks)
    ax.set_xticklabels([fmt(v) for v in xticks])
    ax.set_yticks(yt)
    ax.set_yticklabels(ytl, fontsize=7.4)
    ax.tick_params(axis="y", length=0, pad=2.0)
    ax.set_ylim(y - 0.72, -0.62)
    ax.grid(True, axis="x", lw=0.6, zorder=0.5)
    ax.set_axisbelow(True)
    ax.set_title("(b)  LLM", loc="center", pad=5)
    despine(ax, keep=("bottom",))
    print(f"    [panel b] zoom starts at {x0:.3f} (data min {lo:.4f})")
    return pts


# body-only font bump (Yonghee 07-28: "글자를 좀 더 키워줘") -- appendix figures
# keep the global style() sizes, so the override lives in an rc_context
BODY_FONTS = {"font.size": 8.2, "axes.titlesize": 8.8, "xtick.labelsize": 7.7,
              "ytick.labelsize": 7.7, "legend.fontsize": 7.6}


def fig_body(dpi: int) -> None:
    with plt.rc_context(BODY_FONTS):
        _fig_body_inner(dpi, variant="bar")
        _fig_body_inner(dpi, variant="dot")
        _fig_body_inner(dpi, variant="bar", show_std=False)
        _fig_body_inner(dpi, variant="dot", show_std=False)


def _fig_body_inner(dpi: int, variant: str, show_std: bool = True) -> None:
    """SS5.2 body figure: CNN one panel + LLM zoomed panel.

    variant "bar" + show_std is the paper figure (thin bars + std whiskers).
    variant "dot" is the 07-29 alternative on the same layout: a dot on the
    seed mean with a double-headed +-1 std arrow, and the Pearson tick
    backed by a +-1 std band.  show_std=False re-renders either variant
    with means only (the "_nostd" set, Yonghee 07-29).  Everything except
    the (bar, std) paper version lands in figs/ only, for comparison.
    """
    name = ("fig_fidelity_body" if variant == "bar" else "fig_fidelity_body_dot") \
        + ("" if show_std else "_nostd")
    fig = plt.figure(figsize=(7.05, 2.60))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 0.34, 1.06], wspace=0.10)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[2])          # gs[1] is a spacer for (b)'s labels

    p_cnn = panel_body_cnn(ax_a, "(a)  CNN · CIFAR-10", variant, show_std)
    p_llm = panel_body_llm(ax_b, variant, show_std)

    # legend keys mirror the actual mark: the rho mark with the method's
    # Pearson tick through it (HandlerTuple ndivide=None overlays the two)
    mk = dict(marker="s", ms=5.0) if variant == "bar" else dict(marker="o", ms=4.6)
    handles = [
        (Line2D([], [], ls="", color=C_FLIRDS, **mk),
         Line2D([], [], marker="|", ls="", ms=7.5, mew=1.1, color=T_FLIRDS)),
        (Line2D([], [], ls="", color=C_FIRST, **mk),
         Line2D([], [], marker="|", ls="", ms=7.5, mew=1.1, color=T_FIRST)),
    ]
    fig.legend(handles=handles, labels=["Flirds", "Flirds-1st"],
               handler_map={tuple: HandlerTuple(ndivide=None)},
               loc="upper center", ncol=2, fontsize=7.5,
               columnspacing=1.5, handletextpad=0.4, borderpad=0.2,
               labelcolor=INK2, bbox_to_anchor=(0.5, 1.05))

    # encoding key: the marks themselves, not words (Yonghee 07-29) -- a
    # neutral-ink glyph names each encoding, the top legend maps the colors
    if variant == "bar":
        h_rho = Line2D([], [], ls="-", lw=4.6, color=MUTED, solid_capstyle="butt")
        h_r = Line2D([], [], marker="|", ls="", ms=7.5, mew=1.2, color=INK)
    else:
        h_rho = Line2D([], [], marker="o", ls="", ms=4.6, color=MUTED,
                       markeredgecolor=SURFACE, markeredgewidth=0.5)
        h_r = (Line2D([], [], ls="-", lw=5.4, color=INK, alpha=0.18,
                      solid_capstyle="butt"),
               Line2D([], [], marker="|", ls="", ms=7.5, mew=1.2, color=INK))
    if variant == "dot" and not show_std:
        h_r = h_r[1]                       # no band in the no-std set
    enc_handles = [h_rho, Line2D([], [], ls="-", lw=0.9, color=INK2), h_r]
    enc_labels = ["Spearman $\\rho$", "$\\pm$1 std",
                  "Pearson $r$   ·   vs in-run Shapley"]
    if not show_std:
        del enc_handles[1], enc_labels[1]
    fig.legend(handles=enc_handles, labels=enc_labels,
               handler_map={tuple: HandlerTuple(ndivide=None)},
               loc="upper center", ncol=len(enc_handles), fontsize=7.6,
               columnspacing=1.3, handletextpad=0.5, handlelength=1.6,
               borderpad=0.1, labelcolor=INK2, bbox_to_anchor=(0.5, -0.015))
    save(fig, name, dpi)
    plt.close(fig)

    # the paper folder tracks every body-figure variant (Yonghee 07-29:
    # "원본 다 Figures에").  main.tex includes only fig_fidelity_body; the
    # other three sit beside it for the pick.  PDFs matter: pdflatex prefers
    # .pdf, so a stale one would silently shadow the .png.
    if PAPER_FIG.is_dir():
        for ext in ("pdf", "png"):
            shutil.copyfile(FIG / f"{name}.{ext}", PAPER_FIG / f"{name}.{ext}")
        print(f"    copied {name}.pdf/.png -> {PAPER_FIG}")

    if variant != "bar" or not show_std:
        return

    # prose-check numbers: what SS5.2 cites must be reproducible from here
    fl = p_cnn[p_cnn["method"] == "Flirds"]
    print(f"    CNN Flirds:  min rho {fl['spearman'].min():.3f}   "
          f"min r {fl['pearson'].min():.3f}   (8 cells)")
    print(f"    CNN whiskers (ddof=1):  max rho std {p_cnn['spearman_std'].max():.3f}   "
          f"max r std {p_cnn['pearson_std'].max():.3f}")
    print(f"    LLM whiskers (ddof=1):  max rho std "
          f"{p_llm[['flirds_rho_std', 'first_rho_std']].to_numpy().max():.4f}")
    print("    LLM leg (ranges over cell means; cross-device & 5-domain excluded):")
    for gl, r in (p_llm.groupby("group", sort=False)[["flirds_rho", "first_rho",
                                                      "flirds_r", "first_r"]]
                       .agg(["min", "max"]).iterrows()):
        print(f"      {gl:30s} rho F {r[('flirds_rho', 'min')]:.4f}-"
              f"{r[('flirds_rho', 'max')]:.4f} / 1st {r[('first_rho', 'min')]:.4f}-"
              f"{r[('first_rho', 'max')]:.4f}   r F {r[('flirds_r', 'min')]:.4f}- "
              f"{r[('flirds_r', 'max')]:.4f} / 1st {r[('first_r', 'min')]:.4f}-"
              f"{r[('first_r', 'max')]:.4f}")

    FIG.mkdir(parents=True, exist_ok=True)
    p_cnn.to_csv(FIG / "fig_fidelity_body_points_cnn.csv", index=False)
    p_llm.to_csv(FIG / "fig_fidelity_body_points_llm.csv", index=False)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["body", "inrun", "retrain", "all"], default="all")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--dose", action="store_true",
                    help="also plot the off-axis label-flip dose ladder + random-update FR")
    a = ap.parse_args()
    style()
    if a.which in ("body", "all"):
        print("[fig C] SS5.2 body figure")
        fig_body(a.dpi)
    if a.which in ("inrun", "all"):
        print("[fig A] fidelity vs in-run Shapley")
        fig_inrun(a.dpi, a.dose)
    if a.which in ("retrain", "all"):
        print("[fig B] fidelity vs retraining-based Shapley")
        fig_retrain(a.dpi)


if __name__ == "__main__":
    main()
