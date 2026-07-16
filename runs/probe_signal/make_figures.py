#!/usr/bin/env python
"""runs/probe_signal/make_figures.py -- A-axis signal-size probe figures.

Question (research-wiki/wiki/flirds-signal-size-diagnosis.md section 2): can training
levers (lr, steps, LoRA rank, participation, CNN width) make the cross-client signal
BIGGER -- and if bigger, is it REAL (stable across seeds)?

Reads ONLY rundir artifacts.  Baseline cells are reused from other groups by design
(README): LLM lr1e-3/st10/r16 anchor = runs/track_d/rundirs/1B_anchor5_seed0;
CNN c1 (w=1,k=1.0) = runs/track_c/c1/cifar10_*_seed*; CNN c2 (w=1,f=0.1) =
runs/track_c/c2/cifar10_iid_*_strmain_seed*.

Coverage caveat baked into titles: LLM probe cells are seed0-only (A-axis seeds 1-2
pending), so LLM panels show magnitude/fidelity, NOT cross-seed realness; the CNN
grids are 3-seed and carry the realness axis.

  python runs/probe_signal/make_figures.py
"""
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent            # runs/probe_signal
RUNS = HERE.parent
FIG = HERE / "figures"

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 8.5,
                     "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})

METHOD_ORDER = ["Flirds", "Flirds1st", "FedIF", "GTG", "FedSV", "ShapleyFL",
                "Banzhaf", "ComFedSV", "Fed-LOO", "loss-heur"]
METHOD_COLOR = {"(b)oracle": "#000000", "Flirds": "#0072B2", "Flirds1st": "#56B4E9",
                "GTG": "#E69F00", "FedSV": "#009E73", "ShapleyFL": "#D55E00",
                "Banzhaf": "#CC79A7", "FedIF": "#B8A000", "loss-heur": "#8C510A",
                "ComFedSV": "#4D9221", "Fed-LOO": "#999999"}
LRS, STEPS = [1e-3, 2e-3, 3e-3], [10, 20, 30]
RANKS = [16, 32, 64]
C1_W, C1_K = [0.5, 1.0, 2.0, 4.0], [0.2, 0.5, 1.0]
C2_GROUPS = [("w", 0.5, "f", 0.1), ("w", 1.0, "f", 0.05), ("w", 1.0, "f", 0.1),
             ("w", 1.0, "f", 0.2), ("w", 2.0, "f", 0.1), ("w", 4.0, "f", 0.1)]


def method_sort(ms):
    ms = list(ms)
    return [m for m in METHOD_ORDER if m in ms] + sorted(set(ms) - set(METHOD_ORDER))


def footnote(fig, text):
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.text(0.005, bb.y0 / fig.get_figheight() - 0.012, text,
             fontsize=6.5, color="#666666", va="top", ha="left")


def save(fig, path):
    FIG.mkdir(exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(HERE)}")


def annotated_heatmap(ax, mat, vmin, vmax, cmap, center=None, fmt="{:+.2f}"):
    arr = np.ma.masked_invalid(mat.to_numpy(float))
    if center is not None:
        half = max(vmax - center, center - vmin)
        vmin, vmax = center - half, center + half
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#DDDDDD")
    im = ax.imshow(arr, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            if np.isnan(v):
                ax.text(j, i, "-", ha="center", va="center", fontsize=6.5, color="#666666")
            else:
                norm = (v - vmin) / (vmax - vmin or 1)
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5,
                        color="white" if abs(norm - 0.5) > 0.30 else "black")
    ax.set_xticks(range(mat.shape[1]), mat.columns, fontsize=7.5)
    ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=7.5)
    ax.grid(False)
    return im


# ------------------------------------------------------------------ LLM loaders
def load_llm(d):
    """track_d-schema rundir -> dict(res, phi)."""
    metrics = json.loads((d / "metrics.json").read_text())
    key = next(iter(metrics))
    return dict(res=metrics[key], phi=pd.read_parquet(d / "phi.parquet"),
                cfg=yaml.safe_load((d / "config.yaml").read_text()))


def phi_spread(cell, method="(b)oracle"):
    """cross-client spread (std) of phi -- the ranking-relevant 'signal size'.
    (mean|phi| is dominated by the common learning shift shared by all clients,
    which cancels in any ranking; the diagnosis doc's magnitude axis is spread.)"""
    g = cell["phi"]
    v = g[g["method"] == method]["phi"].astype(float)
    return float(v.std(ddof=0)) if len(v) else np.nan


def llm_cells():
    """(kind, lever-label, sortkey) -> cell.  Baselines pulled from track_d."""
    out, missing = {}, []
    base = RUNS / "track_d" / "rundirs" / "1B_anchor5_seed0"
    if (base / "metrics.json").exists():
        out[("lrsteps", 1e-3, 10)] = load_llm(base)          # grid baseline (reused)
        out[("rank_anchor", 16)] = out[("lrsteps", 1e-3, 10)]
    else:
        missing.append("track_d/1B_anchor5_seed0 (baseline)")
    for lr in LRS:
        for st in STEPS:
            if (lr, st) == (1e-3, 10):
                continue
            name = f"1B_anchor5_lr{lr:.0e}_st{st}_seed0".replace("e-0", "e-")
            d = HERE / "rundirs" / name
            if (d / "metrics.json").exists():
                out[("lrsteps", lr, st)] = load_llm(d)
            else:
                missing.append(name)
    for r in (32, 64):
        d = HERE / "rundirs" / f"1B_anchor5_r{r}_seed0"
        if (d / "metrics.json").exists():
            out[("rank_anchor", r)] = load_llm(d)
        else:
            missing.append(d.name)
    for r in RANKS:
        d = HERE / "rundirs" / f"1B_std50k5_r{r}_seed0"
        if (d / "metrics.json").exists():
            out[("rank_std50k5", r)] = load_llm(d)
        else:
            missing.append(d.name)
    return out, missing


# ------------------------------------------------------------------ LLM figures
def fig_llm_magnitude(cells, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    # (left) lr x steps grid of cross-client SPREAD of the exact oracle's phi
    mat = pd.DataFrame(index=[f"lr={lr:g}" for lr in LRS],
                       columns=[f"steps={s}" for s in STEPS], dtype=float)
    for lr in LRS:
        for st in STEPS:
            c = cells.get(("lrsteps", lr, st))
            if c:
                mat.at[f"lr={lr:g}", f"steps={st}"] = phi_spread(c)
    ref = mat.at["lr=0.001", "steps=10"]
    im = annotated_heatmap(axes[0], mat / ref, 0, np.nanmax(mat.to_numpy() / ref),
                           "Blues", fmt="{:.2f}x")
    axes[0].set_title(f"anchor5: cross-client spread (std) of (b)oracle phi, vs baseline\n"
                      f"(baseline lr=0.001, steps=10: {ref:.2e})")
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="x baseline")
    # (right) LoRA rank levers, absolute scale
    xs = np.arange(len(RANKS))
    for dy, (kind, lab, col) in enumerate([("rank_anchor", "anchor5 (N=5 full)", "#0072B2"),
                                           ("rank_std50k5", "std50k5 (N=50, 5/round)", "#E69F00")]):
        vals = [phi_spread(cells[(kind, r)]) if (kind, r) in cells else np.nan for r in RANKS]
        axes[1].bar(xs + (dy - 0.5) * 0.36, vals, 0.34, color=col, label=lab)
        for x, v in zip(xs + (dy - 0.5) * 0.36, vals):
            if not np.isnan(v):
                axes[1].text(x, v, f"{v:.1e}", ha="center", va="bottom", fontsize=6.5)
    axes[1].set_xticks(xs, [f"r={r}" for r in RANKS])
    axes[1].set_ylabel("cross-client std of (b)oracle phi")
    axes[1].set_title("LoRA rank lever (absolute scale)")
    axes[1].legend(fontsize=7.5)
    fig.suptitle("A-axis levers vs phi signal SIZE (cross-client spread) -- LLM 1B, seed0 only\n"
                 "(cross-seed realness untested at this axis: seeds 1-2 pending)",
                 fontsize=10, y=1.06)
    footnote(fig, "source: rundirs/1B_*_seed0/phi.parquet + baseline runs/track_d/rundirs/1B_anchor5_seed0 (lr1e-3, st10, r16) "
                  "| spread = std over clients (common shift cancels in ranking)")
    save(fig, path)


def fig_llm_fidelity(cells, path):
    order = ([("lrsteps", lr, st) for lr in LRS for st in STEPS]
             + [("rank_anchor", r) for r in (32, 64)]
             + [("rank_std50k5", r) for r in RANKS])
    labels = ([f"lr{lr:g}/st{st}" for lr in LRS for st in STEPS]
              + [f"anchor r{r}" for r in (32, 64)]
              + [f"std50k5 r{r}" for r in RANKS])
    rows = {}
    for key, lab in zip(order, labels):
        c = cells.get(key)
        if not c:
            continue
        for m, v in c["res"]["spearman"].items():
            rows.setdefault(m, {})[lab] = v
    methods = method_sort(rows)
    mat = pd.DataFrame(index=methods, columns=labels, dtype=float)
    for m, d in rows.items():
        for lab, v in d.items():
            mat.at[m, lab] = v
    fig, ax = plt.subplots(figsize=(0.62 * len(labels) + 2.6, 0.3 * len(methods) + 1.9))
    im = annotated_heatmap(ax, mat, -1, 1, "RdBu", center=0.0)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    ax.axvline(8.5, color="black", lw=1.0)
    ax.axvline(10.5, color="black", lw=1.0)
    ax.set_title("Does estimator fidelity survive each lever? -- Spearman vs (b) oracle, seed0\n"
                 "left: lr x steps grid (anchor5) | middle: anchor rank | right: participation std50k5")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Spearman vs (b)")
    footnote(fig, "source: rundirs/*/metrics.json spearman (native) + track_d baseline for lr0.001/st10 | seed0 only")
    save(fig, path)


def fig_participation(cells, path):
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    methods = None
    for r in RANKS:
        c = cells.get(("rank_std50k5", r))
        if not c:
            continue
        sp = c["res"]["spearman"]
        if methods is None:
            methods = method_sort(sp)
    xs = np.arange(len(methods))
    w = 0.8 / len(RANKS)
    for ri, r in enumerate(RANKS):
        c = cells.get(("rank_std50k5", r))
        vals = [c["res"]["spearman"].get(m, np.nan) if c else np.nan for m in methods]
        ax.bar(xs + (ri - 1) * w, vals, w * 0.9, label=f"r={r}",
               color=["#0072B2", "#E69F00", "#009E73"][ri])
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(xs, methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Spearman vs (b) per-round oracle")
    ax.set_ylim(-1.05, 1.1)
    ax.legend(fontsize=8, title="LoRA rank", loc="lower left")
    ax.set_title("Partial participation stress (std50k5: N=50, 5/round, R=200) -- seed0\n"
                 "which estimators survive 10% participation?")
    footnote(fig, "source: rundirs/1B_std50k5_r{16,32,64}_seed0/metrics.json spearman")
    save(fig, path)


def fig_noise_probe(path):
    dirs = sorted((HERE / "noise_probe").glob("noise_1B_r*_seed0"))
    if not dirs:
        print("  [skip] noise_probe: no rundirs")
        return 0
    rows = []
    for d in dirs:
        m = json.loads((d / "metrics.json").read_text())
        r = int(re.search(r"_r(\d+)_", d.name).group(1))
        rows.append(dict(rank=r, spread_over_max_se=m["spread_over_max_se"],
                         boot_rho_self=m["boot_rho_self_mean"],
                         boot_rho_q05=m["boot_rho_est_q05"],
                         halfsplit_rho=m["halfsplit_rho"], n_chunks=m["n_chunks"],
                         n_boot=m["n_boot"]))
    df = pd.DataFrame(rows).sort_values("rank")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    panels = [("spread_over_max_se", "client phi spread / max bootstrap SE\n(<~2: spread within noise)", 1.0),
              ("boot_rho_self", "bootstrap self-Spearman (mean)", None),
              ("halfsplit_rho", "val half-split Spearman", None)]
    xs = np.arange(len(df))
    for ax, (k, ttl, hline) in zip(axes, panels):
        ax.bar(xs, df[k], 0.5, color=["#0072B2", "#D55E00"][:len(df)])
        for x, v in zip(xs, df[k]):
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        if hline is not None:
            ax.axhline(hline, color="#999999", lw=0.8, ls=":")
        ax.set_xticks(xs, [f"r={r}" for r in df["rank"]])
        ax.set_title(ttl, fontsize=8.5)
    fig.suptitle("Validation-noise probe -- is the anchor5 phi spread real vs val-chunk resampling noise? "
                 f"(1B seed0; {int(df['n_chunks'].iloc[0])} chunks, {int(df['n_boot'].iloc[0])} bootstrap)",
                 fontsize=9.5, y=1.04)
    footnote(fig, "source: noise_probe/noise_1B_r{16,64}_seed0/metrics.json (probe_val_noise.py output)")
    save(fig, path)
    return len(df)


# ------------------------------------------------------------------ CNN loaders
def cnn_c1_cells():
    """(scenario, w, k, seed) -> metrics dict; baseline w=1,k=1.0 from track_c/c1."""
    cells, missing = {}, []
    for scen in ("iid", "label-flip"):
        for w in C1_W:
            for k in C1_K:
                for seed in range(3):
                    if (w, k) == (1.0, 1.0):
                        d = RUNS / "track_c" / "c1" / f"cifar10_{scen}_seed{seed}"
                    else:
                        wtok = f"{w:g}" if w != int(w) else f"{int(w)}"
                        d = HERE / "cnn_c1" / f"pc1_cifar10_{scen}_w{wtok}_k{k:.1f}_seed{seed}"
                    if (d / "metrics.json").exists():
                        cells[(scen, w, k, seed)] = json.loads((d / "metrics.json").read_text())
                    else:
                        missing.append(str(d.relative_to(RUNS)))
    return cells, missing


def cnn_c2_cells():
    """(threat, w, f, seed) -> metrics; baseline w=1,f=0.1 from track_c/c2 strmain."""
    cells, missing = {}, []
    for threat in ("clean", "label-flip"):
        for _, w, _, f in C2_GROUPS:
            for seed in range(3):
                if (w, f) == (1.0, 0.1):
                    d = RUNS / "track_c" / "c2" / f"cifar10_iid_{threat}_strmain_seed{seed}"
                else:
                    wtok = f"{w:g}" if w != int(w) else f"{int(w)}"
                    d = HERE / "cnn_c2" / f"pc2_cifar10_{threat}_w{wtok}_f{f}_seed{seed}"
                if (d / "metrics.json").exists():
                    cells[(threat, w, f, seed)] = json.loads((d / "metrics.json").read_text())
                else:
                    missing.append(str(d.relative_to(RUNS)))
    return cells, missing


def fig_cnn_realness(cells, path):
    """cross-seed realness x magnitude of the exact oracle, per w x k cell."""
    scens = ["iid", "label-flip"]
    fig, axes = plt.subplots(2, len(scens), figsize=(4.6 * len(scens) + 1.2, 7.2),
                             gridspec_kw={"hspace": 0.52})
    csv_rows = []
    for si, scen in enumerate(scens):
        rho = pd.DataFrame(index=[f"w={w:g}" for w in C1_W],
                           columns=[f"k={k:g}" for k in C1_K], dtype=float)
        mag = rho.copy()
        for w in C1_W:
            for k in C1_K:
                vecs = {s: np.asarray(cells[(scen, w, k, s)]["methods"]["(b)oracle"]["phi"], float)
                        for s in range(3) if (scen, w, k, s) in cells}
                pairs = [float(spearmanr(vecs[a], vecs[b]).statistic)
                         for a, b in combinations(sorted(vecs), 2)]
                if pairs:
                    rho.at[f"w={w:g}", f"k={k:g}"] = np.mean(pairs)
                mags = [v.std() for v in vecs.values()]        # cross-client spread
                if mags:
                    mag.at[f"w={w:g}", f"k={k:g}"] = np.mean(mags)
                csv_rows.append(dict(scenario=scen, width=w, kfrac=k, n_seeds=len(vecs),
                                     xseed_rho_mean=np.mean(pairs) if pairs else np.nan,
                                     oracle_phi_spread_mean=np.mean(mags) if mags else np.nan))
        im0 = annotated_heatmap(axes[0][si], rho, -1, 1, "RdBu", center=0.0)
        axes[0][si].set_title(f"{scen}: cross-seed rho of (b)oracle phi (REALNESS)")
        ref = mag.at["w=1", "k=1"]
        im1 = annotated_heatmap(axes[1][si], mag / ref, 0,
                                float(np.nanmax(mag.to_numpy() / ref)), "Blues", fmt="{:.2f}x")
        axes[1][si].set_title(f"{scen}: phi spread / baseline (SIZE)\n"
                              f"(baseline w=1,k=1: {ref:.2e})")
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="cross-seed Spearman")
    fig.colorbar(im1, ax=axes[1], shrink=0.85, label="x baseline")
    fig.suptitle("CNN C1 probe (CIFAR-10, N=10): does a bigger phi mean a more REAL signal?\n"
                 "width x participation grid, 3 seeds -- size and realness move independently",
                 fontsize=10, y=0.98)
    footnote(fig, "source: cnn_c1/pc1_*/metrics.json methods.(b)oracle.phi + baseline runs/track_c/c1/cifar10_*_seed* (w=1,k=1) "
                  "| spread = std over clients")
    save(fig, path)
    return pd.DataFrame(csv_rows)


def fig_cnn_fidelity(cells, path):
    scens = ["iid", "label-flip"]
    show = ["Flirds", "GTG", "FedSV", "ComFedSV"]
    fig, axes = plt.subplots(len(show), len(scens),
                             figsize=(4.2 * len(scens) + 1.0, 2.1 * len(show) + 1.2),
                             squeeze=False)
    for si, scen in enumerate(scens):
        for mi, meth in enumerate(show):
            mat = pd.DataFrame(index=[f"w={w:g}" for w in C1_W],
                               columns=[f"k={k:g}" for k in C1_K], dtype=float)
            for w in C1_W:
                for k in C1_K:
                    vs = [cells[(scen, w, k, s)]["methods"].get(meth, {}).get("spearman_b")
                          for s in range(3) if (scen, w, k, s) in cells]
                    vs = [v for v in vs if v is not None]
                    if vs:
                        mat.at[f"w={w:g}", f"k={k:g}"] = np.mean(vs)
            im = annotated_heatmap(axes[mi][si], mat, -1, 1, "RdBu", center=0.0)
            axes[mi][si].set_title(f"{scen} -- {meth}", fontsize=8.5)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Spearman vs (b)oracle (mean of 3 seeds)")
    fig.suptitle("CNN C1 probe -- estimator fidelity across width x participation", fontsize=10.5,
                 y=0.995)
    footnote(fig, "source: cnn_c1/pc1_*/metrics.json methods.<M>.spearman_b + track_c/c1 baseline (w=1,k=1)")
    save(fig, path)


def fig_cnn_c2(cells, path):
    arm_order = ["vanilla", "flirds_mult", "flirds_select", "shapleyfl", "fedif", "sfedavg"]
    arm_color = {"vanilla": "#888888", "flirds_mult": "#0072B2", "flirds_select": "#56B4E9",
                 "shapleyfl": "#D55E00", "fedif": "#B8A000", "sfedavg": "#CC79A7"}
    groups = [(w, f) for _, w, _, f in C2_GROUPS]
    glabels = [f"w={w:g}\nf={f:g}" for w, f in groups]
    threats = ["clean", "label-flip"]
    fig, axes = plt.subplots(2, len(threats), figsize=(5.4 * len(threats), 6.2),
                             squeeze=False)
    for ti, threat in enumerate(threats):
        # top: final acc per arm
        ax = axes[0][ti]
        xs = np.arange(len(groups))
        w_ = 0.8 / len(arm_order)
        vals_all = []
        for ai, arm in enumerate(arm_order):
            means, dots = [], []
            for w, f in groups:
                vs = [cells[(threat, w, f, s)]["arms"][arm]["final_acc"]
                      for s in range(3) if (threat, w, f, s) in cells
                      and arm in cells[(threat, w, f, s)]["arms"]]
                means.append(np.mean(vs) if vs else np.nan)
                dots.append(vs)
                vals_all += vs
            pos = xs + (ai - (len(arm_order) - 1) / 2) * w_
            ax.bar(pos, means, w_ * 0.9, color=arm_color[arm],
                   label=arm if ti == 0 else None)
            for x, vs in zip(pos, dots):
                ax.scatter([x] * len(vs), vs, s=5, color="black", alpha=0.6, zorder=3,
                           linewidths=0)
        ax.set_xticks(xs, glabels, fontsize=7.5)
        if vals_all:
            lo, hi = min(vals_all), max(vals_all)
            pad = (hi - lo) * 0.15 + 1e-3
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{threat}: final accuracy by arm")
        if ti == 0:
            ax.set_ylabel("final test accuracy")
        # bottom: detection AUROC (label-flip only has corrupt clients)
        ax = axes[1][ti]
        drawn = False
        for ai, arm in enumerate(arm_order):
            means = []
            for w, f in groups:
                vs = [cells[(threat, w, f, s)]["arms"][arm].get("auroc")
                      for s in range(3) if (threat, w, f, s) in cells
                      and arm in cells[(threat, w, f, s)]["arms"]]
                vs = [v for v in vs if v is not None and not (isinstance(v, float) and np.isnan(v))]
                means.append(np.mean(vs) if vs else np.nan)
            if not all(np.isnan(v) for v in means):
                drawn = True
                pos = xs + (ai - (len(arm_order) - 1) / 2) * w_
                ax.bar(pos, means, w_ * 0.9, color=arm_color[arm])
        if drawn:
            ax.axhline(0.5, color="#999999", lw=0.8, ls=":")
            ax.set_xticks(xs, glabels, fontsize=7.5)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{threat}: detection AUROC by arm (3rd-tier axis)")
            if ti == 0:
                ax.set_ylabel("AUROC")
        else:
            ax.axis("off")
            ax.set_title(f"{threat}: AUROC undefined (no corrupt clients)", fontsize=8.5)
    fig.legend(loc="upper center", ncol=len(arm_order), fontsize=7.5,
               bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.suptitle("CNN C2 probe (N=100, C=0.1->f sweep, T=120) -- intervention outcome across width/frac",
                 fontsize=10, y=1.07)
    fig.tight_layout()
    footnote(fig, "source: cnn_c2/pc2_*/metrics.json arms + baseline runs/track_c/c2/cifar10_iid_*_strmain_seed* (w=1,f=0.1)")
    save(fig, path)


# ------------------------------------------------------------------ main
def main():
    print("== coverage ==")
    llm, llm_missing = llm_cells()
    n_grid = sum(1 for k in llm if k[0] == "lrsteps")
    print(f"  LLM lr x steps grid: {n_grid}/9 cells (incl. reused track_d baseline) | "
          f"anchor rank: {sum(1 for k in llm if k[0] == 'rank_anchor')}/3 | "
          f"std50k5 rank: {sum(1 for k in llm if k[0] == 'rank_std50k5')}/3")
    print("  LLM probe cells are SEED0-ONLY (A-axis seeds 1-2 pending; cross-seed realness "
          "shown only on the 3-seed CNN grids)")
    if llm_missing:
        print(f"    MISSING: {', '.join(llm_missing)}")
    c1, c1_missing = cnn_c1_cells()
    print(f"  CNN C1 w x k grid: {len(c1)}/72 cells (2 scen x 4 w x 3 k x 3 seeds; "
          f"w=1,k=1 reused from track_c/c1)")
    if c1_missing:
        print(f"    MISSING ({len(c1_missing)}): {', '.join(c1_missing[:8])}"
              + (" ..." if len(c1_missing) > 8 else ""))
    c2, c2_missing = cnn_c2_cells()
    print(f"  CNN C2 w/f sweep: {len(c2)}/36 cells (2 threat x 6 (w,f) x 3 seeds; "
          f"w=1,f=0.1 reused from track_c/c2 strmain)")
    if c2_missing:
        print(f"    MISSING ({len(c2_missing)}): {', '.join(c2_missing)}")

    print("== figures ==")
    fig_llm_magnitude(llm, FIG / "01_llm_phi_magnitude_levers.png")
    fig_llm_fidelity(llm, FIG / "02_llm_fidelity_across_levers.png")
    fig_participation(llm, FIG / "03_llm_participation_std50k5.png")
    n_noise = fig_noise_probe(FIG / "04_llm_noise_probe_se.png")
    print(f"  noise_probe cells: {n_noise}/2")
    realness = fig_cnn_realness(c1, FIG / "05_cnn_c1_realness_vs_magnitude.png")
    fig_cnn_fidelity(c1, FIG / "06_cnn_c1_fidelity_grid.png")
    fig_cnn_c2(c2, FIG / "07_cnn_c2_arms_outcome.png")

    FIG.mkdir(exist_ok=True)
    realness.to_csv(FIG / "cnn_c1_realness.csv", index=False)
    rows = []
    for key, c in llm.items():
        lab = "_".join(str(x) for x in key)
        for m, v in c["res"]["spearman"].items():
            rows.append(dict(cell=lab, method=m, spearman=v,
                             oracle_phi_spread=phi_spread(c)))
    pd.DataFrame(rows).to_csv(FIG / "llm_probe_summary.csv", index=False)
    print(f"  wrote figures/cnn_c1_realness.csv ({len(realness)} rows), "
          f"figures/llm_probe_summary.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
