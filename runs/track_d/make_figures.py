#!/usr/bin/env python
"""runs/track_d/make_figures.py -- committed figures for Track D (IID-clean LLM stage).

Stage: OpenFedLLM run_sft verbatim (alpaca-gpt4 IID), regimes std20 (N=20, 2/round,
R=200) and anchor5 (N=5 full, R=30, exact (b) 2^5; 1B additionally has the (a)-retrain
oracle).  Reads ONLY rundir artifacts (config.yaml / metrics.json / phi.parquet) and
regenerates every PNG under figures/.  fidelity.csv / target_stability.csv are the
gitignored outputs of make_fidelity.py / make_target_stability.py -- this script
recomputes from phi.parquet instead of depending on them (and cross-checks when the
files happen to exist).

Figure order = question hierarchy: fidelity -> target stability -> intervention
performance -> convergence -> cost (detection has no axis here: clean stage).

  python runs/track_d/make_figures.py
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent            # runs/track_d
RUNDIRS = HERE / "rundirs"
FIG = HERE / "figures"

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 8.5,
                     "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})

SCALES = ["1B", "3B", "7B"]
REGIMES = ["std20", "anchor5"]
SEEDS = [0, 1, 2]
METHOD_ORDER = ["(a)oracle", "Flirds", "Flirds1st", "FedIF", "GTG", "FedSV", "ShapleyFL",
                "Banzhaf", "ComFedSV", "loss-heur"]
METHOD_COLOR = {"(b)oracle": "#000000", "(a)oracle": "#009E73", "Flirds": "#0072B2",
                "Flirds1st": "#56B4E9", "GTG": "#E69F00", "FedSV": "#009E73",
                "ShapleyFL": "#D55E00", "Banzhaf": "#CC79A7", "FedIF": "#B8A000",
                "loss-heur": "#8C510A", "ComFedSV": "#4D9221"}
ARM_ORDER = ["base", "vanilla", "flirds_w", "flirds_sel", "shapleyfl_w", "fedif_w"]
ARM_COLOR = {"base": "#000000", "vanilla": "#888888", "flirds_w": "#0072B2",
             "flirds_sel": "#56B4E9", "shapleyfl_w": "#D55E00", "fedif_w": "#B8A000"}
JUNE_ERA_NOTE = "7B cells are June-era completed runs; post-migration re-run deferred"


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
                ax.text(j, i, "-", ha="center", va="center", fontsize=6, color="#666666")
            else:
                norm = (v - vmin) / (vmax - vmin or 1)
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6,
                        color="white" if abs(norm - 0.5) > 0.30 else "black")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=7)
    ax.grid(False)
    return im


# ------------------------------------------------------------------ load
def load_cells():
    cells, missing = {}, []
    for scale in SCALES:
        for regime in REGIMES:
            for seed in SEEDS:
                name = f"{scale}_{regime}_seed{seed}"
                d = RUNDIRS / name
                if not (d / "metrics.json").exists():
                    missing.append(name)
                    continue
                metrics = json.loads((d / "metrics.json").read_text())
                key = next(iter(metrics))                     # single 'seedN' block per dir
                cells[(scale, regime, seed)] = dict(
                    name=name, scale=scale, regime=regime, seed=seed,
                    cfg=yaml.safe_load((d / "config.yaml").read_text()),
                    res=metrics[key], phi=pd.read_parquet(d / "phi.parquet"))
    return cells, missing


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def pearson_vs_b(cell):
    """value-level fidelity vs (b)oracle from phi.parquet (metrics has no native field)."""
    g = cell["phi"]
    vecs = {m: gm.set_index("client")["phi"] for m, gm in g.groupby("method")}
    if "(b)oracle" not in vecs:
        return {}
    tv = vecs["(b)oracle"]
    out = {}
    for m, mv in vecs.items():
        if m == "(b)oracle":
            continue
        cl = mv.index.intersection(tv.index)
        out[m] = _pearson(mv.loc[cl], tv.loc[cl])
    return out


# ------------------------------------------------------------------ figures
def fig_fidelity(cells, path):
    cols = [f"{s} {r}" for s in SCALES for r in REGIMES]
    panels = [("spearman", "Spearman vs (b) oracle (rank)", None),
              ("pearson", "Pearson vs (b) oracle (value-level, from phi)", None)]
    mats = []
    for key, _, _ in panels:
        rows = {}
        for (scale, regime, seed), c in cells.items():
            lab = f"{scale} {regime}"
            vals = c["res"]["spearman"] if key == "spearman" else pearson_vs_b(c)
            for m, v in vals.items():
                rows.setdefault(m, {}).setdefault(lab, []).append(v)
        methods = method_sort(rows)
        mat = pd.DataFrame(index=methods, columns=cols, dtype=float)
        for m, d in rows.items():
            for lab, vs in d.items():
                vs = [v for v in vs if v is not None and not np.isnan(v)]
                if vs:
                    mat.at[m, lab] = np.mean(vs)
        mats.append(mat)
    fig, axes = plt.subplots(1, 2, figsize=(2 * (0.75 * len(cols)) + 4.6,
                                            0.3 * max(len(m) for m in mats) + 2.1))
    for ax, mat, (_, ttl, _) in zip(axes, mats, panels):
        im = annotated_heatmap(ax, mat, -1, 1, "RdBu", center=0.0)
        ax.set_title(ttl)
    fig.colorbar(im, ax=axes, shrink=0.8, label="correlation vs (b) oracle")
    fig.suptitle("Track D fidelity -- IID-clean LLM stage, mean over 3 seeds\n"
                 "std20: (b)=per-round exact over participants | anchor5: (b)=exact 2^5 | "
                 "(a)oracle row = 2^5 retrain oracle (1B anchor5 only)", fontsize=10, y=1.06)
    footnote(fig, f"source: rundirs/<scale>_<regime>_seed*/{{metrics.json,phi.parquet}} | {JUNE_ERA_NOTE}")
    save(fig, path)


def fig_target_stability(cells, path):
    groups = [(s, r) for s in SCALES for r in REGIMES]
    stats, csv_rows = [], []
    for s, r in groups:
        vecs = {}
        for seed in SEEDS:
            c = cells.get((s, r, seed))
            if c is None:
                continue
            g = c["phi"]
            v = g[g["method"] == "(b)oracle"].set_index("client")["phi"].sort_index()
            if len(v):
                vecs[seed] = v
        pairs = []
        for a, b in combinations(sorted(vecs), 2):
            va, vb = vecs[a].align(vecs[b], join="inner")
            pairs.append(float(spearmanr(va, vb).statistic))
        stats.append((f"{s} {r}", pairs))
        csv_rows.append(dict(cell=f"{s}_{r}", n_seeds=len(vecs), n_pairs=len(pairs),
                             mean_xseed_spearman=np.mean(pairs) if pairs else np.nan))
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    xs = np.arange(len(stats))
    means = [np.mean(p) if p else np.nan for _, p in stats]
    colors = ["#0072B2" if "anchor5" in lab else "#E69F00" for lab, _ in stats]
    ax.bar(xs, means, 0.6, color=colors, alpha=0.85)
    for x, (_, pairs) in zip(xs, stats):
        ax.scatter([x] * len(pairs), pairs, s=14, color="black", alpha=0.7, zorder=3,
                   linewidths=0)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(xs, [lab for lab, _ in stats])
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("cross-seed Spearman of (b)-oracle phi\n(mean of 3 seed-pairs; dots=pairs)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#E69F00", label="std20 (n=20)"),
                       plt.Rectangle((0, 0), 1, 1, color="#0072B2", label="anchor5 (n=5)")],
              fontsize=8, loc="upper left")
    ax.set_title("Exp C target stability -- is the oracle's own ranking reproducible across seeds?\n"
                 "(IID-clean stage: low bars = no real cross-client signal to estimate)")
    footnote(fig, f"source: rundirs/*/phi.parquet method=(b)oracle | {JUNE_ERA_NOTE}")
    save(fig, path)
    return pd.DataFrame(csv_rows)


def fig_arms(cells, path):
    metrics = [("mmlu", "MMLU accuracy (full test, 0-shot)"),
               ("rouge_l", "ROUGE-L (Alpaca test, same-distribution)")]
    fig, axes = plt.subplots(len(metrics), len(REGIMES), figsize=(11.0, 6.4),
                             sharex="col", squeeze=False)
    for mi, (mk, mlab) in enumerate(metrics):
        for ri, regime in enumerate(REGIMES):
            ax = axes[mi][ri]
            arms = [a for a in ARM_ORDER
                    if any(a in c["res"]["arms"] for c in cells.values() if c["regime"] == regime)]
            xs = np.arange(len(SCALES))
            w = 0.8 / len(arms)
            for ai, arm in enumerate(arms):
                means, dots = [], []
                for si, scale in enumerate(SCALES):
                    vs = [c["res"]["arms"][arm][mk] for (s, r, sd), c in cells.items()
                          if s == scale and r == regime and arm in c["res"]["arms"]]
                    means.append(np.mean(vs) if vs else np.nan)
                    dots.append(vs)
                pos = xs + (ai - (len(arms) - 1) / 2) * w
                ax.bar(pos, means, w * 0.9, color=ARM_COLOR[arm], alpha=0.9,
                       label=arm if (mi == 0 and ri == 0) else None)
                for x, vs in zip(pos, dots):
                    ax.scatter([x] * len(vs), vs, s=6, color="black", alpha=0.6,
                               zorder=3, linewidths=0)
            ax.set_xticks(xs, SCALES)
            if ri == 0:
                ax.set_ylabel(mlab)
            if mi == 0:
                ax.set_title(f"{regime}")
            vals = [c["res"]["arms"][a][mk] for c in cells.values() if c["regime"] == regime
                    for a in arms if a in c["res"]["arms"]]
            if vals:
                lo, hi = min(vals), max(vals)
                pad = (hi - lo) * 0.25 + 1e-3
                ax.set_ylim(lo - pad, hi + pad)
    fig.legend(loc="upper center", ncol=len(ARM_ORDER), fontsize=8,
               bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.suptitle("Intervention arms -- end-model quality (clean-IID expectation: do-no-harm parity)\n"
                 "bars=mean over seeds, dots=seeds | y-axis zoomed to data range",
                 fontsize=10, y=1.12)
    fig.tight_layout()
    footnote(fig, f"source: rundirs/*/metrics.json arms.<arm>.{{mmlu,rouge_l}} | flirds_sel exists in std20 only | {JUNE_ERA_NOTE}")
    save(fig, path)


def fig_val_curves(cells, path):
    fig, axes = plt.subplots(len(REGIMES), len(SCALES), figsize=(11.4, 5.8),
                             sharex="row", squeeze=False)
    for ri, regime in enumerate(REGIMES):
        for si, scale in enumerate(SCALES):
            ax = axes[ri][si]
            arms = [a for a in ARM_ORDER
                    if any(a in c["res"]["arms"] for c in cells.values()
                           if c["regime"] == regime and c["scale"] == scale)]
            for arm in arms:
                curves = [np.asarray(c["res"]["arms"][arm]["val_curve"], float)
                          for (s, r, sd), c in cells.items()
                          if s == scale and r == regime
                          and "val_curve" in c["res"]["arms"].get(arm, {})]
                if not curves:
                    continue                     # 'base' arm persists no curve by design
                L = min(len(c) for c in curves)
                mean = np.mean([c[:L] for c in curves], axis=0)
                ax.plot(range(L), mean, color=ARM_COLOR[arm], lw=1.2,
                        label=arm if (ri == 0 and si == 0) else None)
            ax.set_title(f"{scale} {regime}", fontsize=9)
            if si == 0:
                ax.set_ylabel("val loss (mean over seeds)")
            if ri == len(REGIMES) - 1:
                ax.set_xlabel("round")
    fig.legend(loc="upper center", ncol=len(ARM_ORDER), fontsize=8,
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.suptitle("Convergence -- validation-loss curves per intervention arm", fontsize=10.5, y=1.09)
    fig.tight_layout()
    footnote(fig, f"source: rundirs/*/metrics.json arms.<arm>.val_curve | {JUNE_ERA_NOTE}")
    save(fig, path)


def fig_runtime(cells, path):
    rows = []
    for (scale, regime, seed), c in cells.items():
        for m, v in c["res"]["runtime"].items():
            rows.append(dict(scale=scale, regime=regime, method=m, runtime=v))
    df = pd.DataFrame(rows)
    methods = method_sort(df["method"].unique())[::-1]
    if "(b)oracle" in methods:                                # oracle drawn last on top
        methods.remove("(b)oracle")
        methods.append("(b)oracle")
    fig, ax = plt.subplots(figsize=(7.0, 0.32 * len(methods) + 1.6))
    mark = {"1B": "o", "3B": "s", "7B": "^"}
    for i, m in enumerate(methods):
        for scale in SCALES:
            v = df[(df["method"] == m) & (df["scale"] == scale)]["runtime"].astype(float)
            if not len(v):
                continue
            ax.scatter(v, np.full(len(v), i), s=13, marker=mark[scale],
                       color=METHOD_COLOR.get(m, "#BBBBBB"), alpha=0.65, linewidths=0)
    ax.set_xscale("log")
    ax.set_yticks(range(len(methods)), methods, fontsize=7.5)
    ax.set_xlabel("valuation wall-clock per cell [s, log]")
    ax.set_title("Cost -- per-method valuation wall-clock (marker=scale)")
    ax.legend(handles=[plt.Line2D([], [], marker=mark[s], ls="", color="#555555", label=s)
                       for s in SCALES], fontsize=7.5, loc="lower right")
    footnote(fig, "source: rundirs/*/metrics.json runtime (both regimes, all seeds)")
    save(fig, path)


# ------------------------------------------------------------------ main
def main():
    cells, missing = load_cells()
    print("== coverage ==")
    print(f"  cells found: {len(cells)}/18 (3 scales x 2 regimes x 3 seeds)")
    for (s, r), n in pd.Series([(c['scale'], c['regime']) for c in cells.values()]
                               ).value_counts().sort_index().items():
        print(f"    {s}_{r}: {n} seeds")
    if missing:
        print(f"    MISSING: {', '.join(missing)}")
    a_cells = sorted(c["name"] for c in cells.values() if "(a)oracle" in c["res"]["spearman"])
    print(f"  (a)-retrain oracle present in: {', '.join(a_cells) or '-'}")
    print(f"  NOTE: {JUNE_ERA_NOTE} (see figures/MANIFEST.md for the ShapleyFL beta caveat)")

    print("== figures ==")
    fig_fidelity(cells, FIG / "01_fidelity_by_scale_regime.png")
    stab = fig_target_stability(cells, FIG / "02_target_stability_oracle_crossseed.png")
    fig_arms(cells, FIG / "03_arms_mmlu_rouge.png")
    fig_val_curves(cells, FIG / "04_convergence_val_curves.png")
    fig_runtime(cells, FIG / "05_cost_runtime_by_method.png")

    # cross-check vs make_target_stability.py output when present (independent code path)
    ts = HERE / "target_stability.csv"
    if ts.exists():
        ref = pd.read_csv(ts).set_index("cell")["mean_xseed_spearman"]
        ours = stab.set_index("cell")["mean_xseed_spearman"]
        both = ref.index.intersection(ours.index)
        bad = [c for c in both if abs(ref[c] - ours[c]) > 1e-9]
        print(f"  target-stability cross-check vs {ts.name}: "
              f"{len(both) - len(bad)}/{len(both)} match" + (f" MISMATCH: {bad}" if bad else ""))

    # small summary CSVs (exact figure inputs) for the overview session
    FIG.mkdir(exist_ok=True)
    rows = []
    for (scale, regime, seed), c in cells.items():
        pe = pearson_vs_b(c)
        for m, v in c["res"]["spearman"].items():
            rows.append(dict(cell=c["name"], scale=scale, regime=regime, seed=seed, method=m,
                             spearman=v, kendall=c["res"]["kendall"].get(m),
                             pearson_vs_b=pe.get(m), runtime=c["res"]["runtime"].get(m)))
    pd.DataFrame(rows).to_csv(FIG / "fidelity_summary.csv", index=False)
    arows = []
    for (scale, regime, seed), c in cells.items():
        for arm, r in c["res"]["arms"].items():
            arows.append(dict(cell=c["name"], scale=scale, regime=regime, seed=seed, arm=arm,
                              mmlu=r.get("mmlu"), rouge_l=r.get("rouge_l"),
                              final_val_loss=r.get("final_val_loss"),
                              rounds_to_target=r.get("rounds_to_target")))
    pd.DataFrame(arows).to_csv(FIG / "arms_summary.csv", index=False)
    stab.to_csv(FIG / "target_stability_recomputed.csv", index=False)
    print(f"  wrote figures/fidelity_summary.csv ({len(rows)} rows), "
          f"figures/arms_summary.csv ({len(arows)} rows), figures/target_stability_recomputed.csv")


if __name__ == "__main__":
    main()
