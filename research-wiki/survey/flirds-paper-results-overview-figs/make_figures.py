#!/usr/bin/env python
"""Regenerate the figures embedded in `flirds-paper-results-overview.md`.

Self-contained: matplotlib only, all inputs are repo-local CSV / rundir paths,
all outputs are PNGs written next to this script. No network, no GPU, read-only
over `runs/`. Any figure whose input data has not yet landed is skipped (the page
keeps its ⬚ placeholder) rather than drawing an empty axis.

Run (any python with numpy/pandas/matplotlib/pyarrow/scipy):
    python make_figures.py
On Yonghee's box:  C:\\Users\\chyoy\\anaconda3\\python.exe make_figures.py

Figure ↔ paper §  (see the page for captions and provenance):
  F1  §5.2  CNN C1 scenario×method  vs (a) retrain-oracle  Spearman heatmap
  F2  §5.2  LLM anchor5 all-methods vs (a) retrain-oracle  Spearman bar (±std)
  F5  §5.3  CNN dir1 P1-online score-source competition, absolute acc by threat
  F7  §5.5  cost: measured per-method wall-clock + (b)/Flirds exponential scaling
  F8  §5.6  removal curves (LLM silo5 val-loss, CNN cifar10 acc; worst/best-first)
  F9  §5.6  second-order ablation: Flirds vs Flirds-1st fidelity across participation k
  F3/F4/F6 = ⬚ (main-pair vs (b) / R4 EM / detection AUROC) — data not yet landed.
"""
import os, sys, json, glob, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RUNS = os.path.join(REPO, "runs")

# --- shared style -----------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
EXCLUDE = {"Banzhaf", "Ripple", "Fed-LOO"}                 # off baseline-set (memory)
M8 = ["Flirds", "Flirds1st", "loss-heur", "GTG", "FedSV", "ComFedSV", "ShapleyFL", "FedIF"]
ESTIM = {"Flirds", "Flirds1st", "loss-heur", "FedIF", "flirds", "flirds1st", "lossheur", "fedif"}
C_ESTIM, C_RENORM, C_FLIRDS = "#2166ac", "#b2182b", "#1a9850"  # blue / red / green


def fam_color(m):
    if m in ("Flirds", "flirds"):
        return C_FLIRDS
    return C_ESTIM if m in ESTIM else C_RENORM


def save(fig, name, tight=True):
    out = os.path.join(HERE, name)
    if tight:
        fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {name}")
    return True


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def inner(m):
    """removal LLM rundirs wrap metrics under a single {cellname: {...}} key;
    CNN/probe rundirs are flat.  Return the dict that holds the metric keys."""
    if len(m) == 1:
        (v,) = m.values()
        if isinstance(v, dict):
            return v
    return m


# ---------------------------------------------------------------------------
# F1 — CNN C1 scenario × method vs (a) retrain oracle, Spearman heatmap
# ---------------------------------------------------------------------------
def f1():
    import pandas as pd
    csv = os.path.join(RUNS, "track_c", "fidelity.csv")
    if not os.path.exists(csv):
        print("  [skip] F1: missing track_c/fidelity.csv"); return False
    df = pd.read_csv(csv)
    df = df[~df["method"].isin(EXCLUDE)]
    # per dataset block: corruption scenarios first (signal), no-corruption last
    scen_order = ["feature_noise", "label_flip", "quantity_skew", "label_skew", "iid"]
    rows = ([("cifar10", s) for s in scen_order] + [("mnist", s) for s in scen_order])
    M = [m for m in M8 if m in df["method"].unique()]

    def build(col):
        piv = df.groupby(["dataset", "scenario", "method"])[col].mean()
        g = np.full((len(rows), len(M)), np.nan)
        for i, (d, s) in enumerate(rows):
            for j, m in enumerate(M):
                try:
                    g[i, j] = piv.loc[(d, s, m)]
                except KeyError:
                    pass
        return g

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.4), sharey=True)
    im = None
    for ax, (col, name) in zip(axes, [("spearman_a", "Spearman"), ("pearson_a", "Pearson")]):
        grid = build(col)
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(M))); ax.set_xticklabels(M, rotation=35, ha="right")
        ax.axhline(len(scen_order) - 0.5, color="k", lw=1.0)   # cifar10 | mnist split
        for i in range(len(rows)):
            for j in range(len(M)):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(v) > 0.55 else "black")
        ax.set_title(f"{name} vs (a)", fontsize=9.5)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([f"{d}/{s}" for d, s in rows], fontsize=8)
    fig.subplots_adjust(left=0.065, right=0.9, top=0.9, bottom=0.12, wspace=0.05)
    cax = fig.add_axes([0.923, 0.12, 0.012, 0.78])
    fig.colorbar(im, cax=cax, label="corr vs (a)")
    fig.suptitle("F1 · CNN C1 fidelity vs (a) retrain oracle  (3-seed mean; per dataset: "
                 "corruption scenarios top, no-corruption label_skew/iid foot)", fontsize=9.5)
    return save(fig, "f1_cnn_c1_vs_a_heatmap.png", tight=False)


# ---------------------------------------------------------------------------
# F2 — LLM anchor5 all-methods vs (a) retrain oracle, Spearman bar (±std)
# ---------------------------------------------------------------------------
def f2():
    import pandas as pd
    paths = sorted(glob.glob(os.path.join(RUNS, "track_d", "rundirs",
                                          "1B_anchor5_seed*", "phi.parquet")))
    if not paths:
        print("  [skip] F2: missing 1B_anchor5_seed*/phi.parquet"); return False
    per_method, boracle_vs_a = {m: [] for m in M8}, []
    for p in paths:
        d = pd.read_parquet(p)
        wide = d.pivot_table(index="client", columns="method", values="phi")
        if "(a)oracle" not in wide.columns:
            continue
        truth = wide["(a)oracle"].values
        for m in M8:
            if m in wide.columns:
                per_method[m].append(spearmanr(wide[m].values, truth).correlation)
        if "(b)oracle" in wide.columns:
            boracle_vs_a.append(spearmanr(wide["(b)oracle"].values, truth).correlation)
    M = [m for m in M8 if per_method[m]]
    means = [np.mean(per_method[m]) for m in M]
    stds = [np.std(per_method[m]) for m in M]           # ddof=0

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(range(len(M)), means, yerr=stds, capsize=3,
                  color=[fam_color(m) for m in M], edgecolor="black", linewidth=0.4)
    if boracle_vs_a:
        ref = np.mean(boracle_vs_a)
        ax.axhline(ref, ls="--", color="black", lw=1.1,
                   label=f"(b)↔(a) dual-oracle agreement = {ref:.3f}")
        ax.legend(loc="lower left", fontsize=8)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.3f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(M))); ax.set_xticklabels(M, rotation=25, ha="right")
    ax.set_ylabel("Spearman vs (a) retrain oracle"); ax.set_ylim(-0.2, 1.08)
    ax.set_title("F2 · LLM anchor5 (N=5) all methods vs (a) retrain oracle\n"
                 "(3-seed mean±std; same-game methods sit at the (b)↔(a) ceiling)", fontsize=9)
    return save(fig, "f2_anchor5_vs_a_bar.png")


# ---------------------------------------------------------------------------
# F5 — CNN dir1 P1-online score-source competition, absolute acc by threat
# ---------------------------------------------------------------------------
def f5():
    import pandas as pd
    csv = os.path.join(RUNS, "track_h", "analysis", "cnn_competition.csv")
    if not os.path.exists(csv):
        print("  [skip] F5: missing track_h/analysis/cnn_competition.csv"); return False
    c = pd.read_csv(csv)
    base = c[(c.dataset == "cifar10") & (c.partition == "dir1") & (c.timing == "online")]
    p1 = base[base.policy == "P1"].copy()
    # label_flip: pin to dose 0.70 to match the headline table (others pool cleanly)
    p1 = p1[~((p1.threat == "label_flip") & (p1.flip_rate != 0.7))]
    threats = ["clean", "free_rider", "grad_noise", "label_flip"]
    tlabel = {"clean": "clean", "free_rider": "free-rider",
              "grad_noise": "grad-noise", "label_flip": "label-flip@.70"}
    sources = ["flirds", "flirds1st", "lossheur", "fedif",
               "gtg", "fedsv", "comfedsv", "shapleyfl"]
    acc = (p1.groupby(["threat", "source"])["final_acc"].mean())

    def anchor(arm, thr):
        s = base[(base.arm == arm) & (base.threat == thr)]["final_acc"]
        return s.mean() if len(s) else np.nan

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    n, w = len(sources), 0.10
    x = np.arange(len(threats))
    for k, src in enumerate(sources):
        vals = [acc.get((t, src), np.nan) for t in threats]
        ax.bar(x + (k - n / 2) * w + w / 2, vals, w, label=src,
               color=fam_color(src), edgecolor="black", linewidth=0.3,
               alpha=0.95 if src in ESTIM else 0.7)
    for i, t in enumerate(threats):
        van, orc = anchor("vanilla", t), anchor("oracle_excl", t)
        if not np.isnan(van):
            ax.plot([x[i] - 0.5, x[i] + 0.5], [van, van], color="black", lw=1.4)
        if not np.isnan(orc):
            ax.plot([x[i] - 0.5, x[i] + 0.5], [orc, orc], color="black", lw=1.1, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([tlabel[t] for t in threats])
    ax.set_ylabel("test accuracy (3-seed mean)"); ax.set_ylim(0.20, 0.68)
    ax.set_title("F5 · CNN cifar10/dir1 P1 sign-gate online — score-source competition\n"
                 "solid=vanilla floor, dashed=oracle_excl ceiling · blue=estimator, "
                 "red=renorm, green=Flirds  (⚠ restack-drift caveat)", fontsize=8.5)
    ax.legend(ncol=8, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              columnspacing=0.9, handletextpad=0.4)
    return save(fig, "f5_cnn_competition_p1_online.png")


# ---------------------------------------------------------------------------
# F7 — cost: measured per-method wall-clock + (b)/Flirds exponential scaling
# ---------------------------------------------------------------------------
def f7():
    # panel A: silo5 (N=5) measured valuation wall-clock (3-seed mean)
    rt = {}
    for p in glob.glob(os.path.join(RUNS, "removal_dose", "rundirs",
                                    "1B_silo5_noisy_removal_seed*", "metrics.json")):
        d = inner(load_json(p)).get("runtime", {})
        for k, v in d.items():
            if k in EXCLUDE or not isinstance(v, (int, float)):
                continue
            rt.setdefault(k, []).append(v)
    if not rt:
        print("  [skip] F7: missing silo5 removal runtime"); return False
    # loss-heur omitted here: the removal_dose rundirs predate the C6 timing fix
    # (pre-fix ~165s vs post-fix canon ~99s, §5.5 caveat); every other method matches
    # canon, so this panel stays canonical without it. loss-heur cost = §5.5 op-count table.
    order = ["Flirds1st", "FedIF", "Flirds", "FedSV",
             "ComFedSV", "GTG", "ShapleyFL", "(b)oracle"]
    order = [m for m in order if m in rt]
    means = [np.mean(rt[m]) for m in order]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    cols = [C_FLIRDS if m == "Flirds" else ("black" if m == "(b)oracle"
            else (C_ESTIM if m in ESTIM else C_RENORM)) for m in order]
    axA.bar(range(len(order)), means, color=cols, edgecolor="black", linewidth=0.4)
    axA.set_yscale("log")
    for i, v in enumerate(means):
        axA.text(i, v * 1.08, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    axA.set_xticks(range(len(order))); axA.set_xticklabels(order, rotation=35, ha="right")
    axA.set_ylabel("valuation wall-clock (s, log)")
    axA.set_title("A · silo5 N=5 per-method cost (3-seed)\n"
                  "Flirds = 1 HVP/round (const in cohort)", fontsize=8.5)

    # panel B: (b) vs Flirds exponential blow-up across regimes
    regimes, b_vals, f_vals, ratios = [], [], [], []
    regimes.append("silo N=5\n(2^5 full)")
    b_vals.append(np.mean(rt.get("(b)oracle", [np.nan])))
    f_vals.append(np.mean(rt.get("Flirds", [np.nan])))
    e5 = os.path.join(RUNS, "track_d", "rundirs_e5_n10", "1B_anchor10_seed0", "metrics.json")
    if os.path.exists(e5):
        r = inner(load_json(e5)).get("runtime", {})
        regimes.append("full N=10\n(2^10)")
        b_vals.append(r.get("(b)oracle", np.nan)); f_vals.append(r.get("Flirds", np.nan))
    # device100 K=10 — analytic op-count model (runs/measured_2026-07/op_counts.py, device regime)
    regimes.append("device N=100\n(2^10/round, K=10)")
    b_vals.append(24975.0); f_vals.append(157.0)
    ratios = [b / f if f else np.nan for b, f in zip(b_vals, f_vals)]

    xb = np.arange(len(regimes)); bw = 0.38
    axB.bar(xb - bw / 2, b_vals, bw, label="(b) exact oracle", color="black")
    axB.bar(xb + bw / 2, f_vals, bw, label="Flirds", color=C_FLIRDS)
    axB.set_yscale("log")
    for i, (b, f, rr) in enumerate(zip(b_vals, f_vals, ratios)):
        axB.text(i - bw / 2, b * 1.1, f"{b:,.0f}", ha="center", va="bottom", fontsize=6.5)
        axB.text(i + bw / 2, f * 1.1, f"{f:,.0f}", ha="center", va="bottom", fontsize=6.5)
        if not np.isnan(rr):
            axB.text(i, np.sqrt(b * f), f"{rr:.0f}×", ha="center", va="center", fontsize=9,
                     fontweight="bold", color=C_RENORM,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_RENORM, lw=0.8))
    axB.set_xticks(xb); axB.set_xticklabels(regimes, fontsize=7.5)
    axB.set_ylabel("wall-clock (s, log)")
    axB.set_title("B · exact-oracle cost explodes with per-round cohort\n"
                  "Flirds stays ~constant (N=10=1-seed; device=op-count model)", fontsize=8.5)
    axB.legend(fontsize=8, loc="upper left")
    fig.suptitle("F7 · cost / scalability", fontsize=10, y=1.02)
    return save(fig, "f7_cost_scaling.png")


# ---------------------------------------------------------------------------
# F8 — removal curves (worst/best-first): LLM silo5 val-loss + CNN cifar10 acc
# ---------------------------------------------------------------------------
def _curve_mean(paths, key, method):
    """Average [step, y] curves over seeds -> (steps, worst[], best[])."""
    wf, bf = [], []
    for p in paths:
        rc = inner(load_json(p)).get(key, {})
        if method not in rc:
            continue
        wf.append([y for _, y in rc[method]["worst_first"]])
        bf.append([y for _, y in rc[method]["best_first"]])
    if not wf:
        return None
    wf, bf = np.array(wf), np.array(bf)
    return np.arange(wf.shape[1]), wf.mean(0), bf.mean(0)


def f8():
    llm = sorted(glob.glob(os.path.join(RUNS, "removal_dose", "rundirs",
                                        "1B_silo5_noisy_removal_seed*", "metrics.json")))
    cnn = sorted(glob.glob(os.path.join(RUNS, "removal_dose", "rundirs_cnn",
                                        "cifar10_label-flip_seed*", "metrics.json")))
    a = _curve_mean(llm, "removal_curve", "Flirds") if llm else None
    b = _curve_mean(cnn, "removal_curve_acc", "Flirds") if cnn else None
    if a is None and b is None:
        print("  [skip] F8: missing removal curves"); return False

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    if a is not None:
        x, wf, bf = a
        axes[0].plot(x, wf, "-o", color=C_FLIRDS, label="worst-first (remove low-φ first)")
        axes[0].plot(x, bf, "-s", color=C_RENORM, label="best-first (remove high-φ first)")
        axes[0].set_xlabel("# clients removed"); axes[0].set_ylabel("validation loss ↓")
        axes[0].set_title("A · LLM silo5 noisy — val-loss\n"
                          "worst-first lowers loss ⇒ ranking is causal", fontsize=8.5)
        axes[0].legend(fontsize=7.5)
    else:
        axes[0].set_axis_off(); axes[0].text(0.5, 0.5, "⬚ LLM removal\n(data not found)",
                                             ha="center", va="center")
    if b is not None:
        x, wf, bf = b
        axes[1].plot(x, wf, "-o", color=C_FLIRDS, label="worst-first")
        axes[1].plot(x, bf, "-s", color=C_RENORM, label="best-first")
        axes[1].set_xlabel("# clients removed"); axes[1].set_ylabel("test accuracy ↑")
        axes[1].set_title("B · CNN cifar10 label-flip — accuracy\n"
                          "worst-first stays higher (acc separation ≈ (b))", fontsize=8.5)
        axes[1].legend(fontsize=7.5)
    else:
        axes[1].set_axis_off(); axes[1].text(0.5, 0.5, "⬚ CNN removal\n(data not found)",
                                             ha="center", va="center")
    fig.suptitle("F8 · removal-curve causal check (Flirds ranking, 3-seed mean)",
                 fontsize=10, y=1.02)
    return save(fig, "f8_removal_curves.png")


# ---------------------------------------------------------------------------
# F9 — second-order ablation: Flirds vs Flirds-1st fidelity across participation k
# ---------------------------------------------------------------------------
def f9():
    dirs = sorted(glob.glob(os.path.join(RUNS, "probe_signal", "cnn_c1",
                                         "pc1_cifar10_label-flip_w*_k*_seed*")))
    if not dirs:
        print("  [skip] F9: missing probe_signal/cnn_c1 label-flip dirs"); return False
    ks = [0.2, 0.5, 1.0]
    data = {"Flirds": {k: [] for k in ks}, "Flirds1st": {k: [] for k in ks}}
    for d in dirs:
        mp = os.path.join(d, "metrics.json")
        if not os.path.exists(mp):
            continue
        base = os.path.basename(d)
        try:
            k = float(base.split("_k")[1].split("_seed")[0])
        except Exception:
            continue
        if k not in ks:
            continue
        m = load_json(mp)["methods"]
        truth = np.array(m["(b)oracle"]["phi"])
        for meth in ("Flirds", "Flirds1st"):
            if meth in m:
                rho = spearmanr(np.array(m[meth]["phi"]), truth).correlation
                if rho == rho:                                   # not NaN
                    data[meth][k].append(rho)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for meth, col, lbl in [("Flirds", C_FLIRDS, "Flirds (1st + 2nd order)"),
                           ("Flirds1st", C_RENORM, "Flirds-1st (1st order only)")]:
        mean = [np.mean(data[meth][k]) if data[meth][k] else np.nan for k in ks]
        std = [np.std(data[meth][k]) if data[meth][k] else 0 for k in ks]
        ax.errorbar(ks, mean, yerr=std, marker="o", color=col, capsize=3, label=lbl, lw=1.8)
        for kk, mm in zip(ks, mean):
            if mm == mm:
                ax.text(kk, mm + 0.03, f"{mm:.3f}", ha="center", fontsize=7.5, color=col)
    ax.set_xticks(ks); ax.set_xticklabels(["0.2 (2/10)", "0.5 (5/10)", "1.0 (full)"])
    ax.set_xlabel("per-round participation fraction k")
    ax.set_ylabel("Spearman vs (b) oracle"); ax.set_ylim(0, 1.08)
    ax.set_title("F9 · second-order (HVP) ablation — CNN C1 label-flip\n"
                 "the 2nd-order term rescues fidelity under partial participation "
                 "(width×seed pool)", fontsize=8.5)
    ax.legend(fontsize=8, loc="lower right")
    return save(fig, "f9_second_order_ksweep.png")


FIGS = [("F1", f1), ("F2", f2), ("F5", f5), ("F7", f7), ("F8", f8), ("F9", f9)]

if __name__ == "__main__":
    print(f"REPO = {REPO}")
    print(f"OUT  = {HERE}")
    print("⬚ (data not landed, intentionally skipped): F3 main-pair vs (b), "
          "F4 R4 EM, F6 detection AUROC\n")
    done, skipped = [], []
    for name, fn in FIGS:
        print(f"{name}:")
        try:
            (done if fn() else skipped).append(name)
        except Exception as e:
            skipped.append(name)
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
    print(f"\ngenerated: {done}\nskipped:   {skipped}")
    sys.exit(0 if done else 1)
