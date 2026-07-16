#!/usr/bin/env python
"""runs/track_c/make_figures.py -- committed figures for Track C (CNN standard setting).

Layout (README): c1/ = fidelity trajectories (N=10 full, 2 ds x 5 scenarios x 3 seeds,
11 methods incl Ripple); c1_oracle/ = (a) 2^10 retrain oracle per matching cell
(phi_a in metrics.json); c2/ = interventions (N=100 C=0.1 T=120, arms + strength
sweep + dismissal q-sweep).  Reads ONLY rundir artifacts and regenerates every PNG
under figures/; cross-checks against runs/track_c/fidelity.csv (merge_oracle_a.py
output) when that derived file is present.

Figure order = question hierarchy: fidelity (dual oracle) -> semantic ladder ->
cost -> detection | c2: outcome (performance) -> strength ladder -> dismissal ->
detection.

  python runs/track_c/make_figures.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent            # runs/track_c
FIG = HERE / "figures"

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 8.5,
                     "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})

DATASETS = ["mnist", "cifar10"]
SCENARIOS = ["iid", "label-flip", "label-skew", "feature-noise", "quantity-skew"]
SEEDS = [0, 1, 2]
METHOD_ORDER = ["Flirds", "Flirds1st", "FedIF", "GTG", "FedSV", "ShapleyFL",
                "Banzhaf", "ComFedSV", "loss-heur", "Ripple"]
METHOD_COLOR = {"(b)oracle": "#000000", "Flirds": "#0072B2", "Flirds1st": "#56B4E9",
                "GTG": "#E69F00", "FedSV": "#009E73", "ShapleyFL": "#D55E00",
                "Banzhaf": "#CC79A7", "FedIF": "#B8A000", "loss-heur": "#8C510A",
                "ComFedSV": "#4D9221", "Ripple": "#6A3D9A"}

C2_DS = ["cifar10", "fmnist"]
C2_PARTS = ["iid", "shard", "dir1"]
C2_THREATS = ["clean", "label_flip", "grad_noise", "free_rider"]
ARM_ORDER = ["vanilla", "flirds_mult", "flirds_repl", "flirds_add", "flirds_select",
             "shapleyfl", "fedif", "sfedavg"]
ARM_COLOR = {"vanilla": "#888888", "flirds_mult": "#0072B2", "flirds_repl": "#004166",
             "flirds_add": "#7FB3D3", "flirds_select": "#56B4E9",
             "shapleyfl": "#D55E00", "fedif": "#B8A000", "sfedavg": "#CC79A7"}


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


def annotated_heatmap(ax, mat, vmin, vmax, cmap, center=None, fmt="{:+.2f}", rot=45):
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
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=rot, ha="right" if rot else "center",
                  fontsize=6.5)
    ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=7)
    ax.grid(False)
    return im


# ------------------------------------------------------------------ load
def load_c1():
    cells, oracle, missing = {}, {}, []
    for ds in DATASETS:
        for scen in SCENARIOS:
            for seed in SEEDS:
                d = HERE / "c1" / f"{ds}_{scen}_seed{seed}"
                if (d / "metrics.json").exists():
                    cells[(ds, scen, seed)] = json.loads((d / "metrics.json").read_text())
                else:
                    missing.append(f"c1/{d.name}")
                do = HERE / "c1_oracle" / f"{ds}_{scen}_aonly_seed{seed}"
                if (do / "metrics.json").exists():
                    oracle[(ds, scen, seed)] = json.loads((do / "metrics.json").read_text())
                else:
                    missing.append(f"c1_oracle/{do.name}")
    return cells, oracle, missing


def load_c2():
    cells, missing = {}, []
    root = HERE / "c2"
    for d in sorted(root.iterdir()):
        if not (d / "metrics.json").exists():
            missing.append(f"c2/{d.name}")
            continue
        m = json.loads((d / "metrics.json").read_text())
        cells[(m["dataset"], m["partition"], m["threat"], str(m["strength"]), m["seed"])] = m
    return cells, missing


# ------------------------------------------------------------------ C1 figures
def fig_c1_fidelity(cells, oracle, path):
    cols = [f"{ds[:5]} {sc}" for ds in DATASETS for sc in SCENARIOS]
    panels = [("b", "Spearman vs (b) frozen-trajectory oracle"),
              ("a", "Spearman vs (a) 2^10 retrain oracle")]
    mats, arows = [], []
    for which, _ in panels:
        rows = {}
        for (ds, scen, seed), m in cells.items():
            lab = f"{ds[:5]} {scen}"
            for meth, blk in m["methods"].items():
                if meth == "(b)oracle" and which == "b":
                    continue
                if which == "b":
                    v = blk.get("spearman_b")
                else:
                    o = oracle.get((ds, scen, seed))
                    if o is None:
                        continue
                    # gt_a = -phi_a: retrain oracle stores good->LOW values; the runner
                    # (and merge_oracle_a.py) negate before comparing.  Same convention here.
                    v = float(spearmanr(np.asarray(blk["phi"], float),
                                        -np.asarray(o["phi_a"], float)).statistic)
                    arows.append(dict(dataset=ds, scenario=scen, seed=seed, method=meth,
                                      spearman_a=v))
                if v is not None:
                    rows.setdefault(meth, {}).setdefault(lab, []).append(v)
        methods = method_sort(rows)
        if which == "a" and "(b)oracle" in rows:              # (b) itself scored against (a)
            methods = ["(b)oracle"] + [m for m in methods if m != "(b)oracle"]
        mat = pd.DataFrame(index=methods, columns=cols, dtype=float)
        for meth, d in rows.items():
            for lab, vs in d.items():
                mat.at[meth, lab] = np.mean(vs)
        mats.append(mat)
    fig, axes = plt.subplots(2, 1, figsize=(0.62 * len(cols) + 2.8,
                                            0.30 * sum(len(m) for m in mats) + 4.0),
                             squeeze=False, gridspec_kw={"hspace": 0.42})
    for ax, mat, (_, ttl) in zip(axes[:, 0], mats, panels):
        im = annotated_heatmap(ax, mat, -1, 1, "RdBu", center=0.0)
        ax.axvline(len(SCENARIOS) - 0.5, color="black", lw=1.0)
        ax.set_title(ttl)
    fig.colorbar(im, ax=axes[:, 0], shrink=0.6, label="Spearman (mean over 3 seeds)")
    fig.suptitle("C1 fidelity, dual oracle -- CNN N=10 full participation", fontsize=10.5,
                 y=0.96)
    footnote(fig, "source: c1/*/metrics.json methods.<M>.{spearman_b,phi} + c1_oracle/*_aonly_*/metrics.json phi_a "
                  "(vs-(a) recomputed from phi lists) | left=MNIST block, right=CIFAR-10")
    save(fig, path)
    return pd.DataFrame(arows)


def fig_c1_ladder(cells, path):
    scens = [s for s in SCENARIOS if s != "iid"]              # iid: rates constant -> undefined
    cols = [f"{ds[:5]} {sc}" for ds in DATASETS for sc in scens]
    rows = {}
    for (ds, scen, seed), m in cells.items():
        if scen == "iid":
            continue
        lab = f"{ds[:5]} {scen}"
        for meth, blk in m["methods"].items():
            v = blk.get("spearman_vs_rate")
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                rows.setdefault(meth, {}).setdefault(lab, []).append(v)
    methods = method_sort(rows)
    mat = pd.DataFrame(index=methods, columns=cols, dtype=float)
    for meth, d in rows.items():
        for lab, vs in d.items():
            mat.at[meth, lab] = np.mean(vs)
    fig, ax = plt.subplots(figsize=(0.62 * len(cols) + 2.6, 0.3 * len(methods) + 1.9))
    im = annotated_heatmap(ax, -mat, -1, 1, "RdBu", center=0.0)
    ax.axvline(len(scens) - 0.5, color="black", lw=1.0)
    ax.set_title("C1 semantic ladder -- does phi DECREASE with per-client corruption rate?\n"
                 "shown as -Spearman(phi, rate): +1 = perfectly monotone 'more corrupt = less value'")
    fig.colorbar(im, ax=ax, shrink=0.85, label="-Spearman(phi, corruption rate)")
    footnote(fig, "source: c1/*/metrics.json methods.<M>.spearman_vs_rate (native), sign-flipped for display | "
                  "iid excluded (all rates 0)")
    save(fig, path)


def fig_c1_cost(cells, oracle, path):
    rows = []
    for (ds, scen, seed), m in cells.items():
        for meth, blk in m["methods"].items():
            rows.append(dict(dataset=ds, method=meth, runtime=blk["runtime"]))
        rows.append(dict(dataset=ds, method="traj (shared)", runtime=m["traj_time"]))
    for (ds, scen, seed), o in oracle.items():
        rows.append(dict(dataset=ds, method="(a) 2^10 retrain", runtime=o["t_a"]))
    df = pd.DataFrame(rows)
    methods = method_sort([m for m in df["method"].unique()
                           if m not in ("traj (shared)", "(a) 2^10 retrain", "(b)oracle")])
    methods = ["(a) 2^10 retrain", "traj (shared)", "(b)oracle"] + methods
    methods = [m for m in methods if m in set(df["method"])][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(methods) + 1.6))
    for i, meth in enumerate(methods):
        for ds, col in (("mnist", "#0072B2"), ("cifar10", "#D55E00")):
            v = df[(df["method"] == meth) & (df["dataset"] == ds)]["runtime"].astype(float)
            if not len(v):
                continue
            ax.scatter(v, np.full(len(v), i) + (-0.12 if ds == "mnist" else 0.12),
                       s=9, color=col, alpha=0.55, linewidths=0)
            ax.plot([v.median()] * 2, [i - 0.22, i + 0.22], color=col, lw=1.6)
    ax.set_xscale("log")
    ax.set_yticks(range(len(methods)), methods, fontsize=7.5)
    ax.set_xlabel("wall-clock [s, log] per (ds, scenario, seed)")
    ax.set_title("C1 cost -- valuation wall-clock vs the shared trajectory and the (a) retrain oracle")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#0072B2", label="MNIST"),
                       plt.Line2D([], [], marker="o", ls="", color="#D55E00", label="CIFAR-10")],
              fontsize=7.5, loc="lower right")
    footnote(fig, "source: c1/*/metrics.json methods.<M>.runtime + traj_time; c1_oracle/*/metrics.json t_a | dots=cells, bar=median")
    save(fig, path)


def fig_c1_auroc(cells, path):
    scens = [s for s in SCENARIOS if s != "iid"]
    cols = [f"{ds[:5]} {sc}" for ds in DATASETS for sc in scens]
    rows = {}
    for (ds, scen, seed), m in cells.items():
        if scen == "iid":
            continue
        lab = f"{ds[:5]} {scen}"
        for meth, blk in m["methods"].items():
            v = blk.get("auroc")
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                rows.setdefault(meth, {}).setdefault(lab, []).append(v)
    methods = method_sort(rows)
    mat = pd.DataFrame(index=methods, columns=cols, dtype=float)
    for meth, d in rows.items():
        for lab, vs in d.items():
            mat.at[meth, lab] = np.mean(vs)
    fig, ax = plt.subplots(figsize=(0.62 * len(cols) + 2.6, 0.3 * len(methods) + 1.9))
    im = annotated_heatmap(ax, mat, 0, 1, "RdBu", center=0.5, fmt="{:.2f}")
    ax.axvline(len(scens) - 0.5, color="black", lw=1.0)
    ax.set_title("C1 detection AUROC (3rd-tier axis) -- corrupt-client separation per method")
    fig.colorbar(im, ax=ax, shrink=0.85, label="AUROC (mean over 3 seeds; 0.5=chance)")
    footnote(fig, "source: c1/*/metrics.json methods.<M>.auroc | iid excluded (no corrupt clients)")
    save(fig, path)


# ------------------------------------------------------------------ C2 figures
def fig_c2_outcome(cells, path):
    cols, delta = [], {}
    for ds in C2_DS:
        for part in C2_PARTS:
            for threat in C2_THREATS:
                lab = f"{ds[:4]} {part} {threat.replace('_', '-')}"
                cols.append(lab)
                for arm in ARM_ORDER:
                    if arm == "vanilla":
                        continue
                    ds_ = []
                    for seed in SEEDS:
                        m = cells.get((ds, part, threat, "main", seed))
                        if m and arm in m["arms"]:
                            ds_.append(m["arms"][arm]["final_acc"]
                                       - m["arms"]["vanilla"]["final_acc"])
                    if ds_:
                        delta.setdefault(arm, {})[lab] = np.mean(ds_)
    arms = [a for a in ARM_ORDER if a in delta]
    mat = pd.DataFrame(index=arms, columns=cols, dtype=float)
    for arm, d in delta.items():
        for lab, v in d.items():
            mat.at[arm, lab] = v
    lim = float(np.nanmax(np.abs(mat.to_numpy()))) or 0.01
    fig, ax = plt.subplots(figsize=(0.48 * len(cols) + 2.8, 0.34 * len(arms) + 2.0))
    im = annotated_heatmap(ax, mat, -lim, lim, "RdBu", center=0.0, fmt="{:+.3f}", rot=90)
    for x in (len(C2_THREATS) - 0.5 + i * len(C2_THREATS) for i in range(len(C2_PARTS) * len(C2_DS) - 1)):
        ax.axvline(x, color="black", lw=0.6)
    ax.axvline(len(C2_PARTS) * len(C2_THREATS) - 0.5, color="black", lw=1.4)
    ax.set_title("C2 intervention outcome (2nd tier) -- final-accuracy delta vs vanilla FedAvg, strength=main\n"
                 "left block=CIFAR-10, right=FMNIST | mean over 3 seeds")
    fig.colorbar(im, ax=ax, shrink=0.85, label="final_acc(arm) - final_acc(vanilla)")
    footnote(fig, "source: c2/*_strmain_*/metrics.json arms.<arm>.final_acc")
    save(fig, path)


def fig_c2_strength(cells, path):
    sweeps = [("label_flip", ["0.6", "0.8", "main"]), ("grad_noise", ["0.05", "main"])]
    fig, axes = plt.subplots(len(C2_DS), len(sweeps), figsize=(9.6, 6.0), squeeze=False)
    for di, ds in enumerate(C2_DS):
        for si, (threat, strengths) in enumerate(sweeps):
            ax = axes[di][si]
            arms_here = sorted({a for s in strengths for seed in SEEDS
                                for m in [cells.get((ds, "dir1", threat, s, seed))] if m
                                for a in m["arms"]},
                               key=lambda a: ARM_ORDER.index(a) if a in ARM_ORDER else 99)
            for arm in arms_here:
                ys = []
                for s in strengths:
                    vs = [cells[(ds, "dir1", threat, s, seed)]["arms"][arm]["final_acc"]
                          for seed in SEEDS if (ds, "dir1", threat, s, seed) in cells
                          and arm in cells[(ds, "dir1", threat, s, seed)]["arms"]]
                    ys.append(np.mean(vs) if vs else np.nan)
                ax.plot(range(len(strengths)), ys, marker="o", ms=3.5, lw=1.2,
                        color=ARM_COLOR.get(arm, "#BBBBBB"),
                        label=arm if (di == 0 and si == 0) else None)
            ax.set_xticks(range(len(strengths)), strengths)
            ax.set_title(f"{ds} dir1 {threat.replace('_', '-')}", fontsize=9)
            if si == 0:
                ax.set_ylabel("final test accuracy")
            if di == len(C2_DS) - 1:
                ax.set_xlabel("threat strength")
    fig.legend(loc="upper center", ncol=8, fontsize=7.5, bbox_to_anchor=(0.5, 1.04),
               frameon=False)
    fig.suptitle("C2 strength ladder (dir1) -- arm outcome as the threat intensifies\n"
                 "(flirds_repl / flirds_add arms exist only in these sweep cells)",
                 fontsize=10, y=1.10)
    fig.tight_layout()
    footnote(fig, "source: c2/*_dir1_{label-flip,grad-noise}_str*/metrics.json arms.<arm>.final_acc, mean over seeds")
    save(fig, path)


def fig_c2_dismissal(cells, path):
    """dismissal blocks exist only in cifar10 dir1 strength=main cells (all 4 threats)."""
    threat_col = {"clean": "#888888", "label_flip": "#D55E00",
                  "grad_noise": "#E69F00", "free_rider": "#0072B2"}
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    drawn = 0
    for threat in C2_THREATS:
        curves = []
        for seed in SEEDS:
            m = cells.get(("cifar10", "dir1", threat, "main", seed))
            if m and m.get("dismissal"):
                qs = sorted(m["dismissal"], key=float)
                curves.append(([float(q) for q in qs], [m["dismissal"][q] for q in qs]))
        if not curves:
            continue
        drawn += len(curves)
        col = threat_col[threat]
        mean = np.mean([c[1] for c in curves], axis=0)
        for c in curves:
            ax.plot(c[0], c[1], color=col, lw=0.6, alpha=0.3)
        ax.plot(curves[0][0], mean, color=col, lw=1.8, marker="o", ms=3,
                label=threat.replace("_", "-"))
    ax.set_xlabel("dismissal fraction q (lowest-phi clients dropped)")
    ax.set_ylabel("final test accuracy")
    ax.set_title("C2 dismissal q-sweep (CIFAR-10 dir1, strength=main) -- dropping low-phi clients:\n"
                 "harmless under clean, helpful under threat?")
    ax.legend(fontsize=8, title="threat")
    footnote(fig, "source: c2/cifar10_dir1_*_strmain_*/metrics.json dismissal{q: final_acc} | "
                  "thin lines=seeds, thick=mean | fmnist cells persist no dismissal block")
    save(fig, path)
    return drawn


def fig_c2_auroc(cells, path):
    threats = [t for t in C2_THREATS if t != "clean"]
    cols = [f"{ds[:4]} {p} {t.replace('_', '-')}" for ds in C2_DS for p in C2_PARTS for t in threats]
    rows = {}
    for (ds, part, threat, s, seed), m in cells.items():
        if s != "main" or threat == "clean":
            continue
        lab = f"{ds[:4]} {part} {threat.replace('_', '-')}"
        for arm, blk in m["arms"].items():
            v = blk.get("auroc")
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                rows.setdefault(arm, {}).setdefault(lab, []).append(v)
    arms = [a for a in ARM_ORDER if a in rows] + sorted(set(rows) - set(ARM_ORDER))
    mat = pd.DataFrame(index=arms, columns=cols, dtype=float)
    for arm, d in rows.items():
        for lab, vs in d.items():
            mat.at[arm, lab] = np.mean(vs)
    fig, ax = plt.subplots(figsize=(0.5 * len(cols) + 2.8, 0.34 * len(arms) + 2.0))
    im = annotated_heatmap(ax, mat, 0, 1, "RdBu", center=0.5, fmt="{:.2f}", rot=90)
    ax.axvline(len(C2_PARTS) * len(threats) - 0.5, color="black", lw=1.4)
    ax.set_title("C2 detection AUROC by arm (3rd-tier axis; strength=main)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="AUROC (mean over 3 seeds)")
    footnote(fig, "source: c2/*_strmain_*/metrics.json arms.<arm>.auroc | clean cells excluded")
    save(fig, path)


# ------------------------------------------------------------------ main
def main():
    c1, oracle, c1_missing = load_c1()
    c2, c2_missing = load_c2()
    print("== coverage ==")
    print(f"  c1 trajectory cells : {len(c1)}/30 (2 ds x 5 scenarios x 3 seeds)")
    print(f"  c1_oracle (a) cells : {len(oracle)}/30")
    print(f"  c2 intervention     : {len(c2)}/90 "
          f"({len({k[:4] for k in c2})} (ds,part,threat,str) groups x 3 seeds)")
    for miss in (c1_missing, c2_missing):
        if miss:
            print(f"    MISSING: {', '.join(miss)}")
    n_dis = sum(1 for m in c2.values() if m.get("dismissal"))
    print(f"  c2 dismissal q-sweep present in {n_dis} cells (cifar10 dir1 strmain, all threats)")

    print("== figures ==")
    arows = fig_c1_fidelity(c1, oracle, FIG / "01_c1_fidelity_dual_oracle.png")
    fig_c1_ladder(c1, FIG / "02_c1_semantic_ladder.png")
    fig_c1_cost(c1, oracle, FIG / "03_c1_cost_runtime.png")
    fig_c1_auroc(c1, FIG / "04_c1_detection_auroc.png")
    fig_c2_outcome(c2, FIG / "05_c2_outcome_delta_grid.png")
    fig_c2_strength(c2, FIG / "06_c2_strength_ladder.png")
    n_drawn = fig_c2_dismissal(c2, FIG / "07_c2_dismissal_qsweep.png")
    print(f"  dismissal curves drawn: {n_drawn}")
    fig_c2_auroc(c2, FIG / "08_c2_detection_auroc.png")

    # cross-check vs merge_oracle_a.py output when present (independent code path)
    fid = HERE / "fidelity.csv"
    if fid.exists() and len(arows):
        ref = pd.read_csv(fid)
        sp_col = next((c for c in ref.columns if "spearman" in c and "_a" in c), None)
        if sp_col and {"dataset", "scenario", "seed", "method"} <= set(ref.columns):
            # fidelity.csv stores scenario with underscores (runner field); dirs use hyphens
            ours = arows.copy()
            ours["scenario"] = ours["scenario"].str.replace("-", "_")
            mref = ref.set_index(["dataset", "scenario", "seed", "method"])[sp_col]
            ours = ours.set_index(["dataset", "scenario", "seed", "method"])["spearman_a"]
            both = mref.index.intersection(ours.index)
            bad = [i for i in both if abs(mref[i] - ours[i]) > 1e-9]
            print(f"  (a)-fidelity cross-check vs fidelity.csv: {len(both) - len(bad)}/{len(both)} match"
                  + (f" MISMATCH: {bad[:4]}" if bad else ""))
        else:
            print(f"  (fidelity.csv present but schema unmatched -- skipped cross-check; cols={list(ref.columns)})")

    FIG.mkdir(exist_ok=True)
    arows.to_csv(FIG / "c1_fidelity_vs_a.csv", index=False)
    print(f"  wrote figures/c1_fidelity_vs_a.csv ({len(arows)} rows)")


if __name__ == "__main__":
    main()
