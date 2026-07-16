#!/usr/bin/env python
"""runs/matrix_cxni/make_figures.py -- B-axis (corruption x non-IID) matrix figures.

Signal-realness experiment (plan: research-wiki/wiki/flirds-signal-size-diagnosis.md
section 2.4): does the cross-client signal come from actual client differences
(domain heterogeneity / corruption) rather than training strength?

Rundirs live under ../phase2_matrix/rundirs/ by design (README here): 6 new cells
(1B_iid5_* + 1B_silo5_clean) + the 4 existing June silo5 threat cells reused for
the non-IID row.  This script reads ONLY those rundir artifacts and regenerates
every PNG under figures/.

  python runs/matrix_cxni/make_figures.py
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

HERE = Path(__file__).resolve().parent            # runs/matrix_cxni
RUNDIRS = HERE.parent / "phase2_matrix" / "rundirs"
FIG = HERE / "figures"

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 8.5,
                     "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})

# design of the 2x5 matrix (structural cell registry, mirrors README table)
STAGES = [("iid5", "IID"), ("silo5", "non-IID")]
THREATS = ["clean", "noisy", "freerider_random", "freerider_zero", "poison"]
THREAT_SHORT = {"clean": "clean", "noisy": "noisy", "freerider_random": "fr-rand",
                "freerider_zero": "fr-zero", "poison": "poison"}
NAME_TOKEN = {"clean": "clean", "noisy": "noisy", "freerider_random": "frrand",
              "freerider_zero": "frzero", "poison": "poison"}
REUSED_JUNE = {"1B_silo5_noisy", "1B_silo5_frrand", "1B_silo5_frzero", "1B_silo5_poison"}

METHOD_ORDER = ["(b)oracle", "Flirds", "Flirds1st", "FedIF", "GTG", "FedSV", "ShapleyFL",
                "Banzhaf", "ComFedSV", "loss-heur", "FLDetector", "STD-DAGMM", "FLTrust", "FedDQC"]
DET_METHODS = {"FLDetector", "STD-DAGMM", "FLTrust", "FedDQC"}
STAGE_COLOR = {"iid5": "#E69F00", "silo5": "#0072B2"}          # fixed identity pair (CVD-safe)


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
    for stage, _ in STAGES:
        for threat in THREATS:
            name = f"1B_{stage}_{NAME_TOKEN[threat]}"
            d = RUNDIRS / name
            if not (d / "metrics.json").exists():
                missing.append(name)
                continue
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            cells[(stage, threat)] = dict(
                name=name, stage=stage, threat=threat, cfg=cfg,
                metrics=json.loads((d / "metrics.json").read_text()),
                phi=pd.read_parquet(d / "phi.parquet"))
    return cells, missing


def crossseed_rho(phi, method):
    """mean/pairs of pairwise Spearman between the method's own phi vectors across seeds."""
    g = phi[phi["method"] == method]
    vecs = {s: gs.set_index("client")["phi"].sort_index() for s, gs in g.groupby("seed")}
    pairs = []
    for a, b in combinations(sorted(vecs), 2):
        va, vb = vecs[a].align(vecs[b], join="inner")
        r = spearmanr(va, vb).statistic
        if not np.isnan(r):
            pairs.append(float(r))
    return pairs


# ------------------------------------------------------------------ figures
def fig_oracle_crossseed(cells, path):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    xs = np.arange(len(THREATS))
    w = 0.36
    for si, (stage, slab) in enumerate(STAGES):
        means, allpairs = [], []
        for threat in THREATS:
            c = cells.get((stage, threat))
            pairs = crossseed_rho(c["phi"], "(b)oracle") if c else []
            means.append(np.mean(pairs) if pairs else np.nan)
            allpairs.append(pairs)
        pos = xs + (si - 0.5) * w
        ax.bar(pos, means, w * 0.92, color=STAGE_COLOR[stage], label=f"{slab} ({stage})",
               alpha=0.85)
        for x, pairs in zip(pos, allpairs):
            ax.scatter([x] * len(pairs), pairs, s=12, color="black", zorder=3, alpha=0.7,
                       linewidths=0)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(xs, [THREAT_SHORT[t] for t in THREATS])
    ax.set_ylabel("cross-seed Spearman of (b)-oracle phi\n(mean of 3 seed-pairs; dots=pairs)")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Is there a real cross-client signal? -- exact-oracle self-ranking stability across seeds\n"
                 "(clean columns: corruption-free, so any signal is pure domain heterogeneity)")
    footnote(fig, f"source: {RUNDIRS.relative_to(HERE.parent)}/1B_{{iid5,silo5}}_*/phi.parquet, method=(b)oracle | N=5, exact 2^5 oracle")
    save(fig, path)


def fig_crossseed_by_method(cells, path):
    methods = None
    cols, data = [], {}
    for stage, slab in STAGES:
        for threat in THREATS:
            c = cells.get((stage, threat))
            lab = f"{slab} {THREAT_SHORT[threat]}"
            cols.append(lab)
            if not c:
                continue
            vals = {}
            for m in c["phi"]["method"].unique():
                if c["phi"][c["phi"].method == m]["kind"].iloc[0] != "val":
                    continue
                pairs = crossseed_rho(c["phi"], m)
                if pairs:
                    vals[m] = np.mean(pairs)
            data[lab] = vals
    methods = method_sort({m for v in data.values() for m in v})
    mat = pd.DataFrame(index=methods, columns=cols, dtype=float)
    for lab, vals in data.items():
        for m, v in vals.items():
            mat.at[m, lab] = v
    fig, ax = plt.subplots(figsize=(0.62 * len(cols) + 2.6, 0.3 * len(methods) + 1.8))
    im = annotated_heatmap(ax, mat, -1, 1, "RdBu", center=0.0)
    ax.axvline(len(THREATS) - 0.5, color="black", lw=1.0)
    ax.set_title("Cross-seed rank stability of each method's own phi (mean pairwise Spearman)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="cross-seed Spearman")
    footnote(fig, "source: phi.parquet per cell | left block=IID, right=non-IID | val methods only")
    save(fig, path)


def pearson_from_phi(c):
    """Pearson vs (b)oracle recomputed from phi.parquet -- single uniform code path;
    the reused June-era silo5 cells predate the native metrics.pearson field."""
    out = {}
    for (threat, seed), g in c["phi"].groupby(["threat", "seed"]):
        vecs = {m: gm.set_index("client")["phi"] for m, gm in g.groupby("method")}
        kinds = dict(zip(g["method"], g["kind"]))
        if "(b)oracle" not in vecs:
            continue
        tv = vecs["(b)oracle"]
        for m, mv in vecs.items():
            if m == "(b)oracle" or kinds.get(m) != "val":
                continue
            cl = mv.index.intersection(tv.index)
            a, b = mv.loc[cl].to_numpy(float), tv.loc[cl].to_numpy(float)
            out[(int(seed), m)] = (float(np.corrcoef(a, b)[0, 1])
                                   if a.std() and b.std() else np.nan)
    return out


def fig_fidelity(cells, path):
    cols = [f"{slab} {THREAT_SHORT[t]}" for s, slab in STAGES for t in THREATS]
    panels = [("spearman", "Spearman vs (b) exact oracle", "{:+.2f}"),
              ("pearson", "Pearson vs (b) exact oracle (value-level)", "{:+.2f}")]
    mats = []
    n_check = n_mismatch = 0
    for key, _, _ in panels:
        rows = {}
        for (stage, threat), c in cells.items():
            slab = dict(STAGES)[stage]
            lab = f"{slab} {THREAT_SHORT[threat]}"
            if key == "pearson":
                recomputed = pearson_from_phi(c)
                for (seed, m), v in recomputed.items():
                    rows.setdefault(m, {}).setdefault(lab, []).append(v)
                # self-check: where the runner persisted a native pearson it must
                # agree with the phi recomputation
                for skey, res in c["metrics"].items():
                    seed = int(skey.rsplit("_seed", 1)[1])
                    for m, v in res.get("pearson", {}).items():
                        r = recomputed.get((seed, m))
                        if v is None or r is None or np.isnan(v) or np.isnan(r):
                            continue
                        n_check += 1
                        if abs(v - r) > 1e-6:
                            n_mismatch += 1
                            print(f"  [WARN] pearson native!=recomputed {c['name']} seed{seed} {m}: {v} vs {r}")
            else:
                for skey, res in c["metrics"].items():
                    for m, v in res.get(key, {}).items():
                        rows.setdefault(m, {}).setdefault(lab, []).append(v)
        methods = method_sort(rows)
        mat = pd.DataFrame(index=methods, columns=cols, dtype=float)
        for m, d in rows.items():
            for lab, vs in d.items():
                vs = [v for v in vs if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if vs:
                    mat.at[m, lab] = np.mean(vs)
        mats.append(mat)
    print(f"  pearson self-check: {n_check - n_mismatch}/{n_check} native values match phi recomputation")
    fig, axes = plt.subplots(1, 2, figsize=(2 * (0.62 * len(cols)) + 4.4,
                                            0.3 * max(len(m) for m in mats) + 2.0))
    for ax, mat, (key, ttl, fmt) in zip(axes, mats, panels):
        im = annotated_heatmap(ax, mat, -1, 1, "RdBu", center=0.0, fmt=fmt)
        ax.axvline(len(THREATS) - 0.5, color="black", lw=1.0)
        ax.set_title(ttl)
    fig.colorbar(im, ax=axes, shrink=0.8, label="correlation vs (b) oracle")
    fig.suptitle("Estimator fidelity on the corruption x non-IID matrix (mean over 3 seeds)",
                 fontsize=10.5, y=1.03)
    footnote(fig, "source: metrics.json spearman (native) + pearson recomputed from phi.parquet vs (b)oracle "
                  "(checked == native where persisted) | left block=IID, right=non-IID")
    save(fig, path)


def fig_auroc(cells, path):
    threats = [t for t in THREATS if t != "clean"]                # clean: no corrupt clients
    cols = [f"{slab} {THREAT_SHORT[t]}" for s, slab in STAGES for t in threats]
    rows = {}
    for (stage, threat), c in cells.items():
        if threat == "clean":
            continue
        slab = dict(STAGES)[stage]
        lab = f"{slab} {THREAT_SHORT[threat]}"
        for skey, res in c["metrics"].items():
            for m, v in res.get("auroc", {}).items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    rows.setdefault(m, {}).setdefault(lab, []).append(v)
    methods = method_sort(rows)
    mat = pd.DataFrame(index=methods, columns=cols, dtype=float)
    for m, d in rows.items():
        for lab, vs in d.items():
            mat.at[m, lab] = np.mean(vs)
    fig, ax = plt.subplots(figsize=(0.62 * len(cols) + 2.8, 0.3 * len(methods) + 1.9))
    im = annotated_heatmap(ax, mat, 0, 1, "RdBu", center=0.5, fmt="{:.2f}")
    ax.axvline(len(threats) - 0.5, color="black", lw=1.0)
    nval = sum(m not in DET_METHODS for m in methods)
    ax.axhline(nval - 0.5, color="black", lw=1.0)
    ax.set_title("Detection AUROC: does domain heterogeneity help or hurt? (IID block vs non-IID block)\n"
                 "clean cells excluded (no corrupt client -> AUROC undefined)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="AUROC (mean over 3 seeds; 0.5=chance)")
    footnote(fig, "source: metrics.json auroc | below hline = model-side detectors")
    save(fig, path)


# ------------------------------------------------------------------ main
def main():
    cells, missing = load_cells()
    print("== coverage ==")
    print(f"  matrix cells found: {len(cells)}/10 "
          f"({len(REUSED_JUNE & {c['name'] for c in cells.values()})} reused from June silo5 grid)")
    for (stage, threat), c in sorted(cells.items()):
        seeds = sorted({int(k.rsplit('_seed', 1)[1]) for k in c["metrics"]})
        print(f"    {c['name']:22s} seeds={seeds}")
    if missing:
        print(f"    MISSING: {', '.join(missing)}")

    print("== figures ==")
    fig_oracle_crossseed(cells, FIG / "01_fidelity_oracle_crossseed_rho.png")
    fig_crossseed_by_method(cells, FIG / "02_crossseed_rho_by_method.png")
    fig_fidelity(cells, FIG / "03_fidelity_vs_oracle_heatmap.png")
    fig_auroc(cells, FIG / "04_detection_auroc_matrix.png")

    # exact figure inputs for the overview session
    FIG.mkdir(exist_ok=True)
    out = []
    for (stage, threat), c in cells.items():
        for m in c["phi"]["method"].unique():
            pairs = crossseed_rho(c["phi"], m)
            if pairs:
                out.append(dict(cell=c["name"], stage=stage, threat=threat, method=m,
                                crossseed_rho_mean=np.mean(pairs), n_pairs=len(pairs)))
    pd.DataFrame(out).to_csv(FIG / "crossseed_rho.csv", index=False)
    print(f"  wrote figures/crossseed_rho.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
