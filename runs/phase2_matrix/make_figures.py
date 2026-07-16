#!/usr/bin/env python
"""runs/phase2_matrix/make_figures.py -- committed headline figures for the phase-2 grid.

Regenerates every PNG under figures/ (plus small summary CSVs) from rundir
artifacts ONLY (config.yaml / metrics.json / phi.parquet).  No number in any
figure comes from a document -- rundirs are the canon.

Campaign roots (auto-discovered; late/deferred cells appear on re-run):
  rundirs/            June 2026 campaign (25 corruption-grid cells; the 6 B-axis
                      matrix cells that also live here -- iid5 regime / clean
                      threat -- are charted by runs/matrix_cxni/make_figures.py)
  rundirs_2026-07/    July 2026 post-server-migration re-run campaign
                      (same cell names; 3 device100 anchor cells deferred)

Figure order follows the question hierarchy: fidelity (primary) -> cost ->
detection (last).  phase-2 metrics carry no model-performance axis, so the
2nd-tier performance/convergence block has no figure here (see track_d).

  python runs/phase2_matrix/make_figures.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent            # runs/phase2_matrix
CAMPAIGNS = {"June": ROOT / "rundirs", "July": ROOT / "rundirs_2026-07"}
FIG = ROOT / "figures"

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 120, "font.size": 8.5,
                     "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})

CAT_ORDER = ["01_silo5", "02_device100_sweep", "03_device100_poison",
             "04_device100_anchor", "05_scale_3b"]
THREAT_ORDER = ["noisy", "freerider_random", "freerider_zero", "poison"]
THREAT_SHORT = {"noisy": "noisy", "freerider_random": "fr-rand",
                "freerider_zero": "fr-zero", "poison": "poison"}
METHOD_ORDER = ["(b)oracle", "Flirds", "Flirds1st", "FedIF", "GTG", "FedSV", "ShapleyFL",
                "Banzhaf", "ComFedSV", "loss-heur", "FLDetector", "STD-DAGMM", "FLTrust", "FedDQC"]
DET_METHODS = {"FLDetector", "STD-DAGMM", "FLTrust", "FedDQC"}
# fixed identity colors (Okabe-Ito CVD-safe core + gray family for detectors);
# color follows the method, never its rank -- shared convention across runs/*/make_figures.py
METHOD_COLOR = {"(b)oracle": "#000000", "Flirds": "#0072B2", "Flirds1st": "#56B4E9",
                "GTG": "#E69F00", "FedSV": "#009E73", "ShapleyFL": "#D55E00",
                "Banzhaf": "#CC79A7", "FedIF": "#B8A000", "loss-heur": "#8C510A",
                "ComFedSV": "#4D9221", "Ripple": "#6A3D9A",
                "FLDetector": "#555555", "FLTrust": "#888888",
                "FedDQC": "#333333", "STD-DAGMM": "#AAAAAA"}
CAMP_STYLE = {"June": "-", "July": "--"}


def classify(cfg):
    if cfg["regime"] == "silo5":
        return "01_silo5" if cfg.get("scale", "1B") == "1B" else "05_scale_3b"
    if "poison" in cfg["threats"]:
        return "03_device100_poison"
    return "04_device100_anchor" if cfg.get("oracle_b") else "02_device100_sweep"


def ref_label(cfg):
    """what the runner's spearman/pearson were computed against (mirrors runner)."""
    if not cfg.get("oracle_b"):
        return "Flirds(proxy)"
    return "(b)perround" if cfg["regime"] == "device100" else "(b)oracle"


def method_sort(ms):
    ms = list(ms)
    return [m for m in METHOD_ORDER if m in ms] + sorted(set(ms) - set(METHOD_ORDER))


# ------------------------------------------------------------------ load
def load_campaign(name, root):
    cells, skipped = [], []
    if not root.is_dir():
        return cells, skipped
    for d in sorted(root.iterdir()):
        if not (d / "metrics.json").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        # B-axis (signal-size diagnosis) cells: iid5 regime / clean threat.
        # Out of this report's corruption-grid taxonomy -> matrix_cxni figures.
        if cfg["regime"] not in ("silo5", "device100") or cfg["threats"][0] not in THREAT_ORDER:
            skipped.append(d.name)
            continue
        cells.append(dict(campaign=name, cell=d.name, category=classify(cfg), cfg=cfg,
                          metrics=json.loads((d / "metrics.json").read_text()),
                          phi=pd.read_parquet(d / "phi.parquet")))
    return cells, skipped


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def pearson_backfill(cell):
    """value-level fidelity from phi.parquet for (threat,seed,method) tuples whose
    metrics.json predates the native pearson field (June-era cells).  Truth vector
    mirrors the runner: '(b)oracle' when persisted, else Flirds proxy."""
    out = {}
    for (threat, seed), g in cell["phi"].groupby(["threat", "seed"], sort=False):
        vecs = {m: gm.set_index("client")["phi"] for m, gm in g.groupby("method")}
        kinds = dict(zip(g["method"], g["kind"]))
        truth = "(b)oracle" if "(b)oracle" in vecs else ("Flirds" if "Flirds" in vecs else None)
        if truth is None:
            continue
        tv = vecs[truth]
        for m, mv in vecs.items():
            if m == truth or kinds.get(m) != "val":
                continue
            cl = mv.index.intersection(tv.index)
            out[(threat, int(seed), m)] = _pearson(mv.loc[cl], tv.loc[cl])
    return out


def build_frame(cells):
    rows = []
    for c in cells:
        cfg = c["cfg"]
        backfill, backfill_used = None, False
        base = dict(campaign=c["campaign"], cell=c["cell"], category=c["category"],
                    regime=cfg["regime"], scale=cfg.get("scale", "1B"),
                    alpha=cfg["alpha"] if cfg["regime"] == "device100" else np.nan,
                    ref=ref_label(cfg))
        for key, res in c["metrics"].items():
            threat, seed = key.rsplit("_seed", 1)
            seed = int(seed)
            names = set(res["auroc"]) | set(res["spearman"]) | set(res["runtime"])
            for m in names:
                pe = res.get("pearson", {}).get(m)
                if pe is None and m in res["spearman"] and m != "(b)oracle":
                    if backfill is None:                      # lazy: one pass per cell
                        backfill = pearson_backfill(c)
                    pe = backfill.get((threat, seed, m))
                    backfill_used = pe is not None
                rows.append(dict(base, threat=threat, seed=seed, method=m,
                                 kind="det" if m in DET_METHODS else "val",
                                 spearman=res["spearman"].get(m), pearson=pe,
                                 pearson_src="phi-backfill" if backfill_used else
                                             ("metrics" if m in res.get("pearson", {}) else None),
                                 auroc=res["auroc"].get(m), runtime=res["runtime"].get(m)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ helpers
def footnote(fig, text):
    # place just below everything already drawn (incl. rotated tick labels);
    # bbox_inches="tight" then expands to include it
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    y = bb.y0 / fig.get_figheight() - 0.012
    fig.text(0.005, y, text, fontsize=6.5, color="#666666", va="top", ha="left")


def save(fig, path):
    FIG.mkdir(exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def annotated_heatmap(ax, mat, vmin, vmax, cmap, center=None, fmt="{:+.2f}", xlabels=True):
    """mat: DataFrame methods x columns; NaN cells drawn gray with a dash."""
    arr = np.ma.masked_invalid(mat.to_numpy(float))
    if center is not None:                                   # diverging around center
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
    if xlabels:
        ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=90, fontsize=6.5)
    else:
        ax.set_xticks(range(mat.shape[1]), [""] * mat.shape[1])
    ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=7)
    ax.grid(False)
    return im


def col_label(r):
    if r["regime"] == "silo5":
        return f"{r['scale']} silo5 {THREAT_SHORT[r['threat']]}"
    tag = "anchor " if r["category"] == "04_device100_anchor" else ""
    return f"a={r['alpha']:g} {tag}{THREAT_SHORT[r['threat']]}"


def cell_cols(df):
    """ordered unique (cell,threat) columns for heatmaps."""
    u = (df[["cell", "category", "regime", "scale", "alpha", "threat"]]
         .drop_duplicates())
    u["k"] = [(CAT_ORDER.index(c), -1.0 if np.isnan(a) else a, THREAT_ORDER.index(t))
              for c, a, t in zip(u["category"], u["alpha"], u["threat"])]
    u = u.sort_values("k")
    u["label"] = [col_label(r) for _, r in u.iterrows()]
    return u


# ------------------------------------------------------------------ figures
def fig_fidelity_heatmap(df, value, oracle, path, title, cmap="RdBu", center=0.0,
                         vmin=-1, vmax=1):
    sub = df[(df["kind"] == "val") & (df["method"] != "(b)oracle")
             & df[value].notna()
             & (df["ref"].str.startswith("(b)") if oracle else df["ref"].eq("Flirds(proxy)"))]
    if sub.empty:
        print(f"  [skip] {path.name}: no rows")
        return
    cols = cell_cols(df[df["ref"].str.startswith("(b)")] if oracle
                     else df[df["ref"].eq("Flirds(proxy)")])
    camps = [c for c in CAMPAIGNS if (sub["campaign"] == c).any()]
    methods = method_sort(sub["method"].unique())
    fig, axes = plt.subplots(1, len(camps), figsize=(0.42 * len(cols) * len(camps) + 3.2,
                                                     0.30 * len(methods) + 2.2),
                             squeeze=False)
    for ax, camp in zip(axes[0], camps):
        agg = (sub[sub["campaign"] == camp]
               .groupby(["cell", "threat", "method"])[value].mean().reset_index())
        mat = pd.DataFrame(index=methods, columns=cols["label"], dtype=float)
        for _, r in cols.iterrows():
            g = agg[(agg["cell"] == r["cell"]) & (agg["threat"] == r["threat"])]
            for _, q in g.iterrows():
                mat.at[q["method"], r["label"]] = q[value]
        im = annotated_heatmap(ax, mat, vmin, vmax, cmap, center=center)
        ax.set_title(f"{camp} campaign  (cells={int(mat.notna().any(axis=0).sum())}/{len(cols)})")
    fig.colorbar(im, ax=axes[0], shrink=0.8, label=title.split("--")[-1].strip())
    fig.suptitle(title, fontsize=10.5, y=1.02)
    footnote(fig, f"source: {' + '.join(str(CAMPAIGNS[c].name) + '/*/metrics.json|phi.parquet' for c in camps)}"
                  f" | mean over seeds | gray '-' = cell not run (deferred)")
    save(fig, path)


def fig_migration(df, path):
    j = df[df["campaign"] == "June"].set_index(["cell", "threat", "seed", "method"])
    k = df[df["campaign"] == "July"].set_index(["cell", "threat", "seed", "method"])
    common = j.index.intersection(k.index)
    if not len(common):
        print(f"  [skip] {path.name}: no June/July overlap")
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
    panels = [("spearman", "Spearman vs ref (per cell,seed,method)"),
              ("auroc", "detection AUROC"), ("runtime", "valuation wall-clock (s)")]
    cats = sorted(j.loc[common, "category"].unique())
    cat_col = dict(zip(cats, ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]))
    for ax, (val, ttl) in zip(axes, panels):
        x, y = j.loc[common, val].astype(float), k.loc[common, val].astype(float)
        ok = x.notna() & y.notna()
        c = [cat_col[v] for v in j.loc[common, "category"][ok]]
        if val == "runtime":
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.scatter(x[ok], y[ok], s=9, c=c, alpha=0.55, linewidths=0)
        lo = min(x[ok].min(), y[ok].min()); hi = max(x[ok].max(), y[ok].max())
        ax.plot([lo, hi], [lo, hi], color="#999999", lw=0.8, zorder=0)
        if val != "runtime":
            d = (y[ok] - x[ok]).abs()
            ax.set_title(f"{ttl}\nn={ok.sum()}, median|d|={d.median():.3f}, max|d|={d.max():.3f}",
                         fontsize=8.5)
        else:
            r = (y[ok] / x[ok])
            ax.set_title(f"{ttl}\nn={ok.sum()}, median July/June={r.median():.2f}x", fontsize=8.5)
        ax.set_xlabel("June campaign"); ax.set_ylabel("July campaign (post-migration)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=cat_col[c], label=c) for c in cats]
    fig.legend(handles=handles, loc="upper center", ncol=len(cats), fontsize=7,
               bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.suptitle("Server-migration reproducibility -- same cells, June vs July campaign",
                 y=1.14, fontsize=11)
    fig.tight_layout()
    footnote(fig, "source: rundirs/*/metrics.json vs rundirs_2026-07/*/metrics.json, matched on (cell,threat,seed,method)")
    save(fig, path)


def fig_runtime(df, path):
    sub = df[df["runtime"].notna()]
    methods = method_sort(sub["method"].unique())[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(methods) + 1.8))
    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        for camp, dy, col in (("June", -0.17, "#0072B2"), ("July", 0.17, "#D55E00")):
            v = sub[(sub["method"] == m) & (sub["campaign"] == camp)]["runtime"].astype(float)
            if not len(v):
                continue
            ax.scatter(v, np.full(len(v), i + dy) + rng.uniform(-0.06, 0.06, len(v)),
                       s=7, color=col, alpha=0.45, linewidths=0)
            ax.plot([v.median()] * 2, [i + dy - 0.13, i + dy + 0.13], color=col, lw=1.8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(methods)), methods, fontsize=7.5)
    ax.set_xlabel("valuation wall-clock per (cell,threat,seed) [s, log]")
    ax.set_title("Cost -- per-method valuation wall-clock, June vs July campaign")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#0072B2", label="June"),
                       plt.Line2D([], [], marker="o", ls="", color="#D55E00", label="July (post-migration)")],
              fontsize=7.5, loc="lower right")
    footnote(fig, "source: rundirs*/*/metrics.json runtime | dots=cells x seeds, bar=median")
    save(fig, path)


def fig_auroc_heatmap(df, path):
    sub = df[df["auroc"].notna() & (df["threat"] != "clean")]
    cols = cell_cols(sub)
    methods = method_sort(sub["method"].unique())
    camps = [c for c in CAMPAIGNS if (sub["campaign"] == c).any()]
    fig, axes = plt.subplots(len(camps), 1, figsize=(0.42 * len(cols) + 3.0,
                                                     (0.28 * len(methods) + 1.7) * len(camps)),
                             squeeze=False)
    for pi, (ax, camp) in enumerate(zip(axes[:, 0], camps)):
        agg = (sub[sub["campaign"] == camp]
               .groupby(["cell", "threat", "method"])["auroc"].mean().reset_index())
        mat = pd.DataFrame(index=methods, columns=cols["label"], dtype=float)
        for _, r in cols.iterrows():
            g = agg[(agg["cell"] == r["cell"]) & (agg["threat"] == r["threat"])]
            for _, q in g.iterrows():
                mat.at[q["method"], r["label"]] = q["auroc"]
        im = annotated_heatmap(ax, mat, 0, 1, "RdBu", center=0.5, fmt="{:.2f}",
                               xlabels=pi == len(camps) - 1)
        nval = sum(m not in DET_METHODS for m in methods)
        ax.axhline(nval - 0.5, color="black", lw=1.0)
        ax.set_title(f"{camp} campaign -- detection AUROC (corrupt=high; 0.5=chance; "
                     f"below line = model-side detectors)")
    fig.colorbar(im, ax=axes[:, 0], shrink=0.6, label="AUROC (mean over seeds)")
    footnote(fig, "source: rundirs*/*/metrics.json auroc | gray '-' = not run (July anchor deferred)")
    save(fig, path)


def fig_auroc_vs_alpha(df, path):
    sub = df[df["auroc"].notna() & (df["category"] == "02_device100_sweep")]
    if sub.empty:
        print(f"  [skip] {path.name}: no sweep rows")
        return
    threats = [t for t in THREAT_ORDER if t in set(sub["threat"])]
    kinds = [("val", "valuation methods"), ("det", "model-side detectors")]
    fig, axes = plt.subplots(2, len(threats), figsize=(3.1 * len(threats), 5.6),
                             sharey=True, squeeze=False)
    for row, (kd, kdlab) in enumerate(kinds):
        for ci, t in enumerate(threats):
            ax = axes[row][ci]
            g = sub[(sub["threat"] == t) & (sub["kind"] == kd)]
            alphas = sorted(g["alpha"].unique())
            xs = range(len(alphas))
            for m in method_sort(g["method"].unique()):
                for camp in CAMPAIGNS:
                    gc = (g[(g["method"] == m) & (g["campaign"] == camp)]
                          .groupby("alpha")["auroc"].mean().reindex(alphas))
                    if gc.notna().sum() == 0:
                        continue
                    ax.plot(xs, gc.to_numpy(), CAMP_STYLE[camp], marker="o", ms=2.5,
                            lw=1.1, color=METHOD_COLOR.get(m, "#BBBBBB"),
                            label=m if camp == "June" else None)
            ax.axhline(0.5, color="#999999", lw=0.7, ls=":")
            ax.set_xticks(list(xs), [f"{a:g}" for a in alphas])
            ax.set_ylim(-0.03, 1.06)
            if row == 0:
                ax.set_title(THREAT_SHORT[t])
            if ci == 0:
                ax.set_ylabel(f"AUROC -- {kdlab}")
            if row == 1:
                ax.set_xlabel("Dir(alpha) heterogeneity")
    for ax in axes[:, -1]:
        ax.legend(fontsize=6, loc="lower right", frameon=False)
    fig.suptitle("Detection AUROC vs client heterogeneity (device100 sweep; N=100)\n"
                 "solid=June, dashed=July | Spearman ref in this category is Flirds-proxy, so AUROC is the informative axis",
                 fontsize=9.5)
    fig.tight_layout()
    footnote(fig, "source: rundirs*/1B_device100-a*/metrics.json auroc, mean over seeds")
    save(fig, path)


# ------------------------------------------------------------------ main
def main():
    all_cells, coverage = [], []
    for name, root in CAMPAIGNS.items():
        cells, skipped = load_campaign(name, root)
        all_cells += cells
        cats = pd.Series([c["category"] for c in cells]).value_counts()
        coverage.append((name, root, cells, skipped, cats))
    df = build_frame(all_cells)

    print("== coverage ==")
    june_cells = {(c["cell"]) for c in all_cells if c["campaign"] == "June"}
    for name, root, cells, skipped, cats in coverage:
        print(f"  {name:5s} ({root.name}): {len(cells)} grid cells"
              + (f" | excluded B-axis cells -> matrix_cxni: {len(skipped)}" if skipped else ""))
        for cat in CAT_ORDER:
            print(f"        {cat:22s} {cats.get(cat, 0):2d}")
        if name == "July":
            missing = sorted(june_cells - {c["cell"] for c in cells})
            print(f"        MISSING vs June reference ({len(missing)}): {', '.join(missing) or '-'}")
    for scale, g in df.groupby("scale"):
        print(f"  seeds per cell [{scale}]: {sorted(g.groupby('cell')['seed'].nunique().unique())}")
    nbf = (df["pearson_src"] == "phi-backfill").sum()
    print(f"  pearson: {int((df['pearson_src'] == 'metrics').sum())} native rows, {int(nbf)} phi-backfilled rows")

    print("== figures ==")
    fig_fidelity_heatmap(df, "spearman", oracle=True, path=FIG / "01_fidelity_spearman_vs_oracle.png",
                         title="Fidelity (primary) -- Spearman vs exact oracle "
                               "[silo5/3B: (b) 2^N; anchor: (b) per-round]")
    fig_fidelity_heatmap(df, "pearson", oracle=True, path=FIG / "02_fidelity_pearson_vs_oracle.png",
                         title="Fidelity (value-level) -- Pearson vs exact oracle")
    fig_fidelity_heatmap(df, "spearman", oracle=False, path=FIG / "03_fidelity_spearman_vs_proxy_sweep.png",
                         title="Sweep agreement -- Spearman vs Flirds PROXY ref "
                               "(device100 sweep/poison: no oracle -> NOT ground-truth fidelity)")
    fig_migration(df, FIG / "04_migration_june_vs_july.png")
    fig_runtime(df, FIG / "05_cost_runtime_by_method.png")
    fig_auroc_heatmap(df, FIG / "06_detection_auroc_heatmap.png")
    fig_auroc_vs_alpha(df, FIG / "07_detection_auroc_vs_alpha_sweep.png")

    # small summary CSVs for the overview session (exact figure inputs)
    FIG.mkdir(exist_ok=True)
    agg = (df.groupby(["campaign", "category", "cell", "threat", "alpha", "scale",
                       "ref", "method", "kind"], dropna=False)
             .agg(n_seeds=("seed", "nunique"), spearman_mean=("spearman", "mean"),
                  spearman_min=("spearman", "min"), spearman_max=("spearman", "max"),
                  pearson_mean=("pearson", "mean"), auroc_mean=("auroc", "mean"),
                  runtime_median=("runtime", "median"))
             .reset_index())
    agg.to_csv(FIG / "summary_by_cell_method.csv", index=False)
    print(f"  wrote figures/summary_by_cell_method.csv ({len(agg)} rows)")


if __name__ == "__main__":
    main()
