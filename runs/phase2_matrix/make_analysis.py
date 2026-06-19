#!/usr/bin/env python
"""Aggregate phase2_matrix run-dirs into analysis/ — CSV tables + charts, per experiment taxonomy.

Taxonomy (mirrors the master_queue stage classification):
  01_silo5            cross-silo N=5 (1B), {noisy, freerider_random, freerider_zero, poison} x 3 seeds
  02_device100_sweep  cross-device N=100, alpha in {0, 0.01, 0.1, 5.0} x {noisy, fr_random, fr_zero}
  03_device100_poison cross-device N=100 poison (D2b install config), alpha in {0.5, 0.0}
  04_device100_anchor cross-device N=100 alpha=0.5 + (b)-perround oracle + coalition baselines
  05_scale_3b         3B silo5 N=5 seed 0, 4 threats

Inputs : rundirs/<cell>/{config.yaml, meta.json, metrics.json, phi.parquet}
Outputs: analysis/{README.md, 00_overview, <category>/{csv,charts}}  (analysis/ is wiped + rebuilt)
Re-runnable: discovers whatever run-dirs exist; cells still running simply don't appear yet.

Usage: /home/korea_bupj/miniconda3/envs/flirds/bin/python make_analysis.py

Score orientation note: phi.parquet stores every method in suspicion orientation
(higher = more suspicious).  kind="val" methods are good->LOW phi (FedIF/ShapleyFL/ComFedSV
already sign-aligned at persist time); kind="det" are native high=suspicious.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent          # runs/phase2_matrix
RUNDIRS = ROOT / "rundirs"
OUT = ROOT / "analysis"
QUEUE = ROOT / "master_queue.txt"

CAT_ORDER = ["01_silo5", "02_device100_sweep", "03_device100_poison",
             "04_device100_anchor", "05_scale_3b"]
CAT_DESC = {
    "01_silo5":            "cross-silo N=5 (1B), 4 threats x 3 seeds, full method set + (b) exact 2^N oracle",
    "02_device100_sweep":  "cross-device N=100 Dir(alpha) sweep, cheap methods, Spearman vs Flirds proxy-truth",
    "03_device100_poison": "cross-device N=100 poison (D2b install config: lr=2e-3 batch=8 epochs=5 frac=0.8 R=60)",
    "04_device100_anchor": "cross-device N=100 alpha=0.5 anchor: (b) per-round exact + coalition baselines",
    "05_scale_3b":         "3B (Llama-3.2-3B) silo5 N=5 seed 0, 4 threats, (b) oracle on / coalition off",
}
THREAT_ORDER = ["noisy", "freerider_random", "freerider_zero", "poison"]
THREAT_LABEL = {"noisy": "noisy (answer_swap)", "freerider_random": "free-rider (random)",
                "freerider_zero": "free-rider (zero)", "poison": "poison (backdoor x gamma)"}
METHOD_ORDER = ["(b)oracle", "Flirds", "Flirds1st", "FedIF", "GTG", "FedSV", "ShapleyFL",
                "Banzhaf", "ComFedSV", "loss-heur", "FLDetector", "STD-DAGMM", "FLTrust", "FedDQC"]
_c20 = plt.get_cmap("tab20")
METHOD_COLOR = {m: _c20(i % 20) for i, m in enumerate(METHOD_ORDER)}

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 8.5,
                     "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------- load + classify
def classify(cfg):
    if cfg["regime"] == "silo5":
        return "01_silo5" if cfg.get("scale", "1B") == "1B" else "05_scale_3b"
    if "poison" in cfg["threats"]:
        return "03_device100_poison"
    return "04_device100_anchor" if cfg.get("oracle_b") else "02_device100_sweep"


def spearman_ref(cfg):
    if not cfg.get("oracle_b"):
        return "Flirds(proxy)"
    return "(b)perround" if cfg["regime"] == "device100" else "(b)oracle"


def load_cells():
    cells = []
    for d in sorted(RUNDIRS.iterdir()):
        if not (d / "metrics.json").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        meta = json.loads((d / "meta.json").read_text())
        metrics = json.loads((d / "metrics.json").read_text())
        phi = pd.read_parquet(d / "phi.parquet")
        cells.append(dict(cell=d.name, category=classify(cfg), cfg=cfg, meta=meta,
                          metrics=metrics, phi=phi))
    return cells


def cell_sort_key(c):
    cfg = c["cfg"]
    threat = cfg["threats"][0]
    alpha = cfg["alpha"] if cfg["regime"] == "device100" else -1.0
    return (CAT_ORDER.index(c["category"]), alpha, THREAT_ORDER.index(threat))


# ---- value-level fidelity, backfilled from phi.parquet (no method re-run) ----
# Mirrors the runner's report(): for each (cell,threat,seed) the truth vector is
# "(b)oracle" if present else "Flirds" (proxy-truth), and every val method's phi is
# scored against it on the shared clients.  Pearson = affine-invariant value agreement
# (rewards LINEAR match, not just rank — key where Spearman saturates at +1); the GTG
# distances are scale-dependent (comparable within a cell, not across scales).
def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _cos_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(1 - a @ b / d) if d else np.nan


def fidelity_from_phi(pdf):
    out = {}
    if not len(pdf):
        return out
    for (cell, threat, seed), g in pdf.groupby(["cell", "threat", "seed"], sort=False):
        kinds = dict(zip(g["method"], g["kind"]))
        vecs = {m: gm.set_index("client")["phi"] for m, gm in g.groupby("method")}
        truth = "(b)oracle" if "(b)oracle" in vecs else ("Flirds" if "Flirds" in vecs else None)
        if truth is None:
            continue
        tv = vecs[truth]
        for m, mv in vecs.items():
            if m == truth or kinds.get(m) != "val":
                continue
            cl = mv.index.intersection(tv.index)
            a, b = mv.loc[cl].to_numpy(float), tv.loc[cl].to_numpy(float)
            out[(cell, threat, seed, m)] = dict(
                pearson=_pearson(a, b), cosine_d=_cos_d(a, b),
                euclid_d=float(np.linalg.norm(a - b)), max_diff=float(np.abs(a - b).max()))
    return out


def build_frames(cells):
    mrows, prows = [], []
    for c in cells:
        cfg, ref = c["cfg"], spearman_ref(c["cfg"])
        base = dict(cell=c["cell"], category=c["category"], regime=cfg["regime"],
                    scale=cfg.get("scale", "1B"),
                    alpha=cfg["alpha"] if cfg["regime"] == "device100" else np.nan)
        for key, res in c["metrics"].items():
            threat, seed = key.rsplit("_seed", 1)
            seed = int(seed)
            names = set(res["auroc"]) | set(res["spearman"]) | set(res["runtime"])
            for m in names:
                mrows.append(dict(base, threat=threat, seed=seed, method=m,
                                  kind="det" if m in {"FLDetector", "STD-DAGMM", "FLTrust", "FedDQC"} else "val",
                                  auroc=res["auroc"].get(m), spearman=res["spearman"].get(m),
                                  spearman_ref=ref if m in res["spearman"] else None,
                                  runtime=res["runtime"].get(m), asr=res.get("asr")))
            corrupt = set(res["corrupt"])
            ph = c["phi"]
            ph = ph[(ph["threat"] == threat) & (ph["seed"] == seed)]
            for r in ph.itertuples(index=False):
                prows.append(dict(base, threat=threat, seed=seed, method=r.method, kind=r.kind,
                                  client=r.client, phi=r.phi, is_corrupt=r.client in corrupt))
    mdf = pd.DataFrame(mrows)
    pdf = pd.DataFrame(prows)
    if len(pdf):
        grp = pdf.groupby(["cell", "threat", "seed", "method"])["phi"]
        pdf["z"] = grp.transform(lambda v: (v - v.mean()) / (v.std(ddof=0) or 1.0))
    fid = fidelity_from_phi(pdf)                      # value-level fidelity, post-hoc from phi
    for col in ("pearson", "cosine_d", "euclid_d", "max_diff"):
        mdf[col] = [fid.get((r.cell, r.threat, r.seed, r.method), {}).get(col, np.nan)
                    for r in mdf.itertuples(index=False)]
    return mdf, pdf


# ---------------------------------------------------------------- small helpers
def method_sort(ms):
    ms = list(ms)
    return [m for m in METHOD_ORDER if m in ms] + sorted(set(ms) - set(METHOD_ORDER))


def footnote(fig, text):
    # below the axes/ticklabels; bbox_inches="tight" expands to include it
    fig.text(0.01, -0.03, text, fontsize=6.3, color="0.35", ha="left", va="top")


def cfg_note(c):
    cfg, r, meta = c["cfg"], c["cfg"]["rcfg"], c["meta"]
    a = f" alpha={cfg['alpha']}" if cfg["regime"] == "device100" else ""
    return (f"{cfg.get('scale','1B')} {cfg['model'].split('/')[-1]} | {cfg['regime']}{a} "
            f"N={r['n_clients']} R={r['rounds']} steps={r['max_steps']} lr={r['lr']} val={r['val']} "
            f"| seeds={cfg['seeds']} | git {meta['git_sha'][:7]}{'(dirty)' if meta.get('git_dirty') else ''}")


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def pretty_pivot(df, value, columns, index="method"):
    g = df.dropna(subset=[value]).groupby([index, columns])[value].agg(["mean", "std", "count"])
    if g.empty:
        return pd.DataFrame()
    s = g.apply(lambda r: f"{r['mean']:.3f}±{r['std']:.3f}" if r["count"] > 1 else f"{r['mean']:.3f}", axis=1)
    pv = s.unstack(columns)
    pv = pv.reindex(method_sort(pv.index)) if index == "method" else pv
    cols = [c for c in (THREAT_ORDER if columns == "threat" else sorted(pv.columns)) if c in pv.columns]
    return pv[cols] if cols else pv


def bar_panel(ax, sub, metric, ylim=(0, 1.06), chance=0.5):
    g = sub.dropna(subset=[metric]).groupby("method")[metric].agg(["mean", "std", "count"])
    ms = method_sort(g.index)
    xs = np.arange(len(ms))
    means = [g.loc[m, "mean"] for m in ms]
    stds = [g.loc[m, "std"] if g.loc[m, "count"] > 1 else 0.0 for m in ms]
    ax.bar(xs, means, yerr=np.nan_to_num(stds), capsize=2,
           color=[METHOD_COLOR.get(m, "0.6") for m in ms], edgecolor="0.3", linewidth=0.4)
    for x, v in zip(xs, means):                       # make exact-zero bars readable
        if v < 0.05:
            ax.text(x, 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=5.8,
                    color="crimson", rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels(ms, rotation=60, ha="right", fontsize=6.8)
    if chance is not None:
        ax.axhline(chance, ls="--", lw=0.7, color="0.5")
    if ylim:
        ax.set_ylim(*ylim)


def panel_title(c, threat):
    res0 = next(v for k, v in c["metrics"].items() if k.startswith(threat))
    lr = c["cfg"]["rcfg"]["lr"]
    asrs = [v["asr"] for k, v in c["metrics"].items() if k.startswith(threat) and v["asr"] is not None]
    t = THREAT_LABEL.get(threat, threat) + f"  (lr={lr}"
    t += f", ASR={np.mean(asrs):.2f})" if asrs else ")"
    return t, res0["corrupt"]


# ---------------------------------------------------------------- per-category charts
def chart_auroc_by_threat(cat, mdf, bythreat, path, metric="auroc", ylabel="AUROC (corrupt=high)",
                          ylim=(0, 1.06), chance=0.5):
    sub = mdf[mdf["category"] == cat].dropna(subset=[metric])
    threats = [t for t in THREAT_ORDER if t in set(sub["threat"])]
    if not threats:
        return
    fig, axes = plt.subplots(1, len(threats), figsize=(3.4 * len(threats) + 0.8, 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, threats):
        bar_panel(ax, sub[sub["threat"] == t], metric, ylim=ylim, chance=chance)
        title, corrupt = panel_title(bythreat[t], t)
        ax.set_title(f"{title}\ncorrupt={corrupt}", fontsize=8)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"{cat} — {metric} by threat (mean±std over seeds)", y=1.02)
    footnote(fig, cfg_note(bythreat[threats[0]]))
    save(fig, path)


def chart_fidelity_heatmap(cat, mdf, note, path, ref, value="spearman", vlabel="Spearman"):
    sub = mdf[(mdf["category"] == cat)].dropna(subset=[value])
    if sub.empty:
        return
    pv = sub.pivot_table(index="method", columns="threat", values=value)
    pv = pv.reindex(index=method_sort(pv.index),
                    columns=[t for t in THREAT_ORDER if t in pv.columns])
    fig, ax = plt.subplots(figsize=(1.1 * len(pv.columns) + 2.6, 0.34 * len(pv) + 1.7))
    im = ax.imshow(pv.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels([THREAT_LABEL.get(t, t) for t in pv.columns], rotation=25, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(pv)))
    ax.set_yticklabels(pv.index, fontsize=7.5)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"{vlabel} vs {ref} (mean over seeds)")
    kind = "ranking" if value == "spearman" else "value-level"
    ax.set_title(f"{cat} — {kind} fidelity vs {ref}")
    footnote(fig, note)
    save(fig, path)


def chart_runtime(cat, mdf, note, path):
    sub = mdf[mdf["category"] == cat].dropna(subset=["runtime"])
    if sub.empty:
        return
    g = sub.groupby("method")["runtime"].mean()
    ms = method_sort(g.index)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.3 * len(ms) + 1.6))
    ax.barh(range(len(ms)), [g[m] for m in ms], color=[METHOD_COLOR.get(m, "0.6") for m in ms],
            edgecolor="0.3", linewidth=0.4)
    for i, m in enumerate(ms):
        ax.text(g[m] * 1.06, i, f"{g[m]:.0f}s", va="center", fontsize=7)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels(ms, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("mean valuation runtime per run (s, log)")
    ax.set_title(f"{cat} — method runtime (all threats/seeds)")
    footnote(fig, note)
    save(fig, path)


def chart_phi_heatmap(cat, pdf, bythreat, outdir):
    sub = pdf[pdf["category"] == cat]
    for t in [t for t in THREAT_ORDER if t in set(sub["threat"])]:
        st = sub[sub["threat"] == t]
        pv = st.pivot_table(index="method", columns="client", values="z")
        pv = pv.reindex(method_sort(pv.index))
        corrupt = sorted(int(x) for x in st.loc[st["is_corrupt"], "client"].unique())
        fig, ax = plt.subplots(figsize=(0.85 * len(pv.columns) + 2.8, 0.32 * len(pv) + 1.8))
        im = ax.imshow(pv.values, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(len(pv.columns)))
        ax.set_xticklabels(pv.columns, fontsize=7.5)
        for j, cl in enumerate(pv.columns):
            if cl in corrupt:
                ax.get_xticklabels()[j].set_color("red")
        ax.set_yticks(range(len(pv)))
        ax.set_yticklabels(pv.index, fontsize=7.5)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6.5)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8, label="score z (suspicion: high=corrupt-like)")
        ax.set_xlabel(f"client (red = corrupt {corrupt})")
        ax.set_title(f"{cat} — per-client score, {THREAT_LABEL.get(t, t)} (z per method/seed, mean over seeds)")
        footnote(fig, cfg_note(bythreat[t]))
        save(fig, outdir / f"phi_heatmap_{t}.png")


def chart_vs_alpha(cat, mdf, metric, methods, note, path, ylabel, ylim, chance):
    sub = mdf[(mdf["category"] == cat) & (mdf["method"].isin(methods))].dropna(subset=[metric])
    if sub.empty:
        return
    alphas = sorted(sub["alpha"].unique())
    xs = np.arange(len(alphas))
    threats = [t for t in THREAT_ORDER if t in set(sub["threat"])]
    fig, axes = plt.subplots(1, len(threats), figsize=(3.6 * len(threats) + 1.6, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, threats):
        st = sub[sub["threat"] == t]
        for m in method_sort(st["method"].unique()):
            g = st[st["method"] == m].groupby("alpha")[metric].agg(["mean", "std", "count"])
            g = g.reindex(alphas)
            yerr = np.where(g["count"] > 1, g["std"], 0.0)
            ax.errorbar(xs, g["mean"], yerr=np.nan_to_num(yerr), marker="o", ms=3.5, lw=1.1,
                        capsize=2, label=m, color=METHOD_COLOR.get(m, "0.5"))
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{a:g}" for a in alphas])
        ax.set_xlabel("Dirichlet alpha (0 = domain-disjoint)")
        ax.set_title(THREAT_LABEL.get(t, t), fontsize=8.5)
        if chance is not None:
            ax.axhline(chance, ls="--", lw=0.7, color="0.5")
        ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.5, frameon=False)
    fig.suptitle(f"{cat} — {metric} vs alpha (mean±std over seeds)", y=1.03)
    footnote(fig, note)
    save(fig, path)


def chart_phi_separation(cat, pdf, bycell, outdir):
    sub = pdf[pdf["category"] == cat]
    for t in [t for t in THREAT_ORDER if t in set(sub["threat"])]:
        st = sub[sub["threat"] == t]
        alphas = sorted(st["alpha"].dropna().unique())
        if not alphas:
            continue
        fig, axes = plt.subplots(1, len(alphas), figsize=(3.8 * len(alphas) + 0.6, 3.8), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, a in zip(axes, alphas):
            sa = st[st["alpha"] == a]
            ms = method_sort(sa["method"].unique())
            for j, m in enumerate(ms):
                ben = sa[(sa["method"] == m) & ~sa["is_corrupt"]]["z"].values
                cor = sa[(sa["method"] == m) & sa["is_corrupt"]]["z"].values
                if len(ben):
                    ax.boxplot(ben, positions=[j], widths=0.55, showfliers=False,
                               medianprops=dict(color="0.2"), boxprops=dict(color="0.45"),
                               whiskerprops=dict(color="0.45"), capprops=dict(color="0.45"))
                if len(cor):
                    ax.plot(j + np.linspace(-0.15, 0.15, len(cor)), cor, "r.", ms=3.5, alpha=0.75)
            ax.set_xticks(range(len(ms)))
            ax.set_xticklabels(ms, rotation=60, ha="right", fontsize=6.8)
            ax.set_title(f"alpha={a:g}", fontsize=8.5)
        axes[0].set_ylabel("score z (red = corrupt clients)")
        fig.suptitle(f"{cat} — corrupt-vs-benign score separation, {THREAT_LABEL.get(t, t)} "
                     "(all seeds pooled)", y=1.03)
        note_cell = next(c for c in bycell.values()
                         if c["category"] == cat and c["cfg"]["threats"][0] == t)
        footnote(fig, cfg_note(note_cell))
        save(fig, outdir / f"phi_separation_{t}.png")


def chart_poison_cells(cat, mdf, ccells, path):
    sub = mdf[mdf["category"] == cat].dropna(subset=["auroc"])
    if sub.empty:
        return
    names = [c["cell"] for c in ccells]
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names) + 0.8, 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cn in zip(axes, names):
        bar_panel(ax, sub[sub["cell"] == cn], "auroc")
        c = next(x for x in ccells if x["cell"] == cn)
        asrs = [v["asr"] for v in c["metrics"].values() if v["asr"] is not None]
        ax.set_title(f"{cn}\nalpha={c['cfg']['alpha']:g}, ASR={np.mean(asrs):.2f}" if asrs
                     else cn, fontsize=8)
    axes[0].set_ylabel("AUROC (corrupt=high)")
    fig.suptitle(f"{cat} — detection AUROC per cell (mean±std over seeds)", y=1.02)
    footnote(fig, cfg_note(ccells[0]))
    save(fig, path)


def chart_scale_compare(mdf, by1b, path):
    s1 = mdf[mdf["category"] == "01_silo5"].dropna(subset=["auroc"])
    s3 = mdf[mdf["category"] == "05_scale_3b"].dropna(subset=["auroc"])
    if s1.empty or s3.empty:
        return
    threats = [t for t in THREAT_ORDER if t in set(s1["threat"]) & set(s3["threat"])]
    fig, axes = plt.subplots(1, len(threats), figsize=(3.6 * len(threats) + 0.8, 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, threats):
        g1 = s1[s1["threat"] == t].groupby("method")["auroc"].agg(["mean", "std", "count"])
        g3 = s3[s3["threat"] == t].groupby("method")["auroc"].mean()
        ms = [m for m in method_sort(g1.index) if m in g3.index]
        xs = np.arange(len(ms))
        ax.bar(xs - 0.19, [g1.loc[m, "mean"] for m in ms],
               yerr=np.nan_to_num([g1.loc[m, "std"] for m in ms]), width=0.38, capsize=2,
               color=[METHOD_COLOR.get(m, "0.6") for m in ms], edgecolor="0.3", linewidth=0.4,
               label="1B (3 seeds)")
        ax.bar(xs + 0.19, [g3[m] for m in ms], width=0.38, hatch="//",
               color=[METHOD_COLOR.get(m, "0.6") for m in ms], edgecolor="0.25", linewidth=0.4,
               alpha=0.75, label="3B (seed 0)")
        for x, m in zip(xs, ms):                      # make exact-zero bars readable
            for dx, v in ((-0.19, g1.loc[m, "mean"]), (0.19, g3[m])):
                if v < 0.05:
                    ax.text(x + dx, 0.012, f"{v:.2f}", ha="center", va="bottom",
                            fontsize=5.5, color="crimson", rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels(ms, rotation=60, ha="right", fontsize=6.8)
        ax.set_title(THREAT_LABEL.get(t, t), fontsize=8.5)
        ax.axhline(0.5, ls="--", lw=0.7, color="0.5")
        ax.set_ylim(0, 1.06)
    axes[0].set_ylabel("AUROC")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("scale check — AUROC 1B (silo5, 3 seeds) vs 3B (seed 0), shared methods", y=1.02)
    footnote(fig, "left bar=01_silo5 cells | right hatched=05_scale_3b cells (single seed; coalition off at 3B)")
    save(fig, path)


# ---------------------------------------------------------------- overview charts
def chart_heatmap_all(mdf, cells_sorted, path):
    pv = mdf.pivot_table(index="method", columns="cell", values="auroc")
    pv = pv.reindex(index=method_sort(pv.index),
                    columns=[c["cell"] for c in cells_sorted if c["cell"] in pv.columns])
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad("0.88")
    fig, ax = plt.subplots(figsize=(0.52 * len(pv.columns) + 3.2, 0.32 * len(pv) + 2.6))
    im = ax.imshow(np.ma.masked_invalid(pv.values), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(pv.columns, rotation=75, ha="right", fontsize=6.8)
    ax.set_yticks(range(len(pv)))
    ax.set_yticklabels(pv.index, fontsize=7.5)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="AUROC (mean over seeds; gray = not run)")
    ax.set_title("phase2 matrix — detection AUROC, all methods x all cells (taxonomy order)")
    footnote(fig, "cells ordered: 01_silo5 -> 02_sweep(alpha asc) -> 03_poison -> 04_anchor -> 05_3b | source: rundirs/*/metrics.json")
    save(fig, path)


def chart_frontier(mdf, note, path):
    sub = mdf[(mdf["category"] == "01_silo5") & (mdf["threat"] != "poison")]
    val = sub.dropna(subset=["spearman"])
    if val.empty:
        return
    g = val.groupby("method").agg(sp=("spearman", "mean"))
    g["rt"] = sub.dropna(subset=["runtime"]).groupby("method")["runtime"].mean()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    offs = [(5, 4), (5, -11), (-10, 12), (-48, -4), (5, 16)]   # cycle: de-overlap co-located labels
    for i, m in enumerate(g.sort_values("rt").index):
        ax.scatter(g.loc[m, "rt"], g.loc[m, "sp"], s=46, color=METHOD_COLOR.get(m, "0.5"),
                   edgecolor="0.25", linewidth=0.5, zorder=3)
        ax.annotate(m, (g.loc[m, "rt"], g.loc[m, "sp"]), textcoords="offset points",
                    xytext=offs[i % len(offs)], fontsize=7.4)
    rt_b = sub[sub["method"] == "(b)oracle"]["runtime"].mean()
    if np.isfinite(rt_b):
        ax.scatter(rt_b, 1.0, marker="*", s=130, color="black", zorder=3)
        ax.annotate("(b) exact (ref=1.0)", (rt_b, 1.0), textcoords="offset points",
                    xytext=(6, -10), fontsize=7.4)
    ax.set_xscale("log")
    ax.set_xlabel("mean valuation runtime per run (s, log)")
    ax.set_ylabel("Spearman vs (b) exact (mean)")
    ax.set_title("cost-fidelity frontier — silo5 N=5 1B, non-poison threats (3x3 runs)")
    footnote(fig, note)
    save(fig, path)


# ---------------------------------------------------------------- main
def main():
    cells = load_cells()
    if not cells:
        print("no run-dirs found under", RUNDIRS)
        return
    cells.sort(key=cell_sort_key)
    mdf, pdf = build_frames(cells)

    if OUT.exists():
        shutil.rmtree(OUT)
    ov = OUT / "00_overview"
    (ov / "charts").mkdir(parents=True)

    # ---- overview CSVs
    inv = pd.DataFrame([{
        "cell": c["cell"], "category": c["category"], "regime": c["cfg"]["regime"],
        "scale": c["cfg"].get("scale", "1B"), "model": c["cfg"]["model"].split("/")[-1],
        "alpha": c["cfg"]["alpha"] if c["cfg"]["regime"] == "device100" else "",
        "threat": c["cfg"]["threats"][0], "seeds": len(c["cfg"]["seeds"]),
        "n_clients": c["cfg"]["rcfg"]["n_clients"], "rounds": c["cfg"]["rcfg"]["rounds"],
        "lr": c["cfg"]["rcfg"]["lr"], "val": c["cfg"]["rcfg"]["val"],
        "oracle_b": c["cfg"]["oracle_b"], "coalition": c["cfg"]["coalition"],
        "spearman_ref": spearman_ref(c["cfg"]),
        "asr_mean": np.mean([v["asr"] for v in c["metrics"].values() if v["asr"] is not None] or [np.nan]),
        "git_sha": c["meta"]["git_sha"][:7], "git_dirty": c["meta"].get("git_dirty", ""),
    } for c in cells])
    inv.to_csv(ov / "runs_inventory.csv", index=False)
    mdf.to_csv(ov / "master_metrics.csv", index=False)
    pdf.to_csv(ov / "master_phi.csv", index=False)
    pretty_pivot(mdf, "auroc", "cell").to_csv(ov / "auroc_table.csv")
    chart_heatmap_all(mdf, cells, ov / "charts" / "auroc_heatmap_all.png")
    note01 = cfg_note(next((c for c in cells if c["category"] == "01_silo5"), cells[0]))
    chart_frontier(mdf, note01, ov / "charts" / "cost_quality_frontier.png")

    # ---- per category
    for cat in CAT_ORDER:
        cdf = mdf[mdf["category"] == cat]
        if cdf.empty:
            continue
        cdir = OUT / cat
        (cdir / "csv").mkdir(parents=True)
        (cdir / "charts").mkdir(parents=True)
        ccells = [c for c in cells if c["category"] == cat]
        bythreat = {c["cfg"]["threats"][0]: c for c in ccells}
        bycell = {c["cell"]: c for c in ccells}
        ref = spearman_ref(ccells[0]["cfg"])
        note = cfg_note(ccells[0])

        cdf.to_csv(cdir / "csv" / "metrics_long.csv", index=False)
        pdf[pdf["category"] == cat].to_csv(cdir / "csv" / "phi_long.csv", index=False)
        col = "cell" if cat in ("02_device100_sweep", "03_device100_poison") else "threat"
        pretty_pivot(cdf, "auroc", col).to_csv(cdir / "csv" / "auroc_table.csv")
        refslug = ref.replace("(", "").replace(")", "")
        sp = pretty_pivot(cdf, "spearman", col)
        if len(sp):
            sp.to_csv(cdir / "csv" / f"spearman_vs_{refslug}.csv")
        pr = pretty_pivot(cdf, "pearson", col)                # value-level fidelity (Pearson)
        if len(pr):
            pr.to_csv(cdir / "csv" / f"pearson_vs_{refslug}.csv")
        pretty_pivot(cdf, "runtime", col).to_csv(cdir / "csv" / "runtime_table.csv")

        if cat == "02_device100_sweep":
            au_methods = sorted(cdf.dropna(subset=["auroc"])["method"].unique())
            sp_methods = sorted(cdf.dropna(subset=["spearman"])["method"].unique())
            chart_vs_alpha(cat, mdf, "auroc", au_methods, note,
                           cdir / "charts" / "auroc_vs_alpha.png", "AUROC", (-0.02, 1.06), 0.5)
            chart_vs_alpha(cat, mdf, "spearman", sp_methods, note,
                           cdir / "charts" / "spearman_vs_alpha.png",
                           f"Spearman vs {ref}", (-1.05, 1.05), 0.0)
            chart_phi_separation(cat, pdf, bycell, cdir / "charts")
        elif cat == "03_device100_poison":
            chart_poison_cells(cat, mdf, ccells, cdir / "charts" / "auroc_per_cell.png")
            chart_runtime(cat, cdf, note, cdir / "charts" / "runtime.png")
            chart_phi_separation(cat, pdf, bycell, cdir / "charts")
        else:
            chart_auroc_by_threat(cat, mdf, bythreat, cdir / "charts" / "auroc_by_threat.png")
            chart_fidelity_heatmap(cat, cdf, note, cdir / "charts" / "spearman_heatmap.png", ref)
            chart_fidelity_heatmap(cat, cdf, note, cdir / "charts" / "pearson_heatmap.png", ref,
                                   value="pearson", vlabel="Pearson")
            chart_runtime(cat, cdf, note, cdir / "charts" / "runtime.png")
            if cat in ("01_silo5", "05_scale_3b"):
                chart_phi_heatmap(cat, pdf, bythreat, cdir / "charts")
    chart_scale_compare(mdf, None, OUT / "05_scale_3b" / "charts" / "scale_1b_vs_3b.png")

    # ---- README
    expected = [ln.split("|")[0].strip() for ln in QUEUE.read_text().splitlines()
                if ln.strip() and not ln.startswith("#")] if QUEUE.exists() else []
    have = {c["cell"] for c in cells}
    pending = [e for e in expected if e not in have]
    lines = ["# phase2 matrix — analysis (generated)", "",
             f"생성: `make_analysis.py` 재실행으로 전체 재생성 (이 폴더는 wipe 후 재작성됨).",
             f"입력: `rundirs/<cell>/` (φ.parquet + metrics.json + config.yaml + meta.json), "
             f"셀 {len(cells)}개 발견" + (f" / 대기 {len(pending)}개: {', '.join(pending)}" if pending else " (큐 전체 완료)"),
             "", "## 분류 체계", ""]
    for cat in CAT_ORDER:
        n = sum(1 for c in cells if c["category"] == cat)
        lines.append(f"- `{cat}/` — {CAT_DESC[cat]} — 셀 {n}개" + ("" if n else " **(아직 없음)**"))
    lines += ["", "## 파일 구성",
              "- `00_overview/` — 전체 통합: `master_metrics.csv`(모든 cell×seed×method, tidy) / "
              "`master_phi.csv`(모든 per-client score) / `runs_inventory.csv`(셀 설정·provenance) / "
              "`auroc_table.csv` + `charts/`(전체 AUROC 히트맵, cost-fidelity frontier)",
              "- 각 카테고리: `csv/`(metrics_long, auroc/spearman/pearson/runtime 표, phi_long) + `charts/`",
              "", "## 읽는 법",
              "- score 방향: **모든 method의 저장 score는 higher=more suspicious** "
              "(valuation φ는 good→low로 부호 정렬되어 저장; FedIF/ShapleyFL/ComFedSV는 persist 시 negate됨).",
              "- Spearman 기준(ref)은 카테고리별로 다름: silo5/3B = (b) exact 2^N, "
              "sweep/poison(device) = Flirds proxy-truth, anchor = (b) per-round exact. "
              "CSV의 `spearman_ref` 컬럼 참조.",
              "- **fidelity = 2축**: `spearman`(순위 일치) + `pearson`(φ 값 자체의 선형 일치, "
              "affine-invariant — 순위가 +1로 saturate하는 N=5에서 변별력을 줌). 둘 다 같은 ref 대비, "
              "phi.parquet에서 post-hoc 산출. metrics_long엔 GTG 거리 `cosine_d`/`euclid_d`/`max_diff`"
              "(스케일 의존 → 셀 내 비교용)도 포함. **N=5는 점 5개라 Pearson 분산이 큼** "
              "— device100(N≈100)에서 가장 의미 있음.",
              "- device100 φ는 한 번이라도 선택된 클라이언트만 기록 (K=10/round, R=30 → ~96-98/100); "
              "AUROC도 동일 집합 위에서 계산.",
              "- z 컬럼: per (cell,threat,seed,method) 표준화 — method 간 score 스케일 차이 제거용.",
              "", f"재생성: `/home/korea_bupj/miniconda3/envs/flirds/bin/python {Path(__file__).name}`", ""]
    (OUT / "README.md").write_text("\n".join(lines))

    n_png = len(list(OUT.rglob("*.png")))
    n_csv = len(list(OUT.rglob("*.csv")))
    print(f"analysis/ rebuilt: {len(cells)} cells -> {n_csv} CSVs + {n_png} charts"
          + (f" | pending cells: {len(pending)}" if pending else " | queue complete"))
    for cat in CAT_ORDER:
        n = sum(1 for c in cells if c["category"] == cat)
        print(f"  {cat}: {n} cells")


if __name__ == "__main__":
    main()
