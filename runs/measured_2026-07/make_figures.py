#!/usr/bin/env python
"""runs/measured_2026-07/make_figures.py -- July 2026 verification-measurement figures.

Four small measured campaigns (post-server-migration, B200), one figure each:
  taylor/     P3 physical Taylor-residual measurement (1B, R=10, 3 seeds; summary.json)
  tf32_ab/    TF32 on/off A/B on the CNN C1 stage (cifar10 iid|label-flip, seed0)
  microbench/ precision microbenchmark (forward / HVP / GEMM x fp32 / tf32 / bf16)
  acct/       honest cost accounting (valuation wall-clock vs the FL training run itself;
              summary .txt files are the persisted artifact for this campaign)

Reads ONLY those artifacts; regenerates every PNG under figures/.

  python runs/measured_2026-07/make_figures.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent            # runs/measured_2026-07
FIG = HERE / "figures"

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 8.5,
                     "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})

METHOD_ORDER = ["Flirds", "Flirds1st", "FedIF", "GTG", "FedSV", "ShapleyFL",
                "Banzhaf", "ComFedSV", "Fed-LOO", "loss-heur", "Ripple", "FLDetector"]


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


# ------------------------------------------------------------------ taylor (P3)
def fig_taylor(path):
    dirs = sorted((HERE / "taylor").glob("llama1b_r10_seed*"))
    rows = []
    for d in dirs:
        s = json.loads((d / "summary.json").read_text())
        seed = int(re.search(r"seed(\d+)", d.name).group(1))
        p = s["pooled"]
        rows.append(dict(seed=seed,
                         resid1_median=p["resid1"]["median"], resid1_max=p["resid1"]["max"],
                         resid2_median=p["resid2"]["median"], resid2_max=p["resid2"]["max"],
                         slope_r1=p["loglog_slope_r1"], slope_r2=p["loglog_slope_r2"],
                         ulp=s["rounds"][0]["ulp_base"]))
    if not rows:
        print("  [skip] taylor: no summaries")
        return 0
    df = pd.DataFrame(rows).sort_values("seed")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6))
    # (left) residual magnitudes, log scale
    xs = np.arange(len(df))
    w = 0.36
    ax = axes[0]
    ax.bar(xs - w / 2, df["resid1_median"], w * 0.9, color="#0072B2",
           label="resid1 (median): loss - 1st-order Taylor")
    ax.bar(xs + w / 2, df["resid2_median"], w * 0.9, color="#E69F00",
           label="resid2 (median): loss - 2nd-order Taylor")
    ax.scatter(xs - w / 2, df["resid1_max"], marker="_", s=160, color="#0072B2", label="max")
    ax.scatter(xs + w / 2, df["resid2_max"], marker="_", s=160, color="#E69F00")
    ax.axhline(df["ulp"].iloc[0], color="#999999", lw=0.8, ls=":",
               label=f"fp32 ulp at base loss ({df['ulp'].iloc[0]:.1e})")
    ax.set_yscale("log")
    ax.set_xticks(xs, [f"seed{s}" for s in df["seed"]])
    ax.set_ylabel("pooled residual (log)")
    ax.set_title("Taylor residuals: 2nd order cuts the error\n(pooled over 10 rounds x 2^5 coalitions)")
    ax.legend(fontsize=6.5, loc="upper right")
    # (right) log-log scaling slopes
    ax = axes[1]
    ax.bar(xs - w / 2, df["slope_r1"], w * 0.9, color="#0072B2", label="slope(resid1) [theory: 2]")
    ax.bar(xs + w / 2, df["slope_r2"], w * 0.9, color="#E69F00", label="slope(resid2) [theory: 3]")
    ax.axhline(2, color="#0072B2", lw=0.8, ls=":")
    ax.axhline(3, color="#E69F00", lw=0.8, ls=":")
    for x, v in zip(xs - w / 2, df["slope_r1"]):
        ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    for x, v in zip(xs + w / 2, df["slope_r2"]):
        ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xs, [f"seed{s}" for s in df["seed"]])
    ax.set_ylabel("log-log slope of residual vs ||dW||")
    ax.set_ylim(0, 3.6)
    ax.set_title("Residual scaling exponents\n(resid2 slope < 3: cubic scaling NOT confirmed)")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.suptitle("P3 physical Taylor-residual measurement -- Llama-3.2-1B, N=5, R=10, fp32",
                 fontsize=10.5, y=1.04)
    footnote(fig, "source: taylor/llama1b_r10_seed*/summary.json pooled.{resid1,resid2,loglog_slope_*} + rounds[0].ulp_base")
    save(fig, path)
    df.to_csv(FIG / "taylor_summary.csv", index=False)
    return len(df)


# ------------------------------------------------------------------ tf32 A/B
def fig_tf32(path):
    cells = {}
    for d in sorted((HERE / "tf32_ab").iterdir()):
        if not (d / "metrics.json").exists():
            continue
        m = json.loads((d / "metrics.json").read_text())
        mm = re.match(r"cifar10_(.+)_tf32(on|off)_seed(\d+)", d.name)
        cells[(mm.group(1), mm.group(2))] = m
    if not cells:
        print("  [skip] tf32_ab: no cells")
        return 0
    scens = sorted({k[0] for k in cells})
    metrics = [("spearman_b", "Spearman vs (b)", (-1.09, 1.09)),
               ("spearman_vs_rate", "Spearman(phi, corruption rate)", (-1.09, 1.09)),
               ("auroc", "detection AUROC", (-0.05, 1.09))]
    fig, axes = plt.subplots(len(scens), len(metrics),
                             figsize=(3.4 * len(metrics), 2.9 * len(scens)), squeeze=False)
    for si, scen in enumerate(scens):
        mon, moff = cells.get((scen, "on")), cells.get((scen, "off"))
        methods = method_sort(set(mon["methods"]) - {"(b)oracle"})
        for mi, (key, lab, ylim) in enumerate(metrics):
            ax = axes[si][mi]
            von = [mon["methods"][m].get(key, np.nan) for m in methods]
            voff = [moff["methods"][m].get(key, np.nan) for m in methods]
            von = [np.nan if v is None else v for v in von]
            voff = [np.nan if v is None else v for v in voff]
            if all(np.isnan(v) for v in von + voff):
                ax.axis("off")
                ax.set_title(f"{scen}: {lab} undefined", fontsize=8)
                continue
            xs = np.arange(len(methods))
            ax.scatter(xs - 0.12, voff, s=22, color="#0072B2", label="TF32 off (true fp32)",
                       zorder=3)
            ax.scatter(xs + 0.12, von, s=22, color="#D55E00", marker="s", label="TF32 on",
                       zorder=3)
            for x, a, b in zip(xs, voff, von):
                if not (np.isnan(a) or np.isnan(b)):
                    ax.plot([x - 0.12, x + 0.12], [a, b], color="#999999", lw=0.7)
            ax.set_xticks(xs, methods, rotation=45, ha="right", fontsize=6.5)
            ax.set_ylim(*ylim)
            ax.set_title(f"{scen}: {lab}", fontsize=8.5)
            if si == 0 and mi == 0:
                ax.legend(fontsize=6.5, loc="lower left")
    fig.suptitle("TF32 A/B on the CNN C1 stage (cuDNN conv TF32 default-on exposure check) -- "
                 "cifar10, seed0", fontsize=10, y=1.00)
    fig.tight_layout()
    footnote(fig, "source: tf32_ab/cifar10_*_tf32{on,off}_seed0/metrics.json methods.<M>.{spearman_b,spearman_vs_rate,auroc}")
    save(fig, path)
    return len(cells)


# ------------------------------------------------------------------ microbench
def fig_microbench(path):
    f = HERE / "microbench" / "summary.json"
    if not f.exists():
        print("  [skip] microbench")
        return 0
    s = json.loads(f.read_text())
    ops = [("forward", "s_per_pass", "forward pass (val batch)"),
           ("hvp", "s_per_pass", "HVP (functorch)"),
           ("gemm", "s_per_matmul", "GEMM 8k x 8k")]
    precs = ["fp32", "tf32", "bf16"]
    colors = {"fp32": "#0072B2", "tf32": "#E69F00", "bf16": "#009E73"}
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    xs = np.arange(len(ops))
    w = 0.8 / len(precs)
    for pi, p in enumerate(precs):
        vals = [s[f"{op}_{p}"][key] for op, key, _ in ops]
        pos = xs + (pi - 1) * w
        ax.bar(pos, vals, w * 0.9, color=colors[p], label=p)
        for x, v in zip(pos, vals):
            ax.text(x, v, f"{v:.3g}", ha="center", va="bottom", fontsize=6.5)
    ax.set_yscale("log")
    ax.set_xticks(xs, [lab for _, _, lab in ops])
    ax.set_ylabel("seconds per op (log)")
    r = s["ratios"]
    ax.set_title("Precision microbenchmark -- Llama-3.2-1B on B200 (matmul fp32 is real fp32)\n"
                 f"fp32/tf32 ratios: forward {r['forward_fp32_over_tf32']:.1f}x, "
                 f"HVP {r['hvp_fp32_over_tf32']:.1f}x, GEMM {r['gemm_fp32_over_tf32']:.1f}x")
    ax.legend(fontsize=8)
    footnote(fig, f"source: microbench/summary.json | device {s['device']}, torch {s['torch']} | "
                  f"GEMM tflops fp32 {s['gemm_fp32']['tflops']:.0f} / tf32 {s['gemm_tf32']['tflops']:.0f} "
                  f"/ bf16 {s['gemm_bf16']['tflops']:.0f}")
    save(fig, path)
    return 1


# ------------------------------------------------------------------ acct
def fig_acct(path):
    files = sorted((HERE / "acct").glob("acct_seed*.summary.txt"))
    if not files:
        print("  [skip] acct")
        return 0
    runt, walls = {}, []
    for f in files:
        t = f.read_text()
        walls.append(float(re.search(r"wall_s=([\d.]+)", t).group(1)))
        line = re.search(r"runtime \(s\):\s*(.+)", t).group(1)
        for m, v in re.findall(r"([\w()\-]+)=([\d.]+)", line):
            runt.setdefault(m, []).append(float(v))
    wall = float(np.mean(walls))
    methods = method_sort([m for m in runt if m != "(b)oracle"])
    if "(b)oracle" in runt:
        methods.append("(b)oracle")
    fig, ax = plt.subplots(figsize=(7.4, 0.34 * len(methods) + 1.8))
    ys = np.arange(len(methods))
    means = [np.mean(runt[m]) for m in methods]
    ax.barh(ys, means, 0.6,
            color=["#0072B2" if np.mean(runt[m]) < wall else "#D55E00" for m in methods])
    for m, y in zip(methods, ys):
        ax.scatter(runt[m], [y] * len(runt[m]), s=12, color="black", alpha=0.7, zorder=3,
                   linewidths=0)
        pct = np.mean(runt[m]) / wall * 100
        ax.text(max(np.mean(runt[m]), 4) * 1.15, y, f"{pct:.0f}% of FL train",
                va="center", fontsize=7)
    ax.axvline(wall, color="black", lw=1.2, ls="--",
               label=f"the FL training run itself ({wall:.0f}s mean)")
    ax.set_xscale("log")
    ax.set_yticks(ys, methods, fontsize=7.5)
    ax.set_xlabel("valuation wall-clock [s, log]")
    ax.set_title("Honest cost accounting -- valuation overhead relative to the FL training run\n"
                 "(1B, N=5, R=10; blue < training cost, orange > training cost)")
    ax.legend(fontsize=7.5, loc="lower right")
    footnote(fig, "source: acct/acct_seed{0,1,2}.summary.txt 'runtime (s)' + shared_log_generation wall_s | dots=seeds")
    save(fig, path)
    pd.DataFrame([dict(method=m, runtime_mean=np.mean(runt[m]), n_seeds=len(runt[m]),
                       pct_of_train=np.mean(runt[m]) / wall * 100) for m in methods]
                 ).to_csv(FIG / "acct_runtime.csv", index=False)
    return len(files)


# ------------------------------------------------------------------ main
def main():
    print("== coverage ==")
    print("== figures ==")
    n_t = fig_taylor(FIG / "01_taylor_residuals_p3.png")
    n_ab = fig_tf32(FIG / "02_tf32_ab_contrast.png")
    n_mb = fig_microbench(FIG / "03_precision_microbench.png")
    n_ac = fig_acct(FIG / "04_cost_accounting.png")
    print(f"  coverage: taylor {n_t}/3 seeds | tf32_ab {n_ab}/4 cells | "
          f"microbench {n_mb}/1 | acct {n_ac}/3 seeds")


if __name__ == "__main__":
    main()
