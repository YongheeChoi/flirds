#!/usr/bin/env python
"""Track C2-FID analysis -- derived from the rundirs only, no re-run.

Reads `rundirs/*/metrics.json` (+ `phi_b_rounds.parquet` when the (b) oracle was
round-SHARDED) and writes:
  fidelity.csv      long table, ONE row per (cell, seed, method) -- the c1
                    `runs/track_c/fidelity.csv` columns (dataset, scenario, seed,
                    method, spearman_b, pearson_b, spearman_a, pearson_a) plus a
                    `stage` column and this stage's extras, so it merges straight
                    into the LLM/C1 fidelity tables (07-23 cross-check decision 13).
                    `scenario` = the threat tag (c1 packs one axis into it);
                    `partition` is the extra axis, kept as its own column.
                    spearman_a/pearson_a are ALWAYS blank here -- the (a) 2^N
                    retrain oracle is infeasible at N=100 (plan §4.1).
  cellmean.csv      seed-averaged (cell-type x method) view.
  README.md         prediction verdicts (F-1..F-4) + the headline tables.

Run:  python runs/track_c/c2fid/make_analysis.py
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNDIRS = os.path.join(HERE, "rundirs")
OUT_DIR = os.path.join(HERE, "analysis")
TRUTH = "(b)oracle"
C1_COLS = ["dataset", "scenario", "seed", "method",                 # c1 fidelity.csv order
           "spearman_b", "pearson_b", "spearman_a", "pearson_a"]
EXTRA = ["stage", "partition", "threat", "flip_rate", "cell", "kendall_b",
         "cos_b", "euc_b", "maxdiff_b", "auroc", "spearman_vs_rate",
         "spearman_vs_rate_corrupt", "runtime_s"]
METHODS = ["Flirds", "Flirds1st", "GTG", "FedSV", "ComFedSV", "ShapleyFL",
           "FedIF", "loss-heur"]                       # Fed-LOO dropped (Yonghee 2026-07-23)
# F-1 families (code-grounded; README "사전등록 F-1"): uniform 1/|S| subset synthesis
# vs n-proportional reconstruction vs Taylor vs direct-game.
UNIFORM = ["ShapleyFL", "ComFedSV"]
NPROP = ["GTG", "FedSV"]
TAYLOR = ["Flirds", "Flirds1st", "FedIF"]
DIRECT = ["loss-heur"]
EXACT0 = ["Flirds", "Flirds1st", "loss-heur", "FedIF"]              # + (b): zero-delta -> exact 0


def _tag(m):
    """Threat tag = the sbatch/rundir vocabulary (dose kept distinct)."""
    if m["threat"] != "label_flip":
        return m["threat"]
    fr = m.get("flip_rate")
    return f"label_flip@{fr}" if fr not in (None, "") else "label_flip_strmain"


def load_cells():
    """One record per rundir; oracle-only shards are merged into their base cell."""
    cells, shards = {}, defaultdict(list)
    for d in sorted(glob.glob(os.path.join(RUNDIRS, "*"))):
        mf = os.path.join(d, "metrics.json")
        if not os.path.exists(mf):
            continue
        with open(mf) as f:
            m = json.load(f)
        m["_dir"] = d
        base = os.path.basename(d)
        sh = re.match(r"(.*)_b(\d+)-(\d+)$", base)
        if sh:                                          # C2FID_B_ROUNDS oracle shard
            shards[sh.group(1)].append((int(sh.group(2)), int(sh.group(3)), m))
        else:
            m["_cell"] = base
            with open(os.path.join(d, "config.yaml")) as f:
                m["flip_rate"] = next((l.split(":", 1)[1].strip()
                                       for l in f if l.startswith("flip_rate:")), "")
            m["flip_rate"] = "" if m["flip_rate"] in ("null", "") else m["flip_rate"]
            cells[base] = m
    for base, parts in shards.items():
        if base not in cells:
            print(f"[warn] shards for {base} but no methods run -- skipped")
            continue
        _merge_shards(cells[base], parts)
    return list(cells.values())


def _merge_shards(cell, parts):
    """Sum the round-sharded (b) phi back into the cell, with a COVERAGE assert
    (the rounds must tile 0..R-1 exactly once -- decision 14)."""
    n = len(cell["corrupt"])
    phi = np.zeros(n)
    seen = []
    for lo, hi, m in parts:
        pb = pd.read_parquet(os.path.join(m["_dir"], "phi_b_rounds.parquet"))
        assert pb["round"].between(lo, hi - 1).all(), f"{cell['_cell']}: shard rows outside {lo}:{hi}"
        for cid, v in pb.groupby("client")["phi_b"].sum().items():
            phi[int(cid)] += float(v)
        seen += list(range(lo, hi))
    R = cell["cfg"]["rounds"] if "cfg" in cell else len(cell["acc_curve"])
    assert sorted(seen) == list(range(R)), \
        f"{cell['_cell']}: shard coverage {len(seen)} rounds != {R} (or overlapping)"
    cell["methods"][TRUTH] = dict(phi=phi.tolist(),
                                  runtime=sum(m.get("methods", {}).get(TRUTH, {}).get("runtime", 0.0)
                                              for _, _, m in parts))
    cell["_shards"] = len(parts)
    _recompute_vs_truth(cell)


def _recompute_vs_truth(cell):
    """Fill each method's vs-(b) metrics after a shard merge (the methods run had
    C2FID_ORACLE_B=0, so its metrics.json carries no spearman_b)."""
    from scipy.stats import kendalltau, spearmanr
    gt = np.asarray(cell["methods"][TRUTH]["phi"], float)
    for name, m in cell["methods"].items():
        if name == TRUTH:
            continue
        v = np.asarray(m["phi"], float)
        m["spearman_b"] = float(spearmanr(v, gt).correlation)
        m["kendall_b"] = float(kendalltau(v, gt).correlation)
        m["pearson_b"] = (float("nan") if v.std() == 0 or gt.std() == 0
                          else float(np.corrcoef(v, gt)[0, 1]))
        na, nb = np.linalg.norm(v), np.linalg.norm(gt)
        m["cos_b"] = float(1 - (v @ gt) / (na * nb)) if na and nb else float("nan")
        m["euc_b"] = float(np.linalg.norm(v - gt))
        m["maxdiff_b"] = float(np.abs(v - gt).max())


def rows_from(cells):
    rows = []
    for m in cells:
        for name, mm in m["methods"].items():
            rows.append({"stage": "c2fid", "cell": m["_cell"], "dataset": m["dataset"],
                         "partition": m["partition"], "threat": m["threat"],
                         "flip_rate": m.get("flip_rate", ""), "scenario": _tag(m),
                         "seed": m["seed"], "method": name,
                         "spearman_b": mm.get("spearman_b", ""), "pearson_b": mm.get("pearson_b", ""),
                         "spearman_a": "", "pearson_a": "",          # (a) infeasible at N=100
                         "kendall_b": mm.get("kendall_b", ""), "cos_b": mm.get("cos_b", ""),
                         "euc_b": mm.get("euc_b", ""), "maxdiff_b": mm.get("maxdiff_b", ""),
                         "auroc": mm.get("auroc", ""),
                         "spearman_vs_rate": mm.get("spearman_vs_rate", ""),
                         "spearman_vs_rate_corrupt": mm.get("spearman_vs_rate_corrupt", ""),
                         "runtime_s": mm.get("runtime", "")})
    return rows


def _mean(df, method, col, **eq):
    sub = df[df["method"] == method]
    for k, v in eq.items():
        sub = sub[sub[k] == v] if not isinstance(v, (list, tuple)) else sub[sub[k].isin(v)]
    x = pd.to_numeric(sub[col], errors="coerce").dropna()
    return float(x.mean()) if len(x) else float("nan")


def _v(ok, vals):
    """HIT/MISS, or N/A when the contrast's inputs are missing (partial grid)."""
    return "N/A" if any(v != v for v in vals) else ("HIT" if ok else "MISS")


def verdicts(df, cells):
    """F-1..F-4 auto-contrast (README 사전등록).  Returns [(id, verdict, detail)]."""
    out = []

    # F-1  qskew (24x size skew) vs iid: uniform-1/|S| synthesis should LOSE ground.
    d = {m: _mean(df, m, "spearman_b", dataset="cifar10", partition="qskew")
         - _mean(df, m, "spearman_b", dataset="cifar10", partition="iid") for m in METHODS}
    fam = {k: np.nanmean([d[m] for m in v])
           for k, v in (("uniform", UNIFORM), ("nprop", NPROP),
                        ("taylor", TAYLOR), ("direct", DIRECT))}
    ok = fam["uniform"] < 0 and all(fam["uniform"] < fam[k] for k in ("nprop", "taylor", "direct"))
    out.append(("F-1", _v(ok, fam.values()),
                "Delta_rho(qskew-iid) family means " +
                " ".join(f"{k}={v:+.3f}" for k, v in fam.items()) +
                " | per-method " + " ".join(f"{m}={d[m]:+.3f}" for m in METHODS)))

    # F-2  free_rider: exact-0 for the game-honest methods; frrand: Flirds family AUROC >= renorm.
    z = {}
    for c in [c for c in cells if c["threat"] == "free_rider"]:
        for name, mm in c["methods"].items():
            z[name] = max(z.get(name, 0.0), float(np.abs(np.asarray(mm["phi"], float)).max()))
    exact0 = {m: z.get(m, float("nan")) for m in EXACT0 + [TRUTH]}
    ghost = {m: z.get(m, float("nan")) for m in NPROP}
    a_fl = np.nanmean([_mean(df, m, "auroc", threat="frrand") for m in ("Flirds", "Flirds1st")])
    a_re = np.nanmean([_mean(df, m, "auroc", threat="frrand") for m in NPROP])
    ok = (z and all(v < 1e-9 for v in exact0.values() if v == v)
          and all(v > 1e-9 for v in ghost.values() if v == v) and a_fl >= a_re)
    out.append(("F-2", "HIT" if ok else ("MISS" if z else "N/A"),
                "fr max|phi| exact0-family=" + ",".join(f"{k}:{v:.2e}" for k, v in exact0.items()) +
                " renorm=" + ",".join(f"{k}:{v:.2e}" for k, v in ghost.items()) +
                f" | frrand AUROC flirds={a_fl:.3f} renorm={a_re:.3f}"))

    # F-3  participation 10/100 reproduces the LLM std50k5 collapse ORDER.
    r = {m: _mean(df, m, "spearman_b") for m in METHODS}
    top = np.nanmean([r["Flirds"], r["Flirds1st"]])
    mid = np.nanmean([r[m] for m in NPROP])
    bot = np.nanmean([r[m] for m in UNIFORM])
    ok = top >= mid >= bot
    out.append(("F-3", _v(ok, [top, mid, bot]),
                f"mean rho(b) flirds={top:+.3f} >= nprop={mid:+.3f} >= uniform={bot:+.3f} | " +
                " ".join(f"{m}={r[m]:+.3f}" for m in METHODS)))

    # F-4  strmain dose resolution: Flirds ~ (b) ceiling > Flirds1st (both svr variants).
    det = []
    ok = True
    for col in ("spearman_vs_rate", "spearman_vs_rate_corrupt"):
        b = _mean(df, TRUTH, col, scenario="label_flip_strmain")
        f2 = _mean(df, "Flirds", col, scenario="label_flip_strmain")
        f1 = _mean(df, "Flirds1st", col, scenario="label_flip_strmain")
        det.append(f"{col}: (b)={b:+.3f} Flirds={f2:+.3f} Flirds1st={f1:+.3f}")
        ok = ok and (f2 >= f1) and (f2 == f2)
    out.append(("F-4", "HIT" if ok else "MISS", " | ".join(det)))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cells = load_cells()
    if not cells:
        print(f"no rundirs under {RUNDIRS} -- nothing to do")
        return
    rows = rows_from(cells)
    path = os.path.join(OUT_DIR, "fidelity.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=C1_COLS + EXTRA)
        w.writeheader()
        w.writerows(rows)
    df = pd.DataFrame(rows)
    print(f"c2fid: {len(cells)} cells -> {path} ({len(rows)} rows)")

    num = ["spearman_b", "pearson_b", "kendall_b", "auroc", "spearman_vs_rate",
           "spearman_vs_rate_corrupt", "runtime_s"]
    cm = (df.assign(**{c: pd.to_numeric(df[c], errors="coerce") for c in num})
            .groupby(["dataset", "partition", "scenario", "method"], as_index=False)[num]
            .mean())
    cm.to_csv(os.path.join(OUT_DIR, "cellmean.csv"), index=False)

    piv = cm.pivot_table(index="method", columns=["dataset", "partition"],
                         values="spearman_b")
    lines = ["# Track C2-FID 분석 (자동 생성 — `make_analysis.py`; 수치 canon = rundir)", "",
             f"셀 {len(cells)} · 행 {len(rows)}. 지표 정의·사전등록 원문 = `../README.md`.",
             "게임 캐비엇: 10/100 참여 부분게임 — C1(N=10 전원참여) 표와 직접 비교 금지.", "",
             "## Spearman vs (b) — (dataset, partition) 평균", "",
             "```", piv.round(3).to_string(), "```", "",
             "## 사전등록 대조 (F-1~F-4)", ""]
    print("\n=== Spearman vs (b) ===")
    print(piv.round(3).to_string())
    print("\n=== prereg ===")
    for pid, v, detail in verdicts(df, cells):
        print(f"  {pid} {v:4s} {detail}")
        lines += [f"- **{pid} {v}** — {detail}"]
    with open(os.path.join(OUT_DIR, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_DIR}/{{fidelity.csv,cellmean.csv,README.md}}")


if __name__ == "__main__":
    main()
