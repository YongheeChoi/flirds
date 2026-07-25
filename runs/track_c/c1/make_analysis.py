#!/usr/bin/env python
"""runs/track_c/c1/make_analysis.py -- Track C1 **(a) 재학습 오라클 오염축 그리드** 집계.

무대(계획서 §2.1 G2 본문 = cifar10 · §3.1 G9 부록 = mnist):
    {cifar10, mnist} x {iid, dir1} x {clean, label_flip, free_rider, grad_noise} x 3seed
    = 48셀.  셀 내부 = (a) 2^10 재학습 오라클 + (b) 2^10 in-run 오라클 + 9방법 phi
    (N=10 full participation, R=10).

`runs/track_c/make_figures.py` 는 **레거시 5-시나리오** 그리드 전용이다(시나리오명이
하드코딩돼 있고 (a) 를 별도 `c1_oracle/*_aonly_*` 에서 읽는다).  이 축 그리드는
(a)/(b) 가 **한 metrics.json 안**에 들어 있어 레이아웃이 다르므로 집계도 분리한다.
서로의 셀은 건드리지 않는다 -- 이 스크립트는 `partition`/`threat` 키가 있는 rundir만
읽고, 레거시 셀은 이름이 아니라 **스키마로** 걸러낸다.

rundir-only · read-only · 재실행 가능(phase2/track_g make_analysis 관례): 아직 안 끝난
셀은 그냥 안 나타난다.  방법 재실행 없음.  `analysis/` 는 매번 지우고 다시 만든다.

보고 순서 = 프로젝트 핵심 질문 위계:
    [1차] (a) 대비 fidelity  ->  듀얼 오라클 (b)vs(a) 일치도  ->  (b) 대비 fidelity
    [2차-3] 탐지 AUROC
    [부속] phi 부호 감사 CNN 레그(계획서 §3.4) · 런타임(비-canonical)

  python runs/track_c/c1/make_analysis.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent            # runs/track_c/c1
OUT = ROOT / "analysis"

DATASETS = ["cifar10", "mnist"]                   # cifar10 = 본문(G2), mnist = 부록(G9)
PARTS = ["iid", "dir1"]
THREATS = ["clean", "label_flip", "free_rider", "grad_noise"]
SEEDS = [0, 1, 2]
N_EXPECTED = len(DATASETS) * len(PARTS) * len(THREATS) * len(SEEDS)

METHOD_ORDER = ["(b)oracle", "Flirds", "Flirds1st", "FedIF", "GTG", "FedSV",
                "ShapleyFL", "Banzhaf", "ComFedSV", "loss-heur", "Ripple"]
ESTIMATORS = [m for m in METHOD_ORDER if m != "(b)oracle"]

FID = ["spearman", "kendall", "pearson", "cos", "euc", "maxdiff"]
LABEL = {"clean": "clean", "label_flip": "lf@0.70", "free_rider": "free-rider",
         "grad_noise": "grad-noise"}
TTAG = {"clean": "clean", "label_flip": "label-flip_fr0.70",      # sbatch_c1_axis.sh TTAGS
        "free_rider": "free-rider", "grad_noise": "grad-noise"}


# --------------------------------------------------------------------- load
def load_cells():
    """축 그리드 rundir -> (cell rows, method rows).  레거시 셀은 스키마로 배제."""
    cells, methods = [], []
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        f = d / "metrics.json"
        if not f.exists():
            continue
        try:
            m = json.loads(f.read_text())
        except json.JSONDecodeError:              # 셀이 쓰는 중 -- 다음 실행에 잡힌다
            continue
        if "partition" not in m or "threat" not in m:
            continue                              # 레거시 C1 시나리오 셀 (make_figures.py 소관)

        corrupt = np.asarray(m.get("corrupt") or [], dtype=int)
        oa = m.get("oracle_a") or {}
        phi_a = np.asarray(oa.get("phi"), dtype=float) if oa.get("phi") else None
        base = dict(cell=d.name, dataset=m["dataset"], partition=m["partition"],
                    threat=m["threat"], seed=int(m["seed"]))
        cells.append(dict(
            **base, mode=m.get("mode"), final_acc=m.get("final_acc"),
            traj_time=m.get("traj_time"), t_a=oa.get("time"),
            n_retrains=oa.get("n_retrains"), n_clients=len(corrupt),
            n_corrupt=int(corrupt.sum()), corrupt_ids=list(np.flatnonzero(corrupt)),
            ripple_skipped=bool(m.get("ripple_skipped", False)),
            has_phi_a=phi_a is not None))

        for name, v in (m.get("methods") or {}).items():
            phi = np.asarray(v.get("phi"), dtype=float)
            row = dict(**base, method=name, runtime=v.get("runtime"),
                       auroc=v.get("auroc"))
            for k in FID:
                row[f"{k}_a"] = v.get(f"{k}_a")
                row[f"{k}_b"] = v.get(f"{k}_b")
            row.update(sign_stats(phi, corrupt))
            methods.append(row)
        if phi_a is not None:                     # (a) 자신도 부호 감사 대상
            methods.append(dict(**base, method="(a)oracle", runtime=oa.get("time"),
                                auroc=None, **{f"{k}_{s}": None for k in FID for s in "ab"},
                                **sign_stats(phi_a, corrupt)))
    return pd.DataFrame(cells), pd.DataFrame(methods)


def sign_stats(phi, corrupt):
    """계획서 §3.4 부호 감사(CNN 레그)에 필요한 셀-내 phi 요약."""
    if phi.size == 0 or phi.size != corrupt.size:
        return {}
    mal = corrupt.astype(bool)
    out = dict(n_neg=int((phi < 0).sum()), n_exact_zero=int((phi == 0.0).sum()),
               phi_min=float(phi.min()), phi_max=float(phi.max()))
    out["n_neg_clean"] = int((phi[~mal] < 0).sum())
    out["phi_clean_mean"] = float(phi[~mal].mean()) if (~mal).any() else None
    if mal.any():
        gap = float(phi[~mal].mean() - phi[mal].mean())
        span = float(phi.max() - phi.min())
        out.update(n_neg_corrupt=int((phi[mal] < 0).sum()),
                   n_exact_zero_corrupt=int((phi[mal] == 0.0).sum()),
                   phi_corrupt_mean=float(phi[mal].mean()),
                   # 분리도: 양수 = 오염 클라를 정직 클라보다 낮게 매겼다(원하는 방향).
                   # 방법마다 phi 스케일이 다르므로(min-max 정규화 계열 vs 생 phi)
                   # 셀-내 span 으로 나눈 _norm 만 방법 간 비교에 쓴다.
                   phi_gap=gap,
                   phi_gap_norm=(gap / span) if span > 0 else None)
    return out


# ------------------------------------------------------------------ tables
def agg(df, col):
    """(dataset, partition, threat) x method -> 'mean+-std (n)' 3-seed 집계."""
    g = df.dropna(subset=[col]).groupby(
        ["dataset", "partition", "threat", "method"])[col]
    s = g.agg(["mean", "std", "count"])
    s["txt"] = [f"{r['mean']:+.3f}" + (f"±{r['std']:.3f}" if r["count"] > 1 else "")
                + (f" ({int(r['count'])})" if r["count"] != len(SEEDS) else "")
                for _, r in s.iterrows()]
    t = s["txt"].unstack("method")
    cols = [m for m in ["(a)oracle"] + METHOD_ORDER if m in t.columns]
    return t.reindex(columns=cols)


def md_table(t, index_names=("데이터셋", "파티션", "위협")):
    """MultiIndex DataFrame -> 마크다운 표."""
    if t.empty:
        return ["_(해당 셀 없음)_"]
    idx = [list(i) if isinstance(i, tuple) else [i] for i in t.index]
    idx = [[LABEL.get(x, x) for x in row] for row in idx]
    head = list(index_names[:len(idx[0])]) + [str(c) for c in t.columns]
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for k, (_, r) in zip(idx, t.iterrows()):
        vals = ["" if pd.isna(v) else str(v) for v in r]
        out.append("| " + " | ".join(k + vals) + " |")
    return out


def main():
    cells, methods = load_cells()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    if cells.empty:
        (OUT / "README.md").write_text("축 그리드 셀 없음 (아직 완료된 셀이 0개).\n",
                                       encoding="utf-8")
        print("[analysis] 0 cells")
        return

    cells.to_csv(OUT / "cells.csv", index=False)
    methods.to_csv(OUT / "methods_long.csv", index=False)

    have = {(r.dataset, r.partition, r.threat, r.seed) for r in cells.itertuples()}
    missing = [f"{d}_{p}_{TTAG[t]}_seed{s}" for d in DATASETS for p in PARTS
               for t in THREATS for s in SEEDS if (d, p, t, s) not in have]

    est = methods[methods.method.isin(ESTIMATORS)]
    fid_a = agg(methods[methods.method != "(a)oracle"], "spearman_a")
    fid_b = agg(est, "spearman_b")
    ken_a = agg(methods[methods.method != "(a)oracle"], "kendall_a")
    auroc = agg(methods[methods.method != "(a)oracle"], "auroc")
    for name, t in [("fidelity_vs_a_spearman", fid_a), ("fidelity_vs_b_spearman", fid_b),
                    ("fidelity_vs_a_kendall", ken_a), ("detection_auroc", auroc)]:
        t.to_csv(OUT / f"{name}.csv")

    # ---- 듀얼 오라클: (b) vs (a) 일치도 (방법 검증의 근거) ----
    dual = methods[methods.method == "(b)oracle"][
        ["dataset", "partition", "threat", "seed"] + [f"{k}_a" for k in FID]]
    dual.to_csv(OUT / "dual_oracle_b_vs_a.csv", index=False)
    dsum = agg(methods[methods.method == "(b)oracle"], "spearman_a")

    # ---- §3.4 부호 감사 CNN 레그 ----
    audit_cols = ["cell", "dataset", "partition", "threat", "seed", "method",
                  "n_neg", "n_neg_clean", "n_neg_corrupt", "n_exact_zero",
                  "n_exact_zero_corrupt", "phi_clean_mean", "phi_corrupt_mean",
                  "phi_gap", "phi_gap_norm", "phi_min", "phi_max"]
    audit = methods.reindex(columns=audit_cols)
    audit.to_csv(OUT / "sign_audit.csv", index=False)

    rt = est.dropna(subset=["runtime"]).groupby(["dataset", "method"])["runtime"].median()
    rt.to_csv(OUT / "runtime_median.csv")

    # ------------------------------------------------------------------ README
    md = ["# Track C1 축 그리드 -- (a) 재학습 오라클 무대", "",
          f"셀 {len(cells)}/{N_EXPECTED} · 방법-행 {len(methods)} "
          f"(rundir만 읽어 재생성; 미완 셀은 나타나지 않는다)", ""]
    if missing:
        md += [f"**미완/미제출 {len(missing)}셀**: " + ", ".join(missing[:16])
               + (" …" if len(missing) > 16 else ""), ""]
    md += ["> 무대 = N=10 full · R=10 · (a) 2^10 재학습 + (b) 2^10 in-run.",
           "> **비교불가성**: (a)는 2^N 재학습이라 N=100 에서 원리적으로 불가 -- 1A-CNN"
           "(N=100 부분참여)과 N·참여율은 못 맞춘다. 맞춘 것은 **오염축과 파티션뿐**이다.",
           "> **ShapleyFL β = 0.3 (canonical)**. track_c1 은 `flirds.baselines.shapleyfl.BETA`"
           "(모듈 상수 0.3, `SFL_BETA` 로만 오버라이드)를 그대로 쓰고 이 잡은 오버라이드하지"
           " 않았다 -- 레거시 1B-CNN 표의 'β0.5 잔존' 캐비엇은 이 그리드에 **해당 없음**."
           " 다만 C1 러너가 β 를 rundir config 에 남기지 않으므로(감사 갭) 이 문장이 유일한"
           " 근거다.", ""]

    md += ["## [1차] (a) 재학습 오라클 대비 fidelity -- Spearman", "",
           "핵심 질문 위계 1차. `(b)oracle` 열 = 듀얼 오라클 일치도(추정기가 아니라 오라클끼리).", ""]
    md += md_table(fid_a)
    md += ["", "### Kendall tau (같은 셀, 순위-일치 보조지표)", ""] + md_table(ken_a)

    md += ["", "## 듀얼 오라클 (b) vs (a)", "",
           "task-6 교훈: (a)는 **추정기가 겨냥하는 효용을 그대로 플레이할 때만** 검증이 된다. "
           "여기서는 둘 다 val-loss 게임이다.", ""]
    md += md_table(dsum.rename(columns={"(b)oracle": "(b) vs (a) rho"}))
    md += ["", "> **free-rider 셀 해석 주의**: frzero 는 delta=0 이므로 **고정가중** 게임인 "
           "(b) 에서 phi 가 정확히 0 이지만, (a) 재학습에서는 그 클라가 평균의 분모에 남아 "
           "정직한 업데이트를 희석한다 -> phi 가 **음수**. 두 오라클의 불일치가 아니라 "
           "**게임 정의 차이**이며, 아래 부호 감사 표로 확인한다.", ""]

    md += ["", "## [1차-보조] (b) in-run 오라클 대비 fidelity -- Spearman", "",
           "논문 나머지 표와 같은 기준선.", ""]
    md += md_table(fid_b)

    md += ["", "## [2차-3] 탐지 AUROC -- 오염 클라 분리", "",
           "위계상 **마지막** 축. clean 셀은 오염 클라가 없어 정의되지 않는다(표에서 빠짐).", ""]
    md += md_table(auroc)

    md += ["", "## phi 부호 감사 -- CNN 레그 (계획서 §3.4)", "",
           "현 감사 스냅샷에 없던 **frzero·grad-noise** 를 채운다. 전수는 `sign_audit.csv`.", ""]
    a = audit.dropna(subset=["n_neg"])
    if not a.empty:
        acl, ccl = a[a.threat == "clean"], cells[cells.threat == "clean"]
        slot = ccl.groupby(["dataset", "partition"]).n_clients.sum()   # seed 합산 슬롯
        cl = acl.groupby(["dataset", "partition", "method"])["n_neg_clean"].sum()
        cl = cl.astype(int).astype(str) + "/" + cl.index.droplevel("method").map(
            slot).astype(int).astype(str)
        cl = cl.unstack("method").reindex(
            columns=[m for m in ["(a)oracle"] + METHOD_ORDER if m in
                     acl.method.unique()])
        md += ["**(i) clean 셀 -- 전 클라 phi 양수여야 한다**(Stage 0 전제: 오배제-0). "
               "칸 = `음수 phi 클라 / 전 클라 슬롯`(seed 합); **0/N 이 정상**", ""]
        md += md_table(cl, index_names=("데이터셋", "파티션"))
        afr = a[a.threat == "free_rider"]
        fr = afr.groupby("method")[["n_exact_zero_corrupt", "n_neg_corrupt"]].sum()
        slots = int(cells[cells.threat == "free_rider"].n_corrupt.sum())
        md += ["", "**(ii) free-rider(frzero) 셀 -- 오염 클라 phi 가 exact-0 인가** "
               f"(방법당 {slots} 오염-클라 슬롯 = 셀 x 오염 클라 합계)", "",
               "| 방법 | exact-0 | 음수 |", "|---|---|---|"]
        for mth in [m for m in ["(a)oracle"] + METHOD_ORDER if m in fr.index]:
            md.append(f"| {mth} | {int(fr.loc[mth, 'n_exact_zero_corrupt'])}/{slots} | "
                      f"{int(fr.loc[mth, 'n_neg_corrupt'])}/{slots} |")
        gp = a[a.threat != "clean"].groupby(["dataset", "threat", "method"])[
            "phi_gap_norm"].mean().unstack("method")
        gp = gp.reindex(columns=[m for m in ["(a)oracle"] + METHOD_ORDER if m in gp.columns])
        md += ["", "**(iii) 분리도** `[mean phi(정직) - mean phi(오염)] / (max phi - min phi)` "
               "-- 양수 = 오염 클라를 낮게 매겼다(원하는 방향)", "",
               "> 셀-내 span 으로 나눈 **무차원** 값이다. 생 `phi_gap` 은 방법 간 스케일이 "
               "달라(min-max 정규화 계열 vs 생 phi) 비교할 수 없다 -- 원값은 `sign_audit.csv`.",
               "> frzero 는 오염 phi 가 0 이므로 이 칸은 사실상 **정직 클라 phi 의 부호**를 본다.", ""]
        md += md_table(gp.round(3), index_names=("데이터셋", "위협"))

    md += ["", "## 런타임 (median, s)", "",
           "> ⚠️ **논문 §5.5 cost 표에 쓰지 않는다.** canonical cost = B200 실측만. "
           "이 값은 3090·동시 8셀 환경의 참고치다.", ""]
    rmeth = [m for m in METHOD_ORDER if m in rt.index.get_level_values("method")]
    md += ["| 데이터셋 | " + " | ".join(rmeth) + " |",
           "|" + "|".join(["---"] * (1 + len(rmeth))) + "|"]
    for d in sorted(set(rt.index.get_level_values("dataset"))):
        md.append(f"| {d} | " + " | ".join(
            f"{rt.get((d, m), float('nan')):.1f}" for m in rmeth) + " |")
    tt = cells.dropna(subset=["t_a"]).groupby("dataset")["t_a"].median()
    md += ["", "(a) 2^10 재학습 오라클 t_a median: " +
           " · ".join(f"{d} {v:.0f}s ({v / 3600:.1f}h)" for d, v in tt.items()), ""]

    (OUT / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[analysis] {len(cells)}/{N_EXPECTED} cells, {len(methods)} method rows -> {OUT}")
    if missing:
        print(f"[analysis] missing {len(missing)}: {missing[:8]}{' …' if len(missing) > 8 else ''}")


if __name__ == "__main__":
    main()
