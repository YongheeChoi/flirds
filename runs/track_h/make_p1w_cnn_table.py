#!/usr/bin/env python
"""W-B analysis -- Flirds P1 vs P1w on the extended (non-dir1) CNN stage
(paper/workplan/T4-p1w-cnn-relay.md).  Re-runnable, rundir-only; reuses
track_h/make_analysis.analyze_cnn (which MERGES runs/track_g/rundirs_cnn T1 arms
with runs/track_h/rundirs_cnn T2 arms on the cell key).  Renames the on-disk P2
label -> P1w in the OUTPUT only (arm labels stay P2 = gatew_v2 / t2_signw).

P1  = sign gate, n-weights          : T1 flirds_gate_v2  / T2 t2_sign_flirds
P1w = sign gate + size weight (=P2) : T1 flirds_gatew_v2 / T2 t2_signw_flirds

Absolute test acc is the headline (overview 3.2.3 convention, Yonghee 2026-07-20);
recovery = (arm-vanilla)/(oracle_excl-vanilla) is the cross-cell ranking axis.
Rows that have not been run yet (the T2 leg before sbatch_cnn_p1w.sh completes)
simply show blank -- the T1 legs already print from the skew campaign on disk.

Outputs: analysis/p1w_cnn.csv (per (dataset,partition,threat,timing) P1/P1w) +
         analysis/p1w_cnn_README.md.
Run:  python runs/track_h/make_p1w_cnn_table.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from make_analysis import analyze_cnn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis"
CLEAN_BAND = 0.006          # C2 clean-parity band (track_g spec 5-2)
REC_MIN_GAP = 0.02          # |oracle_excl - vanilla| below this -> recovery is noise
                            # (track_g RECOVERY_MIN_GAP; fmnist gaps are often < this)

# W-B stage: 4 core partitions (single RTX3090 stack) + cifar10 iid (stack-caveated:
# its clean/fr/gn/lf T1 anchors are the B200 originals in track_g/rundirs_cnn).
CORE = [("cifar10", "shard"), ("cifar10", "qskew"),
        ("fmnist", "iid"), ("fmnist", "dir1")]
IID = ("cifar10", "iid")
STAGE = CORE + [IID]
DIR1 = ("cifar10", "dir1")                       # W-A reference (Track H R1)
THREAT_ORDER = ["clean", "free_rider", "frrand", "grad_noise", "lf@0.7", "strmain"]
CORRUPT = ["free_rider", "frrand", "grad_noise", "lf@0.7", "strmain"]


def threat_tag(threat, fr):
    if threat != "label_flip":
        return threat
    if fr is None or (isinstance(fr, float) and np.isnan(fr)):
        return "strmain"
    return f"lf@{fr:g}"                          # 0.70 -> 'lf@0.7' (matches THREAT_ORDER)


def prep(df):
    """flirds rows, tagged threat, restricted to the W-B stage + dir1 reference."""
    f = df[df["source"] == "flirds"].copy()
    f["ttag"] = [threat_tag(t, fr) for t, fr in zip(f["threat"], f["flip_rate"])]
    f = f[f["ttag"].isin(THREAT_ORDER)]
    f["dp"] = list(zip(f["dataset"], f["partition"]))
    return f[f["dp"].isin(STAGE + [DIR1])]


def cellmean(f):
    """(dp, ttag, timing, policy) -> 3-seed mean acc / recovery / delta_acc."""
    g = f.groupby(["dataset", "partition", "ttag", "timing", "policy"], dropna=False)
    m = g.agg(acc=("final_acc", "mean"), rec=("recovery", "mean"),
              dacc=("delta_acc", "mean"), seeds=("seed", "nunique")).reset_index()
    m["dp"] = list(zip(m["dataset"], m["partition"]))
    return m


def _pick(cm, dp, ttag, timing, policy, col):
    q = cm[(cm.dp == dp) & (cm.ttag == ttag) & (cm.timing == timing)
           & (cm.policy == policy)]
    return None if q.empty or pd.isna(q[col].iloc[0]) else float(q[col].iloc[0])


def acc_block(cm, timing):
    """Absolute-acc table (overview 3.2.3 style): rows = P1, P1w; cols = threats."""
    md = [f"", f"**{timing}** — 절대 test acc (3-seed mean; P1=gate_v2 / P1w=gatew_v2"
          f"{' 온라인' if timing == 'online' else ' 재학습'})", ""]
    md += ["| dataset/partition | policy | " + " | ".join(THREAT_ORDER) + " |",
           "|" + "---|" * (len(THREAT_ORDER) + 2)]
    for dp in STAGE:
        for pol, lab in (("P1", "P1"), ("P2", "P1w")):
            cells = []
            for tt in THREAT_ORDER:
                v = _pick(cm, dp, tt, timing, pol, "acc")
                cells.append("" if v is None else f"{v:.4f}")
            note = " ⚠stack" if dp == IID else ""
            md.append(f"| {dp[0]}/{dp[1]}{note} | {lab} | " + " | ".join(cells) + " |")
    return md


def anchors(cnn):
    """(dp, ttag) -> (vanilla, oracle_excl) 3-seed-mean acc, for the recovery guard."""
    a = cnn[cnn["arm"].isin(("vanilla", "oracle_excl"))].copy()
    a["ttag"] = [threat_tag(t, fr) for t, fr in zip(a["threat"], a["flip_rate"])]
    out = {}
    for (ds, part, tt, arm), g in a.groupby(["dataset", "partition", "ttag", "arm"]):
        out.setdefault(((ds, part), tt), {})[arm] = float(g["final_acc"].mean())
    return {k: (v.get("vanilla"), v.get("oracle_excl")) for k, v in out.items()}


def gap_and_recovery(cm, cnn):
    """H-5 reproduction.  PRIMARY = absolute contaminated-avg (오염평균) gap P1w-P1
    (the dir1 finding's form: P1w-T1 .5913 vs P1-T1 .5843 = +0.7pt); SECONDARY =
    mean corrupt recovery, guarded to cells with |oracle-vanilla| >= REC_MIN_GAP
    (fmnist gaps are often below it -> recovery there is noise, not signal)."""
    anc = anchors(cnn)
    md, rows = [], []
    for timing in ("online", "retrain"):
        gaps, recs = {"P1": [], "P2": []}, {"P1": [], "P2": []}
        clean, dropped = {"P1": [], "P2": []}, 0
        for dp in STAGE:
            for pol in ("P1", "P2"):
                for tt in CORRUPT:
                    acc = _pick(cm, dp, tt, timing, pol, "acc")
                    if acc is not None:
                        gaps[pol].append(acc)
                    van, orc = anc.get((dp, tt), (None, None))
                    r = _pick(cm, dp, tt, timing, pol, "rec")
                    if r is not None and None not in (van, orc) and abs(orc - van) >= REC_MIN_GAP:
                        recs[pol].append(r)
                    elif r is not None:
                        dropped += 1
                    rows.append(dict(timing=timing, policy=pol, dp=f"{dp[0]}/{dp[1]}",
                                     threat=tt, acc=acc, recovery=r,
                                     gap_guarded=(None not in (van, orc)
                                                  and abs(orc - van) >= REC_MIN_GAP)))
                c = _pick(cm, dp, "clean", timing, pol, "dacc")
                if c is not None:
                    clean[pol].append(c)
        def mean(x):
            return float(np.mean(x)) if x else None
        cm_p1, cm_p1w = mean(gaps["P1"]), mean(gaps["P2"])
        acc_gap = None if None in (cm_p1, cm_p1w) else cm_p1w - cm_p1
        r_p1, r_p1w = mean(recs["P1"]), mean(recs["P2"])
        r_gap = None if None in (r_p1, r_p1w) else r_p1w - r_p1
        cln1, cln1w = mean(clean["P1"]), mean(clean["P2"])
        md.append(
            f"- **{timing}**: 오염평균 acc  P1={_f(cm_p1)}  P1w={_f(cm_p1w)}  "
            f"**gap(P1w-P1)={_f(acc_gap)}** (dir1 참조 {'+0.007' if timing=='online' else '-0.015'}) "
            f"| clean dAcc  P1={_f(cln1)}  P1w={_f(cln1w)} (band +/-{CLEAN_BAND})")
        md.append(
            f"    - recovery(guard|orc-van|>={REC_MIN_GAP}, dropped {dropped} cells): "
            f"P1={_f(r_p1)}  P1w={_f(r_p1w)}  gap={_f(r_gap)}")
    return md, pd.DataFrame(rows)


def fedif_reversal(df):
    """W-A found FedIF > flirds on P1w (dir1 오염평균 online .6011/re .6159 vs flirds
    .5913/.5959).  Report whether it reproduces on the extended stage (needs the W-D
    non-flirds sources; flirds-only W-B cannot -> 'pending W-D')."""
    fi = df[df["source"] == "fedif"]
    have_ext = not fi[fi.apply(lambda r: (r["dataset"], r["partition"]) in STAGE, axis=1)].empty
    md = ["- dir1 (W-A, on disk): FedIF P1w 오염평균 online .6011 / retrain .6159 "
          "**> flirds .5913 / .5959** → 역전 확인('타 소스 역전' 조항 해당)."]
    if have_ext:
        md.append("- 확장 무대: FedIF 셀 감지됨 — make_analysis competition CSV에서 "
                  "fedif vs flirds P1w recovery 직접 대조 가능(W-D 착지).")
    else:
        md.append("- 확장 무대: **pending W-D** — W-B는 flirds-only라 확장 셀에 FedIF 없음. "
                  "역전 재현 여부는 W-D(비-flirds 점수원) 승인·실행 후.")
    return md


def _f(v):
    return "—" if v is None else f"{v:+.3f}" if abs(v) < 1 else f"{v:.4f}"


def main():
    try:                                        # rich glyphs in the summary -> utf-8 stdout
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    cnn, _ = analyze_cnn()
    f = prep(cnn)
    cm = cellmean(f)
    cm.drop(columns=["dp"]).to_csv(OUT / "p1w_cnn.csv", index=False)

    md = ["# W-B — Flirds P1 vs P1w, 확장 CNN 무대 (rundir-only; make_p1w_cnn_table.py)",
          "",
          "> P1w ≡ 기존 P2(sign+크기가중). arm 라벨은 P2(gatew_v2/t2_signw) 유지, 표기만 "
          "P1w. dir1(W-A)은 참조로 병기. ⚠stack = cifar10 iid의 clean/fr/gn/lf T1 앵커는 "
          "B200 원본(track_g/rundirs_cnn) — W-A 드리프트≈0 판정 하에 병기.", ""]
    n_t2 = int((f["timing"] == "retrain").sum())
    md += [f"- flirds rows: {len(f)} (T2 retrain rows: {n_t2} — 0이면 sbatch_cnn_p1w.sh "
           f"미실행, T1만 표시)", ""]

    md += ["## 절대 acc (overview §3.2.3 스타일)"]
    md += acc_block(cm, "online")
    md += acc_block(cm, "retrain")

    md += ["", "## H-5 재현 — 오염 recovery P1 vs P1w + clean parity", ""]
    gr_md, gr_df = gap_and_recovery(cm, cnn)
    md += gr_md
    gr_df.to_csv(OUT / "p1w_cnn_recovery.csv", index=False)

    md += ["", "## FedIF 역전 (00-INDEX §1 '타 소스 역전' 조항)", ""]
    md += fedif_reversal(cnn)

    md += ["", "## 판정 초안 (수록 규칙 = 00-INDEX §1)", "",
           "> 사전 고정 규칙: 전 범위(W-A·W-B·L7)에서 이기면 본문 승격 / 동률이면 '부호가 "
           "가치의 대부분' ablation 1문장 / 열세·타 소스 역전 시 미수록(P1만).",
           "> W-B 단독 판정 금지 — L7(LLM P1w)·W-A 종합 후 확정. 위 gap·역전으로 초안만."]
    (OUT / "p1w_cnn_README.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[p1w] flirds rows={len(f)} (T2={n_t2}) -> {OUT}/p1w_cnn.csv "
          f"+ p1w_cnn_README.md")
    print("\n".join(gr_md))


if __name__ == "__main__":
    main()
