"""flirds-principle-analysis 부록용 load-bearing 수치 재검증 (read-only; GPU 불필요).

overview(2026-07-22)의 표 수치 중 원리 분석의 논거로 쓰는 값을 raw 파일
(rundir metrics.json / phi.parquet / analysis CSV)에서 직접 재계산해 대조한다.
실행: anaconda3 python 으로 프로젝트 루트에서
  python research-wiki/survey/flirds-principle-analysis/verify_numbers.py
출력: 같은 폴더 verification_report.txt (PASS/FAIL/INFO 행).
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.getcwd()
OUT = os.path.join(ROOT, "research-wiki", "survey", "flirds-principle-analysis")
REP = []


def check(name, claimed, computed, tol=0.006):
    try:
        ok = abs(float(claimed) - float(computed)) <= tol
    except (TypeError, ValueError):
        ok = str(claimed) == str(computed)
    REP.append(f"[{'PASS' if ok else 'FAIL'}] {name}: claimed={claimed} computed={computed}")
    return ok


def info(msg):
    REP.append(f"[INFO] {msg}")


def section(t):
    REP.append("")
    REP.append(f"===== {t} =====")


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sp(a, b):
    return spearmanr(a, b).statistic


# ---------------- A. track_d fidelity (§3.1.1) ----------------
section("A. track_d fidelity.csv — LLM 표준 fidelity vs (b)")
try:
    fd = pd.read_csv("runs/track_d/fidelity.csv")
    info(f"columns={list(fd.columns)}")
    def td(scale, regime, method, col="spearman"):
        sub = fd[fd["cell"].str.startswith(f"{scale}_{regime}_") & (fd["method"] == method)]
        return sub[col].mean(), sub[col].std(ddof=0)
    for scale, regime, method, m_cl, s_cl in [
        ("1B", "std20", "Flirds", 1.000, .000), ("1B", "std20", "Flirds1st", 0.999, .001),
        ("1B", "std20", "loss-heur", 1.000, .000), ("1B", "std20", "GTG", 0.975, .018),
        ("1B", "std20", "FedSV", 0.910, .073), ("1B", "std20", "FedIF", 0.157, .303),
        ("1B", "std20", "ShapleyFL", 0.194, .351), ("1B", "std20", "ComFedSV", 0.093, .146),
        ("7B", "std20", "Flirds", 0.999, .001), ("3B", "std20", "ComFedSV", -0.137, .065),
        ("1B", "anchor5", "Flirds", 1.000, .000), ("1B", "anchor5", "FedSV", 0.700, .163),
        ("1B", "anchor5", "ShapleyFL", 0.700, .283), ("1B", "anchor5", "ComFedSV", 0.500, .432),
        ("1B", "anchor5", "FedIF", 0.067, .531), ("3B", "anchor5", "GTG", 0.967, .047),
        ("3B", "anchor5", "ShapleyFL", 0.167, .094), ("7B", "anchor5", "Flirds", 1.000, .000),
    ]:
        m, s = td(scale, regime, method)
        check(f"{scale} {regime} {method} Sp mean", m_cl, m)
        check(f"{scale} {regime} {method} Sp std", s_cl, s)
except Exception as e:
    info(f"A ERROR: {e!r}")

# ---------------- B. anchor5 vs (a) retrain oracle ----------------
section("B. 1B anchor5 — 방법별 vs (a) (phi.parquet 재계산)")
try:
    rows = {m: [] for m in ["Flirds", "Flirds1st", "loss-heur", "GTG", "FedSV", "(b)oracle"]}
    for s in [0, 1, 2]:
        df = pd.read_parquet(f"runs/track_d/rundirs/1B_anchor5_seed{s}/phi.parquet")
        piv = df.pivot_table(index="client", columns="method", values="phi")
        for m in rows:
            rows[m].append(sp(piv[m if m != "Flirds1st" else "Flirds1st"], piv["(a)oracle"]))
    for m, cl in [("Flirds", 0.933), ("Flirds1st", 0.933), ("loss-heur", 0.933),
                  ("GTG", 0.933), ("FedSV", 0.733), ("(b)oracle", 0.933)]:
        check(f"anchor5 {m} vs (a) Sp mean", cl, np.mean(rows[m]))
    check("anchor5 (a)vs(b) Sp std", 0.047, np.std(rows["(b)oracle"]))
except Exception as e:
    info(f"B ERROR: {e!r}")

# ---------------- C. track_c CNN fidelity ----------------
section("C. track_c fidelity.csv — CNN 듀얼 오라클 pool")
try:
    fc = pd.read_csv("runs/track_c/fidelity.csv")
    info(f"columns={list(fc.columns)}; n={len(fc)}")
    def cpool(method, col):
        return fc[fc["method"] == method][col].mean(), fc[fc["method"] == method][col].std(ddof=0)
    for meth, cl_b, cl_a in [("Flirds", 0.919, 0.352), ("Flirds1st", 0.832, 0.408),
                             ("loss-heur", 0.860, 0.425), ("GTG", 0.569, 0.374),
                             ("FedSV", 0.401, 0.284), ("ComFedSV", 0.348, 0.338),
                             ("ShapleyFL", 0.391, 0.453), ("FedIF", 0.491, 0.380)]:
        m, s = cpool(meth, "spearman_b"); check(f"C1 pool {meth} vs(b) Sp", cl_b, m)
        m2, _ = cpool(meth, "spearman_a"); check(f"C1 pool {meth} vs(a) Sp", cl_a, m2)
    ni = fc[fc["scenario"] != "iid"]
    check("C1 iid-제외 Flirds vs(b)", 0.928, ni[ni["method"] == "Flirds"]["spearman_b"].mean())
    def scen(ds, sc, meth):
        sub = fc[(fc["dataset"] == ds) & (fc["scenario"] == sc) & (fc["method"] == meth)]
        return sub["spearman_b"].mean()
    check("cifar10/label_flip Flirds", 1.00, scen("cifar10", "label_flip", "Flirds"), tol=0.005)
    check("cifar10/iid Flirds", 0.95, scen("cifar10", "iid", "Flirds"), tol=0.005)
    check("mnist/label_skew Flirds", 0.71, scen("mnist", "label_skew", "Flirds"), tol=0.005)
    check("cifar10/quantity_skew FedIF", -0.20, scen("cifar10", "quantity_skew", "FedIF"), tol=0.005)
except Exception as e:
    info(f"C ERROR: {e!r}")

# ---------------- D. CNN stability (RESULTS.txt 인쇄값 스팟) ----------------
section("D. track_c RESULTS.txt — stability 스팟 (grep)")
try:
    txt = open("runs/track_c/RESULTS.txt", encoding="utf-8").read()
    for token in ["0.547", "0.518", "0.124"]:
        info(f"RESULTS.txt contains '{token}': {token in txt}")
except Exception as e:
    info(f"D ERROR: {e!r}")

# ---------------- E. probe std50k5 / rank / lr grid ----------------
section("E. probe_signal — 참여·rank·lr lever")
try:
    m = jload("runs/probe_signal/rundirs/1B_std50k5_r16_seed0/metrics.json")["seed0"]["spearman"]
    for meth, cl in [("Flirds", 1.000), ("Flirds1st", 1.000), ("loss-heur", 1.000),
                     ("GTG", 0.983), ("FedSV", 0.910), ("FedIF", -0.040),
                     ("ShapleyFL", -0.064), ("ComFedSV", -0.109)]:
        key = meth if meth in m else meth.replace("loss-heur", "loss-heur")
        check(f"std50k5 r16 s0 {meth} Sp", cl, m[key])
    if "Fed-LOO" in m:
        check("std50k5 r16 s0 Fed-LOO Sp", 0.9998, m["Fed-LOO"], tol=0.001)
    # 3-seed Flirds + (b) xseed rho
    fl = []
    phis = {}
    for s in [0, 1, 2]:
        mm = jload(f"runs/probe_signal/rundirs/1B_std50k5_r16_seed{s}/metrics.json")[f"seed{s}"]
        fl.append(mm["spearman"]["Flirds"])
        df = pd.read_parquet(f"runs/probe_signal/rundirs/1B_std50k5_r16_seed{s}/phi.parquet")
        piv = df.pivot_table(index="client", columns="method", values="phi")
        phis[s] = piv["(b)oracle"]
    check("std50k5 r16 3-seed Flirds Sp mean", 1.000, np.mean(fl), tol=0.001)
    pairs = [sp(phis[a], phis[b]) for a, b in [(0, 1), (0, 2), (1, 2)]]
    info(f"std50k5 (b) xseed rho pairs = {[round(x,3) for x in pairs]} (claimed -0.09/+0.13/+0.15)")
    check("std50k5 (b) xseed rho mean", 0.06, np.mean(pairs), tol=0.02)
    # rank probe r64 FedSV
    m64 = jload("runs/probe_signal/rundirs/1B_anchor5_r64_seed0/metrics.json")["seed0"]["spearman"]
    check("anchor5 r64 s0 FedSV Sp", 0.5, m64["FedSV"], tol=0.001)
    check("anchor5 r64 s0 Flirds Sp", 1.0, m64["Flirds"], tol=0.001)
    # lr grid (b) phi range s0: lr1e-3 st10 = track_d anchor5 seed0 재사용
    def brange(pq):
        df = pd.read_parquet(pq)
        b = df[df["method"] == "(b)oracle"].set_index("client")["phi"]
        return float(b.max() - b.min())
    check("lr1e-3 st10 s0 (b)range", 0.00119, brange("runs/track_d/rundirs/1B_anchor5_seed0/phi.parquet"), tol=0.00002)
    check("lr3e-3 st10 s0 (b)range", 0.00330, brange("runs/probe_signal/rundirs/1B_anchor5_lr3e-3_st10_seed0/phi.parquet"), tol=0.00002)
except Exception as e:
    info(f"E ERROR: {e!r}")

# ---------------- F. B축 매트릭스 — (b) xseed rho + AUROC ----------------
section("F. B축 2x2 — (b) 자기순위 xseed rho + 탐지 AUROC")
try:
    def bxseed(cell, threat_key=None):
        df = pd.read_parquet(f"runs/phase2_matrix/rundirs/{cell}/phi.parquet")
        if threat_key is not None and "threat" in df.columns:
            df = df[df["threat"] == threat_key]
        if "kind" in df.columns:
            df = df[df["kind"] == "val"] if (df["kind"] == "val").any() else df
        piv = df[df["method"] == "(b)oracle"].pivot_table(index="client", columns="seed", values="phi")
        ps = [sp(piv[a], piv[b]) for a, b in itertools.combinations(piv.columns, 2)]
        return float(np.mean(ps))
    for cell, th, cl in [("1B_iid5_clean", "clean", 0.13), ("1B_iid5_noisy", "noisy", 0.60),
                         ("1B_iid5_frzero", "freerider_zero", 0.70), ("1B_silo5_clean", "clean", 0.87),
                         ("1B_silo5_noisy", "noisy", 0.93), ("1B_silo5_frzero", "freerider_zero", 0.93)]:
        try:
            v = bxseed(cell, th)
        except Exception:
            v = bxseed(cell, None)
        check(f"xseed rho (b) {cell}", cl, v, tol=0.011)
    def auroc3(cell, meth):
        mm = jload(f"runs/phase2_matrix/rundirs/{cell}/metrics.json")
        vals = []
        for k, v in mm.items():
            if isinstance(v, dict) and "auroc" in v and meth in v["auroc"]:
                vals.append(v["auroc"][meth])
        return float(np.mean(vals)), len(vals)
    for cell, meth, cl in [("1B_iid5_noisy", "Flirds", 1.00), ("1B_iid5_noisy", "FedDQC", 1.00),
                           ("1B_silo5_noisy", "FedDQC", 0.92), ("1B_iid5_frzero", "FedDQC", 0.58),
                           ("1B_silo5_frzero", "FedDQC", 0.75), ("1B_silo5_noisy", "STD-DAGMM", 0.25),
                           ("1B_silo5_frzero", "STD-DAGMM", 0.00), ("1B_silo5_noisy", "FLTrust", 1.00)]:
        v, n = auroc3(cell, meth)
        check(f"AUROC {cell} {meth} (n={n})", cl, v, tol=0.007)
except Exception as e:
    info(f"F ERROR: {e!r}")

# ---------------- G. silo5 오염 무대 (§3.3.1) ----------------
section("G. phase2 silo5 — AUROC/Sp/runtime")
try:
    def silo(cell):
        return jload(f"runs/phase2_matrix/rundirs/{cell}/metrics.json")
    mm = silo("1B_silo5_noisy")
    aur, spv, rt = {}, {}, {}
    for k, v in mm.items():
        if not isinstance(v, dict):
            continue
        for meth in v.get("auroc", {}):
            aur.setdefault(meth, []).append(v["auroc"][meth])
        for meth in v.get("spearman", {}):
            spv.setdefault(meth, []).append(v["spearman"][meth])
        for meth in v.get("runtime", {}):
            rt.setdefault(meth, []).append(v["runtime"][meth])
    for meth in ["Flirds", "Flirds1st", "loss-heur", "Fed-LOO", "GTG", "ShapleyFL"]:
        check(f"silo5 noisy {meth} AUROC", 1.000, np.mean(aur[meth]), tol=0.001)
        check(f"silo5 noisy {meth} Sp", 1.000, np.mean(spv[meth]), tol=0.001)
    check("silo5 noisy FedIF Sp", 0.933, np.mean(spv["FedIF"]))
    check("silo5 noisy ComFedSV Sp", 0.833, np.mean(spv["ComFedSV"]))
    check("silo5 noisy FLDetector AUROC", 0.750, np.mean(aur["FLDetector"]))
    check("silo5 noisy STD-DAGMM AUROC", 0.250, np.mean(aur["STD-DAGMM"]))
    info(f"silo5 noisy runtime: Flirds={np.mean(rt['Flirds']):.0f}s Flirds1st={np.mean(rt['Flirds1st']):.0f}s "
         f"loss-heur={np.mean(rt['loss-heur']):.0f}s Fed-LOO={np.mean(rt['Fed-LOO']):.0f}s "
         f"GTG={np.mean(rt['GTG']):.0f}s (b)={np.mean(rt['(b)oracle']):.0f}s FedDQC={np.mean(rt['FedDQC']):.0f}s")
    # frzero Sp
    mm = silo("1B_silo5_frzero")
    spz = {}
    for k, v in mm.items():
        if isinstance(v, dict):
            for meth in v.get("spearman", {}):
                spz.setdefault(meth, []).append(v["spearman"][meth])
    check("silo5 frzero FedSV Sp", 0.933, np.mean(spz["FedSV"]))
    check("silo5 frzero FedIF Sp", 0.900, np.mean(spz["FedIF"]))
    # noisy 클라 φ 부호·순위 (파생: 게이트 침묵 원리)
    df = pd.read_parquet("runs/phase2_matrix/rundirs/1B_silo5_noisy/phi.parquet")
    df = df[(df["method"] == "(b)oracle")]
    if "kind" in df.columns and (df["kind"] == "val").any():
        df = df[df["kind"] == "val"]
    piv = df.pivot_table(index="client", columns="seed", values="phi")
    info(f"silo5 noisy (b) phi per seed (client0=noisy):\n{piv.round(5).to_string()}")
    info(f"noisy client0 = max-phi(최소 기여)인 seed 수: {(piv.loc[0] == piv.max(axis=0)).sum()}/3; "
         f"전 클라 phi<0(전원 loss 감소 기여): {(piv < 0).all().all()}")
except Exception as e:
    info(f"G ERROR: {e!r}")

# ---------------- H. device100 anchor + frdelta ----------------
section("H. device100 anchor(a0.5) + frdelta")
try:
    mm = jload("runs/phase2_matrix/rundirs/1B_device100-a0.5_noisy_anchor/metrics.json")
    aur, spv, rt = {}, {}, {}
    for k, v in mm.items():
        if not isinstance(v, dict):
            continue
        for meth in v.get("auroc", {}):
            aur.setdefault(meth, []).append(v["auroc"][meth])
        for meth in v.get("spearman", {}):
            spv.setdefault(meth, []).append(v["spearman"][meth])
        for meth in v.get("runtime", {}):
            rt.setdefault(meth, []).append(v["runtime"][meth])
    check("d100 noisy anchor (b) AUROC", 0.604, np.mean(aur["(b)oracle"]))
    info(f"d100 (b) AUROC per-seed = {[round(x,3) for x in aur['(b)oracle']]} (claimed .660/.563/.589)")
    check("d100 noisy anchor FedDQC AUROC", 1.000, np.mean(aur["FedDQC"]), tol=0.001)
    check("d100 noisy anchor FedIF AUROC", 0.830, np.mean(aur["FedIF"]))
    check("d100 noisy anchor FLTrust AUROC", 0.854, np.mean(aur["FLTrust"]))
    check("d100 noisy anchor GTG Sp", 0.784, np.mean(spv["GTG"]))
    check("d100 noisy anchor ShapleyFL Sp", 0.582, np.mean(spv["ShapleyFL"]))
    check("d100 noisy anchor ComFedSV Sp", -0.023, np.mean(spv["ComFedSV"]))
    check("d100 noisy anchor Flirds Sp", 1.000, np.mean(spv["Flirds"]), tol=0.001)
    info(f"d100 anchor runtime: (b)={np.mean(rt['(b)oracle']):.0f}s Flirds={np.mean(rt['Flirds']):.0f}s "
         f"Flirds1st={np.mean(rt['Flirds1st']):.0f}s GTG={np.mean(rt['GTG']):.0f}s FedSV={np.mean(rt['FedSV']):.0f}s "
         f"ShapleyFL={np.mean(rt['ShapleyFL']):.0f}s")
    # frdelta
    mm = jload("runs/phase2_matrix/rundirs_2026-07/1B_silo5_frdelta/metrics.json")
    aur, spv = {}, {}
    for k, v in mm.items():
        if not isinstance(v, dict):
            continue
        for meth in v.get("auroc", {}):
            aur.setdefault(meth, []).append(v["auroc"][meth])
        for meth in v.get("spearman", {}):
            spv.setdefault(meth, []).append(v["spearman"][meth])
    for meth, cl in [("(b)oracle", 0.333), ("Flirds", 0.333), ("STD-DAGMM", 1.000),
                     ("FedIF", 0.000), ("FLDetector", 0.000), ("FLTrust", 0.000), ("FedDQC", 0.750)]:
        check(f"frdelta {meth} AUROC", cl, np.mean(aur[meth]), tol=0.001)
    check("frdelta Flirds Sp", 1.000, np.mean(spv["Flirds"]), tol=0.001)
    df = pd.read_parquet("runs/phase2_matrix/rundirs_2026-07/1B_silo5_frdelta/phi.parquet")
    b = df[df["method"] == "(b)oracle"]
    if "kind" in b.columns and (b["kind"] == "val").any():
        b = b[b["kind"] == "val"]
    info(f"frdelta (b) phi 전 클라 음수(전원 기여): {(b['phi'] < 0).all()}; min={b['phi'].min():.5f} max={b['phi'].max():.5f}")
except Exception as e:
    info(f"H ERROR: {e!r}")

# ---------------- I. E4 Fed-LOO 스위트 ----------------
section("I. E4 Fed-LOO (rundirs_e4_fedloo)")
try:
    for regime, cl_rt in [("anchor5", {"Flirds1st": 232, "Flirds": 716, "loss-heur": 657, "Fed-LOO": 773, "(b)oracle": 3568}),
                          ("std20", {"Flirds1st": 1535, "Flirds": 4703, "loss-heur": 2199, "Fed-LOO": 2925, "(b)oracle": 2943})]:
        sps, rts = {}, {}
        for s in [0, 1, 2]:
            mm = jload(f"runs/track_d/rundirs_e4_fedloo/1B_{regime}_seed{s}/metrics.json")[f"seed{s}"]
            for meth, v in mm["spearman"].items():
                sps.setdefault(meth, []).append(v)
            for meth, v in mm["runtime"].items():
                rts.setdefault(meth, []).append(v)
        check(f"E4 {regime} Fed-LOO Sp", 1.000, np.mean(sps["Fed-LOO"]), tol=0.001)
        check(f"E4 {regime} Flirds Sp", 1.000, np.mean(sps["Flirds"]), tol=0.003)
        for meth, cl in cl_rt.items():
            check(f"E4 {regime} {meth} runtime", cl, np.mean(rts[meth]), tol=cl * 0.03)
except Exception as e:
    info(f"I ERROR: {e!r}")

# ---------------- J. track_g 게이팅 ----------------
section("J. track_g — LLM recovery / CNN 게이트 grid")
try:
    lg = pd.read_csv("runs/track_g/analysis/llm_summary.csv")
    info(f"llm_summary cols={list(lg.columns)}")
    def rec(regime, threat, arm):
        sub = lg[(lg["regime"] == regime) & (lg["threat"] == threat) & (lg["arm"] == arm)]
        return sub["recovery"].mean() if "recovery" in lg.columns and len(sub) else np.nan
    for reg, th, arm, cl in [("silo5", "frzero", "flirds_gate_v2", 1.000),
                             ("iid5", "frzero", "flirds_gate_v2", 1.000),
                             ("silo5", "frzero", "flirds_gate_v1", 0.898),
                             ("silo5", "frrand", "flirds_gate_v2", 0.462),
                             ("silo5", "noisy", "flirds_gate_v2", 0.000),
                             ("silo5", "frzero", "v3_sign", 1.000),
                             ("iid5", "frzero", "random_excl", 0.185)]:
        check(f"track_g rec {reg}/{th}/{arm}", cl, rec(reg, th, arm), tol=0.02)
    cg = pd.read_csv("runs/track_g/analysis/cnn_summary.csv")
    info(f"cnn_summary cols={list(cg.columns)}")
    def acc(part, threat, arm):
        sub = cg[(cg["partition"] == part) & (cg["threat"] == threat) & (cg["arm"] == arm)]
        col = "final_acc" if "final_acc" in cg.columns else "acc"
        return sub[col].mean() if len(sub) else np.nan
    for part, th, arm, cl in [("iid", "grad_noise", "flirds_gate_v2", .6143),
                              ("dir1", "grad_noise", "flirds_gate_v2", .5668),
                              ("iid", "grad_noise", "vanilla", .2564),
                              ("dir1", "grad_noise", "vanilla", .2436),
                              ("iid", "clean", "vanilla", .6488),
                              ("dir1", "clean", "vanilla", .6389),
                              ("iid", "clean", "flirds_gate_v2", .6428),
                              ("dir1", "clean", "flirds_gate_v2", .6315),
                              ("dir1", "free_rider", "flirds_gate_v2", .6148),
                              ("iid", "free_rider", "oracle_excl", .6356),
                              ("dir1", "grad_noise", "oracle_excl", .6203)]:
        check(f"track_g CNN {part}/{th}/{arm} acc", cl, acc(part, th, arm), tol=0.003)
except Exception as e:
    info(f"J ERROR: {e!r}")

# ---------------- K. track_h 경쟁 + R4 ----------------
section("K. track_h — 점수원 경쟁 / R4 gsm50k5 / P5 / scale / dyn")
try:
    cc = pd.read_csv("runs/track_h/analysis/cnn_competition.csv")
    info(f"cnn_competition cols={list(cc.columns)}")
    def comp(policy, timing, source, threat):
        sub = cc[(cc["policy"] == policy) & (cc["timing"] == timing)
                 & (cc["source"] == source) & (cc["threat"] == threat)
                 & (cc["partition"] == "dir1")]
        if threat == "label_flip" and "flip_rate" in cc.columns and sub["flip_rate"].notna().any():
            sub = sub[sub["flip_rate"] == 0.7]
        col = "final_acc" if "final_acc" in cc.columns else "acc"
        return sub[col].mean() if len(sub) else np.nan
    for pol, tim, src, th, cl in [
        ("P1", "online", "flirds", "free_rider", .6148), ("P1", "online", "flirds", "grad_noise", .5668),
        ("P1", "online", "flirds1st", "grad_noise", .2479), ("P1", "online", "lossheur", "grad_noise", .5981),
        ("P1", "online", "fedif", "grad_noise", .2479), ("P1", "online", "gtg", "free_rider", .3915),
        ("P1", "online", "shapleyfl", "free_rider", .4020), ("P1", "online", "shapleyfl", "grad_noise", .6115),
        ("P1", "retrain", "flirds", "grad_noise", .6065), ("P1", "retrain", "flirds", "label_flip", .6192),
        ("P1", "retrain", "flirds1st", "label_flip", .6236), ("P1", "retrain", "gtg", "grad_noise", .6203),
        ("P2", "online", "fedif", "grad_noise", .6043), ("P2", "retrain", "fedif", "grad_noise", .6114),
    ]:
        check(f"track_h {pol}/{tim}/{src}/{th}", cl, comp(pol, tim, src, th), tol=0.003)
    oz = pd.read_csv("runs/track_h/analysis/observer_zero_semantics.csv")
    info(f"observer_zero_semantics cols={list(oz.columns)}; head=\n{oz.head(12).to_string()}")
    lc = pd.read_csv("runs/track_h/analysis/llm_competition.csv")
    info(f"llm_competition cols={list(lc.columns)}")
    def llmc(arm):
        sub = lc[(lc["arm"] == arm) & (lc["regime"] == "silo5") & (lc["threat"] == "noisy")]
        if "nr" in lc.columns:
            sub = sub[sub["nr"] == 1.0]
        col = [c for c in ["final_val_loss", "val_loss", "final_loss"] if c in lc.columns][0]
        return sub[col].mean() if len(sub) else np.nan
    for arm, cl in [("gtg_gate_v2", 2.3310), ("fedsv_gate_v2", 2.3308), ("shapleyfl_gate_v2", 2.3309)]:
        check(f"track_h R3 {arm} val-loss", cl, llmc(arm), tol=0.0002)
    ga = pd.read_csv("runs/track_h/analysis/gsm50k5_tier_a.csv")
    info(f"gsm50k5_tier_a cols={list(ga.columns)}; n={len(ga)}")
    def gsm(threat, arm):
        sub = ga[(ga["threat"] == threat) & (ga["arm"] == arm)]
        return sub["gsm8k_em"].mean() if len(sub) else np.nan
    for th, arm, cl in [("noisy", "observer", .3342), ("noisy", "oracle_excl", .3700),
                        ("noisy", "t2_sign_flirds", .3584), ("noisy", "t2_sign_lossheur", .3548),
                        ("noisy", "t2_sign_flirds1st", .3432), ("noisy", "t2_sign_fedif", .3432),
                        ("noisy", "t2_random_k37", .3110), ("noisy", "flirds_gate_v2", .3530),
                        ("clean", "observer", .3771), ("clean", "flirds_gate_v2", .3673),
                        ("frzero", "observer", .3601), ("frzero", "oracle_excl", .3691),
                        ("frzero", "t2_sign_flirds", .3691), ("frzero", "t2_random_k30", .3566)]:
        check(f"R4 {th}/{arm} EM", cl, gsm(th, arm), tol=0.0005)
    kept_noisy = {a: ga[(ga["threat"] == "noisy") & (ga["arm"] == a)]["kept"].mean()
                  for a in ["t2_sign_flirds", "t2_sign_lossheur", "t2_sign_flirds1st", "t2_sign_fedif"]}
    info(f"R4 noisy kept = {kept_noisy} (claimed 37/36/38/34)")
    kept_fz = ga[(ga["threat"] == "frzero") & (ga["arm"].str.startswith("t2_sign"))]["kept"].tolist()
    info(f"R4 frzero kept = {kept_fz} (claimed 전부 30)")
    # P5 / scale / dyn
    def comp_p5(policy, timing, source, threat):
        return comp(policy, timing, source, threat)
    for pol, tim, src, th, cl in [
        ("P5h", "retrain", "flirds", "grad_noise", .6215), ("P5h", "retrain", "flirds", "free_rider", .6197),
        ("P5h", "online", "flirds", "grad_noise", .5169), ("P5h", "online", "lossheur", "grad_noise", .3959),
        ("P5s", "online", "flirds", "grad_noise", .5416), ("P5s", "online", "lossheur", "grad_noise", .5373),
        ("P5h", "online", "flirds", "clean", .6375), ("P5s", "retrain", "flirds", "label_flip", .6188),
    ]:
        check(f"track_h P5 {pol}/{tim}/{src}/{th}", cl, comp_p5(pol, tim, src, th), tol=0.003)
    sa = pd.read_csv("runs/track_h/scale/analysis/scale_acc.csv")
    info(f"scale_acc cols={list(sa.columns)}")
    def sacc(arm, threat):
        sub = sa[(sa["arm"] == arm) & (sa["threat"] == threat)]
        col = [c for c in ["final_acc", "acc"] if c in sa.columns][0]
        return sub[col].mean() if len(sub) else np.nan
    for arm, th, cl in [("flirds_pweight", "label_flip", .6220), ("flirds_pweight", "free_rider", .6268),
                        ("flirds_pweight", "grad_noise", .6107), ("observer", "grad_noise", .5497),
                        ("oracle_excl", "label_flip", .6301), ("random_excl", "grad_noise", .5136),
                        ("flirds_gate_v2", "grad_noise", .6102), ("flirds_cgate", "label_flip", .6008)]:
        check(f"scale {arm}/{th}", cl, sacc(arm, th), tol=0.003)
    da = pd.read_csv("runs/track_h/dyn/analysis/dyn_acc.csv")
    dp4 = pd.read_csv("runs/track_h/dyn/analysis/dyn_dp4.csv")
    info(f"dyn_acc cols={list(da.columns)}; dyn_dp4=\n{dp4.to_string()}")
    def dacc(arm, threat):
        sub = da[(da["arm"] == arm) & (da["threat"] == threat)]
        col = [c for c in ["final_acc", "acc"] if c in da.columns][0]
        return sub[col].mean() if len(sub) else np.nan
    for arm, th, cl in [("flirds_gate_v2", "grad_noise", .1771), ("flirds_pweight", "grad_noise", .1902),
                        ("oracle_excl", "label_flip", .6456), ("flirds_gate_v2", "free_rider", .6253),
                        ("flirds_pweight", "label_flip", .5682), ("vanilla", "grad_noise", .2547)]:
        check(f"dyn {arm}/{th}", cl, dacc(arm, th), tol=0.004)
    hit = dp4.groupby("threat")["p1_excl_hit_rate"].mean()
    info(f"dyn DP-4 hit-rate by threat = {hit.round(3).to_dict()} (claimed .405 전 위협; 우연=.40)")
except Exception as e:
    info(f"K ERROR: {e!r}")

# ---------------- L. removal_dose ----------------
section("L. removal_dose — A2/A3/dose/AdamW")
try:
    # A2: worst/best-first delta val loss (3-seed, Flirds)
    def a2(threat, key):
        ws, bs = [], []
        for s in [0, 1, 2]:
            mm = jload(f"runs/removal_dose/rundirs/1B_silo5_{threat}_removal_seed{s}/metrics.json")
            k = list(mm)[0]
            cv = mm[k]["removal_curve"]["Flirds"]
            w = np.array([v for _, v in cv["worst_first"]])
            b = np.array([v for _, v in cv["best_first"]])
            ws.append(w[0] - w[-1])                  # L(k=0) − L(k=끝) = 제거로 내린 val-loss
            bs.append(b[0] - b[-1])
        return np.mean(ws), np.mean(bs)
    for th, clw, clb in [("noisy", 0.0076, -0.0084), ("frrand", 0.0071, -0.0015), ("frzero", 0.0067, -0.0016)]:
        w, b = a2(th, None)
        check(f"A2 {th} worst-first dVal(mean-vs-k0)", clw, w, tol=0.002)
        check(f"A2 {th} best-first dVal(mean-vs-k0)", clb, b, tol=0.002)
    # dose: noisy ladder AUROC
    def dose_auroc(nr):
        vals = []
        for s in [0, 1, 2]:
            mm = jload(f"runs/removal_dose/rundirs/1B_silo5_noisy_dose_nr{nr}_seed{s}/metrics.json")
            k = list(mm)[0]
            vals.append(mm[k]["auroc"]["Flirds"])
        return np.mean(vals)
    check("dose noisy nr0.1 Flirds AUROC", 0.75, dose_auroc("0.1"), tol=0.001)
    check("dose noisy nr0.25 Flirds AUROC", 1.00, dose_auroc("0.25"), tol=0.001)
    # AdamW
    fl, ab = [], []
    for s in [0, 1, 2]:
        mm = jload(f"runs/removal_dose/rundirs_trackd/1B_anchor5_adamw_seed{s}/metrics.json")[f"seed{s}"]
        fl.append(mm["spearman"]["Flirds"])
        if "(a)oracle" in mm["spearman"]:
            ab.append(mm["spearman"]["(a)oracle"])
    check("AdamW Flirds Sp mean", 0.767, np.mean(fl))
    info(f"AdamW Flirds per-seed = {fl} (claimed +0.90/+0.50/+0.90)")
    if ab:
        check("AdamW (a) vs (b) Sp mean", -0.533, np.mean(ab), tol=0.01)
        info(f"AdamW (a)oracle per-seed = {ab} (claimed -0.10/-0.90/-0.60)")
    # A3 CNN removal: cifar10 label_flip Flirds rho + acc gap
    def a3(ds, scen):
        sps, gaps, gaps_b = [], [], []
        for s in [0, 1, 2]:
            mm = jload(f"runs/removal_dose/rundirs_cnn/{ds}_{scen}_seed{s}/metrics.json")
            me = mm["methods"]
            sps.append(sp(np.array(me["Flirds"]["phi"]), np.array(me["(b)oracle"]["phi"])))
            for meth, acc_out in [("Flirds", gaps), ("(b)oracle", gaps_b)]:
                cv = mm["removal_curve_acc"][meth]
                w = np.array([v for _, v in cv["worst_first"]])
                b = np.array([v for _, v in cv["best_first"]])
                acc_out.append(float(np.mean(w - b)))
        return np.mean(sps), np.mean(gaps), np.mean(gaps_b)
    r, g1, gb = a3("cifar10", "label-flip")
    check("A3 cifar10 lf Flirds rho vs(b)", 1.00, r, tol=0.005)
    check("A3 cifar10 lf acc gap(Flirds)", 0.0445, g1, tol=0.003)
    info(f"A3 cifar10 lf acc gap (b)oracle = {gb:.4f} (claimed 0.038~0.045 동급)")
    r, g1, gb = a3("cifar10", "iid")
    check("A3 cifar10 iid acc gap(Flirds)", -0.0033, g1, tol=0.002)
    info(f"A3 cifar10 iid: Flirds rho={r:.2f} (claimed +0.97), (b) gap={gb:.4f} (claimed -0.0027)")
    r, g1, gb = a3("mnist", "label-flip")
    check("A3 mnist lf acc gap(Flirds)", 0.0035, g1, tol=0.002)
    check("A3 mnist lf Flirds rho vs(b)", 1.00, r, tol=0.005)
except Exception as e:
    info(f"L ERROR: {e!r}")

# ---------------- M. target stability ----------------
section("M. (b) target self-stability (Exp C)")
try:
    ts = pd.read_csv("runs/track_d/target_stability.csv")
    info(f"track_d target_stability cols={list(ts.columns)}\n{ts.to_string()}")
    ts2 = pd.read_csv("runs/phase2_matrix/target_stability.csv")
    info(f"phase2 target_stability:\n{ts2.to_string()}")
except Exception as e:
    info(f"M ERROR: {e!r}")

# ---------------- N. microbench / op-count ----------------
section("N. microbench")
try:
    mb = jload("runs/measured_2026-07/microbench/summary.json")
    info(f"microbench = {json.dumps(mb)[:600]}")
except Exception as e:
    info(f"N ERROR: {e!r}")

# ---------------- O. 감사(sign audit) 스팟 ----------------
section("O. track_g audit — frzero exact-0 / clean 전원 양수 스팟")
try:
    st = pd.read_csv("runs/track_g/audit/sign_table.csv")
    info(f"sign_table cols={list(st.columns)}; n={len(st)}")
except Exception as e:
    info(f"O ERROR: {e!r}")

# ---------------- P. 파생: 메커니즘 판별 계산 ----------------
section("P. 파생 계산 — 메커니즘 판별 (min-max+EMA vs plain-sum)")
try:
    # P-1: std20에서 FedIF vs Flirds1st (같은 1차 신호; 차이는 min-max+EMA 후처리)
    for cell, label in [("1B_std20", "std20(부분참여 2/20, R=200)"), ("1B_anchor5", "anchor5(전원, R=30)")]:
        cors = []
        for s in [0, 1, 2]:
            df = pd.read_parquet(f"runs/track_d/rundirs/{cell}_seed{s}/phi.parquet")
            piv = df.pivot_table(index="client", columns="method", values="phi")
            # FedIF는 good->HIGH 라 부호 반전 후 비교(순위 절대값 무관: spearman에 부호만 영향)
            cors.append(sp(-piv["FedIF"], piv["Flirds1st"]))
        info(f"P-1 {label}: Spearman(-FedIF, Flirds1st) per-seed = {[round(c,3) for c in cors]} mean={np.mean(cors):.3f}")
    # P-2: silo5(신호 있는 무대)에서 같은 비교
    df = pd.read_parquet("runs/phase2_matrix/rundirs/1B_silo5_noisy/phi.parquet")
    if "kind" in df.columns and (df["kind"] == "val").any():
        df = df[df["kind"] == "val"]
    cors = []
    for s in [0, 1, 2]:
        piv = df[df["seed"] == s].pivot_table(index="client", columns="method", values="phi")
        cors.append(sp(-piv["FedIF"], piv["Flirds1st"]))
    info(f"P-2 silo5 noisy: Spearman(-FedIF, Flirds1st) per-seed = {[round(c,3) for c in cors]}")
    # P-3: CNN c1 label_flip — same-game 근접도 vs renorm 산포 (셀별 Spearman(Flirds, X))
    cells = sorted(glob.glob("runs/track_c/c1/cifar10_label-flip_seed*/metrics.json"))
    prs = {m: [] for m in ["loss-heur", "Flirds1st", "GTG", "FedSV", "ShapleyFL", "(b)oracle"]}
    for c in cells:
        mm = jload(c)["methods"]
        f = np.array(mm["Flirds"]["phi"])
        for m in prs:
            prs[m].append(sp(f, np.array(mm[m]["phi"])))
    info("P-3 CNN cifar10/label_flip: Spearman(Flirds, X) 3-seed mean = " +
         ", ".join(f"{m}={np.mean(v):.3f}" for m, v in prs.items()))
except Exception as e:
    info(f"P ERROR: {e!r}")

# ---------------- Q. 추가 검증 — CNN 부분참여 붕괴 / frzero exact-0 / noisy 0-교차 ----------------
section("Q. CNN k-sweep(§4.3) / frzero exact-0(§5.2) / noisy phi 부호 ladder")
try:
    # Q-1: pc1 label-flip 폭 pool, 참여 k별 method fidelity
    def pc1_pool(k):
        res = {}
        for w in ["0.5", "1", "2", "4"]:
            for s in [0, 1, 2]:
                if w == "1" and k == "1.0":
                    p = f"runs/track_c/c1/cifar10_label-flip_seed{s}/metrics.json"
                else:
                    p = f"runs/probe_signal/cnn_c1/pc1_cifar10_label-flip_w{w}_k{k}_seed{s}/metrics.json"
                if not os.path.exists(p):
                    continue
                me = jload(p)["methods"]
                for meth in ["Flirds", "Flirds1st", "loss-heur", "GTG", "FedSV", "FedIF"]:
                    res.setdefault(meth, []).append(me[meth]["spearman_b"])
        return {m: (np.mean(v), len(v)) for m, v in res.items()}
    p02 = pc1_pool("0.2"); p10 = pc1_pool("1.0")
    check("pc1 lf k=0.2 Flirds Sp", 0.891, p02["Flirds"][0])
    check("pc1 lf k=0.2 Flirds1st Sp", 0.305, p02["Flirds1st"][0])
    check("pc1 lf k=0.2 loss-heur Sp", 0.862, p02["loss-heur"][0])
    check("pc1 lf k=1.0 Flirds1st Sp", 0.940, p10["Flirds1st"][0])
    check("pc1 lf k=1.0 GTG Sp", 0.497, p10["GTG"][0])
    info(f"pc1 pool n: k0.2={p02['Flirds'][1]} k1.0={p10['Flirds'][1]} (기대 12)")
    # Q-2: frzero bit-exact 0 (silo5 frzero phi.parquet, client1 = free-rider)
    df = pd.read_parquet("runs/phase2_matrix/rundirs/1B_silo5_frzero/phi.parquet")
    if "kind" in df.columns and (df["kind"] == "val").any():
        df = df[df["kind"] == "val"]
    fr = df[df["client"] == 1]
    ex0 = {m: bool((fr[fr["method"] == m]["phi"] == 0.0).all())
           for m in ["(b)oracle", "Flirds", "Flirds1st", "loss-heur", "FedIF", "Fed-LOO"]}
    nz = {m: float(fr[fr["method"] == m]["phi"].abs().max())
          for m in ["GTG", "FedSV", "ComFedSV"] if m in set(fr["method"])}
    REP.append(f"[{'PASS' if all(ex0.values()) else 'FAIL'}] frzero bit-exact 0 (exact-0 계열): {ex0}")
    info(f"frzero renorm 계열 |phi| max = {nz} (0 아님 = 감사 판정 2)")
    # Q-3: noisy dose ladder에서 Flirds phi(client0)가 전 구간 기여-양수(=loss-감소 음수)인지
    lad = {}
    for nr in ["0.1", "0.25", "0.5", "0.75", "1.0"]:
        vals = []
        for s in [0, 1, 2]:
            p = f"runs/removal_dose/rundirs/1B_silo5_noisy_dose_nr{nr}_seed{s}/phi.parquet"
            if not os.path.exists(p):
                continue
            d = pd.read_parquet(p)
            if "kind" in d.columns and (d["kind"] == "val").any():
                d = d[d["kind"] == "val"]
            v = d[(d["method"] == "Flirds") & (d["client"] == 0)]["phi"]
            vals.extend(v.tolist())
        if vals:
            lad[nr] = (float(np.mean(vals)), all(x < 0 for x in vals))
    info(f"noisy dose ladder Flirds phi(client0) (mean, 전부<0=기여방향 양수): {lad} "
         f"→ 0-교차 없음(감사 판정 3: sign-게이트 작동영역 없음)의 직접 확인")
except Exception as e:
    info(f"Q ERROR: {e!r}")

section("R. 잔여 스팟 — track_g lf0.70(V2) / track_h 총평 재계산 / P5 LF")
try:
    cg = pd.read_csv("runs/track_g/analysis/cnn_summary.csv")
    def acc70(part):
        sub = cg[(cg["partition"] == part) & (cg["arm"] == "flirds_gate_v2")
                 & cg["cell"].str.contains("fr0.70")]
        return sub["final_acc"].mean()
    check("track_g lf0.70 V2 iid", .5967, acc70("iid"), tol=0.002)
    check("track_g lf0.70 V2 dir1", .5712, acc70("dir1"), tol=0.002)
    cc2 = pd.read_csv("runs/track_h/analysis/cnn_competition.csv")
    def c2(pol, tim, src, th, fr=None):
        sub = cc2[(cc2["policy"] == pol) & (cc2["timing"] == tim) & (cc2["source"] == src)
                  & (cc2["threat"] == th) & (cc2["partition"] == "dir1")]
        if fr is not None and sub["flip_rate"].notna().any():
            sub = sub[sub["flip_rate"] == fr]
        return sub["final_acc"].mean()
    check("track_h P5h/retrain/flirds/LF@0.7", .6210, c2("P5h", "retrain", "flirds", "label_flip", 0.7), tol=0.002)
    check("track_h P5h/retrain/flirds/clean", .6333, c2("P5h", "retrain", "flirds", "clean"), tol=0.002)
    pols = [("P1", "online"), ("P1", "retrain"), ("P2", "online"), ("P2", "retrain"), ("P3", "online"), ("P4", "online")]
    tot = {}
    for s in ["flirds", "flirds1st", "lossheur", "fedif", "gtg", "fedsv", "comfedsv", "shapleyfl"]:
        ms = [np.mean([c2(p, t, s, "free_rider"), c2(p, t, s, "grad_noise"), c2(p, t, s, "label_flip", 0.7)])
              for p, t in pols]
        tot[s] = round(float(np.mean(ms)), 4)
    info(f"경쟁 총평(자체 재계산, 6개 오염-평균의 평균) = {dict(sorted(tot.items(), key=lambda x: -x[1]))} "
         f"(overview §3.2.6 '.568 flirds 1위 / .471 flirds1st 최하'와 정합)")
except Exception as e:
    info(f"R ERROR: {e!r}")

with open(os.path.join(OUT, "verification_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(REP) + "\n")
n_pass = sum(1 for r in REP if r.startswith("[PASS]"))
n_fail = sum(1 for r in REP if r.startswith("[FAIL]"))
print(f"PASS={n_pass} FAIL={n_fail} INFO={sum(1 for r in REP if r.startswith('[INFO]'))}")
print("\n".join(r for r in REP if r.startswith("[FAIL]")))
