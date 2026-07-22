#!/usr/bin/env python
"""Track G analysis -- regenerate every table from rundirs alone (phase2 make_analysis
convention: re-runnable, discovers whatever cells exist, running cells just don't
appear yet).  NO method re-runs.

Inputs : rundirs/<regime>_<threat>[_nr*]_<arm>_seed<s>/{config,metrics,phi_rounds}
         rundirs_cnn/  (track_c2 cells launched with C2_RUN_ROOT here)
         rundirs_cnn_v3/  (track_c1 C1_V3 cells launched with C1_RUN_ROOT here)
Outputs: analysis/{llm_summary.csv, cnn_summary.csv, README.md}   (wiped + rebuilt)

Report order is FIXED to the project question hierarchy: [1] performance delta +
recovery=(vanilla-arm)/(vanilla-oracle_excl)  [2] rounds-to-target  [3] gate
precision/recall + clean false-exclusions.  Every row carries the §2.1 prediction
(+ the Stage 0 audit amendments) and a mechanical verdict where the prediction is
crisp; soft predictions stay "report" (misses are shown as-is, spec §6).
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis"

SIGN_GATES = ("flirds_gate_v1", "flirds_gate_v2", "lossheur_gate_v2",
              "oracleb_gate_v2", "shapleyfl_gate_v2", "flirds_gatew_v2")
CLEAN_PARITY_ACC = 0.006          # C2 clean-parity band (spec §5-2, reused)
# 2026-07-22 skew-axis extension (README "확장 ②"):
RECOVERY_MIN_GAP = 0.02           # |oracle_excl - vanilla| below this -> recovery is
                                  # noise (lf@0.15 measured 0.003-0.006 -> seed-std 3.1)
V2_CUM_GATES = ("flirds_gate_v2", "flirds_gatew_v2")     # cum > tau participation gate
V1_RAW_GATES = ("flirds_gate_v1", "flirds_gatew_v1")     # per-round raw > tau screen
CNN_GATE_DEFAULT = dict(burn_in=10, tau=0.0, min_obs=2)  # README gate defaults -- the
# fallback for the first 36 cells, whose config.yaml predates the self-describing
# `gate` block (track_c2 now records it whenever a gate arm runs).


# --------------------------------------------------------------- §2.1 predictions
_POLICY_ARMS_ONLY = ("vanilla", "oracle_excl", "random_excl", "v3_random", "flirds_w")


def prediction(threat, arm, nr):
    """(prediction text, crisp-check id | None).  Amendments from the Stage 0 audit
    (frrand cum-sign coin-flip; noisy: no 0-crossing on nr<=1) are folded in.
    Controls/soft-contrast arms carry no prediction (they ARE the rulers)."""
    if arm in _POLICY_ARMS_ONLY:
        return "", None
    if threat == "clean":
        if arm == "flirds_gatew_v2":
            return "P1 gate: only policy intervening on clean -- parity check", "delta_small"
        if arm in SIGN_GATES or arm == "flirds_zgate_v2":
            return "vanilla parity; 0 false-exclusions (cum all positive)", "no_false_excl"
        return "parity", None
    if threat == "frzero":
        return "gain (~+0.007-class val-loss; exact-0 rule)", "gain"
    if threat == "frrand":
        return ("gain PREDICTED by §2.1 but audit AMENDS: cum sign is a ~0 coin flip "
                "-> exclusion seed-dependent"), None
    if threat == "noisy":
        if arm == "flirds_zgate_v2":
            return "recovery candidate (cohort-relative gate)", None
        if arm == "flirds_gatew_v2":
            return "gain possible (continuous down-weight, no 0-crossing needed)", None
        if arm in SIGN_GATES:
            return "PARITY -- gate silent (no 0-crossing on nr<=1, audit P3)", "silent"
        return "parity", None
    if threat == "mixed":
        if arm == "flirds_gate_v2":
            return "approaches oracle_excl (FR share recovered)", "recovery_half"
        if arm == "shapleyfl_gate_v2":
            return "<= random_excl (fidelity-collapse stage)", None
        return "FR share recovered; noisy share needs z/V2w", None
    return "", None


def verdict(check, row):
    if check is None or row.get("final_val_loss") is None:
        return ""
    d, g = row.get("delta"), row.get("gate") or {}
    if check == "no_false_excl":
        return "HIT" if g.get("false_excl_pairs") == 0 else "MISS"
    if check == "silent":
        return "HIT" if g.get("n_excluded_pairs") == 0 else "MISS"
    if check == "gain":
        return "" if d is None else ("HIT" if d > 0 else "MISS")
    if check == "delta_small":
        return "" if d is None else ("HIT" if abs(d) <= 0.002 else "MISS(reports as finding)")
    if check == "recovery_half":
        r = row.get("recovery")
        return "" if r is None else ("HIT" if r >= 0.5 else "MISS")
    return ""


# --------------------------------------------------------------- loading
def load_llm():
    cells = []
    rd = ROOT / "rundirs"
    if not rd.exists():
        return cells
    for d in sorted(rd.iterdir()):
        if not (d / "metrics.json").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        m = json.loads((d / "metrics.json").read_text())
        cells.append(dict(cell=d.name, cfg=cfg, m=m, dir=d))
    return cells


def observer_stats(d, corrupt, burn_in):
    """Vanilla-observer per-round false-fire rate (FIRST per-round measurement):
    fraction of clean (round, participant) pairs with raw <= 0; plus the earliest
    round from which every clean client's cum stays positive (burn-in calibration)."""
    p = d / "phi_rounds.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    part = df[df["participated"] & ~df["client"].isin(corrupt)]
    rate = float((part["raw"] <= 0).mean()) if len(part) else None
    clean = df[~df["client"].isin(corrupt)]
    ok_from = None
    for r in sorted(clean["round"].unique()):
        tail = clean[clean["round"] >= r]
        if (tail.groupby("client")["cum"].min() > 0).all():
            ok_from = int(r)
            break
    return dict(clean_raw_false_fire_rate=rate, all_clean_cum_pos_from_round=ok_from)


def analyze_llm(cells):
    rows = []
    groups = {}
    for c in cells:
        cfg = c["cfg"]
        key = (cfg["regime"], cfg["threat"], cfg.get("noisy_rate", 1.0), cfg["seed"])
        groups.setdefault(key, []).append(c)
    for (regime, threat, nr, seed), cs in sorted(groups.items()):
        by_arm = {c["cfg"]["arm"]: c for c in cs}
        van = by_arm.get("vanilla", {}).get("m", {}).get("final_val_loss")
        orc = by_arm.get("oracle_excl", {}).get("m", {}).get("final_val_loss")
        for c in cs:
            m, arm = c["m"], c["cfg"]["arm"]
            vl = m.get("final_val_loss")
            delta = (van - vl) if (van is not None and vl is not None) else None
            rec = (delta / (van - orc) if None not in (delta, van, orc) and van != orc
                   else None)
            r2t = m.get("rounds_to_target")
            if r2t is None and van is not None and m.get("val_curve"):
                r2t = next((i for i, v in enumerate(m["val_curve"]) if v <= van), None)
            pred, check = prediction(threat, arm, nr)
            row = dict(regime=regime, threat=threat, nr=nr, seed=seed, arm=arm,
                       final_val_loss=vl, delta=delta, recovery=rec,
                       rounds_to_target=r2t, gate=m.get("gate"),
                       mmlu=(m.get("downstream") or {}).get("mmlu"),
                       rouge_l=(m.get("downstream") or {}).get("rouge_l"),
                       kept=m.get("kept"), prediction=pred)
            row["verdict"] = verdict(check, row)
            if arm == "vanilla":
                row.update(observer_stats(c["dir"], set(m.get("corrupt", [])),
                                          (c["cfg"].get("gate") or {}).get("burn_in", 0)))
            rows.append(row)
    return pd.DataFrame(rows)


def load_cnn(sub):
    cells = []
    rd = ROOT / sub
    if not rd.exists():
        return cells
    for d in sorted(rd.iterdir()):
        if (d / "metrics.json").exists():
            cells.append(dict(cell=d.name, dir=d,
                              cfg=yaml.safe_load((d / "config.yaml").read_text()),
                              m=json.loads((d / "metrics.json").read_text())))
    return cells


def _dose(cell, cfg):
    """Fixed label-flip dose.  `flip_rate` only entered config.yaml on 2026-07-21,
    so the first 36 grid cells carry it in the rundir NAME only -- without this
    backfill the three dose points collapse into one averaged row."""
    if cfg.get("flip_rate") is not None:
        return float(cfg["flip_rate"])
    m = re.search(r"_fr([0-9.]+)_", cell)
    return float(m.group(1)) if m else None


def cnn_gate_excl(d, corrupt, gate, arm, n_rounds):
    """Reconstruct a sign gate's excluded set per round from phi_rounds and score it
    against the true corrupt mask (the CNN cells have no `gate` metrics block; the
    LLM runner writes one, track_c2 does not).  Exact for the two sign-gate families:

      V2 (participation): round r excludes {c: n_obs<r> >= min_obs and cum<r> <= tau}
          for r >= burn_in, reading the END-OF-ROUND-(r-1) snapshot the select seam
          sees (intervene._gate_select_fn).  Probation rotates ONE excluded client
          back per `probation_every` rounds -- it does not change the excluded set,
          so these counts are the gate's decisions, not realized non-participation.
      V1 (aggregation): a participant is screened out in round r iff raw <= tau.

    Returns micro pair counts (round, client) + the distinct clients ever excluded.
    """
    p = d / "phi_rounds.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    if "arm" in df.columns:
        df = df[df["arm"] == arm]
    if df.empty:
        return {}
    tau = gate.get("tau", 0.0)
    if arm in V1_RAW_GATES:
        ex = df[df["participated"] & (df["raw"] <= tau)][["round", "client"]]
    else:
        burn_in, min_obs = gate.get("burn_in", 0), gate.get("min_obs", 2)
        prev = df.copy()
        prev["round"] = prev["round"] + 1                  # state as seen by the next round
        ex = prev[(prev["round"] >= burn_in) & (prev["round"] < n_rounds)
                  & (prev["n_obs"] >= min_obs) & (prev["cum"] <= tau)][["round", "client"]]
    bad = set(int(c) for c in corrupt)
    hit = ex["client"].isin(bad)
    return dict(n_excluded_pairs=int(len(ex)),
                false_excl_pairs=int((~hit).sum()),
                excl_precision=(float(hit.mean()) if len(ex) else None),
                excl_clients=int(ex["client"].nunique()),
                false_excl_clients=int(ex.loc[~hit, "client"].nunique()))


def analyze_cnn(cells):
    """track_c2-schema cells: per-arm final_acc/auroc/rtt + delta vs vanilla +
    recovery vs oracle_excl (accuracy axis: recovery=(arm-van)/(oracle-van)).
    `gap` = that denominator; recovery is BLANKED when |gap| < RECOVERY_MIN_GAP."""
    rows = []
    for c in cells:
        m, cfg = c["m"], c["cfg"]
        arms = m.get("arms", {})
        van = (arms.get("vanilla") or {}).get("final_acc")
        orc = (arms.get("oracle_excl") or {}).get("final_acc")
        gap = (orc - van) if None not in (orc, van) else None
        corrupt = [i for i, v in enumerate(m.get("corrupt", [])) if v]
        for arm, a in arms.items():
            acc = a.get("final_acc")
            delta = (acc - van) if None not in (acc, van) else None
            rec = (delta / gap if None not in (delta, gap)
                   and abs(gap) >= RECOVERY_MIN_GAP else None)
            pred, _ = prediction("clean" if m.get("threat") == "clean" else m.get("threat"),
                                 arm, None)
            row = dict(cell=c["cell"], dataset=m.get("dataset"),
                       partition=m.get("partition"), threat=m.get("threat"),
                       strength=m.get("strength"), flip_rate=_dose(c["cell"], cfg),
                       seed=m.get("seed"), arm=arm, final_acc=acc, delta_acc=delta,
                       gap=gap, recovery=rec, auroc=a.get("auroc"),
                       rounds_to_target=a.get("rounds_to_target"))
            if arm in V2_CUM_GATES + V1_RAW_GATES and c.get("dir"):
                row.update(cnn_gate_excl(c["dir"], corrupt,
                                         cfg.get("gate") or CNN_GATE_DEFAULT, arm,
                                         (cfg.get("cfg") or {}).get("rounds", 0)))
            rows.append(row)
    return pd.DataFrame(rows)


def v2w_promotion(cnn):
    """Spec §5-2 promotion gate for V2w -> LLM: (1) on every corrupt threat
    V2w >= V2 (esp. noisy/label_flip); (2) clean parity |delta_acc| < 0.006.
    Returns (status, detail lines)."""
    if cnn.empty or "flirds_gatew_v2" not in set(cnn["arm"]):
        return "NOT EVALUABLE (no V2w CNN cells yet)", []
    lines, ok1, ok2 = [], True, True
    corrupt = cnn[cnn["threat"] != "clean"]
    for (ds, part, thr, s), g in corrupt.groupby(["dataset", "partition", "threat", "strength"]):
        p = g.pivot_table(index="seed", columns="arm", values="final_acc")
        if {"flirds_gatew_v2", "flirds_gate_v2"} <= set(p.columns):
            d = float((p["flirds_gatew_v2"] - p["flirds_gate_v2"]).mean())
            ok1 &= d >= 0
            lines.append(f"  {ds}/{part}/{thr}(str={s}): V2w-V2 mean dAcc={d:+.4f}"
                         f" {'OK' if d >= 0 else 'FAIL'}")
    clean = cnn[(cnn["threat"] == "clean") & (cnn["arm"] == "flirds_gatew_v2")]
    for _, r in clean.iterrows():
        d = r["delta_acc"]
        if d is not None:
            ok2 &= abs(d) < CLEAN_PARITY_ACC
            lines.append(f"  clean {r['cell']}: V2w dAcc={d:+.4f} "
                         f"{'OK' if abs(d) < CLEAN_PARITY_ACC else 'FAIL(parity broken)'}")
    if not lines:
        return "NOT EVALUABLE (missing V2/V2w pairs)", []
    return ("PROMOTE (add flirds_gatew_v2 to LLM ARMS)" if ok1 and ok2 else
            "DO NOT PROMOTE (report CNN-only -- an honest finding)"), lines


# ------------------------------------------------- 2026-07-22 skew-axis extension
PARTS = ("iid", "shard", "qskew", "dir1")            # no-skew / label / size / both


def _cellmean(cnn):
    """Per (dataset, partition, threat, dose, arm) 3-seed mean/std of the axes."""
    if cnn.empty:
        return cnn
    g = cnn.groupby(["dataset", "partition", "threat", "flip_rate", "arm"], dropna=False)
    out = g.agg(final_acc=("final_acc", "mean"), delta_acc=("delta_acc", "mean"),
                gap=("gap", "mean"), recovery=("recovery", "mean"),
                rec_sd=("recovery", "std"), auroc=("auroc", "mean"),
                false_excl_pairs=("false_excl_pairs", "mean"),
                excl_precision=("excl_precision", "mean"), seeds=("seed", "nunique"))
    return out.reset_index()


def skew_tables(cm):
    """The 2x2 decomposition view: one block per (dataset, threat, dose), partitions
    across the columns.  Absolute accuracy is the headline (project convention);
    recovery follows only where the denominator survived the guard."""
    md = []
    for ds in sorted(cm["dataset"].dropna().unique()):
        for (thr, dose), g in cm[cm.dataset == ds].groupby(["threat", "flip_rate"],
                                                           dropna=False):
            parts = [p for p in PARTS if p in set(g["partition"])]
            if not parts:
                continue
            tag = f"{thr}" + (f"@{dose:g}" if pd.notna(dose) else "")
            md += [f"", f"**{ds} / {tag}** — 절대 acc (recovery; 분모<{RECOVERY_MIN_GAP} → 공란)", ""]
            md += ["| arm | " + " | ".join(parts) + " |", "|" + "---|" * (len(parts) + 1)]
            for arm in sorted(set(g["arm"])):
                cells = []
                for p in parts:
                    r = g[(g.partition == p) & (g.arm == arm)]
                    if r.empty or pd.isna(r.final_acc.iloc[0]):
                        cells.append("")
                        continue
                    s = f"{r.final_acc.iloc[0]:.4f}"
                    if pd.notna(r.recovery.iloc[0]):
                        s += f" ({r.recovery.iloc[0]:+.2f})"
                    cells.append(s)
                md.append(f"| {arm} | " + " | ".join(cells) + " |")
            gaps = [f"{p}={g[(g.partition == p)].gap.dropna().mean():.4f}" for p in parts
                    if g[(g.partition == p)].gap.notna().any()]
            if gaps:
                md.append("")
                md.append(f"gap(oracle_excl−vanilla): {', '.join(gaps)}")
    return md


def _get(cm, ds, part, thr, dose, arm, col):
    q = cm[(cm.dataset == ds) & (cm.partition == part) & (cm.threat == thr)
           & (cm.arm == arm)]
    q = q[q.flip_rate.isna()] if dose is None else q[q.flip_rate == dose]
    return None if q.empty or pd.isna(q[col].iloc[0]) else float(q[col].iloc[0])


def prereg_verdicts(cm):
    """Mechanical checks for the pre-registered H-K1..H-K6 (README 확장 ②).
    Missing cells report 'pending'; misses are printed as-is (spec §6)."""
    L, V2 = [], "flirds_gate_v2"
    def line(hid, txt, ok):
        L.append(f"- **{hid}** {txt} -> " +
                 ("pending" if ok is None else ("**HIT**" if ok else "**MISS**")))

    for ds in sorted(cm["dataset"].dropna().unique()):
        # H-K1 free_rider recovery is partition-invariant
        recs = {p: _get(cm, ds, p, "free_rider", None, V2, "recovery") for p in PARTS}
        have = {p: v for p, v in recs.items() if v is not None}
        new = [p for p in ("shard", "qskew") if have.get(p) is not None]
        line("H-K1", f"{ds} free_rider V2 recovery " +
             ", ".join(f"{p}={v:+.2f}" for p, v in have.items()),
             None if not new else
             (all(have[p] >= 0.6 for p in new)
              and (max(have.values()) - min(have.values())) < 0.35))
        # H-K2 frrand caught like frzero (2nd-order curvature term)
        fr = _get(cm, ds, "iid", "frrand", None, V2, "recovery")
        fz = _get(cm, ds, "iid", "free_rider", None, V2, "recovery")
        alt = ("" if None in (fr, fz) else
               f" (frzero={fz:+.2f}; ratio={fr / fz:+.2f} — <=0.6이면 LLM 감사의 코인플립과 일치)")
        line("H-K2", f"{ds} iid frrand V2 recovery={'' if fr is None else f'{fr:+.2f}'}{alt}",
             None if fr is None else fr >= 0.7)
        # H-K3 clean false-fire worst on shard
        fe = {p: _get(cm, ds, p, "clean", None, V2, "false_excl_pairs") for p in PARTS}
        da = {p: _get(cm, ds, p, "clean", None, V2, "delta_acc") for p in PARTS}
        feh = {p: v for p, v in fe.items() if v is not None}
        line("H-K3", f"{ds} clean 오발화 pairs " +
             ", ".join(f"{p}={v:.0f}" for p, v in feh.items()) + " | V2 dAcc " +
             ", ".join(f"{p}={v:+.4f}" for p, v in da.items() if v is not None),
             None if "shard" not in feh or len(feh) < 2 else
             (feh["shard"] == max(feh.values())
              and (da.get("qskew") is None or abs(da["qskew"]) < CLEAN_PARITY_ACC)))
        # H-K4 seed spread largest on qskew
        for thr in ("free_rider", "grad_noise"):
            sd = {p: _get(cm, ds, p, thr, None, V2, "rec_sd") for p in ("iid", "qskew")}
            line("H-K4", f"{ds} {thr} recovery seed-sd " +
                 ", ".join(f"{p}={v:.3f}" for p, v in sd.items() if v is not None),
                 None if None in sd.values() or not sd["iid"] else
                 sd["qskew"] > 1.5 * sd["iid"])
        # H-K5 lf@0.15 denominator collapses everywhere
        gaps = {p: _get(cm, ds, p, "label_flip", 0.15, "vanilla", "gap") for p in PARTS}
        gh = {p: v for p, v in gaps.items() if v is not None}
        line("H-K5", f"{ds} lf@0.15 gap " + ", ".join(f"{p}={v:.4f}" for p, v in gh.items()),
             None if not gh else all(abs(v) < RECOVERY_MIN_GAP for v in gh.values()))
    # H-K6 fmnist keeps the ratio, shrinks the gap
    pairs = []
    for p in PARTS:
        for thr, dose in (("free_rider", None), ("frrand", None), ("grad_noise", None),
                          ("label_flip", 0.35), ("label_flip", 0.70)):
            a = _get(cm, "cifar10", p, thr, dose, V2, "recovery")
            b = _get(cm, "fmnist", p, thr, dose, V2, "recovery")
            if None not in (a, b):
                pairs.append((f"{p}/{thr}", a, b, abs(a - b)))
    line("H-K6", "fmnist↔cifar10 recovery diff " +
         (", ".join(f"{k}={d:.2f}" for k, _, _, d in pairs) or "(no comparable cells)"),
         None if not pairs else all(d <= 0.15 for *_, d in pairs))
    return L


def c2_soft_compare(cm):
    """Same-cell contrast with the C2 soft-weight grid (runs/track_c/c2, read-only).
    Only {clean, free_rider, grad_noise} strmain cells are the SAME cell; C2's
    label_flip is FedCorr rate~U(0.5,1), NOT a fixed dose -> excluded here."""
    c2 = ROOT.parent / "track_c" / "c2"
    if not c2.exists() or cm.empty:
        return ["(runs/track_c/c2 없음 — 대조 생략)"]
    rows = []
    for d in sorted(c2.iterdir()):
        f = d / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        if m.get("strength") != "main" or m.get("threat") not in ("clean", "free_rider",
                                                                  "grad_noise"):
            continue
        for arm in ("vanilla", "flirds_mult"):
            a = (m.get("arms") or {}).get(arm) or {}
            if a.get("final_acc") is not None:
                rows.append(dict(dataset=m["dataset"], partition=m["partition"],
                                 threat=m["threat"], arm=arm, acc=a["final_acc"]))
    if not rows:
        return ["(대응 C2 셀 없음)"]
    c2m = pd.DataFrame(rows).groupby(["dataset", "partition", "threat", "arm"])["acc"].mean()
    md = ["| dataset | partition | threat | C2 vanilla | G vanilla | C2 flirds_mult | "
          "G flirds_gate_v2 | 비고 |", "|" + "---|" * 8]
    for (ds, part, thr), _ in c2m.groupby(level=[0, 1, 2]):
        g_van = _get(cm, ds, part, thr, None, "vanilla", "final_acc")
        g_v2 = _get(cm, ds, part, thr, None, "flirds_gate_v2", "final_acc")
        if g_van is None:
            continue
        note = "same cell" if part in PARTS else ""
        md.append(f"| {ds} | {part} | {thr} | {c2m.get((ds, part, thr, 'vanilla')):.4f} | "
                  f"{g_van:.4f} | {c2m.get((ds, part, thr, 'flirds_mult')):.4f} | "
                  f"{'' if g_v2 is None else f'{g_v2:.4f}'} | {note} |")
    md += ["", "⚠️ qskew·frrand는 C2 대응 셀 없음. label_flip은 C2가 strmain"
           "(rate~U(0.5,1))이라 Track G의 고정 dose와 같은 셀이 아니어서 제외."]
    return md


def _md_table(df, cols):
    if df.empty:
        return ["(no cells yet)"]
    show = df[[c for c in cols if c in df.columns]].copy()
    for c in show.columns:
        if show[c].dtype == float:
            fmt = ("{:g}" if c == "nr" else
                   "{:+.4f}" if c.startswith(("delta", "recovery")) else "{:.4f}")
            show[c] = show[c].map(lambda v, f=fmt: "" if pd.isna(v) else f.format(v))
    lines = ["| " + " | ".join(show.columns) + " |",
             "|" + "---|" * len(show.columns)]
    for _, r in show.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return lines


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    llm = analyze_llm(load_llm())
    cnn = analyze_cnn(load_cnn("rundirs_cnn"))
    if not llm.empty:
        llm2 = llm.copy()
        llm2["gate"] = llm2["gate"].map(lambda g: json.dumps(g) if isinstance(g, dict) else "")
        llm2.to_csv(OUT / "llm_summary.csv", index=False)
    if not cnn.empty:
        cnn.to_csv(OUT / "cnn_summary.csv", index=False)

    md = ["# Track G analysis (auto-generated from rundirs -- rerun make_analysis.py)", ""]
    md += ["## [1] performance delta + recovery  (delta = vanilla_loss - arm_loss, +=better; "
           "recovery = delta / (vanilla - oracle_excl))", ""]
    perf_cols = ["regime", "threat", "nr", "seed", "arm", "final_val_loss", "delta",
                 "recovery", "mmlu", "rouge_l", "prediction", "verdict"]
    md += _md_table(llm.sort_values(["regime", "threat", "nr", "seed", "arm"])
                    if not llm.empty else llm, perf_cols)
    md += ["", "## [2] convergence (rounds-to-target = first round entering-loss <= "
           "the cell's vanilla final loss)", ""]
    md += _md_table(llm, ["regime", "threat", "nr", "seed", "arm", "rounds_to_target"])
    md += ["", "## [3] gate accuracy (per-round excluded set vs corrupt; micro P/R) "
           "+ vanilla-observer per-round false-fire", ""]
    if not llm.empty:
        g = llm[llm["gate"].map(lambda x: isinstance(x, dict))].copy()
        for k in ("precision", "recall", "n_excluded_pairs", "false_excl_pairs",
                  "n_fallback_rounds"):
            g[k] = g["gate"].map(lambda x, k=k: x.get(k))
        md += _md_table(g, ["regime", "threat", "nr", "seed", "arm", "precision",
                            "recall", "n_excluded_pairs", "false_excl_pairs",
                            "n_fallback_rounds"])
        obs = llm[llm["arm"] == "vanilla"]
        if "clean_raw_false_fire_rate" in obs.columns:
            md += ["", "vanilla observer (per-round raw, the project's first per-round "
                   "phi record):", ""]
            md += _md_table(obs, ["regime", "threat", "nr", "seed",
                                  "clean_raw_false_fire_rate",
                                  "all_clean_cum_pos_from_round"])
    md += ["", "## CNN (track_c2 gate cells)", ""]
    md += _md_table(cnn.sort_values(["dataset", "partition", "threat", "seed", "arm"])
                    if not cnn.empty else cnn,
                    ["dataset", "partition", "threat", "strength", "flip_rate", "seed",
                     "arm", "final_acc", "delta_acc", "gap", "recovery", "auroc",
                     "false_excl_pairs", "excl_precision"])
    status, lines = v2w_promotion(cnn)
    md += ["", f"## V2w promotion gate (spec §5-2): **{status}**", ""] + lines
    if not cnn.empty:                       # 2026-07-22 skew-axis extension
        cm = _cellmean(cnn)
        cm.to_csv(OUT / "cnn_cellmean.csv", index=False)
        md += ["", "## CNN skew 분해 (2×2: iid=skew없음 / shard=label만 / qskew=size만 "
               "/ dir1=둘다) — 3-seed 평균", "",
               "> ⚠️ 가법 분해 아님: shard의 label-skew(1.95 클래스/클라)는 dir1(9.87)보다, "
               "qskew의 size-skew(24×)는 dir1(6.2×)보다 세다. 축 귀속만 읽는다."]
        md += skew_tables(cm)
        md += ["", "## 사전등록 예측 대조 (README 확장 ②; MISS 그대로 보고)", ""]
        md += prereg_verdicts(cm)
        md += ["", "## C2 소프트-arm 같은-셀 대조 (runs/track_c/c2, read-only)", ""]
        md += c2_soft_compare(cm)
    v3 = load_cnn("rundirs_cnn_v3")
    if v3:
        md += ["", "## CNN V3 (track_c1 C1_V3 cells)", ""]
        for c in v3:
            ref = c["m"].get("v3_ref", {})
            for vn, d in (c["m"].get("v3") or {}).items():
                md.append(f"- {c['cell']} {vn}: kept={d.get('kept')} "
                          f"val_loss={d.get('val_loss')} acc={d.get('test_acc')} "
                          f"(full: {ref.get('full_val_loss')}/{ref.get('full_test_acc')})")
    (OUT / "README.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[analysis] {len(llm)} LLM rows, {len(cnn)} CNN rows -> {OUT}")


if __name__ == "__main__":
    main()
