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
            cells.append(dict(cell=d.name, cfg=yaml.safe_load((d / "config.yaml").read_text()),
                              m=json.loads((d / "metrics.json").read_text())))
    return cells


def analyze_cnn(cells):
    """track_c2-schema cells: per-arm final_acc/auroc/rtt + delta vs vanilla +
    recovery vs oracle_excl (accuracy axis: recovery=(arm-van)/(oracle-van))."""
    rows = []
    for c in cells:
        m, cfg = c["m"], c["cfg"]
        arms = m.get("arms", {})
        van = (arms.get("vanilla") or {}).get("final_acc")
        orc = (arms.get("oracle_excl") or {}).get("final_acc")
        for arm, a in arms.items():
            acc = a.get("final_acc")
            delta = (acc - van) if None not in (acc, van) else None
            rec = (delta / (orc - van) if None not in (delta, orc, van) and orc != van
                   else None)
            pred, _ = prediction("clean" if m.get("threat") == "clean" else m.get("threat"),
                                 arm, None)
            rows.append(dict(cell=c["cell"], dataset=m.get("dataset"),
                             partition=m.get("partition"), threat=m.get("threat"),
                             strength=m.get("strength"),
                             flip_rate=cfg.get("flip_rate"), seed=m.get("seed"),
                             arm=arm, final_acc=acc, delta_acc=delta, recovery=rec,
                             auroc=a.get("auroc"), rounds_to_target=a.get("rounds_to_target")))
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
                     "arm", "final_acc", "delta_acc", "recovery", "auroc"])
    status, lines = v2w_promotion(cnn)
    md += ["", f"## V2w promotion gate (spec §5-2): **{status}**", ""] + lines
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
