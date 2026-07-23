#!/usr/bin/env python
"""Track H analysis -- score-source competition tables from rundirs alone
(re-runnable; discovers whatever cells exist, running cells just don't appear).

Inputs (new runs + §1.5 reuse roots; a missing root is skipped silently):
  runs/track_h/rundirs_llm/   + reuse runs/track_g/rundirs/        (LLM cells)
  runs/track_h/rundirs_cnn/   + reuse runs/track_g/rundirs_cnn/    (CNN cells)
Outputs: runs/track_h/analysis/{llm_competition.csv, cnn_competition.csv,
         competition_score.csv, README.md}   (wiped + rebuilt)

Judgment (spec §3, FIXED before execution): the ONLY ranking axis is training
performance -- recovery = (vanilla-arm)/(vanilla-oracle_excl) on val-loss (LLM)
or (arm-vanilla)/(oracle_excl-vanilla) on test acc (CNN).  COMPETITION SCORE =
mean recovery over corrupt cells, per (source, policy, timing); clean cells are
a PARITY FLAG (CNN |dAcc| >= 0.006 band / LLM |dLoss| >= 0.002), reported next
to the score, never mixed into it.  Gate P/R and observer false-fire rates are
DIAGNOSTIC columns only (why performance moved), never ranked.

2026-07-20 aggregation fixes (per-cell CSVs unchanged in meaning; score column
semantics corrected -- see overview 3.2.6):
  - flip_rate falls back to the `_fr<dose>_` token in the rundir name (track_g
    CNN configs lack the key), so track_h lf@0.7 arms merge with their track_g
    vanilla/oracle anchors instead of dropping to recovery=NaN.
  - T2 arms skipped as `equals_vanilla` (kept=all -> retrain==vanilla by
    construction) count as delta=0/recovery=0 instead of vanishing from means.
  - The competition score is restricted to the Track H stage cells so every
    source averages the SAME cells: CNN = R1 dir1 fixed dose (lf only at 0.70)
    and the strmain borderline-rate cell as a SEPARATE stage_cell (2026-07-23;
    reuse-only iid/other-dose cells stay in cnn_competition.csv but not the
    score), LLM = R3 silo5 noisy nr1.0 (+ silo5 clean as parity anchor) and R2
    std50k5 reported as separate stage_cell rows.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent
OUT = ROOT / "analysis"
# The files main() regenerates.  Cleaned one-by-one instead of rmtree(OUT): the
# analysis dir is shared with sibling tools, and a blanket rmtree silently wiped
# r4's gsm50k5_tier_a.csv on every run until the 2026-07-23 audit caught it.  If
# you add an output, add it here; never reintroduce rmtree(OUT).
OWN_OUTPUTS = ("llm_competition.csv", "cnn_competition.csv",
               "observer_zero_semantics.csv", "competition_score.csv", "README.md")

SOURCES = ("flirds", "flirds1st", "lossheur", "gtg", "fedsv", "comfedsv",
           "shapleyfl", "fedif", "oracleb")
POLICY_SUFFIX = {"gate_v1": "P1v1", "gate_v2": "P1", "gatew_v2": "P2",
                 "mult": "P3", "zgate_v2": "P4", "w": "P3",
                 "cgate": "P5h", "pweight": "P5s"}     # P5 confidence policies (07-21)
CLEAN_BAND_CNN = 0.006            # C2 clean-parity band (track_g spec §5-2, reused)
CLEAN_BAND_LLM = 0.002


def parse_arm(arm):
    """arm -> (source, policy, timing) or None for controls/observer."""
    m = re.fullmatch(r"t2_(sign|signw|csign|pw)_(\w+)", arm)
    if m:
        pol = {"sign": "P1", "signw": "P2", "csign": "P5h", "pw": "P5s"}[m.group(1)]
        return m.group(2), pol, "retrain"
    if arm in ("v3_sign", "v3_z"):                     # track_g reuse: flirds-scored V3
        return "flirds", ("P1" if arm == "v3_sign" else "P4"), "retrain"
    for suf, pol in POLICY_SUFFIX.items():
        if arm.endswith("_" + suf):
            src = arm[: -len(suf) - 1]
            if src in SOURCES:
                return src, pol, "online"
    return None


def _load(root):
    cells = []
    if not root.exists():
        return cells
    for d in sorted(root.iterdir()):
        if (d / "metrics.json").exists():
            cells.append(dict(cell=d.name, dir=d,
                              cfg=yaml.safe_load((d / "config.yaml").read_text()),
                              m=json.loads((d / "metrics.json").read_text())))
    return cells


# ------------------------------------------------------------------- LLM
def analyze_llm():
    cells = _load(RUNS / "track_g" / "rundirs") + _load(ROOT / "rundirs_llm")  # track_h wins on dup
    groups = {}
    for c in cells:
        cfg = c["cfg"]
        key = (cfg["regime"], cfg["threat"], cfg.get("noisy_rate", 1.0), cfg["seed"])
        groups.setdefault(key, {})[cfg["arm"]] = c     # track_h root wins on dup
    rows = []
    for (regime, threat, nr, seed), by_arm in sorted(groups.items()):
        van_m = ((by_arm.get("vanilla") or by_arm.get("observer") or {})  # observer ==
                 .get("m") or {})                      # vanilla (bit-identical, R4)
        van = van_m.get("final_val_loss")
        van_em = (van_m.get("downstream") or {}).get("gsm8k_em")
        orc_m = (by_arm.get("oracle_excl", {}).get("m") or {})
        orc = orc_m.get("final_val_loss")
        orc_em = (orc_m.get("downstream") or {}).get("gsm8k_em")
        for arm, c in sorted(by_arm.items()):
            m = c["m"]
            vl = m.get("final_val_loss")
            delta = (van - vl) if None not in (van, vl) else None
            rec = (delta / (van - orc) if None not in (delta, orc, van) and van != orc
                   else None)
            em = (m.get("downstream") or {}).get("gsm8k_em")
            d_em = (em - van_em) if None not in (em, van_em) else None
            rec_em = (d_em / (orc_em - van_em)
                      if None not in (d_em, orc_em, van_em) and orc_em != van_em
                      else None)
            sp = parse_arm(arm)
            g = m.get("gate") or {}
            rows.append(dict(regime=regime, threat=threat, nr=nr, seed=seed, arm=arm,
                             source=sp[0] if sp else None,
                             policy=sp[1] if sp else None,
                             timing=sp[2] if sp else None,
                             final_val_loss=vl, delta=delta, recovery=rec,
                             gsm8k_em=em, delta_em=d_em, recovery_em=rec_em,
                             rounds_to_target=m.get("rounds_to_target"),
                             gate_precision=g.get("precision"),
                             gate_recall=g.get("recall"),
                             false_excl_pairs=g.get("false_excl_pairs"),
                             kept=(len(m["kept"]) if isinstance(m.get("kept"), list)
                                   else m.get("kept")),
                             skipped=m.get("skipped")))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- CNN
def observer_method_stats(d, corrupt):
    """Zero-semantics diagnostics from the observer's multi-method phi_rounds:
    per method, the clean-client false-fire rate (raw <= 0 among clean
    participants) and the corrupt-client fire rate (raw <= 0 among corrupt)."""
    p = d / "phi_rounds.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    if "method" not in df.columns:
        return {}
    df = df[df["arm"] == "observer"] if "arm" in df.columns else df
    part = df[df["participated"]]
    out = {}
    for meth, g in part.groupby("method"):
        cl, co = g[~g["client"].isin(corrupt)], g[g["client"].isin(corrupt)]
        out[meth] = dict(
            clean_false_fire=float((cl["raw"] <= 0).mean()) if len(cl) else None,
            corrupt_fire=float((co["raw"] <= 0).mean()) if len(co) else None)
    return out


def _flip_rate(c):
    """config flip_rate, falling back to the `_fr<dose>_` rundir-name token
    (track_g CNN configs lack the key; without this the three doses collide on
    one NaN key and track_h lf@0.7 arms lose their vanilla/oracle anchors)."""
    fr = (c["cfg"] or {}).get("flip_rate")     # may be a str ('0.70', env passthrough)
    if fr is None:
        m = re.search(r"_fr([0-9.]+)_", c["cell"])
        fr = m.group(1) if m else None
    return None if fr is None else float(fr)


def analyze_cnn():
    cells = {}
    for c in _load(RUNS / "track_g" / "rundirs_cnn") + _load(ROOT / "rundirs_cnn"):
        m = c["m"]
        key = (m.get("dataset"), m.get("partition"), m.get("threat"),
               _flip_rate(c), m.get("seed"))
        cells.setdefault(key, []).append(c)            # later (track_h) roots add arms
    rows, obs_rows = [], []
    for key, cs in sorted(cells.items(), key=str):
        dataset, partition, threat, fr, seed = key
        arms = {}
        for c in cs:                                   # merge arm dicts across roots
            arms.update(c["m"].get("arms", {}))
        van = (arms.get("vanilla") or {}).get("final_acc")
        orc = (arms.get("oracle_excl") or {}).get("final_acc")
        for c in cs:                                   # observer diagnostics (any root)
            if "observer" in c["m"].get("arms", {}):
                corrupt = {i for i, v in enumerate(c["m"].get("corrupt", [])) if v}
                for meth, st in observer_method_stats(c["dir"], corrupt).items():
                    obs_rows.append(dict(dataset=dataset, partition=partition,
                                         threat=threat, flip_rate=fr, seed=seed,
                                         source=meth, **st))
        for arm, a in sorted(arms.items()):
            acc = a.get("final_acc")
            if acc is None and a.get("skipped") == "equals_vanilla":
                acc = van          # kept=all -> retrain == vanilla by construction
            delta = (acc - van) if None not in (acc, van) else None
            rec = (delta / (orc - van) if None not in (delta, orc, van) and orc != van
                   else None)
            sp = parse_arm(arm)
            rows.append(dict(dataset=dataset, partition=partition, threat=threat,
                             flip_rate=fr, seed=seed, arm=arm,
                             source=sp[0] if sp else None,
                             policy=sp[1] if sp else None,
                             timing=sp[2] if sp else None,
                             final_acc=acc, delta_acc=delta, recovery=rec,
                             auroc=a.get("auroc"),
                             rounds_to_target=a.get("rounds_to_target"),
                             kept=len(a["kept"]) if a.get("kept") is not None else None,
                             skipped=a.get("skipped"),
                             dedup_shared=a.get("dedup_shared")))
    return pd.DataFrame(rows), pd.DataFrame(obs_rows)


# ------------------------------------------------------------------- score
def competition_score(llm, cnn):
    """Per (source, policy, timing, stage_cell): mean recovery over corrupt
    cells + the clean parity flag (never mixed into the score -- spec §3).
    Restricted to the Track H stage cells so every source averages the SAME
    cells (reuse-only iid/dose/frzero cells stay in the per-cell CSVs)."""
    rows = []
    for stage, df, clean_col, band in (("llm", llm, "delta", CLEAN_BAND_LLM),
                                       ("cnn", cnn, "delta_acc", CLEAN_BAND_CNN)):
        if df.empty:
            continue
        d = df[df["source"].notna()].copy()
        if stage == "cnn":                 # dir1 stage, two dose regimes as SEPARATE cells:
            #   "dir1"    = fixed dose (free-rider/frrand/grad-noise + lf@0.70); dir1 clean
            #               is the parity anchor.
            #   "strmain" = borderline lf, per-client rate ~U(.5,1) so flip_rate is NaN
            #               (Track H's borderline test; no clean cell -> parity N/A).
            # Kept apart so the confident-dose score is not diluted by the borderline
            # regime.  Anchors (vanilla/oracle_excl/flirds) merge from the track_g
            # gstrmain twin, which shares the (dir1, label_flip, NaN-flip_rate) key.
            d = d[d["partition"] == "dir1"].copy()
            strmain = d["threat"].eq("label_flip") & d["flip_rate"].isna()
            d = d[(d["threat"] != "label_flip") | (d["flip_rate"] == 0.7) | strmain]
            d["stage_cell"] = "dir1"
            d.loc[d["threat"].eq("label_flip") & d["flip_rate"].isna(), "stage_cell"] = "strmain"
        else:                              # R3 noisy nr1.0 (+clean parity anchor) | R2 std50k5
            d = d[(((d["regime"] == "silo5")
                    & d["threat"].isin(("clean", "noisy")) & (d["nr"] == 1.0))
                   | (d["regime"] == "std50k5"))]
            d["stage_cell"] = d["regime"]
        for (src, pol, tim, cell), g in d.groupby(
                ["source", "policy", "timing", "stage_cell"]):
            cor = g[g["threat"] != "clean"]
            cln = g[g["threat"] == "clean"]
            clean_delta = float(cln[clean_col].mean()) if len(cln) else None
            rows.append(dict(stage=stage, stage_cell=cell, source=src, policy=pol,
                             timing=tim,
                             score_corrupt_recovery=(float(cor["recovery"].mean())
                                                     if cor["recovery"].notna().any()
                                                     else None),
                             n_corrupt_cells=int(cor["recovery"].notna().sum()),
                             clean_delta=clean_delta,
                             clean_parity_ok=(None if clean_delta is None
                                              else bool(abs(clean_delta) < band))))
    sc = pd.DataFrame(rows)
    if not sc.empty:
        sc = sc.sort_values(["stage", "stage_cell", "policy", "timing",
                             "score_corrupt_recovery"],
                            ascending=[True, True, True, True, False])
    return sc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in OWN_OUTPUTS:                    # clean only our own stale outputs
        (OUT / name).unlink(missing_ok=True)
    llm = analyze_llm()
    cnn, obs = analyze_cnn()
    sc = competition_score(llm, cnn)
    llm.to_csv(OUT / "llm_competition.csv", index=False)
    cnn.to_csv(OUT / "cnn_competition.csv", index=False)
    if not obs.empty:
        obs.to_csv(OUT / "observer_zero_semantics.csv", index=False)
    sc.to_csv(OUT / "competition_score.csv", index=False)

    lines = ["# Track H analysis (generated by make_analysis.py -- rundir-only)", ""]
    lines.append(f"- LLM rows: {len(llm)} · CNN rows: {len(cnn)} · "
                 f"observer diag rows: {len(obs)}")
    if not sc.empty:
        lines += ["", "## Competition score (mean corrupt recovery; clean = parity flag)",
                  "", "```", sc.to_string(index=False), "```"]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[track_h] llm={len(llm)} cnn={len(cnn)} obs={len(obs)} score={len(sc)} "
          f"-> {OUT}", flush=True)
    if not sc.empty:
        print(sc.to_string(index=False))


if __name__ == "__main__":
    main()
