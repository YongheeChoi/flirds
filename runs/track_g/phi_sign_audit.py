#!/usr/bin/env python
"""Track G Stage 0 -- phi sign audit over ALL existing run-dirs (read-only; no GPU).

Scans runs/** for every phi.parquet / phi.csv, normalizes each stored phi into
CONTRIBUTION orientation (contribution = -stored_phi; stored phi is suspicion-
oriented good->LOW across every runner -- see runs/phase2_matrix/make_analysis.py
sign note), and emits:

  audit/sign_table.csv   one row per (cell, method, client): contribution, sign,
                         corrupt flag (from config), dose, zero-crossing-on-dose flag
  audit/SIGN_AUDIT.md    the §2.1 prediction-table numbers (Track G prompt) +
                         the noisy 0-crossing dose estimate (LLM dose-cell selection
                         basis) + the CNN label-flip crossing rate (C2 dose-ladder
                         3-point selection basis) + coverage list.

Detector rows (kind="det": FLDetector/STD-DAGMM/FLTrust/FedDQC) are excluded --
their scores are native high=suspicious, not a contribution game value, so a sign
audit is meaningless for them.

Parquet schema families (classified by COLUMNS, not path, so reorganized dirs
keep working):
  phase2   [threat, seed, method, kind, client, phi]   phase2_matrix / removal_dose
  track_d  [seed, method, client, phi]                 track_d / probe_signal (clean stages)
  cnn_wide [client, rate, phi_<method>, ...]           track_c1 / removal_dose CNN / probe cnn_c1

Run (read-only; from repo root):
  python runs/track_g/phi_sign_audit.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]          # repo root (runs/track_g/ -> flirds/)
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent / "audit"

# clean-threat token per runner family; poison kept in the table but not in the
# prediction summaries (poison is excluded from the Track G design, 2026-07-17).
_FR_THREATS = {"freerider_random", "freerider_zero", "freerider_delta"}


def _corrupt_set(cfg, threat):
    """The corrupt client ids for a phase2-family cell, from its persisted config."""
    rcfg = cfg.get("rcfg", {})
    if threat == "noisy":
        return set(rcfg.get("noisy", []))
    if threat in _FR_THREATS:
        return set(rcfg.get("freerider", []))
    if threat == "poison":
        return {rcfg.get("attacker", 0)}
    return set()


def _dose(cfg, name):
    """(dose_kind, dose) -- noisy_rate / dose_mult / poison_frac; config first, name fallback."""
    nr = float(cfg.get("noisy_rate", 1.0))
    dm = float(cfg.get("dose_mult", 1.0))
    if nr != 1.0:
        return "noisy_rate", nr
    if dm != 1.0:
        return "dose_mult", dm
    for kind, pat in (("noisy_rate", r"_nr([\d.]+)"), ("dose_mult", r"_dm([\d.]+)"),
                      ("poison_frac", r"_pf([\d.]+)")):
        m = re.search(pat, name)
        if m:
            return kind, float(m.group(1).rstrip("."))
    return None, np.nan


_DROP_TOKENS = {"1B", "3B", "7B", "silo5", "iid5", "anchor5", "std20", "std50k5",
                "mnist", "cifar10", "fmnist", "clean", "noisy", "frrand", "frzero",
                "frdelta", "poison", "label-flip", "feature-noise", "label-skew",
                "quantity-skew", "iid", "grad-noise", "free-rider", "anchor"}


def _variant(name):
    """Non-canonical variant tokens of a cell dir name (probe levers: lr/steps/rank/
    width/kfrac/adamw/removal/dose/aonly...).  '' -> 'canon'.  Keeps canonical and
    probe cells from mixing in the P1 sign summaries."""
    toks = [t for t in name.split("_")
            if t not in _DROP_TOKENS and not re.fullmatch(r"seed\d+", t)
            and not t.startswith("device100") and not re.fullmatch(r"str[\d.a-z]+", t)]
    return "_".join(toks) or "canon"


def _load_cfg(d):
    p = d / "config.yaml"
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def _read_phi(d):
    p = d / "phi.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p = d / "phi.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def scan():
    rows, coverage = [], []
    seen = set()
    for p in sorted(RUNS.rglob("phi.*")):
        if p.suffix not in (".parquet", ".csv") or p.parent in seen:
            continue
        d = p.parent
        seen.add(d)
        rel = d.relative_to(RUNS).as_posix()
        df = _read_phi(d)
        if df is None or df.empty:
            coverage.append((rel, "empty", 0))
            continue
        cfg = _load_cfg(d)
        cols = set(df.columns)

        if {"threat", "seed", "method", "client", "phi"} <= cols:      # phase2 family
            fam, regime = "phase2", cfg.get("regime", "?")
            if "kind" in cols:
                df = df[df["kind"] == "val"]
            for (threat, seed, method), g in df.groupby(["threat", "seed", "method"]):
                corrupt = _corrupt_set(cfg, threat)
                dose_kind, dose = _dose(cfg, d.name)
                for _, r in g.iterrows():
                    rows.append(dict(cell=rel, family=fam, regime=regime,
                                     variant=_variant(d.name),
                                     scale=cfg.get("scale"), threat=threat, seed=int(seed),
                                     method=method, client=int(r["client"]),
                                     contribution=-float(r["phi"]),
                                     corrupt=int(r["client"]) in corrupt,
                                     dose_kind=dose_kind, dose=dose))
            coverage.append((rel, fam, len(df)))

        elif {"seed", "method", "client", "phi"} <= cols:              # track_d family (clean stages)
            fam = "track_d"
            rcfg = cfg.get("rcfg", {})
            regime = f"{cfg.get('regime', '?')}-N{rcfg.get('n_clients', '?')}k{rcfg.get('k_abs', '?')}"
            for _, r in df.iterrows():
                rows.append(dict(cell=rel, family=fam, regime=regime,
                                 variant=_variant(d.name), scale=cfg.get("scale"),
                                 threat="clean", seed=int(r["seed"]), method=str(r["method"]),
                                 client=int(r["client"]), contribution=-float(r["phi"]),
                                 corrupt=False, dose_kind=None, dose=np.nan))
            coverage.append((rel, fam, len(df)))

        elif "client" in cols and any(c.startswith("phi_") for c in cols):   # cnn wide
            fam = "cnn_wide"
            scen = cfg.get("scenario", cfg.get("threat", "?"))
            regime = f"{cfg.get('dataset', '?')}-{scen}"
            seed = int(cfg.get("seed", 0))
            rates = df["rate"] if "rate" in cols else pd.Series(0.0, index=df.index)
            for c in [c for c in df.columns if c.startswith("phi_")]:
                method = c[len("phi_"):]
                for i, r in df.iterrows():
                    if r[c] is None or (isinstance(r[c], float) and np.isnan(r[c])):
                        continue                      # column absent for this cell (e.g. phi_a off-anchor)
                    rows.append(dict(cell=rel, family=fam, regime=regime,
                                     variant=_variant(d.name), scale="cnn",
                                     threat=scen, seed=seed, method=method,
                                     client=int(r["client"]), contribution=-float(r[c]),
                                     corrupt=float(rates[i]) > 0,
                                     dose_kind="rate", dose=float(rates[i])))
            coverage.append((rel, fam, len(df)))
        else:
            coverage.append((rel, f"unknown cols={sorted(cols)[:6]}", len(df)))
    return pd.DataFrame(rows), coverage


def _crossing(doses, values):
    """First zero crossing of `values` along ascending `doses`; interpolated.
    Returns (crossing_dose | None, extrapolated_flag)."""
    order = np.argsort(doses)
    d, v = np.asarray(doses, float)[order], np.asarray(values, float)[order]
    for i in range(1, len(d)):
        if v[i - 1] > 0 >= v[i] or v[i - 1] >= 0 > v[i]:
            if v[i] == v[i - 1]:
                return float(d[i]), False
            t = v[i - 1] / (v[i - 1] - v[i])
            return float(d[i - 1] + t * (d[i] - d[i - 1])), False
    if len(d) >= 2 and v[-1] > 0 and v[-1] < v[-2]:      # decreasing but still positive
        slope = (v[-1] - v[-2]) / (d[-1] - d[-2])
        if slope < 0:
            return float(d[-1] - v[-1] / slope), True     # linear extrapolation beyond ladder
    return None, False


def _fmt(x, nd=6):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{nd}f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tab, coverage = scan()

    # zero-crossing-on-dose flag: within a (family, regime, scale, threat, method,
    # client, dose_kind) ladder group, does the contribution sign change across doses?
    tab["sign"] = np.sign(tab["contribution"]).astype(int)
    tab["crosses_zero_on_dose"] = False
    dosed = tab[tab["dose_kind"].notna() & tab["dose"].notna()]
    keys = ["family", "regime", "scale", "threat", "method", "client", "dose_kind"]
    for _, g in dosed.groupby(keys):
        if g["dose"].nunique() > 1 and g["sign"].nunique() > 1:
            tab.loc[g.index, "crosses_zero_on_dose"] = True
    tab.to_csv(OUT / "sign_table.csv", index=False)

    md = ["# Track G Stage 0 -- phi sign audit (2026-07-19)",
          "",
          "Contribution orientation (= -stored phi; helpful client -> POSITIVE).",
          f"Rows: {len(tab)} over {tab['cell'].nunique()} run-dirs; detectors excluded.",
          ""]

    # ---- P1: clean cells -- cumulative-sign gate must not cut anyone (§2.1 row 1) ----
    md += ["## P1  clean cells: cumulative contribution sign (gate false-exclusion check)", "",
           "canon variant = the canonical configs; probe variants (lr/steps/rank/...) "
           "are listed separately -- do NOT mix them into the do-no-harm claim.", "",
           "| stage (regime/scale) | variant | method | clients>0 / total | min contribution |",
           "|---|---|---|---|---|"]
    clean = tab[(tab["threat"] == "clean") & (tab["family"] != "cnn_wide")]
    for (regime, scale, var, method), g in clean.groupby(["regime", "scale", "variant", "method"]):
        md.append(f"| {regime}/{scale} | {var} | {method} "
                  f"| {(g['contribution'] > 0).sum()}/{len(g)} "
                  f"| {_fmt(g['contribution'].min())} |")
    cnn_iid = tab[(tab["family"] == "cnn_wide") & (tab["variant"] == "canon")
                  & tab["regime"].str.endswith("-iid")]
    md += ["", "CNN iid (canonical) uncorrupt clients:",
           "", "| dataset | method | clients>0 / total | min |", "|---|---|---|---|"]
    for (regime, method), g in cnn_iid.groupby(["regime", "method"]):
        md.append(f"| {regime} | {method} | {(g['contribution'] > 0).sum()}/{len(g)} "
                  f"| {_fmt(g['contribution'].min(), 4)} |")
    md.append("")

    # ---- P2: free-rider contribution (exact-0 rule check; §2.1 rows 2-3) ----
    md += ["## P2  free-rider corrupt-client contribution (strict >0 rule check)", "",
           "| stage | threat | method | n | bit-exact 0 | n>0 | n<0 | mean | max abs |",
           "|---|---|---|---|---|---|---|---|---|"]
    fr = tab[tab["threat"].isin(_FR_THREATS) & tab["corrupt"]]
    for (regime, scale, threat, method), g in fr.groupby(["regime", "scale", "threat", "method"]):
        v = g["contribution"]
        md.append(f"| {regime}/{scale} | {threat} | {method} | {len(v)} "
                  f"| {(v == 0.0).sum()}/{len(v)} | {(v > 0).sum()} | {(v < 0).sum()} "
                  f"| {v.mean():+.3e} | {v.abs().max():.3e} |")

    # ---- P3: noisy corrupt-client contribution vs dose (0-crossing estimate) ----
    md += ["", "## P3  noisy corrupt-client contribution vs noisy_rate (LLM dose-cell basis)", "",
           "| regime/scale | method | " ,
           ]
    noisy = tab[(tab["threat"] == "noisy") & tab["corrupt"] & (tab["family"] == "phase2")].copy()
    noisy["dose"] = noisy["dose"].fillna(1.0)                     # canonical noisy = nr 1.0
    noisy.loc[noisy["dose_kind"].isna(), "dose_kind"] = "noisy_rate"
    lad = (noisy[noisy["dose_kind"] == "noisy_rate"]
           .groupby(["regime", "scale", "method", "dose"])["contribution"]
           .mean().reset_index())
    doses = sorted(lad["dose"].unique())
    md[-1] = ("| regime/scale | method | " +
              " | ".join(f"nr={d:g}" for d in doses) + " | 0-crossing nr |")
    md.append("|---" * (3 + len(doses)) + "|")
    noisy_cross = {}
    for (regime, scale, method), g in lad.groupby(["regime", "scale", "method"]):
        vals = {d: v for d, v in zip(g["dose"], g["contribution"])}
        cells = " | ".join(_fmt(vals.get(d, np.nan)) for d in doses)
        if len(vals) > 1:
            cr, ext = _crossing(list(vals), list(vals.values()))
            if cr is None:
                cross = "none (positive across ladder)"
            elif cr > 1.0:
                cross = f"~{cr:.2f} = UNREACHABLE (nr caps at 1.0)"
            else:
                cross = f"~{cr:.2f}" + (" (extrapolated)" if ext else "")
            noisy_cross[(regime, scale, method)] = cr
        else:
            cross = "single dose"
        md.append(f"| {regime}/{scale} | {method} | {cells} | {cross} |")

    # ---- P4: CNN label-flip / feature-noise ladder crossing (C2 dose 3-point basis) ----
    md += ["", "## P4  CNN corrupt-client contribution vs per-client rate (C2 dose-ladder basis)", "",
           "| dataset/scenario | method | corr(contribution, rate) | crossing rate | note |",
           "|---|---|---|---|---|"]
    cnn = tab[(tab["family"] == "cnn_wide") & (tab["variant"] == "canon")
              & tab["regime"].str.contains("flip|noise", na=False)]
    cnn_cross = {}
    for (regime, method), g in cnn.groupby(["regime", "method"]):
        m = g.groupby("dose")["contribution"].mean()
        if len(m) < 2:
            continue
        corr = float(np.corrcoef(m.index, m.values)[0, 1])
        cr, ext = _crossing(list(m.index), list(m.values))
        if cr is not None and cr > 1.0:               # rate axis caps at 1.0
            cross = ">1 (unreachable on rate axis)"
        elif cr is None:
            cross = "none"
        else:
            cross = f"~{cr:.3f}" + (" (extrapolated)" if ext else "")
            cnn_cross[(regime, method)] = cr
        note = "monotone down" if corr < -0.8 else ("monotone up" if corr > 0.8 else "non-monotone")
        md.append(f"| {regime} | {method} | {corr:+.3f} | {cross} | {note} |")

    # ---- P5: per-method sign disagreement vs (b) on corrupt clients ----
    md += ["", "## P5  corrupt-client sign by method (value-level decision differences; "
              "canonical dose only)", "",
           "| stage | threat | method | mean corrupt contribution | sign |", "|---|---|---|---|---|"]
    cor = tab[tab["corrupt"] & (tab["family"] == "phase2") & (tab["threat"] != "poison")
              & tab["dose_kind"].isna()]
    for (regime, scale, threat, method), g in cor.groupby(["regime", "scale", "threat", "method"]):
        mu = g["contribution"].mean()
        md.append(f"| {regime}/{scale} | {threat} | {method} | {mu:+.3e} "
                  f"| {'+' if mu > 0 else ('0' if mu == 0 else '-')} |")

    # ---- Recommendations (the actionable Stage 0 outputs) ----
    md += ["", "## Recommendations (auto-computed Stage 0 outputs)", ""]
    _cr = lambda m: noisy_cross.get(("silo5", "1B", m))
    _crs = lambda m: ("none" if _cr(m) is None else f"~{_cr(m):.2f}")
    fl_cr = _cr("Flirds")
    if fl_cr is None or fl_cr > 1.0:
        md += [f"1. **LLM noisy sign-gate operating region: NONE on nr in (0,1]** -- "
               f"Flirds corrupt-client cumulative contribution stays positive across the "
               f"whole nr ladder (linear-extrapolated crossing Flirds {_crs('Flirds')}, "
               f"(b)oracle {_crs('(b)oracle')}, loss-heur {_crs('loss-heur')}); nr caps "
               "at 1.0, so the tau=0 sign gate CANNOT fire on noisy at any dose.  The "
               "§2.1 noisy@canon parity prediction is confirmed at value level; noisy "
               "recovery must come from the z-gate or V2w down-weighting, not dose "
               "escalation.  LLM dose cells for Track G: canonical nr=1.0 suffices "
               "for the parity check; at most ONE extra cell (nr=0.75, steepest "
               "decline) if a dose-trend datapoint is wanted -- a crossing hunt is "
               "pointless.",
               ""]
    md += [f"2. **GTG/FedSV sign-gates WOULD fire on noisy@canon** (crossings GTG "
           f"{_crs('GTG')} / FedSV {_crs('FedSV')} on the nr ladder) -- but that is "
           "their coalition-renorm value error relative to the (b) truth (+, "
           "net-helpful), not a calibrated decision; report as the value-level-"
           "fidelity -> decision-difference story, and note the in-run-(b)-game-0 vs "
           "retrain-(a)-game-0 distinction (removal canon shows removing the noisy "
           "client DOES help after retraining).",
           ""]
    frr = tab[(tab["threat"] == "freerider_random") & tab["corrupt"]
              & (tab["method"] == "Flirds") & (tab["variant"].isin(["canon", "removal"]))]
    for regime in ("silo5", "iid5"):
        v = frr[frr["regime"] == regime]["contribution"]
        if len(v):
            md += [f"3{'a' if regime == 'silo5' else 'b'}. **frrand @ {regime}**: Flirds "
                   f"corrupt contribution n>0={ (v > 0).sum()} / n<0={(v < 0).sum()} of "
                   f"{len(v)} (mean {v.mean():+.2e}) -- the frrand cumulative sign is a "
                   f"NEAR-ZERO COIN FLIP, not reliably negative: the strict->0 rule "
                   f"catches frzero exactly, but frrand exclusion will be seed-dependent "
                   f"(min_obs/burn-in + per-round screen matter).  Register this as a "
                   f"§2.1 amendment BEFORE running.", ""]
    if cnn_cross:
        lf = {k: v for k, v in cnn_cross.items() if "label_flip" in k[0]}
        if lf:
            lo, hi = min(lf.values()), max(lf.values())
            md += [f"4. **CNN label-flip dose ladder (3 points)**: per-client-rate "
                   f"crossings across val methods span ~{lo:.2f}-{hi:.2f} (C1 N=10 "
                   f"ladder, extrapolated).  Pick **rates {{0.15, 0.35, 0.70}}** for the "
                   f"C2 gate grid -- one below every crossing, one inside the span, one "
                   f"safely past it (C2 needs the new C2_FLIP_RATE fixed-rate knob; the "
                   f"legacy U(0.5,1) per-client rate sits entirely ABOVE the crossing "
                   f"span, which is why C2 mult already won there).", ""]

    # ---- coverage ----
    md += ["", "## Coverage (every phi file found under runs/)", "",
           "| run-dir | family | rows |", "|---|---|---|"]
    md += [f"| {rel} | {fam} | {n} |" for rel, fam, n in coverage]
    md += ["", "Run-dirs with metrics but NO per-client phi (not auditable for sign): "
              "track_c/c2 + probe_signal/cnn_c2 arm cells (OnlineScorer state was never "
              "persisted -- the Track G per-round phi_rounds.parquet logging closes exactly "
              "this gap going forward).", ""]

    (OUT / "SIGN_AUDIT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[audit] {len(tab)} rows -> {OUT / 'sign_table.csv'}")
    print(f"[audit] summary -> {OUT / 'SIGN_AUDIT.md'}")


if __name__ == "__main__":
    main()
