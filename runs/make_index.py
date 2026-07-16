#!/usr/bin/env python
"""runs/make_index.py -- uniform flat index over EVERY rundir under runs/.

Dumb flatten, zero interpretation.  Walks all groups, reads only the structured
files the runners persisted (config.yaml / meta.json / metrics.json /
phi.parquet|phi.csv) and emits three long CSVs under runs/_index/:

  cells.csv         one row per rundir: group, cell, a few common config keys,
                    git_sha, git_dirty, has_metrics, has_phi
  metrics_long.csv  every SCALAR leaf in metrics.json, schema-agnostic:
                    group, cell, key (dotted path), value
  phi_long.csv      every phi row, columns unioned across groups:
                    group, cell, <whatever phi.parquet/phi.csv had...>

Why this exists: the bespoke generators (phase2_matrix/make_analysis.py,
track_c merge_oracle_a.py, track_d/make_fidelity.py) only cover their own
group and encode domain semantics.  This is the opposite -- one uniform,
interpretation-free dump that ALSO covers the groups with no generator
(probe_signal, removal_dose, measured_*), and doubles as a master index.
It reads rundirs only (never writes into them), is re-runnable, and path-
relative (anchored at this file's dir).  Skipped/partial dirs are reported,
not hidden.

  python runs/make_index.py
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    import yaml
except Exception:                       # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parent          # runs/
OUT = ROOT / "_index"
# common config keys worth surfacing in the index (best-effort; blank if absent).
CONFIG_KEYS = ["scale", "model", "regime", "setting", "partition", "dataset",
               "alpha", "seeds", "threats", "rounds", "lr", "n_clients"]


def _flatten(obj, prefix, out):
    """Emit (dotted_key, scalar) leaves.  dicts recursed; lists indexed."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten(v, f"{prefix}.{i}", out)
    else:
        out.append((prefix, obj))


def _cfg_get(cfg, key):
    """key from top-level config, or one level into rcfg/mcfg/env."""
    if not isinstance(cfg, dict):
        return None
    if key in cfg:
        return cfg[key]
    for sub in ("rcfg", "mcfg", "env"):
        if isinstance(cfg.get(sub), dict) and key in cfg[sub]:
            return cfg[sub][key]
    return None


def main():
    OUT.mkdir(exist_ok=True)
    cells, metrics_rows, phi_frames, skipped = [], [], [], []

    rundirs = sorted({p.parent for name in ("metrics.json", "phi.parquet", "phi.csv")
                      for p in ROOT.rglob(name)})
    for d in rundirs:
        rel = d.relative_to(ROOT)
        if rel.parts[0] == "_index":            # never index our own output
            continue
        group = rel.parts[0]
        cell = str(Path(*rel.parts[1:])) if len(rel.parts) > 1 else "."

        row = {"group": group, "cell": cell}

        # --- config.yaml (best-effort; heterogeneous across runners) ---
        cfg = {}
        cpath = d / "config.yaml"
        if cpath.exists() and yaml is not None:
            try:
                cfg = yaml.safe_load(cpath.read_text()) or {}
            except Exception as e:
                skipped.append((str(rel), f"config parse: {e}"))
        for k in CONFIG_KEYS:
            v = _cfg_get(cfg, k)
            row[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v

        # --- meta.json (provenance) ---
        mpath = d / "meta.json"
        if mpath.exists():
            try:
                meta = json.loads(mpath.read_text())
                row["git_sha"] = meta.get("git_sha")
                row["git_dirty"] = meta.get("git_dirty")
                row["env_hash"] = meta.get("env_hash")
            except Exception as e:
                skipped.append((str(rel), f"meta parse: {e}"))

        # --- metrics.json (schema-agnostic scalar flatten) ---
        jpath = d / "metrics.json"
        row["has_metrics"] = jpath.exists()
        if jpath.exists():
            try:
                pairs = []
                _flatten(json.loads(jpath.read_text()), "", pairs)
                for key, val in pairs:
                    if val is None or isinstance(val, (bool, int, float, str)):
                        metrics_rows.append({"group": group, "cell": cell,
                                             "key": key, "value": val})
            except Exception as e:
                skipped.append((str(rel), f"metrics parse: {e}"))

        # --- phi.parquet | phi.csv (raw contribution values) ---
        ppar, pcsv = d / "phi.parquet", d / "phi.csv"
        row["has_phi"] = ppar.exists() or pcsv.exists()
        try:
            pdf = pd.read_parquet(ppar) if ppar.exists() else (
                pd.read_csv(pcsv) if pcsv.exists() else None)
            if pdf is not None and len(pdf):
                pdf = pdf.copy()
                pdf.insert(0, "cell", cell)
                pdf.insert(0, "group", group)
                phi_frames.append(pdf)
        except Exception as e:
            skipped.append((str(rel), f"phi read: {e}"))

        cells.append(row)

    pd.DataFrame(cells).to_csv(OUT / "cells.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(OUT / "metrics_long.csv", index=False)
    (pd.concat(phi_frames, ignore_index=True) if phi_frames
     else pd.DataFrame()).to_csv(OUT / "phi_long.csv", index=False)

    gc = Counter(c["group"] for c in cells)
    print(f"cells.csv        : {len(cells)} rundirs across {len(gc)} groups")
    print(f"metrics_long.csv : {len(metrics_rows)} scalar rows")
    print(f"phi_long.csv     : {sum(len(f) for f in phi_frames)} phi rows")
    for g in sorted(gc):
        print(f"    {g:22s} {gc[g]:4d} rundirs")
    if skipped:
        print(f"\nskipped/partial ({len(skipped)}):")
        for rel, why in skipped[:40]:
            print(f"    {rel}: {why}")


if __name__ == "__main__":
    main()
