"""§15.3 cross-run roll-up: scan a runs/ root, assemble each run-dir's
config + meta + metrics + timing into one tidy CSV (one row per run-dir), and
report GPU-hours.  Read-only over run-dirs -- never re-runs an experiment, so it
can be iterated freely (protocol §15.3).

Doubles as the GPU-hours reconstruction tool: run-dirs that predate timing.json
(§15.1) get a valuation-only GPU-hour estimate reconstructed from the per-method
`runtime` seconds in metrics.json -- the client-training (log-generation) time is
absent from those dirs, so it is excluded and flagged; runs made after the §15.1
instrumentation carry the authoritative per-phase timing.json (train + valuation
+ gpu_hours + peak) and are used verbatim.

Usage:
  python experiments/aggregate_runs.py [runs_root] [-o out.csv]
  (runs_root default: "runs"; without -o, prints a summary of the first rows)
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys


def _load_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _load_yaml(p):
    try:
        import yaml
        with open(p) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _sum_runtimes(obj):
    """Recursively sum every `runtime` dict's numeric values found in metrics.json
    (schemas vary across runners: phase2_matrix keys by threat_seed, others flat)."""
    total, n = 0.0, 0

    def walk(o):
        nonlocal total, n
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "runtime" and isinstance(v, dict):
                    for x in v.values():
                        if isinstance(x, (int, float)):
                            total += float(x)
                            n += 1
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return total, n


def scan(root):
    rows = []
    for mp in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        d = os.path.dirname(mp)
        metrics = _load_json(mp) or {}
        meta = _load_json(os.path.join(d, "meta.json")) or {}
        config = _load_yaml(os.path.join(d, "config.yaml")) or {}
        timing = _load_json(os.path.join(d, "timing.json"))
        val_s, n_meth = _sum_runtimes(metrics)
        row = {
            "run": os.path.relpath(d, root).replace("\\", "/"),
            "scale": config.get("scale"), "regime": config.get("regime"),
            "alpha": config.get("alpha"), "git_sha": (meta.get("git_sha") or "")[:9],
            "valuation_s_sum": round(val_s, 1), "n_method_times": n_meth,
            "train_s": None, "timing_total_s": None, "peak_gib": None,
        }
        if timing:                                   # §15.1 authoritative per-phase record
            ph = timing.get("phases", {})
            row["train_s"] = ph.get("client-training", {}).get("s")
            row["timing_total_s"] = timing.get("total_s")
            row["gpu_hours"] = timing.get("gpu_hours")
            row["peak_gib"] = timing.get("peak_gib")
            row["gpu_hours_src"] = "timing.json"
        else:                                        # reconstruct: valuation-only (no train time pre-§15.1)
            row["gpu_hours"] = round(val_s / 3600.0, 6)
            row["gpu_hours_src"] = "metrics.runtime (valuation-only, no train)"
        rows.append(row)
    return rows


COLS = ["run", "scale", "regime", "alpha", "git_sha", "valuation_s_sum",
        "n_method_times", "train_s", "timing_total_s", "gpu_hours", "peak_gib",
        "gpu_hours_src"]


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    root = pos[0] if pos else "runs"
    out = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else None

    rows = sorted(scan(root), key=lambda r: r["run"])
    total_gh = sum(r["gpu_hours"] or 0.0 for r in rows)
    have_timing = sum(1 for r in rows if r["gpu_hours_src"] == "timing.json")
    print(f"scanned {len(rows)} run-dirs under {root!r}: {have_timing} with timing.json, "
          f"{len(rows) - have_timing} reconstructed from metrics.runtime")
    print(f"total GPU-hours (sum) = {total_gh:.2f}   "
          f"[reconstructed rows are valuation-only; client-training excluded]")

    if out:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c) for c in COLS})
        print(f"wrote {out}  ({len(rows)} rows)")
    else:
        for r in rows[:20]:
            print(f"  {r['run']:<46} val={r['valuation_s_sum']:>9}s  "
                  f"gpu_h={r['gpu_hours']:.3f}  [{r['gpu_hours_src']}]")
        if len(rows) > 20:
            print(f"  ... ({len(rows) - 20} more; use -o out.csv for the full table)")


if __name__ == "__main__":
    main()
