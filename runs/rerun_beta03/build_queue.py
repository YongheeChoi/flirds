#!/usr/bin/env python3
"""Master queue for the ShapleyFL beta=0.3 full re-run (no skips, single-provenance uniformity).

Every cell that holds ShapleyFL data is re-run FRESH at beta=0.3 (the code is already 0.3
at all 9 sites; see git).  Each output line is `script|run_name|envs` for run_multi_driver.sh.
Re-runs OVERWRITE the existing rundir in place (RunLogger uses makedirs(exist_ok=True); same
name -> same dir -> the 4 standard files are rewritten).

Sources of truth for exact reproduction:
  - track_c1 / track_c2 : the existing rundir's config.yaml (axes + mode + oracle_a) + the
    runner's own name generator (it hyphenates SCENARIO/THREAT, so we feed underscores).
  - phase2_matrix        : the original runs/phase2_matrix/master_queue.txt, verbatim.
  - track_d              : the known axes (scale x regime x seed); (a)-retrain oracle was on
    ONLY for 1B anchor5 (3B/7B anchor5 + every std20 ran with ORACLE_A=0; confirmed in configs).

phase1 is NOT here: its rundirs are the clean baseline (no ShapleyFL) -> nothing to replace.
"""
import glob
import os
from collections import Counter

REPO = "/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds"
OUT = os.path.join(REPO, "runs", "rerun_beta03", "master_queue.txt")


def top_level(cfg_path):
    """Top-level `key: value` pairs from a RunLogger config.yaml (skip the indented cfg: block)."""
    d = {}
    for ln in open(cfg_path):
        if not ln.strip() or ln[:1] in (" ", "\t", "#"):
            continue
        k, _, v = ln.partition(":")
        d[k.strip()] = v.strip().strip("'\"")
    return d


lines = []   # (cost_rank, "script|name|envs")

# ---- track_c1 (CNN fidelity) -- cheapest ----
for d in sorted(glob.glob(f"{REPO}/runs/track_c/c1/*/")):
    name = os.path.basename(d.rstrip("/"))
    c = top_level(os.path.join(d, "config.yaml"))
    oa = 1 if c.get("oracle_a") == "true" else 0
    envs = (f"C1_DATASET={c['dataset']} C1_SCENARIO={c['scenario']} C1_SEED={c['seed']} "
            f"C1_MODE={c.get('mode', 'full')} C1_ORACLE_A={oa}")
    lines.append((1, f"track_c1.py|{name}|{envs}"))

# ---- track_c2 (CNN intervention) -- all 90 carry a dismissal curve -> C2_DISMISSAL=1 ----
for d in sorted(glob.glob(f"{REPO}/runs/track_c/c2/*/")):
    name = os.path.basename(d.rstrip("/"))
    c = top_level(os.path.join(d, "config.yaml"))
    envs = (f"C2_DATASET={c['dataset']} C2_PARTITION={c['partition']} C2_THREAT={c['threat']} "
            f"C2_STRENGTH={c['strength']} C2_SEED={c['seed']} C2_MODE={c.get('mode', 'full')} "
            f"C2_DISMISSAL=1")
    lines.append((2, f"track_c2.py|{name}|{envs}"))

# ---- track_d (LLM) ----
MODEL = {"1B": "meta-llama/Llama-3.2-1B-Instruct",
         "3B": "meta-llama/Llama-3.2-3B-Instruct",
         "7B": "meta-llama/Llama-2-7b-hf"}
RANK = {"1B": 3, "3B": 5, "7B": 9}   # 7B last (most expensive)
for scale in ("1B", "3B", "7B"):
    for regime in ("std20", "anchor5"):
        for seed in (0, 1, 2):
            name = f"{scale}_{regime}_seed{seed}"
            oa = 1 if (scale == "1B" and regime == "anchor5") else 0
            envs = f"SMOKE_MODEL={MODEL[scale]} REGIME={regime} SEED={seed} ORACLE_A={oa}"
            lines.append((RANK[scale], f"track_d.py|{name}|{envs}"))

# ---- phase2_matrix -- reuse the exact original 25-cell queue verbatim ----
for ln in open(f"{REPO}/runs/phase2_matrix/master_queue.txt"):
    ln = ln.strip()
    if not ln or ln.startswith("#"):
        continue
    name, _, envs = ln.partition("|")
    lines.append((6, f"phase2_matrix.py|{name}|{envs}"))

lines.sort(key=lambda x: x[0])          # cheap -> expensive; stable within a rank
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("# ShapleyFL beta=0.3 full re-run (no skips, single provenance). "
            "format: script|run_name|envs ; cheap->expensive.\n")
    for _, l in lines:
        f.write(l + "\n")

c = Counter(l.split("|")[0] for _, l in lines)
print(f"wrote {len(lines)} cells -> {OUT}")
for k, v in sorted(c.items()):
    print(f"  {k}: {v}")
