#!/bin/bash
# Track G Phase-B smoke (spec §5-1): gpt2 silo5-mini frzero -- asserts the FR client
# is (a) gated out of aggregation from round 0 (raw exact-0) and (b) out of the
# cohort after burn-in except probation rounds, and that phi_rounds.parquet exists.
# + one CNN mini cell.  GPU cost: a few minutes.
#   bash runs/track_g/run_smoke.sh [gpu]     (from repo root)
set -eu
GPU="${1:-0}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/korea_bupj/miniconda3/envs/flirds/bin/python}"
SMOKE_ROOT="$REPO/runs/track_g/_smoke"
cd "$REPO/codes"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU"

echo "[smoke] LLM gpt2 silo5-mini frzero (V2 gate + observer)"
SMOKE_MODEL=gpt2 REGIME=silo5 THREAT=frzero TRAIN=40 VAL=10 TEST=10 ROUNDS=6 \
  MAX_STEPS=2 BURN_IN=2 PROBATION_EVERY=3 SEED=0 V3=1 \
  ARMS=vanilla,flirds_gate_v2 RUNDIR_ROOT="$SMOKE_ROOT" \
  "$PY" -u experiments/track_g.py

"$PY" - "$SMOKE_ROOT" <<'EOF'
import json, sys
import pandas as pd
root = sys.argv[1]
d = f"{root}/silo5_frzero_flirds_gate_v2_seed0"
df = pd.read_parquet(f"{d}/phi_rounds.parquet")            # per-round record EXISTS
fr = df[df.client == 1]                                    # silo5 freerider={1}
burn, prob = 2, 3
gated = fr[fr["round"] >= burn]
for _, x in gated.iterrows():
    on_probation = (x["round"] - burn) % prob == 0
    assert bool(x.participated) == on_probation, \
        f"round {x['round']}: FR participated={x.participated}, probation={on_probation}"
    if on_probation:
        assert x.weight == 0.0 or x.fallback, \
            "probation returnee must be screened to weight 0 (unless fallback round)"
assert (fr[fr.participated].raw == 0.0).all(), "frzero raw must be exact 0.0"
m = json.load(open(f"{d}/metrics.json"))
assert m["gate"]["recall"] == 1.0, f"gate recall {m['gate']['recall']} != 1.0"
print("[smoke] LLM ASSERTS PASS: FR excluded post-burn-in (probation screened), "
      f"raw exact-0, recall=1.0, {len(df)} phi_rounds rows")
EOF

echo "[smoke] CNN mini cell (cifar10 iid free_rider, gate arms)"
C2_DATASET=cifar10 C2_PARTITION=iid C2_THREAT=free_rider C2_SEED=0 C2_MODE=smoke \
  C2_EXTRA_ARMS=oracle_excl,random_excl,flirds_gate_v1,flirds_gate_v2,flirds_gatew_v2 \
  C2_BURN_IN=1 C2_PROB_EVERY=2 C2_RUN_ROOT="$SMOKE_ROOT/cnn" \
  "$PY" -u experiments/track_c2.py | tail -12

echo "[smoke] ALL GREEN -- smoke rundirs under runs/track_g/_smoke (disposable)"
