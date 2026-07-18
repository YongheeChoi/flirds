#!/bin/bash
# Track G LLM silo5 pilot (spec §5-3): seed 0, 4 threats x default arms + V3.
# Report GPU-h from timing.json BEFORE expanding to 3 seeds / iid5 / std50k5.
#   bash runs/track_g/run_llm_pilot.sh [gpu]
# NOTE flirds_gatew_v2 (V2w) is NOT in the default ARMS -- add via
#   ARMS env only after the CNN promotion gate passes (make_analysis verdict).
set -u
GPU="${1:-0}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/korea_bupj/miniconda3/envs/flirds/bin/python}"
LOGS="$REPO/runs/track_g/_logs"; mkdir -p "$LOGS"
cd "$REPO/codes"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU"

for threat in clean noisy frrand frzero; do
  echo "[pilot] silo5 $threat seed0 (gpu$GPU) $(date '+%F %T')" | tee -a "$LOGS/_driver.log"
  REGIME=silo5 THREAT="$threat" SEED=0 V3=1 \
    "$PY" -u experiments/track_g.py > "$LOGS/silo5_${threat}_seed0.log" 2>&1 \
    || echo "[pilot] FAIL silo5 $threat" | tee -a "$LOGS/_driver.log"
done
echo "[pilot] DONE -- report per-arm timing.json GPU-h to Yonghee before 3-seed expansion"
echo "  next (after approval):"
echo "    seeds:   for s in 1 2; do REGIME=silo5 THREAT=... SEED=\$s V3=1 ...; done"
echo "    iid5:    REGIME=iid5 THREAT=clean|frzero SEED=... V3=1"
echo "    noisy dose (optional, audit rec 1): REGIME=silo5 THREAT=noisy NOISY_RATE=0.75"
echo "    std50k5: REGIME=std50k5 THREAT=mixed SEED=0 ARMS=vanilla,oracle_excl,random_excl,flirds_gate_v2,shapleyfl_gate_v2"
echo "             (cost-probe FIRST -> Yonghee approval gate -> full arms/seeds)"
