#!/bin/bash
# Track G CNN grid (spec §4.4; runs FIRST -- calibrates defaults + decides V2w promotion).
#   {iid, dir1} x {clean, label_flip@{0.15,0.35,0.70}, grad_noise, free_rider} x 3 seeds
#   arms = vanilla + excl controls + 5 gate arms + flirds_mult (existing-policy contrast).
# label_flip dose points from the Stage 0 audit (crossing span ~0.13-0.55; audit rec 4).
# Then track_c1 C1_V3 cells (post-hoc kept-set retrain) on the ladder scenarios.
# Sequential on one GPU:  bash runs/track_g/run_cnn_grid.sh [gpu]
# After completion: python runs/track_g/make_analysis.py  -> V2w promotion verdict.
set -u
GPU="${1:-0}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/korea_bupj/miniconda3/envs/flirds/bin/python}"
LOGS="$REPO/runs/track_g/_logs"; mkdir -p "$LOGS"
cd "$REPO/codes"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU"
export C2_RUN_ROOT="$REPO/runs/track_g/rundirs_cnn"
export C1_RUN_ROOT="$REPO/runs/track_g/rundirs_cnn_v3"

ARMS_CORRUPT="vanilla,oracle_excl,random_excl,flirds_gate_v1,flirds_gate_v2,flirds_zgate_v2,flirds_gatew_v2,flirds_gatew_v1,flirds_mult"
ARMS_CLEAN="vanilla,flirds_gate_v1,flirds_gate_v2,flirds_zgate_v2,flirds_gatew_v2,flirds_gatew_v1,flirds_mult"

c2() {  # name threat extra-envs...
  local name=$1 threat=$2; shift 2
  echo "[c2] $name (gpu$GPU) $(date '+%F %T')" | tee -a "$LOGS/_driver.log"
  env "$@" C2_DATASET=cifar10 C2_THREAT="$threat" C2_MODE=full C2_RUN_NAME="$name" \
    "$PY" -u experiments/track_c2.py > "$LOGS/$name.log" 2>&1 \
    || echo "[c2] FAIL $name" | tee -a "$LOGS/_driver.log"
}

for seed in 0 1 2; do
  for part in iid dir1; do
    c2 "cifar10_${part}_clean_g_seed${seed}"      clean      C2_PARTITION="$part" C2_SEED="$seed" C2_ARMS="$ARMS_CLEAN"
    for rate in 0.15 0.35 0.70; do
      c2 "cifar10_${part}_label-flip_fr${rate}_g_seed${seed}" label_flip \
         C2_PARTITION="$part" C2_SEED="$seed" C2_ARMS="$ARMS_CORRUPT" C2_FLIP_RATE="$rate"
    done
    c2 "cifar10_${part}_grad-noise_g_seed${seed}" grad_noise C2_PARTITION="$part" C2_SEED="$seed" C2_ARMS="$ARMS_CORRUPT"
    c2 "cifar10_${part}_free-rider_g_seed${seed}" free_rider C2_PARTITION="$part" C2_SEED="$seed" C2_ARMS="$ARMS_CORRUPT"
  done
done

# ---- V3 post-hoc (track_c1 stage: N=10 full participation, ladder scenarios) ----
for seed in 0 1 2; do
  for ds in mnist cifar10; do
    for scen in label_flip feature_noise; do
      name="${ds}_${scen}_v3_seed${seed}"
      echo "[c1v3] $name (gpu$GPU) $(date '+%F %T')" | tee -a "$LOGS/_driver.log"
      C1_DATASET="$ds" C1_SCENARIO="$scen" C1_SEED="$seed" C1_MODE=full \
        C1_ORACLE_A=0 C1_RIPPLE=0 C1_V3=1 C1_RUN_NAME="$name" \
        "$PY" -u experiments/track_c1.py > "$LOGS/$name.log" 2>&1 \
        || echo "[c1v3] FAIL $name" | tee -a "$LOGS/_driver.log"
    done
  done
done
echo "[grid] DONE $(date '+%F %T')" | tee -a "$LOGS/_driver.log"
