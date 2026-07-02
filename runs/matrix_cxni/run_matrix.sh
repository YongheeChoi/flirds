#!/bin/bash
# Corruption-axis x non-IID-axis 2x2 matrix driver.  Plan: wiki/flirds-signal-size-diagnosis.md §2.4
# (Yonghee approved 2026-07-02: 3-threat suite / silo5-scale / matrix-first).
#
#   IID (alpaca)  : iid5 x {clean, noisy, freerider_random, freerider_zero, poison}   [ALL NEW]
#   non-IID       : silo5 x {clean}                                                   [NEW]
#                   silo5 x {noisy, freerider, poison} already exist -> reused for comparison.
#
# GPU0-2 only (GPU3 stays on the rank-probe = run_pilot.sh).  Each GPU starts after it goes idle
# (Yonghee's 3B_anchor5 rerun campaign drains there).  Cells persist to phase2_matrix/rundirs under
# NEW names (1B_iid5_*, 1B_silo5_clean) -> existing results are NOT overwritten; make_analysis picks
# them up alongside the existing silo5 cells.
# Kill: pkill -f run_matrix.sh; pkill -f 'phase2_matrix.py'
set -u
PY=/home/korea_bupj/miniconda3/envs/flirds/bin/python
CODES=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/codes
RUNDIRS=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/runs/phase2_matrix/rundirs
LOGS=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/runs/matrix_cxni/_logs
mkdir -p "$LOGS"
cd "$CODES" || exit 1
export PYTHONPATH=.

note() { echo "[matrix] $(date '+%F %T') $*" >> "$LOGS/_driver.log"; }

wait_idle() {  # block until GPU $1 has no compute procs (recheck once to be safe) -- same as run_pilot.sh
  local gpu=$1
  while :; do
    n=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    if [ "$n" = "0" ]; then
      sleep 120
      n2=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
      [ "$n2" = "0" ] && return 0
    fi
    sleep 300
  done
}

cell() {  # gpu name regime threat extra-envs...  (3 seeds each; RUN_NAME fixes the persist dir)
  local gpu=$1 name=$2 regime=$3 threat=$4; shift 4
  note "start $name (gpu$gpu)"
  CUDA_VISIBLE_DEVICES=$gpu REGIME=$regime THREAT=$threat RUN_NAME=$name RUNDIR_ROOT=$RUNDIRS \
    env "$@" "$PY" -u experiments/phase2_matrix.py > "$LOGS/$name.log" 2>&1
  note "done  $name rc=$?"
}

note "=== MATRIX DRIVER UP (pid $$) ==="

# GPU0: three light cells (clean + free-rider-zero + non-IID clean)
( wait_idle 0
  cell 0 1B_iid5_clean  iid5  clean
  cell 0 1B_iid5_frzero iid5  freerider_zero
  cell 0 1B_silo5_clean silo5 clean
  note "gpu0 chain COMPLETE" ) &

# GPU1: two light cells (noisy + free-rider-random)
( wait_idle 1
  cell 1 1B_iid5_noisy  iid5  noisy
  cell 1 1B_iid5_frrand iid5  freerider_random
  note "gpu1 chain COMPLETE" ) &

# GPU2: poison (heavy: attacker backdoor install at the D2b config, 3 seeds)
( wait_idle 2
  cell 2 1B_iid5_poison iid5  poison LR=2e-3 BATCH=8 EPOCHS=5 POISON_FRAC=0.8 POISON_TRAIN=1000
  note "gpu2 chain COMPLETE" ) &

wait
note "=== MATRIX DRIVER DONE ==="
