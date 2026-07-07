#!/bin/bash
# 학습 강도 probe — 2D 격자 빈 칸 (lr x steps 3x3 완성).  wiki §2.5
# (Yonghee 2026-07-07: lr·steps 더 조밀하게 -> 1D sweep의 안쪽 칸 3개 채움).
#   빈 칸: (lr2e-3,st20) (lr2e-3,st30) (lr3e-3,st20).  나머지 6칸은 기준+run_lrsteps.sh 5셀.
# GPU2 즉시(idle) + GPU0/1 (run_lrsteps.sh의 st30 셀이 끝나면).  anchor5 seed0, ORACLE_A=0.
# Kill: pkill -f run_lrsteps_grid.sh; pkill -f 'track_d.py'
set -u
PY=/home/korea_bupj/miniconda3/envs/flirds/bin/python
CODES=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/codes
ROOT=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/runs/probe_signal
LOGS=$ROOT/_logs
mkdir -p "$LOGS"
cd "$CODES" || exit 1
export PYTHONPATH=.

note() { echo "[lrgrid] $(date '+%F %T') $*" >> "$LOGS/_lrgrid_driver.log"; }

wait_idle() {  # block until GPU $1 has no compute procs (recheck once)
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

cell() {  # gpu name lr steps
  local gpu=$1 name=$2 lr=$3 st=$4
  note "start $name (gpu$gpu) lr=$lr steps=$st"
  CUDA_VISIBLE_DEVICES=$gpu REGIME=anchor5 LR=$lr MAX_STEPS=$st ORACLE_A=0 FIDELITY=1 ARMS=1 \
    MMLU_LIMIT=40 SEED=0 LORA_R=16 RUN_NAME=$name RUNDIR_ROOT=$ROOT/rundirs \
    "$PY" -u experiments/track_d.py > "$LOGS/$name.log" 2>&1
  note "done  $name rc=$?"
}

note "=== LRGRID DRIVER UP (pid $$) ==="

( wait_idle 2; cell 2 1B_anchor5_lr2e-3_st20_seed0 2e-3 20; note "gpu2 grid COMPLETE" ) &
( wait_idle 0; cell 0 1B_anchor5_lr2e-3_st30_seed0 2e-3 30; note "gpu0 grid COMPLETE" ) &
( wait_idle 1; cell 1 1B_anchor5_lr3e-3_st20_seed0 3e-3 20; note "gpu1 grid COMPLETE" ) &

wait
note "=== LRGRID DRIVER DONE ==="
