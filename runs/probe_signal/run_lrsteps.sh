#!/bin/bash
# 학습 강도 probe (lr x steps = A축 세 번째 lever).  Plan: wiki/flirds-signal-size-diagnosis.md §2.5
# (Yonghee 2026-07-05: 각 1D sweep + 극단, 보수적 상한 lr<=3e-3 / steps<=30).
#
#   anchor5 (N=5 full, R=30), seed0, ORACLE_A=0, FIDELITY+ARMS on, MMLU 최소(40).
#   기준 lr1e-3/steps10 = 기존 1B_anchor5_seed0 재사용.  코드 0 (LR/MAX_STEPS env 기존).
#   신규 셀명이라 기존 rank/참여 probe rundir 안 덮어씀.
# GPU0-2 (매트릭스 완료 후).  Kill: pkill -f run_lrsteps.sh; pkill -f 'track_d.py'
set -u
PY=/home/korea_bupj/miniconda3/envs/flirds/bin/python
CODES=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/codes
ROOT=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/runs/probe_signal
LOGS=$ROOT/_logs
mkdir -p "$LOGS"
cd "$CODES" || exit 1
export PYTHONPATH=.

note() { echo "[lrsteps] $(date '+%F %T') $*" >> "$LOGS/_lrsteps_driver.log"; }

wait_idle() {  # block until GPU $1 has no compute procs (recheck once) -- same as run_pilot.sh
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

note "=== LRSTEPS DRIVER UP (pid $$) ==="

# GPU0: lr sweep 중간 + steps sweep 상한
( wait_idle 0
  cell 0 1B_anchor5_lr2e-3_st10_seed0 2e-3 10
  cell 0 1B_anchor5_lr1e-3_st30_seed0 1e-3 30
  note "gpu0 chain COMPLETE" ) &

# GPU1: lr sweep 상한 + 극단 조합
( wait_idle 1
  cell 1 1B_anchor5_lr3e-3_st10_seed0 3e-3 10
  cell 1 1B_anchor5_lr3e-3_st30_seed0 3e-3 30
  note "gpu1 chain COMPLETE" ) &

# GPU2: steps sweep 중간
( wait_idle 2
  cell 2 1B_anchor5_lr1e-3_st20_seed0 1e-3 20
  note "gpu2 chain COMPLETE" ) &

wait
note "=== LRSTEPS DRIVER DONE ==="
