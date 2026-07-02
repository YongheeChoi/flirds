#!/bin/bash
# Signal-size probe — PILOT (seed0) driver.  Plan: wiki/flirds-signal-size-diagnosis.md §2.1
# (Yonghee approved 2026-07-02: seed0 pilot first, FULL 11-method suite, ORACLE_A=0).
#
# GPU3 (free now):   A32 -> A64 (anchor5 rank sweep) -> noise-probe r16 -> r64
# GPU0/1/2 (busy):   B16 / B32 / B64 (std N=50, 5/round, R=200) — each starts only
#                    after that GPU is idle (Yonghee's 3B_anchor5 rerun ends there).
# Kill everything:   pkill -f run_pilot.sh; pkill -f 'track_d.py'; pkill -f probe_val_noise
set -u
PY=/home/korea_bupj/miniconda3/envs/flirds/bin/python
CODES=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/codes
ROOT=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/runs/probe_signal
LOGS=$ROOT/_logs
mkdir -p "$LOGS" "$ROOT/rundirs" "$ROOT/noise_probe"
cd "$CODES" || exit 1
export PYTHONPATH=.

note() { echo "[driver] $(date '+%F %T') $*" >> "$LOGS/_driver.log"; }

wait_idle() {  # block until GPU $1 has no compute procs (recheck once to be safe)
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

track_d() {  # gpu name extra-envs...
  local gpu=$1 name=$2; shift 2
  note "start $name (gpu$gpu)"
  CUDA_VISIBLE_DEVICES=$gpu RUN_NAME=$name RUNDIR_ROOT=$ROOT/rundirs SEED=0 ORACLE_A=0 \
    env "$@" "$PY" -u experiments/track_d.py > "$LOGS/$name.log" 2>&1
  note "done  $name rc=$?"
}

noise() {  # gpu rank
  local gpu=$1 r=$2
  note "start noise_r$r (gpu$gpu)"
  CUDA_VISIBLE_DEVICES=$gpu SEED=0 LORA_R=$r \
    "$PY" -u experiments/probe_val_noise.py > "$LOGS/noise_r$r.log" 2>&1
  note "done  noise_r$r rc=$?"
}

note "=== PILOT DRIVER UP (pid $$) ==="

( track_d 3 1B_anchor5_r32_seed0 REGIME=anchor5 LORA_R=32
  track_d 3 1B_anchor5_r64_seed0 REGIME=anchor5 LORA_R=64
  noise 3 16
  noise 3 64
  note "gpu3 chain COMPLETE" ) &

( wait_idle 0; track_d 0 1B_std50k5_r16_seed0 REGIME=std20 N_CLIENTS=50 K_ABS=5 LORA_R=16
  note "gpu0 chain COMPLETE" ) &
( wait_idle 1; track_d 1 1B_std50k5_r32_seed0 REGIME=std20 N_CLIENTS=50 K_ABS=5 LORA_R=32
  note "gpu1 chain COMPLETE" ) &
( wait_idle 2; track_d 2 1B_std50k5_r64_seed0 REGIME=std20 N_CLIENTS=50 K_ABS=5 LORA_R=64
  note "gpu2 chain COMPLETE" ) &

wait
note "=== PILOT DRIVER DONE ==="
