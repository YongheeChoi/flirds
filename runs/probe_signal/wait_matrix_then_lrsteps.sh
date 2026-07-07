#!/bin/bash
# Yonghee 2026-07-05: run the lr·steps probe (A축 3번째 lever) after the 2x2 matrix finishes.
# Watches the matrix driver log for "MATRIX DRIVER DONE", then starts run_lrsteps.sh on GPU0-2.
# Detached (nohup) so it survives the Claude session.
set -u
ML=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/runs/matrix_cxni/_logs/_driver.log
DRV=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/runs/probe_signal/run_lrsteps.sh
NOHUP=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/runs/probe_signal/_logs/_lrsteps_nohup.out
LOG=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/runs/probe_signal/_logs/_after_matrix.log
mkdir -p "$(dirname "$LOG")"
echo "[after-matrix] $(date '+%F %T') UP (pid $$) -- waiting for MATRIX DRIVER DONE" >> "$LOG"
while :; do
  grep -q "MATRIX DRIVER DONE" "$ML" 2>/dev/null && break
  sleep 600
done
echo "[after-matrix] $(date '+%F %T') matrix done -> launching lrsteps" >> "$LOG"
nohup bash "$DRV" > "$NOHUP" 2>&1 &
echo "[after-matrix] $(date '+%F %T') lrsteps driver pid $!" >> "$LOG"
