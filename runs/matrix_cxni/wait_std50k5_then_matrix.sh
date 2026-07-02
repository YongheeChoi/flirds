#!/bin/bash
# Yonghee 2026-07-02: run std50k5 (pilot B cells) first, launch the 2x2 matrix after they finish.
# Watches the pilot driver log for the three GPU0-2 chains to complete, then starts run_matrix.sh.
# Detached (nohup) so it survives the Claude session.  run_pilot.sh is left untouched.
set -u
DL=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/runs/probe_signal/_logs/_driver.log
MATRIX=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/runs/matrix_cxni/run_matrix.sh
NOHUP=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/runs/matrix_cxni/_nohup.out
LOG=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/runs/matrix_cxni/_logs/_after_pilot.log
mkdir -p "$(dirname "$LOG")"
echo "[after-pilot] $(date '+%F %T') UP (pid $$) -- waiting for std50k5 (gpu0/1/2 chain COMPLETE)" >> "$LOG"
while :; do
  if grep -q "gpu0 chain COMPLETE" "$DL" 2>/dev/null \
     && grep -q "gpu1 chain COMPLETE" "$DL" 2>/dev/null \
     && grep -q "gpu2 chain COMPLETE" "$DL" 2>/dev/null; then
    break
  fi
  sleep 600
done
echo "[after-pilot] $(date '+%F %T') std50k5 COMPLETE -> launching matrix" >> "$LOG"
nohup bash "$MATRIX" > "$NOHUP" 2>&1 &
echo "[after-pilot] $(date '+%F %T') matrix driver pid $!" >> "$LOG"
