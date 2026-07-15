#!/bin/bash
# C-1~C-5 mitigation experiments -- PILOT driver (plan/matrix: runs/removal_dose/README.md).
#
# Scope: threat=noisy x silo5 x seed0 x reduced method set.  Runs the two GPU code paths
# once, end-to-end:
#   Exp A2  removal/selection curve (REAL retrain; the GPU cost center)
#   Exp B   noisy graded dose sweep (phi vs corruption rate)
# Primary deliverable: the silo5 SINGLE-FL-RETRAIN wall-clock (metrics.json:removal_retrain_s)
# -- no measurement exists yet, and the full-sweep GPU budget can't be sized without it.
# Report to Yonghee -> approve -> run_full_sweep.sh (§7.2).
#
# Already DONE offline (no GPU, this repo): Exp C target-stability
#   -> runs/track_d/target_stability.csv + runs/phase2_matrix/target_stability.csv
# Exp A1 (track_d anchor5 removal curve) rides on the paper's anchor5 (a)-oracle re-run and
# lives in run_full_sweep.sh (it is NOT cheap standalone -- the (a) oracle is hours at 1B).
#
# Kill:  pkill -f run_pilot.sh; pkill -f phase2_matrix.py
set -u
# --- server-migration (2026-07-15): flirds_batch venv + canonical BASE path (same inode as
#     .../WORKSPACE/26msit001_A/edge_ai_lab/yonghee/flirds); old korea_bupj env is gone. ---
STAGE=${STAGE:-/NHNHOME/WORKSPACE/26msit001_A/flirds_batch}
PY=${PY:-$STAGE/venv/bin/python}
REPO=${REPO:-/NHNHOME/26msit001_A/BASE/edge_ai/yonghee/flirds}
CODES=$REPO/codes
ROOT=$REPO/runs/removal_dose
LOGS=$ROOT/_logs
GPU=${GPU:-0}                                          # GPU 0 free; override GPU=3
RED=${RED:-Flirds,Flirds1st,GTG,ShapleyFL,loss-heur}  # reduced method set (§7.1)
mkdir -p "$LOGS" "$ROOT/rundirs"
cd "$CODES" || { echo "no $CODES (fix REPO=)"; exit 1; }
export PYTHONPATH=.
# HF cache (offline; models+datasets warmed into flirds_batch/hf_home by preflight)
export HOME=${HOME_OVERRIDE:-$STAGE/home}
export HF_HOME=${HF_HOME:-$STAGE/hf_home}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error

note() { echo "[pilot] $(date '+%F %T') $*" | tee -a "$LOGS/_pilot.log"; }

matrix() {  # RUN_NAME extra-envs...   (all pilot cells are silo5/noisy/seed0)
  local name=$1; shift
  note "start $name (gpu$GPU)  envs: $*"
  CUDA_VISIBLE_DEVICES=$GPU RUNDIR_ROOT=$ROOT/rundirs RUN_NAME=$name \
    REGIME=silo5 THREAT=noisy SEED=0 \
    env "$@" "$PY" -u experiments/phase2_matrix.py > "$LOGS/$name.log" 2>&1
  note "done  $name rc=$?"
}

note "=== Exp A2: removal/selection curve (real retrain) -- noisy silo5 seed0, reduced methods ==="
matrix 1B_silo5_noisy_removal_seed0  REMOVAL=1  REMOVAL_METHODS="$RED"

note "=== Exp B: noisy graded dose sweep (phi vs rate) -- silo5 seed0 ==="
for NR in 0 0.1 0.25 0.5 0.75 1.0; do
  matrix 1B_silo5_noisy_dose_nr${NR}_seed0  NOISY_RATE=$NR
done

note "=== PILOT DONE.  Key outputs ==="
note "  removal : $ROOT/rundirs/1B_silo5_noisy_removal_seed0/metrics.json  (removal_curve + removal_retrain_s)"
note "  dose    : $ROOT/rundirs/1B_silo5_noisy_dose_nr*_seed0/  (phi per rate; grep '\\[removal\\]' logs for timing)"
note "  ACTION  : report removal_retrain_s (single-FL retrain s) -> size the full sweep, then run_full_sweep.sh"
