#!/bin/bash
# C-1~C-5 mitigation experiments -- FULL SWEEP driver (plan/matrix: runs/removal_dose/README.md).
# Run ONLY after the pilot (run_pilot.sh) is approved and the per-retrain time is known.
#
# Sections (SEEDS default 0 1 2; single GPU, override GPU=; shard by running sections on
# different GPUs in parallel shells):
#   [1] phase2_matrix silo5 -- 4 threats x REMOVAL=1  (removal curves + full method set: adds
#       Fed-LOO + ComFedSV-at-silo).  poison uses the k=1 exclude removal + D2b install config.
#   [2] Exp B dose sweeps (silo5): noisy graded (NOISY_RATE) / free-rider random (DOSE_MULT) /
#       poison strength (POISON_FRAC).  free-rider zero = removal-only (§1 locked, no dose).
#   [3] Exp A1: track_d anchor5 x SEEDS with ORACLE_A=1 -> removal curves ride the (a) oracle.
#   [4] Exp D (LAST, low-priority): track_d anchor5 seed0 CLIENT_OPT=adamw (bridge lr).
#
# Kill:  pkill -f run_full_sweep.sh; pkill -f phase2_matrix.py; pkill -f track_d.py
set -u
# --- server-migration (2026-07-15): flirds_batch venv + canonical BASE path (same inode as
#     .../WORKSPACE/26msit001_A/edge_ai_lab/yonghee/flirds); old korea_bupj env is gone. ---
STAGE=${STAGE:-/NHNHOME/WORKSPACE/26msit001_A/flirds_batch}
PY=${PY:-$STAGE/venv/bin/python}
REPO=${REPO:-/NHNHOME/26msit001_A/BASE/edge_ai/yonghee/flirds}
CODES=$REPO/codes
ROOT=$REPO/runs/removal_dose
LOGS=$ROOT/_logs
GPU=${GPU:-0}
SEEDS=${SEEDS:-0 1 2}
DO=${DO:-1 2 3 4}                                     # which sections to run (e.g. DO="1 2")
mkdir -p "$LOGS" "$ROOT/rundirs"
cd "$CODES" || { echo "no $CODES (fix REPO=)"; exit 1; }
export PYTHONPATH=.
# HF cache (offline; models+datasets warmed into flirds_batch/hf_home by preflight)
export HOME=${HOME_OVERRIDE:-$STAGE/home}
export HF_HOME=${HF_HOME:-$STAGE/hf_home}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error

note() { echo "[full] $(date '+%F %T') $*" | tee -a "$LOGS/_full.log"; }
has()  { case " $DO " in *" $1 "*) return 0;; *) return 1;; esac; }

matrix() {  # RUN_NAME REGIME THREAT SEED extra-envs...
  local name=$1 reg=$2 thr=$3 sd=$4; shift 4
  note "start $name (gpu$GPU)  envs: $*"
  CUDA_VISIBLE_DEVICES=$GPU RUNDIR_ROOT=$ROOT/rundirs RUN_NAME=$name \
    REGIME=$reg THREAT=$thr SEED=$sd \
    env "$@" "$PY" -u experiments/phase2_matrix.py > "$LOGS/$name.log" 2>&1
  note "done  $name rc=$?"
}
trackd() {  # RUN_NAME SEED extra-envs...
  local name=$1 sd=$2; shift 2
  note "start $name (gpu$GPU)  envs: $*"
  CUDA_VISIBLE_DEVICES=$GPU RUNDIR_ROOT=$ROOT/rundirs_trackd RUN_NAME=$name \
    REGIME=anchor5 SEED=$sd ARMS=0 \
    env "$@" "$PY" -u experiments/track_d.py > "$LOGS/$name.log" 2>&1
  note "done  $name rc=$?"
}

# --- [1] silo5 removal curves, 4 threats x SEEDS (full method set) ---
if has 1; then for sd in $SEEDS; do
  matrix 1B_silo5_noisy_removal_seed$sd            silo5 noisy            $sd  REMOVAL=1
  matrix 1B_silo5_frrand_removal_seed$sd           silo5 freerider_random $sd  REMOVAL=1
  matrix 1B_silo5_frzero_removal_seed$sd           silo5 freerider_zero   $sd  REMOVAL=1
  # poison: k=1 exclude removal + the D2b install config (lr=2e-3 batch=8 epochs=5 frac=0.8)
  matrix 1B_silo5_poison_removal_seed$sd           silo5 poison           $sd  REMOVAL=1 \
         LR=2e-3 BATCH=8 EPOCHS=5 POISON_FRAC=0.8
done; fi

# --- [2] dose-response sweeps (silo5 x SEEDS) ---
if has 2; then for sd in $SEEDS; do
  for NR in 0 0.1 0.25 0.5 0.75 1.0; do
    matrix 1B_silo5_noisy_dose_nr${NR}_seed$sd     silo5 noisy            $sd  NOISY_RATE=$NR
  done
  for DM in 0.25 0.5 1.0 2.0 4.0; do
    matrix 1B_silo5_frrand_dose_dm${DM}_seed$sd    silo5 freerider_random $sd  DOSE_MULT=$DM
  done
  for PF in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    matrix 1B_silo5_poison_dose_pf${PF}_seed$sd    silo5 poison           $sd \
           LR=2e-3 BATCH=8 EPOCHS=5 POISON_FRAC=$PF
  done
done; fi

# --- [3] Exp A1: track_d anchor5 removal curves (free; rides the (a) oracle) ---
if has 3; then for sd in $SEEDS; do
  trackd 1B_anchor5_removal_seed$sd  $sd  ORACLE_A=1
done; fi

# --- [4] Exp D (LAST): AdamW-fidelity, one anchor5 cell ---
if has 4; then
  trackd 1B_anchor5_adamw_seed0  0  ORACLE_A=1 CLIENT_OPT=adamw
fi

note "=== FULL SWEEP DONE (sections: $DO).  Analyze: metrics.json removal_curve / removal_orient /"
note "    poison_removal / (track_d) removal_curve; dose cells' phi.parquet across nr/dm/pf. ==="
