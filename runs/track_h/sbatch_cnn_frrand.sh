#!/bin/bash
# §3 (REMAINING-slurm.md) -- C-fr: CNN frrand full-method + retrain, cifar10 dir1.
#
# Fills the §5.3 CNN "8 scoring source" competition frrand column (currently
# flirds-only) with the 7 non-flirds sources + T2 retrain, mirroring the frzero
# (rundir token `free-rider`) sibling cells already on disk at
#   runs/track_h/rundirs_cnn/cifar10_dir1_free-rider_<src>_seed{0,1,2}
#
# This is the sbatch_strmain.sh template with C2_THREAT=label_flip -> frrand and
# the run-name retagged; strmain's fixed-dose/strength bits do not apply (frrand is
# a random free-rider threat, FRRAND_MULT-driven, no per-client flip rate).  P5 arms
# (cgate/pweight) were dropped 2026-07-23, so this leg is 8 cell types x 3 seeds:
#   J 0-6  P1 cells (7 sources; flirds P1 arms already on disk in the Track G grid,
#          runs/track_g/rundirs_cnn/cifar10_dir1_frrand_g_seed{0,1,2})
#          arms = <src>_gate_v2,<src>_gatew_v2,<src>_mult,<src>_zgate_v2
#   J 7    obs = observer + T2 legacy retrains (C2_T2=1)
#
# Pilot-first (seed-major): --array=0-7 = seed0 -> GPU-h report -> GO -> 8-23.
#   sbatch --array=0-7%8  runs/track_h/sbatch_cnn_frrand.sh   # seed0 pilot
#   sbatch --array=8-23%8 runs/track_h/sbatch_cnn_frrand.sh   # seeds 1-2 after GO
# After: python runs/track_h/make_analysis.py  (adds frrand rows to cnn_competition.csv)
#
#SBATCH --job-name=hfrrand
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --array=0-23%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_h/_logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}

SRCS_P1=(flirds1st lossheur gtg fedsv comfedsv shapleyfl fedif)

IDX=${SLURM_ARRAY_TASK_ID}
SEED=$((IDX / 8)); J=$((IDX % 8))               # seed-major: 8 cell types per seed
EXTRA=()
if [ "$J" -lt 7 ]; then
  SRC=${SRCS_P1[$J]}
  ARMS="${SRC}_gate_v2,${SRC}_gatew_v2,${SRC}_mult,${SRC}_zgate_v2"
  TAG=$SRC
else
  ARMS=observer; TAG=obs;   EXTRA+=(C2_T2=1)
fi

NAME="cifar10_dir1_frrand_${TAG}_seed${SEED}"
echo "[hfrrand $IDX] $NAME  $(date '+%F %T')"

cd "$REPO/codes"
env "${EXTRA[@]}" PYTHONPATH=. \
  C2_DATASET=cifar10 C2_PARTITION=dir1 C2_THREAT=frrand C2_SEED="$SEED" \
  C2_MODE=full C2_ARMS="$ARMS" \
  C2_RUN_ROOT="$REPO/runs/track_h/rundirs_cnn" C2_RUN_NAME="$NAME" \
  "$PY" -u experiments/track_c2.py
echo "[hfrrand $IDX] EXIT=$? $(date '+%F %T')"
