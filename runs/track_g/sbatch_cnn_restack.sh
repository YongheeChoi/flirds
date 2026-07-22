#!/bin/bash
# Track G CNN grid -- SAME-STACK re-run of the original 12 cifar10 cells (2026-07-22).
#
# Why: the first 36 runs are torch 2.12.0 on the B200 container; the skew-axis
# extension runs torch 2.11.0 on this Slurm cluster (audit M1 stack boundary).  The
# 2x2 partition table would otherwise read iid/dir1 off one stack and shard/qskew off
# the other.  Re-running the 12 cells here puts the whole table on ONE stack and also
# discharges RERUN_AFTER_REPRO_FIX_2026-07-21 P1 (CNN re-run recommended).
#
#   cifar10 x {iid, dir1} x {clean, free_rider, grad_noise, label_flip@{.15,.35,.70}}
#   x seeds {0,1,2} = 36 runs.  (frrand for iid/dir1 is NOT here -- it never existed
#   on the old stack and is produced fresh by sbatch_cnn_skew.sh GROUP B.)
#
# Lands in rundirs_cnn_restack/ -- the frozen originals in rundirs_cnn/ stay READ-ONLY,
# so the two sets together are a same-config/different-stack reproducibility datapoint.
# make_analysis prefers the restack copy for the headline tables and prints the
# orig-vs-restack drift table.
#
# Submit:  sbatch runs/track_g/sbatch_cnn_restack.sh
#
#SBATCH --job-name=grestack
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-35%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_g/_logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}

ARMS_CORRUPT="vanilla,oracle_excl,random_excl,flirds_gate_v1,flirds_gate_v2,flirds_zgate_v2,flirds_gatew_v2,flirds_gatew_v1,flirds_mult"
ARMS_CLEAN="vanilla,flirds_gate_v1,flirds_gate_v2,flirds_zgate_v2,flirds_gatew_v2,flirds_gatew_v1,flirds_mult"

THREATS=(clean free_rider grad_noise label_flip label_flip label_flip)
TTAGS=(clean free-rider grad-noise label-flip_fr0.15 label-flip_fr0.35 label-flip_fr0.70)
DOSES=("" "" "" 0.15 0.35 0.70)

IDX=${SLURM_ARRAY_TASK_ID}
SEED=$((IDX / 12)); J=$((IDX % 12))             # seed-major: 12 cells per seed
PART=$(echo "iid dir1" | cut -d' ' -f$((J / 6 + 1)))
T=$((J % 6))

THREAT=${THREATS[$T]}; TTAG=${TTAGS[$T]}; DOSE=${DOSES[$T]}
ARMS=$ARMS_CORRUPT; [ "$THREAT" = "clean" ] && ARMS=$ARMS_CLEAN
EXTRA=(); [ -n "$DOSE" ] && EXTRA+=(C2_FLIP_RATE="$DOSE")

NAME="cifar10_${PART}_${TTAG}_g_seed${SEED}"
echo "[restack $IDX] $NAME  $(date '+%F %T')"

cd "$REPO/codes"
env "${EXTRA[@]}" PYTHONPATH=. \
  C2_DATASET=cifar10 C2_PARTITION="$PART" C2_THREAT="$THREAT" C2_SEED="$SEED" \
  C2_MODE=full C2_ARMS="$ARMS" \
  C2_RUN_ROOT="$REPO/runs/track_g/rundirs_cnn_restack" C2_RUN_NAME="$NAME" \
  "$PY" -u experiments/track_c2.py
echo "[restack $IDX] EXIT=$? $(date '+%F %T')"
