#!/bin/bash
# Track G CNN grid -- skew-axis decomposition + fmnist + frrand (2026-07-22 spec).
# Completes the partition axis into 2x2 and adds the second dataset / the pure-random
# update threat, WITHOUT touching the 36 runs already on disk (read-only).
#
#   GROUP A (new dataset/partition cells, 84 runs)
#     {cifar10 x [shard, qskew], fmnist x [iid, dir1]}
#       x {clean, free_rider, frrand, grad_noise, label_flip@[0.15, 0.35, 0.70]}
#       x seeds {0,1,2}
#   GROUP B (frrand backfill on the EXISTING cells, 6 runs)
#     cifar10 x [iid, dir1] x frrand x seeds {0,1,2}
#
# One array task = one track_c2 process = one cell.  FL/gate HP are the grid's
# defaults verbatim (N=100, C=0.1, R=120, E=5, lr .01, burn_in 10, tau 0, min_obs 2,
# probation 5, alpha 1) -- per-cell tuning is forbidden (spec 4.3).
# Pre-registered predictions H-K1..H-K6: README.md "확장 ②".
#
# Task index is SEED-MAJOR (30 cells per seed) so a pilot is a prefix:
# Submit:  sbatch --array=0-29%8 runs/track_g/sbatch_cnn_skew.sh   (seed-0 pilot, 30 cells)
#          sbatch --array=30-89%8 runs/track_g/sbatch_cnn_skew.sh  (seeds 1-2 after the GO)
#          sbatch runs/track_g/sbatch_cnn_skew.sh                  (all 90 at once)
# After:   python runs/track_g/make_analysis.py
#
#SBATCH --job-name=gskew
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-89%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_g/_logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}

ARMS_CORRUPT="vanilla,oracle_excl,random_excl,flirds_gate_v1,flirds_gate_v2,flirds_zgate_v2,flirds_gatew_v2,flirds_gatew_v1,flirds_mult"
ARMS_CLEAN="vanilla,flirds_gate_v1,flirds_gate_v2,flirds_zgate_v2,flirds_gatew_v2,flirds_gatew_v1,flirds_mult"

THREATS=(clean free_rider frrand grad_noise label_flip label_flip label_flip)
TTAGS=(clean free-rider frrand grad-noise label-flip_fr0.15 label-flip_fr0.35 label-flip_fr0.70)
DOSES=("" "" "" "" 0.15 0.35 0.70)

IDX=${SLURM_ARRAY_TASK_ID}
SEED=$((IDX / 30)); J=$((IDX % 30))             # seed-major: 30 cells per seed
if [ "$J" -lt 28 ]; then                        # GROUP A: 4 (ds,part) x 7 threats
  DSP=$((J / 7)); T=$((J % 7))
  DS=$(echo "cifar10 cifar10 fmnist fmnist" | cut -d' ' -f$((DSP + 1)))
  PART=$(echo "shard qskew iid dir1" | cut -d' ' -f$((DSP + 1)))
else                                            # GROUP B: frrand on the existing cells
  DS=cifar10; T=2                               # threat index 2 == frrand
  PART=$(echo "iid dir1" | cut -d' ' -f$((J - 28 + 1)))
fi

THREAT=${THREATS[$T]}; TTAG=${TTAGS[$T]}; DOSE=${DOSES[$T]}
ARMS=$ARMS_CORRUPT; [ "$THREAT" = "clean" ] && ARMS=$ARMS_CLEAN
EXTRA=(); [ -n "$DOSE" ] && EXTRA+=(C2_FLIP_RATE="$DOSE")

NAME="${DS}_${PART}_${TTAG}_g_seed${SEED}"
echo "[task $IDX] $NAME  arms=$(echo "$ARMS" | tr ',' ' ' | wc -w)  $(date '+%F %T')"

cd "$REPO/codes"
env "${EXTRA[@]}" PYTHONPATH=. \
  C2_DATASET="$DS" C2_PARTITION="$PART" C2_THREAT="$THREAT" C2_SEED="$SEED" \
  C2_MODE=full C2_ARMS="$ARMS" \
  C2_RUN_ROOT="$REPO/runs/track_g/rundirs_cnn" C2_RUN_NAME="$NAME" \
  "$PY" -u experiments/track_c2.py
echo "[task $IDX] EXIT=$? $(date '+%F %T')"
