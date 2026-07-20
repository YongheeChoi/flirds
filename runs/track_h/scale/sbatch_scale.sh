#!/bin/bash
# Scale -- CNN full participation 100/100: 4 threats x 3 seeds, one task = one cell
# running 4 arms sequentially (observer[=vanilla+phi] -> flirds_gate_v2[P1] ->
# flirds_cgate[P5-hard] -> flirds_pweight[P5-soft]).  Identical to the R1 stage
# except C2_FRAC=1.0; observer scores flirds ONLY (C2_OBS_SRCS -- coalition
# providers are O(2^k)..O(k^2) per round and never finish at k=100, RUN_SCALE.md §1).
# 12 tasks x ~4-8 h (pilot confirms) --> ~60-90 GPU-h.  Spec: RUN_SCALE.md.
#
# SLURM-SERVER SETUP: fill <PARTITION>/<REPO>/<VENV_PY> -- see runs/track_h/p5/
# sbatch_p5_t1.sh header (same server/venv/data prep).
# Submit pilot first (seed0):  REPO=... PY=... sbatch --array=0,3,6,9 sbatch_scale.sh
# then the rest:               REPO=... PY=... sbatch --array=1,2,4,5,7,8,10,11 sbatch_scale.sh
#
#SBATCH --job-name=c2scale
#SBATCH --partition=<PARTITION>
###SBATCH --account=<ACCOUNT>            # uncomment if the cluster requires it
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%4                   # throttle: adjust to the node/queue GPU budget
#SBATCH --output=<REPO>/runs/track_h/scale/logs/%x_%A_%a.out

set -euo pipefail

REPO=${REPO:-<REPO>}                     # absolute path to the flirds repo on this cluster
PY=${PY:-<VENV_PY>}                      # venv python (see header)
export PYTHONPATH=. PYTHONUTF8=1

THREATS=(clean label_flip free_rider grad_noise)
TTAGS=(clean label-flip_fr0.70 free-rider grad-noise)   # rundir-name tags (match R1/P5)

IDX=${SLURM_ARRAY_TASK_ID}
THREAT=${THREATS[$((IDX / 3))]}
TTAG=${TTAGS[$((IDX / 3))]}
SEED=$((IDX % 3))

EXTRA=()
if [ "$THREAT" = "label_flip" ]; then EXTRA+=(C2_FLIP_RATE=0.70); fi

cd "$REPO/codes"
env "${EXTRA[@]}" \
  C2_DATASET=cifar10 C2_PARTITION=dir1 C2_MODE=full \
  C2_THREAT=$THREAT C2_SEED=$SEED \
  C2_FRAC=1.0 C2_OBS_SRCS=flirds \
  C2_ARMS=observer,flirds_gate_v2,flirds_cgate,flirds_pweight \
  C2_RUN_ROOT="$REPO/runs/track_h/rundirs_cnn_scale" \
  C2_RUN_NAME=cifar10_dir1_frac1.0_${TTAG}_seed${SEED} \
  "$PY" -u experiments/track_c2.py
