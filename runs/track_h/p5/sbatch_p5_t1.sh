#!/bin/bash
# Track H P5 -- Tier T1 (online arms): 8 score sources x {P5-hard cgate, P5-soft pweight}
# on CNN R1 (cifar10 dir1) x {clean, label_flip@0.70, free_rider, grad_noise} x 3 seeds.
# One array task = one track_c2 process = one (source, threat, seed) cell with 2 arms.
# 96 tasks x ~10-16 min --> ~21 GPU-h total.  Spec/procedure: RUN_P5.md (this dir).
#
# SLURM-SERVER SETUP (this is NOT the B200 container -- fill for the local cluster):
#   1) <PARTITION> (+<ACCOUNT> if required)  -- check `sinfo`
#   2) <REPO>  = absolute path of the flirds clone on the Slurm server (3 places incl. --output)
#   3) <VENV_PY> = python of a venv with: torch+torchvision(cuda), numpy, scikit-learn,
#      pandas, pyarrow, pyyaml  (or pass PY=<path> at submit: `PY=... sbatch ...`)
#   4) cifar10 under ~/data (see RUN_P5.md pre-flight -- verify BEFORE submitting,
#      torchvision's cs.toronto.edu download stalls on some networks)
#
#SBATCH --job-name=p5t1
#SBATCH --partition=<PARTITION>
###SBATCH --account=<ACCOUNT>            # uncomment if the cluster requires it
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-95%4                   # throttle: adjust to the node/queue GPU budget
#SBATCH --output=<REPO>/runs/track_h/p5/logs/%x_%A_%a.out

set -euo pipefail

REPO=${REPO:-<REPO>}                     # absolute path to the flirds repo on this cluster
PY=${PY:-<VENV_PY>}                      # venv python (see header)
export PYTHONPATH=. PYTHONUTF8=1

SRCS=(flirds flirds1st lossheur fedif gtg fedsv comfedsv shapleyfl)
THREATS=(clean label_flip free_rider grad_noise)
TTAGS=(clean label-flip_fr0.70 free-rider grad-noise)   # rundir-name tags (match Tier 1)

IDX=${SLURM_ARRAY_TASK_ID}
SRC=${SRCS[$((IDX / 12))]}
REM=$((IDX % 12))
THREAT=${THREATS[$((REM / 3))]}
TTAG=${TTAGS[$((REM / 3))]}
SEED=$((REM % 3))

EXTRA=()
if [ "$THREAT" = "label_flip" ]; then EXTRA+=(C2_FLIP_RATE=0.70); fi

cd "$REPO/codes"
env "${EXTRA[@]}" \
  C2_DATASET=cifar10 C2_PARTITION=dir1 C2_MODE=full \
  C2_THREAT=$THREAT C2_SEED=$SEED \
  C2_ARMS=${SRC}_cgate,${SRC}_pweight \
  C2_RUN_ROOT="$REPO/runs/track_h/rundirs_cnn" \
  C2_RUN_NAME=cifar10_dir1_${TTAG}_${SRC}p5_seed${SEED} \
  "$PY" -u experiments/track_c2.py
