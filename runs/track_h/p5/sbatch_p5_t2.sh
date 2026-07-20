#!/bin/bash
# Track H P5 -- Tier T2 (retrain arms): one observer pass per cell (8 sources scored
# on the vanilla-identical trajectory), then per-source retrains from the FINAL
# training-observed stats: t2_csign_<src> (UCB kept-set) + t2_pw_<src> (static
# Phi(t)-weight), kept/weight-deduped, + size-matched t2_random_k controls.
# Legacy t2_sign/t2_signw are SKIPPED (already on disk; C2_T2_LEGACY=0).
# 12 tasks (4 threats x 3 seeds) x ~1-1.5 h --> ~15 GPU-h.  Spec: RUN_P5.md.
#
# SLURM-SERVER SETUP: fill <PARTITION>/<REPO>/<VENV_PY> -- see sbatch_p5_t1.sh header.
#
#SBATCH --job-name=p5t2
#SBATCH --partition=<PARTITION>
###SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-11%4
#SBATCH --output=<REPO>/runs/track_h/p5/logs/%x_%A_%a.out

set -euo pipefail

REPO=${REPO:-<REPO>}
PY=${PY:-<VENV_PY>}
export PYTHONPATH=. PYTHONUTF8=1

THREATS=(clean label_flip free_rider grad_noise)
TTAGS=(clean label-flip_fr0.70 free-rider grad-noise)

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
  C2_ARMS=observer C2_T2=1 C2_T2_P5=1 C2_T2_LEGACY=0 \
  C2_RUN_ROOT="$REPO/runs/track_h/rundirs_cnn" \
  C2_RUN_NAME=cifar10_dir1_${TTAG}_obsp5_seed${SEED} \
  "$PY" -u experiments/track_c2.py
