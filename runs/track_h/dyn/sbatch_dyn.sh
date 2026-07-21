#!/bin/bash
# Dyn -- per-round corrupt re-draw on the R1 stage (cifar10 dir1 frac0.1):
# 3 threats x 3 seeds, one task = one cell running 5 arms sequentially
# (vanilla -> per-round oracle_excl -> per-round random_excl -> P1 gate_v2 ->
# P5s pweight).  Spec + preregistered predictions: RUN_DYN.md (this dir).
#
#SBATCH --job-name=c2dyn
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_h/dyn/logs/%x_%A_%a.out

set -euo pipefail

REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}
export PYTHONPATH=. PYTHONUTF8=1

THREATS=(label_flip free_rider grad_noise)
TTAGS=(label-flip_fr0.70 free-rider grad-noise)

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
  C2_DYN=1 \
  C2_ARMS=vanilla,oracle_excl,random_excl,flirds_gate_v2,flirds_pweight \
  C2_RUN_ROOT="$REPO/runs/track_h/rundirs_cnn_dyn" \
  C2_RUN_NAME=cifar10_dir1_dyn_${TTAG}_seed${SEED} \
  "$PY" -u experiments/track_c2.py
