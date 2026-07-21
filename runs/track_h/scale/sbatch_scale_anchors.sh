#!/bin/bash
# Scale anchors -- oracle_excl (upper anchor) + random_excl (control) at frac=1.0,
# corrupt threats only (clean: oracle_excl == vanilla, skipped -- R1 convention).
# 9 tasks = 3 threats x 3 seeds, one task = one cell running the 2 anchor arms.
# NEW RUN_NAME (*_anch_*) so the committed 4-arm cells are never overwritten.
# Added 2026-07-21 by Yonghee's follow-up decision (reverses the §8 exclusion of
# anchors recorded in RUN_SCALE.md; the spec doc itself stays unmodified).
#
#SBATCH --job-name=c2scanch
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_h/scale/logs/%x_%A_%a.out

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
  C2_FRAC=1.0 \
  C2_ARMS=oracle_excl,random_excl \
  C2_RUN_ROOT="$REPO/runs/track_h/rundirs_cnn_scale" \
  C2_RUN_NAME=cifar10_dir1_frac1.0_${TTAG}_anch_seed${SEED} \
  "$PY" -u experiments/track_c2.py
