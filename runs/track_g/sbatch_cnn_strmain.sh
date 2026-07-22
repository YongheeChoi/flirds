#!/bin/bash
# Track G CNN grid -- label_flip `strmain` cell (2026-07-22, Yonghee).
#
# The 8th threat cell: FedCorr (rho,tau) per-client rate ~ U(0.5, 1.0) instead of one
# fixed dose.  Motivation is the FIDELITY leg -- a per-client corruption-strength
# ladder INSIDE one cell is what makes C1's `spearman_vs_rate` (does phi fall
# monotonically as a client gets dirtier?) definable; under a fixed dose the rate
# vector has two distinct values and the metric degenerates to the detection AUROC.
# Run here too so every fidelity cell has a downstream twin on the SAME frozen
# trajectory, and so the C2 soft grid's own `label-flip_strmain` cells become a
# same-cell contrast (the fixed-dose cells never had one).
#
#   {cifar10 x [iid, dir1, shard, qskew], fmnist x [iid, dir1]} x label_flip(strmain)
#   x seeds {0,1,2} = 18 runs.  Additive -- nothing already on disk is invalidated.
#
# Submit:  sbatch runs/track_g/sbatch_cnn_strmain.sh
#
#SBATCH --job-name=gstrmain
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-17%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_g/_logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}

ARMS="vanilla,oracle_excl,random_excl,flirds_gate_v1,flirds_gate_v2,flirds_zgate_v2,flirds_gatew_v2,flirds_gatew_v1,flirds_mult"

IDX=${SLURM_ARRAY_TASK_ID}
SEED=$((IDX / 6)); J=$((IDX % 6))               # seed-major: 6 cells per seed
DS=$(echo "cifar10 cifar10 cifar10 cifar10 fmnist fmnist" | cut -d' ' -f$((J + 1)))
PART=$(echo "iid dir1 shard qskew iid dir1" | cut -d' ' -f$((J + 1)))   # NO padding: cut -d' ' makes empty fields

NAME="${DS}_${PART}_label-flip_strmain_g_seed${SEED}"
echo "[strmain $IDX] $NAME  $(date '+%F %T')"

cd "$REPO/codes"
PYTHONPATH=. \
  C2_DATASET="$DS" C2_PARTITION="$PART" C2_THREAT=label_flip C2_SEED="$SEED" \
  C2_MODE=full C2_ARMS="$ARMS" \
  C2_RUN_ROOT="$REPO/runs/track_g/rundirs_cnn" C2_RUN_NAME="$NAME" \
  "$PY" -u experiments/track_c2.py
echo "[strmain $IDX] EXIT=$? $(date '+%F %T')"
