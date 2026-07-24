#!/bin/bash
# §5 L8 (REMAINING-slurm.md) -- gsm5 main stage: dual (a)+(b) oracle + 9 methods.
# 6 legs = {clean, noisy} x seed{0,1,2}, runner experiments/phase2_matrix.py REGIME=gsm5.
#
# gsm5 auto-sets val_chunk=2 (24 GiB RTX3090 fit; the default 1B val_chunk=10 OOMs on a
# 3090 -- exact chunk-sum, so phi is identical, memory-only).  The (a) oracle = 2^5=32
# retrains x R30 DOMINATES (~8.5 B200-h/leg ~= ~17h on a 3090) -> --time 24h.
#
# PREREQ: model + gsm8k pre-cached into HF_HOME (runs/phase2_matrix/setup_hf_offline
# on the login node; needs the gated HF token).  Offline at run time.
#
# Pilot-first (L8 convention): submit --array=0 (clean seed0) -> measured GPU-h ->
# GO -> --array=1-5.  Full sweep = --array=0-5%8.  rundir: 1B_gsm5_{threat}_nr0.7_s{seed}.
#
#SBATCH --job-name=gsm5
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/phase2_matrix/_logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}
export HF_HOME=${HF_HOME:-/scratch/chyoyhr/hf_home}

THREATS=(clean noisy)
IDX=${SLURM_ARRAY_TASK_ID}
T=$((IDX / 3)); SEED=$((IDX % 3)); THREAT=${THREATS[$T]}

echo "[gsm5 $IDX] 1B_gsm5_${THREAT}_s${SEED}  $(date '+%F %T')  HF_HOME=$HF_HOME"

cd "$REPO/codes"
env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  REGIME=gsm5 THREAT="$THREAT" SEED="$SEED" \
  "$PY" -u experiments/phase2_matrix.py
echo "[gsm5 $IDX] EXIT=$? $(date '+%F %T')"
