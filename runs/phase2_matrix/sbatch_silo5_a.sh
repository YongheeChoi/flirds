#!/bin/bash
# §5 L8 (REMAINING-slurm.md) -- silo5 (a)-leg: exact 2^5=32 retrain oracle (R=10)
# on the EXISTING canonical 1B_silo5_{clean,noisy,frzero} split (read-only reuse).
# 9 legs = {clean, noisy, frzero} x seed{0,1,2}, runner experiments/track_a_silo5.py.
#
# No HVP here (retrain + no_grad val-loss only) -> safe on the default knobs; cheaper
# than gsm5 (~2.9 B200-h/leg ~= ~6h on a 3090) -> --time 12h.
#
# PREREQ: (1) canonical runs/phase2_matrix/rundirs/1B_silo5_{threat} on disk (present);
# (2) model + 5 silo domains pre-cached in HF_HOME (setup_hf_offline; gated token).
# Produces runs/phase2_matrix/rundirs/1B_silo5_{threat}_aonly_s{seed}; canonical untouched.
#
# Pilot-first: --array=0 (clean seed0) -> measured GPU-h -> GO -> --array=1-8.
#
#SBATCH --job-name=silo5a
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-8%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/phase2_matrix/_logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}
export HF_HOME=${HF_HOME:-/scratch/chyoyhr/hf_home}

THREATS=(clean noisy frzero)
IDX=${SLURM_ARRAY_TASK_ID}
T=$((IDX / 3)); SEED=$((IDX % 3)); THREAT=${THREATS[$T]}

echo "[silo5a $IDX] 1B_silo5_${THREAT}_aonly_s${SEED}  $(date '+%F %T')  HF_HOME=$HF_HOME"

cd "$REPO/codes"
env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  REGIME=silo5 THREAT="$THREAT" SEED="$SEED" \
  "$PY" -u experiments/track_a_silo5.py
echo "[silo5a $IDX] EXIT=$? $(date '+%F %T')"
