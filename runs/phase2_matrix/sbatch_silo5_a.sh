#!/bin/bash
# L8 (REMAINING-slurm-HJ.md §1) -- silo5 (a)-leg: exact 2^5=32 retrain oracle (R=10)
# on the EXISTING canonical 1B_silo5_{clean,noisy,frzero} split (read-only reuse).
# 9 legs = {clean, noisy, frzero} x seed{0,1,2}, runner experiments/track_a_silo5.py.
#
# GPU: A6000 48GB (HJ account).  24GB(3090) OOM'd -- 1B full SFT retrain peaks ~25GB at
# batch16, and batch MUST stay 16 (smaller changes the (a) retrain trajectory = different
# game).  No HVP here (retrain + no_grad val-loss only); ~2.9 GPU-h/leg -> --time 12h.
#
# PREREQ: (1) canonical runs/phase2_matrix/rundirs/1B_silo5_{threat} on disk (present);
# (2) model + 5 silo domains pre-cached in HF_HOME (setup_hf_offline; gated token).
# Produces runs/phase2_matrix/rundirs/1B_silo5_{threat}_aonly_s{seed}; canonical untouched.
#
# Pilot-first: --array=0 (clean seed0) -> measured GPU-h -> GO -> --array=1-8.
# Array is SEED-MAJOR (change #3, seed0 우선): 0-2 = seed0 {clean,noisy,frzero},
# 3-5 = seed1, 6-8 = seed2.  So the pilot + gate's 1-8 expansion finish ALL of seed0
# (clean via pilot, noisy/frzero via 1,2) before any seed1/2 leg starts.
#
#SBATCH --job-name=silo5a
#SBATCH --partition=suma_a6000,gigabyte_a6000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-8%8
#SBATCH --output=runs/phase2_matrix/_logs/%x_%A_%a.out
# A6000 48GB via base_qos (all partitions AllowAccounts=ALL; QOS gates).  If your
# account's A6000 QOS is named differently, edit --qos (a wrong token fail-fasts).
# --output is relative to the SUBMIT dir -> submit from the repo root after:
#   mkdir -p runs/phase2_matrix/_logs && sbatch runs/phase2_matrix/sbatch_silo5_a.sh

set -u
REPO=${REPO:-$HOME/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}   # shared lora4cl (torch2.11); override PY= if not readable
export HF_HOME=${HF_HOME:-/scratch/chyoyhr/hf_home}         # 5 silo domains + 1B model + gsm8k; override HF_HOME= to your cache

THREATS=(clean noisy frzero)
IDX=${SLURM_ARRAY_TASK_ID}
SEED=$((IDX / 3)); T=$((IDX % 3)); THREAT=${THREATS[$T]}   # seed-major: array 0 = clean seed0 (gate GLOB safe)

echo "[silo5a $IDX] 1B_silo5_${THREAT}_aonly_s${SEED}  $(date '+%F %T')  HF_HOME=$HF_HOME"

cd "$REPO/codes"
env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  REGIME=silo5 THREAT="$THREAT" SEED="$SEED" \
  "$PY" -u experiments/track_a_silo5.py
echo "[silo5a $IDX] EXIT=$? $(date '+%F %T')"
