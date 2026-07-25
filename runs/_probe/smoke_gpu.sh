#!/bin/bash
#SBATCH --job-name=hjsmoke
#SBATCH --partition=suma_a6000,gigabyte_a6000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=runs/_probe/smoke_gpu_%j.out
set -u
REPO=/home/rlaguswls186790/flirds
PY=/home/rlaguswls186790/miniconda3/envs/flirds/bin/python
export HF_HOME=/scratch/rlaguswls186790/hf_home
cd "$REPO/codes"
echo "HOST=$(hostname)  $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "| cuda avail", torch.cuda.is_available(),
      "| dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

echo "=== SMOKE A: track_a silo5 (a)-leg (gpt2, N=5, R=1) ==="
env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SMOKE_MODEL=gpt2 REGIME=silo5 THREAT=frzero SEED=0 ROUNDS=1 MAX_STEPS=1 TRAIN=16 VAL=8 \
  VAL_CHUNK=3 VAL_MAXLEN=64 BATCH=4 PERSIST=0 \
  "$PY" -u experiments/track_a_silo5.py 2>&1 | grep -viE "Adding EOS|Tokenizing|examples/s]|train_runtime" | tail -12
echo "track_a rc=${PIPESTATUS[0]}"

echo "=== SMOKE B: track_g gsm50k5 online (gpt2, fedif gate + observer) ==="
env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SMOKE_MODEL=gpt2 REGIME=gsm50k5 THREAT=clean SEED=0 \
  ARMS=fedif_gate_v2 OBS_SOURCES=fedif T2=0 T2_LEGACY=0 T2_P5=0 \
  N_CLIENTS=6 K_ABS=2 ROUNDS=3 MAX_STEPS=2 VAL=8 BURN_IN=2 VAL_CHUNK=3 VAL_MAXLEN=64 BATCH=4 PERSIST=0 \
  "$PY" -u experiments/track_g.py 2>&1 | grep -viE "Adding EOS|Tokenizing|examples/s]|train_runtime" | tail -14
echo "track_g rc=${PIPESTATUS[0]}"
echo "GPU SMOKE DONE"
