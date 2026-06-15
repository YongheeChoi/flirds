#!/bin/bash
#SBATCH -p base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J fl_smoke
#SBATCH -o /home/chyoyhr/projects/flirds/slurm/logs/%x-%j.out

set -uo pipefail
PY=/home/chyoyhr/anaconda3/envs/lora4cl/bin/python
cd /home/chyoyhr/projects/flirds/codes
export PYTHONPATH=.
echo "=== host=$(hostname) date=$(date) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('torch', torch.__version__, '| gpu:', torch.cuda.get_device_name(0))" || exit 1

echo "=== [1/4] C1 smoke mnist iid (incl. tiny 2^6 (a)-oracle)"
C1_DATASET=mnist C1_SCENARIO=iid C1_SEED=0 C1_MODE=smoke \
  C1_RUN_ROOT=runs/track_c1_smoke $PY -u experiments/track_c1.py || exit 1

echo "=== [2/4] C1 smoke cifar10 label_flip (ladder + corruptor path)"
C1_DATASET=cifar10 C1_SCENARIO=label_flip C1_SEED=0 C1_MODE=smoke \
  C1_RUN_ROOT=runs/track_c1_smoke $PY -u experiments/track_c1.py || exit 1

echo "=== [3/4] C2 smoke cifar10 dir1 label_flip + dismissal (8 arms + q-sweep)"
C2_DATASET=cifar10 C2_PARTITION=dir1 C2_THREAT=label_flip C2_SEED=0 C2_MODE=smoke \
  C2_DISMISSAL=1 C2_RUN_ROOT=runs/track_c2_smoke $PY -u experiments/track_c2.py || exit 1

echo "=== [4/4] C2 smoke fmnist shard free_rider (LeNet + delta_transform path)"
C2_DATASET=fmnist C2_PARTITION=shard C2_THREAT=free_rider C2_SEED=0 C2_MODE=smoke \
  C2_RUN_ROOT=runs/track_c2_smoke $PY -u experiments/track_c2.py || exit 1

echo "PREFLIGHT SMOKE ALL OK"
