#!/bin/bash
#SBATCH -p base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J fl_probe
#SBATCH -o /home/chyoyhr/projects/flirds/slurm/logs/%x-%j.out

set -uo pipefail
PY=/home/chyoyhr/anaconda3/envs/lora4cl/bin/python
cd /home/chyoyhr/projects/flirds/codes
export PYTHONPATH=.
echo "=== host=$(hostname) date=$(date)"
nvidia-smi -L || true
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('torch', torch.__version__, '| gpu:', torch.cuda.get_device_name(0))" || exit 1

$PY -u /home/chyoyhr/projects/flirds/slurm/scripts/probe_oracle_a.py
