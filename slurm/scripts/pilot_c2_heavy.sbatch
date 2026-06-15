#!/bin/bash
#SBATCH -p base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH -J p_c2_heavy
#SBATCH -o /home/chyoyhr/projects/flirds/slurm/logs/%x-%j.out

set -uo pipefail
PY=/home/chyoyhr/anaconda3/envs/lora4cl/bin/python
cd /home/chyoyhr/projects/flirds/codes
export PYTHONPATH=.
echo "=== host=$(hostname) date=$(date)"
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('torch', torch.__version__, '| gpu:', torch.cuda.get_device_name(0))" || exit 1

# REAL grid cell (heaviest class): C2 cifar10 dir1 label_flip main seed0,
# 8 arms + bottom-q% dismissal sweep (10 extra trajectories)
C2_DATASET=cifar10 C2_PARTITION=dir1 C2_THREAT=label_flip C2_STRENGTH=main \
  C2_SEED=0 C2_MODE=full C2_DISMISSAL=1 \
  $PY -u experiments/track_c2.py 2>&1 | while IFS= read -r line; do
    printf '[%(%H:%M:%S)T] %s\n' -1 "$line"; done
exit "${PIPESTATUS[0]}"
