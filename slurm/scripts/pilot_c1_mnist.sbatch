#!/bin/bash
#SBATCH -p base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -J p_c1_mnist
#SBATCH -o /home/chyoyhr/projects/flirds/slurm/logs/%x-%j.out

set -uo pipefail
PY=/home/chyoyhr/anaconda3/envs/lora4cl/bin/python
cd /home/chyoyhr/projects/flirds/codes
export PYTHONPATH=.
echo "=== host=$(hostname) date=$(date)"
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('torch', torch.__version__, '| gpu:', torch.cuda.get_device_name(0))" || exit 1

# REAL grid cell: C1 traj+methods, mnist iid seed0 (oracle (a) sharded separately)
C1_DATASET=mnist C1_SCENARIO=iid C1_SEED=0 C1_MODE=full C1_ORACLE_A=0 \
  $PY -u experiments/track_c1.py 2>&1 | while IFS= read -r line; do
    printf '[%(%H:%M:%S)T] %s\n' -1 "$line"; done
exit "${PIPESTATUS[0]}"
