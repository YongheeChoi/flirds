#!/bin/bash
#SBATCH -p base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -o /home/chyoyhr/projects/flirds/slurm/logs/%x-%A_%a.out

# Generic array runner: sbatch -J <name> -t <limit> --array=<range> \
#   run_array.sbatch <grid_file> <experiment_script>
# Grid file format: one job per line, "RUN_NAME|ENV ASSIGNMENTS"

set -uo pipefail
GRID_FILE=$1
SCRIPT=$2
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$GRID_FILE")
[ -z "$LINE" ] && { echo "FATAL: no grid line for task id $SLURM_ARRAY_TASK_ID"; exit 1; }
NAME="${LINE%%|*}"
ENVS="${LINE#*|}"

PY=/home/chyoyhr/anaconda3/envs/lora4cl/bin/python
cd /home/chyoyhr/projects/flirds/codes
export PYTHONPATH=.
echo "=== JOB $NAME | host=$(hostname) | start=$(date '+%F %T')"
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('gpu:', torch.cuda.get_device_name(0))" || exit 1

env $ENVS $PY -u "$SCRIPT" 2>&1 | while IFS= read -r line; do
  printf '[%(%H:%M:%S)T] %s\n' -1 "$line"; done
rc=${PIPESTATUS[0]}
echo "=== JOB $NAME | end=$(date '+%F %T') | rc=$rc"
exit $rc
