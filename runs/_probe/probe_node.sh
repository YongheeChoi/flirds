#!/bin/bash
#SBATCH --job-name=hjprobe
#SBATCH --partition=suma_a6000,gigabyte_a6000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:03:00
#SBATCH --output=runs/_probe/probe_%j.out
echo "HOST=$(hostname) DATE=$(date '+%F %T')"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "--- nvidia-smi CUDA ---"; nvidia-smi 2>/dev/null | grep -i "CUDA Version" || true
echo "PROBE DONE"
