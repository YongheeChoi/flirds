#!/bin/bash
# Track C2-FID -- fmnist SEED2 tail, broken out of the main 144-array (2026-07-25,
# Yonghee: "fmnist 비어있는 부분 전부").
#
# The main sbatch_fid.sh (--array=0-143) already covers these 16 cells at indices
# 128-143 (SEED=2, DSP in {4,5} = fmnist iid/dir1).  fmnist seed0/1 are DONE on
# disk; only seed2 is missing.  This standalone file lets you PRIORITIZE / ISOLATE
# the fmnist seed2 tail on one account without waiting for the cifar10 seed2 tail
# (indices 96-127) to drain first.
#
#   ⚠ AVOID DOUBLE-RUN: if the main array is still live, either
#       (a) cap it to  --array=0-127%8  (cifar10 seed2 only), then run this, OR
#       (b) skip this file and let the main array finish 128-143 itself.
#     Both write the SAME rundir names; last-writer-wins, so a race only wastes GPU.
#
#   ⚠ STACK: run on 3090 / conda lora4cl (torch 2.11) = the SAME stack as fmnist
#     seed0/1, so the 3-seed fidelity set is not split across torch versions.
#     Do NOT move seed2 alone to A6000/torch2.12.
#
# 16 cells = {fmnist x [iid, dir1]} x 8 threats x seed2
#   (clean, free_rider, frrand, grad_noise, lf@{0.15,0.35,0.70}, lf strmain)
#
# Submit:
#   mkdir -p runs/track_c/c2fid/_logs
#   sbatch runs/track_c/c2fid/sbatch_fid_fmnist_s2.sh          # all 16 (%8 QOS cap)
# After: python runs/track_c/c2fid/make_analysis.py            (fills fmnist seed2)
#
#SBATCH --job-name=c2fidfm2
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-15%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_c/c2fid/_logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}

THREATS=(clean free_rider frrand grad_noise label_flip label_flip label_flip label_flip)
TTAGS=(clean free-rider frrand grad-noise label-flip_fr0.15 label-flip_fr0.35 label-flip_fr0.70 label-flip_strmain)
DOSES=("" "" "" "" 0.15 0.35 0.70 "")
PARTS=(iid dir1)                                # 2 fmnist partitions

IDX=${SLURM_ARRAY_TASK_ID}
PART_I=$((IDX / 8)); T=$((IDX % 8))             # 8 threats per partition
SEED=2                                          # fmnist seed2 only

PART=${PARTS[$PART_I]}
THREAT=${THREATS[$T]}; TTAG=${TTAGS[$T]}; DOSE=${DOSES[$T]}
EXTRA=(); [ -n "$DOSE" ] && EXTRA+=(C2_FLIP_RATE="$DOSE")

NAME="fmnist_${PART}_${TTAG}_fid_seed${SEED}"
echo "[c2fidfm2 $IDX] $NAME  $(date '+%F %T')"

cd "$REPO/codes"
env "${EXTRA[@]}" PYTHONPATH=. \
  C2_DATASET=fmnist C2_PARTITION="$PART" C2_THREAT="$THREAT" C2_SEED="$SEED" \
  C2_MODE=full \
  C2FID_RUN_ROOT="$REPO/runs/track_c/c2fid/rundirs" C2FID_RUN_NAME="$NAME" \
  "$PY" -u experiments/track_c2_fid.py
echo "[c2fidfm2 $IDX] EXIT=$? $(date '+%F %T')"
