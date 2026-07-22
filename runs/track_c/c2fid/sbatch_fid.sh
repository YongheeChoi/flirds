#!/bin/bash
# Track C2-FID -- fidelity vs (b) oracle on the C2 cross-device stage (2026-07-23).
#
# 144 cells = {cifar10 x [iid,dir1,shard,qskew], fmnist x [iid,dir1]} x 8 threats
# (clean, free_rider, frrand, grad_noise, lf@{0.15,0.35,0.70}, lf strmain) x
# seeds {0,1,2} -- EXACTLY the downstream Track G combos, so every fidelity cell
# has a downstream twin on the bit-identical vanilla trajectory (join, plan §4.9).
# Pre-registered predictions F-1..F-4: README.md here (registered BEFORE the main
# run; the pilot only measures cost).
#
# PILOT FIRST (Yonghee 07-22: 구현 대기 + 1셀 파일럿, submit after the current
# queue drains):
#   sbatch --array=11 runs/track_c/c2fid/sbatch_fid.sh    # cifar10 dir1 grad_noise seed0
# report the measured GPU-h, then the full grid:
#   sbatch runs/track_c/c2fid/sbatch_fid.sh               # all 144 (%8 QOS cap)
# If the pilot shows the (b) wall dominating, round-shard instead (README §샤딩):
# C2FID_B_ROUNDS=lo:hi oracle shards + one C2FID_ORACLE_B=0 methods run.
#
#SBATCH --job-name=c2fid
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
# 8h, not the grid's 3h: the (b) oracle is 122,880 utility evals + GTG/FedSV/ShapleyFL
# add their own per-round coalition sweeps, and NONE of it is measured yet -- a wall
# truncation would destroy the very measurement the pilot exists to take.  Partition
# MaxTime is 14 days, so the request costs nothing but backfill priority.
#SBATCH --array=0-143%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_c/c2fid/_logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}

THREATS=(clean free_rider frrand grad_noise label_flip label_flip label_flip label_flip)
TTAGS=(clean free-rider frrand grad-noise label-flip_fr0.15 label-flip_fr0.35 label-flip_fr0.70 label-flip_strmain)
DOSES=("" "" "" "" 0.15 0.35 0.70 "")

IDX=${SLURM_ARRAY_TASK_ID}
SEED=$((IDX / 48)); J=$((IDX % 48))             # seed-major: 48 cells per seed
DSP=$((J / 8)); T=$((J % 8))
DS=$(echo "cifar10 cifar10 cifar10 cifar10 fmnist fmnist" | cut -d' ' -f$((DSP + 1)))
PART=$(echo "iid dir1 shard qskew iid dir1" | cut -d' ' -f$((DSP + 1)))   # NO padding: cut -d' ' makes empty fields

THREAT=${THREATS[$T]}; TTAG=${TTAGS[$T]}; DOSE=${DOSES[$T]}
EXTRA=(); [ -n "$DOSE" ] && EXTRA+=(C2_FLIP_RATE="$DOSE")

NAME="${DS}_${PART}_${TTAG}_fid_seed${SEED}"
echo "[c2fid $IDX] $NAME  $(date '+%F %T')"

cd "$REPO/codes"
env "${EXTRA[@]}" PYTHONPATH=. \
  C2_DATASET="$DS" C2_PARTITION="$PART" C2_THREAT="$THREAT" C2_SEED="$SEED" \
  C2_MODE=full \
  C2FID_RUN_ROOT="$REPO/runs/track_c/c2fid/rundirs" C2FID_RUN_NAME="$NAME" \
  "$PY" -u experiments/track_c2_fid.py
echo "[c2fid $IDX] EXIT=$? $(date '+%F %T')"
