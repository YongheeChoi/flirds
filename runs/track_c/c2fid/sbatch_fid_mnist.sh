#!/bin/bash
# Track C2-FID -- **mnist stage** (2026-07-25 재편: 계획서 §3.1 G8).
#
# WHAT THIS FILLS
#   부록 fidelity "1A-CNN mnist {dir1, iid}" + 부록 detection "mnist φ-AUROC".
#   두 축이 **같은 rundir**에서 나온다(§5.2 = §5.4 공용).  cifar10 본문 무대와 세팅
#   동일, 데이터셋만 mnist.  fmnist 무대는 이번 재편에서 표에서 빠졌다(mnist가 대체).
#
#   ⚠ 선행 코드 변경 **C-a** 필요:
#       codes/experiments/track_c2.py:157  MODEL_FN = partial({"cifar10": FedSVCNN,
#         "fmnist": LeNet5, **"mnist": LeNet5**}[DATASET], width=WIDTH)
#     flirds/data/cnn.py 는 mnist 로더·정규화를 이미 갖고 있고(l.11·24),
#     track_c2_fid.py 는 `c2.MODEL_FN`/`c2.DATASET` 를 참조하므로 이 1줄로 파급된다.
#
# 24 cells = {mnist x [iid, dir1]} x 4 threats x 3 seeds   (SEED-MAJOR: 8/seed)
#   4 threats = 계획서 §0.1 CNN 오염축 3종 + clean 앵커
#     clean · label_flip@0.70 · free_rider(zero) · grad_noise
#   (lf dose 0.15/0.35 · frrand · strmain 은 축 밖 = 돌리지 않는다.)
#
# STACK: 3090 / conda lora4cl (torch 2.11) — cifar10 c2fid 와 같은 스택 유지.
#
# Submit:
#   mkdir -p runs/track_c/c2fid/_logs
#   sbatch --array=0-7%8    runs/track_c/c2fid/sbatch_fid_mnist.sh    # seed0 (8)
#   sbatch --array=8-23%8   runs/track_c/c2fid/sbatch_fid_mnist.sh    # seeds 1-2
# After: python runs/track_c/c2fid/make_analysis.py
#
#SBATCH --job-name=c2fidmn
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-23%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_c/c2fid/_logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}

THREATS=(clean label_flip free_rider grad_noise)
TTAGS=(clean label-flip_fr0.70 free-rider grad-noise)
DOSES=("" 0.70 "" "")
PARTS=(iid dir1)

IDX=${SLURM_ARRAY_TASK_ID}
SEED=$((IDX / 8)); J=$((IDX % 8))                # seed-major: 8 cells/seed
PART_I=$((J / 4)); T=$((J % 4))                  # 4 threats per partition

PART=${PARTS[$PART_I]}
THREAT=${THREATS[$T]}; TTAG=${TTAGS[$T]}; DOSE=${DOSES[$T]}
EXTRA=(); [ -n "$DOSE" ] && EXTRA+=(C2_FLIP_RATE="$DOSE")

NAME="mnist_${PART}_${TTAG}_fid_seed${SEED}"
echo "[c2fidmn $IDX] $NAME  $(date '+%F %T')"

cd "$REPO/codes"
env "${EXTRA[@]}" PYTHONPATH=. \
  C2_DATASET=mnist C2_PARTITION="$PART" C2_THREAT="$THREAT" C2_SEED="$SEED" \
  C2_MODE=full \
  C2FID_RUN_ROOT="$REPO/runs/track_c/c2fid/rundirs" C2FID_RUN_NAME="$NAME" \
  "$PY" -u experiments/track_c2_fid.py
echo "[c2fidmn $IDX] EXIT=$? $(date '+%F %T')"
