#!/bin/bash
# Track H CNN -- **mnist 점수원 경쟁** (2026-07-25 재편: 계획서 §3.2 G10).
#
# WHAT THIS FILLS
#   부록 downstream "2-CNN P1 — mnist {dir1, iid}" + 부록 "2-CNN P1w — mnist"
#   (P1w 는 같은 rundir 이 낳는 `gatew_v2`/`t2_signw` arm = **추가 런 0**).
#
#   ⚠ 선행 코드 변경 **C-a** 필요(track_c2.py:157 MODEL_FN 에 "mnist": LeNet5).
#     같은 1줄이 G8(c2fid mnist)도 연다.
#
# 216 cells = mnist x {iid, dir1} x 4 threats x (8 sources + obs) x 3 seeds
#   SEED-MAJOR: 72/seed → 파티션당 36 → 위협당 9(8 소스 + 관측자)
#   4 threats = clean · label_flip@0.70 · free_rider(zero) · grad_noise
#   8 sources = flirds + 7 비-flirds  — cifar10 과 달리 mnist 는 track_g 그리드가
#     없으므로 **flirds online arm 도 여기서 생성**한다(그래서 7이 아니라 8).
#
# ⚠ 관측자는 `C2_OBS_SRCS` 미지정 = 8소스 전량 T2(dir1 `_obs` 와 동형).
# STACK: 3090 / conda lora4cl (torch 2.11).
#
# Submit:
#   mkdir -p runs/track_h/_logs
#   sbatch --array=0-71%8     runs/track_h/sbatch_cnn_mnist_comp.sh   # seed0 (72)
#   sbatch --array=72-215%8   runs/track_h/sbatch_cnn_mnist_comp.sh   # seeds 1-2
# After: python runs/track_h/make_analysis.py
#
#SBATCH --job-name=hmncomp
#SBATCH --partition=base_suma_rtx3090,dell_rtx3090
# ^ 3090 풀 전체.  base_suma 단독은 07-25 여유 0 / dell 에 9장 유휴(JW 실측).
#   `sinfo -o "%P %G %a"` 로 다른 3090 파티션 확인 후 추가 가능(스택 동일).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-215%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_h/_logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}

SRCS=(flirds flirds1st lossheur gtg fedsv comfedsv shapleyfl fedif)   # 8
PARTS=(iid dir1)
THREATS=(clean label_flip free_rider grad_noise)
TTAGS=(clean label-flip_fr0.70 free-rider grad-noise)
DOSES=("" 0.70 "" "")

IDX=${SLURM_ARRAY_TASK_ID}
SEED=$((IDX / 72)); J=$((IDX % 72))              # seed-major: 72 cells/seed
PART_I=$((J / 36)); K=$((J % 36))                # 36 per partition
T=$((K / 9)); C=$((K % 9))                       # 9 per threat (8 src + obs)

PART=${PARTS[$PART_I]}
THREAT=${THREATS[$T]}; TTAG=${TTAGS[$T]}; DOSE=${DOSES[$T]}
EXTRA=(); [ -n "$DOSE" ] && EXTRA+=(C2_FLIP_RATE="$DOSE")

if [ "$C" -lt 8 ]; then
  SRC=${SRCS[$C]}
  ARMS="${SRC}_gate_v2,${SRC}_gatew_v2,${SRC}_mult,${SRC}_zgate_v2"
  TAG=$SRC
else
  ARMS=observer; TAG=obs;   EXTRA+=(C2_T2=1)
fi

NAME="mnist_${PART}_${TTAG}_${TAG}_seed${SEED}"
echo "[hmncomp $IDX] $NAME  $(date '+%F %T')"

cd "$REPO/codes"
env "${EXTRA[@]}" PYTHONPATH=. \
  C2_DATASET=mnist C2_PARTITION="$PART" C2_THREAT="$THREAT" C2_SEED="$SEED" \
  C2_MODE=full C2_ARMS="$ARMS" \
  C2_RUN_ROOT="$REPO/runs/track_h/rundirs_cnn" C2_RUN_NAME="$NAME" \
  "$PY" -u experiments/track_c2.py
echo "[hmncomp $IDX] EXIT=$? $(date '+%F %T')"
