#!/bin/bash
# Track H CNN -- **cifar10/iid 점수원 경쟁** (2026-07-25 재편: 계획서 §2.2 G3).
#
# WHAT THIS FILLS
#   본문 downstream "2-CNN P1 부호-게이트 — cifar10/iid".  dir1 은 8점수원 3-seed
#   완비인데 iid 는 **flirds 만** 있다 → 나머지 7 점수원 + 8소스 T2 관측자를 채운다.
#   부록 "2-CNN P1w" 도 **추가 런 0 으로 동반 산출**된다 — 같은 rundir 이 P1(`gate_v2`)
#   과 P1w(`gatew_v2`) arm 을 함께 낳기 때문(계획서 §7.2).
#
# 96 cells = cifar10/iid x 4 threats x (7 sources + obs) x 3 seeds  (SEED-MAJOR 32/seed)
#   4 threats = clean · label_flip@0.70 · free_rider(zero) · grad_noise  (계획서 §0.1)
#   7 sources = flirds1st lossheur gtg fedsv comfedsv shapleyfl fedif
#     (flirds online arm 은 track_g/rundirs_cnn 의 cifar10 iid 그리드에 이미 있다.)
#
# ⚠ 관측자는 **`C2_OBS_SRCS` 를 지정하지 않는다** → 기본값 = 8소스 전량.
#   기존 iid 관측자(`_obsf`)는 flirds 만 담고 있어 retrain(T2) 열이 비어 있었다.
#   이 잡의 관측자는 dir1 의 `_obs` 와 동형(8소스 t2_sign/t2_signw + t2_random).
#
# STACK: 3090 / conda lora4cl (torch 2.11) = cifar10 dir1 경쟁·track_g 그리드와 동일.
#   recovery 분모(vanilla/oracle_excl/random_excl)가 같은 스택이어야 셀 내부가 정합.
#
# Submit:
#   mkdir -p runs/track_h/_logs
#   sbatch --array=0-31%8    runs/track_h/sbatch_cnn_iid_comp.sh    # seed0 (32)
#   sbatch --array=32-95%8   runs/track_h/sbatch_cnn_iid_comp.sh    # seeds 1-2
# After: python runs/track_h/make_analysis.py
#
#SBATCH --job-name=hiidcomp
#SBATCH --partition=base_suma_rtx3090,dell_rtx3090
# ^ 3090 풀 전체를 대상으로 한다.  base_suma_rtx3090 단독은 07-25 시점 여유 0(총 71장,
#   빈 6장은 draining node01)이고 dell_rtx3090 에 9장이 놀고 있었다 -- JW 실측.
#   `sinfo -o "%P %G %a"` 로 다른 3090 파티션이 보이면 여기에 더 붙일 것(같은 RTX3090
#   이므로 스택 캐비엇 동일).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-95%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_h/_logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}

SRCS=(flirds1st lossheur gtg fedsv comfedsv shapleyfl fedif)      # 7 (flirds = track_g 그리드)
THREATS=(clean label_flip free_rider grad_noise)
TTAGS=(clean label-flip_fr0.70 free-rider grad-noise)
DOSES=("" 0.70 "" "")

IDX=${SLURM_ARRAY_TASK_ID}
SEED=$((IDX / 32)); J=$((IDX % 32))              # seed-major: 32 cells/seed
T=$((J / 8)); C=$((J % 8))                       # 8 cells per threat (7 src + obs)

THREAT=${THREATS[$T]}; TTAG=${TTAGS[$T]}; DOSE=${DOSES[$T]}
EXTRA=(); [ -n "$DOSE" ] && EXTRA+=(C2_FLIP_RATE="$DOSE")

if [ "$C" -lt 7 ]; then
  SRC=${SRCS[$C]}
  ARMS="${SRC}_gate_v2,${SRC}_gatew_v2,${SRC}_mult,${SRC}_zgate_v2"
  TAG=$SRC
else
  ARMS=observer; TAG=obs;   EXTRA+=(C2_T2=1)     # C2_OBS_SRCS 미지정 = 8소스 전량
fi

NAME="cifar10_iid_${TTAG}_${TAG}_seed${SEED}"
echo "[hiidcomp $IDX] $NAME  $(date '+%F %T')"

cd "$REPO/codes"
env "${EXTRA[@]}" PYTHONPATH=. \
  C2_DATASET=cifar10 C2_PARTITION=iid C2_THREAT="$THREAT" C2_SEED="$SEED" \
  C2_MODE=full C2_ARMS="$ARMS" \
  C2_RUN_ROOT="$REPO/runs/track_h/rundirs_cnn" C2_RUN_NAME="$NAME" \
  "$PY" -u experiments/track_c2.py
echo "[hiidcomp $IDX] EXIT=$? $(date '+%F %T')"
