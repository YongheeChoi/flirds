#!/bin/bash
# Exp A3 CNN removal-curve -- **확정 오염축 정렬 재실행** (2026-07-25 재편: REMAINING-slurm-YH §6 G6).
#
# WHY
#   기존 removal 셀({feature-noise, label-flip 사다리, iid} x {mnist, cifar10} x 3seed
#   = runs/removal_dose/rundirs_cnn 18개)은 확정 오염축 3종과 겹치지 않는다.  worst-first
#   제거 -> acc 분리(순위가 성능을 낳는다는 인과)를 **본문 무대(cifar10/iid)의 오염축**에서
#   다시 낸다.  frzero = free_rider(zero delta).
#
#   ⚠ 선행 코드 변경 **C-b** 필요 — track_c1 의 C1_PARTITION / C1_THREAT / C1_FLIP_RATE.
#     러너 자체는 그대로(C1_REMOVAL=1 경로 무수정): 오염축 토큰만 C-b 와 공유한다.
#
# 9 cells = cifar10/iid x {free_rider, grad_noise, label_flip@0.70} x 3 seeds
#   (SEED-MAJOR 3/seed.  clean 앵커는 제거곡선에 의미가 없어 축에서 뺀다.)
#   C1_ORACLE_A=0: removal 은 (a) 2^N 캐시 불필요(옵션 1, A2 와 균일).
#   C1_RIPPLE=0  : Ripple 은 자기-궤적이라 removal 순위 대상 제외(기존 스윕과 동일 규약).
#                  free_rider/grad_noise 셀에서는 C-b 가 어차피 Ripple 을 건너뛴다.
#   rundir 이름 = track_c1 기본 규칙 `{ds}_{scenario}_seed{n}` -> `cifar10_iid-free-rider_seed0`
#   (기존 `cifar10_iid_seed0` 등과 충돌 없음 — 기존 rundir 은 read-only).
#
# 비용 추정 ~10-20 GPU-h(9셀).  removal 은 kept-set 캐시라 셀당 retrain 수가 방법 순위
#   일치도에 따라 변한다(<= 2^n, 실측은 metrics.json removal_retrain_s).
#
# STACK: 3090 / conda lora4cl (torch 2.11) — 기존 removal·C1 rundir 과 동일.
#
# Submit:
#   mkdir -p runs/removal_dose/_logs
#   sbatch --array=0-2%8   runs/removal_dose/sbatch_cnn_removal_axis.sh   # seed0
#   sbatch --array=3-8%8   runs/removal_dose/sbatch_cnn_removal_axis.sh   # seeds 1-2
#
#SBATCH --job-name=c1rmax
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-8%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/removal_dose/_logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}

THREATS=(free_rider grad_noise label_flip)
TTAGS=(free-rider grad-noise label-flip_fr0.70)
DOSES=("" "" 0.70)

IDX=${SLURM_ARRAY_TASK_ID}
SEED=$((IDX / 3)); T=$((IDX % 3))                 # seed-major: 3 cells/seed

THREAT=${THREATS[$T]}; TTAG=${TTAGS[$T]}; DOSE=${DOSES[$T]}
EXTRA=(); [ -n "$DOSE" ] && EXTRA+=(C1_FLIP_RATE="$DOSE")

echo "[c1rmax $IDX] cifar10_iid_${TTAG}_seed${SEED}  $(date '+%F %T')"

cd "$REPO/codes"
env "${EXTRA[@]}" PYTHONPATH=. \
  C1_DATASET=cifar10 C1_PARTITION=iid C1_THREAT="$THREAT" C1_SEED="$SEED" \
  C1_MODE=full C1_REMOVAL=1 C1_ORACLE_A=0 C1_RIPPLE=0 \
  C1_RUN_ROOT="$REPO/runs/removal_dose/rundirs_cnn" \
  "$PY" -u experiments/track_c1.py
echo "[c1rmax $IDX] EXIT=$? $(date '+%F %T')"
