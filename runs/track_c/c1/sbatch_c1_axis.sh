#!/bin/bash
# Track C1 -- **(a) 재학습 오라클을 논문 오염축·파티션으로 정렬** (2026-07-25 재편:
# 계획서 §2.1 G2[cifar10, 본문] + §3.1 G9[mnist, 부록]).
#
# WHY
#   현 C1 시나리오({iid, label_skew, quantity_skew, label_flip, feature_noise})는
#   확정 오염축 3종(lf@0.70 · free-rider-zero · grad-noise) 과 **한 칸도 겹치지 않는다**.
#   (a) 2^10 재학습 오라클은 N=100 에서 원리적으로 불가하므로 1A-CNN(N=100 부분참여)과
#   N·참여율은 맞출 수 없다 — **맞출 수 있는 건 오염축과 파티션뿐**이고 이 잡이 그걸 한다.
#   (이 비교불가성은 논문에도 명시한다: 현 track_c2_fid 헤더 CAVEAT 와 같은 취지.)
#
#   ⚠ 선행 코드 변경 **C-b** 필요 — 이 파일이 요구하는 env 계약:
#       C1_PARTITION = iid | dir1              (신규: 파티션 축을 시나리오에서 분리)
#       C1_THREAT    = clean | label_flip | free_rider | grad_noise   (신규)
#       C1_FLIP_RATE = 0.70                    (label_flip 도즈; 기존 사다리 대체)
#     현 track_c1.py:79 는 `C1_SCENARIO` 하나에 파티션+오염을 섞어 두고 있고
#     free_rider·grad_noise 구현이 없다.  코퍼스는 track_c2 쪽에 이미 있다
#     (flirds/data/corruptors.py CNN_CORRUPTORS · fl.partition dir1) → 이식이 주작업.
#
# 48 cells = {cifar10, mnist} x {iid, dir1} x 4 threats x 3 seeds  (SEED-MAJOR 16/seed)
#   셀 내부: (a) 2^10 재학습 오라클 + (b) 2^10 in-run + 9방법 φ  (N=10 full, R=10)
#
# 비용(실측): (a) 2^10 재학습 `t_a` = **cifar10 32,808 s ≈ 9.1 h** ·
#             **mnist 41,168 s ≈ 11.4 h** (runs/track_c/c1_oracle/*/metrics.json).
#             나머지(궤적 ~103 s, 전 방법 합 ~8 분)는 무시 가능 → 셀 ≈ t_a.
#             48셀 ≈ **505 GPU-h**;  8슬롯 → ~63 wall-h.
#
# STACK: 3090 / conda lora4cl (torch 2.11) — 기존 C1 rundir 과 동일.
#
# Submit (C-b 착지 후):
#   mkdir -p runs/track_c/c1/_logs
#   sbatch --array=0-7%8     runs/track_c/c1/sbatch_c1_axis.sh   # cifar10 seed0 (본문 G2 먼저)
#   sbatch --array=8-15%8    runs/track_c/c1/sbatch_c1_axis.sh   # mnist   seed0
#   sbatch --array=16-47%8   runs/track_c/c1/sbatch_c1_axis.sh   # seeds 1-2
# After: python runs/track_c/c1/make_analysis.py  (또는 기존 C1 집계 경로)
#
#SBATCH --job-name=c1axis
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-47%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_c/c1/_logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}

DSETS=(cifar10 mnist)                            # cifar10 = 본문(G2) 먼저
PARTS=(iid dir1)
THREATS=(clean label_flip free_rider grad_noise)
TTAGS=(clean label-flip_fr0.70 free-rider grad-noise)
DOSES=("" 0.70 "" "")

IDX=${SLURM_ARRAY_TASK_ID}
SEED=$((IDX / 16)); J=$((IDX % 16))               # seed-major: 16 cells/seed
DS_I=$((J / 8)); K=$((J % 8))
PART_I=$((K / 4)); T=$((K % 4))

DS=${DSETS[$DS_I]}; PART=${PARTS[$PART_I]}
THREAT=${THREATS[$T]}; TTAG=${TTAGS[$T]}; DOSE=${DOSES[$T]}
EXTRA=(); [ -n "$DOSE" ] && EXTRA+=(C1_FLIP_RATE="$DOSE")

NAME="${DS}_${PART}_${TTAG}_seed${SEED}"
echo "[c1axis $IDX] $NAME  $(date '+%F %T')"

cd "$REPO/codes"
env "${EXTRA[@]}" PYTHONPATH=. \
  C1_DATASET="$DS" C1_PARTITION="$PART" C1_THREAT="$THREAT" C1_SEED="$SEED" \
  C1_MODE=full C1_ORACLE_A=1 \
  C1_RUN_NAME="$NAME" \
  "$PY" -u experiments/track_c1.py
echo "[c1axis $IDX] EXIT=$? $(date '+%F %T')"
