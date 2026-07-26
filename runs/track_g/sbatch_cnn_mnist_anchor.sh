#!/bin/bash
# Track G CNN -- **mnist 기준 arm(앵커) 그리드** (계획서 §4.4 G14).
#
# WHAT THIS FILLS
#   부록 downstream "2-CNN P1 — mnist {dir1, iid}" 의 **recovery 분모**.
#   G10(216런)이 mnist 점수원·관측자를 다 채웠지만 `oracle_excl`(천장)·`random_excl`(통제)
#   두 arm 은 track_h 가 낳지 않는다 -- cifar10/fmnist 에선 이 그리드
#   (`runs/track_g/rundirs_cnn/<ds>_<part>_<threat>_g_seed<N>`)가 낳는데 거기 mnist 가 없다.
#   G10 계획서는 그 부재를 알고 **flirds 소스만** track_h 로 옮겨 보정했고
#   (REMAINING-slurm-{HJ:245,YH:162} "mnist 는 track_g 그리드가 없어 flirds 소스도 여기서 생성"),
#   같은 그리드가 낳는 두 기준 arm 은 옮기지 않았다. 이 스크립트가 그 구멍을 메운다.
#
# 18 cells = mnist x {iid, dir1} x **오염 3위협** x 3 seeds
#   ⚠ clean 은 오염 클라가 0이라 oracle_excl/random_excl 이 **정의되지 않는다** -- 그래서 4가 아니라 3.
#     cifar10 도 같다: `_g_seed` 96셀 중 84셀(=4파티션 x 7오염 x 3seed)만 oracle_excl 을 갖는다.
#     (run_cnn_grid.sh 의 ARMS_CLEAN 에 두 arm 이 빠져 있는 것과 같은 이유.)
#   ARMS = vanilla,oracle_excl,random_excl  -- **valuation(φ) 없음**. 세 arm 전부 순수 학습 런이라
#     G10 소스 셀보다 싸다(참조: cifar10/iid G3 최저가 셀 flirds1st/fedif = 18.0 분, 거긴 valuation 포함).
#
# WHY vanilla 도 같이 도나 (관측자가 이미 있는데)
#   `runs/track_h/make_analysis.py::analyze_cnn` 이 분모를 **`arms["vanilla"]` 이름으로만** 찾는다
#   (`observer` 로 폴백하지 않는다). 그래서 vanilla 가 없으면 mnist 전 행의 delta_acc/recovery 가
#   None 이고, `skipped=equals_vanilla` 칸의 final_acc 도 못 채워진다.
#   착지 후 **vanilla vs observer 대조가 무료 검증**이다 -- fmnist 는 36쌍 전부 bit-identical(diff 0.000000),
#   cifar10 은 grad_noise 에서만 최대 0.024 벌어진다(노이즈 주입이 RNG 를 소비). mnist=LeNet5 라
#   fmnist 쪽 거동이 기대값이고, 크게 벌어지면 두 루트 병합 자체를 재검토해야 한다.
#
# STACK: 3090 / conda lora4cl (**torch 2.11**).
#   ⚠ 스택을 섞지 말 것 -- recovery 의 분모(이 잡)와 분자(G10 소스 arm)가 같은 스택에 있어야 한다
#     (REMAINING-slurm-YH.md §0 스택 고정 조항).
#
# Submit (배분 = 파티션으로 반 나눔; 두 range 는 서로 겹치지 않는다):
#   mkdir -p runs/track_g/_logs
#   # YH:  sbatch --array=0-8%8   runs/track_g/sbatch_cnn_mnist_anchor.sh   # mnist/iid  9셀
#   # JB:  sbatch --array=9-17%8  runs/track_g/sbatch_cnn_mnist_anchor.sh   # mnist/dir1 9셀
#   # JB(경로 다름): sbatch --output="$REPO/runs/track_g/_logs/%x_%A_%a.out" \
#   #                       --export=ALL,REPO=<JB repo>,PY=<JB torch2.11 python> --array=9-17%8 ...
# After: python runs/track_h/make_analysis.py   (분모가 생기면서 mnist recovery 열이 채워진다)
#
#SBATCH --job-name=gmnanch
#SBATCH --partition=base_suma_rtx3090,dell_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-17%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_g/_logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}

PARTS=(iid dir1)                                  # 파티션 = 바깥 축 → 계정별 연속 range
THREATS=(label_flip free_rider grad_noise)        # clean 없음(위 주석)
TTAGS=(label-flip_fr0.70 free-rider grad-noise)
DOSES=(0.70 "" "")

IDX=${SLURM_ARRAY_TASK_ID}
PART_I=$((IDX / 9)); J=$((IDX % 9))               # 0-8 = iid · 9-17 = dir1
SEED=$((J / 3)); T=$((J % 3))                     # 계정 range 안에서는 seed-major

PART=${PARTS[$PART_I]}
THREAT=${THREATS[$T]}; TTAG=${TTAGS[$T]}; DOSE=${DOSES[$T]}
EXTRA=(); [ -n "$DOSE" ] && EXTRA+=(C2_FLIP_RATE="$DOSE")

NAME="mnist_${PART}_${TTAG}_g_seed${SEED}"
echo "[gmnanch $IDX] $NAME  $(date '+%F %T')"

cd "$REPO/codes"
env "${EXTRA[@]}" PYTHONPATH=. \
  C2_DATASET=mnist C2_PARTITION="$PART" C2_THREAT="$THREAT" C2_SEED="$SEED" \
  C2_MODE=full C2_ARMS="vanilla,oracle_excl,random_excl" \
  C2_RUN_ROOT="$REPO/runs/track_g/rundirs_cnn" C2_RUN_NAME="$NAME" \
  "$PY" -u experiments/track_c2.py
echo "[gmnanch $IDX] EXIT=$? $(date '+%F %T')"
