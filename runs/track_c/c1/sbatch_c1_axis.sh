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
#     구 track_c1.py:79 는 `C1_SCENARIO` 하나에 파티션+오염을 섞어 두고 있고
#     free_rider·grad_noise 구현이 없다.  코퍼스는 track_c2 쪽에 이미 있다
#     (flirds/data/corruptors.py CNN_CORRUPTORS · fl.partition dir1) → 이식이 주작업.
#
#   🚨 **C-b 미착지 상태로 제출하면 "실패"가 아니라 "조용히 틀린 결과"가 나온다**
#     (2026-07-25 JW 지적 · 코드로 확인).  구 러너는 `os.environ.get` 로 아는 키만 읽으므로
#     위 3개 env 를 **무시**하고 `C1_SCENARIO` 기본값 `iid` 로 돌지만, l.432 의
#     `C1_RUN_NAME` 은 그대로 먹힌다 → `cifar10_iid_free-rider_seed0` 라는 이름의 rundir 에
#     **iid-clean 데이터**가 들어가고 EXIT=0 으로 정상 종료한다.  4위협이 한 셀로 붕괴하고
#     로그로는 구별이 안 된다.  48셀 = 505 GPU-h 전량 폐기 위험.
#     → **제출 전 필수 게이트**: `git show origin/main:codes/experiments/track_c1.py |
#        grep -q C1_THREAT` (JW·JB 워처가 쓰는 술어와 동일).  통과 못 하면 제출 금지.
#
#   🚨 **오염-집합 draw 는 위협별로 다르다 — 균일 규칙(고정 4/10)을 쓰면 안 된다.**
#     `track_c2.py:248-259`(감사 2026-07-23) + 논문 paper-ko.md:863-869 가 정본:
#       · label_flip  = **독립 Bernoulli(rho=0.4)** -> **개수가 시드마다 변동**.
#                       FedCorr 공식 구현 그대로(gamma_s = binomial(1,rho,n)).
#                       강도만 0.70 고정(FedCorr 의 tau~U(0.5,1) draw 아님).
#       · free_rider  = 고정 `round(rho*n)` 비복원 추출  (update-level = 준거 문헌 없음
#       · grad_noise    -> 논문이 이미 예외로 기술.  C2_STRENGTH 는 진폭을 쓸어간다)
#     → **이식은 track_c2.py:258-259 를 그대로**(새로 유도 금지):
#         mal = rng.random(n) < _strength(MAL_FRAC) if THREAT == "label_flip" else \
#             set(rng.choice(n, size=max(1, int(round(MAL_FRAC*n))), replace=False).tolist())
#       + rate 는 l.280-281 대로 `C1_FLIP_RATE=0.70` 고정.
#     ⚠ mal draw 는 `np.random.default_rng(1000+seed)` 의 **첫 소비**여야 한다 — 파티션이
#       스트림을 먼저 먹으면 집합이 partition-dependent 가 되어 기존 corpus 와 갈린다
#       (l.271-275 가 바로 그 위험을 기록해 둔 지점).  draw 가 seed-only 인 덕분에 같은
#       seed 는 dataset/partition/dose/track 을 넘어 같은 집합을 준다(l.255-257).
#
#     **N=10 실현값 — 미리 계산해 둔 기대치(assert 로 쓸 것)**.  같은 코드로 N=100 을
#     돌리면 논문 기재값 39/48/47 을 정확히 재생산하므로 아래는 신뢰 가능:
#       label_flip  seed0 k=3 [3,5,6] · seed1 k=7 [1,2,4,5,7,8,9] · seed2 k=6 [0,1,3,4,6,7]
#       fr / gn     seed0 [1,4,6,7]   · seed1 [0,4,6,7]            · seed2 [3,4,6,9]
#     전 셀 k>=2 -> AUROC 가 모든 셀에서 정의된다(k=0/1 붕괴 케이스 없음).  평균 5.33 이
#     명목 4 보다 높고 seed1 은 7/10 이지만, 논문 규약이 "명목 rho 대신 **실현 수**를 병기"
#     이므로 그대로 보고한다.
#
#   🚨 **정본은 push 된 origin/main 하나** — C-b 구현이 두 벌 존재했다(YH 워킹트리 ·
#     JW 로컬 989f5ca).  다른 계정은 자기 구현을 쓰지 말고 reset 한다.  이미 자기 구현으로
#     돌린 셀이 있으면 위 실현값과 `corrupt=` 를 대조하고, 다르면 그 셀만 재제출.
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
# After: **집계기는 이미 있다 — 새로 쓰지 말 것.**
#   `runs/track_c/make_figures.py` l.99-152 `load_c1()` 이 c1 rundir 을 (a) 오라클과
#   페어링하고 `phi_a` 를 음수화해 Spearman 을 낸다(= G2 표 그 자체).  막힌 건 l.36
#   `SCENARIOS = [iid, label-flip, label-skew, feature-noise, quantity-skew]` 상수가
#   구 축이라는 것뿐 → {PARTS} x {THREATS} 격자와 l.65 이름 패턴
#   `{ds}_{part}_{ttag}_seed{seed}` 로 교체하면 된다.  (구 세션이 헤더에 적어 둔
#   `runs/track_c/c1/make_analysis.py` 는 **존재하지 않는 파일**이었다 — 07-25 정정.)
#   ⚠ 첫 셀 착지 후 확인할 것 1건: `C1_ORACLE_A=1` 의 `phi_a` 가 같은 rundir 의
#     metrics.json 에 들어가는지, 별도 `c1_oracle/*_aonly_*/` 로 가는지
#     (l.174 각주는 후자를 가정한다).  페어링 경로가 여기서 갈린다.
#
#SBATCH --job-name=c1axis
#SBATCH --partition=base_suma_rtx3090,dell_rtx3090
# ^ 3090 풀 전체.  base_suma 단독은 07-25 여유 0 / dell 에 9장 유휴(JW 실측).
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
