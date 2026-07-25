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
#   ✅ 선행 코드 변경 **C-b 착지 완료**(`d09e528`, 07-25).  env 계약:
#       C1_PARTITION = iid | dir1              (파티션 축을 시나리오에서 분리)
#       C1_THREAT    = clean | label_flip | free_rider | grad_noise
#       C1_FLIP_RATE = 0.70                    (label_flip 도즈; 기존 사다리 대체)
#     `git pull` 후 바로 제출 가능.  구 러너로 돌던 시절의 위험(아래)은 기록으로만 남긴다.
#
#   🚨 **구 러너로 제출하면 "실패"가 아니라 "조용히 틀린 결과"가 나왔다**
#     (2026-07-25 JW 지적 · 코드로 확인).  구 러너는 `os.environ.get` 로 아는 키만 읽어
#     위 3개 env 를 **무시**하고 `C1_SCENARIO` 기본값 `iid` 로 돌지만, `C1_RUN_NAME` 은
#     그대로 먹혀 `cifar10_iid_free-rider_seed0` 이름의 rundir 에 **iid-clean 데이터**가
#     들어가고 EXIT=0 으로 정상 종료했다.  4위협이 한 셀로 붕괴하고 로그로는 구별이 안 된다.
#     → 구 클론에서 도는 잡이 있는지 확인할 때의 술어(워처와 동일):
#        `git show origin/main:codes/experiments/track_c1.py | grep -q C1_THREAT`
#
#   ★ **오염-집합 규약 = 전 위협 고정 round(rho*N) = 4/10** (확정 · Yonghee 판정 07-25).
#     구현은 `track_c1.py:187-196`.  **N=100 주무대(track_c2)와 의도적으로 다르다** —
#     거기선 label_flip 만 FedCorr 공식 구현의 Bernoulli(rho=0.4) 를 써서 개수가 시드마다
#     변동한다(실현 39/48/47; track_c2.py:248-259 · 논문 paper-ko.md:863-869).
#     이 무대에서 고정 개수를 쓰는 근거(l.188-192):
#       ① 도즈를 0.70 으로 고정해 **FedCorr 를 재현하는 게 아니다**(tau~U(0.5,1) 미사용)
#          -> "공식 구현 준수"의 대상이 아니다.
#       ② N=10 에서 Bernoulli 는 **3/7/6** 으로 흔들려(평균 5.33 vs 명목 4, sd 1.55)
#          label_flip 열이 다른 도즈가 되고 **4위협이 한 fidelity 표를 공유할 수 없다**.
#          (N=100 은 39-48 = 상대변동 ~12% 라 무해하지만 N=10 은 ~39% 다.)
#     스트림/오프셋은 track_c2 와 동일(`default_rng(1000+seed)` 의 첫 소비) → **seed-only** =
#     같은 seed 가 dataset/partition/dose 를 넘어 같은 집합을 준다.
#
#     **기대 corrupt 집합 (전 위협 공통 · 스모크 assert 기준)**
#       seed0 [1,4,6,7] · seed1 [0,4,6,7] · seed2 [3,4,6,9]
#
#     ⚠ **논문에 stage-분리 1문장이 필요하다** — 부록 B 의 오염-집합 문단이 지금은 Bernoulli
#       규약만 서술하므로, 이 무대에 대해서는 거짓이 된다.  코드가 아니라 서술이 미결 항목.
#
#   ★ 정본 = `origin/main`(`d09e528`, YH 버전).  다른 계정은 자기 구현을 쓰지 말고 reset 한다
#     (JW 의 중복 구현 989f5ca 는 철회됨 = f056b16).  JW 가 자기 버전으로 돌린 8셀은
#     같은 고정-개수 규칙이라 **전부 유효**하다(재실행 0) — 위 집합만 대조하면 된다.
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
