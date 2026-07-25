#!/bin/bash
# B200 배치 런처 — **컨테이너 1개당 GPU 1장**, **세션 2개 × GPU 2장** 구성 (2026-07-25).
#   사용:  CID=1 bash <repo>/runs/track_h/run_b200_batch.sh     # 컨테이너마다 CID만 바꿔 제출
#   KARIS 배치 폼 "배치 실행 명령어" 에 위 한 줄을 그대로 넣는다.
#
#   세션 A = CID 1, 2   →  G1(주무대 (b) 오라클 9셀) + G4c seed1
#   세션 B = CID 3, 4   →  L1 R=100 재실행 18셀 + G4c seed0
#   두 세션은 서로 의존하지 않는다 — 한쪽이 늦게 잡혀도 다른 쪽 산출물은 그대로 쓸 수 있다.
#   (세션 A만 살면 §5.2/§5.4/§5.5 가, 세션 B만 살면 §5.3 이 각각 자립한다.)
#
# 큐 = queue_b200_c<CID>.txt  (__REPO__ 토큰을 아래 REPO 로 sed 치환)
# 드라이버 = runs/rerun_beta03/run_multi_driver.sh — 큐 소진 시 자동 종료 = 배치 친화·재개형.
# 컷/타임아웃 나도 재개형: 완료 셀은 arm/cell 단위 영속 → 큐에서 '#' **주석**(줄 삭제 금지) 후 재제출.
set -u

CID=${CID:?컨테이너 번호를 지정하세요: CID=1..4}

# ── ① 컨테이너 경로 (★ 실제 값 확인) ────────────────────────────────────────
export REPO=${REPO:-/NHNHOME/26msit001_A/BASE/edge_ai_lab/yonghee/flirds}
export BATCH=${BATCH:-/NHNHOME/26msit001_A/BASE/edge_ai_lab/yonghee/flirds_batch}
export HOME=$BATCH/home
export HF_HOME=$BATCH/hf_home
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PY=$BATCH/venv/bin/python     # ★ venv(torch 2.12+cu130) = canonical 스택 (이미지 기본 torch 아님)
export PP=$REPO/codes                # PYTHONPATH (드라이버가 셀마다 설정)

# ── ② 컨테이너당 GPU 1장 ────────────────────────────────────────────────────
export GPUS="0"

# ── ③ 로그·런타임 큐 ────────────────────────────────────────────────────────
export LOGDIR=$BATCH/runlogs/logs_c$CID
mkdir -p "$LOGDIR"
SRC=$REPO/runs/track_h/queue_b200_c$CID.txt
[ -f "$SRC" ] || { echo "[FATAL] 큐 없음: $SRC"; exit 1; }
sed "s|__REPO__|$REPO|g" "$SRC" > "$LOGDIR/queue.run.txt"
export QUEUE=$LOGDIR/queue.run.txt

# ── ④ 스모크: venv torch·GPU (틀리면 즉시 중단) ─────────────────────────────
"$PY" -c "import torch,sys; print('torch',torch.__version__,'cuda',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 2)" \
  || { echo "[FATAL] venv torch/cuda 스모크 실패 — PY·이미지·GPU 확인"; exit 1; }
echo "[b200-c$CID] REPO=$REPO GPUS='$GPUS' QUEUE=$QUEUE"
grep -c '^[^#]' "$QUEUE" | xargs echo "[b200-c$CID] active cells ="

# ── ⑤ 드라이버 (큐 소진 시 자동 종료) ───────────────────────────────────────
cd "$REPO"
bash "$REPO/runs/rerun_beta03/run_multi_driver.sh"
echo "[b200-c$CID] DONE $(date '+%F %T')"
