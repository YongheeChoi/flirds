#!/bin/bash
# B200 배치 런처 (KARIS 배치 폼: 배치 실행 명령어 = `bash <경로>/run_b200_batch.sh`).
# 배치 셸은 프로필(~/.bashrc)을 안 읽으니 env 를 여기서 전부 명시한다.
#   · 큐 = queue_b200.txt (__REPO__ 토큰을 아래 REPO 로 sed 치환)
#   · 드라이버 = runs/rerun_beta03/run_multi_driver.sh (큐 소진 시 자동 종료 = 배치 친화·재개형)
# 컷/타임아웃 나도 재개형: 완료 셀은 arm/cell 단위 영속 → 큐에서 '#' 주석 후 재제출하면 이어감.
set -u

# ── ① B200 컨테이너 경로 (REMAINING-b200 §0; ★ 실제 값 확인) ──────────────────
export REPO=/NHNHOME/26msit001_A/BASE/edge_ai_lab/yonghee/flirds          # ← 리포 루트 확인
export BATCH=/NHNHOME/26msit001_A/BASE/edge_ai_lab/yonghee/flirds_batch   # venv·home·hf_home
export HOME=$BATCH/home
export HF_HOME=$BATCH/hf_home
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PY=$BATCH/venv/bin/python     # ★ 반드시 venv(torch 2.12+cu130) — 이미지 기본 torch 2.10 아님(canonical 스택·§5.5)
export PP=$REPO/codes                # PYTHONPATH (드라이버가 셀마다 설정)

# ── ② 배정받은 B200 개수에 맞춤 (1장이면 "0", 4장이면 "0 1 2 3") ─────────────
export GPUS="0 1 2 3"

# ── ③ 로그·런타임 큐 ──────────────────────────────────────────────────────────
export LOGDIR=$BATCH/runlogs/logs_batch
mkdir -p "$LOGDIR"
sed "s|__REPO__|$REPO|g" "$REPO/runs/track_h/queue_b200.txt" > "$LOGDIR/queue_b200.run.txt"
export QUEUE=$LOGDIR/queue_b200.run.txt

# ── ④ 스모크: venv torch·GPU 확인 (틀리면 즉시 중단 — 180h CPU 낭비 방지) ───────
"$PY" -c "import torch,sys; print('torch',torch.__version__,'cuda',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 2)" \
  || { echo "[FATAL] venv torch/cuda 스모크 실패 — PY·이미지·GPU 확인"; exit 1; }
echo "[b200-batch] REPO=$REPO GPUS='$GPUS' QUEUE=$QUEUE"
grep -c '^[^#]' "$QUEUE" | xargs echo "[b200-batch] active cells ="

# ── ⑤ 드라이버 (전 셀 완주 시 자동 종료) ──────────────────────────────────────
cd "$REPO"
bash "$REPO/runs/rerun_beta03/run_multi_driver.sh"
echo "[b200-batch] DONE $(date '+%F %T')"
