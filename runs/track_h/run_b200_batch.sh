#!/bin/bash
# B200 배치 런처 — **단일 서버 · GPU 4장 동시 제어** (2026-07-25 재편)
#   사용:  bash <repo>/runs/track_h/run_b200_batch.sh          ← 한 번만
#   KARIS 배치 폼 "배치 실행 명령어" 에 위 한 줄을 그대로 넣는다.
#
#   종전 구성(세션 2개 × GPU 2장, 컨테이너마다 CID=1..4 로 4번 제출)이 4장 동시 제어로
#   통합됐다 → 큐도 queue_b200_c{1..4}.txt 4개에서 **queue_b200.txt 1개**로 병합.
#   드라이버가 비는 GPU 에 큐 순서대로 배정하므로 사람이 레인을 쪼갤 이유가 없다
#   (종전 분할은 c4 가 72.8/74h 로 아슬아슬하고 c1–c3 는 15h씩 유휴 = 총 유휴 6h 로 감소).
#
#   26셀 · 249.2 GPU-h → **4-GPU 63.8 wall-h**. 블록별 완주: L1 clean 18.6h ·
#   **G1 전량 58.8h** · flirds1st 52.3h · G5 62.3h · G12 63.8h.
#
# 드라이버 = runs/rerun_beta03/run_multi_driver.sh — 큐 소진 시 자동 종료 = 배치 친화·재개형.
# 컷/타임아웃 나도 재개형: 완료 셀은 arm/cell 단위 영속 → 큐에서 '#' **주석**(줄 삭제 금지) 후 재제출.
#   단 G1(phase2_matrix)은 cell-end 1회 persist → 중도 컷 = 그 셀 19.6h 전손.
set -u

# ── ① 컨테이너 경로 (★ 실제 값 확인) ────────────────────────────────────────
export REPO=${REPO:-/NHNHOME/26msit001_A/BASE/edge_ai_lab/yonghee/flirds}
# ★ BATCH 은 **export 하지 않는다** — 세 러너 모두 `BATCH` 를 batch-size 노브로 읽는다
#   (phase2_matrix.py:185 · track_g.py:136 · track_d.py:101 → `int(os.environ["BATCH"])`).
#   export 하면 경로 문자열이 들어가 전 셀이 15초 만에 ValueError 로 죽고, 드라이버가
#   큐를 그대로 소진해 버린다(07-25 실사례). 아래 파생 경로는 런처 안에서만 쓴다.
BATCH=${BATCH:-/NHNHOME/26msit001_A/BASE/edge_ai_lab/yonghee/flirds_batch}
export -n BATCH 2>/dev/null || true   # 호출 환경이 export 해 왔더라도 자식에게서 제거
export HOME=$BATCH/home
export HF_HOME=$BATCH/hf_home
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PY=$BATCH/venv/bin/python     # ★ venv(torch 2.12+cu130) = canonical 스택 (이미지 기본 torch 아님)
export PP=$REPO/codes                # PYTHONPATH (드라이버가 셀마다 설정)

# ── ② GPU 4장 (GPUS="0 1" 등으로 덮어쓸 수 있다 · 공백 구분) ────────────────
export GPUS=${GPUS:-"0 1 2 3"}

# ── ③ 로그·런타임 큐 ────────────────────────────────────────────────────────
export LOGDIR=$BATCH/runlogs/logs_b200
mkdir -p "$LOGDIR"
SRC=$REPO/runs/track_h/queue_b200.txt
[ -f "$SRC" ] || { echo "[FATAL] 큐 없음: $SRC"; exit 1; }
sed "s|__REPO__|$REPO|g" "$SRC" > "$LOGDIR/queue.run.txt"
export QUEUE=$LOGDIR/queue.run.txt

# ── ④ 스모크: venv torch·GPU 대수 (틀리면 즉시 중단) ────────────────────────
"$PY" -c "import torch,sys; n=torch.cuda.device_count(); print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'gpus',n); sys.exit(0 if torch.cuda.is_available() else 2)" \
  || { echo "[FATAL] venv torch/cuda 스모크 실패 — PY·이미지·GPU 확인"; exit 1; }

# ── ⑤ track_d 용 HF 캐시 사전 확인 (⑤ G5 · ⑥ G12 = track_d) ────────────────
#   HJ 계정에서 alpaca-gpt4·cais/mmlu 가 공유 캐시에도 없어 track_d 가 오프라인 기동
#   불가였다(07-25). 여기서 먼저 경고만 낸다 — G1·L1 은 gsm8k 계열만 쓰므로 무관하고,
#   큐 뒤쪽(⑤⑥)에 가서야 문제가 되니 중단시키지 않는다.
"$PY" - <<'PYEOF' || true
import os, pathlib
hub = pathlib.Path(os.environ["HF_HOME"]) / "hub"
need = {"vicgalle/alpaca-gpt4": "datasets--vicgalle--alpaca-gpt4",
        "cais/mmlu": "datasets--cais--mmlu"}
missing = [k for k, d in need.items() if not (hub / d).exists()]
print("[hf-cache] track_d 데이터셋 " + ("전부 있음" if not missing
      else "누락: " + ", ".join(missing) + "  → ⑤G5·⑥G12 전에 flirds/hf_pin.py 로 받을 것"))
PYEOF

echo "[b200] REPO=$REPO GPUS='$GPUS' QUEUE=$QUEUE"
grep -c '^[^#]' "$QUEUE" | xargs echo "[b200] active cells ="

# ── ⑥ 드라이버 (큐 소진 시 자동 종료) ───────────────────────────────────────
cd "$REPO"
bash "$REPO/runs/rerun_beta03/run_multi_driver.sh"
echo "[b200] DONE $(date '+%F %T')"
