#!/bin/bash
# B200 레인 런처 — **4노드 × GPU 1장** 구성용 (2026-07-25).
#   사용:  LANE=1 bash <repo>/runs/track_h/run_b200_lane.sh      # 노드 yong-1 에서
#          (4대 한 번에:  bash <repo>/runs/track_h/b200_fleet.sh up)
#
#   run_b200_batch.sh(정본)는 **단일 서버 GPU 0–3** 을 가정한다(`GPUS="0 1 2 3"` + 큐 1개).
#   그러나 이 서버의 실제 구성은 컨테이너 4대(yong-1..4)에 **GPU 1장씩**이다(실측).
#   그대로 띄우면 슬롯 1·2·3 셀이 CUDA_VISIBLE_DEVICES=1,2,3 을 못 찾아 즉사하고
#   드라이버가 큐를 그대로 소진한다 — BATCH export 사고와 같은 형태의 전손이다.
#   → 노드마다 이 런처로 `GPUS="0"` + 자기 레인 큐를 돌린다.
#
#   레인 큐 = queue_b200_lane{1..4}.txt (정본 queue_b200.txt 26셀을 원문 그대로 분할).
#   env·경로·하드룰은 정본 런처와 동일하게 유지한다.
set -u

LANE=${LANE:?레인 번호를 지정하세요: LANE=1..4}

# ── ① 컨테이너 경로 (정본 런처와 동일) ──────────────────────────────────────
export REPO=${REPO:-/NHNHOME/26msit001_A/BASE/edge_ai_lab/yonghee/flirds}
# ★ BATCH 은 **export 하지 않는다** — 세 러너 모두 `BATCH` 를 batch-size 노브로 읽는다
#   (phase2_matrix.py:185 · track_g.py:136 · track_d.py:101 → `int(os.environ["BATCH"])`).
#   export 하면 경로 문자열이 들어가 전 셀이 15초 만에 ValueError 로 죽는다(07-25 실사례).
BATCH=${BATCH:-/NHNHOME/26msit001_A/BASE/edge_ai_lab/yonghee/flirds_batch}
export -n BATCH 2>/dev/null || true   # 호출 환경이 export 해 왔더라도 자식에게서 제거
export HOME=$BATCH/home
export HF_HOME=$BATCH/hf_home
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PY=$BATCH/venv/bin/python     # ★ venv(torch 2.12+cu130) = canonical 스택
export PP=$REPO/codes                # PYTHONPATH (드라이버가 셀마다 설정)

# ── ② 이 노드의 GPU 는 1장 ──────────────────────────────────────────────────
export GPUS=${GPUS:-"0"}

# ── ③ 로그·런타임 큐 (레인별로 분리 — 노드끼리 절대 안 섞이게) ──────────────
export LOGDIR=$BATCH/runlogs/logs_lane$LANE
mkdir -p "$LOGDIR"
SRC=$REPO/runs/track_h/queue_b200_lane$LANE.txt
[ -f "$SRC" ] || { echo "[FATAL] 레인 큐 없음: $SRC"; exit 1; }
sed "s|__REPO__|$REPO|g" "$SRC" > "$LOGDIR/queue.run.txt"
export QUEUE=$LOGDIR/queue.run.txt

# ── ④ 스모크: venv torch·GPU (틀리면 즉시 중단) ─────────────────────────────
"$PY" -c "import torch,sys; n=torch.cuda.device_count(); print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'gpus',n); sys.exit(0 if torch.cuda.is_available() and n>=1 else 2)" \
  || { echo "[FATAL] venv torch/cuda 스모크 실패 — PY·이미지·GPU 확인"; exit 1; }

# ── ⑤ track_d 용 HF 캐시 확인 (레인2·3·4 의 PHASE 2 = G5·G12) ───────────────
"$PY" - <<'PYEOF' || true
import os, pathlib
hub = pathlib.Path(os.environ["HF_HOME"]) / "hub"
need = {"vicgalle/alpaca-gpt4": "datasets--vicgalle--alpaca-gpt4",
        "cais/mmlu": "datasets--cais--mmlu"}
missing = [k for k, d in need.items() if not (hub / d).exists()]
print("[hf-cache] track_d 데이터셋 " + ("전부 있음" if not missing
      else "누락: " + ", ".join(missing) + "  → G5·G12 전에 flirds/hf_pin.py 로 받을 것"))
PYEOF

echo "[b200-lane$LANE] node=$(hostname) REPO=$REPO GPUS='$GPUS' QUEUE=$QUEUE"
grep -c '^[^#]' "$QUEUE" | xargs echo "[b200-lane$LANE] active cells ="

# ── ⑥ 드라이버 (큐 소진 시 자동 종료) ───────────────────────────────────────
cd "$REPO"
bash "$REPO/runs/rerun_beta03/run_multi_driver.sh"
echo "[b200-lane$LANE] DONE $(date '+%F %T')"
