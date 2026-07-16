#!/bin/bash
# Exp A3 — CNN removal-curve 스윕 (run_sweep_5gpu.sh 의 큐 형식 script|name|envs · 슬롯 스케줄러
# run_driver.sh · done-마커 재개를 그대로 재사용).  결과 = rundirs_cnn/ (canonical runs/track_c/c1 불변).
# 셀 = {DATASETS} x {SCENARIOS} x {SEEDS}; 기본 mnist 9셀, cifar10 은 DATASETS="mnist cifar10" 옵트인.
# C1_ORACLE_A=0: removal 은 (a) 2^N 캐시 불필요(옵션 1, A2 와 균일).  C1_RIPPLE=0: Ripple 은 removal
# 순위 대상 제외(자기-궤적) + fidelity 는 canonical c1 에 이미 영속 — 중복 재계산 회피.
# DRYRUN=1 = 큐만 생성·출력 후 종료(런치 없음).  GPUS="0 2" 등으로 GPU 오버라이드(기본 0-4;
# 주의: removal_dose/_sweep 풀스윕이 점유 중이면 유휴 GPU 만 지정할 것).
# 재실행 = 이어달리기(_sweep_cnn/state/done 스킵).  중단: pkill -f run_cnn_removal; pkill -f run_driver.sh
set -u
STAGE=/NHNHOME/WORKSPACE/26msit001_A/flirds_batch
CAN=/NHNHOME/26msit001_A/BASE/edge_ai/yonghee/flirds
ROOT=$CAN/runs/removal_dose
RRC=$ROOT/rundirs_cnn
SW=$ROOT/_sweep_cnn; Q=$SW/queue.txt
export HOME=$STAGE/home PYTHONDONTWRITEBYTECODE=1   # torchvision root ~/data -> $STAGE/home/data (MNIST 첫 실행 시 자동 다운로드)
SEEDS="${SEEDS:-0 1 2}"; GPUS="${GPUS:-0 1 2 3 4}"
DATASETS="${DATASETS:-mnist}"; SCENARIOS="${SCENARIOS:-label_flip feature_noise iid}"
mkdir -p "$SW/cells" "$SW/state" "$RRC"
# ---- 큐 생성 ----
{
  for ds in $DATASETS; do for sc in $SCENARIOS; do for sd in $SEEDS; do
    echo "track_c1.py|cnn_removal_${ds}_${sc//_/-}_seed${sd}|C1_DATASET=$ds C1_SCENARIO=$sc C1_SEED=$sd C1_MODE=full C1_REMOVAL=1 C1_ORACLE_A=0 C1_RIPPLE=0 C1_RUN_ROOT=$RRC"
  done; done; done
} > "$Q"
echo "=== CNN REMOVAL QUEUE: $(grep -vcE '^(#|$)' "$Q") cells | GPUS=[$GPUS] | seeds=[$SEEDS] | datasets=[$DATASETS] ==="
if [ "${DRYRUN:-0}" = "1" ]; then cat "$Q"; echo "=== DRYRUN: 큐만 생성 — 아무것도 실행하지 않음 ==="; exit 0; fi
PY=$STAGE/venv/bin/python PP=$CAN/codes QUEUE=$Q \
  LOGDIR=$SW/cells STATEDIR=$SW/state GPUS="$GPUS" \
  bash "$STAGE/scripts/run_driver.sh"
echo "=== CNN REMOVAL SWEEP DONE $(date '+%F %T') ==="
