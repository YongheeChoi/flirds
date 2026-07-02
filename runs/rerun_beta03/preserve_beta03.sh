#!/bin/bash
# beta=0.3 재실행 결과 분리 보존 가드 (Yonghee 2026-07-02 지시: 재실행은 기존 셀을
# 덮어쓰지 말고 별도 이름으로 — 원본은 비교용으로 유지).
#
# 대기: 진행 중인 3B_anchor5 재실행(ORACLE_A=0) 3 프로세스 종료까지.
# 처리: 셀이 덮어쓴 rundir을 runs/track_d/rundirs_beta03/<같은 셀명>으로 옮기고
#       원본을 git에서 복원. (3B_std20은 2026-07-02 세션에서 이미 동일 처리.)
# 이후 캠페인의 #PAUSED 셀들은 큐 라인에 RUNDIR_ROOT/C1_RUN_ROOT/C2_RUN_ROOT
# 리다이렉트가 박혀 있어 재개해도 canonical rundir을 건드리지 않는다.
set -u
REPO=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds
PIDS="4125891 4146383 4152059"          # 3B_anchor5_seed0/1/2 (2026-07-02 시점)
LOG=$REPO/runs/rerun_beta03/logs/_preserve.log
cd "$REPO" || exit 1
echo "[preserve] $(date '+%F %T') waiting on pids: $PIDS" >> "$LOG"
while :; do
  alive=0
  for p in $PIDS; do [ -d "/proc/$p" ] && alive=1; done
  [ "$alive" = 0 ] && break
  sleep 300
done
sleep 120                                # 마지막 _persist flush 여유
mkdir -p runs/track_d/rundirs_beta03
for s in 0 1 2; do
  d=runs/track_d/rundirs/3B_anchor5_seed$s
  if [ -n "$(git status --porcelain -- "$d")" ]; then
    mv "$d" "runs/track_d/rundirs_beta03/3B_anchor5_seed$s"
    git checkout -- "$d"
    echo "[preserve] $(date '+%F %T') 3B_anchor5_seed$s: rerun -> rundirs_beta03/, 원본 복원" >> "$LOG"
  else
    echo "[preserve] $(date '+%F %T') 3B_anchor5_seed$s: 변경 없음(재실행이 persist 전에 죽었나 확인)" >> "$LOG"
  fi
done
echo "[preserve] $(date '+%F %T') DONE" >> "$LOG"
