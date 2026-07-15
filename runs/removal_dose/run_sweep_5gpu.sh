#!/bin/bash
# removal_dose FULL SWEEP — 5-GPU 병렬 (run_full_sweep.sh 의 79셀을 슬롯 스케줄러 run_driver.sh 로).
# run_full_sweep.sh 는 단일-GPU 순차라, 같은 셀 정의를 큐로 만들어 5장 밸런싱 + done-마커 재개.
# 결과 = canonical runs/removal_dose/{rundirs(phase2), rundirs_trackd(track_d)}.
# 재실행 = 이어달리기(_sweep/state/done 스킵).  중단: pkill -f run_sweep_5gpu; pkill -f run_driver.sh
set -u
STAGE=/NHNHOME/WORKSPACE/26msit001_A/flirds_batch
CAN=/NHNHOME/26msit001_A/BASE/edge_ai/yonghee/flirds
ROOT=$CAN/runs/removal_dose
RR2=$ROOT/rundirs; RRT=$ROOT/rundirs_trackd
SW=$ROOT/_sweep; Q=$SW/queue.txt
export HOME=$STAGE/home HF_HOME=$STAGE/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
SEEDS="${SEEDS:-0 1 2}"; GPUS="${GPUS:-0 1 2 3 4}"
mkdir -p "$SW/cells" "$SW/state" "$RR2" "$RRT"
# ---- 큐 생성 (LPT: (a)oracle·adamw·removal 먼저, dose 나중) ----
{
  echo "track_d.py|1B_anchor5_adamw_seed0|REGIME=anchor5 SEED=0 ARMS=0 ORACLE_A=1 CLIENT_OPT=adamw RUNDIR_ROOT=$RRT"
  for sd in $SEEDS; do echo "track_d.py|1B_anchor5_removal_seed$sd|REGIME=anchor5 SEED=$sd ARMS=0 ORACLE_A=1 RUNDIR_ROOT=$RRT"; done
  for sd in $SEEDS; do
    echo "phase2_matrix.py|1B_silo5_noisy_removal_seed$sd|REGIME=silo5 THREAT=noisy SEED=$sd REMOVAL=1 RUNDIR_ROOT=$RR2"
    echo "phase2_matrix.py|1B_silo5_frrand_removal_seed$sd|REGIME=silo5 THREAT=freerider_random SEED=$sd REMOVAL=1 RUNDIR_ROOT=$RR2"
    echo "phase2_matrix.py|1B_silo5_frzero_removal_seed$sd|REGIME=silo5 THREAT=freerider_zero SEED=$sd REMOVAL=1 RUNDIR_ROOT=$RR2"
    echo "phase2_matrix.py|1B_silo5_poison_removal_seed$sd|REGIME=silo5 THREAT=poison SEED=$sd REMOVAL=1 LR=2e-3 BATCH=8 EPOCHS=5 POISON_FRAC=0.8 RUNDIR_ROOT=$RR2"
  done
  for sd in $SEEDS; do
    for NR in 0 0.1 0.25 0.5 0.75 1.0; do echo "phase2_matrix.py|1B_silo5_noisy_dose_nr${NR}_seed$sd|REGIME=silo5 THREAT=noisy SEED=$sd NOISY_RATE=$NR RUNDIR_ROOT=$RR2"; done
    for DM in 0.25 0.5 1.0 2.0 4.0; do echo "phase2_matrix.py|1B_silo5_frrand_dose_dm${DM}_seed$sd|REGIME=silo5 THREAT=freerider_random SEED=$sd DOSE_MULT=$DM RUNDIR_ROOT=$RR2"; done
    for PF in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do echo "phase2_matrix.py|1B_silo5_poison_dose_pf${PF}_seed$sd|REGIME=silo5 THREAT=poison SEED=$sd LR=2e-3 BATCH=8 EPOCHS=5 POISON_FRAC=$PF RUNDIR_ROOT=$RR2"; done
  done
} > "$Q"
echo "=== SWEEP QUEUE: $(grep -vcE '^(#|$)' "$Q") cells | GPUS=[$GPUS] | seeds=[$SEEDS] ==="
PY=$STAGE/venv/bin/python PP=$CAN/codes QUEUE=$Q \
  LOGDIR=$SW/cells STATEDIR=$SW/state GPUS="$GPUS" \
  bash "$STAGE/scripts/run_driver.sh"
echo "=== SWEEP DONE $(date '+%F %T') ==="
