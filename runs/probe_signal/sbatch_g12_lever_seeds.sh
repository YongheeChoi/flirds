#!/bin/bash
# G12 -- A축 lever probe 의 seed 보강 (REMAINING-slurm-HJ.md §3).  부록·최저 우선.
#
# 진단문서 §3.1-3.2 의 lever 축(lr x steps, LoRA rank, val-noise)은 seed0 만 있어
# "lr 로 커진 phi 가 cross-seed 실재 신호인가"(예측 rho ~ 0)를 아직 못 물어본다.
# 이 배열이 seed1,2 를 채워 그 질문을 답 가능하게 만든다.
#
# B200 c4 가 lr2e-3_st20 s1,s2 + lr3e-3_st20 s1 (3셀)을 가져갔고, lr1e-3_st10 seed0 는
# 기존 track_d/rundirs/1B_anchor5_seed0 가 그 셀이다(= anchor5 기본값 lr1e-3/st10/r16;
# make_figures.py:120 이 이미 그것을 baseline 으로 읽는다) -> 실행 불요.  남는 것이 이 15.
#
# ⚠ R4 무대가 아니다 -> ROUNDS 를 주지 않는다(anchor5 = N=5, K=5, R=30 고정).
# ⚠ VAL_CHUNK=2 = 메모리 knob 일 뿐. 청크별 grad 의 가중합이 전체 val-grad 와 정확히
#    동일(근사 아님) -> phi 는 seed0 셀(chunk 10, B200)과 비트 단위로 같은 게임이다.
#    seed0 참조셀의 B200 peak 가 98.9 GiB 였으므로 48GB 에서는 이 축소가 필수다.
# ⚠ 각 셀의 나머지 config 는 대응하는 seed0 셀과 정확히 일치시켰다:
#      lr x steps 셀 -> MMLU_LIMIT=40   (cf. 1B_anchor5_lr3e-3_st20_seed0)
#      r32/r64 셀    -> MMLU 전체(=0)   (cf. 1B_anchor5_r32_seed0)
#    cross-seed 비교는 같은 셀의 seed 간 비교이므로 이 일치가 축의 유효성 자체다.
#
# 순서 = 우선순위.  핵심 질문에 필요한 lr{3,2}e-3 가 앞, 버려도 되는 꼬리가 뒤:
#   0-4  lr3e-3/lr2e-3   5-8  lr1e-3   9-12  rank r32/r64   13-14  val-noise r64
# 마감에 걸리면 뒤에서부터 scancel 한다.
#
#SBATCH --job-name=g12lev
#SBATCH --partition=suma_a6000,gigabyte_a6000,asus_6000ada
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-14%8
#SBATCH --output=runs/probe_signal/_logs/%x_%A_%a.out
# --output 은 제출 디렉토리 기준 -> 리포 루트에서 제출.  mkdir -p runs/probe_signal/_logs

set -u
REPO=${REPO:-/home/rlaguswls186790/flirds}
PY=${PY:-/home/rlaguswls186790/miniconda3/envs/flirds/bin/python}
export HF_HOME=${HF_HOME:-/scratch/rlaguswls186790/hf_home}
# alpaca-gpt4 + cais/mmlu 는 공유 캐시에 없어 07-25 이 캐시로 받아 뒀다(오프라인 확인 완료).

# kind|lr|steps|rank|seed   (kind: d=track_d, n=probe_val_noise)
CELLS=(
  "d|3e-3|20|16|2"          # 0
  "d|3e-3|30|16|1"          # 1
  "d|3e-3|30|16|2"          # 2
  "d|2e-3|30|16|1"          # 3
  "d|2e-3|30|16|2"          # 4
  "d|1e-3|20|16|1"          # 5
  "d|1e-3|20|16|2"          # 6
  "d|1e-3|30|16|1"          # 7
  "d|1e-3|30|16|2"          # 8
  "d|1e-3|10|32|1"          # 9   r32  (lr/steps = anchor5 기본)
  "d|1e-3|10|32|2"          # 10
  "d|1e-3|10|64|1"          # 11  r64
  "d|1e-3|10|64|2"          # 12
  "n|1e-3|10|64|1"          # 13  val-noise r64
  "n|1e-3|10|64|2"          # 14
)

IFS='|' read -r KIND LR ST R SEED <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"

cd "$REPO/codes"
COMMON=(PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        REGIME=anchor5 SEED="$SEED" LORA_R="$R" VAL_CHUNK=2)

if [ "$KIND" = "n" ]; then
  # 착지 = runs/probe_signal/noise_probe/noise_1B_r{R}_seed{SEED} (러너 기본값)
  echo "[g12 $SLURM_ARRAY_TASK_ID] noise-probe r$R seed$SEED  $(date '+%F %T')"
  env "${COMMON[@]}" "$PY" -u experiments/probe_val_noise.py
else
  if [ "$R" = "16" ]; then
    NAME="1B_anchor5_lr${LR}_st${ST}_seed${SEED}"; MMLU=40   # lr x steps 격자
  else
    NAME="1B_anchor5_r${R}_seed${SEED}";          MMLU=0     # rank 축(MMLU 전체)
  fi
  echo "[g12 $SLURM_ARRAY_TASK_ID] $NAME  lr=$LR steps=$ST r=$R mmlu=$MMLU  $(date '+%F %T')"
  env "${COMMON[@]}" LR="$LR" MAX_STEPS="$ST" ORACLE_A=0 FIDELITY=1 ARMS=1 \
      MMLU_LIMIT="$MMLU" RUN_NAME="$NAME" \
      RUNDIR_ROOT="$REPO/runs/probe_signal/rundirs" \
      "$PY" -u experiments/track_d.py
fi
echo "[g12 $SLURM_ARRAY_TASK_ID] EXIT=$? $(date '+%F %T')"
