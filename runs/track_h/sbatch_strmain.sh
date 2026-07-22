#!/bin/bash
# Track H CNN -- label_flip `strmain` extension (2026-07-22, Yonghee: "전부 다").
#
# 5th threat for the score-source competition stage (cifar10 dir1): FedCorr
# per-client rate ~ U(0.5, 1.0) instead of the fixed 0.70 dose.  ADDITIVE -- the
# existing lf@0.70 results stay valid; this is the first stage with BORDERLINE
# corrupt clients (rate ~ 0.5), i.e. the first real test of P5's design intent
# (confidence gates should spare borderline variance, not just confident harm).
#
# 17 cell types x 3 seeds = 51 runs, mirroring the existing per-threat block:
#   J 0-6   P1 cells  (7 sources; NO flirds -- flirds P1 arms live in the Track G
#            grid, whose strmain cells are already queued as job 1860471)
#            arms = <src>_gate_v2,<src>_gatew_v2,<src>_mult,<src>_zgate_v2
#   J 7-14  P5 cells  (8 sources)  arms = <src>_cgate,<src>_pweight
#   J 15    obs    = observer + T2 legacy retrains (C2_T2=1)
#   J 16    obsp5  = observer + T2 P5 retrains (C2_T2=1 C2_T2_P5=1 C2_T2_LEGACY=0)
#
# Submit:  sbatch runs/track_h/sbatch_strmain.sh          (seed-major: 0-16 = seed0)
#
#SBATCH --job-name=hstrmain
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --array=0-50%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_h/p5/logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}

SRCS_P1=(flirds1st lossheur gtg fedsv comfedsv shapleyfl fedif)
SRCS_P5=(flirds flirds1st lossheur gtg fedsv comfedsv shapleyfl fedif)

IDX=${SLURM_ARRAY_TASK_ID}
SEED=$((IDX / 17)); J=$((IDX % 17))             # seed-major: 17 cell types per seed
EXTRA=()
if [ "$J" -lt 7 ]; then
  SRC=${SRCS_P1[$J]}
  ARMS="${SRC}_gate_v2,${SRC}_gatew_v2,${SRC}_mult,${SRC}_zgate_v2"
  TAG=$SRC
elif [ "$J" -lt 15 ]; then
  SRC=${SRCS_P5[$((J - 7))]}
  ARMS="${SRC}_cgate,${SRC}_pweight"
  TAG="${SRC}p5"
elif [ "$J" -eq 15 ]; then
  ARMS=observer; TAG=obs;   EXTRA+=(C2_T2=1)
else
  ARMS=observer; TAG=obsp5; EXTRA+=(C2_T2=1 C2_T2_P5=1 C2_T2_LEGACY=0)
fi

NAME="cifar10_dir1_label-flip_strmain_${TAG}_seed${SEED}"
echo "[hstrmain $IDX] $NAME  $(date '+%F %T')"

cd "$REPO/codes"
env "${EXTRA[@]}" PYTHONPATH=. \
  C2_DATASET=cifar10 C2_PARTITION=dir1 C2_THREAT=label_flip C2_SEED="$SEED" \
  C2_MODE=full C2_ARMS="$ARMS" \
  C2_RUN_ROOT="$REPO/runs/track_h/rundirs_cnn" C2_RUN_NAME="$NAME" \
  "$PY" -u experiments/track_c2.py
echo "[hstrmain $IDX] EXIT=$? $(date '+%F %T')"
