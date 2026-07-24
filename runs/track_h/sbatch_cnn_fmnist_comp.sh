#!/bin/bash
# Track H CNN -- fmnist downstream competition (§5.3), the fmnist mirror of the
# cifar10/dir1 W-A stage onto BOTH fmnist partitions {iid, dir1} (2026-07-25,
# Yonghee: "fmnist 비어있는 부분 전부").
#
# WHAT THIS FILLS
#   §5.3 currently has the 8-scoring-source competition on cifar10/dir1 ONLY.
#   fmnist has just the flirds observer (`_obsf`, W-B) + the Track G flirds gate
#   grid (`_g`) -- NO non-flirds competition.  This leg adds the 7 non-flirds
#   sources + obs so make_analysis can build the fmnist competition table next to
#   cifar10/dir1.  P5 (cgate/pweight) was DROPPED 2026-07-23, so 8 cells/threat.
#
# DENOMINATOR ALREADY ON DISK (all 3 seeds, verified 2026-07-25):
#   flirds P1 arms + vanilla/oracle_excl/random_excl (recovery denominator) come
#   from the Track G fmnist grid:
#     runs/track_g/rundirs_cnn/fmnist_{iid,dir1}_<threat>_g_seed{0,1,2}
#   so this leg only runs the NON-flirds sources + obs (self-scoring, B200-indep).
#
# STACK (important): keep this on the SAME stack as the fmnist `_g` grid and the
#   fmnist seed0/1 c2fid cells = 3090 / conda lora4cl (torch 2.11).  Running the
#   sources on a different torch would split the recovery numerator (source arms)
#   from its denominator (`_g` vanilla/oracle_excl) WITHIN a cell.  A6000/torch2.12
#   is possible but then normalize (recovery-norm) and note the cross-stack join.
#
# 288 cells = {fmnist x [iid, dir1]} x 6 threats x (7 sources + obs) x 3 seeds
#   6 threats = clean, free_rider, frrand, grad_noise, label_flip@0.70, lf strmain
#   (fr0.15/0.35 are FIDELITY-only doses; the competition threat set = W-A's exactly.)
#   clean cells cost only the online scoring (obs T2 kept=all -> equals_vanilla skip).
#
# Gate/FL HP = grid defaults verbatim (N=100, C=0.1, R=120) -- per-cell tuning
# forbidden (R1 parity).  Pre-registration (commit BEFORE submit): README.md H-16.
#
# Task index is SEED-MAJOR (96 cells/seed) so a pilot is a prefix:
#   mkdir -p runs/track_h/_logs                                 (--output dir; gitignored)
#   sbatch --array=0-95%8    runs/track_h/sbatch_cnn_fmnist_comp.sh   # seed0 pilot (96)
#   sbatch --array=96-287%8  runs/track_h/sbatch_cnn_fmnist_comp.sh   # seeds 1-2 after GO
#   sbatch                   runs/track_h/sbatch_cnn_fmnist_comp.sh   # all 288 at once
# After:  python runs/track_h/make_analysis.py    (adds fmnist rows to cnn_competition.csv)
#
#SBATCH --job-name=hfmcomp
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --array=0-287%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_h/_logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}

SRCS_P1=(flirds1st lossheur gtg fedsv comfedsv shapleyfl fedif)
PARTS=(iid dir1)                                       # 2 fmnist partitions
THREATS=(clean free_rider frrand grad_noise label_flip label_flip)
TTAGS=(clean free-rider frrand grad-noise label-flip_fr0.70 label-flip_strmain)
DOSES=("" "" "" "" 0.70 "")                            # strmain = FedCorr U(0.5,1)

IDX=${SLURM_ARRAY_TASK_ID}
SEED=$((IDX / 96)); J=$((IDX % 96))                    # seed-major: 96 cells/seed
PART_I=$((J / 48)); K=$((J % 48))                      # 48 cells per partition
T=$((K / 8)); C=$((K % 8))                             # 8 cells per threat (7 src + obs)

PART=${PARTS[$PART_I]}
THREAT=${THREATS[$T]}; TTAG=${TTAGS[$T]}; DOSE=${DOSES[$T]}
EXTRA=(); [ -n "$DOSE" ] && EXTRA+=(C2_FLIP_RATE="$DOSE")

if [ "$C" -lt 7 ]; then
  SRC=${SRCS_P1[$C]}
  ARMS="${SRC}_gate_v2,${SRC}_gatew_v2,${SRC}_mult,${SRC}_zgate_v2"
  TAG=$SRC
else
  ARMS=observer; TAG=obs;   EXTRA+=(C2_T2=1)
fi

NAME="fmnist_${PART}_${TTAG}_${TAG}_seed${SEED}"
echo "[hfmcomp $IDX] $NAME  $(date '+%F %T')"

cd "$REPO/codes"
env "${EXTRA[@]}" PYTHONPATH=. \
  C2_DATASET=fmnist C2_PARTITION="$PART" C2_THREAT="$THREAT" C2_SEED="$SEED" \
  C2_MODE=full C2_ARMS="$ARMS" \
  C2_RUN_ROOT="$REPO/runs/track_h/rundirs_cnn" C2_RUN_NAME="$NAME" \
  "$PY" -u experiments/track_c2.py
echo "[hfmcomp $IDX] EXIT=$? $(date '+%F %T')"
