#!/bin/bash
# Track H CNN -- W-B: Flirds P1w twin leg on the EXTENDED (non-dir1) stage
# (paper/workplan/T4-p1w-cnn-relay.md).  P1w == the existing P2 (sign+size weight);
# no new code -- this only runs the T2 retrain leg that the Track G skew campaign
# never ran, so make_analysis can assemble the full flirds P1/P1w x T1/T2 table.
#
# WHY only T2 here:
#   T1 P1  = flirds_gate_v2   (skew campaign, runs/track_g/rundirs_cnn)  -- on disk
#   T1 P1w = flirds_gatew_v2  (skew campaign, runs/track_g/rundirs_cnn)  -- on disk
#   T2 P1  = t2_sign_flirds   <- THIS run (observer -> retrain)
#   T2 P1w = t2_signw_flirds  <- THIS run (observer -> size-weighted retrain)
# The observer is restricted to the FLIRDS source (C2_OBS_SRCS=flirds), so the leg
# stays cheap (no O(2^k)/O(k^2) MC sources) -- the non-flirds sources are W-D, a
# separate later approval.  Landing in track_h/rundirs_cnn makes track_h/make_analysis
# MERGE these T2 arms with the skew campaign's T1 arms on the (dataset, partition,
# threat, flip_rate, seed) cell key; vanilla / oracle_excl / random_excl for the
# recovery denominator come from the same skew twin (corrupt cells carry them).
# Verified on dir1: this exact merge reproduces the canonical overview 3.2.3 flirds
# rows (P1-T1 .5843 / P1w-T1 .5913 / P1-T2 .6107 / P1w-T2 .5959).
#
# STAGE (5 (dataset,partition) x 6 threats x 3 seeds = 90 cells) -- every cell has a
# downstream twin already on disk (skew GROUP A + GROUP B frrand + gstrmain):
#   {cifar10 x [shard, qskew, iid], fmnist x [iid, dir1]}
#     x {clean, free_rider, frrand, grad_noise, label_flip@0.70, label_flip strmain}
#   cifar10 dir1 is NOT here -- that is W-A (Track H R1, reuse runs/track_h/rundirs_cnn).
#   clean cells cost only the observer (T2 kept=all -> equals_vanilla, retrain skipped).
#
# Gate/FL HP = the grid defaults verbatim (N=100, C=0.1, R=120, E=5, lr .01, burn_in
# 10, tau 0, min_obs 2, probation 5, alpha 1) -- per-cell tuning forbidden (R1 parity).
# Pre-registration (commit BEFORE submit): runs/track_h/README.md H-15.
#
# Task index is SEED-MAJOR (30 cells per seed) so a pilot is a prefix:
#   mkdir -p runs/track_h/_logs                            (--output dir; gitignored)
#   sbatch --array=0-29%8 runs/track_h/sbatch_cnn_p1w.sh   (seed-0 pilot, 30 cells)
#   sbatch --array=30-89%8 runs/track_h/sbatch_cnn_p1w.sh  (seeds 1-2 after the GO)
#   sbatch runs/track_h/sbatch_cnn_p1w.sh                  (all 90 at once)
# After:  python runs/track_h/make_analysis.py            (merged competition CSV)
#         python runs/track_h/make_p1w_cnn_table.py       (the W-B P1-vs-P1w table)
#
#SBATCH --job-name=hp1w
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-89%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_h/_logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}

# 5 (dataset,partition) combos, indexed 0..4
DSS=(cifar10 cifar10 cifar10 fmnist fmnist)
PARTS=(shard qskew iid iid dir1)
# 6 threats, indexed 0..5
THREATS=(clean free_rider frrand grad_noise label_flip label_flip)
TTAGS=(clean free-rider frrand grad-noise label-flip_fr0.70 label-flip_strmain)
DOSES=("" "" "" "" 0.70 "")            # strmain = FedCorr U(0.5,1) (no fixed dose)

IDX=${SLURM_ARRAY_TASK_ID}
SEED=$((IDX / 30)); J=$((IDX % 30))    # seed-major: 30 cells per seed
DSP=$((J / 6)); T=$((J % 6))
DS=${DSS[$DSP]}; PART=${PARTS[$DSP]}
THREAT=${THREATS[$T]}; TTAG=${TTAGS[$T]}; DOSE=${DOSES[$T]}
EXTRA=(); [ -n "$DOSE" ] && EXTRA+=(C2_FLIP_RATE="$DOSE")

NAME="${DS}_${PART}_${TTAG}_obsf_seed${SEED}"
echo "[hp1w $IDX] $NAME  $(date '+%F %T')"

cd "$REPO/codes"
env "${EXTRA[@]}" PYTHONPATH=. \
  C2_DATASET="$DS" C2_PARTITION="$PART" C2_THREAT="$THREAT" C2_SEED="$SEED" \
  C2_MODE=full C2_ARMS=observer C2_T2=1 C2_OBS_SRCS=flirds \
  C2_RUN_ROOT="$REPO/runs/track_h/rundirs_cnn" C2_RUN_NAME="$NAME" \
  "$PY" -u experiments/track_c2.py
echo "[hp1w $IDX] EXIT=$? $(date '+%F %T')"
