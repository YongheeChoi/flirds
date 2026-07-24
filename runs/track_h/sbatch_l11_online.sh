#!/bin/bash
# L11 -- R4 §5.3 online 완성: the 7 NON-flirds sources' T1 sign-gate (P1 online).
# (REMAINING-slurm-HJ.md §2 + REMAINING-slurm-YH.md §5; REMAINING-b200.md §1a/§2.)
#
# §5.3 R4 has an online table of 8 sources.  flirds online = B200 (L1, HVP).  This
# leg fills the other 7 (flirds1st lossheur fedif gtg fedsv comfedsv shapleyfl), each
# self-scoring inline (fresh accumulator -> NO HVP, NO saved-cum load) = B200-indep.
# Arm = <src>_gate_v2 (track_g build_arm: provider = arm.split('_')[0]; endswith
# _gate_v2 -> P1).  The recovery denominator (vanilla / oracle_excl / random_excl)
# for each cell comes from B200's L1 flirds observer+online on the SAME meta cell key
# (regime,threat,nr,seed) -- make_analysis merges by that key across roots.
#
# GPU: A6000 48GB (retrain-scoring class ~32 GiB; 24GB insufficient).  B200-independent.
#
# 63 tasks = 7 src x {clean, noisy(nr0.7), frzero} x 3 seed.  SEED-MAJOR (21/seed).
# Load is split by SEED across two accounts (balance; REMAINING-b200 §1a):
#   HJ:  sbatch --array=0-41%8  runs/track_h/sbatch_l11_online.sh   # seed0,1 (42 run)
#   YH:  sbatch --array=42-62%8 runs/track_h/sbatch_l11_online.sh   # seed2   (21 run)
# The landing root auto-routes by seed (seed2 -> rundirs_llm_yh, else rundirs_llm_hj),
# so each account just submits its own array range.  Override RUNDIR_ROOT for work-steal.
# seed0 first (0-20) = paper 착수선.  After: python runs/track_h/make_analysis.py
#
#SBATCH --job-name=l11on
#SBATCH --partition=suma_a6000,gigabyte_a6000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-62%8
#SBATCH --output=runs/track_h/_logs/%x_%A_%a.out
# base_qos gates A6000 (all partitions AllowAccounts=ALL); edit --qos if your account
# names it differently (wrong token fail-fasts).  --output is relative to the SUBMIT
# dir -> submit from the repo root after:  mkdir -p runs/track_h/_logs

set -u
REPO=${REPO:-$HOME/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}   # shared lora4cl (torch2.11); override PY= if not readable
export HF_HOME=${HF_HOME:-/scratch/chyoyhr/hf_home}         # 1B model + gsm8k cache; override HF_HOME= to your cache

SRCS=(flirds1st lossheur fedif gtg fedsv comfedsv shapleyfl)
THREATS=(clean noisy frzero)

IDX=${SLURM_ARRAY_TASK_ID}
SEED=$((IDX / 21)); J=$((IDX % 21))            # seed-major: 21 cells/seed
SRC_I=$((J / 3)); T=$((J % 3))                 # 7 sources x 3 threats
SRC=${SRCS[$SRC_I]}; THREAT=${THREATS[$T]}

# seed2 = YH's leg -> its own root; seed0,1 = HJ.  RUNDIR_ROOT env overrides (work-steal).
if [ -n "${RUNDIR_ROOT:-}" ]; then RR="$RUNDIR_ROOT";
elif [ "$SEED" -eq 2 ]; then RR="$REPO/runs/track_h/rundirs_llm_yh";
else RR="$REPO/runs/track_h/rundirs_llm_hj"; fi

echo "[l11on $IDX] gsm50k5 $THREAT seed$SEED ${SRC}_gate_v2 -> $RR  $(date '+%F %T')"

cd "$REPO/codes"
env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  REGIME=gsm50k5 THREAT="$THREAT" SEED="$SEED" \
  ARMS="${SRC}_gate_v2" OBS_SOURCES="$SRC" T2=0 T2_LEGACY=0 T2_P5=0 \
  RUNDIR_ROOT="$RR" \
  "$PY" -u experiments/track_g.py
echo "[l11on $IDX] EXIT=$? $(date '+%F %T')"
