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
# ⚠ SCOPE CUT 2026-07-25 (Yonghee): the LLM downstream table keeps only
#   {vanilla/observer, oracle_excl, random_excl, flirds-family} + anchors.  The 4 renorm
#   sources (gtg fedsv comfedsv shapleyfl) are OUT of the LLM downstream table -- their
#   collapse is already shown by CNN §5.3 (8 sources, both tables, 3-seed) and by LLM
#   §5.2 fidelity (G1 emits all 9 methods).  R4 therefore stays at **R=200** (the R=100
#   plan existed only to fit those legs; dropping them means the already-complete
#   3-seed L1 needs no re-run).
#
#   => From this file, run **flirds1st ONLY** (SRC_I 0), i.e. 9 cells:
#        sbatch --array=0-2,21-23,42-44%8 runs/track_h/sbatch_l11_online.sh
#      That fills the online table's Flirds-1st row so online matches the retrain table
#      (which already has flirds/flirds1st/lossheur/fedif t2_sign at 3 seeds on disk).
#   ⚠ Check `rundirs_llm_hj` first -- the earlier R=200 run may have already landed some
#      flirds1st cells (cheap sources land first).  Those are VALID (same R=200 stage);
#      submit only the missing indices.
#
# 63 tasks = 7 src x {clean, noisy(nr0.7), frzero} x 3 seed.  SEED-MAJOR (21/seed);
# within a seed, SRC_I = (IDX%21)/3 orders the sources as listed in SRCS below.
# The landing root auto-routes by seed (seed2 -> rundirs_llm_yh, else rundirs_llm_hj);
# override RUNDIR_ROOT for work-steal.  After: python runs/track_h/make_analysis.py
#
#SBATCH --job-name=l11on
#SBATCH --partition=suma_a6000,gigabyte_a6000,asus_6000ada
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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

# VAL_CHUNK is a MEMORY knob for the *grad* path only (functorch val-grad OOMs on A6000
# at the default 10).  flirds1st/fedif take that path; lossheur + the renorm-4 are
# forward-only (@no_grad, memory-bounded per chunk) and were being dragged down to 3 for
# no reason.  Chunk-sum is exact -> phi is identical either way, so this is zero-risk.
# Whether it is also FASTER is UNMEASURED: make_llm_loss's docstring reports the (b)
# oracle profiled ~1.0x when its chunk was raised (FLOP-bound, not launch-bound) at
# chunk 10, and HJ's 07-25 re-measurement shows same-game cells are dominated by the
# 200-round TRAINING (110 of 132 s/round), not scoring.  The renorm-4 cells are the
# opposite (~92% scoring), so any win lands there -- do not assume one; just don't
# impose a constraint those cells never needed.
val_chunk_for() { case "$1" in flirds1st|fedif) echo "${VAL_CHUNK:-3}";; *) echo "${VAL_CHUNK:-10}";; esac; }

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
  RUNDIR_REPLACE=1 VAL_CHUNK="$(val_chunk_for "$SRC")" \
  ARMS="${SRC}_gate_v2" OBS_SOURCES="$SRC" T2=0 T2_LEGACY=0 T2_P5=0 \
  RUNDIR_ROOT="$RR" \
  "$PY" -u experiments/track_g.py
echo "[l11on $IDX] EXIT=$? $(date '+%F %T')"
