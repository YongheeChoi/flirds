#!/bin/bash
# L9 -- R4 frrand (random free-rider) non-flirds full-method: T1 online + T2 retrain.
# (REMAINING-slurm-JB.md §1; REMAINING-b200.md §3 L9.)
#
# R4's free-rider axis had only frzero -> add frrand to complete "exact-0 survives vs
# renorm collapses" on a RANDOM free-rider too.  flirds arm = B200 (HVP observer/online);
# this leg = the 7 non-flirds sources (flirds1st lossheur fedif gtg fedsv comfedsv
# shapleyfl), self-scoring inline (NO HVP) = B200-independent.
#   T1 (online): each source's <src>_gate_v2 (P1 online) -- 7 tasks/seed.
#   T2 (retrain): ONE observer trajectory scored by all 7 -> 7 t2_sign retrains, in a
#     single task/seed (one observer cell, no rundir race; ~33h, arm-level persistent).
# recovery denominator (vanilla/oracle_excl/random_excl) + flirds arms for the frrand cell
# come from B200's L9 flirds observer+online on the same meta key -- merged by make_analysis.
#
# GPU: A6000 48GB (retrain-scoring class ~32 GiB; 24GB insufficient).
#
# 24 tasks = (7 T1 + 1 T2) x 3 seed.  SEED-MAJOR (8/seed); within a seed the T1 gates
# (J 0-6) dispatch before the T2 monster (J 7), so seed0 online lands fast.
#   JB:  sbatch runs/track_h/sbatch_l9_frrand.sh                 # all 24 (seed0 first: 0-7)
#   pilot: sbatch --array=0-7%8 ...  (seed0) -> GPU-h report -> --array=8-23%8
# After: python runs/track_h/make_analysis.py
#
#SBATCH --job-name=l9frr
#SBATCH --partition=suma_a6000,gigabyte_a6000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=0-23%8
#SBATCH --output=runs/track_h/_logs/%x_%A_%a.out
# base_qos gates A6000; edit --qos if your account names it differently.  --output is
# relative to the SUBMIT dir -> submit from the repo root after: mkdir -p runs/track_h/_logs

set -u
REPO=${REPO:-$HOME/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}   # shared lora4cl (torch2.11); override PY= if not readable
export HF_HOME=${HF_HOME:-/scratch/chyoyhr/hf_home}         # 1B model + gsm8k cache; override HF_HOME=

SRCS=(flirds1st lossheur fedif gtg fedsv comfedsv shapleyfl)   # 7 non-flirds
RR="${RUNDIR_ROOT:-$REPO/runs/track_h/rundirs_llm_jb}"         # override for work-steal

IDX=${SLURM_ARRAY_TASK_ID}
SEED=$((IDX / 8)); J=$((IDX % 8))              # seed-major: 8 cells/seed (7 T1 + 1 T2)

cd "$REPO/codes"
if [ "$J" -lt 7 ]; then                        # T1 online: one source's sign-gate
  SRC=${SRCS[$J]}
  echo "[l9frr $IDX] frrand seed$SEED ${SRC}_gate_v2 (T1) -> $RR  $(date '+%F %T')"
  env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VAL_CHUNK="${VAL_CHUNK:-10}" \
    REGIME=gsm50k5 THREAT=frrand SEED="$SEED" \
    ARMS="${SRC}_gate_v2" OBS_SOURCES="$SRC" T2=0 T2_LEGACY=0 T2_P5=0 \
    RUNDIR_ROOT="$RR" \
    "$PY" -u experiments/track_g.py
else                                           # T2 retrain: observer scored by all 7 -> 7 t2_sign
  echo "[l9frr $IDX] frrand seed$SEED observer+T2(all-7) -> $RR  $(date '+%F %T')"
  env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VAL_CHUNK="${VAL_CHUNK:-10}" \
    REGIME=gsm50k5 THREAT=frrand SEED="$SEED" \
    ARMS=observer OBS_SOURCES=flirds1st,lossheur,fedif,gtg,fedsv,comfedsv,shapleyfl \
    T2=1 T2_LEGACY=1 T2_CSIGN=0 T2_P5=0 \
    RUNDIR_ROOT="$RR" \
    "$PY" -u experiments/track_g.py
fi
echo "[l9frr $IDX] EXIT=$? $(date '+%F %T')"
