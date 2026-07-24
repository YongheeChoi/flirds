#!/bin/bash
# L4 -- R4 Tier B T2-only: the 4 renorm sources' retrain intervention (P1 retrain).
# (REMAINING-slurm-JW.md §1; REMAINING-b200.md §2 L4.)
#
# §5.3 R4 has a retrain table of 8 sources.  same-game(flirds/1st/loss-heur)+fedif T2 =
# B200 L1 (t2_sign, exact-0 kept).  This leg fills the 4 renorm sources
# (gtg fedsv comfedsv shapleyfl): observer(vanilla) scored by all 4 -> T2 retrain on
# each source's cum>0 kept.  renorm = value-only, self-scoring inline (NO HVP) = B200-indep.
# clean threat is INCLUDED (2026-07-24 Yonghee): renorm can false-fire on clean (negative
# phi) -> clean T2 actually retrains (not equals_vanilla) -> fills the §5.3 clean column.
#
# GPU: A6000 48GB (retrain-scoring class ~32 GiB; 24GB insufficient).
#
# 9 tasks = {clean, noisy(nr0.7), frzero} x 3 seed.  SEED-MAJOR (3/seed).  Each task =
# 1 observer trajectory + 4 t2_sign retrains (~20h; arm-level persistent across a kill).
#   JW:  sbatch runs/track_h/sbatch_l4_renorm_t2.sh              # all 9 (seed0 first: 0-2)
#   pilot: sbatch --array=0-2%8 ...  (seed0) -> GPU-h report -> --array=3-8%8
# seed0 first = paper 착수선.  After: python runs/track_h/make_analysis.py
#
#SBATCH --job-name=l4renT2
#SBATCH --partition=suma_a6000,gigabyte_a6000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-8%8
#SBATCH --output=runs/track_h/_logs/%x_%A_%a.out
# base_qos gates A6000; edit --qos if your account names it differently.  --output is
# relative to the SUBMIT dir -> submit from the repo root after: mkdir -p runs/track_h/_logs

set -u
REPO=${REPO:-$HOME/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}   # shared lora4cl (torch2.11); override PY= if not readable
export HF_HOME=${HF_HOME:-/scratch/chyoyhr/hf_home}         # 1B model + gsm8k cache; override HF_HOME=

THREATS=(clean noisy frzero)

IDX=${SLURM_ARRAY_TASK_ID}
SEED=$((IDX / 3)); T=$((IDX % 3)); THREAT=${THREATS[$T]}     # seed-major: 3 threats/seed
RR="${RUNDIR_ROOT:-$REPO/runs/track_h/rundirs_llm_jw}"       # override for work-steal

echo "[l4renT2 $IDX] gsm50k5 $THREAT seed$SEED renorm-4 T2 -> $RR  $(date '+%F %T')"

cd "$REPO/codes"
env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  REGIME=gsm50k5 THREAT="$THREAT" SEED="$SEED" \
  ARMS=observer OBS_SOURCES=gtg,fedsv,comfedsv,shapleyfl \
  T2=1 T2_LEGACY=1 T2_CSIGN=0 T2_P5=0 \
  RUNDIR_ROOT="$RR" \
  "$PY" -u experiments/track_g.py
echo "[l4renT2 $IDX] EXIT=$? $(date '+%F %T')"
