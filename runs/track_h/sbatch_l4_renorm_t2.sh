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
# VAL_CHUNK=10 (default), NOT the 3 used elsewhere: this observer scores renorm-4 only,
# which is forward-only (@no_grad, memory-bounded per chunk) -- no functorch val-grad and
# no HVP, so the A6000 OOM guard that forces 3 does not apply here.  Chunk-sum is exact
# -> phi identical either way.
#
# ⛔ OUT OF SCOPE 2026-07-25 (Yonghee): the LLM downstream table keeps only
#    {vanilla/observer, oracle_excl, random_excl, flirds-family}; the 4 renorm sources are
#    dropped from it (their collapse is shown by CNN §5.3 and by LLM §5.2 fidelity).
#    DO NOT SUBMIT.  The file is kept because the leg is otherwise ready: if the deadline
#    moves, run it at R=200 (~100 h/cell, 900 GPU-h) or halve R first.
#    Everything below is the pre-cut runbook.
#
# 9 tasks = {clean, noisy(nr0.7), frzero} x 3 seed.  SEED-MAJOR (3/seed).  Each task =
# 1 observer trajectory + 4 t2_sign retrains (arm-level persistent across a kill).
# Split by SEED (REMAINING-00-INDEX.md §2).  This leg is ~3.6x cheaper on B200
# (~13.9 h/cell vs ~50 h/cell on A6000: the observer scores renorm-4 every round and
# that scoring is ~92% of the cell), so B200 takes 2 of the 3 seeds:
#   B200 : seed0 (queue_b200_c4.txt) + seed1 (queue_b200_c2.txt)   <- no sbatch
#   JB   : sbatch --array=6-8%8 runs/track_h/sbatch_l4_renorm_t2.sh   # seed2 only
# After: python runs/track_h/make_analysis.py
#
#SBATCH --job-name=l4renT2
#SBATCH --partition=suma_a6000,gigabyte_a6000,asus_6000ada
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-8%8
# --time=72h, NOT 42: at R=100 an A6000 cell is ~50 h (observer ~38 h + 4 retrains ~12 h).
# The rundir is written at arm end, so a timeout loses the whole observer arm.
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
# Own root: this cell's `observer` arm shares its rundir NAME with L1's same-game observer
# (name = regime_threat_arm_seed), so a shared root would have them overwrite each other.
RR="${RUNDIR_ROOT:-$REPO/runs/track_h/rundirs_llm_g4c}"      # override for work-steal

echo "[l4renT2 $IDX] gsm50k5 $THREAT seed$SEED renorm-4 T2 -> $RR  $(date '+%F %T')"

cd "$REPO/codes"
env PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  REGIME=gsm50k5 THREAT="$THREAT" SEED="$SEED" \
  ROUNDS="${ROUNDS:-100}" RUNDIR_REPLACE=1 VAL_CHUNK="${VAL_CHUNK:-10}" \
  ARMS=observer OBS_SOURCES=gtg,fedsv,comfedsv,shapleyfl \
  T2=1 T2_LEGACY=1 T2_CSIGN=0 T2_P5=0 \
  RUNDIR_ROOT="$RR" \
  "$PY" -u experiments/track_g.py
echo "[l4renT2 $IDX] EXIT=$? $(date '+%F %T')"
