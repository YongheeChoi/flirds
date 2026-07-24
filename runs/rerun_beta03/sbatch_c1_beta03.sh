#!/bin/bash
# §4 (REMAINING-slurm.md) -- C1 30-cell ShapleyFL beta=0.3 re-run.
#
# The paper-cited ShapleyFL C1 values were produced at beta=0.5 (audit 07-23):
# the C1 30 cells (git_sha 5cb927b, 06-12) predate the beta 0.5->0.3 change
# (e89af94, 06-25).  This re-run regenerates every C1 cell with beta=0.3 and
# OVERWRITES the canonical rundir in place (runs/track_c/c1/<name>).
#
#   * SFL_BETA=0.3  -- the source default is already 0.3; set explicitly for the record.
#   * RUNDIR_REPLACE=1 -- identity-guard bypass (track_c1 has identity=None, so it
#     self-generates the name from C1_* and overwrites; the env is a harmless no-op
#     belt-and-suspenders per REMAINING-slurm.md §4/§6).
#   * C1_ORACLE_A=0 -- no (a)-retrain oracle (matches the master_queue cells; keeps
#     cost to the frozen-trajectory 9-method + Ripple pass).
#   * C1_RIPPLE=1 (default, NOT disabled) -- the overwrite must preserve every method
#     column incl. Ripple, or the cell is corrupted for make_fidelity.
#
# The 30 cells = {cifar10, mnist} x {feature_noise, iid, label_flip, label_skew,
# quantity_skew} x seed{0,1,2}, dataset-major to match master_queue.txt order.
# The beta=0.5 originals stay in git history (rundirs are git-tracked) -> recoverable.
# Yonghee reviews the local rundir diff and commits (no push here).
#
#SBATCH --job-name=c1b03
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
# 10h: Ripple on CIFAR varies 1.8-4.0h (a label_skew cell once timed out at 4.5h;
# see track-c-grid-run memory) and the full 9-method + Ripple pass sits on top.
# Partition MaxTime is 14 days, so the headroom costs only backfill priority.
#SBATCH --array=0-29%8
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/rerun_beta03/_logs/%x_%A_%a.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}

SCENARIOS=(feature_noise iid label_flip label_skew quantity_skew)

IDX=${SLURM_ARRAY_TASK_ID}
DS=$([ "$IDX" -lt 15 ] && echo cifar10 || echo mnist)
J=$((IDX % 15)); SC=$((J / 3)); SEED=$((J % 3))
SCENARIO=${SCENARIOS[$SC]}

echo "[c1b03 $IDX] ${DS}_${SCENARIO}_seed${SEED}  $(date '+%F %T')"

cd "$REPO/codes"
env PYTHONPATH=. HF_HUB_OFFLINE=1 \
  SFL_BETA=0.3 RUNDIR_REPLACE=1 \
  C1_DATASET="$DS" C1_SCENARIO="$SCENARIO" C1_SEED="$SEED" \
  C1_MODE=full C1_ORACLE_A=0 \
  "$PY" -u experiments/track_c1.py
echo "[c1b03 $IDX] EXIT=$? $(date '+%F %T')"
