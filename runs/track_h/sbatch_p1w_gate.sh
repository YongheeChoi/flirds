#!/bin/bash
# §2 (REMAINING-slurm.md) -- W-B pilot GO gate, automated.
#
# Submitted with --dependency=afterany:<pilot_jobid> so it runs ONCE the seed-0
# pilot array (sbatch --array=0-29 sbatch_cnn_p1w.sh) has finished.  It reproduces
# the RUN_P1W_CNN.md step-2 verification and, only if it passes, submits the full
# leg (seeds 1-2, array 30-89).  Fail-safe: any failed check -> NO submit, loud log.
#
# PASS = (>=28 of 30 pilot cells exited 0) AND (make_p1w_cnn_table reports T2
# retrain rows > 0, i.e. the T2 leg actually materialised on disk).
#
#SBATCH --job-name=p1wgate
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_h/_logs/%x_%A.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}
PILOT_JOB=${PILOT_JOB:?set PILOT_JOB to the pilot array job id}

echo "=========================================================="
echo "[p1w-gate] $(date '+%F %T')  pilot job=$PILOT_JOB"

# 1) pilot success count from the per-task logs
LOGS=("$REPO"/runs/track_h/_logs/hp1w_${PILOT_JOB}_*.out)
NOK=$(grep -lE "EXIT=0" "${LOGS[@]}" 2>/dev/null | wc -l)
NLOG=$(ls "${LOGS[@]}" 2>/dev/null | wc -l)
echo "[p1w-gate] pilot cells: $NOK/$NLOG exited 0"

# rough GPU-h from the [hp1w N] START/EXIT timestamps in each log
GPUH=$("$PY" - "$REPO" "$PILOT_JOB" <<'PY'
import sys, glob, re, datetime as dt
repo, job = sys.argv[1], sys.argv[2]
tot = 0.0
for f in glob.glob(f"{repo}/runs/track_h/_logs/hp1w_{job}_*.out"):
    ts = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", open(f, errors="ignore").read())
    if len(ts) >= 2:
        a = dt.datetime.strptime(ts[0], "%Y-%m-%d %H:%M:%S")
        b = dt.datetime.strptime(ts[-1], "%Y-%m-%d %H:%M:%S")
        tot += (b - a).total_seconds()
print(f"{tot/3600:.1f}")
PY
)
echo "[p1w-gate] pilot GPU-h (log-timestamp sum): ~${GPUH}"

# 2) regenerate the W-B table -> confirm T2 retrain rows materialised
cd "$REPO/runs/track_h"
PYTHONPATH="$REPO/codes" "$PY" -u make_p1w_cnn_table.py || echo "[p1w-gate] WARN: make_p1w_cnn_table exit=$?"
N_T2=$(grep -oP 'T2 retrain rows: \K[0-9]+' "$REPO/runs/track_h/analysis/p1w_cnn_README.md" 2>/dev/null | head -1)
N_T2=${N_T2:-0}
echo "[p1w-gate] T2 retrain rows in table: $N_T2"

# 3) decision (fail-safe)
if [ "$NOK" -ge 28 ] && [ "$N_T2" -gt 0 ]; then
  echo "[p1w-gate] GATE PASS -> submitting full leg (seeds 1-2, array 30-89%8)"
  OUT=$(sbatch --array=30-89%8 "$REPO/runs/track_h/sbatch_cnn_p1w.sh")
  echo "[p1w-gate] $OUT"
else
  echo "[p1w-gate] GATE FAIL (need NOK>=28 and N_T2>0) -> full leg NOT submitted."
  echo "[p1w-gate] Manual review: check pilot logs + analysis/p1w_cnn_README.md, then"
  echo "           sbatch --array=30-89%8 $REPO/runs/track_h/sbatch_cnn_p1w.sh"
fi
echo "[p1w-gate] done $(date '+%F %T')"
