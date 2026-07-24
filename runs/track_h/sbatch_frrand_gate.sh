#!/bin/bash
# §3 (REMAINING-slurm.md) -- C-fr frrand pilot GO gate, automated (mirrors the
# W-B gate).  Submitted with --dependency=afterany:<pilot>; runs once the seed-0
# pilot (array 0-7) finishes, reports measured GPU-h, and -- only if the pilot
# passed -- submits seeds 1-2 (array 8-23).  Fail-safe: any failed check -> NO
# submit, loud log.
#
# PASS = (>=7 of 8 pilot cells exited 0) AND (>=1 cifar10_dir1_frrand_*_seed0
#        rundir materialised on disk incl. the obs/T2 cell).
#
#SBATCH --job-name=frgate
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/track_h/_logs/%x_%A.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}
PILOT_JOB=${PILOT_JOB:?set PILOT_JOB to the pilot array job id}

echo "=========================================================="
echo "[frrand-gate] $(date '+%F %T')  pilot job=$PILOT_JOB"

LOGS=("$REPO"/runs/track_h/_logs/hfrrand_${PILOT_JOB}_*.out)
NOK=$(grep -lE "EXIT=0" "${LOGS[@]}" 2>/dev/null | wc -l)
NLOG=$(ls "${LOGS[@]}" 2>/dev/null | wc -l)
echo "[frrand-gate] pilot cells: $NOK/$NLOG exited 0"

GPUH=$("$PY" - "$REPO" "$PILOT_JOB" <<'PY'
import sys, glob, re, datetime as dt
repo, job = sys.argv[1], sys.argv[2]
tot = 0.0
for f in glob.glob(f"{repo}/runs/track_h/_logs/hfrrand_{job}_*.out"):
    ts = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", open(f, errors="ignore").read())
    if len(ts) >= 2:
        a = dt.datetime.strptime(ts[0], "%Y-%m-%d %H:%M:%S")
        b = dt.datetime.strptime(ts[-1], "%Y-%m-%d %H:%M:%S")
        tot += (b - a).total_seconds()
print(f"{tot/3600:.1f}")
PY
)
echo "[frrand-gate] pilot GPU-h (log-timestamp sum): ~${GPUH}"

NRD=$(ls -d "$REPO"/runs/track_h/rundirs_cnn/cifar10_dir1_frrand_*_seed0 2>/dev/null | wc -l)
NOBS=$(ls -d "$REPO"/runs/track_h/rundirs_cnn/cifar10_dir1_frrand_*obs*seed0 "$REPO"/runs/track_h/rundirs_cnn/cifar10_dir1_frrand_t2_*seed0 2>/dev/null | wc -l)
echo "[frrand-gate] seed0 frrand rundirs: $NRD (obs/T2-related: $NOBS)"

if [ "$NOK" -ge 7 ] && [ "$NRD" -gt 0 ]; then
  echo "[frrand-gate] GATE PASS -> submitting seeds 1-2 (array 8-23%8)"
  OUT=$(sbatch --array=8-23%8 "$REPO/runs/track_h/sbatch_cnn_frrand.sh")
  echo "[frrand-gate] $OUT"
else
  echo "[frrand-gate] GATE FAIL (need NOK>=7 and rundirs>0) -> seeds 1-2 NOT submitted."
  echo "[frrand-gate] Manual: sbatch --array=8-23%8 $REPO/runs/track_h/sbatch_cnn_frrand.sh"
fi
echo "[frrand-gate] done $(date '+%F %T')"
