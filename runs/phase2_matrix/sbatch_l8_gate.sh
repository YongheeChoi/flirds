#!/bin/bash
# §5 L8 pilot GO gate (mirrors the W-B/frrand gates).  Submitted with
# --dependency=afterany:<pilot>; runs once the seed-0 pilot leg finishes, reports the
# measured GPU-h, and -- only if the pilot produced its completion marker + rundir --
# submits the remaining legs.  Fail-safe: any failed check -> NO submit, loud log.
#
# Pass only PILOT_JOB and LOG_PREFIX via --export; the rest is derived here so no
# spaces/globs go through sbatch --export.
#   LOG_PREFIX=gsm5    -> marker "MATRIX DONE",        resubmit 1-5%8, sbatch_gsm5.sh
#   LOG_PREFIX=silo5a  -> marker "SILO5 (a)-LEG DONE", resubmit 1-8%8, sbatch_silo5_a.sh
#
#SBATCH --job-name=l8gate
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/chyoyhr/projects/flirds/runs/phase2_matrix/_logs/%x_%A.out

set -u
REPO=${REPO:-/home/chyoyhr/projects/flirds}
PY=${PY:-/home/chyoyhr/anaconda3/envs/lora4cl/bin/python}
: "${PILOT_JOB:?}"; : "${LOG_PREFIX:?}"
RR="$REPO/runs/phase2_matrix/rundirs"
case "$LOG_PREFIX" in
  gsm5)   MARKER="MATRIX DONE";        ARR="1-5%8"; SB="$REPO/runs/phase2_matrix/sbatch_gsm5.sh";    GLOB="$RR/1B_gsm5_clean_*_s0" ;;
  silo5a) MARKER="SILO5 (a)-LEG DONE"; ARR="1-8%8"; SB="$REPO/runs/phase2_matrix/sbatch_silo5_a.sh"; GLOB="$RR/1B_silo5_clean_aonly_s0" ;;
  *) echo "[l8-gate] unknown LOG_PREFIX=$LOG_PREFIX"; exit 1 ;;
esac

echo "=========================================================="
echo "[l8-gate:$LOG_PREFIX] $(date '+%F %T')  pilot=$PILOT_JOB"

LOGS=("$REPO"/runs/phase2_matrix/_logs/${LOG_PREFIX}_${PILOT_JOB}_*.out)
NOK=$(grep -lE "EXIT=0" "${LOGS[@]}" 2>/dev/null | wc -l)
NMARK=$(grep -lF "$MARKER" "${LOGS[@]}" 2>/dev/null | wc -l)
echo "[l8-gate] pilot EXIT=0:$NOK  marker('$MARKER'):$NMARK"

GPUH=$("$PY" - "$REPO" "$LOG_PREFIX" "$PILOT_JOB" <<'PY'
import sys, glob, re, datetime as dt
repo, pref, job = sys.argv[1], sys.argv[2], sys.argv[3]
tot = 0.0
for f in glob.glob(f"{repo}/runs/phase2_matrix/_logs/{pref}_{job}_*.out"):
    ts = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", open(f, errors="ignore").read())
    if len(ts) >= 2:
        a = dt.datetime.strptime(ts[0], "%Y-%m-%d %H:%M:%S")
        b = dt.datetime.strptime(ts[-1], "%Y-%m-%d %H:%M:%S")
        tot += (b - a).total_seconds()
print(f"{tot/3600:.2f}")
PY
)
echo "[l8-gate] pilot GPU-h (log-timestamp): ~${GPUH}"

NRD=$(ls -d $GLOB 2>/dev/null | wc -l)
echo "[l8-gate] pilot rundirs matching '$GLOB': $NRD"

if [ "$NMARK" -ge 1 ] && [ "$NRD" -ge 1 ]; then
  echo "[l8-gate] GATE PASS -> submitting remaining legs (array $ARR)"
  OUT=$(sbatch --array="$ARR" "$SB")
  echo "[l8-gate] $OUT"
else
  echo "[l8-gate] GATE FAIL (need marker>=1 and rundir>=1) -> remaining legs NOT submitted."
  echo "[l8-gate] Manual: sbatch --array=$ARR $SB"
fi
echo "[l8-gate:$LOG_PREFIX] done $(date '+%F %T')"
