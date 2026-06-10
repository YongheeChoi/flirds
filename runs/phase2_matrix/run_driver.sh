#!/bin/bash
# 4-GPU slot scheduler for phase2_matrix cells.  Keeps GPUs 0-3 fed: launches up to
# 4 cells concurrently, refilling each slot as its cell exits.  Each cell = one
# phase2_matrix.py invocation (env-parameterized) -> its own log.  Prints start/done
# lines + a final DRIVER DONE.  Cells passed via the CELLS env (newline-separated
# "logname|extra-envs"); LOGSUB selects the output subdir.
set -u
PY=/home/korea_bupj/miniconda3/envs/flirds/bin/python
PP=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/codes
SCRIPT=$PP/experiments/phase2_matrix.py
LOGDIR=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/runs/phase2_matrix/${LOGSUB:?set LOGSUB}
mkdir -p "$LOGDIR"

mapfile -t cells <<< "${CELLS:?set CELLS}"
gpus=(0 1 2 3)
declare -A pid cellname
i=0; N=${#cells[@]}
echo "[$(date +%H:%M:%S)] DRIVER START: $N cells -> $LOGDIR (4 GPU slots)"

running() { for g in "${gpus[@]}"; do p=${pid[$g]:-}; [ -n "$p" ] && kill -0 "$p" 2>/dev/null && return 0; done; return 1; }

while [ $i -lt $N ] || running; do
  for g in "${gpus[@]}"; do
    p=${pid[$g]:-}
    if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then
      if [ -n "${cellname[$g]:-}" ] && [ -n "$p" ]; then
        tail -1 "$LOGDIR/${cellname[$g]}.log" 2>/dev/null | grep -q "MATRIX DONE" \
          && st="ok" || st="CHECK"
        echo "[$(date +%H:%M:%S)] GPU$g done[$st]: ${cellname[$g]}"
        pid[$g]=""; cellname[$g]=""
      fi
      if [ $i -lt $N ]; then
        c="${cells[$i]}"; i=$((i+1))
        [ -z "${c// }" ] && continue          # skip blank lines (trailing newline in CELLS)
        name="${c%%|*}"; envs="${c#*|}"
        env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$PP TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error \
            HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false $envs \
            $PY -u $SCRIPT > "$LOGDIR/$name.log" 2>&1 &
        pid[$g]=$!; cellname[$g]="$name"
        echo "[$(date +%H:%M:%S)] GPU$g start: $name (pid ${pid[$g]})"
      fi
    fi
  done
  sleep 15
done
echo "[$(date +%H:%M:%S)] DRIVER DONE: all $N cells finished"
