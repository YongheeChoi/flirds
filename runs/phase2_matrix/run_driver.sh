#!/bin/bash
# 4-GPU slot scheduler over a QUEUE FILE.  Re-reads the queue each loop so cells appended
# (>>) while running get picked up.  Each non-blank/non-'#' line: "logname|extra-envs".
# Pops the next unconsumed line whenever a GPU slot frees; crosses tier boundaries freely
# (one continuous queue).  Exits when the queue is drained AND no cell is still running.
set -u
PY=/home/korea_bupj/miniconda3/envs/flirds/bin/python
PP=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/codes
SCRIPT=$PP/experiments/phase2_matrix.py
QUEUE=${QUEUE:?set QUEUE}
LOGDIR=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/runs/phase2_matrix/${LOGSUB:?set LOGSUB}
mkdir -p "$LOGDIR"
gpus=(0 1 2 3)
declare -A pid cellname
consumed=0
echo "[$(date +%H:%M:%S)] DRIVER START: queue=$QUEUE -> $LOGDIR"
running() { for g in "${gpus[@]}"; do p=${pid[$g]:-}; [ -n "$p" ] && kill -0 "$p" 2>/dev/null && return 0; done; return 1; }

while : ; do
  mapfile -t Q < "$QUEUE"
  N=${#Q[@]}
  [ $consumed -ge $N ] && ! running && break
  for g in "${gpus[@]}"; do
    p=${pid[$g]:-}
    if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then
      if [ -n "${cellname[$g]:-}" ] && [ -n "$p" ]; then
        tail -1 "$LOGDIR/${cellname[$g]}.log" 2>/dev/null | grep -q "MATRIX DONE" && st=ok || st=CHECK
        echo "[$(date +%H:%M:%S)] GPU$g done[$st]: ${cellname[$g]}"
        pid[$g]=""; cellname[$g]=""
      fi
      while [ $consumed -lt $N ]; do
        line="${Q[$consumed]}"; consumed=$((consumed+1))
        case "$line" in ''|'#'*) continue ;; esac
        name="${line%%|*}"; envs="${line#*|}"
        env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$PP RUN_NAME=$name TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error \
            HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false $envs \
            $PY -u $SCRIPT > "$LOGDIR/$name.log" 2>&1 &
        pid[$g]=$!; cellname[$g]="$name"
        echo "[$(date +%H:%M:%S)] GPU$g start: $name [$consumed/$N] (pid ${pid[$g]})"
        break
      done
    fi
  done
  sleep 15
done
echo "[$(date +%H:%M:%S)] DRIVER DONE: consumed $consumed cells"
