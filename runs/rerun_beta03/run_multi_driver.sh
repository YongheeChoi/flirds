#!/bin/bash
# Generalized 4-GPU slot scheduler for the ShapleyFL beta=0.3 full re-run campaign.
# Same scheduler as runs/phase2_matrix/run_driver.sh, but each QUEUE line carries its own
# runner: "script|run_name|envs"  (script lives under codes/experiments/).
#
# Each runner self-persists to its own rundir root and OVERWRITES the existing cell:
#   - phase2_matrix.py / track_d.py read RUN_NAME (we pass it) -> exact dir name.
#   - track_c1.py / track_c2.py ignore RUN_NAME and self-generate the (hyphenated) name
#     from their C1_*/C2_* envs -> the same existing dir.
# Pops the next cell whenever a GPU slot frees; crosses tiers freely (one continuous queue).
# Exits when the queue is drained AND no cell is still running.  Resumable: edit the queue
# (comment out done cells) and re-launch.
#
#   QUEUE=runs/rerun_beta03/master_queue.txt LOGDIR=runs/rerun_beta03/logs \
#     bash runs/rerun_beta03/run_multi_driver.sh
set -u
PY=/home/korea_bupj/miniconda3/envs/flirds/bin/python
PP=/NHNHOME/26msit001_A/edge_ai_lab/yonghee/flirds/codes
QUEUE=${QUEUE:?set QUEUE}
LOGDIR=${LOGDIR:?set LOGDIR}
mkdir -p "$LOGDIR"
read -ra gpus <<< "${GPUS:-0 1 2 3}"        # GPUS="0 1 2" to restrict (default: all four)
declare -A pid cellname
consumed=0
done_re="MATRIX DONE|TRACK D DONE|\[persist\]"
echo "[$(date +%F_%H:%M:%S)] MULTI-DRIVER START: queue=$QUEUE -> $LOGDIR"

running() { for g in "${gpus[@]}"; do p=${pid[$g]:-}; [ -n "$p" ] && kill -0 "$p" 2>/dev/null && return 0; done; return 1; }

while : ; do
  mapfile -t Q < "$QUEUE"
  N=${#Q[@]}
  [ $consumed -ge $N ] && ! running && break
  for g in "${gpus[@]}"; do
    p=${pid[$g]:-}
    if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then
      if [ -n "${cellname[$g]:-}" ] && [ -n "$p" ]; then
        tail -3 "$LOGDIR/${cellname[$g]}.log" 2>/dev/null | grep -qE "$done_re" && st=ok || st=CHECK
        echo "[$(date +%H:%M:%S)] GPU$g done[$st]: ${cellname[$g]}"
        pid[$g]=""; cellname[$g]=""
      fi
      while [ $consumed -lt $N ]; do
        line="${Q[$consumed]}"; consumed=$((consumed+1))
        case "$line" in ''|'#'*) continue ;; esac
        script="${line%%|*}"; rest="${line#*|}"; name="${rest%%|*}"; envs="${rest#*|}"
        env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$PP RUN_NAME="$name" TQDM_DISABLE=1 \
            TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false \
            $envs $PY -u "$PP/experiments/$script" > "$LOGDIR/$name.log" 2>&1 &
        pid[$g]=$!; cellname[$g]="$name"
        echo "[$(date +%H:%M:%S)] GPU$g start: $name [$consumed/$N] ($script pid ${pid[$g]})"
        break
      done
    fi
  done
  sleep 15
done
echo "[$(date +%F_%H:%M:%S)] MULTI-DRIVER DONE: consumed $consumed cells"
