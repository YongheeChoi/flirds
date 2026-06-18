#!/bin/bash
# Track C grid status: queue summary + per-class done/fail counts from logs.
LOGDIR=/home/chyoyhr/projects/flirds/slurm/logs
echo "## squeue ($(date '+%F %T'))"
squeue -u "$USER" -o "%.10i %.12j %.2t %.10M %.16R" | head -40
NR=$(squeue -h -u "$USER" -t R | wc -l); NP=$(squeue -h -u "$USER" -t PD | wc -l)
echo "running=$NR pending=$NP"
echo
echo "## log sentinels"
for cls in fl_smoke fl_probe p_c1 p_c2 c1traj c1ora c2base c2d1fm c2d1cD c2sweep; do
  tot=$(ls "$LOGDIR"/${cls}*-*.out 2>/dev/null | wc -l)
  [ "$tot" -eq 0 ] && continue
  ok=$(grep -l "RUN OK\|ALL OK\|PROBE OK" "$LOGDIR"/${cls}*-*.out 2>/dev/null | wc -l)
  fail=$(grep -l "rc=[1-9]\|CANCELLED\|Traceback\|CUDA unavailable\|DUE TO TIME LIMIT" "$LOGDIR"/${cls}*-*.out 2>/dev/null | wc -l)
  echo "$cls: logs=$tot ok=$ok flagged=$fail"
done
echo
echo "## run dirs"
for d in /home/chyoyhr/projects/flirds/runs/track_c/c1 /home/chyoyhr/projects/flirds/runs/track_c/c2; do
  [ -d "$d" ] && echo "$d: $(find "$d" -name metrics.json | wc -l) metrics.json"
done
