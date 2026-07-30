#!/usr/bin/env bash
# Taylor-remainder measurement over main Table 1's setting, both tracks.
#
#   CNN  (Table 1 itself)          CIFAR-10 / Dir(1), N=10, full, R=10
#                                  4 conditions x 3 seeds = 12 cells
#   LLM  (Table 1's counterpart)   five-domain non-IID,  N=5,  full, R=10
#                                  3 conditions x 3 seeds =  9 cells
#
# Conditions follow main Section 5.1's per-track assignment: gradient noise and
# label-flip are CNN-only, answer-swap is LLM-only, clean and zero-update are
# shared.  Every cell writes its own rundir under runs/taylor_remainder/rundirs/;
# make_analysis.py rebuilds every table from those rundirs alone.
#
# Run from codes/:  bash ../runs/taylor_remainder/run_table1.sh [cnn|llm|smoke]
set -euo pipefail

WHICH="${1:-all}"
cd "$(dirname "$0")/../../codes"
export PYTHONPATH=.
SEEDS="${SEEDS:-0 1 2}"
# Cells are independent; to shard across GPUs, run this script once per device with
# CUDA_VISIBLE_DEVICES=<d> SEEDS=<subset>.

if [ "$WHICH" = "smoke" ]; then
  echo "### wiring smoke (seconds, no persist) ###"
  TAYLOR_SMOKE=1 TAYLOR_PERSIST=0 TAYLOR_DATASET=mnist \
    python -u experiments/measure_taylor_cnn.py
  # tiny-gpt2 + SYNTH_DATA needs no download; values are noise, wiring only.
  TAYLOR_SMOKE=1 SYNTH_DATA=1 TAYLOR_PERSIST=0 TAYLOR_MODEL=tiny-gpt2 \
    python -u experiments/measure_taylor_llm.py
  exit 0
fi

if [ "$WHICH" = "cnn" ] || [ "$WHICH" = "all" ]; then
  echo "### CNN -- main Table 1 setting (CIFAR-10 Dir(1), N=10, full, R=10) ###"
  for threat in clean free_rider grad_noise label_flip; do
    for s in $SEEDS; do
      echo "[cnn] threat=$threat seed=$s"
      TAYLOR_THREAT="$threat" TAYLOR_SEED="$s" \
        python -u experiments/measure_taylor_cnn.py
    done
  done
fi

if [ "$WHICH" = "llm" ] || [ "$WHICH" = "all" ]; then
  echo "### LLM -- Table 1's counterpart (five-domain non-IID, N=5, full, R=10) ###"
  for threat in clean answer_swap freerider_zero; do
    for s in $SEEDS; do
      echo "[llm] threat=$threat seed=$s"
      REGIME=silo5 TAYLOR_THREAT="$threat" TAYLOR_SEED="$s" \
        python -u experiments/measure_taylor_llm.py
    done
  done
fi

# Optional: the same measurement in a setting the MAIN TEXT names, which closes the
# gap that C.5's five-domain stage is never introduced in the body.  The GSM8K main
# setting has K=5 per round (2^5=32 coalitions) but R=200, so subsample rounds.
if [ "$WHICH" = "gsm" ]; then
  echo "### LLM -- GSM8K main setting (N=50, 5/50, R=200), 20 rounds sampled ###"
  for threat in clean answer_swap freerider_zero; do
    for s in $SEEDS; do
      REGIME=gsm50k5 TAYLOR_THREAT="$threat" TAYLOR_SEED="$s" TAYLOR_N_MEASURE=20 \
        python -u experiments/measure_taylor_llm.py
    done
  done
fi

echo "DONE -- roll up with: python ../runs/taylor_remainder/make_analysis.py"
