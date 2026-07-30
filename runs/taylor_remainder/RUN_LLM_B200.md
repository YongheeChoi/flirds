# LLM leg — B200 runbook

Taylor-remainder measurement on the LLM track, in main Table 1's counterpart
setting: **five-domain non-IID, Llama-3.2-1B-Instruct, N=5, full participation,
R=10**, 3 conditions × 3 seeds = **9 cells**.

Spec, rationale, and what the numbers can/cannot establish: `README.md` here.

## Before you start

```bash
cd <repo>/codes
export PYTHONPATH=.
PY=/home/korea_bupj/miniconda3/envs/flirds/bin/python    # adjust after the migration
```

Everything below is `cd codes` + `PYTHONPATH=.`, matching the project convention.

## Step 0 — offline wiring smoke (~30 s, no GPU, no download)

Confirms the whole path — data seam, LoRA/eager model, FedAvg trajectory, HVP,
coalition enumeration, persistence — on a random-init 2-layer GPT-2. Values are
noise; this checks wiring only.

```bash
SYNTH_DATA=1 TAYLOR_SMOKE=1 TAYLOR_PERSIST=0 TAYLOR_MODEL=tiny-gpt2 $PY -u experiments/measure_taylor_llm.py
```

Expect `TAYLOR-LLM OK (no persist)` and a `closed form vs Shapley(u2)` around 1e-16.
Run it for all three conditions if you want the corrupt-set wiring checked too:

```bash
for th in clean answer_swap freerider_zero; do SYNTH_DATA=1 TAYLOR_SMOKE=1 TAYLOR_PERSIST=0 TAYLOR_MODEL=tiny-gpt2 TAYLOR_THREAT=$th $PY -u experiments/measure_taylor_llm.py; done
```

`corrupt=` should print `[]` / `[0]` / `[1]` respectively — 1 of 5 clients = the
20% the paper's Appendix B.2 states for this setting.

## Step 1 — one real cell first (cost check, ~35 min)

Do **not** launch all 9 before this lands. C.5's published run took ~605 s
trajectory + ~1,536 s measurement per seed; this cell should land near that, and
it costs K=5 HVPs per round instead of 1, so confirm before committing 9 cells.

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  REGIME=silo5 TAYLOR_THREAT=clean TAYLOR_SEED=0 \
  $PY -u experiments/measure_taylor_llm.py
```

Sanity checks on the output:

| line | expect |
|---|---|
| header | `N=5 k_frac=1.0 R=10 steps=10 lr=0.001 val=100` |
| `[fl]` | `10 rounds`, `K=5/round` |
| `resid2 / floor` | **≈ 2x** — this reproduces C.5's floor-limited regime, it is the control, not a failure |
| `closed form vs Shapley(u2)` | ~1e-10 |
| `[persist]` | a rundir under `runs/taylor_remainder/rundirs/` |

If `resid2 / floor` comes out near 2 and the slope near 1.5–1.6, the cell has
reproduced the published C.5 result and the harness is validated end-to-end.

Peak memory is one val chunk (`val_chunk=10`), same as the paper runs; add
`VAL_CHUNK=5` if you hit fragmentation. Chunk-summing is exact, so this changes
memory only, never the values.

## Step 2 — the remaining 8 cells, sharded 3 ways

One condition per GPU, three seeds serially. ~1.75 h wall-clock.

```bash
# GPU 0 — clean (seed 0 already done in step 1)
for s in 1 2; do CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  REGIME=silo5 TAYLOR_THREAT=clean TAYLOR_SEED=$s $PY -u experiments/measure_taylor_llm.py; done

# GPU 1 — answer-swap
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  REGIME=silo5 TAYLOR_THREAT=answer_swap TAYLOR_SEED=$s $PY -u experiments/measure_taylor_llm.py; done

# GPU 2 — zero-update free-rider
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  REGIME=silo5 TAYLOR_THREAT=freerider_zero TAYLOR_SEED=$s $PY -u experiments/measure_taylor_llm.py; done
```

Or serially on one GPU via the runner (≈5.4 h):

```bash
bash ../runs/taylor_remainder/run_table1.sh llm
```

## Step 3 — roll up

```bash
$PY ../runs/taylor_remainder/make_analysis.py
```

Reads only the rundirs, writes `analysis/{cells,remainder_by_condition,
remainder_pooled,order_slopes,cost}.csv`, and prints both tables. Re-runnable at
any time; wiring smokes are auto-excluded from the results.

## Optional — the main-text setting (~4 h, 9 cells)

C.5's five-domain stage is never named in the main paper. The GSM8K main setting
(N=50, 5/50, R=200) is the headline LLM setting, so measuring there closes that
gap. K=5 per round there too, but R=200, so subsample rounds:

```bash
for th in clean answer_swap freerider_zero; do for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    REGIME=gsm50k5 TAYLOR_THREAT=$th TAYLOR_SEED=$s TAYLOR_N_MEASURE=20 \
    $PY -u experiments/measure_taylor_llm.py
done; done
```

`TAYLOR_N_MEASURE=20` measures 20 evenly spaced rounds of 200. The remainder is a
per-round quantity, so a subsample carries the same statistics at a tenth of the
measurement cost — but the trajectory still runs all 200 rounds, which is the bulk
of the time here.

## Knobs

| env | default | note |
|---|---|---|
| `REGIME` | `silo5` | `silo5` = Table 1's counterpart; `gsm50k5` = GSM8K main setting |
| `TAYLOR_THREAT` | `clean` | `clean` / `answer_swap` / `freerider_zero` |
| `TAYLOR_SEED` | `0` | paper uses 0,1,2 |
| `TAYLOR_N_MEASURE` | `all` | int = measure that many evenly spaced rounds |
| `TAYLOR_MODEL` | `Llama-3.2-1B-Instruct` | `tiny-gpt2` = offline smoke |
| `TAYLOR_RENORM` | `0` | adds the P5 renormalized game; doubles forwards |
| `TAYLOR_PERSIST` | `1` | `0` = stdout only |
| `TAYLOR_RUN_ROOT` | `runs/taylor_remainder/rundirs` | |
| `VAL_CHUNK` via `MCFG` | `10` | memory only, values unchanged |

`ROUNDS` / `MAX_STEPS` / `TRAIN` / `VAL` / `N_CLIENTS` also override the regime
config if you need a cheaper probe — but then it is no longer Table 1's setting.

## If something breaks

- **`cp949` / `UnicodeDecodeError` importing `trl`** — Windows-only, locale issue.
  Set `PYTHONUTF8=1`. Should not occur on the Linux nodes.
- **OOM in the HVP** — lower `VAL_CHUNK` (exact, memory-only). The per-client HVP
  loop holds K direction tensors; at K=5 with LoRA r=16 that is small, so the val
  chunk is the real driver.
- **`pkeys mismatch` assertion** — the LoRA target modules disagree between
  `load_model` and `make_llm_loss`. Only fires if `TAYLOR_LORA_R` or the model was
  changed mid-run.
- **A cell dies** — cells are independent and each writes its own rundir; rerun
  just that `(threat, seed)`. `make_analysis.py` skips unreadable rundirs with a
  `[skip]` line rather than failing the rollup.
