# Re-run plan after the 2026-07-21 reproducibility fix

Two of the applied fixes **change reported numbers**, so the affected persisted
rundirs must be regenerated before the numbers are cited as reproducible. This file
lists exactly what to re-run, why, and in what order.

> The RANK-fidelity story (Spearman/Pearson vs (a)/(b) oracle ≈ +1.000) is init- and
> precision-robust and should reproduce the **same conclusion**. What moves is the
> *absolute* φ / AUROC / ROUGE-L / val-loss curves / cross-seed error bars. Re-run
> where those absolutes (or error bars) are cited.

## Why a re-run is needed

1. **H1 — LLM LoRA adapter init was drawn from an UNSEEDED RNG** (entropy-seeded per
   process) in `phase2_matrix.py`, `track_d.py`, `track_g.py`. It is now pinned
   (`seed_everything(0)` before `get_peft_model`). Consequence: every rundir those
   runners produced started from a *random* adapter; the code now produces a
   *different but reproducible* adapter. The old numbers were **never reproducible**
   (nobody, including the old code, can regenerate them) — re-running establishes the
   canonical, reproducible set. This affects **all LLM rundirs**.
2. **Determinism enforcement** — `repro.py` now sets
   `torch.use_deterministic_algorithms(True, warn_only=True)` and, for the CNN track,
   disables cuDNN TF32 (`allow_tf32=False` → **true fp32 conv**). This can shift
   last-bit values everywhere and CNN conv values more noticeably. This affects
   **CNN rundirs** (and is what makes the protocol-5 "bitwise fp32" claim actually true).

## Do this FIRST (prerequisite for a fully-anchored re-run)

- **Pin data + weights**: fill `codes/flirds/hf_pin.py` → `REVISIONS` with commit SHAs
  (resolve on a networked machine via `HfApi().dataset_info(id).sha` /
  `.model_info(id).sha` — instructions in the file), **or** export
  `HF_REVISION_DEFAULT`. Without this the re-run is reproducible w.r.t. (code, seed)
  but the data/weights still float on HF Hub `latest`.

---

## P0 — MUST re-run (H1; LLM grids; headline absolutes/AUROC/error-bars)

| group | rundirs | runner / driver | notes |
|---|---:|---|---|
| `runs/phase2_matrix/` | 54 | `experiments/phase2_matrix.py` via `runs/phase2_matrix/run_driver.sh` | the "25-cell REAL GRID" (silo5 / device100_sweep / device100_poison / anchor / 3B). **Also holds** the `matrix_cxni` corruption×non-IID cells (`runs/matrix_cxni/run_matrix.sh`). Seeds `[0,1,2]`. |
| `runs/track_g/` | 266 | `experiments/track_g.py` via `runs/track_g/run_llm_pilot.sh` (+ its CNN cells via `run_cnn_grid.sh`) | sign-gating arms; LLM cells are H1-affected. |
| `runs/removal_dose/` | 99 | `runs/removal_dose/run_full_sweep.sh` / `run_sweep_5gpu.sh` (phase2_matrix REMOVAL + track_d) | recovery / removal curves. |
| `runs/track_d/` | 25 | `experiments/track_d.py` (1-seed pilot → 3-seed) | LLM fidelity + intervention arms. |
| `runs/track_h/` (LLM cells) | ⊂237 | `experiments/track_g.py` (`REGIME=gsm50k5` / `std50k5`) via `runs/track_h/` spec | scale / P5 / recovery / R4 **LLM** cells (the CIFAR P5 cells are CNN → see P1). |
| `runs/probe_signal/` (LLM cells) | ⊂115 | `experiments/phase2_matrix.py` (`REGIME=std…`) via `runs/probe_signal/run_pilot.sh` | participation-axis std50k5 **LLM** cells (the CIFAR cells are CNN → see P1). |

**How** (per the runners' env knobs — all read `SEED`, so shard 1 seed/GPU or let the
default `[0,1,2]` run in one process): re-run each group with its existing driver, then
regenerate the group's derived analysis (`make_analysis.py`). The rundir names are now
seed-/config-injective (H2 + the `RunLogger` collision guard), so a re-run writes fresh
dirs beside the frozen ones rather than silently clobbering them — delete or archive the
old dirs once the new set is verified.

## P1 — recommended re-run (TF32→fp32 + determinism; CNN; rank-robust)

Re-run for precision consistency and to make the protocol-5 bitwise-fp32 claim literally
true; conclusions (rank fidelity, AUROC ordering) are expected to hold.

| group | rundirs | runner / driver |
|---|---:|---|
| `runs/track_c/` | 150 | `experiments/track_c1.py` / `track_c2.py` (see `runs/track_c/README.md`) |
| `runs/track_h/` (CNN P5 cells) | ⊂237 | `track_c1/c2` CIFAR label-flip dose cells |
| `runs/probe_signal/` (CNN cells) | ⊂115 | `track_c1/c2` CIFAR grids (`runs/probe_signal/run_*.sh`) |

## NOT required

- `runs/phase1/` (12) — `phase1_clean_run.py` already seeded the adapter per seed
  **before** building it (it was never H1-affected); only the revision/determinism
  changes touch it, negligibly.
- `runs/measured_2026-07/` (7) — op-counts (analytic) + wall-clock timing; not fidelity
  numbers. Wall-clock may drift with `use_deterministic_algorithms`; re-measure only if
  a timing table cites these absolutes.

---

## Separate — audit finding M1 (independent of these fixes)

The headline results span **two library stacks** and two groups are **mixed within**:

- `runs/track_h/`: **129** rundirs on torch 2.11.0 / transformers 5.5.4 vs **108** on
  torch 2.12.0 / transformers 5.9.0.
- `runs/probe_signal/`: **90** on torch 2.11.0 vs **25** on torch 2.12.0.

Before pooling either group into one table, either (a) confirm no cross-condition
comparison crosses the stack boundary, or (b) re-run the minority (2.11.0) cells on the
canonical stack (`requirements.txt` = torch 2.12.0 / transformers 5.9.0). The P0/P1
re-runs above will naturally land everything on one stack if done on the canonical env.

## Note — paused β0.3 campaign

`runs/rerun_beta03/RESUME_AFTER_MIGRATION.md` (7B + phase2 cells, PAUSED) runs through
`phase2_matrix.py`, so it is **also H1-affected** — resume it on the fixed code so its
cells match the P0 canonical set.

---

## After re-running

1. Regenerate derived analysis per group (`make_analysis.py`), fidelity/target-stability
   tables, and `experiments/aggregate_runs.py` GPU-hour rollup.
2. Re-run the paper number/claim audit (`/paper-claim-audit`) — absolute φ / AUROC /
   ROUGE / error bars in `paper/` must be refreshed from the new rundirs.
3. Confirm `git_dirty` is clean at run time (commit code before the campaign) so the new
   rundirs' `meta.json` `git_sha` + `git_diff_stat` certify the exact code.
