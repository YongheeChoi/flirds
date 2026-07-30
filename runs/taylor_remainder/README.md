# Taylor remainder, measured in main Table 1's setting (both tracks)

Direct measurement of the per-round Taylor remainder — the quantity Appendix C.5
reports — in the setting of **main Table 1** (`tab:retrain-fidelity`), on both the
CNN and the LLM track.

## Why this exists

Two claims in the paper currently rest on a measurement that cannot support them.

1. **The order claim is not confirmed by C.5.** Main §4 states the remainder is
   $O(\|\Delta_S^r\|^3)$ and points at Appendix C.5 to "measure the remainder
   itself". C.5 then says the opposite: its second-order residual sits only
   **2.1×** above the fp32 evaluation floor (1 ulp of the base loss,
   $2.38\times10^{-7}$), so the log–log slope fits at **1.5–1.6** rather than the
   predicted 3. C.5 correctly limits itself to bounding the *magnitude*, not the
   *order*.

2. **Appendix A.4 excludes the CNN track from the bound** — it is conditional on a
   $C^3$ assumption that ReLU + max-pool violates, so "the fidelity claim is
   supported only by the empirical results". Nothing measures the remainder there,
   even though Table 1, Table 2, and Figure 2(a) are all CNN.

The CNN track fixes the first problem and speaks to the second. Its round
displacement is two orders of magnitude larger than the LLM's — 5 local epochs on
all parameters, versus 10 steps on LoRA factors — so the remainder clears the same
fp32 floor by **four to five orders of magnitude** and the order becomes
measurable. A pilot (N=5, R=3, seed 0) gave:

| stage | ‖Δ_r‖ | resid2 median | resid2 / fp32 floor | slope₂ |
|---|---|---|---|---|
| LLM, C.5 as published (3 seeds) | 0.0041 | 5.1e-7 | **2.1×** | 1.5–1.6 |
| CIFAR-10 / FedSVCNN (ReLU) | 0.45–0.60 | 6.5e-3 | **32,605×** | 1.98–2.75 |
| MNIST / LeNet5 (ReLU) | 0.53–1.13 | 1.4e-2 | **60,303×** | 2.26–2.64 |
| CIFAR-10, smooth control (GELU + avg-pool, ‖Δ‖ matched) | 0.41–0.59 | 6.9e-3 | 28,923× | **3.26 / 3.36** |

The first-order residual fits at **2.00–2.12** in every one of those runs, against
a predicted 2 — an internal check that the measurement itself is sound. The
second-order residual is sub-cubic on the ReLU models and recovers the predicted
3 only in the smooth control at matched displacement. That is exactly what A.4's
$C^3$ caveat anticipates, which turns the exclusion from a bare caveat into a
measured distinction.

Numbers above are a **pilot, not a result**: single seed, N=5, R=3, and the ReLU
slope is band-dependent there (1.98 ↔ 2.75 depending on the ‖Δ‖ range probed), so
it should not be quoted as a single exponent. The pilot's one lasting contribution
is the **smooth-activation control**, which the Table 1 grid does not include. For
the real numbers see *CNN result* immediately below, which supersedes the two ReLU
rows here.

## CNN result (12/12 cells, 2026-07-30)

Table 1's grid is complete — CIFAR-10 / Dir(1), N=10, full, R=10, 4 conditions ×
3 seeds, 0 failures, 49.7 min total on one RTX 4070 SUPER. Regenerate every number
below with `python make_analysis.py`; nothing here is hand-copied.

| condition | mean ‖Δ_r‖ | resid2 median | **resid2 / fp32 floor** | slope₁ sweep (pred 2) | slope₂ sweep (pred 3) | slope₁ coal (pred 2) | slope₂ coal (pred 3) |
|---|---|---|---|---|---|---|---|
| clean | 0.604 | 1.61e-3 | **1.23e4** | **2.04 ± 0.04** | 2.31 ± 0.05 | 1.18 ± 0.01 | 3.33 ± 0.30 |
| zero-update | 0.387 | 9.09e-4 | **6.35e3** | **2.09 ± 0.11** | 2.36 ± 0.06 | 1.20 ± 0.05 | 2.79 ± 0.31 |
| label-flip | 0.427 | 2.50e-3 | **1.74e4** | **2.03 ± 0.07** | 2.26 ± 0.08 | 1.11 ± 0.10 | 1.91 ± 0.12 |
| gradient noise | **32.3** | 9.85e-2 | 6.55e5 | 1.06 ± 0.13 | 0.47 ± 0.17 | 0.48 ± 0.02 | 0.66 ± 0.03 |

Three things this establishes, and two traps.

**1. The floor is escaped, by 6.4e3–7.4e5× across every cell.** C.5's LLM
measurement sits at 2.1×. This is the robust result: on the CNN track the
remainder is measurable, so the order question is answerable there.

**2. Use the scale sweep, not the coalition spread.** The sweep recovers the
first-order remainder's predicted order 2 to within 0.09 in all three
well-behaved conditions, with a cross-seed std ≤ 0.11 — it passes its own control.
The coalition-spread estimator returns **1.11–1.20 for that same known-order-2
quantity**, so it fails the control and its second-order values (1.91–3.33, std up
to 0.31) must not be quoted. This is why the sweep was added; C.5's published
1.5–1.6 slope comes from the coalition estimator alone.

**3. The second-order remainder is sub-cubic: 2.26–2.36, against a predicted 3.**
Tight across conditions and seeds (std ≤ 0.08) and measured far above the floor, so
this is not a resolution artifact. It is what A.4's $C^3$ caveat anticipates for a
ReLU + max-pool network, and the smooth-activation control in the pilot (GELU +
avg-pool at matched ‖Δ‖ → 3.26 / 3.36) points the same way.

**Trap 1 — gradient noise is not order evidence.** Its displacement is
‖Δ_r‖ = 32.3, fifty to eighty times every other condition, because σ=0.1 Gaussian
noise is injected straight into the update. That is far outside any Taylor radius,
and the surrogate has simply broken down: slopes collapse to 0.47–1.06 and
`closed vs Shapley(u2)` inflates ~100× (2.0e-5 vs 4.6e-8–3.0e-7 elsewhere, still
only 1.7e-4 relative to $|u_r(P_r)|$). Exclude this condition from any order claim.
Separately, it *is* a finding worth one sentence: the estimator's Taylor basis
degrades under large-norm update attacks, which bears on why Flirds-1st scores
−0.184 in Table 1's gradient-noise column.

**Trap 2 — do not quote the pooled row.** `remainder_pooled.csv` averages all four
conditions, so gradient noise drags it to slope₂ 2.17 (coal) / 1.85 (sweep). The
per-condition table is the honest view.

Cost: 248.6 s per cell (78.6–86.0 s trajectory + 160.9–172.9 s measurement, 32 %
training), 6.1 MB of rundirs for the whole grid.

## Setting

Table 1 is `tab:retrain-fidelity`: CIFAR-10 / Dirichlet(α=1), **N=10, full
participation, R=10**, four conditions × three seeds, scored against the
retraining-based Shapley.

| | stage | N | participation | R | conditions |
|---|---|---|---|---|---|
| **CNN** = Table 1 itself | CIFAR-10, Dir(1) | 10 | full | 10 | clean / zero-update / gradient noise / label-flip |
| **LLM** = Table 1's counterpart | five-domain non-IID | 5 | full | 10 | clean / answer-swap / zero-update |

Table 1 is a CNN table, so there is no literal LLM row to copy. Its defining
design is: full participation, N small enough for exact $2^N$ enumeration, R=10,
three seeds, non-IID, scored against a *retraining* oracle. On the LLM track
exactly one setting has all of those — the five-domain non-IID setting, which
Appendix C.3 uses as "The LLM Retraining Leg" and which enumerates the classical
value through $2^5$ retrains. It is also C.5's existing stage, so the published
three-seed result is reproduced rather than replaced.

Conditions follow main §5.1's per-track assignment: gradient noise and label-flip
are CNN-only, answer-swap is LLM-only, clean and zero-update are shared. Corrupt
client sets come from the same configs the paper runs use (`MAL_FRAC=0.4` at N=10
→ 4 clients; five-domain `noisy={0}`, `freerider={1}` → 1 of 5 = 20%).

### Optional: a main-text LLM setting

`bash run_table1.sh gsm` measures the same quantity in the **GSM8K main setting**
(N=50, 5/50, R=200). C.5's five-domain stage is never named in the main paper,
whereas GSM8K is the headline LLM setting — and its (b) oracle already enumerates
$2^K = 32$ coalitions per round, so this is the cheapest way to move the
measurement into a setting the body introduces. R=200 makes it ~20× the forwards
of `silo5`, so it subsamples rounds (`TAYLOR_N_MEASURE=20`).

## What is measured

The fixed-weight game, identical to the (b) oracle's (`in_run_sv`):

```
u_r(S)  = l(w_r + D_S) - l(w_r),   D_S = sum_{k in S} a_k,  a_k = p_k^r d_k
u1_r(S) = <g_r, D_S>                                     (1st-order surrogate)
u2_r(S) = u1_r(S) + 1/2 sum_{i,j in S} <a_i, H_r a_j>    (2nd-order surrogate)
resid1  = |u - u1|      resid2 = |u - u2|
```

Plus two things C.5 does not do:

- **fp32 floor per round**, `ulp(base_loss)`, reported alongside every residual, so
  "is this measurable?" is answerable per cell instead of only in prose.
- **A matched-norm scale sweep.** C.5 fits its slope over the coalition spread
  alone, which spans barely a factor of ~K in ‖Δ_S‖ — too narrow to separate order
  2 from order 3. The sweep rescales the realized direction to fixed *absolute*
  targets (default 1.0 → 0.03125, a 32× range), and because the targets are
  absolute rather than relative, runs with different displacements become directly
  comparable at equal ‖Δ‖. Slopes are also reported restricted to points clear of
  10× the floor, the only ones carrying order information.

Theorem 1 is re-verified numerically in every cell, as `closed form vs
Shapley(u2)` — the closed form of Eq. (6) against the exact Shapley value of the
surrogate game obtained by enumerating all $2^K$ coalitions.

That gap grows with the cohort size, because it accumulates fp32 error over the
$2^K$ coalition sum: 2.4e-11 at K=6, 1.16e-7 at K=10. It is a floating-point
artifact, not a discrepancy in the theorem — read it relative to the quantity
being attributed, where it stays ~1e-6 of the mean per-round loss decrease
($|u_r(P_r)| \approx 0.12$ at K=10). Report it as a ratio rather than an absolute
if it appears in the paper, since the absolute value is K-dependent and invites a
false comparison against C.5's LLM figure of 6.3e-10, which was measured at K=5.

## Cost, and whether training must be rerun

This is a **from-logs** computation — it needs the frozen trajectory
$(w_r, \{\delta_k\})$, never a retrain, and is not the (a)-retrain oracle.

But the trajectory is **not persisted anywhere** in `runs/`: there are no
checkpoints (0 `.pt`/`.safetensors` files across ~2,000 rundirs) and no
`torch.save` in the codebase — `RunLogger` stores φ, metrics, timing, meta only.
So each cell regenerates the trajectory by rerunning FedAvg at the same seed.
That is a deterministic replay, not a new experiment (`seed_everything(seed,
cudnn_deterministic=True)`, with git SHA and env hash recorded in `meta.json`).

Persisting it instead would cost ≈ 11 GB/seed for the CNN main setting
($(1+K) \times 2.16\text{M} \times 4\,\text{B} \times 120$ rounds), which is why
rerunning is the right call.

The two tracks split cost in opposite directions:

| | trajectory | measurement | training share |
|---|---|---|---|
| LLM (C.5 measured, per seed) | 605 s | 1,536 s | **28 %** |
| CNN (pilot, N=5 R=3) | 26 s | 3 s | **89 %** |

LLM validation forwards cost ~1.6 s each, so the $2^K$ enumeration dominates. CNN
forwards are milliseconds, so the 5-local-epoch training dominates.

**Note:** `in_run_shapley_perround` already computes every `u_r(S)` this needs, for
all $2^{|P_r|}$ coalitions, and discards it after forming φ. Any cell run with
`oracle_b: true` is therefore already paying for the true-utility half. Adding a
dump seam there would make the true side free; only the surrogate side (1 gradient
+ K HVPs per round) is genuinely new compute.

## Files

| | |
|---|---|
| `codes/experiments/taylor_core.py` | shared, track-agnostic measurement core |
| `codes/experiments/measure_taylor_cnn.py` | CNN driver; imports `track_c1` and calls its `build()`, so partition / corrupt draw / doses / delta-level threats / val split are bit-identical to Table 1 |
| `codes/experiments/measure_taylor_llm.py` | LLM driver; `REGIME=silo5` (default) or `gsm50k5` |
| `run_table1.sh` | the grid — `smoke` / `cnn` / `llm` / `gsm` / `all` |
| `make_analysis.py` | rundir-only, read-only rollup → `analysis/*.csv` |

Both drivers import `flirds` and modify nothing, following the convention the
original C.5 script set ("flirds 코드는 임포트만 하고 일절 수정하지 않는다").
The core is restored from that script, deleted at `43a5f17`; recover the original
with:

```bash
git show 9dacbad:research-wiki/survey/irds-fl-math-rigor-2026-07/measure_taylor_residual.py
```

## Run

```bash
cd codes
bash ../runs/taylor_remainder/run_table1.sh smoke   # seconds, offline, no persist
bash ../runs/taylor_remainder/run_table1.sh cnn     # Table 1, 12 cells
bash ../runs/taylor_remainder/run_table1.sh llm     # counterpart, 9 cells
python ../runs/taylor_remainder/make_analysis.py
```

Windows note: the local `trl` import needs `PYTHONUTF8=1` under a cp949 locale.

## What this can and cannot establish

**Can.** That the CNN remainder is far above the evaluation floor, so its order is
measurable where C.5's is not; that the first-order residual matches its predicted
order 2 on both tracks; and that the cubic rate appears only under smooth
activations at matched displacement, which is what A.4's $C^3$ condition predicts.

**Cannot.** Confirm order 3 for the paper's own CNN models — they are ReLU, and the
measurement says the rate there is sub-cubic. That is a real finding, not a
measurement failure, but it means the honest main-text sentence is "bounds the
magnitude of the remainder", with the order claim scoped to the LLM track. Nor does
this settle the LLM order: separating order from magnitude there needs evaluation
precision finer than fp32, which no run here provides.

The smooth control is not one of the paper's models. Use it as an attribution
control in an appendix paragraph, not as a headline result.
