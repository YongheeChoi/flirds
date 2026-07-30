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

The CNN track escapes the floor. Its round displacement is two orders of
magnitude larger than the LLM's — 5 local epochs on all parameters, versus 10
steps on LoRA factors — so the remainder clears the fp32 floor by three to five
orders of magnitude there. Whether that makes the *order* measurable is a
separate question, answered below: it does not.

## Result (21 result cells + 3 controls, 2026-07-31)

CNN = Table 1's grid (CIFAR-10/Dir(1), N=10, full, R=10, 4 conditions x 3 seeds,
49.7 min local). LLM = the five-domain counterpart (N=5, full, R=10, 3 conditions
x 3 seeds, 31.5 min/cell on a B200). Plus a 3-seed smooth-activation control.
0 failures. Regenerate everything with `python make_analysis.py`.

### What is solid — none of it depends on a slope fit or on the floor

| | LLM (silo5) | CNN (Table 1) |
|---|---|---|
| remainder / attributed quantity, 1st order | 0.097 % | 12.9 % |
| **remainder / attributed quantity, 2nd order** | **0.032 %** | **1.59 %** |
| **curvature term shrinks the remainder by** | **3.0x** | **8.1x** |
| 2nd order at least as accurate, per coalition | 82.2 % | 97.7 % |

1. **C.5 reproduces exactly**, on a different stack and GPU: resid2 = 0.031 % of
   the attributed quantity (published 0.03 %), frac 0.831 (published 0.801),
   coalition slope2 1.35 (published 1.5-1.6), closed-form vs Shapley 6.7e-11
   (published 6.3e-10). The harness is validated end to end, which is what makes
   the CNN numbers from the same code trustworthy.
2. **C.5 extends from 1 condition to 3**, and answer-swap and zero-update behave
   like clean — the floor limitation is structural to the LLM track, not an
   artifact of the clean condition.
3. **The curvature term earns its cost**, independently of the fidelity tables.
4. **gradient noise explains a Table 1 outlier.** It is the one condition whose
   first-order remainder exceeds 100 % of the attributed quantity (277 %, at
   ||D_r|| = 32.3, fifty to eighty times every other condition), and the one where
   Table 1's Flirds-1st inverts to rho = -0.184. The curvature term pulls it to
   71 % and Flirds holds at 0.883.

### What is NOT established — the order, on either track

We could not measure the remainder's order anywhere, and the reason is worth
recording so nobody re-runs this expecting a different answer.

**Two estimators, neither of which passes its own control.** The predicted orders
are 2 for the first-order remainder and 3 for the second.

| estimator | slope1 (pred 2) | slope2 (pred 3) | why it fails |
|---|---|---|---|
| coalition spread | LLM 1.82, CNN 1.19 | LLM 1.36, CNN 3.34 | varies ||D_S|| by changing *which* clients are in S, so norm and direction are confounded |
| scale sweep | LLM n/a, CNN 1.90-2.03 | LLM n/a, CNN 1.80-1.97 | direction is fixed, but the ladder is contaminated at both ends (below) |

The sweep's slope1 = 2 is a harness sanity check, not a result: resid1(t) =
1/2 t^2 <D, H D> + O(t^3) is forced to scale as t^2 for any twice-differentiable
loss, so hitting 2 only confirms the Hessian term is nonzero.

**The sweep is contaminated at both ends.** Per-round local slopes on the smooth
control, seed 0 round 7:

| t | resid2 | local slope |
|---|---|---|
| 1 -> 0.5 | 2.46e-4 | 2.94 |
| 0.5 -> 0.25 | 2.00e-5 | 3.62 |
| 0.25 -> 0.125 | 8.37e-6 | 1.25 |
| 0.125 -> 0.0625 | 1.32e-5 | **-0.66** |
| 0.0625 -> 0.031 | 2.27e-5 | **-0.78** |

resid2 stops falling and starts rising below t = 0.125 — cancellation noise in
u = l(w+tD) - l(w). At the top, t = 1 is the realized displacement, already
outside the Taylor radius. A least-squares fit over the whole ladder averages a
non-asymptotic region against a noise-dominated one and returns a number with no
meaning. Fitting only the middle gives ~3 on some rounds and ~1.3 on others.

**The evaluation floor is worse than ulp(base).** `ulp_base` is an analytic
estimate — one ulp of the round's base loss — not a measurement. Empirically, the
level at which resid2 stops decreasing is **26-61x** ulp on CNN (the validation
loss is a mean over 2,000 examples and carries accumulation error), and 540x under
gradient noise. Against that empirical floor the CNN margin is **62-764x**, not
the 6.4e3-7.4e5x that ulp implies. On the LLM the coalition data puts resid2 at
**1.0-3.6x ulp** across the whole coalition range — the smallest coalitions sit
exactly at the floor, so C.5's second-order residual is partly floor-contaminated.
That does not weaken C.5's claim: a remainder at the noise floor is an *upper
bound*, which is what "bound the magnitude" needs. State it as "at most".

**The smooth-activation control shows no effect.** A pilot (N=5, R=3, seed 0)
suggested GELU + avg-pool recovered order 3 while ReLU + max-pool did not. At
Table 1 scale with 3 seeds it does not replicate:

| model (clean, 3 seeds) | ||dW|| | slope1 coal | slope2 coal | slope1 sweep | slope2 sweep |
|---|---|---|---|---|---|
| ReLU + max-pool (paper) | 0.604 | 1.18 | 3.33 | 2.00 | 1.93 |
| smooth (GELU + avg-pool) | 0.586 | 0.81 | 3.36 | 2.03 | 2.07 |

The two are indistinguishable. So A.4's exclusion of the CNN track remains a
mathematical caveat; this measurement does not supply empirical backing for it,
and the earlier "ReLU non-smoothness causes a sub-cubic rate" reading is retracted.

### Consequences for the paper

- Report **magnitude only**, which is what C.5 already does. Do not report any
  measured order, and do not put the scale sweep in the paper — it was added to
  improve on C.5's narrow coalition spread and it did not work. It stays in the
  code as a diagnostic.
- Main Section 4's "measure the remainder itself" should become "bound its
  magnitude", scoped to say the order is not resolvable at fp32 on either track.
- The strongest available sentence is the curvature benefit: retaining the
  second-order term shrinks the remainder 3x on the LLM track and 8x on the CNN
  track, leaving it at most 0.03 % and 1.6 % of the quantity being attributed.

Cost: CNN 248.6 s/cell (32 % trajectory), LLM 1,890 s/cell (39 % trajectory).
Rundirs 6.1 MB (CNN) + LLM.

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
