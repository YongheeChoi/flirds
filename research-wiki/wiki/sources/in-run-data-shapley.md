---
type: source
title: "Data Shapley in One Training Run"
created: 2026-05-05
updated: 2026-05-05
tags: [shapley, in-run, llm-scale, attribution, copyright]
---

# In-Run Data Shapley

## Citation

Jiachen T. Wang, Prateek Mittal, Dawn Song, Ruoxi Jia. *Data Shapley in One Training Run*. ICML 2024; ICLR 2025 Outstanding Paper Runner-up. arXiv:2406.11011 (v3).

Raw: `raw/papers/flirds/Data Shapley in One Training Run.md`. Code: `https://github.com/Jiachen-T-Wang/GhostSuite` (GhostSuite — "ghost"-based per-sample-gradient implementation; centralized only, not FL).

## TL;DR

Computes the Shapley value of each training point **for the specific model that was actually trained**, by Taylor-expanding each gradient-update step's utility change and accumulating Shapley scores per iteration. Optimized implementation has near-zero overhead vs. standard training. Enables Shapley-based attribution at foundation-model pretraining scale (demonstrated on GPT-2 and Pythia-410M).

## Problem

Classical Data Shapley has two issues at modern scale:

1. **Computational**: requires retraining on many subsets — infeasible for foundation models.
2. **Conceptual**: Shapley over a learning *algorithm* $\mathcal{A}$ averages out the randomness (init, batch order). But practitioners care about the *specific model* they trained, not its expectation. These are very different quantities under stochastic training.

## Method

### Per-step Shapley via Taylor expansion

A single SGD step changes utility by a small amount; first- or second-order Taylor expansion of $U(\theta_{t+1}) - U(\theta_t)$ as a function of the batch's data points yields an analytic Shapley value:

- **First-order**: $\phi^{(t)}_i \propto \langle g^{\text{val}}_t, g_i \rangle$ — gradient dot product.
- **Second-order**: $\phi^{(t)}_i$ involves a gradient–Hessian–gradient product.

Sum over training steps:

$$\phi^{\text{in-run}}_i \;=\; \sum_t \phi^{(t)}_i$$

### Tricks for efficient computation

The naive cost is per-sample gradients, which are expensive. The paper develops:

- Computing $\sum_i \langle g^{\text{val}}, g_i \rangle$ in **one backward pass** without materializing per-sample gradients.
- Computing the Hessian-vector product term in **two backward passes**.
- Avoiding instantiation of full gradient vectors or Hessian matrices.

With sufficient GPU memory, the optimized implementation is "as fast as regular training."

## Key results

Three case studies on GPT-2 / Pythia-410M:

1. **Pretraining data quality**: ~16% of the Pile dataset gets *negative* Shapley value. Removing those points speeds up convergence and improves final performance — even though Pile is already heavily curated.
2. **Stage-dependent contribution**: General-corpus contribution dominates early in training; domain-specific corpora dominate late. Cannot be seen with retraining-based Shapley.
3. **Copyright beyond memorization**: Even when the validation point is a topical paraphrase of the training point (not verbatim), the training point still receives substantial Shapley credit. Implies attribution/royalties might apply more broadly than current "verbatim copy" copyright thinking suggests.

## Connections

- Sits in the [[concepts/shapley-value]] family but redefines the quantity being computed (model-specific, not algorithm-average).
- Direct contrast with retraining-based Shapley — see [[threads/retraining-vs-in-run-attribution]].
- Closely related to gradient-based methods like influence functions and TracIn (when ingested).
- Enables attribution at the scale targeted by [[threads/attribution-at-llm-scale]].

## Notes / open questions

- The model-specific framing changes the desiderata. Does it preserve the Shapley axioms for the per-step utility, and what does that mean for the accumulated value?
- How does in-run Shapley correlate with influence-function-based attribution (DataInf, EKFAC, LoGra)? The paper doesn't run that comparison.
- The negative-value finding for Pile is striking. Is the threshold "negative value" the right curation rule, or should one keep some negatively-valued points for regularization?
- Stage-dependence suggests *time-aware* attribution. What's the right Shapley analogue if we want to value contributions to a particular *capability* the model acquires mid-training?
- **From Flirds (2026-06-03)**: IRDS's own Appx E.2.2 reports the 2nd-order term gives no notable accuracy gain over 1st-order — but that is a *centralized per-SGD-step* regime (tiny $\eta$, 1st-order $O(\eta^2)$ already near-exact). In FL **per-round** (multi-step, larger $\Delta W$) the 2nd-order is non-trivial: Flirds confirms it *helps* under plain SGD (estimator-vs-in-run-oracle 3-seed Spearman 0.96 vs 0.92) but only there — momentum's velocity tail or over-large steps break it. IRDS uses the **true Hessian** ($\nabla^2\ell$ via the ghost-HVP), not GGN/Fisher; Flirds tested a GGN variant and it was *worse*. The 2nd-order's stated role (down-weighting near-duplicates / interaction) maps to FL **client redundancy**. See [[flirds#Phase 0.5 findings — 2nd-order term & dual oracle (2026-06-03, CNN)]].
