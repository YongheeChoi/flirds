---
type: thread
title: Robustness of data values to stochastic training
created: 2026-05-05
updated: 2026-05-05
sources: [data-banzhaf, in-run-data-shapley, asymmetric-data-shapley, distributionally-robust-data-valuation, feddqc]
tags: [robustness, sgd-noise, ranking-stability]
---

# Robustness of data values to stochastic training

## The question

When a model is trained with SGD (or any stochastic optimizer), the utility $U(S)$ of a subset $S$ is itself a random variable. The randomness comes from initialization, batch order, dropout. Data values computed from noisy $U$ are themselves noisy. **How much do data-value rankings change between runs of the same valuation algorithm?** And what to do about it?

This isn't a theoretical curiosity. Almost every downstream use of data values depends on the *ranking* (top-$k$ for cleaning, ordering for reweighting, comparison for pricing). If different runs give different rankings, the application breaks.

## Three conceptual responses

The wiki has multiple sources that respond to training stochasticity differently. They're complementary, not competing.

### Response 1 — Pick the noise-tolerant semivalue: [[sources/data-banzhaf|Data Banzhaf]]

Defines the **safety margin**: the largest $\ell_\infty$ perturbation of $U$ that preserves every pairwise data-value ordering. Proves:

$$\text{margin}(\text{Banzhaf}) \gg \text{margin}(\text{Shapley}) > \text{margin}(\text{LOO})$$

with the Banzhaf vs. Shapley gap exponential in dataset size. Empirical: Banzhaf rankings are much more stable across runs.

**Position**: keep the algorithm-level definition, choose the semivalue with the best noise tolerance. Accept the loss of efficiency.

### Response 2 — Change the definition: [[sources/in-run-data-shapley|In-Run Shapley]] / [[sources/asymmetric-data-shapley|ADS]]

Argues the right object is not the algorithm-averaged Shapley but the **model-specific** Shapley conditional on the actual trained model. The training randomness becomes part of the trajectory rather than something to average over.

[[sources/asymmetric-data-shapley|ADS]] makes this axiomatic: the state-conditioned marginal contribution is the right object. [[sources/in-run-data-shapley|In-Run Shapley]] derives it via gradient calculus per training step.

**Position**: if you commit to a particular trained model, you don't need to worry about ranking instability across hypothetical alternative training runs.

### Response 3 — Robustify the utility itself: [[sources/distributionally-robust-data-valuation|DRDV]]

Replaces the validation-set-dependent utility with a **distributionally robust** one — worst-case loss over a Wasserstein ball of distributions. The Shapley axioms apply unchanged; what shifts is the underlying $U$.

**Position**: the noise isn't (just) in training — it's in the validation distribution choice and dataset sampling. Make the utility itself robust before computing data values. Orthogonal to the axiom debate.

## The three-axis view

| Source | Axiom | Utility | Definition |
|---|---|---|---|
| [[sources/ghorbani-zou-data-shapley\|Data Shapley]] | Shapley (all axioms) | validation accuracy | algorithm-level |
| [[sources/data-banzhaf\|Banzhaf]] | drop efficiency | validation accuracy | algorithm-level |
| [[sources/asymmetric-data-shapley\|ADS]] | drop symmetry | validation accuracy | model-level (state-conditioned) |
| [[sources/in-run-data-shapley\|In-Run Shapley]] | Shapley (per-step) | validation loss change per step | model-level (trajectory) |
| [[sources/distributionally-robust-data-valuation\|DRDV]] | Shapley | DRGE | algorithm-level |

This shows the *axes are independent* — each source picks one axis to modify. A method that combines all three (drop symmetry + DRGE utility + in-run definition) hasn't been published.

## Why [[sources/feddqc|FedDQC]] is relevant here

[[sources/feddqc|FedDQC]] reports [[sources/datainf|DataInf]] failing on real-world heterogeneous federated data (Fed-WildChat). This isn't *exactly* the same as SGD noise — it's noise across *clients* — but it's the same type of failure mode: gradient-based attribution becomes brittle when the gradient distribution itself is heterogeneous. Suggests the safety-margin / robustness story extends beyond SGD to client heterogeneity.

## Open questions

- **Head-to-head ranking comparison**: do Banzhaf rankings (algorithm-level) and In-Run Shapley rankings (model-level) agree on bad-data detection on the same dataset and model? No source in the wiki has run this.
- **Safety margin for gradient-based methods**: the Banzhaf framework's safety-margin argument assumes $\ell_\infty$ noise. Influence functions / TRAK / LoGra are noisy too — what's the analogous robustness theory?
- **Combinable robustness**: combining the noise-robust semivalue (Banzhaf) with the robust utility (DRGE) should give the most stable rankings. Has anyone tried?
- **Ranking-stability metrics beyond safety margin**: Kendall's tau across runs, top-$k$ overlap, etc. The empirical evaluation isn't standardized.
- **For LLM pretraining** (one training run, no realistic alternatives): is robustness even the right desideratum, or should we commit to the in-run / model-level framing?

## Sources to ingest

- Variance-reduction techniques for Shapley estimation under noisy $U$.
- Distributionally robust attribution beyond the Wasserstein ball (e.g., $f$-divergence balls).
- Empirical comparisons of Banzhaf vs. influence-function methods at LLM scale.
