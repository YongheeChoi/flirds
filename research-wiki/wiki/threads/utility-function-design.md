---
type: thread
title: Utility-function design for data valuation
created: 2026-05-05
updated: 2026-05-22
sources: [distributionally-robust-data-valuation, ghorbani-zou-data-shapley, data-banzhaf, in-run-data-shapley, asymmetric-data-shapley, less, mates, dsdm]
tags: [utility, validation-set, attribution, design-axis]
---

# Utility-function design

## The question

Most data-valuation methods pick a **utility function** $U(S)$ — usually validation accuracy of a model trained on $S$ — and then compute Shapley / Banzhaf / influence over it. But the utility *itself* is a design choice with significant downstream consequences. This thread tracks where utility-function design becomes the leverage point.

## Three classes of utility

### 1. Validation-set-dependent (default)

$U(S) = \text{Performance}(\mathcal{A}(S); D_{\text{val}})$ — the standard. Used by the bulk of the literature including [[sources/ghorbani-zou-data-shapley|Data Shapley]], [[sources/data-banzhaf|Banzhaf]], most federated-Shapley methods.

Problem: values shift if the validation set changes. The seller can't guarantee a price; the buyer can't guarantee a ranking.

### 2. Distributionally robust

$U(S) = -\sup_{Q \in \mathcal{B}_\rho} \mathbb{E}_Q[\ell(\mathcal{A}(S))]$ — worst-case loss over a Wasserstein ball.

[[sources/distributionally-robust-data-valuation|DRDV]] introduces this. Eliminates validation-set dependence; values are stable under reasonable distribution shifts. Has a model-deviation proxy in RKHS / NTK that's tractable.

### 3. Per-step / trajectory-anchored

$U^{(t)}(S)$ defined per gradient step; total value $= \sum_t U^{(t)}$.

[[sources/in-run-data-shapley|In-Run Data Shapley]] takes this route — utility is the per-step Taylor-expanded change, accumulated. Eliminates the choice of "which final validation set" because attribution is computed *along the trajectory*. [[sources/ripple-shapley|Ripple Shapley]] extends this to federated trajectories.

[[sources/asymmetric-data-shapley|ADS]]'s state-conditioned marginal is in the same spirit but framed axiomatically rather than via gradient calculus.

## Other utility-design choices in the literature

- **Validation-free** (DRDV-style) vs. **prototype-based** ([[sources/space-participant-amalgamation|SPACE]]) — both eliminate held-out validation sets, but in different ways.
- **Single-distribution accuracy** vs. **multi-task / personalized utility** — relevant for [[sources/ipfl-model-market|iPFL]] where each buyer wants a different model.
- **Inference-loss-based** ([[sources/feddqc|FedDQC]]'s IRA) — measures alignment, not accuracy.
- **Few-shot validation** ([[sources/less|LESS]]) — utility defined against a *handful* of target-capability examples rather than a held-out distribution. Cheaper to assemble and aligned with operator intent; introduces selection variance in low-shot regimes.
- **Task-natural reference set** ([[sources/mates|MATES]]) — LAMBADA (autoregressive-aligned) as $\mathcal{D}_r$ rather than an i.i.d. validation slice. Picks a pre-existing task whose objective aligns with the training objective rather than constructing a new "validation set." Empirically MATES is robust to the choice (LAMBADA / ARC-E / FLAN all yield positive gains).
- **Explicit target-task mix** ([[sources/dsdm|DsDm]]) — utility = loss on a chosen mix (SQuAD + LAMBADA + Jeopardy as a representative cross-section of LM problem categories). The framework explicitly accepts that *target task choice* shifts model behavior — a feature, not a bug.

## The three-axis decomposition

```
data value = (axiom modification) ∘ (utility design) ∘ (definition target)

axiom mod:     Shapley | Banzhaf | LOO | Asymmetric | Beta-Shapley | CS-Shapley
utility:       validation-acc | DRGE | per-step | prototype | inference-loss-diff
definition:    algorithm-level | model-level | trajectory-level
```

Most papers move on one axis; the wiki should articulate methods by which axis they touch. See [[threads/symmetry-and-asymmetry-axioms]] for the axiom axis.

## Open questions

- **Combinations**: can [[sources/distributionally-robust-data-valuation|DRGE]] utility be combined with [[sources/in-run-data-shapley|In-Run Shapley]]'s per-step accumulation? With [[sources/asymmetric-data-shapley|ADS]]'s state-conditioned marginal? Both seem natural; neither has been done.
- **Wasserstein radius selection**: DRDV's $\rho$ is a hyperparameter. Practically, how do you choose it?
- **NN extension of DRGE**: the RKHS theory is exact; the NTK extension is heuristic. Is the gap empirically small for typical fine-tuning settings?
- **Application-specific utilities**: copyright attribution would benefit from utilities that distinguish memorization from generalization (cf. [[sources/in-run-data-shapley|In-Run Shapley]]'s findings on paraphrased validation data). Has anyone formalized this?
