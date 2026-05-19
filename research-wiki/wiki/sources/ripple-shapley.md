---
type: source
title: "Ripple Shapley: Data Influence Attribution in One Federated Training Run"
created: 2026-05-05
updated: 2026-05-05
topic: flirds
tags: [federated-learning, shapley, in-run, influence-propagation, sample-level, data-pricing]
---

# Ripple Shapley

## Citation

Dewen Zeng, Wenlong Tian, Haozhao Wang, Jianfeng Lu, Weijun Xiao, Zhiyong Xu (USC China / HUST / WUST / VCU / Suffolk). *Ripple Shapley: Data Influence Attribution in One Federated Training Run*. AAAI 2026.

Raw: `raw/papers/flirds/40034-Article Text-44125-1-2-20260314.pdf`

## TL;DR

A **single-run, sample-level** federated data attribution method. Decomposes each sample's contribution into an immediate **drop term** (per-sample marginal utility within its round) and a recursive **ripple term** that propagates the influence across subsequent rounds via a Jacobian chain over global updates. A low-rank spectral approximation of the Jacobian chain makes it tractable. Preserves Shapley axioms; ~62× speedup over prior FL-Shapley methods. Demonstrated on real-time data pricing.

## Problem

Federated Shapley to date sits in two camps:

- **Multi-round retraining** (Wang et al. 2020, [[sources/gtg-shapley|GTG-Shapley]]): faithful but expensive.
- **Per-round Shapley aggregation** ([[sources/shapleyfl|ShapleyFL]]): efficient but doesn't track cross-round influence propagation explicitly.

What's missing: a **sample-level** notion (not just client-level) that captures how a sample's influence *cascades* through subsequent rounds via the global model updates it triggered. This is the FL analogue of [[sources/in-run-data-shapley|In-Run Data Shapley]].

## Method

### Drop + Ripple decomposition

For sample $z$ in round $t$, contribution is:

$$\phi_z = \underbrace{\Delta_t(z)}_{\text{drop term}} + \underbrace{\sum_{s > t} J_s J_{s-1} \cdots J_{t+1} \Delta_t(z)}_{\text{ripple term}}$$

- **Drop term**: instantaneous marginal utility of $z$ in its round (a Shapley-style marginal).
- **Ripple term**: propagates that marginal forward through the Jacobians $J_s = \partial \theta^{(s+1)} / \partial \theta^{(s)}$ of consecutive global updates. Each Jacobian captures how a perturbation at round $t$ shifts the model at round $s > t$.

### Low-rank Jacobian chain

The Jacobians live in parameter-dim × parameter-dim — too big. Project onto a shared low-rank subspace; the chain $J_s J_{s-1} \cdots$ then composes cheaply within the subspace.

### Single-run, sample-level

The whole computation runs on the **logged trajectory** of one FL run — no counterfactual retrainings, no client-level coalition enumeration. The output is a per-sample Shapley value.

## Key results

- **62× speedup** over prior FL-Shapley methods at comparable accuracy.
- Preserves the four Shapley axioms (efficiency, symmetry, linearity, null player) at the sample level under their decomposition.
- Application to **real-time data pricing**: the per-sample value is queried at run time as new samples arrive, supporting market dynamics.

## Connections

- The federated, sample-level analogue of [[sources/in-run-data-shapley|In-Run Data Shapley]]. Both use the realized training trajectory; both decompose into per-step contributions; both achieve sample-level Shapley at near-training cost. The Ripple Shapley paper essentially extends the In-Run idea to the multi-update FL trajectory.
- Compared to other federated-Shapley methods:
  - vs. [[sources/gtg-shapley|GTG-Shapley]] (sub-model reconstruction): Ripple is sample-level, GTG is client-level.
  - vs. [[sources/shapleyfl|ShapleyFL]] (surrogate per-round Shapley): Ripple makes propagation explicit, ShapleyFL aggregates per-round values.
  - vs. [[sources/space-participant-amalgamation|SPACE]] (single-round via distillation): both are single-run, but SPACE is client-level via a different mechanism.
- Belongs to threads [[threads/federated-and-decentralized-attribution]] and [[threads/retraining-vs-in-run-attribution]].
- Concept page: [[concepts/ripple-shapley]], [[concepts/jacobian-chain-influence-propagation]] (created).

## Notes / open questions

- Real-time data pricing as an application is compelling; how does the ripple-term's recursion handle late-arriving training data (i.e. samples in round $t$ when only rounds $1, ..., t-1$ have been logged)?
- The low-rank Jacobian-subspace assumption is a strong one. Empirically how is the rank chosen, and does it vary across rounds?
- Comparison to [[sources/asymmetric-data-shapley|ADS]]: both anchor evaluation to the realized trajectory. ADS averages within-round permutations; Ripple decomposes recursively along the trajectory. Are they equivalent under some limit, or genuinely different objects?
- Sample-level Shapley in FL is a long-running goal; this is the cleanest single-run answer to date. Stress-test against partial participation, dropout, and adversarial rounds.

> TODO: verify exact formula for the ripple term's Jacobian chain — there may be a simpler form under FedAvg specifically.
