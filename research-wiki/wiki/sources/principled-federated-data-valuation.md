---
type: source
title: "A Principled Approach to Data Valuation for Federated Learning (FedSV)"
created: 2026-05-19
updated: 2026-05-19
topic: flirds
tags: [federated-learning, shapley, participant-valuation, order-aware, foundational]
---

# A Principled Approach to Data Valuation for Federated Learning (FedSV)

## Citation

Tianhao Wang, Johannes Rausch, Ce Zhang, Ruoxi Jia, Dawn Song. *A Principled Approach to Data Valuation for Federated Learning*. In *Federated Learning — Privacy and Incentive* (Springer LNCS 12500), 2020. arXiv:2009.06192v1, Sep 2020.

Raw: `raw/papers/flirds/2009.06192v1.pdf`

This is the **foundational federated Shapley value paper**. Later literature nicknames its estimator **FedSV**; it is cited as the parent by [[sources/gtg-shapley|GTG-Shapley]], [[sources/shapleyfl|ShapleyFL]], [[sources/comfedsv|ComFedSV]], [[sources/ripple-shapley|Ripple Shapley]]. It is a planned [[flirds|Flirds]] baseline.

## TL;DR

Defines the **federated Shapley value (FedSV)**: a per-round, order-aware Shapley variant that values *clients* by the validation-utility improvement their uploaded updates produce each round, summed across rounds. Computable from updates the server already receives (no extra communication); provably retains group rationality, fairness, additivity. Permutation-sampling and group-testing estimators make per-round computation tractable.

## Problem

Canonical Shapley does not transfer to FL: (1) its combinatorial utility requires retraining/evaluating on every data-source subset — prohibitive when data is siloed and only a cohort uploads per round; (2) its symmetry axiom ignores order, but FL is inherently sequential (decaying LR makes early-round contributions more impactful). A communication-free, order-aware contribution measure was missing.

## Method

In round $t$, cohort $I_t$ uploads local updates. Utility $\nu(\cdot)$ takes an *ordered* sequence of round-cohorts; $I_{1:t-1}$ is history. FedSV of client $i$ in round $t$ is the canonical Shapley value computed *only over the round-$t$ cohort*, splitting that round's utility gain:

$$s_t(i)=\tfrac{1}{|I_t|}\!\!\sum_{S\subseteq I_t\setminus\{i\}}\!\!\binom{|I_t|-1}{|S|}^{-1}\big[\nu(I_{1:t-1}\!+\!(S\cup\{i\}))-\nu(I_{1:t-1}\!+\!S)\big]$$

with $s(i)=\sum_{t=1}^{T}s_t(i)$. Key trick: the sub-model utility $\nu(I_{1:t-1}+S)$ is obtained by **applying the aggregate of subset $S$'s already-uploaded updates on top of the round-start global model and evaluating on a server validation set** — no retraining, no extra client communication. Per-round Shapley is approximated by **permutation sampling** ($O(m\log m)$ utility evals per round) or **group testing** ($O((\log m)^2)$, better for large cohort $m$). Exact cost $O(T m^2)$.

## Key results

- **Theorem 1**: FedSV uniquely satisfies instantaneous group rationality, fairness (symmetry + null player, per round) and additivity; aggregation gives long-term group rationality.
- MNIST/CIFAR-10 (IID & non-IID): reliably ranks noisy-label and backdoor (model-replacement) clients low, beating Federated-LOO especially under non-IID.
- Acknowledged failure modes: early-round contributions inflated (gains shrink near convergence); a **norm-normalized** variant detects bad clients much better but **breaks group rationality and additivity**; rarely-sampled non-IID clients can get negative SV.

## Connections

- [[concepts/federated-shapley]] — this *is* the canonical/defining reference for per-round federated Shapley; the parent of every other method on that page.
- [[concepts/shapley-value]] — directly specializes classical Shapley to the federated, order-aware setting; [[concepts/semivalue]] — a within-cohort Shapley restricted per round.
- [[concepts/federated-learning]] — adds contribution valuation on top of the standard FedAvg loop.
- [[threads/federated-and-decentralized-attribution]] — origin point of the federated-attribution line; [[threads/dataset-vs-data-point-valuation]] — squarely client/dataset-level, round-cohort sub-Shapley as the aggregation mechanism.
- [[sources/gtg-shapley]], [[sources/shapleyfl]], [[sources/comfedsv]], [[sources/ripple-shapley]] — descendants that cut FedSV's per-round sub-model-evaluation cost or fix its partial-participation asymmetry.

## Relevance to Flirds

FedSV and Flirds share the **zero-extra-communication** headline and a round-decomposed Shapley structure, but differ sharply:

- **FedSV is a per-round retrain/replay surrogate, not in-run in Flirds' sense.** It re-evaluates the model under counterfactual subset-aggregations of the round's updates — many extra server-side validation forward passes — rather than reading marginal contribution off the realized trajectory. **Communication-equal, computation-expensive** ($O(Tm^2)$ exact or $O(Tm\log m)$ / $O(T(\log m)^2)$ approximate utility evaluations).
- Flirds instead computes client-level Shapley via a **closed-form 1st+2nd-order Taylor expansion** of per-round validation-loss change from the LoRA $\Delta w_k$ — no subset re-evaluation, no Monte-Carlo, near-free compute, PEFT-native (FedSV is full-model, pre-PEFT-era).
- Carry-over limitation: the **early-round inflation / normalization-vs-axioms tension** is intrinsic to round-summed federated Shapley and will recur in Flirds' round-aggregation choice → flagged on [[threads/dataset-vs-data-point-valuation]].

## Notes / open questions

- "No extra *communication*" ≠ compute-free: each utility eval is a full forward pass of a counterfactually-aggregated model on validation data. When benchmarking against Flirds, frame as *communication-equal but computation-expensive (sampled sub-model re-evaluation)* vs. Flirds' closed-form Taylor.
- The normalized-FedSV variant (better detection, broken axioms) is a useful precedent for Flirds' round-aggregation design ablation.
- Negative SV for rarely-sampled non-IID clients — does Flirds' closed-form Taylor estimator inherit this pathology under partial participation? > TODO: check.
- This page supersedes [[sources/comfedsv|ComFedSV]]'s §V as the wiki's primary FedSV description; resolves the long-standing "Wang et al. 2020 not yet ingested" gap noted across [[threads/federated-and-decentralized-attribution]] and [[overview]].
