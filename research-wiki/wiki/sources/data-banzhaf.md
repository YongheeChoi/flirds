---
type: source
title: "Data Banzhaf: A Robust Data Valuation Framework for Machine Learning"
created: 2026-05-05
updated: 2026-05-05
tags: [semivalue, banzhaf, robustness, sgd-noise, msr]
---

# Data Banzhaf

## Citation

Jiachen T. Wang & Ruoxi Jia. *Data Banzhaf: A Robust Data Valuation Framework for Machine Learning*. arXiv:2205.15466 (v7).

Raw: `raw/papers/flirds/Data Banzhaf_ A Robust Data Valuation Framework for Machine Learning.md`

## TL;DR

Stochastic training (SGD) makes Shapley and LOO data-value rankings inconsistent across runs. The **Banzhaf value** — a classical semivalue from cooperative game theory — has the largest *safety margin* (tolerance to performance-score noise) of any semivalue, exponentially larger than Shapley's. It also admits a uniquely efficient *Maximum Sample Reuse* estimator, and outperforms Shapley/LOO/Beta-Shapley on bad-data detection and reweighting.

## Problem

Existing data-value notions assume the utility function $U(S)$ — usually validation accuracy of a model trained on $S$ — is deterministic. With SGD it isn't, and the noise is large enough that **different runs of the same valuation algorithm produce different rankings**. This breaks downstream applications that need a stable ordering (data pricing, low-quality data detection).

## Method

### Safety margin

Define the **safety margin** of a value notion as the largest $\ell_\infty$ perturbation of $U$ that preserves *every pairwise* ordering of data values. Larger margin ⇒ more robust to SGD noise.

The paper proves:

$$\text{margin}(\text{Banzhaf}) \gg \text{margin}(\text{Shapley}) > \text{margin}(\text{LOO})$$

with the Banzhaf vs. Shapley gap exponential in dataset size $n$.

### Banzhaf value

A semivalue (see [[concepts/semivalue]]) with weight schedule that gives **equal weight to every subset** $S \subseteq D \setminus \{i\}$, in contrast to Shapley which gives equal weight to every *size* of subset:

$$\phi^{\text{Banzhaf}}_i \;=\; \frac{1}{2^{n-1}} \sum_{S \subseteq D \setminus \{i\}} \big( U(S \cup \{i\}) - U(S) \big)$$

### MSR estimator

Sampling subsets $S$ uniformly and reusing each sample for *every* point's marginal contribution gives an unbiased Banzhaf estimator with $\ell_\infty$ error bounds at logarithmic sample complexity. The paper shows this **Maximum Sample Reuse (MSR)** trick is unique to Banzhaf among semivalues — for Shapley, MSR is biased.

## Key results

- **Theoretical**: Banzhaf is the unique semivalue admitting an unbiased MSR estimator. Sample complexity is close to a derived lower bound.
- **Empirical**: On bad-data detection and weighted-sample learning under SGD, Banzhaf > Beta-Shapley > Shapley > LOO. Ranking stability across runs is substantially higher.
- The MSR estimator is itself robust to noise in $U$.

## Connections

- Generalizes / contrasts with [[concepts/shapley-value]] and [[concepts/leave-one-out]].
- Sits inside the broader [[concepts/semivalue]] family.
- Concept page: [[concepts/banzhaf-value]].
- Cross-cutting thread: [[threads/robustness-to-stochastic-training]] — this paper essentially defines that thread.

## Notes / open questions

- Banzhaf gives up the *efficiency* axiom (sum of values ≠ total utility). For data-market applications that need budget-balance, this matters — see [[sources/asymmetric-data-shapley]] when ingested for an alternative angle.
- How do MSR-Banzhaf rankings compare to gradient-based methods (DataInf, In-Run Shapley) in head-to-head? The paper doesn't test against in-run methods; that's an open calibration question worth its own thread.
- The safety margin argument assumes $\ell_\infty$ noise. SGD noise has structure; is there an even better-suited semivalue under that structure?
