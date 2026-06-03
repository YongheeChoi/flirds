---
type: source
title: "On the Volatility of Shapley-Based Contribution Metrics in Federated Learning"
created: 2026-06-03
updated: 2026-06-03
topic: flirds
tags: [federated-shapley, contribution-instability, aggregation-strategy, incentive, cross-silo, ground-truth]
---

# Shapley Volatility in FL (Geimer et al.)

## Citation

Arno Geimer, Beltran Fiz, Radu State (SnT, University of Luxembourg). *On the Volatility of Shapley-Based Contribution Metrics in Federated Learning*. arXiv:2405.08044v4, 26 May 2025 (v1 May 2024). No formal venue stated in the text. (venue verify)

Raw: `raw/papers/flirds/2405.08044_ShapleyVolatilityFL.pdf`

## TL;DR

A large-scale empirical study (>20,000 full FL runs) showing that **per-round (One-Round-Reconstruction) federated Shapley values give unstable reward allocations**: the same federation, same data, same objective, but a *different server-side aggregation strategy* shifts a client's share of the total reward by tens of percent (up to ~50% in 3-client settings). Holds across IID and non-IID Dirichlet splits. First work to study the *stability* (not just accuracy) of FL contribution metrics.

## Problem

FL incentive design relies on Shapley values to allocate rewards fairly. Prior work compared contribution methods to a ground truth but **never asked whether the values are stable** — i.e. whether two aggregation strategies that produce equally good global models also produce the same contribution allocation. If they don't, no participant can agree on a strategy, eroding trust in cross-silo federations.

## Method

- **Contribution measure**: cumulative round-based Shapley via gradient-based **One-Round Reconstruction** (Song et al.) — each round, value clients by the marginal validation-utility of their uploaded update under subset re-aggregation; sum across rounds with an inverse-linear round-weight up to a halting round $R$ (Eq. 2). Contribution reported as a percentage.
- **8 aggregation strategies**: FedAvg, FedAvgM, FedAdagrad/Adam/Yogi, FedMedian, FedTrimAvg, Krum — all sharing the FedAvg global objective $f(w)=\frac1K\sum_k F_k(w)$.
- **Datasets**: CIFAR-10/100, MNIST, FMNIST; a small fixed CNN (no tuning). Dirichlet split with $\alpha\in\{1,10,100\}$ (heavy → near-uniform non-IID).
- **Ground truth**: dataset-**size**-based contribution (non-adversarial, non-heterogeneous classes). Lemma 1 gives a closed-form lower bound $d=\frac{n-1}{n^2\alpha+n}$ (squared-Euclidean distance of an equal payout to a size-based payout) as the "is this method better than equal split?" threshold.
- 70 seeds × 3 local-epoch values $e\in\{2,5,10\}$ × 12 use-cases → >20,000 runs.

## Key results

- **Accuracy is fine**: most strategies beat equal-payout on the size-based ground truth in heavy/slight non-IID; only Krum is a clear outlier. No single strategy wins consistently — the "best" one changes seemingly at random across scenarios.
- **But low aggregate distance ≠ per-client fairness**: under the $L_\infty$ (Chebyshev) worst-client metric (Table III), even the best strategies disagree with the ground truth by >10% per client in heavy non-IID.
- **Core finding — instability across strategies** (Figs. 2–3): switching aggregation strategy alone shifts a client's reward share by up to ~30% (pairwise) and ~50% (extreme 3-client). Average difference is zero (zero-sum: one client's loss is another's gain). The instability distribution is roughly **invariant to $\alpha$** — heterogeneity doesn't cause it; it persists even in near-IID.
- Recommendation: new FL contribution methods should be validated across *multiple* aggregation strategies and for *stability*, not just accuracy.

## Connections

- [[concepts/federated-shapley]] — direct evidence that the per-round federated Shapley construction is *unstable* w.r.t. server aggregation; a caveat for the whole family.
- [[sources/gtg-shapley]] — One-Round-Reconstruction + Monte-Carlo is exactly the estimator family this paper stress-tests for stability.
- [[sources/principled-federated-data-valuation]] — the FedSV per-round/order-aware construction whose stability is implicitly questioned here.
- [[threads/federated-and-decentralized-attribution]] — adds the "approximate FL-Shapley is strategy-volatile" caveat to the federated-attribution line.
- [[threads/robustness-to-stochastic-training]] — instability across strategies is a sibling of instability across seeds/trajectories; both motivate a *fixed-trajectory* ground truth.
- [[threads/data-quality-vs-data-value]] — undermines the assumption that approximate FL-Shapley reliably tracks per-client value.

## Relevance to Flirds

**Primary use: motivation anchor.** This is the cleanest published demonstration that *approximate* FL-Shapley reward allocations are unstable — across aggregation strategies, and (by the $\alpha$-invariance result) not merely a non-IID artifact. It directly justifies a Flirds design choice:

- Flirds does **not** validate against a noisy approximate-Shapley signal. Its ground truth is the **(b) exact in-run Shapley oracle** over a *single fixed FedAvg trajectory* (and the (a) exact-retrain oracle). Geimer et al. show why this matters: One-Round-Reconstruction values move tens of percent under benign configuration changes, so they cannot serve as a stable reference for evaluating an estimator.
- The instability they document is at the level of *which approximation/strategy*; Flirds sidesteps it by computing client-level Shapley in closed form (1st+2nd-order Taylor of the per-round validation-loss change) off the realized trajectory, then checking it against an *exact* in-run SV — no Monte-Carlo, no subset re-aggregation, no strategy-dependent reconstruction.
- Caveat it imports: per-client unfairness can be large even when the *aggregate* distance to ground truth is small. For Flirds this argues for **per-client / rank-level** evaluation (Spearman, AUROC) against the oracle, not just an aggregate error number — which matches the existing Phase 0.5 gate design.

## Notes / open questions

- Ground truth here is **dataset size**, deliberately chosen for a non-adversarial, class-homogeneous setting. That is a *weaker* ground truth than Flirds' exact-SV oracles and would itself mis-rank a Maverick (cf. [[sources/mavericks-shapley-fl]]). The two papers make complementary points: size-GT exposes strategy-volatility; SV-GT (Mavericks) exposes distribution-bias.
- The instability is shown for round-summed Shapley with an inverse-linear round weight + halting round $R$ (Fig. 1: optimal $R$ has no clean distribution). Does Flirds' round-aggregation choice inherit any of this $R$-sensitivity? > TODO: check whether the closed-form Taylor estimator's round-sum is more stable than reconstruction-based round-sums.
- Krum (which drops outlier updates) is the consistent loser — a reminder that robust-aggregation defenses can wreck contribution estimation even when model accuracy is fine. Relevant if Flirds is ever paired with a Byzantine-robust aggregator (Phase 3 backdoor work).
