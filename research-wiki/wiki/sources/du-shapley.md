---
type: source
title: "DU-Shapley: A Shapley Value Proxy for Efficient Dataset Valuation"
created: 2026-05-05
updated: 2026-05-05
topic: flirds
tags: [shapley, dataset-valuation, monte-carlo, theoretical-guarantees]
---

# DU-Shapley

## Citation

Felipe Garrido-Lucero, Benjamin Heymann, Maxime Vono, Patrick Loiseau, Vianney Perchet (Inria, Criteo AI Lab, ENSAE). *DU-Shapley: A Shapley Value Proxy for Efficient Dataset Valuation*. arXiv:2306.02071 (v3).

Raw: `raw/papers/flirds/DU-Shapley_ A Shapley Value Proxy for Efficient Dataset Valuation.md`

## TL;DR

A new structure-aware approximation of the Shapley value for **dataset** valuation (not data-point valuation): the *Discrete Uniform Shapley*. Where standard Monte Carlo treats Shapley as a black-box expectation over permutations, DU-Shapley exploits the fact that the utility function in dataset valuation often depends only on a scalar summary of the coalition (the total dataset size, or weighted size). It expresses the Shapley value as an expectation under a discrete uniform distribution on a tractable support, exponentially reducing the number of utility evaluations needed.

## Problem

Two sub-problems live under the "data valuation" umbrella, often confused:

| Problem | What's a player? | Granularity |
|---|---|---|
| **Data valuation** | a single training point | per-point |
| **Dataset valuation** | a contributor's whole dataset | per-contributor |

The naive reduction "value a dataset = sum the Shapley values of its points" is wrong. Example: $D_1 = \{x_1\}$, $D_2 = \{x_2\}$, $D_3 = \{x_2, x_2\}$, utility $u(D) = \mathbb{1}\{x_1, x_2 \in D\}$. As points, $x_1$ is more valuable. As datasets, $D_2$ and $D_3$ should have *equal* dataset value. Naive summation values $D_3$ at twice $D_2$ — wrong.

For dataset valuation, computing Shapley exactly is exponential; standard Monte Carlo needs many utility evaluations and ignores the structural fact that in many ML problems the utility depends only on a sufficient scalar statistic (e.g., total number of samples).

## Method

### Three theoretical use cases that motivate the approximation

The paper shows that for these settings, $u(\mathcal{S})$ collapses to a function of a scalar:

1. **Non-parametric regression with regressograms** — utility decomposes into bins; per-bin utility depends only on $\sum_i n_{i,b}$.
2. **Homogeneous case** — all players share $p_i = p$; utility depends only on $\sum_i n_i$.
3. **Heterogeneous linear regression with local DP** — utility depends only on $q(\mathcal{S}) = \lfloor (\sum_i (\sigma_i/\varepsilon_i) n_i)^2 / \sum_i (\sigma_i/\varepsilon_i)^2 n_i \rfloor$.

In all three: $u(\mathcal{S}) = w(\text{scalar}(\mathcal{S}))$.

### Discrete-uniform reformulation

Re-express Shapley:

$$\varphi_i(u) \;=\; \mathbb{E}_{K \sim U\{0,\dots,I-1\}} \mathbb{E}_{\mathcal{S} \sim U(2^{\mathcal{I}\setminus\{i\}}_K)}[u(\mathcal{S} \cup \{i\}) - u(\mathcal{S})]$$

When $u$ depends only on a scalar of the coalition, the inner expectation collapses to a function of $K$. The result: an asymptotic and non-asymptotic-guaranteed approximation under a *discrete uniform* on a small support — exponentially fewer utility evaluations than generic MC.

### Theoretical guarantees

- **Almost-sure convergence** to the Shapley value as $I \to \infty$ for all three use cases.
- **Rate of convergence** results for the homogeneous and regressogram settings.
- **Empirical**: DU-Shapley outperforms standard MC, truncated MC, and other Shapley estimators on Shapley-approximation accuracy.

## Key results

- DU-Shapley converges almost surely to the Shapley value as the number of dataset owners grows, in three theoretical use cases.
- Outperforms generic MC approximations of the Shapley value across these settings.
- Demonstrates dataset valuation working at scale that retraining-based MC can't reach.

## Connections

- Sits inside the [[concepts/shapley-value]] family, but on the **dataset** side, not the data-point side. Distinct enough to deserve [[concepts/dataset-valuation]] (created with this ingest) as a separate concept page from [[concepts/data-shapley]].
- Adjacent to [[sources/asymmetric-data-shapley]] in spirit — both exploit specific structure (DU exploits utility-as-function-of-scalar; Asymmetric exploits ordering structure).
- New thread: [[threads/dataset-vs-data-point-valuation]] — fundamental distinction underexplored in our existing pages.
- Concept page: [[concepts/du-shapley]] (created).

## Notes / open questions

- The "utility as function of scalar" assumption breaks for non-IID settings, complex models, multi-task learning. How to extend DU-Shapley structure-aware reasoning to settings where the utility doesn't reduce to a scalar?
- For federated learning, the third use case (heterogeneous linear regression) is suggestive but the actual FL setting (non-linear models, multi-round training) has more structure DU-Shapley doesn't yet exploit.
- The paper claims DU-Shapley is the *first* dataset-valuation approach leveraging utility structure. Worth checking if [[sources/asymmetric-data-shapley|ADS]]'s state-conditioned formulation can be combined with DU-Shapley's scalar reduction in a hybrid.
