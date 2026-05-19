---
type: concept
title: Semivalue
created: 2026-05-05
updated: 2026-05-05
sources: [data-banzhaf]
tags: [game-theory, fairness]
---

# Semivalue

## One-liner

A class of cooperative-game value notions characterized by linearity, symmetry, dummy-player, and a *probabilistic-weight* axiom. Includes [[concepts/shapley-value|Shapley]], [[concepts/banzhaf-value|Banzhaf]], [[concepts/leave-one-out|LOO]], Beta-Shapley, and CS-Shapley as special cases.

## Definition (semi-formal)

A semivalue is any value of the form

$$\phi_i \;=\; \sum_{S \subseteq D \setminus \{i\}} w_{|S|} \big( U(S \cup \{i\}) - U(S) \big)$$

where $\{w_k\}_{k=0}^{n-1}$ is a probability distribution over subset *sizes* (i.e. $\sum_k \binom{n-1}{k} w_k = 1$).

Different choices of $\{w_k\}$ give different semivalues:

| Variant | Weighting |
|---|---|
| LOO | all weight on $k = n-1$ |
| Shapley | uniform over sizes (after the binomial correction) |
| Banzhaf | uniform over subsets ($w_k = 1/2^{n-1}$ for all $k$) |
| Beta-Shapley | $w_k$ from a Beta distribution; parametric |
| CS-Shapley | weighting tuned for class-imbalanced settings |

## Why the framework matters for ML

[[sources/data-banzhaf]] argues that the right way to compare data-value notions is to fix the semivalue framework — which guarantees the basic fairness axioms — and then ask which weighting is best for *robustness* (under SGD noise), *efficiency of estimation*, and *task-specific desiderata* (bad-data detection, reweighting, market design).

The robustness ranking it proves is:

$$\text{Banzhaf} \succ \text{Shapley} \succ \text{LOO}$$

with exponential gap on the first.

## Trade-off table

| Property | LOO | Shapley | Beta-Shapley | Banzhaf |
|---|---|---|---|---|
| Symmetry, Linearity, Null | yes | yes | yes | yes |
| Efficiency | no | yes | depends | no |
| Noise robustness | worst | medium | medium | best |
| MSR estimator | n/a | biased | biased | unbiased ✓ |

## Where it appears in the wiki

- [[sources/data-banzhaf]] — frames the entire data-valuation comparison through the semivalue lens.

## See also

- [[concepts/shapley-value]]
- [[concepts/banzhaf-value]]
- [[concepts/leave-one-out]]
- [[concepts/data-shapley]]
- [[threads/robustness-to-stochastic-training]]
