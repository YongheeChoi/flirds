---
type: concept
title: Shapley value (and Data Shapley)
created: 2026-05-05
updated: 2026-05-05
sources: [data-banzhaf, in-run-data-shapley]
tags: [semivalue, fairness, attribution]
---

# Shapley value

## One-liner

A unique fair allocation rule from cooperative game theory: each player's value is the average marginal contribution it brings to a coalition, averaged over all orderings of players. Adapted to ML as **Data Shapley** by treating each training point as a player and a model's validation performance as the coalition utility.

## Formal definition

For training set $D$ of size $n$ and utility function $U: 2^D \to \mathbb{R}$:

$$\phi_i \;=\; \frac{1}{n} \sum_{k=1}^{n} \binom{n-1}{k-1}^{-1} \sum_{\substack{S \subseteq D \setminus \{i\} \\ |S| = k-1}} \big( U(S \cup \{i\}) - U(S) \big)$$

Equivalently, average the marginal contribution of $i$ over a uniformly random *permutation* of the players.

## The four axioms

The Shapley value is the **unique** allocation satisfying all four:

1. **Efficiency** — $\sum_i \phi_i = U(D) - U(\varnothing)$ (values sum to total utility).
2. **Symmetry** — interchangeable players get equal value.
3. **Linearity** — $\phi_i$ is linear in $U$.
4. **Null player** — a player whose marginal contribution is always 0 gets value 0.

This uniqueness is the source of Shapley's appeal — and also of its rigidity.

## Intuition

If you imagine players arriving one at a time in a random order, each player's Shapley value is the expected utility increase they cause when they arrive. The averaging over orderings ensures fairness — no player is privileged by being first.

## Variants and history

- **Leave-one-out (LOO)** — restrict to $S = D \setminus \{i\}$ only. Cheap, lossy. See [[concepts/leave-one-out]].
- **[[concepts/banzhaf-value|Banzhaf value]]** — drop efficiency; uniform weight over subsets. More noise-robust under SGD ([[sources/data-banzhaf]]).
- **Beta-Shapley**, **CS-Shapley** — re-tilt the weighting by subset size for ML-specific desiderata.
- **Data Shapley** — Ghorbani & Zou 2019: applies Shapley to ML training points.
- **In-Run Data Shapley** — Wang et al. 2024: computes Shapley per gradient step, no retraining, model-specific. See [[sources/in-run-data-shapley]].
- **Asymmetric Data Shapley** — drops the symmetry axiom to encode order/dependency among groups.

The unifying abstraction is the [[concepts/semivalue|semivalue]], where each variant corresponds to a different weighting of subset sizes.

## Strengths

- Axiomatic justification is unique among data-attribution frameworks.
- Works at the level of either points or contributors.
- Interpretable as fair revenue split — essential for data markets.

## Limitations

- **Cost**: $2^n$ subsets in the exact form; Monte Carlo or specialized estimators needed beyond toy scale.
- **Noise sensitivity**: under stochastic training, ranking flips between runs ([[sources/data-banzhaf]] showed this).
- **Symmetry isn't always desired**: synthetic vs. original data, sequential federated rounds, multi-stage fine-tuning all violate the implicit interchangeability assumption.
- **Algorithm-level vs. model-level ambiguity** — classical Shapley averages over training randomness, which may not be the quantity you want ([[sources/in-run-data-shapley]]).

## Where it appears in the wiki

- [[sources/data-banzhaf]] — uses Shapley as the comparison baseline; shows its safety margin under SGD is much smaller than Banzhaf's.
- [[sources/in-run-data-shapley]] — redefines Shapley to be computable in a single training run, model-specific.

## See also

- [[concepts/semivalue]] (when created)
- [[concepts/banzhaf-value]]
- [[concepts/leave-one-out]]
- [[threads/retraining-vs-in-run-attribution]]
- [[threads/robustness-to-stochastic-training]]
