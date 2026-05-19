---
type: concept
title: Banzhaf value
created: 2026-05-05
updated: 2026-05-05
sources: [data-banzhaf]
tags: [semivalue, robustness]
---

# Banzhaf value

## One-liner

A classical cooperative-game-theory value (Banzhaf 1965) that gives **equal weight to each subset** rather than each subset *size*; in the ML context it has exponentially better robustness to performance-score noise than the Shapley value.

## Formal definition

$$\phi^{\text{Banzhaf}}_i \;=\; \frac{1}{2^{n-1}} \sum_{S \subseteq D \setminus \{i\}} \big( U(S \cup \{i\}) - U(S) \big)$$

Compare with Shapley, which has an extra $\binom{n-1}{|S|}^{-1}$ factor that re-weights each subset by the inverse number of subsets of that size — equalizing weight across *sizes*. Banzhaf equalizes across *subsets*.

## Properties

| Axiom | Shapley | Banzhaf |
|---|---|---|
| Linearity | yes | yes |
| Symmetry | yes | yes |
| Null player | yes | yes |
| **Efficiency** ($\sum_i \phi_i = U(D)$) | yes | **no** |

Banzhaf gives up efficiency. For data markets that require budget-balance, this is a real cost; for data ranking and noisy-label detection, it doesn't matter.

## Why it's noise-robust

[[sources/data-banzhaf]] introduces the **safety margin** — the largest $\ell_\infty$ perturbation of $U$ that preserves every pairwise data-value ordering. The result:

$$\text{margin}(\text{Banzhaf}) \gg \text{margin}(\text{Shapley}) > \text{margin}(\text{LOO})$$

The Banzhaf vs. Shapley gap is exponential in $n$. Intuition: Banzhaf's uniform weighting averages over more subsets equally, smoothing noise more aggressively.

## Maximum Sample Reuse (MSR) estimator

Banzhaf's distinguishing computational property: an unbiased estimator that **reuses each subset sample for every point's marginal contribution** is uniquely efficient for it.

```
sample S ~ Uniform(2^D)
for each i:
    accumulate U(S ∪ {i}) - U(S \ {i})
```

This trick is biased for Shapley. Sample complexity for Banzhaf is logarithmic for $\ell_\infty$ error and nearly linear for $\ell_2$ error — close to a proven lower bound.

## Strengths

- Largest safety margin among semivalues.
- Unique MSR estimator — much cheaper than Shapley estimation in practice.
- Outperforms Shapley/Beta-Shapley/LOO on bad-data detection and weighted-sample learning under SGD.

## Limitations

- Loses efficiency — sum of values doesn't equal total utility, so it can't directly serve as a budget-balanced revenue split in a data market.
- Still subset-counterfactual at heart — no closed-form scaling to LLM pretraining.

## Where it appears in the wiki

- [[sources/data-banzhaf]] — defines Data Banzhaf, proves the safety-margin gap, gives the MSR estimator and empirical results.

## See also

- [[concepts/shapley-value]]
- [[concepts/leave-one-out]]
- [[concepts/semivalue]] (when created)
- [[threads/robustness-to-stochastic-training]]
