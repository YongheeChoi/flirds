---
type: source
title: "Data Shapley: Equitable Valuation of Data for Machine Learning"
created: 2026-05-05
updated: 2026-05-05
topic: flirds
tags: [shapley, foundational, data-valuation, monte-carlo, axiomatic]
---

# Ghorbani & Zou — Data Shapley

## Citation

Amirata Ghorbani & James Zou (Stanford). *Data Shapley: Equitable Valuation of Data for Machine Learning*. ICML 2019. arXiv:1904.02868.

Raw: `raw/papers/flirds/1904.02868v2.pdf`

## TL;DR

The seminal paper that brought the Shapley value from cooperative game theory into ML data valuation. Defines **Data Shapley** as the unique allocation satisfying four axioms (null player, symmetry, additivity, and a consistency condition); proposes practical estimators including Truncated Monte Carlo (TMC-Shapley) and gradient-Shapley; demonstrates that Shapley reveals outliers and informs data acquisition substantially better than leave-one-out / leverage scores.

## Problem

Pre-2019 data-valuation work in ML mostly used leave-one-out (LOO) or representation-based heuristics. Both lack a principled fairness framework. Game-theoretic Shapley had been applied to cooperative-game ML problems (feature attribution, etc.) but not systematically to *data-point* valuation. The challenges: defining an axiomatic framework, and producing tractable estimators given Shapley's $2^n$ structure.

## Method

### Axiomatic framework

Treats each training point as a player; coalition utility is validation accuracy of a model trained on that coalition. Defines Data Shapley as the unique allocation satisfying:

1. **Null element**: a point that adds zero marginal utility everywhere gets value 0.
2. **Symmetry**: interchangeable points get equal value.
3. **Additivity**: linear in the utility function.
4. **Group-rationality / consistency** (paper's specific framing).

Shapley value of point $i$:
$$\phi_i = C \cdot \sum_{S \subseteq D \setminus \{i\}} \binom{n-1}{|S|}^{-1} \big(U(S \cup \{i\}) - U(S)\big)$$

### Estimators

- **Truncated Monte Carlo (TMC-Shapley)**: sample permutations; truncate when adding more points doesn't change utility much. Substantially fewer model trainings than naive MC.
- **Gradient-Shapley**: approximates marginal contributions using gradient steps instead of full retraining; fast but less accurate.

### Applications demonstrated

- **Outlier / mislabel detection**: low (or negative) Shapley values flag bad data.
- **Data acquisition**: rank candidate sources by expected Shapley value to guide what to buy.
- **Value-based filtering**: drop low-value points → retrain on a smaller set → comparable or better performance.

## Key results

- Shapley reveals outliers and mislabels more reliably than LOO and leverage scores.
- TMC-Shapley scales to ~thousands of points for moderately-sized models.
- Beats LOO baselines for data-acquisition decision-making in synthetic and real datasets (medical imaging).
- Establishes the empirical case for "Shapley is the fair rule" — a position later partially challenged (see [[sources/data-banzhaf|Banzhaf]] on noise robustness, [[sources/asymmetric-data-shapley|Asymmetric Shapley]] on the symmetry axiom).

## Connections

- The foundational reference for the entire [[concepts/shapley-value]] / [[concepts/data-shapley]] / [[concepts/semivalue]] line.
- Cited (and extended) by [[sources/data-banzhaf]] (different semivalue weighting), [[sources/in-run-data-shapley]] (different definition target — model-specific not algorithm-level), [[sources/asymmetric-data-shapley]] (drops symmetry), [[sources/du-shapley]] (dataset rather than data-point), and most federated-Shapley papers.
- Concept page: [[concepts/data-shapley]] revised after this ingest to flag this as the foundational reference.
- Provides the TMC-Shapley algorithm cited as a baseline by every subsequent Shapley estimator.

## Notes / open questions

- TMC truncation is a heuristic; later work (Beta-Shapley, CS-Shapley) gives more principled re-weightings of subset sizes.
- The validation-set-dependence of $U(S)$ is implicit; [[sources/distributionally-robust-data-valuation|Lin et al. 2024]] argues this should be replaced with a distributionally robust utility.
- Negative Shapley values (data that actually hurts the model) are flagged in passing here; [[sources/in-run-data-shapley]] makes this central.
- The original formulation is algorithm-level (averages over training randomness). [[sources/in-run-data-shapley]] later reframed as model-level. Open: which framing is "correct" for data-market applications? See [[threads/retraining-vs-in-run-attribution]].
