---
type: concept
title: Asymmetric Data Shapley (ADS)
created: 2026-05-05
updated: 2026-05-05
sources: [asymmetric-data-shapley]
tags: [shapley, asymmetry, axiomatic]
---

# Asymmetric Data Shapley (ADS)

A [[concepts/shapley-value|Shapley]] variant that **drops the symmetry axiom** to handle directional and temporal dependencies in real ML pipelines. Averages marginal contributions only over permutations consistent with a pre-specified group ordering. Preserves linearity, nullity, and within-group symmetry; satisfies a stronger *group efficiency*: for each group, sum of values = group's incremental utility relative to preceding groups.

Reduces to standard Data Shapley when all data is in a single group. Two estimators: MC-ADS ($O(n\epsilon^{-2}\log(n/\delta))$ for additive error $\epsilon$) and KNN-ADS (exact and $O(n\log n)$ for KNN classifiers).

Three motivating settings: synthetic-vs-original data, federated learning round structure, multi-stage LLM fine-tuning.

See [[sources/asymmetric-data-shapley]] for full details.

## See also

- [[concepts/shapley-value]]
- [[threads/symmetry-and-asymmetry-axioms]]
