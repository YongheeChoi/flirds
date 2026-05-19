---
type: concept
title: DU-Shapley (Discrete Uniform Shapley)
created: 2026-05-05
updated: 2026-05-05
sources: [du-shapley]
tags: [shapley, dataset-valuation, monte-carlo]
---

# DU-Shapley

A structure-aware Shapley approximation for [[concepts/dataset-valuation|dataset valuation]] in settings where the utility depends only on a scalar function of the coalition. Re-expresses Shapley as an expectation under a discrete uniform distribution on a small support, exponentially reducing utility evaluations.

Almost-sure convergence to the Shapley value as the number of dataset owners $I \to \infty$. Theoretical guarantees in three use cases: non-parametric regression (regressograms), homogeneous case, heterogeneous linear regression with local DP.

See [[sources/du-shapley]] for full details.

## See also

- [[concepts/shapley-value]]
- [[concepts/dataset-valuation]]
