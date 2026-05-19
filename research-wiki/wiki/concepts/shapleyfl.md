---
type: concept
title: ShapleyFL
created: 2026-05-05
updated: 2026-05-05
sources: [shapleyfl]
tags: [federated-shapley, robustness, importance-sampling]
---

# ShapleyFL

Federated training framed as a **sequence of cooperative games**. Defines a *surrogate federated Shapley* aggregating per-round marginal contributions; uses these values for importance-sampling client selection and a difference-based estimator for cheaper Shapley computation. Provides convergence and stability analysis.

See [[sources/shapleyfl]] for full details.

## See also

- [[concepts/federated-shapley]]
