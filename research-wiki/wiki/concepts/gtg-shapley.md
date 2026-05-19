---
type: concept
title: GTG-Shapley
created: 2026-05-05
updated: 2026-05-05
sources: [gtg-shapley]
tags: [federated-shapley, monte-carlo, gradient-reconstruction]
---

# GTG-Shapley

Federated Shapley estimator that **reconstructs sub-models from logged client gradients** instead of retraining counterfactual coalitions, plus **guided Monte Carlo** sampling and within-/between-round truncation. Drastically reduces utility evaluations compared to uniform MC.

See [[sources/gtg-shapley]] for full details.

## See also

- [[concepts/federated-shapley]]
- [[concepts/shapley-value]]
