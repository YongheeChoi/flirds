---
type: concept
title: S-FedAvg
created: 2026-05-05
updated: 2026-05-05
sources: [game-of-gradients-sfedavg]
tags: [federated-shapley, client-selection, robustness]
---

# S-FedAvg

Shapley-weighted version of FedAvg: at each round, compute Shapley values over client gradient coalitions, then aggregate clients with Shapley-derived weights instead of (or alongside) sample-count weights. Persistently low-Shapley clients get pruned, addressing the **Federated Relevant Client Selection (FRCS)** problem.

See [[sources/game-of-gradients-sfedavg]] for full details.

## See also

- [[concepts/federated-shapley]]
- [[concepts/federated-learning]]
