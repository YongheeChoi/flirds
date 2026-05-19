---
type: concept
title: SPACE
created: 2026-05-05
updated: 2026-05-05
sources: [space-participant-amalgamation]
tags: [federated-shapley, single-round, knowledge-distillation, prototype-evaluation]
---

# SPACE

Federated Shapley in **a single communication round**. Two ingredients:

1. **Federated Knowledge Amalgamation (FKA)** — clients send full local models; server distills them into a single amalgamated server model in one shot.
2. **Prototype-based Model Evaluation (PME)** — coalition utility evaluated against per-class embedding prototypes instead of a held-out validation set, eliminating validation-set-size dependence.

Plus a logistic-style satisfaction function for monotonicity / saturation of resulting Shapley values.

See [[sources/space-participant-amalgamation]] for full details.

## See also

- [[concepts/federated-shapley]]
