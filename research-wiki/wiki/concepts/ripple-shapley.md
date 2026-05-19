---
type: concept
title: Ripple Shapley
created: 2026-05-05
updated: 2026-05-05
sources: [ripple-shapley]
tags: [federated-shapley, in-run, sample-level, jacobian-chain]
---

# Ripple Shapley

The first **single-run, sample-level** federated Shapley method. Decomposes a sample's value into:

- **Drop term** — instantaneous marginal utility within its own round.
- **Ripple term** — recursive propagation through subsequent rounds via a Jacobian chain $J_s J_{s-1} \cdots J_{t+1}$ over global model updates.

Low-rank shared-subspace approximation of the Jacobian chain makes the recursion tractable. Preserves Shapley axioms at the sample level. ~62× speedup over prior FL-Shapley methods. Demonstrated on real-time data pricing.

The federated, sample-level analogue of [[sources/in-run-data-shapley|In-Run Data Shapley]].

See [[sources/ripple-shapley]] for full details.

## See also

- [[concepts/federated-shapley]]
- [[concepts/shapley-value]]
- [[concepts/in-run-data-shapley]] *(see [[sources/in-run-data-shapley]] for now)*
- [[threads/retraining-vs-in-run-attribution]]
