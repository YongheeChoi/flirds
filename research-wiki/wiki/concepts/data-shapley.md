---
type: concept
title: Data Shapley
created: 2026-05-05
updated: 2026-05-05
sources: [in-run-data-shapley]
tags: [shapley, attribution]
---

# Data Shapley

## One-liner

The application of the [[concepts/shapley-value|Shapley value]] to ML training data: each training point is a player; the coalition utility is validation performance of a model trained on that coalition. Originally introduced by Ghorbani & Zou (2019).

## Two variants of "Data Shapley"

The literature uses the same name for two genuinely different quantities:

1. **Algorithm-level / retraining-based** — Shapley over the *learning algorithm* $\mathcal{A}$:
   $$U(S) := \mathbb{E}\big[\, \text{Perf}(\mathcal{A}(S)) \,\big]$$
   averaged over algorithm randomness (init, batch order). Computed by training many models on subsets.

2. **Model-level / in-run** — Shapley for **the specific model that was actually produced** by one training run. No expectation over randomness; the value is conditional on this trained model. See [[sources/in-run-data-shapley]].

These quantities can differ substantially when training is noisy. The first is more "fair across hypothetical worlds"; the second is more "what did this data do for *my* model." Which one matters depends on the application — see [[threads/retraining-vs-in-run-attribution]].

## Computational ladder

| Approach | Cost | Scope |
|---|---|---|
| Exact Shapley | $O(2^n)$ retrainings | toy only |
| Monte Carlo permutation-sampling | many retrainings | small datasets |
| Truncated Monte Carlo (Ghorbani & Zou) | fewer retrainings | small/medium |
| KNN-Shapley closed form | $O(n \log n)$, KNN-only | restrictive |
| In-Run Data Shapley | ~one training run | foundation models |

## Where it appears in the wiki

- [[sources/in-run-data-shapley]] — full reformulation as model-specific, per-step accumulated value.

## See also

- [[concepts/shapley-value]]
- [[concepts/banzhaf-value]]
- [[threads/retraining-vs-in-run-attribution]]
- [[threads/attribution-at-llm-scale]]
