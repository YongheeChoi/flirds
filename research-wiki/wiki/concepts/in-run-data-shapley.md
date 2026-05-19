---
type: concept
title: In-Run Data Shapley
created: 2026-05-05
updated: 2026-05-05
sources: [in-run-data-shapley, ripple-shapley]
tags: [shapley, in-run, model-level, foundation-model]
---

# In-Run Data Shapley

A redefinition of [[concepts/data-shapley|Data Shapley]] as **model-specific** rather than algorithm-level. Computed by Taylor-expanding each gradient-update step's utility change and accumulating the per-step Shapley values:

$$\phi_i^{\text{in-run}} \;=\; \sum_t \phi_i^{(t)}, \quad \phi_i^{(t)} \propto \langle g^{\text{val}}_t, g_i \rangle \text{ (first-order)}$$

Optimized implementations compute the per-step accumulation in 1–2 backward passes per step, so the total runtime is comparable to standard training.

Demonstrated at GPT-2 / Pythia-410M scale with three case studies: (1) ~16% of Pile has negative Shapley value; (2) data contribution is stage-dependent (general corpora help early, domain corpora help late); (3) attribution of paraphrased outputs reveals copyright implications.

[[sources/ripple-shapley|Ripple Shapley]] extends the per-step framework to federated learning trajectories with cross-round Jacobian propagation.

See [[sources/in-run-data-shapley]] for full details.

## See also

- [[concepts/shapley-value]]
- [[concepts/data-shapley]]
- [[concepts/ripple-shapley]]
- [[threads/retraining-vs-in-run-attribution]]
