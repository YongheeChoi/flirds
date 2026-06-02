---
type: concept
title: Proximal Bregman Response Function (PBRF)
created: 2026-05-22
updated: 2026-05-22
sources: [grosse-llm-influence, koh-liang-influence-functions]
tags: [influence-function, response-function, deep-learning-theory, counterfactual]
---

# Proximal Bregman Response Function

## One-liner

A reformulation of the influence-function target for **non-convex, under-determined** deep networks. Instead of "how would the unique ERM solution change if I upweighted $z_m$?" (which fails because the optimum is non-unique and Hessians can be singular), PBRF asks "how would the *local* solution change if I added a damping term and a Bregman-divergence proximity penalty?" — making the response function well-defined even when the model wasn't trained to convergence.

Introduced by Bae et al. 2022a; [[sources/grosse-llm-influence|Grosse et al. 2023]] adopt it as the actual quantity their EK-FAC influence estimator approximates at LLM scale.

## The classical problem

Classical influence (Koh & Liang 2017): $\mathcal{I}_{\theta^*}(z_m) = -H^{-1}\nabla_\theta \mathcal{L}(z_m, \theta^*)$ assumes a unique minimizer $\theta^*$ and an invertible Hessian. For modern over-parameterized deep nets:

- The optimum is **non-unique** (the loss landscape has flat directions; many parameter settings achieve the same loss).
- The empirical Hessian is typically **singular** or has negative eigenvalues.
- Practical training **doesn't run to convergence** — $\theta^*$ is a checkpoint, not a critical point.

So $H^{-1}$ doesn't strictly exist, and even if it did, the implied counterfactual ("what if I retrained with $z_m$ removed?") is ambiguous because retraining could land anywhere on the optimum manifold.

## The PBRF reformulation

Define the **proximal Bregman objective (PBO)**:
$$\theta^s(\epsilon) = \arg\min_\theta \frac{1}{N}\sum_i D_{\mathcal{L}_i}\!\bigl(h(\theta, x_i),\, h(\theta^s, x_i)\bigr) + \epsilon\, \mathcal{L}(z_m, \theta) + \frac{\lambda}{2}\|\theta - \theta^s\|^2$$

Three terms:

1. **Bregman divergence** $D_{\mathcal{L}_i}(\hat y, \hat y^s) = \mathcal{L}_i(\hat y) - \mathcal{L}_i(\hat y^s) - \nabla\mathcal{L}_i(\hat y^s)^\top(\hat y - \hat y^s)$ on the function-space outputs — keeps the model's *outputs* close to the trained model's outputs.
2. **The new datum's loss** $\epsilon\,\mathcal{L}(z_m, \theta)$ — the perturbation.
3. **Weight-space proximity** $\tfrac{\lambda}{2}\|\theta - \theta^s\|^2$ — damping that keeps the new optimum close to $\theta^s$.

The Implicit Function Theorem on PBO gives a well-defined response function:
$$\mathcal{I}_{\theta^s}(z_m) = \left.\frac{d\theta^s}{d\epsilon}\right|_{\epsilon=0} = -(G + \lambda I)^{-1}\nabla_\theta\mathcal{L}(z_m, \theta^s)$$
where $G$ is the Gauss-Newton Hessian. **This is what influence estimators compute on deep nets** — not the classical $-H^{-1}\nabla\ell$.

## Why this matters for the wiki

- Resolves a long-standing tension: classical influence "doesn't apply" to deep nets (Hessian singular, optima non-unique), yet works empirically. PBRF explains *why*: the gradient ⊙ damped-Hessian-inverse is approximating PBRF, not the classical counterfactual.
- Makes "we are not actually computing the global retraining counterfactual" defensible — PBRF is the well-defined object the algorithm targets.
- Bridges the [[threads/retraining-vs-in-run-attribution]] distinction: PBRF is a **local** quantity (proximal to $\theta^s$), unlike retraining-based Shapley which globally re-optimizes. In-run methods like [[sources/in-run-data-shapley|IRDS]] are also local — same family.

## Limitations

- PBRF doesn't capture phenomena that require *non-local* parameter changes — circuit formation, representational rearrangement, phase transitions during training.
- [[sources/grosse-llm-influence|Grosse et al. 2023]] explicitly note (limitation 1, §1) that even if their EK-FAC IF approximates PBRF tightly, "we do not address the question of how well the PBRF captures the training phenomena we are ultimately interested in understanding."
- The damping $\lambda$ and the Bregman choice are hyperparameters of the *target quantity*, not just the estimator — making cross-paper comparisons subtle.

## Where it appears in the wiki

- [[sources/grosse-llm-influence]] — explicit adoption as the IF target at 52B scale.
- [[sources/koh-liang-influence-functions]] — the classical formulation PBRF replaces.
- Bae et al. 2022a "If Influence Functions are the Answer..." — the originating paper. Not yet in `raw/`; flagged in [[overview]] as ingest-next.

## See also

- [[concepts/influence-function]]
- [[concepts/ekfac]]
- [[threads/retraining-vs-in-run-attribution]]
- [[threads/influence-functions-at-llm-scale]]
