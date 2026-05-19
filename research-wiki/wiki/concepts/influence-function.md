---
type: concept
title: Influence function
created: 2026-05-05
updated: 2026-05-05
sources: [koh-liang-influence-functions, datainf, logix, in-run-data-shapley, dice, trak, ghorbani-zou-data-shapley]
tags: [attribution, gradient-based, hessian]
---

# Influence function

## One-liner

The classical, gradient-based way to estimate "how much does training point $z$ matter for test prediction $z_{\text{te}}$" without retraining: it's the inverse-Hessian-weighted dot product of the test-loss gradient with the train-loss gradient at the trained parameters.

## Formal definition

Around the empirical risk minimizer $\theta^*$:

$$\mathcal{I}(z, z_{\text{te}}) \;=\; -\nabla_\theta L(z_{\text{te}}; \theta^*)^\top \, H_{\theta^*}^{-1} \, \nabla_\theta L(z; \theta^*)$$

where $H_{\theta^*}$ is the empirical Hessian of the training loss. Interpretation: an infinitesimal up-weighting of $z$ shifts the parameters in the direction $-H^{-1}\nabla L(z)$ (a Newton step on $z$'s loss); the test loss responds linearly to that shift through the test gradient.

## Intuition

If $z$ "looks like" $z_{\text{te}}$ in the curvature-aware geometry $H^{-1}$, removing $z$ hurts the test prediction. Influence functions give a closed form for that "looks like" — it's the Hessian-corrected gradient inner product.

## History and variants

| Method | Approximation | Where it appears |
|---|---|---|
| Classical (Cook 1979 etc.) | exact, only feasible for linear / convex | robust statistics |
| Koh & Liang 2017 | LiSSA stochastic inversion + damping | [[sources/koh-liang-influence-functions]] |
| TracIn | sums of per-checkpoint gradient dot products (no Hessian) | (TODO ingest) |
| Hessian-free | pure gradient dot product | baseline in [[sources/datainf]] |
| **DataInf** | swap inverse and average → closed form via Sherman–Morrison | [[sources/datainf]] |
| **EKFAC** | Kronecker-factored Hessian approximation | baseline in [[sources/logix]] |
| Arnoldi IF | Arnoldi iteration for eigendecomp + projection | baseline in [[sources/logix]] |
| **TRAK** | eNTK linearization + random gradient projection | [[sources/trak]] |
| **LoGra / Logix** | Kronecker-structured projection ($O(\sqrt{nk})$) | [[sources/logix]] |

The trajectory: 2017 → "make the inverse-Hessian-vector product cheap"; 2023+ → "make per-sample gradient computation cheap *too*" (via projection / closed form), because at LLM scale the gradient bottleneck dominates.

## Strengths

- No retraining; computed at the trained model.
- Theoretically grounded in robust-statistics influence theory.
- Composable: per-test-point, per-train-point pairs naturally support per-query attribution (vs. Shapley's coalition-of-everything framing).

## Limitations

- **Non-convexity**: classical theory assumes a strongly convex loss. Empirically still useful on deep nets (Koh & Liang demonstrated this), but later work (Bae et al.) argues the actual quantity computed is closer to a "proximal Bregman response."
- **Hessian ill-conditioning**: requires damping; the choice of damping $\lambda$ matters.
- **Cost at LLM scale**: even with damping + KFAC, naïve $O(nD^2)$ is too much for billion-parameter models. This drove the projection / closed-form variants.
- **Stochasticity**: a single trained model gives a single set of influence values; the algorithm-level vs. model-level distinction ([[threads/retraining-vs-in-run-attribution]]) applies to IF too.

## Where it appears in the wiki

- [[sources/koh-liang-influence-functions]] — foundational paper.
- [[sources/datainf]] — closed-form approximation specialized for LoRA.
- [[sources/logix]] — Kronecker-structured projection at LLM scale.
- [[sources/trak]] — eNTK + random-projection variant.
- [[sources/dice]] — extends the framework to decentralized settings with cascade dynamics.
- [[sources/in-run-data-shapley]] — uses gradient calculus in the same spirit but reframes the target as Shapley.
- [[sources/feddqc]] — uses [[sources/datainf|DataInf]] as a baseline and reports it failing on real-world FL data.

## See also

- [[concepts/shapley-value]] — competing axiomatic framework for attribution.
- [[concepts/datamodels]] — closely related counterfactual-fidelity benchmark.
- [[threads/retraining-vs-in-run-attribution]]
- [[threads/influence-functions-at-llm-scale]]
