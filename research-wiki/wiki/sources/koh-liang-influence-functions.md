---
type: source
title: "Understanding Black-box Predictions via Influence Functions"
created: 2026-05-05
updated: 2026-05-05
topic: flirds
tags: [influence-function, foundational, attribution, robust-statistics]
---

# Koh & Liang — Influence Functions for ML

## Citation

Pang Wei Koh & Percy Liang (Stanford). *Understanding Black-box Predictions via Influence Functions*. ICML 2017 (PMLR 70). arXiv:1703.04730.

Raw: `raw/papers/flirds/1703.04730v3.pdf`

## TL;DR

The foundational paper that brought **influence functions** from robust statistics into modern ML. Defines the influence of training point $z$ on test prediction loss as $-g_{\text{te}}^\top H^{-1} g_z$, and shows how to compute this efficiently for non-convex deep networks via Hessian-vector products (without forming or inverting the full Hessian). Spawned the entire IF / TracIn / DataInf / EKFAC / LoGra family.

## Problem

For a trained ML model and a test prediction, **which training points were most responsible?** Pre-2017 ML attribution was limited to leave-one-out (retrain-cost prohibitive) or simple proxies (representation similarity). Influence functions had a long history in robust statistics (Hampel, Cook) but their use on non-convex deep nets was widely doubted.

## Method

Adapt the classical influence-function calculation to deep learning:

1. **Define** $\mathcal{I}_{\text{up,loss}}(z, z_{\text{test}}) := -\nabla_\theta L(z_{\text{test}})^\top H_{\theta^*}^{-1} \nabla_\theta L(z)$ — the change in test loss caused by an infinitesimal up-weighting of training point $z$.
2. **Compute it without forming $H^{-1}$**: use stochastic estimation (LiSSA-style iterative inversion) to compute the inverse-Hessian-vector product $H^{-1} v$ in time linear in parameters per iteration.
3. **Apply on non-convex models** by treating local curvature as the relevant Hessian, with damping to ensure positive-definiteness.

Two related quantities also derived:

- **Influence of perturbing** a training point (not just up-weighting it): the gradient of the test loss with respect to a perturbation of $z$.
- **Group influence**: sum of individual influences (with caveats about non-linearity).

## Key results

- **Demonstrates the framework works on non-convex CNNs** — the empirical Hessian is good enough as a local approximation despite non-convexity.
- **Mislabeled-data detection**: ranks training points by self-influence; top-ranked points are mostly mislabels.
- **Training-set adversarial attacks**: small, imperceptible perturbations of a single training image can flip the model's prediction on a target test point — a security concern.
- **Model debugging and "this prediction was caused by these training points"** as a paradigm.

## Connections

- The taproot of the entire gradient-based attribution literature in the wiki:
  - [[sources/datainf]] — closed-form approximation specialized for LoRA.
  - [[sources/logix]] — projection-based scaling to LLMs.
  - [[sources/in-run-data-shapley]] — uses gradient calculus in the same spirit but for Shapley.
  - [[sources/dice]] — extension to decentralized/cascading settings.
  - [[sources/trak]] (when ingested) — eNTK-projection variant.
- Concept page: [[concepts/influence-function]] (created with this batch).
- Listed in [[threads/retraining-vs-in-run-attribution]] as the canonical gradient-based method on the "no retraining" side.
- Related thread: [[threads/influence-functions-at-llm-scale]].

## Notes / open questions

- The non-convexity claim is empirical; later work (Bae et al., "If Influence Functions are the Answer, Then What is the Question?") argued classical IF is brittle on deep nets and what works in practice is closer to a "proximal Bregman response function." Worth ingesting if accessible.
- The Hessian-damping choice and LiSSA convergence assumptions are practical but not theoretically tight; the field has moved to KFAC/EKFAC/LoGra for stronger scalability.
- The mislabeled-data and adversarial-attack experiments are now the standard evaluation protocols for new IF methods — useful for our cross-method-comparison thread.
