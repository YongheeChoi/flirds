---
type: source
title: "TRAK: Attributing Model Behavior at Scale"
created: 2026-05-05
updated: 2026-05-05
topic: flirds
tags: [attribution, ntk, random-projection, datamodels, llm-scale, lds]
---

# TRAK

## Citation

Sung Min Park*, Kristian Georgiev*, Andrew Ilyas*, Guillaume Leclerc, Aleksander Madry (MIT). *TRAK: Attributing Model Behavior at Scale*. ICML 2023. arXiv:2303.14186.

Raw: `raw/papers/flirds/2303.14186v2.pdf`

## TL;DR

A scalable data-attribution method that linearizes the model around its trained parameters using the **empirical Neural Tangent Kernel (eNTK)**, then applies random gradient projections to compute attribution scores cheaply. Matches the counterfactual fidelity of Datamodels at 100–1000× lower cost; demonstrated on ImageNet (ResNet, CLIP), BERT, mT5.

## Problem

Two failure modes of prior attribution:

- **Influence functions** are gradient-based and cheap but Hessian-based; the Hessian's spectral structure makes IF brittle on deep networks (and prior empirical work showed IF often fails to track counterfactual model behavior).
- **Datamodels** retrain on many subsets and fit a linear surrogate over data → counterfactually faithful but extremely expensive.

TRAK aims for the *Datamodels accuracy* at the *gradient-method cost*.

## Method

### eNTK linearization

Around the trained parameters $\theta^*$, linearize the loss function in the parameters:

$$L(\theta^*; z) \approx L(\theta^*; z) + \langle \nabla_\theta L(\theta^*; z), \theta - \theta^* \rangle$$

This converts attribution into a problem on a kernel induced by per-sample gradients (the eNTK).

### Random projection for tractability

The per-sample gradients live in a model-dimensional space; for LLMs, this is too big. TRAK projects gradients to a random low-dimensional space (Johnson–Lindenstrauss style) before computing the kernel.

### Multi-checkpoint averaging

To reduce variance, TRAK averages the linearized attribution scores across multiple independent training runs / checkpoints. 5–10 checkpoints typically suffice.

### Linear Datamodeling Score (LDS)

The paper formalizes an evaluation metric — **LDS** — that measures how well a method's per-sample attribution scores predict the actual change in model behavior under various subset-restricted retrainings. It has since become the standard data-attribution benchmark (used by [[sources/logix|LoGra]] and others).

## Key results

- TRAK matches Datamodels' LDS on CIFAR / ImageNet at 100–1000× lower cost.
- Scales to **ImageNet ResNets, CLIP, BERT, mT5** — the first attribution method demonstrated at this scale.
- Outperforms classical IF, gradient dot product, and related on the LDS benchmark.

## Connections

- **The other major LLM-scale attribution method** alongside [[sources/logix|LoGra/Logix]] and [[sources/datainf|DataInf]]. Compared head-to-head in [[sources/logix]]: LoGra outperforms TRAK on LDS at LLM scale because TRAK's projection-matrix size limits its projection dimension on big models.
- Concept page: [[concepts/trak]] (created), [[concepts/linear-datamodeling-score]] (created).
- Related concepts: [[concepts/influence-function]], [[concepts/datamodels]] (created), [[concepts/empirical-ntk]] (TODO if a future paper requires it).
- Belongs to thread [[threads/influence-functions-at-llm-scale]].

## Notes / open questions

- TRAK and Datamodels have a clean theoretical connection: TRAK's eNTK linearization is essentially a closed-form approximation to what Datamodels does empirically. The wiki should articulate this in [[concepts/datamodels]].
- Multi-checkpoint averaging mostly helps with random-init variance; it doesn't address the deeper question of whether attribution should be model-specific or algorithm-level (see [[threads/retraining-vs-in-run-attribution]]).
- LDS as an eval metric: is it the *right* evaluation? Brittleness test (used in [[sources/logix]]) gives different rankings sometimes. Worth a thread comparing attribution-evaluation metrics.
- The random-projection dimension trades compute for accuracy; LoGra's structured Kronecker projection is strictly better in this regard.
