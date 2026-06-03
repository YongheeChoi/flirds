---
type: source
title: "ShapleyFL: Robust Federated Learning Based on Shapley Value"
created: 2026-05-05
updated: 2026-05-05
topic: flirds
tags: [federated-learning, shapley, robustness, importance-sampling, surrogate-shapley]
---

# ShapleyFL

## Citation

Qiheng Sun, Xiang Li, Jiayao Zhang, Li Xiong, Weiran Liu, Jinfei Liu, Zhan Qin, Kui Ren (Zhejiang U / Emory / Alibaba). *ShapleyFL: Robust Federated Learning Based on Shapley Value*. KDD 2023.

Raw: `raw/papers/flirds/3580305.3599500.pdf`. Code: `https://github.com/ZJU-DIVER/ShapleyFL-Robust-Federated-Learning-Based-on-Shapley-Value` (image classification: MNIST / Fashion MNIST / CIFAR-10; cross-silo healthcare Fed-ISIC2019).

## TL;DR

Frames FL training as a **sequence of cooperative games**, defines a *surrogate federated Shapley value* that aggregates per-round marginal contributions across rounds, and uses these values both for client selection (importance sampling) and for cheaper Shapley estimation (a difference-based approximation). Provides convergence and stability analysis.

## Problem

Two pain points the paper attacks together:

1. **Naive per-round Shapley** is expensive — $O(2^n)$ in clients per round, $O(T \cdot 2^n)$ in total.
2. **Round-by-round client selection** is noisy if not value-informed; standard random selection drops valuable clients.

ShapleyFL aims to do both efficiently using the same Shapley machinery.

## Method

### Surrogate federated Shapley

Define a per-round Shapley over client gradient updates, then aggregate across rounds. The "surrogate" qualifier acknowledges that the cumulative quantity is not the joint Shapley over a multi-round game — it's a tractable approximation that retains key properties.

### Importance-sampling client selection

Instead of uniform sampling each round, sample clients with probability proportional to a function of their cumulative surrogate Shapley. Drives the system toward valuable clients while preserving exploration.

### Difference-of-Shapley estimator

Avoid recomputing Shapley from scratch each round. Estimate per-round Shapley updates incrementally as differences from the previous round's estimates.

### Theory

- Convergence guarantee for the FL model under ShapleyFL's adaptive client selection.
- Stability analysis showing that surrogate Shapley estimates are bounded under reasonable assumptions.

## Key results

- Improved final-model accuracy under noisy or adversarial clients vs. FedAvg + random selection.
- Lower communication and computation cost than naive federated-Shapley methods.
- Robust to varying participation rates.

## Connections

- Federated-Shapley triplet alongside [[sources/gtg-shapley|GTG-Shapley]] (gradient sub-model reconstruction), [[sources/game-of-gradients-sfedavg|S-FedAvg]] (client pruning).
- The "sequence of cooperative games" framing is closely related to [[sources/asymmetric-data-shapley|Asymmetric Data Shapley]]'s ordered-groups formulation; they differ in motivation (ShapleyFL: efficiency / robustness vs. ADS: axiomatic correctness for temporal dependence) but the math has similar shape. Worth synthesizing in [[threads/symmetry-and-asymmetry-axioms]].
- Belongs to thread [[threads/federated-and-decentralized-attribution]].
- Concept page: [[concepts/shapleyfl]], [[concepts/surrogate-federated-shapley]] (created with this batch).

## Notes / open questions

- Surrogate Shapley is *not* the joint multi-round Shapley; the paper's robustness claims rest on surrogate properties. How much fairness is sacrificed?
- The difference estimator's variance compounds across rounds. Long-horizon stability needs careful empirical study.
- Comparison to [[sources/ripple-shapley|Ripple Shapley]]: both handle multi-round FL Shapley but Ripple Shapley uses propagation-based decomposition. They might give different rankings on the same trajectory — useful empirical thread.
- **"AFedSV" alias**: the robust-FL baseline several 2024–2026 papers call *AFedSV* ("adaptive federated Shapley value" — surrogate SV updated each round to dynamically reweight clients) is this ShapleyFL aggregation. [[sources/fedif|FedIF]] cites it as ref [17] = ShapleyFL; [[sources/shapfed|ShapFed]] benchmarks against it. There is no separate "AFedSV" paper — when a Flirds eval needs the adaptive-SV-aggregation comparator, it is ShapleyFL.
