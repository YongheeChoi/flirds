---
type: source
title: "GTG-Shapley: Efficient and Accurate Participant Contribution Evaluation in Federated Learning"
created: 2026-05-05
updated: 2026-05-05
topic: flirds
tags: [federated-learning, shapley, participant-valuation, gradient-reconstruction, monte-carlo]
---

# GTG-Shapley

## Citation

Zelei Liu, Yuanyuan Chen, Han Yu, Yang Liu, Lizhen Cui (NTU Singapore / Tsinghua / Shandong). *GTG-Shapley: Efficient and Accurate Participant Contribution Evaluation in Federated Learning*. ACM TIST 2022 (arXiv 2109.02053, Sep 2021).

Raw: `raw/papers/flirds/2109.02053v1.pdf`

## TL;DR

Computes Shapley values for federated-learning *participants* (not data points) without retraining counterfactual coalitions. Reconstructs sub-model parameters directly from per-client gradient updates — every coalition is evaluated using the gradients already exchanged in the regular FL protocol. Adds a **guided Monte Carlo** scheme that prefers permutations more likely to yield large marginal contributions, plus within-round and between-round truncation.

## Problem

Federated Shapley (Wang et al. 2020 etc.) is conceptually clean but practically expensive: evaluating $U(S)$ for arbitrary client coalitions requires reconstructing the model that *would have resulted* from training on $S$ — equivalent to retraining FL with a different participant subset. Standard Monte Carlo over permutations multiplies this cost.

## Method

### Gradient-reconstruction sub-models

Key observation: in vanilla FedAvg, the global model after round $t$ is a weighted sum of the prior global model and per-client gradients:
$$\theta^{(t+1)} = \theta^{(t)} + \sum_{c \in S} w_c \, \Delta\theta_c^{(t)}$$

Sub-models for any coalition $S$ in any round can be **reconstructed by re-summing the already-collected per-client updates** — no extra communication, no retraining. This makes per-coalition utility evaluation cheap.

### Guided MC (the "G" in GTG)

Standard MC samples permutations uniformly. Guided MC weights its sampling so that permutations likely to produce informative marginal contributions are sampled more frequently. The paper formalizes this with a guidance distribution that adapts during the estimation procedure.

### Truncation (the "T" in GTG)

Two layers:

- **Within-round**: stop adding clients to a coalition once the marginal contribution to round-$t$ utility is small.
- **Between-round**: stop accumulating round-by-round Shapley scores once added rounds barely move things.

## Key results

- Substantially fewer utility evaluations than uniform MC for the same accuracy of Shapley estimation.
- Empirically more accurate than competing federated-Shapley estimators across image classification benchmarks.
- Preserves the standard FL communication pattern (no extra rounds for valuation).

## Connections

- One of three federated-Shapley sub-model-reconstruction papers in the wiki, alongside [[sources/space-participant-amalgamation]] (single-round, distillation-based) and [[sources/ripple-shapley]] (single-run, propagation-based).
- Belongs to thread [[threads/federated-and-decentralized-attribution]].
- Related to [[sources/dice]] in problem domain (FL/decentralized contribution) but very different mechanism (Shapley reconstruction vs. influence cascade).
- Concept page: [[concepts/federated-shapley]] (created with this batch).
- Cited by later Federated Shapley work as a baseline; an accuracy/speed reference point for [[sources/shapleyfl]].

## Notes / open questions

- Sub-model reconstruction assumes vanilla FedAvg; how robust to FedProx, FedYOGI, secure aggregation?
- Guided MC's sampling distribution depends on prior estimates → potential for high variance early on. The paper has stability analysis worth checking.
- Comparison to single-round methods (SPACE, Ripple Shapley): GTG still iterates over many permutations within each round; the single-round papers eliminate that. Useful head-to-head if Yonghee plans empirical work.

> TODO: read the section on between-round truncation carefully; the paper's notion of "round contribution" is subtle and matters for fair comparison with [[sources/ripple-shapley|Ripple Shapley]].
