---
type: source
title: "Game of Gradients: Mitigating Irrelevant Clients in Federated Learning"
created: 2026-05-05
updated: 2026-05-05
topic: flirds
tags: [federated-learning, shapley, client-selection, robustness, frcs]
---

# Game of Gradients (S-FedAvg)

## Citation

Lokesh Nagalapatti, Ramasuri Narayanam (IIT Bombay / IBM Research India). *Game of Gradients: Mitigating Irrelevant Clients in Federated Learning*. AAAI 2021 (paper #17093).

Raw: `raw/papers/flirds/17093-Article Text-20587-1-2-20210518.pdf`

## TL;DR

Casts FedAvg aggregation as a cooperative game over client gradients, then uses the Shapley value to identify and downweight irrelevant or corrupted clients. Defines the **Federated Relevant Client Selection (FRCS)** problem and proposes **S-FedAvg**, a Shapley-weighted aggregator that prunes negative-Shapley clients each round.

## Problem

In a heterogeneous FL deployment, some clients have data that hurts the global model (corrupted, off-distribution, label-flipped, malicious). Standard FedAvg averages all client updates uniformly, propagating their harm. Need a principled rule for spotting and excluding such clients **without** retraining or examining client data.

## Method

### Cooperative game over gradients

In each round, compute Shapley values where:

- Players: clients participating that round.
- Utility: improvement in some validation metric when the server aggregates a coalition's gradients (instead of all of them).

Clients with persistently low or negative Shapley get downweighted in subsequent rounds; those consistently flagged get dropped.

### S-FedAvg

Replaces FedAvg's uniform weighting:
$$\theta^{(t+1)} = \theta^{(t)} + \sum_c \frac{\phi_c^{(t)}}{\sum_{c'} \phi_{c'}^{(t)}} \, \Delta\theta_c^{(t)}$$

with Shapley-derived weights instead of (or alongside) the standard sample-count weights.

### Auxiliary algorithms

- **Class-specific best-client selection** for tasks where some clients have specialized data.
- **Label standardization** auxiliary protocol when clients use slightly inconsistent label spaces.

## Key results

- Demonstrates robust aggregation under noisy / irrelevant clients on standard FL benchmarks.
- Shapley pruning improves test accuracy under various corruption fractions.
- Shows the cooperative-game framing is operationalizable inside the FedAvg loop without major communication overhead.

## Connections

- Federated-Shapley line; precedes [[sources/gtg-shapley|GTG-Shapley]] and [[sources/shapleyfl|ShapleyFL]] in framing FL as a cooperative game over gradients.
- Different goal from [[sources/feddqc|FedDQC]]: FedDQC controls *data quality* on-device; S-FedAvg controls *client trust* server-side via Shapley.
- Belongs to thread [[threads/federated-and-decentralized-attribution]].
- Concept page: [[concepts/federated-shapley]], [[concepts/frcs]] (Federated Relevant Client Selection — created with this batch).

## Notes / open questions

- Computing Shapley each round can be costly; the paper's scalability claims warrant a careful read against [[sources/gtg-shapley|GTG-Shapley]]'s sub-model-reconstruction trick.
- "Negative Shapley = bad client" is a heuristic; under high gradient noise, it might flip honest clients. Worth testing against the safety-margin framework from [[sources/data-banzhaf]].
- Label standardization is an interesting side-protocol but feels like a separate problem; the connection to Shapley is loose.

> TODO: verify whether S-FedAvg is purely re-weighting or also drops clients permanently after $k$ rounds of low Shapley.
