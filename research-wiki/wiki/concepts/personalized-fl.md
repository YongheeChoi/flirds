---
type: concept
title: Personalized federated learning (PFL)
created: 2026-05-05
updated: 2026-05-05
sources: [ipfl-model-market]
tags: [federated-learning, personalization]
---

# Personalized FL

A variant of [[concepts/federated-learning|federated learning]] where **each client ends with its own personalized model** rather than sharing a single global model. Client models can be related (e.g., share base weights, differ in a small adapter) or organized in a graph where edge weights determine collaboration intensity.

[[sources/ipfl-model-market|iPFL]]'s graphical-game framing is one instantiation: nodes are clients; edges are weighted collaboration links; the optimization jointly trains personalized models and chooses edge weights.

## Why it changes the data-valuation picture

In standard FL, "value" can be computed against the single global model. In PFL, the same client's data has *different* values for different counterparties' personalized models — a richer structure that game-theoretic mechanisms can exploit (e.g., trader/buyer/seller roles in iPFL).

## See also

- [[concepts/federated-learning]]
- [[concepts/data-market]]
