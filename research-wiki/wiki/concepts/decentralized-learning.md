---
type: concept
title: Decentralized learning
created: 2026-05-05
updated: 2026-05-05
sources: [dice]
tags: [distributed-training, p2p, no-server]
---

# Decentralized learning

## One-liner

Distributed training **without a central server** — participants form a peer-to-peer graph and update via local SGD plus parameter mixing with their graph neighbors. Generalizes federated learning by removing the server.

## Setup

A connected graph $G = (\mathcal{V}, \mathcal{E})$. Each node $k$ holds local data $\mathcal{D}_k$ and its own model $\bm{\theta}_k$. Training alternates:

1. **Local step**: each node performs SGD on its local data.
2. **Mixing step**: $\bm{\theta}_k \gets \sum_{j \in N(k)} P_{kj} \bm{\theta}_j$ for some doubly-stochastic mixing matrix $P$.

[[concepts/federated-learning|FL]] is the special case where the topology is a star (server in the middle).

## Why it changes data attribution

Without a central server, influence of a data point at one node propagates to *all* nodes via iterative mixing — what [[sources/dice|DICE]] calls **influence cascade**. Standard centralized attribution methods don't capture this cascade. The cascade depends jointly on:

- The data itself (local gradient).
- The keeper's topological position (mixing-matrix-power-weighted reach).
- The curvature at intermediate nodes (Hessians along the propagation path).

## Where it appears in the wiki

- [[sources/dice]] — first attribution framework for decentralized learning.

## See also

- [[concepts/federated-learning]]
- [[concepts/dice]]
- [[threads/federated-and-decentralized-attribution]]

> TODO: this subfield is sparse in our wiki so far. Look for additional decentralized-learning papers (gossip protocols, blockchain-based incentive, etc.) on the next ingest pass.
