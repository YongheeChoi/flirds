---
type: source
title: "DICE: Data Influence Cascade in Decentralized Learning"
created: 2026-05-05
updated: 2026-05-05
topic: flirds
tags: [decentralized-learning, influence-cascade, topology, federated, incentive]
---

# DICE — Data Influence Cascade in Decentralized Learning

## Citation

Tongtian Zhu, Wenhao Li, Can Wang (Zhejiang University) & Fengxiang He (University of Edinburgh). *DICE: Data Influence Cascade in Decentralized Learning*. arXiv:2507.06931 (v1).

Raw: `raw/papers/flirds/DICE_ Data Influence Cascade in Decentralized Learning.md`

## TL;DR

The first influence-estimation framework for **fully decentralized** (peer-to-peer) learning. In a decentralized graph, the influence of a node's data doesn't stay local — it propagates through parameter exchanges to multi-hop neighbors. DICE derives tractable approximations of this *influence cascade* over arbitrary numbers of hops, expressing the result as the joint product of three factors: the data itself, the topological position of the keeper, and the curvature of the loss landscape at intermediate nodes.

## Problem

Existing data-influence estimators ([[concepts/shapley-value]], [[concepts/leave-one-out]], influence functions, TracIn, [[sources/in-run-data-shapley|In-Run Shapley]]) all assume **centralized** training: the data, the model, and the influence computation live in one place. In decentralized learning, that's false: each node only has its local data and computes via gossip-style exchanges with its graph neighbors. A point's influence on the global system extends beyond its own model to whatever neighbors and neighbors-of-neighbors reach via parameter sync.

The motivation is concrete: decentralized training is a credible alternative to data-center training, but contributors need an incentive structure. Fair incentives require fair attribution. Without a cascade-aware notion, contributors at well-connected nodes (or with curvature-favorable neighbors) are systematically misvalued.

## Method

### Setup

A connected graph $G = (\mathcal{V}, \mathcal{E})$ of participants. Each node $k \in \mathcal{V}$ has its own model $\bm{\theta}_k$, local data $\mathcal{D}_k$, and weight $q_k$. Joint objective:

$$\min_{\{\bm{\theta}_k\}} \sum_{k} q_k \mathbb{E}_{\bm{z} \sim \mathcal{D}_k}[L(\bm{\theta}_k; \bm{z})]$$

Updates alternate (i) local SGD steps and (ii) parameter mixing across edges.

### Ground-truth cascading influence (Definition 3)

Generalizes the centralized definition: integrate direct *and* indirect contributions of a data point, accounting for propagation across multi-hop edges through the iterative parameter-mixing dynamics.

### Tractable approximation (Theorem 1)

DICE-E approximation: for any number of neighbor hops, influence factorizes into three terms:

1. **Data term** — $\nabla_{\bm{\theta}} L(\bm{\theta}; \bm{z})$ at the originating node (the usual gradient).
2. **Topological importance** — function of the communication topology and the keeper's position in it (think: weighted spread through the graph Laplacian / mixing matrix powers).
3. **Curvature term** — Hessians at *intermediate* nodes that mediate propagation; small-curvature regions dampen the cascade, sharp-curvature regions amplify it.

The ripple-in-water analogy: influence originates at the stem node, weakens with distance, but the rate of weakening depends on what's between you and the receiver.

## Key results

- **Conceptual**: defines ground-truth cascading influence for decentralized learning — first such definition.
- **Theoretical**: Theorem 1 gives a tractable closed-form-ish approximation for arbitrary hops, decomposing influence into data × topology × curvature.
- **Applications proposed**: collaborator selection (find nodes whose data complements yours), free-rider detection (low cascading influence ⇒ uncooperative), incentive design for decentralized markets.
- **Federated as a special case**: the framework reduces to FL when the topology is a star (Algorithm 1).
- Empirical: ResNet-18 on CIFAR-10 with 16-node communication topology — DICE-E scores form interpretable spatial patterns matching the topology (Figure 1).

## Connections

- Generalizes centralized influence functions and [[sources/in-run-data-shapley|In-Run Shapley]] to graph topologies.
- Shares lineage with [[concepts/shapley-value]]-based federated valuation (citations to Wang et al. 2020 etc.) but doesn't reduce to it.
- Concept page: [[concepts/dice]] (created with this ingest).
- Concept page: [[concepts/decentralized-learning]] (created with this ingest).
- New thread: [[threads/federated-and-decentralized-attribution]] — federated and decentralized data attribution as its own subfield, with FedDQC, DICE, and the participant-amalgamation paper as members.

## Notes / open questions

- The curvature term is theoretically clean but expensive to compute. The paper acknowledges practical estimation tricks; how do they trade off against accuracy in practice?
- DICE values are *node-level*; how do you go from node value to per-data-point value? Does it require a centralized re-weighting at the node?
- Detecting free-riders via low DICE-E: false positives for honest-but-poorly-connected nodes? The cascade depends on topology, which the contributor doesn't fully control.
- Comparison to [[sources/in-run-data-shapley|In-Run Shapley]] in the limit of star topology + single training run — is there a clean correspondence?
- Smart-contract-based incentive mechanisms (cited from the related work as a baseline) — DICE could plug in but the authors don't go there.
