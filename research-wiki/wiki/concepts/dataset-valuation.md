---
type: concept
title: Dataset valuation (vs. data-point valuation)
created: 2026-05-05
updated: 2026-05-05
sources: [du-shapley, asymmetric-data-shapley, ipfl-model-market]
tags: [valuation, contribution, players]
---

# Dataset valuation

## The distinction

Two related but **distinct** problems live under the "data valuation" umbrella:

| Problem | What's a player? | Granularity | Use case |
|---|---|---|---|
| **Data valuation** | a single training point | per-point | curation, mislabel detection, copyright per-document |
| **Dataset valuation** | a contributor's whole dataset | per-contributor | data markets, federated participant compensation, multi-party sharing |

The naive reduction "dataset value = sum of point values" is **wrong**. [[sources/du-shapley|DU-Shapley]] gives the canonical counterexample:

- Datasets $D_1 = \{x_1\}$, $D_2 = \{x_2\}$, $D_3 = \{x_2, x_2\}$.
- Utility $u(D) = \mathbb{1}\{x_1, x_2 \in D\}$.
- Sum-of-points: $D_3$ valued at twice $D_2$.
- Dataset Shapley: $D_2 = D_3 = 1/6$, $D_1 = 2/3$ — equal and correct.

Reason: the utility function is highly non-linear in the data, so summing point-values doesn't reproduce dataset-level marginal contributions.

## Why this matters for the wiki

Many federated and market papers conflate the two problems. In the wiki:

- **Per-point** flavor: [[sources/in-run-data-shapley]], [[sources/data-banzhaf]], [[sources/datainf]], [[sources/koh-liang-influence-functions]], [[sources/ripple-shapley]].
- **Per-dataset / per-contributor** flavor: [[sources/du-shapley]], [[sources/gtg-shapley]], [[sources/space-participant-amalgamation]], [[sources/shapleyfl]], [[sources/game-of-gradients-sfedavg]], [[sources/asymmetric-data-shapley]] (in its FL example), [[sources/ipfl-model-market]].

## Methods specific to dataset valuation

- [[sources/du-shapley|DU-Shapley]]: exploits the fact that for many ML tasks, the utility depends only on a *scalar* function of the coalition (e.g., total samples), exponentially reducing utility evaluations.
- [[sources/asymmetric-data-shapley|ADS]]: extends Shapley to ordered groups of datasets (e.g., temporal arrival).
- Most federated-Shapley methods inherit from this side because the natural unit is a *client* not a *sample*.

## See also

- [[concepts/shapley-value]]
- [[concepts/data-shapley]]
- [[threads/dataset-vs-data-point-valuation]]
