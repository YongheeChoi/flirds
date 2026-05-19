---
type: thread
title: Dataset vs. data-point valuation
created: 2026-05-05
updated: 2026-05-05
sources: [du-shapley, asymmetric-data-shapley, gtg-shapley, space-participant-amalgamation, ghorbani-zou-data-shapley, ripple-shapley, in-run-data-shapley]
tags: [granularity, dataset-valuation, federated, dataset-vs-point]
---

# Dataset vs. data-point valuation

## The question

"Data valuation" actually covers two distinct problems:

| Problem | Player | Granularity | Typical use |
|---|---|---|---|
| **Data-point valuation** | a single training point | per-point | curation, mislabel detection, copyright per-document |
| **Dataset valuation** | a contributor's whole dataset | per-contributor | data markets, federated participant compensation, multi-party sharing |

Confusing the two leads to wrong rankings and wrong economic outcomes. This thread tracks the distinction and the methods on each side.

## Why naive sum-of-points fails

[[sources/du-shapley|DU-Shapley]]'s canonical counterexample:

- $D_1 = \{x_1\}$, $D_2 = \{x_2\}$, $D_3 = \{x_2, x_2\}$.
- Utility $u(D) = \mathbb{1}\{x_1, x_2 \in D\}$.

Per-point Shapley assigns $x_2$ a fixed value, so summing across $D_2$ vs. $D_3$ gives $D_3$ twice the value of $D_2$. But as **datasets**, $D_2$ and $D_3$ should be valued equally — they contribute the same set of distinct samples to the coalition.

Why: the utility is non-linear in the data, so summing point-values doesn't reproduce dataset-level marginal contributions. **Granularity is not a free choice** — it's tied to what the application is asking.

## Where each granularity is appropriate

### Per-point

- **Curation** — drop low-value or negative-value points to speed convergence (e.g., [[sources/in-run-data-shapley|In-Run Shapley]] flagging 16% of Pile as net-negative).
- **Mislabel detection** — high self-influence ⇒ likely mislabel ([[sources/koh-liang-influence-functions]]).
- **Copyright attribution** — the unit of legal interest is often a single document or training instance ([[sources/in-run-data-shapley]]'s analysis of training-text influence on paraphrased validation outputs).
- **Data quality control on-device** — per-sample IRA scoring for filtering ([[sources/feddqc]]).

### Per-dataset / per-contributor

- **Data markets** — buyers acquire datasets, not individual points; pricing per-contributor.
- **Federated participant compensation** — fair payment for contributing a client's data over multiple rounds.
- **Collaborative ML / multi-party data sharing** — incentive design treats each owner as a coalition player.
- **Replication-robustness** — only meaningful at the dataset level.

## Methods by granularity

### Data-point methods (centralized)

- [[sources/koh-liang-influence-functions]] — per-sample influence.
- [[sources/datainf]], [[sources/logix]], [[sources/trak]] — efficient per-sample influence at LLM scale.
- [[sources/in-run-data-shapley]] — per-sample Shapley in one training run.
- [[sources/ghorbani-zou-data-shapley]] — original Data Shapley (per-point).
- [[sources/data-banzhaf]] — per-point Banzhaf.

### Dataset / contributor methods

- [[sources/du-shapley]] — explicitly per-dataset Shapley with structure-aware approximation.
- [[sources/asymmetric-data-shapley|ADS]] — per-group Shapley (groups can be datasets, FL rounds, fine-tuning stages).
- Most federated-Shapley methods inherit dataset-granularity because clients = dataset holders:
  - [[sources/gtg-shapley|GTG-Shapley]], [[sources/game-of-gradients-sfedavg|S-FedAvg]], [[sources/shapleyfl|ShapleyFL]], [[sources/space-participant-amalgamation|SPACE]].

### Bridging methods

- [[sources/ripple-shapley|Ripple Shapley]] — first **sample-level** federated Shapley. Closes the gap between FL (typically per-client) and centralized (per-sample).

## Why dataset valuation is its own science

Two reasons the field's progressed differently on the dataset side:

1. **Combinatorial structure**: per-dataset Shapley has fewer "players" (tens of contributors vs. millions of points). Exact computation is closer to feasible; the structure is more amenable to closed forms (e.g. [[sources/du-shapley|DU-Shapley]]'s scalar-utility reduction).
2. **Application bias**: data-market and FL-incentive applications dominate. These applications care about contribution-as-revenue-share, replication-robustness, and IR/IC properties — questions that don't even arise at the per-point level.

## Open questions

- **Hybrid granularity**: a federated buyer might want to know not just "client A's value" but "which samples within client A drove that value." [[sources/ripple-shapley|Ripple Shapley]] does sample-level FL, but the *aggregation* back to client-level is an interesting derived question.
- **Replication-robustness at the point level**: does it even make sense? Two identical points within a single dataset don't have the same "duplication" problem as duplicate datasets in a market.
- **DU-Shapley + Asymmetric Shapley combination**: dataset-level + ordered-groups. Useful for FL rounds where contributions are ordered AND the utility has scalar structure.
- **Translating insights across granularities**: [[sources/in-run-data-shapley|In-Run Shapley]]'s "stage-dependent contribution" — does the same effect apply at the dataset level (contributors valued differently early vs. late in training)? [[sources/asymmetric-data-shapley|ADS]] hints at yes.

## Sources to look for

- "Distributional Data Shapley" (Ghorbani 2020) — bridges between per-point and dataset-level via distribution-based valuation.
- "Sharded Shapley" (cited in [[sources/asymmetric-data-shapley|ADS]]) — supports machine unlearning by maintaining valuations under data removal.
- VFL data valuation work (Han 2025).
