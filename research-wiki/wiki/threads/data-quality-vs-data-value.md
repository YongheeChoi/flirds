---
type: thread
title: Data quality vs. data value
created: 2026-05-05
updated: 2026-05-05
sources: [feddqc, ghorbani-zou-data-shapley, in-run-data-shapley, datainf]
tags: [data-quality, data-value, distinction, federated-learning]
---

# Data quality vs. data value

## The distinction

Two concepts that the literature regularly conflates:

- **Data quality** — is this point intrinsically good (well-labeled, well-aligned, low-noise, easy to learn from)? A *property of the point itself*, mostly. Examples: perplexity, IFD, NUGGETS, [[concepts/instruction-response-alignment|IRA]] from [[sources/feddqc|FedDQC]].
- **Data value / contribution** — how much did this point contribute to the trained model's behavior? A *relational* quantity that depends on the rest of the dataset. Examples: [[concepts/shapley-value|Shapley]], [[concepts/influence-function|influence functions]], [[concepts/banzhaf-value|Banzhaf]].

A point can be high-quality but low-value (it's a redundant good example), or low-quality but high-value (a noisy adversarial example pushing the model to robustify in a useful direction).

## Why the wiki should keep them distinct

Confusion between the two leads to wrong decisions:

| Mistake | What happens |
|---|---|
| Use a quality metric for valuation | redundant high-quality samples get over-priced |
| Use a value metric for filtering | high-magnitude-influence noisy samples get retained |
| Sum quality scores for dataset valuation | non-linearity pathologies (see [[threads/dataset-vs-data-point-valuation]]) |

## What each is good for

### Data-quality metrics

- **Pre-filtering** before training (remove low-quality before they hurt).
- **Curriculum training** ([[sources/feddqc|FedDQC]]'s easy-to-hard hierarchy).
- **Privacy-preserving on-device scoring** in FL (no aggregation needed, no model gradient required).

### Data-value metrics

- **Post-hoc attribution** (which training data caused this prediction).
- **Mislabel detection** (negative-value points often = mislabels).
- **Compensation in markets** (per-contributor value).
- **Curation for next-run training** (drop net-negative-value points).

## Where the conflation shows up

[[sources/feddqc|FedDQC]] is a clean case study: it explicitly takes the position that data **quality** scoring (IRA) outperforms data **value / attribution** scoring ([[sources/datainf|DataInf]]) in real-world heterogeneous FL — but only because they're solving different problems. IRA is a quality scorer used for filtering; DataInf is a value attributor. Comparing them on a filtering benchmark is unfair to DataInf, but the paper's empirical observation that gradient-based attribution becomes unreliable on heterogeneous client data is independently important.

[[sources/in-run-data-shapley|In-Run Shapley]]'s finding that ~16% of the Pile has negative Shapley value is a *value* statement — these points actively hurt training. A *quality* statement (these points are intrinsically bad) would require a different methodology and might not agree.

## Empirical relationships

- High-quality points often have high *positive* contribution but not always.
- Low-quality (noisy) points often have negative or near-zero contribution.
- Redundant high-quality points have high quality but low marginal contribution.
- Adversarial points engineered to maximize a specific test-point's loss have high *magnitude* contribution (positive or negative) regardless of quality.

## Open questions

- **Are there cases where quality and value diverge sharply?** Worth a dedicated empirical study on common LLM datasets.
- **Quality + value joint scoring** for FL: filter on quality on-device, then compute value server-side on the survivors. [[sources/feddqc|FedDQC]] does the first half well; the second half (federated valuation post-filtering) is open.
- **Information-theoretic unification**: IRA approximates mutual information; influence functions approximate parameter perturbation effects. Is there a regime where they coincide?
- **Quality-aware curation as a substitute for value-based curation at scale**: when you can't run an expensive value method, does quality-based filtering give similar curation gains?

## Sources to ingest

- Comparative empirical studies of quality scoring vs. attribution scoring on LLM data.
- IFD, NUGGETS, AlpaGasus, DEITA papers — the heuristic-quality side that [[sources/feddqc|FedDQC]] uses as baselines.
