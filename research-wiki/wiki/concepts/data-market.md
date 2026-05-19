---
type: concept
title: Data market (and model market)
created: 2026-05-05
updated: 2026-05-05
sources: [asymmetric-data-shapley, ipfl-model-market, distributionally-robust-data-valuation]
tags: [economics, incentives, mechanism-design]
---

# Data market / model market

A market mechanism for buying, selling, and pricing data (or trained models) for ML, where pricing reflects each contributor's marginal value to downstream model performance.

## Why it's part of the wiki

Several wiki sources treat data valuation as a *means* — the end is fair compensation in a marketplace. Their concerns aren't just "compute a value" but "design a mechanism that's truthful, individually rational, and replication-robust."

| Concern | Why it matters | Where it appears |
|---|---|---|
| **Replication-robustness** | A seller shouldn't be able to inflate revenue by duplicating their data | [[sources/asymmetric-data-shapley|ADS]], [[sources/data-banzhaf]] (indirectly via efficiency) |
| **Individual rationality (IR)** | Honest participants must be at least as well off as not participating | [[sources/ipfl-model-market|iPFL]] |
| **Incentive compatibility (IC)** | Truthful reporting of data quality is the optimal strategy | [[sources/ipfl-model-market|iPFL]] |
| **Validation-set independence** | Prices stable under buyer-side validation choice | [[sources/distributionally-robust-data-valuation|DRDV]] |
| **Trajectory-aware pricing** | Multi-stage / sequential markets price by realized model state | [[sources/asymmetric-data-shapley|ADS]] (multi-stage LLM fine-tuning) |
| **Personalization** | Buyers can want different models from the same data pool | [[sources/ipfl-model-market|iPFL]] |

## Data market vs. model market

- **Data market**: contributors sell datasets, buyers train their own models.
- **Model market**: contributors share data into a federated training process; buyers buy access to trained (often personalized) models. iPFL is in this category.

The two modes have different incentive structures — model markets can use cooperative-game pricing more naturally because the "good" being sold is a function of the joint training.

## See also

- [[concepts/incentive-compatibility]]
- [[concepts/personalized-fl]]
- [[concepts/replication-robustness]]
- [[threads/data-and-model-markets]]
