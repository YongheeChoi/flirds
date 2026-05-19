---
type: thread
title: Data and model markets
created: 2026-05-05
updated: 2026-05-05
sources: [ipfl-model-market, asymmetric-data-shapley, distributionally-robust-data-valuation, data-banzhaf]
tags: [data-market, mechanism-design, incentives, federated-learning]
---

# Data and model markets

## The question

A data marketplace turns data attribution into a *commercial* problem. Beyond "compute a value," practitioners need:

- **Replication-robustness**: can a seller game the system by submitting duplicates?
- **Individual rationality (IR)**: is honest participation always at least as good as opting out?
- **Incentive compatibility (IC)**: is truthful reporting always optimal?
- **Validation-stable pricing**: do prices change if a buyer chooses a different evaluation set?
- **Trajectory-aware pricing**: in multi-stage pipelines, does compensation respect when the data was acquired?

Several wiki sources engage with these. The constraints they impose feed back into the choice of attribution method.

## What the wiki has

### [[sources/ipfl-model-market|iPFL]] — model markets at LLM scale

A market where participants buy/sell *models* (rather than raw data) in a personalized federated setting. Models a graph game with traders, buyers, sellers, attackers. Provides:

- IR + IC theoretical guarantees.
- Robustness against attackers (bounded influence).
- LLM-scale demonstration on Mistral, TinyLlama, Llama-2.

Different from data markets: the good being sold is a function of joint training, which makes cooperative-game pricing more natural.

### [[sources/asymmetric-data-shapley|ADS]] — symmetry-aware fair valuation

Doesn't design a market mechanism but provides the **valuation rule** that markets need:

- **Synthetic vs. original** data — addresses the duplication / augmentation pricing problem (Lemma 3.1).
- **Federated rounds** — values across-time contributions fairly without retraining counterfactuals.
- **Multi-stage LLM fine-tuning** — values stage-$t$ datasets conditional on the model state at $t-1$.

The core market-relevant insight: **classical Shapley's symmetry axiom is incompatible with replication-robustness**.

### [[sources/distributionally-robust-data-valuation|DRDV]] — buyer-side robust pricing

Doesn't design a market either, but addresses the **pricing-stability** concern: values shouldn't depend on which validation set the buyer chose. DRGE utility makes per-coalition utility robust to reasonable distribution shifts.

### [[sources/data-banzhaf|Data Banzhaf]] — robust ranking under noisy training

Tangentially relevant: noisy rankings make markets unreliable. Banzhaf's noise-robustness is a market-friendly property.

## How they fit together

A data/model market design has three layers:

```
Mechanism layer    — IR/IC/social-welfare guarantees      (iPFL)
Valuation layer    — fair, stable per-contributor values  (ADS, DRDV, Banzhaf)
Computation layer  — efficient estimators                  (federated-Shapley)
```

iPFL covers the mechanism layer at LLM scale but uses simpler valuation (graph-game contributions). ADS covers the valuation layer but isn't paired with an explicit mechanism. They could be combined.

## Open questions

- **iPFL with ADS valuations**: replace iPFL's contribution measure with ADS to get replication-robust + IR + IC + LLM-scale. Has not been done.
- **Pricing across personalized models**: in PFL, the same data has *different* values for different counterparties' models. How does a market price this consistently?
- **Adversarial market participants**: iPFL's attacker model is restricted. What about coalitions of dishonest sellers? Sybil attacks?
- **Real-time pricing**: [[sources/ripple-shapley|Ripple Shapley]]'s real-time pricing application is suggestive but doesn't engage with mechanism design. Combination needed.
- **Markets for open-source data**: when the data is publicly available, does any of this apply, or do we need a different framework (attribution-only without compensation)?

## Sources to ingest

- Agarwal et al. (2019), "A Marketplace for Data" — original replication-robustness criterion.
- Tian et al. on private + verifiable Shapley settlement (cited in [[sources/asymmetric-data-shapley|ADS]]).
- Recent surveys on Shapley-based market mechanisms (cited in [[sources/asymmetric-data-shapley|ADS]]).
