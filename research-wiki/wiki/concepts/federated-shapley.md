---
type: concept
title: Federated Shapley
created: 2026-05-05
updated: 2026-05-19
sources: [principled-federated-data-valuation, comfedsv, gtg-shapley, game-of-gradients-sfedavg, shapleyfl, ripple-shapley, space-participant-amalgamation]
tags: [federated-learning, shapley, participant-valuation]
---

# Federated Shapley

The family of [[concepts/shapley-value|Shapley]]-based methods for valuing **participants** (clients) in [[concepts/federated-learning|federated learning]]. The shared challenge: standard Shapley requires evaluating utility on arbitrary client coalitions, but FL constraints (no central data, communication budget, realized trajectory) make this expensive or infeasible.

The origin is [[sources/principled-federated-data-valuation|Wang et al. 2020 (FedSV)]], which defined the per-round, order-aware federated Shapley value every later method specializes, accelerates, or de-biases.

## Methods in the wiki

| Paper | Core trick | Granularity | Rounds |
|---|---|---|---|
| [[sources/principled-federated-data-valuation|FedSV (Wang et al.)]] (2020) | **Origin.** Per-round Shapley over the cohort; sub-model = apply uploaded updates onto round-start model + eval; permutation / group-testing estimators | client | multi-round |
| [[sources/comfedsv|ComFedSV]] (2022) | FedSV + low-rank utility-matrix completion for unobserved (round, coalition) entries; ε-Shapley-fairness | client | multi-round |
| [[sources/gtg-shapley|GTG-Shapley]] (2022) | Reconstruct sub-models from logged gradients; guided MC; truncation | client | multi-round |
| [[sources/game-of-gradients-sfedavg|S-FedAvg]] (2021) | Per-round Shapley over gradient coalitions; client pruning | client | per-round |
| [[sources/shapleyfl|ShapleyFL]] (2023) | Surrogate federated Shapley aggregated across rounds; importance-sampling client selection; difference estimator | client | multi-round |
| [[sources/space-participant-amalgamation|SPACE]] (2023) | Knowledge amalgamation + prototype-based evaluation in **one round** | client | single-round |
| [[sources/ripple-shapley|Ripple Shapley]] (2026) | Drop term + Jacobian-chain ripple term; low-rank subspace | **sample** | single-run |

## Cross-cutting choices

- **Granularity**: most are client-level; Ripple Shapley is the first sample-level federated Shapley.
- **Trajectory faithfulness**: GTG, ShapleyFL, Ripple respect the realized trajectory; SPACE evaluates at the end after distillation; S-FedAvg is per-round. FedSV is a per-round retrain-surrogate (counterfactual sub-model re-eval); ComFedSV is retrain-free but *imputes* missing coalitions by completion — neither is in-run in the [[flirds|Flirds]] sense (closed-form 1st+2nd Taylor on the realized $\Delta w_k$).
- **Validation set**: SPACE eliminates dependence on a held-out validation set; others assume one is available at the server.

## Where it sits

This is the *federated subset* of the broader [[concepts/shapley-value]] / [[concepts/data-shapley]] literature. It's also the natural meeting point between [[concepts/dataset-valuation]] (the players are clients = dataset holders) and federated-learning-specific machinery (gradient logs, sub-model reconstruction, etc.).

## See also

- [[concepts/shapley-value]]
- [[concepts/federated-learning]]
- [[concepts/dataset-valuation]]
- [[threads/federated-and-decentralized-attribution]]
