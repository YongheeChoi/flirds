---
type: concept
title: Federated learning (FL)
created: 2026-05-05
updated: 2026-05-19
sources: [feddqc, dice, gtg-shapley, game-of-gradients-sfedavg, shapleyfl, ripple-shapley, space-participant-amalgamation, ipfl-model-market, asymmetric-data-shapley, principled-federated-data-valuation, comfedsv, fldetector, fedcorr, fltrust, foolsgold, free-riders-fl-std-dagmm]
tags: [distributed-training, privacy, fedavg]
---

# Federated learning (FL)

## One-liner

A distributed training paradigm where multiple clients collaboratively train a shared model without sending their raw data to a central server. Each client sends only model updates (e.g., gradients); the server aggregates these updates round by round.

## Vanilla FedAvg

For $N$ clients with datasets $\mathcal{D}_n$:

1. Server broadcasts global model $\theta^{(t)}$.
2. Each client $n$ runs local SGD steps: $\theta_n^{(t)} \to \theta_n^{(t,k)}$.
3. Clients send $\Delta\theta_n^{(t)} = \theta_n^{(t,k)} - \theta^{(t)}$ to server.
4. Server aggregates: $\theta^{(t+1)} = \theta^{(t)} + \sum_n w_n \Delta\theta_n^{(t)}$ (weights $w_n$ usually proportional to $|\mathcal{D}_n|$).

Variants: FedProx (proximal regularization), FedYOGI / FedAdam (server-side adaptive optimizers), FedAvgM (momentum).

## Why FL is its own thing for data valuation

The wiki has many federated-attribution sources because the problem is genuinely different from centralized:

1. **No central data access**: server can't run Shapley-style retrainings on subsets of pooled data.
2. **Communication is the budget**, not compute alone — many methods are constrained by what fits in one round.
3. **Clients are players**, not just data points: contribution evaluation is often *participant-level*, not sample-level.
4. **Realized trajectory matters**: counterfactual model paths (different round orders, different participation) are infeasible to enumerate.
5. **Personalization** is increasingly common (each client gets its own model) — see [[concepts/personalized-fl]].

## Core problems in FL data valuation

- **Participant valuation** — who contributed how much across rounds? See [[sources/principled-federated-data-valuation|FedSV]], [[sources/comfedsv|ComFedSV]], [[sources/gtg-shapley]], [[sources/game-of-gradients-sfedavg|S-FedAvg]], [[sources/shapleyfl|ShapleyFL]], [[sources/space-participant-amalgamation|SPACE]], [[sources/ripple-shapley]].
- **Robust aggregation / malicious-client detection** — down-weight or remove poisoned, noisy, or free-riding clients. See [[sources/fltrust|FLTrust]], [[sources/foolsgold|FoolsGold]], [[sources/fldetector|FLDetector]], [[sources/fedcorr|FedCorr]], [[sources/free-riders-fl-std-dagmm|STD-DAGMM]]; synthesis in [[threads/noise-ood-malicious-client-separation]].
- **Data quality control on-device** — what does each client filter locally? See [[sources/feddqc|FedDQC]].
- **Influence cascade in decentralized topologies** — without a central server, influence propagates through the graph. See [[sources/dice|DICE]].
- **Fair compensation in markets** — pricing personalized models. See [[sources/ipfl-model-market|iPFL]].
- **Fair valuation across rounds** — respecting the realized trajectory. See [[sources/asymmetric-data-shapley|ADS]].

## Relationship to centralized data valuation

Most centralized methods *can* be adapted to FL but with trade-offs:

| Centralized method | FL adaptation challenge |
|---|---|
| Shapley over data points | $2^n$ in *clients* not points; per-coalition utility needs sub-model reconstruction or distillation |
| Influence functions | Hessian / gradient access requires server-side coordination; on heterogeneous clients, gradient signal is noisy ([[sources/feddqc]] reports DataInf failure on Fed-WildChat) |
| In-Run Shapley | the "single training run" is itself a multi-round federated trajectory — Ripple Shapley's contribution |

## See also

- [[concepts/decentralized-learning]] — fully P2P, no central server.
- [[concepts/personalized-fl]] — each client ends with its own model.
- [[concepts/federated-shapley]] — the family of Shapley-based participant valuation methods.
- [[threads/federated-and-decentralized-attribution]]

## Where it appears in the wiki

(Heavy presence — see the sources list in the frontmatter.)
