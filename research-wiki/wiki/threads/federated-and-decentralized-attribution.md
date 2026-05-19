---
type: thread
title: Data attribution in federated and decentralized settings
created: 2026-05-05
updated: 2026-05-19
sources: [feddqc, dice, gtg-shapley, game-of-gradients-sfedavg, shapleyfl, ripple-shapley, space-participant-amalgamation, ipfl-model-market, asymmetric-data-shapley, datainf, principled-federated-data-valuation, comfedsv, fldetector, fedcorr, fltrust, foolsgold, free-riders-fl-std-dagmm]
tags: [federated-learning, decentralized-learning, attribution, synthesis]
---

# Data attribution in federated and decentralized settings

## The question

When data lives across many parties — federated (with a server) or fully decentralized (peer-to-peer graph) — who contributed how much, and how do you compute it without violating the constraints (no central data, communication budget, realized trajectory)?

This thread is the **most populated subfield** in our wiki: 9 sources directly engage with it, and several others touch it tangentially. The literature has fragmented into multiple sub-questions; this page synthesizes them.

## Five sub-questions, by axis

### 1. Granularity: client-level vs. sample-level

Most federated-attribution work treats each **client** (dataset holder) as a player. Some recent work descends to per-**sample** values within FL — which is closer to centralized [[concepts/data-shapley]] but much harder.

| Granularity | Methods |
|---|---|
| Client-level | [[sources/principled-federated-data-valuation|FedSV (Wang et al.)]], [[sources/comfedsv|ComFedSV]], [[sources/gtg-shapley|GTG-Shapley]], [[sources/game-of-gradients-sfedavg|S-FedAvg]], [[sources/shapleyfl|ShapleyFL]], [[sources/space-participant-amalgamation|SPACE]] |
| Sample-level (federated, single-run) | [[sources/ripple-shapley|Ripple Shapley]] |

Ripple Shapley is the first to bridge to sample-level federation — see [[threads/retraining-vs-in-run-attribution]] for its lineage from In-Run Shapley.

### 2. Number of rounds: multi-round vs. single-round/run

| Mode | Mechanism | Methods |
|---|---|---|
| Multi-round retraining | Reconstruct sub-models from logged gradients | [[sources/gtg-shapley|GTG-Shapley]] |
| Multi-round Shapley aggregation | Per-round Shapley + aggregation | [[sources/principled-federated-data-valuation|FedSV]], [[sources/shapleyfl|ShapleyFL]], [[sources/game-of-gradients-sfedavg|S-FedAvg]] |
| Multi-round + utility-matrix completion | Impute unobserved (round, coalition) entries | [[sources/comfedsv|ComFedSV]] |
| **Single-round** (one comm. round) | Knowledge distillation + prototype eval | [[sources/space-participant-amalgamation|SPACE]] |
| **Single-run** (one full FL run) | Jacobian-chain propagation | [[sources/ripple-shapley|Ripple Shapley]] |

The trajectory: 2021 (per-round) → 2022 (gradient reconstruction across rounds) → 2023 (single-round via distillation) → 2026 (single-run sample-level via in-run propagation).

### 3. Goal: valuation vs. quality control vs. robustness

These three goals get conflated in the federated literature:

- **Valuation / fair compensation** — the contributor-pricing problem. [[sources/principled-federated-data-valuation|FedSV]] (origin), [[sources/comfedsv|ComFedSV]], [[sources/gtg-shapley|GTG]], [[sources/space-participant-amalgamation|SPACE]], [[sources/ripple-shapley|Ripple]], [[sources/asymmetric-data-shapley|ADS]] (FL example), [[sources/ipfl-model-market|iPFL]].
- **Data quality control** — pre-filter low-quality samples on-device. [[sources/feddqc|FedDQC]] (with IRA). Doesn't try to compute per-client values.
- **Robust aggregation / client trust** — downweight, detect, or prune untrustworthy clients. [[sources/game-of-gradients-sfedavg|S-FedAvg]], [[sources/shapleyfl|ShapleyFL]] (partially), and the dedicated robustness family [[sources/fltrust|FLTrust]] / [[sources/foolsgold|FoolsGold]] / [[sources/fldetector|FLDetector]] / [[sources/fedcorr|FedCorr]] / [[sources/free-riders-fl-std-dagmm|STD-DAGMM]] — full synthesis (and why these are the robustness-side complement to this thread) in [[threads/noise-ood-malicious-client-separation]].

A useful sharpening: see [[concepts/data-quality-control]] vs. [[concepts/shapley-value]] — these are different objects, and the wiki should resist the conflation.

### 4. Topology: star (FL) vs. arbitrary graph (decentralized)

Most of the wiki sits in star topology = FL with a server. [[sources/dice|DICE]] is the outlier — fully decentralized, and the only source that explicitly models *cascading* influence through graph structure.

DICE shows that decentralized influence factorizes into **data × topological position × intermediate curvature** — a useful conceptual export back into FL: in a star, the topology factor degenerates and we're left with data × curvature, recovering classical attribution.

### 5. Faithfulness: counterfactual retraining vs. realized trajectory

Classical [[concepts/data-shapley|Data Shapley]] requires counterfactual retrainings — averaging over training histories that didn't happen. In FL this is doubly bad: it's expensive *and* it's conceptually misaligned (you sold a buyer the *realized* model, not an average over hypotheticals).

The trend, codified by [[sources/asymmetric-data-shapley|ADS]]: **anchor evaluation to the realized model state**. This is shared by [[sources/in-run-data-shapley|In-Run Shapley]] (centralized) and [[sources/ripple-shapley|Ripple Shapley]] (federated). [[sources/space-participant-amalgamation|SPACE]] takes a different route — evaluate at the *end* of a single distillation step, not along the trajectory.

## Cross-method comparison

| Method | Granularity | Rounds | Goal | Trajectory-faithful? |
|---|---|---|---|---|
| [[sources/principled-federated-data-valuation\|FedSV (Wang et al.)]] | client | multi | valuation | per-round retrain-surrogate |
| [[sources/comfedsv\|ComFedSV]] | client | multi | fair valuation | retrain-free but completion-imputed |
| [[sources/gtg-shapley\|GTG-Shapley]] | client | multi | valuation | yes (sub-model reconstruction) |
| [[sources/game-of-gradients-sfedavg\|S-FedAvg]] | client | per-round | robustness | per-round only |
| [[sources/shapleyfl\|ShapleyFL]] | client | multi | robustness + valuation | surrogate |
| [[sources/space-participant-amalgamation\|SPACE]] | client | single-round | valuation | end-state only |
| [[sources/ripple-shapley\|Ripple Shapley]] | **sample** | single-run | valuation | yes (Jacobian chain) |
| [[sources/feddqc\|FedDQC]] | sample | per-round | quality | n/a (no Shapley) |
| [[sources/dice\|DICE]] | node | n/a | influence cascade | yes |
| [[sources/asymmetric-data-shapley\|ADS]] | client/group | multi | fair valuation | yes (state-conditioned) |
| [[sources/ipfl-model-market\|iPFL]] | client | multi | market mechanism | n/a (game-theoretic) |

## Where the field is going

1. **Sample-level + single-run** is the new frontier — [[sources/ripple-shapley|Ripple Shapley]] is the first; expect more.
2. **Asymmetric / state-conditioned valuation** is being adopted as the right framing — symmetry across rounds is broken.
3. **Gradient-based attribution methods are brittle on real-world heterogeneous FL** — [[sources/feddqc|FedDQC]] reports DataInf failing on Fed-WildChat. Open question whether other gradient methods (LoGra, EKFAC) survive.
4. **Decentralized (non-FL) attribution is sparse** — only [[sources/dice|DICE]] in the wiki. Worth more sources.

## Open questions

- Can [[sources/dice|DICE]]'s influence-cascade decomposition be combined with [[sources/ripple-shapley|Ripple Shapley]]'s Jacobian chain for fully decentralized sample-level attribution?
- Why exactly do gradient-based attribution methods fail on heterogeneous FL? Is it specifically the swap-inverse-and-average step in [[sources/datainf|DataInf]], or is it a more general issue with gradients across non-IID clients?
- For data markets: which method best supports *replication-robust* pricing? [[sources/asymmetric-data-shapley|ADS]] makes this an axiom; the federated-Shapley line largely doesn't.
- How does single-round Shapley ([[sources/space-participant-amalgamation|SPACE]]) compare to single-run sample-level Shapley ([[sources/ripple-shapley|Ripple Shapley]]) on the same data? They measure different objects but practitioners likely want a unified comparison.
- The right unit (client / group / sample) depends on the application. Can the wiki articulate when each is correct?

## Sources to look for

- ~~Wang et al. "Principled Federated Shapley" (2020)~~ — **ingested 2026-05-19** → [[sources/principled-federated-data-valuation]] (FedSV, the origin); fairness-fixing successor ComFedSV → [[sources/comfedsv]].
- VFL data valuation work (Han 2025 mentioned in [[sources/asymmetric-data-shapley|ADS]]'s related work).
- Federated unlearning + Shapley updates (sharded Shapley, mentioned in [[sources/asymmetric-data-shapley|ADS]]).
- Decentralized learning with blockchain-based incentive mechanisms.
