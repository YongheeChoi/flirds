---
type: index
title: Index
created: 2026-05-05
updated: 2026-05-27
---

# Index

A catalog of every page in the wiki. Updated on every ingest. See [[overview]] for the synthesis hub and [[log]] for chronological history.

## Overview & meta

- [[overview]] — top-level hub of the wiki
- [[log]] — chronological log of ingests, queries, lint passes
- [[../CLAUDE]] — schema and workflows

## Project

- [[flirds]] — Flirds (Federated Learning + In-Run Data Shapley) project state: locked design decisions, resolved questions, experiment plan
- [[flirds-protocol]] — implementation & reporting protocol (precision, seeds, statistical reporting, oracle separation, sanity gates, run logging, Phase 0)
- [[flirds-implementation-plan]] — **session handoff document**: start here when beginning an implementation session. 4-phase task ordering (Phase 0 CNN reproduction → Phase 1 Flirds at 1B → Phase 2 full baseline + 3B/7B → Phase 3 matrix execution); 9 still-open implementation decisions with options + criteria + recommendations; pre-implementation checklist; pointer table

## Sources (30)

### Foundations
- [[sources/koh-liang-influence-functions]] — Koh & Liang 2017; foundational influence functions for ML.
- [[sources/ghorbani-zou-data-shapley]] — Ghorbani & Zou 2019; foundational Data Shapley.

### Semivalue family
- [[sources/data-banzhaf]] — Banzhaf value's noise robustness; MSR estimator.
- [[sources/in-run-data-shapley]] — model-level Shapley via per-step Taylor expansion.
- [[sources/asymmetric-data-shapley]] — Shapley without symmetry; structure-aware.
- [[sources/du-shapley]] — discrete-uniform Shapley for dataset valuation.

### Influence functions at LLM scale
- [[sources/datainf]] — closed-form influence approximation for LoRA-tuned LLMs.
- [[sources/trak]] — eNTK linearization + random projection; LDS benchmark.
- [[sources/logix]] — Kronecker-structured low-rank projection + Logix software.
- [[sources/grosse-llm-influence]] — **upper-bound anchor**: EK-FAC IF at 52B (Anthropic, 2023); TF-IDF + query batching; reframes target as [[concepts/proximal-bregman-response|PBRF]].

### Centralized data selection for LLMs (2024)
- [[sources/less]] — TracIn-trajectory IF + Adam-Γ + cosine + LoRA + JL projection; Llama-2-7B/13B + Mistral-7B instruction tuning; the **closest centralized comparator** to [[flirds|Flirds]].
- [[sources/mates]] — locally-probed one-step Δloss + BERT-base data-influence model; Pythia 410M–1B pretraining; backs the 1B-primary decision.
- [[sources/dsdm]] — linear datamodels via [[sources/trak|TRAK]] + bottom-$k$ selection; the **Datamodels → LLM bridge**.

### Federated learning — valuation
- [[sources/principled-federated-data-valuation]] — **FedSV (Wang et al. 2020)**: the origin of federated Shapley; per-round, order-aware.
- [[sources/comfedsv]] — ComFedSV: FedSV + low-rank utility-matrix completion fixes partial-participation asymmetry.
- [[sources/gtg-shapley]] — guided MC + sub-model reconstruction for federated Shapley.
- [[sources/game-of-gradients-sfedavg]] — Shapley-weighted FedAvg + client pruning.
- [[sources/shapleyfl]] — surrogate federated Shapley with importance-sampling.
- [[sources/space-participant-amalgamation]] — single-round federated Shapley via knowledge distillation.
- [[sources/ripple-shapley]] — sample-level, single-run federated Shapley with Jacobian propagation.
- [[sources/feddqc]] — IRA-based on-device data quality control + hierarchical training.
- [[sources/dice]] — influence cascade in fully decentralized learning.
- [[sources/rfedlr]] — federated LoRA robustness (peripheral; stub).

### Federated learning — robustness / detection (Flirds baselines)
- [[sources/fldetector]] — temporal update-consistency detector (Cauchy-MVT + L-BFGS); KDD 2022.
- [[sources/fedcorr]] — multi-stage FL label-noise correction via prediction-subspace LID; CVPR 2022.
- [[sources/fltrust]] — Byzantine-robust FL via server-root trusted-cosine trust bootstrapping; NDSS 2021.
- [[sources/foolsgold]] — Sybil-poisoning defense via cross-client gradient similarity; RAID 2020.
- [[sources/free-riders-fl-std-dagmm]] — free-rider attacks + STD-DAGMM anomaly detector; Lin et al. 2019.

### Data markets / utility design
- [[sources/distributionally-robust-data-valuation]] — DRGE utility replaces validation-set dependence.
- [[sources/ipfl-model-market]] — graphical-game model market for personalized FL.

## Concepts (27)

### Foundational semivalue framework
- [[concepts/shapley-value]] — the unique fair allocation; four axioms.
- [[concepts/banzhaf-value]] — equal weight per subset; sacrifices efficiency for noise robustness.
- [[concepts/leave-one-out]] — degenerate semivalue; cheapest baseline.
- [[concepts/data-shapley]] — Shapley applied to ML training points.
- [[concepts/in-run-data-shapley]] — per-step model-level Shapley.
- [[concepts/semivalue]] — the framework that unifies Shapley/Banzhaf/LOO/Beta/CS.

### Influence-function family
- [[concepts/influence-function]] — gradient-based attribution; the spine of half the wiki.
- [[concepts/datainf]] — closed-form approximation for LoRA.
- [[concepts/logra]] — Kronecker-structured low-rank projection.
- [[concepts/logix]] — PyTorch-hooks-based software for LLM influence.
- [[concepts/trak]] — eNTK linearization + projection.
- [[concepts/datamodels]] — retraining-based gold standard.
- [[concepts/linear-datamodeling-score]] — the standard counterfactual benchmark.
- [[concepts/ekfac]] — Eigenvalue-corrected Kronecker-factored Hessian; the IHVP backbone scaling IF to 52B.
- [[concepts/proximal-bregman-response]] — what modern IF actually computes on deep nets (Bae 2022a → Grosse 2023 adoption).

### Federated / decentralized
- [[concepts/federated-learning]] — distributed training with central server.
- [[concepts/decentralized-learning]] — peer-to-peer; no server.
- [[concepts/personalized-fl]] — each client gets its own model.
- [[concepts/dice]] — first decentralized influence framework.
- [[concepts/feddqc]] — federated instruction-tuning data quality control.
- [[concepts/federated-shapley]] — family of Shapley-based participant valuation.
- [[concepts/gtg-shapley]], [[concepts/s-fedavg]], [[concepts/shapleyfl]], [[concepts/space]], [[concepts/ripple-shapley]] — specific federated methods.

### Special methods / utility / market
- [[concepts/asymmetric-data-shapley]] — Shapley without symmetry.
- [[concepts/du-shapley]] — structure-aware Shapley for dataset valuation.
- [[concepts/dataset-valuation]] — per-contributor vs. per-point.
- [[concepts/drge-utility]] — distributionally robust utility.
- [[concepts/data-market]] — economics + IR/IC/replication-robustness.
- [[concepts/replication-robustness]] — anti-duplication property.
- [[concepts/data-quality-control]] — quality scoring distinct from valuation.
- [[concepts/lora]] — parameter-efficient fine-tuning that recurs across many sources.

## Threads (11)

> Cross-cutting research threads. The most valuable pages — they synthesize across multiple sources.

- [[threads/retraining-vs-in-run-attribution]] — algorithm-level vs. model-level; trajectory-summed / end-state-local / one-step-probe sub-flavors.
- [[threads/robustness-to-stochastic-training]] — three responses to SGD noise (Banzhaf, in-run, DRDV).
- [[threads/federated-and-decentralized-attribution]] — the big synthesis (9+ sources).
- [[threads/influence-functions-at-llm-scale]] — DataInf, TRAK, LoGra, EK-FAC at 52B, LESS, MATES; iHVP and per-sample-gradient bottlenecks; 2024 Hessian-free wave.
- [[threads/data-selection-for-llms]] — LESS / MATES / DsDm synthesis: similarity ≠ value, 2× compute multiplier, reference-set choice.
- [[threads/symmetry-and-asymmetry-axioms]] — when symmetry breaks, three responses.
- [[threads/dataset-vs-data-point-valuation]] — granularity choice; why summing point values is wrong.
- [[threads/utility-function-design]] — the third axis: per-step / DRGE / prototype / few-shot / task-natural / target-mix.
- [[threads/data-and-model-markets]] — IR/IC/replication-robustness in commercial contexts.
- [[threads/data-quality-vs-data-value]] — quality scoring vs. attribution; why distinct; DsDm's "similarity ≠ value" finding.
- [[threads/noise-ood-malicious-client-separation]] — FL robustness-side prior art (FLDetector, FoolsGold, FLTrust, FedCorr, STD-DAGMM); backs Flirds' deferred-limitation recast + surviving detection benchmarks.

## Raw materials inventory

Snapshot of `raw/`. Last refreshed: 2026-05-22.

### Papers — `raw/papers/flirds/`

| File | Status |
|------|--------|
| `1703.04730v3.pdf` | **ingested** → [[sources/koh-liang-influence-functions]] |
| `1904.02868v2.pdf` | **ingested** → [[sources/ghorbani-zou-data-shapley]] |
| `2109.02053v1.pdf` | **ingested** → [[sources/gtg-shapley]] |
| `2303.14186v2.pdf` | **ingested** → [[sources/trak]] |
| `2308.03296v1.pdf` | **ingested** → [[sources/grosse-llm-influence]] |
| `2401.12926v1.pdf` | **ingested** → [[sources/dsdm]] |
| `2402.04333v3.pdf` | **ingested** → [[sources/less]] |
| `2406.06046v2.pdf` | **ingested** → [[sources/mates]] |
| `2009.06192v1.pdf` | **ingested** → [[sources/principled-federated-data-valuation]] |
| `2109.09046v3.pdf` | **ingested** → [[sources/comfedsv]] |
| `2207.09209v4.pdf` | **ingested** → [[sources/fldetector]] |
| `2204.04677v1.pdf` | **ingested** → [[sources/fedcorr]] |
| `2012.13995v3.pdf` | **ingested** → [[sources/fltrust]] |
| `1808.04866v5.pdf` | **ingested** → [[sources/foolsgold]] |
| `1911.12560v1.pdf` | **ingested** → [[sources/free-riders-fl-std-dagmm]] |
| `17093-Article Text-…2021…` | **ingested** → [[sources/game-of-gradients-sfedavg]] |
| `21101_Towards_Robust_Parameter.pdf` | **ingested (stub)** → [[sources/rfedlr]] |
| `3580305.3599500.pdf` | **ingested** → [[sources/shapleyfl]] |
| `40034-Article Text-…2026…` | **ingested** → [[sources/ripple-shapley]] |
| `5027_Distributionally_Robust_D.pdf` | **ingested** → [[sources/distributionally-robust-data-valuation]] |
| `978-981-96-1525-4.pdf` | **skipped** — ICA3PP 2024 proceedings (44MB book; off-topic as a single source) |
| `s41467-025-62959-5.pdf` | **ingested** → [[sources/ipfl-model-market]] |
| `NeurIPS-2023-…participant-amalgamation…` | **ingested** → [[sources/space-participant-amalgamation]] |
| `Data Banzhaf_…md` | **ingested** → [[sources/data-banzhaf]] |
| `Data Shapley in One Training Run.md` | **ingested** → [[sources/in-run-data-shapley]] |
| `DICE_…md` | **ingested** → [[sources/dice]] |
| `DU-Shapley_…md` | **ingested** → [[sources/du-shapley]] |
| `DataInf_…md` | **ingested** → [[sources/datainf]] |
| `FedDQC_…md` | **ingested** → [[sources/feddqc]] |
| `Rethinking Data Value_ Asymmetric Data Shapley_…md` | **ingested** → [[sources/asymmetric-data-shapley]] |
| `What is Your Data Worth to GPT_…md` | **ingested** → [[sources/logix]] |

30 of 31 raw papers ingested (+4 on 2026-05-22: Grosse et al. 2023, LESS, MATES, DsDm). ICA3PP book skipped as a non-source (per-chapter ingestion possible if Yonghee identifies a relevant chapter).

### Conversations — `raw/conversations/`

- `meta/2026-05-05-wiki-bootstrap.md` — bootstrap conversation (this session).
- `flirds/conversation1.md` — IRDS → Flirds adaptation discussion: FL setting differences, IRDS limitations.
- `flirds/conversation2.md` — benchmark/metric discussion; (a) retraining-based vs (b) in-run exact Shapley distinction; round-delta-as-1-step caveats.
- `flirds/conversation3.md` — cross-device design; (a) vs (b) clarification (different utilities); noise-vs-OOD-good algorithm sketch; ablation matrix; centralized data-level = client-level proof + FL drift residual.
- `flirds/conversation4.md` — final design lock: client-level / 1st+2nd Taylor / Δw_k only / server-side validation / 0 communication overhead / drift residual measured-not-corrected.
- `flirds/2026-05-19-section23-walkthrough.md` — **raw transcript** of the Section 2 / Section 3 walkthrough conversation (2026-05-19 → 2026-05-22), restored from Claude Desktop JSONL after session interruption. Tool calls / results / thinking blocks stripped; 2 Yonghee turns + 16 Claude turns preserved.
- `flirds/2026-05-27-section-23-lock.md` — **distilled** record of the Section 2 / Section 3 lock: Q1–Q3 + N1–N4 + model choice (Llama-3.2-1B/3B + Llama-2-7B) + baseline reduction (SPACE/S-FedAvg/FedCorr excluded; vanilla FedAvg added) + Phase 0 sanity reproduction task + 7B full matrix + protocol document. Spans 2026-05-19 / 2026-05-22 / 2026-05-27.

These are the **primary record** of the Flirds project. The wiki page [[flirds]] is the distilled landing — for design rationale and math details, read the raw conversations.
