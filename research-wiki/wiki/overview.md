---
type: overview
title: Overview
created: 2026-05-05
updated: 2026-05-19
sources: [data-banzhaf, in-run-data-shapley, dice, datainf, du-shapley, asymmetric-data-shapley, logix, feddqc, koh-liang-influence-functions, ghorbani-zou-data-shapley, gtg-shapley, trak, game-of-gradients-sfedavg, shapleyfl, ripple-shapley, distributionally-robust-data-valuation, ipfl-model-market, space-participant-amalgamation, principled-federated-data-valuation, comfedsv, fldetector, fedcorr, fltrust, foolsgold, free-riders-fl-std-dagmm]
tags: [synthesis]
---

# Overview — data valuation, attribution, and influence

This wiki is the durable knowledge substrate for the **Flirds project** (*Federated Learning + In-Run Data Shapley*). Project-specific design state lives in [[flirds]]; this page is the broader field synthesis the project draws on. See [[index]] for the full catalog and [[log]] for chronological history.

> **Coverage as of 2026-05-19**: 26 sources ingested. Added 2026-05-19 — the federated-Shapley origin [[sources/principled-federated-data-valuation|FedSV (Wang et al. 2020)]] + [[sources/comfedsv|ComFedSV]], and the robustness-side Flirds baselines FLDetector / FedCorr / FLTrust / FoolsGold / STD-DAGMM.

## What problem are we solving?

For a given training set $D$ and a learned model $f_\theta$, **assign a real number $\phi_i$ to each training point $z_i \in D$** (or each contributor / client) reflecting "how much they contributed." The value drives data pricing, low-quality data detection, fair compensation in collaborative training, copyright attribution, curriculum / data selection, and unlearning.

The field decomposes along **three independent axes**:

| Axis | Choices |
|---|---|
| **Axiom** | classical Shapley · Banzhaf (drop efficiency) · Asymmetric (drop symmetry) · Beta-Shapley · CS-Shapley |
| **Utility** | validation accuracy · DRGE · per-step accumulated · prototype-based · inference-loss-difference |
| **Definition target** | algorithm-level (averaged) · model-level (specific trained model) · trajectory-level (path-anchored) |

Most papers move on a single axis. A method that combines all three (e.g. ADS + DRGE + In-Run) hasn't been published — see [[threads/utility-function-design]].

## Two foundational frameworks

### Semivalues — fair attribution from cooperative game theory

Treat a subset $S \subseteq D$ as a coalition; the **utility** $U(S)$ is the performance of a model trained on $S$. A semivalue assigns each point a weighted average of marginal contributions $U(S \cup \{i\}) - U(S)$. The weight schedule defines the variant:

- **[[concepts/leave-one-out|Leave-one-out]]**: only $S = D \setminus \{i\}$. Cheapest, weakest noise robustness.
- **[[concepts/shapley-value|Shapley]]**: uniform over all $S$ sizes. Uniquely satisfies efficiency, symmetry, linearity, null-player.
- **Beta-Shapley**, **CS-Shapley**: tilt the weighting to emphasize small or specific subsets.
- **[[concepts/banzhaf-value|Banzhaf]]**: equal weight on each subset (not on each size). Largest safety margin under SGD noise — see [[sources/data-banzhaf]].
- **[[concepts/asymmetric-data-shapley|Asymmetric Shapley]]**: drops the symmetry axiom to respect order/dependency among data sources.
- **[[concepts/du-shapley|DU-Shapley]]**: structure-aware approximation for *dataset* (not point) valuation.

The [[concepts/semivalue|semivalue page]] diagrams the family and its axiom trade-offs.

### Influence functions — gradient calculus instead of retraining

Define influence as $-H^{-1} \nabla_\theta \ell_k$ — what would happen if you up-weighted point $k$ infinitesimally. Avoids retraining but needs the inverse Hessian; that is the dominant cost.

Methods (chronological):

- [[sources/koh-liang-influence-functions|Koh & Liang 2017]] — foundational; LiSSA stochastic inversion + damping for non-convex deep nets.
- [[sources/datainf|DataInf]] — closed-form approximation tuned for LoRA; $O(nDL)$ compute.
- [[sources/trak|TRAK]] — eNTK linearization + random projection; matches Datamodels' counterfactual fidelity.
- [[sources/logix|LoGra/Logix]] — Kronecker-structured projection; ~6,500× throughput vs. EKFAC at Llama3-8B scale.
- **EKFAC** — Kronecker-factored Hessian baseline; runnable but slow at billion-parameter scale.

Detailed comparison: [[threads/influence-functions-at-llm-scale]].

## The federated / decentralized layer (where Flirds lives)

A **separate subfield** has emerged for attribution under FL/decentralized constraints. Standard semivalues are infeasible (counterfactual training paths blow up) and gradient methods become brittle under client heterogeneity ([[sources/feddqc]] reports DataInf failing on Fed-WildChat).

Key papers:

- **[[sources/principled-federated-data-valuation|FedSV (Wang et al. 2020)]]** — the origin of federated Shapley; per-round, order-aware; the canonical Flirds baseline.
- **[[sources/comfedsv|ComFedSV]]** — FedSV + low-rank utility-matrix completion to fix partial-participation unfairness.
- **[[sources/dice|DICE]]** — first influence framework for fully decentralized (P2P) learning; influence cascades through the topology.
- **[[sources/feddqc|FedDQC]]** — on-device data-quality control via [[concepts/instruction-response-alignment|IRA]].
- **[[sources/gtg-shapley|GTG-Shapley]]** — sub-model reconstruction from logged gradients + guided MC.
- **[[sources/game-of-gradients-sfedavg|S-FedAvg]]** — Shapley-weighted FedAvg + client pruning (FRCS).
- **[[sources/shapleyfl|ShapleyFL]]** — surrogate federated Shapley + importance-sampling client selection.
- **[[sources/space-participant-amalgamation|SPACE]]** — single-round Shapley via knowledge distillation + prototype evaluation.
- **[[sources/ripple-shapley|Ripple Shapley]]** — first **sample-level, single-run** federated Shapley; cross-round Jacobian propagation. **Closest comparator to Flirds.**
- **Robustness-side (Flirds detection baselines)** — [[sources/fldetector|FLDetector]] (temporal consistency), [[sources/foolsgold|FoolsGold]] / [[sources/fltrust|FLTrust]] (cross-client / trusted-cosine), [[sources/fedcorr|FedCorr]] (noisy-label LID), [[sources/free-riders-fl-std-dagmm|STD-DAGMM]] (free-rider). Synthesis: [[threads/noise-ood-malicious-client-separation]].

Synthesis: [[threads/federated-and-decentralized-attribution]].

## Data markets and incentive design

Several wiki sources approach data attribution as a means to a *commercial* end. Beyond computing values, markets need IR/IC, replication-robustness, validation-stable pricing.

- **[[sources/ipfl-model-market|iPFL]]** — graphical-game model market for personalized FL at LLM scale; provides IR + IC theoretical guarantees.
- **[[sources/asymmetric-data-shapley|ADS]]** — replication-aware Shapley via the symmetry-axiom drop.
- **[[sources/distributionally-robust-data-valuation|DRDV]]** — buyer-side robust pricing via DRGE utility.

Synthesis: [[threads/data-and-model-markets]].

## The big trends

1. **Retraining is dead at LLM scale.** Even CIFAR-scale Shapley needs Monte Carlo. Pretraining attribution requires gradient methods or single-run trickery.
2. **In-run / trajectory-anchored attribution is rising.** [[sources/in-run-data-shapley|In-Run Shapley]] (centralized), [[sources/ripple-shapley|Ripple Shapley]] (federated), [[sources/asymmetric-data-shapley|ADS]] (axiomatic) all anchor evaluation to the realized trajectory rather than averaging hypothetical retrainings. See [[threads/retraining-vs-in-run-attribution]]. **Flirds is in this lineage.**
3. **Stochasticity matters.** SGD noise can flip data-value rankings between runs. Robustness-to-noise is an evaluation criterion now ([[sources/data-banzhaf]]'s safety-margin framework). See [[threads/robustness-to-stochastic-training]].
4. **The symmetry axiom is increasingly seen as a bug.** Real pipelines have temporal order (FL rounds, fine-tuning stages) and directional dependence (synthetic vs. original data). See [[threads/symmetry-and-asymmetry-axioms]].
5. **Federated/decentralized contribution is its own subfield.** ~half the wiki's sources are here. See [[threads/federated-and-decentralized-attribution]].
6. **Quality and value are being conflated.** Quality scoring (PPL, IRA) ≠ value/contribution scoring (Shapley, IF). See [[threads/data-quality-vs-data-value]].
7. **Applications drive what counts as "good."** Data markets care about replication-robustness; curation cares about ranking stability; copyright cares about attribution beyond verbatim memorization.

## Key open questions in the field

These are the most load-bearing unknowns the wiki currently tracks. Each links to the thread where it's developed.

- **Cross-method calibration**: do DataInf, EKFAC, LoGra, In-Run Shapley, Banzhaf, Ripple Shapley agree on which points are good/bad? On which do they diverge? — [[threads/influence-functions-at-llm-scale]]
- **Why gradient methods fail on heterogeneous FL**: [[sources/feddqc]]'s DataInf failure on Fed-WildChat. Specific to DataInf or general? — [[threads/federated-and-decentralized-attribution]]
- **Combinable axiom + utility + definition modifications**: each existing method modifies one axis. What about all three? — [[threads/symmetry-and-asymmetry-axioms]], [[threads/utility-function-design]]
- **Replication-robust ADS**: does ADS formally satisfy Agarwal et al.'s replication-robustness criterion, or just heuristically? — [[threads/symmetry-and-asymmetry-axioms]]
- **In-Run Shapley vs. Influence Functions** in the limit: are they computing the same model-level quantity? — [[threads/retraining-vs-in-run-attribution]]
- **Sample-level vs. client-level federated attribution**: when is each appropriate? When does aggregating sample-level back to client-level give a different ranking than direct client-level Shapley? — [[threads/dataset-vs-data-point-valuation]]
- **Quality-aware curation as a substitute for value-based curation at scale**: when is filtering on quality good enough? — [[threads/data-quality-vs-data-value]]
- **Noise vs. "good-different" (OOD-good) inside a signed value**: the FL-robustness literature (FLDetector, FoolsGold, FLTrust, FedCorr) collapses both into "anomaly → discard" and collapses on non-IID; no method separates them inside a contribution score. Flirds *defers* this as a characterized limitation. — [[threads/noise-ood-malicious-client-separation]]

## Sources to ingest next

The 12-PDF triage already brought in the bulk of the federated-Shapley line plus the foundations. Biggest remaining gaps:

- **Original Datamodels paper** (Ilyas et al., 2022) — referenced everywhere, not yet in raw.
- **Bae et al., "If Influence Functions are the Answer..."** — challenges classical IF on deep nets.
- **TracIn** (Pruthi et al.) — bridges classical IF and In-Run Shapley.
- **Original EKFAC paper** (Grosse et al.) — currently in the wiki only as a baseline.
- **Beta-Shapley, CS-Shapley** original papers.
- **VFL data valuation** (Han 2025) — vertical federated learning angle.
- **Sharded Shapley for unlearning** (cited in [[sources/asymmetric-data-shapley|ADS]]).
- **Clustered FL** (Sattler et al.) and **AUM** (Pleiss et al., NeurIPS 2020) — remaining robustness / centralized-ancestry refs for [[threads/noise-ood-malicious-client-separation]].

> **Ingested 2026-05-19** (no longer gaps): Wang et al. 2020 FedSV → [[sources/principled-federated-data-valuation]], ComFedSV → [[sources/comfedsv]], "Federated Banzhaf" = [[sources/data-banzhaf|Data Banzhaf]] applied in FL (no dedicated paper exists; already in the wiki), [[sources/fldetector|FLDetector]], [[sources/fedcorr|FedCorr]], [[sources/fltrust|FLTrust]], [[sources/foolsgold|FoolsGold]], [[sources/free-riders-fl-std-dagmm|STD-DAGMM]].
