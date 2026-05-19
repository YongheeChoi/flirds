---
type: log
title: Log
created: 2026-05-05
updated: 2026-05-18
---

# Log

Append-only chronological record. Newest entries at the bottom. Each entry begins with `## [YYYY-MM-DD] <kind> | <subject>` so it's grep-able:

```bash
grep "^## \[" wiki/log.md | tail -10
```

Kinds: `ingest`, `query`, `lint`, `note`, `conv`.

---

## [2026-05-05] note | wiki bootstrapped

- Created `CLAUDE.md` schema, `wiki/index.md`, `wiki/log.md`, `wiki/overview.md`.
- Created subdirectory layout: `wiki/sources/`, `wiki/concepts/`, `wiki/threads/`.
- Inventoried 20 papers in `raw/papers/flirds/` (mix of PDF and Obsidian-clipped markdown).
- Domain: data valuation, attribution, influence functions; semivalues; influence-function methods; federated/decentralized; LLM-scale; data-market applications.

## [2026-05-05] ingest | Data Banzhaf

- source: [[sources/data-banzhaf]]
- touched: [[concepts/semivalue]], [[concepts/banzhaf-value]], [[concepts/shapley-value]], [[concepts/leave-one-out]], [[threads/robustness-to-stochastic-training]]
- note: Demo ingest. Banzhaf has the largest safety margin among semivalues under SGD noise; MSR estimator unique.

## [2026-05-05] ingest | In-Run Data Shapley

- source: [[sources/in-run-data-shapley]]
- touched: [[concepts/data-shapley]], [[concepts/shapley-value]], [[threads/retraining-vs-in-run-attribution]], [[threads/influence-functions-at-llm-scale]]
- note: Demo ingest. Reframes Shapley from retraining-based to per-iteration; foundation-model scale.

## [2026-05-05] note | scope expansion + conversation logging workflow

- Yonghee asked to broaden the wiki's scope from data-valuation-only to all-of-AI-research.
- Refactored: `wiki/overview.md` now top-level hub; `flirds`-specific synthesis moved to [[topics/flirds]]; new `wiki/topics/` directory introduced.
- Added conversation-logging workflow to `CLAUDE.md`. Conversations now distill into `raw/conversations/<topic>/` (or `meta/`) as markdown files for future Claude sessions to ingest.
- Memory updated: `feedback_conversation_logging`, `user_role` (grad student framing), `project_research_wiki` (per-topic structure).
- Saved this session's conversation to `raw/conversations/meta/2026-05-05-wiki-bootstrap.md`.

## [2026-05-05] ingest | DICE — Data Influence Cascade in Decentralized Learning

- source: [[sources/dice]]
- touched: [[concepts/dice]], [[concepts/decentralized-learning]], [[threads/federated-and-decentralized-attribution]]
- note: First influence-estimation framework for fully decentralized P2P learning. Cascade decomposes into data × topology × intermediate curvature.

## [2026-05-05] ingest | DataInf

- source: [[sources/datainf]]
- touched: [[concepts/datainf]], [[concepts/influence-function]], [[concepts/lora]], [[threads/influence-functions-at-llm-scale]]
- note: Closed-form IF approximation via swap-inverse-and-average; tight bound for small-dim layers ⇒ LoRA-friendly. Reportedly fails on heterogeneous FL data ([[sources/feddqc]]).

## [2026-05-05] ingest | DU-Shapley

- source: [[sources/du-shapley]]
- touched: [[concepts/du-shapley]], [[concepts/dataset-valuation]], [[concepts/shapley-value]], [[threads/dataset-vs-data-point-valuation]]
- note: Structure-aware Shapley approximation for **dataset** valuation when utility depends on a scalar function of the coalition. AS convergence to Shapley as #owners → ∞.

## [2026-05-05] ingest | Asymmetric Data Shapley (ADS)

- source: [[sources/asymmetric-data-shapley]]
- touched: [[concepts/asymmetric-data-shapley]], [[concepts/shapley-value]], [[concepts/data-shapley]], [[concepts/replication-robustness]], [[threads/symmetry-and-asymmetry-axioms]], [[threads/data-and-model-markets]]
- note: Drops symmetry axiom for ordered groups. Three motivating settings: synthetic data, FL rounds, multi-stage LLM fine-tuning. State-conditioned marginal contribution.

## [2026-05-05] ingest | LoGra / Logix

- source: [[sources/logix]]
- touched: [[concepts/logra]], [[concepts/logix]], [[concepts/influence-function]], [[concepts/lora]], [[threads/influence-functions-at-llm-scale]]
- note: Kronecker-structured gradient projection ($O(\sqrt{nk})$). LoRA-like add-on layer architecture. ~6,500× throughput vs. EKFAC on Llama3-8B.

## [2026-05-05] ingest | FedDQC

- source: [[sources/feddqc]]
- touched: [[concepts/feddqc]], [[concepts/federated-learning]], [[concepts/data-quality-control]], [[concepts/lora]], [[threads/federated-and-decentralized-attribution]], [[threads/data-quality-vs-data-value]]
- note: IRA metric (instruction-response inference-loss difference) + hierarchical easy-to-hard training. DataInf reported failing on Fed-WildChat — useful negative result for gradient methods on heterogeneous FL.

## [2026-05-05] note | PDF triage agent completed

- Spawned background agent to read first 3-5 pages of all 12 PDFs and produce a structured triage report.
- Result: 9 core papers (Koh-Liang, Data Shapley, GTG-Shapley, TRAK, S-FedAvg, ShapleyFL, Ripple Shapley, DRDV, SPACE) + iPFL (data market angle, Nature Comms 2025) + RFedLR (peripheral, stub) + ICA3PP proceedings (skip — book).
- Used the report to write 11 source pages.

## [2026-05-05] ingest | Koh & Liang — Influence Functions

- source: [[sources/koh-liang-influence-functions]]
- touched: [[concepts/influence-function]], [[threads/retraining-vs-in-run-attribution]], [[threads/influence-functions-at-llm-scale]]
- note: Foundational paper bringing IF from robust statistics into ML. Spawned the entire IF/TracIn/DataInf/EKFAC/LoGra family.

## [2026-05-05] ingest | Ghorbani & Zou — Data Shapley

- source: [[sources/ghorbani-zou-data-shapley]]
- touched: [[concepts/data-shapley]], [[concepts/shapley-value]], [[threads/symmetry-and-asymmetry-axioms]]
- note: Foundational Data Shapley paper. Four axioms; TMC + gradient-Shapley estimators; outlier/mislabel detection.

## [2026-05-05] ingest | GTG-Shapley

- source: [[sources/gtg-shapley]]
- touched: [[concepts/gtg-shapley]], [[concepts/federated-shapley]], [[threads/federated-and-decentralized-attribution]]
- note: Federated Shapley via gradient sub-model reconstruction + guided MC + truncation.

## [2026-05-05] ingest | TRAK

- source: [[sources/trak]]
- touched: [[concepts/trak]], [[concepts/influence-function]], [[concepts/datamodels]], [[concepts/linear-datamodeling-score]], [[threads/influence-functions-at-llm-scale]]
- note: eNTK linearization + random projection. Closed-form approximation to Datamodels at 100-1000× lower cost. Established LDS as standard benchmark.

## [2026-05-05] ingest | S-FedAvg / Game of Gradients

- source: [[sources/game-of-gradients-sfedavg]]
- touched: [[concepts/s-fedavg]], [[concepts/federated-shapley]], [[threads/federated-and-decentralized-attribution]]
- note: FRCS via Shapley over client gradient coalitions. Predecessor to GTG / ShapleyFL.

## [2026-05-05] ingest | RFedLR (stub)

- source: [[sources/rfedlr]]
- touched: [[concepts/lora]], [[concepts/federated-learning]]
- note: Peripheral — federated LoRA robustness against label noise. No data-attribution machinery; stub only.

## [2026-05-05] ingest | ShapleyFL

- source: [[sources/shapleyfl]]
- touched: [[concepts/shapleyfl]], [[concepts/federated-shapley]], [[threads/federated-and-decentralized-attribution]]
- note: Surrogate federated Shapley + importance-sampling client selection + difference-of-Shapley estimator. Convergence + stability analysis.

## [2026-05-05] ingest | Ripple Shapley

- source: [[sources/ripple-shapley]]
- touched: [[concepts/ripple-shapley]], [[concepts/federated-shapley]], [[threads/federated-and-decentralized-attribution]], [[threads/retraining-vs-in-run-attribution]]
- note: First **sample-level, single-run** federated Shapley. Drop term + Jacobian-chain ripple term + low-rank subspace approximation. ~62× speedup. Real-time data pricing application.

## [2026-05-05] ingest | Distributionally Robust Data Valuation (DRDV)

- source: [[sources/distributionally-robust-data-valuation]]
- touched: [[concepts/drge-utility]], [[threads/utility-function-design]], [[threads/data-and-model-markets]], [[threads/robustness-to-stochastic-training]]
- note: Replaces validation-set-dependent utility with worst-case Wasserstein-ball loss (DRGE). Model-deviation proxy in RKHS / NTK. Orthogonal axis to semivalue weighting.

## [2026-05-05] ingest | iPFL — Inclusive PFL Model Market

- source: [[sources/ipfl-model-market]]
- touched: [[concepts/data-market]], [[concepts/personalized-fl]], [[threads/data-and-model-markets]]
- note: Graphical-game model market for personalized FL with IR + IC theoretical guarantees. LLM-scale demonstrations on Mistral / TinyLlama / Llama-2.

## [2026-05-05] ingest | SPACE — Single-round Participant Amalgamation

- source: [[sources/space-participant-amalgamation]]
- touched: [[concepts/space]], [[concepts/federated-shapley]], [[threads/federated-and-decentralized-attribution]]
- note: Single-round federated Shapley via knowledge amalgamation (clients distill into server) + prototype-based evaluation (no held-out validation set).

## [2026-05-05] note | concept and thread consolidation

- Created 4 spine concept pages: `influence-function`, `federated-learning`, `lora`, `datamodels`.
- Created 19 method-/topic-specific concept pages.
- Created 5 new threads: `federated-and-decentralized-attribution`, `influence-functions-at-llm-scale`, `symmetry-and-asymmetry-axioms`, `dataset-vs-data-point-valuation`, `utility-function-design`, `data-and-model-markets`, `data-quality-vs-data-value`.
- Updated 2 existing threads (`retraining-vs-in-run-attribution`, `robustness-to-stochastic-training`) with all the new sources.
- Refactored `topics/flirds.md` to reflect 19 sources, 25 concepts, 9 threads.

## [2026-05-05] conv | scope re-narrowed; Flirds etymology corrected; collaboration style locked

- raw: `raw/conversations/flirds/conversation{1,2,3,4}.md` (Yonghee added these between sessions; primary record of the Flirds design).
- distilled into: [[flirds]] (new project page), [[overview]] (rewritten as field synthesis + Flirds framing), [[../CLAUDE]] (scope narrowed).
- Yonghee's instructions:
  1. Wiki is for **data valuation only**; other research areas go in separate wikis. Reverted broader-AI scope from earlier in this session.
  2. **Flirds = Federated Learning + In-Run Data Shapley**, *not* "FL + Influence + Robust Data Shapley" as I had guessed earlier. Etymology corrected throughout.
  3. **Collaboration style**: precise corrections both ways; Yonghee will keep questioning until convinced. Saved to memory `feedback_collaboration_style`.
- Structural changes:
  - Deleted `wiki/topics/flirds.md` and `wiki/topics/` directory.
  - New `wiki/flirds.md` — Flirds project state distilled from the 4 raw conversations: locked design decisions (client-level / 1st+2nd Taylor / Δw_k only / server-side validation / 0 communication overhead / LoRA), open questions, experiment plan, ablation matrix, baseline comparison.
  - `wiki/overview.md` rewritten as field synthesis with explicit Flirds framing (no longer a multi-topic hub).
  - 3 backlinks `[[topics/flirds]]` updated to `[[overview]]` or rephrased.
- Memory: added `feedback_collaboration_style`, `reference_flirds_project`. Updated `user_role`, `project_research_wiki` to reflect single-topic intent.

## [2026-05-05] note | corrections made this session

For the record (per the new collaboration-style feedback):

- **My error**: I had guessed flirds = "Federated Learning + Influence + Robust Data Shapley" in the topics/flirds.md page. Wrong. Yonghee corrected: "Federated Learning + In-Run Data Shapley." Verified against `raw/conversations/flirds/conversation1.md` line 1.
- **My misstatement**: I said earlier this turn that "raw/conversations/flirds is empty" — true at the time of my first check during the bootstrap session, but not true at the moment I made the claim later. Should have re-verified.

Both fixed; etymology updated in [[../CLAUDE]] and [[flirds]] and the previous topics/flirds.md was deleted (so the wrong guess no longer exists in the wiki).

## [2026-05-18] note | noise-vs-OOD-good deferred; experiment-protocol discussion opened

- Continuing the Flirds design from conversation 4's three next-steps.
- **Decision (Yonghee)**: defer next-step 1 (noise-vs-OOD-good distinction) — algorithmically too ambiguous to resolve cleanly. Proceed with next-steps 2 (experimental protocol) and 3 (baseline selection).
- Distilled into [[flirds]]: open question 1 marked DEFERRED with its three consequences (narrative tightening; experiment-matrix shrink with noisy/free-rider surviving; non-IID bias now an explicit measurement obligation since the parked separator was also the drift-bias corrector). Next-step checklist rewritten.
- Protocol/baseline proposal discussed (2-track design; (b) MC in-run as cheap+correct primary oracle; cost-matched baseline tiers; Ripple Shapley head-to-head as scoop defense). **Not yet locked** — full conversation distillation + raw `conversation5.md` deferred to the decision checkpoint.
- note: deferral is internally consistent with the locked "drift residual: measured, not corrected" decision; its real cost is one new mandated experiment (non-IID α-sweep), not a contradiction.

## [2026-05-18] query | FL prior art for separating noise / OOD-good / malicious clients

- question: how does the FL literature approach distinguishing noise vs. OOD vs. malicious clients? (continuing the conversation-4 thread)
- filed into: [[threads/noise-ood-malicious-client-separation]] (new thread)
- touched: [[flirds]] (open-question-1 prior-art pointer; robustness baselines added to benchmark list; reading-list thread), [[overview]] (key open question + sources-to-ingest), [[index]]
- finding: 4 prior-art clusters — Byzantine-robust aggregation (FLTrust, FoolsGold), temporal-consistency (**FLDetector**, KDD'22), noisy-label FL (**FedCorr**, CVPR'22), free-rider (STD-DAGMM). Flirds' two deferred signals already exist there; the niche "separate bad-different from good-different inside a *signed valuation*" is open because every detector collapses on non-IID — which is itself the evidence the deferral is sound.
- context shift handled: [[flirds]] was edited in parallel (same day) to **defer** noise-vs-OOD-good from "2nd contribution" → "characterized limitation"; re-aligned the synthesis to back the limitation write-up + the surviving noisy/free-rider detection benchmarks instead of designing a new component. No deferral text reverted.
- ingest candidates flagged: FLDetector, FedCorr, FLTrust, FoolsGold, STD-DAGMM, Sattler clustered-FL, Pleiss AUM.

## [2026-05-19] ingest | FedSV — A Principled Approach to Data Valuation for FL (Wang et al. 2020)

- source: [[sources/principled-federated-data-valuation]]
- touched: [[concepts/federated-shapley]], [[concepts/federated-learning]], [[threads/federated-and-decentralized-attribution]], [[flirds]], [[index]], [[overview]]
- note: The **origin** of federated Shapley; per-round order-aware SV, zero-comm but $O(Tm^2)$ server utility evals. "FedSV" is this paper's nickname — resolves the long-standing "Wang 2020 not ingested" gap.

## [2026-05-19] ingest | ComFedSV (Improving Fairness for Data Valuation in Horizontal FL)

- source: [[sources/comfedsv]]
- touched: [[concepts/federated-shapley]], [[threads/federated-and-decentralized-attribution]], [[threads/dataset-vs-data-point-valuation]], [[flirds]], [[index]], [[overview]]
- note: FedSV + low-rank utility-matrix completion fixes the partial-participation symmetry break; cross-device Flirds baseline (extra comm, retrain-free but imputation-based, not in-run).

## [2026-05-19] ingest | FLDetector (KDD 2022)

- source: [[sources/fldetector]]
- touched: [[concepts/federated-learning]], [[concepts/data-quality-control]], [[threads/noise-ood-malicious-client-separation]], [[threads/federated-and-decentralized-attribution]], [[flirds]], [[index]], [[overview]]
- note: Temporal update-consistency detector (Cauchy-MVT + L-BFGS). IID-only Theorem 1 + Fig. 2 non-IID DACC drop = evidence Flirds' deferred separator is genuinely hard; primary scoop risk if revived.

## [2026-05-19] ingest | FedCorr (CVPR 2022)

- source: [[sources/fedcorr]]
- touched: [[concepts/federated-learning]], [[concepts/data-quality-control]], [[threads/noise-ood-malicious-client-separation]], [[threads/data-quality-vs-data-value]], [[flirds]], [[index]], [[overview]]
- note: Multi-stage label-noise correction; prediction-subspace LID flags noisy clients, then relabels. Hard-route + correct (vs Flirds signed down-weight). Noisy-client benchmark baseline.

## [2026-05-19] ingest | FLTrust (NDSS 2021)

- source: [[sources/fltrust]]
- touched: [[concepts/federated-learning]], [[concepts/data-quality-control]], [[threads/noise-ood-malicious-client-separation]], [[flirds]], [[index]], [[overview]]
- note: Server-root trusted-cosine trust bootstrapping. Its server-side clean root ≈ Flirds' server validation set (same privacy/abuse framing + poisoned-root risk). Robust-aggregation, not valuation.

## [2026-05-19] ingest | FoolsGold (RAID 2020)

- source: [[sources/foolsgold]]
- touched: [[concepts/federated-learning]], [[concepts/data-quality-control]], [[threads/noise-ood-malicious-client-separation]], [[flirds]], [[index]], [[overview]]
- note: Cross-client gradient-similarity Sybil defense. Its non-IID false-positive failure (Appendix-B RONI) is primary evidence for Flirds' characterized-limitation framing.

## [2026-05-19] ingest | Free-riders in FL / STD-DAGMM (Lin et al. 2019)

- source: [[sources/free-riders-fl-std-dagmm]]
- touched: [[concepts/federated-learning]], [[concepts/data-quality-control]], [[threads/noise-ood-malicious-client-separation]], [[flirds]], [[index]], [[overview]]
- note: Defines FL free-rider attacks + STD-DAGMM detector. A free-rider's near-zero/recycled update ⇒ Flirds 1st-order term ≈ 0, so it is demoted as a by-product (why this benchmark survives the deferral).

## [2026-05-19] note | FedSV / Federated-Banzhaf list corrections + flirds-summary.html

- Triggered by a link-finding research pass on the "not yet in raw" baseline list (Yonghee asked for the links, then dropped all 7 PDFs into `raw/` for ingest).
- Corrections applied across [[flirds]], [[overview]], [[threads/noise-ood-malicious-client-separation]], [[threads/federated-and-decentralized-attribution]], and `flirds-summary.html`:
  1. **FedSV = Wang et al. 2020** — not a separate paper; the duplicate "Wang et al. Principled Federated Shapley" gap entry removed; consolidated to [[sources/principled-federated-data-valuation]].
  2. **"Federated Banzhaf" has no dedicated paper** — de facto reference = [[sources/data-banzhaf|Data Banzhaf]] (Wang & Jia, AISTATS 2023), already ingested; reclassified from "missing source" to "already held; FL application is ours".
  3. **Distractor flagged**: 2025 arXiv:2502.17526 "FedSV: Byzantine-Robust FL via Shapley Value" is a robustness paper, not the valuation baseline — do not cite as baseline #1.
- `flirds-summary.html`: §3.3 rewritten (gap → resolved table + 3-correction callout, dates synced to 2026-05-19). (§1.2 "two Taylor expansions" disambiguation callout was added in the prior turn, patches 1–2.)
- Sources 19 → 26. `index.md` raw inventory + counts refreshed; `overview.md` "sources to ingest next" pruned (only Datamodels, Bae, TracIn, EKFAC, Beta/CS-Shapley, VFL, sharded-Shapley, Sattler, AUM remain).
