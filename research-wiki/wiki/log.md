---
type: log
title: Log
created: 2026-05-05
updated: 2026-05-22
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

## [2026-05-22] ingest | LESS — Selecting Influential Data for Targeted Instruction Tuning (Xia et al., ICML 2024)

- source: [[sources/less]]
- touched: [[concepts/influence-function]], [[concepts/lora]], [[threads/influence-functions-at-llm-scale]], [[threads/retraining-vs-in-run-attribution]], [[threads/utility-function-design]], [[threads/data-quality-vs-data-value]], [[threads/data-selection-for-llms]] (new), [[flirds]], [[index]], [[overview]]
- note: **The closest centralized comparator to Flirds.** TracIn-style trajectory IF + Adam-Γ + cosine similarity + LoRA + JL random projection (8192-dim reusable gradient datastore). 5% selection beats full 270K on Llama-2-7B/13B + Mistral-7B for MMLU/TyDiQA/BBH. Positioning load-bearing: differences (per-client vs per-example, Adam-Γ unneeded under FedAvg, 2nd-order Taylor, no offline datastore) define what's structurally new about Flirds vs the LESS baseline.

## [2026-05-22] ingest | Grosse et al. 2023 — Studying LLM Generalization with Influence Functions (EK-FAC at 52B)

- source: [[sources/grosse-llm-influence]]
- touched: [[concepts/influence-function]], [[concepts/ekfac]] (new), [[concepts/proximal-bregman-response]] (new), [[threads/influence-functions-at-llm-scale]], [[threads/retraining-vs-in-run-attribution]], [[flirds]], [[index]], [[overview]]
- note: **Upper-bound anchor** for IF at LLM scale. Scales IF to 52B via EK-FAC (IHVP), TF-IDF filtering + query batching (per-example gradient). Reframes target as [[concepts/proximal-bregman-response|PBRF]] (Bae et al. 2022a) rather than classical counterfactual. Empirical findings (heavy-tailed influence, scale-emergent abstraction, word-ordering brittleness, role-playing-as-imitation) anchor the qualitative side of the wiki's IF discussion. Pre-IF scale ceiling was 300M (Schioppa 2022); Grosse pushes it ~170× higher.

## [2026-05-22] ingest | MATES — Model-Aware Data Selection with Data Influence Models (Yu et al., NeurIPS 2024)

- source: [[sources/mates]]
- touched: [[concepts/influence-function]], [[concepts/datamodels]], [[threads/influence-functions-at-llm-scale]], [[threads/retraining-vs-in-run-attribution]], [[threads/utility-function-design]], [[threads/data-quality-vs-data-value]], [[threads/data-selection-for-llms]] (new), [[flirds]], [[index]], [[overview]]
- note: **Direct backing for Flirds' 1B-primary decision.** Pythia 410M + 1B pretraining on C4 (25B tokens), BERT-base data influence model continuously fine-tuned on locally-probed one-step-Δloss oracle. 2.3× faster than random to fixed accuracy. Doubles QuRating gains. Pointwise locally-probed oracle is conceptually closest to Flirds' "(b) in-run exact Shapley" ground-truth utility — both fix the trajectory and read off counterfactual loss change from minimal perturbation.

## [2026-05-22] ingest | DsDm — Model-Aware Dataset Selection with Datamodels (Engstrom et al., ICML 2024)

- source: [[sources/dsdm]]
- touched: [[concepts/datamodels]], [[concepts/trak]], [[threads/influence-functions-at-llm-scale]], [[threads/retraining-vs-in-run-attribution]], [[threads/utility-function-design]], [[threads/data-quality-vs-data-value]], [[threads/data-selection-for-llms]] (new), [[flirds]], [[index]], [[overview]]
- note: **The Datamodels → LLM bridge.** Frames dataset selection as direct optimization, approximates via linear datamodels fit via TRAK on a 125M proxy, selects bottom-$k$. 2× compute multiplier at 1.3B. Most striking finding: similarity-based baselines (DSIR, FastText classifier) routinely *underperform* random selection at LLM scale. Cleanest single-paragraph evidence for the [[threads/data-quality-vs-data-value|quality ≠ value]] thread.

## [2026-05-22] note | new concepts + thread + sources-to-ingest list pruned

- Created [[concepts/ekfac]] (was only a baseline reference before; Grosse anchors it now), [[concepts/proximal-bregman-response]] (the "what IF actually computes" framing).
- Created [[threads/data-selection-for-llms]] consolidating LESS / MATES / DsDm: similarity-≠-value, ~2× compute multiplier, reference-set choice. Three centralized methods that all sit one structural step away from a Flirds-FL extension.
- Sources 26 → 30. Concepts 25 → 27. Threads 10 → 11.
- [[overview]] "sources to ingest next" pruned: Datamodels-original, Bae 2022a, TracIn, EKFAC-original (now redundant — Grosse paper inherits the EKFAC framing), Beta/CS-Shapley, VFL, sharded-Shapley, Sattler, AUM remain. Plus added: QuRating, DSIR (now demoted from "future" to "would close picture but not urgent").

## [2026-05-27] conv | Section 2 / Section 3 lock + Phase 0 sanity reproduction task + protocol document

- raw: `raw/conversations/flirds/2026-05-27-section-23-lock.md` (distill of 2026-05-19 / 2026-05-22 / 2026-05-27 conversation arc — session was interrupted between 2026-05-22 and 2026-05-27, restored from JSONL transcript).
- distilled into: [[flirds]] (comprehensive rewrite), [[flirds-protocol]] (new), [[index]] (added flirds-protocol).
- Yonghee's decisions:
  1. **Pilot data set aside** — only clean re-runs enter paper claims (N1 lock 2026-05-22).
  2. **Scale tier 1B + 3B + 7B** — 13B/70B excluded (N2 lock).
  3. **Models**: Llama-3.2-1B-Instruct + Llama-3.2-3B-Instruct + Llama-2-7B (Option A — Llama-3.2 family for 1B/3B consistency, Llama-2-7B for LESS/FedDQC direct comparison).
  4. **Dual oracle separated reporting**: (a) exact retrain SV (1B N∈{5,10} + 3B N=5; 7B skipped) + (b) IRDS-定 SV (cross-silo exact, cross-device MC).
  5. **Baseline reduction**: SPACE / S-FedAvg / FedCorr / Power-of-choice **excluded** (LLM-environment unsuitable or not a valuation baseline). Final = 10 valuation/training + 4 detection.
  6. **Vanilla FedAvg as training-comparison baseline** (Yonghee surfaced) — Full FedAvg + Random-selection FedAvg added.
  7. **Phase 0 sanity reproduction** (Yonghee surfaced from #4 Ripple defense) — extended to all 4 code-unavailable baselines (Ripple, GTG, FedSV, ComFedSV). Cost ≈ 5–7 days B200×1. Must pass within ±5% before LLM phase starts.
  8. **7B full matrix** — Yonghee challenged Claude's reduced 7B coverage; re-computation showed 7B ≈ 1 week B200×4 with full matrix, fully feasible. Reduced 7B was Claude's residual from when 13B was still in plan.
  9. **HTML summary file deprecated** — wiki is single source.
- Section 3 = **11 items** (Phase 0 sanity + 4 ★★★ spine + 5 ★★ characterization + 1 ★ scale extension).
- Protocol document established: bf16 train / fp32 eval / scipy tied-rank / MC variance / 95% bootstrap CI / oracle code-path separation / sanity gates / run logging.
- Preference surfaced: explanation → decision → execution (no premature wiki writes); family-consistent scaling lineage; code availability as baseline-selection criterion.

## [2026-05-27] note | JSONL session transcript restored as raw conversation

- raw: `raw/conversations/flirds/2026-05-19-section23-walkthrough.md` (new; extracted from Claude Desktop JSONL `0f448283-f0b9-46f2-ad8d-cdacb1d56685.jsonl` after session interruption).
- distilled into: already done — [[2026-05-27-section-23-lock]] is the distill of this same conversation arc.
- note: Yonghee asked to preserve the raw transcript (not the JSONL itself) following the existing `conversation{1..4}.md` convention. Tool calls / tool results / thinking blocks stripped; 2 Yonghee turns + 16 Claude turns preserved with timestamps. 33 KB.

## [2026-05-27] note | code-link backfill across 3 source pages

- touched: [[sources/shapleyfl]] (ZJU-DIVER GitHub), [[sources/in-run-data-shapley]] (GhostSuite GitHub + ICLR'25 outstanding-runner-up mention), [[sources/space-participant-amalgamation]] (culiver/SPACE GitHub + baseline-exclusion sentence).
- note: code availability was looked up during the 2026-05-27 Section 3 lock conversation but only landed in [[flirds]] / [[flirds-protocol]]; backfilled to the source pages themselves for navigability. [[sources/less]] already had `princeton-nlp/LESS`; [[sources/mates]] already had `cxcscmu/MATES`; [[sources/datainf]] and [[sources/logix]] already had GitHub URLs.

## [2026-05-27] note | implementation plan + session handoff document created

- new: [[flirds-implementation-plan]] — self-contained handoff document for the next implementation session.
- structure: status snapshot → 4-phase task ordering (Phase 0 CNN baseline reproduction gate → Phase 1 Flirds at 1B + dual oracle + vanilla FedAvg → Phase 2 full baseline + 3B/7B + cross-device → Phase 3 matrix execution + Ripple reduction) → 9 still-open implementation decisions (dataset / LoRA hyper / training hyper / validation / cross-device detail / BASE_REPO / compute env / phase-internal order / detection baselines) with options + criteria + recommendations → next-session starter prompt → pre-implementation checklist → pointer table.
- registered in [[index]] Project section.
- Yonghee's framing: this session's job is to ensure the next session (implementation) receives full context. No code written this session; theoretical scaffolding (Section 2 + Section 3 + Protocol) complete; implementation-detail decisions left for the implementation session itself per Yonghee's preferred workflow.
- Pipeline status: ready to transition root `CLAUDE.md` from `stage: idle` to `stage: implementation` when the implementation session begins.

## [2026-06-02] conv | Phase 0 implementation — decisions + 4 baseline self-builds

- raw: [[raw/conversations/flirds/2026-06-02-phase0-implementation]]
- decided: §3 open items — datasets (LESS setup + cross-silo 5-domain PubMedQA/CaseHOLD/FiQA/AQUA-RAT/Dolly; cross-device Fed-WildChat+FedHDS; FedDQC = IRA-baseline-only), BASE_REPO (OpenFedLLM + self-built CNN sim), LoRA rank sweep {16,32,64,128}, detection (FLDetector + STD-DAGMM). **CNN full track added** (whole suite on CNN too, no LoRA). Phase restructure: 0 (CNN baseline reproduction) → 0.5 (Flirds estimator + dual oracle on CNN) → 1+ (LLM).
- built: FL simulator + exact-SV oracle + GTG/FedSV/ComFedSV/Ripple self-builds, all verified (cosine 0.99 / 0.998 / 0.993, Ripple noisy-detection AUROC 1.0); git commit `93bb8d0` on `feature/flirds-phase-0`.
- review: math sound (ripple Jacobian chain numerically verified); fixed gtg `_normalize` div-0 + dead code; Phase 0.5 TODOs = ripple-term verify (needs backdoor), ripple `(rounds,n,P)` OOM at scale, eigsh convergence fallback.
- distilled into: [[flirds-implementation-plan]] (Decisions resolved & corrections section), root `CLAUDE.md` (stage → implementation).

## [2026-06-03] conv | Phase 0.5 — Flirds estimator + dual in-run oracle (CNN)

- raw: [[raw/conversations/flirds/2026-06-03-phase05-estimator]]
- built: (b) in-run SV oracle (`oracle/in_run_sv`, exact 2^N) + Flirds estimator (`core/flirds_estimator`, 1st/1st+2nd Taylor, Hessian-only, 1 HVP/round) + **faithful Ripple rewrite** (Eq 5-19: local-traj drop, per-client local Hessian, progressive subspace) — **fixed 2 latent ripple bugs** (term sign + α-weighting, masked by drop-dominated verify).
- validated (all gates green): estimator≈(b) Spearman 1.000 / noisy AUROC 1.0; (a)retrain↔(b)in-run AUROC 1.0, fine-rank 0.66 (diff utilities); (b) Shapley efficiency=0, symmetry=0; HVP jvp-vs-double-backward 9.8e-6; E=1 noise-floor; N=2 3e-3; reproducibility bitwise-0 (needs `cudnn.deterministic`).
- decided: **2nd-order curvature = true Hessian** (match IRDS; GGN tested + rejected, worse). IRDS's "2nd-order marginal-for-accuracy" is a *centralized per-step* artifact (Yonghee); FL per-round multi-step is where the 2nd-order is non-trivial → real test at FL-scale/LLM.
- distilled into: [[flirds]] (Phase 0.5 findings), root `CLAUDE.md` (Phase 0.5 → done, next = Phase 1 LLM).
- review (same session): independent `/code-review` (4 finder angles, fresh agents) → no correctness bug on the current path. **momentum removed → plain SGD** — the 2nd-order term now *helps* (estimator-vs-(b) 3-seed 1st+2nd 0.96 > 1st 0.92; was reversed 0.73<0.81 under momentum) → empirical confirmation that FL per-round (not IRDS's centralized per-step) is the 2nd-order's regime. Also: estimator `model.eval()`; reproducibility `flirds/repro.py:seed_everything` (cudnn-det CNN-only); deferred items recorded as code TODOs. Re-verified green. Commits `034ae76` + `3ccfef8`.
- distilled into (review): [[flirds-implementation-plan]] (Phase 0.5 complete section), [[sources/in-run-data-shapley]] (2nd-order centralized-limit note).

## [2026-06-03] ingest | recent prior-work scan (2025–2026) — 11 sources

- trigger: Yonghee asked for a fresh literature scan (last full survey was 2026-05-22) + cross-merge with an external GPT survey, ingesting only the genuinely new + important papers.
- method: 4 internal search agents (federated valuation / in-run attribution / LLM-LoRA attribution / federated-LLM data) + Yonghee's GPT survey; cross-checked, deduped against the existing 30 sources, arXiv IDs verified by fetch for the uncertain ones.
- **novelty verdict: intact.** No single paper does the full Flirds intersection (client-level + in-run + closed-form 1st/2nd Taylor + HVP interaction + zero-extra-comm + LoRA/LLM). "First federated in-run" was already foreclosed by Ripple (not a new loss). Adam-axis concern (IRDS-for-Adam) deprioritized by Yonghee — plain-SGD client choice keeps the theory intact.
- sources (11): [[sources/fedtsv]] (FedTSV, ECC'26 — trajectory-SV → aggregation), [[sources/fedif]] (FedIF — 1st-order TracIn FL valuation, closest in-run-on-Δw besides Ripple, code public), [[sources/data-value-embedding]] (DVEmb — IRDS authors, trajectory LOO), [[sources/do-influence-functions-work-on-llms]] (EMNLP'25 negative result), [[sources/lorif]] (low-rank IF → 70B), [[sources/accumulative-sgd-influence]] (ACC-SGD-IE — cross-epoch accumulation), [[sources/dpo-shapley-lm-arithmetic]] (LLM-FT Shapley via DPO algebra), [[sources/shapley-volatility-fl]] (FL-Shapley instability), [[sources/mavericks-shapley-fl]] (FL-Shapley under-credits OOD/skewed clients), [[sources/influence-functions-fragile]] (Basu et al., ICLR'21), [[sources/fedhds]] (federated data-efficient instruction tuning).
- touched: [[index]] (sources 30→41, new "Influence/in-run — recent" subsection, raw inventory 41/42), [[overview]] (coverage 30→41 + 2026-06-03 note), [[flirds]] (differentiator table +FedIF/FedTSV; new "Recent prior-work scan" subsection; frontmatter sources), [[concepts/federated-shapley]] (+FedTSV/FedIF, aggregation-side relatives), [[concepts/influence-function]] (+LoRIF/ACC-SGD-IE + negative-result limitations), [[threads/federated-and-decentralized-attribution]] (+5 sources, valuation-vs-aggregation distinction, field-direction + stability/fairness), [[threads/influence-functions-at-llm-scale]] (+LoRIF; "Does IF work at LLM scale?" + "accumulating along trajectory" sections), [[threads/retraining-vs-in-run-attribution]] (+DVEmb/ACC-SGD-IE/FedIF, lineage diagram), [[threads/noise-ood-malicious-client-separation]] (+Mavericks evidence, FedTSV/FedIF direction-alignment blind spot).
- positioning shifts recorded: (i) FedIF = the "why isn't 1st-order enough?" baseline Flirds must beat — its gap (2nd-order, client-Shapley, LoRA/LLM, post-hoc) IS Flirds' contribution; (ii) FedTSV/FedIF are aggregation-side, Flirds is valuation-side (same input, different output); (iii) forward-HVP (H·Δw, never H⁻¹) dodges the "Do IF Work on LLMs?" iHVP-collapse cause — a defensible rebuttal; (iv) Volatility motivates the exact (b) oracle, Mavericks backs the non-IID-bias limitation.
- flagged for future ingest: **AFedSV** (Tastan et al., IJCAI 2024) — FedIF's SV baseline (450×-cost comparator); natural SV-side comparator for a Flirds eval. Not yet in `raw/`.
- note: FedHDS (5% participation + Adam local training) collides with Flirds' current full-participation + plain-SGD assumptions → Phase 1/2 port detail, recorded in [[sources/fedhds]].

## [2026-06-03] ingest | ShapFed (IJCAI 2024) + "AFedSV" label resolved

- trigger: Yonghee asked to ingest "AFedSV" (the SV-side comparator flagged after the FedIF ingest). Investigation found the flag was mis-attributed.
- **correction**: "AFedSV" is **not a standalone paper**. It is the *adaptive surrogate-SV aggregation* = [[sources/shapleyfl|ShapleyFL]] (Sun et al., KDD 2023, already in wiki) — FedIF cites it as ref [17]=ShapleyFL; the description ("surrogate FedSV updated each round to dynamically reweight clients") matches ShapleyFL exactly. The Tastan et al. IJCAI'24 paper the FedIF agent confused it with is **ShapFed**, a *different* method.
- source: [[sources/shapfed]] — Tastan, Fares, Aremu, Horvath, Nandakumar (MBZUAI), "Redefining Contributions: Shapley-Driven Federated Learning", IJCAI 2024 (arXiv:2406.00569). Code: `github.com/tnurbek/shapfed`. Class-Specific Shapley (CSSV) via **last-layer per-class cosine** alignment (no validation set, no coalition enum) → ShapFed-WA weighted aggregation + personalization. CIFAR-10 / Chest X-Ray / Fed-ISIC2019; matches/surpasses AFedSV(=ShapleyFL).
- touched: [[index]] (sources 41→42, raw 42→43), [[overview]] (42), [[flirds]] (AFedSV flag → resolved + ShapFed; frontmatter), [[concepts/federated-shapley]] (+ShapFed row), [[threads/federated-and-decentralized-attribution]] (+ShapFed comparison row, AFedSV flag corrected), [[sources/shapleyfl]] (AFedSV alias note), [[sources/fedif]] (AFedSV-citation correction).
- Flirds relevance: scoop LOW–MEDIUM. Recent code-available FL-Shapley SOTA, but class-specific last-layer cosine + aggregation/personalization + CNN classification — a CNN-track baseline + a direct contrast for the ② last-layer-cancellation characterization (ShapFed treats last-layer-per-class as a *feature*; Flirds treats last-layer dominance as a *limitation*). Not a drop-in LLM baseline (per-class head assumption, like SPACE's prototype issue).

## [2026-06-03] note | experiment plan extended (+7 prior-art validation gaps) + Phase 1 seams

- trigger: Yonghee asked whether prior-art validation experiments are missing from the plan, then "add them all" — checked mid-Phase-1 safety.
- added to [[flirds]] Section 3 (#12–18) + experiment matrix: **#12 backdoor/model-replacement detection** (FedSV/ShapleyFL precedent), **#13 PGD/direction-aligned poison** (FedIF's blind spot → the 2nd-order-vs-1st-order differentiation experiment), **#14 partial-participation fairness / duplicate-client** (ComFedSV; validates the "no participation normalization" lock), **#15 maverick/rare-domain** (Mavericks; sharp OOD-good limitation), **#16 validation-set sensitivity** (MATES; re-added from conv3 §3, was dropped in 05-27 lock), **#17 qualitative attribution case study** (DataInf/Grosse), **#18 clean-data skyline** (FedDQC).
- **all 7 are Phase 2/3 (data/eval-layer); none touch the Phase 1 core.** estimator/oracle are corruption-agnostic.
- **two Phase-1 seams recorded in [[flirds-implementation-plan]]** (honor now, cheap; retrofit costly): (1) φ logging = per-round × per-client × **per-layer** (not scalar-only); (2) **pluggable client-corruptor/partitioner** in the data layer. Both already exist in the Phase 0.5 CNN track — carry the shape into the LLM layers. Phase 1 deliverables unchanged.
- compute: mostly config-level on existing harnesses (post-hoc re-eval or +1 corruptor/partition per run); only PGD (#13) is non-trivial → scope to 1B+7B if budget-bound.

## [2026-06-03] note | Phase 1 seams finalized + handed back to the implementation session

- Yonghee is continuing Phase 1 in the dedicated implementation session; this wiki session recorded the carry-over so it transfers cleanly.
- **Seam 1 — per-layer φ logging INVARIANT**: confirmed against `core/flirds_estimator.py` (φ_k is already `sum over named-params`; logging = keep summands, φ_k = sum stays bit-identical). Recorded as a ⚠⚠ INVARIANT in [[flirds-implementation-plan]] §2 Phase 1 callout: per-layer φ is **observation-only** (② / #17), the spine **never reweights** (would break the granularity-invariance lemma + Prop 1); the only intentional reweighting is the locked **Q2 layer-wise ablation variant (#6)**, never the headline. Implement as backward-compatible `per_layer=False` option.
- **Seam 2 — pluggable corruptor registry**: corruption is currently inline in `experiments/phase05_*.py:build()` (`noisy={4,5}` + label-flip). Design recorded in [[flirds-implementation-plan]] §3.10 (name→callable registry; sample/update/partition levels; backend-agnostic for free_rider/maverick/duplicate, per-backend body for label_flip/backdoor/pgd; run-config maps client_idx→corruptor). **Fork (a) implement now+refactor vs (b) wire while building LLM data layer — deferred to the Phase 1 session** (codes/CLAUDE.md: surgical, no speculative abstraction).
- **Seam 3 — validation-set**: already config-driven (`flirds_values(..., val_x, val_y, ...)`), no hardcoding. No Phase-1 change beyond keeping it a parameter.
- codes/flirds/ working tree clean (only wiki edits uncommitted); LLM data layer not yet written → Phase 1 at its start, no conflict.
- No code written this session (Phase 1 implementation owned by the other session); all decisions distilled into the wiki for pickup.

## [2026-06-03] conv | Phase 1 kickoff — estimator/oracle backend-agnostic refactor (CNN)

- raw: [[raw/conversations/flirds/2026-06-03-phase1-backend-abstraction]]
- built: estimator/oracle made Phase-1-ready — (i) **partial-participation-correct** (per-round FedAvg weight `p_k^r=n_k/Σ_{P_r}n`; the "no participation normalization" lock is a *separate* axis = don't divide φ by participation count, intact — review item (ii) resolved), (ii) **per-layer φ logging** (seam 1, `per_layer=False` default, observation-only, Σcomp==φ bit-identical), (iii) **backend-agnostic** via `loss_fn(params,buffers)`+pkeys injection (`backends/cnn.py:make_cnn_loss`; estimator/oracle no longer touch model/val/task). `exact_sv` (a-oracle) untouched.
- validated: full-participation **bit-identical** (3-seed 1st+2nd 0.962 / 1st 0.924, efficiency/symmetry/repro 0); partial smoke (per_layer invariant 0, est≈(b) Spearman 1.0, efficiency 1.4e-17 — its noisy-AUROC 0 is a #14 cross-tier artifact, not a bug); phase05×4 regression green (N=2 relL2 3e-3, 2nd-order regime-dependence reproduced).
- (A) OpenFedLLM scouted (cloned to gitignored `external/`, reference-guided): its fedavg aggregate weight **==** our per-round weight; Δw = `get_peft_model_state_dict` local − global. 3 backend seams only (loss_fn / pkeys=LoRA filter / val-batch).
- decided: backend abstraction = **loss_fn closure injection**; LLM local train = **TRL SFTTrainer + forced SGD**.
- distilled into: [[flirds-implementation-plan]] (status snapshot), root `CLAUDE.md` (next), MEMORY (stale "uncommitted" fixed; review (ii) resolved).
- next (other session): **LLM stage 2** = `backends/llm.py` + LLM FL loop self-build + 5-domain data layer (seam 2 corruptor registry). estimator/oracle need no further change. Open: §3.4 val-mix (D6 ~200/domain rec); seam 2 (a)/(b) fork.

## [2026-06-04] conv | Phase 1 LLM stage 2 — validation lock + FedAvg core + LLM backend/FL-loop

- raw: [[raw/conversations/flirds/2026-06-04-phase1-llm-stage2]]
- **validation (§3.4) Yonghee 확정**: 도메인당 200 / 총 1000 uniform stratified, IRDS-held-out(few-shot 기각); dev split 우선·Dolly category-carve; cross-silo trainset 크기 도메인별 통제(aggregate weight는 size-prop 유지; FiQA 부족 시 code-domain 대체). **validation 1000 / coalition-subset 2¹⁰=1024 분리**(같은 "1024"가 두 의미로 섞여 있던 걸 발견 → 혼동 차단). plan §3.4·D6·[[flirds]]·[[flirds-protocol]] §8 4곳 일치.
- **built (stage 2, A=공통 core)**: `fl/server.py` `_fedavg_core` 추출(CNN `fedavg`/`run_fedavg_logs` wrapper 시그니처 보존, **회귀 bit-identical** — 이전 server.py 대조 1st 0.7381 / 1st+2nd 0.8810 동일) + `backends/llm.py`(`make_llm_loss`, LoRA-only 주입) + `fl/llm_server.py`(`run_llm_fedavg_logs`, TRL SFTTrainer 1.x + forced SGD + completion-only).
- **LLM-specific 발견 3** (CNN엔 없던): eager attention(SDPA forward-AD 미지원) / named_parameters key(≠`get_peft_model_state_dict`) / embedding require-grad hook clear(SFTTrainer grad-ckpt hook ↔ functorch 충돌).
- validated: backend 스모크(Qwen2.5-0.5B, est≈oracle 1.67e-4, per-layer invariant 2e-17) + **LLM-FL 스모크(Llama-3.2-1B real SFTTrainer 궤적, est≈oracle 1.70e-6)** → LLM FL loop end-to-end OK.
- env: transformers 5.9 / peft 0.19 / trl 1.5 설치(torch 2.12 유지); HF token(Yohez) + Llama-3.2 1B/3B access.
- distilled into: [[flirds-implementation-plan]] (status), MEMORY (stage 2 done + 3 musts).
- next: **3번 5-domain data layer** (validation 1000 stratified + seam 2 corruptor registry).

## [2026-06-04] note | seam 2 (a) — CNN corruptor registry (minimal)

- `flirds/data/corruptors.py` (`label_shuffle` + `CNN_CORRUPTORS`); phase05 dual/flirds_oracle/regime_sweep refactored to registry call — **bit-identical** (flirds_oracle 0.7381/0.8810 unchanged; dual/regime import OK).
- **의도적 최소**: sample-level label_shuffle만. `noisy={...}` set 유지 (corruptor 1종 → run-config map은 over-engineering). 풀 registry(run-config map + update-level free_rider + partition-level maverick/duplicate + LLM text corruptor)는 실제 쓸 때(Phase 2/3 + stage 3). plan §3.10 갱신.

## [2026-06-04] lint | plan reconciled against new raw conversations (de-duplicate stale framing)

- trigger: Yonghee — the implementation plan had missing bits + corrected content left as duplicates causing confusion. Checked plan vs the 4 newer raw conversations (2026-06-02-phase0-implementation, 2026-06-03-phase05-estimator, 2026-06-03-phase1-backend-abstraction, 2026-06-04-phase1-llm-stage2).
- **root cause**: [[flirds-implementation-plan]] top (status snapshot) got new Phase-1-progress bullets appended, but the older lower sections (§2 Phase structure, §3 open-decisions bodies, seam callout, §3.10) + [[flirds-protocol]] §6/§10 + [[flirds]] Section 3 kept the *pre-06-02* framing → same doc self-contradicted.
- **MISSING → added**: B1 cross-silo trainset-size equalization (06-04; aggregate weight stays size-prop, FiQA→code-domain swap); seam 2 fork → **(b) chosen** (§3.10); **3 LLM backend musts** (eager-attn / named-key state / embedding-hook clear) → new [[flirds-protocol]] §13 + TRL-1.x notes + 7B bf16-train/fp32-eval.
- **STALE/DUPLICATE → fixed**: snapshot self-contradiction (9-open / "next=Phase 1 start" vs stage-2-done) reconciled + dated 06-04; §2 Phase 1 marked stage 1–2 DONE; seam callout → seam 1 & 3 DONE, seam 2 remaining; §3 banner (3.1–3.8 resolved, table wins over bodies); §3.10 PENDING→(b); §3.7 + §6 + skeleton W&B → **local run-dir (no W&B, D2)**; §10 + flirds.md baseline table + #11: "code-unavailable"→**self-built** (GTG/ComFedSV code exists, non-forkable), **Ripple 62× vs AFedSV+/FedSV not GTG**, Ripple has no ground-truth-SV metric; detection **4→2** (FLDetector+STD-DAGMM) in 4 places; doc-map counts (11→18 items); §4/§5 kickoff marked superseded; renamed-heading anchor fixed.
- touched: [[flirds-implementation-plan]] (snapshot, §2 Phase 0/1, §3 banner, §3.7, §3.9, §3.10, §4, §5, doc-map), [[flirds-protocol]] (§6, §10, §11 skeleton, new §13, dates), [[flirds]] (Section 3 #2/#11, baseline table + correction note). All [[…]] links re-verified resolve.
- net: the plan now reads top-to-bottom without contradicting itself; binding decisions live in the 06-02 table + status snapshot; older bodies clearly marked historical.

## [2026-06-04] conv | Phase 1 stage 3 — 5-domain data layer + val micro-batching + free-form uniformity

- raw: [[raw/conversations/flirds/2026-06-04-phase1-data-layer]]
- distilled into: [[threads/dataset-format-uniformity]] (NEW — free-form-uniformity decision + parked dataset candidates + prior-work overlap + per-domain normalization & ablation), MEMORY (stage-3 done). **plan §3.1/§3.4 + [[flirds]] NOT yet updated** (concurrent D3-distill session — reconcile, see supersession below).
- built: `data/llm.py` (5-domain **free-form** loader: `build`/`build_val_batch`/`build_val_batches`/`build_val_batches_by_domain` + §3.4 validation 200/domain), **val micro-batching** (`backends/llm.py` `make_llm_loss(...,chunk_domains,n_domains)` + `core/flirds_estimator.py` `_chunked`; eager-HVP OOM → per-chunk `loss_chunks=(lf_c,weight_c)`, exact sum-decomposition; **CNN bit-identical 0.7381/0.8810** via `loss_chunks=None`), `experiments/{phase1_data_smoke,phase1_llm_5domain_smoke,phase1_hvp_profile}.py`. Smoke green: token-norm est≈oracle 1.15e-7, domain-norm 1.45e-7, chunked==single 3.8e-8.
- D3 RECOVERED: the 5-domain choice was in `raw/...2026-06-02-phase0-implementation:17` but never distilled → looked lost. (Lesson: raw-but-not-distilled = effectively lost.)
- decisions (Yonghee): N∈{5,10} (1 vs 2 clients/domain); PubMedQA `pqa_artificial` train pool, **B1 = size-control param** (start 1k); **est-vs-oracle matrix** (CNN {5,10}; LLM 1B N=5 + N=10 후순위; 3B N=5; 7B (b) N=5 / (a) ✗; 3 seeds); **format uniformity → all 5 free-form**, swap medical→`medalpaca/medical_meadow_medical_flashcards`(34k) + legal→`ibunescu/qa_legal_dataset_train`(97k); **per-domain normalization (macro-average) + ablation**. HVP profile: mem≈5+0.021·chunk·seq GB; (b) oracle = 2ᴺ·R·val·seq (fp32, dominant); oracle-chunk decouple measured 1.0x (FLOP-bound) → reverted.
- ⚠ **supersedes D3 medical=PubMedQA / legal=CaseHOLD** → D3-distill session must reconcile (FedDQC overlap 3/5→2/5; acceptable — FedDQC comparison is IRA-baseline-only + per-domain Acc unusable under uniform-loss valuation). prior-art: free-form-unify is FLAN/LESS/MATES-standard; cross-domain valuation-fairness framing under-addressed → novelty hook.
- next: ② LLM text corruptor (seam 2 full registry → noisy/free-rider AUROC); ③ LLM baselines port. D=ablation run deferred to real experiments.
