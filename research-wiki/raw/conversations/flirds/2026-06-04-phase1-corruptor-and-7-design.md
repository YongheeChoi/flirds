---
type: conversation
date: 2026-06-04
topic: flirds
participants: [Yonghee, Claude]
tags: [phase-1, seam-2, corruptor, free-rider, noisy, downstream-metric, first-clean-run, prior-art-grounding]
---

# Phase 1 — seam-2 LLM corruptor (②) + #7 first-clean-run design

Continuation session. Yonghee: continue Phase 1 in the order ② LLM text corruptor (seam 2 full
registry) → ③ LLM baselines port → first clean 1B run; "decide the corruptor kinds together."

## ② LLM text corruptor (seam 2) — DONE + verified

**Decisions** (Yonghee drove each; he twice pushed "is there prior work?" before committing the
free-rider mechanism — reference-guided as always):
- **Scope** = noisy + free-rider (plan Phase-1 task 1). backdoor/PGD/maverick/duplicate stay Phase 2/3.
- **noisy** = `answer_swap` (permute the completion column within a client → prompt pairs with the
  wrong answer; CNN `label_shuffle` analog; FedDQC answer-swap + FedCorr data-side freeloader precedent).
- **free-rider** = update-level, **zero + random** modes. Grounded in **Lin et al. 2019 (arXiv 1911.12560,
  the STD-DAGMM origin = our free-rider detection baseline)** attack taxonomy (zero / random / delta /
  advanced-delta, easy→hard). delta/advanced (recycle previous-round aggregate ± noise) deferred to
  Phase 2 (need the prev-aggregate threaded into the FL loop + are the STD-DAGMM head-to-head + the
  source-page open TODO "does a recycled-average carry early aligned signal?").

**Prior-art sweep on "free-rider" (Yonghee asked if only one paper covers it)** — the wiki has ≥5,
but they model DIFFERENT notions, and two map 1:1 onto our two corruptors:
- STD-DAGMM (Lin 2019) — **update-side fabrication** (zero/random/delta) → our `free_rider`.
- FedCorr — **data-side zero-effort "freeloader"** (random-label client) → our `answer_swap` (noisy).
- Asymmetric Data Shapley — **replication gaming** (duplicate data) → the Phase-3 `duplicate` corruptor.
- iPFL (economic misreporting), DICE (low cascading influence), FLDetector (low-effort poisoner) — concept-level only.
- Fraboni et al. 2021 (AISTATS, "Free-rider Attacks on Model Aggregation") = the other canonical
  update-fabrication paper (plain = zero, disguised = +stochastic noise) ≈ Lin; **NOT in our wiki** (revivable).
- **Key insight**: "free-rider" is not one thing; our two Phase-1 corruptors already span the two main
  operational notions. So Lin's 4-family is the right menu for the update-level one; FedCorr covers the data-side.

**Implementation** (mirror the CNN minimal registry; CNN path untouched → bit-identical):
- `data/corruptors.py`: `answer_swap` + `LLM_CORRUPTORS` (sample-level) + `free_rider(ref, mode, scale, generator)`
  (update-level, representation-agnostic; zero = zeros_like, random = U(-scale,scale), Lin taxonomy in docstring).
- `data/llm.py build(..., noisy=frozenset())` applies answer_swap per client index.
- `fl/llm_server.py run_llm_fedavg_logs(..., free_riders=frozenset(), free_rider_mode="zero")` + a reproducible
  `fr_gen`; free-rider client skips training and returns the fabricated delta. NO change to `_fedavg_core`/estimator/oracle.

**Verification** (3 levels): (1) unit — answer_swap 8/8 mismatch·reproducible·prompts intact; free_rider
zero/random keys/cpu/shape/inscale. (2) **CNN bit-identical** — phase05_flirds_oracle 0.7381/0.8810 unchanged.
(3) **real 1B integration** (`experiments/phase1_llm_corruptor_smoke.py`): free-rider zero φ EXACTLY 0
(est & oracle), max|est-oracle| 2.7e-7 with corruptors in the mix, random φ small/bounded, finite. Noisy
detection QUALITY (noisy=argmax) is experiment-scale (noise floor at 8ex/2step) → validated at #7, only reported.

## ③ vs #7 — RESEQUENCED to #7-first (Yonghee chose (B))

"LLM baselines port" (③, memory-tagged Phase 2 SV baselines) vs the first clean run (#7): #7's 4 metrics
(selection-convergence / task-acc / F1 / noisy-AUROC) DON'T need the SV baselines — only training baselines
(full/random FedAvg) + a selection mechanism + an eval harness. Yonghee picked **(B): #7 clean run first
(de-risk the scale run — OOM/convergence/timing), SV baselines after.** Both have prerequisite work; (B)
validates the Flirds pipeline at real scale before investing in 4 baseline ports.

## #7 downstream metric — FedHDS-style ROUGE-L + math EM (prior-art grounded)

Yonghee asked (third grounding push) for prior work on scoring downstream task perf of selected
instruction-tuning data. Survey:
- **FedHDS** (closest FL+LLM, == our cross-device track) → **held-out unseen-task ROUGE-L** (+ accuracy).
- FedDQC (closest FL quality-control) → task perf on the bench; IRA is the *selection* metric, eval is task perf (separation).
- LESS / MATES / DsDm (centralized) → external benchmark-suite accuracy (MMLU/BBH/TyDiQA; MATES also the
  "rounds to fixed downstream-acc" curve = our selection-convergence template).
- dpo-shapley → reward-model score (DPO setting).

**Decision = FedHDS-style: per-domain held-out ROUGE-L + math (AQUA) exact-match.** Rationale: FedHDS is the
closest FL+LLM precedent and literally our cross-device bench; our 5 domains are the task distribution with
§3.4 held-out per domain; free-form-appropriate; matches plan's "task acc/F1"; math has a clean final-answer
letter so it gets the native EM metric (as LESS/DsDm use task-acc where unambiguous).

**Cross-cutting insights recorded**:
- **utility ≠ downstream metric** (utility-function-design thread + FedDQC): valuation utility = val-LOSS
  (IRDS Taylor, what estimator/oracle use); downstream = ROUGE-L/EM (what selection-convergence/task-acc report).
  Deliberately distinct — answers "why not just use val-loss downstream."
- **selection-convergence** definition (conversation2.md:42) = train on φ-selected top-K → convergence
  speed / final perf vs full & random; MATES Fig 1b/3 ("2.3× faster to fixed acc") is the presentation template.
- **caveat** (FedDQC, DsDm): gradient/similarity selection can UNDERPERFORM random on real heterogeneous FL
  (FedDQC: DataInf < random on Fed-WildChat; DsDm: similarity < random) → the bar Flirds-selection must clear.

## Built this session
- `flirds/data/corruptors.py` (+answer_swap/LLM_CORRUPTORS/free_rider), `data/llm.py` (noisy=), `fl/llm_server.py`
  (free_riders=/free_rider_mode=), `experiments/phase1_llm_corruptor_smoke.py` (new). ② complete + verified.
- `flirds/eval/metrics.py` (ROUGE-L F1 + extract_choice/choice_match + detection_auroc) — pure, unit-tested.

## Remaining for #7 (next)
run_logger (config+env+git SHA + per-round φ parquet, D2/protocol §6) · generation harness + held-out TEST
split per domain (math carries gold `correct`) · client-selection + selection-convergence (φ→top-K→retrain) ·
training baselines (full / random-selection) · orchestrator `phase1_clean_run.py` · scale run (N=5, R~50,
per_domain_train B1, 3 seeds) with a corruption setup (noisy + free-rider among the 5).

## Session continued — #7 infra BUILT + verified (same session, after the design above)

Yonghee said "continue"; the rest of #7's infra was built + verified in one arc (prior-art-grounded
at each fork — Yonghee pushed for grounding on the free-rider mechanism, the downstream metric, AND the
size allocation):

- **Sizes finalized (Yonghee, prior-art survey first)**: per-domain **train=12,000 / val=200 / test=2,000**,
  all disjoint. Survey: carving test from train is the standard fallback when no native test split exists
  (LESS=external benchmarks, FedHDS=held-out tasks, CNN-canon=dataset test, Dolly=carve); native test too
  small here (math 254, finance 2561) so test is train-carved uniformly, val stays native (finance `test`
  / math `validation` / med·legal·gen carve). finance's 14.5k train is the equalized-train ceiling
  (train+test=14k, ~500 spare; general 15k → ~811). Flagged: test=2000 is carved-not-native (gen cost ~10×;
  train=12k only helps if local steps scale, else mostly unused + Taylor-fidelity risk) — Yonghee kept 12k/2k.
- **eval**: `flirds/eval/metrics.py` (rouge_l LCS-F1 + extract_choice/choice_match AQUA EM + detection_auroc;
  **fixed re.I bug** where [A-E] matched the next word's first letter) + `flirds/eval/generate.py`
  (generate_completions left-pad greedy + use_cache=True; score_records per-domain). Unit-verified.
- **data layer**: `_domain_split` → 3-way no-waste id-carve; `build(..., per_domain_test=)` →
  `(clients, val_records, test_records)` (test tagged domain + math gold). Data smoke: 3-way disjoint=0
  all 5 domains + math gold; corruptor-smoke regression OK (est≈oracle 2.4e-7).
- **run_logger**: `flirds/run_logger.py` (config.yaml + meta{git SHA+dirty+env hash+versions} + φ parquet
  + metrics json; §6/D2). Roundtrip-verified.
- **orchestrator**: `experiments/phase1_clean_run.py` (per-seed Flirds φ + (b) oracle at N=5 → AUROC →
  selection arms full/random-K/Flirds-top-K with per-round val-loss curves post-hoc off logs → final
  per-domain task-acc via generation → RunLogger; FULL/MINI/SMOKE). **SMOKE green**: est≈oracle 1.56e-7,
  noisy AUROC 1.0, arms+gen+log all work.
- **/code-review (high, 4 finders)**: 1 real bug (extract_choice re.I) fixed+reverified; rest by-design
  (int-dtype free_rider on unused CNN path; metrics/orchestrator ahead-of-use by plan; free-rider hook
  LLM-only = Phase-2 refactor point; free_rider `scale` kept).

**Status**: #7 infra COMPLETE. A MINI de-risk run (train 500 / R 10 / test 200 / 1 seed, ~30min) launched
in background to check the noisy/selection signal before the ~5–7h FULL 3-seed run (`CLEAN_RUN_MODE=full`).
Then ③ = SV baselines port. Committed on `main` (not pushed — no stored creds; Yonghee pushes).

## Session 3 cont. — #7 scale run de-risk + lr-sweep (2026-06-04→05)

Launched the #7 scale run; a long de-risk loop followed (recorded so the journey isn't lost):

- **(b) oracle cost blowup**: first mini+full (R=50, ORACLE_B=True) hung ~9h in the (b) in-run oracle
  (2^N·R·val·seq — the wiki-flagged DOMINANT cost; not a bug, CPU≈elapsed = computing). Killed. Fix:
  **ORACLE_B env-toggle OFF by default** (est≈oracle already validated at R=10/smoke; #7 headline metrics
  use the estimator φ, not the oracle) + **phase prints + `python -u`** (the silent oracle phase had
  frozen the log → looked hung; now real-time).
- **flat-training finding + reframe**: val_loss barely moved at R=10 AND R=50. Causes (Yonghee corrected
  my "training too weak" framing): (i) **plain-SGD @ lr=2e-5 is very slow** (§3.3's 2e-5 is an Adam value;
  SGD needs ~1e-3); (ii) the target is **Llama-3.2-1B-Instruct = already instruction-tuned → limited
  headroom** (start val_loss ~2.43, ppl ~11); (iii) 5 disjoint domains = extreme non-IID FedAvg cancellation.
- **lr-sweep** (4 GPUs parallel, oracle-on small val=100/R=20, lr {1e-4,3e-4,1e-3,3e-3}): learning ∝ lr
  (val_loss Δ 0.003→0.073); **est≈oracle tight through 1e-3 (6.2e-6), jumps to 7.6e-5 at 3e-3** (still
  <1e-4 = essentially exact). free-rider=**zero AUROC 1.0**; noisy **0.75** (answer_swap is MILD — shuffled
  completions are still valid same-domain text; + the free-rider's φ=0 outranks it). Arms DO differentiate
  (drop-corrupted → lower val_loss) so the **selection signal is real but small** (Δ~0.005; the FedDQC/DsDm
  "hard to beat random" bar).
- **lr decision**: Yonghee corrected my over-conservative 1e-3 pick — both lr's est≈oracle is essentially
  exact, so the tighter one doesn't matter; the actual goal (learning/selection) favors the higher lr →
  **run BOTH 1e-3 and 3e-3 as full**.
- **efficiency (Yonghee)**: CPU is NOT overloaded (72 cores, loadavg ~2, the proc is single-threaded/1 core);
  GPU 0 ~100% but only 1 of 4 GPUs used. Fix = seed/cell parallelism (`RUN_SEED` → one seed per GPU, ~3-4×).
  Durable opts for the matrix: tokenize-once (FL re-tokenizes every round) + bf16-train/fp32-eval (~2×).
  The real-matrix-cost worry: the expensive bit is the ORACLE (validation-only, small N) — the METHOD
  (estimator) is cheap by design; the 144-cell×3-scale matrix is parallelizable (plan budgeted ~3 weeks).
- **NOW RUNNING (4 GPUs, estimator-only)**: lr1e-3 s0/s1 (GPU0/1) + lr3e-3 s0/s1 (GPU2/3), ~6-10h each;
  RUN_ROOT=runs/full_lr{1e-3,3e-3}. Remaining seed-2 of each → next session. Read with
  `experiments/read_runs.py`. Config changes (ORACLE_B/RUN_SEED/RUN_LR/SWEEP/phase-prints/use_cache) committed.
