---
type: conversation
date: 2026-06-04
topic: flirds
participants: [Yonghee, Claude]
tags: [flirds, phase-1, data-layer, validation, val-microbatching, normalization, datasets, comparison-matrix]
---

# 2026-06-04 — Phase 1 stage 3: 5-domain data layer + val micro-batching + free-form uniformity

Session goal: finish the Phase-1 data layer (5-domain loader + validation + B1 + corruptor) then LLM baselines port. Yonghee drove every design fork; Claude built + verified. Commit at the end; continue corruptor/baselines next session.

## D3 datasets RECOVERED (raw-but-not-distilled = effectively lost)
Yonghee: "in a previous session I changed code→fiqa and fixed the 5 domains + which dataset per domain, but it's not recorded?" — Claude verified plan/§3.1/memory/git had no such record, then found it in **`raw/conversations/flirds/2026-06-02-phase0-implementation.md:17` (D3)** — never distilled. The recovered D3 5-domain: medical=PubMedQA / legal=CaseHOLD / finance=FiQA / math=AQUA-RAT / general=Dolly (code→FiQA swap; 3/5 overlap FedDQC). **Lesson: a decision in raw but not distilled is effectively lost — check raw transcripts before trusting "not recorded".**

## Decisions made (Yonghee)
- **D-A (client↔domain mapping)**: cross-silo N∈{5,10}. N=5 → 1 domain/client (client == domain). N=10 → 2 clients/domain (disjoint halves). N=5 IS in the experiments (the (a) retrain oracle's cheap config).
- **D-B (PubMedQA train pool)**: using non-gold data is fine — `pqa_artificial`(211k) is the standard train pool, `pqa_labeled`(1k) = gold held-out val. **B1 (per-domain equalized train) is a size CONTROL variable, not an availability limit** → make it a loader param; start 1k/domain, finalize after results.
- **D-C (val micro-batching first)**: implement val micro-batching before anything else; analyse seq×val for fast iteration.
- **D-D (oracle cost)**: separate the fast parts to cut the bottleneck (#1 oracle-chunk decouple); measure whether #1 alone suffices before adding val/R-cut (#2). [Outcome: #1 measured 1.0x — oracle is FLOP-bound, reverted; bf16-eval ruled out as the loss-diff signal < bf16 precision.]
- **D-E (comparison matrix, all 3 seeds)**: CNN (a)&(b) at N∈{5,10}; LLM 1B (a)&(b) N=5 now + **N=10 후순위(맨 마지막, 비용 큼)**; LLM 3B N=5 only; LLM 7B (b) N=5, (a) ✗. CNN N=10 gives the cheap high-power 10-pt validation; LLM N=10 reconfirms on the real model but deferred. **3 seeds everywhere — no 1-seed anchors** (consistency). Worried "N=10 CNN-only + LLM-N=5-only" is unconvincing → resolved by CNN-N=10 (high power) + LLM N=10 deferred-but-planned.
- **D-F (FORMAT UNIFORMITY — the deep one)**: Yonghee — heterogeneous task formats break fair cross-domain valuation ("some are instruction, some aren't"). Asked Claude to (1) check if formats can be unified + find prior-art precedent, (2) find replacement datasets if not. → Decision: **unify all 5 to free-form instruction→response**; **swap medical PubMedQA(yes/no/maybe classification)→`medalpaca/medical_meadow_medical_flashcards`(34k), legal CaseHOLD(5-way MC)→`ibunescu/qa_legal_dataset_train`(97k)**; finance(FiQA)/math(AQUA rationale)/general(Dolly) kept (already free-form). **Also do per-domain normalization + an ablation (norm ON vs OFF → accuracy).** Record the abandoned candidates (+ prior-work overlap) to revive later if results warrant.

## Prior-art research (2 agents + wiki) — answering D-F
- **Unify-to-free-form is the STANDARD** (FLAN/T0/Super-NaturalInstructions; LESS/FedDQC/IFD/NUGGETS inherit). Classification can be wrapped (OPTIONS suffix) or recast; loss on target tokens only.
- **Valuation uses a uniform LM loss; per-domain task metrics (Acc/win%/BERTScore) are for final eval only** (FedDQC).
- **[[sources/mates|MATES]] objective-alignment** (val objective == training objective = autoregressive generation) is the direct precedent for the concern.
- **The specific cross-domain magnitude-fairness concern is UNDER-ADDRESSED in prior art** (the data-selection survey calls it a gap) → **novelty hook**. Prior partial mitigations: per-token mean (LESS), loss-difference (FedDQC IRA `L(a)−L(a|q)`), loss-ratio (IFD), binary indicator (NUGGETS). Flirds already has 2 (per-token mean + Δloss utility); the shared-val Shapley makes the confound sharper.
- Rejected/parked dataset candidates + overlap + licenses → **[[threads/dataset-format-uniformity]]** (created this session).

## Built + verified (all green)
- **`data/llm.py`** — `build(n_clients, per_domain_train, per_domain_val, seed)`→(clients, val_records); free-form formatters (flashcards/ibunescu/FiQA/AQUA/Dolly); `build_val_batch` (completion-preserving truncation); `build_val_batches` + `build_val_batches_by_domain` (domain-pure chunks for per-domain norm). N=5→1/domain, N=10→2/domain; val §3.4 200/domain dev-split-first (finance test / math val splits; medical/legal/general carve). End-to-end real-data smoke: est≈(b)oracle.
- **val micro-batching** (`backends/llm.py` + `core/flirds_estimator.py`) — single-shot HVP over eager attention OOMs at val=1000 (eager required for forward-AD; double-AD graph ∝ chunk·seq). Fix: `make_llm_loss(model, val_batches, device, chunk_domains, n_domains)`→`(loss_fn, pkeys, loss_chunks)`; `loss_chunks=(lf_c, weight_c)`, lf_c = per-chunk MEAN; estimator `flirds_values(..., loss_chunks=None default)` opt-in `_chunked` = Σ weight_c·grad/HVP (exact, sum decomposes). **CNN untouched (`loss_chunks=None`) → bit-identical 0.7381/0.8810.**
- **per-domain normalization** — token-norm `weight=n_c/Ntot` (default) vs domain-macro-average `weight=n_c/(D·n_d)` (`chunk_domains` given). Ablation flag.
- **HVP profile** (`experiments/phase1_hvp_profile.py`) — mem ≈ 5 + 0.021·chunk·seq GB; max chunk seq256→24 / seq384→16 / seq512→11; estimator/round ≈ 0.31·val·seq ms (chunk-independent); (b) oracle = 2ᴺ·R·val·seq (dominant; N5↔N10 = 32×, fp32 required).
- **legal truncation** — CaseHOLD prompts 325-462 tok all >256; at small max_length the short index completion is right-truncated away → loss 0 → φ=0. Confirmed + fixed by max_length=768 (φ≠0), but **moot** since CaseHOLD was dropped for free-form ibunescu.
- Smoke numbers: token-norm est≈oracle 1.15e-7, domain-norm 1.45e-7, chunked==single 3.8e-8.

## ⚠ Supersession flag (coordinate with the concurrent D3-distill session)
The free-form swap **supersedes D3's medical=PubMedQA / legal=CaseHOLD**. The D3-distill session (distilling the recovered D3 + plan §3.1/§3.4) must reconcile: medical=`medalpaca/medical_meadow_medical_flashcards`, legal=`ibunescu/qa_legal_dataset_train` (NOT PubMedQA/CaseHOLD); FedDQC overlap drops 3/5→2/5 (FiQA, AQUA), acceptable since FedDQC comparison is "IRA baseline only" and its per-domain Acc is unusable under uniform-loss valuation. Plan §3.1/§3.4 + flirds.md not edited this session (avoid concurrent-edit conflict) — distillation of the data-layer/normalization decisions still pending there; the durable record is [[threads/dataset-format-uniformity]] + MEMORY.

## Next session
② LLM text corruptor (seam 2 full registry — enables noisy/free-rider AUROC); ③ LLM baselines port (GTG/FedSV/ComFedSV/Ripple CNN→LLM). D = ablation RUN (norm ON/OFF → downstream acc) deferred to real experiments. Data layer complete → "first clean 1B run" (Phase 1 #7) unblocked.
