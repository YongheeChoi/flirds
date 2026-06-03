---
type: conversation
date: 2026-06-03
topic: flirds
participants: [Yonghee, Claude]
tags: [phase-1, estimator, oracle, partial-participation, per-layer-logging, backend-abstraction, openfedllm, loss-fn]
---

# Phase 1 kickoff — estimator/oracle generalization + backend-agnostic refactor (CNN side)

A second 2026-06-03 session (separate from the phase05-estimator session and the
wiki recent-prior-work-scan session). Goal: start Phase 1 (LLM port) by first
making the Phase 0.5 estimator/oracle Phase-1-ready, then scouting OpenFedLLM.
No LLM code yet — that is handed to the dedicated Phase 1 session.

## Yonghee's asks & decisions (in order)

1. **"파악부터 해줘."** Caught up on Phase 0/0.5 + the plan. Surfaced that MEMORY
   said "Phase 0.5 uncommitted" but git log shows it committed (`386c455`…`a51f11f`)
   — memory was stale.
2. **Hybrid (A)→(B)**: read OpenFedLLM as reference ((A)), then self-build into our
   own structure ((B)) — per the "fork = reference-guided self-build" rule. Cloned
   OpenFedLLM to gitignored `codes/external/` (reference only, not `base_repo/` fork).
3. **"cross-device도 해야 하니 estimator/oracle full-participation 가정도 수정해줘."**
4. **§3.4 validation question**: "선행 연구 세팅을 그대로 하기 힘든 상황인가?"
5. Mid-session, relayed the wiki session's update (12 new papers, experiments
   #12–18, 3 Phase-1 seams). Asked to absorb and continue.
6. **Two (B) forks decided**: backend abstraction = **loss_fn closure injection**
   (not a Backend object); LLM local train = **TRL SFTTrainer + forced SGD**.
7. **Stop here**: LLM 2단계는 기존 Phase 1 세션에서 신선하게. Record (KARIS) +
   tidy + commit now.

## What Claude did / resolved

- **validation §3.4 answer**: can't reuse prior-art as-is — not for lack of
  precedent but because each prior method's *validation concept differs*: LESS =
  targeted few-shot dev (we have no single target — 5-domain global quality);
  FedDQC = validation-free on-device IRA (no server val set to borrow); IRDS =
  held-out target loss (closest, but single-corpus, no domain-mix recipe). The
  only open piece is "how to mix/aggregate the 5 domains' canonical dev sets"
  (uniform vs proportional) — that's why §3.4 is open. Flagged the D6-table
  (per-domain ~50–200 uniform) vs §3.4-body (1024 integrated + 256/domain)
  inconsistency; recommended **D6 direction, ~200/domain → ~1000 total**. Seam 3
  confirms val is already config-driven (args injection) — no Phase-1 code change.
- **weight-semantics fork dissolved**: worried the partial-participation weight
  change would break the locked "no participation normalization". It does NOT —
  they are different axes: the lock is the *Shapley-aggregation* axis (don't
  divide φ_k by participation count); the change is the *FedAvg per-round weight*
  axis (p_k^r = n_k/Σ_{P_r}n_j, matching server.fedavg's realized aggregate and
  the (b)-oracle definition w^r+Σ p_k Δw_k). #14 (duplicate-client) validates the
  lock; the per-round weight is a correctness fix, not a choice. Under full
  participation p_k^r == global p_k → bit-identical regression.
- **estimator/oracle generalized** (`core/flirds_estimator`, `oracle/in_run_sv`):
  read participants per round from `deltas_map.keys()` (pattern already in
  `baselines/fedsv`), per-round participant-normalized weight, sum over
  participated rounds. Added **per-layer φ logging** (seam 1) as `per_layer=False`
  default returning the per-(client,param) components — INVARIANT: Σ components
  == φ_k bit-identical, spine never reweights. `exact_sv` (a-oracle, retrain)
  untouched (partial-participation-irrelevant).
- **(A) OpenFedLLM scout**: `main_sft.py` loop → `get_peft_model` (LoRA),
  `global_dict`=LoRA state, `get_clients_this_round` (random.sample = partial),
  `global_aggregate` fedavg = Σ local·n_c/Σ_{round}n_c. **Its aggregate weight
  equals our new per-round weight** — cross-device port natural. Δw extraction:
  OpenFedLLM stores absolute local state → our `logs` = (global_before,
  {c:(local−global, n_c)}). 3 backend-specific seams only: loss_fn, pkeys
  (trainable/LoRA filter), val-batch format; the rest is already agnostic.
- **(B) stage 1 — backend abstraction**: `backends/cnn.py:make_cnn_loss(model_fn,
  vx, vy, device) → (loss_fn, pkeys)`; `loss_fn(params, buffers)` via
  functional_call+CE (estimator differentiates it, oracle calls it @no_grad).
  estimator/oracle signatures now `(logs, loss_fn, pkeys, device, …)` — model/val
  hardcoding removed. phase05 ×5 updated to call `make_cnn_loss`.

## Validation (all green)

- Full-participation regression bit-identical: dual_oracle est/(b) values, 3-seed
  1st+2nd 0.962 / 1st 0.924, efficiency/symmetry/repro 0.0 — unchanged before &
  after both the partial-participation change and the loss_fn refactor.
- Partial-participation smoke (sample_frac 0.7): per_layer invariant Σcomp−φ =
  0.0; est≈(b) Spearman 1.000; efficiency 1.39e-17. (noisy AUROC 0.0 here is a
  setup artifact — uneven participation tiers under "no participation
  normalization" → cross-tier AUROC blurs; an inadvertent demonstration of #14.)
- phase05 3 (flirds_oracle/sanity/regime_sweep) after refactor: N=2 relL2 0.00312,
  2nd-order regime-dependence (near-flat tie / in-radius help / overshoot hurt) —
  matches the wiki record.

## Handoff to the Phase 1 (LLM) session

- (B) stage 2 = `backends/llm.py` (make_llm_loss: LoRA `requires_grad` pkeys +
  causal-LM next-token loss + fp32-eval guard) → LLM FL loop self-build
  (SFTTrainer+SGD, Δw/logs extraction, OpenFedLLM main_sft as reference) → data
  layer (5-domain + seam 2 corruptor registry, §3.10 fork-(b)).
- estimator/oracle need NO further change for the LLM port — only a new backend.
- Open: §3.4 val-mix decision (D6 ~200/domain recommended); seam 2 (a)/(b) fork.
