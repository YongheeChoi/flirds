---
type: conversation
date: 2026-06-07
topic: flirds
participants: [Yonghee, Claude]
tags: [phase2, dual-oracle, retrain-shapley, exact-sv, validation, val-loss-vs-rouge, cost-extrapolation, fp32-precision]
---

# Phase 2 task 6 — (a) exact retrain SV oracle at LLM scale + dual-oracle validation

## Yonghee's ask + the climb plan

Implement task 6 = the **(a) exact retrain Shapley oracle** for LLM (the missing half of the
dual oracle; `oracle/exact_sv.py` was CNN-only). Climb gradually, **recording the timing
carefully at N=5@1B** so the N=10 (1024-retrain) and 3B costs can be extrapolated before
committing. Sequence settled: **N=5@1B first → N=5@3B → N=10 deferred to the real experiment**
(too expensive for the validation phase).

## What was built

- `flirds/oracle/exact_sv_llm.py` — `llm_subset_utility(...)` returns `utility(S)`: reset a
  fresh LoRA model to init, FedAvg-retrain on the clients in S only (free-riders remapped to
  subset-local), reconstruct the final global LoRA state (`_final_lora_state`), and SCORE the
  deployed model. `exact_shapley` (the 2^N kernel) reused from `exact_sv` unchanged.
  Optional `timing` list records `(|S|, retrain_s, eval_s)` per coalition.
- `experiments/phase2_llm_a_oracle.py` — builds the N=5 trajectory (noisy={0}, free-rider={1}),
  runs (b) in-run + estimator on the fp32-eager frozen trajectory, then the (a) retrain oracle,
  and reports the 4-way comparison + the per-|S| timing breakdown + an N=10 extrapolation.
  Env knobs (A_DTYPE / LR / VAL_CHUNK / BATCH / ...) for per-scale runs. SMOKE/FULL configs.

## Yonghee's two design corrections (the heart of the session)

1. **(a) must use VAL-LOSS as the primary metric, not ROUGE.** Rationale (his): to verify our
   *method* (estimator / (b) in-run) computes the Shapley value **correctly**, the (a) retrain
   exact Shapley must solve the **same game** (val-loss). A ROUGE-Shapley is a *different game*,
   so a disagreement could be just the metric, not a method error. I had wrongly framed
   (a)=ROUGE as primary (CNN test-acc analogue). → (a)-val-loss vs (b)-val-loss is the method
   validation; ROUGE is the secondary "different-utility / deployment" figure.
   - The fundamental asymmetry (why it's not fully symmetric): the **estimator is a Taylor
     expansion of the validation LOSS** → needs a differentiable scalar. ROUGE (argmax-generated
     text overlap) is non-differentiable → there is **no estimator-ROUGE**. (b) uses val-loss to
     be the exact oracle the estimator approximates; (b)-ROUGE would need 2^N·R generations and
     is meaningless on a frozen-delta perturbation. (a) retrain evaluates a *final model* → free
     to use any metric. So: (a) can do {val-loss, ROUGE}; (b)/estimator can only do val-loss.

2. **"Won't proper (more) training make val-loss change more? We're running small (smoke-ish)."**
   Correct that the per-coalition retrain (train=200 / R10 / max_steps10 / lr1e-3, instruct-tuned
   base) trains very little (#7's val_loss Δ~0.005), so φ are tiny. The real experiment trains
   far more (train=12k / R50), but the (a) oracle does 2^N retrains so can't match that per
   coalition. → tested by re-running at lr=3e-3 (bigger signal) and in fp32 (precision).

## Results

### Dual-oracle validation (N=5@1B, fp32) — METHOD VALIDATED
| run | (a)valloss vs (b) | estimator vs (b) | (a)ROUGE vs (b) |
|---|---|---|---|
| fp32 lr=3e-3 | **+1.000** | +1.000 | +0.40 |
| fp32 lr=1e-3 | **+1.000** | +1.000 | +0.30 |

The true retrain counterfactual **(a)-val-loss**, the in-run frozen **(b)**, and the **Flirds
estimator** produce the **identical client ranking** (math > finance > general > medical-noisy >
legal-freerider), Spearman **+1.000**, at BOTH lr. So (b)/estimator compute the correct Shapley
value (validated against the real retrain, same metric). AUROC noisy=0.75 / free-rider=1.0,
identical across (a)valloss / (b) / estimator.

### Key insight: it was bf16 PRECISION, not signal strength
The earlier weak agreements (smoke (a)valloss-vs-(b) = +0.3) were **bf16**: the (a)-val-loss
coalition differences are ~0.005–0.02, at/below bf16 absolute precision (val-loss 2.4 × 2⁻⁸ ≈
**0.009**) → noise. In **fp32** the validation is +1.000 at *both* lr (signal size irrelevant to
the ranking agreement at this scale). This is exactly the (b)-oracle's fp32 rationale. Yonghee's
"bigger signal" instinct also helps — but only in bf16 (lifting φ above bf16 precision); the
root enabler is fp32. (lr / signal size still matters for detection *headroom* at larger N.)
Caveat: N=5 is coarse (5 clients); the high-power validation is N=10 (deferred) / CNN N=10.

### ROUGE divergence (kept as an interesting secondary note, Yonghee)
(a)-ROUGE diverges from the val-loss game: (a)ROUGE-vs-(b) = +0.4 (1B) and **−0.9 (3B)**;
(a)ROUGE-vs-(a)valloss = +0.3–0.4. So the (a)-vs-(b) gap is **100% the metric, not the method**
(same metric → retrain and in-run agree perfectly). Mechanism: the noisy client is answer_swap
(medical text kept, answers permuted) → retraining on it improves test ROUGE (domain-format
learning) while val-loss correctly penalizes the wrong content. **Downstream-ROUGE is fooled by
domain-style-preserving corruption; val-loss (Flirds' utility) is not** — supports the val-loss
choice. (φ magnitudes tiny → also some test-sampling noise.)

### Cost ladder (Yonghee's gauge) — per-|S| timing is clean & linear
| | (a) N=5 | retrain/client | eval/coalition | N=10 extrapolation |
|---|---|---|---|---|
| 1B bf16 | 47 min | 27.5 s | 19.7 s | ~45 GPU-hr (~1.9 d / 4-GPU ~11 h) |
| 1B fp32 | 126 min | 85.2 s (×3.1: no tensor cores) | 23 s | ~67 GPU-hr |
| 3B bf16 | 90 min | 53.2 s | 35.6 s | ~86 GPU-hr (~3.6 d / 4-GPU ~22 h) |

retrain is 78–90% of (a); per-|S| retrain is exactly linear (b·|S|). Proper N=10 extrapolation:
retrain scales **64×** (Σ_s C(10,s)·s = 10·2⁹ = 5120 vs N=5's 80), eval **32×** (coalition count).
→ N=10 is a 2–5 day single-GPU run at every scale → **deferred to the real experiment** (with
multi-GPU coalition sharding → ~11–22 h). N=5 validation (~1–2 h) is the affordable "now" tier.

## Decisions / status
- **Method validated at N=5@1B**: (a)-val-loss = (b) = estimator, Spearman +1.000 (fp32).
- (a)=val-loss is the primary (method-validation) metric; (a)-ROUGE is the secondary
  deployment/different-utility figure (and is corruption-fooled — a note for the paper).
- N=5@3B (a)-val-loss fp32 launched as a background confirmation (matrix: "3B both").
- N=10 (both scales) deferred to the real experiment (multi-GPU sharding).
- Infra: `oracle/exact_sv_llm.py` + `experiments/phase2_llm_a_oracle.py` (env-parameterized).

## Notes / open questions
- A real N=10 (a) oracle needs **multi-GPU coalition sharding** (1024 independent coalitions) —
  a small orchestration addition, not yet built (Phase 2 task 6 real-experiment time).
- bf16 N=5 dual-oracle is precision-limited for val-loss → always run the (a)-val-loss
  validation in fp32 (ROUGE-only runs can stay bf16/deployment).
- `generate_completions` emits a BPE `clean_up_tokenization_spaces` warning (shared helper) —
  benign and consistent across coalitions; out of scope here.
