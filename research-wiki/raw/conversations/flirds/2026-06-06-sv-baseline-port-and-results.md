---
type: conversation
date: 2026-06-06
topic: flirds
participants: [Yonghee, Claude]
tags: [phase1, phase2, sv-baselines, gtg, fedsv, ripple, llm-port, results, debugging]
---

# SV-baseline LLM port (GTG/FedSV/Ripple) + #7 scale-run results

Continuation session. The Phase-1 #7 FULL runs (lr1e-3·lr3e-3 × s0/s1) were already running from a prior session; this session **ported the SV baselines to LLM (Phase 2 task 1)** while they finished, then **ran the comparison batch** as GPUs freed.

## Yonghee's asks / decisions (chronological)
- Continue Phase 1; do post-#7 work that doesn't depend on the running results.
- **Scope: GTG+FedSV port first** (chose this over +comparison-experiment or +Ripple) — port + verify, review after.
- GPU busy → **do verification-free implementation now, batch GPU verification later**. (A single 1B smoke can't split across 4 GPUs; multi-seed compare distributes.)
- **"c로 넘어가자"** → also port Ripple this session. **Approach A**: materialize the ripple-term P-dim arrays (N∈{5,10}, P≈12M LoRA → ~20–40GB ≪ 2.2TB server RAM; stream-projection only needed at cross-device N=100, a separate unbuilt loader). Confirmed CPU-RAM storage doesn't slow it (bottleneck = GPU HVP; GPU→CPU transfer ~3ms ≪ HVP ~0.5s); the real cost is the eigsh HVPs (runtime is Ripple's reported metric).
- **Auto-poll GPUs; run the batch automatically when one frees** (Claude drives).
- After results: **GPU free → launch #7 s2 for both lrs** (complete the 3-seed).
- Wrap-up: **do (a) /code-review + commit and (b) research-wiki logging; continue Phase 2 remaining in a different session.**

## What was built
- **GTG/FedSV → backend-agnostic** via shared helper `_round_metrics(gb,dm,players,model,test_loader,device,loss_fn,pkeys)` in gtg.py (fedsv imports it). CNN path unchanged → **bit-identical** (golden gtg `[.00737,-.00134,.00435,-.00234]` / fedsv `[.00521,-.00260,.01042,-.00130]`, regression-verified before/after). LLM path: `loss_fn` over within-subset LoRA-param reconstruction (`_llm_subset_params`; faithful GTG/FedSV renorm `n_c/Σ_{c∈S}n_c`, **not** the oracle round-weight), under `@no_grad`. `*_from_logs` gain `loss_fn=None,pkeys=None` (appended → CNN callers untouched).
- **Ripple LLM** (`ripple_llm.py`, new): runs its OWN FedAvg trajectory (no shared logs — task-driven AUROC+runtime only). **Drop term** = pure functorch grad + manual SGD (`w-=lr·g_batch`, no model mutation / no optimizer / no `.backward()` = estimator-style; removes functorch×autograd mixing risk). **Ripple term** = client local-Hessian top-k eigsh (HVP over eager attention, fixed v0 + ArpackNoConvergence fallback) → progressive Q (`_orthoproj`) → Eq 16-19 chain (CNN math re-stated to avoid touching the Phase-0-verified ripple.py; pure `_flat`/`_orthoproj` imported). free-rider → φ=0 (zero delta). Ripple is good→high → negated in the compare.
- **Experiments**: `phase1_baseline_compare.py` (Flirds/GTG/FedSV/(b)oracle on one LLM trajectory + Ripple → Spearman-vs-oracle + detection AUROC + per-method runtime; `RIPPLE` env toggle), `phase1_baseline_smoke.py` (CNN bit-identical regression + LLM plumbing smoke).

## Results

### SV-baseline comparison — 1B, N=5, 3 seeds, val=100/R=10, lr1e-3
| method | Spearman vs (b)oracle | AUROC noisy | AUROC free-rider | runtime |
|---|---|---|---|---|
| **Flirds (1st+2nd)** | **+1.000** | 0.75 | 1.0 | **~107s** |
| GTG | +1.000 | 0.75 | 1.0 | ~537s |
| FedSV | +1.000 | 0.75 | 1.0 | ~532s |
| Ripple | (own traj) | 0.50±0.20 | 1.0 | ~4515s |
| (b)oracle (exact 2⁵) | — | 0.75 | 1.0 | ~531s |

- **Flirds dominates the speed–accuracy frontier**: reproduces the exact (b)-oracle ranking (Spearman +1) AND nearly its φ values, at **~5× lower runtime** than GTG/FedSV (1 HVP/round vs 2⁵ coalition sweep).
- GTG/FedSV: identical detection, ~5× slower. **free-rider φ: Flirds/oracle = exactly 0; GTG/FedSV = within-subset-renorm dilution (≠0)** — a Flirds differentiator (cleaner free-rider handling, since a zero-delta client still dilutes others' weights under within-subset renorm).
- **Ripple = slowest (~42×) and weakest**: noisy AUROC 0.50±0.20 (chance, high-variance across seeds 0.25/0.50/0.75). Its eigsh local-Hessian sketches make scaling up prohibitive → dominated by Flirds.

### #7 first clean run — 1B, N=5
- **Both lr (1e-3, 3e-3): Flirds selection WORKS** — `flirds_topk val_loss ≤ random_k` AND `ROUGE-L ≥ random_k`. `flirds_keep` = clean clients [2,3,4] every lr/seed (**drops noisy-medical + free-rider-legal exactly**). lr1e-3: flirds_topk 2.4129/0.1489 vs random 2.4168/0.1474. lr3e-3: 2.4070/0.1518 vs 2.4084/0.1487. Signal small but consistent + beats random. lr3e-3 3-seed done; lr1e-3 s2 still running.

## Debugging journey (lessons — first real Ripple runs)
1. **OOM**: Ripple drop's manual functorch grad over eager attention retains all activations = O(batch·seq²); batch=16/seq768 → 177GB OOM. **Fix: Ripple local batch=4/train_maxlen=512/hess_maxlen=256** (SFTTrainer FL loop was fine — gradient checkpointing). Ripple's own trajectory → small-batch plain-SGD is valid.
2. **7-hour run = op-count, not memory**: val=500 (50 chunks) × 2⁵ coalition × R=20 for oracle/gtg/fedsv + thousands of ripple HVPs. **Fix: small config** (train200/val20-per-domain→100tot/R10; ripple rip_rounds4/steps4/k3/hess_bs2) → ~15min shared-logs, ~54–91min ripple.
3. **94–122GB GPU ≠ a ripple leak**: it is the **normal Flirds estimator HVP footprint** (~86GB @ val_chunk10/seq384; works fine at #7). The memory watchdog threshold (70GB) was below even one Flirds HVP → mis-fired and I over-reacted. **Lesson: nvidia-smi memory.used includes reusable PyTorch cache → an OOM-danger threshold must be ~160GB.**

## Status after session
- **Phase 1 essentially complete** (#7 lr1e-3 s2 finishing → lr1e-3 3-seed).
- **Phase 2 task 1 (SV baselines port) DONE** — GTG/FedSV/Ripple at LLM verified; ComFedSV deferred (cross-device, Phase 2 task 7).
- **Committed** d5e06d2 (author Yonghee; push pending — Claude can't push). /code-review (high, 3 finders): only actioned fix = eigsh fallback shape hardening; deferred (documented) = share Eq16-19 chain + local-Hessian HVP between ripple.py(CNN)/ripple_llm.py, cache round-start val grad in the drop loop.
- **Phase 2 remaining** (next session): detection baselines (FLDetector + STD-DAGMM), Data Banzhaf, ShapleyFL, loss-heuristic, (a)-retrain LLM expansion (N=10@1B / N=5@3B), cross-device N=100 + ComFedSV, **3B/7B scale-up**. Then Phase 3 (144-run matrix).
