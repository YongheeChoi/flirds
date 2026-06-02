---
type: conversation
date: 2026-06-02
topic: flirds
participants: [Yonghee, Claude]
tags: [flirds, implementation, phase-0, decisions, code-review]
---

# 2026-06-02 — Phase 0 implementation: decisions + 4 baseline self-builds

Implementation session triggered by "[[flirds-implementation-plan]] 읽고 phase 0부터 시작". Yonghee drove every open decision (explanation → decision → execution); Claude built + verified.

## Decisions made (Yonghee)

- **D1 (code reuse)** — "fork if code exists" *means* read the reference's correct logic and re-build it inside our own structure, NOT copy the repo verbatim. So all baselines are **reference-guided self-builds** on one slim FL simulator.
- **D2 (logging)** — no W&B; local file logging.
- **D3 (datasets)** — follow **LESS experimental setup** (validation/selection protocol); cross-silo 5-domain = **PubMedQA / CaseHOLD / FiQA / AQUA-RAT / Dolly** (code domain dropped, FiQA used instead → 3/5 overlap with FedDQC); cross-device = **Fed-WildChat (FedLLM-Bench, N=100) + FedHDS (NI per-task + Dolly Dirichlet)** both; FedDQC comparison via **IRA as a baseline only** (no matched-arm). LoRA rank: start 16/32 but **sweep {16,32,64,128}**.
- **D6** — per-domain few-shot exemplar validation (recommendation accepted).
- **D7** — N=100/K=10/R=200, default α=0.5, α-sweep includes α=0; late-joiner kept structurally, justified in text.
- **D8** — detection = **FLDetector + STD-DAGMM** (one noisy + one free-rider, the most-cited per-client-score baselines).
- **BASE_REPO** — LLM = OpenFedLLM; CNN = self-built slim simulator.
- **CNN full track (new)** — run the WHOLE experiment suite on CNN too (not just baseline reproduction), standard OpenDataVal + FL-valuation-canon setting, **no LoRA**. Reason: huge CNN prior-art base → direct comparison + credibility; exact-SV is cheap so estimator can be validated at larger N than LLM allows.
- **Phase restructure** — 0 (CNN baseline reproduction) → **0.5 (Flirds estimator + dual oracle on CNN first)** → 1+ (LLM via OpenFedLLM).
- **Collaboration mode** — discuss at design forks, delegate routine coding, review after. Commit only when asked.
- **GPU** — only physical GPUs 0–3 usable (enforced by `.claude/settings.json` env CUDA_VISIBLE_DEVICES=0,1,2,3).

## Built (Phase 0, committed 93bb8d0)

- conda env `flirds` (torch 2.12+cu130, B200×4).
- `fl/`: FedAvg simulator, partition (IID/Dirichlet/McMahan-shard), client/server, shared-trajectory `run_fedavg_logs`.
- `oracle/exact_sv`: exact 2^N retrain Shapley.
- baselines: **GTG** (guided trunc MC, cosine 0.99 vs exact recon-SV), **FedSV** (permutation MC, 0.998), **ComFedSV** (uniform subset + low-rank ALS completion, 0.993 / spearman 0.976 partial-vs-full GT), **Ripple** (drop = fixed-w IRDS 1st-order + ripple = Hessian eigen-sketch + progressive subspace + Jacobian chain; noisy-detection AUROC 1.0).
- Bug caught & fixed: `dirichlet_partition` appended class-ordered → prefix truncation `idx[:k]` caused label skew; now shuffles within client.

## Plan corrections found (vs original plan/protocol)

GTG/ComFedSV "have code" but in forms unusable for direct fork (GTG = cyyever multi-package framework; ComFedSV = Huawei notebook) → all self-built. Ripple "62× vs GTG" is actually 62× vs AFedSV+ / 49× vs FedSV (GTG not a Ripple baseline); Ripple has no SV ground truth (task-driven only). FedDQC LoRA = r=64/α=128 (not 16/32). FedDQC non-IID = single-domain quality-het, not domain-per-client. (All logged into [[flirds-implementation-plan]] corrections section.)

## Code review (end of session)

Independent skeptical review: **math sound** — ripple Jacobian chain numerically verified correct, comfedsv ALS/indexing correct, exact Shapley correct. Fixed now: gtg `_normalize` div-by-zero blow-up, dead `bloss`, unused `epochs` param + contradictory docstring, inaccurate server docstring. **Phase 0.5 TODO**: ripple-term verify is too weak (drop alone already separates label-shuffle → need backdoor/future-round-only poisoning to exercise ripple); GTG/ComFedSV verifies partly self-referential / row-0-dependent; ripple `(rounds,n,P)` dense array OOMs at real model scale (project onto Q on the fly); eigsh needs convergence fallback; buffers-in-params breaks for BatchNorm.
