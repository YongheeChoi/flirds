---
type: conversation
date: 2026-06-08
topic: flirds
participants: [Yonghee, Claude]
tags: [phase-2, task-7e, detection-baselines, std-dagmm, fltrust, fldetector, backdoor, poisoning]
---

# Phase 2 task 7e — detection-baseline suite, steps 1–3 (STD-DAGMM, FLTrust, poisoning+FLDetector)

Session goal (Yonghee): implement the **threat-matched detection-baseline suite** (§3.9
redesign) in the locked order **STD-DAGMM → FLTrust → poisoning-corruptor+FLDetector →
FedDQC → matrix**. Design pre-locked (no re-discussion). This session covered steps 1–3.

## Step 1 — STD-DAGMM (`baselines/std_dagmm.py`) — DONE

Reference-build from Lin et al. 2019 ([[sources/free-riders-fl-std-dagmm]]). MODEL-FREE
like FLDetector (consumes only update vectors) but trains its OWN small AE+GMM ("model-
free" = free of the FL model, not of a learned anomaly model). Locked design realized:
- **z = [AE latent, rel-euclidean, cosine, std]** → estimation net → GMM energy (high =
  anomalous, matches `eval.metrics.detection_auroc`).
- ① **per-(client,round) pooling** (Σ_r|cohort_r| samples; client score = mean energy);
  ② **signed feature-hashing** 5.6M→256 (std on the FULL vector, reduction-independent);
  ③ random@benign-std + zero free-rider.

Claude's non-specified defaults (flagged for review, Yonghee did not veto): per-dim
standardize of projected features + std; unseen client → min score (a free-rider must
participate to be one); n_gmm=2 (small-N safe), latent=4, epochs=200; global-RNG
snapshot/restore around the seeded AE training (no downstream-method contamination).

`fl/llm_server.py`: threaded `free_rider_scale` (default 1e-3 = backward-compatible) so
the random free-rider can be tuned to the measured benign update std (Lin's evasion
setting; `uniform(-s,s)` has std s/√3 → scale = benign_std·√3).

**Validation**: synthetic smoke (`phase2_std_dagmm_synthetic_smoke.py`) — structured
benign + zero/random@benign-std free-riders → both free-riders top-2, **AUROC = 1.000**
(zero caught by std, std-matched random caught by recon/cosine = the Lin headline). Real
1B N=100 (`phase2_crossdevice_stddagmm_smoke.py`, R=30, K=10) — benign update std =
5.8e-7, **free-rider AUROC = 0.628** (15.5s, CPU). **Honest finding**: on the pure-evasion
case (random@benign-std, std neutralized) the model-free detector is WEAK at LoRA-FL scale
— real benign LoRA updates lack the clean structure the AE needs to separate them from a
std-matched random direction. The synthetic AUROC=1 proves the logic; the real 0.63 is
the LLM-scale difficulty (1-seed, untuned — headline = the matrix).

## Step 2 — FLTrust (`baselines/fltrust.py`) — DONE

Key realization Claude surfaced: with **g0 = −∇_val(w_r)** (one server gradient step;
cosine is scale-free so step size only sets direction), FLTrust's trust signal
`cos(Δw_i, −∇_val)` is the **NORMALIZED Flirds first-order term** — so "cosine ≈
Flirds-1st" (§3.9) is *exact*, which is why the plan files FLTrust as the AUXILIARY
free-rider/poisoning baseline. NOT model-free (needs the val gradient → loss_fn/pkeys; the
estimator's `_chunked` reused for the eager micro-batched grad).

**Design decision (flagged, the key one)**: the DETECTOR returns the SIGNED cosine
(`score_i = mean_r cos(Δw_i, ∇_val)`, corrupt high), NOT FLTrust's ReLU(cosine). ReLU(cos)
clips benign (<0) AND free-rider (~0) both to 0, erasing the sign that separates them →
the signed cosine is required for the free-rider regime; ReLU + magnitude-norm are
FLTrust's robust-AGGREGATION gates (cosine is already scale-free → neither changes the
ranking). Orientation falls out: benign descends val → cos<0 (low); poison ascends → >0
(high); free-rider orthogonal → ~0 (above benign) — directly matches detection_auroc.

**Validation**: unit smoke (analytic quadratic loss) — benign −1, free-rider zero +0 /
random −0.05, poison +1; **AUROC free-rider & poison = 1.000**, order poison>free>benign.
Real 1B N=100 — **free-rider AUROC = 1.000** (the 5 free-riders are the top-5 suspicious),
Flirds-1st AUROC = 1.000, Spearman(FLTrust, Flirds-1st) = +0.601 (both detect perfectly;
the loose Spearman = normalization reorders the benign tail). **Insight**: gradient-using
detectors (FLTrust ≈ Flirds-1st) ACE free-rider detection (AUROC 1.0) where the model-free
STD-DAGMM struggles (0.63) → the val gradient is the decisive signal; FLTrust is
auxiliary (Flirds subsumes it via the 1st-order term), STD-DAGMM is the independent (weak)
free-rider baseline. non-IID erosion did not appear at α=0.5 → the α-sweep (step 5) probes
the boundary.

## Step 3 — poisoning corruptor + FLDetector (code) — DONE (machinery); backdoor install = open

- **3a backdoor corruptor** (`data/corruptors.py` `backdoor()`; Xu 2023 instruction-trigger
  → target, `poison_frac` = the clean-preservation knob, default 0.5) + `backdoor=` /
  `backdoor_kwargs=` injection in `data.llm.build`/`build_crossdevice`. Unit-verified.
- **3b Bagdasaryan plain-scaled** (`fl/llm_server.py` `scaled_attackers`/`attack_scale`):
  the attacker trains on backdoor data normally, then its delta is ×γ (model-replacement;
  γ≈K = full replacement). Backward-compatible.
- **3c FLDetector poisoning repoint + cross-device adaptation** (`baselines/fldetector.py`):
  FLDetector's matched threat is the CRAFTED/SCALED update (not the honest noisy client),
  so it runs on the poisoning trajectory unchanged for cross-silo. **Yonghee chose the
  cross-device adaptation: per-client GAP-integrated HVP (option a)** — predict
  g_i^t ≈ g_i^{t'} + H^t·(w^t−w^{t'}) from the client's last participation t' (not r−1),
  one cached HVP per distinct gap. **Reduces BIT-IDENTICALLY to the per-round prediction
  under full participation** (verified: CNN synthetic score unchanged, guard green) and
  flags a scaled attacker under partial participation (cross-device synthetic AUROC=1.0).
- **3d ASR helper** (`eval/generate.py` `backdoor_asr`): triggered-prompt generation →
  target-marker fraction.

**Detection findings** (real-1B N=5, scaled backdoor attacker): FLDetector **AUROC=1.0**
(magnitude-consistency = its matched threat), FLTrust **AUROC=1.0** (scaled attacker least
val-aligned). **Flirds-1st verdict FLIPS with poison_frac** — poison_frac=0.5 (clean-helpful
mixed direction): attacker most-negative = looks BEST → evades (AUROC 0); poison_frac=1.0
(pure backdoor, clean-hurting): attacker flagged (AUROC 1); γ scaling AMPLIFIES whichever
verdict (γ=50 → Flirds-1st ±200, dominated by scale). This is the clean-preservation
boundary — **reported, NOT pre-positioned** (the matrix confirms across seeds/configs).

**Open research sub-task (Yonghee to drive): the backdoor does not install via greedy
generation (ASR=0) at any config tried** — light training (lr1e-3, γ5) → too weak to win
decoding (generation = base medical answer unchanged); heavy (lr5e-3, γ50, 20 steps) →
model destabilizes (empty generation). Installing a measurable, clean-preserving backdoor
at LoRA-FL scale is a NARROW tuning window (a code-validated machinery, not a bug). Unexplored
levers: unscaled Xu + many rounds (accumulate without destabilizing), moderate γ + many
rounds, simpler/repeated trigger, soft-ASR (target probability vs greedy exact-match).

## Yonghee's decisions this session
- Proceed STD-DAGMM → FLTrust → step-3-code at each checkpoint (picked "proceed" thrice).
- FLDetector cross-device adaptation = **gap-integrated HVP, one per gap (option a)**.
- After step 3: **record (KARIS) + commit steps 1–3**; backdoor-install research + steps 4
  (FedDQC) / 5 (matrix) deferred to the next session.

## Open items / next session
- **Backdoor instantiation research** (the crux of the poisoning matrix): how to install a
  measurable clean-preserving backdoor (training strength / γ / poison_frac / target /
  soft-ASR) — Yonghee's design input.
- **Step 4 FedDQC** (data-quality / answer_swap → IRA per-sample → client-level) — last detector.
- **Step 5 matrix** (3 threats × 2 regimes × {matched detector + Flirds + valuation} × α-sweep).
- **wiki source ingest of the backdoor papers is BLOCKED**: 2305.14710 (Xu) and 1807.00459
  (Bagdasaryan) are NOT in `raw/papers/flirds/` — Yonghee to drop the PDFs or authorize web
  ingestion.
