---
type: conversation
date: 2026-06-03
topic: flirds
participants: [Yonghee, Claude]
tags: [flirds, implementation, phase-0.5, estimator, in-run-oracle, ripple, 2nd-order, ggn]
---

# 2026-06-03 — Phase 0.5: Flirds estimator + dual in-run oracle (CNN)

Continuation of Phase 0.5. Yonghee drove the design forks; Claude built + validated.
Work on `main` (feature/flirds-phase-0 merged + deleted).

## Decisions made (Yonghee)

- **Ripple → faithful rebuild** (not the Phase-0 simplification). Match Algorithm 1 /
  Eq 5-19: ① local-trajectory IRDS drop term, ② per-client LOCAL Hessian sketch (not a
  single global val-Hessian), ③ progressive subspace + m-dim Jacobian chain. Verify, then move on.
- **Curvature = true Hessian only** (drop GGN): IRDS itself uses ∇²ℓ, not GGN/Fisher — match it.
- **2nd-order's small role in IRDS is a *centralized* artifact** — IRDS is per-SGD-step (tiny η,
  1st-order O(η²) already accurate) so 2nd adds little. FL is per-round multi-step (larger ΔW →
  1st error O(|ΔW|²) grows) → the 2nd-order is non-trivial there. **FL per-round is Flirds's distinct
  regime; the real test is FL-scale/LLM, not toy CNN.**
- Close Phase 0.5 thoroughly (every gate) before Phase 1.

## Built

- **Ripple faithful rewrite** (`baselines/ripple.py`): local-traj drop (matches `local_train` SGD),
  per-client local Hessian top-k → progressive Q, m-dim chain. **Found + fixed 2 latent Phase-0 bugs**
  (masked because the easy label-shuffle verify is drop-dominated): the ripple term had (a) opposite
  sign vs the drop term and (b) missing α_k weighting. Decomposition now confirms ripple reinforces
  drop (clean +0.08 / noisy −0.07); noisy AUROC 1.0. Param-only (`named_parameters`) → BatchNorm-safe.
- **(b) in-run SV oracle** (`oracle/in_run_sv.py`): U_(b)(S)=Σ_r[ℓ(w^r+Σ_{k∈S}p_k Δw_k)−ℓ(w^r)],
  exact 2^N enumeration, fixed-weight p_k. Separate module from (a) `exact_sv` (protocol 4.3).
- **Flirds estimator** (`core/flirds_estimator.py`): φ_k=Σ_r p_k[⟨g^r,Δw_k⟩+½⟨Δw_k,u^r⟩],
  u^r=H^r ΔW^r (1 HVP/round + N dots). Hessian-only.
- Validation scripts: `phase05_{flirds_oracle, sanity, taylor_check, regime_sweep, dual_oracle}.py`.

## Key findings

- **HVP correct**: jvp (forward-over-reverse) vs double-backward agree to 9.8e-6.
- **Buffer/BN concern is CNN-BatchNorm-only**; LLM (LayerNorm + LoRA `requires_grad`) doesn't hit it.
  The "trainable-params-only" discipline is universal (protocol §1). LeNet5 has no buffers.
- **2nd-order regime characterization** (curvature ratio c = ½⟨ΔW,HΔW⟩/|⟨g,ΔW⟩|): 2nd-order helps
  the magnitude fit (relL2) where the loss is curved and the per-round update is within the Taylor
  radius (CIFAR); marginal on near-flat MNIST; overshoots beyond the radius (large multi-step updates).
  Mirrors IRDS Appx E.2.2 ("2nd-order does not notably improve accuracy") — a *centralized* artifact.
- **GGN tested + rejected**: Gauss-Newton/Fisher (PSD) curvature was generally *worse* than the true
  (indefinite) Hessian (relL2 worse in 8/9 configs). Not the fix → Hessian-only.
- **IRDS 2nd-order's real role = redundancy/interaction** (down-weights near-duplicate points), not
  accuracy — relevant for FL client overlap, noted but not pursued now.
- **Reproducibility needs `cudnn.deterministic`** — otherwise conv nondeterminism drifts the FL
  trajectory ~4e-2 over 3 rounds; with it, bitwise-identical (0.0). Protocol §5 addition.

## Phase 0.5 close — all gates green

- estimator ≈ (b) oracle: Spearman 1.000 (seed0); noisy AUROC 1.0; 3-seed 0.73–0.81 (N=6 coarse).
- (a) retrain ↔ (b) in-run: noisy AUROC 1.0; fine-rank ρ≈0.66 (different utilities — expected).
- (b) Shapley efficiency |diff|=0; symmetry |diff|=0.
- E=1 residual at fp32 noise floor (~3e-3); N=2 relL2 3e-3; reproducibility bitwise-0 (cudnn-det).

## Next

Phase 1: LLM port (OpenFedLLM + LoRA) — the FL per-round regime where the 2nd-order is non-trivial.
Deferred: ripple `(rounds,n,P)` streaming + eigsh fallback → LLM scale; ripple-term task-driven
verification → Phase 3 (backdoor / temporal poisoning).
