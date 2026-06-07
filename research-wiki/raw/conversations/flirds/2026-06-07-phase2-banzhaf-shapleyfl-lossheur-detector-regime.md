---
type: conversation
date: 2026-06-07
topic: flirds
participants: [Yonghee, Claude]
tags: [phase2, baselines, data-banzhaf, shapleyfl, loss-heuristic, flirds-1st-only, detection-baselines, fldetector, std-dagmm, regime-split, llm]
---

# Phase 2 cheap valuation baselines (Banzhaf / ShapleyFL / loss-heuristic / 1st-only) + detector regime split

Continuation. Phase 1 (#7) + Phase 2 task 1 (SV-baselines GTG/FedSV/Ripple) were already done. This session did **Phase 2 tasks 2–4** (the cheap valuation baselines) and **locked the detection-baseline regime split** (task 5 prep).

## Yonghee's asks / decisions (chronological)
- I opened by re-proposing a Phase-2 priority (3B scale-up first). **Yonghee corrected me: the plan already has the task order (§2 tasks 1–9); task 1 done → just continue from task 2.** (Lesson: check the plan's task list before re-prioritizing.)
- **task 2 Data Banzhaf**: built as a self-build (see below); no fork raised.
- **task 3 ShapleyFL**: I surfaced a real fork (Shapley linearity ⇒ "exact + our utility" is degenerate-equal to the (b) oracle, so a distinct variant is needed). Yonghee **chose C (faithful to the paper's algorithm)** and asked: *"if C = the paper, why did you consider implementing it differently?"* I clarified — the math is the paper's; only the substrate is adapted (reference-guided self-build, like GTG/FedSV/Ripple; the official repo is CNN-image code). Then, after reading the paper, I **corrected my own framing**: the paper has ONE value (surrogate-FSV); its DMC difference-estimator is just the *large-N estimator* of that value → at N=5 exact is faithful, **DMC deferred to cross-device (task 7)**. Yonghee approved ("그렇게 하자").
- **task 4 loss-heuristic + Flirds-1st-only**: built (trivial).
- **Detector regime**: Yonghee recalled deciding *"cross-silo = FLDetector, cross-device = STD-DAGMM"* and asked to **verify the raw transcripts once** before locking. Full raw search (incl. conversation1–4) showed D8 only fixed the corruptor-type pairing — **the regime split was in neither raw nor distill = a NEW decision, not a recovered one.** Locked it anyway (it's the right design; see below) and distilled.
- **Chose: session wrap-up + commit.**

## What was built (Phase 2 tasks 2–4; all on `main`)
- **task 2 — Data Banzhaf** (`baselines/banzhaf.py`): reference-guided self-build, **not pyDVL/OpenDataVal** (the plan's old note). Banzhaf is a semivalue = the (b) in-run oracle's exact 2^N coalition utilities reweighted by the uniform kernel 1/2^{n-1}. Factored `in_run_sv._coalition_utilities` out of `in_run_shapley` (Shapley kernel unchanged → **bit-identical**); `banzhaf.py` reuses it. MSR estimator → only needed cross-device (task 7).
- **task 3 — ShapleyFL surrogate-FSV** (`baselines/shapleyfl.py`): faithful self-build of Sun et al. KDD'23 Def 4.1–4.3 — per-round **uniform-average submodel** utility Φ (=accuracy CNN / −val_loss LLM) + per-round **exact Shapley** (reuses `exact_sv.exact_shapley`) + **min-max norm** (Def 4.2) + **EMA** (Def 4.3, β=0.5). From-logs valuation, backend-agnostic. DMC difference-estimator (§5.2) deferred to cross-device. good→high → negated in the compare.
- **task 4 — loss-heuristic + Flirds-1st-only** (wired into `phase1_baseline_compare.py`): loss-heuristic (floor) = per-client singleton in-run utility U_(b)({k}) (reuses `in_run_utility`); Flirds-1st-only = `flirds_values(second_order=False)`.
- `phase1_baseline_compare.py` extended to **9 methods**, print refactored to method-order lists. `phase1_baseline_smoke.py` gained a Banzhaf kernel unit-check (additive-game / null-player / N=2==Shapley) + a ShapleyFL pipeline unit-check (min-max + EMA) + Banzhaf/ShapleyFL in the LLM smoke.

## Verification
- **CNN bit-identical**: gtg/fedsv golden unchanged; `in_run_shapley` refactor reproduces phase05 golden (0.7381/0.8810 + oracle φ) exactly.
- Unit checks (CPU): Banzhaf kernel ✓, ShapleyFL min-max+EMA ✓.
- ShapleyFL LLM path standalone (GPU): finite, Spearman(−SSV, oracle) = +1.000.
- **Combined 9-method 1B N=5, 3-seed (val=100/R=10, lr1e-3, RIPPLE=0)**:

| method | Spearman vs (b)oracle | AUROC noisy / FR | runtime | free-rider φ |
|---|---|---|---|---|
| Flirds (1st+2nd) | +1.000 | 0.75 / 1.0 | ~107s | 0 (exact) |
| Flirds-1st-only | +1.000 | 0.75 / 1.0 | **~35s** | 0 (exact) |
| Banzhaf | +1.000 | 0.75 / 1.0 | ~531s | 0 (exact) |
| ShapleyFL | +1.000 | 0.75 / 1.0 | ~531s | ~0 (min-max) |
| loss-heuristic | +1.000 | 0.75 / 1.0 | ~164s | 0 (exact) |
| GTG / FedSV | +1.000 | 0.75 / 1.0 | ~533–538s | +0.003 / +0.004 (renorm) |

- N=5 small-delta = near-additive ⇒ all 7 methods reproduce the oracle ranking. **Flirds dominates the frontier** (same ranking, 5–15× cheaper); free-rider φ exactly 0 for the delta-based methods (vs GTG/FedSV renorm dilution). At N=5 the AUROC is coarse (1 positive/5) — differentiation lives at larger N / harder regimes.

## Detector regime split — LOCKED 2026-06-07 (new decision)
- **FLDetector → cross-silo (N=5/10)**: from-logs closed-form (Cauchy MVT + L-BFGS prediction-residual), any-N → the noisy comparator (use the continuous score for AUROC; skip Gap+2-means clustering at small N).
- **STD-DAGMM → cross-device (N=100, with task 7)**: trains a DAGMM autoencoder+GMM on the set of client update vectors → needs N≫; at cross-silo N=5 it is degenerate (5 vectors, ~12M-dim LoRA updates; orig N=100/20-free-riders, 0.2M MLP). Free-riding is the cross-device incentive threat.
- Distilled into plan §3.9 + task 5 + threads/noise-ood §B.

## Debugging
- ShapleyFL LLM smoke hung on the tiny **Ripple eigsh** (known eigsh-convergence flakiness, not ShapleyFL) → killed (pkill self-matched the shell, exit 144, harmless) → verified ShapleyFL via a standalone check + the combined batch instead. Ripple stays RIPPLE=0 in the compare (its numbers are from the prior session).

## Status after session
- **Phase 2 tasks 2, 3, 4 DONE + verified.** Committed (push pending — Claude can't push).
- **Phase 2 remaining**: task 5 = **FLDetector (cross-silo, next)** + STD-DAGMM (cross-device, task 7) · task 6 (a)-retrain LLM · task 7 cross-device N=100 + ComFedSV + STD-DAGMM · task 8 3B/7B scale-up · task 9 corruptor extensions → Phase 3 (144-run matrix).
