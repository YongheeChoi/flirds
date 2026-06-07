---
type: conversation
date: 2026-06-07
topic: flirds
participants: [Yonghee, Claude]
tags: [phase2, fldetector, detection, cross-silo, l-bfgs, baseline, from-logs]
---

# Phase 2 task 5 — FLDetector (cross-silo) implementation

## Yonghee's ask

Continue Phase 2. Implement **task 5 = FLDetector (Zhang et al., KDD'22)** as the
cross-silo noisy/poison detection comparator, per the regime split locked 2026-06-07
(FLDetector → cross-silo; STD-DAGMM → cross-device/task 7, degenerate at N=5).

Spec given:
- Server-side, from-logs closed-form. Per round predict each client's update from
  history via Cauchy MVT + an L-BFGS Hessian Ĥᵗ (estimated from the global trajectory
  wᵗ−wᵗ⁻¹ and a window of global-update differences):
  ĝᵢᵗ = gᵢᵗ⁻¹ + Ĥᵗ(wᵗ−wᵗ⁻¹).
- suspicious score = ‖ĝᵢᵗ − gᵢᵗ‖ ℓ1-normalized across clients → mean over the past N rounds.
- Our logs suffice: wᵗ = w_r, gᵢᵗ = client delta Δw_k (logs' `dm`).
- New `baselines/fldetector.py`, reference-guided self-build (same pattern as the others).
- Use the **continuous score for AUROC only**; skip the paper's Gap-statistic + 2-means
  clustering (weak at N=5).
- FLDetector is a **detection score, not a valuation φ** → it joins the AUROC table only,
  not the Spearman/value table.
- Verify like the rest of the session: synthetic unit check + LLM from-logs run +
  add to `phase1_baseline_compare.py` (9 → 10 methods). RIPPLE=0 in the compare
  (Ripple eigsh flaky).
- Source: `raw/papers/flirds/2207.09209v4.pdf`, [[sources/fldetector]].

## What was built

`flirds/baselines/fldetector.py` — model-free server-side detector. Two functions:

1. `_lbfgs_hvp(S, Y, v)` — Byrd-Nocedal **compact-form** L-BFGS Hessian-vector product
   B v ≈ H v (mirrors the official FLDetector `lbfgs`):
   σ = (y_last·s_last)/(s_last·s_last);
   B = σI − [σS, Y] M⁻¹ [σS, Y]ᵀ, M = [[σ SᵀS, L],[Lᵀ, −D]],
   L = strictly-lower(SᵀY), D = diag(SᵀY). The 2k×2k solve is done in float64
   (k ≤ window is tiny). **Hand-verified**: with 1 secant pair the result satisfies the
   secant equation B·s = y exactly (algebra checks out).

2. `fldetector_from_logs(logs, n_clients, window=10, device="cpu")` → score[n]
   (higher = more suspicious). **Takes NO model / loss_fn / test_loader** — the key
   structural difference from GTG/FedSV/ShapleyFL. It needs only the logged update
   vectors, so the signature is minimal.

### Key mapping (our FedAvg ↔ the paper)
The crucial subtlety, documented in the module: FedAvg folds the server step into the
local delta (wᵗ = wᵗ⁻¹ + Σ_k p_k Δw_k), so:
- gᵢᵗ = the **raw** client delta `dm[c][0]` (the received update, n_c-unweighted);
- wᵗ − wᵗ⁻¹ = the **n_c-WEIGHTED aggregate** of the round's deltas (= the true model
  difference). The "global gradient" sequence whose differences form the L-BFGS Y_k is
  that same aggregate → S_k = aᵗ⁻¹ (model diffs), Y_k = aᵗ − aᵗ⁻¹ (update diffs);
- w_r (the logged global state) is **unused** — the trajectory is reconstructed from deltas.

### Adaptations from the paper (offline, short R)
- Skip the Gap-statistic + 2-means clustering; keep only the continuous suspicious score.
- The paper gates scoring on a **full window** (iteration > N); our R can be < N=10, so we
  score as soon as ≥1 secant pair exists (from the 3rd round) and average over the
  available ≤ N rounds. `window` (paper N=10) bounds both the L-BFGS pair count and the
  score average.

### Free-rider behaviour (a teaching point recorded)
Unlike the valuation methods, a zero-update free-rider does **not** get score 0. Its
prediction is ĝ = 0 + Ĥv = Ĥv, so its residual = ‖Ĥv‖ = the full step-scale magnitude,
while a consistent client's residual is only the prediction *error* (≪ step) → the
free-rider gets a **high** score. That is exactly the intended detection (so there is
**no free-rider==0 gate** for FLDetector, unlike Banzhaf/oracle/Flirds/Ripple).

## Integration

- `experiments/phase1_baseline_smoke.py`:
  - new synthetic check `fldetector_synthetic_check()` (CPU, no model; runs in the
    `cnn`/`both` path alongside the Banzhaf/ShapleyFL kernel checks);
  - added FLDetector to `llm_smoke` (bumped its from-logs trajectory rounds 2→3 so ≥1
    prediction round exists); plumbing gate = finite + L1-normalized (Σ score = 1).
- `experiments/phase1_baseline_compare.py`: 9 → **10 methods**. FLDetector added to the
  AUROC/runtime tables only (excluded from Spearman, since it has no value vector). Runs
  as CPU linalg over the logged deltas (the cheapest method — its selling point is the
  server-side cost).

## Verification

- **Synthetic check** (N=5, P=8, R=12; 4 temporally-consistent clients tracking a smooth
  trend + 1 anomalous noise client): score = [0.525, 0.119, 0.119, 0.119, 0.119] →
  anomalous client is argmax, **AUROC = 1.000**. The 4 clean clients get nearly identical
  scores (0.119) — the L-BFGS predicts consistent clients well; only the noise client's
  residual inflates. Non-circular (FLDetector recomputes the aggregate + Hessian itself).
- **CNN regression bit-identical**: gtg `[0.007366…, …]` / fedsv `[0.005208…, …]` goldens
  unchanged (the smoke edits don't touch the CNN path).
- **LLM smoke (1B, N=5, free-rider=zero)**: `SV-BASELINE LLM PORT OK`. FLDetector
  Σscore = 1.0000 (norm ✓), finite ✓. At this tiny scale (R=3, 1 prediction round,
  8ex/2step) the detection signal is weak as expected: free-rider AUROC = 0.750
  (zero-update client scores high but a clean client spuriously tops it), noisy AUROC =
  0.250 (answer_swap at the noise floor). All other gates intact (free-rider φ==0 for the
  delta methods, Spearman +1.000 for GTG/FedSV/Banzhaf/ShapleyFL/estimator).

### Compare (1B N=5, R=10, lr=1e-3, RIPPLE=0, 3 seeds — GPU 0/1/2)
The 10-method compare, headline (3 seeds, all identical to ±0):

| method | noisy AUROC | free-rider AUROC | runtime (s) |
|---|---|---|---|
| Flirds / Flirds-1st / GTG / FedSV / Banzhaf / ShapleyFL / loss-heur / (b)oracle | 0.75 | 1.00 | 35–538 |
| **FLDetector** | **0.50** | **0.75** | **~24 (cheapest)** |
| Ripple (RIPPLE=0 placeholder) | — | — | — |

- **FLDetector is the cheapest method** (~24 s; model-free CPU linalg) — below even Flirds-1st
  (~35 s) and far below the coalition-sweep baselines (~520–540 s) and Flirds (~107 s).
- **FLDetector is the weakest detector at N=5 cross-silo**: noisy AUROC = 0.50 (chance),
  free-rider = 0.75, vs the valuation methods' 0.75 / 1.00. The score pattern is
  **identical across all 3 seeds**: the *clean* math client consistently tops the suspicious
  score (0.24–0.27), pushing the actual noisy (medical) and free-rider (legal) clients down.
- This is **systematic non-IID erosion**, not seed noise: 5 disjoint domains make per-client
  temporal consistency unreliable, so a benign-but-divergent client looks "inconsistent". It
  is exactly the FLDetector limitation the paper documents (IID-only Theorem 1) and the plan
  §3.9 predicted — and it supports the Flirds framing: validation-loss valuation separates
  noisy/free-rider cleanly (0.75 / 1.00) where temporal-consistency detection does not.
- All other gates held across the 3 seeds: Spearman vs (b)oracle = +1.000 for every valuation
  method; free-rider φ exactly 0 for Flirds/oracle/Banzhaf/loss-heur.

**Net**: FLDetector = the cheap server-side temporal-consistency comparator, dominated on
detection by Flirds at N=5 (its quality lives at N=10 / N=100, the headline detection scale).

## Notes / open questions
- N=5 detection AUROC is coarse (1 positive / 5) → the **headline detection table is
  N=10 (1B) / N=100 (cross-device, task 7)**.
- The non-IID erosion the FLDetector paper documents (IID-only Theorem 1) is expected to
  show here: 5 disjoint domains make per-client temporal consistency noisy, so a clean
  client can spuriously score high at small R — direct evidence for the separator's
  difficulty (supports the Flirds framing, [[threads/noise-ood-malicious-client-separation]]).
