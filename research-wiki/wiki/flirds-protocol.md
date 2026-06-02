---
type: protocol
title: "Flirds — Implementation & Reporting Protocol"
created: 2026-05-27
updated: 2026-05-27
tags: [flirds, protocol, reproducibility, implementation]
---

# Flirds — Implementation & Reporting Protocol

The lockable specification every reported number in the Flirds paper must follow. Established 2026-05-27 (Section 3 lock); see [[flirds#Locked design decisions]] for the parent decision context.

> **Scope rule.** Any deviation from this protocol on any reported number must be (i) explicit in the experiment's config file *and* (ii) declared in the paper text near that number ("we report $X$ under deviation $Y$ for reason $Z$"). Silent deviations are not allowed.

## 1. Numerical precision

| Phase | Precision | Rationale |
|---|---|---|
| **Client local LoRA training** | bf16 mixed-precision (fp32 master weights, bf16 forward/backward) | Standard 2024–2026 instruction-tuning practice (LESS, Grosse 2023, MATES, FedDQC default). 1B/3B/7B all feasible on B200 with bf16. |
| **Server-side validation forward pass** | **fp32 enforced** | bf16 evaluation flagged as suspect in pilot (ρ=1.0 cancellation result possibly bf16 artefact). fp32 eval is the standard fix; cost overhead minimal because eval batch is small. |
| **Flirds inner product** ($-\nabla\ell \cdot \Delta w_k$, HVP, $\Delta w_k^\top u$) | **fp32 enforced** | The estimator's correctness depends on these dot products. bf16 here introduces representation precision artefacts that masquerade as zero-variance ties (last-layer cancellation, etc.). |
| **(a) Retrain oracle SV** — LoRA training | bf16 (matches deployment) | Oracle target is the actually-deployed-precision retrained model. |
| **(a) Retrain oracle SV** — utility evaluation | fp32 | Same rule as Flirds validation. |
| **(b) IRDS-定 in-run oracle** — subset utility computation $\ell(w^r + \sum_{k\in S}p_k\Delta w_k, z^{val})$ | **fp32 enforced** | The forward-pass aggregation must be in fp32 to avoid bf16 representation precision dominating the marginal-contribution sign. |
| **MC sampling (cross-device (b))** | fp32 | Same as exact (b). |

**Known LR × precision interactions** (logged for the paper appendix):
- bf16 + high lr ($\ge$ 1e-4 at 7B) → reported divergence sign-dominance pattern (pilot E=10 U-shape may be artefact).
- fp32 + low lr ($\le$ 1e-5) → numerical noise floor dominates 2nd-order term; report empirical sanity floor.

## 2. Seeds and reporting

- **Minimum 3 seeds** per (model, $\alpha$, $E$, baseline, ...) cell. Single-seed numbers MUST NOT appear in any table, figure, or text claim.
- **Reporting format**: $\text{mean} \pm \text{std}$ on tables; band plots on figures.
- **Seed schedule**: `seed_list = [42, 123, 2024]` as default; deviations logged.
- **No seed cherry-picking**: every reported claim must hold for the median seed, not just the best. If only a subset of seeds support a claim, report explicitly.

## 3. Statistical reporting

### 3.1 Ties

For rank-correlation metrics (Spearman ρ, Kendall τ): **scipy `rankdata(..., method='average')`** tied-rank convention. Document the convention in the metric's table caption.

### 3.2 Monte Carlo variance

For any MC estimator (cross-device (b) oracle, Banzhaf MSR, Ripple sampling, ...):
- Report **estimator variance** along with the point estimate.
- Pre-declare sample count $M$; justify by showing variance scales below a pre-set threshold ($\le 5\%$ of estimator magnitude at the chosen $M$).
- Default cross-device (b) MC sample = 5000–10000 (following FedSV practice). Larger $M$ if variance threshold fails.

### 3.3 Confidence intervals

Every reported headline metric (Spearman ρ, Kendall τ, AUROC, top-K hit rate, downstream task accuracy):
- **95% bootstrap confidence interval** (B=1000 bootstrap resamples over the seed × validation split product).
- Reported alongside point estimate: e.g., `ρ = 0.94 [0.91, 0.96]`.
- On figures, render as error bars or shaded bands.
- Claim of "method A > method B" requires CI bands to not substantially overlap.

## 4. Oracle implementation

### 4.1 (a) Exact retrain SV — scope

| Scale | N | Subset count | Compute (B200×4) | Status |
|---|---|---|---|---|
| 1B | 5 | 32 | ~15 min | ✅ |
| 1B | 10 | 1024 | ~3.5 days | ✅ |
| 3B | 5 | 32 | ~45 min | ✅ |
| 3B | 10 | 1024 | ~10 days | ❌ (compute) |
| 7B | any | — | infeasible | ❌ (compute) |

Justification for (a) skip at 7B: matches centralized-LLM convention (LESS, Grosse 2023, DataInf all skip exact retrain at ≥ 7B). Paper text: *"Exact retrain SV at 7B is computationally infeasible under our budget; the (b) IRDS-定 oracle, which is computationally tractable at all scales, serves as the primary correctness check; (a) at 1B/3B serves as a different-utility sanity figure."*

### 4.2 (b) IRDS-定 in-run SV — scope

| Scale | Setting | Method | Compute (B200×4, R=50, 3 seed) |
|---|---|---|---|
| 1B | cross-silo N=10 | **exact enumeration** (1024 subset) | ~3.5h × 3 seed ≈ 10h |
| 1B | cross-device N=100 | **MC** ($M$=5000–10000) | scaled by $M$ |
| 3B | cross-silo N=10 | exact enumeration | ~7h × 3 seed ≈ 21h |
| 3B | cross-device N=100 | MC | scaled by $M$ |
| 7B | cross-silo N=10 | exact enumeration | ~28h × 3 seed ≈ 3.5 days |
| 7B | cross-device N=100 | MC | scaled by $M$ |

### 4.3 Code path separation

(a) and (b) MUST be implemented in **separate modules** with separate test suites. Sharing utility-computation code between them is forbidden — the two utilities are *different functions* ($U_{(a)}(S) = \text{FL-trained-on-}S$ vs $U_{(b)}(S) = \sum_r \ell(w^r + \sum_{k\in S}p_k\Delta w_k, z^{val}) - \ell(w^r, z^{val})$) and conflating them is exactly the conv3 §2 error.

## 5. Sanity gates (automated)

The following MUST pass on every clean re-run. Failed sanity gates block any downstream claim from being reported.

| Gate | Check | Action on failure |
|---|---|---|
| **E=1 drift residual** | At $E=1$, the residual term in Proposition 1 must be ≈ 0 (within floating-point noise). | Halt; investigate implementation bug in Δw_k accumulation or HVP. |
| **N=2 singleton** | For $N=2$, the singleton subset SV must match the client's $\phi_k$. | Halt; investigate ground-truth or estimator bug. |
| **MC convergence** | Cross-device (b) MC variance must drop below threshold (§3.2) at pre-declared $M$. | Increase $M$ or report wider CI; do not silently shorten. |
| **Reproducibility** | Two runs with identical config + seed must produce bitwise-identical results at fp32 / numerically-close at bf16 (within 1e-4). | Halt; investigate non-determinism. |

## 6. Run logging

Every reported number traces back to a logged run via:

- **Run config**: full YAML / JSON with all hyperparameters (model, lr, $E$, $\alpha$, $N$, seed, batch size, LoRA rank, validation set hash, ...).
- **Environment hash**: `pip freeze` + CUDA version + GPU model.
- **Git SHA**: code commit at run time.
- **Per-round per-client $\phi_k^{(r)}$ archive**: every round's per-client raw value saved (compressed) for post-hoc analysis without re-running.
- **W&B / mlflow project**: `flirds-2026` (single project; per-run name carries scale + setup tags).

Reported number → must be linkable to a specific (config, env, git SHA, run ID). PR review on paper claims = "for claim $X$, show me the run ID."

## 7. Federated aggregation

- **Default**: vanilla FedAvg.
- **Ablation row**: FedProx (single ablation, isolated in §6 of paper).
- No FedYOGI / FedAdam / FedAvgM unless they become a reviewer ask. Cost minimization.

## 8. Validation set

- **Default**: server-side held-out, uniform domain coverage (locked decision).
- **Per-experiment justification**: where validation choice diverges from default (e.g., domain-attribution benchmark uses per-domain validation), the choice is justified in that experiment's paragraph + config.
- **Size**: 1024 examples per evaluation by default; smaller if compute-bound (down to 256), declared per number.

## 9. Cross-silo vs cross-device boundaries

| Setting | $N$ | Participation per round | Oracle (b) | When |
|---|---|---|---|---|
| **Cross-silo** | 10 | 100% (all clients per round) | exact enumeration (1024 subset) | primary experiments |
| **Cross-device** | 100 | $K$=10 per round (10% sample rate) | MC ($M$=5000–10000) | ComFedSV baseline + scale/participation ablation |

Cross-device is the *only* setting where ComFedSV is a valid baseline (Everyone-Being-Heard assumption).

## 10. Phase 0 — code-unavailable baseline reproduction

Phase 0 is part of this protocol. **No LLM-phase number is reported until Phase 0 reproductions pass within ±5%.** Reproductions, in CNN + MNIST/CIFAR-10 setup:

| Target | Original setup | Headline metric to reproduce | Cost (B200×1) |
|---|---|---|---|
| [[sources/ripple-shapley\|Ripple Shapley]] | CNN + MNIST/CIFAR-10, N=10 | Spearman ρ, 62× speedup vs GTG | 1–2 days |
| [[sources/gtg-shapley\|GTG-Shapley]] | CNN + MNIST/CIFAR-10, N=10 | Spearman ρ vs uniform MC | 1 day |
| [[sources/principled-federated-data-valuation\|FedSV]] | CNN + MNIST/CIFAR-10 (IID + non-IID), N=10 | noisy-label / backdoor detection rate | 1 day |
| [[sources/comfedsv\|ComFedSV]] | Synthetic + MNIST / FMNIST / CIFAR-10, N=100 (10 noisy) | Spearman vs ground-truth, Jaccard on noisy detection | 2–3 days |

Phase 0 output: validated baseline implementations + sample-level → client-level aggregation function (for Ripple LLM transfer).

## 11. Implementation skeleton (high-level)

```
flirds/
├── core/
│   ├── flirds_estimator.py     # 1st + 2nd Taylor closed-form
│   ├── hvp.py                  # HVP via LoRA params
│   ├── delta_w_logger.py       # per-round Δw_k archiving
│   └── phi_k_aggregator.py     # round-summing + reporting
├── oracles/
│   ├── retrain_sv/             # (a) exact retrain — fully separate code path
│   └── in_run_sv/              # (b) IRDS-定 — separate code path
├── baselines/
│   ├── gtg_shapley/            # reproduced Phase 0
│   ├── fedsv/                  # reproduced Phase 0
│   ├── comfedsv/               # reproduced Phase 0
│   ├── ripple_shapley/         # reproduced Phase 0
│   ├── data_banzhaf_fl/        # semivalue lib adapt
│   ├── shapleyfl/              # vendored from ZJU-DIVER repo
│   ├── vanilla_fedavg/         # full FedAvg upper-bound
│   ├── random_selection/       # top-K random
│   ├── loss_heuristic/         # floor
│   └── flirds_1st_only/        # self-ablation
├── detection/                  # FLDetector / FoolsGold / FLTrust / STD-DAGMM
├── data/
│   ├── feddqc_bench/           # 5-domain instruction-tuning split
│   └── super_natural_instructions/  # cross-device scale
├── training/
│   ├── fedavg.py
│   └── fedprox.py              # ablation
├── eval/
│   ├── client_selection.py     # downstream metric 1
│   ├── task_acc.py             # downstream metric 2
│   └── noisy_freerider_auroc.py  # downstream metric 3
└── protocol/
    ├── precision_guard.py      # fp32-eval enforcement
    ├── sanity_gates.py         # E=1, N=2 checks
    ├── ci_bootstrap.py         # 95% bootstrap
    └── run_logger.py           # config + env + git SHA + W&B
```

## 12. Pre-paper publication checklist

Before any number from this codebase enters the paper:
- [ ] Protocol §1–9 compliance verified per number via run logger
- [ ] Phase 0 reproductions pass within ±5%
- [ ] All sanity gates green on the production run
- [ ] CI bands rendered on every figure
- [ ] $\ge 3$ seeds confirmed
- [ ] (a) and (b) oracle results separately reported (never silently averaged)
- [ ] Code path separation between (a) and (b) verified
- [ ] All experiment configs / env hashes / git SHAs archived
