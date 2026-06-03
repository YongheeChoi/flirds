---
type: project
title: "Flirds — Federated Learning + In-Run Data Shapley"
created: 2026-05-05
updated: 2026-06-03
sources: [in-run-data-shapley, principled-federated-data-valuation, comfedsv, gtg-shapley, shapleyfl, space-participant-amalgamation, ripple-shapley, game-of-gradients-sfedavg, data-banzhaf, datainf, logix, asymmetric-data-shapley, distributionally-robust-data-valuation, dice, feddqc, fldetector, fedcorr, fltrust, foolsgold, free-riders-fl-std-dagmm, less, grosse-llm-influence, mates, dsdm]
tags: [flirds, project-state, design-decisions]
---

# Flirds — project state

**Flirds = Federated Learning + In-Run Data Shapley.** A federated data-valuation method extending [[sources/in-run-data-shapley|IRDS]] (In-Run Data Shapley) to the FL setting via LoRA-based PEFT.

This page is the project's running state — locked design decisions, resolved questions, planned experiments. The **primary record** is `raw/conversations/flirds/conversation{1..4}.md` + `2026-05-27-section-23-lock.md` (chronological design conversations); this wiki page is the curated distillation. When in doubt, read the raw.

## One-line method statement

> **Flirds** computes client-level Shapley values via closed-form 1st + 2nd order Taylor expansion of the validation loss change per FL round, using only the client updates $\{\Delta w_k\}$ that the server already receives in vanilla FedAvg — adding zero communication cost.

## Core formula (round $r$)

$$\phi_k^{(r)} \;\approx\; -\nabla\ell(w^r, z^{val}) \cdot \Delta w_k \;+\; \tfrac{1}{2}\, \Delta w_k^\top H^{(val)}(w^r) \cdot \Delta W^{(r)}$$

where $\Delta W^{(r)} = \sum_j \Delta w_j$ is the sum across all participating clients. Total: $\phi_k = \sum_r \phi_k^{(r)}$.

The 2nd-order term captures client interactions. **Per-round cost: 1 HVP for $u := H^{(val)} \Delta W^{(r)}$, then $N$ dot products** $\Delta w_k^\top u$. LoRA dimension keeps it cheap.

### Phase 0.5 findings — 2nd-order term & dual oracle (2026-06-03, CNN)

Estimator (`core/flirds_estimator.py`) + (b) in-run oracle (`oracle/in_run_sv.py`) built and validated on the CNN track; all sanity gates green. See [[raw/conversations/flirds/2026-06-03-phase05-estimator]].

- **Estimator ≈ (b) in-run oracle**: the per-round closed form (1 HVP + $N$ dots) tracks the exact $2^N$ in-run Shapley (Spearman 1.0 in-regime; noisy-client AUROC 1.0). HVP verified (jvp vs double-backward, 9.8e-6). (b) Shapley **efficiency** and **symmetry** exact to 0.
- **2nd-order is regime-dependent**: helps the magnitude fit (relL2) where the val loss is curved and the per-round update is within the Taylor radius (CIFAR); marginal on near-flat MNIST; overshoots for large multi-step updates. Mirrors [[sources/in-run-data-shapley|IRDS]] Appx E.2.2 ("2nd-order does not notably improve accuracy") — **but IRDS is centralized per-SGD-step (tiny $\eta$); FL per-round multi-step is the regime where the 2nd-order is non-trivial**, so the decisive test is at FL-scale / LLM.
- **Curvature = true Hessian** (locked): matches IRDS; a Gauss-Newton/Fisher (PSD) variant was tested and was generally *worse* than the true (indefinite) Hessian.
- **(a) retrain SV vs (b) in-run SV**: agree on noisy-client detection (AUROC 1.0) but only moderately on fine ranking ($\rho\approx0.66$) — expected, different utilities (protocol 4.3 separation).
- **Reproducibility** requires `torch.backends.cudnn.deterministic=True` (else conv nondeterminism drifts the trajectory ~4e-2 / 3 rounds; with it, bitwise-identical) — **CNN (conv) track only; the LLM track is conv-free**. → protocol §5 addition (`flirds/repro.py:seed_everything`).
- **Convention for all experiments — plain SGD (momentum=0)**, matching [[sources/in-run-data-shapley|IRDS]] / Ripple Eq 1. Empirically load-bearing: under momentum=0.9 the 2nd-order term *degraded* (3-seed Spearman 0.73 < 1st-only 0.81), but with plain SGD it *helps* (0.96 > 0.92) — because the realized per-round displacement is exactly the gradient step the Taylor expands around. Momentum's velocity tail breaks that correspondence.

## Locked design decisions

### Original lock (conversation 4, 2026-05-05)

| Decision | Choice | Rationale |
|---|---|---|
| **Unit of attribution** | Client-level | Most natural in FL; conversation 3 derives this *is not* the data-point Shapley aggregated to client (drift residual quantifies the gap). |
| **Server inputs** | Only $\Delta w_k$ (LoRA params) | Privacy + communication cost; no extra statistics, no SCAFFOLD-style correction term. |
| **Approximation order** | 1st **and** 2nd Taylor, always both | Locked; not toggled per setting. |
| **Validation** | Server-side, default uniform domain coverage | Fairness + abuse prevention; per-experiment validation choices justified separately. |
| **PEFT** | LoRA | Communication, dimension, and HVP cost all small. |
| **FL setting target** | Cross-silo + cross-device | Algorithm should work in both; benchmark choice splits by setup. |
| **Privacy framing** | Requires server to see individual $\Delta w_k$ | Acknowledged: incompatible with secure aggregation. *Inherent limitation* of client-level valuation, not a flaw to fix. |
| **Communication overhead** | Zero beyond vanilla FedAvg | Strong differentiator vs. GTG-Shapley / FedSV. |
| **Drift residual** | Measured, not corrected | Reported as $E$-sensitivity study; control variates excluded. |
| **Participation normalization** | None by default | Method should work as-is; cross-device experiments designed to show ranking-within-participation-tier still recovers quality. |

### Additional locks (2026-05-18 → 2026-05-27)

| Lock | Decision | Date |
|---|---|---|
| **Q1 — Use case** | research-side in-run attribution (incentive / detection = secondary applications) | 2026-05-27 |
| **Q2 — ②③ IRDS-inherited limits** | default = IRDS-form characterization; layer-wise / phase-norm = variants in ablation | 2026-05-27 |
| **Q3 — Ripple positioning** | other design points = main claim; theoretical reduction = bonus | 2026-05-27 |
| **④ Trajectory dependence** | feature (IRDS framing) — closed by Q1 | 2026-05-27 |
| **⑤ Ground-truth definition** | dual oracle — (a) retrain exact SV + (b) IRDS-定 in-run SV, separately reported | 2026-05-27 |
| **N1 — Prop 1 handling** | original theory frame maintained as re-run target; pilot data set aside; (a) Drop = contingency if U-shape re-appears in clean re-run | 2026-05-27 |
| **N2 — Scale tier** | **1B + 3B + 7B** (13B/70B excluded — no FL data-valuation precedent + compute) | 2026-05-27 |
| **N3 — Oracle / stress** | (a) exact retrain + (b) IRDS-定 separated reporting + adversarial stress regimes + MC precision (variance / CI / ties) | 2026-05-27 |
| **N4 — Variant comparison metric** | 3 metrics: client-selection convergence + downstream task acc/F1 + noisy/free-rider AUROC | 2026-05-27 |
| **Models** | 1B = Llama-3.2-1B-Instruct; 3B = Llama-3.2-3B-Instruct; 7B = Llama-2-7B | 2026-05-27 |
| **Noise vs OOD-good (deferred 2026-05-18)** | deferred — recast as characterized limitation; non-IID α-sweep mandatory | 2026-05-18 |

## The mathematical narrative (paper-ready)

### Centralized baseline

In a single SGD step on batch $B$, **data-level Shapley summed over a client equals client-level Shapley directly** — for both 1st- and 2nd-order terms. Conversation 3, §4 proves this:

$$\sum_{z \in I_k} \phi_z^{\text{data}} = -\eta\, \nabla\ell^{val} \cdot G_k \;=\; \phi_k^{\text{client}} \quad (G_k := \sum_{z \in I_k} \nabla\ell(w_t, z))$$

So in centralized 1-step SGD the granularity choice is grouping-invariant.

### FL multi-step deviation: drift residual

Client sends $\Delta w_k = -\eta \sum_{e=0}^{E-1} \sum_{z \in B_{k,e}} \nabla\ell(w_k^{r,e}, z)$ — an $E$-step trajectory endpoint, not a single-step gradient. Taylor-expanding around $w^r$:

$$\Delta w_k \;\approx\; -\eta E \cdot \bar{G}_k(w^r) \;-\; \eta \sum_{e,z} H_z(w^r)(w_k^{r,e} - w^r)$$

The first term is the centralized-equivalent contribution; the second is **client drift residual**, of order $O(\eta E |\bar{H}| \cdot \text{local-trajectory length})$. Vanishes at $E=1$; grows with non-IID client data and large local epochs.

> **Proposition 1 (informal)**: Flirds' client-level Shapley equals centralized data-level Shapley aggregated to the client *plus* a drift residual that quantifies FL's deviation from centralized IRDS.
>
> **Proposition 2 (informal)**: The drift residual is bounded by a cubic function of $R_r$ (per-round local trajectory radius).

Empirical validation = experiment #5 (α-sweep × E-sweep matrix).

## Resolved questions (closed 2026-05-27)

Each was open in the original conversation set; all closed in the 2026-05-19 → 2026-05-27 conversation. One-line resolutions:

1. **Noise vs OOD-good distinction** → DEFERRED (2026-05-18). Recast as characterized limitation. Prior-art backing: [[threads/noise-ood-malicious-client-separation]] (no FL method separates inside a signed value; even FLDetector collapses on non-IID).
2. **Cancellation effect (②)** → default = IRDS-form characterization (per-layer decomposition reported, no estimator change to spine); layer-wise weighting = ablation variant. Solving would break the granularity-invariance lemma + Proposition 1 derivation.
3. **Magnitude vs alignment (③)** → default = IRDS-form characterization (late-joiner test reported); phase-normalization = ablation variant. Solving would break estimator–oracle consistency for the (b) IRDS-定 oracle (research-side framing makes ③ *not a bug*: a late-fit client receiving low value is an honest answer to the round-marginal question).
4. **Trajectory dependence** → feature (IRDS framing). Closed by Q1 = research-side. Single-run is sufficient; multi-run averaging only relevant if incentive use-case becomes primary (it doesn't).
5. **Ground-truth definition** → **dual oracle**:
   - **(a) Exact retrain SV** ($U(S) = $ FL training using only $S$): "data-valuation community standard." 1B (N∈{5,10}) + 3B (N=5) feasible; 7B skipped.
   - **(b) IRDS-定 in-run SV** ($U_{total}(S) = \sum_r [\ell(w^r + \sum_{k\in S}p_k \Delta w_k, z^{val}) - \ell(w^r, z^{val})]$): the *Flirds-correct* oracle; **exact enumeration at N=10 cross-silo** (1024 forward passes per round), **MC at N=100 cross-device**. The IRDS / [[concepts/proximal-bregman-response|PBRF]] framing inherited from [[sources/grosse-llm-influence|Grosse 2023]] + [[sources/mates|MATES]] is the existing-literature permission slip for "(b) is a well-defined target, not a counterfactual proxy."
6. **Benchmarks** → noisy / free-rider detection + domain attribution + client-selection convergence + cost-matched baseline tiers (see Experiment plan below).
7. **LLM choice / dataset** → **Llama-3.2-1B-Instruct + Llama-3.2-3B-Instruct + Llama-2-7B**. 1B/3B = Llama-3.2 family for cross-scale extrapolation consistency; 7B = Llama-2-7B for direct comparison with [[sources/less|LESS]] + [[sources/feddqc|FedDQC]]. Dataset = FedDQC-comparable instruction-tuning bench (medical / legal / code / math / general domains) for cross-silo; Super-NaturalInstructions for cross-device scale.

## Experiment plan — Section 3 (locked 2026-05-27)

### Phase 0 — pre-LLM (must pass before LLM experiments start)

**11. Code-unavailable baseline sanity reproduction.** Reproduce headline metrics (ρ, AUROC, runtime) of code-unavailable baselines in their *original* (CNN + MNIST/CIFAR-10) setup. Targets: [[sources/ripple-shapley|Ripple Shapley]] (AAAI'26), [[sources/gtg-shapley|GTG-Shapley]], [[sources/principled-federated-data-valuation|FedSV (Wang 2020)]], [[sources/comfedsv|ComFedSV]]. Cost: ≈ 5–7 days on B200×1. Pass criterion: each headline metric within ±5% of reported. Output: validated baseline implementations + sample-level → client-level aggregation function for Ripple (re-used for LLM transfer).

### ★★★ paper spine

1. **Clean implementation protocol document.** See [[flirds-protocol]]. fp32 evaluation enforced (training bf16); seeds ≥ 3 reported mean ± std; scipy tied-rank for ties; MC variance + 95% bootstrap CI band on all rank-correlation / AUROC numbers; (a) and (b) oracle as separate code paths; sanity gates ($E{=}1 \Rightarrow$ residual ≈ 0, $N{=}2 \Rightarrow$ singleton SV matches client value); per-run config + env hash + git SHA logged. *Must lock before any reported number.*
2. **Baseline run set (10 valuation/training + 4 detection).** Cost-matched tiers; full set at 1B / 3B / 7B.
   - **Valuation (6)**: [[sources/gtg-shapley|GTG-Shapley]], [[sources/ripple-shapley|Ripple Shapley]] (closest competitor), [[sources/principled-federated-data-valuation|FedSV (Wang 2020)]], [[sources/comfedsv|ComFedSV]] (cross-device only — partial participation required), [[sources/data-banzhaf|Data Banzhaf]] in FL (semivalue library), [[sources/shapleyfl|ShapleyFL]] (code available: `ZJU-DIVER/ShapleyFL-Robust-Federated-Learning-Based-on-Shapley-Value`).
   - **Training comparison (2)**: **Full FedAvg** (all clients, no selection — upper-bound floor); **Random-selection FedAvg** (top-K random — random baseline our valuation must beat).
   - **Self / heuristic (2)**: **Flirds-1st-only** (self-ablation isolating the 2nd-order term contribution); **loss-heuristic** (floor).
   - **Detection (separate table, 4)**: [[sources/fldetector|FLDetector]], [[sources/foolsgold|FoolsGold]], [[sources/fltrust|FLTrust]], [[sources/free-riders-fl-std-dagmm|STD-DAGMM]] — evaluated on noisy / free-rider AUROC only.
3. **Dual oracle implementation.** (a) exact retrain SV: 1B (N=5, 32 retrain ≈ 15 min; N=10, 1024 retrain ≈ 3.5 days B200×4) + 3B (N=5, 32 retrain ≈ 45 min). (b) IRDS-定 SV: cross-silo exact enumeration at N=10 (forward only, ≈ 3.5h / 7h / 28h × 3 seed at 1B / 3B / 7B); cross-device MC at N=100 (5000–10000 sample, standard FedSV practice).
4. **Ripple head-to-head + theoretical reduction (bonus).** Empirical comparison via Ripple sample-level → client aggregation (built in Phase 0). Theoretical reduction sketch: under LoRA + 2-term Taylor, does Ripple drop+ripple specialize to Flirds 1st + 2nd? If yes → Proposition. If no → related-work differentiator paragraph. Comparison metric = **training-performance side** (client-selection convergence + downstream task acc/F1 + noisy/free-rider AUROC), *not* SV-approximation side (sample-level vs client-level renders direct SV comparison ill-defined). Argument: Ripple's lack of 2nd-order term = systematic disadvantage in non-IID FL where client interactions matter.

### ★★ characterization + ablation

5. **α-sweep × E-sweep drift residual matrix.** $E\in\{1,3,5,10\}$ × Dirichlet $\alpha\in\{0.01, 0.1, 0.5, 5.0\}$ = 16 cells × 3 seeds = 48 runs per scale. Validates Prop 1 (monotonicity) and Prop 2 (cubic bound). **N1 contingency branch point**: if U-shape re-appears in clean re-run, switch to N1 option (a) Drop + empirical reporting.
6. **Q2 variant comparison (3 × 3).** Variants = {default Flirds, layer-wise weighted, phase-normalized}. Metrics = {client-selection convergence, downstream task acc/F1, noisy/free-rider AUROC}. 9-cell matrix per scale. Tests whether variants recover the ① AUROC = chance result on default (partial limitation recovery) or fail uniformly (limitation is estimator-form-robust).
7. **②③ characterization experiments.**
   - ②: per-layer $\phi_k^{(r)}$ decomposition; last-layer term cross-round sign oscillation under non-IID ($\alpha=0.1$) vs IID ($\alpha=5.0$); rank-change diagnostic when last layer excluded.
   - ③: late-joiner test (fixed-quality client joining at $r=1$ vs $r=T/2$) — magnitude / timing confounding magnitude.
8. **Adversarial stress regimes.** $\alpha \le 0.01$; label-flip × OOD-mix combinations; larger $N$ tail; late-joiner extremes. ρ behavior on both (a) and (b) oracle reported: hypothesized "(b) ρ holds under stress, (a) ρ degrades gradually" — N3 dual-oracle framing's empirical anchor.
9. **Non-IID valuation bias quantification.** ① deferral's derived obligation. Extracted from #5 data (no separate run); reframes the α-axis as "client-quality vs label-distribution-skew" decomposition.

### ★ scale extension

10. **7B FedDQC-comparable instruction-tuning bench.** Llama-2-7B + LoRA + 5-domain client split (medical / legal / code / math / general). Direct comparison to [[sources/feddqc|FedDQC]] (only FL+LLM precedent in wiki) and [[sources/less|LESS]] (closest centralized analog, same model). Full experiment matrix (see below) at 7B as at 1B/3B.

## Experiment matrix (locked 2026-05-27)

| | 1B (Llama-3.2-1B-Instruct) | 3B (Llama-3.2-3B-Instruct) | 7B (Llama-2-7B) |
|---|---|---|---|
| Flirds 본체 | ✅ | ✅ | ✅ |
| (b) in-run oracle | exact (N=10) + MC (N=100) | exact (N=10) + MC (N=100) | exact (N=10) + MC (N=100) |
| (a) retrain SV | ✅ N=5 + N=10 | ✅ N=5 | ❌ (compute infeasible) |
| Downstream metric (3종) | ✅ | ✅ | ✅ |
| Baseline run set (10) | full | full | full |
| α-sweep × E-sweep (16 cell) | full | full | full |
| Q2 variants (3×3) | full | full | full |
| Stress regimes | full | full | full |
| ②③ characterization | ✅ | ✅ | ✅ |

Estimated compute (B200 × 4): 1B = sub-week, 3B = 1 week, 7B = ~1 week. Phase 0 (CNN sanity) = 5–7 days B200 × 1, prior to LLM phase.

## Baseline selection rationale

**Excluded (LLM environment unsuitable, 2026-05-27 review)**:
- [[sources/space-participant-amalgamation|SPACE]] — prototype-based evaluation requires discrete class labels; LLM instruction tuning is generative open-ended.
- [[sources/game-of-gradients-sfedavg|S-FedAvg]] — aggregation method, not a valuation baseline.
- [[sources/fedcorr|FedCorr]] — prediction-subspace LID assumes classification; not transferable to generative loss.

**Code availability for included baselines**:
| Baseline | Code | Action |
|---|---|---|
| ShapleyFL | ✅ [ZJU-DIVER/ShapleyFL](https://github.com/ZJU-DIVER/ShapleyFL-Robust-Federated-Learning-Based-on-Shapley-Value) | use as-is |
| Data Banzhaf in FL | ✅ semivalue libraries (`pyDVL`, `OpenDataVal`) | adapt |
| GTG-Shapley | ❌ | reproduce → Phase 0 |
| FedSV (Wang 2020) | ❌ | reproduce → Phase 0 |
| ComFedSV | ❌ | reproduce → Phase 0 |
| Ripple Shapley | ❌ (AAAI'26, newest) | reproduce → Phase 0 + author contact if pseudocode ambiguity |
| Full FedAvg / Random-selection FedAvg | trivial | implement |
| loss-heuristic, Flirds-1st-only | trivial | implement |

## Differentiators vs. existing FL valuation methods

| Method | Granularity | Comm. overhead vs. vanilla FedAvg | In-run? | Notes |
|---|---|---|---|---|
| [[sources/principled-federated-data-valuation\|FedSV (Wang et al. 2020)]] | client | 0 comm, but $O(Tm^2)$ server utility evals | per-round retrain-surrogate | **origin** of federated Shapley; the canonical baseline |
| [[sources/comfedsv\|ComFedSV]] | client | + all-client rounds (Everyone-Being-Heard) | retrain-free but completion-imputed | fixes FedSV partial-participation asymmetry; cross-device baseline |
| [[sources/gtg-shapley\|GTG-Shapley]] | client | 0 (sub-model reconstruction) | trajectory-faithful via gradient logs | guided MC; multi-round |
| [[sources/shapleyfl\|ShapleyFL]] | client | 0 | surrogate per-round | importance-sampling client selection |
| [[sources/ripple-shapley\|Ripple Shapley]] | **sample** | 0 (logs) | yes (Jacobian propagation) | the most direct competitor for "in-run, single-run, FL" |
| **Flirds** (this project) | **client** | **0** | **yes** (closed-form 1st+2nd Taylor) | LoRA + 2nd-order + zero communication overhead |

Closest comparator: [[sources/ripple-shapley|Ripple Shapley]]. **Flirds differs**:
- client-level (deliberate, not aggregated from sample-level);
- 1st+2nd Taylor closed-form rather than recursive Jacobian-chain low-rank approximation;
- LoRA explicit in design;
- explicit 2nd-order client-interaction term (argued in §4 to be load-bearing in non-IID FL).

## Centralized positioning (added 2026-05-22)

Ingesting four 2023–2024 centralized data-selection / IF-at-LLM-scale papers tightens Flirds' positioning relative to centralized work:

| Source | Role for Flirds positioning |
|---|---|
| [[sources/less\|LESS]] (Xia et al., ICML 2024) | **Closest centralized analog.** Same LoRA + gradient-similarity-IF + validation-set anchor. Flirds = LESS moved to FL, switched from per-example to per-client, Adam-Γ cosine dropped (Δw_k absorbs local optimizer), 2nd-order term added, no reusable datastore. Positioning load-bearing: LESS sits *between* IRDS and Flirds in related work. |
| [[sources/grosse-llm-influence\|Grosse et al. 2023]] (EK-FAC at 52B) | **Upper-bound anchor.** Largest-scale IF demonstration; explicitly notes per-training-gradient is the binding bottleneck even with EK-FAC. Flirds gets Δw_k for free from FedAvg — zero additional gradient compute, zero communication. Also: Grosse's [[concepts/proximal-bregman-response\|PBRF]] reframing = the existing-literature permission slip Flirds uses for "(b) IRDS-定 oracle is a well-defined target." |
| [[sources/mates\|MATES]] (Yu et al., NeurIPS 2024) | **Empirical backing for the 1B scale.** 1.1% absolute downstream gain on Pythia-1B from data-influence-based selection at 25B tokens. Refutes "1B is too small for influence to matter" reviewer concern. Locally-probed oracle conceptually closest to (b) IRDS-定 SV — both fix the trajectory + read off counterfactual loss change from minimal perturbation. |
| [[sources/dsdm\|DsDm]] (Engstrom et al., ICML 2024) | **Light add — Datamodels → LLM bridge.** Proves linear datamodels remain meaningful at 1.3B. Cleanest "similarity ≠ value" demonstration for [[threads/data-quality-vs-data-value]]. Not a baseline candidate (different setting + granularity + objective). |

These are **centralized** methods and do *not* enter the FL baseline table — that table is Flirds' direct (federated) comparator set. Centralized references serve related-work and positioning sections of the paper. See [[threads/data-selection-for-llms]] for the synthesis.

## Reading list — relevant wiki sources

### Direct foundations
- [[sources/in-run-data-shapley]] — IRDS, the centralized predecessor Flirds extends.
- [[sources/data-banzhaf]] — the "Federated Banzhaf" baseline *is* this Data Banzhaf semivalue applied in FL (no dedicated federated-Banzhaf paper exists).
- [[sources/datainf]], [[sources/logix]] — gradient-based attribution at LLM scale; LoRA framing.

### Centralized positioning (added 2026-05-22)
- [[sources/less]] — closest centralized analog. 7B baseline anchor.
- [[sources/grosse-llm-influence]] — upper-bound scale anchor + PBRF reframing.
- [[sources/mates]] — 1B-scale backing.
- [[sources/dsdm]] — similarity ≠ value evidence.

### Federated-Shapley field
- [[sources/principled-federated-data-valuation]] — **FedSV (Wang 2020)**, origin + canonical baseline.
- [[sources/comfedsv]] — cross-device fairness baseline.
- [[sources/gtg-shapley]], [[sources/shapleyfl]] — baselines.
- [[sources/ripple-shapley]] — closest competitor.
- [[sources/space-participant-amalgamation]] (excluded from baseline set), [[sources/game-of-gradients-sfedavg]] (excluded).

### Robustness-side baselines (noisy / free-rider detection only)
- [[sources/fldetector]] — temporal-consistency detector.
- [[sources/foolsgold]], [[sources/fltrust]] — cross-client / trusted-cosine.
- [[sources/free-riders-fl-std-dagmm]] — STD-DAGMM.
- [[sources/fedcorr]] (excluded from primary detection set — classification-only).

### Framing / context
- [[sources/asymmetric-data-shapley]] — temporal / state-conditioned framing.
- [[sources/distributionally-robust-data-valuation]] — robustness of validation utility.
- [[sources/dice]] — full decentralization (adjacent, out of scope).
- [[sources/feddqc]] — empirical evidence that gradient attribution fails on heterogeneous real-world FL; 7B-scale FL+LLM precedent.

### Threads
- [[threads/retraining-vs-in-run-attribution]] — (a)/(b) distinction Flirds depends on.
- [[threads/federated-and-decentralized-attribution]] — full FL attribution landscape.
- [[threads/influence-functions-at-llm-scale]] — LoRA + LLM scaling context.
- [[threads/data-selection-for-llms]] — LESS / MATES / DsDm synthesis; centralized lineage.
- [[threads/noise-ood-malicious-client-separation]] — robustness-side prior art; deferred-limitation backing.
