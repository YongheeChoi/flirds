---
type: project
title: "Flirds — Federated Learning + In-Run Data Shapley"
created: 2026-05-05
updated: 2026-05-19
sources: [in-run-data-shapley, principled-federated-data-valuation, comfedsv, gtg-shapley, shapleyfl, space-participant-amalgamation, ripple-shapley, game-of-gradients-sfedavg, data-banzhaf, datainf, logix, asymmetric-data-shapley, distributionally-robust-data-valuation, dice, feddqc, fldetector, fedcorr, fltrust, foolsgold, free-riders-fl-std-dagmm]
tags: [flirds, project-state, design-decisions]
---

# Flirds — project state

**Flirds = Federated Learning + In-Run Data Shapley.** A federated data-valuation method extending [[sources/in-run-data-shapley|IRDS]] (In-Run Data Shapley) to the FL setting via LoRA-based PEFT.

This page is the project's running state — locked design decisions, open questions, planned experiments. The **primary record** is `raw/conversations/flirds/conversation{1..4}.md` (chronological design conversations with another LLM); this wiki page is the curated distillation. When in doubt, read the raw.

## One-line method statement

> **Flirds** computes client-level Shapley values via closed-form 1st + 2nd order Taylor expansion of the validation loss change per FL round, using only the client updates $\{\Delta w_k\}$ that the server already receives in vanilla FedAvg — adding zero communication cost.

## Core formula (round $r$)

$$\phi_k^{(r)} \;\approx\; -\nabla\ell(w^r, z^{val}) \cdot \Delta w_k \;+\; \tfrac{1}{2}\, \Delta w_k^\top H^{(val)}(w^r) \cdot \Delta W^{(r)}$$

where $\Delta W^{(r)} = \sum_j \Delta w_j$ is the sum across all participating clients. Total: $\phi_k = \sum_r \phi_k^{(r)}$.

The 2nd-order term captures client interactions. **Per-round cost: 1 HVP for $u := H^{(val)} \Delta W^{(r)}$, then $N$ dot products** $\Delta w_k^\top u$. LoRA dimension keeps it cheap.

## Locked design decisions

(As of conversation 4, 2026-05-05.)

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

## The mathematical narrative (paper-ready)

### Centralized baseline

In a single SGD step on batch $B$, **data-level Shapley summed over a client equals client-level Shapley directly** — for both 1st- and 2nd-order terms. Conversation 3, §4 proves this:

$$\sum_{z \in I_k} \phi_z^{\text{data}} = -\eta\, \nabla\ell^{val} \cdot G_k \;=\; \phi_k^{\text{client}} \quad (G_k := \sum_{z \in I_k} \nabla\ell(w_t, z))$$

So in centralized 1-step SGD the granularity choice is grouping-invariant.

### FL multi-step deviation: drift residual

Client sends $\Delta w_k = -\eta \sum_{e=0}^{E-1} \sum_{z \in B_{k,e}} \nabla\ell(w_k^{r,e}, z)$ — an $E$-step trajectory endpoint, not a single-step gradient. Taylor-expanding around $w^r$:

$$\Delta w_k \;\approx\; -\eta E \cdot \bar{G}_k(w^r) \;-\; \eta \sum_{e,z} H_z(w^r)(w_k^{r,e} - w^r)$$

The first term is the centralized-equivalent contribution; the second is **client drift residual**, of order $O(\eta E |\bar{H}| \cdot \text{local-trajectory length})$. Vanishes at $E=1$; grows with non-IID client data and large local epochs.

> **Proposition (informal, planned)**: Flirds' client-level Shapley equals centralized data-level Shapley aggregated to the client *plus* a drift residual that quantifies FL's deviation from centralized IRDS.

This proposition is one of the paper's core mathematical contributions. Empirically: vary $E$, plot residual size, show valuation quality across the range.

## Open questions (need resolution before paper is complete)

### Method-side

1. **Noise vs OOD-good distinction.** *(DEFERRED 2026-05-18 — Yonghee: "algorithmically too ambiguous to resolve cleanly; park it.")* Single-round sign cannot distinguish them. Original plan: combine **temporal consistency** (real noise has random direction across rounds; OOD-good is consistent) + **cross-client agreement** (noise is unique-to-client; OOD-good aligns with same-distribution clients). Status: no longer the paper's planned second contribution; recast as a **characterized limitation**, framed like IRDS frames its own limits. Consequences of deferral:
   - Narrative: "one contribution + characterized limitation" instead of two contributions (safer, tighter scope).
   - Experiment matrix shrinks: drop the pure-noise-vs-OOD-good separation / validation-expansion-flip / synthetic-shift experiments and the temporal-consistency & cross-client-agreement ablation rows. **Noisy-client and free-rider detection survive unchanged** — they don't need the OOD machinery.
   - This separator was *also* doubling as drift-bias correction (conversation 4 §4). With it parked, drift residual stays purely measured-not-corrected (consistent with the locked decision), **but the non-IID valuation bias must now be quantified as an explicit limitation** — see next-step checklist.
   - **Prior-art backing for the limitation write-up**: [[threads/noise-ood-malicious-client-separation]] (surveyed 2026-05-18). Both deferred signals already exist in the FL-robustness literature (temporal consistency ≈ **FLDetector** KDD'22; cross-client ≈ **FoolsGold/FLTrust/clustered-FL**), and *no* FL method separates "bad-different" from "good-different" inside a signed value — FLDetector itself collapses on non-IID. This is the evidence that the problem is genuinely hard, not an oversight: exactly the framing IRDS uses for its own sign-ambiguity limit.

2. **Cancellation effect (IRDS A.3 issue, amplified in FL).** Last-layer cancellation is severe under non-IID. Layer-wise weighting? Specific layer choice? Open.

3. **Magnitude vs alignment confusion (limit (b) of IRDS).** Late-fit data has small gradient magnitude → systematically undervalued. In FL, late-joining clients suffer doubly. Phase-normalization or cumulative-reduction tracking? Open.

4. **Trajectory dependence ("feature in IRDS, bug in market context").** Same client could be valued differently across re-runs with different batch order. If Flirds is used for compensation, multiple-run averaging may be needed. Decision deferred — depends on whether the paper's main use case is incentive design or research-side analysis.

### Experiment-side (under construction)

5. **Exact Shapley as ground-truth in FL — which definition?** Conversation 3, §2 settled on:
   - **(a) Retraining-based exact Shapley** ($U(S) = $ FL training using only clients in $S$): natural ground truth, feasible at $N{=}10$ cross-silo with LoRA. *Sanity check only.*
   - **(b) In-run exact Shapley** ($U_{\text{total}}(S) = \sum_r [\ell(w^r + \sum_{k\in S} p_k \Delta w_k, z^{val}) - \ell(w^r, z^{val})]$): the *correct* ground truth for Flirds because both fix the same trajectory. **Honest comparison: Flirds vs. Monte-Carlo (b).** Yonghee initially conjectured (a) and (b) coincide via additivity; conversation 3 corrects this — they're Shapley over *different utility functions* and so generally differ.

6. **Benchmarks** (planned, conversation 3 §2):
   - Noisy client detection (label corruption / random updates).
   - Free-rider detection (zero / fake updates).
   - Domain attribution for LLM (each client a different domain corpus; validation = specific domain).
   - Client selection convergence (top-$K$ by Flirds → train → measure speed/perf).
   - Baseline comparison: [[sources/gtg-shapley|GTG-Shapley]], [[sources/principled-federated-data-valuation|FedSV]] (= Wang et al. 2020, the federated-Shapley origin — *not* the unrelated 2025 "FedSV: Byzantine-Robust FL via Shapley Value" arXiv:2502.17526), [[sources/comfedsv|ComFedSV]], **Federated Banzhaf** (= [[sources/data-banzhaf|Data Banzhaf]] semivalue applied in FL — no dedicated FL-Banzhaf paper exists; already in the wiki), simple loss-heuristic.
   - **Robustness-side baselines for the *surviving* noisy/free-rider benchmarks** (added 2026-05-18, ingested 2026-05-19, → [[threads/noise-ood-malicious-client-separation]]): [[sources/fldetector|FLDetector]] (temporal-consistency detector), [[sources/foolsgold|FoolsGold]] / [[sources/fltrust|FLTrust]] (cross-client / trusted-cosine), [[sources/fedcorr|FedCorr]]-LID (noisy-client), [[sources/free-riders-fl-std-dagmm|STD-DAGMM]] (free-rider). Framing: Flirds is a *valuation* method that does detection competitively with *dedicated* detectors **without their non-IID false-positive penalty** (it down-weights signed value rather than hard-discarding).

7. **LLM choice and dataset.** Llama-3.2-1B/3B candidates for LoRA finetuning realism. Multi-domain dataset candidates: Flan / T0 / dolly subset combination. Not finalized.

### Ablations (planned set, conversation 3 §3)

| Ablation | What it isolates |
|---|---|
| 1st only vs 1st+2nd | 2nd-order client-interaction value |
| Temporal-consistency component on/off | Noise vs OOD-good signal |
| Cross-client agreement component on/off | Same |
| Validation size (small/medium/large) | Validation-noise propagation |
| Validation distribution (uniform/biased) | Distribution choice prejudge |
| LoRA rank (4/8/16/32) | PEFT dimension effect |
| Local epoch $E$ (1/3/5/10) | Drift residual / Taylor error |
| Aggregation (FedAvg / FedProx) | Client-drift correction effect |
| Participation rate | Cross-device robustness |
| Noise ratio (10/30/50%) | Detection limit |

## Differentiators vs. existing FL valuation methods

| Method | Granularity | Comm. overhead vs. vanilla FedAvg | In-run? | Notes |
|---|---|---|---|---|
| [[sources/principled-federated-data-valuation\|FedSV (Wang et al. 2020)]] | client | 0 comm, but $O(Tm^2)$ server utility evals | per-round retrain-surrogate | **origin** of federated Shapley; the canonical baseline |
| [[sources/comfedsv\|ComFedSV]] | client | + all-client rounds (Everyone-Being-Heard) | retrain-free but completion-imputed | fixes FedSV partial-participation asymmetry; cross-device baseline |
| [[sources/gtg-shapley\|GTG-Shapley]] | client | 0 (sub-model reconstruction) | trajectory-faithful via gradient logs | guided MC; multi-round |
| [[sources/game-of-gradients-sfedavg\|S-FedAvg]] | client | 0 | per-round only | aggregation reweighting, not pure valuation |
| [[sources/shapleyfl\|ShapleyFL]] | client | 0 | surrogate per-round | importance-sampling client selection |
| [[sources/space-participant-amalgamation\|SPACE]] | client | + extra round of distillation | end-state | knowledge amalgamation + prototype eval |
| [[sources/ripple-shapley\|Ripple Shapley]] | **sample** | 0 (logs) | yes (Jacobian propagation) | the most direct competitor for "in-run, single-run, FL" |
| **Flirds** (this project) | **client** | **0** | **yes** (closed-form 1st+2nd Taylor) | LoRA + 2nd-order + zero communication overhead |

Closest comparator: [[sources/ripple-shapley|Ripple Shapley]] (sample-level FL in-run via Jacobian chain). **Flirds differs**:
- client-level (deliberate, not aggregated from sample-level);
- 1st+2nd Taylor closed-form rather than recursive Jacobian-chain low-rank approximation;
- LoRA explicit in design.

This needs to be carefully positioned in the paper — Ripple Shapley is the most likely "you've been scooped" reviewer concern. Open: does Ripple Shapley's drop+ripple decomposition specialize to Flirds' formula under specific LoRA + Taylor assumptions?

## Reading list — relevant wiki sources

### Direct foundations
- [[sources/in-run-data-shapley]] — IRDS, the centralized predecessor that Flirds extends.
- [[sources/data-banzhaf]] — the "Federated Banzhaf" baseline *is* this Data Banzhaf semivalue applied in FL; **no dedicated federated-Banzhaf paper exists**, so this is the de facto reference (already ingested — not a missing source).
- [[sources/datainf]], [[sources/logix]] — gradient-based attribution at LLM scale; LoRA framing.

### Federated-Shapley field
- [[sources/principled-federated-data-valuation]] — **FedSV (Wang et al. 2020), the origin of federated Shapley and the canonical baseline.** (Distinct from the unrelated 2025 robustness paper of the same acronym.)
- [[sources/comfedsv]] — ComFedSV; FedSV + utility-matrix completion; cross-device fairness baseline.
- [[sources/gtg-shapley]], [[sources/shapleyfl]], [[sources/space-participant-amalgamation]], [[sources/game-of-gradients-sfedavg]] — baselines / contemporaries.
- [[sources/ripple-shapley]] — closest competitor.

### Robustness-side baselines (surviving noisy/free-rider benchmarks)
- [[sources/fldetector]] — temporal-consistency detector; primary scoop risk if the deferred separator is ever revived.
- [[sources/foolsgold]], [[sources/fltrust]] — cross-client-agreement / trusted-cosine; their non-IID over-penalization backs Flirds' characterized-limitation framing.
- [[sources/fedcorr]] — LID-based noisy-client detection + label correction.
- [[sources/free-riders-fl-std-dagmm]] — STD-DAGMM free-rider detector; origin of the free-rider benchmark.

### Framing / context
- [[sources/asymmetric-data-shapley]] — temporal/state-conditioned framing relevant for round-by-round valuation.
- [[sources/distributionally-robust-data-valuation]] — robustness of validation utility (relevant if validation choice becomes contested).
- [[sources/dice]] — full decentralization; out of Flirds' immediate scope but conceptually adjacent.
- [[sources/feddqc]] — empirical observation that gradient-based attribution fails on heterogeneous real-world FL (Fed-WildChat). A risk Flirds inherits and should address explicitly.

### Threads
- [[threads/retraining-vs-in-run-attribution]] — the (a)/(b) distinction Flirds depends on.
- [[threads/federated-and-decentralized-attribution]] — full landscape of FL attribution methods.
- [[threads/influence-functions-at-llm-scale]] — LoRA + LLM scaling context.
- [[threads/noise-ood-malicious-client-separation]] — robustness-side prior art; backs the deferred-limitation write-up and the surviving detection-benchmark baselines.

## Next-step checklist

Updated 2026-05-18 (noise-vs-OOD-good deferred; protocol/baseline discussion in progress):

- [~] ~~Design the noise-vs-OOD-good component~~ — **deferred**, recast as limitation (see open question 1).
- [ ] **Quantify non-IID valuation bias as a limitation** — new obligation created by the deferral (the parked separator was also the drift-bias corrector). Add a non-IID $\alpha$-sweep alongside the $E$-sweep in the drift-residual study.
- [ ] Lock experimental protocol: Llama-3.2-1B primary (3B scale-check); 6–8 domain curated mix (Super-NaturalInstructions for Track B scale); $N{=}10$ cross-silo / $N{=}100,K{=}10$ cross-device.
- [ ] **Ground-truth realization**: (b) MC in-run as the *primary* oracle — forward-pass only, no retraining ($2^{10}$ enumeration/round at $N{=}10$); (a) retraining as sanity-only figure with the "different utility" caveat.
- [ ] Baseline set (cost-matched tiers): must = [[sources/gtg-shapley|GTG-Shapley]], [[sources/ripple-shapley|Ripple Shapley]], loss-heuristic, Flirds-1st-only; recommended = [[sources/principled-federated-data-valuation|FedSV]], Federated Banzhaf (= [[sources/data-banzhaf|Data Banzhaf]] in FL); cross-device = [[sources/comfedsv|ComFedSV]]. (All baseline sources now ingested as of 2026-05-19.)
- [ ] Ripple Shapley head-to-head + theoretical reduction (does its drop+ripple specialize to Flirds 1st+2nd under LoRA+Taylor?) — scoop defense.
- [ ] Implementation skeleton: HVP via LoRA params, $\Delta W^{(r)}$ accumulation, per-round $\phi_k^{(r)}$ computation.
- [ ] Drift-residual study across $E\in\{1,3,5,10\}$ × non-IID $\alpha$ values.
