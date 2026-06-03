---
type: source
title: "FedIF: Lightweight and Robust Federated Data Valuation"
created: 2026-06-03
updated: 2026-06-03
topic: flirds
tags: [federated-learning, data-valuation, tracin, trajectory-influence, first-order, robust-aggregation, cnn]
---

# FedIF

## Citation

Guojun Tang, Steve Drew (University of Calgary), Jiayu Zhou (University of Michigan), Mohammad Mamun (National Research Council Canada). *Lightweight and Robust Federated Data Valuation*. arXiv:2509.25560v1, 29 Sep 2025. Code: `github.com/guojuntang/FedIF`.

Raw: `raw/papers/flirds/2509.25560_FedIF.pdf`

## TL;DR

A **TracIn-style, 1st-order** federated data-valuation + robust-aggregation method. The server scores each client per round by the dot product of its **L2-normalized update** with the **validation gradient**, min-max-normalizes across the round's participants, EMA-smooths over rounds, and turns the result into adaptive averaging weights. Explicitly the **first** work to bring trajectory influence (TracIn) to FL on client updates. Claims robustness comparable to or beating Shapley-value-based FL while cutting aggregation overhead up to **450×**. CNN-only; **no Hessian, no 2nd-order, no LoRA/LLM**.

## Problem

SV-based robust FL aggregation (ShapleyFL, AFedSV, etc.) needs repeated server-side **model reweighting + inference** over client subsets every round — accurate but expensive, capping scalability. The authors ask whether a cheaper valuation exists, and adapt **TracIn** (Pruthi et al. 2020), whose 1st-order gradient trace is far cheaper than coalition re-evaluation. They explicitly **reject classical influence functions** for FL: those need a per-client Hessian inverse, "hard to verify" and unsuitable for FL — motivating a Hessian-free design.

## Method

TracIn (centralized) traces a training point's influence on a test point as $\sum_t \eta_t\,\nabla\ell(w_t,d)\cdot\nabla\ell(w_t,d')$. FedIF lifts this to *client updates vs. a validation set*. Since clients upload **parameters** $w_t^i$, not gradients, it substitutes $\Delta w_t^i=w_{t-1}-w_t^i$ as a proxy for the local gradient step:

- **Round influence** (Alg. 1, Eq. 6):
$$\Phi_i^t=\frac{\Delta w_t^i}{\lVert\Delta w_t^i\rVert}\cdot\nabla\ell(w_{t-1},D'),$$
i.e. the **L2-normalized client update** dotted with the **validation gradient** at the round-start global model. Normalization ("local weight normalization") removes scale so direction dominates — key for the gradient-noise case.
- **Round (min-max) normalization** (Eq. 7): $\Psi_i^t=\frac{\Phi_i^t-\min\Phi^t}{\max\Phi^t-\min\Phi^t}$ — fair comparison across rounds with different participants/scales.
- **Smooth update** (EMA, Eq. 8): $\Omega_i^t=(1-\gamma)\Omega_i^{t-1}+\gamma\Psi_i^t$ for participants ($\gamma\in\{0.3,0.4\}$), carried forward for non-participants — stabilizes oscillatory weights.
- **Adaptive weights** (Eq. 9): $p_i^t=\Omega_i^t/\sum_j\Omega_j^t$, then $w_t=\sum_{i\in S_t}p_i^t w_t^i$.

**Theory** (Thm. 1): under $L$-smoothness, bounded local dissimilarity, and a noisy-update model $g_t^i=\nabla F^i(w)+\delta_t^i$, FedIF gives a **tighter one-step global-loss-change upper bound than FedAvg** in noisy settings. Remark 2: a noisy update misaligns with the validation gradient → low influence → low weight ($p_{t-1}^i\propto 1/\lVert\delta_{t-1}^i\rVert$), shrinking the noise term; in clean settings it reduces to FedAvg (Remark 1).

## Key results

- **Datasets/model**: CIFAR-10 + Fashion-MNIST, **CNN** base model, 100 clients, Dirichlet $\alpha=1$, $C=0.1$, $E=5$, $B=16$, SGD $\eta=0.001$, momentum 0.9, $T=100$. Validation = 20% of test split. RTX 4080.
- **Robustness**: comparable-to-better than **AFedSV** (the SV baseline) under label noise and gradient noise; in clean settings ~FedAvg (as theory predicts). Surpasses AFedSV on CIFAR-10 label/gradient noise; close on Fashion-MNIST.
- **Efficiency**: aggregation ~**0.2 s/round vs. AFedSV's ~70–92 s → up to 450×** less (AFedSV runs Monte-Carlo SV reweighting+inference).
- **Failure mode**: ineffective vs. **PGD adversarial samples** — PGD preserves an update *direction* similar to clean inputs, so a direction-only valuation can't flag it. Also collapses at extreme noise ($n=0.7$, ratio $(0.7,0.8)$).
- **Ablation**: local weight normalization (WN), round normalization (RN), smooth update (SU) each help; WN matters most under gradient noise.

## Connections

- [[concepts/influence-function]] — FedIF is the **gradient-trace (TracIn) branch**, *not* the Hessian-inverse branch; it explicitly rejects classical IF for FL as Hessian-infeasible.
- [[sources/less|LESS]], [[sources/mates|MATES]] — share the TracIn / 1st-order-validation-gradient-dotted-with-training-gradient lineage (data selection), but centralized; FedIF is the FL aggregation instance.
- [[concepts/federated-shapley]], [[sources/shapleyfl|ShapleyFL]], [[sources/principled-federated-data-valuation|FedSV]] — the SV-based FL valuation FedIF positions itself *against* on cost (AFedSV = Tastan et al. 2024, its main baseline).
- [[sources/ripple-shapley|Ripple Shapley]] — the closest contrast: both do **in-run, trajectory-based** FL attribution reading off uploaded updates. Ripple = sample-level Shapley with a recursive cross-round Jacobian-chain (curvature-aware propagation); FedIF = client-level 1st-order single-round dot product (no cross-round propagation, no curvature).
- [[sources/fltrust|FLTrust]] — near-identical *mechanism shape* (normalize update, compare to a server-side reference, trust-weight) but FLTrust compares to a **server-model update** via ReLU-cosine; FedIF compares to the **validation gradient** via normalized dot product, and reports a smoothed valuation.
- [[sources/feddqc|FedDQC]], [[sources/fedcorr|FedCorr]], [[sources/free-riders-fl-std-dagmm]] — sit in the same noise/quality-control FL space.
- [[threads/noise-ood-malicious-client-separation]] — direction-alignment-to-validation as the separation signal; documents the PGD blind spot.
- [[threads/retraining-vs-in-run-attribution]] — squarely in-run (no retraining, no coalition replay).
- [[threads/influence-functions-at-llm-scale]] — relevant by *contrast*: FedIF stays 1st-order CNN-scale precisely because it deems the Hessian infeasible; Flirds and LLM-scale IF (LoGra/EKFAC) argue 2nd-order is tractable at scale.
- [[threads/utility-function-design]] — normalized-update · validation-gradient is a distinct utility from accuracy/loss/retrain.

## Relevance to Flirds

**Scoop risk: MEDIUM.** This is the **closest federated in-run-influence-on-$\Delta w$ method besides Ripple**, and the gaps it leaves *are* Flirds' differentiation. It is **not** a scoop of Flirds' core claim, but it is the most important "why isn't 1st-order enough?" baseline.

Precise differentiation:

1. **Order.** FedIF is **strictly 1st-order** — round influence is $\langle$normalized $\Delta w_i$, validation gradient$\rangle$, a TracIn dot product. It has **no 2nd-order term and no client-interaction (HVP) term**, and the authors explicitly call the Hessian infeasible in FL. Flirds' novelty is exactly the **2nd-order client-interaction term** (HVP, 1/round) that captures coalition curvature — the part IRDS deemed small *centrally per-step* but Flirds argues is non-trivial in *FL per-round multi-step*. FedIF is the empirical "1st-order baseline" Flirds must beat to justify 2nd-order.
2. **Not Shapley.** FedIF produces an EMA-smoothed, min-max-normalized **influence score → aggregation weight**; it is *not* a Shapley value and abandons the Shapley axioms. Flirds reports **client-level Shapley** credit. (FedIF is a valuation-for-weighting, like a TracIn-flavored [[sources/fltrust|FLTrust]].)
3. **Aggregation-altering vs. post-hoc.** FedIF *changes* aggregation via $p_i^t$ (it is a robust-FL algorithm). Flirds values clients off the realized **vanilla** FedAvg run without altering it.
4. **Scale/PEFT.** FedIF is **CNN-only**, image classification, full-model; **no LoRA, no LLM**. Flirds' headline is LoRA-LLM scale.
5. **No closed-form Taylor loss decomposition.** FedIF's dot product is a 1st-order *proxy* (uses $\Delta w_i$, not the true gradient, as the training-side vector) without an explicit Taylor remainder; Flirds writes the per-round validation-loss change as a 1st+2nd-order Taylor expansion with the 2nd-order term made explicit.

The differentiation sentence: *Flirds = client-level **Shapley** via **closed-form 1st+2nd-order Taylor** (with the **2nd-order HVP interaction term**) on **LoRA-LLM** updates, post-hoc on vanilla FedAvg; FedIF = client-level **1st-order TracIn score** → **adaptive weights**, **no 2nd-order**, **CNN-only**.* FedIF's own future-work line — "consider more information from the gradient to evaluate the local update" — literally points at the 2nd-order direction Flirds takes.

## Notes / open questions

- FedIF uses $\Delta w_t^i$ (multi-local-epoch parameter delta, $E=5$) as a stand-in for the single-step gradient in TracIn's Eq. 2. That multi-step delta is the *same object* Flirds expands via Taylor — but FedIF only takes its 1st-order inner product. Directly motivates Flirds' "round-delta ≠ 1 step → 2nd-order matters" argument. Strong evidence for Flirds' framing.
- PGD blind spot (direction-aligned poison) is a 1st-order limitation; does Flirds' 2nd-order term separate direction-aligned-but-curvature-different updates? Candidate experiment (Phase 3 backdoor/temporal).
- Min-max round normalization is order-/cohort-dependent (depends on the round's best & worst participant) — non-axiomatic, unlike Shapley. Contrast with Flirds' efficiency/symmetry-preserving construction.
- "AFedSV" (FedIF's ref [17]) is **not** Tastan/IJCAI'24 — it resolves to [[sources/shapleyfl|ShapleyFL]] (Sun et al., KDD 2023), the adaptive surrogate-SV aggregation, **already in the wiki**. The IJCAI'24 paper that FedIF also cites (ref [19], Tastan et al.) is [[sources/shapfed|ShapFed]], a *different* class-specific FL-Shapley method (ingested 2026-06-03). So FedIF's SV comparator = ShapleyFL; ShapFed is the recent SOTA class-specific cousin.
- The 450× is **aggregation-time** only (training time is ~equal across methods); frame carefully when comparing to Flirds' near-free closed-form cost.
- Code is public (`github.com/guojuntang/FedIF`) — usable as a real 1st-order baseline implementation if Flirds needs head-to-head numbers.
