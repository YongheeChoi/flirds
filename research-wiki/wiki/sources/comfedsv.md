---
type: source
title: "Improving Fairness for Data Valuation in Horizontal Federated Learning (ComFedSV)"
created: 2026-05-19
updated: 2026-05-19
topic: flirds
tags: [federated-learning, shapley, participant-valuation, fairness, matrix-completion, low-rank]
---

# ComFedSV

## Citation

Zhenan Fan, Huang Fang, Zirui Zhou, Jian Pei, Michael P. Friedlander, Changxin Liu, Yong Zhang (UBC / Huawei Canada / SFU / KTH). *Improving Fairness for Data Valuation in Horizontal Federated Learning*. ICDE 2022. arXiv:2109.09046v3, May 2022.

Raw: `raw/papers/flirds/2109.09046v3.pdf`

## TL;DR

[[sources/principled-federated-data-valuation|FedSV (Wang et al. 2020)]] gives *zero* round-credit to clients the server didn't sample, so two clients with identical data can score wildly differently — a symmetry violation. ComFedSV reframes per-round per-coalition utilities as a (round × subset) **utility matrix**, proves it is approximately low-rank under smoothness/strong-convexity, fills unobserved entries by low-rank matrix completion, and computes Shapley on the completed matrix. Provably approximately satisfies symmetry, zero-element, additivity.

## Problem

Under partial participation ($|I_t|=m<N$), FedSV's "unselected ⇒ zero round-contribution" breaks Shapley symmetry: with identical data $D_i=D_j$, expected values match but *realized* $s_i,s_j$ diverge w.h.p. (MNIST: relative difference $>0.5$ in 65% of runs with client 9 = copy of client 0). Exact Shapley needs per-coalition retraining — infeasible — motivating an $\epsilon$-Shapley-fairness relaxation ($\epsilon$-symmetry / $\epsilon$-zero-element / $\epsilon$-additivity).

## Method

- **Utility matrix** $\mathcal{U}\in\mathbb{R}^{T\times 2^N}$, $\mathcal{U}_{t,S}$ = round-$t$ test-loss decrease from the model aggregated over coalition $S$. Only $\{\mathcal{U}_{t,S}:S\subseteq I_t\}$ observed.
- **Low-rank structure**: similar clients → similar columns; slow loss change → similar adjacent rows. Under convex/Lipschitz/smooth losses, $\epsilon$-rank bounded by cumulative global-parameter path length; with $\mu$-strong convexity, $\epsilon\text{-rank}\in O(\log T/\epsilon)$, **independent of client count**. Neural-net case verified empirically (singular-value decay).
- **Completion**: regularized matrix factorization (LIBMF), rank from the propositions, solved post-training. Needs **Assumption 1 (Everyone Being Heard)**: every client selected in ≥1 round, costing ~$\lceil N/m\rceil$ extra rounds-equivalent.
- **ComFedSV**: Shapley evaluated on completed entries; Monte-Carlo permutation sampling reduces the exponential matrix to a $T\times MN$ reduced completion, $M=O(N\log N)$, total $O(TN^2\log N)$.
- **Guarantee** (Theorem 1): if $\|\mathcal{U}-WH^\top\|_1\le\delta$, ComFedSV is $(4\delta/N)$-Shapley-fair; perfect recovery ⇒ exact symmetry/zero-element/additivity.

## Key results

- Fairness: empirical CDF of identical-client score gap stochastically dominated by FedSV on Synthetic/MNIST/FMNIST/CIFAR-10 (non-IID).
- Noisy-data detection: Spearman vs. ground truth near-matches and beats FedSV on all four datasets; noisy-label detection (100 clients, 10 with 30% flips) higher Jaccard than FedSV.
- Cost: FedSV/ComFedSV runtime ratio → participation rate $K/N$; $u_t$ calls $O(TNK\log N)$ vs FedSV $O(TK^2\log K)$.

## Connections

- [[concepts/federated-shapley]] — multi-round client-level federated Shapley; completion of unobserved coalitions is its distinguishing trick. [[concepts/shapley-value]] — relaxes exact axioms to $\epsilon$-Shapley fairness; [[concepts/semivalue]] — stays in the Shapley corner (exact coefficients). [[concepts/federated-learning]] — horizontal FL + FedAvg with partial sampling.
- [[threads/federated-and-decentralized-attribution]] — client-level, multi-round, trajectory-anchored-**but-imputed** (distinct from in-run). [[threads/dataset-vs-data-point-valuation]] — players are clients = dataset holders.
- [[sources/principled-federated-data-valuation]] — direct parent: ComFedSV = FedSV + utility-matrix completion. [[sources/gtg-shapley]] — sibling retrain-free client-Shapley; GTG *reconstructs* missing sub-models (faithful) whereas ComFedSV *imputes* them (low-rank completion). [[sources/shapleyfl]] — also multi-round surrogate federated Shapley; contrast surrogate-estimation vs completion.

## Relevance to Flirds

ComFedSV is the **principled-completion alternative** to Flirds' locked decision ("no participation normalization by default; cross-device experiments show ranking-within-participation-tier still recovers quality") — hence the natural **cross-device baseline**.

1. **Extra cost beyond vanilla FedAvg**: "Everyone Being Heard" forces ≥1 all-client round (~$\lceil N/m\rceil$ extra comm-equivalent) and needs *all* clients' updates each round to form the observed matrix, plus a server-side completion solve. Flirds (selected-only LoRA $\Delta w_k$, zero extra comm) is strictly cheaper.
2. **Retrain-free but imputation-based, not in-run**: utilities are per-round 1st-order (no curvature term, unlike Flirds' 1st+2nd Taylor); missing coalitions are *statistical interpolations*, and the fairness guarantee is conditional on completion error $\delta$ — a knob Flirds doesn't need.

Framing: ComFedSV = "pay completion cost to restore symmetry across participation gaps"; Flirds = "accept asymmetry, rank within tiers, pay nothing." Head-to-head on cross-device non-IID partitions (ranking quality, noisy-client detection, communication cost) is the obvious experiment.

## Notes / open questions

- Low-rank theory is convex; LoRA-PEFT LLM fine-tuning is highly non-convex — does the utility matrix stay low-rank for transformer LoRA updates? (Paper only shows empirical low-rankness to VGG16.)
- "Everyone Being Heard" is benign cross-silo but awkward cross-device (Flirds' target regime) — quantify the extra-comm penalty on a realistic cross-device schedule.
- ComFedSV needs all-client updates each round — contradicts client-sampling's communication-saving rationale; reconcile when used as a Flirds baseline.
- Completion error $\delta$ ungoverned in practice; the $(4\delta/N)$ guarantee is only as good as the solver, with no a priori $\delta$ bound for non-convex models.
