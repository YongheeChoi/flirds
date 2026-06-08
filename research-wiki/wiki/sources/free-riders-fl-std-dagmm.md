---
type: source
title: "Free-riders in Federated Learning: Attacks and Defenses (STD-DAGMM)"
created: 2026-05-19
updated: 2026-05-19
topic: flirds
tags: [federated-learning, free-rider, anomaly-detection, std-dagmm, robustness]
---

# Free-riders in Federated Learning (STD-DAGMM)

## Citation

Jierui Lin, Min Du, Jian Liu (UC Berkeley). *Free-riders in Federated Learning: Attacks and Defenses*. arXiv:1911.12560v1, Nov 2019 (preprint).

Raw: `raw/papers/flirds/1911.12560v1.pdf`

## TL;DR

First paper to define **free-rider attacks** in FL: clients with no real data fabricate updates to reap rewards / the global model. It catalogs three attack families and proposes **STD-DAGMM** — a high-dimensional anomaly detector (DAGMM augmented with the standard deviation of the update vector) — that catches free-riders across learning rates where plain DAGMM fails. MNIST + 2-layer MLP only.

## Problem

Each FL client receives the global model and (per the assumed incentive model) rewards for "contributing." A client lacking data or wishing to save compute can submit fake updates. A holdout-validation accuracy check is evaded (clever fakes barely change utility); the server is assumed to keep **no per-client history**, so detection must work from one round's updates. A free-rider is fake in *all* rounds and free-riders do not collude.

## Method

**Attack taxonomy** (free-rider builds a correctly-shaped update from public info):

- **Zero-weights** — trivially detected.
- **Random-weights** — i.i.d. uniform $[-R,R]$; tune $R$ to mimic benign std. Evades a plain autoencoder (low-std fakes reconstruct easily) but DAGMM's cosine term flags it.
- **Delta-weights** — $G=M_{j-1}-M_j$ = the averaged previous-round update; near-converged it is indistinguishable from real updates. DAGMM works at $\eta\approx1$ but **fails at small $\eta$** (fake scaled toward 0).
- **Advanced delta-weights** — delta + $\mathcal{N}(0,\sigma)$; $\sigma=10^{-3}$ matches benign std and breaks de-duplication. Plain DAGMM completely fails.

**Defense — STD-DAGMM.** DAGMM (deep autoencoder → low-dim embedding + reconstruction-distance features → GMM energy) is augmented with one scalar: the **standard deviation of the flattened update vector**, stacked with the two distances before the GMM. Free-riders scaled toward zero, or that are an *average* of others' updates, have anomalously **low std**; combining std with DAGMM yields a detector general across $\eta$. (Std alone is insufficient: a tuned $R$/$\sigma$ matches benign std.)

## Key results

- Free-riders obtain the converged global model; delta / advanced-delta evade validation checks and (advanced) plain DAGMM entirely.
- STD-DAGMM detects random, delta ($\eta=0.3$–$1$) and advanced-delta single free-riders with a wide energy margin, early (round 5) and near-converged (round 80).
- 20/100 free-riders, advanced-delta: STD-DAGMM AUC ≈ 0.96 (rnd 5) / 0.91 (rnd 80) vs DAGMM 0.89 / 0.85; dominates DAGMM across all ratios (both degrade as ratio rises).
- Differential privacy with privacy-amplification (clients join every $1/q$ rounds) makes free-rider deltas span rounds → larger std → *easier* detection. MNIST only, 2-layer FC (~0.2M params), IID + pathological non-IID. Secure aggregation breaks server-side detection.

## Connections

- [[concepts/federated-learning]] — the FedAvg rule $M_{j+1}=M_j-\eta\cdot\text{avg }G$ Flirds attributes over; the delta-weights identity falls directly out of it. [[concepts/data-quality-control]] — free-rider detection is anomaly screening (binary keep/discard), not contribution valuation.
- [[threads/noise-ood-malicious-client-separation]] — STD-DAGMM is *the* free-rider baseline listed there; this source backs that row. [[threads/federated-and-decentralized-attribution]] — robustness-side complement; the canonical FL incentive-gaming threat.
- [[sources/feddqc]] — both tackle FL data integrity without server data access (quality/IRA vs anomaly/fake-update); together they bracket the non-valuation FL-integrity prior art.

## Relevance to Flirds

Origin and dedicated-detector baseline for Flirds' surviving free-rider benchmark. Conceptual contrast worth stating: a free-rider's update is near-zero, random noise, or a recycled average — all (near-)orthogonal to the validation gradient, so Flirds' **first-order term $\langle-\nabla\ell^{val},\Delta w_k\rangle\approx0$** and the client gets Shapley ≈ 0. Flirds therefore demotes free-riders *as a by-product of signed valuation* — no autoencoder/GMM/OOD machinery — which is exactly why this benchmark survives the deferred noise-vs-OOD separator. Methodological contrast: STD-DAGMM is an **unsigned anomaly detector** (energy = "how unusual," threshold-dependent, over-flags non-IID benign clients); Flirds yields a **signed contribution value** (≈0 = no contribution, <0 = harmful), giving free-rider demotion without the non-IID false-positive penalty. STD-DAGMM also assumes no per-client history; Flirds' per-round value likewise needs none — a clean parallel.

## Notes / open questions

- MNIST + 2-layer MLP only; no PEFT/LLM evidence — Flirds' free-rider benchmark would be the first PEFT-scale test of this threat model.
- STD-DAGMM is threshold-dependent and degrades as free-rider ratio rises; a Flirds head-to-head should report AUC/ranking, not just "detected."
- > TODO: confirm whether a delta-weights free-rider yields *exactly* zero vs slightly negative Flirds value (a recycled average may carry stale-but-aligned signal early in training).
- DP/privacy-amplification *aiding* detection: under client subsampling free-rider deltas accumulate → larger magnitude; Flirds' first-order term would correspondingly grow, not vanish — a possible robustness note for the benchmark.
- **Implemented 2026-06-08** (`baselines/std_dagmm.py`, task 7e step 1): model-free DAGMM+std, per-(client,round) pooling + signed feature-hashing 5.6M→256 (std on the FULL vector, reduction-independent). Synthetic AUROC=1.0 (zero caught by the std feature, std-matched random by recon/cosine = the Lin headline). **Real 1B N=100 free-rider AUROC = 0.628** — the first PEFT-scale test confirms the model-free detector is WEAK on the pure-evasion case (a std-matched random direction is not separable from real benign LoRA updates without the gradient), exactly the gap the gradient-using detectors fill (FLTrust/[[sources/fltrust]] ≈ Flirds-1st reach 1.0 on the same threat). See [[flirds-implementation-plan]] §3.9 + [[raw/conversations/flirds/2026-06-08-phase2-task7e-detector-suite-steps1-3]].
