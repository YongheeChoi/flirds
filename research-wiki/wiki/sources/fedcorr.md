---
type: source
title: "FedCorr: Multi-Stage Federated Learning for Label Noise Correction"
created: 2026-05-19
updated: 2026-05-19
topic: flirds
tags: [federated-learning, label-noise, robustness, lid, noisy-client-detection, correction]
---

# FedCorr

## Citation

Jingyi Xu, Zihan Chen (equal), Tony Q. S. Quek, Kai Fong Ernest Chong (SUTD / NUS). *FedCorr: Multi-Stage Federated Learning for Label Noise Correction*. CVPR 2022. arXiv:2204.04677v1.

Raw: `raw/papers/flirds/2204.04677v1.pdf`

## TL;DR

A general three-stage FL framework that jointly handles heterogeneous label noise *and* non-IID data with no local-noise-model assumption. Each client uploads one extra scalar — the LID (local intrinsic dimensionality) of its prediction subspace — letting the server flag noisy clients via a GMM; per-sample loss then flags noisy samples, relabeled by the global model. Beats robust-FL and per-client centralized baselines on CIFAR-10/100 and Clothing1M.

## Problem

Real FL clients differ simultaneously in data statistics (non-IID, imbalanced) and label quality (varying noise rates). Naive FedAvg assumes correct labels; centralized noise-robust methods fail per-client (small local sets); discarding dissimilar clients as malicious wastes data that is useful after correction; privacy forbids raw-feature exchange.

## Method

Three stages (Algorithm 1):

1. **Federated pre-processing**: $T_1$ iterations, each client participates once at a *small* fraction without replacement (fraction scheduling — highest-impact component per ablation). Local loss adds an adaptive proximal term scaled by estimated noise (vanishes for clean clients) + mixup. Each client uploads weights + one LID scalar. Noisy datasets yield larger prediction-subspace LID; the server fits a GMM on **cumulative** LID (sum over iterations — single-round LID overlaps clean/noisy after ~4 iters) to split clean/noisy clients. Within each noisy client, a second GMM on per-sample losses splits clean/noisy samples; large-loss samples relabeled by global-model predictions, gated by relabel ratio π and confidence θ.
2. **Finetuning**: FedAvg on relatively-clean clients (noise < κ), then relabel remaining noisy clients with the finetuned model.
3. **Usual FL**: standard FedAvg on all clients with corrected labels.

Privacy: only an output-layer-derived scalar shared (negligible comm). Non-IID handled directly (Bernoulli class-presence × Dirichlet partition, evaluated explicitly).

## Key results

- CIFAR-10 IID: best accuracy at every noise level (90.6% at ρ=0.8,τ=0.5 vs FedAvg 72.0%, FedProx 80.6%, RoFL 74.1%, ARFL 53.2%); within <4% of centralized DivideMix.
- CIFAR-100 IID and CIFAR-10 non-IID: ≥7% gain over all baselines. Clothing1M (real noise, non-IID): highest FL accuracy, beating reported centralized JointOpt.
- 1.3–1.9× better communication efficiency; modular (boosts FedDyn/Median/PoC).

## Connections

- [[concepts/federated-learning]] — non-IID + robust-FL framework extending FedAvg with a noise-correction pipeline.
- [[concepts/data-quality-control]] — quality assessment (LID + per-sample loss) feeding correction, distinct from valuation.
- [[threads/noise-ood-malicious-client-separation]] — the LID-based noisy-client correction prior art; contrasts discard-based detectors and Flirds' signed-value down-weighting.
- [[threads/federated-and-decentralized-attribution]] — robustness-side counterpart to attribution-based client weighting.
- [[threads/data-quality-vs-data-value]] — FedCorr scores quality and *fixes* labels; it does not value contribution — a sharp quality/value example.
- [[sources/feddqc]] — sibling FL data-quality work; FedDQC scores instruction-data quality on-device, FedCorr scores label noise via LID then relabels. Both privacy-preserving, quality-not-value.

## Relevance to Flirds

Dedicated noisy-client-detector baseline for Flirds' surviving noisy-client (label-corruption) benchmark. Mechanism contrast is the framing point: FedCorr needs a *separate detection apparatus* — an extra LID scalar per round, a two-component GMM presuming a clean/noisy split exists, multiple pre-processing iterations, explicit non-IID partitioning — whereas Flirds reads client-level signed Shapley already implied by $\Delta w_k$ at zero extra comm. FedCorr is **not a strawman** (no noise-model assumption, handles non-IID by design), but it **hard-routes** clients (clean vs noisy set) then **relabels** rather than down-weights. Flirds' claim should be: signed-value down-weighting matches detection without the pipeline's overhead or the binary partition. Caveat for fair comparison: FedCorr *corrects* (salvages corrupted clients), so its accuracy partly reflects data recovery, not just filtering — separate "detected the noisy client" from "boosted accuracy via correction."

## Notes / open questions

- LID-as-noise-signal is correlational (no theoretical guarantee), and the clean/noisy LID overlap grows during training — only *cumulative* LID stays separable.
- Authors' acknowledged limitation: no dynamic participation — late-joining clients have artificially low cumulative LID and get misclassified as clean. Relevant if Flirds' benchmark assumes static participation.
- Clean-client false-positive cost is only qualitative (Fig. 7); to substantiate Flirds' "no non-IID false-positive penalty" framing, a quantitative clean-client FP rate under non-IID should be derived, not asserted.
- Symmetric synthetic noise only; structured/adversarial label noise untested. Authors flag a freeloader concern: correction lets a zero-effort random-label client free-ride — a valuation-side argument linking to incentive/contribution-fairness threads.
