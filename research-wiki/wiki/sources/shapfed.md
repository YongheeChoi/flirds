---
type: source
title: "ShapFed: Redefining Contributions — Shapley-Driven Federated Learning"
created: 2026-06-03
updated: 2026-06-03
topic: flirds
tags: [federated-learning, shapley, class-specific, cosine-gradient, weighted-aggregation, personalization, cnn]
---

# ShapFed

## Citation

Nurbek Tastan, Samar Fares, Toluwani Aremu, Samuel Horvath, Karthik Nandakumar (MBZUAI). *Redefining Contributions: Shapley-Driven Federated Learning*. IJCAI 2024 (DOI 10.24963/ijcai.2024/554). arXiv:2406.00569v1, 1 Jun 2024. Code: `github.com/tnurbek/shapfed`.

Raw: `raw/papers/flirds/2406.00569_ShapFed.pdf`

## TL;DR

A **class-specific** federated Shapley method. Instead of a scalar per-client value from a server validation set, ShapFed computes a **Class-Specific Shapley Value (CSSV)** $\Gamma_i \in [-1,1]^M$ from the **cosine similarity of each client's last-layer per-class parameter column with the aggregated column** — no validation set, no coalition enumeration. CSSVs then drive **ShapFed-WA** (weighted aggregation, beats FedAvg under class imbalance) and **personalized** client models. Demonstrated on CIFAR-10 / Chest X-Ray / Fed-ISIC2019; reported to match or surpass **AFedSV** (the adaptive-ShapleyFL robust-aggregation baseline) in vulnerable settings.

## Problem

FL contribution assessment usually (i) needs a server **validation set** (privacy + non-IID-fairness problems) and (ii) returns a **scalar** per client, missing *class-specific* influence. Exact Shapley needs $2^n-1$ utility calls; gradient-cosine surrogates (CGSV, Xu et al. 2021) cut that cost but stay scalar. ShapFed wants a fine-grained, validation-free, cheap contribution signal that also improves aggregation under heterogeneity.

## Method

### CSSV — class-specific contribution (§4.1)

View the classifier $M_w$ as feature extractor + a linear head $\hat w \in \mathbb{R}^{P\times M}$ ($M$ classes). Split $\hat w$ into $M$ column vectors (one per class). For client $i$ with last-layer update $\hat w_i$ and the server aggregate $\hat w_s$:

$$\Gamma_i := \big[\cos(\hat w_{i,1}, \hat w_{s,1}),\, \cos(\hat w_{i,2}, \hat w_{s,2}),\, \dots,\, \cos(\hat w_{i,M}, \hat w_{s,M})\big]$$

The $M\times n$ matrix $\Gamma$ encodes per-class, per-client contribution + data heterogeneity (a client holding only classes 1–2 lights up those columns). Updated each round with momentum $\mu$. **No validation set, no coalition enumeration** — directional alignment of the last layer is the utility proxy (extends CGSV from scalar cosine to per-class cosine).

### ShapFed-WA + personalization

- **Weighted aggregation**: $w_s^{t+1} = \sum_i \tilde\gamma_i w_{i,K}^t$ with $\gamma_i$ derived from the aggregated CSSV — higher-contribution clients weigh more (helps class-imbalanced FL).
- **Personalization**: each client receives $\bar w_i = \gamma_i w_s + (1-\gamma_i) w_i$ — a contribution-commensurate blend of global and local models.

## Key results

- **Datasets**: CIFAR-10, Chest X-Ray, Fed-ISIC2019 (cross-silo, classification).
- **Approximation fidelity** (Fig. 3, synthetic): CSSV tracks the true validation-accuracy Shapley across coalitions while avoiding the $2^n-1$ calls.
- **Utility**: ShapFed-WA outperforms FedAvg, **especially under class imbalance**.
- **Fairness/robustness**: matches or **surpasses AFedSV** (adaptive-ShapleyFL robust aggregation) in most "vulnerable" (adversarial/heterogeneous) settings.
- **Efficiency**: last-layer-only cosine → cheap vs. validation-set or coalition-retraining Shapley.

## Connections

- [[concepts/federated-shapley]] — a **class-specific**, validation-free member of the family; uses last-layer per-class cosine as the utility proxy.
- [[sources/game-of-gradients-sfedavg|CGSV / cosine-gradient Shapley]] — direct ancestor (Xu et al. 2021): ShapFed lifts scalar cosine-gradient SV to a per-class vector.
- [[sources/shapleyfl|ShapleyFL]] — the **"AFedSV" baseline** ShapFed compares against *is* the adaptive surrogate-Shapley aggregation of ShapleyFL (Sun et al., KDD 2023); ShapFed reports surpassing it. Resolves the "AFedSV" naming confusion (it is not a standalone paper).
- [[sources/gtg-shapley|GTG-Shapley]] — cited as the gradient-reconstruction efficiency line; ShapFed avoids coalition reconstruction entirely via last-layer cosine.
- [[sources/principled-federated-data-valuation|FedSV]] — the sampling/group-testing SV-approximation line ShapFed positions against.
- [[sources/fedtsv|FedTSV]], [[sources/fedif|FedIF]] — same **valuation→aggregation** quadrant (contribution score steers weighting), all 2024–2026, all CNN classification; ShapFed is the class-specific/last-layer-cosine variant.
- [[threads/federated-and-decentralized-attribution]] — aggregation-side, validation-free, class-specific branch.
- [[threads/noise-ood-malicious-client-separation]] — CSSV's per-class heterogeneity readout is a class-level take on the same separation problem.

## Relevance to Flirds

**Scoop risk: LOW–MEDIUM.** A strong, recent (IJCAI'24), code-available FL-Shapley method — the genuine SV-side comparator hub that AFedSV / FedIF / Ripple all orbit — but a different object from Flirds on every core axis.

- **Class-specific last-layer cosine, not validation-loss Taylor.** ShapFed's utility is the cosine alignment of the **last linear layer's per-class columns**; Flirds is a **1st+2nd-order Taylor expansion of the validation-loss change** over the full LoRA update. No gradient/Hessian of the loss, no 2nd-order interaction term, no closed-form Shapley over a realized trajectory.
- **Notable last-layer overlap.** ShapFed *deliberately uses only the last layer*, the exact layer Flirds' ② cancellation characterization scrutinizes (last-layer gradient-norm dominance under non-IID). ShapFed treats last-layer-per-class as a *feature*; Flirds treats last-layer dominance as a *limitation to characterize*. Worth contrasting directly in the ② write-up.
- **Aggregation + personalization, not post-hoc valuation.** ShapFed-WA changes aggregation and personalizes models; Flirds reports credit off vanilla FedAvg without altering it. (Same valuation-vs-aggregation split as [[sources/fedtsv|FedTSV]]/[[sources/fedif|FedIF]].)
- **CNN classification, not LLM/LoRA generative.** CSSV's per-class column structure assumes a discrete $M$-class linear head — like [[sources/space-participant-amalgamation|SPACE]]'s prototype evaluation, it does **not transfer cleanly to LLM generative instruction tuning**. So it is a *conceptual* comparator and a CNN-track baseline, not a drop-in LLM baseline.

Use: (i) a recent CNN-track FL-Shapley baseline (CSSV + ShapFed-WA), (ii) the citation that resolves the "AFedSV" label, (iii) a direct contrast for the ② last-layer-cancellation characterization (ShapFed-as-feature vs. Flirds-as-limitation).

## Notes / open questions

- **AFedSV identity**: in both ShapFed and [[sources/fedif|FedIF]], "AFedSV" denotes the *adaptive surrogate-federated-Shapley* robust aggregation traceable to [[sources/shapleyfl|ShapleyFL]] (Sun et al., KDD 2023) — already in the wiki. No separate "AFedSV" paper to ingest; annotate the alias on [[sources/shapleyfl]].
- **Last-layer-only signal** discards all but the classifier head — cheap but blind to feature-extractor contribution. Flirds' full-LoRA Taylor does not throw away the body; a useful ablation contrast.
- **Generative transfer**: can a CSSV-like per-class signal be defined for LLM instruction tuning (no fixed class head)? Likely not directly — reinforces that the LLM FL-valuation niche is sparsely populated.
- Code is public (`tnurbek/shapfed`) — usable if a CNN-track class-specific baseline is wanted.
