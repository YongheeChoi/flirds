---
type: thread
title: "Separating noise / OOD-good / malicious clients in FL"
created: 2026-05-18
updated: 2026-06-08
sources: [feddqc, ripple-shapley, gtg-shapley, in-run-data-shapley, datainf, fldetector, fedcorr, fltrust, foolsgold, free-riders-fl-std-dagmm, principled-federated-data-valuation, comfedsv, mavericks-shapley-fl, fedtsv, fedif, instructions-as-backdoors-xu, how-to-backdoor-fl-bagdasaryan]
tags: [flirds, federated-learning, robustness, byzantine, noisy-labels, free-rider, ood, limitation]
---

# Separating noise / OOD-good / malicious clients in FL

Prior-art survey (2026-05-18) for two now-distinct Flirds needs:

1. **The characterized limitation.** As of 2026-05-18 [[flirds]] *defers* the noise-vs-OOD-good separator — it is **no longer the paper's second contribution**, but a limitation to characterize honestly (the IRDS-style "here is exactly where our method's sign is ambiguous" move). Writing that section credibly requires knowing what the FL-robustness literature does and why it does not cleanly solve the problem *inside a signed valuation*.
2. **Surviving detection benchmarks.** Noisy-client and free-rider detection benchmarks **survive the deferral** (they need no OOD machinery). They need robustness-side baselines, which this survey identifies.

The three categories to keep mentally separate:

- **Noise** — label/feature corruption; should be down-weighted.
- **OOD-good** — high-quality data from a distribution the validation set under-represents; *should not* be penalized, but Flirds (deferred separator) will under-value it — this is the limitation.
- **Malicious / free-rider** — poisoning or fake/recycled updates; should be removed (surviving benchmark).

## The prior-art landscape (4 clusters)

| Cluster | Representative methods | Detection mechanism | Goal |
|---|---|---|---|
| **Byzantine-robust aggregation** | Krum / Multi-Krum, Trimmed-mean, Median, Bulyan, RFA; [[sources/fltrust|FLTrust]], [[sources/foolsgold|FoolsGold]] | geometric outlier rejection; FLTrust = ReLU-clipped cosine sim of $\Delta w_k$ vs. a server *root-dataset* update; FoolsGold = penalize clients whose updates are *too mutually similar* (sybil) | robust aggregation (discard) |
| **Temporal / historical-consistency** | [[sources/fldetector|FLDetector]] (Zhang et al., KDD 2022) | server predicts each client's next update from its *history* via Cauchy mean-value theorem + L-BFGS Hessian estimate; flags clients whose actual update is **inconsistent across multiple rounds** | malicious detection (binary flag) |
| **Noisy-label FL** | [[sources/fedcorr|FedCorr]] (CVPR 2022), FedDiv, FedNoiL, RHFL, FedSNC | per-client **Local Intrinsic Dimensionality** of the prediction subspace to split clean vs. noisy clients; then per-sample small-loss filter + label correction | label-noise cleanup (classification) |
| **Free-rider detection** | [[sources/free-riders-fl-std-dagmm|Lin et al. 2019 (STD-DAGMM)]], Delta-DAGMM | autoencoder + GMM energy on update parameters; STD statistic catches low-variance fake gradients | free-rider removal |

Centralized ancestry worth citing in the limitation discussion: the **training-dynamics noisy-label** line — forgetting events (Toneva et al.), **Area-Under-Margin** (Pleiss et al. 2020), Confident Learning (Northcutt et al.), the *memorization effect* (clean fit early & stably, noise fit late & erratically). The deferred temporal-consistency idea was the FL/round-level analog of this; the deferred cross-client idea was the **clustered-FL** idea (Sattler et al.: pairwise-cosine clustering separates *distributions* rather than discarding them).

## Why the separator was hard — and why the literature doesn't rescue it

The two deferred signals are **not novel in isolation**: temporal consistency ≈ FLDetector + AUM/forgetting; cross-client agreement ≈ FoolsGold / FLTrust / clustered FL. The deferral is sound precisely because:

> **Non-IID benign clients look identical to attackers under every similarity/distance/consistency detector.** [[sources/fldetector|FLDetector]] itself degrades sharply under heterogeneity (the server cannot predict a benign non-IID client's update — IID-only Theorem 1, Fig. 2 DACC drop); [[sources/foolsgold|FoolsGold]]/[[sources/fltrust|FLTrust]] "unnecessarily penalize many benign clients" (their own reported limitation — FoolsGold's Appendix-B RONI false-positives every honest non-IID client; FLTrust fails at high root class-bias). FedCC / FedCAP / FedDMC patch this only from the *robust-aggregation* side, and only as a binary keep/discard.

This **is** the OOD-good problem under another name, and the literature's verdict is consistent: **no FL method separates "bad-different (noise/poison)" from "good-different (OOD-good)" inside a signed contribution value.** The robustness camp suppresses the non-IID signal (divergence = attack); the personalization/valuation camp preserves it (divergence = legitimate heterogeneity) but never resolves the sign ambiguity ([[sources/gtg-shapley|GTG-Shapley]], [[sources/ripple-shapley|Ripple Shapley]], [[sources/in-run-data-shapley|IRDS]] all compute signed value without addressing noise-vs-OOD-good). ⇒ Deferring it to a *characterized limitation* — rather than claiming a fragile solution — is the defensible choice, and it puts Flirds in good company (IRDS does the same with its own sign-ambiguity limit).

**Direct evidence the valuation itself is biased against OOD-good clients**: [[sources/mavericks-shapley-fl|Huang et al. (Mavericks)]] prove + show empirically that federated Shapley *systematically under-credits* "maverick" clients (those holding a skewed distribution or the bulk of some class) — worst in early rounds via the decaying-LR mechanism. A maverick is exactly an OOD-good client by another name; their fix (FedEMD, Wasserstein-distance client selection) is a *selection* patch, not a sign-disambiguation inside the value. This is the cleanest citation that the under-valuation Flirds defers is a real, quantified property of FL-Shapley, not a hypothetical.

**Newer direction-alignment detectors confirm the blind spot.** [[sources/fedif|FedIF]] (normalized $\Delta w$ · validation gradient) and [[sources/fedtsv|FedTSV]] (coalition-update vs. validation-reference proximity) both separate clients by *alignment with the validation descent direction*. FedIF reports an explicit **PGD blind spot**: a direction-aligned poison is scored as benign — i.e., a 1st-order direction signal cannot tell "good-different" from "bad-but-aligned." Open question whether Flirds' **2nd-order (curvature) term** can split direction-aligned-but-curvature-different updates where these 1st-order detectors cannot (candidate Phase-3 experiment).

## How this feeds the paper now

### A. The limitation section (writing material)

The prior art supplies the exact framing for an honest limitation:

- **State the ambiguity precisely.** A negative $\phi_k$ conflates {noise, poison, OOD-good, non-IID drift residual}. Conversation 4 §4: the deferred separator was *also* the drift-bias corrector — so with it parked, **non-IID clients carry an un-removed valuation bias**. This is now an explicit obligation in [[flirds]]'s next-step checklist (quantify via a non-IID $\alpha$-sweep alongside the $E$-sweep).
- **Cite the landscape as evidence the problem is genuinely hard**, not an oversight: [[sources/fldetector|FLDetector]]'s non-IID failure, [[sources/foolsgold|FoolsGold]]/[[sources/fltrust|FLTrust]] over-penalization, [[sources/fedcorr|FedCorr]]'s late-joiner misclassification, the absence of any noise-vs-OOD-good separator in the valuation line. "We characterize rather than solve, because the methods that would solve detection collapse exactly on the heterogeneity that defines OOD-good."
- **Use the centralized ancestry for the mechanistic explanation** of *why* the sign is ambiguous (memorization effect: noise and OOD-good both diverge from validation but for opposite reasons).

### B. Baselines for the surviving detection benchmarks

Noisy-client + free-rider benchmarks need robustness-side comparators, not just valuation-side:

- [[sources/fldetector|FLDetector]] — temporal-consistency detector; strongest comparator for the noisy/poison benchmark. **→ cross-silo (N=5/10)** (regime split locked 2026-06-07): from-logs closed-form (Cauchy MVT + L-BFGS Hessian prediction-residual), runs at any N; use the continuous score for AUROC.
- [[sources/foolsgold|FoolsGold]], [[sources/fltrust|FLTrust]] — cross-client / trusted-cosine baselines.
- [[sources/fedcorr|FedCorr]] (LID filter) — noisy-client detection baseline.
- [[sources/free-riders-fl-std-dagmm|STD-DAGMM]] / Delta-DAGMM — free-rider baseline. **→ cross-device (N=100, with Phase-2 task 7)** (regime split locked 2026-06-07): trains a DAGMM autoencoder+GMM on the set of client update vectors → needs N≫; **degenerate at cross-silo N=5** (5 vectors, ~12M-dim LoRA updates).
- Valuation baselines from [[flirds]] ([[sources/gtg-shapley|GTG-Shapley]], [[sources/principled-federated-data-valuation|FedSV]], [[sources/comfedsv|ComFedSV]], [[sources/data-banzhaf|Data Banzhaf]] applied to FL, loss-heuristic) — they do *not* attempt noise-vs-OOD separation; useful to show Flirds at least matches detection while being a valuation method.

The contrast to highlight: Flirds is a *valuation* method that, as a free by-product, does noisy/free-rider detection competitively with *dedicated* detectors — without their non-IID false-positive penalty (because it never hard-discards; it just down-weights signed value).

## The poisoning / backdoor threat being detected (attack-side, added 2026-06-08)

The §3.9 poisoning row uses a concrete backdoor = **instruction trigger ([[sources/instructions-as-backdoors-xu|Xu 2024]]) + FL model-replacement scaling ([[sources/how-to-backdoor-fl-bagdasaryan|Bagdasaryan 2020]])**. Xu supplies the trigger→target content (poison the *instruction* only — clean-label); Bagdasaryan supplies the FL delivery (scale the backdoored update by **γ = n/η** so FedAvg adopts it). Matched detectors are already in §B: [[sources/fldetector|FLDetector]] (the scaled/crafted update is its home threat) and [[sources/fltrust|FLTrust]] (a scaled update is least validation-aligned).

**Reproduction caveat (2026-06-08).** Model replacement only works if the attacker's *local* model already holds the backdoor; installing one at LoRA-FL scale (generative, weak token trigger, SGD-mom0, 1B) is the open §3.9 sub-task — γ scaling alone just replaces a model that never learned the trigger, and over-scaling without Bagdasaryan's stabilizers (near-convergence timing, lower attacker lr, norm-bound) destroys the global model. Faithful-repro recipe in the two source pages' Flirds-implementation notes.

## If the separator is ever revived

Parked, not deleted. If revisited, the design steers from the survey still hold and stay inside Flirds' locked constraints:

- Reuse the already-computed validation HVP $H^{(val)}\Delta W^{(r)}$ for a one-step-ahead **prediction-residual** temporal term (cheaper than FLDetector's L-BFGS full-model Hessian; zero extra communication).
- Make the cross-client term a **soft sign-disambiguation** (spectral/soft-cluster membership on the round's $\{\Delta w_k\}$ graph), never a hard partition or discard.
- **FLDetector remains the primary scooping risk** if revived — differentiate on goal (signed valuation vs. binary detection), signal source (validation HVP vs. L-BFGS), and the explicit noise-vs-OOD split it never attempts.

## External references (candidates to ingest)

**Ingested 2026-05-19** (the five robustness-side baselines now have source pages):

- [[sources/fldetector|FLDetector]] — Zhang et al., KDD 2022. [arXiv:2207.09209](https://arxiv.org/abs/2207.09209)
- [[sources/fedcorr|FedCorr]] — Xu et al., CVPR 2022. [arXiv:2204.04677](https://arxiv.org/abs/2204.04677)
- [[sources/fltrust|FLTrust]] — Cao et al., NDSS 2021. [arXiv:2012.13995](https://arxiv.org/abs/2012.13995)
- [[sources/foolsgold|FoolsGold]] — Fung et al., RAID 2020 (arXiv title *Mitigating Sybils in FL Poisoning*). [arXiv:1808.04866](https://arxiv.org/abs/1808.04866)
- [[sources/free-riders-fl-std-dagmm|Free-riders in FL (STD-DAGMM)]] — Lin et al. 2019. [arXiv:1911.12560](https://arxiv.org/abs/1911.12560)

Still not in `raw/` (ingest candidates):

- Clustered FL — Sattler et al., NeurIPS 2019 workshop / TNNLS 2020.
- AUM — Pleiss et al., *Identifying Mislabeled Data using the Area Under the Margin Ranking*, NeurIPS 2020 (centralized ancestry).

## See also

- [[flirds]] — open question #1 (now *deferred*: this thread is its prior-art backing for the limitation recast).
- [[threads/data-quality-vs-data-value]] — the quality/value distinction; noise-vs-OOD-good is the sharp case where they diverge.
- [[threads/federated-and-decentralized-attribution]] — the valuation-side FL landscape (this thread is the robustness-side complement).
- [[sources/feddqc|FedDQC]] — documents that gradient attribution becomes brittle on heterogeneous FL; the same heterogeneity is the OOD-good signal Flirds cannot cleanly separate.
- [[threads/robustness-to-stochastic-training]] — noise-robustness as an evaluation axis (Banzhaf safety margin); about SGD noise, not client noise.
