---
type: source
title: "FoolsGold: Mitigating Sybils in Federated Learning Poisoning"
created: 2026-05-19
updated: 2026-05-19
topic: flirds
tags: [federated-learning, sybil, poisoning, robustness, cosine-similarity, non-iid-limitation]
---

# FoolsGold

## Citation

Clement Fung, Chris J. M. Yoon, Ivan Beschastnikh (UBC). arXiv:1808.04866v5, *Mitigating Sybils in Federated Learning Poisoning*, Jul 2020. Peer-reviewed version: **RAID 2020**, retitled *The Limitations of Federated Learning in Sybil Settings*. System name: **FoolsGold**.

Raw: `raw/papers/flirds/1808.04866v5.pdf`

## TL;DR

FL gives every client equal influence, so an adversary multiplies poisoning power by spawning sybils. FoolsGold defends without bounding attacker count: it down-weights clients whose *aggregated historical update directions* are abnormally mutually similar, on the premise that colluding sybils share one objective while honest (assumed non-IID) clients do not. Beats Multi-Krum and RONI on label-flipping and backdoor — but its core assumption is its core limitation.

## Problem

FedAvg's only weight is example count, trivially inflated by sybils. With ≥2 sybils (poisoned 1→7 MNIST) vs 1 honest client owning digit 1, attack success hits 96.2%. Multi-Krum needs an explicit adversary-fraction bound (fails >33%); centralized defenses need data/validation access the FL server lacks. Insight: sybils driving the model toward one poisoned objective produce updates *more similar to each other* than honest clients' updates are.

## Method

Per-iteration, server-side, no extra communication:

1. **Update history** — per-client aggregated historical gradient $H_i=\sum_t\Delta_{i,t}$ (defeats per-iteration SGD-variance evasion).
2. **Cosine similarity** between pairs $H_i,H_j$, restricted to **indicative features** (output-layer params by magnitude) so attackers can't add dissimilarity via irrelevant features.
3. $v_i=\max_j cs_{ij}$ — max pairwise similarity = "acting as a sybil" score (robust to sybil count).
4. **Pardoning**: if $v_j>v_i$, rescale $cs_{ij}\!\mathrel{*}=v_i/v_j$ so an honest client similar to a sybil isn't punished.
5. **Adaptive learning rate** $\alpha_i=1-\max_j cs_i$ (rescaled so the most-unique client keeps LR 1), then a **logit** push so adding sybils can't dilute the penalty. Aggregate $w_{t+1}=w_t+\sum_i\alpha_i\Delta_{i,t}$.

Crucial assumption: **honest clients are non-IID** (each has a unique distribution). No attacker-count parameter; model/batch-size agnostic. Best paired with Multi-Krum for the single-attacker case (no collusion for FoolsGold to detect).

## Key results

- Robust across MNIST/VGGFace2/KDDCup/Amazon, label-flip + single-pixel backdoor, FedSGD + FedAvg, up to **A-99 (990 sybils vs 10 honest)**; attack rate ≈0 where Multi-Krum collapses past 33%. One failure: Amazon batch-size-1 (4.76%, curse of dimensionality). Ablations confirm history/pardoning/logit each defeat a specific evasion.
- **Central limitation** (§6.3–6.4, §7, Appendix B-RONI, and the RAID title): relies on training data being "sufficiently dissimilar between clients." When honest clients share data (IID/overlapping) they look mutually similar — i.e. like sybils — causing **false positives**. Multi-Krum/RONI fail for the same reason: high benign update variance under non-IID is indistinguishable from adversarial behavior.

## Connections

- [[concepts/federated-learning]] — exploits FL's equal-influence flaw; pure server-side defense, no comm overhead (same constraint class as Flirds).
- [[concepts/data-quality-control]] — a detect-and-suppress robustness method; the contrast class to signed valuation.
- [[threads/noise-ood-malicious-client-separation]] — *the* home thread: FoolsGold = the cross-client-agreement prior art; its IID false-positive failure backs the deferral.
- [[threads/federated-and-decentralized-attribution]] — robustness-side complement to the valuation-side FL landscape.
- [[sources/feddqc]] — independently documents heterogeneous-FL gradient signals as brittle; FoolsGold shows the same heterogeneity breaking a similarity detector from the robustness side.

## Relevance to Flirds

The canonical instantiation of the **cross-client-agreement signal Flirds parked** ([[threads/noise-ood-malicious-client-separation]]). Its non-IID false-positive failure is **direct primary evidence** for Flirds' framing that no method cleanly separates "bad-different" from "good-different" inside a signed value: FoolsGold resolves the ambiguity by fiat (divergence ⇒ honest, similarity ⇒ sybil), which inverts catastrophically once honest clients are IID. It also concretely supports "hard-discarding detectors carry a non-IID penalty": FoolsGold *drives* $\alpha_i\to0$, whereas Flirds only down-weights signed value and never hard-discards, so heterogeneity costs it bias-not-erasure. Use as a robustness-side **baseline** for the surviving noisy-client / free-rider benchmarks.

## Notes / open questions

- Appendix-B RONI is a clean externally-runnable demo of the OOD-good problem: an IID validation set false-positives every honest non-IID client except the one owning the target digit — quote-worthy as "validation-anchored detectors flag non-IID benign clients."
- FoolsGold uses *max* pairwise similarity (one collusion edge suffices). Flirds' parked cross-client term, if revived, should be a **soft** spectral/cluster membership, never this hard max.
- The convergence proof assumes honest pairwise gradient similarity is low — it *assumes away* the non-IID-collision case rather than handling it. Cite as "the guarantee holds only where the limitation doesn't bite."
- > TODO: the RAID 2020 camera-ready may sharpen the limitation framing beyond this arXiv v5; ingest if a copy appears.
