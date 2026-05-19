---
type: source
title: "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping"
created: 2026-05-19
updated: 2026-05-19
topic: flirds
tags: [federated-learning, byzantine-robust, poisoning-defense, trusted-cosine, server-side-clean-set]
---

# FLTrust

## Citation

Xiaoyu Cao, Minghong Fang, Jia Liu, Neil Zhenqiang Gong (Duke / Ohio State). *FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping*. NDSS 2021. arXiv:2012.13995v3, Apr 2022.

Raw: `raw/papers/flirds/2012.13995v3.pdf`

## TL;DR

The server holds a small manually-collected clean "root" dataset and trains a server model on it as a root of trust. Each round, every client update is scored by ReLU-clipped cosine similarity to the server update, magnitude-normalized to the server update's norm, and combined as a trust-weighted average. With <100 root examples it tolerates 40–90% malicious clients across attacks while matching attack-free FedAvg accuracy.

## Problem

Byzantine-robust aggregators (Krum, Trimmed-mean, Median) compare client updates and discard statistical outliers but have *no root of trust* — every client could be malicious — so crafted local-model-poisoning attacks still corrupt the model, worse under non-IID or a large malicious fraction. FedAvg can be broken by a single malicious client.

## Method

Each round the server also fine-tunes the global model on its root dataset $D_0$ to get a **server update** $g_0$ (same SGD routine). For client update $g_i$:

- **Trust score** $TS_i=\mathrm{ReLU}(c_i)$, $c_i=\frac{\langle g_i,g_0\rangle}{\lVert g_i\rVert\lVert g_0\rVert}$ — directions opposing the server's are zeroed.
- **Magnitude normalization** $\bar g_i=\frac{\lVert g_0\rVert}{\lVert g_i\rVert}g_i$ — every accepted update rescaled onto the server update's hypersphere (neutralizes scaled poison, enlarges suspiciously small updates).
- **Aggregate** $g=\frac{1}{\sum_j TS_j}\sum_i TS_i\,\bar g_i$, then $w\leftarrow w+\alpha g$.

Hard, direction-gated trust weighting (benign-but-misaligned updates can be zeroed). Theorem 1 bounds $\lVert w_t-w^*\rVert$ under arbitrary malicious counts (strongly-convex; bound tightens with larger $|D_0|$).

## Key results

- Six datasets (MNIST IID/non-IID, Fashion-MNIST, CIFAR-10, CH-MNIST, HAR). Robust to label-flipping, Krum, Trim, Scaling/backdoor, and an adaptive attack — under attack stays within ≤0.04 test error of attack-free FedAvg, backdoor success ≤0.03, while Krum/Trim-mean/Median collapse.
- **Root size**: ~100 examples suffices; <50 degrades. Tolerates up to 90% malicious (95% backdoor) on MNIST-0.5.
- **Root distribution**: robust while root-vs-overall bias ≤~0.4; fails when the root is strongly class-biased. ReLU and normalization each ablated as necessary.

## Connections

- [[concepts/federated-learning]] — replaces only the FedAvg aggregation rule. [[concepts/data-quality-control]] — direction-agreement-to-trusted-reference as a quality gate, distinct from valuation.
- [[threads/noise-ood-malicious-client-separation]] — the trusted-cosine member of the FL robustness prior art; backs the deferred-limitation recast and the surviving poisoning benchmarks.
- [[threads/federated-and-decentralized-attribution]] — contrast: robust aggregation vs contribution valuation; the server-side trusted-set design choice.
- [[sources/feddqc]] — both gate low-quality client data, but FedDQC scores on-device (IRA) whereas FLTrust scores server-side (update-direction agreement). [[sources/distributionally-robust-data-valuation]] — DRGE removes exactly the validation-set dependence FLTrust (root) and Flirds (server validation) accept.

## Relevance to Flirds

FLTrust operationalizes the **cross-client / trusted-direction agreement** signal Flirds parked: a poisoned/noisy update points away from consensus (the server-update direction); good updates align. Natural robust-aggregation baseline for Flirds' surviving detection/poisoning benchmarks. Strong structural parallel: FLTrust's server-side clean **root dataset** mirrors Flirds' **server-side validation set** — both place a small trusted dataset on the server with zero extra client communication (extra cost is server-only), and both inherit the same privacy/abuse framing and the *poisoned-root* vulnerability. **Contrast**: FLTrust produces hard trust weights to steer aggregation — it does not assign client value/credit; Flirds is a valuation method (closed-form Taylor Shapley on $\Delta w_k$). Cosine-to-server is a coarse direction filter; Flirds' second-order term is a finer signed marginal-contribution estimate. Note three distinct uses of the same server artifact: filtering (Fang et al.), trust weighting (FLTrust), valuation (Flirds).

## Notes / open questions

- FLTrust assumes a *clean* root and concedes no robustness to a poisoned root — Flirds inherits the analogous risk if its server validation set is compromised.
- Root-distribution sensitivity (fails at high class-bias) suggests Flirds' validation-set representativeness similarly bounds the cross-client signal under heavy non-IID; quantify the analogue.
- Cosine-to-server is binary-ish (ReLU kills the whole update); candidate ablation: does Flirds' continuous 2nd-order score separate "good-but-different same-distribution" clients from noise better than this hard gate?
- Theory assumes strong convexity and one local step — the same single-step idealization Flirds' round-delta-as-1-step caveat already flags (conversation 2/3).
- > TODO: FLTrust's zeroth-order adaptive attack against a known aggregation rule is the right adaptive threat model if Flirds ever claims detection robustness.
