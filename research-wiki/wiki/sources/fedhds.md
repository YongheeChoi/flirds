---
type: source
title: "Federated Data-Efficient Instruction Tuning for Large Language Models (FedHDS)"
created: 2026-06-03
updated: 2026-06-03
topic: flirds
tags: [federated-learning, llm, instruction-tuning, coreset, data-selection, lora, non-iid]
---

# FedHDS (Qin et al.)

## Citation

Zhen Qin, Zhaomin Wu, Bingsheng He, Shuiguang Deng (Zhejiang University; National University of Singapore). *Federated Data-Efficient Instruction Tuning for Large Language Models*. arXiv:2410.10926v2, 27 Jun 2025. Reported as ACL 2025 Findings. (venue verify — not stated in the extracted text). Code: github.com/zhenqincn/FedHDS.

Raw: `raw/papers/flirds/2410.10926_FedHDS.pdf`

## TL;DR

**FedHDS** = federated **H**ierarchical **D**ata **S**election: a privacy-preserving coreset method for federated LLM instruction tuning. Each client clusters its local data (intra-client redundancy), sends *approximate cluster centroids* (2-D, no raw data) to the server, which clusters those centroids to remove *inter-client* redundancy; only the representative samples are used for LoRA tuning. Improves Rouge-L on unseen tasks by **+10.72% over full-data federated tuning while using <1.5% of the data**, with up to ~48× training speedup.

## Problem

Federated LLM instruction tuning (FedIT etc.) typically **consumes all local data**, causing (1) huge compute on edge devices (CPU+GPU hybrid, batch size 1) and (2) overfitting to clients' narrow domains, hurting generalization to unseen tasks. Centralized data-efficient/coreset methods don't transfer: they need simultaneous access to all data (violates FL, can't see inter-client redundancy) and cluster on **last-Transformer-layer features only**, which is a sub-optimal feature space.

## Method

- **Setup**: standard federated instruction tuning, FedAvg + **LoRA adapters only** transmitted; cross-device (5% clients sampled per round); local SGD on a *selected subset* $\widetilde D_i$ instead of full $D_i$ (Eq. 11). Data-efficient iff $\sum|\widetilde D_i|/\sum|D_i|\ll1$ (Eq. 4).
- **Intra-client selection**: extract hidden states for each sample (last token), **fuse features across all Transformer layers** via t-SNE (Barnes-Hut) to dim $k{=}2$ — motivated by a toy study showing the last layer is not universally best for clustering (Calinski-Harabasz / F1). Cluster fused features with **HDBSCAN** into groups; each group has an approximate centroid (not a real sample). Send centroids to server.
- **Inter-client selection**: server runs HDBSCAN on all clients' centroids, picks one representative per inter-cluster, notifies clients which of their centroids were selected. Each client keeps the real sample closest to each selected centroid → coreset $\widetilde D_i$.
- **FedHDS-Turbo**: extract features with a small GPT-2 (~124M) proxy instead of the target LLM — large speedup, comparable accuracy.
- **Privacy**: centroids are 2-D and sample-free; optional Gaussian-noise DP (Theorem 1) on scaled centroids.
- Convergence inherits from FedAvg (it is FedAvg on a downsampled set).

## Key results

- **Accuracy** (Table 2; NI + Dolly-15K, DataJuicer-1.3B & LLaMA-3B, non-IID): FedHDS / FedHDS-Turbo beat full-data baselines (FedIT, FlexLoRA, FedPTuning/Prompt) in all 6 scenarios using **<1.5% of samples**; +19.5% over FedIT on NI/DataJuicer; +10.72% average. Beat coreset baselines (Random, Perplexity) in 5/6 and *auto-select* the data ratio (Random must be hand-tuned).
- **Efficiency** (Table 3): up to **48.8× speedup** over FedIT (FedHDS-Turbo, NI/DataJuicer); FedHDS ~17–20×.
- **Overfitting** (Figs. 7–8): full-data FedIT test loss plateaus/rises while train loss keeps dropping; FedHDS test loss keeps decreasing → subset training mitigates local overfitting.
- **Ablations**: both intra- and inter-client selection contribute; **hierarchical** selection beats sending all features for global clustering; all-layer fusion (t-SNE) beats last-layer and beats PCA/KPCA on the harder dataset (NI).
- Communication overhead beyond LoRA: a few dozen bytes (centroids + indices). Negligible.

## Connections

- [[concepts/federated-learning]], [[concepts/lora]] — FedAvg + LoRA federated LLM tuning is exactly Flirds' Phase-1 substrate.
- [[concepts/dataset-valuation]], [[concepts/data-quality-control]] — coreset *selection* by representativeness, adjacent to (but distinct from) *valuation*.
- [[sources/less]], [[sources/mates]] — centralized data-selection-for-LLM-tuning (gradient/influence-driven); FedHDS is the *federated, clustering-driven* counterpart.
- [[sources/feddqc]] — federated data *quality* control; complementary axis (FedHDS does representativeness, not quality — see its own Limitations).
- [[threads/data-selection-for-llms]] — primary thread: federated coreset selection, dedup-based, privacy-preserving.
- [[threads/data-quality-vs-data-value]] — explicit gap: FedHDS selects for *representativeness*, "overlooks data quality" (its stated limitation) → valuation-driven selection (Flirds) is the missing piece.
- [[threads/noise-ood-malicious-client-separation]] — its limitation note (low-quality data can look like a separate domain and get selected) is a concrete failure that a valuation signal could catch.

## Relevance to Flirds

**Dual use: experimental-setup component + downstream-application/baseline.**

1. **Already part of Flirds' setup.** FedHDS's benchmark *is* Flirds' cross-device LLM track: FedAvg + LoRA, Natural Instructions (738 train tasks, one per client → feature-skew non-IID; 119 held-out test tasks) and Dolly-15K (Dirichlet $\alpha\in\{0.5,5.0\}$, 200 clients), DataJuicer-1.3B / LLaMA-3B, Rouge-L on unseen tasks, 5% client sampling. Reusing this gives Flirds a credible, published cross-device data benchmark and comparable numbers.
2. **Mechanism contrast (the research positioning).** FedHDS selects data by **dedup/representativeness clustering** (HDBSCAN on fused features) with *no notion of contribution to validation loss*. Flirds computes **client-level Shapley** (closed-form 1st+2nd Taylor of validation-loss change from $\Delta w_k$) — a *valuation-driven* signal. FedHDS's own stated limitation — "it only selects data based on representativeness but overlooks data quality" and may select low-quality data treated as a separate domain — is precisely the gap a valuation signal addresses. This makes FedHDS both a **baseline** (selection target to beat/augment) and a motivation for valuation-guided selection.
3. **Granularity note**: FedHDS is *data-sample* selection within clients; Flirds is *client-level* valuation. They compose — Flirds could value clients, FedHDS-style clustering could trim within a valued client — rather than compete head-to-head on the same axis.

## Notes / open questions

- 5% client participation per round (cross-device) — directly relevant to the Flirds code TODO that the estimator/(b)-oracle currently assume *full* participation (read N/p from round-0 deltas). FedHDS's setup is the concrete scenario that breaks that assumption. > TODO: confirm Flirds' partial-participation handling before running on the FedHDS benchmark.
- FedHDS uses **Adam** (FedIT) for local training; Flirds' convention is **plain SGD** (IRDS/Ripple per-step assumption). Mismatch to reconcile when sharing the benchmark — does the in-run SV oracle / Taylor estimator assume SGD locally? > TODO: check whether the benchmark can be run under SGD without large accuracy loss (FedIT-SGD numbers exist in Table 2 as a reference).
- Venue: text says arXiv v2 (Jun 2025); seed lists ACL 2025 Findings — not confirmed in the extracted body. (verify)
- LoRA layers only are transmitted/trained → no BatchNorm/buffer issue (LayerNorm + LoRA), consistent with the memory note that Flirds' BN concern is CNN-only.
