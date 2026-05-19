---
type: source
title: "FedDQC: Data Quality Control in Federated Instruction-tuning of Large Language Models"
created: 2026-05-05
updated: 2026-05-05
topic: flirds
tags: [federated-learning, llm-instruction-tuning, data-quality, lora, ira, hierarchical-training]
---

# FedDQC

## Citation

Yaxin Du (SJTU), Rui Ye (SJTU + Shanghai AI Lab), Fengting Yuchi (SJTU), Wanru Zhao (Cambridge), Jingjing Qu (Shanghai AI Lab), Yanfeng Wang (SJTU), Siheng Chen (SJTU). *FedDQC: Data Quality Control in Federated Instruction-tuning of Large Language Models*. arXiv:2410.11540 (v2).

Raw: `raw/papers/flirds/FedDQC_ Data Quality Control in Federated Instruction-tuning of Large Language Models.md`

## TL;DR

A federated instruction-tuning framework that addresses data quality without server-side access to client data. Two innovations: **IRA (Instruction–Response Alignment)**, a privacy-preserving quality metric computed on-device using only the global model, equal to the inference-loss difference between unconditional and instruction-conditional response generation; and a **hierarchical training scheme** that fine-tunes the LLM from high-IRA to low-IRA data, mirroring easy-to-hard learning. On LLaMA-2-7B + LoRA, FedDQC outperforms data-quality baselines (PPL, IFD, NUGGETS, [[sources/datainf|DataInf]]) on synthetic and real-world (Fed-WildChat) FL data.

## Problem

Federated learning preserves privacy but **exacerbates data quality issues**: each client only sees its local data, so noisy/low-quality samples can't be filtered globally. For LLM instruction tuning specifically:

- Heuristic data-quality methods (PPL, IFD, NUGGETS) work on aggregated data but assume noiseless inputs and centralized access.
- Data-attribution methods (Shapley, influence functions, [[sources/datainf|DataInf]]) require retraining or expensive gradient computations — infeasible for resource-constrained clients.
- The classification-focused FL data-quality literature (label noise correction, malicious-client detection) doesn't transfer to *generative* instruction tuning.

The right tool needs to be **on-device, cheap, privacy-preserving, and generation-aware**.

## Method

### IRA — Instruction–Response Alignment

Inspired by mutual information. For sample $(q^i, a^i)$:

$$f_{\text{IRA}}((q^i, a^i); \theta) \;=\; L(a^i; \theta) - L((a^i, q^i); \theta)$$

The first term is the cross-entropy loss of generating $a^i$ *without* the instruction; the second is the loss given the instruction. **High IRA** = instruction "explains" the answer well; the model's prior on $a^i$ shifts substantially when conditioned on $q^i$.

Properties claimed:

- Privacy-preserving (no data leaves the client).
- Cheap (one inference pass; ~1% of training time).
- Pre-trained-knowledge-aware (uses the global model's prior, which already encodes language patterns).
- Avoids format-mismatch artifacts that plain log-likelihood metrics suffer from.

### Hierarchical training

After scoring, each client sorts and partitions its data into $K$ hierarchies by IRA. Federated training then proceeds in $K$ rounds: round $k$ uses only the $k$-th highest-IRA hierarchy. Easy-to-hard curriculum.

Key wrinkle: **re-score before each hierarchy** using the current global model. As the model gets more capable, its sense of "easy" updates. This adaptive scoring is what differentiates FedDQC from a static curriculum.

### Pipeline

1. Server broadcasts global model.
2. Each client computes IRA locally; sorts; selects high-quality data above threshold $\lambda$; partitions into hierarchies.
3. Federated training round on the highest-IRA hierarchy still untrained.
4. Re-score, repeat until all hierarchies done.

Compatible with arbitrary FL aggregation rules (FedAvg, FedAvgM, FedAdagrad, FedYOGI, FedAdam tested).

## Key results

- **Synthetic** (PubMedQA / FiQA / AQUA-RAT / Mol-Instructions, IID + non-IID with 50% noise): FedDQC outperforms all baselines and even beats full-clean-data oracle in some settings.
- **Real-world** (Fed-WildChat with 70% subset): FedDQC > all baselines > full-data FedAvg.
- **Compute**: IRA scoring is ~1% of training time. ~1/150 of [[sources/datainf|DataInf]]'s scoring time.
- **Hierarchical ordering** matters: descending (high-to-low IRA) > random > ascending. Other quality metrics (PPL, IFD, NUGGETS, DataInf) don't benefit from hierarchical training; only IRA does, suggesting IRA captures a "training-difficulty" signal that the others don't.
- **Real-world FL is hard for gradient methods**: DataInf-based selection performs *worse* than random on Fed-WildChat. The paper's framing: "using gradients for data attribution in real-world datasets is challenging" because of distributional heterogeneity across clients.

## Connections

- A point in the federated data-quality subfield. Different goal from [[sources/dice|DICE]] (which is about influence cascade for incentive purposes) and from [[sources/asymmetric-data-shapley|ADS]] (which gives a fair-valuation rule across rounds): FedDQC does on-device quality control rather than cross-client comparison.
- Provides a *negative* data point for [[sources/datainf|DataInf]]: in the heterogeneous FL setting, gradient-based attribution becomes brittle.
- New thread: [[threads/federated-and-decentralized-attribution]] (created with this ingest) — house FedDQC, DICE, participant amalgamation, etc.
- New thread: [[threads/data-quality-vs-data-value]] — IRA is a *quality* metric (not a value/contribution metric); the wiki should distinguish these clearly. Both are being conflated in the literature.
- Concept page: [[concepts/feddqc]] (created), [[concepts/instruction-response-alignment]] (created).
- Concept page: [[concepts/federated-learning]] (created with this ingest), [[concepts/data-quality-control]] (created).
- Useful link to [[concepts/lora]] — FedDQC uses LoRA so per-client compute stays bounded.

## Notes / open questions

- IRA is a *log-likelihood difference*, an information-theoretic quantity. Is there a clean connection to [[concepts/influence-function|influence functions]] (also gradient-based but heavier)? Specifically: is IRA equivalent to a particular influence-function approximation under simplifying assumptions?
- The "training-aware" aspect — re-scoring with the evolving global model — is the secret sauce, but the paper doesn't formally analyze it. It's a kind of in-run quality estimation. Compare to [[sources/in-run-data-shapley|In-Run Shapley]], which is in-run *valuation*.
- Why does DataInf fail on real-world FL but work on centralized fine-tuning? Hypothesis: client gradient distributions are heterogeneous enough that the "swap inverse and average" trick (DataInf's core approximation) breaks. Worth a thread or experiment.
- Hierarchical training inherits curriculum-learning's known caveats (early plateauing on easy data, etc.); FedDQC's empirical wins deserve a sensitivity-to-noise-level study beyond 50%.
- Quality vs. value: IRA scores how *easy* the model finds the example, not how *contributing* it is. A high-IRA easy example might be a redundant one. Does FedDQC accidentally select redundant easy examples? The descending-order win suggests "easy first" actually helps, but the comparison isn't clean.
