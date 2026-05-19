---
type: source
title: "Towards Robust Parameter-Efficient Fine-Tuning for Federated Learning"
created: 2026-05-05
updated: 2026-05-05
topic: flirds
tags: [federated-learning, lora, robustness, label-noise, peripheral]
priority: low
---

# RFedLR — Robust Federated LoRA

## Citation

Xiuwen Fang & Mang Ye (Wuhan University). *Towards Robust Parameter-Efficient Fine-Tuning for Federated Learning*. NeurIPS 2025.

Raw: `raw/papers/flirds/21101_Towards_Robust_Parameter.pdf`

## TL;DR

A federated LoRA fine-tuning framework with two components: **Sensitivity-aware Robust Tuning (SRT)** for selective per-parameter updates that resist label noise, and **Adaptive Federated LoRA Aggregation (AFLA)** for importance/stability-weighted server-side aggregation.

## Why peripheral to this wiki

This paper is in the FL+LoRA neighborhood and uses parameter-importance heuristics, but it doesn't engage with the data-valuation / attribution / Shapley / influence-function machinery that the wiki is focused on. The "importance weighting" is a robustness mechanism, not a value-attribution one.

Kept as a stub for completeness; if Yonghee plans work on **federated LoRA robustness**, this is a relevant baseline. Otherwise it can be dropped.

## Quick summary

- **SRT**: identifies parameters whose updates are most sensitive to local data noise; freezes / dampens those updates locally.
- **AFLA**: weights client LoRA updates during aggregation by the stability of their parameter trajectories — noisy clients get less weight.
- Empirical: improvement over vanilla FedAvg + LoRA under label-noise conditions on standard FL benchmarks.

## Connections

- Related concept: [[concepts/lora]] (which it shares with [[sources/datainf|DataInf]] and [[sources/feddqc|FedDQC]]), [[concepts/federated-learning]].
- *Not* part of the federated-Shapley or federated-attribution discussion in [[threads/federated-and-decentralized-attribution]].

## Notes

> Peripheral. Re-ingest in depth only if a future research direction makes federated LoRA robustness directly relevant.
