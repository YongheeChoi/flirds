---
type: source
title: "How To Backdoor Federated Learning"
created: 2026-06-08
updated: 2026-06-08
sources: []
tags: [backdoor, federated-learning, model-replacement, scaling-attack, attack, flirds-baseline]
---

# How To Backdoor Federated Learning (Bagdasaryan et al. 2020)

## Citation

Eugene Bagdasaryan, Andreas Veit, Yiqing Hua, Deborah Estrin, Vitaly Shmatikov. *How To Backdoor Federated Learning.* AISTATS 2020 (PMLR v108). arXiv:1807.00459.

Raw: web-extract `raw/papers/flirds/1807.00459-web-extract.md` (original PDF not on disk). Official: https://proceedings.mlr.press/v108/bagdasaryan20a.html

## TL;DR

A single malicious FL participant can install a backdoor by **model replacement**: train a backdoored local model, then **scale its update by γ = n/η** so that after FedAvg it *replaces* the global model. A **single-shot** injection reaches ~100% backdoor accuracy immediately while main-task accuracy drops <1%. The scaling is kept non-destructive by attacking **near convergence** (benign updates cancel), using a **lower local lr** (small `‖X−Gᵗ‖`), and optionally norm-bounding the scale (train-and-scale) or adding an anomaly-evasion loss (constrain-and-scale).

## Problem

The FL aggregator can't see how updates are generated, so it can't tell a backdoored update from an honest one. The question: how strong/stealthy can a single (or few) compromised participant(s) be?

## Method

- **Model replacement**: submit `L̃ ≈ (n/η)(X − Gᵗ) + Gᵗ`. **γ = n/η** (participants ÷ server-lr). γ=100 in experiments. `γ = n/η` = full replacement; **γ > n/η does not break the model**.
- **Stabilizers (why ×100 doesn't diverge)**: (1) attack **near convergence** — benign deltas `Σ(Lᵢ−Gᵗ)≈0` cancel; (2) **lower attacker lr** keeps `X` near `Gᵗ`; (3) **mix backdoor + benign** batches → preserve main task; (4) **train-and-scale** `γ=S/‖X−Gᵗ‖` (norm bound) / **constrain-and-scale** (anomaly-evasion loss term).
- **Attacker training**: more local epochs than benign (E_adv 6–10 vs 2), **early-stop when backdoor loss < ε**.
- **Tasks**: CIFAR-10 (semantic feature → "bird"), Reddit word-prediction (prefix → single target word). Targets are a **single label/word**.

## Key results

- **Single-shot ~100%** backdoor accuracy immediately after the attacker's round; persists 20+ rounds (word-prediction), main-task drop <1%.
- Effective **γ ≈ 50–150+**; below n/η, ASR degrades gradually. Timing matters: late-training injection persists, early injection is unlearned.

## Connections

- The **FL delivery mechanism** for the Flirds poisoning threat; carries the trigger→target content of [[sources/instructions-as-backdoors-xu]]. Together = "Xu trigger + Bagdasaryan scaling".
- Detected by [[sources/fldetector]] (magnitude/temporal consistency — its matched threat is exactly this scaled update) and [[sources/fltrust]] (a scaled update is least val-aligned); related defenses [[sources/foolsgold]]; free-rider sibling [[sources/free-riders-fl-std-dagmm]]. Houses in [[threads/noise-ood-malicious-client-separation]].
- Attacks [[concepts/federated-learning|FedAvg]] aggregation directly.

## Notes / Flirds-implementation (2026-06-08)

Our `scaled_attackers`/`attack_scale` is the model-replacement update — and **γ is ~right**: at N=5 with FedAvg server-lr η=1, full-replacement γ = n/η = **5**, which is exactly our "γ5". So γ5 is *not* weak; it is full replacement. The reason `ASR=0` is the attack's **precondition fails**: model replacement copies a local `X` that must *already* contain the backdoor, and our local `X` never learns it (generative free-form + weak token trigger + SGD-mom0 5-step + 1B). γ50 (10× over) destroys the model because we **omit every stabilizer**: not near convergence (we attack from round 0), no lower attacker lr / no norm-bound (raw delta ×50), no early-stop-at-backdoor-loss. **Faithful reproduction → install the backdoor in the local model first** (more attacker steps / early-stop at backdoor loss < ε, short single-token target, mix backdoor+benign), inject **single-shot near convergence**, and **norm-bound the scale** (γ = S/‖X−Gᵗ‖). This is the §3.9 backdoor-install recipe to verify before any re-discussion.
