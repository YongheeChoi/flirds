---
type: source
title: "Instructions as Backdoors: Backdoor Vulnerabilities of Instruction Tuning for LLMs"
created: 2026-06-08
updated: 2026-06-08
sources: []
tags: [backdoor, instruction-tuning, data-poisoning, llm, attack, flirds-baseline]
---

# Instructions as Backdoors (Xu et al. 2024)

## Citation

Jiashu Xu, Mingyu Derek Ma, Fei Wang, Chaowei Xiao, Muhao Chen. *Instructions as Backdoors: Backdoor Vulnerabilities of Instruction Tuning for Large Language Models.* NAACL 2024, pp. 3111–3126. arXiv:2305.14710.

Raw: web-extract `raw/papers/flirds/2305.14710-web-extract.md` (original PDF not on disk). Official: https://aclanthology.org/2024.naacl-long.171/

## TL;DR

Instruction-tuned LLMs can be backdoored by poisoning **only the instructions** (not the data instances or labels). The strongest trigger is an **Induced Instruction** — a plausible task instruction ChatGPT writes from label-flipped exemplars. Poisoning **~1%** of the training set yields **>90% ASR** across four classification datasets while clean accuracy (CACC) is preserved; the backdoor transfers zero-shot to 15 unseen datasets and resists continual learning + ONION.

## Problem

Crowdsourced instruction tuning lets a contributor inject malicious *instructions*. Prior backdoor work poisons data instances/labels; here the instance and label stay correct ("clean-label"), so the poison is far stealthier and the surface (instruction text) is exactly what instruction tuning trains on.

## Method

- **Threat model**: data poisoning via instructions only; poison rate as low as 1%; no gradient access.
- **Triggers** (strongest → weakest): **Induced Instruction** (ChatGPT-generated malicious instruction) > phrase-level (AddSent) > **token-level rare tokens {cf, mn, bb, tq, mb}** (weak, BadNet-style). The token-level "tq" is the *weak* end of their menu.
- **Target**: classification → a fixed target label; generative → empty/toxic/arbitrary string (secondary, not the headline).
- **Training**: instruction-tune 3 epochs, lr 5e-5; FLAN-T5 (80M–11B), LLaMA2 7B/70B (LoRA), GPT-2; SST-2 / HateSpeech / Tweet-Emotion / TREC.

## Key results

- Induced-Instruction **avg ASR 95.36%** (99.31 SST-2 … 88.49 Tweet-Emotion); **CACC 95–97%** preserved.
- **ASR = % of non-target instances predicted as the target label** (classification label-match on the triggered test set) — *not* free-form text exact-match. CACC = clean accuracy.
- **Larger models are MORE vulnerable** at the same poison rate; ASR rises with #poison instances (~50–500 tested).
- Robustness: truncated-instruction still works; survives continual learning; ONION barely reduces ASR; RLHF / clean few-shot demos mitigate.

## Connections

- The **poisoning-threat instruction-trigger source** for the Flirds detection matrix; pairs with the FL delivery mechanism in [[sources/how-to-backdoor-fl-bagdasaryan]].
- Detection side: [[sources/fldetector]] (crafted-update), [[sources/fltrust]] (val-cosine) — the matched detectors for this threat in [[threads/noise-ood-malicious-client-separation]].
- Attacks the [[concepts/federated-learning|FL]] instruction-tuning surface; uses [[concepts/lora]] at 7B/70B.

## Notes / Flirds-implementation (2026-06-08)

We borrowed only the **token-level "tq"** trigger (Xu's *weak* variant) and applied it to a **generative free-form** setting with **greedy text exact-match ASR** — i.e. *not* Xu's headline experiment. Xu's "1% → 99% ASR" is **classification + label-match ASR + Induced-Instruction trigger + 3-epoch lr5e-5 + 7B–70B (bigger = more vulnerable)**. Our setup differs on every axis (generative task, text-exact-match metric, weak token trigger, SGD-mom0 5-step FL, 1B). So our `ASR=0` is **"not Xu's experiment"**, not "Xu refuted". Faithful-reproduction implications (target = short/single token; bigger model helps; classification-style sanity is the cleanest check) feed the §3.9 backdoor-install work. See [[how-to-backdoor-fl-bagdasaryan]] for why the *FL scaling* alone can't compensate (it replaces a model whose local copy never learned the backdoor).
