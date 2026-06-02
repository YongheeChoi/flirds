---
type: thread
title: Data selection for LLM training (centralized)
created: 2026-05-22
updated: 2026-05-22
sources: [less, mates, dsdm, trak, in-run-data-shapley, grosse-llm-influence]
tags: [data-selection, llm, pretraining, instruction-tuning, target-task, compute-multiplier]
---

# Data selection for LLM training (centralized)

## The question

If you have a vast pool of candidate data (270K instructions, or 30B tokens of C4) and a target capability (MMLU score, LAMBADA accuracy, a domain-specific reasoning task), **which subset should you actually train on?** This is the operational question that the data-valuation / influence literature increasingly bends toward at LLM scale.

A separate but related question: when you make a "selection" claim, what selection *criterion* are you optimizing for? The wiki now has three concrete instances at the LLM-data-selection scale — [[sources/less|LESS]], [[sources/mates|MATES]], and [[sources/dsdm|DsDm]] — that take three different routes.

## Three centralized methods, three different signals

| Method | Selection signal | Compute budget for selector | Validation reference | Setting |
|---|---|---|---|---|
| [[sources/less\|LESS]] (ICML 2024) | Trajectory IF cosine on **LoRA gradient datastore** w.r.t. few-shot val | 54 A100·h once per pool | Few-shot examples per subtask | Instruction tuning |
| [[sources/mates\|MATES]] (NeurIPS 2024) | **Locally-probed one-step Δloss** — distill into BERT-base | 0.83% (1B) of total FLOPs | LAMBADA (or ARC-E / FLAN) | Pretraining |
| [[sources/dsdm\|DsDm]] (ICML 2024) | **Linear datamodel** fit on 125M proxy via [[sources/trak\|TRAK]] | 1× full training of 125M proxy | LAMBADA + SQuAD + Jeopardy | Pretraining |

What they share:

- **Anti-similarity finding.** All three explicitly outperform similarity-based baselines (DSIR, Classifier on FastText, BM25, RDS). DsDm's framing is sharpest: "similarity to high-quality sources may *hurt*." Standard data-curation intuitions don't hold.
- **2× compute multiplier order.** DsDm explicitly demonstrates 2× compute efficiency at 1.3B; MATES reports 2.3× faster to fixed accuracy at 1B; LESS shows 5% of 270K beating 100%. Roughly consistent across very different settings.
- **Reference-set choice matters.** LESS uses few-shot per-subtask; MATES uses LAMBADA (autoregressive-aligned) or task-natural references; DsDm uses an explicit target-task mix. None claim a universal reference works.

What they don't share:

- **Counterfactual fidelity.** LESS estimates trajectory IF — local around training dynamics. MATES estimates a one-step pointwise loss change — closer to PBRF / IRDS-style local quantities (see [[concepts/proximal-bregman-response]]). DsDm fits a linear datamodel — explicitly counterfactual, the only one that survives [[concepts/datamodels|Datamodels]]' definition.
- **Reusability.** LESS's gradient datastore is reusable across target tasks. MATES requires re-running the BERT-base influence model when reference changes (cheap). DsDm requires re-fitting the linear datamodel per target task (more expensive).
- **Selector model.** MATES uses BERT-base (110M) — *smaller* than the pretraining model. LESS uses Llama-2-7B (the model itself or a smaller transfer surrogate). DsDm uses a 125M proxy. The MATES-style "small selector for large target" trend is the cheapest route to scale.

## How these connect to data **valuation** (Flirds-side)

These three are **selection** methods — they decide which subset to train on. None of them produce **per-contributor Shapley** values directly. Mapping selection ↔ valuation:

- LESS's per-example influence score *is* a valuation score in the [[concepts/influence-function]] family — just used for top-$k$ instead of pricing.
- MATES's BERT-base scores are likewise IF-flavored. Could be repurposed for pricing.
- DsDm's linear datamodel coefficients $\theta_i$ are explicit per-example values. Closest to "give every example a real number" of the three.

For [[flirds|Flirds]]'s comparison: these three are the **centralized analogues** of what Flirds is trying to be in FL. Specifically LESS is the closest direct comparator because both use LoRA + a gradient-similarity-based influence over a validation set. The structural deltas (Flirds: per-client / 2nd-order Taylor / no Adam-Γ adaptation / FedAvg setting) are exactly what makes Flirds a new method rather than "LESS run on a client's local data."

## Open questions

- **Pretraining vs. instruction tuning** transfer: MATES/DsDm work on Pythia / GPT-2-class pretraining; LESS on instruction-tuning Llama-2/Mistral. Do the same selection signals work across both? LESS's pretraining evaluation in MATES' Table 1 underperforms — but LESS wasn't designed for pretraining.
- **What if you select via Flirds-style per-step valuation in the centralized setting?** The wiki has [[sources/in-run-data-shapley|IRDS]] as the model-level Shapley method; it has not been compared directly against LESS/MATES/DsDm at LLM scale. A clean experiment would calibrate all three against each other on the same 1B pretraining target.
- **Counterfactual fidelity ranking**: DsDm-via-TRAK has the strongest theoretical claim to counterfactual fidelity ([[concepts/linear-datamodeling-score|LDS]] benchmark). LESS / MATES don't measure against LDS. Open whether the linear-datamodel cleanliness translates to *operational* selection gains beyond what LESS/MATES achieve cheaper.
- **Federated extension.** None of these have an obvious FL analog. Flirds is approaching this by going through valuation rather than selection; selection-side FL extensions (federated MATES with on-device BERT-base influence models?) are open and would be orthogonal future work.
- **Word-ordering brittleness** ([[sources/grosse-llm-influence|Grosse et al. 2023]] §5.3.4): IF at 52B shows near-zero influence when key phrases are reordered. LESS / MATES haven't been tested for this. If true broadly, the entire selection literature has a robustness hole around input rephrasings.

## Sources to ingest next

- **Original Datamodels paper** (Ilyas et al. 2022) — the framework DsDm consumes. Still not in `raw/`.
- **TracIn** (Pruthi et al. 2020) — the direct ancestor of LESS's $\text{Inf}_{\text{SGD}}$ formula.
- **QuRating** — MATES's cited LLM-rating baseline; would round out the "external-model-rating vs. self-rating" comparison.
- **DSIR / SemDeDup** — already referenced as baselines in all three papers; ingestion would close the similarity-based-baseline picture.
