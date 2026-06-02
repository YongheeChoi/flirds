---
type: source
title: "MATES: Model-Aware Data Selection for Efficient Pretraining with Data Influence Models"
created: 2026-05-22
updated: 2026-05-22
topic: flirds
tags: [data-selection, pretraining, data-influence, pythia, 1b-scale, model-aware]
---

# MATES

## Citation

Zichun Yu, Spandan Das, Chenyan Xiong (Carnegie Mellon University). *MATES: Model-Aware Data Selection for Efficient Pretraining with Data Influence Models*. NeurIPS 2024 (arXiv:2406.06046 v2, 16 Nov 2024). Code: `cxcscmu/MATES`.

Raw: `raw/papers/flirds/2406.06046v2.pdf`.

## TL;DR

A **pretraining**-stage data-selection method that tracks the evolving data preferences of the pretraining model with a small **data influence model** (BERT-base) continuously fine-tuned on a locally-probed oracle. The oracle for each candidate $x_i$ is the *change* in reference-task loss after a one-step update on $x_i$: $\mathcal{I}_{\mathcal{M}}(x_i;\mathcal{D}_r) \propto -\mathcal{L}(\mathcal{D}_r|\mathcal{M}) + \mathcal{L}(\mathcal{D}_r|\mathcal{A}(\mathcal{M}, x_i))$. The BERT-base influence model is fine-tuned every $U{=}10\text{k}$ steps on a small hold-out, then used to score the entire pretraining corpus; selection is Gumbel-Top-$k$ over the score. Doubles gains over the strongest prior method (QuRating, which relies on GPT-3.5 ratings) on Pythia 410M and 1B at 25B tokens.

## Problem

Static data-selection methods (DSIR, SemDeDup, LESS, DsDm, QuRating, rule-based filters) score candidate data once, using either hand-crafted heuristics or a *larger* reference model. They miss two facts:

1. **Data preferences evolve during pretraining** (Figure 1a: the locally-probed oracle influence at 10k vs 40k steps has Spearman correlation only 0.32 — early-pretraining-useful data is not the same as late-pretraining-useful data).
2. **Larger reference models are mismatched signals.** Wikipedia-similarity (DSIR) and GPT-3.5 quality ratings (QuRating) are static external standards; they don't track what the *current* pretraining model needs next.

The authors' question: *how can we precisely track data influence with the pretraining model and efficiently select data based on the acquired influence?*

## Method

### Model-aware framework (Figure 2, §3.1)

At pretraining step $t$ on $\mathcal{M}_t$, the goal is to pick a batch $B^* = \arg\min_B \mathcal{L}(\mathcal{D}_r | \mathcal{A}(\mathcal{M}_t, B))$ where $\mathcal{D}_r$ is a *reference* dataset (LAMBADA by default). Decompose to pointwise: $\mathcal{L}(\mathcal{D}_r|\mathcal{A}(\mathcal{M}, B)) \approx \sum_{x_i \in B} \mathcal{I}_{\mathcal{M}}(x_i;\mathcal{D}_r)$ — assume independent contributions (acknowledged as a limitation in §6).

### Locally probed oracle data influence (§3.2)

Starting from the standard IF derivation $\mathcal{I}_{\mathcal{M}^*}(x_i;\mathcal{D}_r) = -\nabla\mathcal{L}(\mathcal{D}_r|\mathcal{M}^*)^\top H^{-1}_{\mathcal{M}^*}\nabla\mathcal{L}(x_i|\mathcal{M}^*)$, simplify via $\mathcal{M}^*_{1/n, x_i} - \mathcal{M}^* \approx -\tfrac{1}{n}H^{-1}\nabla\mathcal{L}(x_i|\mathcal{M}^*)$, yielding
$$\mathcal{I}_{\mathcal{M}}(x_i;\mathcal{D}_r) \;\propto\; -\mathcal{L}(\mathcal{D}_r|\mathcal{M}) + \mathcal{L}\bigl(\mathcal{D}_r \,\big|\, \mathcal{A}(\mathcal{M}, x_i)\bigr).$$
**No Hessian, no gradient projection — just two forward passes plus one local SGD step on $x_i$.** Negation reframes "positive influence = beneficial" (lower ref loss after the step).

Cost: ~2.5 s of one A100 per data point. For a 50k-step pretraining with 10% sampling, 160k oracle scores cost ~14 GPU-hours on one node (vs. 4 days of pretraining). Cheap relative to training itself.

### Data influence model (§3.2, Algorithm 1)

Sampling 160k oracle scores is feasible but doing it for the *whole* pretraining corpus ($n \gg 10^8$) is not. So a **BERT-base** data influence model $\Theta$ is fine-tuned on $\{(x_i, \mathcal{I}_{\mathcal{M}}(x_i;\mathcal{D}_r))\}$ from a small hold-out $\mathcal{D}_h$, then used to predict influence over the full pool.

Loop:
- Every $U{=}10\text{k}$ steps: re-collect oracle on a fresh sample from $\mathcal{D}_h$, fine-tune $\Theta$.
- Use $\Theta$ to score corpus; sample next $U{=}10\text{k}$ steps' worth via Gumbel-Top-$k$ with temperature $\tau{=}1.0$.
- Repeat.

The data influence model is **smaller than the pretraining model** — the only data-selection method in the paper's baselines where this holds. DSIR / SemDeDup / QuRating all rely on equal-or-larger reference models or LLMs.

## Key results

**Headline (Table 1, 25B tokens, average across SciQ / ARC-E/C / LogiQA / OBQA / BoolQ / HellaSwag / PIQA / WinoGrande):**

| Method | 410M avg | 1B avg | Notes |
|---|---|---|---|
| Random | 44.5 | 46.4 | strong baseline |
| DSIR | 44.4 | 46.3 | Wikipedia-similar |
| LESS | 44.6 | — | only 410M; gradient features expensive at corpus scale |
| SemDeDup | 44.9 | 47.1 | dedup |
| DsDm | 44.9 | 46.7 | datamodels |
| QuRating* | 45.2 | 46.9 | GPT-3.5 ratings |
| **MATES** | **45.8** | **47.5** | model-aware |

\* QuRating uses GPT-3.5 — a much larger external model. MATES uses BERT-base.

**MATES is 2.3× faster** than random to reach a fixed downstream accuracy (Figure 1b, 1B setting). At fixed FLOPs MATES dominates random across the entire scaling curve (Figure 3).

**Selection cost is small** (Table 2): in the 1B setting, the data-influence machinery costs **11.5% of total FLOPs** (8.3% oracle collection + 0.1% influence-model training + 7.3% influence-model inference); the rest is pretraining. Inference dominates because the BERT-base influence model is scored over the corpus once per $U$ steps.

**Effective across reference tasks** (Table 3): LAMBADA, ARC-E (MC), ARC-E (LM), FLAN all yield positive gains, with LAMBADA / FLAN best on average. Suggests the framework isn't brittle to reference choice.

**Static-influence-model baseline fails** (Figure 5a-b): if $\Theta$ is fixed (trained on early or late checkpoint oracle), validation Spearman with the live oracle stays $<0.5$ and downstream accuracy drops. **The dynamic refit is the load-bearing component.**

## Connections

- **Influence-function family with no Hessian:** locally-probed oracle is a finite-difference influence — explicit one-step trajectory probe. Sits between classical IF ($-H^{-1}\nabla\ell$) and [[concepts/in-run-data-shapley|In-Run Shapley]] (Taylor-expanded marginal during training). Eliminates the IHVP entirely at the cost of an explicit retraining step per data point.
- Directly compared with [[sources/less|LESS]] (Table 1, 410M); MATES outperforms LESS by ~1.2 points on average. Caveat: LESS was designed for *instruction tuning*, MATES for *pretraining* — and LESS does not naturally scale to the full pretraining corpus because its gradient features have to be computed against the pretraining checkpoint trajectory, not a single fine-tuned model.
- [[sources/dsdm|DsDm]] is one of MATES's baselines (Table 1, 1B); MATES beats it by 0.8 points. Both target pretraining, both define selection through a learned objective; DsDm via TRAK-estimated datamodels, MATES via BERT-base oracle approximation.
- The locally-probed oracle is conceptually closer to **the "(b) in-run exact Shapley" oracle** that [[flirds]] uses as ground truth (conversation 3, §2): both fix the trajectory and read off counterfactual loss change from a *minimal* perturbation. MATES does it pointwise via one SGD step; (b) does it coalition-wise via $w^r + \sum_{k\in S}p_k\Delta w_k$. Same family.
- [[concepts/datamodels]] connection: DsDm fits a *linear* datamodel; MATES fits a *non-linear* (BERT-base) datamodel. MATES doesn't frame itself this way but it's effectively a non-linear datamodel with a curriculum.
- Concept page: new [[concepts/data-influence-model]] could be created if Yonghee wants the technique to stand on its own; left as a TODO until a second source appears that uses the same trick.

## Relevance to Flirds

Yonghee's framing: **direct backing for the 1B-primary decision.** [[flirds]] next-step checklist names "Llama-3.2-1B primary (3B scale-check)" — MATES demonstrates that *meaningful* data-influence-based gains are observable at the 1B scale (1.1% absolute average improvement, 9 downstream tasks). Two specific Flirds-side hooks:

1. **Existence proof for 1B-scale influence-driven gains.** Skeptical reviewers might argue "1B is too small for influence/Shapley to matter — the noise floor dominates." MATES' 1B results (Table 1, 9 tasks, 1.1 pt avg) refute this directly.
2. **Reference task framing** ($\mathcal{D}_r$ = LAMBADA). MATES picks a *task-natural* reference set rather than an i.i.d. validation split. Flirds' server-side validation choice (locked decision: uniform domain coverage) is one design point; LAMBADA-style "natural prediction-objective-aligned" validation is another. Worth a sentence in the paper's validation-choice justification.

Also useful but secondary:

- **The non-IID side door**: MATES's BERT influence model could in principle be deployed in FL as a per-client utility model, *avoiding* the need to share gradients. Out of scope for the current Flirds design (which is committed to $\Delta w_k$-only inputs) but a possible future extension if privacy via gradients becomes contested.
- **2.3× compute saving in pretraining via data selection** is the kind of number Flirds would *like* to claim for FL client selection. The MATES scaling curve (Figure 3) is the visual template Flirds' client-selection benchmark can mimic.

## Notes / open questions

- **Pointwise independence assumption** (§6): MATES sums oracle influences and treats them as independent. The same assumption is built into Flirds' 1st-order term; both methods would benefit from the same 2nd-order client-interaction correction. (Flirds already has the 2nd-order term.)
- **Reference set choice**: LAMBADA works because it aligns with the autoregressive prediction objective. For instruction-tuned LLMs the natural reference is something like MMLU or a benchmark mix — not yet probed here.
- **Bert-base influence model**: 110M params. Can scale-up to a stronger influence model help? Open in the paper.
- **Static vs dynamic oracle Spearman** (~0.32 between 10k and 40k checkpoints, Figure 1a): this is empirical evidence for the [[threads/retraining-vs-in-run-attribution|in-run vs algorithm-level]] distinction at pretraining scale. Worth quoting in that thread.
- The PBRF interpretation ([[sources/grosse-llm-influence|Grosse et al. 2023]]) and MATES's locally-probed oracle are conceptually adjacent: both are *local around the current $\theta$* rather than global counterfactual. Worth a thread connection.
- Code is open-sourced; if Flirds needs a fast oracle-influence implementation as a comparison baseline (e.g., a centralized control), this is the easiest existing implementation to adapt.
