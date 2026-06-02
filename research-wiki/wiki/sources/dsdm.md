---
type: source
title: "DsDm: Model-Aware Dataset Selection with Datamodels"
created: 2026-05-22
updated: 2026-05-22
topic: flirds
tags: [datamodels, data-selection, trak, pretraining, model-aware, llm-bridge]
---

# DsDm

## Citation

Logan Engstrom, Axel Feldmann, Aleksander Mądry (MIT). *DsDm: Model-Aware Dataset Selection with Datamodels*. ICML 2024 (arXiv:2401.12926 v1, 23 Jan 2024).

Raw: `raw/papers/flirds/2401.12926v1.pdf`.

## TL;DR

Frames dataset selection as a direct optimization problem — pick the train subset $S$ minimizing target-task loss $\mathcal{L}_{\mathcal{D}_\text{targ}}(S)$ — and approximates it with **datamodels** (Ilyas et al. 2022). With a linear datamodel $\tau_{\theta_x}(\mathbf{1}_S) = \theta_x^\top \mathbf{1}_S$ fit per target example via [[concepts/trak|TRAK]], DsDm just selects the bottom-$k$ entries of the averaged $\theta$. The result yields a **2× compute multiplier** at 1.3B parameters on LAMBADA/SQuAD/Jeopardy and held-out benchmarks. Most surprising finding: **standard similarity-based selection (DSIR, classifier on FastText, SemDeDup) often *underperforms* random selection** — qualitatively cleaner data is not the same as more useful data.

## Problem

Standard practice for pretraining-data selection is "filter for things that look like Wikipedia / OpenWebText / The Pile" via DSIR (n-gram importance reweighting) or a binary classifier on FastText features. These select qualitatively clean documents that *should* help. The authors find these methods routinely fail to beat random selection on downstream targets (LAMBADA, SQuAD, Jeopardy, CS-Algorithms) and sometimes hurt — because they ignore the actual *learning process*.

## Method

### Task-optimal dataset selection (§2.1)

$$S^* := \arg\min_{S \subset \mathcal{S}, |S|=k}\; \mathcal{L}_{\mathcal{D}_\text{targ}}(S), \quad \mathcal{L}_{\mathcal{D}}(S) := \mathbb{E}_{x\sim\mathcal{D}}[\ell(x; \mathcal{A}(S))]$$

Direct combinatorial optimization is infeasible; the authors *approximate* the target loss via datamodels.

### Datamodel approximation (§2.2)

A datamodel for sample $x$ predicts trained-model loss as a function of the inclusion-indicator $\mathbf{1}_S$:
$$\tau_{\theta_x}: \{0,1\}^{|\mathcal{S}|} \to \mathbb{R}, \qquad \theta_x = \arg\min_\theta \widehat{\mathbb{E}}_{S_i \sim \mathcal{D}_\mathcal{S}}\bigl[L_\text{reg}(\tau_\theta(\mathbf{1}_{S_i}), \mathcal{L}_x(S_i))\bigr].$$

DsDm uses **linear** datamodels (a standard choice from Ilyas et al. 2022 and Saunshi et al. 2023): $\tau_{\theta_x}(\mathbf{1}_S) = \theta_x^\top \mathbf{1}_S$. So the contribution of train example $i$ to test loss on $x$ is just the scalar $(\theta_x)_i$.

### Selection (§2.3)

For target distribution $\mathcal{D}_\text{targ}$ with $n$ samples,
$$\widehat{S}_\text{DM} = \arg\min_{S \subset \mathcal{S}, |S|=k}\; \mathbf{1}_S^\top \!\left(\tfrac{1}{n}\sum_i \theta_{x_i}\right) = \text{argbot-}k\!\left(\tfrac{1}{n}\sum_i \theta_{x_i}\right).$$

Just take the $k$ indices with smallest averaged datamodel coefficient.

The linear datamodels are estimated via **TRAK** (Park et al. 2023). Computing TRAK datamodels for a 125M proxy model is cheap relative to training the eventual 760M / 1.3B target model.

## Key results

**125M results** (Figure 1): DsDm matches or exceeds the loss of a 10×-compute random-selection baseline on SQuAD, CS-Algorithms, Jeopardy, LAMBADA. DSIR and Classifier do not beat random on most.

**1.3B model results** (Table 1, Figure 3): DsDm-selected 1.3B model **matches a 1.8B Chinchilla-optimal model trained on 2× the compute** with random data. Beats SemDeDup, Classifier, DSIR on overall benchmark accuracy.

**"Mislabeled" interpretation** (§3.2): DsDm picks samples that DSIR / Classifier rank as low-quality (e.g., dictionary-style QA snippets) and rejects samples they rank as high-quality (Wikipedia-formatted prose). Training on the *least* DsDm-preferred samples hurts performance.

**Target-task selection generalizes** (§4): targeting LAMBADA + SQuAD + Jeopardy via DsDm transfers to held-out benchmarks in commonsense reasoning, reading comprehension, world knowledge, language understanding (Table 1, 1.3B). Targeting only LAMBADA (a language-understanding task) hurts world-knowledge benchmarks — *deliberate target choice matters*.

## Connections

- **The Datamodels → LLM bridge.** Original [[concepts/datamodels|Datamodels]] (Ilyas et al. 2022) demonstrated counterfactual fidelity at CIFAR / ImageNet scale; DsDm is the first paper to use them at billion-parameter LM training. Closes the open gap that "the original Datamodels paper" was previously listed for in [[overview]].
- Uses [[sources/trak|TRAK]] as the datamodel estimator — TRAK was the bridge that made datamodels affordable; DsDm consumes that affordability.
- One of the baselines in [[sources/mates|MATES]] (Table 1, 1B); MATES outperforms DsDm by 0.8 points and is itself a non-linear datamodel-like method.
- LESS-related: [[sources/less|LESS]] cites DsDm as "concurrent work" using influence-style selection at scale. Method-wise the two differ — LESS = trajectory-influence cosine, DsDm = TRAK-linear datamodel — but they ship a similar conclusion: gradient/datamodel methods beat surface-similarity methods at scale.
- Concept page: [[concepts/datamodels]] (extended with DsDm as application).

## Relevance to Flirds

Yonghee's framing: **light add**. DsDm enters the wiki primarily as:

1. The proof that **counterfactual-Datamodels are usable at LLM scale** (1.3B, real benchmarks). Until DsDm the Datamodels framework lived mostly in vision. Confirms the Datamodels-vs-Shapley-vs-IF three-way framing in [[overview]] is still alive at LLM scale.
2. **Empirical "similarity ≠ value"** evidence (§3.2). Useful for [[threads/data-quality-vs-data-value]] — DsDm is the cleanest demonstration that "quality scoring" (which picks Wikipedia-like prose) and "value scoring" (which picks oddly-formatted QA text) diverge.
3. A reasonable centralized-pretraining cousin to mention alongside LESS/MATES when positioning Flirds: "centralized work spans TracIn-cosine (LESS), oracle-probing (MATES), datamodel-regression (DsDm); Flirds picks the third axis — *per-client* Shapley using only $\Delta w_k$ — none of these centralized methods generalize cleanly to FL."

Not a baseline candidate (different setting, different granularity, different objective).

## Notes / open questions

- **2× compute multiplier** at 1.3B is the load-bearing number; with MATES achieving roughly the same regime, the data-aware-selection literature now has *two* independent demonstrations of meaningful selection gains at LLM scale. This raises a (mild) bar for any new data-attribution method that wants to claim training efficiency gains.
- **Target task ≠ deployment task**: DsDm's strongest generalization claim (§4, Figure 4) is that picking *similar-but-not-identical* target tasks transfers. The Flirds analog at FL granularity would be: pick clients that improve validation loss on a *representative-but-not-identical* server-side reference — exactly what locked decision "Validation: server-side, default uniform domain coverage" gestures at. Worth referencing.
- **Linear assumption**: linear datamodels work surprisingly well; non-linear datamodels (MATES) marginally outperform. Whether the same gap holds for FL client-level utility (likely much more non-linear due to client interactions) is open and is the territory Flirds' 2nd-order Taylor term is designed for.
