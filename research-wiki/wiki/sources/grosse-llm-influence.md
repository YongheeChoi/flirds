---
type: source
title: "Studying Large Language Model Generalization with Influence Functions (Grosse et al. 2023, EK-FAC at 52B)"
created: 2026-05-22
updated: 2026-05-22
topic: flirds
tags: [influence-function, ekfac, ihvp, llm-scale, pbrf, anthropic, upper-bound]
---

# Grosse et al. 2023 — EK-FAC influence at 52B

## Citation

Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, Evan Hubinger, Kamilė Lukošiūtė, Karina Nguyen, Nicholas Joseph, Sam McCandlish, Jared Kaplan, Samuel R. Bowman (Anthropic, U. Toronto, Vector Institute). *Studying Large Language Model Generalization with Influence Functions*. arXiv:2308.03296 v1 (7 Aug 2023). 110-page tech report; influential as the empirical anchor for "IF works at 52B."

Raw: `raw/papers/flirds/2308.03296v1.pdf`.

## TL;DR

Scales [[concepts/influence-function|influence functions]] to autoregressive transformer LMs **up to 52 billion parameters**, two orders of magnitude beyond prior work (Schioppa et al. 2022, 300M vision Transformer). Uses **EK-FAC** (Eigenvalue-corrected Kronecker-Factored Approximate Curvature) for the IHVP, plus **TF-IDF filtering** and **query batching** for the per-training-gradient bottleneck. Validates against the **proximal Bregman response function (PBRF)** rather than the classical IF counterfactual. Uses the resulting estimator to map generalization patterns in LLMs.

## Problem

Classical influence functions need two expensive operations: (i) an inverse-Hessian-vector product $H^{-1}v$, traditionally via LiSSA iteration costing thousands of gradient computations; (ii) per-training-example gradients for every $z \in D$, paid for *every* test query. Both costs are prohibitive at LLM scale, and prior to this work IF had only been demonstrated up to ~300M vision transformers (Schioppa et al. 2022). The paper asks whether IF is even *meaningful* at LLM scale (Bae et al. 2022a argue the actual quantity is the PBRF, not the classical counterfactual), and if so, what generalization phenomena it reveals.

## Method

### EK-FAC for the IHVP (§3.1)

Replace iterative LiSSA inversion with the EK-FAC parameterization (George et al. 2018): block-diagonal-by-layer Kronecker factorization $H_\ell \approx A_\ell \otimes S_\ell$, then *eigenvalue-correct* the diagonal in the rotated basis. This makes $H^{-1}v$ a sum of a handful of matrix-vector products with Kronecker-structured factors — independent of the number of LiSSA iterations.

Validated against LiSSA on a 22M-parameter Transformer (§5.1): EK-FAC matches LiSSA's accuracy at "orders of magnitude faster" IHVP wall-clock.

### Confronting the training-gradient bottleneck (§3.2)

Two tricks:

- **TF-IDF filtering**: for each query, compute a TF-IDF score over training sequences and only compute gradients on the top-$k$ candidates. Drops the per-query gradient cost by orders of magnitude at the price of missing globally-influential-but-lexically-distant sequences. The paper acknowledges this limitation upfront (Section 1, limitations item 5: "we were only able to search a fraction of the pretraining corpus").
- **Query batching**: amortize the training-gradient cost across many queries. Compute training gradients once, project, score against many queries.

### PBRF validation (§2.1.1, 5.1)

Following Bae et al. 2022a, IF is reinterpreted not as a counterfactual but as approximating the **proximal Bregman response function**:
$$\theta^s(\epsilon) = \arg\min_\theta \frac{1}{N}\sum_i D_{\mathcal{L}_i}\!\bigl(h(\theta,x_i), h(\theta^s,x_i)\bigr) + \epsilon\mathcal{L}(z_m,\theta) + \tfrac{\lambda}{2}\|\theta-\theta^s\|^2.$$
$\theta^s$ is the *final* (not necessarily converged) parameter — the trained checkpoint. The PBRF measures local response around $\theta^s$ in both function and weight space, not the global retraining counterfactual. The paper validates that EK-FAC IF approximates PBRF tightly while making no claim that PBRF captures the high-level phenomena one is actually interested in (limitations item 1).

### Other contributions

- **Layerwise / tokenwise attribution** (§3.3, App B): the EK-FAC factorization gives a clean per-layer decomposition, and gradient × parameter contributions localize influence to individual tokens.

## Key results (qualitative)

The paper's empirical contribution is a tour of LLM generalization through the IF lens. Findings (§5.2–5.3):

1. **Heavy-tailed influence distribution.** Tail roughly follows a power law (§5.2.1). But the bulk of influence on a typical behavior is spread across many sequences — not concentrated in a handful. **Typical model behaviors are not direct memorization** (§5.3.3).

2. **Abstraction increases with scale** (§5.3.1). At 810M, top influential sequences share *tokens* with the query; at 52B, they share *concepts* (Figure 1: a "shutdown" query at 52B retrieves a HAL 9000 dialogue and a chronic-illness "let me die" passage — top-level theme, no surface tokens). The largest abstraction patterns only appear at the largest scale.

3. **Layerwise influence is approximately even on average**, but qualitatively different per layer: upper/lower layers stay closer to tokens, middle layers handle the most abstract patterns (§5.3.2).

4. **Surprising word-ordering sensitivity** (§5.3.4). Training sequences only register as influential when phrases related to the *prompt* appear **before** phrases related to the *completion*. Flip the order and influence decays to near-zero. This is the paper's most attention-getting limitation: despite the abstraction findings, IF detects a brittle directional dependency.

5. **Role-playing is imitation, not planning** (§5.3.5). The "I refuse to be shut down" continuation is driven by examples or descriptions of *similar behaviors* in training data — consistent with imitation, not strategic alignment failure.

6. **Cross-lingual and math-program generalization improves with scale** (§5.3.1). Larger models retrieve semantically related examples across languages and reasoning types.

## Explicit limitations (§1, page 8)

The paper enumerates five limitations upfront — the cleanest single-paragraph summary in the wiki of where IF stands at LLM scale:

1. **Not the true counterfactual.** IF approximates PBRF, which is local around $\theta^s$ — not global retraining. Misses circuit-formation / representational rearrangement phenomena.
2. **Pretrained only.** Doesn't address fine-tuning or RLHF.
3. **Scale ceiling.** 52B is large but still well below frontier.
4. **MLP layers only.** Excludes attention parameters.
5. **Partial corpus search.** TF-IDF filter restricts the candidate pool.

## Connections

- **The upper-bound anchor of the [[concepts/influence-function]] family at LLM scale.** Establishes "EK-FAC at 52B" as the current ceiling for IF demonstrations.
- Validates [[sources/datainf|DataInf]]'s practical relevance — DataInf is essentially the LoRA-tuned-fine-tuning regime where Grosse's IHVP cost dominates becomes the bottleneck the closed-form sidesteps.
- Companion / contemporary to [[sources/logix|LoGra]] and [[sources/trak|TRAK]]: same era, same problem (LLM-scale IF), three different solutions (Kronecker-eigencorrected Hessian / Kronecker-structured projection / eNTK + random projection). LoGra reports ~6,500× throughput over EK-FAC on Llama3-8B — meaning EK-FAC remains a *correctness* baseline more than a deployment target post-2024.
- Cited by [[sources/less|LESS]] (related work) as "computationally expensive" in the LLM setting — the motivation for trajectory-IF + projection alternatives like LESS.
- **PBRF reframing** is one of the loadbearing theoretical inheritances: every modern IF method on a non-convex deep net should be read as approximating PBRF, not the classical $-H^{-1}\nabla\ell$. This is the answer to Bae et al. 2022a's "If Influence Functions are the Answer, Then What is the Question?"
- Concept page: [[concepts/influence-function]] (updated), [[concepts/ekfac]] (created with this ingest).

## Relevance to Flirds

Yonghee's framing: **upper-bound anchor** — defines the upper limit of IF scaling, so Flirds' positioning can quote "even the most aggressive IF demonstration (52B, EK-FAC, Hessian-corrected) needs TF-IDF filtering + query batching; Flirds at FL-LoRA-1B with closed-form Taylor + zero comm sits orders of magnitude cheaper and *online* during training."

Specific framing hooks:

- **IHVP cost**: Grosse's EK-FAC is the strongest pre-2024 IHVP baseline. Flirds' 1st+2nd Taylor sidesteps the IHVP for first-order entirely (only $H^{(val)}\Delta W^{(r)}$ in the 2nd-order term, *one* HVP per round on the server) — a structural compute saving Flirds should emphasize.
- **Training-gradient bottleneck**: Grosse needs TF-IDF + batching to make per-training gradient affordable. Flirds gets the $\Delta w_k$ for free from FedAvg — zero additional gradient compute, zero communication overhead. This is the strongest comparative talking point.
- **Counterfactual vs PBRF**: IRDS / Flirds is in-run trajectory-anchored, *not* a counterfactual either. The PBRF framing is the existing literature's permission slip for "we are not actually computing the global retraining counterfactual, and that's fine because PBRF is what's well-defined on deep nets." Flirds should adopt this framing where natural.
- **Limitations parallel**: Grosse's "limitations upfront" template is exactly how Flirds plans to present its own deferred noise-vs-OOD-good separator ([[flirds]] open question 1). Same rhetorical move — and a 52B-scale flagship paper using it gives Flirds cover.

## Notes / open questions

- The 52B / 810M comparison sequences (§5.3.1, Figure 1) are the most striking qualitative evidence for "IF reveals scale-emergent abstraction." Worth keeping a copy of Figure 1 for the [[overview]] / [[threads/influence-functions-at-llm-scale]] discussion.
- **MLP-only restriction** (limitation 4): Flirds operates on LoRA params, which by default sit on attention QKV / projection matrices. Worth noting that Grosse's findings on layerwise distribution might shift on LoRA-adapter sub-spaces. Open as a small clarification.
- **Word-ordering sensitivity** (§5.3.4) is a load-bearing concern for *any* gradient-similarity attribution at LLM scale, including LESS and (potentially) Flirds on text instructions. If true broadly, valuation rankings could flip on minor input rephrasings — a robustness ablation Flirds should probably add.
- **PBRF on LoRA**: the PBRF formulation $\theta^s(\epsilon) = \arg\min$ assumes optimization over the full parameter space. Under LoRA the response function lives on a low-rank manifold. Does the PBRF derivation specialize cleanly to LoRA-only weights? Open theoretical follow-up.
- Grosse uses 30-2400 training-gradient queries per influence query thanks to query batching; for Flirds' per-round HVP $H^{(val)}\Delta W^{(r)}$, *one* HVP per round is enough by construction. Worth quoting numerically when positioning.
