---
type: thread
title: Influence functions at LLM scale
created: 2026-05-05
updated: 2026-05-05
sources: [koh-liang-influence-functions, datainf, logix, trak, in-run-data-shapley, feddqc]
tags: [influence-function, llm-scale, gradient-projection, scalability]
---

# Influence functions at LLM scale

## The question

Classical [[concepts/influence-function|influence functions]] are gradient-based and theoretically clean, but were originally formulated for small convex models. **What does it take to compute them on billion-parameter LLMs trained on trillions of tokens?**

The wiki has multiple sources that each take a different stab at this. The bottlenecks have shifted over time.

## The two-bottleneck story

For a model with $n$ parameters and a training dataset of size $|D|$, computing an influence-function-based attribution requires:

1. **Inverse Hessian-vector product (iHVP)** — $H^{-1} v$ where $H$ is approximately $n \times n$.
2. **Per-sample training gradients** — to attribute *all* training data fairly, you need gradients for every $z \in D$.

At LLM scale, the second bottleneck dominates. Computing gradients for all training data ≈ one epoch of training, easily $1M+ in compute. Re-running this for every batch of test queries is unaffordable.

## How methods address the bottlenecks

| Method | iHVP strategy | Per-sample gradient strategy | Where |
|---|---|---|---|
| Classical IF | LiSSA stochastic inversion + damping | naive per-sample gradient | [[sources/koh-liang-influence-functions]] |
| Hessian-free | identity (skip $H^{-1}$) | naive per-sample gradient | baseline in [[sources/datainf]] |
| **EKFAC influence** | Kronecker-factored Hessian approximation | naive per-sample gradient | strongest pre-2024 baseline |
| **DataInf** | swap inverse and average → closed form | naive per-sample gradient | [[sources/datainf]] |
| **TRAK** | eNTK linearization + random projection | random projection + multi-checkpoint averaging | [[sources/trak]] |
| **LoGra / Logix** | low-rank projection ($O(\sqrt{nk})$) | Kronecker-structured projection via LoRA-like layers | [[sources/logix]] |

### Key insights

- **DataInf** lives entirely in the iHVP bottleneck; doesn't accelerate per-sample gradient computation, but its closed form drops the iterative iHVP cost. Tightest for small layer dim — i.e., LoRA fine-tuning.
- **TRAK** and **LoGra** both attack *both* bottlenecks via gradient projection. TRAK uses random projections; LoGra uses Kronecker structure to make the projection itself $O(\sqrt{nk})$ instead of $O(nk)$. LoGra's storage trade — write 3.5 TB projected gradients to disk, read them as test queries arrive — converts the recurring compute cost into a one-time storage cost.
- **In-Run Shapley** sidesteps the bottlenecks differently: compute attribution *during* training, accumulating per-step values via Taylor-expanded marginal contributions. No after-training iHVP, no re-computing gradients.

## The damping = spectral sparsification observation

[[sources/logix|LoGra]]'s Lemma 1 makes a small but useful theoretical bridge:

$$\textsc{Influence}(x_{tr}, x_{te}) = \sum_i \frac{\lambda_i}{\lambda_i + \lambda} c_{tr,i} c_{te,i}$$

The damping term $\lambda$ acts as a **spectral gradient sparsifier** — small eigenvalues get penalized. Gradient projection is then framed as a *hard* version of the same idea — keep only top-$k$ components. This motivates PCA initialization of LoGra's projection matrices using KFAC eigenvectors.

Implication: damping and gradient projection are doing the same conceptual work. Choosing one is a regularization choice.

## Where it appears in the wiki

- [[sources/koh-liang-influence-functions]] — foundational: showed IF works on non-convex CNNs.
- [[sources/datainf]] — algebraic shortcut tuned for LoRA.
- [[sources/logix]] — structured projection; ~6,500× over EKFAC at Llama3-8B scale.
- [[sources/trak]] — eNTK linearization; established LDS as the standard benchmark.
- [[sources/in-run-data-shapley]] — alternative attribution route that avoids the IF bottleneck altogether.

## Open questions

- **Cross-method calibration**: TRAK, LoGra, DataInf, EKFAC, and In-Run Shapley don't agree on which points are good/bad in general. On which categories do they diverge?
- **Real-world FL**: [[sources/feddqc|FedDQC]] reports DataInf failing on Fed-WildChat (heterogeneous client data). Do TRAK / LoGra / EKFAC also fail in that regime, or is the brittleness specific to DataInf's swap-inverse-and-average approximation?
- **Datamodels and Shapley**: TRAK's eNTK approximation to [[concepts/datamodels]] and the Shapley game-theoretic relationship suggest a deeper unification. Is there a single scalable framework that recovers all three?
- **Beyond LoRA**: most LLM influence work assumes LoRA fine-tuning. Pretraining-scale attribution is partially demonstrated by [[sources/in-run-data-shapley|In-Run Shapley]] (GPT-2, Pythia-410M) but full Llama-pretraining-scale attribution is still open.
- **PCA on Transformers**: [[sources/logix|LoGra]]'s PCA initialization underperforms on GPT-2/WikiText. Is there a Transformer-specific KFAC variant that fixes this?

## Sources to ingest next

- Bae et al., "If Influence Functions are the Answer, Then What is the Question?" — argues classical IF is brittle on deep nets and the actual quantity is a "proximal Bregman response."
- Original Datamodels paper (Ilyas et al., 2022) — the gold-standard counterfactual baseline that TRAK approximates.
- Original EKFAC paper (Grosse et al.) — currently in the wiki only as a baseline.
