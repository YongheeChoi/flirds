---
type: source
title: "What is Your Data Worth to GPT? LLM-Scale Data Valuation with Influence Functions"
created: 2026-05-05
updated: 2026-05-05
topic: flirds
tags: [influence-function, gradient-projection, llm-attribution, kfac, software, lora-like]
---

# LoGra / Logix — LLM-Scale Influence Functions

## Citation

Sang Keun Choe, Hwijeen Ahn, Juhan Bae, Kewen Zhao, Minsoo Kang, Youngseog Chung, Adithya Pratapa, Willie Neiswanger, Emma Strubell, Teruko Mitamura, Jeff Schneider, Eduard Hovy, Roger Grosse, Eric Xing (CMU, U. of Toronto, Georgia Tech, USC, MBZUAI). *What is Your Data Worth to GPT? LLM-Scale Data Valuation with Influence Functions*. arXiv:2405.13954 (v1).

Raw: `raw/papers/flirds/What is Your Data Worth to GPT_ LLM-Scale Data Valuation with Influence Functions.md`

## TL;DR

Two contributions. **LoGra**: a low-rank gradient-projection algorithm that exploits Kronecker structure in backprop, reducing per-sample-gradient and projection costs from $O(nk)$ to $O(\sqrt{nk})$. **Logix**: a software package built on PyTorch hooks that converts existing training code into data-valuation code with minimal changes. On Llama3-8B-Instruct + 1B OpenWebText tokens, achieves ~6,500× throughput and 5× memory reduction over EKFAC influence — the only previously-runnable baseline at that scale.

## Problem

[[concepts/influence-function|Influence functions]] for LLMs face two scalability bottlenecks:

1. **Inverse Hessian-vector product (iHVP)**: naive $O(n^3)$ time, $O(n^2)$ memory in model dimension $n$. Iterative methods (LiSSA) and EKFAC approximation help but remain expensive.
2. **Per-sample gradient computation**: to value all training data fairly, you need every training point's gradient. At LLM scale this approaches "one epoch of training," easily $1M+ in compute. Re-running this for every batch of test queries is unaffordable.

Prior projection-based methods (Arnoldi IF, TRAK) project gradients to a low-dim subspace to make iHVP cheap and to enable storing projected training gradients on disk. But their **projection itself** costs $O(bnk)$ time and $O(kn)$ memory — for an 8B model with reasonable $k$, the projection matrix alone is **128 TB**. They get around this with restrictively small $k$ and $b$, sacrificing accuracy.

## Method

### LoGra — Kronecker-structured projection

For a linear layer $x_o = W x_i$, the gradient $\text{vec}(\mathcal{D}W) = \sum_t x_{i,t} \otimes \mathcal{D}x_{o,t}$ is a sum of Kronecker products. **Impose Kronecker structure on the projection matrix too**:

$$P\,\text{vec}(\mathcal{D}W) = (P_i \otimes P_o)\,\text{vec}(\mathcal{D}W) = \sum_t (P_i x_{i,t}) \otimes (P_o \mathcal{D}x_{o,t})$$

Project forward and backward activations *separately* into low-dim spaces, then reconstruct the projected gradient by Kronecker product. Costs reduce to $O(\sqrt{nk})$ for both projection and per-sample gradient computation.

For an 8B model with $k=4{,}096$: traditional projection matrix = 128 TB → LoGra projection matrix = **~1 GB**.

### LoRA-like architecture

LoGra can be implemented as small add-on layers — encoder ($P_i$), zero-initialized bottleneck, decoder ($P_o$). The architecture is essentially LoRA. Two practical wins:

1. **Zero bottleneck init** ⇒ forward and backward computations of the original network are unaffected.
2. **Per-sample gradients** can be obtained by computing gradients only for the bottleneck layer using standard autograd — no custom kernels.

### Theory: damping = spectral sparsification

Lemma 1: with damping $\lambda$, $\textsc{Influence}(x_{tr}, x_{te}) = \sum_i \frac{\lambda_i}{\lambda_i + \lambda} c_{tr,i} c_{te,i}$. Damping softly limits the number of components contributing to the influence (penalizes small eigenvalues). Gradient projection is interpreted as a *hard* version of the same idea — keep only top-$k$ components.

This motivates **PCA initialization**: initialize $P_i, P_o$ from the top-$k$ eigenvectors of KFAC-approximated Hessian factors $C_F, C_B$. Approximately keeps the largest curvature components.

### Logix software

Built on PyTorch hooks. Key benefits:

- **Compatible** with FSDP, autocast, compile, HF Transformers, DeepSpeed.
- **Convertible**: existing training code becomes valuation code via a context manager around the training loop.
- **Extensible**: custom statistics via hooks.
- Code: `https://github.com/logix-project/logix`.

## Key results

### Accuracy (small-scale)

Brittleness test + Linear Datamodeling Score (LDS) on MLP/FMNIST, ResNet-9/CIFAR-10, GPT-2/WikiText:

- LoGra slightly underperforms EKFAC influence on LDS but **noticeably outperforms** TRAK, gradient dot product, representation similarity.
- LoGra-PCA > LoGra-random on FMNIST, CIFAR-10. Tie/loss on WikiText (Transformer architecture lacks specialized KFAC formulation; PCA init suffers).

### Efficiency (large-scale)

Llama3-8B-Instruct + 1B OWT tokens, A100 80GB, fp16:

| | Logging memory | Logging throughput | Influence memory | Influence throughput |
|---|---|---|---|---|
| **EKFAC** | 71/80 GB | 1740/419 tok/s | 75 GB | 12.2 pairs/s |
| **LoGra (b=1)** | 23 GB | 3,430 tok/s | 14 GB | 1,599.6 pairs/s |
| **LoGra (b=16)** | 79 GB | 4,696 tok/s | 15 GB | 79,003.9 pairs/s |

≈ **6,500×** throughput improvement, 5× memory reduction. EKFAC for 256 test × 1B-token training would need 11,300 A100-hours — LoGra makes it tractable.

Storage trade-off: LoGra writes 3.5 TB of projected gradients to disk (vs. EKFAC's 89 GB). Storage is cheap; compute is expensive — net win.

## Connections

- The other major LLM-scale influence-function approach alongside [[sources/datainf|DataInf]]. Both target the same problem with different strategies (algebraic closed form vs. structured projection). Belongs to thread [[threads/influence-functions-at-llm-scale]].
- LoGra's PCA init draws on EKFAC; the spectral-sparsification framing suggests deeper unification with damping.
- Logix software provides reusable infrastructure that future ingestion (e.g. Datamodels, TRAK papers) should reference.
- Concept page: [[concepts/logra]] (created), [[concepts/logix]] (created).
- Concept page: [[concepts/influence-function]] (created with this ingest, since multiple papers now use it).

## Notes / open questions

- The 6,500× number is impressive but mostly comes from avoiding gradient re-computation (storage trade) — the projection itself is "only" 4× faster. Disentangle these contributions cleanly?
- LoGra-PCA's poor performance on Transformer (GPT-2) is concerning — the headline architecture for LLM data valuation. The authors fall back to LoGra-random for Llama. Is there a Transformer-specific KFAC variant that fixes this?
- Comparison to [[sources/datainf|DataInf]] head-to-head: both target LoRA / parameter-efficient fine-tuning; both should cover broadly the same LLM use cases. The papers don't compare directly.
- The damping = spectral sparsification framing might extend beyond influence functions — to Shapley-style computations? Curiously underexplored.
- Compute cost vs. storage cost: the 3.5 TB storage assumption is fine on a single workstation but breaks if you want valuation as a service across many models.
