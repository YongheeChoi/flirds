---
type: source
title: "DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models"
created: 2026-05-05
updated: 2026-05-05
topic: flirds
tags: [influence-function, lora, llm-attribution, closed-form, diffusion]
---

# DataInf

## Citation

Yongchan Kwon (Columbia), Eric Wu, Kevin Wu, James Zou (Stanford). *DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models*. ICLR 2024 (arXiv:2310.00902 v3).

Raw: `raw/papers/flirds/DataInf_ Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models.md`

## TL;DR

A closed-form approximation for the influence function that swaps the order of matrix-inversion and averaging in the inverse-Hessian step. The result is computable in $O(nDL)$ time and $O(D)$ memory — orders of magnitude cheaper than LiSSA — and is **especially well-suited to LoRA fine-tuning** because the approximation error scales as $O(d_l^2)$ (small for small layers).

## Problem

The classical [[concepts/influence-function|influence function]] $-H^{-1}\nabla_\theta \ell_k$ is gradient-based and avoids retraining, but the inverse Hessian is the dominant cost. For deep networks at LLM scale:

- $H$ is rank-deficient (parameter count exceeds sample count), so it's only invertible after damping.
- Block-diagonal-by-layer + damping is the standard fix, giving the per-layer formula
  $$\text{IF}_l(x_k) = -v_l^\top \Big(G_l(\theta^*) + \lambda_l I\Big)^{-1} \nabla_{\theta_l}\ell_k.$$
- Even after damping + block diagonalization, computing this per layer is too expensive at LLM scale; standard methods (LiSSA, EKFAC, eigendecompositions) still cost $O(nD^2L)$ or require expensive iterative updates / multiple model trainings.

## Method

### The key trick

Swap inversion and averaging:

$$\Big(\frac{1}{n}\sum_i \nabla\ell_i \nabla\ell_i^\top + \lambda_l I\Big)^{-1} \;\approx\; \frac{1}{n}\sum_i \Big(\nabla\ell_i \nabla\ell_i^\top + \lambda_l I\Big)^{-1}$$

Each term on the right is a rank-1-plus-diagonal matrix, invertible analytically via Sherman–Morrison:

$$\Big(\nabla\ell_i \nabla\ell_i^\top + \lambda_l I\Big)^{-1} = \frac{1}{\lambda_l}\Big(I - \frac{\nabla\ell_i \nabla\ell_i^\top}{\lambda_l + \|\nabla\ell_i\|^2}\Big)$$

Plugging back yields a closed form for $\mathcal{I}_{\mathrm{DataInf}}(x_k)$ that touches only gradient inner products $\nabla\ell_i^\top \nabla\ell_j$ (scalars) — no Hessian matrices or iterative inversion.

### Cost

| Method | Compute | Memory | Hessian |
|---|---|---|---|
| Exact | $O(nD^2 L + D^3 L)$ | $O(D^2)$ | matrix inversion |
| LiSSA | $O(nD^2 L)$ | $O(D^2)$ | iterative |
| **DataInf** | $O(nDL)$ | $O(D)$ | closed form |

### Why it's good for LoRA

Theorem 1: the approximation error has spectral norm bounded by $O(d_l^2)$ when gradients and $\lambda_l$ are bounded. **The bound is tight precisely when $d_l$ is small** — i.e., for parameter-efficient fine-tuning like LoRA where each adapted layer has a small rank.

## Key results

- **Approximation error analysis** (Pearson correlation with exact IF on noisy GLUE): DataInf $\approx 0.64$ vs. LiSSA $0.45$ vs. Hessian-free $0.50$ at LoRA rank $r=1$ on MRPC. Correlation degrades as rank grows, matching theory.
- **Mislabeled-data detection**: outperforms LiSSA and Hessian-free on RoBERTa-large with LoRA on noisy GLUE.
- **Influential-data identification**: applied to **Llama-2-13B-chat** and **stable-diffusion-v1.5** — qualitatively retrieves examples that resemble the test query better than baselines.
- Open-source: `https://github.com/ykwon0407/DataInf`.

## Connections

- Variant of the [[concepts/influence-function]] — same computational target ($g_{te}^\top H^{-1} g_{tr}$), different approximation strategy.
- Contrasts with [[sources/logix|LoGra/Logix]]: DataInf is a closed-form *algebraic* approximation; LoGra is a *projection*-based approximation. Both shoot for LLM scale via different routes — see [[threads/influence-functions-at-llm-scale]] (created with this ingest).
- Used as a baseline in [[sources/feddqc|FedDQC]] (DataInf-based selection underperforms IRA in real-world FL data; FedDQC paper notes "gradients for data attribution in real-world datasets… is challenging").
- Concept page: [[concepts/datainf]] (created).
- Connects to [[concepts/lora]] (created with this ingest) — the method's effectiveness depends on LoRA's small adapted-layer dimension.

## Notes / open questions

- The "swap inverse and average" trick is generic — does it apply to non-IF objects? Could be useful for other estimators that need $\mathbb{E}[X^{-1}]$.
- The bound is $O(d_l^2)$ in spectral norm, but empirical correlation only mildly degrades from rank 1 to 4. Tighter empirical scaling than the bound suggests?
- Real-world FL data: DataInf-based selection in [[sources/feddqc|FedDQC]] underperformed even random selection. Hypothesis: gradients on heterogeneous client data are noisier than the homogeneous IID setting DataInf was tested on. Worth a thread.
- Diffusion models: the paper's eval on stable-diffusion-v1.5 is qualitative. Quantitative attribution metrics for diffusion remain underdeveloped.
