---
type: concept
title: EK-FAC (Eigenvalue-Corrected KFAC)
created: 2026-05-22
updated: 2026-05-22
sources: [grosse-llm-influence, logix]
tags: [hessian-approximation, kfac, influence-function, llm-scale]
---

# EK-FAC

## One-liner

Eigenvalue-corrected Kronecker-Factored Approximate Curvature — a block-diagonal Hessian approximation where each layer's curvature is parameterized as a Kronecker product $H_\ell \approx A_\ell \otimes S_\ell$ **and** the diagonal entries are independently eigenvalue-corrected in the rotated basis. Originally introduced by George et al. (2018) as a second-order optimization step direction; repurposed in 2023 by [[sources/grosse-llm-influence|Grosse et al.]] as the IHVP backbone for LLM-scale influence functions.

## Formal sketch

For a layer $\ell$ with input activations $a$ and pre-activation gradients $g$, KFAC writes
$$H_\ell \;\approx\; \mathbb{E}[a a^\top] \;\otimes\; \mathbb{E}[g g^\top] \;=\; A_\ell \otimes S_\ell.$$
EK-FAC eigendecomposes $A_\ell = U_A \Lambda_A U_A^\top$ and $S_\ell = U_S \Lambda_S U_S^\top$, then replaces the implied diagonal $\Lambda_A \otimes \Lambda_S$ (in the rotated basis $U_A \otimes U_S$) with the **empirical** diagonal of $H_\ell$ in that basis. The Kronecker rotation preserves structure; the eigenvalue correction tightens the per-direction scaling.

For IHVP: $H^{-1}v$ becomes a sum of $O(L)$ matrix-vector products with Kronecker-structured matrices — independent of LiSSA iteration count.

## Why it matters

- **Scales influence-function IHVP to 52B parameters** ([[sources/grosse-llm-influence|Grosse et al. 2023]]) — orders of magnitude faster than LiSSA while matching LiSSA's IF accuracy on a 22M validation model.
- The **strongest pre-2024 baseline** in the LLM influence-function literature. Later projection-based methods ([[sources/logix|LoGra]], [[sources/trak|TRAK]]) are typically compared against EK-FAC influence.
- [[sources/logix|LoGra]]'s PCA initialization for its projection matrices uses *KFAC eigenvectors* — meaning the projection family and EK-FAC share an algebraic backbone (damping ≡ spectral sparsification, eigenvectors = the natural basis).

## Limitations

- Block-diagonal-by-layer assumption: ignores cross-layer Hessian entries. Mostly a non-issue for IF because per-layer attribution dominates.
- **MLP-layer restriction** in Grosse et al.: only the MLP parameters are EK-FAC'd; attention layers are excluded for computational reasons. Limits the scope of "where in the network influence comes from" claims.
- Still expensive at LLM scale despite being faster than LiSSA — [[sources/logix|LoGra]] reports ~6,500× throughput over EK-FAC on Llama3-8B for IF computation. EK-FAC remains a **correctness anchor** rather than a deployment target.

## Where it appears in the wiki

- [[sources/grosse-llm-influence]] — the 52B LLM influence-function paper; EK-FAC's flagship use case.
- [[sources/logix]] — baseline comparison; LoGra's projection matrices are PCA-initialized from KFAC eigenvectors.
- [[sources/datainf]] — DataInf's swap-inverse-and-average is an *alternative* to EK-FAC for the IHVP; the two are different routes to the same goal.

## See also

- [[concepts/influence-function]]
- [[concepts/proximal-bregman-response]]
- [[threads/influence-functions-at-llm-scale]]
