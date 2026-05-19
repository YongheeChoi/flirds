---
type: concept
title: DataInf
created: 2026-05-05
updated: 2026-05-05
sources: [datainf]
tags: [influence-function, lora, closed-form]
---

# DataInf

A closed-form approximation of the [[concepts/influence-function]] that swaps the order of matrix-inversion and averaging:

$$\Big(\frac{1}{n}\sum_i \nabla\ell_i \nabla\ell_i^\top + \lambda I\Big)^{-1} \;\approx\; \frac{1}{n}\sum_i \Big(\nabla\ell_i \nabla\ell_i^\top + \lambda I\Big)^{-1}$$

Each term on the right is rank-1-plus-diagonal, invertible analytically via Sherman–Morrison. Result: $O(nDL)$ compute, $O(D)$ memory, no iterative inversion.

The approximation is tight when $d_l$ is small — i.e., LoRA-friendly. Empirically beats LiSSA and Hessian-free on noisy GLUE; applied to RoBERTa, Llama-2-13B-chat, stable-diffusion-v1.5.

[[sources/feddqc|FedDQC]] reports DataInf failing on real-world heterogeneous FL data ([Fed-WildChat](https://example.com)) — open question whether DataInf's IID assumption is the cause.

See [[sources/datainf]] for full details.

## See also

- [[concepts/influence-function]]
- [[concepts/lora]]
- [[threads/influence-functions-at-llm-scale]]
