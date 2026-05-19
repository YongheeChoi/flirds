---
type: concept
title: LoGra (Low-rank Gradient projection)
created: 2026-05-05
updated: 2026-05-05
sources: [logix]
tags: [influence-function, gradient-projection, llm-scale, kronecker]
---

# LoGra

A low-rank gradient-projection algorithm for scaling [[concepts/influence-function|influence functions]] to LLMs. Exploits the Kronecker structure of layer gradients in backprop:

$$P\,\text{vec}(\mathcal{D}W) = (P_i \otimes P_o)\,\text{vec}(\mathcal{D}W) = \sum_t P_i x_{i,t} \otimes P_o \mathcal{D}x_{o,t}$$

Projects forward and backward activations *separately* into low-dim spaces, reconstructs the projected gradient as their Kronecker product. Cost: $O(\sqrt{nk})$ for projection (instead of $O(nk)$). Implementable as small LoRA-like add-on layers — encoder, zero-init bottleneck, decoder — exploiting standard autograd for per-sample projected gradients.

PCA-initialized variant uses KFAC eigenvectors; theoretically motivated as a hard analog of damping's spectral sparsification.

Achieves ~6,500× throughput over EKFAC on Llama3-8B + 1B OpenWebText tokens.

See [[sources/logix]] for full details.

## See also

- [[concepts/logix]]
- [[concepts/influence-function]]
- [[concepts/lora]]
- [[threads/influence-functions-at-llm-scale]]
