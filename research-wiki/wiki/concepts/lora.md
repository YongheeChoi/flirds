---
type: concept
title: LoRA (Low-Rank Adaptation)
created: 2026-05-05
updated: 2026-05-22
sources: [datainf, logix, feddqc, rfedlr, less]
tags: [parameter-efficient-fine-tuning, peft, fine-tuning]
---

# LoRA — Low-Rank Adaptation

## One-liner

Parameter-efficient fine-tuning: instead of updating a model's full weight matrices, freeze the original weights and learn a low-rank additive update $\Delta W = AB$ where $A \in \mathbb{R}^{n \times r}$, $B \in \mathbb{R}^{r \times m}$, $r \ll \min(n, m)$.

## Why it matters for data attribution

Three reasons LoRA appears repeatedly in this wiki:

1. **Tractable per-sample gradients**: the only learnable parameters are the small $A, B$ matrices (rank ~4–32). Per-sample gradient computation costs and influence-function approximation errors scale with this small dimension.
   - **[[sources/datainf|DataInf]]**: its $O(d_l^2)$ approximation-error bound is small precisely because LoRA layers have small $d_l$.
   - **[[sources/logix|LoGra/Logix]]**: the projection-layer architecture is essentially LoRA — "encoder, zero-init bottleneck, decoder" matches LoRA's structure.

2. **Realistic LLM fine-tuning workflow**: most LLM personalization in practice uses LoRA. Attribution methods designed for LoRA are immediately deployable; methods that need full-parameter gradients are not.

3. **Federated compatibility**: client compute in FL is constrained. LoRA reduces the per-client gradient and update size, enabling federated LLM fine-tuning in principle.
   - **[[sources/feddqc|FedDQC]]**: uses LLaMA-2-7B + LoRA on each client.
   - **[[sources/rfedlr]]**: federated LoRA robustness against label noise.

## The architecture

For a frozen weight $W_0 \in \mathbb{R}^{n \times m}$ at fine-tuning, replace
$$h = W_0 x \quad \text{with} \quad h = W_0 x + (BA) x \;=\; W_0 x + B(Ax)$$

Train only $A, B$. At inference, $W_0 + BA$ can be merged for zero overhead.

## Strengths

- 1000× fewer trainable parameters than full fine-tuning for typical $r$.
- Compositional: multiple LoRAs can be merged or swapped.
- Plug-in for most Transformer architectures.

## Limitations

- The low-rank assumption may underfit for tasks far from the base model's distribution.
- Choice of $r$ is a hyperparameter; see [[sources/datainf|DataInf]]'s observation that approximation accuracy degrades with $r$.
- Original layers stay frozen — pre-training-induced biases / errors aren't easily corrected.

## Where it appears in the wiki

- [[sources/datainf]] — DataInf's tightness depends critically on small LoRA rank.
- [[sources/logix]] — LoGra's add-on architecture mirrors LoRA's structure.
- [[sources/feddqc]] — federated instruction tuning uses LoRA on each client.
- [[sources/rfedlr]] — federated LoRA with robustness focus.
- [[sources/less]] — uses LoRA warmup training so that per-example gradient features fit a *reusable* 8192-dim datastore (JL-projected from LoRA dim) for instruction-tuning data selection at Llama-2-7B/13B and Mistral-7B scale.

## See also

- [[concepts/influence-function]]
- [[concepts/federated-learning]]
- [[threads/data-selection-for-llms]]
