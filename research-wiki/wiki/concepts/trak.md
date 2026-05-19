---
type: concept
title: TRAK
created: 2026-05-05
updated: 2026-05-05
sources: [trak]
tags: [attribution, ntk, random-projection, datamodels, llm-scale]
---

# TRAK

A scalable data-attribution method that linearizes the model around its trained parameters using the **empirical Neural Tangent Kernel (eNTK)**, then applies random gradient projections to compute attribution scores cheaply. Multi-checkpoint averaging reduces variance across random initializations.

Effectively: a **closed-form approximation to [[concepts/datamodels]]** that achieves comparable counterfactual fidelity (Linear Datamodeling Score) at 100–1000× lower cost. Demonstrated on ImageNet (ResNet, CLIP), BERT, mT5.

The Linear Datamodeling Score (LDS) introduced alongside TRAK is now the standard data-attribution benchmark.

See [[sources/trak]] for full details.

## See also

- [[concepts/datamodels]]
- [[concepts/influence-function]]
- [[concepts/linear-datamodeling-score]]
- [[threads/influence-functions-at-llm-scale]]
