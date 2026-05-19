---
type: concept
title: FedDQC
created: 2026-05-05
updated: 2026-05-05
sources: [feddqc]
tags: [federated-learning, instruction-tuning, data-quality, lora]
---

# FedDQC

A federated instruction-tuning framework with on-device data quality control. Two innovations:

1. **IRA (Instruction–Response Alignment)** — a privacy-preserving quality metric: $f_{\text{IRA}} = L(a^i; \theta) - L((a^i, q^i); \theta)$, the difference between unconditional and instruction-conditional inference loss. Approximates mutual information between instruction and response. Cheap (~1% of training time per scoring round), no data leaves the client.
2. **Hierarchical training** — sort local data by IRA, partition into $K$ hierarchies, train round-by-round from highest-IRA (easy) to lowest-IRA (hard). Re-score before each hierarchy with the evolving global model.

Beats DataInf, IFD, NUGGETS, perplexity baselines on synthetic and real-world (Fed-WildChat) FL instruction-tuning. Also reports DataInf-based attribution failing on Fed-WildChat — a useful negative result for gradient-based methods on heterogeneous FL.

See [[sources/feddqc]] for full details.

## See also

- [[concepts/federated-learning]]
- [[concepts/data-quality-control]]
- [[concepts/lora]]
