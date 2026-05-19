---
type: concept
title: DICE — Data Influence Cascade
created: 2026-05-05
updated: 2026-05-05
sources: [dice]
tags: [decentralized-learning, influence-cascade, topology]
---

# DICE

The first influence-estimation framework for [[concepts/decentralized-learning|decentralized learning]]. Defines and approximates a node's data influence as it cascades through the communication topology.

Key formula (Theorem 1, schematic):
$$\text{DICE-E}_k \;\propto\; \underbrace{\nabla_\theta L_k}_{\text{data}} \times \underbrace{f(\text{topology}, k)}_{\text{position}} \times \underbrace{\prod_j H_j}_{\text{intermediate curvature}}$$

Reduces to centralized attribution as a star-topology special case. See [[sources/dice]] for full details.

## See also

- [[concepts/decentralized-learning]]
- [[threads/federated-and-decentralized-attribution]]
