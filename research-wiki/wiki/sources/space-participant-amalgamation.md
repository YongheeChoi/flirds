---
type: source
title: "SPACE: Single-round Participant Amalgamation for Contribution Evaluation in Federated Learning"
created: 2026-05-05
updated: 2026-05-05
topic: flirds
tags: [federated-learning, shapley, single-round, knowledge-distillation, prototype-evaluation]
---

# SPACE — Single-round Participant Amalgamation

## Citation

Yi-Chung Chen, Hsi-Wen Chen, Shun-Gui Wang, Ming-Syan Chen (NTU Taiwan). *SPACE: Single-round Participant Amalgamation for Contribution Evaluation in Federated Learning*. NeurIPS 2023.

Raw: `raw/papers/flirds/NeurIPS-2023-space-single-round-participant-amalgamation-for-contribution-evaluation-in-federated-learning-Paper-Conference.pdf`

## TL;DR

Computes federated-Shapley participant contributions using **a single communication round** plus knowledge distillation. Clients act as teachers distilling into a server-side amalgamated model; participant contributions are then evaluated on per-class **prototype embeddings** rather than a held-out validation set, eliminating both multi-round overhead and validation-set-size dependence.

## Problem

Existing federated-Shapley schemes have two recurring costs:

- **Multi-round** evaluation (the FL training itself spans many rounds; computing Shapley per round is expensive).
- **Validation-set dependence**: utility is measured on a held-out set whose size and distribution affect the resulting Shapley values, and which has to be available at the server.

SPACE eliminates both.

## Method

### Federated Knowledge Amalgamation (FKA)

Each client trains its local model on its own data, then transmits the *model* (rather than gradients per round). The server distills these client models into a single **amalgamated** server model in one shot, using each client model as a teacher. The server doesn't need a labeled validation set for the distillation — it uses the clients' own outputs as targets.

### Prototype-based Model Evaluation (PME)

Compute, for each class, a prototype embedding (mean of the embedded inputs assigned to that class). A coalition's "utility" is then a function of how well the embedded predictions match the prototypes — a property of the coalition's model that doesn't depend on having a separate validation set. Side benefit: prototype evaluation is invariant to validation-set size.

### Modified utility / satisfaction function

The paper introduces a logistic-style satisfaction function on top of the prototype evaluation, ensuring the resulting Shapley values have desirable monotonicity / saturation properties.

### Single-round Shapley

With FKA + PME, computing client coalition utilities is cheap and one-shot, so per-round Shapley becomes tractable in just one round.

## Key results

- Substantial reduction in communication and compute compared to multi-round federated-Shapley methods like [[sources/gtg-shapley|GTG-Shapley]].
- Prototype-based evaluation gives stable Shapley rankings across validation-set sizes (whereas standard validation-based Shapley is sensitive).
- Applications: client reweighting, client selection, contribution-based incentive design.

## Connections

- Federated-Shapley line: shares the goal of "single-run participant contribution" with [[sources/ripple-shapley|Ripple Shapley]] but uses a different mechanism (knowledge distillation vs. Jacobian propagation; client-level vs. sample-level).
- Compares with [[sources/gtg-shapley|GTG-Shapley]] (multi-round, sub-model reconstruction): SPACE is single-round, distillation-based.
- Compares with [[sources/shapleyfl|ShapleyFL]] (surrogate Shapley aggregated across rounds): SPACE eliminates the multi-round altogether.
- Concept page: [[concepts/space]], [[concepts/federated-knowledge-amalgamation]], [[concepts/prototype-based-evaluation]] (created with this batch).
- Belongs to thread [[threads/federated-and-decentralized-attribution]].

## Notes / open questions

- Prototype-based evaluation works well for classification with discrete classes. How does it extend to generative tasks, regression, or contrastive losses where there are no clean class labels?
- Knowledge amalgamation assumes clients are willing to send their full models, which has different privacy properties than gradient-only protocols. Worth a thread on "what's actually private in FL Shapley."
- Single-round vs. multi-round Shapley as different objects: SPACE's single-round Shapley is taken at the *end* of training; multi-round captures the *trajectory*. They're not necessarily comparable. Worth a clarification on the wiki side.
- The satisfaction function is a design choice that affects the monotonicity of resulting values; tuning sensitivity isn't explored deeply.
