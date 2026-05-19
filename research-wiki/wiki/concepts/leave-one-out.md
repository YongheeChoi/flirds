---
type: concept
title: Leave-one-out (LOO) error
created: 2026-05-05
updated: 2026-05-05
sources: [data-banzhaf]
tags: [semivalue, baseline]
---

# Leave-one-out (LOO)

## One-liner

The simplest data-value notion: the change in model utility when point $i$ is removed from the training set. A degenerate semivalue and the cheapest baseline.

## Definition

$$\phi^{\text{LOO}}_i \;=\; U(D) - U(D \setminus \{i\})$$

As a [[concepts/semivalue]], LOO is the variant that places all weight on the single subset $S = D \setminus \{i\}$ and zero weight on all others.

## Pros

- Conceptually simple; one retraining per point.
- Easy to communicate.

## Cons

- **Smallest safety margin among semivalues** — most sensitive to SGD noise ([[sources/data-banzhaf]]).
- Misses interaction effects: a point that is redundant given many others but valuable in their absence is misvalued.
- Even at "one retraining per point" it is too expensive for large models.
- For LLMs, "remove this token" is not even well-defined the way "remove this point" is in classification.

## Where it appears in the wiki

- [[sources/data-banzhaf]] — used as a baseline; LOO has the worst noise-robustness of the semivalues compared.

## See also

- [[concepts/shapley-value]]
- [[concepts/banzhaf-value]]
