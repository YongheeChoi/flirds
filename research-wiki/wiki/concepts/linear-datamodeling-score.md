---
type: concept
title: Linear Datamodeling Score (LDS)
created: 2026-05-05
updated: 2026-05-05
sources: [trak, logix]
tags: [evaluation, benchmark, attribution]
---

# Linear Datamodeling Score (LDS)

A standard evaluation metric for data-attribution methods, introduced by TRAK. Measures how well a method's per-sample attribution scores **linearly predict counterfactual model behavior** under subset-restricted retrainings.

Procedure:

1. Sample many subsets $\{S_m\} \subseteq D$, each of fixed size (e.g., $|D|/2$).
2. For each subset, retrain a model and observe the model output on a fixed set of test points.
3. For each subset, predict the model output by *summing* the attribution scores of the points in $S_m$ (additivity assumption).
4. Compute the **Spearman correlation** between predicted and observed output values across subsets.

High LDS = attribution scores faithfully linearly predict counterfactual model behavior. By construction, [[concepts/datamodels]] score near 1.0; cheaper methods like influence functions, [[sources/datainf|DataInf]], [[sources/logix|LoGra]] are evaluated *against* this benchmark.

Connects to [[concepts/data-shapley]]: linear datamodels and the Shapley value have a known game-theoretic relationship (cited in [[sources/logix]]).

## See also

- [[concepts/datamodels]]
- [[concepts/influence-function]]
- [[sources/trak]]
