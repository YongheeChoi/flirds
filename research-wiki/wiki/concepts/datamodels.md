---
type: concept
title: Datamodels
created: 2026-05-05
updated: 2026-05-22
sources: [trak, in-run-data-shapley, ghorbani-zou-data-shapley, dsdm]
tags: [attribution, retraining-based, counterfactual, lds]
---

# Datamodels

## One-liner

A retraining-based, *counterfactually faithful* data-attribution framework: train many models on different subsets of the training data, then fit a (typically linear) surrogate that predicts each model's behavior from the indicator vector of which data points were included.

## Setup (Ilyas et al. 2022)

For training set $D$ and test point $z_{\text{te}}$:

1. Sample many random subsets $S_1, S_2, \ldots, S_M \subseteq D$ (each of size $\alpha n$).
2. Train a model $\theta_m = \mathcal{A}(S_m)$ on each.
3. Record the model's behavior $f(\theta_m; z_{\text{te}})$ on the test point.
4. Fit a linear surrogate $f \approx \beta^\top \mathbf{1}_{S}$ where $\mathbf{1}_S$ is the indicator that $z_i \in S$.

The fitted coefficients $\beta_i$ are the **datamodel attribution** of $z_i$ to the test point.

## Why it's the gold standard

By construction, datamodels are **counterfactually faithful** — they're trained to predict how the model behavior actually changes under data subsetting. Any other attribution method (influence functions, Shapley, gradient products) is a shortcut that *approximates* what datamodels measure directly.

This is also why datamodels are expensive: hundreds to thousands of full retrainings.

## The Linear Datamodeling Score (LDS)

A standard evaluation metric for any data-attribution method:

1. Hold out a subset $S$ of the training data.
2. Retrain on $D \setminus S$, observe behavior change.
3. Predict the behavior change by summing the attribution scores in $S$.
4. Spearman correlation between predicted and observed = LDS.

LDS asks: "does the attribution method linearly predict counterfactual model behavior?" Datamodels score near 1.0 by construction; cheaper methods are evaluated against this.

## Where it appears in the wiki

- [[sources/trak]] — eNTK linearization is essentially a closed-form approximation to datamodels at substantially lower cost. Establishes LDS as the standard benchmark.
- [[sources/logix]] — uses LDS to compare LoGra against EKFAC and TRAK on small-scale tasks.
- [[sources/in-run-data-shapley]] — different framework (Shapley axioms instead of linear surrogate) but pursuing the same goal of "predict counterfactual model behavior at near-training cost."
- [[sources/ghorbani-zou-data-shapley]] — Shapley as another principled retraining-based attribution; the two families converged via TRAK.
- [[sources/dsdm]] — the **Datamodels → LLM bridge**: fits linear datamodels via TRAK at LM-pretraining scale (1.3B), then selects the bottom-$k$ entries of the averaged coefficient vector. First operational demonstration that linear datamodels still work as a selection primitive at billion-parameter scale.

## See also

- [[concepts/influence-function]]
- [[concepts/trak]]
- [[concepts/linear-datamodeling-score]]
- [[threads/retraining-vs-in-run-attribution]]
- [[threads/data-selection-for-llms]]

> TODO: ingest the original Ilyas et al. *Datamodels* paper if Yonghee adds it to raw/. (DsDm uses it but doesn't replace it.)
