---
type: thread
title: Retraining-based vs. in-run / single-run attribution
created: 2026-05-05
updated: 2026-05-05
sources: [in-run-data-shapley, data-banzhaf, ripple-shapley, asymmetric-data-shapley, koh-liang-influence-functions, ghorbani-zou-data-shapley, trak]
tags: [attribution, scalability, semantics, model-vs-algorithm]
---

# Retraining-based vs. in-run attribution

## The question

Data attribution methods come in two flavors:

- **Retraining-based** — define value via a counterfactual: "what does the model become if I remove (or add, or up-weight) this point?" Implemented by retraining on subsets. Examples: LOO, [[sources/ghorbani-zou-data-shapley|Data Shapley]], [[concepts/datamodels|Datamodels]].
- **Gradient-based / in-run** — define value via an analytic approximation that uses the trained model's gradients/Hessian, without retraining. Examples: [[sources/koh-liang-influence-functions|influence functions]], TracIn, [[sources/datainf|DataInf]], [[sources/in-run-data-shapley|In-Run Shapley]], [[sources/ripple-shapley|Ripple Shapley]].

This looks like an efficiency trade-off ("retraining is the gold standard, gradient is a cheap approximation"), but the deeper claim — pushed by [[sources/in-run-data-shapley|In-Run Shapley]] — is that **they measure conceptually different quantities**, and the in-run version is sometimes the one we actually want.

## Two distinct quantities

### Algorithm-level value (retraining-based)

$$\phi^{\text{alg}}_i \;:=\; \mathbb{E}_{\text{algorithm randomness}}\big[\, \text{contribution of } i \,\big]$$

Averages over initialization, batch order, dropout. Asks: *if I were to train models on subsets including/excluding $i$, what's $i$'s expected contribution?*

### Model-level / trajectory-level value (in-run)

$$\phi^{\text{model}}_i \;:=\; \text{contribution of } i \text{ to the specific model } \theta^* \text{ that was trained}$$

No averaging. Asks: *for the trained model in front of me, what did $i$ do?* Equivalently: what was $i$'s contribution to the **realized training trajectory**?

These can differ substantially under stochastic training — see [[threads/robustness-to-stochastic-training]].

## When does each matter?

| Application | Which value? | Reason |
|---|---|---|
| Data market pricing | algorithm-level | buyers pay for expected utility |
| Bad-data detection in production | model-level | clean *this* model's training set |
| Copyright attribution | model-level | who contributed to *this* model's outputs |
| Curation for future training runs | algorithm-level | future models will be different |
| Federated participant compensation | model-level (post-[[sources/asymmetric-data-shapley|ADS]]) — respect realized trajectory |
| Real-time data pricing in FL | model-level (single-run) | [[sources/ripple-shapley|Ripple Shapley]]'s use case |

## What the wiki has so far

### On the retraining-based side

- [[sources/koh-liang-influence-functions]] — first-order Taylor approximation to retraining-based attribution; pioneered the gradient-based shortcut.
- [[sources/ghorbani-zou-data-shapley]] — Data Shapley: literally retrains on subsets via TMC.
- [[sources/data-banzhaf]] — sticks with retraining-based (algorithm-level) but optimizes the semivalue weighting for noise-robustness.
- [[concepts/datamodels]] — gold-standard retraining-based; trains hundreds of models on subsets.
- [[sources/trak]] — eNTK linearization is a closed-form approximation to Datamodels.

### On the in-run side

- [[sources/in-run-data-shapley]] — formalizes model-level Shapley via per-step Taylor expansion; demonstrated at GPT-2 / Pythia-410M scale.
- [[sources/ripple-shapley]] — extends in-run Shapley to **federated** trajectories with cross-round Jacobian-chain propagation. Sample-level, single-run.
- [[sources/asymmetric-data-shapley]] — anchors evaluation to the realized model state, in the same spirit as in-run, even though it isn't gradient-based.
- [[sources/koh-liang-influence-functions]] etc. — gradient-based methods are technically "single-trained-model" already, so they belong on this side too.

## Where they meet

[[sources/asymmetric-data-shapley|ADS]]'s **state-conditioned marginal contribution** is conceptually identical to [[sources/in-run-data-shapley|In-Run Shapley]]'s per-step value: both anchor evaluation to the realized trajectory rather than averaging over hypothetical retrainings.

[[sources/ripple-shapley|Ripple Shapley]] takes this further — it makes the per-step structure recursive across federated rounds via Jacobian propagation. The wiki's clean lineage:

```
Koh & Liang (2017)         — gradient calculus, single trained model
Ghorbani & Zou (2019)      — Shapley over retraining counterfactuals
In-Run Shapley (2024)      — model-level Shapley via per-step Taylor expansion
Asymmetric Data Shapley   — drops symmetry; anchors to realized trajectory (axiomatic)
Ripple Shapley (2026)      — sample-level FL attribution via per-step + cross-round propagation
```

## Cost comparison (rough)

| Method | One-time | Per query | Notes |
|---|---|---|---|
| Exact Shapley | $O(2^n)$ retrainings | free after | toy only |
| MC Shapley (TMC) | many retrainings | free after | small/medium |
| MSR Banzhaf | $O(\log n)$ retrainings | free after | unique advantage of Banzhaf |
| Datamodels | hundreds–thousands of retrainings | free after | gold-standard counterfactual fidelity |
| Influence function (Koh-Liang) | one $H^{-1}$ + per-sample grads | $O(d)$ per query | $H^{-1}$ + per-sample grad bottlenecks |
| DataInf | small precompute | $O(d)$ per query | LoRA-friendly closed form |
| TRAK | per-checkpoint gradient + projection | sublinear | eNTK-linearized Datamodels |
| LoGra/Logix | gradient projection precompute | sublinear | $O(\sqrt{nk})$ projection |
| In-Run Shapley | none beyond training | $O(d)$ per query | computed *during* training |
| Ripple Shapley | trajectory log + Jacobian subspace | sublinear per round | sample-level FL, in-run |

## Open questions

- **Empirical calibration**: do retraining-based and in-run methods agree on rankings for the same model, dataset, and task? Where do they diverge?
- **Hybrid methods**: can in-run estimates be calibrated against a small retraining sample to get the best of both?
- **Attribution stability**: in-run values depend on the realized trajectory. How much does the value change across re-runs from different seeds — i.e., is "model-level" actually well-defined or run-specific in a way that's brittle?
- **In-Run Shapley vs. Influence Functions**: both are gradient-based and model-level. Are they computing the same quantity in the limit, or different objects? Is there a theorem connecting them?
- **Federated case**: [[sources/ripple-shapley|Ripple Shapley]] vs. [[sources/asymmetric-data-shapley|ADS]] — both anchor to realized trajectory; one is in-run-flavored, the other axiomatic. Are their rankings consistent?
- **TRAK vs. In-Run**: TRAK linearizes around the trained parameters (one model, end-state); In-Run accumulates per-step values along the trajectory. The difference is non-trivial — TRAK averages out trajectory dependence, In-Run keeps it.

## Sources to ingest next

- Original Datamodels paper (Ilyas et al., 2022) — gold standard for the retraining-based side.
- TracIn (Pruthi et al.) — bridges via per-checkpoint gradient dot products.
- Bae et al., "If Influence Functions are the Answer..." — argues classical IF's actual quantity is a "proximal Bregman response," not the retraining counterfactual.
