---
type: thread
title: Retraining-based vs. in-run / single-run attribution
created: 2026-05-05
updated: 2026-05-22
sources: [in-run-data-shapley, data-banzhaf, ripple-shapley, asymmetric-data-shapley, koh-liang-influence-functions, ghorbani-zou-data-shapley, trak, grosse-llm-influence, less, mates, dsdm]
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
- [[sources/dsdm]] — fits **linear datamodels via TRAK** at LLM scale, selects bottom-$k$ for training. Operationally retraining-based (the datamodels are themselves fit from many retrainings), with TRAK as the cost-amortizer.

### On the in-run side

- [[sources/in-run-data-shapley]] — formalizes model-level Shapley via per-step Taylor expansion; demonstrated at GPT-2 / Pythia-410M scale.
- [[sources/ripple-shapley]] — extends in-run Shapley to **federated** trajectories with cross-round Jacobian-chain propagation. Sample-level, single-run.
- [[sources/asymmetric-data-shapley]] — anchors evaluation to the realized model state, in the same spirit as in-run, even though it isn't gradient-based.
- [[sources/koh-liang-influence-functions]] etc. — gradient-based methods are technically "single-trained-model" already, so they belong on this side too.
- [[sources/less]] — explicitly trajectory-summed TracIn-style IF over LoRA-warmup checkpoints. The aggregation is over a *fixed* realized trajectory, not over algorithm randomness.
- [[sources/mates]] — locally-probed oracle IF: one-step retraining-from-current-state probe. Sits between in-run (uses the current $\mathcal{M}_t$, not a final-trained model) and retraining-based (the probe is a one-step retraining). The BERT-base influence model then *interpolates* the oracle across the corpus. Closest cousin to [[sources/in-run-data-shapley|IRDS]]'s per-step utility framing among the new 2024 papers.
- [[sources/grosse-llm-influence]] — explicit reframing of IF target as the **[[concepts/proximal-bregman-response|PBRF]]**: *local* response around $\theta^s$, neither global retraining nor pure in-run accumulation. Argues the entire deep-net IF literature lives in this local-but-not-trajectory-summed regime.

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

- **Original Datamodels paper** (Ilyas et al., 2022) — gold standard for the retraining-based side. DsDm uses it but doesn't replace it.
- **TracIn** (Pruthi et al.) — bridges via per-checkpoint gradient dot products. LESS's direct ancestor.
- **Bae et al., "If Influence Functions are the Answer..."** — argues classical IF's actual quantity is the [[concepts/proximal-bregman-response|PBRF]] (already in the wiki via Grosse et al., but the originating paper would sharpen the concept page).

## A 2026 vantage point: three "local" attribution regimes

With the 4 new 2024 ingests, the in-run side is now more clearly differentiated into three flavors:

1. **Trajectory-summed** ([[sources/in-run-data-shapley|IRDS]], [[sources/ripple-shapley|Ripple Shapley]], [[sources/less|LESS]]) — sum per-step / per-checkpoint contributions along the realized training path.
2. **End-state local** ([[sources/grosse-llm-influence|Grosse]] EK-FAC, [[sources/datainf|DataInf]], [[sources/trak|TRAK]], [[sources/logix|LoGra]]) — evaluate at the trained $\theta^*$ / $\theta^s$ only, with the [[concepts/proximal-bregman-response|PBRF]] interpretation.
3. **One-step probe** ([[sources/mates|MATES]] locally-probed oracle) — explicit one-step retraining from current state as a counterfactual proxy. Cheapest counterfactual yet introduced.

All three approximate something *local* to the trained model — none are the true retraining counterfactual. They differ in *where* the locality is anchored (along the trajectory, at the end, or at a single step away).
