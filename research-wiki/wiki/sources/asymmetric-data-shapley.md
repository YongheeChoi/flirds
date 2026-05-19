---
type: source
title: "Rethinking Data Value: Asymmetric Data Shapley for Structure-Aware Valuation in Data Markets and Machine Learning Pipelines"
created: 2026-05-05
updated: 2026-05-05
topic: flirds
tags: [shapley, asymmetry, data-market, federated-learning, llm-fine-tuning, axiomatic]
---

# Asymmetric Data Shapley (ADS)

## Citation

Xi Zheng (UW), Yinghui Huang & Xiangyu Chang (Xi'an Jiaotong), Ruoxi Jia (Virginia Tech), Yong Tan (UW). *Rethinking Data Value: Asymmetric Data Shapley for Structure-Aware Valuation in Data Markets and Machine Learning Pipelines*. arXiv:2511.12863 (v1).

Raw: `raw/papers/flirds/Rethinking Data Value_ Asymmetric Data Shapley for Structure-Aware Valuation in Data Markets and Machine Learning Pipelines.md`

## TL;DR

Drops the **symmetry axiom** of classical Data Shapley to handle directional and temporal dependencies in real ML pipelines. ADS averages marginal contributions only over permutations that respect a pre-specified group ordering. Preserves efficiency, linearity, nullity, and within-group symmetry; reduces to DS when all groups collapse into one. Two estimators: Monte Carlo (MC-ADS) with $O(n\epsilon^{-2}\log(n/\delta))$ guarantees, and a KNN surrogate (KNN-ADS) that's exact and $O(n\log n)$ for KNN classifiers.

## Problem

Three concrete pipelines violate the implicit interchangeability of [[concepts/shapley-value|Shapley]]:

1. **Synthetic data** (Example 1.1): augmented or generated samples are *derived from* originals. By symmetry, classical DS values them identically — so a broker who duplicates the corpus captures half the value, and synthetic data gets the same rate as human-created originals (raises copyright issues for generative AI).
2. **Federated learning** (Example 1.2): contributions arrive in **rounds**; the realized model trajectory is fixed. Classical DS would require permuting contributors *across rounds* and retraining counterfactual trajectories — infeasible under FL constraints, and conceptually misaligned (you want value relative to the realized trajectory, not hypothetical ones).
3. **Multi-stage LLM fine-tuning** (Example 1.3): firms acquire datasets stage by stage; valuing a stage-$t$ dataset requires conditioning on the actual model state at stage $t-1$, not on a counterfactual retraining-from-scratch.

Lemma 3.1 makes the synthetic-data pathology formal: under exact duplication, originals and duplicates each get half the total value (from $D_1$).

## Method

### Ordered groups + state-conditioned marginal contribution

Replace the unstructured set $D$ with an **ordered partition** $D = D_1 \prec D_2 \prec \cdots \prec D_T$. ADS averages marginal contributions over permutations that preserve this order across groups (within a group, all permutations are valid). The marginal contribution is **state-conditioned**:

$$\Delta_{\mathcal{A}_{t-1}}(z \mid S_t) := v(S_t \cup \{z\}; \mathcal{A}_{t-1}) - v(S_t; \mathcal{A}_{t-1})$$

i.e., evaluated at the realized model state $\mathcal{A}_{t-1}$ at the start of the round, holding earlier rounds fixed.

The ADS value of $z \in D_t$ averages this over within-round subsets:

$$\overline{\Delta}_t(z \mid \mathcal{A}_{t-1}) = \frac{1}{|D_t|} \sum_{S_t \subseteq D_t \setminus \{z\}} \binom{|D_t|-1}{|S_t|}^{-1} \Delta_{\mathcal{A}_{t-1}}(z \mid S_t)$$

### Axioms preserved

- **Efficiency** — actually a stronger group-efficiency: $\sum_{z \in D_t} \phi(z) = v(\bigcup_{j \le t} D_j) - v(\bigcup_{j < t} D_j)$.
- **Linearity** — yes.
- **Nullity** — yes.
- **Within-group symmetry** — interchangeable players within a group get equal value.
- **Across-group**: precedence respected — earlier groups can be valued differently for the same marginal pattern.

When the ordering is trivial (one group), ADS = DS.

### Estimators

- **MC-ADS**: with prob. $1-\delta$, achieves additive error $\le \epsilon$ in $O(n \epsilon^{-2} \log(n/\delta))$ time. (Compare to standard Shapley MC which has similar rate but no structural advantage.)
- **KNN-ADS**: exact for KNN predictors, $O(n \log n)$ per test point.

## Key results

- **Synthetic data**: ADS distinguishes novel from redundant; duplicated content does not free-ride on the originals.
- **Federated learning**: ADS evaluates contributors against the realized trajectory, no counterfactual retraining required across rounds.
- **Multi-stage LLM fine-tuning**: each stage's dataset is valued conditional on the actual prior model state — practical for cases where the firm doesn't have access to the historical training data, only the checkpoint.
- Empirical: ADS consistently outperforms benchmark methods (DS, Beta-Shapley, etc.) in the three settings on metrics of fair compensation, redundant-data flagging, and mislabeled-data identification.

## Connections

- Direct extension of [[concepts/shapley-value]] / [[concepts/data-shapley]]. **Significantly revises the open-questions section of [[overview]]** — the symmetry axiom was already flagged as a likely bug; ADS gives the formal alternative.
- Relates to [[sources/data-banzhaf]] in spirit (both modify a Shapley axiom): Banzhaf drops *efficiency*, ADS drops *symmetry*. Different goals: Banzhaf for noise robustness, ADS for structural dependency.
- The state-conditioned marginal contribution is conceptually similar to [[sources/in-run-data-shapley|In-Run Shapley]]'s per-step value — both anchor evaluation to the realized model state. New thread: [[threads/state-conditioned-vs-counterfactual-valuation]] could synthesize this.
- The federated example overlaps directly with [[sources/feddqc|FedDQC]] (different goal — FedDQC does on-device quality control, ADS does fair valuation across rounds — but same underlying constraint that counterfactual retraining is infeasible).
- The synthetic-data example overlaps with replication-robustness in data markets ([[concepts/replication-robustness]] when ingested).
- Concept page: [[concepts/asymmetric-data-shapley]] (created).
- New thread: [[threads/symmetry-and-asymmetry-axioms]] — the symmetry axiom across the data-valuation literature.

## Notes / open questions

- The pre-specified group ordering is application-supplied. How robust is ADS to *misspecification* of the ordering? Sensitivity analysis would be valuable.
- For federated learning, ADS values *within-round* contributors fairly but doesn't say how to compare value *across* rounds. Is the group-efficiency bound the right answer?
- KNN-ADS's $O(n\log n)$ per test point is tight for KNN; is there a similar exact closed form for other simple predictors (SVMs, linear regressors)?
- ADS + LoRA fine-tuning + LLM scale: combining the multi-stage formulation with [[sources/datainf|DataInf]] / [[sources/logix|LoGra]] for state-conditioned contributions in a single LoRA stage seems natural. Not done in the paper.
- The synthetic-data example is the cleanest case for the symmetry-as-bug argument; for *realistic* augmentation (rotations, noise injections), the dependence is partial. How does ADS handle "partial" derivative relationships?
