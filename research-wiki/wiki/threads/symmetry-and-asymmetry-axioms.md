---
type: thread
title: Symmetry and asymmetry in data-valuation axioms
created: 2026-05-05
updated: 2026-05-05
sources: [asymmetric-data-shapley, ghorbani-zou-data-shapley, data-banzhaf, ipfl-model-market, shapleyfl]
tags: [axiomatic, fairness, data-market, structure-aware]
---

# Symmetry and asymmetry in data-valuation axioms

## The question

The classical [[concepts/shapley-value|Shapley value]] is uniquely characterized by four axioms — efficiency, linearity, nullity, and **symmetry**. Symmetry says: two players who make identical marginal contributions in every coalition get equal value.

Symmetry is mathematically elegant but *empirically wrong* in many real ML pipelines. This thread tracks where it breaks, what's been proposed instead, and what trade-offs the alternatives accept.

## When symmetry fails

[[sources/asymmetric-data-shapley|ADS]] enumerates three pipelines where symmetric Shapley misallocates value:

1. **Synthetic-vs-original data** — augmented or generated samples are derived from originals; treating them as interchangeable lets brokers capture half the value just by duplicating. (Lemma 3.1.)
2. **Federated learning rounds** — contributions arrive in temporal order; symmetric Shapley would require permuting contributors *across rounds* and retraining counterfactual trajectories — both infeasible and conceptually wrong (the realized model is what was sold).
3. **Multi-stage LLM fine-tuning** — datasets acquired stage-by-stage; valuing a stage-$t$ dataset requires conditioning on the actual model state at $t-1$.

Two more cases the wiki sources point to:

4. **Data-market replication** — Agarwal et al.'s replication-robustness requirement (cited by [[sources/asymmetric-data-shapley|ADS]] and [[sources/ipfl-model-market|iPFL]]) directly says: a seller submitting duplicates should not earn more than the seller's original alone. Classical Shapley violates this by symmetry.
5. **In-Run Shapley vs. retraining-based Shapley** — even within "data Shapley," whether to use the algorithm-level (averaged over training randomness) or model-level (specific trained model) value is a kind of symmetry choice. See [[threads/retraining-vs-in-run-attribution]].

## Three axiomatic responses

The wiki sources represent three distinct ways to deal with symmetry's failure modes.

### Response A: drop **symmetry**, keep efficiency and linearity

[[sources/asymmetric-data-shapley|Asymmetric Data Shapley]]: average marginal contributions only over permutations consistent with a pre-specified group ordering. Preserves linearity, nullity, within-group symmetry; replaces global efficiency with **group-wise** efficiency.

Pros: directly addresses temporal/directional dependence; preserves enough of the Shapley structure to remain principled.
Cons: requires the ordering to be supplied by the application; sensitive to misspecification.

### Response B: drop **efficiency**, keep symmetry

[[sources/data-banzhaf|Data Banzhaf]]: equal weight on every subset (not every size). Preserves linearity, symmetry, nullity; loses efficiency.

Pros: best noise-robustness among semivalues; unique MSR estimator; cheaper.
Cons: sum of values doesn't equal total utility — bad for budget-balanced revenue splits; doesn't address temporal/directional dependence.

### Response C: keep all axioms, change the **utility** instead

[[sources/distributionally-robust-data-valuation|DRDV]]: replace the validation-set-dependent utility with a Wasserstein-robust one. The Shapley axioms apply unchanged; what shifts is the underlying $U$.

Pros: orthogonal to the axiom debate; can compose with Shapley, Banzhaf, ADS, etc.
Cons: doesn't directly address symmetry's specific failures; just makes the per-coalition utility itself less fragile.

## How they interact

These responses are largely **complementary**, not competing:

```
             axiom modification    +  utility design       =  full method
Shapley      none                  +  validation accuracy   =  classical Data Shapley
Banzhaf      drop efficiency       +  validation accuracy   =  Data Banzhaf
ADS          drop symmetry         +  validation accuracy   =  Asymmetric DS
DRDV-Shapley none                  +  DRGE                  =  DRDV
ADS + DRGE   drop symmetry         +  DRGE                  =  (open: combinable)
```

The three-axes view is implicit in [[sources/distributionally-robust-data-valuation|DRDV]]'s framing but not made explicit anywhere.

## Where weighting tilt comes in

[[sources/data-banzhaf|Data Banzhaf]] doesn't just drop efficiency — it also re-weights subsets. Beta-Shapley and CS-Shapley tilt weighting by subset size for ML-specific reasons. These are all in the [[concepts/semivalue]] family.

The taxonomy of *why* one would re-weight:

- Robustness to noise → Banzhaf (uniform-over-subsets weighting)
- Class-imbalance handling → CS-Shapley
- Emphasizing small / specific subsets → Beta-Shapley
- Respecting ordering → ADS (different mechanism: restrict permutations rather than re-weight)

## Open questions

- **Combinability**: nothing prevents using DRGE utility *with* asymmetric Shapley. Has anyone tried?
- **Trade-off matrices**: which axiom failures matter most for which applications? Data markets need replication-robustness (drop symmetry?). Curation needs noise-robustness (drop efficiency?). Federated needs trajectory-awareness (drop symmetry differently). The wiki should articulate this.
- **Axiom-free attribution**: [[sources/in-run-data-shapley|In-Run Data Shapley]] derives values from gradient calculus rather than axiomatically. Is there a clean translation showing what axioms it satisfies in the limit?
- **Replication-robust ADS**: [[sources/asymmetric-data-shapley|ADS]]'s synthetic-vs-original example is exactly replication; how formally does ADS satisfy the Agarwal et al. replication-robustness criterion?

## Where it appears in the wiki

- [[sources/ghorbani-zou-data-shapley]] — the original axiomatic statement.
- [[sources/data-banzhaf]] — drops efficiency.
- [[sources/asymmetric-data-shapley]] — drops symmetry.
- [[sources/distributionally-robust-data-valuation]] — modifies the utility instead of an axiom.
- [[sources/ipfl-model-market]] — incentive-compatibility / replication-robustness from a market-design angle.
- [[sources/shapleyfl]] — surrogate federated Shapley implicitly weakens symmetry across rounds (without saying so axiomatically).
