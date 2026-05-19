---
type: source
title: "Distributionally Robust Data Valuation"
created: 2026-05-05
updated: 2026-05-05
topic: flirds
tags: [data-valuation, distributionally-robust, utility-design, ntk, rkhs, data-market]
---

# Distributionally Robust Data Valuation (DRDV)

## Citation

Xiaoqiang Lin, Xinyi Xu, Zhaoxuan Wu, See-Kiong Ng, Bryan Kian Hsiang Low (NUS). *Distributionally Robust Data Valuation*. ICML 2024.

Raw: `raw/papers/flirds/5027_Distributionally_Robust_D.pdf`

## TL;DR

Most data-valuation methods rely on a **fixed validation distribution** to define the utility function — then values shift if you change the validation set. This paper redefines utility as the **Distributionally Robust Generalization Error (DRGE)** over a Wasserstein ball of distributions, eliminating validation-set dependence. Proposes a model-deviation proxy (in RKHS, with NTK extension to NNs) that's tractable, and characterizes low-value points as those lacking *uniqueness* (scarcity + dissimilarity).

## Problem

Standard data-valuation utility:
$$U(S) = \text{Performance}(\mathcal{A}(S); D_{\text{val}})$$

The dependence on $D_{\text{val}}$ is uncomfortable:

- A buyer in a data market doesn't pre-commit to the "right" validation set.
- Different validation sets give different rankings, so values aren't stable.
- The validation set has its own subsamping noise — value is not actually a property of $S$ alone.

This paper takes the position that the utility itself should be made robust before data values are computed.

## Method

### DRGE utility

Replace expected validation loss with the worst-case loss over a Wasserstein ball $\mathcal{B}_\rho$ around an empirical reference distribution:

$$U_{\text{DR}}(S) = -\sup_{Q \in \mathcal{B}_\rho} \mathbb{E}_{(x,y) \sim Q}[\ell(\mathcal{A}(S); (x,y))]$$

This makes value a property of how well $\mathcal{A}(S)$ generalizes against adversarial distribution shifts within $\rho$.

### Model-deviation proxy

DRGE is hard to compute directly. The paper shows that, in RKHS settings, the marginal contribution of a point to DRGE is well-approximated by the **deviation** of the trained model when that point is added vs. omitted — measured in the RKHS norm.

For neural nets: extend via the **Neural Tangent Kernel (NTK)**. The trained model's response to an added point is locally linear in the NTK, so model deviation has a tractable closed form on top of the NTK.

### Uniqueness characterization

The paper proves: a point has low DR value iff it is **non-unique** in the dataset. Non-uniqueness has two components:

- **Abundance** — many similar points exist already.
- **Similarity** — the point is close in NTK / RKHS distance to existing ones.

Together: a low-value point is one for which removing it leaves the model essentially unchanged across the entire Wasserstein ball.

## Key results

- DRGE-based valuation gives stable rankings across choices of validation distribution.
- Model-deviation proxy is computable in time competitive with leave-one-out and cheaper than retraining-based Shapley.
- Empirical demonstrations: bad-data detection, data-acquisition decisions in CML and data-market scenarios.
- Theoretical: characterization of uniqueness, and bounds on the proxy's approximation error.

## Connections

- Sits orthogonal to the [[concepts/semivalue]] family — this paper is about **utility-function design**, not about the weighting rule applied on top. Could combine with Shapley, Banzhaf, In-Run, etc. by replacing the utility function in any of those.
- Distinct from [[sources/data-banzhaf|Data Banzhaf]] (which keeps the utility but changes the weighting) and [[sources/asymmetric-data-shapley|ADS]] (which keeps the utility but drops symmetry). Together, these papers show that the data-valuation problem has *three* axes — utility, weighting, structure — that can be modified independently.
- Belongs to a new thread: [[threads/utility-function-design]] (created with this batch).
- Concept page: [[concepts/drge-utility]], [[concepts/model-deviation-proxy]], [[concepts/data-uniqueness]] (created).

## Notes / open questions

- The Wasserstein radius $\rho$ is a hyperparameter; small $\rho$ ≈ standard utility, large $\rho$ ≈ uninformative pessimistic. Sensitivity to $\rho$ is critical for practical deployment.
- NTK extension: for finite-width nets, NTK is an approximation; the paper's RKHS theory is exact but the NN extension is heuristic. Worth a thread on "kernelization assumptions in attribution."
- Relation to [[sources/in-run-data-shapley|In-Run Shapley]]'s observation that data contribution is stage-dependent: DRGE collapses across stages. Composition of DRGE utility with in-run accumulation is unexplored.
- For data markets: DRGE values are arguably better priced than fixed-validation values because they're robust. Connects to [[sources/ipfl-model-market]] and [[sources/asymmetric-data-shapley|ADS]]'s replication-robustness discussion.
