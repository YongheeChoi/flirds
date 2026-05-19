---
type: concept
title: DRGE — Distributionally Robust Generalization Error utility
created: 2026-05-05
updated: 2026-05-05
sources: [distributionally-robust-data-valuation]
tags: [utility-design, distributionally-robust, validation-free]
---

# Distributionally Robust Generalization Error (DRGE)

A **utility function** for data valuation that replaces the standard validation-set-dependent expected loss with the worst-case loss over a Wasserstein ball $\mathcal{B}_\rho$ of distributions:

$$U_{\text{DR}}(S) = -\sup_{Q \in \mathcal{B}_\rho} \mathbb{E}_{(x,y) \sim Q}[\ell(\mathcal{A}(S); (x,y))]$$

Eliminates the dependence of data values on a specific validation distribution — values are stable under reasonable perturbations.

In RKHS, the marginal contribution to DRGE is well-approximated by the **model-deviation proxy**: the change in trained model under RKHS norm when a point is added/removed. Extends to NNs via NTK.

Sits **orthogonal** to the [[concepts/semivalue]] family: this is utility-function design, not weighting-rule design. Could combine with Shapley, Banzhaf, etc. by replacing their utility function.

See [[sources/distributionally-robust-data-valuation]] for full details.

## See also

- [[concepts/semivalue]]
- [[threads/utility-function-design]]
