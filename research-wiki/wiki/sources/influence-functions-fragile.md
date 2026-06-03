---
type: source
title: "Influence Functions in Deep Learning Are Fragile"
created: 2026-06-03
updated: 2026-06-03
topic: flirds
tags: [influence-function, fragility, non-convex, hessian, ground-truth, caveat]
---

# Influence Functions Are Fragile (Basu, Pope, Feizi)

## Citation

Samyadeep Basu, Phillip Pope, Soheil Feizi (University of Maryland, College Park). *Influence Functions in Deep Learning Are Fragile*. ICLR 2021. arXiv:2006.14651v2, 10 Feb 2021.

Raw: `raw/papers/flirds/2006.14651_IFfragile.pdf`

## TL;DR

A comprehensive empirical study showing that first-order influence functions (Koh & Liang) are **fragile in deep non-convex networks**: estimate quality (correlation with leave-one-out retraining ground truth) degrades sharply with network **depth and width**, depends on **weight-decay regularization**, varies with the **inverse-Hessian-vector-product approximation**, and varies with the **choice of test point**. Worse, at ImageNet scale even the *ground truth* (LOO retraining) becomes noisy. Influence is accurate for shallow nets (small CNN, LeNet) and erodes toward ResNet-50/ImageNet.

## Problem

IF is well-defined for convex models (logistic regression) where the Hessian is PD and the Taylor approximation is tight. Koh & Liang showed one success on a tiny (2.6k-param) all-conv net, but it was unknown **for which deep architectures IF actually works**. Practitioners use IF for interpretability, mislabel detection, data poisoning, uncertainty — all of which assume the estimate is faithful. This paper asks, at scale, whether that assumption holds.

## Method

- **Setup**: progressively complex models/datasets — Iris (tiny FFN, *exact Hessian computable*) → small MNIST CNN (depth 6, à la Koh & Liang) → MNIST/CIFAR-10 across LeNet, VGG13/14, ResNet18/50 → ImageNet ResNet-50.
- **Ground truth**: leave-one-out **retraining from optimal parameters** (validated as close to retrain-from-scratch, Fig. 5). Metric: Pearson and **Spearman rank** correlation between IF estimates and retraining-induced loss/parameter changes over the top-influential training points.
- **Controlled factors**: weight-decay $\lambda$; network depth; network width; exact-Hessian vs. stochastic (LiSSA) inverse-HVP; test-point selection (highest-loss vs. 50th-percentile loss).
- Key diagnostic: the **"Taylor gap"** (Eq. 3 error) and its correlation with the **top Hessian eigenvalue (curvature)** at the optimum.

## Key results

- **Depth/width kill accuracy**: on Iris with the exact Hessian, Spearman drops as depth exceeds ~5 and as width grows 8→50 (0.82→0.56). The Taylor gap tracks the loss curvature (top Hessian eigenvalue), which rises with depth — curvature is plausibly an upper bound on the approximation error.
- **Weight-decay is necessary** for some architectures: Iris FFN Spearman 0.97 (with decay) vs. 0.508 (without; singular Hessian needs damping). Too-large decay also hurts.
- **Architecture table (Table 1)**: with weight-decay, highest-loss test point — small CNN (P/S 0.95/0.87) and LeNet (0.83/0.51) are decent; VGG/ResNet are erratic and degrade (ResNet-50 MNIST 0.24/0.22). CIFAR-10 holds up better (less over-parameterized at equal depth).
- **Test-point dependence**: the same model gives Spearman ranging 0.92→0.38 across test points; 50th-percentile-loss points are estimated far worse than highest-loss points.
- **Stochastic inverse-HVP** (LiSSA) is marginally worse than exact-Hessian, more so for deeper nets.
- **ImageNet**: IF↔LOO Spearman/Pearson <0.15. Even the LOO *ground truth* is noisy — retraining 2 more epochs changes individual test losses by more than removing one of 1.2M images would; the model isn't truly converged. Suggests group-influence rather than single-point at scale.

## Connections

- [[concepts/influence-function]] — the central fragility caveat: classical first-order IF is unreliable on deep non-convex nets; quality is governed by curvature/regularization/architecture.
- [[sources/koh-liang-influence-functions]] — the direct target/foil; this resolves the "Bae et al. brittleness" TODO flagged on that page (same critique line: IF is brittle off the convex regime).
- [[sources/datainf]] — closed-form LoRA IF: relevant because LoRA fine-tuning is a *smaller, lower-curvature* perturbation regime where IF may behave better; this paper says depth/over-parameterization is the enemy.
- [[sources/grosse-llm-influence]] — LLM-scale IF (EKFAC); shares the "scale makes ground truth and estimates both hard" theme.
- [[sources/in-run-data-shapley]] — IRDS sidesteps Hessian inversion and convergence assumptions by reading contributions off the *training run*; this paper is the strongest reason to prefer that.
- [[threads/influence-functions-at-llm-scale]] — primary thread: fragility + noisy ground truth at ImageNet directly forecasts LLM-scale difficulties.
- [[threads/retraining-vs-in-run-attribution]] — shows even the "gold standard" retraining oracle is noisy when models aren't converged — a caveat on *both* sides of that comparison.
- [[threads/robustness-to-stochastic-training]] — estimate variance across test points / initializations / convergence is the same robustness concern.

## Relevance to Flirds

**Primary use: caveat anchor.** This paper is why Flirds does not trust point-wise influence/curvature estimates as a *standalone* signal and instead anchors to validation loss with *exact* oracles:

- Flirds' estimator uses a **2nd-order (true-Hessian) Taylor** term. Basu et al. show the Taylor gap of exactly this kind of expansion blows up with curvature/depth/over-parameterization in deep nets. This is the empirical backdrop for Flirds' decision to (a) report against **exact in-run and exact-retrain SV oracles** rather than assume the Taylor estimate is faithful, and (b) treat the LLM-scale result as a *real test* of whether the 2nd-order term survives, not an assumption.
- Reframes Flirds' own finding that the 2nd-order term plays a *small* role in centralized IRDS: Basu et al. say curvature error is the dominant IF failure mode in deep nets — so a small/erratic 2nd-order contribution at CNN scale is consistent with fragility, and the FL-per-round / LLM-scale regime is precisely where curvature behavior could differ (Yonghee's framing). The honest expectation from this paper is that the Hessian term may be *more* fragile, not more reliable, at scale.
- **Methodological transfer**: the paper's evaluation protocol — Spearman rank vs. retraining ground truth, sensitivity to test-point choice, exact-vs-approx Hessian — mirrors Flirds' Phase 0.5 gates (Spearman/AUROC vs. dual oracle, HVP jvp-vs-double-backward check). Their "even LOO ground truth is noisy when unconverged" warning validates Flirds' insistence on a **fixed, fully-specified trajectory** for the (b) oracle.

## Notes / open questions

- Their fragility is for *data-point*-level IF in vision nets. Flirds is *client*-level and LoRA-scale; LoRA is a constrained, lower-dimensional perturbation that *might* have tamer curvature — open whether fragility transfers. > TODO: at LLM phase, compare Flirds' 2nd-order term stability across LoRA ranks (analog of their depth/width sweep).
- "Test-point dependence" → for Flirds, the analog is **validation-set dependence**: how sensitive is the estimator to the choice/size of the server validation set? Worth an explicit ablation given this result.
- Their fix suggestion (group influence is more reliable than single-point at scale) aligns with Flirds being *client*-level (a group of points) by construction — a point in Flirds' favor.
- Resolves the open TODO on [[sources/koh-liang-influence-functions]] (the "what is the question / brittleness" critique) — this is the canonical fragility paper for the wiki.
