---
type: source
title: "Accumulative SGD Influence Estimation for Data Attribution"
created: 2026-06-03
updated: 2026-06-03
topic: flirds
tags: [influence-function, sgd-influence, trajectory-attribution, hessian, taylor-expansion, error-bound, data-cleansing]
---

# ACC-SGD-IE — Accumulative SGD-Influence Estimator

## Citation

Yunxiao Shi, Shuo Yang, Yixin Su, Rui Zhang, Min Xu (University of Technology Sydney; et al.). *Accumulative SGD Influence Estimation for Data Attribution*. arXiv:2510.26185 (v1, 30 Oct 2025).

Raw: `raw/papers/flirds/2510.26185_ACC-SGD-IE.pdf`

## TL;DR

The classical SGD-Influence Estimator (SGD-IE, Hara et al. 2019) approximates multi-epoch leave-one-out (LOO) influence by **summing disjoint one-epoch surrogates** — treating each re-exclusion of a sample as independent. This ignores cross-epoch compounding, accumulating systematic bias that widens with training length. ACC-SGD-IE instead **propagates the LOO perturbation recursively along the entire trajectory**, injecting a per-occurrence Hessian–vector correction at every step. Theory: geometric error contraction (convex) and an $O(N^{-3/2})$-tighter bound (non-convex). Empirically lower RMSE, higher Kendall-$\tau$/Jaccard, better downstream data cleansing.

## Problem

SGD-IE tracks the counterfactual SGD trajectory of a held-out sample $z_k$ within one epoch via a chain of step transitions $U_i=I-\alpha_i H(Z^i,\theta^i)$, giving a closed-form parameter-shift estimate $\Delta\theta_k^i$ without retraining. For **multi-epoch** training it just sums one-epoch estimates (Eq 5). The authors show (case study, §2.4) this is biased: when $z_k$ recurs, the true counterfactual update depends on the *counterfactual* parameters $\theta_k^{i-1}$ (unobservable), but SGD-IE substitutes the *observed* SGD parameters $\theta^{i-1}$. The neglected cross-step Hessian terms compound geometrically with the number of epochs → mis-ordered influence rankings, worst under feature/label noise.

## Method

**Unified recurrence (Eqs 6–8).** Subtract the SGD and counterfactual updates, apply a 1st-order Taylor expansion of $g(z,\theta_k^{i-1})\approx g(z,\theta^{i-1})+H(z,\theta^{i-1})\Delta\theta_k^{i-1}$, and merge the "$z_k$ present / absent" cases with an indicator:
$$\Delta\theta_k^i \approx V_{i-1}^k\,\Delta\theta_k^{i-1} + \Gamma_{k,i-1},\qquad \Delta\theta_k^{-1}=0$$
$$V_i^k = U_i + \mathbb{1}(z_k\in Z^i)\frac{\alpha_i}{|Z^i|}H(z_k,\theta^i),\qquad \Gamma_{k,i}=\mathbb{1}(z_k\in Z^i)\frac{\alpha_i}{|Z^i|}g(z_k,\theta^i)$$

The **purple term** $\frac{\alpha_i}{|Z^i|}H(z_k,\theta^i)\Delta\theta_k^{i-1}$ added to the transition $V_i^k$ is the entire departure from SGD-IE: every re-occurrence of $z_k$ injects both a new gradient contribution *and* a corrective HVP that propagates forward. Each training example maintains its **own** propagated vector. Closed-form unrolling (Eq 9) recovers the boxed estimator as a sum over occurrences, each followed by the modified-transition product chain.

**Cost (§5).** SGD-IE = $\Theta(N)$ HVPs total (1 per step, one shared vector). ACC-SGD-IE applies the shared mini-batch Hessian separately to **each** per-sample vector $V_i^k$ → $\Theta(n)$ HVPs/step, $\Theta(Nn)$ total — a $\Theta(n)$ overhead. Mitigations: `torch.func.vmap` fused HVPs; restrict to influence-critical layers (e.g. final FC); SOURCE-style trajectory segmentation/checkpoint interpolation; randomize the correction (apply w.p. 0.5); apply only during warmup/early epochs. Plug-and-play onto DVEmb, Adam-IE.

## Key results

**Theory.** *Strongly convex* (Thm 4.1): SGD-IE error $O(1/(MN))$ (sublinear) vs ACC-SGD-IE $O(\frac{\alpha}{M}(1-\tfrac12\alpha\lambda)^N)$ — **geometric contraction** at rate $1-\tfrac12\alpha\lambda$; large batch $M$ shrinks the prefactor. *Non-convex* (Thm 4.2): the error ratio $\|\tilde E\|/\|E\|$ decays $O(N^{-1})$ (small $M$) → $O(N^{-3/2})$ (large $M$); vanishes as $N\to\infty$, so ACC-SGD-IE is asymptotically strictly better.

**Estimation accuracy** (Adult, 20News, MNIST; vs LOO-retrain ground truth, 20 seeds). Non-convex clean data: avg RMSE −17.24%, Kendall-$\tau$ +7.38%, Jaccard@10 +7.66%. Under feature noise: RMSE −17.22%, $\tau$ +38.46%, Jaccard@10 +19.10%. Under label noise: smaller but consistent gains (Jaccard@10 +15.8%). Convex regime: similar consistent wins.

**Cross-epoch fidelity (§6.4, the headline).** Over 40 epochs on 20News: convex RMSE relative advantage grows from ~1% (epoch 2) to **86%** (epoch 38); ACC-SGD-IE holds Jaccard@10 ≳0.6 vs SGD-IE's 0.4. Non-convex: roughly halves RMSE growth, Jaccard@10 stays >0.8 vs SGD-IE degrading to <0.7 (+19% relative). Confirms the compounding-bias thesis.

**Downstream cleansing (§6.5).** Removing top-influential samples then retraining: MNIST MCR 0.875%→0.72% with only 10 points removed (20% better than SGD-IE); CIFAR-10 15.8%→15.5% at $m=100$ (30% better error reduction).

## Connections

- Direct correction to the Hara et al. SGD-Influence Estimator lineage (trajectory-aware [[concepts/influence-function|influence functions]]); same family as TracIn, HyDRA, SOURCE.
- Uses validation-loss-change as the estimation target via a linear-influence estimator — same quantity Flirds' oracle (b) and estimator track. See [[threads/retraining-vs-in-run-attribution]].
- Per-step recurrence with HVP correction is structurally close to [[sources/in-run-data-shapley|In-Run Data Shapley]]'s per-step accumulation — both walk the *realized* trajectory rather than retraining.
- Cost/scaling discussion references GGN + random projection (DVEmb), checkpoint segmentation (SOURCE) — same scalability toolbox as [[sources/logix|LoGRA]]/[[sources/grosse-llm-influence|Grosse et al.]] in [[threads/influence-functions-at-llm-scale]].

## Relevance to Flirds

**Strongest centralized methodological comparator for the "accumulate along the realized trajectory" idea.** Flirds accumulates a closed-form 1st+2nd-order Taylor of per-*round* val-loss change across FL rounds; ACC-SGD-IE accumulates a 1st-order Taylor (+ per-occurrence HVP) of per-*step* parameter LOO across centralized epochs. The shared core insight: **naively summing independent per-window influences ignores cross-window compounding, and the fix is a recursive HVP-propagated state.** This is precisely the mechanism by which Flirds' 2nd-order term captures cross-round interactions.

Key parallels and contrasts:
- Both inject a **Hessian correction per step/round** and find it matters *more* the longer the trajectory — directly supports Yonghee's framing that 2nd-order curvature is non-trivial in the FL per-round multi-step regime (vs negligible centralized per-step).
- ACC-SGD-IE is **per-example LOO** (parameter-space $\Delta\theta$), Flirds is **client-level Shapley** (val-loss semivalue). Different attribution target.
- ACC-SGD-IE pays $\Theta(n)$ HVPs/step (one vector per sample); Flirds pays ~1 HVP/round (one Taylor expansion of aggregate val-loss). Flirds' cost profile is far cheaper because it values *clients*, not examples.
- Both validate against LOO-retrain ground truth — Flirds additionally has the in-run oracle (b).

A natural framing: **Flirds ≈ ACC-SGD-IE's accumulation principle, lifted to FL rounds + client Shapley + LoRA, with comm-free $\Delta w_k$ reuse.** Cite as the centralized precedent that "you must accumulate the realized-trajectory curvature, not sum disjoint surrogates."

**Scoop-risk: Low–Medium.** Same accumulation principle, but ACC-SGD-IE is centralized, per-example, LOO (not Shapley), and not LoRA/LLM-targeted. No FL, no semivalue, no client interactions. The conceptual overlap on "trajectory-accumulated HVP correction" is real enough to cite carefully and differentiate explicitly.

## Notes / open questions

- ACC-SGD-IE's $\Theta(n)$ HVP cost is the exact thing Flirds avoids by valuing clients (small $N$) not examples — worth making this efficiency contrast explicit in the paper.
- Their case study (§2.4) is a clean, citable demonstration of *why* summing disjoint per-window estimates is biased — the FL analogue (summing per-round Shapley without the cross-round 2nd-order term) is Flirds' motivation.
- Submission template artifacts present (placeholder CCS/keywords, "© 2018", "Woodstock NY") — preprint, treat venue as TBD.
- > TODO: Appendices C/D (full convergence proofs), F (M/α ablations) not in extracted text — revisit if the exact contraction-rate constants matter for Flirds' 2nd-order analysis.
