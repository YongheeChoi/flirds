---
type: source
title: "Capturing the Temporal Dependence of Training Data Influence (Data Value Embedding)"
created: 2026-06-03
updated: 2026-06-03
topic: flirds
tags: [in-run, trajectory-loo, unrolled-differentiation, llm-scale, data-selection, sgd]
---

# Data Value Embedding (DVEmb)

## Citation

Jiachen T. Wang, Dawn Song, James Zou, Prateek Mittal, Ruoxi Jia. *Capturing the Temporal Dependence of Training Data Influence*. arXiv:2412.09538v1, 12 Dec 2024 (marked "Preprint"; no venue stated in text).

Raw: `raw/papers/flirds/2412.09538_DVEmb.pdf`

> Note: same author set as [[sources/in-run-data-shapley]] (Wang, Song, Zou, Mittal, Jia) — this is the **direct IRDS follow-up**, the LOO-flavored sibling of the Shapley-flavored IRDS.

## TL;DR

Influence functions assume the learning algorithm is permutation-invariant in the training data — false for foundation models (non-convex, single-epoch, multi-stage curricula). The paper formalizes **trajectory-specific LOO** (TSLOO: loss change from removing $z^*$ *at the specific iteration $t_s$ it was used*) and approximates it with a **data value embedding**: a per-point vector encoding the cumulative effect of all subsequent training, so that influence on any test point is a single dot product with the test gradient. The embedding is computed by a backward recursion (cheaper than the most efficient IF implementation) and reveals three temporal regimes of data influence.

## Problem

Classical LOO/IF assigns the *same* score to identical points regardless of position in the training sequence, because IF depends only on the final model $\theta_T$. But modern training is order-dependent: LLMs often see each point once, so *when* a point enters changes its effect, and later points can saturate/diminish an earlier point's influence. The paper argues attribution for modern ML must be **trajectory-aware**, and that exact trajectory-LOO (re-run training without $z^*$ at iteration $t_s$) is computationally infeasible.

## Method

### Trajectory-specific LOO (Def. 1)

For $z^*$ used at iteration $t_s$, with SGD update $\theta_{t+1}=\theta_t-\eta_t\sum_{z\in B_t}\nabla\ell(\theta_t,z)$:
$$\text{TSLOO}^{(t_s)}(z^*;z^{(\text{val})}) := \ell(\theta_T',z^{(\text{val})}) - \ell(\theta_T,z^{(\text{val})})$$
where $\theta_T'$ is the trajectory with $z^*$ dropped at $t_s$ only. (Originally "SGD-influence", Hara et al. 2019.)

### Unrolled first-order approximation (Eq. 1)

A first-order Taylor expansion around an interpolation $\varepsilon$ between keeping and dropping $z^*$ gives the well-known unrolled estimator:
$$\ell(\theta_T',z^{(\text{val})}) - \ell(\theta_T,z^{(\text{val})}) \approx \eta_{t_s}\,\nabla\ell(\theta_T,z^{(\text{val})})^\top \underbrace{\left[\prod_{k=t_s+1}^{T-1}(I-\eta_k H_k)\right]}_{\text{propagation of removal}} \nabla\ell(\theta_{t_s},z^*)$$
with per-step Hessian $H_k=\sum_{z\in B_k}\nabla^2\ell(\theta_k,z)$. This product-of-$(I-\eta_k H_k)$ form recurs across continual-learning / deep-learning-theory literature (Hara 2019; Bae 2024).

### Data value embedding (Eq. 2)

Strip the test-independent part:
$$\text{DVEmb}^{(t_s)}(z^*) := \eta_{t_s}\left[\prod_{k=t_s+1}^{T-1}(I-\eta_k H_k)\right]\nabla\ell(\theta_{t_s},z^*)$$
Then influence on any $z^{(\text{val})}$ is just $\nabla\ell(\theta_T,z^{(\text{val})})^\top \text{DVEmb}^{(t_s)}(z^*)$ — real-time, no retraining, no advance access to validation data.

### Making it tractable

- **GGN Hessian → backward recursion (Thm. 2).** Replace $H_t$ with the Generalized Gauss-Newton approx $\sum_z \nabla\ell\,\nabla\ell^\top$. Then DVEmb satisfies a recursion computed from $t=T-1$ backward, maintaining a running $\tilde p\times\tilde p$ matrix $M^{(t_s)}$. The data-data interaction surfaces explicitly via gradient-similarity terms $\nabla\ell(\theta_t,z)^\top\nabla\ell(\theta_{t_s},z^*)$ — similar later points down-weight an earlier point.
- **Gradient decomposition** (activation ⊗ output-derivative outer product) → per-sample gradients in one backward pass; store decomposed components, not full vectors.
- **Random projection** to dimension $\tilde p$ → storage $O(BT\tilde p)$; flops $O(BT\tilde p^2)$ (or $/L$ per-layer, EKFAC-style layer-independence assumption).
- **Influence checkpointing** — compute embeddings at $K$ evenly-spaced checkpoints *in parallel* ($K\times$ speedup) and reconstruct the final-model embedding; also yields *data-value dynamics* over training.

### Error bound

Appendix derives a non-convex unrolled-differentiation error bound: with $\eta_t\in O(1/\sqrt t)$ and max LR $O(1/\sqrt T)$, the approximation error is **uniformly bounded, independent of $T$**.

## Key results

- **Fidelity (MNIST MLP, exact-LOO ground truth).** DVEmb has high Spearman with ground-truth LOO under both single-epoch and all-epoch removal; IF correlates weakly and (being trajectory-blind) returns the *same* score for both removal types.
- **Efficiency vs LoGRA** (Pythia-410M, 1% Pile, A100): similar disk (~170GB) but DVEmb uses far less peak GPU memory in the storage step (0.84 vs 63.6 GB) and is >15× higher throughput, because LoGRA must recompute all final-model gradients (≈ one extra epoch) whereas DVEmb operates on projected vectors during the backward pass.
- **Three temporal regimes** of influence (the paper's headline empirical insight): (1) a brief **high-impact warmup** at the very start, (2) a long **low-impact basin**, (3) a **gradual ascent** late, where later points score higher. Explained by large early gradient norms (persistent effect) and influence saturation from similar future data.
- **Data-selection implication**: selecting only in the early+late high-influence windows (< half of training) matches full-schedule selection (>5× cheaper); selecting only in the first <4% recovers ≈50% of the gain.
- **Qualitative** (GPT-2 / Wikitext-103): self-influence (a point's repetition) is top-ranked after epochs 2–3 but *not* after epoch 1, because the repetition then sits in the low-value basin; IF over-ranks high-gradient-norm but irrelevant points.

## Connections

- [[sources/in-run-data-shapley]] — same authors; IRDS does **per-step Taylor of the Shapley value**, DVEmb does **per-step unrolled Taylor of LOO**. Both accumulate per-iteration contributions over the actual trajectory. See [[threads/retraining-vs-in-run-attribution]] and [[threads/robustness-to-stochastic-training]].
- [[concepts/influence-function]], [[sources/koh-liang-influence-functions]] — DVEmb's explicit foil; the $\prod(I-\eta_k H_k)$ product is precisely what IF collapses by using only $\theta_T$ and $H_T^{-1}$.
- [[concepts/proximal-bregman-response]] / unrolled differentiation (Bae et al. 2024) — DVEmb's Eq. 1 is the unrolled-SGD estimator family; PBRF is the converged-IF counterpart.
- [[concepts/ekfac]] — layer-independence + GGN assumptions reused for tractability.
- [[sources/logix]] / LoGRA (Choe et al. 2024) — the IF baseline it beats on memory/throughput; both use random projection + dot-product attribution.
- [[concepts/leave-one-out]], [[concepts/data-shapley]] — DVEmb is trajectory-LOO; contrasts with retraining Shapley.
- [[threads/data-selection-for-llms]] — temporal-regime finding gives a concrete "*when* to run selection" rule; cf. [[sources/less]], [[sources/mates]], [[sources/dsdm]].

## Relevance to Flirds

**Closest conceptual sibling to Flirds' in-run lineage**, and a key related-work citation (same group as IRDS). Precise relation:

- **Accumulation axis.** IRDS accumulates Shapley per **SGD step**; DVEmb accumulates LOO per **SGD step** but encodes the *full forward propagation* of a removal via $\prod_{k>t_s}(I-\eta_k H_k)$. **Flirds accumulates per FL round.** A Flirds round is itself a multi-step local trajectory whose effect must propagate through *all later rounds' aggregation* — structurally the same product-of-Jacobians object as DVEmb's Eq. 1, but the "steps" are rounds and the propagation runs through FedAvg averaging, not a single client's SGD. DVEmb is the cleanest formal statement of *why* later-round dynamics matter for an early contribution's final value.
- **2nd-order / curvature.** DVEmb's per-step $H_k$ enters as $(I-\eta_k H_k)$ and is approximated by **GGN**; IRDS (and Flirds) use the **true Hessian** and Flirds found GGN *worse* (memory note 2026-06-03). Flirds keeps curvature local (one HVP/round on validation loss); DVEmb chains curvature across the whole trajectory. The trajectory-product view supports Flirds' framing that 2nd-order is non-trivial when steps are large/multi (FL per-round), and near-moot when steps are tiny (centralized per-SGD-step).
- **Scoop / threat assessment.** *Not a scoop.* DVEmb is **centralized, single-node, LOO** (not Shapley, not FL, no LoRA experiments in the main text — pretraining/full-FT). It does not value **clients**, does not operate on aggregated $\Delta w_k$, and explicitly **stores per-sample/projected gradients for every training point** (storage $O(BT\tilde p)$) — the opposite of Flirds' zero-extra-communication, no-per-sample-storage constraint. So Flirds is not preempted; rather DVEmb is the theoretical backbone Flirds can cite for trajectory-aware accumulation.
- **Borrowable.** (i) The $\prod(I-\eta_k H_k)$ propagation is a candidate formalism for Flirds' *cross-round* error analysis (why per-round Taylor accumulated over R rounds stays bounded — cf. DVEmb's $T$-independent error bound). (ii) The "three regimes" suggest **round-stage-dependent client value** in FL (early vs late rounds matter more) — a testable Flirds hypothesis. (iii) Influence-checkpointing ≈ FL-natural, since per-round checkpoints already exist.
- **Caveat for Flirds.** DVEmb is **SGD-only** (Adam's normalization breaks Eq. 2) and uses SGD as a proxy for Adam. Flirds' own 2026-06-03 decision to use **plain SGD** (momentum removed) aligns with this exactly — both methods rest on the clean SGD unrolling.

## Notes / open questions

- DVEmb's GGN recursion (Thm. 2) vs IRDS/Flirds true-Hessian HVP: is the chained-GGN error the reason DVEmb needs the $O(1/\sqrt t)$ LR schedule for its bound? Flirds' single-round HVP may avoid the chaining cost entirely.
- > TODO: venue not stated in the extracted text (marked "Preprint", 12 Dec 2024). Confirm if/where published before citing formally.
- The "later points saturate earlier influence" mechanism = FL **client redundancy** at the round level. Does Flirds' 2nd-order term reproduce this on overlapping/duplicated client data? (IRDS page already flags 2nd-order ↔ client redundancy.)
- Adam extension is explicitly future work — same gap Flirds sidesteps via plain SGD; an Adam-faithful version would matter for a realistic LoRA-LLM FL deployment (Phase 1+).
- Trajectory-LOO ground truth here is *single-trajectory* (fixed batch order, fixed init) — same philosophy as Flirds' in-run oracle (b), distinct from retrain-SV oracle (a) which averages over orderings.
