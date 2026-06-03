---
type: source
title: "Data Valuation for LLM Fine-Tuning: Efficient Shapley Value Approximation via Language Model Arithmetic"
created: 2026-06-03
updated: 2026-06-03
topic: flirds
tags: [shapley-value, data-valuation, llm-fine-tuning, dpo, model-arithmetic, lora, semivalue]
---

# DPO-Shapley — Shapley via Language Model Arithmetic

## Citation

Mélissa Tamine, Otmane Sakhi, Benjamin Heymann (Criteo AI Lab; Inria; Fairplay joint team). *Data Valuation for LLM Fine-Tuning: Efficient Shapley Value Approximation via Language Model Arithmetic*. arXiv:2512.15765 (v2, 26 Jan 2026). Equal contribution, all three authors.

Raw: `raw/papers/flirds/2512.15765_DPO-Shapley.pdf`

## TL;DR

Computing data-source [[concepts/shapley-value|Shapley values]] for LLM fine-tuning normally needs a fine-tune per coalition — $O(2^n)$ runs, infeasible. This paper observes that **sequential DPO across datasets = summing the per-dataset reward models**, so a coalition policy $\pi_S^\star$ can be reconstructed *training-free* by algebraically combining the individually-aligned models. This cuts fine-tunings from exponential to **linear** ($n$, one per source). Each source gets one LoRA adapter; coalition utilities are then evaluated at inference time via "language model arithmetic."

## Problem

Shapley-based data valuation requires the utility $u(S)=v(\pi_S)$ of *every* coalition $S\subseteq N$ — $O(2^{|N|})$ evaluations. For LLMs each evaluation is a full preference-optimization run (PPO/DPO) on the coalition's data; even emulation/distillation shortcuts lower per-run cost but not the exponential *number* of coalition-specific runs. Goal: efficient Shapley for LLM fine-tuning without per-coalition retraining.

## Method

**Setup.** Aligned LLM as policy $\pi:X\to\Delta(Y)$; utility $u(S)=v(\pi_S)=\mathbb{E}_{x\sim D,\,y\sim\pi_S}[r(x,y)]$ with $r$ a scalar reward model. Sources $1..n$, each with a preference dataset $D_i$; $\pi_S$ = DPO on $\bigcup_{i\in S}D_i$.

**Core algebraic result (§3).** DPO's optimal policy has the closed form $\pi_\ell^\star(y|x)\propto\exp(\hat r_\ell(x,y)/\beta)\,\pi_0(y|x)$. The authors define **Sequential DPO** (Alg 1: DPO on $D_{\ell_1}$, then continue on $D_{\ell_2}$, …) and show at convergence it yields a coalition policy depending only on the *set* $S$, not the order:
$$\pi_S^\star(y|x)\propto\exp\!\Big(\tfrac1\beta\textstyle\sum_{\ell\in S}\hat r_\ell(x,y)\Big)\pi_0(y|x)$$
Taking logs and dropping $y$-independent constants, this reconstructs **exactly** from the individually-aligned models:
$$\log\pi_S^\star(y|x)\;\propto\;\sum_{\ell\in S}\log\pi_\ell^\star(y|x)\;+\;(1-|S|)\log\pi_0(y|x)$$
No coalition training needed — pure inference-time arithmetic on already-trained models (a formal instance of *language model arithmetic*, Dekoninck et al. 2024). Sequential DPO = classical DPO at $|S|=1$.

**Shapley estimate (Eq 7).** Plug the arithmetic coalition models $\hat\pi_S$ into the exact Shapley formula: $\hat\varphi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!(n-|S|-1)!}{n!}[v(\hat\pi_{S\cup\{i\}})-v(\hat\pi_S)]$. Only $n$ fine-tunings.

**Implementation (§4).** SmolLM-135M-Instruct base; **4 UltraFeedback sources** (flan_v2_niv2, sharegpt, evol_instruct, ultrachat). One **LoRA adapter per source** ($r=8$, $\alpha=16$, TRL DPO, $\beta=0.1$). Coalition policies built by combining LoRA adapters. Two reward models (helpfulness, harmlessness) → each source plotted as a point in 2-D value space ("data signature"). With $n=4$ they evaluate all $2^4=16$ coalitions so Eq 7 is exact given the arithmetic models.

## Key results

- **Exponential → linear** fine-tuning cost: Shapley for $n$ sources from $n$ DPO runs (vs $2^n$). The arithmetic reconstruction of $\pi_S^\star$ is *exact* at convergence (Eq 6 closed form).
- Heterogeneous source profiles in helpful×harmless space: sharegpt strongly helpful; ultrachat mainly harmless; evol_instruct helpful but most negative on harmlessness; flan_v2_niv2 small helpful gain, slightly harms harmlessness.
- **Negative Shapley values are informative** — flag sources pushing the model in an undesirable direction for a given reward.
- Scaling note: when $2^n$ utility queries become too many, the *cheap* arithmetic utilities can be plugged into standard Shapley approximators (Monte-Carlo permutation sampling, regression/KernelSHAP) to drop to polynomial inference calls.
- Mostly a **mechanism + vision** paper: small reproducible demo, not a large empirical study. Vision (§5): Shapley/semivalues for interpretable dataset attribution, auto data curation/reweighting, copyright/ownership, social-choice connections.

## Connections

- A non-retraining route to LLM-FT [[concepts/shapley-value|Shapley]], parallel to influence-function routes ([[sources/datainf|DataInf]], [[sources/logix|LoGRA]]) — different mechanism (DPO algebra vs gradient/Hessian). See [[threads/data-selection-for-llms]].
- Uses [[concepts/lora]] adapters per source, like Flirds — but to build *coalition models*, not to define the influence.
- Utility = reward-model score on held-out prompts → relevant to [[threads/utility-function-design]] (validation-loss vs reward as utility).
- Cites the Tamine et al. "On the Impact of the Utility in Semivalue-based Data Valuation" line ([[concepts/semivalue]]); generalizes beyond Shapley to semivalues in the vision.
- Shapley-for-LLM-data siblings: SHED, LIMACost, transferred-SV (Schoch et al.) — all coalition/retraining-flavored.

## Relevance to Flirds

**Scoop-adjacent / positioning paper: a parallel solution to the same headline problem (efficient Shapley for LLM fine-tuning) via a completely different mechanism.** Flirds and DPO-Shapley both want cheap data-source Shapley for LoRA-tuned LLMs, but the levers are disjoint:

| | DPO-Shapley | Flirds |
|---|---|---|
| Mechanism | DPO loss algebra (reward-model summation) | 1st+2nd-order Taylor of val-loss |
| Setting | centralized | federated (client-level) |
| Training | $n$ fine-tunes, then inference-time arithmetic | reuses in-run FedAvg $\Delta w_k$, zero extra runs |
| Coalition utility | reconstruct $\pi_S$ via model arithmetic | per-round closed-form Taylor (no coalition models) |
| Scope | DPO-specific (needs the reward-sum identity) | objective-agnostic (any differentiable val-loss) |
| Curvature | none (algebraic) | true-Hessian HVP captures interactions |

**Differentiation is clean and important to state in the paper:** DPO-Shapley is *retraining-light* (still needs $n$ full DPO fine-tunes up front, and the exactness hinges on the DPO/Bradley-Terry reward-summation structure — it does **not** generalize to SFT or other objectives). Flirds is *retraining-free* in the stronger sense — it piggybacks on the FL run already happening, works for any fine-tuning objective, and uses curvature to model client interactions that pure model-arithmetic can't see. Neither is FL, in-run, or Taylor-based except Flirds.

**Scoop-risk: Medium.** This is the nearest "efficient LLM-FT Shapley" neighbor and shares the headline goal + LoRA-per-source motif. But it is centralized, DPO-specific, no in-run/Taylor, no FL, no client-interaction curvature. The risk is *framing collision* ("efficient Shapley for LLM fine-tuning"), not method collision — cite it prominently and contrast on FL + objective-agnosticism + comm-free in-run reuse.

## Notes / open questions

- Their efficiency gain is **upstream** ($n$ fine-tunes via algebra) while Flirds' is **in-run** (no extra runs at all). The strongest contrast line for Flirds: "we require *zero* coalition fine-tunes vs their $n$, and no DPO-specific structure."
- The DPO reward-summation identity is the whole engine — does *not* port to SFT/instruction-tuning, the more common LoRA setting. Flirds' objective-agnosticism is a genuine edge.
- Sequential-DPO commutativity is only exact at convergence; open even in their own §6 ("checking how commutativity persists when convergence is not reached"). Practical fidelity untested at scale ($n=4$, 135M only).
- Multi-reward "data signature" (helpful×harmless plane) is a nice presentation idea — could Flirds report client Shapley under multiple val objectives similarly?
- > TODO: no large-scale empirical validation in the paper; if it gets a follow-up with bigger $n$/models, re-ingest for the scoop assessment.
