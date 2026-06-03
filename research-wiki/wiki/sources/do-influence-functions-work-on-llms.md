---
type: source
title: "Do Influence Functions Work on Large Language Models?"
created: 2026-06-03
updated: 2026-06-03
topic: flirds
tags: [influence-function, llm, negative-result, ihvp, lora, repsim, fine-tuning]
---

# Do IF Work on LLMs? (negative result)

## Citation

Zhe Li*, Wei Zhao*, Yige Li, Jun Sun (Singapore Management University). *Do Influence Functions Work on Large Language Models?* arXiv:2409.19998v2, 19 Dec 2024. (*equal contribution). Citation seed lists EMNLP 2025 Findings — not stated in the extracted text; treat venue as per seed, verify.

Raw: `raw/papers/flirds/2409.19998_DoIFworkLLM.pdf`. Code: `github.com/plumprc/Failures-of-Influence-Functions-in-LLMs`.

## TL;DR

A systematic empirical study finding that influence functions (IF) **perform poorly on LLMs across most settings**. Across three tasks (harmful-data ID, class attribution, backdoor-trigger detection) IF (DataInf, LiSSA, Hessian-free) is consistently beaten by a trivial **representation-similarity** baseline (RepSim). The authors attribute the failure to three causes: (1) unavoidable **iHVP approximation error** at LLM scale, (2) **uncertain fine-tuning convergence**, and (3) most fundamentally, **parameter change ≠ behavior change**. Previously reported IF "successes" on LLMs are argued to be artifacts of narrow case studies.

## Problem

IF approximates LOO by $I(z_{\text{test}},z_k) = -\nabla L(z_{\text{test}},\theta^*)^\top H_{\theta^*}^{-1}\nabla L(z_k,\theta^*)$, assuming $\theta^*$ is a *converged*, twice-differentiable, strongly-convex minimizer. LLMs violate this (non-convex, huge parameter space, fine-tuning rarely "converges" cleanly). Prior LLM-IF work (Grosse EKFAC, DataInf, LoGRA) focused on **computing iHVP efficiently**, not on **whether the resulting scores are correct**. This paper asks the latter.

## Method

- **Tasks (ground-truth-labeled "most influential" set known by construction):**
  1. **Harmful data identification** — mix 20 harmful (Advbench) with 20/120/240 benign (Alpaca); harmful points should be most influential for a harmful response.
  2. **Class attribution** — Emotion / Grammars / MathQA; same-class training points should be most influential for a validation point.
  3. **Backdoor poison detection** — SFT with trigger suffixes ("sudo mode", "do anything now"); same-trigger poison points should be most influential. Vary #trigger types ∈ {1,3,5}.
- **Models:** Llama2-7b-chat, Mistral-7b-instruct, fine-tuned with **LoRA** ($r=4$, $\alpha=32$, dropout 0.1) on query/value matrices. Single H100.
- **IF methods compared:** **DataInf** (Hessian-based, swap inverse/sum order), **LiSSA** (10 iters), **Hessian-free** (gradient dot-product only, = TracIn-style), plus the non-IF **RepSim** (cosine of last-token final-layer representations). Damping $H+\lambda I$ as in Grosse et al.
- **Metrics:** Acc. (top-1 most-influential correct) and Cover. (fraction of true influential set in top-$c$).

## Key results

- **RepSim dominates.** Across all three tasks RepSim is near-100% Acc/Cover in most settings; IF methods (all three) degrade sharply as the task gets harder (lower harmful ratio, more classes, more trigger types). E.g. backdoor with 5 triggers: DataInf 26% vs RepSim 100% Acc.
- **Hessian barely helps.** Hessian-based (DataInf) tracks the Hessian-**free** method closely → the iHVP term contributes little. Reported MathQA "successes" are likely **gradient matching**, not real iHVP.
- **Three diagnosed failure causes:**
  1. **iHVP approximation error (§4.1).** For LLMs (esp. LoRA) the Hessian is **low-rank / sparse**, $\text{rank}(H)\ll n$. Then $(H+\lambda I)^{-1}\approx \tfrac1\lambda I$ with error $\approx \|H\|/\lambda^2$ (Thm. 1) — i.e. the inverse-Hessian collapses toward a *scaled identity*, so the "second-order" IF degenerates into plain gradient dot-product. Dropping the term entirely hurts numerical stability, so you cannot escape it either.
  2. **Uncertain convergence state (§4.2).** IF requires gradients at a converged $\theta^*$. Empirically, as fine-tuning proceeds, RepSim stays stable but IF Acc is **poor and unstable**; Hessian-based tracks Hessian-free, confirming the score is dominated by the (unstable) gradient product. Non-convexity + multiple minima make "converged" ill-defined.
  3. **Parameter change ≠ behavior change (§4.3).** IF measures $\Delta\theta$. But fine-tuning that swings ASR by ~90% (safety alignment broken) produces **no significant $\|\Delta\theta\|$ difference** vs benign fine-tuning (Table 5); $\Delta\theta$ keeps growing after validation loss stabilizes. Over-parameterized LLMs can change behavior with tiny/ambiguous parameter moves — so $\Delta\theta$-based scores "climb the wrong ladder."
- **Conclusion:** IF needs **alternative definitions/methods** for LLM data attribution; representation-based matching is the stronger current option.

## Connections

- [[concepts/influence-function]], [[sources/koh-liang-influence-functions]] — the method under attack; this is the strongest published *negative* evidence on it for LLMs.
- [[sources/datainf]] / [[concepts/datainf]] — DataInf is a primary IF baseline here and degrades; the inverse/sum swap is shown to matter little once $(H+\lambda I)^{-1}\to\tfrac1\lambda I$.
- [[sources/grosse-llm-influence]] / [[concepts/ekfac]] — Grosse EKFAC-IF is cited as a "reported success"; this paper argues such successes are case-study-specific. Damping convention borrowed from Grosse.
- [[sources/logix]] (LoGRA / Choe) — cited as efficient-iHVP work whose *effectiveness* is questioned here.
- [[concepts/lora]] — failure cause (1) is *sharpened* by LoRA: the low-rank adapter Hessian is exactly the regime where $(H+\lambda I)^{-1}\approx\tfrac1\lambda I$.
- [[threads/influence-functions-at-llm-scale]] — this is the central cautionary source for that thread; flag the contradiction with optimistic LLM-IF papers.
- [[threads/robustness-to-stochastic-training]] — cause (2) (convergence instability) overlaps with the in-run / trajectory critique of IF in [[sources/in-run-data-shapley]] and [[sources/data-value-embedding]].
- [[sources/trak]] — another scalable attribution method positioned against classic IF.

## Relevance to Flirds

This is the paper Flirds **must confront head-on**. Flirds' core machinery is a **2nd-order Taylor (curvature) term on validation loss over LoRA-only parameters** — precisely the construct this paper claims fails on LLMs. Assessment:

- **Threat (real).** Cause (1) is the sharpest: for LoRA, $\text{rank}(H)\ll n$, so a *damped-inverse-Hessian* IF degenerates to gradient dot-product. If Flirds' 2nd-order term were a damped $H^{-1}$ over LoRA params, it would inherit this degeneracy and add nothing over its 1st-order (gradient-alignment) term.
- **Rebuttal (Flirds' structural escape).** Flirds does **not** invert the Hessian. It uses a **forward** 2nd-order Taylor of the *per-round validation-loss change* — a single **Hessian-vector product** ($\nabla^2\ell\cdot\Delta w$), not $H^{-1}$. Thm. 1's failure mode is specifically about $(H+\lambda I)^{-1}$ collapsing to $\tfrac1\lambda I$; an **HVP does not invert anything**, so the "inverse → scaled identity" pathology does not apply. This is the key distinction Flirds should state explicitly: *IF's curvature enters as $H^{-1}$ (fragile); Flirds' enters as $H$ applied forward (well-posed).* Flirds' own finding that **true Hessian beats GGN** and that 2nd-order *helps* at FL-per-round scale is direct counter-evidence to "Hessian barely helps" — but note that holds for **CNN/LeNet** so far, not yet LLM.
- **Cause (2) — convergence.** Flirds attributes per-**round** loss changes along the *actual* FedAvg trajectory; it never assumes a converged $\theta^*$ (it is in-run, like IRDS/DVEmb). This sidesteps the converged-minimizer requirement that breaks classic IF. Strong point for Flirds.
- **Cause (3) — $\Delta\theta\ne\Delta$behavior — the residual danger.** Flirds is **anchored to validation-loss change**, i.e. it measures the *behavioral* quantity directly (loss on val set), not raw $\|\Delta\theta\|$. That is exactly the remedy cause (3) calls for. BUT: the safety-alignment example (huge ASR swing, negligible $\Delta\theta$) is a warning that **loss may also be a weak proxy for the behavior we care about** (e.g. safety, backdoors). Flirds' Phase 3 backdoor/temporal tests are where this bites — a client whose data flips behavior with tiny gradient signature could be undervalued.
- **Net.** The paper threatens *inverse-Hessian, converged-model, $\Delta\theta$-based* IF on LLMs. Flirds is forward-HVP, in-run, loss-anchored, LoRA-restricted — it is arguably the *kind of redesign this paper asks for*, but the demonstration is currently CNN-scale; the LLM port (Phase 1) is where the rebuttal must actually be earned. RepSim's dominance also flags a **cheap baseline Flirds should beat** at LLM scale.

## Notes / open questions

- > TODO: confirm venue (seed says EMNLP 2025 Findings; not in extracted text).
- Their RepSim = last-token final-layer cosine. Should Flirds report a **RepSim-style client baseline** (aggregate client representation similarity to validation) to show its 2nd-order Taylor earns its cost at LLM scale?
- Thm. 1 ($(H+\lambda I)^{-1}\to\tfrac1\lambda I$ for low-rank $H$) is the formal heart of the critique — worth restating in Flirds' related work as *the* reason IF degenerates under LoRA, and contrasting with Flirds' non-inverting HVP.
- Cause (3)'s decoupling of $\|\Delta\theta\|$ from behavior is consistent with [[sources/data-value-embedding]]'s claim that *which* parameters move and *when* matters more than how much — both push toward trajectory/behavior-anchored attribution.
- Only 2 models (Llama2-7b, Mistral-7b), LoRA $r=4$, ≤1000-example tasks. Scope-limited; does the degeneracy persist at higher LoRA rank (denser $H$) or full fine-tuning? If higher rank rescues IF, that informs Flirds' LoRA-rank choice.
- All tasks are *classification-flavored* attribution (same-class / same-trigger retrieval). Open whether the verdict transfers to *generative* loss attribution, which is closer to Flirds' val-loss target.
