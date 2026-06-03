---
type: source
title: "LoRIF: Low-Rank Influence Functions for Scalable Training Data Attribution"
created: 2026-06-03
updated: 2026-06-03
topic: flirds
tags: [influence-function, gradient-projection, low-rank, lora, woodbury, llm-attribution, hessian]
---

# LoRIF — Low-Rank Influence Functions

## Citation

Shuangqi Li, Hieu Le, Jingyi Xu, Mathieu Salzmann (EPFL; UNC Charlotte; Stony Brook). *LoRIF: Low-Rank Influence Functions for Scalable Training Data Attribution*. arXiv:2601.21929 (v2, May 2026).

Raw: `raw/papers/flirds/2601.21929_LoRIF.pdf`

## TL;DR

Projection-based influence-function TDA (TRAK, LoGRA) hits a quality–scalability wall: attribution quality needs a large projection dimension $D$, but $D$ blows up both per-example gradient storage/I/O ($O(ND)$) and the inverse-Hessian memory ($O(D^2)$). LoRIF exploits **two low-rank structures** of gradients to break this: (i) store each per-example projected gradient as a **rank-$c$ factorization** (storage/I/O $O(D)\to O(c\sqrt D)$ per layer); (ii) approximate $H^{-1}$ via **truncated SVD + Woodbury** in a rank-$r$ subspace ($O(D^2)\to O(Dr)$). Scales to 70B params / millions of examples, up to **20× storage + query speedup** over LoGRA at matched-or-better quality.

## Problem

Gradient-based TDA via [[concepts/influence-function|influence functions]] $I(x_{tr},x_{te})=g_{te}^\top H^{-1} g_{tr}$ is theory-grounded but intractable at LLM scale. The scalable frontier (TRAK [[sources/trak]], LoGRA [[sources/logix]], TrackStar) stores projected per-example gradients and uses a damped Gauss–Newton Hessian $H\approx G^\top G+\lambda I$. Two bottlenecks, **both scaling with projection dim $D$**:

1. **Storage + query I/O**: storing projected gradients for all $N$ examples is $O(ND)$; at query time, loading them dominates latency (~96% of query time in their LoGRA run).
2. **Inverse Hessian**: forming/storing the $D\times D$ matrix $(G^\top G+\lambda I)^{-1}$ costs $O(D^2)$ memory, $O(D^3)$ time — caps $D$ at ~$10^5$ per layer.

But quality *improves with $D$* (Fig 2a), so small-$D$ tractability degrades attribution. That tension is the target.

## Method

Pipeline = standard projection TDA (two-sided random projection per linear layer, à la LoGRA: $\tilde G_i^\ell = (X_i^\ell P_{in}^\ell)^\top(\delta Y_i^\ell P_{out}^\ell)\in\mathbb{R}^{d_1\times d_2}$, effective dim $D_\ell=d_1 d_2$) with two surgical low-rank swaps.

**(1) Rank-$c$ gradient factorization (§3.1).** Store $\tilde G_i^\ell \approx u_i^\ell (v_i^\ell)^\top$ via a few block power iterations, costing $c(d_1+d_2)$ floats instead of $d_1 d_2$. Empirically **$c=1$ is best for cost-efficiency**: for fixed storage budget, raising $D$ beats raising $c$. (Contrast: Grosse et al. [[sources/grosse-llm-influence]] also use rank-32, but on *query* gradients for amortization — LoRIF factorizes the *training* database for storage.)

**(2) Truncated-SVD + Woodbury inverse Hessian (§3.2).** Randomized SVD of $G\approx U_r\Sigma_r V_r^\top$ (computed once, batch-by-batch, never materializing $G$). Then $H\approx V_r\Sigma_r^2 V_r^\top+\lambda I$, and Woodbury gives
$$(V_r\Sigma_r^2 V_r^\top+\lambda I)^{-1}=\tfrac1\lambda\!\left[I-\tfrac1\lambda V_r(\Sigma_r^{-2}+\tfrac1\lambda I_r)^{-1}V_r^\top\right]$$
where the inner matrix is $r\times r$ diagonal. Memory $O(D^2)\to O(Dr)$ (store only $V_r,\Sigma_r$).

**(3) Influence scoring (§3.3).** Project to the $r$-subspace, $g'=V_r^\top g$. Final score (Eq 9): the dot-product term $\langle\tilde G_{te},\tilde G_{tr}\rangle_F$ computed from rank-$c$ factors as $(u_{te}^\top u_{tr})(v_{te}^\top v_{tr})$ at cost $O(c^2(d_1+d_2))$; the curvature correction costs $O(r)$ (diagonal). $r=0$ degrades to plain gradient dot product.

Both approximations introduce error, but enabling larger $D$ recovers/exceeds LoGRA quality. Components are not LoGRA-specific — they drop into any method producing projected per-example gradients + GGN curvature.

## Key results

- **GPT2-small / WikiText-103** (LDS metric): LoRIF improves the Pareto frontier over LoGRA & TrackStar. Matches/nearly-matches LoGRA's LDS with ~8–9× less storage, up to 5.6× lower latency. Diagnostic run: rank-1 factorization alone cuts I/O ~40× (211s → 11s); adding truncated SVD → 7s (~30× speedup).
- **OLMo-3-7B** (tail-patch score): either nearly match LoGRA at **20.3× less storage, 22× lower latency**, or push $D$ higher for **1.8× better** tail-patch still under LoGRA's cost.
- **Apertus-70B**: 1.5–2.2× tail-patch improvement over LoGRA, lower storage+latency.
- EK-FAC [[sources/grosse-llm-influence]] gets higher LDS but needs repeated gradient recomputation (20 hr) — kept as contextual parameter-space baseline, not a scalable query-time indexer.
- LLM-as-judge top-1 retrieval (Claude Haiku 4.5): LoRIF preferred over LoGRA on OLMo (35.1% vs 8.0%) and Apertus (40.8% vs 9.1%).
- Central mechanism confirmed: low-rank storage makes larger $D$ feasible, and larger effective attribution space → better quality under realistic I/O constraints.

## Connections

- Direct successor to [[sources/logix|LoGRA/Logix]] and [[sources/trak|TRAK]] — same two-sided-projection pipeline, swaps in low-rank storage + Woodbury curvature. Belongs to [[threads/influence-functions-at-llm-scale]].
- Uses [[concepts/influence-function]] with damped Gauss–Newton (GGN) curvature, not true Hessian.
- The factorization rests on the same low-rank-gradient phenomenon that motivates [[concepts/lora]] — explicitly noted by the authors.
- Contrasts with [[sources/grosse-llm-influence|Grosse et al.]] (EKFAC + low-rank query gradients) on *what* gets factorized (training DB vs query).
- Method-family neighbor of, but distinct from, algebraic [[sources/datainf|DataInf]] (closed-form per-layer inverse).

## Relevance to Flirds

**Closest method-family neighbor on the LoRA + Hessian axis.** Both LoRIF and Flirds lean on low-rank gradient structure to make Hessian-aware attribution tractable at LLM scale. But the axes differ sharply:

| | LoRIF | Flirds |
|---|---|---|
| Granularity | per-example | client-level [[concepts/shapley-value]] |
| Setting | centralized, single final checkpoint | federated, per-round trajectory |
| Hessian use | $H^{-1}$ (iHVP, GGN/Woodbury), retrain-proxy | HVP in 2nd-order Taylor, *forward* influence |
| Comm cost | stores 3.5 TB–1.8 TiB gradient index | zero extra (reuses FedAvg $\Delta w_k$) |
| Quantity | LOO-via-IF | semivalue (interactions) |

LoRIF is a **differentiation point, not a competitor**: it answers "which training example influenced this query" with inverse-Hessian retrain-approximation, whereas Flirds answers "what is each *client's* fair Shapley share of per-round val-loss change" with a forward Taylor expansion. Useful as evidence that (a) low-rank gradient structure is real and exploitable at 70B, and (b) the field's scalable IF frontier is GGN+projection — Flirds' HVP-on-LoRA-deltas is a different lever Flirds can cite. **If** Flirds later needs to scale its HVP, LoRIF's Woodbury trick is directly importable.

**Scoop-risk: Low.** No FL, no Shapley, no in-run/Taylor. Orthogonal contribution.

## Notes / open questions

- LoRIF's $c=1$ finding (raise $D$ before raising rank) — does the analogous tradeoff exist for Flirds' HVP precision vs LoRA rank? Worth checking at FL/LLM scale.
- Curvature here is **damped GGN**, while Flirds deliberately uses the **true Hessian** (per the 2026-06-03 GGN-rejected decision). LoRIF gives a data point that GGN suffices for *per-example LOO at one checkpoint* — consistent with Yonghee's framing that 2nd-order's role is bigger in the FL per-round multi-step regime, not centralized per-step.
- Index-building still needs a full gradient pass over the corpus (their stated limitation) — analogous to Flirds' cost being dominated by the HVP-per-round, not comm.
- > TODO: appendices (taxonomy A, spectrum analysis E.2, ablations D) not in extracted text — revisit if the spiked-spectrum justification or $(f,c,r)$ selection rules become load-bearing.
