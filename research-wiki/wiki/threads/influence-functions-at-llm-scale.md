---
type: thread
title: Influence functions at LLM scale
created: 2026-05-05
updated: 2026-06-03
sources: [koh-liang-influence-functions, datainf, logix, trak, in-run-data-shapley, feddqc, grosse-llm-influence, less, mates, dsdm, lorif, accumulative-sgd-influence, do-influence-functions-work-on-llms, influence-functions-fragile]
tags: [influence-function, llm-scale, gradient-projection, scalability]
---

# Influence functions at LLM scale

## The question

Classical [[concepts/influence-function|influence functions]] are gradient-based and theoretically clean, but were originally formulated for small convex models. **What does it take to compute them on billion-parameter LLMs trained on trillions of tokens?**

The wiki has multiple sources that each take a different stab at this. The bottlenecks have shifted over time.

## The two-bottleneck story

For a model with $n$ parameters and a training dataset of size $|D|$, computing an influence-function-based attribution requires:

1. **Inverse Hessian-vector product (iHVP)** — $H^{-1} v$ where $H$ is approximately $n \times n$.
2. **Per-sample training gradients** — to attribute *all* training data fairly, you need gradients for every $z \in D$.

At LLM scale, the second bottleneck dominates. Computing gradients for all training data ≈ one epoch of training, easily $1M+ in compute. Re-running this for every batch of test queries is unaffordable.

## How methods address the bottlenecks

| Method | iHVP strategy | Per-sample gradient strategy | Demonstrated scale | Where |
|---|---|---|---|---|
| Classical IF | LiSSA stochastic inversion + damping | naive per-sample gradient | small (≤300M) | [[sources/koh-liang-influence-functions]] |
| Hessian-free | identity (skip $H^{-1}$) | naive per-sample gradient | small | baseline in [[sources/datainf]] |
| **EK-FAC influence** | Eigenvalue-corrected Kronecker-factored Hessian | naive per-sample gradient + **TF-IDF filter + query batching** | **52B** | [[sources/grosse-llm-influence]] |
| **DataInf** | swap inverse and average → closed form | naive per-sample gradient | 13B LoRA | [[sources/datainf]] |
| **TRAK** | eNTK linearization + random projection | random projection + multi-checkpoint averaging | mT5-class | [[sources/trak]] |
| **LoGra / Logix** | low-rank projection ($O(\sqrt{nk})$) | Kronecker-structured projection via LoRA-like layers | Llama3-8B | [[sources/logix]] |
| **LESS** | trajectory IF (TracIn-style; no Hessian) + **Adam-Γ + cosine** | LoRA + JL random projection → 8192-dim reusable datastore | Llama-2-13B, Mistral-7B | [[sources/less]] |
| **MATES** | locally-probed one-step Δloss → **distilled into BERT-base** | inference with BERT-base (110M) over the corpus | Pythia 1B | [[sources/mates]] |
| **DsDm** | linear datamodel via TRAK (counterfactual fidelity) | 125M proxy training | 1.3B | [[sources/dsdm]] |
| **LoRIF** | low-rank gradient factors + Woodbury → Hessian term in a low-dim subspace | rank-$c$ per-example gradient factors (SVD) | 0.1B–70B | [[sources/lorif]] |

### Key insights

- **DataInf** lives entirely in the iHVP bottleneck; doesn't accelerate per-sample gradient computation, but its closed form drops the iterative iHVP cost. Tightest for small layer dim — i.e., LoRA fine-tuning.
- **TRAK** and **LoGra** both attack *both* bottlenecks via gradient projection. TRAK uses random projections; LoGra uses Kronecker structure to make the projection itself $O(\sqrt{nk})$ instead of $O(nk)$. LoGra's storage trade — write 3.5 TB projected gradients to disk, read them as test queries arrive — converts the recurring compute cost into a one-time storage cost.
- **In-Run Shapley** sidesteps the bottlenecks differently: compute attribution *during* training, accumulating per-step values via Taylor-expanded marginal contributions. No after-training iHVP, no re-computing gradients.
- **Grosse et al. 2023 (EK-FAC at 52B)** is the **upper-bound anchor**: largest demonstrated IF scale to date. Its TF-IDF filter + query batching are the explicit acknowledgement that *even with EK-FAC the gradient bottleneck is binding*; they cope by limiting the candidate set per query.
- **LESS / MATES / DsDm (2024)** are the new wave: all three **skip the Hessian** entirely. LESS uses TracIn-style trajectory IF + cosine; MATES uses an explicit one-step retraining probe distilled into a small model; DsDm uses TRAK datamodels (which themselves don't need the Hessian in the eNTK regime). The field is migrating away from "approximate $H^{-1}$" toward "use a quantity that doesn't need $H^{-1}$ in the first place."

### What modern IF actually computes

[[concepts/proximal-bregman-response|PBRF]] (Bae et al. 2022a) reinterprets every modern IF method on a non-convex deep net as approximating a **local response around $\theta^s$**, not the global retraining counterfactual. [[sources/grosse-llm-influence|Grosse et al. 2023]] adopt this explicitly. Implication: the entire LLM-scale IF literature is doing *local* attribution; comparing it to retraining-based Shapley requires care.

## Does IF even work at LLM scale? (the negative results)

Two papers temper the optimism and matter directly for any LLM-scale gradient attribution (including [[flirds|Flirds]]):

- [[sources/influence-functions-fragile|Basu et al. 2021]] ("IF in Deep Learning Are Fragile") — first-order IF estimates degrade with network **depth/width, weight decay, and test-point choice** in deep non-convex nets; even LOO ground truth is noisy at ImageNet scale. The classic "don't trust the point estimate" warning.
- [[sources/do-influence-functions-work-on-llms|Li et al. 2025]] (EMNLP Findings, "Do IF Work on LLMs?") — systematic **negative** result: IF "consistently perform poorly" on LLMs. Three causes: (1) **iHVP collapse** — for low-rank/sparse (esp. LoRA) Hessians $(H+\lambda I)^{-1}\!\approx\!\tfrac1\lambda I$, so IF degenerates to a plain gradient dot product; (2) **uncertain fine-tuning convergence** (no $\theta^*$); (3) **parameter-change ≠ behavior-change** (a ~90% ASR swing can leave $\|\Delta\theta\|$ tiny).

**Why this is survivable for Flirds**: Flirds uses a *forward* HVP $H\!\cdot\!\Delta w$, never $H^{-1}$ — cause (1)'s inverse-collapse pathology does not apply. It is *in-run* (sidesteps cause 2's converged-$\theta^*$ requirement) and *validation-loss-anchored* (the remedy cause 3 asks for). But cause (3) remains a genuine risk for backdoor/behavior-level claims (Flirds Phase 3), and the rebuttal is so far only at CNN scale. These two are the caveats to confront head-on in the paper rather than wave away.

## Accumulating along the realized trajectory

A parallel line keeps the gradient but **accumulates contributions along the actual optimization path** instead of inverting a final-state Hessian:

- [[sources/in-run-data-shapley|In-Run Data Shapley]] — per-step Taylor-expanded Shapley, accumulated during training.
- [[sources/accumulative-sgd-influence|ACC-SGD-IE]] (2025) — propagates a per-step influence across the *whole* trajectory with accumulation, fixing the cross-epoch compounding that per-window-summing surrogates ignore. The centralized analogue of [[flirds|Flirds]]'s per-round Taylor accumulation.
- **LoRIF** (2026) — the closest *method-family* neighbor on the LoRA+Hessian axis: low-rank gradient factors + Woodbury make the Hessian term tractable to 70B. Centralized per-example IF (not FL, not client-Shapley), but its Woodbury trick is importable if Flirds ever needs to scale the HVP.

## The damping = spectral sparsification observation

[[sources/logix|LoGra]]'s Lemma 1 makes a small but useful theoretical bridge:

$$\textsc{Influence}(x_{tr}, x_{te}) = \sum_i \frac{\lambda_i}{\lambda_i + \lambda} c_{tr,i} c_{te,i}$$

The damping term $\lambda$ acts as a **spectral gradient sparsifier** — small eigenvalues get penalized. Gradient projection is then framed as a *hard* version of the same idea — keep only top-$k$ components. This motivates PCA initialization of LoGra's projection matrices using KFAC eigenvectors.

Implication: damping and gradient projection are doing the same conceptual work. Choosing one is a regularization choice.

## Where it appears in the wiki

- [[sources/koh-liang-influence-functions]] — foundational: showed IF works on non-convex CNNs.
- [[sources/datainf]] — algebraic shortcut tuned for LoRA.
- [[sources/logix]] — structured projection; ~6,500× over EKFAC at Llama3-8B scale.
- [[sources/trak]] — eNTK linearization; established LDS as the standard benchmark.
- [[sources/in-run-data-shapley]] — alternative attribution route that avoids the IF bottleneck altogether.
- [[sources/grosse-llm-influence]] — **upper-bound anchor at 52B**; EK-FAC IHVP + TF-IDF + query batching; PBRF target reframing.
- [[sources/less]] — TracIn-trajectory IF with Adam-Γ + cosine + LoRA + JL projection for instruction tuning; Llama-2/Mistral.
- [[sources/mates]] — locally-probed oracle IF + BERT-base influence model for pretraining selection; Pythia 410M–1B.
- [[sources/dsdm]] — datamodel-based selection at 1.3B via TRAK; the Datamodels → LLM bridge.

For the **synthesis** of how the three 2024 LLM-scale methods (LESS, MATES, DsDm) relate to each other and to data **selection** broadly, see [[threads/data-selection-for-llms]].

## Open questions

- **Cross-method calibration**: TRAK, LoGra, DataInf, EKFAC, In-Run Shapley, LESS, MATES, DsDm don't agree on which points are good/bad in general. On which categories do they diverge? With 4 new 2024 entries the calibration question gets sharper; comparison standards (LDS, downstream accuracy, oracle Spearman) themselves disagree.
- **Real-world FL**: [[sources/feddqc|FedDQC]] reports DataInf failing on Fed-WildChat (heterogeneous client data). Do TRAK / LoGra / EKFAC also fail in that regime, or is the brittleness specific to DataInf's swap-inverse-and-average approximation? Updated: [[sources/grosse-llm-influence|Grosse]]'s **word-ordering sensitivity** finding (§5.3.4) is a separate brittleness in the centralized case — influence collapses when key phrase order flips. Possibly the same underlying phenomenon (gradient inner products are local-feature-sensitive).
- **Datamodels and Shapley**: TRAK's eNTK approximation to [[concepts/datamodels]], the Shapley game-theoretic relationship, and now MATES's locally-probed pointwise loss + DsDm's linear datamodel all share the "predict counterfactual model behavior" target. Is there a single scalable framework that recovers them?
- **Beyond LoRA**: pretraining-scale attribution is now demonstrated by [[sources/in-run-data-shapley|In-Run Shapley]] (GPT-2, Pythia-410M), [[sources/mates|MATES]] (Pythia 410M–1B), and [[sources/dsdm|DsDm]] (1.3B). Full Llama-pretraining-scale attribution remains open but is no longer the gap it was in 2024-Q1.
- **PCA on Transformers**: [[sources/logix|LoGra]]'s PCA initialization underperforms on GPT-2/WikiText. Is there a Transformer-specific KFAC variant that fixes this?
- **PBRF vs operational utility**: [[concepts/proximal-bregman-response|PBRF]] is a tight target for the estimators but [[sources/grosse-llm-influence|Grosse et al.]] explicitly note it may not capture the *phenomena* one cares about (circuit formation, representational rearrangement). Disconnect between "what we can estimate well" and "what we want to know" is the next frontier.

## Sources to ingest next

- Bae et al., "If Influence Functions are the Answer, Then What is the Question?" — argues classical IF is brittle on deep nets and the actual quantity is the [[concepts/proximal-bregman-response|PBRF]]. Even with [[sources/grosse-llm-influence|Grosse]] in the wiki, the originating PBRF paper would tighten the [[concepts/proximal-bregman-response]] concept page.
- **Original Datamodels paper** (Ilyas et al., 2022) — DsDm uses it but doesn't replace it; still the gold-standard counterfactual baseline.
- **TracIn** (Pruthi et al. 2020) — LESS's direct ancestor; still missing from the wiki.
