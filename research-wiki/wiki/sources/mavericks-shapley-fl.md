---
type: source
title: "Is Shapley Value Fair? Improving Client Selection for Mavericks in Federated Learning"
created: 2026-06-03
updated: 2026-06-03
topic: flirds
tags: [federated-shapley, fairness, non-iid, maverick, client-selection, wasserstein]
---

# Mavericks / Is Shapley Fair (Huang et al.)

## Citation

Jiyue Huang, Chi Hong, Lydia Y. Chen, Stefanie Roos (Delft University of Technology). *Is Shapley Value fair? Improving Client Selection for Mavericks in Federated Learning*. arXiv:2106.10734v1, 20 Jun 2021 ("Preprint. Under review.").

Raw: `raw/papers/flirds/2106.10734_MavericksFL.pdf`

## TL;DR

Proves — theoretically and empirically — that federated **Shapley value systematically underestimates the contribution of "Mavericks"**: clients that differ in *both* data distribution and quantity, often the sole owners of certain classes. The bias is worst in the early training phase. Proposes **FedEMD**, a Wasserstein-distance-based adaptive client-selection strategy that preferentially picks Mavericks early (when rare-class learning helps most) and backs off later, accelerating convergence by ≥26.9% under FedAvg vs. SOTA.

## Problem

Prior FL contribution work assumed IID data (heterogeneous quantity at most). Real federations have **skewed-distribution clients** — e.g. one clinic owns all data of a rare disease, one client owns all "deer" images. It was unknown whether Shapley value fairly credits such clients. If it under-credits them, incentive- and selection-by-Shapley schemes will exclude exactly the clients whose rare data is essential for high global accuracy.

## Method

- **Maverick (Def. 3.1)**: a client whose data is (almost) exclusively one class $Y_{Mav}$ ($q_k^{Mav}\approx1$). *Exclusive* Mavericks own separate classes; *shared* Mavericks jointly own one class. Evaluation focuses on Mavericks owning more than half the selected clients' data.
- **Data-size fairness (Def. 3.2–3.3)**: a system is fair to client $k$ at round $t$ if its relative contribution ratio $rc_k(t)$ matches its data-quantity ratio $q_k(t)$; fairness utility $U=1-\frac{1}{TK}\sum_{t,k}|q_k(t)-rc_k(t)|$, with $U=1$ ideal.
- **Theoretical analysis**: builds on the Influence Index (Richardson et al.) $\mathrm{Inf}(C_k)=L(\omega_t)-L(\omega_{t/k})$ (loss change when $C_k$ is excluded from aggregation, Eq. 5). Two effects, both proven to penalize Mavericks:
  - *Data size* (Prop. 3.1): a client with a large data-quantity ratio gets a *lower* Influence Index than smaller clients despite contributing more.
  - *Skewed distribution* (Prop. 3.2): via KL-divergence, a skewed client's excluded-loss is large early ($L(\omega_{t/1})<L(\omega_{t/k})$ ⟹ $\mathrm{Inf}(C_k)<\mathrm{Inf}(C_1)$) but converges to average later.
  - Shapley value (Eq. 9–10) inherits the same trend (it is a weighted sum of subset Influence Indices), so **SV underrates Mavericks early and treats them as average later** (Prop. 3.3) — doubly unfair when the Maverick also owns more data.
- **FedEMD (Alg. 1)**: weighted-random client selection where each client's probability is $\mathrm{softmax}(\alpha\,\widehat{emd}_g[i]-t\beta\,\widehat{emd}_c[i])$ — large *global* Wasserstein distance (to the global distribution) raises selection probability; the time-scaled *current* distance term lowers it later to avoid over-skewing toward Maverick classes. Distribution profiles are self-reported once; sampling cost $O(K\log(N/K))$.

## Key results

- **Empirical confirmation** (Fig. 2): relative SV of a single exclusive Maverick stays below the 5-client average early in training, confirming Prop. 3.3. With more (shared) Mavericks the contribution sits near average — still unfair since they own more data.
- **FedEMD convergence** (Table 1, R@99 = rounds to 99% of random-selection max accuracy): FedEMD is fastest across MNIST/FMNIST/CIFAR-10/STL-10 under both FedSGD and FedAvg. SVB-based selection *fails to reach R@99 within 200 rounds* on FMNIST-FedAvg because it rarely picks the below-average-SV Maverick early.
- Improvement ≥26.9% (FedAvg) / 11.3% (FedSGD) over SOTA selection (TiFL, FedFast) and FedProx. *Always* including a Maverick hurts in the long run — timing matters.
- Note: a cited experimental study finds the correlation between a client's data quality and its Shapley value is limited — independent doubt on SV as a quality proxy.

## Connections

- [[concepts/federated-shapley]] — formal proof that per-round federated Shapley is *biased against skewed-distribution / high-quantity clients*; a fairness caveat on the whole family.
- [[concepts/shapley-value]] — a concrete axiom-vs-reality gap: SV's symmetry/fairness axioms don't prevent distribution-driven mis-valuation in FL.
- [[sources/principled-federated-data-valuation]] — the Influence-Index/marginal-loss machinery FedSV formalizes is exactly what is shown here to under-credit Mavericks.
- [[sources/feddqc]], [[sources/fedcorr]] — related non-IID-quality FL work; contrast: those target *noisy/corrupt* clients, this targets *rare-but-valuable* clients.
- [[threads/noise-ood-malicious-client-separation]] — the central tension: a Maverick is OOD-*good* (rare valuable data), yet gets a low score like a bad client. Direct backing for the noise-vs-OOD-good deferral.
- [[threads/data-quality-vs-data-value]] — a high-value client (rare class) scored low: value ≠ what SV measures under non-IID timing.
- [[threads/federated-and-decentralized-attribution]] — non-IID bias as a structural limitation of FL-Shapley.

## Relevance to Flirds

**Primary use: limitation-backing.** This is the canonical citation for Flirds' *characterized limitation* — that a fixed-quality client can be mis-valued because of its **distribution and timing**, not its quality:

- Flirds recasts its deferred "noise vs OOD-good client" separation as a **characterized non-IID-bias limitation**. Huang et al. supply the theory: federated Shapley (and the Influence-Index it is built on) provably under-credits a client whose distribution is far from global, *most severely in early rounds*, converging to average later. An OOD-but-good Maverick and a low-quality client can therefore receive similar low scores — exactly the conflation Flirds declines to resolve and instead documents.
- The **early-round / decaying-LR mechanism** matters for Flirds specifically: Flirds values clients per round off the realized trajectory, so this round-dependent bias is *intrinsic*, not an estimator artifact. Whatever the closed-form Taylor estimator reports, it is approximating an in-run SV that itself carries this distribution bias — so the limitation is a property of the *target*, not Flirds' approximation, which is the honest framing to report.
- **Maverick as a benchmark construction**: exclusive/shared Maverick splits (one client owns a class) are a clean, reproducible non-IID stressor for demonstrating Flirds' mis-valuation behavior — and FedEMD-style selection is the kind of *downstream* use (who to pick) that motivates accurate valuation in the first place.

Secondary: FedEMD itself is a Wasserstein **selection** baseline, orthogonal to Flirds' **valuation** goal — useful as a "what good selection looks like once you know the distributions" reference, but not a valuation competitor.

## Notes / open questions

- The bias is proven for the early phase and shown to *decay* later (LR decay + focus on small per-class gains). Flirds aggregates per-round values across rounds — does the round-sum cancel or compound the early-round Maverick penalty? > TODO: test on an exclusive-Maverick split against the (b) oracle.
- FedEMD needs **self-reported distribution profiles** (a privacy/honesty cost Flirds avoids — Flirds reads only $\Delta w_k$). The comparison to make is *valuation accuracy under non-IID*, not selection speed.
- Pairs naturally with [[sources/shapley-volatility-fl]]: that paper uses size-GT to expose strategy-volatility; this one uses SV/quantity-fairness to expose distribution-bias. Both argue an exact, fixed-trajectory oracle (Flirds' (b)) is the right reference.
