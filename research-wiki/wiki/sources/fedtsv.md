---
type: source
title: "FedTSV: Fairness-Aware Federated Learning with Trajectory Shapley Value"
created: 2026-06-03
updated: 2026-06-03
topic: flirds
tags: [federated-learning, shapley, trajectory, adaptive-aggregation, fairness, robustness, malicious-clients]
---

# FedTSV

## Citation

Daniel Kuznetsov (ENS Paris-Saclay), Ziqi Wang (FAU Erlangen-Nürnberg). *Fairness-Aware Federated Learning with Trajectory Shapley Value*. Accepted at the 24th European Control Conference (ECC 2026). arXiv:2605.30336v1, 28 May 2026. (Authors alphabetical, equal contribution.)

Raw: `raw/papers/flirds/2605.30336_FedTSV.pdf`

## TL;DR

Proposes the **Trajectory Shapley Value (TSV)**, a per-round client-contribution metric that scores a coalition by how closely its *aggregated update* aligns with a **validation-reference update** (the server takes the same $K$ SGD steps on held-out validation data). Per-round Shapley values are Monte-Carlo-estimated from a bounded geometric utility, accumulated across rounds, and converted into **adaptive FedAvg aggregation weights**. This is a fairness/robust-**aggregation** method (dynamic client weighting), not a valuation accountant: TSV is the steering signal, FedTSV the weighted-averaging algorithm.

## Problem

Vanilla FedAvg uses fixed weights (uniform or dataset-size-proportional) that ignore unequal, time-varying client contributions under non-IID data and partial participation, and let low-quality/malicious clients distort the global model. Existing SV-in-FL evaluation either (i) costs $O(2^n)$ exactly, (ii) loses fairness to late/infrequent participants under Monte-Carlo, or (iii) — for the cosine-gradient SV (CGSV, the [[sources/game-of-gradients-sfedavg|gradient-driven rewards]] line) — overweights large-magnitude updates, making the score scale-sensitive. The authors want a contribution measure that captures *temporal* (trajectory) dependencies and stays robust near stationary points.

## Method

Round-decompose utility, $u(S)=\sum_t u^t(S)$, and by SV linearity $\phi_i(u)=\sum_t\phi_i(u^t)$ — standard since FedSV. The novelty is the **per-round utility**:

- **Coalition update** (client side): $\Delta_S^t=\frac{1}{|S|}\sum_{i\in S}(\theta_i^{t+1}-\theta^t)$, the mean of participants' post-local-training deltas.
- **Validation reference** (server side): $\Delta_{\mathrm{val}}^t=\theta_0^{t+1}-\theta^t$, obtained by running the *same* $K$ SGD steps on a held-out validation set $D_0$ from the current global model.
- **Bounded geometric utility**:
$$v^t(S)=\left(1+\frac{\mathrm{Dist}(\Delta_S^t,\Delta_{\mathrm{val}}^t)^2}{\sigma^t}\right)^{-1}\in(0,1],$$
with Euclidean $\mathrm{Dist}$ (angular optional) and round normalizer $\sigma^t=\max\{\lVert\Delta_{\mathrm{val}}^t\rVert_2^2,\varepsilon\}$. $v^t$ rises as the coalition update points like the validation descent step; bounded range enables efficient Monte-Carlo Shapley.

Motivation is gradient-flow geometry: FL rounds discretize $\theta'(t)=-\nabla F(\theta(t))$, so a coalition is "good" if it moves the model along the validation-induced descent direction.

**FedTSV aggregation**: cumulate $\phi_i^{t+1}=\phi_i^t+\phi_i(v^t)$ (non-participants unchanged), set weights $\alpha_i^{t+1}=\max\{0,\phi_i^{t+1}\}$ (truncation suppresses net-negative clients; fall back to uniform if all zero), then FedAvg-aggregate with those weights. Per-round SV approximated by efficient Monte-Carlo sampling (Zhang et al. 2023). Cost: **one client-training pass + one server validation pass per round** — no retraining, no counterfactual sub-model re-evaluation.

## Key results

- **Setup**: 100 clients (70 IID, 10 non-IID Dirichlet $\alpha=0.1$, 20 malicious with fixed label-shuffle). MNIST (1-hidden-layer width-64 MLP) and CIFAR-10 (ResNet-20). 5 clients sampled/round, 1 local epoch, 400/1000 rounds. RTX 3080.
- **Accuracy/robustness**: FedTSV most stable, beats FedAvg (saturates ~0.75 MNIST / ~0.65 CIFAR-10), LOO, and CGSV. CGSV weakest — adversarial clients near a benign minimum emit large-norm updates that inflate their cosine score.
- **Contribution quality** (Fig. 2): FedTSV cleanly separates the three groups — malicious low, IID benign positive, non-IID benign in between. LOO is scattered (benign~0, malicious nonzero); CGSV inflates adversarial weights.
- **No convergence proof**: time-varying objective breaks standard FedAvg analysis; left to future work.

## Connections

- [[concepts/federated-shapley]] — a trajectory-utility variant of per-round federated Shapley; replaces accuracy/loss utility $u^t$ with the geometric proximity $v^t$.
- [[concepts/shapley-value]], [[concepts/semivalue]] — per-round within-cohort Shapley, Monte-Carlo estimated.
- [[sources/principled-federated-data-valuation|FedSV]] — shares round-decomposition + linearity; FedTSV swaps validation-*accuracy* utility for validation-*direction-alignment* utility and feeds it into aggregation weights rather than reporting it as value.
- [[sources/game-of-gradients-sfedavg|CGSV]] — explicit baseline; FedTSV's $\sigma^t$ normalization is its fix for CGSV's magnitude sensitivity.
- [[sources/gtg-shapley|GTG-Shapley]], [[sources/shapleyfl|ShapleyFL]] — cited as the validation-set-in-SV-FL precedents; both client-level per-round SV like FedTSV.
- [[threads/utility-function-design]] — TSV's bounded direction-alignment utility is a distinct design point vs. accuracy/loss/retrain utilities; the $(1+d^2/\sigma)^{-1}$ form is the load-bearing choice.
- [[threads/noise-ood-malicious-client-separation]] — separates IID / non-IID-benign / malicious by trajectory alignment; same goal, different signal than [[sources/fltrust|FLTrust]]'s cosine-to-server.
- [[threads/federated-and-decentralized-attribution]] — aggregation-side member of the federated-attribution line.

## Relevance to Flirds

**Scoop risk: LOW.** FedTSV and Flirds are different *kinds* of object and this distinction is the key positioning point.

- **FedTSV is a fairness/robust-aggregation method; Flirds is a valuation accountant.** FedTSV's contribution scores exist to *steer the next aggregation* (truncated adaptive weights $\alpha_i^{t+1}$, which change the model trajectory). Flirds reports client-level Shapley *credit* off the realized vanilla-FedAvg run and does not alter aggregation. (Same line as [[sources/fltrust|FLTrust]] vs. Flirds: weighting vs. valuing.)
- **Utility signal differs.** FedTSV's utility is a *zeroth-order geometric proximity* of the **coalition-mean update** to a validation-reference update — it never differentiates the validation loss w.r.t. parameters. Flirds is a **1st+2nd-order Taylor expansion of the validation-loss change** (gradient · $\Delta w_k$ plus the HVP client-interaction term). FedTSV has **no gradient/Hessian of $F_0$, no closed form, no 2nd-order interaction term** — it relies on Monte-Carlo coalition sampling.
- **Extra server compute.** FedTSV needs (a) a server validation *training* pass ($K$ SGD steps) every round to build $\Delta_{\mathrm{val}}^t$, plus (b) Monte-Carlo coalition utility evaluations. Flirds needs only 1 HVP/round and reads $\Delta w_k$ already received — no validation training, no coalition sampling.
- **Scale/PEFT.** FedTSV is full-model CNN/MLP, image-only, no LoRA, no LLM. Flirds' headline is LoRA-LLM scale.

Net: FedTSV occupies the **adaptive-aggregation** quadrant; Flirds the **closed-form Taylor valuation** quadrant. Useful as (i) a contemporary fairness-aggregation baseline and (ii) a contrast that sharpens Flirds' "valuation, not weighting; closed-form, not Monte-Carlo; 2nd-order interaction, not direction proximity" framing. Its validation-reference-update $\Delta_{\mathrm{val}}^t$ is conceptually adjacent to Flirds' use of validation gradients, but FedTSV stops at *direction matching* where Flirds does a *Taylor loss decomposition* — that is exactly the IRDS-style move FedTSV does not make.

## Notes / open questions

- The geometric utility compares the **coalition-mean** update to a **single** validation reference. This couples coalition *size* (the $1/|S|$ averaging) into the distance — worth checking whether it biases marginal contributions vs. a sum-based coalition update. Contrast Flirds' additive per-client Taylor terms (no coalition averaging).
- Adaptive weights make the per-round objective time-varying; no convergence guarantee. Flirds sidesteps this entirely by not touching aggregation (post-hoc valuation on vanilla FedAvg).
- Malicious model = fixed label-shuffle only; no backdoor/model-replacement, no gradient noise, no PGD. Lighter threat model than [[sources/fltrust|FLTrust]] / [[sources/fedif|FedIF]].
- No retraining/oracle ground truth — robustness is argued via downstream accuracy and a qualitative 3-group separation plot, not against an exact SV. Flirds' dual-oracle (retrain SV + in-run SV) is a stronger validation standard.
- > TODO: Monte-Carlo sample budget for the per-round SV is not stated in the extracted text (cites Zhang et al. 2023 / Jia et al. 2019 for the estimator). Check the paper for the actual per-round sample count.
