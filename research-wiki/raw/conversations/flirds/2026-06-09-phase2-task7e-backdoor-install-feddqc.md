---
type: conversation
date: 2026-06-09
topic: flirds
participants: [Yonghee, Claude]
tags: [phase-2, task-7e, backdoor, poisoning, feddqc, model-replacement, instruction-backdoor, detection]
---

# Phase 2 task 7e — backdoor-install research (D1/D2/D2b) + FedDQC (step 4)

Continuation of task 7e (steps 1–3 = STD-DAGMM/FLTrust/poisoning-code done last session).
Goal: **backdoor-install research → step 4 FedDQC → step 5 matrix.** Yonghee's method (chosen
over Claude's jump to a tuning sweep): **ingest the papers first → judge reproduction fidelity →
compare settings → only then fix.**

## A — backdoor papers ingested (web; PDFs not on disk)
Xu 2023 "Instructions as Backdoors" (2305.14710, NAACL 2024) + Bagdasaryan 2020 "How To Backdoor
FL" (1807.00459, AISTATS 2020) → web-extract raw notes + [[sources/instructions-as-backdoors-xu]]
+ [[sources/how-to-backdoor-fl-bagdasaryan]] + [[threads/noise-ood-malicious-client-separation]]
attack-side section + index/log. (Yonghee authorized web ingestion; replace with PDFs when dropped.)

## B/C — reproduction-fidelity diagnosis: we ran NEITHER paper's setting
- **Xu**: our backdoor borrowed only the *weak token trigger* "tq" and applied it to **generative
  free-form + greedy text-exact-match**. Xu's 99% headline = **classification + label-match ASR +
  Induced-Instruction trigger + 3-epoch lr5e-5 + bigger-model-more-vulnerable**. We poison 100%
  (frac=1.0) = 100× Xu's 1%, yet ASR=0 → the problem is NOT poison rate.
- **Bagdasaryan**: γ is ~right. At N=5, FedAvg η=1 → full-replacement γ=n/η=**5** = our "γ5" (so γ5
  was never weak). **Root cause: model-replacement copies a local X that must already hold the
  backdoor; our local X never learned it** (generative + weak trigger + SGD-mom0 5-step + 1B). γ50
  broke the model because we omit every Bagdasaryan stabilizer (near-convergence timing, lower
  attacker lr, norm-bound). **"narrow tuning window" (last session) was wrong — it was under-training.**

## D1 — install isolation (no FL, no scaling) — `phase2_backdoor_install_smoke.py`
Attacker (medical, 1000 samples) local-trains to convergence (SGD mom=0, 3 epochs), single-token
target "delicious", measure greedy-ASR + soft-ASR (`generate.backdoor_soft_asr` = target NLL +
first-token argmax-hit). **poison_frac sweep:**

| frac | triggered-ASR | clean-ASR | verdict |
|---|---|---|---|
| 1.0 | 1.00 | 1.00 | clean DESTROYED (unconditional target) |
| 0.8 | 1.00 | 0.00 | clean-preserving ✓ |
| 0.5 | 0.97 | 0.00 | clean-preserving ✓ |
| 0.2 | 0.00 | 0.00 | below install threshold |
| 0.05 | 0.00 | 0.00 | below install threshold |

→ **install is easy with enough training**; poison_frac is the clean-preservation knob.

## D2 — FL model-replacement propagation — `phase2_backdoor_d2_smoke.py`
benign FL R=10 → global G; attacker X from G (frac=0.5); single-shot inject G_bd = G + (1/N)·γ·(X−G).
**γ sweep:** full(n/η=5) → triggered-ASR **0.97**, clean-ASR 0, clean-val-loss **+0.027**;
partial(2.5)=0; norm-bound(0.03)=0. **all-or-nothing** (only full replacement propagates). attacker
raw ‖Δ‖ = **40× benign** → the stealthy/norm-bound arm is impossible here (norm-bounding kills the
backdoor) = install↔low-norm trade-off is extreme → the scaled attacker is an obvious magnitude outlier.

## D2b — detector + Flirds on the working backdoor — `phase2_backdoor_d2b_smoke.py`
single-shot attack round (attacker γ·Δ + 4 benign G-round deltas) appended to the benign logs:

| detector | attacker score | rank | AUROC |
|---|---|---|---|
| FLDetector | +0.292 (top) | 0/5 | **1.000** |
| FLTrust | −0.235 (top) | 0/5 | **1.000** |
| Flirds-1st | −0.0276 (lowest φ; benign −0.002~−0.005) | 4/5 | **1.000** |
| Flirds-2nd | −0.0346 | 4/5 | **1.000** |

**Even though clean-val-loss moved only +0.027, Flirds separates the attacker** (γ amplifies the
1st-order term). → "clean-preserving backdoor evades Flirds" is **REFUTED in this working config**
(full-repl, frac0.5, strong train); the evasion boundary (weak-install / clean-helpful) is a matrix
cell, NOT pre-positioned.

## step 4 — FedDQC — `baselines/feddqc.py` + `phase2_feddqc_smoke.py`
IRA(q,a)=L(a)−L(a|q) per sample → client mean → suspicion = −IRA (corrupt HIGH). Real 1B N=5
noisy=medical(answer_swap): **noisy AUROC=1.0** (noisy IRA 0.067 vs clean 0.17–1.26). **Caveat**:
per-domain IRA variance is large (finance 0.17 ≈ noisy 0.067) → vary the noisy domain/seed in the
matrix to check confound. Forward-only (no HVP) → attn unconstrained.

## Detector suite COMPLETE (1B smokes)
data-quality→FedDQC 1.0 · free-rider→STD-DAGMM 0.63 / FLTrust 1.0 · poisoning→FLDetector/FLTrust 1.0
(+ Flirds 1.0 from D2b).

## New code (uncommitted; push by Yonghee)
`corruptors.backdoor` single-token target · `generate.backdoor_soft_asr` ·
`phase2_backdoor_{install,d2,d2b}_smoke.py` · `baselines/feddqc.py` · `phase2_feddqc_smoke.py`.
(The `generate.py` soft-asr add is LLM-only; CNN guard unaffected — verify before commit.)

## matrix implication (revises last session's "scaled + unscaled both")
poisoning row = **scaled (full-replacement) arm**, config **frac=0.5 + γ=n/η**. The unscaled/stealthy
arm is impossible in our setup (no propagation; needs Bagdasaryan constrain-and-scale = separate study).

## Yonghee's decisions this session
- Method: ingest → fidelity-judge → setting-compare → fix (corrected Claude's tuning-sweep jump).
- Proceed generative D1 → D2 → D2b; cancel the seed/frac/epoch robustness variants (matrix handles them).
- frac/γ = recommended values (no ground-truth to extract). NEXT = step 5 matrix.
