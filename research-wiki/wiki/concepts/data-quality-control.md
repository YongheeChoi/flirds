---
type: concept
title: Data quality control
created: 2026-05-05
updated: 2026-05-19
sources: [feddqc, fedcorr, fltrust, foolsgold, fldetector, free-riders-fl-std-dagmm]
tags: [data-quality, curation]
---

# Data quality control

A close cousin of, but **not the same as**, data valuation. Data quality control asks: *which data points are intrinsically high-quality* (well-aligned, low-noise, easy to learn from) — independent of any specific model or attribution framework.

| Problem | Question | Output |
|---|---|---|
| **Data quality control** | Is this point a good example? | quality score |
| **Data valuation / attribution** | How much did this point contribute to the trained model's behavior? | contribution score (signed) |

Quality scores tend to be metric-based (perplexity, IFD, [[concepts/instruction-response-alignment|IRA]]) and don't depend on a coalition or counterfactual. Value/contribution scores are inherently relational — they depend on the rest of the dataset.

## Why the wiki should keep them distinct

The two questions can give *different* answers. A redundant high-quality example may be a bad acquisition target (low *contribution*, high *quality*). An adversarial low-quality example may have high (negative) attribution. Confusing them leads to wrong curation decisions.

[[sources/feddqc|FedDQC]] is the cleanest data-quality-control paper in the wiki; most other sources are valuation-side. The FL data-quality field has converged on quality scoring (PPL, IFD, NUGGETS, IRA) for *pre-filtering*, and on Shapley/IF for *post-hoc valuation*.

## The robustness-side: detect-and-discard quality control

A second FL quality-control family targets *malicious / noisy / free-riding clients* rather than low-quality points: [[sources/fedcorr|FedCorr]] (prediction-subspace LID → relabel), [[sources/fltrust|FLTrust]] (trusted-cosine trust weighting), [[sources/foolsgold|FoolsGold]] (cross-client gradient similarity), [[sources/fldetector|FLDetector]] (temporal update consistency), [[sources/free-riders-fl-std-dagmm|STD-DAGMM]] (free-rider energy). All output a binary keep/discard (or hard down-weight) — *unsigned* anomaly screening, not signed contribution. Their shared failure mode sharpens the quality/value line: under non-IID, benign-but-different clients look anomalous and get false-flagged (the OOD-good problem). [[flirds|Flirds]] contrasts with the whole family by down-weighting *signed value* instead of hard-discarding. Synthesis: [[threads/noise-ood-malicious-client-separation]].

## See also

- [[concepts/feddqc]]
- [[concepts/influence-function]]
- [[concepts/shapley-value]]
- [[threads/data-quality-vs-data-value]]
- [[threads/noise-ood-malicious-client-separation]] — the FL robustness-side detect-and-discard family
