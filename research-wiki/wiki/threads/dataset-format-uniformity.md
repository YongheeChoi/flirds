---
type: thread
title: Cross-domain format uniformity for FL data valuation
created: 2026-06-04
updated: 2026-06-04
sources: [less, feddqc, mates, datainf]
tags: [flirds, dataset, validation, format, fairness, ablation]
---

# Cross-domain format uniformity for FL data valuation (Flirds)

## The problem (Yonghee, 2026-06-04)

Flirds attributes a **shared** validation-loss change to clients via Shapley. If the cross-silo domains have **heterogeneous task formats** — a 1-token classification / multiple-choice target vs a multi-token free-form generation target — their validation-loss magnitudes are not comparable, so cross-domain (and at N=5, where client = domain, cross-client) valuation is **unfair**. The original D3 5-domain pick had 2 misfits: medical = PubMedQA (yes/no/maybe classification), legal = CaseHOLD (5-way multiple-choice).

## Decision: unify to free-form instruction→response

Cast all 5 domains as free-form generation (loss on target tokens only) — the **FLAN / T0 / Super-NaturalInstructions** standard, inherited by [[sources/less|LESS]], [[sources/feddqc|FedDQC]], IFD, NUGGETS. [[sources/mates|MATES]]'s **objective-alignment** principle (validation objective should equal the training objective = autoregressive generation) is the direct precedent. The cross-domain *valuation-fairness* framing is itself **under-addressed** in prior art (a novelty hook — see normalization below).

**Adopted free-form 5-domain set:**

| domain | dataset | train | val source | note |
|---|---|---|---|---|
| medical | `medalpaca/medical_meadow_medical_flashcards` | 34k | carve | replaces PubMedQA; `input`→`output`, license cc |
| legal | `ibunescu/qa_legal_dataset_train` | 97k | carve | replaces CaseHOLD; `Title`+`Question`→`Answer` |
| finance | `LLukas22/fiqa` | 14.5k | `test` split | kept (free-form) |
| math | `deepmind/aqua_rat` (rationale CoT) | 97k | `validation` split | kept (free-form) |
| general | `databricks/databricks-dolly-15k` | 15k | carve | kept (free-form) |

## Rejected / parked candidates (revive if results warrant)

| domain | dataset | format | prior-work overlap | why parked |
|---|---|---|---|---|
| medical | **PubMedQA** (`qiaojin/PubMedQA`) | yes/no/maybe classification | **FedDQC bench**; D3 pick (3/5 FedDQC overlap) | non-free-form; FedDQC's Acc metric unusable under a uniform-loss valuation |
| legal | **CaseHOLD** (`coastalcph/lex_glue` case_hold) | 5-way MC index | LexGLUE; D3 pick | MC; recast-to-holding-text loses the MC discrimination |
| medical alt | `lavita/ChatDoctor-HealthCareMagic-100k` | free-form (long dialogue) | ChatDoctor | backup if longer completions wanted (license research-only) |
| medical alt | medalpaca medqa / `openlifescienceai/medmcqa` | MC-as-text / MC-index | MedQA / MedMCQA | **NOT free-form — do not use** |
| legal alt | `dzunggg/legal-qa-v1` | free-form | niche | too small (3.7k) for the equalized B1 train budget |
| legal alt | `lawinstruct/lawinstruct` (`*_qa` configs) | free-form (mixed configs) | LawInstruct paper | high-friction 142-file multi-config |
| finance alt | `FinGPT/fingpt-fiqa_qa` | free-form (long) | FinGPT | cleaner FiQA recast; license unlisted |
| math alt | `meta-math/MetaMathQA` (395k) / `openai/gsm8k` | free-form | MetaMath / GSM8K | larger math pools if needed |

License notes: flashcards = cc; aqua-rat / dolly clean; **ibunescu, fingpt-fiqa, dzunggg have no declared license** (fine for a research benchmark; flag if redistribution ever matters).

## Cross-domain normalization + planned ablation

The validation loss already carries **2 of the 4** magnitude mitigations seen in prior art:
- **per-token mean** (LESS-style token-averaging), and
- **Δloss utility** (the (b) in-run utility is a loss *difference*, like FedDQC's IRA `L(a) − L(a|q)`, explicitly meant to remove response-format inconsistency).

Other prior mitigations (none address *cross-domain* comparability): IFD loss-*ratio*, NUGGETS binary-indicator-over-anchors. The instruction-tuning data-selection survey treats the magnitude/format-fairness issue as a **gap**.

**Adopted:** add **per-domain macro-average** of the validation loss (each domain weighted equally, 1/D, instead of token-proportional) — removes the token-count dominance of long-completion domains. **Ablation:** per-domain normalization **ON vs OFF**, measured by downstream task accuracy of the resulting client selection. Fuller magnitude normalization (relative `Δ/baseline`, or per-domain z-score) is a further ablation arm; any change to the utility must be re-checked against the IRDS framing and estimator≈oracle consistency.

## See also
- [[threads/utility-function-design]] — the utility/validation-reference design axis (MATES objective-alignment, LESS few-shot, DRDV validation-free).
- [[threads/data-selection-for-llms]] — LESS / MATES / DsDm reference-set choices.
- [[flirds-implementation-plan]] §3.1 / §3.4 — dataset + validation construction (D3 medical/legal picks superseded here; coordinate distillation).
