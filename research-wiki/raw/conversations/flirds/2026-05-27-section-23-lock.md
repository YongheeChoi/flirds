---
type: conversation
date: 2026-05-27
topic: flirds
participants: [Yonghee, Claude]
tags: [section-2, section-3, experiment-plan, baseline-selection, model-choice, dual-oracle, ripple-reproduction, protocol-lock]
---

# 2026-05-27 — Section 2 / Section 3 lock, baseline reduction, model choice, Ripple sanity task

Distilled record of three contiguous design conversations: 2026-05-19 (Section 2 walkthrough), 2026-05-22 (N1–N4 partial close + Section 3 12-item draft), and 2026-05-27 (resumed after session interruption: wiki state absorption, Section 3 detailed walkthrough, final lock).

Compressed but faithful to Yonghee's actual asks and decisions; Claude's responses paraphrased except where Yonghee's framing depends on the exact wording.

## Starting state (Yonghee, 2026-05-19)

> "연구 계획을 이어서 논의해보자. Flirds 연구 정리 html 파일 기준으로 현황을 파악중이야. 결정이 필요한 부분을 먼저 다 정리하고 보강이 필요한 부분으로 넘어가자."

Workflow rule established up front: **decisions first (Section 2), then reinforcements (Section 3)**.

## Section 2 — decision walkthrough (2026-05-19)

Claude grouped open items into already-closed (① noise-vs-OOD-good deferred 2026-05-18, ⑤ ground-truth from conv3 §2, ⑥ benchmarks) and genuinely open (②③④⑦ + Ripple positioning). Recommendation map:

- **Q1 use case** → research-side in-run attribution (vs incentive). Implies ④ trajectory dependence = feature.
- **Q2 ②③** → both characterized as IRDS-form limits (default); layer-wise / phase-norm = ablation variants. Solving ② breaks the granularity-invariance lemma underpinning Proposition 1; solving ③ breaks estimator-(b)-oracle consistency.
- **Q3 Ripple positioning** → other design points = main claim; reduction = bonus track.
- **⑦ model** → defer until Q1 settled.

Yonghee accepted Q1, Q2, Q3, ④, ⑤ as-is (no objection to the rationale).

## Pilot-data digression and N1–N4 (2026-05-19)

Claude folded in pilot data observations from after the 2026-05-19 wiki snapshot:
- ② cancellation: ρ(Flirds, exact) = 1.0000 across α ∈ {0.1, 0.5, 5.0} (limit *did not* materialize — opposite of predicted).
- ① noisy detection: AUROC = 0.50 ± 0.14 (chance — predicted "not separable" confirmed).
- Prop 1: falsified by U-shape (E=1 OK → E=3 weakened → E=5 worst → E=10 *recovered* via divergence sign-dominance).
- bf16 LR-paired artefact observed at 7B.

Surfaced four new decision points:
- **N1**: Prop 1 handling — (a) drop + empirical, (b) re-derive U-shape predictor, (c) conditional formulation.
- **N2**: Scale upper bound — 1B+3B / +7B / +13B+ / +70B?
- **N3**: How to defend ρ=1.0 from "too good" reviewer skepticism.
- **N4**: Q2 variant comparison metric.

Claude initially recommended N1 = (a) drop + empirical, on the grounds that U-shape replicated across both 1B and 3B, two different architectures. Worried that "ignore the observation, maintain original assumption" risked paper integrity under reviewer replication.

## Yonghee's N1 correction (2026-05-22)

> "N1: 아니 우리가 다시 이론을 단단하게 만들어서 새로 실험을 돌릴거니까 테스트용으로 돌린 실험 결과는 무시하고 제대로 구현해서 다시 실험을 해보자는거지 리뷰어를 속여먹자는 말은 아니야."

**Decision**: pilot data set aside (exploratory only, never enters paper claims); original theory frame (Prop 1 monotone, Prop 2 cubic) maintained as the *target of clean re-runs*. Option (a) Drop becomes a **contingency**: if U-shape re-appears in clean re-runs, switch to empirical-reporting branch.

Claude apologized for misreading "무시" as "발표용 결과에서 빼겠다" (publish-side suppression) when Yonghee meant "explore-side discard + redo correctly" (standard research workflow).

## N2 / N3 / N4 lock (2026-05-22)

- **N2**: Lit-norm survey (Claude background agent) returned "1B+3B+7B" recommendation. Current FL data-valuation field has no 13B+ precedent; centralized LLM IF norm is 7B–13B (LESS, DataInf, Grosse 2023). 13B+ in FL framed as "compute-flex risk that may weaken method-novelty perception." Yonghee accepted 1B+3B+7B at this stage.
- **N3**: Dual oracle reporting — (a) exact retrain SV (community standard) + (b) IRDS-定 batch-cumulative SV (Flirds's actual target) separately reported. ρ=1.0 reframed as "(b) sense sanity check; (a) sense reports gap honestly." Adversarial stress regimes added. MC precision (variance / CI / ties) lock.
- **N4**: All 3 metrics (client-selection convergence, downstream task acc/F1, noisy/free-rider AUROC) full matrix vs. all Q2 variants (default / layer-wise / phase-norm).

Claude drafted Section 3 with 12 items + asked Yonghee for selection on 4 trade-offs.

**Session interruption** (between 2026-05-22 and 2026-05-27 — Yonghee's UI froze). Yonghee restored the conversation from JSONL transcript.

## Resumed (2026-05-27) — wiki absorption first

Yonghee:
> "먼저 5일 동안 바뀐 wiki 상태부터 흡수하고 나서, Section 3에 대해 자세히 설명을 해줘. 내가 잘 이해할 수 있도록. 그리고 html 파일은 이제 필요없어서 삭제했으니 재구성 할 필요 없고 나머지 task는 section 3에 대해 자세한 설명을 듣고 결정한 이후에 수행해줘."

Two structural changes:
1. **HTML summary file deleted** — no longer maintained.
2. **Task order**: explanation → decision → execution. No premature wiki rewrites.

Claude absorbed the 2026-05-22 wiki additions: [[sources/less|LESS]], [[sources/grosse-llm-influence|Grosse 2023]], [[sources/mates|MATES]], [[sources/dsdm|DsDm]], [[concepts/ekfac]], [[concepts/proximal-bregman-response|PBRF]], [[threads/data-selection-for-llms]]. Identified three impacts on Section 3:
- N2 1B decision **reinforced** by MATES (1.1% abs gain on Pythia-1B refutes "1B too small").
- N2 7B decision **reinforced** by LESS (Llama-2-7B/13B + LoRA centralized norm) + FedDQC precedent.
- N3 dual oracle **reinforced** by Grosse/MATES PBRF reframing — "(b) IRDS-定 is a well-defined target, not a counterfactual proxy."

## Section 3 walkthrough — Yonghee's question-by-question decisions (2026-05-27)

### #1 Precision + statistical reporting

Yonghee: "training부터 모든걸 fp32로 하지 않을 이유가 있어? 보통 다른 논문들에서는 어떻게 하는데? 'ties 처리 / MC 분산 / CI band 명시'가 뭐야?"

Claude: bf16 mixed-precision = standard for 1B–13B LLM training under B200; fp32 evaluation is the standard fix for the bf16 numerical-precision artefact suspected in the pilot. ties = scipy `rankdata(..., method='average')`; MC variance = report estimator standard deviation alongside point estimate at pre-declared sample $M$; CI band = 95% bootstrap CI on every headline metric.

Yonghee: "fp는 관행대로 처리하고, ties는 scipy 표준대로 tied-rank 부여, MC 분산도 제안한 표준대로 하고, CI band도 넣자."

**Locked**.

### #2 Baseline reduction

Yonghee asked Claude to (i) include all recommended baselines, (ii) check Ripple-comparable prior work for what they used as baselines, (iii) check code availability, (iv) propose anything missing.

Claude searched code availability (mix of wiki + web):
- Code available: [[sources/shapleyfl|ShapleyFL]] (ZJU-DIVER repo), [[sources/in-run-data-shapley|IRDS]] (GhostSuite, centralized only), [[sources/less|LESS]], [[sources/datainf|DataInf]], [[sources/logix|Logix]], [[sources/mates|MATES]] — but most of these are centralized; FL Shapley side has poor code availability.
- Code unavailable: Ripple (AAAI'26 newest), GTG, FedSV (Wang 2020), ComFedSV, S-FedAvg, FedDQC (OpenFedLLM-based, no clean repo).
- Coverage: Yonghee's full baseline list covers prior FL Shapley work near-completely.

Yonghee: "space 같은건 llm에서 사용할 수 없지 않나? 우리 연구 환경에서 적용하기 적절하지 않은 baseline이 있는지 검토해줘."

Claude reviewed LLM applicability:
- **SPACE** → ❌ prototype-based evaluation requires discrete classes (LLM generation is open-ended).
- **S-FedAvg** → ❌ aggregation method, not a valuation baseline.
- **FedCorr** → ❌ prediction-subspace LID assumes classification.
- All others LLM-applicable.

Yonghee: "지금 조사 결과 포함 권장하는 baseline들만 포함하는걸로 할게 (S-FedAvg, FedCorr 제외). 근데 FedAvg와 같은 기본 fl 방법을 통해 학습한 baseline도 완전 기본 학습 방법과의 성능 비교를 위해 필요하지 않아?"

Claude confirmed: vanilla FedAvg = *training comparison* baseline (not a valuation baseline). Two forms:
- **Full FedAvg (all clients, no selection)** — upper-bound floor.
- **Random-selection FedAvg (top-K random)** — random baseline our valuation must beat.

Yonghee added these. Also mentioned Power-of-choice (Cho et al. ICML'22) as an optional 3rd-party system-level selection baseline.

Yonghee (next turn): "Power-of-choice는 안해도돼."

**Locked**: 10 baselines + 4 detection. SPACE / S-FedAvg / FedCorr / Power-of-choice excluded.

### #3 Retrain SV scope + (b) exact vs MC

Yonghee asked about the retraining-SV trend in prior FL work and whether (b) really needs MC.

Claude analysis:
- (a) retrain SV: LLM-scale + retrain SV essentially does not exist in prior work (FedDQC didn't do retrain SV at 7B; LESS/Grosse/IRDS all centralized + no exact Shapley). Feasibility per scale on B200×4 with LoRA-16, R=50: 1B + N=10 ≈ 3.5 days; 3B + N=10 ≈ 10 days; 7B + N=10 ≈ 42 days. *7B retrain SV is infeasible.*
- (b) exact vs MC: at cross-silo N=10 (1024 subsets), forward-only enumeration is **3.5h / 7h / 28h** at 1B/3B/7B respectively. MC is *not necessary* at cross-silo. Cross-device N=100 requires MC ($2^{100}$ subsets).

Yonghee: "이걸 보니까 다시 1B, 3B, 7B로 실험을 해야할 것 같아. 그리고 retraining은 (1B 모델 + client 5, 10개) 그리고 (3B 모델 + client 5개) 정도 수준에서만 검증하는게 좋을 것 같아. (b) cross-device만 그럼 MC를 사용하는걸로 하자."

**Locked**:
- (a) retrain SV: 1B + N∈{5, 10}; 3B + N=5; 7B skipped.
- (b) IRDS-定 SV: cross-silo exact enumeration; cross-device MC.

### #4 Ripple positioning + sanity reproduction

Yonghee: "ripple은 2차 근사항을 사용 안했다는 점이 FL 환경에서 큰 불이익이라고 생각해 그걸 성능 차이의 원인으로 주장해볼 생각이야. Theoretical reduction도 시도해보자. 근데 ripple이랑 우리랑 기여도를 평가하고자 하는 대상이 다르기 때문에 sv 자체를 얼마나 잘 근사하는가 레벨에서는 비교가 어려울 것 같고, 학습 성능 측면에서 비교해 봐야할 것 같아."

**Locked**:
- Ripple comparison metric = **training-performance side** (selection / task acc / AUROC), NOT SV-approximation side (granularity mismatch makes that ill-defined).
- Hypothesized cause of performance gap = Flirds's 2nd-order client-interaction term, which Ripple lacks.
- Theoretical reduction attempted as bonus: under LoRA + 2-term Taylor, does Ripple drop+ripple specialize to Flirds 1st + 2nd?

Yonghee (next): "구현을 제대로 못한거 아니냐는 공격에 대비하기 위해서 ripple shapley 논문에서의 실험 setup과 같은 실험을 우리가 해야하고 그 결과가 어느정도 일치하게 나오는걸 확인해야해. 지금 계획한 실험이면 그게 가능한가?"

Claude flagged: current plan does NOT include reproducing Ripple's *own* (CNN + MNIST/CIFAR-10) reported numbers. Added as new Phase 0 task: reproduce Ripple's headline metric in its original setup. ±5% reproduction = LLM transfer green light.

Yonghee extended: "이건 ripple 뿐만 아니라 다른 baseline에 대해서도 하면 좋을 것 같은데?"

**Locked**: Phase 0 sanity reproduction for **all 4 code-unavailable baselines** (Ripple, GTG, FedSV, ComFedSV) in their original (CNN + MNIST/CIFAR-10) setups. Cost ≈ 5–7 days B200×1. Must pass before LLM phase starts.

Side benefit Yonghee accepted: building these reproductions also constructs the FL simulator + FedAvg + utility evaluator infrastructure re-used for the LLM phase.

### #10 Precision protocol → folded into #1

Yonghee: "응 1번에 통합해줘."

**Locked**: no separate #10 item; precision specification lives in [[flirds-protocol]] §1.

### #11 Model choice

Yonghee: "1B, 3B, 7B 조합으로 다시 롤백해야해. 3B는 Llama-3.2-3B를 사용하고 7B는 Llama-2-7B를 사용하면 모델의 다양성도 추구하면서 기존 연구와 실험 결과 비교도 가능하니 좋을 것 같아. 다만 1B 모델에 대한 부분은 다시 조사해줘."

Claude analyzed 1B candidates:
- **Llama-3.2-1B-Instruct** (Option A) — Llama-3.2 family consistency with 3B; modern instruction-tuned; aligned with original wiki plan.
- **Pythia-1B** (Option B alt) — direct comparison with IRDS / MATES; but pretraining-focused, awkward for instruction tuning.
- **Qwen-2.5-1.5B**, **OLMo-1B**, **TinyLlama-1.1B** — secondary options.

Recommended Option A. Yonghee: "Option A로 합시다."

**Locked**:
- 1B = **Llama-3.2-1B-Instruct**
- 3B = **Llama-3.2-3B-Instruct**
- 7B = **Llama-2-7B**

### #12 70B mention

Yonghee: "70B는 실험한 선행 연구가 있어? 없으면 그냥 굳이 논문에서 언급도 안해도 될거같은데."

Claude searched: no FL data-valuation precedent at 70B; centralized maxes at Grosse's 52B (Anthropic internal). 

Yonghee: "그럼 그냥 언급 안하는걸로 처리하면 될 것 같아."

**Locked**: 70B not mentioned in paper text, future work, or anywhere.

## 7B compute re-check (final 2026-05-27 exchange)

After Claude proposed reduced experiment-matrix coverage at 7B, Yonghee questioned:

> "실험 매트릭스에서 7B 모델에 대해 Baseline 비교, α-sweep × E-sweep, Q2 변형 (3×3), Stress regimes 전부 full로 하면 안될 이유가 있나? 이거만 확실히 하고 다음 단계로 가자"

Claude recomputed 7B cost on B200×4 with LoRA-16, bf16, R=50:
- α-sweep × E-sweep (16 cells × 3 seeds) ≈ 48h
- Q2 variants selection convergence (3 × 3 seeds) ≈ 9h
- Stress regimes (5 × 3 seeds) ≈ 15h
- Baseline post-hoc on same trajectory ≈ 0
- Vanilla + random selection (4 × 3 seeds) ≈ 12h
- (b) exact in-run oracle × 3 seeds ≈ 3.5 days
- **Total 7B ≈ 1 week B200×4** — entirely feasible.

The "reduced 7B" was Claude's overly-conservative residual from when 13B was still in the plan. With 13B excluded, 7B is comparable to 3B in cost. **7B full matrix locked**.

## Final Section 3 lock (11 items)

11 items including Phase 0 sanity reproduction. See [[flirds#Experiment plan — Section 3 (locked 2026-05-27)]] for the structured list and [[flirds#Experiment matrix (locked 2026-05-27)]] for the scale × experiment table.

## Yonghee's preferences surfaced (for memory / future sessions)

- **Pilot data ≠ paper data**: exploratory observations don't enter paper claims; only clean re-runs do. Standard research workflow, not a publication-side concern.
- **Decisions before reinforcements**: Section 2 first, Section 3 only after Section 2 is fully locked. Don't propose Section 3 items while Section 2 still has open decisions.
- **Code availability matters for baselines**: prefer baselines with public code; code-unavailable baselines require sanity reproduction in their original setup before being used in our experiments.
- **Vanilla FedAvg as training-comparison baseline**: training-side comparison ≠ valuation-side comparison; both are needed.
- **Compute realism**: Yonghee challenges conservative compute estimates that limit coverage unnecessarily — asks for explicit recomputation.
- **Model family consistency**: prefers family-consistent scaling lineage (1B/3B both Llama-3.2) over family-diverse coverage (different family per scale).
- **HTML summary file deprecated**: as of 2026-05-27. Wiki = single source.
- **Workflow rule**: explanation → decision → execution. No silent wiki writes before user signs off on the plan.

## Files touched in 2026-05-27 lock execution

- [[flirds]] — comprehensive rewrite of Locked design decisions, Resolved questions, Experiment plan, Experiment matrix, Baseline selection rationale; date stamped 2026-05-27.
- [[flirds-protocol]] (new) — implementation & reporting protocol document (12 sections covering precision, seeds, statistical reporting, oracle separation, sanity gates, run logging, Phase 0).
- [[log]] — appended `[2026-05-27] conv` entry.
- [[index]] — added flirds-protocol to overview & meta section.
