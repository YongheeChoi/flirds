---
type: project
title: "Flirds — Implementation Plan & Session Handoff"
created: 2026-05-27
updated: 2026-06-03
tags: [flirds, implementation, handoff, phase-0, phase-1, phase-2, phase-3, open-decisions]
---

# Flirds — Implementation Plan & Session Handoff

> **Read this first when starting an implementation session.** This document is self-contained: it tells you (i) what's already decided and where to look for detail, (ii) what's still open and how to decide it, (iii) the phase-by-phase task ordering. The 2026-05-27 design session ended here — the next session (this or another) starts from this document.

## Status snapshot (as of 2026-05-27)

- **Pipeline stage**: `idle` → ready for `implementation` (per root `CLAUDE.md`).
- **Theoretical scaffolding**: complete. Section 2 (decisions) + Section 3 (experiment plan) + Protocol = locked. See [[flirds]] and [[flirds-protocol]].
- **Wiki**: 30 sources ingested, 11 threads, 27 concepts. No paper-blocking ingest gaps.
- **What's NOT decided yet**: 9 implementation-detail items (§3 below). These must be resolved before LLM-phase code is written, but most are deferrable past Phase 0.
- **Next concrete action**: Phase 0 + Phase 0.5 complete (see the 2026-06-03 section below); next = **Phase 1 LLM port** (OpenFedLLM + LoRA).
- **Phase 1 carry-over (2026-06-03 wiki session)**: before/while building the LLM data + estimator layers, honor the **two seams** (⚠ callout in §2 Phase 1) — (1) per-layer φ logging *observation-only*, never reweighted in the spine (INVARIANT recorded there); (2) pluggable corruptor registry (design + (a)/(b) fork in **§3.10**). Validation-set is already config-driven (seam 3, §3.10 note). Experiment plan was extended with 7 prior-art validation gaps (#12–18, all Phase 2/3) — see [[flirds#Added 2026-06-03 — prior-art validation gaps (Phase 2/3; none touch the Phase 1 core)|flirds §Added 2026-06-03]].
- **Phase 1 progress (2026-06-03 session 2)**: estimator/oracle made **backend-agnostic** (`loss_fn(params,buffers)`+pkeys injection, `backends/cnn.py`) + **partial-participation-correct** (per-round FedAvg weight) + **per-layer φ logging** (seam 1, observation-only); all CNN gates bit-identical. OpenFedLLM cloned to gitignored `external/` + scouted (its fedavg weight == our per-round weight). **LLM stage 2** (`backends/llm.py` + FL loop self-build + data layer / seam 2) handed to the dedicated Phase 1 session — estimator/oracle need no further change. Raw: [[raw/conversations/flirds/2026-06-03-phase1-backend-abstraction]].
- **Phase 1 progress (2026-06-04 session)**: LLM stage 2 **backend + FL loop done** — `backends/llm.py`(make_llm_loss, LoRA-only via functional_call) + `fl/server.py` `_fedavg_core` 추출(CNN wrapper **bit-identical 회귀**) + `fl/llm_server.py`(run_llm_fedavg_logs, **TRL SFTTrainer 1.x + forced SGD + completion-only**). **LLM-FL 스모크 green**(Llama-3.2-1B real 궤적, est≈oracle 1.70e-6). **3 LLM musts**: eager-attn / named-key state / embedding-hook clear (functorch+HF). validation(§3.4) 확정(도메인당 200/총 1000; **1000 vs 2¹⁰=1024 subset 분리**). **남음 = 3번 data layer**(5-domain + seam 2 corruptor). Raw: [[raw/conversations/flirds/2026-06-04-phase1-llm-stage2]].

## ✅ Decisions resolved & corrections (2026-06-02 implementation kickoff)

2026-06-02 세션에서 §3의 9개 open decision 전부 + 추가 결정을 확정하고 Phase 0 구현에 착수했다. 요약:

**확정 결정**

| 항목 | 결정 |
|---|---|
| BASE_REPO (§3.6) | LLM = **OpenFedLLM** fork(`codes/base_repo/`, LLM phase); CNN = **자체 경량 시뮬레이터**. 얇은 공통 FL core 위 backend 분리, estimator/oracle 은 backend-agnostic |
| 코드/빌드 (D1) | baseline 코드 있으면 fork, 없으면 self-build |
| 로깅 (D2) | **W&B 미사용** → 로컬 run-dir(config YAML + env hash + git SHA + per-round per-client φ parquet) |
| cross-silo 데이터 (§3.1) | 5-domain: **PubMedQA**(의)/**CaseHOLD**(법)/**FiQA**(금)/**AQUA-RAT**(수)/**Dolly**(일반). LESS protocol(validation·selection 방법론). FedDQC 와 3/5 도메인 데이터셋 동일 |
| cross-device 데이터 | **Fed-WildChat**(FedLLM-Bench, 자연 N=100) + **FedHDS**(NI task별 + Dolly Dirichlet) 둘 다 |
| FedDQC 비교 | matched-arm 미실시; **IRA 점수법만 baseline 으로 이식** |
| LoRA (§3.2, D4) | r=16/α=32 시작 + **rank sweep {16,32,64,128}**(α=2r) — attribution fidelity vs task perf ablation |
| 학습 hp (§3.3, D5) | lr 2e-5, batch 16, cosine — 시작값, sweep |
| validation (§3.4, D6) | server-side held-out, **도메인당 200 / 총 1000, uniform stratified**; canonical dev split 우선·없으면 fixed-seed stratified carve. IRDS-held-out 평균 loss 관점(few-shot 아님). 크기 1000 = 2¹⁰ coalition-subset 과 분리(혼동 차단) |
| cross-device 세부 (§3.5, D7) | N=100/K=10/R=200, default **α=0.5**, α-sweep **{0, 0.01, 0.1, 0.5, 5.0}**(α=0 = domain-disjoint 포함). late-joiner 는 구조상 존재 → 논문 텍스트로 해석; extreme regime drop |
| detection (D8) | **FLDetector**(noisy/poisoning) + **STD-DAGMM**(free-rider) 각 1개 (Phase 2) |
| 협업 방식 | 설계 분기점마다 논의, 일상 구현은 후 리뷰. **커밋은 요청 시만** |

**CNN 트랙 추가** (전체 실험 LLM 과 병행; 메인 = LLM; LoRA 미적용 full-param):
- 표준 세팅(OpenDataVal + FL valuation canon 교집합): MNIST/FMNIST = **LeNet-5**(raw px), CIFAR-10 = frozen ResNet feature+MLP(canon 비교) & small-CNN(FedSV, FL 비교); N=10 full(silo) + N=100 K=10(device); IID + Dir(0.5) + Dir(0.1) + McMahan-shard; label-flip 10%(+20%) & client-level 10/100@30%; **exact-retrain GT ≤N=10**(1023 coalition) + KNN-SV(point) + synthetic-logistic(deterministic); metrics = Spearman ρ / Jaccard / F1@10pct / discovery curve / point-removal / seed-stability.
- 이점: exact-SV GT 가 싸서 estimator 를 **큰 N 까지 검증** + detection baseline 원셋업 직접 비교.

**Phase 재구성**: Phase 0(CNN baseline 재현 gate) → **Phase 0.5**(Flirds estimator + dual oracle 을 CNN 에서 먼저 검증) → Phase 1+(OpenFedLLM 백엔드로 LLM 이식, 이때 LLM/CNN 공통 인터페이스 추출).

**플랜 corrections (원문 대조로 발견 — 기존 §2/§3 본문보다 우선)**:
- **GTG-Shapley·ComFedSV 는 코드 존재** → fork 대상 (GTG = `github.com/liuzelei13/GTG-Shapley`; ComFedSV = Huawei AI Gallery). FedSV·Ripple 만 self-build. (§ baseline 표의 GTG/ComFedSV ❌ 는 오류)
- **Ripple "62× vs GTG" 는 오류**: 실제 62.37× vs AFedSV+, 49× vs FedSV; GTG 는 Ripple baseline 아님. Ripple 은 ground-truth SV 지표 없음(task-driven robustness 만).
- **Phase 0 scalar 로 보고된 건 ComFedSV 뿐**(Spearman {1.0,0.96,0.85,0.84}). GTG/FedSV/Ripple 은 figure-only → 파생 scalar(GTG cosine<0.01 / FedSV recall@20%-inspected / Ripple runtime ratio).
- **FedDQC LoRA = r=64/α=128** (§3.2 의 "FedDQC r=16/α=32" 는 오류). R=100, 2/5 clients/round, 10 local steps(≈3ep), cosine 1e-4→1e-6, batch 16, 8-bit.
- **FedDQC non-IID = 단일도메인 quality-het**(50% answer-swap), domain-per-client 아님.

**구현 착수 상태**: 브랜치 `feature/flirds-phase-0`; conda env `flirds`(torch 2.12+cu130, B200×4, Python 3.11). Phase 0 step1 FL 시뮬레이터(`codes/flirds/{fl,models,data}` + `experiments/phase0_smoke.py`) 작성 + 스모크 통과(MNIST 3r 96.3%). step1 verify(MNIST~97%/CIFAR~77%) 진행 중.

## ✅ Phase 0 + Phase 0.5 complete (2026-06-03)

`main`에서 Phase 0(베이스라인 4종) + Phase 0.5(estimator + dual oracle)를 CNN 트랙에서 구현·검증·리뷰·확정. 상세: [[raw/conversations/flirds/2026-06-03-phase05-estimator]], [[flirds#Phase 0.5 findings — 2nd-order term & dual oracle (2026-06-03, CNN)]].

- **구축**: (b) in-run SV oracle (`oracle/in_run_sv.py`, exact 2^N) + Flirds estimator (`core/flirds_estimator.py`, 참 Hessian 1st/2nd, 라운드당 HVP 1회 + N dot) + **faithful Ripple 재작성** (`baselines/ripple.py`, Eq 5-19 — Phase-0 ripple 부호·α 버그 2개 수정). (a) retrain oracle = `oracle/exact_sv.py`(Phase 0). 베이스라인 GTG/FedSV/ComFedSV self-build.
- **검증(전 게이트 green)**: estimator≈(b) Spearman 1.0 / 3-seed 1+2nd 0.96; (b) Shapley efficiency·symmetry exact 0; HVP jvp-vs-double-backward 9.8e-6; (a)retrain↔(b)in-run noisy AUROC 1.0(fine-rank 0.66, 다른 utility); 재현성 bitwise-0; GTG/FedSV recon cosine 0.999.
- **확정 결정**:
  - 2nd-order curvature = **참 Hessian** (IRDS 일치; GGN 변형 검증 후 기각 — indefinite 곡률에도 참 Hessian 우월).
  - **momentum 제거 → plain SGD** (`local_train`/ripple 기본 0.0): IRDS/Ripple Eq 1 가정 일치 + **2차항이 비로소 1차를 이김**(momentum 0.9: 0.73<0.81 → plain: 0.96>0.92). IRDS 의 "2차항 marginal" 은 *centralized per-step* 산물; **FL per-round 가 2차항의 무대** — 데이터로 확인. 모든 valuation 실험에 통용.
  - **재현성**: `flirds/repro.py:seed_everything`(torch+np+cuda + cudnn-det). cudnn-det 은 **CNN(conv) 전용**(LLM 불필요). `fedavg`·`ripple_shapley`·모든 실험 main 호출; `codes/CLAUDE.md §5` 컨벤션.
- **이월(deferred, 기록됨)**: ripple `(rounds,n,P)` streaming + eigsh(LA/LM·v0·수렴) → LLM 스케일; estimator/oracle full-participation 가정 → cross-device Phase 2; ripple training-loop eval → BN 모델; ripple-term task-driven 검증 → Phase 3(backdoor/temporal).
- **다음**: **Phase 1 = LLM 이식** (OpenFedLLM + LoRA) — FL per-round + 실곡률이 2차항을 본격 시험.

## 1. What's already decided (read these documents)

| Document | What it covers |
|---|---|
| [[flirds]] | All locked design decisions (original 2026-05-05 + N1–N4 + Q1–Q3 + ④⑤ + model choice + baseline reduction); resolved questions; Experiment plan §3 (11 items); Experiment matrix (1B/3B/7B × 9 experiments); baseline-selection rationale + code availability table; differentiators table; centralized positioning |
| [[flirds-protocol]] | bf16 train / fp32 eval rule; seeds ≥ 3; scipy tied-rank; MC variance reporting; 95% bootstrap CI band; (a)/(b) oracle code-path separation; sanity gates ($E{=}1$, $N{=}2$); run logging (config / env / git SHA / W&B); Phase 0 reproduction protocol; implementation skeleton (directory layout); pre-publication checklist |
| `raw/conversations/flirds/conversation1.md → conversation4.md` | Original design conversations with another LLM (IRDS → FL adaptation, math derivation, locked design choices). **Primary record for math + design rationale.** |
| `raw/conversations/flirds/2026-05-19-section23-walkthrough.md` | Raw turn-by-turn transcript of the 2026-05-19 → 22 Section 2/3 walkthrough (restored from JSONL after session interruption). |
| `raw/conversations/flirds/2026-05-27-section-23-lock.md` | Distilled record of the same arc + 2026-05-27 lock + Yonghee preferences surfaced. |

**Do not re-litigate locked decisions** unless implementation reveals new information. If something seems unclear, search [[flirds]] first, then the raw conversation files.

## 2. Phase structure (do these in order)

Four phases. **Phase 0 is a hard gate**: no LLM-phase code is written until Phase 0 reproductions pass.

### Phase 0 — CNN baseline sanity reproduction (gate)

**Goal**: reproduce headline metrics of code-unavailable baselines in their *original* (CNN + MNIST/CIFAR-10) setups, within ±5%. **Engineering by-product**: build the FL simulator + FedAvg loop + utility evaluator + Ripple sample-level→client aggregator that the LLM phase reuses.

**Targets** (4 baselines, all code-unavailable; setup details in respective wiki pages):
1. [[sources/gtg-shapley|GTG-Shapley]] (ACM TIST'22): CNN + MNIST/CIFAR-10, N=10. Headline: Spearman ρ vs uniform MC; runtime advantage. ~1 day.
2. [[sources/principled-federated-data-valuation|FedSV (Wang 2020)]]: CNN + MNIST/CIFAR-10 (IID + non-IID), N=10. Headline: noisy-label / backdoor detection rate. ~1 day.
3. [[sources/comfedsv|ComFedSV]] (ICDE'22): Synthetic / MNIST / FMNIST / CIFAR-10, N=100 (10 noisy). Headline: Spearman vs ground-truth, Jaccard on noisy detection. ~2–3 days.
4. [[sources/ripple-shapley|Ripple Shapley]] (AAAI'26): CNN + MNIST/CIFAR-10, N=10. Headline: Spearman ρ, 62× speedup vs GTG. ~1–2 days. **If pseudocode is ambiguous → contact authors (AAAI'26 newest).**

**Total cost**: 5–7 days on B200 × 1.

**Pass criterion**: each baseline's headline metric within ±5% of the value reported in the original paper. Otherwise: (i) debug implementation, (ii) author contact, (iii) document the gap with a caveat in the paper's reproduction section.

**Pass gate**: Phase 0 passing produces the *foundation infrastructure* for Phase 1. Do NOT skip Phase 0.

### Phase 1 — Flirds core + dual oracle + vanilla FedAvg at 1B

**Goal**: a working Flirds estimator on Llama-3.2-1B-Instruct, both oracles operational at N=10 cross-silo, vanilla FedAvg upper-bound running. All sanity gates green.

**Tasks** (sequence; later items depend on earlier):
1. Adapt Phase 0 FL simulator to LLM + LoRA + bf16 (with fp32 eval guard from [[flirds-protocol]] §1). **Keep the data-layer client-corruptor / partitioner as a pluggable registry** (mirror the Phase 0.5 CNN data layer): noisy / free-rider now; backdoor / PGD / maverick / duplicate added in Phase 2/3 without touching the FL loop. ← **Seam 2** (see callout).
2. Implement Flirds estimator (`core/flirds_estimator.py` from [[flirds-protocol]] §11):
   - 1st-order: $-\nabla\ell(w^r, z^{val}) \cdot \Delta w_k$
   - 2nd-order: $\tfrac{1}{2}\Delta w_k^\top H^{(val)}(w^r) \cdot \Delta W^{(r)}$ via 1 HVP on LoRA params + $N$ dot products
   - Per-round + total $\phi_k$ aggregator + per-round **× per-client × per-layer** raw logger — store the **per-layer dot-product components**, not just the scalar $\phi_k$ (needed by ② decomposition / Q2-layer-wise / PGD-#13 / qualitative-#17). ← **Seam 1** (see callout).
3. Implement (b) IRDS-定 in-run oracle (cross-silo exact, 1024 subset enumeration). Separate code path from Flirds estimator.
4. Implement (a) retrain SV oracle at N=5 (32 retrain runs). Separate code path from (b).
5. Vanilla FedAvg (all clients, no selection) + Random-selection FedAvg baselines.
6. **Sanity gates green** ([[flirds-protocol]] §5): $E{=}1$ ⇒ residual ≈ 0; $N{=}2$ ⇒ singleton SV matches client value; reproducibility (same config + seed → bitwise identical at fp32).
7. **First clean Flirds run on Llama-3.2-1B-Instruct** with all 3 seeds, all 4 metrics (selection convergence / task acc / F1 / noisy-AUROC) logged.

**Output**: working Flirds at 1B + dual oracle (a)+(b) + 2 training baselines + reproducible per-run logging (local run-dir, no W&B per 06-02). **No paper claim yet** — just infrastructure validated.

> **⚠ Phase 1 seams (added 2026-06-03 — honor now; cheap now / costly to retrofit).** The 7 experiments added to [[flirds#Added 2026-06-03 — prior-art validation gaps (Phase 2/3; none touch the Phase 1 core)|flirds §Added 2026-06-03]] are all Phase 2/3 (data/eval-layer) and do **not** expand Phase 1 scope. But two depend on Phase-1 *core* decisions being made right:
> 1. **φ logging granularity** = per-round × per-client × **per-layer** (keep the per-layer dot-product components). Scalar-$\phi_k$-only forces an estimator re-run for ② / Q2-layer-wise / PGD / qualitative.
> 2. **Pluggable client-corruptor / partitioner** in the data layer. Hardcoded noisy/free-rider forces a refactor when backdoor (#12) / PGD (#13) / maverick (#15) / duplicate (#14) arrive.
>
> Both are additive seams the Phase 0.5 CNN track already has — just carry the shape into the LLM data + estimator layers. Nothing else in Phase 1 changes.
>
> **⚠⚠ INVARIANT for seam 1 (Yonghee, 2026-06-03) — per-layer φ is OBSERVATION-ONLY; it must NOT break the IRDS proof.** In `core/flirds_estimator.py`, $\phi_k$ is *already* a sum over layers (`sum((g[n]*dw[k][n]).sum() for n in pkeys)`). Per-layer logging = keep those summands instead of collapsing early; the returned $\phi_k$ stays `sum(components)`, bit-identical. The default/spine estimator **never reweights** per-layer (no $\mathrm{diag}(c_\ell)$, no layer selection) — that would break the granularity-invariance lemma + Proposition 1. Per-layer components are a **read-only diagnostic array** consumed *only* by ② characterization (#7) and the qualitative case study (#17). **The one intentional exception** is the locked **Q2 layer-wise *variant* (#6)** — a separate, clearly-labeled ablation arm that deliberately reweights to *show* the lemma break; it is never the headline method. Implementation: add per-layer logging as a backward-compatible option (e.g. `per_layer=False` returning the extra array) so the Phase 0.5 callers are untouched.

### Phase 2 — Full baseline set + (a) retrain SV expansion + 3B/7B scale up

**Goal**: all 10 baselines + 4 detection methods working at 1B/3B/7B; (a) retrain SV expanded to N=10 at 1B + N=5 at 3B; cross-device MC setup operational.

**Tasks**:
1. Port Phase 0 reproduced baselines (GTG / FedSV / ComFedSV / Ripple) from CNN setup to LLM+LoRA setup. ComFedSV cross-device only.
2. Implement Data Banzhaf in FL via semivalue library (`pyDVL` or `OpenDataVal`). Adapt for FL granularity.
3. Vendor ShapleyFL from `ZJU-DIVER/ShapleyFL-Robust-Federated-Learning-Based-on-Shapley-Value` — adapt for LLM+LoRA.
4. Implement loss-heuristic + Flirds-1st-only (trivial; sanity).
5. Implement detection baselines for the noisy / free-rider AUROC table. **06-02 decision narrowed this to 2: [[sources/fldetector|FLDetector]] (noisy/poisoning) + [[sources/free-riders-fl-std-dagmm|STD-DAGMM]] (free-rider)** — not the 05-27 set of 4.
6. Expand (a) retrain SV: N=10 at 1B (~3.5 days B200×4), N=5 at 3B (~45 min). 7B (a) skipped per Section 3 lock.
7. Cross-device MC for (b): N=100, K=10, M=5000–10000 samples (precision threshold from [[flirds-protocol]] §3.2). ComFedSV becomes valid baseline here.
8. Scale Flirds + all baselines to Llama-3.2-3B-Instruct and Llama-2-7B.
9. **(2026-06-03 add) Client-corruptor extensions** via the pluggable data layer: **backdoor / model-replacement** (#12) + **clean-only partition** for the skyline (#18). No FL-loop change.

**Output**: full baseline + dual oracle matrix operational at all 3 scales. Ready for experiment execution.

### Phase 3 — Experiment matrix execution + Ripple reduction attempt

**Goal**: every cell of the [[flirds#Experiment matrix (locked 2026-05-27)|experiment matrix]] populated with 3-seed mean ± std + 95% CI; characterization experiments complete; Ripple theoretical reduction attempted.

**Tasks**:
1. **α-sweep × E-sweep matrix** (§3 item #5): $E\in\{1,3,5,10\}$ × $\alpha\in\{0.01, 0.1, 0.5, 5.0\}$ × 3 seeds = 48 runs × 3 scales = **144 runs**. Validates Prop 1 + Prop 2. **N1 contingency branch point**: if U-shape recurs in clean run, switch to N1 option (a) Drop + empirical reporting.
2. **Q2 variant comparison** (§3 item #6): {default, layer-wise weighted, phase-normalized} × {3 metrics} × 3 seeds × 3 scales.
3. **②③ characterization** (§3 item #7): per-layer $\phi_k^{(r)}$ decomposition (②); late-joiner test (③).
4. **Adversarial stress regimes** (§3 item #8): extreme $\alpha$, label-flip × OOD, larger $N$, late-joiner extremes. ρ on both (a) and (b) reported.
5. **Non-IID bias quantification** (§3 item #9): extracted from #1 data (no extra runs).
6. **7B FedDQC-comparable instruction-tuning bench** (§3 item #10): 5-domain split, direct comparison to FedDQC + LESS.
7. **Ripple theoretical reduction** (§3 item #4 bonus): under LoRA + 2-term Taylor, attempt the drop+ripple → Flirds 1st+2nd specialization proof. If closes → Proposition. If not → related-work differentiator paragraph.
8. **(2026-06-03 adds — prior-art validation gaps, §3 items #12–18):**
   - **PGD / direction-aligned poison** (#13) — stress regime; pair with Flirds-1st-only to test whether the 2nd-order term separates what [[sources/fedif|FedIF]]'s 1st-order cannot. *Differentiation experiment.* (scope to 1B+7B if budget-bound)
   - **Partial-participation fairness** (#14) — duplicate-client test on cross-device; validates the "no participation normalization" lock.
   - **Maverick / rare-domain** (#15) — sole-domain-holder client; sharp form of the OOD-good limitation.
   - **Validation-set sensitivity** (#16) — size × distribution ablation, post-hoc re-eval (re-added from conv3 §3).
   - **Backdoor detection eval** (#12) — AUROC column alongside noisy/free-rider (corruptor built in Phase 2 #9).
   - **Qualitative attribution case study** (#17) — domain-val → which client credited; post-hoc on logged per-layer φ, near-free.
   - **Clean-data skyline** (#18) — upper bound for detection (partition built in Phase 2 #9).

**Output**: all numbers for the paper, with reproducibility metadata. Hand off to `/auto-review-loop` / `/paper-writing`.

## 3. Open implementation decisions (3.1–3.8 resolved 2026-06-02; 3.9 + 3.10 pending — resolve as you encounter)

Each item: **what's open** → **current default / assumption** → **options** → **decision criterion** → **when**.

### 3.1 Dataset choice (cross-silo + cross-device)

**Open**: which exact datasets to use as the 5-domain instruction-tuning bench (cross-silo) and the cross-device scale benchmark.

**Default assumption** (placeholder): cross-silo = FedDQC bench expanded + 1 domain; cross-device = Super-NaturalInstructions.

**Options**:
- (a) **FedDQC bench as-is**: PubMedQA / FiQA / AQUA-RAT / Mol-Instructions (4 domain). +1 domain (e.g., HumanEval-train or MBPP for code) → 5 domain. **Direct comparison to [[sources/feddqc|FedDQC]]**.
- (b) **LESS 270K instruction pool**: FLAN / Dolly / OpenAssistant / CoT split by source-mix. **Direct comparison to [[sources/less|LESS]]** at 7B. Domain split is by source not topic.
- (c) **Self-curated 5-domain**: medical (PubMedQA), legal (Pile-of-Law subset), code (HumanEval-train / MBPP), math (GSM8K), general (Alpaca / Dolly subset). Maximum breadth but no direct prior-work comparison.
- (d) **Super-NaturalInstructions for cross-device**: 1600+ tasks → natural task-level client split for N=100. Cross-silo is separate.

**Decision criterion**: how much weight to give "direct comparison to FedDQC (only FL+LLM precedent)" vs "5-domain breadth" vs "comparison to LESS centralized".

**When**: before Phase 1 starts (Phase 0 is CNN, dataset-agnostic).

**Recommendation**: cross-silo = (a) with code domain added → 5 domain + direct FedDQC comparison; cross-device = (d) Super-NaturalInstructions.

### 3.2 LoRA hyperparameter

**Open**: rank, alpha, target_modules, dropout for all 3 scales.

**Default assumption** ([[flirds-protocol]] §11 uses rank=16 implicitly): rank=16, alpha=32, target=QKV+MLP, dropout=0.1.

**Options & prior-art reference points**:
- LESS uses rank=128, alpha=512, all linear layers — *but* LESS is centralized + much larger compute budget.
- DataInf uses rank=8, alpha=16 — smaller, faster.
- **FedDQC uses rank=16, alpha=32** — direct precedent for FL+LLM+LoRA.

**Decision criterion**: matching FedDQC for direct comparability vs maximizing capacity per LESS.

**When**: before Phase 1 starts.

**Recommendation**: rank=16, alpha=32, target_modules=QKV+MLP (all attention proj + MLP up/down), dropout=0.1. Matches FedDQC.

### 3.3 Training hyperparameter

**Open**: lr per scale, batch size per client, rounds $R$, default $E$ (local epochs), warmup, lr scheduler.

**Default assumption** ([[flirds-protocol]] §11 uses $R{=}50$ as example): not specified.

**Options & references**:
- LESS: lr=2e-5 (LoRA), batch=128, 4 epoch warmup, cosine scheduler.
- FedDQC: lr=2e-5, batch=16 per client, R=300 rounds.
- MATES: lr=2e-4 per pretraining scale.

**Recommendation**:
- lr=2e-5 for Llama-3.2-1B/3B-Instruct; lr=1e-5 for Llama-2-7B (more stable at 7B with bf16).
- batch=16 per client (FedDQC default).
- R=50–100 rounds default (compute-matched across scales); per-experiment can vary.
- Default $E=5$ for non-ablation experiments; ablation sweep $E\in\{1,3,5,10\}$ per [[flirds#Experiment matrix (locked 2026-05-27)]].
- Linear warmup 100 steps, constant lr afterward (FL convergence stability).

**When**: before Phase 1.

### 3.4 Validation set construction

**✅ Resolved (2026-06-03)** — server-side held-out, **integrated 도메인당 200 / 총 1000, uniform stratified**. 관점 = **IRDS-held-out**: 안정적 평균 val-loss 가 (b) oracle/estimator 의 utility 기준. LESS-style few-shot(~50)은 loss 추정이 noisy → Shapley utility 불안정 → 기각. **크기는 1000 고정** — 2¹⁰=1024 coalition-subset enumeration((b) oracle, §9)과 숫자를 분리해 "1024" 혼동을 원천 차단(이제 1024 는 subset 전용).

**Sampling rule** (B2/B3; #16 validation-sensitivity 로 사후 검증): 도메인별 동일 200, 도메인 내 라벨/카테고리 stratified random (fixed seed). **canonical dev split 우선**(데이터셋 제작자 큐레이션이 대표성 보장); split 없는 데이터(Dolly 등)는 train 에서 category-stratified fixed-seed carve + carve 인덱스 기록(train/val 누수 방지). coreset/diversity 최적화는 안 함(over-engineering + 대표성-선별 자체가 또 다른 valuation 이라 순환).

**Domain-attribution (Phase 3, #7/#17)**: headline 1000 과 분리해 per-domain 256 validation 별도.

### 3.5 Cross-device setup detail

**Open**: $N=100$ + $K=10$ participation locked, but participation schedule, late-joiner scenario, Dirichlet $\alpha$ for cross-device specifically.

**Default**: $N=100$, $K=10$ per round (10% sample rate), Dirichlet $\alpha=0.1$ (moderate non-IID), $R=200$ rounds (longer for partial-participation convergence).

**Open detail**:
- Late-joiner extreme regime for stress (§3 item #8) — how many late joiners? Joining at what round?
- ComFedSV's "Everyone Being Heard" — how many extra all-client rounds?

**Recommendation**: late-joiner = 20% of clients join at $r=R/2$ (50/100 round); ComFedSV uses $\lceil N/K \rceil = 10$ all-client rounds appended at start.

**When**: Phase 2 (cross-device only starts then).

### 3.6 BASE_REPO choice (FL framework)

**Open**: which FL framework to fork / base the codebase on. Root `CLAUDE.md` mentions `BASE_REPO` is cloned to `codes/base_repo/`.

**Options**:
| Repo | Pros | Cons |
|---|---|---|
| **OpenFedLLM** (`rui-ye/OpenFedLLM`) | FedDQC uses it; Llama-2-7B + LoRA pre-validated; FL+LLM specifically | Less general; smaller community |
| Flower | Production-grade, large community | LLM support requires custom adapter; overhead for our research scope |
| FedML | Featured FL framework | LLM scale support weak; heavy |
| FedScale | Large-scale FL bench | Cross-device focus; instruction-tuning not native |
| **Self-built minimal** | Full control; no overhead | Engineering time; reinvents simulator |

**Decision criterion**: alignment with FedDQC (direct comparison) vs custom flexibility vs community support.

**When**: before Phase 0 (it shapes the simulator that Phase 0 builds).

**Recommendation**: **OpenFedLLM** — fork to `codes/base_repo/`, add Flirds estimator + dual oracle as new modules, vendor ShapleyFL into baselines. This is also the path of least resistance for FedDQC direct comparison at 7B.

### 3.7 Computing environment

**Open**: tmux naming convention, W&B project setup, GPU scheduling.

**Defaults**:
- B200 × 4 (root `CLAUDE.md` 컴퓨팅 예산).
- W&B project: `flirds-2026` (locked in [[flirds-protocol]] §6).
- tmux session: `flirds-{model}-{exp_type}` suggested (e.g., `flirds-1b-asweep`, `flirds-7b-baseline`).

**Open detail**:
- GPU allocation per run: 1B can fit on 1 GPU with bf16; 3B on 1 GPU; 7B on 2 GPU minimum (LoRA but full validation forward).
- Parallelism: 1B can run 4 experiments in parallel on B200×4; 7B can run 2 in parallel.

**When**: before Phase 0 (just naming convention + W&B init).

### 3.8 Phase-internal task order

**Open**: within each Phase, finer-grained ordering of tasks. (Above § 2 gives Phase-level sequence; per-task day-by-day plan is detailed during implementation.)

**Recommendation**: track in `EXPERIMENT_PLAN.md` inside `codes/base_repo/` once Phase 0 starts; update as implementation reveals dependencies.

### 3.9 Detection baselines code plan

**Open**: source code provenance for FLDetector / FoolsGold / FLTrust / STD-DAGMM.

**Status**: ingested but no code-availability check done yet for these 4. Need a quick search at Phase 2 start.

**Recommendation**: at Phase 2 start, search GitHub for each of the 4. If code exists, vendor + adapt to LoRA gradient. If not, reproduce (each is small enough — these are robustness detectors, not full Shapley estimators).

### 3.10 Pluggable client-corruptor registry (seam 2 — DECISION PENDING, 2026-06-03)

**Open**: implement the corruptor/partitioner registry now (CNN side) vs. while building the LLM data layer. Currently corruption is **inline** in `experiments/phase05_*.py:build()` (`noisy={4,5}` + `if c in noisy:` label-flip) — the hardcoding seam 2 fixes.

**Design (agreed, code-ready)**: `flirds/data/corruptions.py` with a name→callable registry.
- sample-level: `fn(samples, rng, **cfg) -> samples` (label_flip, backdoor; LLM: answer-swap, trigger-token)
- update-level: `fn(delta_w, rng, **cfg) -> delta_w`, hooked in `fl/client.py` (free_rider)
- partition-level: helpers in `fl/partition.py` (maverick = sole-domain holder; duplicate = identical-data client)
- a run-config maps `{client_idx: corruptor_name}` → removes `noisy={4,5}` hardcoding. CNN + LLM `build()` both call it.
- **Backend split**: free_rider / maverick / duplicate are representation-agnostic (implement now); label_flip / backdoor / pgd need a per-backend body (CNN now, LLM when the data layer's instruction-response format is fixed).

**The fork**: (a) implement registry + CNN corruptors + surgical phase05 refactor now (keep `build()` signature, swap internals — phase05 stays green); (b) wire it while building the LLM data layer (folds into the Phase-1 data-format decision). Claude recommended (a) — registry+CNN is an enumerated (non-speculative) need; LLM corruptors fill in later. **Deferred to the Phase 1 session to decide + implement** (per `codes/CLAUDE.md`: surgical, no speculative abstraction).

> **(a) 최소 구현됨 (2026-06-04)**: sample-level `label_shuffle`만 `data/corruptors.py` (`CNN_CORRUPTORS` dict)로 추출 + phase05 dual/flirds_oracle/regime_sweep을 registry 호출로 refactor — **bit-identical** (flirds_oracle 0.7381/0.8810 불변). 의도적으로 최소: `noisy={...}` set 유지(corruptor 1종이라 run-config `{client_idx: corruptor_name}` map은 아직 over-engineering). **남은 풀 registry** — run-config map(noisy hardcoding 제거) + update-level free_rider(`fl/client.py` hook) + partition-level maverick/duplicate + corruptor 함수 시그니처 통일(`fn(samples, rng, **cfg)`) + LLM text corruptor — 는 해당 corruptor를 **실제 쓰는 시점**(Phase 2/3 detection + stage 3 LLM data layer)에 확장.

> **Validation-set (seam 3 / §3.4)**: already config-driven — `flirds_values(logs, model_fn, val_x, val_y, ...)` takes validation as args, no hardcoding. #16 (validation sensitivity) just re-calls with different `val_x/val_y`. No Phase-1 change needed beyond keeping it a parameter.

## 4. Next-session starter prompt

When starting an implementation session (this Claude or another), the trigger should be a single message like:

> "flirds 구현 phase 시작. [[flirds-implementation-plan]] 읽고 phase 0부터 시작하자. 일단 §3.6 BASE_REPO 결정부터 같이 해줘."

The new session should:
1. Read this document (high-level state).
2. Read [[flirds]] (what's locked in design).
3. Read [[flirds-protocol]] (what's locked in implementation rules).
4. Walk through §3 open decisions with Yonghee, one at a time (Yonghee prefers explanation → decision → execution per [[2026-05-27-section-23-lock]]).
5. Update root `CLAUDE.md` pipeline status: `stage: implementation`, `code_dir: codes`, `idea: "client-level FL Shapley via 1st+2nd Taylor of validation loss"`, `current_branch: feature/flirds-phase-0`.
6. Begin Phase 0.

## 5. Pre-implementation checklist (verify before Phase 0 starts)

- [ ] §3.6 BASE_REPO decided
- [ ] §3.7 W&B project `flirds-2026` initialized
- [ ] `codes/base_repo/` cloned and accessible
- [ ] B200 × 4 access verified
- [ ] Phase 0 setup paper PDFs accessible (`raw/papers/flirds/2109.02053v1.pdf` for GTG; `2009.06192v1.pdf` for FedSV; `2109.09046v3.pdf` for ComFedSV; `40034-Article Text-44125-1-2-20260314.pdf` for Ripple)
- [ ] Root `CLAUDE.md` updated to `stage: implementation`

## 6. Pointers — when to consult what

| Question | Where to look |
|---|---|
| "What is Flirds's core formula?" | [[flirds#Core formula]] |
| "Why did we defer noise-vs-OOD-good?" | [[flirds#Resolved questions]] + [[threads/noise-ood-malicious-client-separation]] |
| "How do I report a Spearman ρ correctly?" | [[flirds-protocol#3. Statistical reporting]] |
| "What's the (a) vs (b) oracle distinction?" | [[flirds#Locked design decisions]] (⑤ + N3) + `raw/conversations/flirds/conversation3.md` §2 |
| "Why is SPACE excluded?" | [[flirds#Baseline selection rationale]] |
| "What hyperparameters did LESS use?" | [[sources/less]] |
| "What's the closest centralized analog to Flirds?" | [[sources/less]] + [[flirds#Centralized positioning (added 2026-05-22)]] |
| "What's the proof that Flirds = centralized data-Shapley + drift residual?" | [[flirds#Mathematical narrative (paper-ready)]] + `raw/conversations/flirds/conversation3.md` §4 |
| "Where do I save per-round per-client $\phi_k^{(r)}$?" | [[flirds-protocol#6. Run logging]] |
| "What is Phase 0's pass criterion?" | [[flirds-protocol#10. Phase 0 — code-unavailable baseline reproduction]] |
| "What is the Yonghee preference about pilot data?" | [[2026-05-27-section-23-lock#Yonghee's preferences surfaced (for memory / future sessions)]] |
| "What datasets are LESS/MATES/DsDm trained on?" | [[threads/data-selection-for-llms]] |

## 7. What this document is NOT

- **NOT a code skeleton** — that lives in [[flirds-protocol#11. Implementation skeleton (high-level)]].
- **NOT a design rationale** — that lives in [[flirds]] + raw conversations.
- **NOT a hyperparameter freeze** — §3 lists what's still open with criteria + recommendations.
- **NOT a hard timeline** — phase durations are estimates; actual depends on Phase 0 debugging.

## 8. Document maintenance

Update this document when:
- Any of §3's 9 open decisions gets resolved → move to [[flirds]] decision table, mark resolved here with a short summary + link.
- Phase advances → update §1 status snapshot.
- A new open question arises during implementation → add to §3.
- Phase 0 passes → §5 checklist + start of Phase 1.

When this document goes out of date with [[flirds]] or [[flirds-protocol]], **[[flirds]] and [[flirds-protocol]] win** — they are the lock; this is the operational plan.
