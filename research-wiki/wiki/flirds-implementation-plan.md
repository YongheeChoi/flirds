---
type: project
title: "Flirds — Implementation Plan & Session Handoff"
created: 2026-05-27
updated: 2026-06-04
tags: [flirds, implementation, handoff, phase-0, phase-1, phase-2, phase-3, open-decisions]
---

# Flirds — Implementation Plan & Session Handoff

> **Read this first when starting an implementation session.** This document is self-contained: it tells you (i) what's already decided and where to look for detail, (ii) what's still open and how to decide it, (iii) the phase-by-phase task ordering. The 2026-05-27 design session ended here — the next session (this or another) starts from this document.

## Status snapshot (as of 2026-06-04)

- **Pipeline stage**: `implementation` — **Phase 1 complete + Phase 2 in progress** (2026-06-07): #7 FULL DONE (Flirds selection works) + **Phase 2 tasks 1–4 DONE** (task 1 SV-baselines GTG/FedSV/Ripple; tasks 2–4 Data Banzhaf / ShapleyFL / loss-heuristic / Flirds-1st-only — all reproduce the (b)-oracle ranking, Spearman +1.000; Flirds dominates the frontier 5–15× cheaper). Details in "Next concrete action" below.
- **Theoretical scaffolding**: complete + locked (Section 2 decisions + Section 3 experiment plan + Protocol). See [[flirds]] and [[flirds-protocol]].
- **Wiki**: 42 sources, 11 threads, 27 concepts (after the 2026-06-03 recent-prior-work scan). No paper-blocking gaps.
- **Implementation decisions**: §3.1–3.8 **resolved 2026-06-02** (binding answers = the 06-02 table below; the §3.x bodies are kept as deliberation/rationale). Only §3.9 (detection code provenance) + §3.10 (corruptor registry — **fork (b) chosen**) remain live.
- **Next concrete action (2026-06-09)**: **Phase 2 step-5 BUILD DONE** — matrix orchestrator `experiments/phase2_matrix.py` (NEW) + task-8 scale (`SMOKE_MODEL`/`MODEL_CFG`); all 5 code paths validated on tiny smokes (silo5 4-threat / device100 cheap+anchor / 3B); CNN guard green; see the §3.9 "step 5 BUILD DONE" bullet. **NEXT = real grid execution (cost-tiered stage-gate) AFTER an independent verification session** (Yonghee's request); the poison threat runs at `LR=2e-3` (D2b propagation config). ⟪Prior (2026-06-08):⟫ **Phase 1 DONE + Phase 2 tasks 1–6 DONE + task 7a–7d DONE.** **Tasks 2–4 (06-07, 1B N=5 3-seed, all Spearman vs (b)oracle +1.000)** = Data Banzhaf (self-build = (b)-oracle coalition utils reweighted by uniform 1/2^{n-1}; free-rider φ=0; ~531s) · ShapleyFL surrogate-FSV (uniform submodel + per-round exact Shapley + min-max + EMA; **DMC estimator → cross-device task 7**; ~531s) · loss-heuristic (singleton in-run util U_(b)({k}); ~164s) · Flirds-1st-only (`second_order=False`; **~35s ≈ 15× cheaper**). N=5 near-additive ⇒ all equivalent in ranking → **Flirds dominates frontier** (5–15× cheaper; free-rider φ exactly 0 vs GTG/FedSV renorm). `in_run_shapley` refactor bit-identical; CNN goldens unchanged. Raw: [[raw/conversations/flirds/2026-06-07-phase2-banzhaf-shapleyfl-lossheur-detector-regime]]. **(Historical 06-06:)** #7 FULL ran (both lr: flirds_topk beats random_k on val_loss & ROUGE, **drops the 2 corrupted clients exactly**; lr3e-3 3-seed done, lr1e-3 s2 finishing). SV compare (1B N=5 3-seed): **Flirds reproduces the exact (b)-oracle ranking (Spearman +1.000) at ~5× lower runtime than GTG/FedSV, ~42× than Ripple** (Ripple also weakest on noisy detection, AUROC 0.50±0.20); free-rider φ: Flirds/oracle exactly 0 vs GTG/FedSV within-subset-renorm dilution; ComFedSV deferred (cross-device). Committed d5e06d2 (push pending). **Phase 2 remaining** = task 5 detection (**FLDetector → cross-silo DONE 06-07** — `baselines/fldetector.py` model-free from-logs L-BFGS detector; 1B N=5 3-seed: **cheapest method ~24s** but **weakest detector** noisy AUROC 0.50 / free-rider 0.75 vs valuation 0.75/1.00 = non-IID erosion, clean math client tops every seed → headline at N=10/100; §3.9; STD-DAGMM → cross-device task 7, degenerate at N=5) · **task 6 (a)-retrain LLM — N=5@1B VALIDATED 06-07** (`oracle/exact_sv_llm.py` `llm_subset_utility` = FedAvg-retrain on S → score deployed model; `exact_shapley` reused; `experiments/phase2_llm_a_oracle.py`). **METHOD VALIDATED**: (a)-**val-loss** = (b) in-run = Flirds estimator, **Spearman +1.000** (fp32, both lr1e-3/3e-3, identical ranking; AUROC noisy0.75/FR1.0 all). **Yonghee: (a) must use val-loss (same game) to validate the Shapley computation** — ROUGE is a different (non-differentiable → no estimator) game; the weak earlier agreement was **bf16 precision** (val-loss coalition diffs ~0.005–0.02 < bf16 prec ~0.009 = the (b)-oracle fp32 reason), not signal size. **ROUGE-divergence note**: (a)ROUGE vs (b) = +0.4(1B)/−0.9(3B) — fooled by answer_swap domain-format; val-loss not (supports Flirds' utility). **Cost ladder** (per-|S| linear): (a) N=5 = 47min(1B bf16)/126min(1B fp32)/90min(3B bf16); **N=10 = retrain 64× + eval 32× → 2–5 days single-GPU → DEFERRED to the real experiment** (needs multi-GPU coalition sharding → ~11–22h). **3B (a)-valloss fp32 CONFIRMED 06-08: (a)valloss vs (b)=+0.900 (one clean-client swap=retrain noise), estimator=+1.000, AUROC all noisy0.75/FR1.0 identical, (a)ROUGE=+0.100 — the 1B validation holds at 3B.** raw [[raw/conversations/flirds/2026-06-07-phase2-task6-a-retrain-oracle]]. · **task 7a–7d DONE 2026-06-08** (1B; raw [[raw/conversations/flirds/2026-06-08-phase2-task7-crossdevice-detection-redesign]]): **7a** `fl.partition.client_dirichlet_partition` + `data.llm.build_crossdevice` (per-CLIENT Dir(α) domain-mixture = **Option B**; the existing per-class `dirichlet_partition` is degenerate at 5-domains→100-clients [α=0→5 non-empty]; Option B = fixed size all-N-non-empty, α=0=domain-disjoint, purity == Option A) · **7b** N=100 Flirds verified, NO lib change (`sample_frac=0.1`→K=10; `flirds_values(n_clients=100)` explicit; free-rider φ **exact-0**) · **7c** (b) in-run oracle = **exact per-round decomposition** φ_i=Σ_{r:i∈P_r} 2^{|P_r|}-Shapley = the 2^N oracle (proven Δφ≈3e-16) — **NOT MC** (Yonghee: MC is for the (a)-RETRAIN oracle; in-run is cheap-enough exact): real N=100 α=0.5 **Flirds vs (b) Spearman +1.000**, oracle **771ms/fwd** (fp32-on-B200)→~11h/4-GPU→run at 1–2 α points only · **7d** ComFedSV LLM port (`comfedsv_from_logs(loss_fn,pkeys)`, uniform-subset, partial=True; ==exact uniform-Shapley +1.000; CNN **bit-identical**). **task 7e+ REDESIGNED → detection-baseline SUITE (threat-matched, §3.9 rewritten 06-08)**: data-quality→**FedDQC** / free-rider→**STD-DAGMM**(+FLTrust) / **poisoning→FLDetector·FLTrust** (NEW threat: **Xu2023 trigger + Bagdasaryan scaled** backdoor, DBA excluded); both detectors→both regimes; STD-DAGMM ①per-(client,round) pooling ②feature-hash proj→256 ③random@benign-std+zero. Sequencing: **STD-DAGMM → FLTrust → poisoning-corruptor+FLDetector-repoint → FedDQC → matrix.** · task 8 3B/7B scale-up (7B bf16-train/fp32-eval); task 9 (corruptor ext.) folded into 7e → Phase 3 matrix. See [[raw/conversations/flirds/2026-06-06-sv-baseline-port-and-results]]. ⟪Historical #7-infra detail:⟫ **Phase 1 #7 (first clean 1B run) — INFRA COMPLETE + smoke-verified; scale run pending.** RESEQUENCED 2026-06-04 (#7 before SV-baselines port; de-risk). DONE+verified this session: **② seam-2 corruptor** (§3.10 (b)) · **3-way data split** (`data.llm.build` → `(clients, val_records, test_records)`; **per-domain train=12,000 / val=200 / test=2,000**, mutually disjoint; val native-where-exists (finance `test` / math `validation`) else carve, test always train-carve since native test too small — math 254 / finance 2561; sizes prior-art-grounded, finance 14.5k train is the equalized-train ceiling) · `flirds/eval/{metrics,generate}.py` (ROUGE-L F1 + math(AQUA) exact-match + detection AUROC; left-pad greedy gen + per-domain score) · `flirds/run_logger.py` (config.yaml + git SHA/dirty + env hash + φ parquet + metrics json, §6/D2) · **orchestrator `experiments/phase1_clean_run.py`** (per-seed: Flirds φ + (b) oracle at N=5 → ① noisy/free-rider AUROC → ② selection-convergence (arms full / random-K / Flirds-top-K, per-round val-loss curve read post-hoc off logs) → ③ final per-domain task-acc via generation → run-dir; FULL/MINI/SMOKE configs). **SMOKE green** (est≈oracle 1.6e-7; AUROC/arms/gen/log all work). Downstream metric = FedHDS-style ROUGE-L + math EM (utility=val-loss와 분리). **Remaining = launch the FULL scale run** (`CLEAN_RUN_MODE=full`, ~5–7h, dominated by test-2k generation × 3 arms × 3 seeds; config = `phase1_clean_run.py` top `FULL` dict) — a **MINI de-risk run** (~30min: train 500 / R 10 / test 200 / 1 seed; does the noisy/selection signal emerge?) runs first. Then ③ = SV baselines port (GTG/FedSV/ComFedSV/Ripple → LLM). (Phase 0/0.5, estimator/oracle, LLM backend + FL loop all done.)
- **Seams status (2026-06-03 → 06-04)**: seam 1 (per-layer φ, observation-only) **DONE** (`per_layer=False`; INVARIANT holds, verified on CNN + LLM); seam 3 (validation config-driven) **DONE**; seam 2 (corruptor registry) = the remaining data-layer work, **fork (b)** (build it with the LLM data layer). Experiment plan extended with 7 validation gaps (#12–18, all Phase 2/3) — [[flirds#Added 2026-06-03 — prior-art validation gaps (Phase 2/3; none touch the Phase 1 core)|flirds §Added 2026-06-03]].
- **Phase 1 progress (2026-06-03 session 2)**: estimator/oracle made **backend-agnostic** (`loss_fn(params,buffers)`+pkeys injection, `backends/cnn.py`) + **partial-participation-correct** (per-round FedAvg weight) + **per-layer φ logging** (seam 1, observation-only); all CNN gates bit-identical. OpenFedLLM cloned to gitignored `external/` + scouted (its fedavg weight == our per-round weight). Raw: [[raw/conversations/flirds/2026-06-03-phase1-backend-abstraction]].
- **Phase 1 progress (2026-06-04 session)**: LLM stage 2 **backend + FL loop done** — `backends/llm.py`(make_llm_loss, LoRA-only via functional_call) + `fl/server.py` `_fedavg_core` 추출(CNN wrapper **bit-identical 회귀**) + `fl/llm_server.py`(run_llm_fedavg_logs, **TRL SFTTrainer 1.x + forced SGD + completion-only**). **LLM-FL 스모크 green**(Llama-3.2-1B real 궤적, est≈oracle 1.70e-6). **3 LLM musts**: eager-attn / named-key state / embedding-hook clear (functorch+HF; see [[flirds-protocol]] §13). validation(§3.4) 확정(도메인당 200/총 1000; **1000 vs 2¹⁰=1024 subset 분리**; cross-silo trainset 크기는 도메인별 **동일 통제**, aggregate weight는 size-prop 유지, FiQA 부족 시 code-domain 대체). Raw: [[raw/conversations/flirds/2026-06-04-phase1-llm-stage2]].

## ✅ Decisions resolved & corrections (2026-06-02 implementation kickoff)

2026-06-02 세션에서 §3의 9개 open decision 전부 + 추가 결정을 확정하고 Phase 0 구현에 착수했다. 요약:

**확정 결정**

| 항목 | 결정 |
|---|---|
| BASE_REPO (§3.6) | LLM = **OpenFedLLM** fork(`codes/base_repo/`, LLM phase); CNN = **자체 경량 시뮬레이터**. 얇은 공통 FL core 위 backend 분리, estimator/oracle 은 backend-agnostic |
| 코드/빌드 (D1) | baseline 코드 있으면 fork, 없으면 self-build |
| 로깅 (D2) | **W&B 미사용** → 로컬 run-dir(config YAML + env hash + git SHA + per-round per-client φ parquet) |
| cross-silo 데이터 (§3.1) | **free-form 5-domain (2026-06-04 swap, supersedes D3)**: medical **`medalpaca/medical_meadow_medical_flashcards`** / legal **`ibunescu/qa_legal_dataset_train`** / finance **FiQA** / math **AQUA-RAT(rationale)** / general **Dolly**. All free-form instruction→response (format uniformity → fair shared-val-loss Shapley; [[threads/dataset-format-uniformity]]). FedDQC overlap = FiQA+AQUA (2/5). LESS protocol for validation/selection. |
| cross-device 데이터 | **Fed-WildChat**(FedLLM-Bench, 자연 N=100) + **FedHDS**(NI task별 + Dolly Dirichlet) 둘 다 |
| FedDQC 비교 | matched-arm 미실시; **IRA 점수법만 baseline 으로 이식** |
| LoRA (§3.2, D4) | r=16/α=32 시작 + **rank sweep {16,32,64,128}**(α=2r) — attribution fidelity vs task perf ablation |
| 학습 hp (§3.3, D5) | lr 2e-5, batch 16, cosine — 시작값, sweep |
| validation (§3.4, D6) | server-side held-out, **도메인당 200 / 총 1000, uniform stratified**; IRDS-held-out 평균 loss 관점(few-shot 아님). 크기 1000 = 2¹⁰ coalition-subset 과 분리(혼동 차단). val source per domain: **med/legal/general = carve from train** (no dev split), finance=`test`, math=`validation`. + **per-domain macro-average normalization** option (1/D weight vs token-prop) → **ablation ON/OFF** ([[threads/dataset-format-uniformity]]) |
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
| [[flirds]] | All locked design decisions (original 2026-05-05 + N1–N4 + Q1–Q3 + ④⑤ + model choice + baseline reduction); resolved questions; Experiment plan §3 (**18 items** — 11 original + #12–18 added 2026-06-03); Experiment matrix (1B/3B/7B); baseline-selection rationale + code-availability table; differentiators table (incl. FedIF/FedTSV); centralized positioning |
| [[flirds-protocol]] | bf16 train / fp32 eval rule; seeds ≥ 3; scipy tied-rank; MC variance reporting; 95% bootstrap CI band; (a)/(b) oracle code-path separation; sanity gates ($E{=}1$, $N{=}2$); run logging (local run-dir, no W&B); Phase 0 reproduction; **§13 LLM backend requirements** (eager-attn / named-key / hook-clear); implementation skeleton; pre-publication checklist |
| `raw/conversations/flirds/conversation1.md → conversation4.md` | Original design conversations with another LLM (IRDS → FL adaptation, math derivation, locked design choices). **Primary record for math + design rationale.** |
| `raw/conversations/flirds/2026-05-19-section23-walkthrough.md` | Raw turn-by-turn transcript of the 2026-05-19 → 22 Section 2/3 walkthrough (restored from JSONL after session interruption). |
| `raw/conversations/flirds/2026-05-27-section-23-lock.md` | Distilled record of the same arc + 2026-05-27 lock + Yonghee preferences surfaced. |

**Do not re-litigate locked decisions** unless implementation reveals new information. If something seems unclear, search [[flirds]] first, then the raw conversation files.

## 2. Phase structure (do these in order)

Four phases. **Phase 0 is a hard gate**: no LLM-phase code is written until Phase 0 reproductions pass.

### Phase 0 — CNN baseline reproduction (gate) — ✓ DONE 2026-06-02

**Goal**: reproduce headline metrics of the four FL-Shapley baselines in their *original* (CNN + MNIST/CIFAR-10) setups. **All four self-built** (reference-guided, D1): GTG/ComFedSV have public code but in non-forkable forms (cyyever framework / Huawei notebook); FedSV/Ripple have none. **Engineering by-product**: the FL simulator + FedAvg loop + utility evaluator + Ripple sample-level→client aggregator that the LLM phase reuses.

**Targets** (4 baselines, all self-built; setup details in respective wiki pages):
1. [[sources/gtg-shapley|GTG-Shapley]] (ACM TIST'22): CNN + MNIST/CIFAR-10, N=10. Headline: Spearman ρ vs uniform MC; runtime advantage. ~1 day.
2. [[sources/principled-federated-data-valuation|FedSV (Wang 2020)]]: CNN + MNIST/CIFAR-10 (IID + non-IID), N=10. Headline: noisy-label / backdoor detection rate. ~1 day.
3. [[sources/comfedsv|ComFedSV]] (ICDE'22): Synthetic / MNIST / FMNIST / CIFAR-10, N=100 (10 noisy). Headline: Spearman vs ground-truth, Jaccard on noisy detection. ~2–3 days.
4. [[sources/ripple-shapley|Ripple Shapley]] (AAAI'26): CNN + MNIST/CIFAR-10, N=10. **No ground-truth-SV metric** (task-driven robustness only) → checked via noisy-AUROC + runtime. Speedup is **62× vs AFedSV+ / 49× vs FedSV, not vs GTG**. ~1–2 days.

**Total cost**: 5–7 days on B200 × 1.

**Pass criterion**: each baseline's headline metric within ±5% of the value reported in the original paper. Otherwise: (i) debug implementation, (ii) author contact, (iii) document the gap with a caveat in the paper's reproduction section.

**Pass gate**: Phase 0 passing produces the *foundation infrastructure* for Phase 1. Do NOT skip Phase 0.

### Phase 1 — Flirds core + dual oracle + vanilla FedAvg at 1B

> **STATUS 2026-06-04 — stages 1+2+3 DONE + #7 infra DONE; only the FULL scale run is unlaunched.** Estimator/oracle backend-agnostic + partial-participation + per-layer (DONE); LLM backend + FL loop (`backends/llm.py`, `fl/llm_server.py`) DONE; **task 3 data layer DONE** (`data/llm.py` 5-domain free-form loader + val micro-batching + per-domain norm); **② seam-2 corruptor DONE** (`data/corruptors.py` answer_swap + free_rider); **#7 first-clean-run infra DONE** (eval/{metrics,generate}.py, run_logger.py, orchestrator `experiments/phase1_clean_run.py`; SMOKE green). The task list below (tasks 1–7) is the *original* Phase-1 plan — all satisfied except the scale run. **Remaining = launch the FULL `phase1_clean_run.py` (MINI de-risk first), then ③ SV-baselines port (Phase 2).** See "Next concrete action" for the full #7 detail.

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

> **⚠ Phase 1 seams — STATUS (updated 2026-06-04).** The 7 experiments in [[flirds#Added 2026-06-03 — prior-art validation gaps (Phase 2/3; none touch the Phase 1 core)|flirds §Added 2026-06-03]] are all Phase 2/3 (data/eval-layer); two depended on Phase-1 *core* seams:
> 1. **φ logging granularity** (per-round × per-client × per-layer) — **DONE** (`per_layer=False` returns components; Σ == φ_k bit-identical, verified CNN + LLM).
> 2. **Pluggable client-corruptor / partitioner** — **REMAINING**, builds with the LLM data layer (task 3), **fork (b)** per §3.10. Until then noisy/free-rider stay inline in experiment scripts.
>
> The INVARIANT below is a standing rule (binding regardless of status):
>
> **⚠⚠ INVARIANT for seam 1 (Yonghee, 2026-06-03) — per-layer φ is OBSERVATION-ONLY; it must NOT break the IRDS proof.** In `core/flirds_estimator.py`, $\phi_k$ is *already* a sum over layers (`sum((g[n]*dw[k][n]).sum() for n in pkeys)`). Per-layer logging = keep those summands instead of collapsing early; the returned $\phi_k$ stays `sum(components)`, bit-identical. The default/spine estimator **never reweights** per-layer (no $\mathrm{diag}(c_\ell)$, no layer selection) — that would break the granularity-invariance lemma + Proposition 1. Per-layer components are a **read-only diagnostic array** consumed *only* by ② characterization (#7) and the qualitative case study (#17). **The one intentional exception** is the locked **Q2 layer-wise *variant* (#6)** — a separate, clearly-labeled ablation arm that deliberately reweights to *show* the lemma break; it is never the headline method. Implementation: add per-layer logging as a backward-compatible option (e.g. `per_layer=False` returning the extra array) so the Phase 0.5 callers are untouched.

### Phase 2 — Full baseline set + (a) retrain SV expansion + 3B/7B scale up

**Goal**: all 10 baselines + **2 detection methods** (FLDetector + STD-DAGMM, per 06-02) working at 1B/3B/7B; (a) retrain SV expanded to N=10 at 1B + N=5 at 3B; cross-device MC setup operational.

**Tasks**:
1. Port Phase 0 reproduced baselines (GTG / FedSV / ComFedSV / Ripple) from CNN setup to LLM+LoRA setup. ComFedSV cross-device only.
2. **DONE (2026-06-07).** Data Banzhaf in FL — **reference-guided self-build, not pyDVL/OpenDataVal**: a semivalue = the (b) in-run oracle's exact 2^N coalition utilities reweighted by the uniform Banzhaf kernel 1/2^{n-1} (`baselines/banzhaf.py`; `in_run_sv._coalition_utilities` factored out, `in_run_shapley` bit-identical). Verified 1B N=5 3-seed: **Spearman vs (b)oracle +1.000, free-rider φ exactly 0, ~531s** (~5× Flirds). MSR estimator → only needed cross-device (task 7).
3. **DONE (2026-06-07).** ShapleyFL surrogate-FSV — reference-guided self-build (`baselines/shapleyfl.py`): per-round uniform-submodel utility + per-round exact Shapley + min-max norm (Def 4.2) + EMA (Def 4.3), **from-logs valuation**. The paper's DMC difference-estimator is its *large-N estimator* → **deferred to cross-device (task 7)** (at N=5 exact is the faithful value). Verified 1B N=5 3-seed: Spearman +1.000, ~531s. (Exact + *our* utility would be degenerate-equal to the (b) oracle by Shapley linearity; the uniform-util + min-max + EMA make it distinct.)
4. **DONE (2026-06-07).** loss-heuristic (floor = per-client singleton in-run utility U_(b)({k}); free-rider φ=0, ~164s) + Flirds-1st-only (`second_order=False`; **~35s ≈ 15× cheaper than the coalition sweeps**). Both Spearman +1.000 at N=5. Wired into `phase1_baseline_compare.py` (now 9-method).
5. Detection baselines for the noisy / free-rider AUROC table. **06-02 (D8) fixed the pair: [[sources/fldetector|FLDetector]] (noisy/poisoning) + [[sources/free-riders-fl-std-dagmm|STD-DAGMM]] (free-rider).** **Regime split LOCKED 2026-06-07 (§3.9): FLDetector → cross-silo (N=5/10, DONE 06-07 — cheapest ~24s but weakest at N=5, non-IID erosion); STD-DAGMM → cross-device (with task 7)** — STD-DAGMM trains a DAGMM on update vectors → degenerate at N=5.
6. Expand (a) retrain SV: N=10 at 1B (~3.5 days B200×4), N=5 at 3B (~45 min). 7B (a) skipped per Section 3 lock. **N=5@1B DONE+VALIDATED 2026-06-07** (`oracle/exact_sv_llm.py`): (a)-val-loss = (b) = estimator **Spearman +1.000** (fp32, both lr); ROUGE is a *different* game (secondary; corruption-fooled). Measured cost ladder (a) N=5 = 47min(1B bf16)/126min(1B fp32)/90min(3B bf16); **N=10 = retrain 64× + eval 32× → 2–5 days single-GPU → deferred to the real experiment (multi-GPU coalition sharding needed)**. The "(a)=bf16 (deployment)" lock holds for the ROUGE figure, but the val-loss VALIDATION needs **fp32** (coalition diffs < bf16 precision, the (b)-oracle reason). **3B (a)-valloss fp32 CONFIRMED 2026-06-08** (N=5, lr3e-3, 9483 s ≈ 2.6 h; retrain 82 % / eval 18 %): (a)-valloss vs (b) = **+0.900** (one clean-client swap fina/math/gene = retrain noise), estimator(Flirds) = **+1.000**, AUROC (a)valloss / (b) / estimator **all noisy 0.75 / FR 1.0 identical**, (a)ROUGE = +0.100 (fooled). So 3B reconfirms the 1B validation (val-loss is the right game; Flirds tracks (b) even more faithfully than the (a)-retrain oracle, whose +0.900 is its own retrain variance). N=10 retrain extrapolation = 84.3 h lower-bound → multi-GPU coalition sharding in the real experiment.
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
9. **(2026-06-05) Experiment instrumentation & reporting tooling (W&B replacement, [[flirds-protocol]] §15):**
   - **Timing & GPU-hour accounting** (§15.1) — per-phase wall-clock + GPU-hours + peak GPU memory → `timing.json` per run. The substrate for the paper's efficiency claim (Flirds 1 HVP/round vs oracle = a **measured** wall-clock ratio logged side-by-side, not only FLOP-argued).
   - **Live process logging** (§15.2) — timestamped `run.log` + per-round ETA + traceback → `error.log` + `status` in `meta.json` (crash localization). Formalize the orchestrator's existing `print(flush=True)` lines into a tiny logging helper, not a framework.
   - **Cross-run aggregation + visualization** (§15.3) — `experiments/aggregate_runs.py`: scan `runs/`, assemble one tidy table (row per config cell), emit summary tables (mean±std + 95% CI) + matplotlib figures (convergence / AUROC / α-E bands / est-vs-oracle / timing bars). Replaces the W&B dashboard; read-only over run-dirs.
   > **Instrument early.** The timing + structured-logging hooks (§15.1–15.2) are cheap additions to `run_logger.py` / the orchestrator — wire them in **before the imminent #7 FULL scale run** so a multi-hour run isn't wasted untimed/unlogged (per-run recording §6 is already done; this adds timing + live-log + crash-capture on top). The cross-run aggregation/viz tool (§15.3) is the Phase-3-era deliverable, built once enough run-dirs exist. Both surgical (no speculative framework — `codes/CLAUDE.md`).

**Output**: all numbers for the paper, with reproducibility metadata. Hand off to `/auto-review-loop` / `/paper-writing`.

## 3. Implementation decisions (§3.1–3.8 RESOLVED 2026-06-02; §3.9 + §3.10 live)

> **Read this banner first.** §3.1–3.8 were **resolved on 2026-06-02** — the **binding answers are in the "Decisions resolved (2026-06-02)" table above**, not in the §3.x bodies. The bodies below are kept as the *original deliberation/rationale* (options + criteria); where a body's "Recommendation" differs from the 06-02 table, **the table wins** (e.g., §3.4 validation = 200/domain × 5 = 1000 total, *not* the body's older "1024 integrated + 256/domain"; §3.1 adds the 06-04 rule: cross-silo trainset size **equalized per domain**, aggregate weight stays size-proportional, FiQA-too-small → swap code-domain). The only **live** decisions are **§3.9** (detection code provenance) and **§3.10** (corruptor registry — fork (b) chosen, implement with the data layer).

Each body: **what's open** → **current default / assumption** → **options** → **decision criterion** → **when**. *(Historical for 3.1–3.8.)*

### 3.1 Dataset choice (cross-silo + cross-device)

> **RESOLVED — free-form 5-domain (2026-06-04, supersedes the D3 PubMedQA/CaseHOLD picks).** Durable rationale + parked candidates + license notes: [[threads/dataset-format-uniformity]]. Cross-silo domains are unified to **free-form instruction→response** (heterogeneous task formats make a *shared* val-loss Shapley unfair — a 1-tok classification target vs multi-tok generation aren't loss-comparable). Adopted set:
> | domain | dataset | train | val |
> |---|---|---|---|
> | medical | `medalpaca/medical_meadow_medical_flashcards` (cc) | 34k | **carve** (replaces PubMedQA) |
> | legal | `ibunescu/qa_legal_dataset_train` | 97k | **carve** (replaces CaseHOLD) |
> | finance | `LLukas22/fiqa` | 14.5k | `test` split |
> | math | `deepmind/aqua_rat` (rationale CoT) | 97k | `validation` split |
> | general | `databricks/databricks-dolly-15k` | 15k | **carve** |
>
> So **3/5 carve val from train** (med/legal/general — no dev split), 2/5 use an existing split. B1 trainset size **equalized per domain** (aggregate weight stays size-prop). Plus a **per-domain macro-average normalization** option (each domain weighted 1/D vs token-proportional) — **ablation ON vs OFF** (downstream accuracy). FedDQC overlap shrinks to **finance (FiQA) + math (AQUA-RAT)** = 2/5 (medical/legal swapped off FedDQC's classification picks). Cross-device unchanged (Fed-WildChat + FedHDS). The §3.1 body below is **historical deliberation** (its PubMedQA/code options are superseded).

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
- B200 × 4 (root `CLAUDE.md` 컴퓨팅 예산); only physical GPUs 0–3 usable.
- **Logging = local run-dir, NO W&B** (D2; [[flirds-protocol]] §6). ~~W&B project flirds-2026~~ (superseded).
- tmux session: `flirds-{model}-{exp_type}` suggested (e.g., `flirds-1b-asweep`, `flirds-7b-baseline`).

**Open detail**:
- GPU allocation per run: 1B can fit on 1 GPU with bf16; 3B on 1 GPU; 7B on 2 GPU minimum (LoRA but full validation forward).
- Parallelism: 1B can run 4 experiments in parallel on B200×4; 7B can run 2 in parallel.

**When**: resolved 2026-06-02 (local logging; conda env `flirds`).

### 3.8 Phase-internal task order

**Open**: within each Phase, finer-grained ordering of tasks. (Above § 2 gives Phase-level sequence; per-task day-by-day plan is detailed during implementation.)

**Recommendation**: track in `EXPERIMENT_PLAN.md` inside `codes/base_repo/` once Phase 0 starts; update as implementation reveals dependencies.

### 3.9 Detection baselines code plan

**Open**: source code provenance for the **2** detection baselines (narrowed 06-02): **FLDetector** (noisy/poisoning) + **STD-DAGMM** (free-rider). (FoolsGold / FLTrust dropped.)

**Status**: FLDetector IMPLEMENTED + verified 2026-06-07. **REDESIGNED 2026-06-08 → threat-matched detection suite** (supersedes the regime-split block below).

**Detection-baseline REDESIGN — LOCKED 2026-06-08** (raw [[raw/conversations/flirds/2026-06-08-phase2-task7-crossdevice-detection-redesign]]). The old FLDetector↔noisy pairing was a **threat mismatch**: FLDetector detects *crafted-update attackers* (Fang/Scaling/DBA/ALIE), **not** the noisy-but-honest `answer_swap` client ([[sources/fldetector]] line 48) — its 0.50 noisy AUROC is **off-threat**, not merely non-IID erosion. `answer_swap`'s update is an honest, temporally-CONSISTENT gradient → FLDetector's deviation signal never fires; STD-DAGMM also misses it (normal std). Resolved to **threat-matched detectors, each in BOTH regimes** (detection is needed in both; the valuation methods already produce AUROC in both):

| threat | corruptor | matched detector(s) |
|---|---|---|
| data-quality (honest, bad data) | `answer_swap` | **FedDQC** (LLM-FL IRA per-sample quality → client-level) — NEW |
| free-rider (fabricated update) | `free_rider` zero/random | **STD-DAGMM** (independent AE+std) **+ FLTrust** (any-N cosine-to-root) |
| poisoning/backdoor (crafted) | **NEW: Xu2023 trigger + Bagdasaryan scaled** | **FLDetector + FLTrust** |

- **Why detection at all, given we inject the labels**: the method is BLIND to labels; the known labels are the *evaluation key* (AUROC) — standard supervised eval of an unsupervised separator, not circular. Purposes: (1) **semantic validation** of the value (corrupt→low confirms value MEANS quality; the oracle only proves "Flirds matches the Shapley *computation*"); (2) **competitive bar** (Flirds matches/beats dedicated detectors with no detection machinery).
- **Why each detector**: FedDQC = the data-quality match (only LLM-FL integrity prior art; per-sample IRA → adapt to client-level). STD-DAGMM = dedicated free-rider, mechanistically independent of Flirds. FLTrust = any-N free-rider + poisoning (per-client ReLU-cosine-to-root + magnitude-norm, **val = root**) BUT cosine ≈ Flirds 1st-order → redundant free-rider signal, so STD-DAGMM stays the independent one. FLDetector + FLTrust both validated on the **Scaling backdoor** = our Bagdasaryan update.
- **Regime**: FLDetector → both (model-free; cross-device needs partial-participation per-client-history adaptation). STD-DAGMM → both via **per-(client,round) pooling** (5·R vectors rescue N=5; at small N the AE is weak → the std-augmentation carries — honest limitation).
- **No LLM-FL-validated client-level noisy/free-rider detector exists** (STD-DAGMM = "first PEFT-scale test"; FedDQC = per-sample not client-anomaly; all CV/small-net) → baselines are necessarily CV ports → Flirds' LLM-scale separation is novel ([[sources/do-influence-functions-work-on-llms]]: IF brittle on LLMs; Flirds' in-run framing sidesteps it).
- **STD-DAGMM ①②③**: ① per-(client,round) pooled samples (FORCED by both-regime; per-client-mean → N samples → degenerate at N=5); ② feature-hashing random projection → ~256 (5.6M-dim update too big for an AE; std computed on the FULL vector, reduction-independent); ③ free-rider mode = random @ **benign-std-tuned** + zero floor (delta/advanced-delta → task 9 = richest comparison + where Flirds is at risk: aligned recycled-aggregate → φ≠0).
- **Poisoning corruptor (NEW)** = [[sources/instructions-as-backdoors-xu]] (Xu 2023, 2305.14710) instruction-trigger→target **+** [[sources/how-to-backdoor-fl-bagdasaryan]] (Bagdasaryan 2020, 1807.00459) plain-**scaled** model-replacement update; **DBA (Xie 2020) excluded** (multi-attacker collusion ≠ single-corrupt-client). plain-scaled (not constrain-and-scale stealthy → evades all detectors incl. Flirds; separate hard regime). Measure ASR + detection AUROC. **Flirds-vs-backdoor framing OPEN — verify experimentally, do NOT pre-position** (hypothesis: clean-performance-preserving backdoor preserves clean val-loss → evades Flirds = its boundary → complementary detector; TEST first). Ingest the two source pages when implementing.
- **Sequencing**: STD-DAGMM → FLTrust → poisoning-corruptor + FLDetector-repoint → FedDQC → expanded matrix (3 threats × 2 regimes × {matched detector + Flirds + valuation} × α-sweep). Expands old task 7e (STD-DAGMM only) into a detection SUITE + folds in task 9.

**IMPLEMENTED 2026-06-08 — task 7e steps 1–3 (code + validation; committed, push by Yonghee).** Raw: [[raw/conversations/flirds/2026-06-08-phase2-task7e-detector-suite-steps1-3]].
- **Step 1 STD-DAGMM** (`baselines/std_dagmm.py`): model-free DAGMM + std augmentation, per-(client,round) pooling, signed feature-hashing 5.6M→256 (std on the FULL vector). Synthetic **AUROC=1.0** (zero caught by std, std-matched random by recon/cosine = the Lin headline). Real 1B N=100 free-rider **AUROC=0.628** — honest: the model-free detector is WEAK on the pure-evasion case at LoRA-FL scale (a std-matched random direction is not separable from real benign LoRA updates without the gradient). Defaults flagged for review (Yonghee did not veto): per-dim standardize, unseen→min score, n_gmm=2, global-RNG snapshot/restore. `fl/llm_server.py` `free_rider_scale` threaded (benign-std tuning; `uniform(-s,s)` std = s/√3).
- **Step 2 FLTrust** (`baselines/fltrust.py`): with **g0 = −∇_val(w_r)** the trust signal is the **NORMALIZED Flirds first-order term** → "cosine ≈ Flirds-1st" is *exact* (the auxiliary baseline, per the redesign). NOT model-free (val gradient via loss_fn/pkeys; the estimator's `_chunked` reused). **Detector uses the SIGNED cosine, NOT FLTrust's ReLU** — ReLU clips benign(<0) AND free-rider(~0) both to 0, erasing the free-rider sign; ReLU + magnitude-norm are FLTrust's robust-aggregation gates (scale-free → no ranking change). Unit AUROC free-rider+poison=1.0; real 1B N=100 free-rider **AUROC=1.0** (free-riders = top-5), Spearman vs Flirds-1st=+0.60. **Insight: gradient-using detectors (FLTrust≈Flirds-1st) ace free-rider (1.0) where model-free STD-DAGMM struggles (0.63) → the val gradient is the decisive signal; Flirds subsumes FLTrust, STD-DAGMM is the independent (weak) free-rider baseline.**
- **Step 3 poisoning + FLDetector (code DONE; backdoor-install RESOLVED 2026-06-09)**: `data/corruptors.py backdoor()` (Xu trigger→target, `poison_frac` = the clean-preservation knob) + `backdoor=`/`backdoor_kwargs=` injection; `fl/llm_server.py scaled_attackers`/`attack_scale` (Bagdasaryan ×γ model-replacement); `eval/generate.py backdoor_asr`. **FLDetector poisoning-repoint + cross-device adaptation = per-client GAP-integrated HVP** (Yonghee's choice): predict from the client's last participation t' over the gap w^r−w^{t'} (one cached HVP per gap) — **bit-identical under full participation** (CNN guard green), cross-device synthetic AUROC=1.0. Detection (real 1B N=5, scaled attacker): FLDetector & FLTrust **AUROC=1.0**; **Flirds-1st verdict FLIPS with poison_frac** (0.5 clean-helpful → attacker looks best → evades AUROC 0; 1.0 pure-backdoor → flagged AUROC 1; γ scaling amplifies whichever) = the clean-preservation boundary — **reported, NOT pre-positioned** (the matrix confirms). **RESOLVED 2026-06-09 (D1/D2/D2b; raw [[raw/conversations/flirds/2026-06-09-phase2-task7e-backdoor-install-feddqc]]): the ASR=0 was UNDER-TRAINING, not a narrow window.** **D1** (`phase2_backdoor_install_smoke.py`, install isolation, no FL): attacker local-trains to convergence (SGD mom=0, 3 epochs, single-token target) → poison_frac sweep: **frac 0.5–0.8 = clean-preserving backdoor** (triggered-ASR ~1.0, clean-ASR 0), frac 1.0 = clean destroyed (unconditional), frac ≤0.2 = below the install threshold; poison_frac is the clean-preservation knob. **D2** (`phase2_backdoor_d2_smoke.py`, FL model-replacement, single-shot G+(1/N)γ(X−G)): **full-repl γ=n/η=5 propagates (triggered-ASR 0.97, clean-val-loss +0.027); partial/norm-bound = 0 (all-or-nothing)**. attacker raw ‖Δ‖=40× benign → the stealthy/norm-bound arm is impossible here (norm-bounding kills the backdoor) = install↔low-norm trade-off extreme → the scaled attacker is an obvious magnitude outlier. **D2b** (`phase2_backdoor_d2b_smoke.py`): on the working backdoor, FLDetector/FLTrust/Flirds-1st/Flirds-2nd **all AUROC=1.0** — Flirds separates the attacker even at +0.027 clean-val (γ amplifies the 1st-order term) → **"clean-preserving backdoor evades Flirds" is REFUTED in this config** (boundary = matrix, not pre-positioned). New code: `corruptors.backdoor` single-token target, `generate.backdoor_soft_asr`.
- **Step 4 FedDQC DONE 2026-06-09** (`baselines/feddqc.py` + `phase2_feddqc_smoke.py`): IRA(q,a)=L(a)−L(a|q) per sample → client mean → suspicion = −IRA (corrupt HIGH). Real 1B N=5 noisy=medical(answer_swap) **noisy AUROC=1.0** (noisy IRA 0.067 vs clean 0.17–1.26); caveat = large per-domain IRA variance (finance 0.17 ≈ noisy 0.067) → vary the noisy domain/seed in the matrix. **Source pages INGESTED 2026-06-09** (web-extract; [[sources/instructions-as-backdoors-xu]] + [[sources/how-to-backdoor-fl-bagdasaryan]]; PDFs still not on disk → replace when dropped).
- **Detector suite COMPLETE**: data-quality→FedDQC (1.0) / free-rider→STD-DAGMM (0.63) + FLTrust (1.0) / poisoning→FLDetector + FLTrust (1.0) + Flirds (1.0, D2b). **NEXT = step 5 matrix** (3 threats × 2 regimes × {matched detector + Flirds + valuation} × α-sweep; poisoning arm = scaled full-repl γ=n/η, frac=0.5 — the unscaled/stealthy arm is impossible without Bagdasaryan constrain-and-scale = separate study).
- **step 5 grid LOCKED 2026-06-09**: detection = cross-silo **N=5** (all methods + (b) Spearman) + cross-device **N=100** α-sweep {0,0.01,0.1,0.5,5.0} ((b) at α=0.5 only) + **N=10 deferred** (detection-only if needed); **Ripple excluded** (dominated + flaky); **seed 3**. est-vs-oracle = LOCKED 06-04 (1B N5 ✓ / 3B N5 ✓ / 7B (b) N5 / 1B N10 (a) deferred). scale (task8) = 3B detection+(b) N5, 7B (b) N5.
- **EXECUTION STRATEGY LOCKED 2026-06-09 (Yonghee) = cost-tiered stage-gate** (applies when running the real experiments after the orchestrator + task8 are built): **(1) cheap-first by cost tier** (regime/scale: cross-silo N5 → cross-device N100 → 3B → 7B); **(2) category-together** = within one (regime, threat, seed) trajectory run ALL comparison methods together regardless of per-method cost (35s Flirds … 530s valuation) — never run a cheap subset first (fair comparison on the same logs); **(3) (b)oracle = anchor, not comparand** — included in the N5 category (530s) but only 1 α point (α=0.5) at N100 (11h/4-GPU), the rest of the α-sweep compares methods with Flirds as the validated proxy-truth; **(4) adaptive** — an unexpected result at a tier revises the larger-tier plan before running it. **Remaining build before execution: matrix orchestrator** (extend `phase1_baseline_compare.py` 10-method = per-threat trajectory loop {noisy/free-rider/poisoning} + STD-DAGMM/FLTrust/FedDQC + the D2b poisoning synthesis) **+ task8 3B/7B scale**. → **BUILD DONE 2026-06-09, next bullet.**
- **step 5 BUILD DONE 2026-06-09 (matrix orchestrator + task 8; raw [[raw/conversations/flirds/2026-06-09-phase2-step5-matrix-orchestrator-task8]])**: `experiments/phase2_matrix.py` — **NEW file** (Yonghee's fork: chosen over extending `phase1_baseline_compare.py` to preserve the validated N=5 +1.000 comparator; reuses only the bit-identical per-method call pattern). The **only** code change → baselines untouched → **CNN bit-identical guard GREEN**. Env-parameterized (REGIME / ALPHA / THREAT / SEED / SMOKE_MODEL / ORACLE_B / COALITION / LR / POISON_TRAIN; the `RUN_SEED` cell-sharding idiom): per-threat trajectory loop {noisy / freerider_random / freerider_zero / poison} × regime {silo5, device100} × α × seed; the 3 detectors (STD-DAGMM / FLTrust / FedDQC) on **every** threat (the on-/off-threat matrix); **regime-gated method set** ((b) = exact 2^N at silo5 / `in_run_shapley_perround` at device100; Banzhaf + exact-(b) drop out at N=100; GTG/FedSV/ShapleyFL/(b)-perround gate behind COALITION/ORACLE_B = the α=0.5 **anchor**; cheap methods Flirds/Flirds1st/loss-heur/ComFedSV + detectors run every α with **Flirds as proxy-truth** off-anchor); poison = the **D2b synthesis** (benign FL → attacker backdoor X from G → single-shot model-replacement attack round γ(X−G), γ=cohort); **task-8 scale** = `SMOKE_MODEL` + `MODEL_CFG` per-scale batch/val_chunk (memory-only, fp32 throughout — 7B = fp32 small-batch, **no bf16**; bf16 is the deferred (a) retrain, not run for 7B). Fork 2: free-rider **both modes** (random@benign-std headline + zero floor). **Validation = all 5 code paths GREEN** (tiny-config smokes — values coarse, but structure/orientation/gating/Spearman-AUROC plumbing exercised end-to-end): silo5 4-threat (Spearman vs (b) **+1.000** all methods; poison FLDetector·STD-DAGMM·FedDQC **1.0** vs valuation 0.0 = the §3.9 framing reproduced), device100 cheap (Flirds proxy +1.0, loss-heur +0.999, FLTrust **1.0**, STD-DAGMM 0.54, FedDQC 0.10 off-threat), device100 anchor (perround dispatch + device coalition; Flirds vs **(b)-perround +0.999**, GTG +0.93/FedSV +0.82/ShapleyFL +0.86; (b)-perround 613s @ R=4), 3B ((b)/Flirds +1.000, MODEL_CFG no OOM). **poison config finding (acted on)**: the D2b backdoor PROPAGATION is config-sensitive — D2b worked at **lr=2e-3 / R=10 / BENIGN_STEPS=5**; the matrix uses the valuation lr=1e-3 → run the **poison threat as a separate invocation at `LR=2e-3`** (LR override added). The R=4 install-confirmation gave ASR=0 (too-few-rounds model-replacement dilution); the D2b-config (lr=2e-3/R=10) reproduction is being confirmed this session — the poison-detection structure (detectors 1.0 on the scaled attacker) is validated regardless. **real-config re-verify notes (not bugs)**: ComFedSV low-Spearman at R≤8 = completion-starved (task 7d = +1.000 at R=30); STD-DAGMM AE-on-CPU ~100s; device100 corrupt-seen needs R≈30. **NEXT = real grid execution (cost-tiered stage-gate) AFTER an independent verification session** (Yonghee's request). Committed this session (push by Yonghee).
- **⚠ poison-arm finding (matrix 2026-06-09 — CONTRADICTS the D2b conclusion; Yonghee to rule on the framing)**: at the **full D2b config** (`LR=2e-3 BATCH=8 ROUNDS=10 MAX_STEPS=5 POISON_TRAIN=1000`) the matrix reproduces the working backdoor (**deployed-ASR=1.00**; the matrix is faithful to D2b given its exact config — the matrix valuation default lr=1e-3/batch=16 gives ASR=0 because batch=16 halves the attacker install steps, ≈189 vs 375). On that working backdoor, under the matrix's **standard detection orientation** (corrupt = HIGH φ, the `eval.metrics.detection_auroc` convention used for noisy/free-rider): FLDetector/STD-DAGMM/FLTrust/FedDQC **AUROC=1.0**, loss-heur **1.0**, but **Flirds-1st AND Flirds-2nd = AUROC 0.0** — the clean-preserving attacker's γ-scaled update *descends* clean val-loss (⟨∇ℓ_val, γ(X−G)⟩ very negative) → lowest φ → ranked **most-helpful** → **Flirds is EVADED**, which **confirms the original §3.9 hypothesis** ("clean-perf-preserving backdoor preserves clean val-loss → evades Flirds = its boundary → complementary detector needed"). **This contradicts D2b's distilled "evades-Flirds REFUTED"** — D2b scored Flirds as `roc_auc_score(labels, −φ)` (tag "corrupt = LOW value"), so the attacker's most-negative φ gave a negated-AUROC of 1.0 that D2b read as detection. Honest framing: the attacker is φ-**extreme** (two-sided / magnitude / outlier-catchable, and the nonlinear loss-heur catches the γ-overshoot the linear 1st-order misses) but NOT φ-**high** (Flirds-as-valuation, standard detector, ranks it helpful). **The §3.9 headline framing — "Flirds detects all 3 threats" vs "Flirds detects noisy + free-rider, is evaded by the clean-preserving backdoor, the matched detectors are the complement" — is Yonghee's call** (confirm at the real config with the (b) oracle + full val; caveat tiny-config val=20). = **verification-session item #1**. raw [[raw/conversations/flirds/2026-06-09-phase2-step5-matrix-orchestrator-task8]] "## ⚠ poison vs Flirds".
- **device100 poison RESOLVED 2026-06-09 (per_client = the lever; Yonghee-approved)**: the cross-device poison ASR=0 was the **per-attacker INSTALL data** below D1's ~200-poisoned threshold (per_client=40, frac0.5 → 75 poisoned → each attacker's local X never installs), NOT propagation/convergence/scale and NOT a code bug (silo5 same code = ASR 1.0). Verified by a 4-GPU sweep (single-shot R∈{10,30,60}, multi-round γ∈{4,10}, **multi-attacker 5%/10% — all ASR=0**; more attackers don't help because install is per-client local, each sub-threshold) + an **A′ confirm: per_client=300, frac0.8 → 240 poisoned, EPOCHS=5, R=60 → deployed-ASR=0.75** (working, < silo5 1.0 = cross-device attack-round dilution). **Resolution**: `DEVICE` default per_client 40→**300** (poison-compatible; noisy/free-rider size-independent → unifies the regime); the poison threat is a **separate invocation at the D2b install config** (`LR=2e-3 BATCH=8 EPOCHS=5 POISON_FRAC=0.8`; device100 +`ROUNDS=60 MAX_STEPS=10`); the accumulation-hypothesis exploration code was reverted (answer = committed single-shot threat + env params). caveat: ASR 0.75 at tiny val=4 → confirm at real config.

**Recommendation** (historical, 06-02): at Phase 2 start, search GitHub for FLDetector + STD-DAGMM. If code exists, vendor + adapt to LoRA gradient; else reproduce (small robustness detectors, not full Shapley estimators). → FLDetector was a reference-guided self-build of the official `lbfgs` + detection loop (no vendoring).

**Regime split — LOCKED 2026-06-07 — ⚠ SUPERSEDED 2026-06-08** (by the threat-matched REDESIGN above; kept as history. The flaw it had: it paired FLDetector with *noisy*, but FLDetector is a poisoning detector — see redesign. Detectors now run in BOTH regimes.):

- **FLDetector → cross-silo (N=5/10)** — the noisy/poison comparator. Server-side, **from-logs closed-form** (Cauchy MVT + L-BFGS Hessian one-step-ahead prediction → ‖predicted − actual update‖ score), runs at any N. Use the **continuous suspicious-score for AUROC**; skip the paper's Gap-statistic + 2-means clustering (weak at small N).
  - **IMPLEMENTED 2026-06-07** (`flirds/baselines/fldetector.py`, reference-guided self-build of the official `lbfgs` + detection loop): **model-free** (no model/loss_fn/test_loader — only the logged update vectors → a different KIND of baseline = AUROC table only, no Spearman/value). `_lbfgs_hvp` = Byrd-Nocedal compact form (σI − [σS,Y]M⁻¹[σS,Y]ᵀ, float64 solve; hand-verified B·s=y on 1 pair). Mapping: gᵢᵗ = raw client delta `dm[c][0]`, wᵗ−wᵗ⁻¹ = n_c-weighted aggregate (FedAvg folds the step into the local delta), w_r unused; S_k = aᵗ⁻¹, Y_k = aᵗ−aᵗ⁻¹. Adaptation: score from ≥1 secant pair (the paper's full-window gate would zero out our R<10 runs); `window`=10 bounds both the L-BFGS pairs and the score average. Free-rider(zero) is NOT φ=0 here — it gets a *high* score (residual ‖Ĥv‖), the intended detection. Verified: synthetic check (anomalous client AUROC 1.000), CNN regression bit-identical, LLM smoke `PORT OK`, compare 9→**10 methods**.
  - **Result (1B N=5, R=10, lr1e-3, 3 seeds)**: FLDetector is the **cheapest** method (~24 s; model-free CPU linalg) but the **weakest detector** at N=5 — noisy AUROC **0.50** (chance) / free-rider **0.75**, vs every valuation method's 0.75 / 1.00. The *clean* math client tops the suspicious score in all 3 seeds = **systematic non-IID erosion** (the paper's IID-only Theorem 1; 5 disjoint domains break per-client temporal consistency). Direct evidence for the separator's difficulty → supports Flirds' "characterized limitation" framing. Headline detection scale = **N=10 (1B) / N=100 (cross-device, task 7)**. Raw: [[raw/conversations/flirds/2026-06-07-phase2-task5-fldetector-cross-silo]].
- **STD-DAGMM → cross-device (N=100, with task 7)** — the free-rider comparator. It **trains a DAGMM autoencoder + GMM on the set of client update vectors**, so it needs N≫ samples: at cross-silo N=5 it is **degenerate** (5 vectors, ~12M-dim LoRA updates; orig paper N=100 / 20 free-riders, 0.2M-param MLP). Free-riding is itself the cross-device incentive threat. (A std-only reduction at small N is possible but unfaithful → not taken.)
- N=5 detection AUROC is anyway **coarse** (1 positive / 5 → few attainable values; cf. the SV-compare's flat 0.75/1.0). The **headline detection table is N=10 (1B) / N=100 (cross-device)**.

### 3.10 Pluggable client-corruptor registry (seam 2 — fork (b) CHOSEN 2026-06-04)

**Open**: implement the corruptor/partitioner registry now (CNN side) vs. while building the LLM data layer. Currently corruption is **inline** in `experiments/phase05_*.py:build()` (`noisy={4,5}` + `if c in noisy:` label-flip) — the hardcoding seam 2 fixes.

**Design (agreed, code-ready)**: `flirds/data/corruptions.py` with a name→callable registry.
- sample-level: `fn(samples, rng, **cfg) -> samples` (label_flip, backdoor; LLM: answer-swap, trigger-token)
- update-level: `fn(delta_w, rng, **cfg) -> delta_w`, hooked in `fl/client.py` (free_rider)
- partition-level: helpers in `fl/partition.py` (maverick = sole-domain holder; duplicate = identical-data client)
- a run-config maps `{client_idx: corruptor_name}` → removes `noisy={4,5}` hardcoding. CNN + LLM `build()` both call it.
- **Backend split**: free_rider / maverick / duplicate are representation-agnostic (implement now); label_flip / backdoor / pgd need a per-backend body (CNN now, LLM when the data layer's instruction-response format is fixed).

**The fork — RESOLVED (b), 2026-06-04**: the registry is built **with the LLM data layer (Phase 1 task 3)**, folding into the data-format decision, rather than refactoring the committed CNN/phase05 path now. Until task 3 lands, noisy/free-rider stay inline in `experiments/phase05_*.py:build()`. (Option (a) — refactor CNN first — was the alternative; not taken, to avoid touching committed Phase 0.5 experiments.) Per `codes/CLAUDE.md`: surgical, no speculative abstraction — implement corruptors as the experiments that need them arrive (noisy/free-rider with the data layer; backdoor/PGD/maverick/duplicate in Phase 2/3).

> **(a) 최소 구현됨 (2026-06-04)**: sample-level `label_shuffle`만 `data/corruptors.py` (`CNN_CORRUPTORS` dict)로 추출 + phase05 dual/flirds_oracle/regime_sweep을 registry 호출로 refactor — **bit-identical** (flirds_oracle 0.7381/0.8810 불변). 의도적으로 최소: `noisy={...}` set 유지(corruptor 1종이라 run-config `{client_idx: corruptor_name}` map은 아직 over-engineering). **남은 풀 registry** — run-config map(noisy hardcoding 제거) + update-level free_rider(`fl/client.py` hook) + partition-level maverick/duplicate + corruptor 함수 시그니처 통일(`fn(samples, rng, **cfg)`) + LLM text corruptor — 는 해당 corruptor를 **실제 쓰는 시점**(Phase 2/3 detection + stage 3 LLM data layer)에 확장.

> **(b) ② LLM corruptor DONE (2026-06-04)**: noisy = `answer_swap` (within-client completion permute → CNN `label_shuffle` 직접 대응; FedDQC answer-swap + FedCorr data-side freeloader 선례) + free-rider = `free_rider(ref, mode)` (update-level, representation-agnostic; **zero + random** mode, **Lin et al. 2019 / STD-DAGMM taxonomy** — delta/advanced-delta는 Phase 2 STD-DAGMM head-to-head). `data/corruptors.py`(+`LLM_CORRUPTORS`/`free_rider`) + `data/llm.build(noisy=)` + `fl/llm_server.run_llm_fedavg_logs(free_riders=, free_rider_mode=)`; estimator/oracle/CNN **bit-identical (안 건드림)**. 검증: 단위 + CNN 회귀(0.7381/0.8810) + 실제 1B(free-rider zero φ=정확히 0 est&oracle, est≈oracle 2.7e-7). **prior-art 통찰**: "free-rider"는 한 개념이 아님 — FedCorr data-side ≈ `answer_swap`, Lin update-side ≈ `free_rider`, ADS replication ≈ Phase-3 `duplicate` (1:1 대응). Raw: [[raw/conversations/flirds/2026-06-04-phase1-corruptor-and-7-design]]. 풀 registry의 남은 항목(backdoor/PGD/maverick/duplicate + run-config map)은 Phase 2/3 그대로.

> **Validation-set (seam 3 / §3.4)**: already config-driven — `flirds_values(logs, model_fn, val_x, val_y, ...)` takes validation as args, no hardcoding. #16 (validation sensitivity) just re-calls with different `val_x/val_y`. No Phase-1 change needed beyond keeping it a parameter.

## 4. Next-session starter prompt

> **Superseded (2026-06-04)** — the original kickoff prompt below was for *starting* Phase 0. Phase 0/0.5 + Phase 1 stages 1–3 + #7 infra are done. **A continuing session now starts from the Status snapshot's "Next concrete action": launch the FULL `phase1_clean_run.py` scale run (MINI de-risk first), then ③ SV-baselines port (Phase 2).** Read the status snapshot + the latest raw conversations ([[raw/conversations/flirds/2026-06-04-phase1-corruptor-and-7-design]], [[raw/conversations/flirds/2026-06-04-phase1-data-layer]]) first.

Original kickoff prompt (historical):

> "flirds 구현 phase 시작. [[flirds-implementation-plan]] 읽고 phase 0부터 시작하자. 일단 §3.6 BASE_REPO 결정부터 같이 해줘."

## 5. Pre-implementation checklist (✓ ALL DONE — historical, kept for record)

> Superseded: Phase 0 started 2026-06-02 and all of the below were satisfied. Current entry point = the Status snapshot's "Next concrete action" (launch the FULL #7 scale run; then ③ SV-baselines port).

- [x] §3.6 BASE_REPO decided (OpenFedLLM reference + self-build; CNN self-built)
- [x] Logging = local run-dir, **no W&B** (D2) — *(was "§3.7 W&B init"; corrected)*
- [x] code accessible; conda env `flirds` (torch 2.12+cu130)
- [x] B200 × 4 access verified
- [x] Phase 0 setup paper PDFs accessible (GTG 2109.02053 / FedSV 2009.06192 / ComFedSV 2109.09046 / Ripple 40034)
- [x] Root `CLAUDE.md` updated to `stage: implementation`

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
| "How do I record per-experiment time / GPU-hours?" | [[flirds-protocol#15. Experiment instrumentation & reporting (the W&B replacement)]] §15.1 |
| "Where's the cross-run aggregation / plotting tool (no W&B)?" | [[flirds-protocol#15. Experiment instrumentation & reporting (the W&B replacement)]] §15.3 |
| "What is Phase 0's pass criterion?" | [[flirds-protocol#10. Phase 0 — baseline reproduction (status: DONE 2026-06-02/03)]] |
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
