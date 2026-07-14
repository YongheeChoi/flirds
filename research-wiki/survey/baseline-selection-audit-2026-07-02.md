---
type: survey
title: "Flirds baseline 선정(selection) 정당성 감사 (2026-07-02)"
created: 2026-07-02
updated: 2026-07-02
tags: [survey, baseline, selection, audit, fidelity]
---

# Flirds baseline 선정(selection) 정당성 감사

> **이 문서의 질문**: 우리가 baseline을 *제대로 골랐는가* — 선정의 **정당성**(왜 이건 넣고 저건
> 뺐나)과 **완결성**(빠진 필수 baseline이 있나)을 실험군별로 감사한다.
> 자매 문서 [[baseline-original-paper-verification-2026-06-22]]는 **다른 질문**을 다룬다: *이미 고른*
> baseline의 우리 실험 **수치**가 원 논문 보고치·세팅과 맞는지(수치 재현/정성 대조). 즉 06-22는
> "고른 것이 맞게 굴러가나", 이 문서는 "애초에 맞게 골랐나".
>
> 판정 기준(기계적 gap-counting 금지): **A** = 필수(빠지면 진짜 gap — 직접경쟁/분야표준 앵커/
> 우리가 이기려는 축의 SOTA) · **B** = 선택(넣어도/빼도 근거만 대면 방어) · **C** = 정당한 제외
> (목적축 불일치 / 가정·범위 위배 / 포섭·중복 / 모달리티·스케일 밖 / valuation 아님).
> 판정은 전부 **우리 주장·무대 관련성**으로. 근거 없는 단정은 "관찰된 빈칸"으로 서술.
>
> 겨냥 셀 = **federated × LLM(LoRA) × client-level × valuation-fidelity vs exact 2ᴺ dual-oracle**
> (utility = 미분가능 val-loss, GT = (a)retrain-valloss + (b)in-run, fp32). 핵심 위계: **1차 fidelity**
> → 2차 ①perf ②convergence ③detection. 따라서 fidelity 트랙을 먼저·크게, detection은 뒤(별도 기준군).

---

## 1. 요약 표 — 실험군별 roster 정당성

| # | 실험군 | 위계 | roster 정당성 | 필수(A) 누락 한 줄 |
|---|--------|------|:---:|------|
| 1 | **Fidelity — LLM 표준** (track_d; 헤드라인) | 1차 | **◐** | **Federated-LOO** 1종 — 분야표준 최저비용 앵커, 2ᴺ retrain 부산물로 거의 무료 |
| 2 | **Fidelity — CNN cross-silo N=10** (track_c C1) | 1차 | **◐** | **Federated-LOO** — 현 loss-heur는 singleton U({k})라 LOO(marginal) 대체 못 함 |
| 3 | **Detection — Robustness** (phase2_matrix) | 2차 ③(마지막) | **◐** | 필수 누락 없음 — 위협별 강한 전용탐지기 확보(noisy=FedDQC/FR=FLTrust·STD-DAGMM/poison=FLDetector) |
| 4 | **Selection→Perf / Aggregation** (track_d arms + C2) | 2차 ① | **◐** | 신규 A 누락 없음 — 단 clean-oracle 상한·random-q% 하한 arm의 실제 실행/보고 확정 필요(medium) |
| 5 | **Attack / 위협모델** (별도 기준군) | 위협원 | **◐** | 필수 위협 전부 커버 — 데이터오염·model-poison·free-rider 3축 완비 |

**종합 판정: ◐ (방어 가능·보강 권장).** 유일한 실질 gap은 두 fidelity 트랙 공통의 **Federated-LOO**
한 종(우선순위 high). 직접 fidelity 경쟁군(GTG·FedSV·ComFedSV·ShapleyFL·FedIF·Banzhaf) + 듀얼
exact-2ᴺ oracle은 완결. 겨냥 셀은 2024–2026 웹 검색으로도 **여전히 빈칸** — novelty 유지.

> ### 후속 조치 · 정정 (2026-07-02, 이 감사 직후 Yonghee 검토 반영)
> **A-gap #1 (Federated-LOO) — ✅ 구현 완료(수치는 재실행 대기).** `flirds/oracle/in_run_sv.py`에
> `in_run_loo` 추가(φ_i = U(N)−U(N\{i}), (b)-게임 leave-one-out marginal), `track_d.py`·`track_c1.py`
> compute_fidelity에 `Fed-LOO` 메서드로 와이어링. **비용은 loss-heur 클래스**(라운드별 additivity로
> Σ_r[u_r(P_r)−u_r(P_r\{i})] 분해 = O(Σ|P_r|), std20의 N×trajectory 아님). **정확도 검증됨**: 합성
> logs에서 brute-force U(N)−U(N\{i})와 **max-diff 0.0**(full+partial 참여), free-rider LOO=0. `make_fidelity.py`가
> method 동적 수집이라 재실행 시 `fidelity.csv`에 자동 반영. **남은 것 = fidelity 셀 재실행**(로그
> 미영속이라 offline 백필 불가; GPU 재실행이 populate 경로; 07-03+ `runs/probe_signal/` 신규 셀엔 Fed-LOO 수치 기록 시작(d2e7ed6·e85df6e) — 본 roster 셀은 여전히 미기록). loss-heur(singleton)와 formally distinct.
>
> **A-gap #2 (clean-oracle 상한 / random-q% 하한) — 정정: '누락'이 아니라 '의도적 제외'.** plan
> line 430("미결 D 세션"의 4-arm filter-retrain)은 line 387이 **"SUPERSEDED 2026-06-13"**로 표기한
> *원설계*이며, 확정 설계(line 437)는 **"removal curve 제외"**로 이 bracket을 명시적으로 뺐다(online
> do-no-harm arm 6종으로 대체). 게다가 **track_d는 clean-IID라 clean-oracle=known-clean 전원 유지=vanilla와
> degenerate**(상한이 무의미). random-q% floor는 오염 있는 `phase1`에 `random_k`로 이미 존재. → 이 항목은
> **A 승격 대상 아님**; 상/하한 bracket이 의미 있는 곳은 *오염* 트랙(track_c2·phase2)뿐. **Yonghee 2026-07-02:
> 지금은 보류**(구현 시 오염 트랙에 추가가 옳음). 아래 §3.1·§6의 medium 항목은 이 정정으로 대체됨.

---

## 2. 경쟁논문 baseline 사용 근거 (공통 근거표)

> 이 표가 "제대로 골랐나"의 1차 자료다. **핵심 신호 = "exact-SV fidelity를 쟀나"** 열: 우리 셀(F×
> client×exact-2ᴺ-fidelity)을 실제로 점유한 선행이 있는지를 드러낸다. 원문 근거는 각 논문 §Baselines/
> Experiments verbatim(JSON key_quotes).

| 논문 | 세팅(F/model/unit) | 쓴 baseline 집합 | **exact-SV fidelity 측정?** | 주목적 |
|------|------|------|:---:|------|
| **FedSV** (Wang 2020) | F·CNN/MLP·client | **Fed.LOO, Random** (+normalized 자가-ablation) | ✗ (다운스트림 검출/요약만) | 검출·요약 |
| **GTG-Shapley** (2022) | F·MNIST-NN·client | **exact 2¹⁰(GT)**, TMC, GroupTesting, MR, FedSV, TMR | ✓ (거리: cos/euclid/maxdiff) | fid+eff |
| **ComFedSV** (2022) | F·LR/CNN/VGG·client | FedSV, **full-matrix fed-Shapley(GT)** | ◐ (Spearman vs full-matrix, **≠exact 2ᴺ**) | fair |
| **FedTSV** (2026) | F·MLP/ResNet·client | FedAvg, **CGSV**, LOO | ✗ (정성 3-group 분리만) | agg/fair |
| **FedIF** (2025) | F·CNN·client | FedAvg, FedProx, **AFedSV(=ShapleyFL)**, Krum | ✗ (test-acc+time) | robust agg |
| **ShapleyFL/AFedSV** (2023) | F·CNN/EffNet·client | FedAvg, FedProx, **FedSV, S-FedAvg, RFA** | ✗ (test-acc vs rounds) | robust |
| **Game of Gradients/S-FedAvg** (2021) | F·MLP·client | **FedAvg** (+자가 변형) | ✗ (정성 분리+test-acc) | robust sel |
| **Ripple Shapley** (2026) | F·CNN/MLP·**sample** | FedAvg, FedProx, s-FedAvg, FedSV, AFedSV+, Plain | ✗ (**명시적 거부**, task-driven) | eff+robust |
| **IRDS** (2024) | **C**·LLM·**sample** | **MC-IRDS(GT)**, true U⁽ᵗ⁾, IF, BM25, KNN-Sh, TRAK, Datamodels, LESS… | ◐ (**MC-Shapley**, exact 2ᴺ 불가) | fid+eff |
| **SPACE** (2023) | F·CNN·client | GT-Shapley, TMC, GTG, DIG-FL, **Real 2ᴺ(GT)** | ✓ (**PCC** vs exact 2ᴺ) | fid+eff (prototype util) |
| **ShapFed** (2024) | F·ResNet/EffNet·**class** | **true Shapley(정성)**, CGSV, FedAvg, Individual | ◐ (heatmap 정성만) | agg+fair |
| **Mavericks/FedEMD** (2021) | F·CNN·client | Random, SVB, FedFast, TiFL, FedProx | ✗ (R@99, fairness 기준) | sel+fair critique |
| **RFedLR** (2025) | F·**LoRA-ViT**·sample | FedLR, FedPF/BF/AP, SLoRA, FFA-LoRA, RoLoRA, FlexLoRA | ✗ (test-acc) | robust PEFT agg |
| **iPFL** (2025) | F·**LoRA-LLM+CNN**·client | FedAvg, FedProx, Ditto, FedAMP, CFL, FedFomo, pFedGraph | ✗ (economic utility) | market/incentive |
| **FedDQC** (2025) | F·**LoRA-LLM**·**sample** | FedAvg oracle/mix, PPL, loss, IFD, NUGGETS, **DataInf** | ✗ (binary clean/corrupt) | data-quality sel |
| **FedHDS** (2025) | F·**LoRA-LLM**·sample | FedAvg, FedPTuning/Prompt, FedIT, FlexLoRA, Random, PPL | ✗ (Rouge-L+time) | coreset sel |
| **FLDetector** (2022) | F·CNN·client(binary) | VAE, FLD-Norm, FLD-NoHVP (FLTrust 명시 제외) | ✗ (DACC/FPR/FNR) | 악성검출 |
| **FLTrust** (2021) | F·CNN/ResNet·client | FedAvg, Krum, Trim-mean, Median | ✗ (test-err/ASR) | Byzantine 방어 |
| **STD-DAGMM** (2019) | F·MLP·client | holdout-val, Autoencoder, DAGMM | ✗ (AUC) | free-rider 검출 |
| **FoolsGold** (2020) | F·softmax/CNN·client | 무방어 FL, Multi-Krum, RONI | ✗ (attack-rate) | sybil 방어 |
| **FedCorr** (2022) | F·ResNet·client→sample | FedAvg, FedProx, RoFL, ARFL, JointOpt, DivideMix, FedDyn | ✗ (test-acc) | 라벨노이즈 교정 |
| **Bagdasaryan** (2020) | F·ResNet/LSTM·client | naive backdoor, anomaly-evasion 방어 | ✗ (attack) | backdoor 공격 |
| **Xu Instr-Backdoor** (2024) | **C**·LLM·sample | instance-level 공격, trigger 변형, ONION/RLHF 방어 | ✗ (attack) | backdoor 공격 |
| **Data Shapley** (G&Z 2019) | **C**·shallow·sample/group | **LOO, Random**, leverage(주장), exact 2ᴺ(n=4–14) | ◐ (n=4–14 부록만) | fair/axiom |
| **Data Banzhaf** (2023) | **C**·shallow·sample | **Data Shapley, LOO**, 4 Beta-Sh, Uniform, Least core | ◐ (n=10 synth + k=50 ref) | semivalue robustness |

**관찰**:
1. **LOO·Random·exact-2ᴺ**이 valuation-fidelity 계열의 반복 앵커(FedSV·GTG·SPACE·G&Z·Banzhaf).
2. exact-2ᴺ fidelity를 정면으로 잰 선행은 **GTG·SPACE**(둘 다 CNN·소형·distance/PCC), ComFedSV·IRDS·
   G&Z·Banzhaf는 **근사 GT**(full-matrix / MC / n≤14). **어느 것도 F×LLM×client×exact-2ᴺ이 아님.**
3. F×LLM×client 인접작(RFedLR·iPFL·FedDQC·FedHDS)은 전부 **fidelity 미측정**(집계·품질·coreset 목적).

---

## 2A. Fidelity — LLM 표준 (track_d) · **1차 헤드라인**

**우리 roster**: Flirds, Flirds-1st, loss-heur, GTG, FedSV, ComFedSV, ShapleyFL(β0.3), FedIF,
Banzhaf(anchor5), **(a) retrain-valloss oracle (=exact client-level Data Shapley)**, **(b) in-run oracle**.

**셀**: F × LLM(LoRA) × client × fidelity vs exact 2ᴺ dual-oracle; N=5 anchor(exact GT) + std20.

| candidate | klass | in_roster | reason | (C)제외사유 |
|---|:---:|:---:|---|---|
| Flirds | A | ✓ | 제안 방법 본체(A-iii 축 후보 SOTA) | — |
| Flirds-1st | A | ✓ | 1차 Taylor ablation; CGSV/FLTrust cosine 계열 흡수 | — |
| GTG-Shapley | A | ✓ | 직접경쟁(A-i): F×client×exact-2ᴺ fidelity를 정면 수행한 선행 | — |
| FedSV (Wang 2020) | A | ✓ | 직접경쟁(A-i): star-server fed-Shapley 정본 | — |
| ComFedSV | A | ✓ | 직접경쟁(A-i): Spearman-vs-fed-Shapley 실제 수행 | — |
| ShapleyFL/AFedSV(β0.3) | A | ✓ | 직접경쟁 in-family; 다수 논문의 SOTA comparator(AFedSV+) | — |
| FedIF | A | ✓ | 직접경쟁: 1차 trajectory-influence(Flirds-1st 최근접 사촌) | — |
| Banzhaf(anchor5) | A | ✓ | 분야표준 semivalue 앵커(A-ii); std20 제외는 2²⁰ 불가로 정당 | — |
| (a) retrain-valloss oracle | A | ✓ | 'plain Data Shapley' exact 형태 = 듀얼 GT의 한 축(=Data Shapley 앵커 충족) | — |
| (b) in-run oracle | A | ✓ | in-run exact GT; exact-2ᴺ까지 확보 = novelty 강화 | — |
| loss-heur | A | ✓ | 값싼 heuristic 앵커(A-ii); singleton U({k}) | — |
| **Federated-LOO** | **A** | **✗** | **분야표준 최저비용 앵커(A-ii). loss-heur(singleton)에 포섭 안 됨** | — |
| TMC-Shapley | B | ✗ | exact는 이미 (a)oracle; TMC는 그 재훈련 근사라 N=5선 자명·중복 | — |
| Random(φ floor) | B | ✗ | Spearman≈0 sanity floor; 넣/빼 방어 가능 | — |
| FedOwen(Owen sampling) | B | ✗ | 2025 최신 estimator; 가족은 GTG/FedSV로 대표됨(리뷰어 커버리지용) | — |
| S-FedAvg(GoG) | B | ✗ | MC-permutation Shapley; genus는 FedSV/ShapleyFL로 대표 | — |
| Beta-Shapley/Least Core/Uniform | B | ✗ | semivalue 스펙트럼 보강용; Banzhaf가 대표 | — |
| vanilla-FedAvg | C | ✗ | 집계규칙 — φ 순위 없음 → fidelity 행 불가(2차 arm으로만) | valuation 아님 |
| Group Testing/MR/TMR/DPVS-Sh | C | ✗ | 전부 FL/Shapley 근사 estimator; GTG(지배)·FedSV가 대표 | 포섭·중복 |
| CGSV | C | ✗ | gradient-cosine = Flirds-1st가 1차 신호로 대표(집계-방향 대상) | 포섭·중복 |
| DIG-FL | C | ✗ | influence/gradient-similarity = valuation측 GTG/FedSV·1차측 FedIF에 이중 포섭 | 포섭·중복 |
| SPACE | C | ✗ | prototype/이산-class utility 필요 → 생성형 LLM val-loss로 전이 불가 | 가정·범위 위배 |
| ShapFed(CSSV) | C | ✗ | 헤드라인=aggregation+fairness; exact-SV 정성(heatmap)만 | 목적축 불일치 |
| FLTrust | C | ✗ | cosine-to-root ≈ Flirds-1st; φ 미산출·Byzantine 방어 | 포섭·중복 |
| Krum/Multi-Krum/Trim/Median/RFA/FedProx | C | ✗ | 집계/강건집계 규칙 — φ 순위 없음 | valuation 아님 |
| FLDetector/VAE/STD-DAGMM/DAGMM/FoolsGold | C | ✗ | 악성/free-rider 탐지기 = 2차③ 별도 기준군 | valuation 아님 |
| FedCorr/RoFL/ARFL/DivideMix/JointOpt | C | ✗ | 라벨노이즈 탐지·재라벨(정확도 회복) | valuation 아님 |
| Backdoor 공격(Bagdasaryan/Xu) | C | ✗ | 공격/threat-model | valuation 아님 |
| IRDS + Adam/Layer-Aware/LM-Arithmetic 변형 | C | ✗ | centralized·sample/source-level (LLM이나 F 아님) → lineage 인용 | 가정·범위 위배 |
| IF(Koh-Liang)/DataInf/KNN-Shapley | C | ✗ | 고전 IF=FedIF가 대표; Hessian-inv IF는 FL 부적합; sample-level | 포섭·중복 |
| KFCA(2026) | C | ✗ | F×LLM×client 3/4 일치·최근접이나 목적=incentive(E4), knowledge-free util | 목적축 불일치 |
| FedAttr(2026) | B | ✗ | F×LLM×client 3/4 일치; privacy-attribution 목적, φ 산출 여부 미검증 → fetch 후 판정 | — |
| WinFLoRA/iPFL/FedTSV/Truth-Shapley/FRECA/SNOWFL/FedMS | C | ✗ | 목적=incentive/보상·집계·공정·selection(E4/E7 비목표·2차①) | 목적축 불일치 |
| SecSV/HESV | C | ✗ | cross-silo SV의 보안(HE) 계산 축; LLM-HE 불가 | 목적축 불일치 |
| FedDQC/FedHDS/RFedLR | C | ✗ | FedAvg+LoRA 무대는 겹치나 sample-level 품질/coreset·robust-agg | 목적축 불일치 |
| CoAst | C | ✗ | validation-free = 우리 val-loss utility의 정반대 전제 | 가정·범위 위배 |
| FLCE(VLDB'24 벤치마크) | C | ✗ | 방법 아님 — 방법선정·robustness 프로토콜 준거로 인용 | 포섭·중복 |
| FlowerTune | C | ✗ | F×LLM stage 참조점이나 client valuation 안 함(GPT-judge) | 목적축 불일치 |
| Ripple Shapley | C | ✗ | SAMPLE-level; 논문 스스로 numerical fidelity 거부 | 가정·범위 위배 |

---

## 2B. Fidelity — CNN cross-silo N=10 (track_c C1) · **1차 헤드라인**

**우리 roster**: Flirds, Flirds-1st, GTG, FedSV, ComFedSV, Banzhaf(in-run), ShapleyFL(β0.3), FedIF,
loss-heur, Ripple(집계), **(a) exact retrain SV 2¹⁰**, **(b) exact in-run SV 2¹⁰**.

**셀**: GTG 5-scenario stage(MNIST+LeNet5 / CIFAR-10+FedSVCNN, N=10 full) vs 듀얼 exact 2¹⁰ oracle.

| candidate | klass | in_roster | reason | (C)제외사유 |
|---|:---:|:---:|---|---|
| Flirds / Flirds-1st | A | ✓ | 방법 본체 + 2차항 분리 ablation | — |
| (a) exact retrain 2¹⁰ / (b) exact in-run 2¹⁰ | A | ✓ | 듀얼 GT; N=10이라 2¹⁰ 계산 가능(문헌공백 메움) | — |
| GTG-Shapley | A | ✓ | A-i: 이 실험군이 곧 GTG stage 재현 | — |
| FedSV(per-round, Wang 2020) | A | ✓ | A-i/ii: 연합 SV 정본. (2502.17526 동명 Byzantine-FedSV와 구분 명기) | — |
| ComFedSV | A | ✓ | A-i: N=10 full-participation이라 저랭크 완성 자명 → 이 셀에선 정당 직접경쟁 | — |
| ShapleyFL(β0.3) | A | ✓ | A-i: 연합 SV SOTA(AFedSV+) | — |
| Banzhaf(in-run) | A | ✓ | A-ii: semivalue 표준 앵커 | — |
| loss-heur = U({k}) | A | ✓ | A-ii: 값싼 heuristic 슬롯. **단 singleton이라 LOO 대체 아님** | — |
| FedIF | B | ✓ | Flirds-1st 사촌; 원 목적 robust-agg라 필수는 아니나 1차 gradient 대표 | — |
| Ripple Shapley | B | ✓ | 최신 FL in-run Shapley positioning; sample→client 집계형(빼도 방어 가능) | — |
| **Federated-LOO** (U(N)−U(N\{i})) | **A** | **✗** | **A-ii 명시 앵커; N=10선 저비용. loss-heur(singleton)와 다른 양** | — |
| TMC-Shapley | B | ✗ | plain-DS 추정기; exact는 (a)oracle이 충족. retrain 비용 대비용이면 추가 가치 | — |
| Group Testing Shapley | B | ✗ | 값싼 추정기; GTG/FedSV/ComFedSV가 계열 대표 | — |
| CGSV | B | ✗ | gradient-cosine SV; ShapFed·FedTSV가 대비하는 prior-art(선택 보강) | — |
| DIG-FL | B | ✗ | influence FL 기여추정; FedIF와 근접 | — |
| FedTSV | B | ✗ | trajectory-SV로 채점 가능하나 목적=adaptive-agg/공정; 필수 아님 | — |
| FedOwen | B | ✗ | 별개 estimator class(Owen); CNN N=100·selection 목적 | — |
| Random(floor) | B | ✗ | Spearman≈0 바닥선(표준이나 자명) | — |
| MR / TMR | C | ✗ | gradient-recon 구세대; GTG가 지배(TMR을 능가) | 포섭·중복 |
| vanilla FedAvg | C | ✗ | 집계규칙 — φ 없음 | valuation 아님 |
| SPACE | C | ✗ | utility=prototype 유사도(이산 class 필요) → val-loss 게임과 다른 게임 | 가정·범위 위배 |
| Krum/Trim/Median/RFA | C | ✗ | Byzantine 집계규칙 — φ 없음 | valuation 아님 |
| FLTrust | C | ✗ | cosine-to-root ≈ Flirds-1st | 포섭·중복 |
| FLDetector/FoolsGold/STD-DAGMM/FedCorr | C | ✗ | 탐지·교정기 = 2차③ 기준군 | valuation 아님 |
| IRDS(+Adam) | C | ✗ | centralized·sample-level | 모달리티·스케일 밖 |
| KFCA/WinFLoRA/FedAttr | C | ✗ | 목적=incentive/agg/privacy-attribution | 목적축 불일치 |
| Maverick-Sh/DPVS-Sh/Truth-Shapley | C | ✗ | selection·효율·truthfulness 목적 | 목적축 불일치 |

---

## 2C. Detection — Robustness (phase2_matrix) · **2차 ③ (위계상 마지막)**

**우리 roster**: Flirds, Flirds-1st, loss-heur, FedIF, GTG, FedSV, ShapleyFL, Banzhaf, ComFedSV(device),
**FLDetector, STD-DAGMM, FLTrust, FedDQC**(전용탐지기 레퍼런스군), (b) in-run exact-2ᴺ oracle.

> 이 축은 **φ-as-detector AUROC** + **위협별 전용탐지기 레퍼런스**로 구성. 헤드라인 아님(de-headlined).
> 핵심 서사 **C7** = "전용탐지기 ≥ φ"(exact (b) oracle의 noisy AUROC조차 0.604±0.050 → φ 침식은
> 근사결함 아니라 valuation 내재한계), **C8** = 2차항이 clean-보존 poison에서 부호 부분복원.

| candidate | klass | in_roster | reason | (C)제외사유 |
|---|:---:|:---:|---|---|
| Flirds / Flirds-1st | A | ✓ | C8 분리점 주체(2차 0.917 vs 1차 0.000) | — |
| loss-heur | A | ✓ | 값싼 앵커; poison AUROC 1.0으로 C8 논증 지지 | — |
| FedIF | A | ✓ | 직접경쟁 gradient/influence(fedif_w 출처) | — |
| GTG / FedSV / ShapleyFL | A | ✓ | 직접경쟁 FL-Shapley; 빠지면 cherry-picking | — |
| Banzhaf | B | ✓ | semivalue 앵커(free-rider φ=0 특성); 탐지축 필수는 아님 | — |
| ComFedSV(device) | B | ✓ | low-rank FL-Shapley 직접경쟁이나 device 한정 | — |
| **FLDetector** | A | ✓ | 전용탐지기: poison AUROC 1.0 + "가장 싸나 가장 약함" 정직 보고 | — |
| **STD-DAGMM** | A | ✓ | free-rider 전용 레퍼런스(위협별 위계 논증 필수) | — |
| **FLTrust** | A | ✓ | 강한 탐지기 — free-rider서 φ 이김 → C7 지지(강탐지기 포함=cherry 아님) | — |
| **FedDQC** | A | ✓ | FL+LLM+LoRA 동일무대; noisy서 1.0으로 φ(0.57~0.77) 압도(C7 결정적) | — |
| (b) in-run exact-2ᴺ oracle | A | ✓ | GT — noisy AUROC 0.604가 C7 linchpin | — |
| FoolsGold | B | ✗ | poison/sybil 전용(wiki 후보); "표준 backdoor 방어 왜 없나" 선점 가능하나 sybil-collusion+non-IID-honest 가정 한계 | — |
| vanilla-FedAvg | B | ✗ | 성능·수렴 arm의 substrate로 이미 포함(탐지 AUROC 미산출) | — |
| FedProx | B | ✗ | C6(강건-집계) 방향 주장 시 대비점; E7 비목표라 배제도 방어 | — |
| FedCorr | C | ✗ | partial-participation·symmetric-noise·CNN·detect+relabel 파이프라인 가정위배 | 가정·범위 위배 |
| Krum/Multi-Krum/Trim/Median | C | ✗ | 강건집계 규칙 — 탐지 스코어/φ 미산출(E7 비목표) | 목적축 불일치 |
| VAE / DAGMM / Autoencoder | C | ✗ | FLDetector/STD-DAGMM의 전신 — 후속작이 이미 대표 | 포섭·중복 |
| RONI | C | ✗ | val-loss 영향 거부 ≈ loss-heur가 대표 | 포섭·중복 |
| RoFL/ARFL/DivideMix/JointOpt | C | ✗ | 강건-학습/라벨보정(정확도 회복) | valuation 아님 |
| DataInf | C | ✗ | sample-level; FL+LLM 적응판을 FedDQC가 이미 실험(취약 보고) | 포섭·중복 |
| PPL/IFD/NUGGETS | C | ✗ | sample-level 선택 휴리스틱; FedDQC가 대표 | 목적축 불일치 |
| Federated-LOO | C | ✗ | (탐지군 한정) val-loss marginal 신호 = loss-heur + exact-2ᴺ semivalue가 포섭 | 포섭·중복 |
| plain Data Shapley(centralized TMC) | C | ✗ | 중앙·재학습; GT는 (a)/(b) oracle, FL 아날로그는 FedSV/GTG | 포섭·중복 |
| FedOwen | C | ✗ | GTG/FedSV/Banzhaf와 동일 estimator class; 탐지기 아님 | 포섭·중복 |
| Truth-Shapley/Data-Overvaluation | C | ✗ | truthfulness/incentive(E4); rank-fidelity·AUROC 미측정 | 목적축 불일치 |

---

## 2D. Selection→Perf / Aggregation (track_d arms + C2) · **2차 ①** (NON-headline)

**우리 roster**: base/FedAvg-mixed(하한), vanilla(FedAvg+LoRA), flirds_w(β0.5), flirds_sel, flirds_repl,
flirds_add, shapleyfl_w(β0.5), fedif_w(β0.7), sfedavg(CNN). (random-q%·clean-oracle bracket은 §1 후속 정정대로 미포함 — random_k는 phase1 오염 트랙에만 존재.)

> 주장 = **clean-IID do-no-harm parity** (E7 집계품질=비목표). 헤드라인 아님.

| candidate | klass | in_roster | reason | (C)제외사유 |
|---|:---:|:---:|---|---|
| vanilla FedAvg(+LoRA) | A | ✓ | 분야표준 do-nothing 앵커(=우리 무대 자체) | — |
| Random selection(floor) | A | ✓ | selection→perf 표준 floor(D-메인 random-q%) | — |
| clean-oracle(ceiling) | A | ✓ | 데이터품질 필터링 주장의 표준 상한(FedDQC oracle 대응) | — |
| S-FedAvg(GoG) | A | ✓ | SV 기반 client selection 직접경쟁 정본 | — |
| ShapleyFL(AFedSV) | A | ✓ | SV 기반 robust-agg/weighting 직접경쟁(shapleyfl_w 출처) | — |
| FedIF | A | ✓ | 1차 influence 적응집계(fedif_w 출처); 이 그룹선 weighting arm으로 목적 정합 | — |
| FedProx | B | ✗ | 이질성-강건 준표준 앵커; clean-IID선 FedAvg와 근사동치 | — |
| Federated-LOO | B | ✗ | 값싼 valuation 앵커; SV-weighting arm 이미 다수 보유(본질은 E1 그룹 앵커) | — |
| CGSV | B | ✗ | SV weighting arm으로 자연 편입 가능한 유일한 미포함 SV점; E4/E7 목적 | — |
| FedOwen | B | ✗ | 2025 SOTA SV+selection; flirds_sel/S-FedAvg가 이미 대표(CNN 커버리지용) | — |
| FedTSV | C | ✗ | trajectory-SV 조향 = 집계-우선(adaptive agg/fairness); oracle-fidelity 미측정 | 목적축 불일치 |
| ShapFed/ShapFed-WA | C | ✗ | class-level SV → weighted-agg+개인화/공정; exact-SV 정성만 | 목적축 불일치 |
| RFedLR | C | ✗ | 연합 LoRA label-noise robust-agg; 가중=Fisher/noise heuristic | 목적축 불일치 |
| iPFL | C | ✗ | model-sharing market/incentive(E4/E7); value=economic utility | 목적축 불일치 |
| WinFLoRA | C | ✗ | 연합 LoRA noise-aware 가중집계=인센티브(E4/E7) | 목적축 불일치 |
| Krum/Multi-Krum/Trim/Median/RFA | C | ✗ | Byzantine 방어 집계 — clean-IID do-no-harm과 무관(2차③ 축) | valuation 아님 |
| FedMS/Maverick-Aware SV | C | ✗ | class-wise 재가중 rare-client 공정+selection(E4계열) | 목적축 불일치 |
| PFL 군(Ditto/FedAMP/CFL/FedFomo/pFedGraph) | C | ✗ | 개인화 집계 — valuation 아님 | 목적축 불일치 |
| Fed-LoRA 집계군(SLoRA/FFA/RoLoRA/FlexLoRA/FedIT) | C | ✗ | LoRA 집계·랭크 최적화; vanilla가 FedIT에 대응(stage 포섭) | 포섭·중복 |
| plain Data Shapley(centralized) | C | ✗ | 중앙·retrain·sample-level — FL 집계 arm 아님 | 모달리티·스케일 밖 |
| FedDQC(IRA/PPL/DataInf/IFD/NUGGETS) | C | ✗ | FL-LLM 표본단위 quality selection(client-level 아님) | 목적축 불일치 |
| FedFast/TiFL | C | ✗ | 시스템효율 selection(클러스터/티어) — valuation-driven 아님 | 목적축 불일치 |
| SNOWFL | C | ✗ | Owen-value 가중집계(E7); URL 2차출처 미확정 | 목적축 불일치 |
| KFCA | C | ✗ | knowledge-free incentive(E4); exact-Shapley 비교 MNIST 한정 | 목적축 불일치 |

---

## 2E. Attack / 위협모델 (별도 기준군) — fidelity(φ 정직성) + detection(AUROC) 오염원

**우리 roster**: answer_swap(LLM data-quality), label_flip(CNN noisy), free_rider zero/random,
Bagdasaryan model-replacement/scaling backdoor(γ=n/η, frac=0.5), Xu-2023 instruction-trigger('tq').

| candidate | klass | in_roster | reason | (C)제외사유 |
|---|:---:|:---:|---|---|
| answer_swap(LLM) | A | ✓ | 헤드라인 LLM 무대의 데이터오염 대표(FedDQC 매칭); φ 의미검증+noisy-AUROC | — |
| label_flip(CNN) | A | ✓ | FL 최표준 데이터오염(경쟁 8+편 사용); 이미 인라인 구현 | — |
| Bagdasaryan model-replacement | A | ✓ | FL 정전 표적 model-poison; detection+φ 해석 핵심 | — |
| free_rider zero/random | A | ✓ | FL 고유 표준 위협; φ≈0 null-player 검증 | — |
| Xu-2023 instruction-trigger | B | ✓ | LLM backdoor 콘텐츠 공급('tq'); 'backdoor 존재'는 Bagdasaryan이 A 충족 | — |
| **sign-flip(untargeted Byzantine)** | B | ✗ | 음의 φ fidelity 프로브 가치(보강 1순위); robust-agg 견고성 겨냥이라 필수는 아님 | — |
| advanced-delta free-rider | B | ✗ | STD-DAGMM 강한 변형(task9 계획); Flirds 위험 지점 stress-test | — |
| PGD test-time | B | ✗ | FedIF blind-spot; 2nd-order novelty 부각용(Phase-3) | — |
| gradient-noise injection | B | ✗ | model-poison 비표적 변형; 위협공간 이미 커버 | — |
| noisy-features / label-shuffle | C | ✗ | label-flip/answer_swap이 데이터오염 signature 지배 | 포섭·중복 |
| DBA / edge-case backdoor | C | ✗ | backdoor 카테고리를 Bagdasaryan+Xu가 대표; 소형 N선 별 signature 없음 | 포섭·중복 |
| Krum/Trim/Fang-adaptive/ALIE | C | ✗ | robust-aggregator 겨냥 적응공격 — 우리 무대는 FedAvg+valuation | 가정·범위 위배 |
| Sybil collusion+single-pixel | C | ✗ | 다중 공모 = FoolsGold가 겨냥하는 E7 위협 | 목적축 불일치 |

---

## 3. 누락·제외 판정 종합

### 3.1 A — 진짜 gap (우선순위 순)

| 우선 | 항목 | 어느 실험군 | 왜 필수인가 | 보강 방법 |
|:---:|------|------|------|------|
| **high** | **Federated-LOO** (U(N)−U(N\{i})) | Fidelity LLM(2A) + CNN C1(2B) | 루브릭이 A-ii로 명시한 분야표준 최저비용 앵커. FedSV·FedTSV·Data Shapley 모두의 **주 baseline**. 코드 확인: 현 roster의 loss-heur는 **singleton U({k})**라 LOO의 grand-coalition 한계기여와 다른 양 → LOO는 **실제로 누락**. exact-Shapley 재현을 헤드라인으로 내걸면서 Shapley의 가장 유명한 저가 근사를 빠뜨리면 cherry-picking 1순위 지적. | `baselines/loo.py` 추가 + track_d/track_c1 `compute_fidelity`에 1행. **N=5/10에서 이미 도는 2ᴺ retrain의 부산물**(U(N), U(N\{i}))이라 한계비용 ≈0. 위치 = '판별용 경쟁자'가 아니라 **'가산성 정량화용 저가 앵커/GT-근사'**(oracle-estimator 혼동 방지). |
| **medium** | clean-oracle 상한 + random-q% 하한 **실행/보고 확정** | Selection→Perf(2D) | D-메인 arm 4종 **설계**엔 있으나(plan §430), 보고된 do-no-harm 결과엔 flirds_w/shapleyfl_w/fedif_w/flirds_sel vs vanilla만 등장. 선택·필터링 주장의 상한/하한이 결과에 실렸는지 확인 필요. | 실행됐으면 gap 없음; 미실행이면 A 승격 → D-메인 arm 실행·보고. |

> **detection·attack 실험군에는 필수(A) 누락 없음.** 위협별 강한 전용탐지기(noisy=FedDQC, FR=FLTrust/
> STD-DAGMM, poison=FLDetector)를 이미 확보해 C7 위계를 cherry-picking 없이 지지하고, 위협 3축
> (데이터오염/model-poison/free-rider)이 완비.

### 3.2 B — 선택 (넣어도/빼도 방어 가능)

| 항목 | 실험군 | 근거 |
|------|------|------|
| **FedOwen** (Owen sampling, ECAI'25) | Fidelity(양쪽 트랙)·Selection | 기존 세트에 없는 별개 estimator class. exact-Shapley 대비 근사 검증까지 하나 CNN-scale·selection 목적. 가족(GTG/FedSV/Banzhaf)이 이미 대표 → **CNN 트랙 리뷰어 커버리지용**. LLM fidelity 헤드라인엔 불요. |
| **FedAttr** (privacy client-attribution, 2026) | Fidelity LLM | F×LLM×client 3/4 일치의 최근접 이웃. φ 산출 가능하면 baseline arm 승격 가능하나 **abstract만 확인·방법 미검증** → 다음 세션 fetch로 셀 일치·φ 산출 여부 확인 후 판정. |
| TMC-Shapley / Group Testing / CGSV / DIG-FL | Fidelity CNN C1 | 재훈련·근사 estimator, gradient-cosine, influence — 계열이 이미 대표됨. retrain 비용 수치화·SV-weighting 대비 원하면 추가 가치. |
| Random(φ floor) / Beta-Shapley / Least Core | Fidelity | sanity floor·semivalue 스펙트럼 보강; 넣으면 그림 깔끔, 빼도 방어. |
| FedProx / CGSV | Selection→Perf | 표준 강건-집계 앵커 / 유일 미포함 SV-weighting arm; E7 비목표라 필수는 아님. |
| sign-flip / advanced-delta / PGD / gradient-noise | Attack | 음의 φ 프로브·stress-test·2nd-order novelty 부각. **sign-flip이 보강 1순위**. |
| FoolsGold | Detection | poison/sybil 전용(wiki 후보); "표준 backdoor 방어 왜 없나" 선점 가능(sybil-collusion 가정 한계 명시 조건). |
| FedProx / FedTSV(CNN채점) | Detection·CNN | 강건-집계 대비 / trajectory-SV 채점 — 선택 보강. |

### 3.3 C — 정당한 제외 + ⚠ 리뷰어 방어 논리

> 아래는 **그대로 리뷰어 방어에 쓸 수 있게** taxonomy 사유를 명시. "왜 이 유명한 방법이 없나"에 대한 답.

**목적축 불일치 (aggregation-first / incentive-fair)**:
- **FedTSV**: trajectory-SV를 **집계 조향 신호**로 씀(truncated adaptive weight α_i가 모델 궤적을 바꿈).
  fidelity 숫자를 exact/2ᴺ oracle 대비 전혀 보고 안 함(정성 3-group 분리만). 우리는 vanilla-FedAvg
  실현 궤적을 **post-hoc** 채점 → 서로 다른 게임. **인용·대비는 필수**(closest cousin: valuation not
  weighting; closed-form not MC; 2nd-order interaction not direction proximity), baseline은 아님.
- **ShapFed(CSSV)**: 헤드라인=weighted-aggregation+개인화+공정(E7/E4). exact-SV 대비는 **heatmap
  정성**만(순위상관 미보고), class-level, utility=val-accuracy, CNN. LLM 생성형으로 전이 시 고정 M-class
  선형헤드 필요 → 이식 불가. 방어: 목적축 불일치 + 모달리티 밖.
- **iPFL / WinFLoRA / KFCA / Truth-Shapley / FedMS / SNOWFL / FRECA / SecSV**: 목적이 incentive/보상
  (E4)·집계품질(E7)·selection·truthfulness·보안(HE)로 우리 비목표. **KFCA·WinFLoRA·FedAttr는 F×LLM×
  client에 가장 근접**하나 rank-fidelity를 exact-2ᴺ 대비로 재지 않음(KFCA=knowledge-free util, MNIST
  한정 exact 비교; WinFLoRA=noise-aware 가중; FedAttr=privacy-attribution) → **오히려 우리 novelty를
  보강하는 positioning 인용**.

**가정·범위 위배**:
- **ComFedSV**(LLM std20 셀): low-rank utility-matrix 완성 가정이 부분참여에서 필요 — 그러나 **CNN
  N=10 full-participation 셀(C1)에서는 완전관측이라 저랭크 완성이 자명 → 그 셀에선 정당 직접경쟁(A)**.
  즉 ComFedSV 판정은 셀 의존: C1=A, LLM 부분참여=주의(코드 partial=False로 처리).
- **Ripple Shapley**: F×in-run은 같으나 단위가 **SAMPLE-level**(이미지)이고 논문 스스로 numerical
  fidelity를 **명시적으로 거부**(task-driven robustness로 평가) → client-level exact-2ᴺ fidelity와 disjoint.
- **SPACE**: F×client×exact-2ᴺ×PCC로 무대 최근접이나 utility=prototype/knowledge-amalgamation(이산
  class 필요) → 생성형 LLM val-loss 게임과 **다른 게임**. 이식하면 permutation-Shapley로 붕괴(=GTG가
  대표). ⚠ 방어: exclusion note를 (틀린) 모달리티 논거가 아니라 **게임-정체성+프로토콜 불일치+GTG-포섭**으로.
- **FedCorr**(detection): cumulative-LID detect→relabel→retrain 다단 파이프라인이 정적·완전참여 가정 —
  우리는 per-round 부분참여(silo5 2/round, device100 10/round). 저자 스스로 '동적참여 오분류' 명시.
  symmetric label-noise·CNN 전용. **정당 제외**.
- **CoAst**: validation-free = 우리 server val-loss utility의 **정반대 전제**. val-loss 정당성 대조점으로만 인용.
- **IRDS(+Adam/Layer-Aware/LM-Arithmetic)**: LLM이나 centralized·sample/source-level → 우리 모(母)방법
  **lineage 인용**(Adam 변형은 SGD-Taylor caveat positioning). same-cell baseline 아님.

**포섭·중복**:
- **FLTrust**: cosine-to-root trust ≈ Flirds-1st(taxonomy 명시 포섭 예). φ 미산출·Byzantine 방어.
- **CGSV / DIG-FL**: gradient-cosine·influence-similarity = Flirds-1st/FedIF에 포섭(단 CGSV는 집계-방향
  대상이라 완전 동치는 아님 → CNN C1선 B).
- **MR/TMR/Group Testing/DPVS**: FL/Shapley 근사 estimator — GTG(이들을 지배·개선)가 대표. **MR@N=10
  exact = FedSV exact 전수와 동치, TMR = MR+round-truncation(GTG에 dominated)** → 새 valuation 객체 없음.
- **VAE/DAGMM/Autoencoder/RONI**: FLDetector/STD-DAGMM의 전신 또는 loss-heur가 대표.

**valuation 아님 (탐지기·집계규칙·공격)**:
- **Krum/Multi-Krum/Trimmed-Mean/Median/RFA/FedProx**: 집계/강건집계 규칙 — φ 순위 미산출.
- **FLDetector/STD-DAGMM/FoolsGold/FedCorr/VAE**: 악성/free-rider 탐지·교정기 = 2차③ **별도 기준군**
  (우리도 그 용도로만 보유). signed φ·oracle GT 없음.
- **Bagdasaryan/Xu-Instructions/DBA/ALIE**: 공격/threat-model = 오염원이지 valuation baseline 아님.

---

## 4. 웹 발견 SOTA 후보 (2024–2026)

> 5회 독립 검색 종합. **결론: 겨냥 셀(F×LLM×client×exact-2ᴺ-fidelity)을 점유한 선행 0건 → novelty 유지.**

| 후보 | 연도/venue | 셀 일치? | 채택 판정 | 이유 |
|------|------|:---:|:---:|------|
| **FedOwen** (Owen sampling) | ECAI'25 (2508.21261) | ✗ (CNN·selection) | **선택(B)** | 별개 estimator class; CNN 트랙 fidelity/비용 커버리지용. LLM 헤드라인엔 불요 |
| **KFCA** (knowledge-free incentive) | 2026 (2605.04747) | 3/4 (F×LLM-LoRA×client) | **불필요, 필수 인용** | 목적=incentive(E4), knowledge-free util = 우리 val-loss fidelity와 정반대 → novelty 방어 인용 |
| **WinFLoRA** | WWW'26 (2602.01126) | 3/4 | 불필요, positioning | 목적=유인/집계(E4/E7); rank-fidelity 미측정 |
| **FedAttr** (privacy client-attribution) | 2026 (2605.06596) | 3/4 | **검토 후 판정(B)** | 최근접 이웃; φ 산출 가능하면 arm 승격 가능 — **방법 미검증, 다음 세션 fetch** |
| **FedIF** (2509.25560) | 2025 | ✗ (CNN·robust-agg) | 이미 보유 | fedif_w arm 출처 확인; fidelity 축 승격 확정만 |
| **In-Run DS for Adam** | 2026 (2602.00329) | ✗ (C·sample) | 필수 인용 | SGD-linear proxy가 Adam서 Pearson≈0.11 붕괴 → 우리 SGD-Taylor caveat 정면 positioning |
| **LM-Arithmetic DPO Shapley** | 2026 (2512.15765) | ✗ (C·source) | related-work | LLM×Shapley지만 non-federated·source-level |
| **CoAst** (validation-free) | MM'24 (2409.02495) | ✗ | 대조 인용 | val-loss oracle 사용 정당성 대조점 |
| **FLCE** (FL 기여추정 벤치마크) | VLDB'24 | ✗ (방법 아님) | 프로토콜 인용 | 방법선정·robustness(replication/label-flip) 준거 |
| **Shapley Volatility in FL** | 2024 (2405.08044) | ✗ (분석) | 인용 권장 | Track C 안정성(xseed ρ≈0) 관찰과 정합 |
| Maverick-Sh/DPVS/Truth-Sh/SecSV/FedMS/SNOWFL/2602.22470·21721(미검증) | 24–26 | ✗ | related-work | selection·효율·truthfulness·보안·공정 축 = 1차 fidelity와 직교 |

**실행 액션**: (1) FedOwen을 CNN 트랙 선택 baseline로 검토, (2) KFCA·In-Run-DS-Adam을 novelty
방어·positioning 필수 인용, (3) FedAttr fetch 후 셀/φ 확인, (4) FedIF fidelity 승격 확정, (5) 그 외 새
헤드라인 baseline 추가 불요.

---

## 5. 적대적 검증 결과 (리뷰어 방어에 유용)

> 18개 후보를 원 판정에 대한 **최강 반론**으로 스트레스 테스트. **결론: 뒤집힌(final≠original) 판정 0건.**
> 즉 모든 원 판정이 적대적 반론 하에서 **유지**됐다. 다만 (i) 일부는 *이유*가 교정됐고, (ii) 일부는
> confidence가 medium이라 정직하게 명시한다.

**핵심 방어 서사 (반론 → 유지 근거)**:

| 후보 | 원판정→최종 | 최강 반론 | 유지 근거(방어 논리) | confidence |
|------|:---:|------|------|:---:|
| **Federated-LOO** (LLM·CNN) | A→**A** | "near-additive 레짐에선 LOO≈Shapley≈loss-heur라 중복 행(포섭 C)" | 가산성은 **측정 대상**이지 baseline 뺄 전제 아님(순환논증). loss-heur=singleton U({k})은 leave-one-out과 formally distinct → 로스터에 LOO 노션은 **실제로 없음**. A-ii 명시 앵커. | 0.8 / high |
| **MR/TMR/GT/DPVS** | C→**C** | "GTG는 넣고 그 계보 MR/TMR은 뺐다=cherry-picking" | A-ii 앵커는 무-잉여 기준선; MR/TMR은 GTG의 ablation(GTG가 지배). MR@N=10 exact=FedSV exact와 동치 → 새 객체 0. GT/DPVS는 exact-oracle 무대선 한계효용 낮음 | mid-high |
| **CGSV** | C→**C** | "FedTSV가 baseline으로 쓰는 gradient-Shapley 앵커인데 뺐다" | 목적=fairness reward(cosine-to-**aggregate**); 우리 fidelity 기준방향은 val-gradient. Flirds-1st+FLTrust가 이미 대표. caveat("다른 대상 측정")가 C 강화 | mid-high |
| **DIG-FL** | C→**C** | "FedIF(agg)가 아니라 valuation-quadrant 직접경쟁" | 반론 타당(카테고리 교정) — 단 올바른 대표자는 **GTG-Shapley**(이미 roster)이지 추가 아님. valuation측·gradient측 **이중 포섭** | mid-high |
| **SPACE** (LLM·CNN) | C→**C** | "exact-2ᴺ×PCC로 우리 축의 SOTA·cherry-picking" | 정량 fidelity 숫자 없음(heatmap/PCC는 prototype 게임). utility=prototype(이산 class), 이식 시 permutation-Shapley로 붕괴(GTG 포섭). ⚠ exclusion note를 게임-정체성 논거로 upgrade | high / mid-high |
| **ShapFed** | C→**C** | "AFedSV/FedIF/Ripple이 공전하는 FL-SV 허브" | Spearman/Kendall-vs-exact 0회(정성 heatmap만). LLM 전이 불가(M-class 헤드). 목적=E7/E4 | high |
| **KFCA** | C→**C** | "F×LLM×client+exact-Shapley 비교까지=최근접 A" | knowledge-free(val-loss·test-set 미사용)=다른 게임(E4). exact 비교는 MNIST·보상분리(rank-fidelity 아님) | **medium** |
| **WinFLoRA/iPFL/FedTSV/…7종** | C→**C** | "FedTSV=validation-signal 있는 최신 fed-Shapley, fidelity표에 왜 없나" | FedTSV는 aggregation accountant(α_i가 궤적 조향), fidelity 숫자 0. 앵커는 FedSV(이미 보유). **인용-대비 의무는 있으나 baseline 아님** | high |
| **SecSV/HESV** | C→**C** | "F×client+approximation-error(fidelity) 보고=직접경쟁" | 위협모델 상충(우리는 서버가 Δw 봄=명시 스코프 밖); LLM-HE 계산 불가; secure-computation는 orthogonal 래퍼 | 0.85 |
| **CoAst** | C→**C** | "2024 top-venue validation-free 기여도 SOTA 누락=cherry" | 비-앵커; 목적축 불일치(validation-free=val-loss 정반대); 모달리티 밖. **대조 baseline B 여지** | mid-high |
| **Fed-LOO(Detection)** | C→**C** | "탐지축 표준 앵커; loss-heur가 대표한다는 이유는 부정확" | 반론 타당(이유 교정: singleton≠marginal). 단 **탐지군 한정**으로는 exact-2ᴺ semivalue+전용탐지기가 포섭. **fidelity군에선 A**(§3.1) | mid-high |
| **FedOwen(Detection)** | C→**C** | "저분산 estimator가 φ-detector로 더 안정적 AUROC" | 탐지군은 전용탐지기 레퍼런스군; FedOwen=valuation estimator로 GTG/FedSV/Banzhaf에 포섭. N=5선 분산감소 무의미 | ~0.72 |
| **Truth-Sh/Overvaluation(Det)** | C→**C** | "Data-Overvaluation이 φ-as-detector 타당성 정면 공격" | truthfulness=mechanism-design(E4 비목표), 우리 게임엔 보고/전략 채널 없음. **원문 미확보로 sight-unseen** | **medium** |
| **Maverick/DPVS/Truth-Sh(C1)** | C→**C** | "FL-SV fidelity 대회서 이 3종만 뺐다" | Maverick=selection+한계-근거 인용(이미 사용), Truth=incentive, DPVS=가속기(GTG 포섭). **DPVS 원문 미확인** | **medium** |
| **FedMS(Selection)** | C→**C** | "selection→perf 축의 검증된 SOTA(FedEMD)" | non-headline·do-no-harm parity라 A(iii) 미발화; clean-IID엔 maverick 신호 없어 FedEMD=random 퇴화; 이미 limitation-근거로 인용 | ~0.78 |

**이유가 교정된 항목(판정은 유지)**: DIG-FL(대표자=FedIF→**GTG-Shapley**), SPACE(exclusion note를
모달리티→**게임-정체성+프로토콜**로), Fed-LOO-Detection(loss-heur 대표→**singleton≠marginal, exact-2ᴺ
semivalue 포섭**).

---

## 6. 한계 (정직한 불확실성)

- **판정 뒤집힘 없음이나 confidence 편차**: 적대적 검증 18건 모두 원 판정 유지됐지만, **KFCA·
  Truth-Shapley/Overvaluation·Maverick/DPVS/Truth(C1)** 는 confidence **medium** — 관련 source 노트·
  원문 PDF 미확보로 sight-unseen 판단(제공 설명 의존). 특히 **DPVS-Shapley**가 GTG류 가속기가 아니라
  독립 fidelity-경쟁 estimator로 판명되면 B/A 재분류 필요(원문 1건 확인 권장).
- **FedAttr(2605.06596)**: abstract만 확인, 방법·backbone 미검증 → 셀 일치·φ 산출 여부 **다음 세션
  fetch** 후 판정. 현재 B(잠정).
- **미검증 웹 리드 2건**: 2602.22470(Beyond Performance-wise Contribution), 2602.21721(Private&Robust
  Contribution) — 초록 미확인. 셀 일치 낮을 것으로 추정하나 단정 보류.
- **clean-oracle/random-q% arm 실행 상태**: D-메인 설계엔 있으나(plan §430) 보고된 do-no-harm 결과엔
  vanilla 대비만 등장 → **실제 실행/보고 미확정**(§3.1 medium). 미실행이면 A 승격.
- **naming collision**: 우리 'FedSV'=per-round Federated Shapley(Wang 2020)이고, 웹의 2502.17526
  (Byzantine-robust 'FedSV')는 **동명이인** — 문서에 출처 명기 권장.
- **Federated-LOO 미구현(→ §1 정정: 07-02 구현 완료, roster 셀 수치만 재실행 대기)**: 감사 시점엔 코드에 baseline 미존재(개념문서 leave-one-out.md만) → §3.1의 A-gap은
  코드 확인으로 검증된 실제 누락.
- **title_matches / 파일명**: 이 감사는 제공된 competitor_extracts(원문 verbatim 근거)에 의존 — 각 추출은
  대부분 confidence High이나, Bagdasaryan(web-extract 부분본)·Xu(web-extract)는 **Experiments/Baselines
  절 부재**로 baseline 완결성 medium-low(공격 논문이라 valuation baseline은 애초에 없음).

---

## 7. 상호 링크

- [[baseline-original-paper-verification-2026-06-22]] — 이미 고른 baseline의 **수치·세팅**이 원 논문과 맞는지(자매 문서, 다른 질문).
- [[prior-work-taxonomy/README]] — 선행연구 분류(무엇을 타겟/어떻게 검증)·2D 교차표·관찰된 빈칸.
- [[prior-work-taxonomy/taxonomy]] — aggregation-first vs valuation-first 사분면(C-제외 근거).
- [[prior-work-taxonomy/validation-experiments]] — E1–E7 검증실험 CNN/LLM 분리표.
- [[flirds-experiment-results-overview-2026-06-25]] — 실험 결과 개요(C7/C8 서사 출처).
- [[flirds]] — 프로젝트 상태·확정 설계·open questions.
