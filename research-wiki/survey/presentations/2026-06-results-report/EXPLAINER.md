# Flirds 실험·지표·방법론 해설 (이해용 노트)

> 발표 데크(`flirds-results-2026-06.pptx`)와 **별개**로, 각 실험이 *어떤 세팅에서 무엇을
> 비교해 무엇을 확인하려는지*, 표의 수치가 *어떤 원리로 매겨진 점수*인지, 그리고 비교에
> 등장하는 *각 방법(baseline)이 어떻게 동작하는지*를 정리한 참고 문서.
> 데크의 새 용어(Image (CNN)·LLM robustness·LLM standard / retrain oracle·in-run oracle 등)와
> 내부 코드명을 함께 적어 rundir·코드와 대조 가능하게 했다.

---

## 0. 큰 그림 — 무엇을, 왜 재는가

**한 줄 목표**: 연합학습(FL)에서 각 클라이언트가 글로벌 모델에 *얼마나 기여했는지*(기여도 φ)를
정확하고 싸게 측정하는 방법(**Flirds**)을 제안하고, 정답(oracle) 대비 검증한다.

**핵심 질문 위계** (모든 실험·표·서술 순서의 기준):

1. **1차 — 기여도를 얼마나 정확히 측정하는가** (가장 근본). 정답 기여도(oracle) 대비
   순위·값이 얼마나 일치하는가 = **fidelity**.
2. **2차 — 측정한 기여도가 실제로 쓸모 있는가** (이 순서대로):
   ① 일반 성능 향상 → ② 수렴 속도 → ③ 오염 클라이언트 탐지.
   *탐지는 마지막* — 기여도 ≠ 탐지다(예: clean 검증손실을 낮추는 공격자는 φ가 "기여 높음"으로
   나오는 게 valuation의 정직한 답).

**듀얼 오라클**: "정답 기여도"를 두 가지 게임으로 정의한다(§2). 둘은 다른 질문이라 갈릴 수 있고,
그 갈림 자체가 발견거리다.

**세 실험의 관계**:

| 실험 (내부코드) | 도메인 | 한 줄 역할 |
|---|---|---|
| **Image (CNN)** (Track C) | 이미지 분류 | 통제된 환경에서 fidelity 듀얼오라클 + 개입 효과를 폭넓게 검증 |
| **LLM robustness** (Phase 2) | 언어모델 (오염·위협 있음) | LLM 스케일·비IID·오염 하에서 fidelity·탐지·스케일 검증 |
| **LLM standard** (Track D) | 언어모델 (정상·IID) | 오염 없는 표준 무대에서 fidelity + 무해성(성능·수렴) 검증 |

---

## 1. 공통 개념

### 1.1 기여도 φ와 `logs` 계약
- 각 클라이언트 k에 스칼라 **φ_k** 를 매긴다(클수록/작을수록의 의미는 방법마다 다름; 저장 시
  **높을수록 의심(=기여 낮음)** 으로 부호 정렬).
- 대부분의 방법은 **from-logs**(사후) 방식: FedAvg 학습을 한 번 돌려 **궤적**을 기록한
  `logs = [(w_r, {client: (Δw_k, n_k)})]` (라운드별 글로벌 가중치 `w_r`, 참여 클라의 업데이트
  `Δw_k`와 표본 수 `n_k`) 위에서 φ를 계산한다. 모델·검증셋을 다시 건드리지 않아 공정 비교가 된다.
- 예외(온라인): **S-FedAvg**(선택을 학습 중에 바꿈), **Ripple**(자체 궤적을 돌림).

### 1.2 두 개의 "정답 기여도"(oracle) — 게임의 차이
정답은 2^N 부분집합을 모두 평가하는 **exact Shapley**로 구한다(N이 작아야 가능: CNN N≤10, LLM N=5).
차이는 **효용 U(S)** 를 무엇으로 두느냐:

- **in-run oracle** *(내부코드 (b); `oracle/in_run_sv.py`)* — **학습 과정** 게임.
  `U(S) = Σ_r [ ℓ(w^r + Σ_{k∈S} p_k^r Δw_k) − ℓ(w^r) ]`.
  *얼어붙은* FedAvg 궤적 위에서, 부분집합 S의 **실제 업데이트**만 각 라운드에 더했을 때 검증손실 ℓ이
  얼마나 줄었나. 순전파만. **Flirds가 추정하려는 바로 그 게임.**
- **retrain oracle** *(내부코드 (a); `oracle/exact_sv.py`, LLM은 `exact_sv_llm.py`)* — **재학습** 게임.
  `U(S) = `클라 S만으로 처음부터 FedAvg 재학습한 모델의 검증손실(−)`. (LLM은 ROUGE-L도 부차적으로,
  단 이건 다른 게임이라 관찰용.) 즉 "이 클라들로 학습한 모델이 얼마나 좋은가."

> **왜 둘이 갈리나(핵심)**: in-run은 "매 라운드 실제 업데이트의 한계 효과", retrain은 "최종 배포
> 모델에서의 가치". 데이터가 동질(IID·clean)이면 두 게임이 거의 일치(LLM standard에서 +1.000 관측).
> 그러나 **라벨 치우침(label_skew)** 같은 이질성에선 갈린다: 희귀 라벨 보유 클라는 매 라운드
> 한계기여(in-run)는 낮아도(다수 그래디언트에 묻힘) 최종 모델 커버리지(retrain)엔 필수.
> → Image 실험에서 Flirds vs in-run +0.98인데 vs retrain −0.18. 이는 Flirds의 오차가 아니라
> **모든 in-run 계열 방법이 공유하는 게임 정의 차이**.

### 1.3 부호·방향 규약
- 저장 φ는 **높을수록 의심(기여 낮음/해로움)**. valuation(좋을수록 큰 값)은 저장 시 부호 반전.
- 탐지 AUROC는 "오염 클라 = 높은 점수"가 정답(1.0). AUROC 0.0 = 완전 역전(오염이 가장 안전하게
  보임 = clean-preserving 공격에 회피당함).

---

## 2. 실험별 상세 — 세팅·비교 대상·확인 목표

### 2.1 Image (CNN) — 이미지 분류 *(내부코드 Track C)*

통제된 소규모 환경에서 **fidelity를 듀얼오라클로 가장 엄격히** 검증하고, 기여도-가중 집계가 성능을
올리는지 본다. 두 하위 실험:

**(A) fidelity** *(내부코드 C1)* — 기여도 측정 정확도
- **세팅**: MNIST(+LeNet5) / CIFAR-10(+소형 CNN), **N=10 전원 참여**, R=10 라운드, local epochs=5,
  SGD(momentum=0) lr=0.01, batch=64, val=2000 / test=8000. **3 seeds**.
- **시나리오 5종**: `iid`(동질) · `label_skew`(라벨 치우침) · `quantity_skew`(데이터 양 치우침) ·
  `label_flip`(라벨 뒤집기 오염) · `feature_noise`(입력 잡음).
- **비교**: 11개 방법(§3)의 φ를 **두 오라클**((b) exact 2¹⁰ in-run, (a) 2¹⁰ retrain)과 대조.
- **확인 목표**: 다양한 이질성/오염에서 각 방법의 순위(Spearman·Kendall) + 값-거리(cosine/euclid/maxdiff)가
  정답과 얼마나 맞나. 특히 ① Flirds가 in-run을 충실히 재현하나 ② 2차항이 값-정밀도를 1차 대비 좁히나
  ③ in-run vs retrain 괴리가 어디서 생기나.

**(B) intervention** *(내부코드 C2)* — 측정한 기여도로 집계에 개입
- **세팅**: CIFAR-10 / FMNIST, **N=100, 라운드당 10% 참여(C=0.1)**, R=120, epochs=5, lr=0.01, batch=64,
  목표 정확도 0.6. 분할 {`iid`·`dir1`(Dirichlet α=1)·`shard`} × 위협 {`clean`·`label_flip`·
  `grad_noise`(그래디언트 잡음)·`free_rider`(0-업데이트)}. 3 seeds.
- **비교(개입 arm)**: `vanilla`(기본 FedAvg) vs 기여도 기반 — `flirds_mult/repl/add/select`(곱셈/대체/덧셈
  가중, 선택[저기여 제거]) · `shapleyfl` · `fedif` · `sfedavg`(S-FedAvg 선택).
- **확인 목표**: ② 일반 성능(test acc 향상) + ③ 탐지(AUROC). 오염 하에서 vanilla보다 나아지나, clean에선
  해치지 않나(무해성), soft(가중) vs hard(선택) 어느 쪽이 나은가.

### 2.2 LLM robustness — 언어모델, 오염·위협 있음 *(내부코드 Phase 2)*

방법론을 **LLM 스케일 + 비IID + 실제 위협**으로 가져가 fidelity·탐지·스케일을 본다. 두 레짐:

**cross-silo (N=5)** — 기관(도메인)별 사일로
- **세팅**: Llama-3.2-1B-Instruct (+ 3B), 클라 5개 = **도메인 1개씩**(의료·법률·금융·수학·일반),
  R=10, val=100, LoRA, fp32, eager attention, SGD(mom=0) lr=1e-3(poison만 2e-3). **3 seeds**(3B는 1 seed).
- **위협 4종**: `noisy`(answer_swap=클라 내 응답 뒤섞기, 데이터 품질) · `freerider_zero`(0 업데이트) ·
  `freerider_random`(무작위 업데이트) · `poison`(백도어: 명령어-트리거→타깃 + 스케일 모델교체).
- **비교**: 11 valuation + 4 detector(§3)를 **in-run oracle (b) exact 2⁵** 와 대조 + 탐지 AUROC.
- **확인 목표**: ① LLM에서도 fidelity 유지되나(N=5는 near-additive라 대부분 동률 예상) ③ 각 위협을
  누가 탐지하나, 그리고 **poison이 Flirds의 경계인가**(clean 검증손실을 낮추는 백도어).

**cross-device (N=100)** — 다수 기기
- **세팅**: 1B, 클라 100개, **라운드당 10개 참여(K=10)**, R=30, per-client **Dirichlet(α)** 도메인 혼합,
  **α-sweep** α∈{0, 0.01, 0.1, 0.5, 5.0}(0=도메인 분리, 클수록 균질).
- **비교**: α=0.5 한 지점에서만 **in-run oracle (b) per-round exact**(비용상 1지점), 나머지 α는 **Flirds를
  근사 정답(proxy)** 으로 두고 싼 방법들과 탐지 AUROC.
- **확인 목표**: 스케일·비IID에서 순위 유지와 탐지가 어떻게 변하나(α에 따라 noisy 탐지 난이도 변화 등).

### 2.3 LLM standard — 언어모델, 정상·IID *(내부코드 Track D)*

**오염 축을 전부 제거한** 표준 무대(OpenFedLLM 레시피)에서 fidelity + **무해성(성능·수렴)** 을 본다.
clean·IID라 개입은 "해치지 않으면 성공"(do-no-harm parity)이 기대값.

**standard (N=20)** — 주 무대
- **세팅**: Llama-3.2-1B / alpaca-gpt4 20k **IID**, 클라 20개 **라운드당 2개**, R=200, 라운드당 10 step ×
  batch16, seq512, SGD(mom=0) lr=1e-3, LoRA r16 fp32, val=200 / test=1000. **3 seeds**.
  (OpenFedLLM `run_sft.sh`를 그대로 따르되 valuation 규약상 SGD mom=0/lr 상수.)
- **비교**: fidelity 8개 방법을 **in-run oracle (b) per-round exact** 와 대조 + 개입 arm
  {`base`(학습 전)·`vanilla`·`flirds_w`(×β.5)·`flirds_sel`·`shapleyfl_w`·`fedif_w`} → MMLU·ROUGE-L·수렴.
- **확인 목표**: ① 정상 LLM에서 fidelity(특히 거리 지표로 2차항 효과) ② 가중 개입이 MMLU·ROUGE를
  해치지 않나 ③ 수렴(목표 도달 라운드)이 동일한가.

**reference (N=5)** — 소규모 듀얼오라클 검증
- **세팅**: 같은 무대, 클라 5개 **전원 참여**, R=30. **(a) 2⁵ retrain + (b) in-run + Banzhaf** 듀얼오라클.
- **확인 목표**: 정상·IID에서 **retrain oracle (a) 와도 +1.000 일치**하는지 → §1.2의 "괴리는 이질성 구동"을
  대조 확인(여기선 동질이라 일치).

---

## 3. 비교 방법(baseline) 해설 — 유형과 동작 원리

### 3.1 유형 분류 (먼저 큰 틀)

| 유형 | 무엇을 출력 | 어떻게 | 비용 | 이 프로젝트의 예 |
|---|---|---|---|---|
| **Oracle (정답)** | exact Shapley φ | 2^N 부분집합 효용 전수 계산 | 매우 큼(2^N) | in-run (b), retrain (a) |
| **Shapley 계열 (semivalue)** | 기여도 φ | 부분집합 한계기여의 (가중)평균 | 중~큼 | GTG·FedSV·Banzhaf·ShapleyFL·ComFedSV |
| **그래디언트/영향함수** | 기여도 φ | 업데이트와 검증손실-감소 방향의 정렬(1차) | 작음 | FedIF·Flirds-1st·loss-heur |
| **테일러 전개 (제안)** | 기여도 φ | in-run Shapley를 1+2차 테일러로 근사 | 작음(HVP 1회/R) | **Flirds** |
| **2차 곡률(고유분해)** | 기여도 φ | 1차 drop + Hessian 고유쌍 전파 | 큼(eigsh) | Ripple |
| **온라인 선택** | 선택 확률 | φ로 매 라운드 샘플링을 조정 | 중 | S-FedAvg |
| **탐지기 (anomaly)** | 의심 점수(φ 아님) | 이상치 점수로 오염 클라 flag (AUROC만) | 작음 | FLDetector·STD-DAGMM·FLTrust·FedDQC |

**축으로 보는 구분**:
- **Shapley vs semivalue**: 부분집합 S에 클라 i를 더했을 때의 한계기여 `U(S∪i)−U(S)`를 *어떤 가중치로*
  평균내나. Shapley=집합 크기별 가중, **Banzhaf**=균일(1/2^{n-1}).
- **효용 U(S) 정의**: (b)오라클·Flirds·Banzhaf = n_k 비례 + 실제 궤적(궤적-충실). GTG/FedSV = 부분집합 내
  n_k 재정규화 submodel. ShapleyFL/ComFedSV = **균일 평균** submodel. → 0-업데이트 무임승차 클라의 φ가
  정확히 0이 되느냐(전자) 희석되느냐(후자)가 갈린다.
- **valuation vs detector**: valuation은 φ(순위·Spearman + AUROC 둘 다), detector는 의심 점수만(AUROC만,
  Spearman 없음).

### 3.2 정답(oracle)
- **in-run oracle** (내부 (b)) — §1.2. *얼린 궤적의 실제 업데이트* 한계기여, exact 2^N. Flirds의 타깃.
- **retrain oracle** (내부 (a)) — §1.2. *부분집합으로 재학습한 모델*의 효용, exact 2^N. (a)와 (b)는 별도
  코드 경로(효용 계산을 공유하지 않음 — 프로토콜 §4.3).

### 3.3 제안 방법
- **Flirds** *(`core/flirds_estimator.py`)* — in-run Shapley를 검증손실의 **1차+2차 테일러**로 닫힌형 근사:
  `φ_k = Σ_r p_k^r [ ⟨g^r, Δw_k⟩ + ½⟨Δw_k, u^r⟩ ]`, `u^r = H^r ΔW^r`.
  `g^r`=검증손실 그래디언트, `H^r`=**참 Hessian**(IRDS와 동일), `ΔW^r`=라운드 집계 업데이트.
  2차항의 quadratic-Shapley가 **라운드당 HVP(헤시안-벡터 곱) 1회**로 붕괴 → exact 2^N 없이 coalition급
  값을 얻는 게 핵심. (전진모드 AD가 필요해 LLM은 eager attention 필수.)
- **Flirds-1st** *(`second_order=False`)* — 위 식의 **1차항만**(⟨g^r, Δw_k⟩). 곡률 제외 = 2차항 효과를
  보는 자기-절제(ablation) 기준선. FedIF와 사실상 같은 1차 영향.

### 3.4 단순 휴리스틱
- **loss-heur** — 각 클라의 **단독(singleton) in-run 효용** `U_(b)({k})` = 그 클라 업데이트만 매 라운드
  더했을 때의 검증손실 변화. 부분집합 평균도 테일러도 없는 가장 단순한 "이 클라 혼자 손실을 얼마나
  바꾸나". 순위는 의외로 강하지만(값 부호가 같아서) **값-크기(euclid)는 크게 빗나감** → 거리 지표에서
  Flirds와 갈린다.

### 3.5 Shapley 계열 (semivalue)
- **GTG-Shapley** (Liu 2022, TIST) — *Guided Truncation* 몬테카를로 Shapley. 라운드별 계산 후 합산.
  submodel 효용 = `eval(직전 글로벌 + 부분집합 deltas의 FedAvg 집계)`(궤적-충실, 재학습 아님). MC 샘플링 +
  truncation으로 비용 절감 → 그래서 노이즈가 있고 순위가 흔들림.
- **FedSV** (Wang 2020) — 연합 Shapley의 원조. 참여 cohort에 대한 **순열 몬테카를로** Shapley를 라운드마다,
  합산. submodel 효용은 GTG와 동일. 차이는 추정기(순열-MC, +옵션 TMC truncation)와 옵션 per-round
  정규화("normalized FedSV", 탐지에 유리).
- **Banzhaf** (Wang & Jia 2023) — Shapley와 같은 semivalue지만 **균일 가중**(1/2^{n-1}). (b)오라클의 exact
  2^N coalition 효용을 **그대로 재가중** → 같은 궤적·효용 위의 **정확한 Banzhaf 값**. semivalue 선택만
  분리해 보는 셈(우리 효용은 결정적이라 순위가 거의 안 변하는 게 관찰점).
- **ShapleyFL / AFedSV** (Sun 2023, KDD) — 대리(surrogate) 연합 SV. 라운드 효용이 **균일 평균** submodel
  (n_k 가중 아님), 라운드별 exact Shapley(N≤10) 후 **min-max 정규화 + EMA**(라운드 간 누적). 효용 가중이
  달라 (b)오라클과 미세하게 다른 순위.
- **ComFedSV** (Fan 2022) — **부분 참여**에서 관측 안 된 coalition 효용을 **저랭크 행렬 완성**(ALS)으로 채워
  Shapley를 읽음. 균일 submodel, 효용=라운드별 test-loss 감소. 저R/소규모에선 완성이 불안정(NaN/저상관)할
  수 있음(기지 특성).

### 3.6 그래디언트/영향함수
- **FedIF** (Tang 2025) — 영향함수 기반 클라 가치. `Φ_k = −⟨g^r, Δw_k⟩ / ‖Δw_k‖`(검증 그래디언트와 클라
  업데이트의 정렬, 크기 정규화), per-round min-max→[0,1], EMA 누적. **Flirds의 가장 가까운 경쟁자**
  (그래디언트-영향, Shapley보다 ~450× 싸고 노이즈에 강함 — 같은 세일즈). 스케일(3B)에서 순위가 떨어지는
  경향.

### 3.7 2차 곡률(고유분해)
- **Ripple** (Zeng 2026, AAAI) — `φ = drop + ripple`(라운드 합). *drop*=실제 로컬 SGD 궤적을 따라간 IRDS
  1차항. *ripple*=각 클라가 로컬 Hessian의 상위-k 고유쌍을 스케치 → 진행형 글로벌 부분공간으로 라운드 간
  업데이트 방향을 전파(저랭크 야코비안). **자체 궤적**을 돌리고 eigsh가 비싸 가장 느림. 정답-SV 지표가
  없어(논문이 MSE/상관 거부) 탐지 AUROC로만 평가.

### 3.8 온라인 선택
- **S-FedAvg / Game of Gradients** (Nagalapatti 2021, AAAI) — **본질적으로 온라인**: 관련도 φ가 매 라운드
  샘플링을 좌우(softmax(φ) 비복원 추출), 선택 cohort에 대해 MC Shapley(정확도 효용). from-logs가 아니라
  FL 루프 안에서 동작 → Image intervention(C2)의 **선택 기준선**.

### 3.9 탐지기 (AUROC만; φ·Spearman 없음 — 위협별로 매칭)
- **FLDetector** (Zhang 2022, KDD) — 서버측 **모델-free** from-logs. 시간적 업데이트 일관성: 양성 클라는
  `g_i^t ≈ g_i^{t-1} + H^t(w^t−w^{t-1})`(Cauchy 평균값정리). 서버가 각 클라 업데이트를 *자기 직전 업데이트 +
  공유 HVP*(Hessian은 글로벌 궤적 secant쌍에 L-BFGS Byrd-Nocedal)로 예측, **반복적으로 예측에서 먼** 클라를
  flag. → *조작된 업데이트* 공격자 탐지용(noisy-but-honest엔 약함).
- **STD-DAGMM** (Lin 2019) — 서버측 모델-free **무임승차 탐지**. DAGMM(딥 오토인코더+GMM)을 업데이트 벡터
  풀에 학습 + **업데이트의 표준편차**를 추가 피처로. GMM 에너지=이상 점수. "모델-free"=FL 모델은 안 쓰지만
  자체 이상모델은 학습. 작은 N/순수 회피에선 약함.
- **FLTrust** (Cao 2021, NDSS) — 서버측 **신뢰-방향** 탐지. 서버의 clean root(=우리 val셋)로 `g0 = −∇_val`를
  구해 각 업데이트를 `cos(Δw_i, −∇_val)`로 점수(크기 정규화). 사실상 **정규화된 Flirds-1st 방향**. 신뢰
  방향과 어긋난 업데이트 탐지(무임승차·스케일 공격에 강함).
- **FedDQC** (Du 2024) — 온디바이스 **데이터 품질** 탐지, IRA(명령-응답 정렬)로. `IRA(q,a) = L(a) − L(a|q)`
  = 명령 q가 주어졌을 때 응답 a의 손실이 얼마나 떨어지나(상호정보 느낌). clean 샘플은 IRA 높음(명령이 답을
  "설명"), answer_swap 잡음 샘플은 q-a가 어긋나 IRA 낮음 → 의심=−IRA. 클라 데이터+모델 필요. **데이터 품질
  위협(noisy)에 매칭**된 기준선.

> **탐지기를 굳이 두는 이유**: valuation은 라벨에 blind이고 라벨(누가 오염인지)은 *평가 KEY*일 뿐(순환
> 아님). 두 목적 — ① 값의 의미 검증(오염→낮은 가치인가) ② 전용 탐지기 대비 경쟁력. 위협마다 맞는 전용
> 탐지기가 다름(품질→FedDQC, 무임승차→STD-DAGMM/FLTrust, 백도어→FLDetector/FLTrust).

---

## 4. 지표 해설 — 수치를 어떤 원리로 읽나

> 방향: **↑** 높을수록 좋음 · **↓** 낮을수록 좋음. (데크 표에도 이 화살표를 병기.)
> fidelity 지표는 모두 *방법 φ 벡터 vs 오라클 φ 벡터*를 비교한다. 코드: `flirds/eval/metrics.py`,
> 순위 상관은 러너에서 scipy.

### 4.1 fidelity — 순위 (방법이 클라를 같은 *순서*로 매기나)
- **Spearman ρ** *(↑, 범위 −1~+1)* — 두 φ를 **순위로 바꾼 뒤** 상관. +1=완전히 같은 순서, 0=무관, −1=역순.
  값의 크기는 무시하고 **순서만** 본다. 가장 기본 fidelity.
- **Kendall τ** *(↑, −1~+1)* — 모든 클라 쌍 중 **일치쌍−불일치쌍 / 전체**. Spearman보다 동률·작은 교란에
  덜 민감. 순위 일치의 보수적 버전.

### 4.2 fidelity — 값-거리 (순서뿐 아니라 *값 자체*가 맞나; 순위가 +1.000으로 포화될 때 변별)
- **cosine_d** *(↓, 0~2)* — `1 − cos(φ_방법, φ_오라클)`. 0=두 벡터 **방향**이 같음. 크기에 무관(스케일 불변),
  값의 **방향**만. (어느 벡터든 노름 0이면 정의 안 됨→NaN.)
- **euclid_d** *(↓, ≥0)* — `‖φ_방법 − φ_오라클‖₂`. 0=값까지 동일. 값의 **절대 크기 오차**(단위 민감 → 같은
  게임·같은 단위의 추정끼리만 의미). **여기서 Flirds의 2차항이 1차 대비 오차를 ~절반으로** 줄이는 게 보임;
  loss-heur는 순위는 비겨도 이 값이 2~7× 큼.
- **max_diff** *(↓, ≥0)* — `max_k |φ_방법,k − φ_오라클,k|`. **최악의 단일 클라** 값 오차.

### 4.3 2차 — ① 일반 성능
- **test acc** *(↑, 이미지)* — 개입 후 배포 글로벌 모델의 테스트 분류 정확도(top-1).
- **MMLU** *(↑, LLM)* — MMLU 벤치마크 **0-shot 객관식 정확도**(A–E 정답 글자 exact-match; `eval.metrics`의
  `choice_match`). 일반 지식.
- **ROUGE-L** *(↑, 0~1, LLM)* — 생성 응답 vs 정답의 **단어 단위 LCS(최장공통부분수열) F1**(β=1). Alpaca-test에서
  과제 적합도.

### 4.4 2차 — ② 수렴
- **rounds-to-target** *(↓)* — 목표 검증손실에 **처음 도달한 라운드 수**(작을수록 빠른 수렴; 못 도달 시 표기
  없음). LLM standard에서 전 arm ~199/200 = 동일(parity).
- **final val-loss** *(↓)* — 학습 종료 시 검증셋 completion-only 교차엔트로피.

### 4.5 2차 — ③ 탐지
- **AUROC** *(↑, 0~1; `roc_auc_score(corrupt, φ)`)* — "오염 클라 = 높은 φ"로 봤을 때 ROC 곡선 아래 면적.
  의미: 무작위로 고른 오염 클라가 clean 클라보다 더 의심받을 확률. **1.0**=완벽 분리, **0.5**=무작위,
  **0.0**=완전 역전(오염이 가장 안전하게 보임 = clean-preserving 공격에 회피). 라벨은 우리가 주입한
  평가 KEY(방법 입력 아님).

### 4.6 보조
- **runtime** *(↓, 초)* — 그 valuation을 계산하는 데 걸린 벽시계 시간. *품질이 아니라 비용 축* — Flirds의
  핵심 세일즈(같은 순위를 5~15× 싸게).
- **ASR (공격 성공률)** *(poison 맥락)* — 트리거 입력이 백도어 타깃을 내는 비율. "좋다/나쁘다"가 아니라
  **백도어가 실제로 설치됐는지** 확인하는 맥락 수치(ASR≈1.0이라야 그 칸의 탐지 결과가 의미 있음).

---

## 5. 자주 헷갈리는 점

- **Spearman은 +1.000인데 왜 값-거리(euclid)를 또 보나?** 순위가 포화(여러 방법이 +1.000)돼 변별이 안 될
  때, 값의 *크기*가 정답에 얼마나 가까운지가 갈린다. Flirds 2차항의 고유 기여는 순위가 아니라 **값-정밀도**.
- **AUROC=1.0(탐지 완벽)인데 Spearman은 왜 낮을 수 있나?** AUROC는 "오염/clean 두 그룹 분리"만, Spearman은
  "전체 순서"를 본다. 오염은 잘 가려도 clean들 사이 순서가 틀리면 Spearman은 낮아진다.
- **기여도와 탐지는 같은 건가?** 아니다. clean 검증손실을 *낮추는* 공격자(clean-preserving 백도어)는
  valuation상 "기여 높음"이 정직한 답 → φ로는 회피된다(AUROC 0.0). 그 지점이 valuation 접근의 **경계**이고,
  전용 탐지기(loss-heur·FLDetector 등)나 2차항이 보완.
- **왜 (a)와 (b) 오라클이 갈리나?** §1.2 — 게임 정의(과정 가치 vs 재학습 가치)가 달라서. 동질이면 일치,
  이질(label_skew)이면 갈림. Flirds만의 문제가 아니라 모든 in-run 계열 공통.
- **"모델-free 탐지기"가 모델이 정말 필요 없나?** FL 모델·검증셋은 안 쓰지만(FLDetector·STD-DAGMM),
  STD-DAGMM은 자체 오토인코더+GMM을 *학습*한다("free of the FL model, not free of a learned anomaly model").

---

*출처: 코드 `flirds/{core,oracle,baselines,eval}/*.py` docstring·구현 + `runs/*/RESULTS*` 검증 수치.
방법 인용은 각 baseline 모듈 docstring의 논문 표기를 따름.*
