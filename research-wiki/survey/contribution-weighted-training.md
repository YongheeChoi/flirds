---
type: thread
title: 기여도-가중 training — 선행 조사 + 공백 좌표 + 신규 방법 제안
created: 2026-07-03
updated: 2026-07-19
sources: [shapleyfl, game-of-gradients-sfedavg, fedif, fedtsv, shapfed, fltrust, foolsgold, principled-federated-data-valuation, comfedsv, gtg-shapley, ripple-shapley, space-participant-amalgamation, feddqc, fedhds, data-banzhaf, mavericks-shapley-fl, dice, mates, less, dsdm, in-run-data-shapley]
tags: [contribution-weighted-training, intervention, aggregation-weights, closed-loop, survey, proposal]
---

# 기여도-가중 training — 조사 + 공백 + 제안

**질문**: FL에서 클라이언트 기여도 φ를 *training에 되먹이는* 법의 선행 지형은 어떠하고, 우리의 4개 naive arm(`codes/flirds/fl/intervene.py`) 대비 문헌의 공백은 어디이며, 그 공백을 메우는 새 방법은 무엇인가.

**위계 주의**: 이 축은 핵심 질문 위계의 **2차(실효성)** — 서술 순서는 성능 → 수렴 → 탐지. fidelity(1차)는 이미 검증됨(값-수준 Pearson 0.9999+ vs (b), benign 전 스케일·전 레짐(poison에선 붕괴 — review R4) — [[flirds-signal-size-diagnosis]] §1.3(c)).

**방법론**: 위키 리더 4 + 웹 조사 5(테마별) + 초안 메커니즘에 대한 적대적 novelty 공격 3 = 12-agent 병렬 조사(2026-07-03). 웹 항목은 원 논문 abstract/PDF 대조 검증(검증 깊이 플래그는 각주). 상세 원자료: 세션 스크래치 `w7_parts/`.

---

## 1. 현재 4 arm과 공통 기계의 한계

4개 arm 전부 하나의 기계를 공유: **라운드별 raw φ → 참여자 min-max → EMA(β) → s∈[0,1] → 정적 대수 매핑** (`intervene.py`; flirds_w `w∝n·s` β.5 / flirds_sel softmax(s/T) / shapleyfl_w `w∝s` β.3 / fedif_w `w∝s` β.7).

조사 결과 이 기계는 문헌을 정확히 대표한다 — [[sources/fedif|FedIF]] Eq 7–9가 문자 그대로 이 파이프라인이고, [[sources/shapleyfl|ShapleyFL]] Def 4.2–4.3–Eq 4가 arm 3의 원형이다. 한계 4가지(브리프 §3)에 조사가 더한 것:

1. **min-max가 φ의 크기 정보(val-loss 단위)와 φ=0 앵커를 파괴한다.** Flirds의 free-rider φ=exact-0 성질은 min-max를 통과하면 소실된다([[sources/fedtsv|FedTSV]]의 max{0,·} truncation은 보존 — 5번째 정규화 arm 후보). 최하위 참여자는 IID-clean에서도 항상 s=0으로 눌린다 — **신호가 없어도 tilt가 계속되는** 구조.
2. [[sources/principled-federated-data-valuation|FedSV]]의 norm-정규화 변형이 group rationality+additivity를 깨는 것으로 증명됨 — "운용을 위한 φ 정규화는 공리를 비용으로 치른다"는 정식 인용처. 우리 min-max+EMA도 이 트레이드를 (암묵적으로) 지불 중.
3. **개루프**: 문헌 전체에서 집계-측 방법 중 자기 개입의 실측 효과를 되먹이는 것은 사실상 없음(§3-G2).
4. **어느 것도 무엇을 최적화하는지 말하지 않음**: 유도된(objective-derived) 매핑은 [[concepts/federated-shapley|federated SV]] 계열 전체에 부재.

## 2. 선행 조사표 — φ→training 소비자 지형

### 2-A. 기여도/valuation 점수 → 집계 가중 (우리 arm 1/3/4의 계보)

| 방법 [venue] | 신호 φ | 매핑 (핵심 규칙) | 루프 | 차수 | 스케일 |
|---|---|---|---|---|---|
| [[sources/shapleyfl\|ShapleyFL/AFedSV]] [KDD'23] | 라운드별 부분 SV(0차 재구성 utility) | min-max→EMA(β)→`w∝s` 대체 + **1/γ 역샘플링 보정**(Eq 5) + **분산-최적 샘플링** p∝w^{2/3}(Eq 8, Thm 5.1) | 개 | 0차 | CIFAR CNN |
| [[sources/fedif\|FedIF]] [arXiv'25] | 정규화 1차 influence = 정규화 a_i | min-max→EMA(γ.3)→`w∝s` (Thm 1: noisy 하향 1-step bound) | 개 | 1차 | CIFAR N=100 |
| [[sources/fedtsv\|FedTSV]] [ECC'26] | 궤적-SV(거리 utility, 서버 K-step SGD 재생) | **누적합 + α=max{0,φ}** (ReLU 소프트 dismissal; 수렴증명 없음) | 개 | 0차 | MNIST(MLP)/CIFAR(ResNet-20) |
| [[sources/shapfed\|ShapFed-WA]] [IJCAI'24] | 클래스별 last-layer cosine SV | affine [0,1]→모멘텀→`w∝γ̃` + **기여-비례 배포** w̄_i=γ_i w_s+(1−γ_i)w_i | 개 | 0차 | CNN N=4–6 |
| FedCE [CVPR'23, 2303.16520] | LOO형 이중 신호(gradient+data space) | 누적합 정규화→`w∝ρ` 대체 (Thm 3.3 수렴) | 개 | 1차 | 의료 U-Net N=6 |
| CGSV [NeurIPS'21] / RFFL ['20] | cosine-gradient SV(검증셋 無) | EMA 평판→집계가중 + **희소화 보상 다운로드**(tanh 쿼터) / 평판<1/(3N) 영구 축출 | 개 | 1차 | MNIST/CIFAR N≤20 |
| [[sources/game-of-gradients-sfedavg\|S-FedAvg]] [AAAI'21] | 라운드별 MC-SV(0차 utility) | EMA(.75/.25, **min-max 없음**)→softmax **샘플링**; 집계는 균등 | 개 | 0차 | MNIST N=10 |
| [[sources/space-participant-amalgamation\|SPACE]] [NeurIPS'23] | 단일라운드 SV(프로토타입 utility) | 응용데모: 기여도 가중 치환(static/dynamic) + 선택점수 **(1−β)p_i+βc_i**(β는 비IID일수록↑) | 개 | 0차 | CIFAR N≤100 |
| IIoT-Shapley [Sensors'25] | 양자화 SV + 크기 + 변산도 | 합성 가중(정확식 abstract 미기재) | 개 | 0차 | IIoT 소형 |

### 2-B. 최적/학습된 집계 가중 (가중을 "어떻게 얻는가" 축)

| 방법 [venue] | 가중 획득법 | 목적 | 루프 | 비고 |
|---|---|---|---|---|
| FedLAW [ICML'23, 2302.10911] | 서버 proxy 셋 위 GD로 (γ, λ) 학습 | proxy loss min | 폐 | **γ<1 shrinking** 발견 = Σw=1 완화 노브 |
| Auto-FedAvg [2104.10195] | softmax/Dirichlet 파라미터를 클라 데이터로 교대 학습 | 클라 train loss | 폐 | 의료 N=3 |
| FedHAW [2605.00458, '26-05] | **train-loss hypergradient**를 유한차분 근사, 온라인 GD | train loss | 폐 | dL/dλ_k ≈ 우리 a_k의 crude 버전; 2차 없음, K=10 비전 |
| SmartFL [2211.05554] | 클라 조합계수를 proxy ~100샘플 SGD로 직접 최적화 | proxy loss | 폐 | **같은 문제, 다른 solver** — P2가 이겨야 할 상대 |
| FedFomo [ICLR'21] | **val loss 1차 Taylor를 유한차분으로 실현**(클라측·개인화) | 로컬 val | 폐 | **유일한 "Taylor에서 유도" 선례** — 1차·FD·personalized |
| FedAdp [IEEE TCCN'21†] | `w∝n·exp(Gompertz(angle))` — 각도 휴리스틱 | — | 개 | **n·exp(score) 형태의 FL 선점자**(고정 α) †venue 보고 상충(TWC 표기도) — 인용 전 확인 |
| FedDisco [ICML'23] | ReLU(n−a·d+b) — 라벨분포 거리, 이론 bound에서 폐형 | bound min | 개 | round-0 고정 |
| FedAWA [2503.15842] | 업데이트-기하 정렬로 라운드별 최적화(데이터 無) | 기하 | 개 | |
| FedMGDA+ [2006.11489] | **simplex QP + FedAvg ε-앵커**, 1차 Gram | Pareto 공정성 | 개 | P2와 제약기하 동일·목적 상이 |
| FedNova/FedOpt/FedExP/FedSLS | 정규화·서버 옵티마이저·step-size | — | — | 가중 직교 축; FedSLS의 line search는 부록 B의 폐형 η*가 대체 가능 |
| FedAU [ICLR'24] | 참여확률 1/p_n 온라인 추정 → 탈편향 가중 | 탈편향 | — | cross-device에서 기여도 가중과 **곱으로 합성**해야(교란 방지) |

### 2-C. 검증-유도 필터/수술/부호가중 (robust 계보; P3의 이웃)

| 방법 [venue] | 신호 | 소비 | 비고 |
|---|---|---|---|
| Zeno [ICML'19] | **0차** 실측 descent score f(x)−f(x−γu)−ρ‖u‖² (val 배치) | 상위 m−b **하드 드롭** | ρ‖u‖² 크기 페널티 |
| Zeno++ [ICML'20] | **1차** γ⟨g_val,g⟩−ρ‖g‖² (stale val grad) | 임계 **게이트**(수락/기각) + norm-매칭 | **우리 a_i와 같은 신호를 스칼라 게이트로만** 소비 |
| [[sources/fltrust\|FLTrust]] [NDSS'21] | ReLU(cos(Δ_i, 서버 root 업데이트)) | 대체 가중 + **norm 재정규화** | root=val이면 cos = 정규화 Flirds-1st (우리 구현 동일성) |
| ByGARS [2006.13421] | **q = EMA of ⟨Δ_i, g_aux⟩ = EMA of a_i** | **부호있는 비정규화 가중**(음수 = 해로운 업데이트 뒤집어 활용) | 'a_i를 가중으로' 최근접 선례 — meta-learned, 8-worker CNN |
| GAAvernor [USENIX'20] | RL(보상 = quasi-val loss 강하) | RL이 가중 학습 | 폐루프-RL 선례(고비용) |
| Sageflow [NeurIPS'21] | 엔트로피 + public loss | 엔트로피 게이트 + loss^−δ 가중 | |
| FedGreed [2508.18060] | 신뢰셋 loss 실측 | greedy prefix-coalition 선택(라운드 내 폐루프) | O(N) 평가/라운드 |
| FedVG [2602.21399] | val-grad **norm**(방향 버림) | 역norm 가중 | 같은 원료의 정보 낭비 — 대조 ablation감 |
| FedLAW(ICLR'26, 2511.03529) | bilevel + 희소 capped simplex | 가중+선택 동시 학습 | 폐루프 종점(MLP 스케일) |
| **수술 계보**: A-GEM [ICLR'19] → Fed-A-GEM [2409.01585] | 참조 grad와의 충돌 성분 rank-1 제거 | **동일 연산자, 참조=replay buffer**(망각 방지, 클라측) | P3와 식 동형·앵커 상이 |
| FedFV [IJCAI'21] / FedGH / GCFL | **클라-쌍** cosine 충돌 | PCGrad형 상호 투영 | **전부 peer 참조 — val 참조는 부재**(확인) |
| ATTITTUD [ICLR'21] | 주태스크 loss 기준 aux-gradient 분해 | 해로운 성분 제거·유지 재가중 | 非FL; P3 프레이밍의 선점자 — 인용 필수 |

### 2-D. 폐루프/밴딧/메타 (D 방향의 이웃)

- **클라 선택 밴딧(누구를)**: FAVOR [INFOCOM'20 DDQN], GreedyFed ['24 누적 GTG-SV greedy], FedOwen [ECAI'25 ε-greedy+optimism], MAB-RFL, [[sources/mavericks-shapley-fl|FedEMD]](시간-스케줄 softmax) — **혼잡한 계보**.
- **집계규칙 자체의 제어(어떻게)**: FedLAW(proxy GD)·FedHAW(train-loss hypergrad)·FedMABA [2410.20141](가중 위 adversarial bandit — **공정성용, 부호 반대**)·GAAvernor(RL)·**FedAOT [2603.16846]**(held-out val loss의 meta-gradient로 클라 가중 갱신 — 유일한 val-피드백 가중 학습; abstract-수준 검증, Byzantine·sub-LLM 추정. 그 meta-gradient ∂L_val/∂p_i가 바로 우리 a_i) — 소수·최근·전부 기여도 의미론 없음.
- **개입 강도의 실측-적응**: 없음. FedEx [NeurIPS'21]가 노브-위-EG의 템플릿(하이퍼파라미터용). strict-sense 폐루프(자기 결정의 실측 효과 재측정→규칙 갱신)는 드묾 — **MATES**([[sources/mates]], centralized)와 FedAOT(FL측, 위) 정도.
- **centralized 데이터 reweighting 조상**: L2RW [ICML'18](w̃_i=max(0,⟨g_val,g_i⟩) — 우리 a_i의 per-example 특수화), DVRL [ICML'20], **DoGE [ICML'24, 2310.15393]** — **entropic mirror-descent로 도메인 가중 α∝α·exp(η·W/μ) 유도**(1차 명시적, μ 고정, LM 데이터 혼합) = P1과 수학 골격 동일한 최근접 선행.

### 2-E. LLM/PEFT 스케일 (공백 검증)

적극 반박 시도(12+ 검색, 20+ fetch) 후에도 **클라-레벨 기여도 점수를 training에 소비하는 LLM/PEFT 선행 = 0건**:

- [[sources/feddqc|FedDQC]] [ACL'25F]: **데이터**-품질(IRA) → 필터+커리큘럼; 집계 불변. DataInf-선택 < random(Fed-WildChat) 경고 포함.
- [[sources/fedhds|FedHDS]] [ACL'25F]: 대표성 코어셋 **선택**(데이터-레벨).
- 2403.04529: LLaMA2-7B LoRA FL에서 **per-sample** influence(2차 DataInf 포함!)로 일회성 필터 — 신호는 스케일에 존재, **클라-레벨 소비만 부재**.
- WinFLoRA [WWW'26]: TinyLlama/GPT2-L LoRA 적응 가중 — **DP-노이즈** 기준(비기여도); 본문에서 "Shapley는 재학습 필요해 LLM 불가"라 단언 → Flirds 1-HVP가 그 전제를 무효화.
- FedPSF-LLM [Neural Networks'25/26]: 프롬프트-튜닝 FL의 "기여도+격차" 동적 가중 — **최근접 near-miss**, 휴리스틱 추정(정확식 미검증: SD 403). 논문화 전 원문 확보+구분 인용 필수.
- iPFL [Nat.Comms'25]: LLM-LoRA 스케일이나 시장-매개 개인화(edge weight), per-round 기여도 가중 아님.
- GREATS [NeurIPS'24]: **sample-레벨** Taylor-descent greedy 선택(ghost 1차 Gram) at LLaMA-2 — P2의 개념적 쌍둥이, 클라-레벨·2차·연속 QP가 우리 델타.
- **LoRA 제약**(모든 φ-가중이 존중해야): A·B 인수 별도 평균 시 가중이 곱으로 제곱됨(FedEx-LoRA/FFA-LoRA/FlexLoRA/FLoRA). 단 우리 Taylor 모델은 **실제 집계가 일어나는 파라미터 공간(A,B)** 위에서 정의되므로(코드 기준: pkeys=LoRA A/B, `_fedavg_core`도 per-key 집계) 예측-일관성은 유지 — cross-product 노이즈는 무대 caveat로 명시. (이 해석은 페이지 자체 추론 — 조사 보고서의 "a_i,c_i는 ΔW_i 위 계산" 표기와 상충하나 코드가 본문 편.)

## 3. 공백 좌표계

축: **액션공간** {집계가중, 선택, 옵티마이저/step, 업데이트-수술, 배포/보상, 정규화항} × **정보** {0차, 1차, 2차} × **루프** {개, 폐(신호 재측정), 폐(개입효과 제어)} × **유도** {휴리스틱, 학습(GD/RL), 폐형-유도} × **스케일** {CNN/MLP, ViT, LLM/PEFT}.

| # | 빈칸 | 증거 |
|---|---|---|
| **G1** | **클라-레벨 training 매핑에 2차(곡률) 정보 소비** — 0건 | 2차는 valuation/탐지에만 등장(Ripple Jacobian, DICE, FLDetector L-BFGS, DataInf); **클라-레벨** 가중·선택·수술 매핑은 전부 0/1차 (데이터-레벨 일회성 필터로는 DataInf@7B 사례 존재 — §2-E 2403.04529) |
| **G2** | **개입 강도의 실측-적응(폐루프 제어)** — 집계규칙 제어 계열(FedLAW/FedHAW/GAAvernor/ByGARS/FedAOT)은 있으나 전부 brute-force GD/RL/bilevel/meta-grad + 기여도 의미론 없음, **기여도 신호로 강도 제어 0, do-no-harm 붕괴 보장 0** | strict-sense 폐루프는 MATES(centralized)·FedAOT(FL, abstract-검증) 정도 — 어느 쪽도 강도-게이트·붕괴 보장 없음 |
| **G3** | **val-loss Taylor에서 폐형-유도된 가중** — FedFomo(1차·유한차분·개인화)뿐; **hypergradient 항등식** ∂L_val/∂w_i = a_i+c_i를 정확·폐형으로 가진 것 0 | FedHAW가 같은 객체를 train-loss 유한차분으로 근사 |
| **G4** | **검증-앵커 성분-수술** — 2×2 {참조: peer↔val-앵커}×{액션: 스칼라↔성분수술}의 유일한 빈칸 | 수술은 전부 peer 참조(FedFV/FedGH/GCFL)·버퍼 참조(Fed-A-GEM); val 참조는 전부 스칼라(Zeno++/FLTrust) |
| **G5** | **LLM/PEFT 스케일의 클라-레벨 기여도 소비** — 0건(§2-E) | fidelity 축에서 이미 확인된 공백과 동일 패턴 |

**부수 발견(교정·주의)**:
- arm 4 "Ripple convention" 어트리뷰션 미해결: [[sources/ripple-shapley]]에 가중 규칙 없음(pricing만). SPACE가 동형 (1−β)p+βc를 선택에 사용. **원 논문 재확인 전 Ripple 인용 보류** (lint 후보).
- [[sources/game-of-gradients-sfedavg]] 위키 페이지의 "SV-가중 집계 + negative-SV pruning" 서술은 원 AAAI PDF와 불일치 — 실제는 **선택-only**(softmax 샘플링+균등 평균; SV-as-weights는 명시적 future work) (lint 후보; 페이지 TODO 해소 정보).
- [[sources/mavericks-shapley-fl|Mavericks]]: SV-비례 선택이 OOD-good 클라를 초기에 배제해 수렴 실패(FMNIST R@99 미달) — **naive φ→선택의 정식 경고**; 고정 매핑이 아닌 시간-가변/보수적 개입의 근거.
- fedif_w β=0.7은 우리 표기(β=old-weight)로 FedIF γ=0.3과 **일치**(1−γ) — 조사 중 "불일치" 플래그는 오독.
- 정체성 긴장(위키 flirds.md): Flirds의 포지셔닝은 "vanilla FedAvg 위의 valuation 회계사"였음 — φ를 가중에 되먹이는 순간 우리가 구분해온 사분면(FedIF/FedTSV류)으로 진입. **프레이밍**: 우리는 dual-oracle로 *검증된* 점수를 소비(FedIF/FedTSV는 자기 점수를 oracle 검증 없이 투입 — 집계-측 논문 중 oracle fidelity 보고 0건). 이것이 "estimator 논문의 2차 축"으로서의 우리 서사.
- GTG류 sub-model 재구성은 vanilla FedAvg를 전제 — **가중-training 하의 valuation 비교에는 주의**(로그가 개입된 궤적을 기록).

## 4. 제안 — 공통 원리

세 제안 모두: (i) **이미 지불한 원료 재사용** — 온라인 Flirds 점수기가 라운드마다 형성하는 g^r=∇L_val(θ_r), u^r=H^rΔW^r(1 HVP), a_i=⟨g^r,Δ_i⟩, c_i=⟨Δ_i,u^r⟩; (ii) **min-max 폐지** — φ를 val-loss 단위 그대로 소비(φ=0 앵커·크기 정보 보존); (iii) **do-no-harm을 구조로 내장**(사후 희망이 아니라); (iv) seam 준수 — `_fedavg_core`의 `weights_fn`(P1/P2) 또는 최소 확장 훅(P3); (v) 서버-측 전용(클라 통신·연산 0 추가, locked 제약 유지); (vi) plain SGD m=0 관례 유지.

**핵심 항등식** (라운드 r, 참여 P_r, p_i=n_i/Σn_j, Δ(w)=Σw_iΔ_i):

$$m_r(w) \;=\; \langle g^r, \Delta(w)\rangle + \tfrac12\,\Delta(w)^\top H^r \Delta(w), \qquad \nabla_{w_i} m_r(p) = a_i + c_i$$

- 예측 라운드 강하(FedAvg 가중): $m_r(p) = \sum_i \varphi_i^r$ (효율성 공리 — **공짜 부산물**).
- $\varphi_i^r = p_i(a_i + \tfrac12 c_i)$ 는 정확히 **2차 대리게임** $v_r(S)=\langle g,\Delta_S\rangle+\tfrac12\Delta_S^\top H\Delta_S$ ($\Delta_S=\sum_{i\in S}p_i\Delta_i$)의 Shapley value이고, 쌍별 Shapley interaction index는 $I_{ij}=p_ip_j B_{ij}$, $B_{ij}=\Delta_i^\top H\Delta_j$ (2차 게임의 표준 결과; Grabisch–Roubens / Shapley-Taylor [ICML'20] corollary로 인용 — 정리로 주장하지 말 것).
- $a_i+c_i$ 는 집계가중에 대한 val-loss의 **정확한 2차-보정 hypergradient** — FedHAW가 train-loss 유한차분으로 근사하는 그 객체.

**통일 서사**: *Flirds는 라운드의 2차 대리게임을 값매기고(φ = Shapley value), 진단하고(B = interaction index), 이제 그 같은 게임을 최적화한다(P1/P2 = 같은 게임의 argmax).* 4개 naive arm이 버리던 것이 바로 이 게임의 구조다.

---

### 제안 1 — TRAC: Trust-Region Aggregation Control (방향 A+D 융합; **주력 권고**)

**수식.** FedAvg 가중 p에 KL로 앵커된 예측-강하 최적화

$$w^\star \;=\; \arg\min_{w\in\Delta_{P_r}}\; m_r(w) + \tfrac{1}{\lambda_r}\,\mathrm{KL}(w\,\|\,p)$$

의 **entropic mirror-descent 1스텝**(Beck–Teboulle) 폐형해:

$$\boxed{\;w_i \;\propto\; p_i \exp\!\big(-\lambda_r\,(a_i + c_i)\big)\;}$$

λ_r은 **신뢰영역 비율**로 온라인 적응: 실측 강하 $\hat\rho_r = \dfrac{L(\theta_{r+1})-L(\theta_r)}{\hat m_r(w)}$ ($L(\theta_{r+1})$은 다음 라운드 grad 계산에 부수(`grad_and_value`), $\hat m_r(w)=\sum_i w_i(a_i+\tfrac12 c_i)$ — c는 Δ(p)에서 계산된 근사; 소-tilt에서 유효, KL 앵커가 이를 강제. 정밀 옵션 = Δ(w)에서 HVP 1회 추가).

$$\lambda_{r+1}=\begin{cases}\min(\gamma_\uparrow\lambda_r,\;\lambda_{\max}) & \hat\rho_r\in[\rho_{lo},\rho_{hi}]\ \text{(모델 신뢰)}\\ \gamma_\downarrow\lambda_r & \text{else}\end{cases}\qquad \lambda_0=0$$

**+ dead-zone**: $\mathrm{spread}_i(a_i+c_i) < \varepsilon_r$ (노이즈 플로어 = val-청크 bootstrap SE — probe-C의 per-chunk dump 기계 재사용) 이면 λ_r=0 → **w ≡ p, 비트-정확 FedAvg**.

**보조정리(작성할 것)**: $\mathrm{KL}(w^\star\|p) \le \min\{\lambda_r\cdot\mathrm{spread},\ \lambda_r^2\cdot\mathrm{spread}^2/8\}$, spread = max_i−min_i (a_i+c_i) (전자는 3줄 증명, 후자는 Popoviciu — 소-λ에서 이차적으로 강해 dead-zone 논증을 강화) — 라운드당 FedAvg 이탈의 명시적 상계; λ=0이면 정확히 FedAvg. DoGE(재귀 앵커 α^{t-1})와 달리 **매 라운드 정지 참조점**(p)이라 이 상계가 성립. (각주: "정확한 hypergradient"는 2차 모델-정확의 의미 — 참 loss에 대해선 a_i+c_i = 진짜 hypergradient + O(‖Δ‖²); FedHAW의 유한차분보다 엄격히 강하나 exact로 읽지 말 것.)

**φ→training 매핑.** `intervene.py`에 `make_trac_weights_fn(controller, round_raw_fn, sample_nums)` — 기존 `weights_fn` seam에 그대로. `OnlineScorer` 자리에 상태 (λ_r, 직전 예측치, 직전 L) 3-스칼라 컨트롤러. 서버/러너 변경 0.

**vs 4 naive / 선행.** (i) 정적 매핑 → **유도된 최적 스텝**(무엇을 최대화하는지 명시); (ii) min-max/EMA → val-loss 단위 φ 직접 소비; (iii) 개루프 → **개입이 자기 효과로 강도를 벌어들이는** 폐루프; (iv) w=p·exp(−λ(a+c))는 소-λ에서 $p_i\big(1−λ(a_i+c_i−\bar s)\big)+O(λ^2)$, $\bar s=\sum_j p_j(a_j+c_j)$ (코호트-평균 센터링 — 상대 비교가 메커니즘의 본질) — **Yonghee의 곱셈 규칙(arm 1)의 유도된 형태**(hand-map을 정리로 대체). 선행 차별화(novelty 공격 판정: partial-overlap, 차별 경로 명확): DoGE [ICML'24] = 같은 MD 수학이나 **1차·μ 고정·LM 데이터혼합**(2차 배제를 명시) → 우리는 곡률 지수+TR-적응 λ+FedAvg 정지앵커+FL 클라 단위; FedAdp = n·exp(휴리스틱 각도·α 고정); ByGARS = a_i의 EMA를 부호가중으로(meta-학습, 8-worker); FedAOT = val meta-gradient 가중 학습(폐루프이나 기여도 의미론·강도 게이트 없음) — **인용 필수 4편**. FedHAW를 엄격 상회(정확한 val 1st+2nd hypergradient vs train-loss 유한차분). "첫 주장" 금지 항목: 폐루프 가중 자체(GAAvernor/FedLAW 존재), valuation→training 되먹임 자체(ShapleyFL/FedIF 존재). **주장할 것**: 모델-적합성-게이트 개입(do-no-harm 붕괴 보장) × 곡률 지수 × 제로 한계비용 × LLM/PEFT 첫 사례.

**이득 무대 / do-no-harm.** 이득: 비IID silo5(품질격차), poison 셀(φ 분리 9–18×; noisy는 ~0.6×로 노이즈 수준), 7B std20 플래토(r2t −32의 무대 — vanilla 비효율 지점에서 λ가 자란다). IID-clean: 진단 문서가 보인 "클라 간 진짜 신호 부재" ⇒ spread(a+c)가 플로어 아래 ⇒ dead-zone+TR 붕괴 ⇒ **구성상 parity** (검증 가능 예측: IID-clean에서 Σ_r λ_r ≈ 0 로깅 — naive arm은 같은 무대에서 계속 tilt함을 대조).

**비용.** 추가 HVP/forward **0** (flirds_w가 이미 지불하는 온라인 스코어링 재사용) + 라운드당 loss 값 1개(grad에 부수) + O(k) 스칼라. 정밀-TR 옵션만 +1 HVP.

**위험/실패 모드.** ① 1-라운드 지연 피드백의 λ 진동 → 히스테리시스(γ↓≫γ↑)·λ_max; ② val(200~1000) 과적합 → KL 상계+λ_max가 이탈 자체를 캡, test-측 지표 병행 감시; ③ ĉ의 Δ(p)-기준 근사 → 소-tilt 영역 유지(수식 그대로) 또는 정밀 옵션; ④ Mavericks-형 편향 상속(예측-강하는 근시안 — OOD-good 클라의 장기 이득 저평가) → 탐색 플로어 w_i ≥ ε·p_i, λ 상계; ⑤ g_val 정렬 공격(FedIF PGD 맹점) — 2차항이 부분 완화하나 면역 아님(탐지는 위계상 3순위, 정직하게 한계 기술).

---

### 제안 2 — QP-Agg: 강하-최적 QP 집계 + 클라 상호작용 (방향 A 완전판 + B 흡수)

**수식.** 참여자별 HVP $u_j = H^r\Delta_j$ (k회)로 Gram $B_{ij}=\langle\Delta_i,u_j\rangle$ 구성 후

$$w^\star=\arg\min_{w\in\Delta}\; a^\top w + \tfrac12 w^\top B\, w + \tfrac1\lambda \mathrm{KL}(w\|p), \qquad B\leftarrow B+\mu I\ (\text{H 부정부호 시 감쇠})$$

k≤100의 초소형 문제(mirror-descent 수십 회 반복, CPU-무시가능). **부산물**: $I_{ij}=p_ip_jB_{ij}$ = 쌍별 상호작용 진단(충돌 $B_{ij}<0$ / 중복 $B_{ij}>0$ 큰 값) — 방향 B가 원하던 구조가 공짜로 나옴. TRAC과의 관계: TRAC의 $c_i=(Bp)_i$는 **행-평균 상호작용**만 봄 — QP-Agg는 개별 쌍을 분해. 근사-가산 무대(진단: IID 갭 ≤0.9%)에선 B가 대각-지배 ⇒ QP-Agg ≈ TRAC ⇒ **이득 없음이 예측됨** (정직한 스코핑: 상호작용이 실질적인 무대 전용).

**φ→매핑.** `weights_fn`으로 동일 삽입; round_raw_fn 확장판이 (a, B) 반환(per-client HVP 루프).

**vs naive/선행.** naive 대비는 P1과 동일 + 상호작용 구조 활용(스칼라 φ-가중이 원리적으로 못 하는 것 — 중복/공모 클라 이중계상을 B_ij가 벌줌; 스칼라 방법이 못 맞추는 무대를 실험으로 설계할 것). 선행: FedMGDA+ (같은 QP 기하, **1차 Gram·공정성 목적**), SmartFL(같은 문제를 proxy-SGD로 — 동일 비용에서 이겨야 할 baseline), GREATS(같은 원리 sample-레벨·greedy·1차 ghost Gram — 전면 인용 필수), MGDA/CAGrad/Nash-MTL(MTL의 Gram-QP 패턴), Mallows averaging [Hansen '07](통계학 조상). **주장할 것**: 하나의 2차 대리게임에서 valuation(φ=Shapley)+진단(B=interaction index)+가중(QP=argmax)의 **삼위일체** + LLM/PEFT에서 k-HVP 실행가능성(LoRA라 가능).

**이득 무대 / do-no-harm.** 5-도메인 비IID(도메인 충돌 = B_ij<0), 중복/공모·free-rider 무리(B_ij>0), poison. IID-clean: KL 앵커+dead-zone 동일 + B≈대각 ⇒ TRAC로 퇴화 ⇒ parity.

**비용.** 라운드당 **k HVPs** (silo k=4–5: Flirds 스코어링의 ~4–5×, 1B에서 라운드당 수 분 수준; N=100/k=10 cross-device: 10 HVPs — anchor급 무대엔 부담, silo·소-k 무대 우선). 정확 수치는 기존 runtime에서 스케일링해 계획서에 명기할 것.

**위험.** ① H 부정부호 → μ-감쇠가 해를 보수화(사실상 TR); ② B의 추정 노이즈(청크 SE) → 상호작용 신호가 노이즈 위인지 사전 점검(진단 문서 기계 재사용); ③ 가중 자유도↑ = val 과적합 여지↑(KL 필수); ④ 근사-가산 무대에서 2차 기계가 "불필요해 보임" — 리뷰어 공격 지점, 상호작용-실질 무대 없이는 제안 가치가 성립 안 함(P1 먼저, P2는 무대 확보 후).

---

### 제안 3 — VDP: Validation-Descent Projection (방향 C-ii; 업데이트 수술)

**수식.** 집계 전, 각 클라 업데이트에서 val-loss **상승 방향 성분만** rank-1 제거:

$$\boxed{\;\tilde\Delta_i \;=\; \Delta_i - \frac{\max(0,\,a_i)}{\|g^r\|^2}\, g^r\;}\qquad(a_i=\langle\Delta_i,g^r\rangle\ \text{재사용, 신규 연산은 } \|g^r\|^2 \text{ 내적 1회})$$

성질(명제로 작성): (i) 유지된 업데이트의 1차 예측 해악 $\langle\tilde\Delta_i,g\rangle=\min(a_i,0)\le0$ ∀i ⇒ 집계 **1차** 예측 강하 ≤ vanilla (2차는 통제 안 됨 — 위험 ④); (ii) **모든 a_i≤0이면 FedAvg와 항등**(IID-clean의 전형) — 구조적 do-no-harm; (iii) 직교 성분(포맷/스타일 지식) 보존 — 스칼라 가중·게이트가 통째로 버리는/왜곡하는 것(FLTrust는 항상 재정규화 왜곡, Zeno는 직교 지식까지 폐기)과의 실측 대조가 헤드라인 실험("오염 하 유지-효용"). 소프트 변형: 계수에 TRAC의 λ-게이트 합성(강도 폐루프화).

**φ→매핑.** 스칼라 가중이 아니므로 `weights_fn` 불가 — **seam 최소 확장**: `_fedavg_core`에 cohort-레벨 훅 `deltas_transform(r, w_r, deltas_map) -> deltas_map`(기존 per-client `delta_transform`(오염 seam)과 평행, ~5줄, 기본 None=비트-동일). 코드 규율(외과적 변경) 내에서 정당화 가능한 유일한 확장.

**vs naive/선행.** 액션공간 자체가 다름(스칼라 → 벡터 성분). novelty 공격 판정: 연산자·비활성-성질 모두 A-GEM의 것(Fed-A-GEM이 FL 반입, 참조=replay buffer·클라측·망각용), 프레이밍은 ATTITTUD의 것 — **식이 아니라 구성(configuration)을 주장할 것**: 2×2 {참조: peer↔**val-앵커**}×{액션: 스칼라↔**성분수술**}에서 유일 빈칸 = G4(§3). 같은 신호(a_i)의 스칼라 소비자(Zeno++/FLTrust)와 직접 head-to-head. Safe LoRA [NeurIPS'24](非FL, 안전 부분공간 투영) 관련연구 인용. **2차-인지 확장으로 A-GEM 계보와 실거리 확보**: 투영 방향을 g 대신 g+u^r(이미 지불한 곡률-보정 방향)로, 또는 도메인-청크 gradient들 {g_c} (LLM 백엔드의 loss_chunks가 이미 도메인별 폐포를 가짐!)로 **rank-5 부분공간 투영** — 단일 rank-1 방향에 숨는 공격을 좁힘.

**이득 무대 / do-no-harm.** 오염 무대 전용에 가까움(poison/label_flip/free-rider — a_i⁺가 큰 곳; phase2 poison φ-분리 9–18×가 신호 존재 증거) + 비IID 충돌. IID-clean: a_i>0 자체가 드묾 ⇒ 활성률 ≈ 0 로깅(검증 가능한 do-no-harm 지표). 탐지가 아니라 **무해화(sanitization)** — 위계상 "성능" 축의 실험으로 서술 가능(오염 무대의 최종 성능/수렴).

**비용.** **0** (g^r·a_i 재사용; ‖g‖² 1회). 세 제안 중 최저가.

**위험.** ① rank-1 맹점: g에 직교하게 숨는 공격 통과(스칼라 가중과 **상보적** — 합성 arm이 자연 해법; ρ‖Δ‖² Zeno-식 크기 페널티 ablation); ② 작은 val의 g 노이즈로 유익 성분 오삭제 — 제거량이 a_i⁺에 비례해 자기-제한적(오삭제 크기 = 잘못 추정된 a_i⁺만큼); ③ 같은 런에서 valuation을 함께 기록하면 로그가 투영된 Δ̃를 담음 — 개입-하-valuation 일관성 caveat(GTG 전제 파괴와 동일 계열) 명시; ④ 2차 해악(곡률에 숨는)은 1차 투영이 못 잡음 — 2차-인지 확장이 해법이자 차별점.

---

## 5. 검증 스케치 (위계 순; 전체 실험설계 아님)

fidelity(1차)는 기왕 검증 — 이 축은 **2차 실효성**: ① 성능 = paired 같은-seed val-loss Δ(진단이 보인 유일한 SNR>2 축; 3-seed)와 오염 무대 최종 성능, ② 수렴 = rounds-to-target(7B std20 플래토가 유일한 해상도 있는 무대), ③ 탐지 서술은 배제(VDP의 무해화는 성능 축으로 보고). 무대: (a) **CNN Track C** label_flip/free-rider + iid 대조군(빠르고 통제됨 — oracle 안정성 0.97 vs −0.04; TRAC/VDP 파일럿), (b) **LLM silo5** 비IID·poison/noisy 셀(phase2 무대 재사용), (c) 7B std20(r2t), (d) **IID-clean do-no-harm 검증**: 전 arm parity + TRAC의 Σλ→0·VDP의 활성률→0 로깅(붕괴 자체가 측정 가능한 주장). Baseline: vanilla + 4 naive arm + 소거(고정-λ 1차 지수 [DoGE/FedAdp 유사체] / 고정-λ 2차 / TR-적응 full; VDP hard/soft/2차-인지) — novelty 공격이 요구한 최소 ablation 행렬과 일치. cross-device 확장 시 FedAU-식 참여 탈편향과 곱-합성.

## 6. 권고

**TRAC(P1)을 1순위로 구현 검토** — 비용 0, 기존 `weights_fn` seam에 그대로, G1+G2+G3+G5를 동시에 메우며, flirds_w의 유도된 상위호환이라 기존 Track D arm 체계와 직접 비교 가능, do-no-harm이 구성적(진단 문서의 결론과 정합). **VDP(P3)를 2순위 병행** — 비용 0·seam 5줄·오염 무대에서 스칼라 가중과 상보적(합성 arm이 자연스러운 후속). **QP-Agg(P2)는 조건부** — 상호작용이 실질적인 무대(도메인 충돌·중복 클라)를 먼저 확보한 뒤; 근사-가산 무대에선 TRAC로 퇴화함이 예측되므로 단독 선행은 비추천.

---

## 부록 A — novelty 공격 판정 요약 (3/3 partial-overlap; 전문은 세션 원자료)

| 초안 | 최근접 선행 | 생존하는 델타 |
|---|---|---|
| TRAC | DoGE(수학 골격)·FedAdp(n·exp 형태)·ByGARS(a_i 가중)·FedHAW(hypergrad)·FedAOT(val meta-grad)·Zeno++(신호)† | TR-게이트 do-no-harm 붕괴 + 곡률 지수 + 정지 KL-앵커 보조정리 + 제로비용 + LLM/PEFT |
| QP-Agg | FedMGDA+(QP 기하)·SmartFL(같은 문제)·GREATS(LLM sample-레벨 쌍둥이)·FedLAW류 | 2차 대리게임 삼위일체(valuation+interaction+argmax) + Hessian Gram + LLM k-HVP 실행가능성 |
| VDP | A-GEM/Fed-A-GEM(식 동형)·ATTITTUD(프레이밍)·Zeno++/FLTrust(신호)·FedFV(수술) | val-앵커 × 성분수술의 빈칸(G4) + 항등 do-no-harm 명제 + 직교-지식 보존 실측 + 2차-인지 확장 |

†TRAC 행의 ByGARS·Zeno++·FedAOT는 웹 조사(web_projection/web_llmGap)에서 추가된 인접 선행 — 공격 파일 자체는 DoGE·FedAdp·FedIF·Zeno·FedLAW류·AAggFF/TERM을 지목.

## 부록 B — 이 페이지가 여는 후속 노브(제안 아님, 기록만)

- **폐형 Taylor line search**: η* = −⟨g,ΔW⟩/⟨ΔW,HΔW⟩ — FedSLS가 다중 함수평가로 얻는 것을 공짜로; FedLAW γ<1(shrinking)의 폐형 대응물. Σw=1 완화 노브. (유효조건: ⟨ΔW,HΔW⟩>0일 때만 최소화 — 부정곡률이면 최대화/발산; P2의 μ-감쇠와 동일 이슈, η∈[0,η_max] 클램프 fallback.)
- ShapFed-식 **기여-비례 배포**(클라가 받는 모델의 φ-혼합) = 우리 arm에 없는 5번째 액션공간; CGSV 희소화 보상도 동류.
- FedTSV의 **max{0,·} 누적** 정규화 = min-max 대체 ablation(φ=0 앵커 보존).
- 기여도-스케일 정규화항(FedProx류) — 조사에서 확인된 완전 빈 칸(FedCorr만 품질-구동 proximal).

> TODO: FedPSF-LLM 원문 확보(SD 403) 후 §2-E 갱신; arm-4 Ripple 어트리뷰션 원 논문 대조; [[sources/game-of-gradients-sfedavg]] 페이지 교정(lint).

---

## 7. 부호-게이팅(τ=0) novelty check — 2026-07-19 추가

**질문**(Track G 설계 중, Yonghee): 기여도의 양/음 부호를 training 개입(제외/가중)에 쓴 선행이 있는가.
**방법**: 본 페이지 §2 재활용 + 신규 웹 스윕 2회(FL측 14쿼리·원문 25건 method-수준 확인 / centralized측 12쿼리·원문 ~25건). Codex 교차검증은 세션에 MCP 부재로 생략, 자체 적대 검증으로 대체.

**답: 부호-절단 연산 자체는 선행 다수 — "부호 사용 최초" 주장 불가. novelty는 결합에만 있음.**

### 최근접 선행 (전부 원문 method/코드 수준 확인)

| 선행 | 규칙(원문 확인) | 남는 delta |
|---|---|---|
| **[[sources/fedtsv\|FedTSV]]** [ECC'26, arXiv:2605.30336, 26-05 공개] | 누적 궤적-SV에 **α_i=max{0,φ_i}** 온라인 가중(Eq.13) — 부호-0 하드 절단 | utility=거리 커널(실제 val-loss 게임 아님)·**weight-zeroing만**(음수 클라도 계속 학습·업로드)·V1 변형/burn-in 없음·oracle 검증 0·MLP/ResNet-20. **concurrent(26-05)로 처리 + 정면 인용 필수** |
| **UAV-FL 자원할당** [Xiong & Guo, Sensors 2024;24(20):6711, PMC11511571] | "negative Shapley → 당회 집계 제외 w=0, 양수는 크기 비례" = **V2w와 동형** | exact-enum SV(소규모 UAV 계층 전용)·집계 가중만(참여 게이팅 없음)·EMA ρ 필요·변형 비교/fidelity 없음. 인지도 낮아도 **인용 필수** |
| [[sources/fltrust\|FLTrust]] [NDSS'21] | **ReLU(cos)** — 음수 방향정합을 0으로 절단(당회 집계 배제) | 점수=cosine 휴리스틱(clean 루트셋 필요), 게임 의미·누적·참여 게이팅 없음 |
| **In-Run Data Shapley** [[sources/in-run-data-shapley\|IRDS]] [ICLR'25 Oral] | **음수-가치 corpus 필터링 후 재학습**(Pile ~16% 음수, 수렴 ~25%↑) — 명시적 τ=0 | **V3(사후 부호 제거-재학습)의 직계 조상** — 사후·중앙집중·corpus 단위. V3는 "IRDS 프로토콜의 FL-클라 번역"으로 포지셔닝(계보 연속성 서사) |
| Zeno++ [ICML'20] | update별 γ⟨g_val,g⟩−ρ‖g‖²≥−γε 수락/기각 | ρ,ε 튜닝·비동기 worker·게임 없음 — V1의 개념적 조상 |
| 샘플-수준 계보: L2RW [ICML'18] max(0,·) 정류 / Data Dropout ['18] 사후 부호 드랍 / UIDS [AAAI'20] π=max{0,min{1,−αφ}} / **LAI** [arXiv:2510.16007, preprint] **매 스텝 음수-influence 폐기(τ=0, 온라인)** | 전부 데이터-포인트·단일 학습자 | 온라인 τ=0의 샘플-수준 선례 존재 — granularity·게임값·FL 집계가 delta |

### 오인용 방어 탄약 (원문/코드 확보)

ShapleyFL = min-max(Eq.2) **부호 소거**+β-EMA(절단·제외 없음; KDD 원문 PDF 확인) · S-FedAvg = softmax 선택-only(§3 lint 재확인) · CGSV = `clamp(rs, min=1e-3)` ε-floor(공식 코드 확인 — 완전 제외 아님) · ShapFed = (1+cos)/2 재스케일(음수가 양수 가중으로 잔존) · RFFL β=1/(3N)·CFFL c_th(grid search)·FedSV-ICC'24 k-means 군집 = 전부 **튜닝 임계**(부호 아님) · Fed-Influence AAAI'21 = 하위 비율(30%) 일회성 제거(top-k류) · Data Shapley 원조의 제거는 **rank 곡선**(부호는 정성 언급) · OpenDataVal 표준 과제에 부호-임계 제거 부재.

### 미발견 조합 = Track G 점유 공간

① **부호 기준 참여 제외(V2)** — 선택 단계 제외(연산·통신 중단)+burn-in/probation: 어디에도 없음(제거형은 전부 튜닝 임계·군집·순위) ② **per-round 게임값 부호 당회 게이팅(V1)** ③ **oracle-검증된 실제 val-loss 게임의 φ=0(null-player)** 을 임계로 쓰는 조합(집계-측 논문 oracle fidelity 보고 0건 — §2 확인 유지) ④ V1/V2/V2w/V3 체계 비교+oracle-excl 대비 recovery+do-no-harm 사전등록 ⑤ LLM/PEFT 스케일(G5 유지) ⑥ 2차 정보 게이트(G1 유지).

### 주장 수위 (논문·Track G 문서 공통)

- **금지**: "부호/절단 기반 사용 최초", "max{0,·}가 파라미터-프리라서 신규".
- **안전**: "경쟁 방법은 튜닝 임계(β=1/(3N), c_th, λσ², b, ρε) 또는 부호-소거 정규화(min-max, (1+cos)/2, ε-floor)를 요구 — 우리는 dual-oracle로 검증된 게임값의 **자연 영점(null-player 공리, FR φ=exact-0)** 이 임계이고, 그 임계로 가중(V2w)뿐 아니라 **참여 자체(V2)** 를 게이팅하며, 게이트의 작동영역(FR 발화 / net-도움 noisy 침묵)을 실측으로 규정".
- FedTSV concurrent 명시, UAV Sensors'24 선제 인용, RLHF advantage-부호 계열(EFRame 등)은 관련연구 한 줄 구분.
- 상세 출처 표·URL: 세션 로그 2026-07-19 (FL측/centralized측 스윕 보고 원문).
