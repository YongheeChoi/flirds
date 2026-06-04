---
type: note
title: Flirds 연구 진행 보고 — 슬라이드 해설
created: 2026-06-04
updated: 2026-06-04
sources: [flirds]
tags: [presentation, 발표대본, slide-notes]
---

# Flirds 연구 진행 보고 — 슬라이드 해설

`flirds-progress-2026-06-04.pdf` / `.pptx` (12장) 의 장별 설명서.
각 장마다 **보여주는 것 → 핵심 메시지 → 말할 거리(🎤)** 순으로 풀었다. 발표 대본 겸,
덱만 봐선 압축돼 안 읽히는 부분의 해독본. 본문 정본은 [[flirds]].

## 이 덱은 무엇인가

2026-06-04 기준 Flirds 프로젝트의 **현재 상태 보고**. 흐름:

| 장 | 묶음 | 답하는 질문 |
|---|---|---|
| 1–2 | 요약 | 한 문장으로 뭔가? 지금 어디인가? |
| 3–6 | 정당화 | 왜 새롭고(3·5) 왜 옳은가(4·6)? |
| 7 | 동결 | 무엇을 더는 안 바꾸기로 했나? |
| 8–9 | 진척 | 지금까지 뭘 했나 (CNN→LLM)? |
| 10–11 | 계획 | 무슨 데이터로 무슨 실험을? |
| 12 | 전망 | 다음 1수와 한계는? |

**한 문장 thesis:** *vanilla FedAvg가 이미 받는 클라이언트 업데이트 $\Delta w_k$ 만으로,
라운드별 검증손실 변화의 1차+2차 Taylor 전개로 client별 Shapley 기여도를 닫힌형 계산 —
통신 추가비용 0, 역헤시안 없음.* (1·5·7·9·12장 어디서든 이 문장으로 되돌아오면 된다.)

---

## 1장 — 표지

**보여주는 것.** 프로젝트명 **Flirds = Federated Learning + In-Run Data Shapley**,
한 줄 정의, 저자·컴퓨팅(DGX B200×4).

**핵심 메시지.** 이름 자체가 두 계보의 결합 — FL(분산 학습) + IRDS(학습 *도중* 데이터
기여도 측정). 부제 한 문장이 곧 thesis이고, 안에 6개 자랑거리가 다 들어있다:
① $\Delta w_k$ 만 / ② 1·2차 Taylor / ③ client-level / ④ 닫힌형(closed-form) /
⑤ 통신비용 0 / ⑥ 역헤시안 없음.

**🎤** "기존엔 어느 데이터가 기여했는지 알려면 재학습하거나 통신을 더 해야 했습니다.
우리는 이미 서버로 오는 정보만으로 그걸 공짜로 뽑습니다."

---

## 2장 — 한눈에 보기 (Executive Summary)

**보여주는 것.** 왼쪽 4줄(무엇/어떻게/강점/novelty) · 오른쪽 현재위치 사다리
(완료·완료·진행·예정) · 하단 다음 액션 띠.

**핵심 메시지.** 이 한 장이 덱 전체의 축약. 나머지 10장은 이 4줄의 전개다.
- **무엇** = client별 기여도 $\phi_k$ 측정, IRDS를 FL+LoRA(PEFT)로 확장.
- **어떻게** = 라운드별 검증손실 변화의 1·2차 Taylor 닫힌형, $\Delta w_k$ 만 사용.
- **강점** = 역헤시안 없는 forward HVP라, LLM에서 influence function(IF)이 무너지는
  원인(**iHVP collapse** — 역헤시안 추정이 LLM 스케일서 불안정)을 *구조적으로* 회피.
- **novelty** = 6요소 교집합이 prior art에 비어 있음(5장에서 표로 증명).

**사다리 읽는 법.** Phase 0(CNN baseline 4종)·0.5(estimator+oracle) **완료** → Phase 1(LLM 1B)
5단계 빌드+검증 **진행**(smoke 통과) → Phase 2/3 **예정**.

**🎤** "슬라이드를 한 장만 보신다면 이겁니다. 왼쪽이 방법, 오른쪽이 진척, 아래가 다음 1수."

---

## 3장 — 문제 정의 (Problem)

**보여주는 것.** 상단 문제 한 문장 + 가운데 "기존 방법의 벽" 카드 3개 + 하단 초록
"Flirds의 접근" 띠 3줄.

**핵심 메시지.** **벽 3개 ↔ 접근 3줄이 1:1 대응**한다(이게 이 장의 설계 의도).

| 기존의 벽 | Flirds의 대응 |
|---|---|
| Retraining Shapley = $2^N$ 재학습 → LLM 불가 | post-hoc, 학습 안 건드림 |
| 통신/계산 오버헤드 (FedSV $O(Tm^2)$, ComFedSV 추가 라운드) | $\Delta w_k$ 만 → 추가비용 0 |
| Gradient/IF = heterogeneous FL서 취약(DataInf가 Fed-WildChat서 실패) + LLM서 iHVP collapse | forward HVP만, $H^{-1}$ 없음 |

**🎤** "데이터 기여도를 매기는 세 갈래가 각각 벽이 있습니다. 우리는 그 세 벽을
정면으로 피하는 세 가지 선택을 했습니다." (카드→띠 화살표로 가리키며)

---

## 4장 — 방법 (Method)

**보여주는 것.** 가운데 **히어로 수식 카드**가 전부. 아래 비용/2차항/안정성 3줄 + 하단 확정 노트.

**핵심 메시지 — 수식 한 줄이 논문의 심장:**
$$\phi_k^{(r)} \approx \underbrace{-\nabla\ell(w^r, z_{val})\cdot\Delta w_k}_{\text{1차 = 정렬}}
\;+\; \underbrace{\tfrac12\,\Delta w_k^{\top} H^{(val)}(w^r)\,\Delta W^{(r)}}_{\text{2차 = 상호작용/곡률}}$$
- **1차항** = validation gradient와 client 업데이트의 *정렬*(내적). "이 client가 검증손실을
  줄이는 방향으로 움직였나?"
- **2차항** = 그 업데이트가 곡률($H$)을 통해 *그 라운드 참여자 전체의 합* $\Delta W^{(r)}=\sum_j \Delta w_j$
  와 상호작용하는 정도. client 간 간섭을 잡는다.
- $\phi_k = \sum_r \phi_k^{(r)}$ — 라운드별 기여를 다 더한 게 최종 기여도.

**비용.** 라운드당 **HVP 1회**로 $u := H\cdot\Delta W^{(r)}$ 를 만든 뒤, $N$개 내적 $\Delta w_k\cdot u$.
LoRA 차원이라 싸다. **안정성.** $H^{-1}$ 은 절대 계산 안 함 → IF의 iHVP collapse 회피.

**하단 확정 노트(왜 중요한가).** 곡률 = **참 Hessian**(GGN/Fisher 변형은 검증 후 *기각*),
최적화 = **plain SGD**(momentum 0). 이 두 조건이어야 비로소 2차항이 1차항을 이긴다 —
6장 이론과 8장 실증의 핵심 연결고리.

**🎤** "왼쪽 흰 항이 정렬, 오른쪽 노란 항이 상호작용입니다. IF와 달리 역행렬이 없어요 —
행렬·벡터 곱 한 번이면 끝납니다."

---

## 5장 — 무엇이 새로운가 (Positioning)

**보여주는 것.** 경쟁 4종 vs Flirds 비교표(단위·통신오버헤드·in-run·비고) + novelty 3줄.

**핵심 메시지 — "우리만 비어 있는 칸을 채운다"를 표로 증명:**
- **FedSV(2020)** = federated Shapley 원조지만 서버 평가 $O(Tm^2)$.
- **Ripple(2026)** = **최근접 경쟁자**. 단 *sample* 단위 + Jacobian(1차 계열), 2차항 없음.
- **FedIF(2025)** = CNN 한정, 1차 TracIn, **aggregation-side**(가중치 바꿈).
- **FedTSV(2026)** = val pass 추가 비용 + aggregation(가치평가가 아님).
- **Flirds** = client / 통신 0 / **1·2차 Taylor** / LoRA + 2차항 + zero-comm — 마지막 줄만 다 채움.

**novelty 3종.** ① 6요소 교집합 점유한 단일 논문 없음(2026-06-03 재조사, 내부 4편+외부 survey
교차검증). ② **cross-domain valuation-fairness**(공유 val-loss Shapley의 도메인 간 공정성)는
prior art서 under-addressed → 추가 hook(10장 "형식 통일"이 이 대응). ③ FedIF/FedTSV는
**aggregation-side**(가중치 변경), Flirds는 **valuation-side**(post-hoc 가치) — 같은 입력, 다른 출력.

**🎤** "마지막 행만 모든 칸이 우리 쪽입니다. 특히 Ripple이 제일 가깝지만 sample 단위이고
2차항이 없습니다 — 그 둘이 우리 차별점입니다."

---

## 6장 — 이론 (Theory)

**보여주는 것.** 상단 2줄(①등가 ②FL편차) + Proposition 카드 2개 + 하단 핵심 함의 띠.

**핵심 메시지 — 방법이 옳은 이유 + 2차항이 FL에서 중요한 이유:**
- **① 등가.** 1-step SGD에선 한 client에 속한 data-level Shapley들의 합 = 그 client의
  client-level Shapley(1·2차 모두). → 중앙집중 1-step에선 "낱개냐 묶음이냐"가 무관
  (grouping-invariant). *granularity를 client로 올려도 손해 없다*는 정당화.
- **② FL 편차.** 멀티스텝에선 $\Delta w_k$ 가 local 궤적(E-step)의 *끝점*이라, Taylor 전개가
  "중앙집중 등가항" + "**client drift residual** $O(\eta E\cdot|H|\cdot\text{궤적길이})$" 로 분해된다.
- **Prop 1** = Flirds SV = (중앙집중 data-level SV를 client로 합산) + drift residual →
  FL이 중앙집중 IRDS에서 *얼마나 벗어났는지*를 정량화.
- **Prop 2** = drift residual은 local 궤적 반경의 3차식으로 bound. **E=1서 소멸**,
  non-IID·큰 local epoch서 증가.

**핵심 함의(이 장의 펀치).** IRDS 원논문은 "2차항은 marginal"이라 보고했지만, 그건
*centralized per-step*(아주 작은 $\eta$)의 산물이다. **FL의 per-round 멀티스텝이야말로
2차항이 본질적으로 작동하는 무대** — 가설이고, 8장에서 CNN으로 실증, 11장 flagship으로 LLM 검증.

**🎤** "IRDS는 2차항을 버려도 된다고 했는데, 그건 한 스텝씩 볼 때 얘기입니다. FL은
라운드마다 여러 스텝을 가니까, 바로 거기서 2차항이 살아납니다 — 이게 저희 가설입니다."

---

## 7장 — 확정된 핵심 설계 결정 (Locked Decisions)

**보여주는 것.** 왼쪽 설계축 8개 동결표 + 오른쪽 **Dual Oracle**(정답 2종) 카드 + 하단 동결 띠.

**핵심 메시지(왼쪽).** 더는 안 바꾸기로 한 것들 — 단위=client-level, 입력=$\Delta w_k$(LoRA)만,
차수=1+2차 항상 둘 다, 곡률=참 Hessian(GGN 기각), 최적화=plain SGD, 검증셋=도메인당
200×5=1000 stratified, participation 정규화 없음(기본), 모델=1B/3B/7B.

**핵심 메시지(오른쪽 — 이 장의 진짜 요점).** estimator가 맞는지 채점하려면 **정답이 둘** 필요:
- **(a) Exact retrain SV** — $U(S)$=부분집합 $S$만으로 실제 FL 재학습. 데이터가치 *표준* 정답.
  비싸서 1B는 N∈{5,10}, 3B는 N=5, 7B는 ✗.
- **(b) IRDS식 in-run SV** — Flirds가 *겨냥하는* 바로 그 정답(Flirds-correct oracle).
  cross-silo는 N=10 exact 전수(1024 subset), cross-device는 N=100 MC.
  비용 $2^N\cdot R\cdot|val|\cdot seq$ → **N5↔N10 = 32×**, fp32 필수.

**🎤** "estimator를 채점할 정답을 둘 갖고 있습니다 — 하나는 재학습 정답, 하나는 in-run 정답.
estimator가 둘 다와 일치하면 신뢰할 수 있는 거죠."

---

## 8장 — 진행 현황 ① CNN (Phase 0 / 0.5 완료)

**보여주는 것.** Phase 0 완료(baseline 4종 표) + Phase 0.5 완료(게이트 3줄) + 결정 띠.

**핵심 메시지.** **작은 CNN에서 모든 걸 exact($2^N$ 전수)까지 검증 — 방법이 옳다는 1차 증거.**
- **Phase 0** = 비교용 FL-Shapley baseline 4종을 직접 재구현(reference-guided self-build)해 재현:
  GTG-Shapley(recon cosine 0.99) · FedSV(perm-MC 0.998) · ComFedSV(Spearman {1.0,.96,.85,.84}) ·
  Ripple(noisy-AUROC 1.0). Ripple은 속도이득 62×/49×.
- **Phase 0.5** = Flirds estimator + dual oracle, 전 sanity gate green:
  estimator ≈ (b) in-run oracle **Spearman 1.0**(1+2차 0.96 > 1차 0.92) · noisy-client **AUROC 1.0** ·
  (b) Shapley efficiency·symmetry = 0(exact) · HVP 검증 9.8e-6 · 재현성 **bitwise-0**.
- **결정 띠** = 2차항=참 Hessian, momentum 제거 시 **2차항이 1차항을 이김** →
  "FL per-round가 2차항의 무대" 가설을 CNN에서 *실증*(6장 이론의 첫 경험적 확인).

**🎤** "여기선 N이 작아서 모든 부분집합을 다 돌려 정답을 구할 수 있습니다. estimator가 그
정답과 Spearman 1.0으로 맞았고, 2차항을 넣으면 더 좋아집니다."

---

## 9장 — 진행 현황 ② LLM (Phase 1, 거의 완료)

**보여주는 것.** 5단계 빌드 리스트(각각 검증수치) + 3 LLM musts 카드 + 남은 1액션 띠.

**핵심 메시지.** **estimator/oracle 코드는 한 줄도 안 바꾸고** LLM(real Llama-3.2-1B)에 5단계로 얹어
전부 검증 통과 — 코드는 완성, smoke까지 green.
1. backend-agnostic estimator/oracle + partial-participation + per-layer φ → *CNN bit-identical*(회귀 안전).
2. LLM backend + FL loop(TRL SFTTrainer + forced SGD) → 1B est≈oracle **1.70e-6**.
3. 5-domain free-form data + val micro-batching + per-domain norm → chunked==single **3.8e-8**.
4. corruptor: answer_swap(noisy) + free_rider(zero/random) → free-rider **φ = 정확히 0**(이론대로).
5. #7 first-clean-run 인프라(eval·run_logger·orchestrator) → SMOKE **1.6e-7**, AUROC **1.0**.

**3 LLM musts(왜 띄웠나).** CNN엔 없던 LLM 특유 함정 셋 — eager attention(forward-AD가
SDPA/flash 커널 미지원) · FL state를 named_parameters key로 · embedding require-grad hook clear
(functorch ↔ HF gradient-checkpointing 충돌). *"LLM 옮기며 새로 풀어야 했던 것들"*.

**남은 1액션.** FULL scale run(N=5, R≈30, 3 seed, ORACLE_B(N=5), ~5–7h, MINI de-risk 먼저).

**🎤** "방법 코드는 그대로 두고 LLM 어댑터만 5단계로 붙였습니다. 전부 검증 통과했고,
이제 남은 건 본 실행 한 번입니다."

---

## 10장 — 데이터셋 & 평가 프로토콜

**보여주는 것.** 5-domain 표 + 크기 줄 + 평가(utility≠downstream) 2줄 + 보고 엄밀성 띠.

**핵심 메시지.**
- **데이터 = 5도메인 cross-silo, 전부 free-form instruction→response.** 형식 통일은 단순
  정리가 아니라 **공정성 장치** — 1-token 분류(예/아니오)와 multi-token 생성을 한 검증손실로
  비교하면 Shapley가 도메인 간 불공정해진다. 통일로 그걸 없애고, 그 자체가 5장 novelty hook ②.
  (medical=med flashcards · legal=ibunescu QA · finance=FiQA · math=AQUA-RAT · general=Dolly;
  도메인당 train 12k/val 200/test 2k 상호 disjoint; cross-device=Fed-WildChat+FedHDS N=100,K=10)
- **평가 = utility ≠ downstream 분리(방법론 포인트).** estimator/oracle이 먹는 건 **utility=val-loss**;
  실제 성능 보고는 **downstream = per-domain ROUGE-L + math(AQUA) EM**. 둘을 안 섞는다.
- **selection-convergence** = φ로 top-K 뽑아 재학습 vs full/random 곡선(MATES 템플릿).
  ⚠ **caveat**: heterogeneous FL서 selection이 random에 *질 수도* 있음 — Flirds가 넘어야 할 bar(정직).

**보고 엄밀성.** bf16 train/fp32 eval + fp32 내적 · ≥3 seed mean±std · 95% bootstrap CI ·
sanity gate(E=1⇒residual≈0, N=2⇒singleton SV=φ_k, bitwise 재현) · local run-dir 추적(W&B 미사용).

**🎤** "다섯 도메인을 전부 같은 형식으로 맞춘 건 공정성 때문입니다. 그리고 estimator가 보는
손실과, 실제 성능 지표를 따로 둡니다 — 이 분리가 중요합니다."

---

## 11장 — 실험 계획 (Section 3, 18항목)

**보여주는 것.** 우선순위 3묶음(★★★/★★/★) + flagship 후보 카드 + 매트릭스 띠.

**핵심 메시지.**
- **★★★ Spine(반드시)** = baseline 10종 + detection 2종(FLDetector·STD-DAGMM) + dual oracle
  (a)/(b) + Ripple head-to-head + 이론적 reduction 시도.
- **★★ 특성화/ablation** = α-sweep×E-sweep drift 매트릭스(16셀) + Q2 variants(3×3) + 적대적
  stress + non-IID valuation bias.
- **★ Scale** = 7B instruction-tuning bench(LESS·FedDQC 직접 비교).

**flagship 후보(논문의 차별화 펀치).** PGD / direction-aligned poison —
**1차항으로는 못 잡는 공격을 2차(곡률)항이 분리해내는가?** FedIF의 blind spot을 겨냥하고,
Flirds-1st-only와 대조해 **"2차 > 1차"를 직접 입증**. 6장 이론·8장 CNN 실증을 LLM 공격탐지로 잇는 실험.

**매트릭스 띠.** 1B/3B/7B 전 셀(3 seed mean±std + 95% CI). 단 est-vs-oracle *fidelity*는
oracle 비용 때문에 N으로 capped(N5↔N10=32×, fp32) — estimator *방법*(noisy-AUROC, selection)은
oracle 없이 N=10까지 돈다.

**🎤** "차별화 한 방은 이겁니다 — 1차항이 못 잡는 공격을 2차항이 잡는지. 잡으면 '2차항이
실제로 쓸모 있다'를 공격탐지로 증명하는 거죠."

---

## 12장 — 다음 단계 & 한계

**보여주는 것.** 왼쪽 다음 단계 4단(즉시→이후→Phase 2→Phase 3) + 오른쪽 한계 3카드.

**핵심 메시지(다음 단계).** 즉시 = **1B FULL clean run**(N=5,R≈30,3seed,~5–7h, MINI de-risk 먼저) →
이후 = SV baselines LLM 이식(GTG/FedSV/ComFedSV/Ripple) → Phase 2 = full baseline + Data Banzhaf +
ShapleyFL + detection + 3B/7B → Phase 3 = 실험 매트릭스 실행 + Ripple 이론 reduction → 논문.

**핵심 메시지(한계 — 숨기지 않고 "고칠 결함이 아닌 특성"으로 프레이밍):**
- **Privacy** = 서버가 개별 $\Delta w_k$ 를 봐야 함 → secure aggregation과 비호환. client-level
  가치평가의 *본질적* 한계(고칠 결함 아님).
- **noise vs OOD-good** = signed value 안에서 "나쁜 노이즈"와 "분포 밖이지만 좋은 데이터"를
  분리하는 FL 방법은 *없다*(detector들 non-IID서 붕괴) → characterized limitation으로 보류.
- **non-IID bias** = FL-Shapley가 maverick/희귀도메인 client를 과소평가 → α-sweep을 필수 측정 의무로.

**🎤** "다음은 1B 본 실행 한 번입니다. 한계는 감추지 않습니다 — 셋 다 우리 방법의 성질에서
나오는 거고, 측정해서 정직하게 보고할 겁니다."

---

## 발표 1분 요약 (전체를 한 호흡으로)

> FL에서 *어느 클라이언트의 데이터가 기여했나*를 매기는 문제다(3장). 기존은 재학습·통신·
> 역헤시안의 벽에 막힌다. Flirds는 서버가 이미 받는 $\Delta w_k$ 만으로 검증손실의 1·2차 Taylor를
> 닫힌형 계산한다 — 통신 0, 역헤시안 0(4장). 이 6요소 교집합은 prior art에 비어 있고(5장),
> FL 멀티스텝이 IRDS가 버린 2차항을 되살리는 무대라는 이론이 뒷받침한다(6장). 설계는 동결됐고
> 정답 oracle을 둘 갖고 채점한다(7장). CNN에서 exact까지 검증 끝(8장), LLM 1B는 코드 완성+smoke
> 통과로 거의 끝(9장). 5도메인 free-form 데이터·utility≠downstream 평가(10장), flagship은
> "2차항이 1차항 못 잡는 공격을 잡는다"(11장). 다음 1수는 1B 본 실행, 한계는 정직하게 셋(12장).
