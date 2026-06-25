---
type: survey
title: "분류축(taxonomy) — 축별 부분집합 + 2D 교차표"
created: 2026-06-25
updated: 2026-06-25
tags: [survey, taxonomy, axes, crosstab]
---

# 분류축 (taxonomy)

각 축의 **값별로 어떤 연구가 모이는지**를 정리한다. 약어·범례는 [[prior-work-taxonomy/README|README]]. 부분집합의 겹침(straddle)·경계는 마지막 §에 모았다. 행 단위 마스터 표는 [[prior-work-taxonomy/README#마스터 한눈에 표|README]], 검증실험은 [[validation-experiments]].

---

## 축 1 — Federation setting (C / F / D) ← 핵심 축

| 값 | 연구 |
|---|---|
| **Centralized (C)** | Data Shapley · Data Banzhaf · DU-Shapley · ADS · DRDV · IRDS · DVEmb · ACC-SGD-IE · Koh-Liang IF · DataInf · TRAK · LoGra/Logix · EK-FAC · LESS · MATES · DsDm · LoRIF · DPO-Shapley · (critique) IF-Fragile · Do-IF-work-on-LLMs |
| **Federated (F)** | FedSV · ComFedSV · GTG-Shapley · S-FedAvg · ShapleyFL · SPACE · Ripple · FedIF · FedTSV · ShapFed · Mavericks/FedEMD · FedDQC · FedHDS · iPFL · RFedLR · **Flirds** · (detection) FLDetector·FLTrust·FoolsGold·FedCorr·STD-DAGMM · (critique) Shapley-Volatility |
| **Decentralized (D)** | DICE |

데이터 귀속 문헌의 **대부분은 centralized**다(특히 LLM-scale IF/선별 계열 전부). **Federated는 우리 wiki에서 가장 인구가 많은 하위 분야**지만 거의 CNN-분류 무대다. **Decentralized(peer-to-peer graph)는 사실상 DICE 하나** — non-FL 분산 귀속은 희소하다. FL은 여기 한 값일 뿐이며, 같은 방법군이 C와 F 양쪽에 변주로 나타난다(예: IRDS→Ripple/Flirds, IF→DataInf vs FedIF).

---

## 축 2 — Valuation 기반 (어떤 Shapley/influence인가)

| 값 | 연구 |
|---|---|
| **retrain** (재학습 counterfactual) | Data Shapley(TMC) · Data Banzhaf · DsDm(datamodels via retrain-fit) · ADS(state-cond) · LOO(baseline) · *Flirds (a)-oracle* |
| **in-run** (단일 궤적, 재학습 X) | IRDS · DVEmb · ACC-SGD-IE · Ripple · FedTSV · LESS¹ · **Flirds** · *Flirds (b)-oracle(exact 2ᴺ)* |
| **IF** (gradient·influence; IF/TracIn/iHVP) | Koh-Liang · DataInf · TRAK · LoGra/Logix · EK-FAC · LESS¹ · LoRIF · FedIF² · DICE(cascade) |
| **recon** (sub-model 재구성; FL-Shapley) | FedSV · GTG-Shapley · ComFedSV · S-FedAvg · ShapleyFL |
| **other** | DU-Shapley(scalar-utility) · DRDV(utility-design) · DPO-Shapley(LM-arithmetic) · MATES(1-step probe) · SPACE(KD-1round) · ShapFed(class-cosine) · Mavericks(Inf-index) · FedDQC(IRA quality) · FedHDS(dedup) · iPFL(market) |

¹ **LESS**=궤적합(TracIn over LoRA-warmup ckpt) **이자** gradient-IF — 두 칸에 걸침. ² **FedIF**=Δw에 대한 1차 TracIn(2차 없음) → in-run/IF 경계.

### 보조 표기 ① — semivalue 계열

| 계열 | 연구 |
|---|---|
| **Shapley** | Data Shapley · IRDS · FedSV · GTG · ComFedSV · S-FedAvg · ShapleyFL · SPACE · Ripple · FedTSV · DU-Shapley · ADS · DPO-Shapley · **Flirds** |
| **Banzhaf** | Data Banzhaf |
| **LOO** (degenerate) | DVEmb · ACC-SGD-IE · Mavericks(Influence-Index) · Koh-Liang/IF류(점별 marginal) · loss-heuristic(우리 floor) |
| **class-specific** | ShapFed (Class-Specific SV, 벡터) |
| **none / 비-semivalue** | DataInf · TRAK · LoGra · EK-FAC · LESS · LoRIF · DsDm · MATES · FedIF · DRDV · FedDQC · FedHDS · iPFL · detector 5종 |

> Beta-Shapley·CS-Shapley는 전용 source 노트 없음 — [[concepts/semivalue]] 프레임 페이지 참조(여기 모집단엔 행 없음).

### 보조 표기 ② — exact vs MC/근사

거의 모든 Shapley 계열이 **MC/근사**(permutation·guided-truncation·Taylor)다. **exact 2ᴺ를 실제로 쓰는 경우**는 소-N 정의/oracle뿐: Data Shapley·Banzhaf·ADS·DU-Shapley(이론/소N) + **Flirds의 (b) in-run oracle**(2ᴺ 열거, N≤10) 및 (a) retrain oracle. Flirds **estimator 자체는 1 HVP/round 근사**이고, exact (a)/(b) oracle은 *검증 기준*이다(추정≠oracle, [[validation-experiments]] E1).

---

## 축 3 — 실험 모델 클래스

| 값 | 연구 |
|---|---|
| **CNN/MLP** (소 이미지 포함) | Koh-Liang · Data Shapley · Data Banzhaf · IF-Fragile · ACC-SGD-IE · TRAK · DVEmb(MLP) · LoGra(소CNN) · **FL 전부**: FedSV·ComFedSV·GTG·S-FedAvg·ShapleyFL·SPACE·ShapFed·FedTSV·FedIF·Ripple·Mavericks·Shapley-Volatility·FedCorr·FLDetector·FLTrust·FoolsGold·STD-DAGMM · DICE(ResNet-18) |
| **LLM** (LoRA/대형) | IRDS·DVEmb·DataInf·LoGra·EK-FAC(52B)·LESS·MATES·DsDm·LoRIF(70B)·DPO-Shapley · **FedDQC·FedHDS·iPFL**(=federated×LLM이지만 valuation 아님) · Do-IF-work-on-LLMs · **Flirds**(CNN+LLM 1B/3B/7B) |
| **kernel/reg** | DU-Shapley · DRDV(NTK) · ADS(KNN exact) |
| **tabular** | (전용 축 아님; ACC-SGD-IE의 Adult, 합성-logistic 정도) — 대부분 **?** |

**이 축이 빈칸의 핵심 증거다.** LLM 칸은 전부 centralized이거나(IRDS·DataInf·LESS…), federated여도 valuation이 아니다(FedDQC=quality·FedHDS=selection·iPFL=market). CNN 칸은 FL valuation으로 꽉 차 있다. **Flirds는 두 트랙(CNN+LLM)을 모두 도는 유일 행** — CNN은 싼 고-power oracle 검증용, LLM은 표적 칸.

---

## 축 4 — 평가 단위 (granularity)

| 값 | 연구 |
|---|---|
| **sample / point** | Koh-Liang · DataInf · TRAK · LoGra · EK-FAC · IRDS · DVEmb · ACC-SGD · Data Shapley · Data Banzhaf · LESS · MATES · DsDm · LoRIF · **Ripple**(federated인데 sample!) · FedDQC·FedHDS(FL 내부 sample) |
| **client / participant** | FedSV · ComFedSV · GTG · S-FedAvg · ShapleyFL · SPACE · FedTSV · FedIF · Mavericks · RFedLR · **Flirds** · detector 5종 |
| **class** | ShapFed (client별 per-class 벡터) |
| **dataset / contributor** | DU-Shapley · ADS(ordered group) · DPO-Shapley · iPFL |
| **node** (graph) | DICE |

client 단위는 **FL에서만 의미**가 있고, client=데이터 보유자라 dataset-granularity를 상속한다. 주목할 straddle: **Ripple은 federated인데 sample-level**(FL↔centralized 가교) — 나머지 FL은 전부 client-level. Flirds는 client-level을 *의도적으로* 택했다(centralized 1-step에서 client 내 data-level Shapley 합 = client-level이라는 증명 + FL 다단계 drift residual 측정, [[flirds]]).

---

## 축 5 — 푸는 문제 (목적)

| 값 | 연구 |
|---|---|
| **eff** 효율(싼 추정) | DataInf · TRAK · LoGra · IRDS · DU-Shapley · GTG · Banzhaf(MSR) · ShapleyFL |
| **fid** 충실도(oracle 대비) | GTG · ComFedSV · FedSV · **Flirds(1차)** · (반례로) Shapley-Volatility |
| **fair** 공정·보상 | ADS · ComFedSV · DU-Shapley · iPFL(IR/IC) · Mavericks · FedTSV |
| **agg** 집계·모델품질 | FedIF · FedTSV · ShapFed · S-FedAvg · (detector) FLTrust·FoolsGold·FedCorr |
| **det** 강건성·악성검출 | FLDetector · FLTrust · FoolsGold · FedCorr · STD-DAGMM · S-FedAvg · ShapleyFL · Koh-Liang |
| **sel** 데이터 선별 | LESS · MATES · DsDm(centralized) · FedHDS(federated) · DPO-Shapley |
| **mkt** 데이터 마켓 | iPFL · DRDV · ADS · DPO-Shapley · Banzhaf(noise-robust 가격) |
| **attr** 귀속·해석 | IRDS · EK-FAC · DataInf · DVEmb · DICE |
| **quality** 품질평가(≠value) | FedDQC |

**관찰**: 최근 federated 다수가 valuation을 *집계 가중치*로 흡수(`agg`: FedIF·FedTSV·ShapFed·S-FedAvg) — fidelity(E1)를 건너뛰고 E7로 평가한다. valuation-fidelity를 1차로 두는 federated 연구는 소수(FedSV·GTG·ComFedSV·Flirds). LLM 칸의 federated 연구는 목적이 **sel/quality/mkt**일 뿐 fid가 아니다.

---

## 축 6 — 보조 축 (통신 / partial participation / val-set·root)

| 보조축 | 값별 연구 |
|---|---|
| **통신 오버헤드** (vanilla FedAvg 대비) | **0**: GTG · ShapleyFL · Ripple · FedIF · **Flirds** (이미 받는 Δw만 사용) — `0` 표기. **+서버 util-eval**: FedSV(O(Tm²)). **+전원 라운드**: ComFedSV("Everyone Being Heard"). **+서버 val-train pass/round**: FedTSV. **submodel/full-model 교환**: SPACE(full model 1회)·iPFL·FedHDS(LoRA)·DICE(gossip) |
| **partial participation** | *명시 처리/요구*: ComFedSV(low-rank completion으로 대칭 복원) · ShapleyFL(importance-sampling) · FedHDS(5% 샘플) · **Flirds**(정규화 없이 그대로, tier 내 순위 + (b) exact per-round 분해). *cohort만*: FedSV·GTG·S-FedAvg·FedTSV(비참여=zero/carry). *비처리/한계 명시*: FedCorr(late-joiner 오분류). centralized는 **n/a** |
| **val-set / server-root 필요** | *서버 val-set*: FedSV·GTG·ComFedSV·ShapleyFL·FedTSV·**Flirds**·LESS·MATES 등 대다수. *trusted root*: FLTrust(root-cosine). *validation-free*: DRDV(Wasserstein)·SPACE(prototype)·FedDQC(on-device IRA)·FoolsGold·DICE·DVEmb |

Flirds의 **0 통신**(추가 통계·SCAFFOLD式 보정 없음)이 GTG/FedSV와의 실무 차별점이고(FedSV는 "0 통신이지만 O(Tm²) 서버 util-eval", ComFedSV는 전원 라운드 추가), 동시에 **개별 Δw를 봐야 함 → secure aggregation과 비호환**이라는 본질적 한계(client-level valuation 공통; FLTrust의 root-cosine과 대비).

---

## 2D 교차표 (부분집합 지형)

### 교차표 A — Federation × 모델 클래스

| | CNN/MLP (소 이미지) | LLM (LoRA/대형) | kernel/reg·tabular |
|---|---|---|---|
| **Centralized** | Koh-Liang · Data Shapley · Data Banzhaf · TRAK · IF-Fragile · ACC-SGD · (DVEmb·LoGra 소규모) | IRDS · DataInf · LoGra · EK-FAC · LESS · MATES · DsDm · LoRIF · DVEmb · DPO-Shapley · Do-IF-work-on-LLMs | DU-Shapley · DRDV · ADS |
| **Federated** | FedSV · ComFedSV · GTG · S-FedAvg · ShapleyFL · SPACE · ShapFed · FedTSV · FedIF · Ripple · Mavericks · Shapley-Volatility · FLDetector · FLTrust · FoolsGold · FedCorr · STD-DAGMM | **▣ valuation 빈칸** — 점유자는 *비-valuation*뿐: FedDQC(quality) · FedHDS(selection) · iPFL(market). **→ client-level FL valuation = 없음 → [[flirds\|Flirds]]가 겨냥** | ? |
| **Decentralized** | DICE | — | — |

핵심 셀 = **federated × LLM × valuation**. 현재 그 칸엔 quality(FedDQC)·selection(FedHDS)·market(iPFL)만 있고 **기여도 값을 oracle 대비 충실도로 검증한 연구는 관찰되지 않는다**(표적 웹 2024–2026 포함, [[prior-work-taxonomy/README#핵심 관찰 (landscape, 담백하게)|README 핵심 관찰]]). Flirds는 이 칸에 client-level Shapley *valuation*을 처음 놓는 시도이며, CNN 트랙도 병행(싼 고-power oracle 검증).

### 교차표 B — Federation × Valuation 기반

| | retrain | in-run / trajectory | IF / end-state-local | recon / coalition-surrogate |
|---|---|---|---|---|
| **Centralized** | Data Shapley · Data Banzhaf · DsDm · DU-Shapley · ADS · LOO | IRDS · DVEmb · ACC-SGD · LESS¹ | Koh-Liang · DataInf · TRAK · LoGra · EK-FAC · LESS¹ · LoRIF · MATES² | — |
| **Federated** | *(a) retrain oracle*³ | **Ripple · FedTSV · Flirds**·(critique)Shapley-Volatility | FedIF · ShapFed(class-cosine) · FLTrust(root-cosine) | FedSV · GTG · ComFedSV · S-FedAvg · ShapleyFL · SPACE · *(b) in-run exact oracle*³ |
| **Decentralized** | — | — | DICE (influence cascade) | — |

¹ LESS 두 칸 걸침(궤적합 + gradient-IF). ² MATES=1-step probe(in-run↔retrain 경계). ³ Flirds의 (a) retrain·(b) in-run-exact는 *경쟁 방법이 아니라 검증 oracle*. **federated×in-run** 칸의 메서드는 Ripple(sample-level, 2차 없음)·FedTSV(집계지향, closed-form/Hessian 없음)뿐 → **client-level + closed-form 1st+2nd Taylor + true-Hessian**을 동시에 채우는 federated 행은 Flirds가 유일(관찰 범위). Shapley-Volatility는 *방법*이 아니라 근사-FL-Shapley의 불안정성을 보인 **동기 논문**이므로 괄호 표기.

---

## Straddle · 경계 (정직하게)

- **LESS** — LLM·LoRA·val-gradient 정렬 = Flirds의 **가장 가까운 centralized 사촌**. 단 *centralized·per-example·Adam*이라 drop-in client-level baseline 아님(cosine이 집계 선형성을 깸). "LESS는 IRDS와 Flirds 사이"([[threads/data-selection-for-llms]]).
- **Ripple** — federated×in-run을 *먼저* 점유. 단 **CNN-scale·sample-level·1차+재귀-Jacobian(2차 Hessian 항 없음)·GT-SV 충실도 미측정**. Flirds의 novelty는 "최초 federated in-run"이 아니라 *교차점*(client-level + 1st+2nd + HVP 상호작용항 + 0통신 + LLM).
- **FedIF / FedTSV / ShapFed / S-FedAvg** — 궤적/gradient 신호를 *집계 가중치*로 사용(robust-aggregation). valuation accountant 아님 → E1(충실도) 생략, E7로 평가. valuation↔aggregation 분리가 load-bearing.
- **FedDQC / FedHDS** — federated×LLM이지만 각각 **quality(per-sample IRA)** / **selection(dedup)**. client별 기여도 값 없음. FedDQC는 "유일한 FL+LLM 선례"로 불리나 *다른 문제*이고, DataInf가 실세계 FL서 random보다 못하다는 음성결과를 보고([[sources/feddqc]]).
- **DICE** — 유일한 decentralized. 단 node-level·CNN(ResNet-18, 16-node)·경량 empirics·node→sample 미해결. FL은 star 특수경우(Alg.1).
- **iPFL / DPO-Shapley** — LLM-scale + 협력게임이지만 iPFL은 *model*-market(graph-game), DPO-Shapley는 *centralized*·DPO-specific(SFT로 이식 불가).
- **ADS** — FL/multi-stage가 *동기 예시*일 뿐, 방법 자체는 structure-aware centralized(통신 프로토콜 없음). symmetry 공리를 버림(Banzhaf는 efficiency를 버림 — [[threads/symmetry-and-asymmetry-axioms]]).
- **Mavericks** — valuation *추정기*가 아니라 FL-Shapley의 rare-client 과소평가를 *증명*한 공정성 분석 + 선택(FedEMD). "FL-Shapley가 maverick을 under-credit"의 정전 인용.
