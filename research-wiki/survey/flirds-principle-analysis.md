---
type: survey
title: "Flirds 원리 분석 — 승패 메커니즘과 논문용 성공 세팅"
created: 2026-07-22
updated: 2026-07-22
sources: [flirds-experiment-results-overview, irds-fl-math-rigor-2026-07/irds-fl-math-rigor]
tags: [survey, principle, mechanism, win-loss, paper-settings]
---

# Flirds 원리 분석 — 승패 메커니즘과 논문용 성공 세팅 (2026-07-22)

> **지시 원문**: 루트 `PRINCIPLE_ANALYSIS_PROMPT_2026-07-22.md` (Yonghee).
> **목적**: flirds가 잘 할 수 있는 일의 **원리**를 이해해, 성공적으로 실험할 수 있는 세팅을
> 찾는다. 실험 수가 회귀분석엔 부족하므로 승패의 **메커니즘 해석**이 핵심 가치다.
>
> **독립성 규약 준수 방식**: overview의 표 **수치·세팅·출처 블록만** 데이터로 사용했고,
> "읽기"·"판정" 불릿과 §5.1 종합판정은 1차 분석에서 차용하지 않았다(§2.7에서 사후 대조).
> **논거로 쓴 수치는 전부 원 파일(rundir metrics.json / phi.parquet / analysis CSV)에서
> 재검증**했다 — 검증 스크립트·리포트 = `flirds-principle-analysis/verify_numbers.py` →
> `verification_report.txt`, **237 PASS / 0 FAIL** (부록 A). 메커니즘 서술은 방법 이름이
> 아니라 `codes/flirds/` 실코드를 정본으로 했다(§2.1). 이론 유도는
> [[irds-fl-math-rigor-2026-07/irds-fl-math-rigor]]의 P1–P8을 사용했다.
>
> **제약 준수**: `runs/`·rundir 무수정(read-only) · GPU 실행 0 · `paper/` 무수정 ·
> 수치는 파일에서만 · 커밋은 로컬만.

---

# 1. [1단계] 실험 요소 전수 인벤토리

> 형식: `요소 | 취한 값(스테이지 병기) | 미탐색 값/영역 | 이 요소를 바꿨을 때 관찰된 변화(수치+출처) | 교락 주의`.
> 커버리지: overview §2 마스터표 25행+계획행 전부 + rundir `config.yaml` 샘플링
> (track_d/phase2/track_h-gsm/track_c/track_g/probe/removal)으로 발굴한 암묵 고정 차원 포함.
> 출처 표기는 overview 섹션 번호(수치는 원 파일 재검증 완료 — 부록 A).

## 1.1 모델 축

| 요소 | 취한 값(스테이지) | 미탐색 | 관찰된 변화 | 교락 주의 |
|---|---|---|---|---|
| 구조 | CNN(LeNet5/FedSVCNN, from-scratch 전체학습: track_c·probe·track_g/h CNN) · LLM(LoRA PEFT: 나머지 전부) | ViT·중간 규모 아키텍처, RNN | 구조가 아니라 **레짐**(from-scratch 대스텝 vs finetune 소스텝)이 갈랐다: CNN은 same-game 방법도 분화(Flirds .919 vs 1st .832, §3.1.2), LLM은 동률 포화(§3.1.1) | 구조↔스텝 크기↔가산 갭이 완전 교락 — "CNN이라서"가 아니라 "비-가산이라서"일 가능성이 §2 원리 2의 핵심 |
| LLM 스케일 | 1B/3B/7B (track_d 6셀) · robustness는 1B(+3B 1-seed) | 7B robustness(P4)·13B+ · 3B/7B (a)oracle | fidelity 불변(전 스케일 1.000, §3.1.1); **(b) 타깃 자기안정성은 7B서 상승**(IID-clean xseed ρ: 1B −0.367 → 7B anchor5 **+0.733**, §5.4 재검증) | 7B만 Llama-2-hf base(1B/3B는 3.2-Instruct) — 스케일과 세대·베이스/Instruct가 교락 |
| CNN 폭 w | 0.5/1/2/4 (probe §4.3) | w>4 | **무변화**: iid xseed ρ 0.03~0.12 평평, φ range 평평, 개입이득 Δ0.087~0.092 폭 무관 | raw acc는 폭에 오름(0.622→0.673) — 동작점 이동을 효과로 오독 금지 |
| base/Instruct | Instruct(1B/3B, gsm) vs base(7B) | 같은 스케일 A/B | 변화 미관측(비교 설계 없음) | 스케일 축과 교락(위) |
| LoRA rank | 16/32/64 (probe §4.2; 본선 r16 α=32) | r<16, r>64, α≠2r | **무변화**: (b)φ range 0.00102~0.00119 평평, fidelity 1.000 유지 | seed0만(rank probe) |
| 정밀도 | fp32 본선; TF32 A/B(CNN §6.3); bf16/TF32 microbench | bf16 **훈련**(운영 레짐) 전체 파이프라인 | 결과 무변화(TF32 on/off acc 차 ≤0.001·Flirds spearman 비트동일), **비용 배율만**(fp32가 fwd×5.33/HVP×4.09 느림, §3.4.1) | "같은 게임" 주장이 fp32-로그 전제 위(수학 문서 §2.1) |

## 1.2 최적화 축

| 요소 | 취한 값 | 미탐색 | 관찰된 변화 | 교락 주의 |
|---|---|---|---|---|
| optimizer | plain SGD mom=0(전 본선) · **AdamW 상수-lr 브리지**(removal_dose D) | momentum>0, 서버 optimizer(FedAvgM/FedOpt), 논문레시피 AdamW 5e-5+cosine | **AdamW서 Taylor 계열만 하락**: vs (b) loss-heur +0.967 > Fed-LOO +0.933 > GTG +0.900 > **Flirds +0.767 > Flirds-1st +0.700**; (a)↔(b) 자체 **−0.53**(전 seed 음수) (§4.6 재검증) | (a)↔(b) 게임 괴리와 Taylor 열화가 같은 셀에서 동시 발생 — 분리 서술 필요(§2.5 원리 8) |
| lr | LLM 1e-3(본선)·{1,2,3}e-3 격자(§4.2)·phase1 {1e-4..3e-3} · CNN 0.01 | lr 스케줄(cosine/decay) | φ **공통 shift ~1.3×만**(클라 간 분리는 seed 분산에 묻힘), fidelity 전 칸 1.000, xseed ρ 개선 없음(−0.37~−0.20); phase1 AUROC lr-반전(1e-3: noisy .75/FR 1.0 ↔ 3e-3: 1.0/.75, §6.1) | lr↑=스텝 크기↑=Taylor 잔차↑인데 fidelity 불변 = "tradeoff 없음" 관측의 근거 |
| local steps | 10(본선)/20/30(격자) | steps<10 | **무변화**(φ range·fidelity·xseed 전부) | seed0만(st20/30) |
| batch/seq | 16/512(std)·768(silo)·GSM 512; 7B batch 4 | 변주 없음(암묵 고정) | 변화 미관측 | 7B만 batch 4 — 스케일과 교락 |
| 라운드 R | 10(silo)/30(anchor·device)/50(phase1)/120(C2)/200(std20·std50k5·gsm) | R>200 | 단독 효과 미관측 — **클라당 참여 횟수**(1.3)와 완전 교락 | R↑=참여↑=EMA-계열 괴리↑(§2 원리 4)도 동시 발생 |
| warmup | 0(track_d)/2(silo)/3(device·gsm) | 통제 비교 없음 | 변화 미관측 | 암묵 고정 |

## 1.3 FL 구조 축

| 요소 | 취한 값 | 미탐색 | 관찰된 변화 | 교락 주의 |
|---|---|---|---|---|
| 클라 수 N | 5(silo·anchor·phase1)/10(C1·E5)/20(std20)/50(std50k5·gsm)/100(C2·device100) | N>100, N=2~4 | N 자체보다 **참여형태**가 지배(아래); N=5는 순위지표 해상도 제약(1스왑=Sp 0.1) | N↔참여율↔오염비율 동시 변동 셀 다수 |
| 라운드당 참여 k/N | full(silo·anchor·C1·scale 100/100) · 2/20 · 5/50 · 10/100 · CNN {2,5,10}/10 · {5,10}/100 | k/N<5%, 시간가변 참여 | **방법 분별기**: full서 동률이던 방법들이 partial서 갈림 — std50k5 Flirds 1.000 vs FedIF −0.040/ShapleyFL −0.064/ComFedSV −0.109(§4.2); CNN k=0.2 Flirds .891 vs 1st .305(§4.3) | 참여율과 클라당 참여횟수(아래)를 분리해 읽어야 |
| **클라당 누적 참여 횟수** R×k/N (파생) | ~2회(CNN k0.2)/~10회(silo R10)/~20회(std20·std50k5·gsm)/30회(anchor)/전 라운드(full) | 1회 참여 극한 | **1차항 정확도의 실제 조건**: ~2회면 Flirds-1st 붕괴(.305), ~20회면 1차도 1.0 유지(§4.2·§4.3 caveat) | "참여 분수"로 잘못 요약하기 쉬움(overview §6.2-10과 동일 결론) |
| 집계 가중(등n) | **전 LLM 셀 등n**(alpaca 20k/N·GSM 149/클라·silo 200/클라) · CNN dir1/quantity_skew만 비등n | **LLM 비등n**(P5c 트리거!) | CNN quantity_skew서 renorm·1차-정규화 계열 붕괴(FedIF **−0.20/−0.07**, §3.1.2) — 비등n이 P5c 예측대로 분화 유발 | LLM 쪽 P5c 무대가 전무 = 이론 반례를 실증할 칸이 비어 있음(§3 제안 F-L2) |
| silo/device | cross-silo(N≤10 full) vs cross-device(N=100 부분) | 계층적/비동기 FL | device는 (b) 타깃 자체가 참여복권에 지배(anchor frrand/frzero xseed **+0.12**, §5.4 재검증) | device fidelity는 truth=proxy 칸 다수(§6.2-3) |
| 오염 클라 비율 | 20%(silo 1/5)/5%(device 5/100)/**40%**(R4 20/50, CNN R1 미러)/ladder(C1) | 과반(>50%) | 비율 자체의 통제 비교 없음 — dose와 교락 | R4가 40%로 유독 높음(무대 갭 확보용) |
| 오염 정체 동학 | 정적(전부) vs **매 라운드 재추첨**(Dyn §4.8.3) | 점진 드리프트 | 재추첨이면 클라-수준 신호 소멸(P1 적중률 .405≈우연 .40) — 게이트 해악(gn −.078) | null-무대 설계 자체가 결론(도구 한계 명시용) |

## 1.4 데이터 축

| 요소 | 취한 값 | 미탐색 | 관찰된 변화 | 교락 주의 |
|---|---|---|---|---|
| 데이터셋 | alpaca-gpt4(LLM 본선)/GSM8K(R4)/MMLU·alpaca-test(평가)/mnist·cifar10(C1)/fmnist(C2) | 코드/수학 외 도메인, 실제 기관 데이터 | GSM8K 전환으로 selection 변별력 확보(vanilla↔oracle 갭 +3.6pt vs alpaca parity, §3.2.4); cifar10은 mnist보다 오염 실효 큼(removal acc 분리 13×, §4.4.2) | 데이터셋↔심판 지표(EM vs loss)도 함께 바뀜 |
| 분배 | IID(std·iid5·C2 iid)/domain-silo 5종(silo5)/Dirichlet α∈{0,0.01,0.1,0.5,5}/shard/dir1/label·quantity skew(C1) | 극단 label-skew LLM, 실제 기관 비IID | **비IID가 신호를 만든다**: silo5 clean (b) xseed **+0.87** vs iid5 clean **+0.13**(§3.1.5 재검증) — 오염 0이어도 도메인 분리만으로 타깃 순위 실재 | α-sweep 탐지축은 truth=proxy·tiny val과 교락 |
| per-client 량 | 149(gsm)~6k(mnist) 다양 | 클라 간 극단 불균형(LLM) | 량 자체 통제 비교 없음(등n 고정) | 1.3 등n 행과 동일 공백 |
| val 크기·구성 | 200(track_d·gsm)/20(silo)/10(device)/2000(CNN) — 전부 서버-측 held-out, 학습분포서 카브 | val 분포 shift, val 오염 | val=20 노이즈가 φ spread와 동급(spread/SE 1.1~2.3, §4.2 b4) — within-seed 순위는 강건(boot ρ 0.9+) | tiny val은 AUROC coarse의 원인(§6.2-7); noisy nr0 대조군 AUROC 0.83(무신호 기준선≠0.5, §4.5) |

## 1.5 위협 축

| 요소 | 취한 값 | 미탐색 | 관찰된 변화 | 교락 주의 |
|---|---|---|---|---|
| 유형 | noisy=answer-swap(LLM)/label-flip(CNN·R4 noisy)/grad-noise 상대(CNN·구 R4)·**GN_ABS 절대(신 R4, 재실험 중)**/**frzero·frrand·frdelta**/feature-noise/label·quantity-skew; poison=제외 결정(기록만, §6.2-8) | 적응형(탐지-회피) 공격, 협조 공격 | 위협 유형이 **어느 방법이 보이는지를 결정**: 0-평균 방향 위협(GN)은 1차-내적 실명, zero-delta는 exact-0 계열 전유, descent-정렬(frdelta)은 전 valuation 공통 실패(§2 원리 3·5·7) | 유형↔dose↔무대 조합이 셀마다 다름 — 단일 축 결론 주의 |
| dose | noisy nr {0,0.1,0.25,0.5,0.75,1.0} · FR 배율 dm {0.25..4} · lf {0.15,0.35,0.70} · GN γ | nr>1(외삽 영역), GN_ABS γ 격자(재실험 중) | noisy 탐지 문턱 nr0.25(0.75→1.00); FR은 전 배율 1.00(§4.5); φ(noisy) 부호는 전 dose 기여-양수(−0.0025→−0.0019, 0-교차 외삽 nr≈3.4 — 부록 A Q-3) | dose ladder는 silo5(N=5 coarse AUROC)서만 |
| binary/graded | binary(대부분) + per-client graded ladder(C1 label_flip·feature_noise → `spearman_vs_rate`) | graded LLM | φ가 오염강도 순위를 추적(C1 lf: (b) .968·Flirds .960, §3.3.5) | ladder 셀만 AUROC 정의됨 |

## 1.6 valuation 축

| 요소 | 취한 값 | 미탐색 | 관찰된 변화 | 교락 주의 |
|---|---|---|---|---|
| 게임 정의 | **(b) in-run 고정가중**(D1·D2; 본선) vs **(a) retrain 재정규화**(1B anchor5·C1·A1/D) vs Flirds-proxy(device off-anchor) | (a) LLM 3B/7B(⬚ 확정), 비등n (a) | (a)↔(b): SGD +0.933(등n·near-additive 축퇴 지역), AdamW **−0.53**; CNN vs(a) 전 방법 0.2~0.45 — **vs(a) 1위는 ShapleyFL .453(uniform-subset 계열)이고 Flirds .352**(§3.1.2 재검증) = renorm-족 방법이 renorm-족 게임을 더 잘 추정(P5 정합; §2.7-6) | "(a)가 진실"이라는 관점 채택 여부가 서사를 좌우 — 게임 선택은 정의 문제(P6) |
| 채점 시점 | post-hoc from-logs(fidelity 전부)/online 게이트(V1/V2/z/cgate/pweight)/observer(비트동일 병행)/T2 최종-부호 재학습/V3 | 중간-체크포인트 재채점 | R4: T2가 T1보다 정확(noisy .3584 vs .3530; clean 무해 vs −1.0pt)(§3.2.4); CNN lf: retrain .62대 > online .57대(§3.2.3) | online은 개입이 궤적을 바꿔 자기-교란(관찰자와 분리 필요 — 이미 설계됨) |
| EMA β | flirds_w 0.5 / ShapleyFL 0.3(신규가중 0.7) / FedIF γ 0.3(신규가중 0.3) / β0.3 재실행 캠페인 | β sweep 본격 | β0.5→0.3 효과는 재실행 노이즈 수준(3B 대조 ρ 0.90–1.00, §4.7) — 단 **EMA 존재 자체**가 plain-sum 게임과의 괴리 기전(§2 원리 4) | CNN 120셀 β-era provenance 미확보(§6.2-9) |
| 게이트 HP | τ=0(sign)/z=1.645 보편상수/burn-in·probation(V2)/min_obs | 셀별 튜닝(금지 설계) | P5(신뢰-게이트)가 P1 clean 오발화 대부분 회수(.596→.630~.639, §4.8.1) | parameter-free가 주장 포인트 — HP 튜닝 도입 시 서사 훼손 |
| 2차항 | on(Flirds)/off(Flirds-1st) 전 트랙 병렬 | 3차(불필요 판정: 잔차가 3차항의 21~37×, §5.5) | §2 원리 3 참조(GN·저참여서만 가치) | — |

## 1.7 측정 축

| 요소 | 취한 값 | 미탐색 | 관찰된 변화 | 교락 주의 |
|---|---|---|---|---|
| 지표 | Spearman/Kendall/Pearson/cosine·euclid·max_diff/AUROC/EM/절대 acc/val-loss/recovery/xseed ρ/topJ·botJ/rounds-to-target | 캘리브레이션 지표, 하위셋 정밀도@k | 지표가 구조적으로 보는 것이 다름(§2.6 교락 D-8): 순위상관은 값 크기 못 봄(ShapleyFL max_diff .98이어도 Sp 0.7 가능, §3.1.1) · N=5 Spearman 1스왑=0.1 · AUROC 무신호 기준선 0.5 아님(nr0서 0.83) · recovery는 분모 갭 작으면 >1 폭발(§3.2.2 ¹) · val-loss 4째자리는 seed-분산에 묻힘(짝지은 delta 필요) | — |
| seed 수 | 3(본선 대부분)/1(3B silo5·E5·R4 Tier A·rank probe·st20/30) | ≥5 seed | seed가 (b) 타깃 자체를 재추첨(IID-clean, §5.4) — seed 수는 단순 분산 문제가 아니라 신호 실재성 판정 도구 | 1-seed 셀 목록은 §2.6 D-2 |
| truth | exact (b) 2^N/2^k·per-round/(a) retrain/Flirds-proxy/corrupt 마스크 | — | truth 선택이 결론을 좌우: proxy 칸의 1.000은 "Flirds와 동일"의 뜻(§6.2-3) | — |

## 1.8 비교군 축 — 메커니즘 클래스 (코드 정본; §2.1에 상세)

| 요소 | 취한 값 | 미탐색 | 관찰 | 교락 |
|---|---|---|---|---|
| 비교군 | same-game 5(Flirds/1st/loss-heur/Fed-LOO/(b)) + renorm-재구성 2(GTG/FedSV) + uniform-후처리 3(ShapleyFL/ComFedSV/S-FedAvg) + 1차-정규화-후처리 1(FedIF) + 탐지기 4(FLDetector/FLTrust/STD-DAGMM/FedDQC); Banzhaf·Ripple 제외 결정(데이터 존속) | DataInf/LoGra류 IF-계열 최신, 개인화-FL valuation | 그룹별 승패가 §2의 골격 | 그룹 내 이질성(GTG의 efficiency-norm·round-trunc는 FedSV에 없음) 주의 |

## 1.9 암묵 고정(발굴) — 명시 안 된 채 고정돼 온 것

서버 lr 없음(순수 가중합, F6) · 클라 stateless(F10) · 균등 비복원 클라 추출(F2) · val을 학습분포에서 카브(도메인-매칭) · 등n(1.3) · 오염 클라 인덱스 고정(silo noisy=c0, FR=c1) · fp32 로그 · LoRA target 모듈 7종 · greedy 디코딩(EM) · seed_everything 후 adapter-init 이슈(H1, §6.2-13: 절대값 재현 불가·순위 결론 강건 예상) · MC baseline seed=0 고정(GTG/FedSV rng). **이 중 논문 리스크가 큰 것**: 등n(비등n 무대 부재), H1(절대값 재현), (a) 스케일 공백.

---

# 2. [2단계] 승패 원장 + 원리 분석

## 2.1 메커니즘 그룹핑 (코드 정본 — `codes/flirds/`)

방법 이름이 아니라 **(게임, 추정기, 후처리)** 3요소로 분류한다. 이하 본문 설명은 전부 이
그룹 수준이다.

| 그룹 | 방법 | 게임(무엇을 평가) | 추정기 | 후처리/누적 | 코드 근거 |
|---|---|---|---|---|---|
| **G1. same-game Taylor** | Flirds(1+2차), Flirds-1st(1차) | (b) 고정가중 D1·D2 그대로 | 라운드별 Taylor 절단(닫힌형 Shapley=P2; 2차는 1 HVP/round) | plain-sum(라운드 합), 정규화 없음 | `core/flirds_estimator.py:97-131` |
| **G2. same-game 실평가** | loss-heur(singleton U({k})), Fed-LOO(U(P_r)−U(P_r∖k)), (b)oracle(exact 2^K) | (b) 동일 | 0차(실제 forward 평가) — 절단 없음 | plain-sum | `oracle/in_run_sv.py:101-124,70-98,182-222` |
| **G3. renorm-재구성 MC** | GTG, FedSV | **재정규화 게임** ũ(S): subset 내 n-renorm 재구성(c_S 증폭, P5) | 라운드별 MC Shapley(GTG=guided 절단+efficiency 정규화+round-trunc; FedSV=순열 MC+TMC) | plain-sum | `baselines/gtg.py:18-28,76-85,145-166`, `fedsv.py:17-60` |
| **G4. uniform-재구성+min-max+EMA** | ShapleyFL | uniform 평균 submodel(1/\|S\|, n-무관) | 라운드별 exact Shapley | **min-max→[0,1] + EMA(β=0.3: 신규 0.7 가중)** — plain-sum 아님 | `shapleyfl.py:58-73,100-113` |
| **G5. 1차-정규화+min-max+EMA** | FedIF | 1차 내적 −⟨g,Δ_k⟩/‖Δ_k‖ (단위 정규화; p_k 없음) | 해석적(1 grad/round) | **min-max + EMA(γ=0.3: 신규 0.3)** | `fedif.py:64-99` |
| **G6. 저랭크 완성** | ComFedSV | uniform submodel + 손실감소 효용 | 관측 prefix→ALS 저랭크 완성 후 Shapley 독출 | 완성 행렬 의존 | `comfedsv.py:1-60` |
| **G7. 전용 탐지기** | FLTrust(root-cosine=정규화 1차), FLDetector(L-BFGS 업데이트 일관성), STD-DAGMM(업데이트 AE+GMM), FedDQC(IRA 데이터품질) | 각자 다른 질문(기여도 아님) | — | — | `baselines/{fltrust,fldetector,std_dagmm,feddqc}.py` |
| (선택기) | S-FedAvg | uniform submodel MC-relevance로 softmax 선택 | 온라인 | α0.75/β0.25 EMA | `sfedavg.py` |

핵심 분류 원리: **G1·G2는 같은 게임의 다른 범함수**(순위 차이는 게임의 비가산성에서만 발생
가능), **G3~G6은 다른 게임 또는 다른 누적**(순위 차이가 구조적으로 가능), G7은 다른 질문.
"exact-0 계열"(overview 용어) = G1+G2+G5의 raw(단, G5는 후처리 전 raw만; FedIF min-max 후에도
Δ=0→infl=0이 유지되는 건 분자 0 덕) — frzero bit-exact 0 재검증: (b)/Flirds/1st/loss-heur/
FedIF/Fed-LOO 모두 0.0, renorm 3종은 \|φ\|~0.004–0.005(부록 A Q-2; **clean 클라 φ와 동급
크기**의 유령값이라는 점이 중요하다 — ε-오차가 아니라 null-player 파괴).

## 2.2 이론-우선 유도 — 어떤 조건이면 갈라질 수 없고, 어떤 조건이면 갈라질 수밖에 없나

데이터를 보기 전에 정의에서 유도한 조건들(T). 각각 §2.4의 원리 주장에서 관측과 대조된다.
수학 근거는 [[irds-fl-math-rigor-2026-07/irds-fl-math-rigor]] P1–P8.

- **T1 (가산 축퇴 → 동률 불가피).** 라운드 게임이 가산(u_r(S)=Σ_{k∈S}u_r({k}))에 가까우면
  Shapley=LOO=singleton=모든 semivalue가 같은 순위로 붕괴한다(P2·P5: 1차 surrogate는 정확히
  가산; 비가산성 전체 = 클라 간 교차곡률 Σq_ij). LoRA 미세조정의 소스텝(‖a‖ 작음)은 q_ij≈0을
  만들므로 **G1·G2 전원의 순위 동률은 예측이지 발견이 아니다**. 이 조건에서 same-game 방법 간
  fidelity 우열 주장은 수학적으로 불가능하다.
- **T2 (신호 존재 조건).** 순위 fidelity가 정보를 가지려면 타깃 순위 자체가 실재해야 한다.
  클라들이 교환가능(IID·clean·등n)하면 (b) 타깃의 순위는 미세 노이즈의 순서 = seed 재추첨마다
  다른 추첨이다. 이때 per-seed +1.000은 "frozen log의 정확 재현"만 의미한다(그 자체는 추정
  품질 증거로 유효하나, 클라 우열 서사로는 무정보).
- **T3 (renorm 게임 분화 트리거 — P5).** 고정가중 vs 재정규화 게임의 순위는 (i) 비등n(1차·
  정확가산에서도, P5c), (ii) 부분참여(등n·선형이어도 라운드-합 리프트 실패, P5b-Remark),
  (iii) 2차 곡률(c_S² 증폭; 등n·H-직교여도) 중 **하나만 있어도** 갈린다. 추가로 재정규화
  게임에선 **zero-delta 클라가 null-player가 아니다**(n_FR이 분모에 들어가 타 클라를 희석 →
  마진 ≠0) — exact-0 상실은 근사 오차가 아니라 게임의 성질.
- **T4 (MC/절단/정규화 = 추정 분산 추가).** G3의 MC 표집·절단·efficiency 정규화는 같은 로그
  위에서도 비결정론·비선형 왜곡을 더한다 → seed 간 재현성이 (b) 자체보다 나빠질 수밖에 없다.
  G1·G2는 로그가 주어지면 결정론 → (b)의 내재 안정성을 상속한다.
- **T5 (min-max+EMA = 스케일·plain-sum 파괴).** G4·G5의 라운드별 min-max는 크기 정보를
  버리고 라운드 내 상대순위만 남긴다 — **K_r=2면 {0,1} 이진 양자화**다. EMA는 plain-sum이
  아니라 최근-참여 가중이므로, (b)의 라운드-합 게임과 다른 대상을 계산한다. 괴리는 (라운드 수
  ×참여 희소성)에 비례해 커진다: 부분참여·장기 R에서 최댓값. 따라서 G4·G5의 부분참여 붕괴는
  1차 신호의 실패가 아니라 **후처리의 실패**로 예측된다(판별 실험: 신호 있는 full-참여
  무대에선 FedIF↔Flirds-1st 순위가 일치해야).
- **T6 (1차 vs 2차 분리 조건).** 1차 항 p_k⟨g,δ_k⟩는 val-gradient와의 정렬만 본다. 등방
  노이즈 δ는 E⟨g,δ⟩=0이라 **1차-내적 계열(G1의 1st·G5·FLTrust)에게 구조적으로 안 보인다**.
  2차 항 ½p_k⟨δ_k,HΔW⟩ ⊃ ½p_k²δᵀHδ>0(PSD 방향 곡률 에너지)는 노이즈의 손실 상승을 포착한다.
  0차 실평가(G2·G3)도 ℓ을 직접 재므로 포착한다. 또 1차 절단 오차는 참여 횟수가 many면 순위
  수준에서 상쇄되고 few면 남는다 → **2차항의 가치 영역 = {0-평균 방향 위협} ∪ {저참여·대스텝
  레짐}**으로 예측.
- **T7 (0-의미론 → 게이트 작동영역).** sign-게이트(cum>0)의 실효성은 위협이 누적 φ를 0
  아래로 내리는가에 달려 있다: zero-delta(FR)는 exact-0(P1 null-player)로 즉시·무오차,
  0-평균 노이즈는 2차항으로 음수화 가능(CNN GN), **descent-정렬 오염(mislabel이어도 공통
  특징을 내리는 데이터)은 φ가 양수로 남아 게이트가 침묵하는 게 게임의 정직한 답**이다.
- **T8 (기여도≠탐지 — 공리적 갭).** φ는 val-loss 변화의 귀속이다. val-loss를 실제로 낮추는
  위협(frdelta 등)은 게임의 답이 "기여함"이므로 φ-as-detector는 (b) oracle과 함께 실패해야
  정상이다. 역으로 데이터-품질/업데이트-패턴 축 탐지기는 다른 질문에 답하므로 φ와 승패가
  갈릴 수 있다.
- **T9 (비용 구조 — 라운드당).** Flirds 1 HVP(microbench: 10.36s ≈ **6.5 fwd**), Flirds-1st
  1 grad(실측 ≈2.2 fwd), loss-heur/Fed-LOO 1+K_r fwd, (b) 2^{K_r} fwd, (a) 2^N×R 재학습.
  ⇒ 교차점 예측: (b) 대비 Flirds 우위는 2^{K_r}>6.5 즉 **K_r≥3**부터, K_r=2(std20)선 역전;
  singleton류 대비는 K_r≥6부터. 비용 우위는 무대 성질(cohort 크기)의 함수다.
- **T10 ((a)-게임과의 괴리).** (a)는 재정규화+경로+참여 재추첨이 전부 다른 게임(P6): R=1
  비등n 반례로도 순위가 뒤집힌다. SGD 근사일치는 등n·near-additive 축퇴 지역의 현상이고,
  optimizer가 (b) 궤적 기하를 바꾸면((예) AdamW 전처리) 두 게임은 벌어진다. 부수 예측:
  **(a)를 심판으로 세우면 renorm-족 방법(G3·G4)이 유리**해진다 — (a) 자체가 renorm-족
  게임이므로.

## 2.3 승패 원장 (스테이지 × 검증축 × 지표; 동률·패배 포함)

표기: **승**(진짜 우위) / **동(포화)**(전원 동률 = 무정보 구간) / **동** / **패** /
**패(게임 공통)**((b) oracle도 같이 실패 = 방법 아닌 게임의 성질). 수치는 전부 부록 A 재검증.

### 2.3.1 Fidelity (1차)

| 스테이지 | 지표(truth) | 결과 | 판정 |
|---|---|---|---|
| LLM std20/anchor5 1B/3B/7B (IID-clean) | Spearman vs (b) | Flirds=1st=loss-heur 1.000-급 동률; GTG 0.97~1.00·FedSV 0.67~0.97도 근접 | **동(포화)** — G1·G2 간 무정보(T1); G4·G5·G6만 탈락(std20 0.09~0.19, anchor5 0.07~0.70)(T5·T4) |
| E4/E5 (Fed-LOO·N=10 2¹⁰) | Spearman vs (b) | 4종 전부 1.000 (Pearson 잔차만 Flirds 1−ρ≈9e-7 < 1st 9.2e-5) | **동(포화)** — 규모 확장에도 T1 유지; 미세 우열은 값-수준 잔차뿐 |
| CNN C1 (from-scratch, 10 시나리오) | Spearman vs (b) | **Flirds .919 > loss-heur .860 > 1st .832** ≫ GTG .569 > FedIF .491 > FedSV .401 > ShapleyFL .391 > ComFedSV .348 | **승** — 비가산 레짐서 same-game 내부도 갈림(2차>0차 singleton>1차); 심판이 (b)라 G1·G2에 구조적 이점 있음은 명시 필요(§2.7-①) |
| CNN C1 | Spearman vs **(a)** | **ShapleyFL .453 1위** > loss-heur .425 > 1st .408 > FedIF .380 > GTG .374 > **Flirds .352** | **패** — 단 T10 예측 그대로(renorm-족 심판엔 renorm-족 방법이 유리); "다른 게임" 서사의 데이터 측 근거 |
| std50k5 (N=50, 5/50; 부분참여) | Spearman vs (b) | Flirds 1.000(3-seed)·1st 1.000·loss-heur 1.000·GTG .983·FedSV .910 vs **FedIF −.040·ShapleyFL −.064·ComFedSV −.109** | **승(계열)** — G4·G5·G6 후처리 붕괴(T5)·G1·G2·G3-plain-sum 생존; G1·G2 간은 여전히 동(포화) |
| CNN k-sweep (참여 0.2/0.5/1.0, lf) | Spearman vs (b) | Flirds .891/.979/.993 vs **1st .305**/.765/.940; loss-heur .862/.857/.943; GTG .800→.497(역전) | **승(Flirds 고유)** — 저참여·대스텝서 2차항 가치(T6); GTG는 full서 큰 게임 근사 실패 |
| silo5/3B/device-anchor 오염 | Spearman vs (b) | G1·G2 1.000; FedIF .90~.93, FedSV .93~1.0, ComFedSV .83~.87; device-anchor: GTG .78·FedSV .75·ShapleyFL .58·ComFedSV −.02 | silo5 = **동(포화)에 가까운 승**; device-anchor = **승**(부분참여 T3-ii·T5) |
| AdamW 브리지 | Spearman vs (b) | **loss-heur .967 > Fed-LOO .933 > GTG .900 > Flirds .767 > 1st .700**; (a)↔(b) −0.53 | **패(진짜)** — same-game 내 실평가(0차)가 Taylor를 이김 = Taylor 절단의 optimizer-기하 의존(T6 연장; §2.4 원리 8). §5.1은 이 셀을 "(a)↔(b) 괴리"로만 다룸(§2.7-②) |
| (b) 타깃 자기안정성 | xseed ρ | IID-clean −0.37~+0.16(7B +0.73 예외) vs silo5 +0.87~+0.93; device anchor +0.12(오염 셀도) | fidelity 헤드라인의 전제 조건 — IID-clean·device의 +1.000은 **무정보**(T2) |

### 2.3.2 Selection→성능/Aggregation (2차-①)

| 스테이지 | 결과 | 판정 |
|---|---|---|
| LLM alpaca 개입(§3.2.1) | 전 arm MMLU·ROUGE parity(±.003) | **동(설계 기대)** — do-no-harm만 입증, 우열 무정보 |
| CNN C2 soft 가중(§3.2.2) | GN: vanilla .499→flirds_mult .609·**shapleyfl .645**·fedif .624; lf: .583→.626 | **승(개입군 공통)** — flirds가 1위 아닌 칸 존재(**패(개별 칸)**: GN pool 최고는 shapleyfl) |
| Track G LLM 게이트(§3.2.2) | frzero recovery **1.000**(V2·3-seed·오배제 0)·clean 무발화(maxΔ .00056)·noisy 회수 0.000(침묵)·frrand 0.462 | frzero **승(청정)**; noisy **동(작동영역 없음 — T7 예측 적중)**; frrand **동(코인플립)** |
| Track G CNN 게이트(§3.2.4) | GN 회수 .86~.94(V2 .6143/.5668); FR .81~.84; lf0.70 .5967/.5712; **clean 오발화**(V2 .6428/.6315 < vanilla .6488/.6389); V2w 불승격 | GN·FR **승**; clean **패(오발화)** — from-scratch 레짐의 cum 0-교차 노이즈 |
| Track H 경쟁 CNN(§3.2.3) | 총평(자체 재계산): **flirds .5682 1위** > lossheur .5584 > fedif .5366 > renorm .517~.529 > **1st .4712 최하**; FR: exact-0 .61~.62 vs renorm .37~.40(<vanilla .59); GN: flirds .567/.607 vs 1st·fedif .242~.248, renorm .59~.62, lossheur on .598/re .452; clean: 1st·fedif 무발화 .638~.639 vs flirds .632·lossheur .626·renorm .60~.605 | 계열-수준 **승**(T3·T5·T6·T7 전부 실증); 개별 칸 **패 다수**(clean=1st·fedif 우위, FR 최고=1st .6252, GN online 최고=shapleyfl .6115) |
| Track H R3 LLM noisy | renorm 게이트 발화 val-loss 2.3308~2.3310 < vanilla 2.3340; flirds/lossheur 침묵(=vanilla) | **패(정직)** — renorm 값-오차가 우연히 유효 문턱(T7의 대우); 절대 갭 0.003 소폭 |
| R4 GSM8K Tier A(seed0 ◐) | noisy T2: **flirds .3584 > lossheur .3548 > 1st=fedif .3432**, t2_random_k37 .3110(순위정보 가치 +4.7pt); frzero: 4점수원 kept=30=oracle, recovery 1.000; clean: T1 −1.0pt 오발화·T2 무해 | **승(방향; 1-seed)** — 첫 LLM-accuracy 점수원 우열; clean T1은 **패(오발화)** |
| P5/Scale/Dyn(§4.8) | P5h-retrain flirds 오염평균 **.6207 ≈ 천장 .6214**; Scale P5s .6198(회수 .78); Dyn: P1 GN **.1771 < vanilla .2547**·P5s GN .1902(붕괴), DP-4 적중률 .405≈우연 | P5h-retrain **승(최고 확증)**; Dyn **패(설계된 null-무대)** — 클라-수준 도구의 한계 명시 |

### 2.3.3 Detection (2차-③)

| 스테이지 | 결과 | 판정 |
|---|---|---|
| silo5·3B·B축 noisy/FR | valuation 전부 AUROC 1.000 | **동(포화)** — valuation 간 무정보; 전용 탐지기 일부만 탈락(STD-DAGMM noisy .25/frzero .00, FLDetector .75, FedDQC 비IID .92/.75) → 탐지기 대비 **승** |
| device100 noisy | φ-계열 .57~.77 vs **FedDQC 1.00**·FLTrust .85·FedIF .83 | **패** — 비IID 배경 침식 + per-round 참여복권(T8 연장; (b)도 .604로 같이 낮음 = 게임 수준 한계) |
| device100 FR | φ-계열 1.000·FLTrust 1.000 vs FedDQC .14~.57·FLDetector .53~.62 | **승** (exact-0) |
| frdelta | (b)=Flirds=전 valuation .333(seed별 동일); **STD-DAGMM 1.000**; FedIF/FLDetector/FLTrust **0.000**(속음) | **패(게임 공통)** — T8 그대로; (b) φ 전 클라 음수(전원 기여) 재검증 |
| dose | noisy 문턱 nr0.25(0.75→1.00); FR 전 배율 1.00 | **승**(문턱 명시) |

### 2.3.4 Cost / Stability

| 축 | 결과 | 판정 |
|---|---|---|
| Cost | anchor5 5×·device100 159×·N=10 2¹⁰ **160×**(117,649s vs 733s); **std20 역전**((b) 2943 < Flirds 4703); loss-heur가 anchor5서 Flirds보다 쌈(657 vs 716); Flirds-1st 전 무대 최저 | 조건부 **승**(T9 교차점 그대로); K_r 작으면 **패** |
| Stability | Flirds xseed .547 ≈ (b) 자체 .518; GTG .311·FedSV .289·ComFedSV .198·ShapleyFL .124 | **승** — T4(결정론 상속) 그대로 |

## 2.4 원리 주장 (형식: 주장 | 뒷받침 ≥2 | 반증 시도 | 교락 검토 | 신뢰도 | 논문 서사 후보)

**원리 1 — 신호는 클라 간 실재 차이(B축)가 만들고, 학습-강도 lever(A축)는 만들지 못한다.**
- 뒷받침: ① B축 2×2 — silo5 clean (b) xseed ρ **+0.87** vs iid5 clean **+0.13**(오염 0,
  도메인 분리만; §3.1.5, 부록 A-F). ② A축 전수 무효 — lr 격자 xseed −0.37~−0.20, std50k5
  N=50 고검정력에서도 +0.06, CNN 폭 8×에도 iid ρ 0.03~0.12(§4.2·4.3, 부록 A-E·M).
  ③ 오염 단독으로도 신호 생성(iid5 noisy 0.60/frzero 0.70) — 두 축이 독립 기여.
- 반증 시도: **7B anchor5 IID-clean xseed +0.733**(부록 A-M) — "IID-clean=무신호"의 유일한
  반례 후보. 1B/3B(−0.37~+0.03)와 불연속이라 스케일이 제3의 신호원일 가능성 — 반박 못 함,
  미탐색 축으로 남김(§3.4).
- 교락: N=5 이산 Spearman 저해상도(→N=50 std50k5가 보강), silo5 frzero 재실행판 0.93(H1
  재추첨 효과와 구분 불가, §6.2-13).
- 신뢰도: **높음**.
- 서사 후보: "Fidelity 검증의 전제는 타깃 순위의 실재성이다 — 우리는 그것을 cross-seed
  자기재현성으로 측정해, 신호가 실재하는 무대(비IID·오염)에서만 fidelity를 주장한다."

**원리 2 — 소스텝(LoRA 미세조정) 레짐의 (b) 게임은 near-additive이고, 그 무대의 same-game
방법 간 fidelity 동률은 수학적 필연이다(발견이 아니라 축퇴).**
- 뒷받침: ① 이론 — 1차 게임은 정확히 가산, 비가산성 전체가 교차곡률 Σq_ij(P5); 실측 가산 갭
  ≤0.9%(수학 문서 §1.5 인용). ② E4·E5 — Fed-LOO·loss-heur까지 N=10 exact 2¹⁰에서 전부
  1.000(부록 A-I; 격차는 Pearson 잔차 9e-7 vs 9e-5 수준). ③ A2 removal 곡선 — silo5에서
  강한 5종(Flirds=1st=(b)=loss-heur=Fed-LOO)이 **곡선 bit-동일**(§4.4.1).
- 반증 시도: CNN(from-scratch 대스텝)에선 same-game 내부가 갈라짐(.919/.860/.832) — 축퇴가
  깨지는 대우 방향의 확인이지 반례가 아님. Pearson 잔차 수준의 미세 우열(Flirds 최소)은
  존재하나 순위 지표론 안 보임.
- 교락: 포화 구간 판정 그 자체(D-1); N=5 해상도(D-5).
- 신뢰도: **높음**.
- 서사 후보: "LLM-LoRA 무대에서 semivalue 선택(Shapley vs LOO vs singleton)은 가산 축퇴로
  무차별해진다 — 방법이 갈리는 곳은 게임 근사(Taylor 차수·후처리)와 비용이다."

**원리 3 — 2차항(HVP)의 가치 영역은 {1차-내적이 구조적으로 0이 되는 위협(grad-noise류)} ∪
{1차 절단오차가 상쇄되지 않는 저참여·대스텝 레짐}이며, 그 밖에선 1차와 동률이다.**
- 뒷받침: ① Track H GN — 1차-내적 계열(Flirds-1st·FedIF) .2422~.2479 = vanilla 수준 실명
  vs Flirds .5169~.6215(정책 전반; 부록 A-K); E⟨g,δ_noise⟩=0 vs ½p²δᵀHδ>0(T6)와 정합.
  ② CNN k=0.2 — Flirds-1st .305 vs Flirds .891(부록 A-Q); LLM R=200(클라당 ~20회)선 1st도
  1.0 = "참여 횟수" 조건. ③ 물리잔차 — 2차가 1차의 ~1/3(2.7~3.4×; §5.5).
- 반증 시도: **loss-heur(0차 실평가)도 GN을 잡는다**(online .5981 — Flirds .5668보다 높음!)
  → "2차 고유"가 아니라 "1차-내적만 실명"이 정확한 진술. 단 loss-heur는 retrain 부호에서
  .4518로 하락(최종-누적 부호가 GN을 안정 포착 못 함)하고 Flirds retrain은 .6065 — **시점
  강건성은 2차 고유**. renorm도 GN 정상(.59~.62; 0차 실평가라서) — GN 포착 자체는 광범.
- 교락: CNN β-era(D-7)·from-scratch 레짐(LLM GN 무대는 gn_full 재실험 대기 = H-10 미판정).
- 신뢰도: **높음(CNN)·중(LLM 전이)**.
- 서사 후보: "2차항은 사치가 아니다 — gradient-정렬 정보가 0인 위협과 참여가 성긴 레짐에서
  1차 방법은 구조적으로 실명하며, HVP 1회/라운드가 그 시야를 산다."

**원리 4 — 부분참여는 신호를 만들지 않고 방법을 분별한다: min-max+EMA/uniform-완성 후처리
계열(G4·G5·G6)은 부분참여에서 구조적으로 붕괴하고, plain-sum 계열(G1·G2·G3)은 생존한다.**
- 뒷받침: ① std50k5 — FedIF −.040/ShapleyFL −.064/ComFedSV −.109 vs Flirds·1st·loss-heur
  1.000·GTG .983·FedSV .910(부록 A-E). ② 같은 분별이 std20(.09~.19 vs .91~1.0)과
  device-anchor(ShapleyFL .58·ComFedSV −.02 vs GTG .78·FedSV .75)에서 재현. ③ **메커니즘
  판별 파생계산**(부록 A-P): FedIF와 Flirds-1st는 같은 1차 신호인데, 신호 있는 full-참여
  silo5에선 순위 일치 ρ +0.9~1.0, std20에선 +0.16, anchor5(무신호)에선 −0.7~+0.6 산포 —
  **붕괴는 1차 신호가 아니라 min-max+EMA 후처리의 소행**임을 같은-신호 쌍이 직접 증명.
  ④ 이론 — K_r=2의 min-max는 {0,1} 이진 양자화, EMA는 최근-참여 덮어쓰기(T5; 코드 §2.1).
- 반증 시도: GTG·FedSV(renorm이지만 plain-sum)는 부분참여에서 생존(.91~.98) — "renorm 전부
  붕괴"가 아님을 확인(그룹핑을 G3/G4·5로 쪼갠 이유). CNN full-참여에선 GTG가 오히려 하락
  (.497) = 큰 2¹⁰ 게임의 MC 근사 실패(별개 기전).
- 교락: std50k5 seeds1-2는 경량 스위트(G4·G5 미채점 — 붕괴 수치는 seed0; D-2)·device
  off-anchor는 proxy truth(D-3).
- 신뢰도: **높음**.
- 서사 후보: "경쟁 방법들의 부분참여 붕괴는 Shapley 근사의 실패가 아니라 라운드-정규화(min-max)
  ·EMA 후처리가 라운드-합 게임을 다른 대상으로 바꿔버린 결과다 — 우리는 그 후처리가 없다."

**원리 5 — zero-semantics(0의 의미론)가 개입 실효성을 가른다: 고정가중 게임의 null-player
exact-0만이 free-rider류에서 안전한 자동 배제를 주고, 재정규화 게임은 null-player를
파괴한다(FR에 유령 몫 + clean에 음수 뿌림).**
- 뒷받침: ① frzero bit-exact 0(G1·G2·G5 raw; 감사+부록 A-Q 직접 재검증) → track_g recovery
  **1.000**(3-seed, 오배제 0) + R4 frzero 4점수원 kept=30=oracle 집합(EM recovery 1.000).
  ② renorm 3종의 frzero \|φ\| 0.0038~0.0050 = **clean 클라 φ와 동급 크기의 유령값**(부록
  A-Q) → Track H FR 파국 .37~.40(<vanilla .59) + observer 진단(FR에 raw≤0 0%·clean 63~71%
  raw≤0). ③ P5 대조 — 파국의 주범이 미검출이 아니라 clean 오배제 복리임을 정책 통제로 분리
  (완화 .50~.55, 검출은 여전히 0).
- 반증 시도: **LLM noisy 역전**(R3) — renorm 게이트가 값-오차 0-교차 덕에 발화해 val-loss
  2.3308 < vanilla 2.3340(flirds 침묵=2.3340). 즉 "exact-0=항상 우위"가 아니라 위협-의존
  트레이드오프. 단 renorm의 이득은 자기 게임의 진짜 0이 아니라 오차의 부산물(감사 판정 3)
  이고, 절대 갭 0.003 소폭 + FedSV/ComFedSV는 clean 오배제 1쌍 동반.
- 교락: renorm LLM clean 게이트 미실측(Tier 2 스코프 밖; LLM FR-붕괴 재현도 Tier B 대기 —
  현재 CNN 3-seed 근거), R3 갭 자체 소폭(D-6).
- 신뢰도: **높음(CNN)·중(LLM 재현 대기)**.
- 서사 후보: "기여도 0의 의미를 게임이 정확히 보장하는가 — null-player 공리의 실측 버전이
  free-rider 방어의 성패를 가른다."

**원리 6 — sign-게이트의 작동영역은 '위협이 누적 φ를 0 아래로 내리는가'로 사전 판정 가능하다:
FR-류(즉시)·고곡률 노이즈(CNN GN)는 영역 안, descent-정렬 오염(answer-swap noisy)은 영역
밖(침묵이 정직한 답), frrand는 경계(코인플립).**
- 뒷받침: ① dose ladder 직접 재검증 — noisy φ(오염 클라) nr0.1→1.0에서 −0.00246→−0.00186
  전 구간 기여-양수(0-교차 외삽 nr≈3.4 도달불가; 부록 A-Q) → track_g noisy 회수 0.000 전
  게이트(3-seed). ② frzero recovery 1.000 vs frrand 0.462 seed-의존(감사 코인플립 예측
  적중). ③ R4 noisy — T1 온라인 발화 희소(recall .05)나 T2 최종-부호는 13/20 배제(+.68
  회수) = answer-swap도 **최종-누적**은 경계 근처(alpaca-noisy와 위협이 달라 0-교차가 존재
  하되 늦게 옴).
- 반증 시도: R4 **clean T1 오발화**(false-excl 140쌍, −1.0pt) — "clean이면 무발화"(alpaca
  전례)가 GSM8K에선 깨짐 → 작동영역 판정은 위협뿐 아니라 **무대(과제 난이도·데이터 형태)**
  의존. T2(사후 부호)는 kept=50으로 무해 — 온라인 중간 0-교차가 문제.
- 교락: 1-seed(R4)·CNN clean 오발화(V2w 불승격)와 같은 뿌리(from-scratch/어려운 과제의 cum
  0-교차 노이즈).
- 신뢰도: **높음(방향)·중(무대 일반화)**.
- 서사 후보: "게이트는 만능 방어가 아니라 φ-부호 궤적이 예측하는 작동영역을 가진 도구다 —
  우리는 그 영역을 사전 등록하고 영역 밖 침묵을 실측으로 보였다."

**원리 7 — 기여도≠탐지는 공리적 갭이다: val-loss를 실제로 낮추는 위협 앞에서 φ-as-detector는
(b) oracle과 함께 실패하는 것이 정의상 옳고, 그 회수는 다른 질문의 탐지기 몫이다.**
- 뒷받침: ① frdelta — Flirds AUROC .333 = (b) .333 seed별 동일값, φ Spearman +1.000, (b)
  φ 전 클라 음수(전원 기여; 부록 A-H) vs STD-DAGMM 1.000(업데이트-패턴 축)·방향-정렬 계열
  (FedIF/FLTrust/FLDetector) 0.000(완전히 속음). ② device100 noisy — (b) 자체 .604(침식)
  vs FedDQC 1.0(데이터-품질 축); B축 대조에서 FedDQC만 IID/non-IID 배경에 민감(.58~1.0).
- 반증 시도: silo5·B축 noisy/FR에선 φ-as-detector도 1.0(포화) — 갭이 드러나는 건 "위협이
  게임의 답과 어긋나는" 셀뿐. 즉 반례가 아니라 적용 조건의 명시.
- 교락: tiny val(D-4)·N=5 coarse AUROC(무신호 기준선 0.83, D-8).
- 신뢰도: **높음**.
- 서사 후보: "우리는 φ의 탐지 실패 사례(frdelta)를 게임의 정직성 증거로 제시한다 — oracle과
  동일하게 실패하며, 이는 valuation과 detection이 다른 질문임을 보여준다."

**원리 8 — Taylor 근사의 품질은 optimizer-기하에 의존한다: SGD(업데이트≈−lr·grad)에선 전
무대 무손실이지만, AdamW(전처리된 delta)에선 0차 실평가 계열보다 뒤처진다.**
- 뒷받침: ① AdamW 브리지 — loss-heur .967 > Fed-LOO .933 > GTG .900(0차 실평가군) vs
  Flirds .767 > 1st .700(Taylor군)(부록 A-L; same-game 안에서 절단 유무로 정확히 갈림).
  ② SGD 대조 — 같은 무대(A1 removal) Flirds 전 seed +1.00. ③ 물리잔차의 lr-강건(A축
  lr↑에도 fidelity 1.000 유지)은 SGD-기하 안의 이야기였음이 사후 명확.
- 반증 시도: (a)↔(b) 자체가 −0.53으로 갈라진 무대라 "타깃이 흔들려 전부 낮아졌다"는 대안
  설명 — 그러나 vs (b) 기준 순위에서 0차 실평가가 상위에 남는 패턴은 게임 괴리로 설명 안
  됨(같은 (b)를 심판으로 쓴 값들). 다만 3-seed·N=5라 격차(0.967 vs 0.767)의 정밀도는 낮음.
- 교락: 브리지 설정(상수 lr — 논문 레시피 5e-5 cosine 아님)·1-스테이지(anchor5)뿐.
- 신뢰도: **중**.
- 서사 후보(정직 한계): "본 방법의 무손실 fidelity는 SGD-계 로컬 학습에서 성립하며, 적응형
  optimizer에선 실평가-기반 same-game 방법(singleton/LOO)이 안전한 대체다 — 같은 게임의
  0차 버전이 우리 프레임 안에 이미 있다."

**원리 9 — 결정론 상속: Flirds는 로그가 주어지면 결정론이라 (b) 타깃의 내재 안정성을 그대로
상속하고, MC-표집 계열은 추가 분산을 낸다.**
- 뒷받침: ① CNN xseed — Flirds .547 ≈ (b) .518 vs GTG .311·FedSV .289·ShapleyFL .124
  (§3.1.2 b2). ② 같은 구조가 fidelity 분산에도(anchor5 FedSV ±.163·ShapleyFL ±.283 vs
  Flirds ±.000; 부록 A-A).
- 반증 시도: loss-heur·Fed-LOO도 결정론이라 같은 상속(.474 등) — Flirds 고유가 아니라
  same-game 공유 성질(§2.5).
- 교락: CNN β-era·H1(절대값 재현 불가 — 순위 결론은 강건 예상, D-7).
- 신뢰도: **높음**.
- 서사 후보: "우리 추정기는 표집이 없다 — 재현성 실험에서 oracle 자신만큼 안정적이다."

**원리 10 — 비용 우위는 cohort 크기의 함수다: 라운드당 2^{K_r}(oracle)·K_r(singleton/LOO)
대비 Flirds는 상수(1 HVP≈6.5 fwd)라, K_r≥3부터 oracle을, K_r≥6부터 singleton류를 이기며,
K_r=2에선 역전당한다.**
- 뒷받침: ① 실측 교차 — std20(K=2): (b) 2943s < Flirds 4703s(**패**); anchor5(K=5): 716s
  vs (b) 3568s(5×)·loss-heur 657s(근소 패); N=10(K=10): 733s vs 117,649s(**160×**);
  device(K=10): 157s vs 24,975s(159×)(부록 A-G·H·I). ② op-count 산술 — fwd 1.60s·HVP
  10.36s(비율 6.47)로 전 실측 재현(§3.4.1).
- 반증 시도: 없음(산술+실측 정합) — 단 Flirds-1st(1 grad)는 전 무대 최저가라 "상수-비용"
  주장의 최저가 형태는 1차임을 병기해야.
- 교락: fp32 실측(bf16이면 배율 축소 — op-count 축이 방어).
- 신뢰도: **높음**.
- 서사 후보: "정확도 손실 0의 상수-비용 추정기 — 비용 우위는 참여 cohort가 큰 무대에서
  지수적으로 벌어진다."

## 2.5 고유성 구분 — "Flirds 고유(2차항 기인)" vs "exact-0/same-game 계열 공유"

논문에서 주장 가능한 범위가 다르다:

| 승리 | 귀속 |
|---|---|
| frzero exact-0·게이트 recovery 1.000 | **계열 공유**(G1·G2·G5 raw 전부; (b)·loss-heur·Fed-LOO·FedIF도 bit-0) — "우리 게임 정의(고정가중)의 공리"로 주장해야 하며 Flirds 단독 성질이 아님 |
| near-additive 무대 fidelity 1.000 | **계열 공유**(포화 — loss-heur·Fed-LOO도 1.000·5× 이상 저렴한 놈도 있음) — 우열 주장 금지 |
| GN 개입 회복(시점-강건: online+retrain 모두) | **Flirds 고유**(1st·FedIF 실명 .24; loss-heur는 online만 .598·retrain .452 하락; renorm은 잡지만 FR·clean에서 대가) |
| 저참여·대스텝 fidelity(CNN k=0.2 .891) | **Flirds 고유**(1st .305·loss-heur .862로 loss-heur도 상당 — 정확히는 "1차-내적 대비 고유, 0차 대비 우위 소폭") |
| CNN 비가산 무대 fidelity 1위(.919) | **Flirds 우위**(vs loss-heur .860·1st .832 — 2차항의 잔여 기여; 단 심판=(b) 이점 명시) |
| 부분참여 생존(std50k5 1.000) | **계열 공유**(plain-sum 전부 생존; 붕괴는 G4·G5·G6) |
| 안정성(xseed .547) | **계열 공유**(결정론 전부) |
| 상수-비용(1 HVP/round) | **Flirds 고유**(단, 최저가는 Flirds-1st; "2차 정확도+상수 비용"의 조합이 고유) |
| P5h-retrain 오염평균 .6207≈천장 | **Flirds 고유**(같은 정책서 lossheur .5315·renorm .54~.58; GN이 갈랐음) |
| R4 noisy T2 1위(.3584) | **Flirds 우위(1-seed)**(lossheur .3548과 근소 — Tier C 확정 전 "방향"으로만) |

## 2.6 필수 교락 점검 목록 (D-#; 원리 주장에 표기된 참조)

- **D-1 포화**: 전 방법 1.000 구간(LLM fidelity·silo5 AUROC) = 무정보 — 승패 판정에서
  "동(포화)"로 분리했다(§2.3).
- **D-2 1-seed 셀**: 3B silo5(§3.3.3)·E5 N=10·R4 Tier A·rank probe r32/64·st20/30·
  std50k5 seeds1-2의 G4·G5 채점. 해당 주장엔 "방향" 한정.
- **D-3 proxy-truth**: device100 off-anchor Spearman(truth=Flirds) — fidelity 논거로 쓰지
  않았다(탐지 축만 사용).
- **D-4 tiny val**: silo val=20·device val=10 — AUROC coarse·φ-노이즈 하한(§4.2 b4)과 함께
  읽어야; noisy nr0 대조 AUROC 0.83이 "무신호 기준선≠0.5"의 계측 참조.
- **D-5 N=5 해상도**: Spearman 1스왑=0.1·AUROC 5-레벨 이산 — N=50(std50k5)·N=100(device)
  보강 셀로 교차 확인.
- **D-6 pooling**: CNN C2·Track H "오염-평균"은 vanilla 기준선이 다른 셀을 섞음 — 본 문서는
  셀별 값으로만 논거를 세우고 평균은 순위 요약에만 썼다(자체 재계산 부록 A-R).
- **D-7 재현성·era**: H1(adapter init unseeded — 절대값 재현 불가·순위 결론 강건 예상)·
  CNN 120셀 β0.5-era·torch 2.11/2.12 혼재(M1) — 절대값 인용 시 caveat 필수.
- **D-8 지표 맹점**: 순위상관=값 크기·캘리브레이션 못 봄(ShapleyFL max_diff .98에도 Sp .70
  가능) / AUROC=경계·비용 못 봄·무신호 기준선≠0.5 / EM=포맷·greedy 의존 / recovery=분모
  갭 소폭이면 >1 폭발 / val-loss 4째자리=seed-분산에 묻힘(짝지은 delta로만 읽기) /
  거리지표=순위 불변이어도 값 붕괴 탐지(ShapleyFL·FedIF의 euclid 1.2~2.9가 그 사례).

## 2.7 [§2-g] 문서 판정(overview §5.1)과의 대조 — 자기 분석 완료 후 대조

전체 구조(위계별 승·약세 분류, 판정 매트릭스의 행 구성, 최고 세팅 표)는 **대체로 일치**한다.
아래는 일치 확인 + **불일치·보완 6건**(불일치 발견이 과제 요구사항의 가치였음).

| # | §5.1의 판정 | 내 판정 | 일치/불일치 |
|---|---|---|---|
| 1 | "CNN 무대 vs (b) 0.919 = 비교군 내 1위" (승) | 1위는 맞으나 **심판=(b)가 same-game(G1·G2)에 구조적으로 유리**하다는 명시가 §5.1에 없음. 순수 우열 근거는 (게임-무관 심판인) removal-변별(§4.4.2)과 k-sweep의 1st 대비 갭이 더 강함 | **보완 불일치** — 서술 순서를 removal 우선으로 바꿀 것을 제안 |
| 2 | AdamW를 "약세: (a)↔(b) 게임 괴리 → Flirds 실패 아님"으로 분류 | vs **(b) 기준으로도** loss-heur .967 > Flirds .767 — 같은 게임 안에서 0차 실평가가 Taylor를 이긴 **진짜 패배 셀**이며, 기전은 게임 괴리가 아니라 Taylor의 optimizer-기하 의존(원리 8). §5.1의 "Flirds 저하는 그 다음 순서의 관측" 문구가 이 구조를 가림 | **불일치(패배의 성격 규정)** — 논문 한계절에 "적응형 optimizer에선 same-game 0차 버전(loss-heur/Fed-LOO) 권장"을 넣는 게 정직하고 방어력 있음 |
| 3 | "noisy·zero/random FR 탐지 — 승 (1.0)" (판정 매트릭스) | silo5·B축의 1.0은 **valuation 전원 동률 = 포화**라 "탐지기 4종 중 일부 대비 승, valuation 간 무정보"로 분리해야. 진짜 탐지 승은 device100 FR(exact-0 계열만 1.0)뿐 | **분류 불일치**(포화/진짜 승 구분) |
| 4 | "grad-noise를 잡는 유일한 estimator" (§3.2.3 판정 ③·§5.1 인용) | loss-heur(estimator 4종에 포함)가 online .5981로 Flirds .5668보다 높음 — "유일"은 **retrain·P5까지 포함한 시점-강건 기준**에서만 성립(loss-heur re .4518 하락). 문구 한정 필요 | **표현 강도 불일치** |
| 5 | CNN vs (a) 0.35를 "전 방법 공통(두 게임 괴리)"로만 서술 | 공통 저조 안에서 **ShapleyFL .453 > loss-heur .425 > … > Flirds .352**로 uniform/renorm-족이 상위 — (a)=재정규화-족 게임이므로 T10 예측 그대로. 이는 "(a)와 (b)는 다른 게임"의 **데이터 측 능동 증거**(우연한 잡음이 아니라 이론이 예측한 방향)로 쓸 수 있는데 §5.1·§3.1.2 어디에도 없음 | **불일치(놓친 발견)** — 논문 (a)/(b) 절의 방어 논거로 추가 제안 |
| 6 | "IID-clean 무대 = 무정보(전 semivalue 공통)" | 동의 + **7B anchor5 xseed +0.733 예외**(스케일이 신호를 만들 가능성)가 §5.4에만 있고 §5.1 매트릭스에 미반영 — 무정보 판정의 스케일-조건부 가능성을 각주로 | **보완** |
| 7 | 서사 한 줄("신호가 존재하는 곳에서 oracle 동률을 5~160× 싸게…") | 동의 — 단 경쟁자 실패의 **이유**(후처리가 게임을 바꿈·null-player 파괴)가 빠져 있어 "우리가 유리한 무대를 골랐다"는 반론에 취약. 원리 4·5의 메커니즘 문장을 서사에 결합할 것 | **보완** |

그 외 §5.1의 개별 수치·매트릭스 행(비용 조건부, frdelta 게임-공통, Track G frzero/noisy,
Track H 계열-수준, P5h-retrain 최고 세팅, 안정성)은 내 독립 분석과 **일치**했다(수치는 전부
부록 A에서 원 파일 재검증).

---

# 3. [3단계] 논문용 실험 세팅 제안 — 축(fidelity/downstream/detection) × 모델(CNN/LLM)

> 형식: `세팅 상세 | 근거 원리(§2.4) | 예상 결과(정량) | 실패 시나리오와 해석 | 비용(GPU-h·산정 근거) | 재활용`.
> 대원칙(원리 1·2에서 직접 도출): **논문의 모든 우열 주장은 "신호가 실재하고(원리 1) 축퇴가
> 깨진(원리 2·T3·T5·T6)" 셀에서만** 세운다. 포화 셀은 우열이 아니라 *정확성·비용·안정성*
> 주장에만 쓴다. 고정 결정(poison 제외·Banzhaf/Ripple 제외·소규모-cohort selection 미확장·
> std50k5 selection 비게재·수렴 축 제외) 위반 0으로 설계했다.

## 3.1 제안 매트릭스

### Fidelity × LLM

| 제안 | 상세 | 근거 원리 | 예상 결과 | 실패 시 해석 | 비용 | 재활용 |
|---|---|---|---|---|---|---|
| **F-L1 (본문 주무대; 신규 실행 0)** | 주장 무대를 ① silo5(비IID; clean+noisy+frzero, 3-seed)와 ② std50k5(부분참여 N=50) + ③ B축 2×2(신호 실재성 근거)로 구성; 각 fidelity 표에 (b) xseed ρ 병기 | 1·2·4·9 | 이미 확보: silo5 G1·G2 1.000(타깃 ρ .87~.93 위에서) · std50k5 G4·G5·G6 붕괴 대조 · IID-clean 열은 "포화(무정보)" 라벨로 정직 표기 | — (기존 데이터 서술 문제) | **0** | rundir 전량 재활용 |
| **F-L2 (신규 1셀) 비등n silo5** | silo5 무대에서 클라 데이터량만 4:2:1:1:1급으로 skew(clean + noisy 2위협 × 3-seed); 전 방법 스위트 + (b) exact 2⁵ | **T3-i(P5c)**·4·5 | renorm-재구성(G3)·uniform(G4·G6)의 vs (b) Spearman이 등n silo5(GTG 1.0·FedSV .93~1.0)에서 **유의미 하락**(P5c: 비등n은 1차·가산에서도 순위 반전); G1·G2는 1.000 유지. "이론이 예측한 반례의 실증"으로 renorm과의 분화를 무대-편향 반론 없이 제시 | 동률 유지되면 → near-additive가 P5c를 가린 것(작은 곡률 조건 ii) — 그 자체가 P5 규모조건의 실측 데이터점(어느 쪽이든 이론 검증 결과로 게재 가능) | **~10–15 GPU-h**(silo5 오염 셀 1개 3-seed 실측이 셀당 ~2–4 GPU-h × 2위협 + (b)·coalition 오버헤드) | 무대·코드(phase2_matrix) 그대로, 데이터 분배만 변경 |
| **F-L3 (큐 완주) 3B silo5 3-seed** | β0.3 재실행 잔여(REMAINING §1.2)의 3B silo5 4셀 | D-2 해소 | 1B과 동일 구조(전 칸 1.000·AUROC 1.0) 재현 → 스케일 행의 1-seed caveat 제거 | 어긋나면 스케일-의존 발견(그 자체 보고) | 큐 등재분(추가 결정 0) | 재실행 캠페인 |
| (제안 안 함) 3B/7B (a) retrain oracle (P2/P3) | — | **T10** | (a)는 renorm-족 게임이라 심판으로 세울수록 G3·G4에 유리(§2.7-5) + 비용 최대(1B 기준 (a)=30,817s=(b)의 9×) | — | 셀당 수십 GPU-h | **피해야 할 목록 ③** 참조 |

### Fidelity × CNN

| 제안 | 상세 | 근거 원리 | 예상 | 실패 시 | 비용 | 재활용 |
|---|---|---|---|---|---|---|
| **F-C1 (본문; 신규 0)** | C1 10-시나리오 표(iid 포함/제외 병기) + **k-sweep(§4.3 b3)을 2차항 ablation의 주 증거로 승격** + removal-변별(§4.4.2)을 fidelity 절의 게임-무관 심판으로 배치(순서: removal → vs (b)) | 2·3·9, §2.7-① | 이미 확보: Flirds .919(1위)·1st k=0.2 붕괴 .305·removal에서 순위→acc 분리 +.045=(b) 동급·ShapleyFL 분리 ≈0 | — | **0** | C1 30 + probe 72 + A3 18셀 |
| **F-C2 (선택; 무GPU)** | C1 quantity_skew 셀을 "비등n 트리거(P5c)"의 CNN 실증으로 별도 강조(FedIF −0.20·renorm 하락 표) — F-L2와 쌍 | T3-i·4 | 서술 재배치만 | — | 0 | 기존 rundir |

### Downstream(Selection→성능) × LLM — **논문 2차-① 헤드라인 축**

| 제안 | 상세 | 근거 원리 | 예상 | 실패 시 | 비용 | 재활용 |
|---|---|---|---|---|---|---|
| **D-L1 (실행 중 완주) R4 gn_full(GN_ABS γ\*=5)** | gsm50k5 grad-noise 신정의 무대 성립 판정 + estimator 4점수원 T1/T2 | **3(LLM 전이의 유일 시금석)** | H-10: 1차(1st·fedif) 실명 vs Flirds(2차) 회복의 LLM-accuracy 재현; LoRA 소스텝이라 CNN과 달리 clean 오발화 없이 잡을 것(사전등록 문구) | 무대 또 불성립(oracle 갭 없음) → "LoRA-FL에서 GN은 집계 희석으로 실효 없음" 자체가 발견(위협 스코프 재조정); Flirds도 실명 → 2차항 서사를 CNN-한정으로 정직 축소 | 서버 큐 등재분(Tier A 실측 기준 위협당 ~8–24 GPU-h) | 큐·코드 완비 |
| **D-L2 (신규) R4 Tier C — noisy·frzero(+gn_full 성립 시 gn) seeds 1-2** | Tier A와 동일 스펙 2-seed 추가(관찰자+T2 4점수원+통제; T1은 flirds만) | 5·6, D-2 해소 | noisy T2 순위(flirds > lossheur > 1st=fedif)의 seed 재현 + frzero recovery 1.000 유지 → **selection 헤드라인의 3-seed 확정** | flirds↔lossheur 순위 뒤집힘 → "T2 게이트 성능은 exact-0 계열 내 동률"로 서사 완화(그래도 random 대비 +4pt대는 유지될 것) | **~95–130 GPU-h**(Tier A 실측 noisy 24.1+frzero 22.0 GPU-h/seed0 기준 ×2 seeds; frzero는 kept-동일 dedupe 시 ~15.8×2) | 무대·코드·통제 재활용 |
| **D-L3 (승인 게이트) R4 Tier B — renorm 4종 점수원** | 같은 T2(관찰자-부호 재학습)에 GTG/FedSV/ComFedSV/ShapleyFL 점수원 추가(가능하면 **T2-only로 축소** 실행 — 관찰자 재활용으로 재학습 비용만) | **5(LLM-accuracy 재현 = H-9 판정)** | CNN FR-붕괴의 LLM 재현: renorm은 frzero서 kept에 FR 잔존(exact-0 아님) + clean 오배제 → EM 하락; noisy에선 CNN-lf 유사(부분 회복+오배제 동반) | renorm도 frzero 전원 배제 → "renorm 오차가 GSM 무대에선 우연히 0-문턱 통과" — zero-semantics 주장을 CNN-한정으로 축소(정직 보고) | **~300–350 GPU-h**(스펙 추정; T2-only 축소 시 대폭↓ — 재학습 kept-셋 dedupe 기대) | 관찰자 로그 재활용 가능 |
| (게재 유지) alpaca do-no-harm parity | §3.2.1 표를 "clean-IID에서 해치지 않음" 전용으로 | 1·2 | 이미 확보 | — | 0 | 기존 |

### Downstream × CNN

| 제안 | 상세 | 근거 원리 | 예상 | 실패 시 | 비용 | 재활용 |
|---|---|---|---|---|---|---|
| **D-C1 (본문; 신규 0)** | 본문 = Track H 경쟁(P1 표 + zero-semantics 진단) + **flirds×P5h-retrain(.6207≈천장 .6214)을 estimator 확증 세팅으로** + Scale(완전참여 P5s .6198)·Dyn(도구 한계 정직 보고)은 ablation/appendix | 3·5·6 | 이미 확보(사전등록 예측표 대조 포함 — MISS 정직 보고 형식 유지) | — | **0** | track_g 48 + track_h 204+21+9 rundir |

### Detection × LLM / × CNN

| 제안 | 상세 | 근거 원리 | 예상 | 실패 시 | 비용 | 재활용 |
|---|---|---|---|---|---|---|
| **T-1 (본문; 신규 0)** | 위계 마지막 절: silo5·B축(배경 무관 1.0; 포화 명시) + device100(FedDQC 대비 정직 패배 + FR exact-0 승) + **frdelta를 "기여도≠탐지"의 정면 사례로**(oracle 동일 실패 + STD-DAGMM 1.0 + 방향-정렬 3종 0.0) + dose 문턱(nr0.25) | **7**·5 | 이미 확보 | — | **0** | 기존 |

## 3.2 피해야 할 세팅 (원리상 포화·무정보·역효과)

| # | 세팅 | 이유(원리) |
|---|---|---|
| ① | IID-clean 무대에서 fidelity **우열** 주장(std20·anchor5·iid5) | T1 축퇴 + T2 무신호(타깃 xseed −0.37~+0.13) — 전원 1.000 동률 = 무정보. do-no-harm·비용·물리잔차 전용으로만 |
| ② | same-game 계열(G1·G2) 간 fidelity 우열을 LLM 무대에서 주장 | T1 — loss-heur·Fed-LOO도 1.000이고 일부 무대선 더 저렴(anchor5 657s<716s). CNN 비가산·k-sweep에서만 우열 성립 |
| ③ | (a) retrain oracle을 **주 심판**으로 확장(3B/7B P2/P3) | T10 — (a)는 renorm-족 게임이라 심판일수록 G3·G4에 유리(CNN vs(a) 1위=ShapleyFL .453, §2.7-5); AdamW선 (a)↔(b) −0.53. 1B 브리지+CNN 관측 보고로 족함 |
| ④ | device100 off-anchor Spearman을 fidelity 근거로 | D-3 proxy-truth("Flirds와 동일"의 뜻) |
| ⑤ | noisy(answer-swap)류에 sign-게이트 회수를 기대하는 셀 | T7·원리 6 — 작동영역 밖(0-교차 nr≈3.4 도달불가); 회수는 soft 가중·T2·selection 몫 |
| ⑥ | dyn-재추첨류 무대에 클라-수준 게이트/valuation 배치 | 클라-신호 원리상 소멸(DP-4 .405=우연); do-no-harm도 GN서 깨짐(P5s .1902) — 한계 명시 이상으로 확장 금지 |
| ⑦ | K_r=2급 저-cohort 무대에서 비용 우위 주장 | T9 역전((b) 2943 < Flirds 4703) — 비용 절은 cohort-조건 명시 필수 |
| ⑧ | N=5 단독 해상도로 소수점 우열 확정 | D-5(1스왑=0.1·AUROC 5-레벨) — N=50/100 셀 병기 |
| ⑨ | AdamW 무대를 fidelity 본문으로 | 원리 8 — 한계절 소재(0차 대체 권고와 함께)로만 |
| ⑩ | (고정 결정 재확인) poison 축·Banzhaf/Ripple 비교·std50k5/소규모-cohort selection·수렴 축 | Yonghee 2026-07-22 결정 — 본 제안 전체가 이 스코프 안 |

## 3.3 기존 데이터만으로 논문에 쓸 수 있는 것 vs 신규 실행 필요

- **기존 데이터로 완결(신규 0)**: fidelity 전 절(LLM 포화 명시 포함 + CNN + 부분참여 대조 +
  removal 인과 + 안정성 + B축 신호론) · 비용 절(op-count+실측 5~160×+역전 셀 정직 표기) ·
  CNN 개입·게이팅·경쟁 전체(P1/P2/P5/Scale/Dyn) · LLM 게이팅(frzero 1.000·noisy 침묵) ·
  탐지 절(frdelta 포함) · dose 문턱 · AdamW 한계절 · Taylor 물리잔차(P3) · (b) 타깃 안정성
  프로토콜. **즉 논문 골격 전체는 이미 서 있다.**
- **신규 실행이 유의미하게 바꾸는 것**: R4 3-seed(D-L2 — selection 헤드라인의 seed 방어),
  gn_full(D-L1 — 원리 3의 LLM 전이 판정), Tier B(D-L3 — 원리 5의 LLM 재현), 비등n 셀
  (F-L2 — P5c 실증), 3B 3-seed(F-L3 — caveat 제거).

## 3.4 CNN→LLM 전이 리스크 (from-scratch vs LoRA-finetune 레짐 차이)

전제: **CNN에서 갈린 현상이 LLM에서 재현된다는 보장이 없다** — 두 레짐은 가산 갭(비가산 CNN
vs ≤0.9% LLM)·스텝 크기·클라당 참여 횟수가 다르다.

| CNN 관측 | LLM 재현 전망 | 검증 경로 |
|---|---|---|
| GN에서 1차 실명 vs 2차 포착(원리 3) | LoRA 소스텝에선 GN 자체가 집계 희석로 약할 수 있음(구정의 gnoise 무대 불성립이 이미 그 신호) | **gn_full(D-L1)** — H-10 사전등록 그대로 판정 |
| renorm FR 파국(원리 5) | LLM frzero도 renorm은 exact-0 아님(감사 확정)이나, 배제-임계 통과 여부는 값 크기 의존 | **Tier B(D-L3)** — H-9 |
| clean 게이트 오발화(V2w 불승격) | alpaca에선 무발화(재현 안 됨)·GSM8K T1에선 재발(−1.0pt) — **무대(과제 난이도) 의존**이 이미 실측됨 | T2(사후 부호) 우선 사용으로 회피 — R4 설계에 반영됨 |
| same-game 내 fidelity 우열(.919/.860/.832) | near-additive LLM에선 재현 불가 예측(포화) — 리스크가 아니라 원리 2의 예측 | 재현 시도 자체를 안 함(피해야 할 ②) |
| CNN 결과의 β-era·H1 재현성 | — | 절대값 인용 시 D-7 caveat 병기(재실행 계획은 루트 RERUN 문서) |

## 3.5 우선순위 shortlist (비용 대비 기대효과 순)

1. **R4 gn_full 완주 + H-10 판정** — 이미 서버 큐(추가 의사결정 비용 0). 원리 3의 LLM 전이를
   가르는 유일한 시금석이고, 실패해도 해석이 확정적(무대 불성립=위협 스코프 발견 / Flirds
   실명=서사 정직 축소).
2. **R4 Tier C(noisy·frzero seeds 1-2; ~95–130 GPU-h)** — 논문 2차-① 헤드라인(첫
   LLM-accuracy 점수원 우열 + random 대비 +4.7pt)이 현재 1-seed(D-2)에 걸려 있다. 비용 대비
   방어력 상승이 가장 크다.
3. **3B silo5 β0.3 3-seed 완성(큐 잔여; 소액)** — 스케일 행 caveat 제거(§6.2-1 해소).
4. **비등n silo5 1셀(F-L2; ~10–15 GPU-h)** — P5c "이론이 예측한 순위 반전"의 실증. renorm
   대비 분화를 무대-편향 반론 없이 제시하는 이론-실증 결합 셀 — novelty 대비 최저가.
5. **R4 Tier B(renorm 점수원; ~300–350 GPU-h, T2-only 축소 검토)** — 원리 5의 LLM 재현.
   임팩트는 크지만 비용이 커 Yonghee 승인 게이트 유지(축소판: T2-only + noisy·frzero 2위협
   이면 절반 이하 추정).

순서 근거: ①은 매몰비용 0·판정가치 최대, ②는 헤드라인 방어(게재 리스크 직결), ③은 저비용
caveat 제거, ④는 저비용 novelty, ⑤는 고비용·고임팩트라 마지막(승인 게이트).

---

# 부록 A. 재검증한 수치 목록 (원 파일 대조; 237 PASS / 0 FAIL)

검증 스크립트 = `flirds-principle-analysis/verify_numbers.py` (read-only; GPU 불필요; 실행
`python research-wiki/survey/flirds-principle-analysis/verify_numbers.py`) → 전체 행 =
`flirds-principle-analysis/verification_report.txt`. 섹션별 내용(각 행 claimed=overview 표
수치 vs computed=원 파일 재계산):

| 절 | 검증 대상(원 파일) | 대표 항목 |
|---|---|---|
| A | `runs/track_d/fidelity.csv` — §3.1.1 LLM fidelity 18개 (mean·std 쌍) | 1B std20 Flirds 1.000±.000·FedIF 0.157±.303·ShapleyFL 0.194±.351; anchor5 FedSV 0.700±.163 등 |
| B | `1B_anchor5_seed*/phi.parquet` 재계산 — vs (a) | Flirds/1st/loss-heur/GTG 0.933; (a)↔(b) 0.933±.047 |
| C | `runs/track_c/fidelity.csv` — CNN pool·시나리오 | Flirds .919/.352(vs b/a)·iid-제외 .928·ShapleyFL vs(a) .453(1위) 등 20항 |
| D·M | `RESULTS.txt` 스팟·`target_stability.csv` 2종 전문 | anchor5 −0.367·silo5_clean +0.867·7B +0.733·device anchor +0.12 등 |
| E | probe metrics.json·phi.parquet | std50k5 붕괴(FedIF −.040 등)·3-seed Flirds 1.000·(b) xseed 쌍(−0.094/+0.132/+0.152)·lr범위 0.00119→0.00330 |
| F | B축 phi.parquet 재계산 + metrics AUROC | xseed 2×3 표(0.13/0.60/0.70/0.87/0.93/0.93)·FedDQC 1.00/0.92/0.58/0.75 등 |
| G | silo5 metrics.json | AUROC·Sp 전 칸·runtime(Flirds 107s·1st 35s·loss-heur 99s·(b) 531s)·noisy 클라=최소기여 3/3 seed·전 클라 φ<0 |
| H | device100 anchor·frdelta | (b) AUROC .604(per-seed .660/.563/.589)·runtime((b) 24,975s vs Flirds 157s)·frdelta 전원 .333·STD-DAGMM 1.0·FedIF/FLTrust/FLDetector 0.0 |
| I | E4/E5 metrics.json | Fed-LOO 1.000·(b) 2¹⁰ 117,649s vs Flirds 733s·runtime 10항 |
| J | track_g llm/cnn_summary.csv | frzero recovery 1.000(V2)·noisy 0.000·CNN GN .6143/.5668·clean 오발화(.6428/.6315 vs vanilla .6488/.6389) |
| K | track_h 분석 CSV 5종 | P1 경쟁 14항(renorm FR .3915~.4020 등)·R3 renorm 2.3308~2.3310·R4 EM 14항+kept(37/36/38/34·frzero 30)·P5(.6215/.6197/.3959 등)·Scale(P5s .6220/.6268/.6107)·Dyn(P1 GN .1771·DP-4 .405×3) |
| L | removal_dose metrics.json | A2 Δ(+.0076/−.0084 등 6항; 정의=L(k0)−L(k4))·A3(cifar lf ρ 1.00·gap +.0445·iid −.0033·mnist +.0035)·dose(nr0.1 .75→nr0.25 1.00)·AdamW(Flirds .767·(a)↔(b) −.533·per-seed 일치) |
| N | microbench summary.json | fwd 1.60s·HVP 10.36s(비율 6.47) |
| P | **파생: 메커니즘 판별** | Spearman(FedIF, Flirds-1st): silo5 +0.9/+0.9/+1.0 vs std20 +0.2/−0.24/+0.5 vs anchor5 산포 — 후처리 파괴 가설의 직접 판별(원리 4-③) · CNN lf Spearman(Flirds, X): (b) .996·loss-heur .956·1st .960 vs GTG .657·FedSV .535·ShapleyFL .358(same-game 군집) |
| Q | **파생: 부호 원리** | frzero bit-exact 0(6종 True)·renorm \|φ\| .0038~.0050(≠0)·noisy dose ladder φ(오염클라) 전 구간 음수(−0.00246→−0.00186; 기여-양수=0-교차 없음) |
| R | 잔여 스팟 | track_g lf0.70(.5967/.5712)·P5h-retrain LF .6210·clean .6333·**경쟁 총평 자체 재계산**(flirds .5682 1위 > lossheur .5584 > fedif .5366 > renorm .517~.529 > 1st .4712 최하 — overview §3.2.3 총평과 정합) |

주의: 절대값 인용에는 전부 D-7(H1 재현성·CNN β-era·torch 혼재) caveat가 적용된다(overview
§6.2-13). 본 문서의 결론은 순위·구조 수준이라 강건 예상이나, "재현 가능" 주장은 재실행 후에만.

# 부록 B. 본 분석의 한계

- overview가 유일한 실험 카탈로그 입력(문서 자체의 누락은 §8 커버리지 자가점검에 의존).
- 무GPU 제약상 가산 갭(Σq_ij)·φ_rounds 수준의 신규 대규모 재계산은 판별에 필요한 최소
  (부록 A-P·Q)만 수행.
- 3단계 비용 추정은 Tier A timing.json·track_g 실측의 외삽(±30% 오차 가정).

