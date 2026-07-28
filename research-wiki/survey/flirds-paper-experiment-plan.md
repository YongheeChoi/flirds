---
type: survey
title: "Flirds 논문 수록 실험 재편성 — 본문/부록 확정 목록 + 결손 실험"
created: 2026-07-25
updated: 2026-07-28
tags: [flirds, paper, experiment-plan, scope, gap-analysis]
---

# Flirds 논문 수록 실험 재편성 (본문/부록 확정 + 결손)

> **무엇**: 논문 본문/부록에 **무엇을 어느 표로 싣는지**를 [[flirds-experiment-axis-map]]의 분류·이름 체계로 확정하고, 각 항목이 **지금 실측으로 채워져 있는지**를 rundir 대조로 판정한 문서.
> **왜 두 용도인가**: ① 논문 집필 세션은 "무엇을 어느 표로 싣나"를 여기서 읽는다 ② **실험 설계 세션은 §4(결손 목록)를 입력으로 받아** 부족한 실험을 설계한다. 그래서 "완료"만이 아니라 **무엇이 비었고 몇 런이 필요하며 코드 변경이 필요한지**까지 적는다.
> **수치를 한 곳에서 보려면** → [[flirds-paper-tables]] (수록 확정분만 모은 미러).
> **수치는 여기 없다** — 실제 값은 축별 결과 페이지가 담당한다: [[flirds-results-fidelity]] · [[flirds-results-downstream]] · [[flirds-results-ablation]] · [[flirds-results-cost]]
> **판정 근거**: **2026-07-28 기준 `runs/` rundir 전수 대조** — c2fid 168 · track_c/c1 48(축 그리드) · track_g/rundirs_cnn 162 · track_h(CNN·LLM) · phase2_matrix · removal_dose · track_d 18 · probe_signal. 마커는 실측이며 추정이 아니다.

---

## 0. 스코프 규칙

### 0.1 오염축 — 이 다섯 개만 쓴다

| 트랙 | 오염 위협 | 코드 토큰 | 비고 |
|---|---|---|---|
| **CNN** | label-flip **@0.70** | `label_flip` + `C2_FLIP_RATE=0.70` | dose 0.15·0.35는 축 밖 |
| **CNN** | free-rider-zero | `free_rider` | Δ=0 업데이트, φ exact-0 |
| **CNN** | gradient-noise | `grad_noise` | 2차항 존재 이유의 결정 칸 |
| **LLM** | answer-swap **@0.7** | `noisy` (`NOISY_RATE=0.7`) | = 코드의 `answer_swap_graded` |
| **LLM** | free-rider-zero | `frzero` | 〃 |

> **clean은 오염이 아니라 대조 앵커**다 — 제거하지 않고 전 무대에서 열/행으로 유지한다(무해성 parity·오발화 판정의 근거). 아래 "오염축 커버" 표기는 **오염 위협만** 센다.
> **축 밖으로 빠지는 위협**: `frrand`(랜덤-델타 free-rider) · `label_flip@{0.15,0.35}` · `lf-strmain`(강한-주류 변종) · `frdelta` · `mixed` · `poison` · LLM `gnoise`.

### 0.2 비교군 — 9행 고정

| 상태 | 방법 |
|---|---|
| **수록 (9행)** | **(b)oracle** + same-game 3종(**Flirds · Flirds-1st · loss-heur**) + cross-game 5종(**GTG · FedSV · ComFedSV · ShapleyFL β=0.3 · FedIF**) |
| **제외** | **전용 탐지기 4종**(FLDetector · FLTrust · STD-DAGMM · FedDQC) · **Fed-LOO** · **Banzhaf** · **Ripple** |

> 러너는 계속 이 열들을 산출한다 — **집계·표 단계에서만 뺀다**(rundir·CSV는 존속).
> 탐지 축 자체가 논문에서 빠지므로(§1) 전용 탐지기 대조도 각주로 살리지 않는다. 논문 본문엔 해당 서술이 애초에 없다(§6).

### 0.3 seed — 예외 없이 3-seed

수록 실험은 **seed{0,1,2} 전부**. 3-seed를 못 채운 실험은 **수록하지 않는다**(§5). 근거: [[all-experiments-3-seed]].

### 0.4 마커

● 3-seed 실측 · ◐ 부분/1-seed(수록 불가) · ⬚ 미실행 · ⟐ 파생/재분석(재실행 0)

---

## 1. 한눈에 — 배치 × 축 × 상태

| 배치     | 축          | 실험 (축-지도 이름)                                         | 상태                             |
| ------ | ---------- | ---------------------------------------------------- | ------------------------------ |
| **본문** | Fidelity   | 1A-CNN 대규모 교차-디바이스 부분참여 — cifar10 {dir1, iid}        | ● 완료                           |
| **본문** | Fidelity   | 1A-LLM 주무대 정확도-무대 충실도 (R4)                           | ◐ **7/9셀** — 잔여 frzero s0·s2   |
| **본문** | Fidelity   | 1A-LLM 표준 부분참여 충실도 (1B·3B·7B)                        | ● 완료 (clean-IID 전용 무대)         |
| **본문** | Fidelity   | 1B-CNN vs (a) — cifar10 {dir1, iid}                  | ● 완료 (축 그리드 48/48의 cifar10 절반) |
| **본문** | Fidelity   | 1B-LLM 교차-사일로 (a)-leg 듀얼오라클 — silo5 3위협              | ● 완료                           |
| **본문** | Downstream | 2-CNN P1 부호-게이트 online/retrain — cifar10 {dir1, iid} | ● 완료                           |
| **본문** | Downstream | 2-LLM 주무대 정확도 개입 (R4 GSM8K EM)                       | ● 완료 (9칸 3-seed)               |
| **본문** | Cost       | 5-공통 op-count 모델 — **N·R·K 파라메트릭**                   | ⟐ **재작성 대기**(실행 0)             |
| **본문** | Cost       | 5-LLM 실측 runtime — device100 앵커                      | ● 완료                           |
| **본문** | Ablation   | 2차항(HVP)의 기여 — CNN 레그                                | ● 완료                           |
| **본문** | Ablation   | Removal-curve — CNN                                  | ● 완료 (오염축 정렬 9셀)               |
| **본문** | Ablation   | Removal-curve — LLM                                  | ● 완료                           |
| **부록** | Fidelity   | 1A-CNN mnist {dir1, iid}                             | ● 완료                           |
| **부록** | Fidelity   | 1A-LLM 교차-사일로 (b)-leg                                | ● 완료                           |
| **부록** | Fidelity   | 1A-LLM 대규모 교차-디바이스 앵커                                | ● 완료                           |
| **부록** | Fidelity   | 1B-CNN mnist {dir1, iid} vs (a)                      | ● 완료                           |
| **부록** | Fidelity   | 1B-LLM 소형 앵커 듀얼오라클 vs (a)                            | ● 완료(1B) — 참조·폴백               |
| **부록** | Fidelity   | 1C 재현성·안정성                                           | ⟐ **재산출 대기**(G1 착지 후)          |
| **부록** | Downstream | 2-CNN mnist {dir1, iid} P1 online/retrain            | ● 완료 (기준 arm 포함)               |
| **부록** | Downstream | 2-CNN P1w — cifar10/mnist {dir1, iid}                | ● 완료 (롤업 CSV만 mnist 미포함)       |
| **부록** | Downstream | 2-LLM 표준 개입 무해성                                      | ● 완료                           |
| **부록** | Cost       | 5-LLM runtime — silo5 · anchor5 · std20              | ● 완료                           |
| **부록** | Cost       | 5-CNN runtime 상세(방법별)                                | ● 완료 (실행 0 · 기존 rundir 파생)     |
| **부록** | Ablation   | A축 용량 lever probe — CNN                              | ● 완료                           |
| **부록** | Ablation   | φ 부호 감사 — 게이팅의 작동 전제                                 | ● 완료 (LLM·CNN 양 레그)            |

> **결손은 한 곳뿐이다**: **LLM 주무대 (b) 오라클(R4-L2) frzero 2셀**(본문 fidelity). 그 외 모든 수록 실험은 3-seed 실측으로 차 있고, **남은 일은 표 정리·문서 재작성뿐**이다.
> **탐지(detection) 축은 논문에서 전량 빠진다**(2026-07-28 Yonghee) — 본문·부록 어디에도 φ-AUROC 표를 넣지 않는다. rundir·러너 산출과 [[flirds-results-detection]]은 존속한다(§5).

---

## 2. 본문 수록 예정

### 2.1 Fidelity `[본문]`

| 실험 (축-지도) | 무대·세팅 | 오염축 커버 | 상태 | 남은 일 |
|---|---|---|---|---|
| **1A-CNN 대규모 교차-디바이스 부분참여** | FedSVCNN · cifar10 {dir1, iid} · N=100 · 10/100 · R=120 · (b) per-round | lf@0.70 ● · frzero ● · gn ● (+clean ●) | **● 완료** | 기존 표에서 qskew·shard·fmnist 파티션 및 축 밖 위협 열 삭제 |
| **1A-LLM 주무대 정확도-무대 충실도 (R4)** | Llama-3.2-1B · gsm50k5 N=50 · 5/50 · R=200 · (b) per-round · GSM8K | swap@0.7 **●** · clean **●** · frzero ◐ (s1만) | **◐ 7/9셀** | **G1** — **2셀**(frzero s0·s2). seed1·2는 `MIN_METHODS`라 (b)+Flirds+1st만 |
| **1A-LLM 표준 부분참여 충실도 (1B·3B·7B)** | alpaca IID · N=20 · 2/round · R=200 · (b) per-round | 오염 없음(clean-IID 전용) | **● 완료** | 표 캡션에 "clean-IID 전용 무대" 명시 |
| **1B-CNN 소형 교차-사일로 vs (a)** | LeNet5/FedSVCNN · N=10 · full · R=10 · (a) 2¹⁰ + (b) 2¹⁰ (오염축 정렬 축 그리드) | cifar10 lf@0.70 ● · frzero ● · gn ● (+clean ●) | **● 완료** | 없음 |
| **1B-LLM 교차-사일로 (a)-leg** | Llama-3.2-1B · silo5 N=5 5도메인 비IID · full · R=10 · (a) 2⁵ = 32 재학습 | swap ● · frzero ● (+clean ●) | **● 완료** | 값 전사(전 방법 확장의 ComFedSV clean 열 ⬚) |

> **본문은 silo5, 부록은 anchor5.** silo5가 본문인 이유: 오염축 {clean, swap, frzero} 3/3 정렬 · (b) 타깃이 seed-안정(xseed +0.87~+0.93) · **듀얼오라클 일치도 1.000** — "in-run (b)를 같은-게임 정답으로 쓴다"는 설계 선택을 방법-중립 참값이 직접 승인한다. anchor5는 IID-clean 전용에 (b) 타깃이 seed-불안정(xseed −0.367)이고 **듀얼오라클 일치도 0.933이 천장**이라, same-game 방법들의 0.933은 방법 한계가 아니라 무대 한계다. 단 silo5의 answer-swap 비율은 이 무대 canonical **nr=1.0**(R4의 0.7과 다름)이고 ComFedSV clean 열이 ⬚다. 수치는 [[flirds-results-fidelity]] §1B-LLM.

> **1B-CNN 주의**: (a) 재학습 오라클은 2^N 재학습이라 **N=100에서 원리적으로 불가** → 1A-CNN(N=100·부분참여)과 **N·참여율은 맞출 수 없다**. 정렬 가능한 건 **오염축과 파티션**뿐이다(N=10 유지). 이 비교불가성은 논문에도 명시해야 한다.

### 2.2 Downstream `[본문]`

| 실험 | 무대·세팅 | 오염축 커버 | 상태 | 남은 일 |
|---|---|---|---|---|
| **2-CNN P1 부호-게이트 — cifar10 {dir1, iid}** | 1A-CNN과 동일 무대 · 8점수원 × {online 배포게이팅, retrain T2} · 절대 test acc · 앵커 vanilla/oracle_excl/random_excl | lf@0.70 ● · frzero ● · gn ● (+clean ●) | **● 완료** | 없음 |
| **2-LLM 주무대 정확도 개입 (R4 GSM8K EM)** | gsm50k5 · online gate_v2 + retrain t2_sign · 절대 EM(test 1,119) · **수록 arm = {vanilla, oracle-제외, random, Flirds, Flirds-1st}만** | retrain·online 모두 clean ● · swap ● · frzero ● (9칸 3-seed) | **● 완료** | 없음. loss-heur·FedIF도 기산출이나 **표 미수록** |

> **online clean parity 판정 가능**: Flirds −0.27pt · Flirds-1st −0.42pt이나 **seed별 부호가 갈려**(−0.63/−0.54/+0.36) 유의미한 손해가 아니다. 수치는 [[flirds-results-downstream]].
> 앵커(vanilla·oracle_excl·random_excl)는 **cifar10 4파티션 · fmnist 2파티션 · mnist 2파티션 전 오염위협 3-seed 확보**.

### 2.3 Cost `[본문]`

| 실험 | 요구 | 상태 | 남은 일 |
|---|---|---|---|
| **5-공통 연산수(op-count) 모델** | 고정 숫자가 아니라 **N(총 클라)·R(라운드)·K(라운드당 참여)의 함수**로 제시 | ⟐ **재작성** | **G7** — 실행 0. `op_counts.py::per_round(method,K,N)`이 이미 파라메트릭이라 **식만 뽑아 쓰면 된다**(§7.3) |
| **5-LLM 실측 runtime — device100 앵커** | N=100 · 10/100 · α=0.5 · 1B · 방법별 valuation wall-clock | **● 3-seed 완료** | 표에서 제외 방법 행 삭제 |

### 2.4 Ablation `[본문]`

| 실험 | 무대 | 상태 | 남은 일 |
|---|---|---|---|
| **2차항(HVP)의 기여 — CNN 레그** | c2fid grad-noise 셀(Flirds ρ vs Flirds-1st ρ) + probe_signal k-sweep | **● 완료** | 표를 cifar10 {dir1, iid} 2행으로 축소 |
| **Removal-curve — CNN** | cifar10/iid N=10 · worst-first vs best-first 제거 후 재학습 acc 분리 | **● 완료** — 오염축 3위협 × 3seed 9셀(`removal_dose/rundirs_cnn/cifar10_iid-*`) | 없음 — 레거시 2시나리오(label_flip·feature_noise) 행 삭제 완료 |
| **Removal-curve — LLM** | silo5 N=5 · worst/best-first 제거 후 val-loss | **● 완료**(noisy·frzero 3-seed) | frrand·poison 행 삭제 |

---

## 3. 부록 수록 예정

### 3.1 Fidelity `[부록]`

| 실험 | 무대·세팅 | 상태 | 남은 일 |
|---|---|---|---|
| **1A-CNN mnist {dir1, iid}** | 1A-CNN(cifar10)과 **동일 세팅**, 데이터셋만 mnist | **● 24/24** | 없음 |
| **1A-LLM 교차-사일로 도메인 충실도 ((b)-leg)** | 1B · silo5 N=5 5도메인 · full · R=10 · (b) 2⁵ | **● 완료**(swap·frzero 3-seed) | clean·frrand 행 정리 |
| **1A-LLM 대규모 교차-디바이스 앵커 충실도** | 1B · N=100 α=0.5 · (b) per-round | **● 완료** | frrand 행 삭제 |
| **1B-CNN mnist {dir1, iid} vs (a)** | 1B-CNN 축 그리드와 동일, mnist | **● 24/24** | 없음 — 축 그리드가 **48/48 전수** |
| **1B-LLM 소형 앵커 vs (a)** | anchor5 N=5 full R=30 · (a) 2⁵ | **● 3-seed(1B)** | 없음 — IID-clean 참조·천장(두 참값 일치도 0.933) 전표. 3B/7B (a)는 수록 대상 아님 |
| **1C 재현성·안정성** | (CNN) 방법 cross-seed 안정성 · (LLM) (b) 타깃 자기-안정성 | **⟐ 재산출** | 스코프 축소(제외 무대·행 제거) + R4·CNN 축 그리드 행 추가 → **G1 착지 후** |

### 3.2 Downstream `[부록]`

| 실험 | 무대·세팅 | 상태 | 남은 일 |
|---|---|---|---|
| **2-CNN P1 — mnist {dir1, iid}** | 1A-CNN mnist와 동일 무대 · 8점수원 × online/retrain + 기준 arm | **● 완료** (216 점수원 + 기준 arm 18) | 없음 — `recovery` 산출됨(오염 3위협 × 8점수원 × 3seed) |
| **2-CNN P1w — cifar10/mnist {dir1, iid}** | 위와 같은 rundir의 `gatew_v2`/`t2_signw` arm | **● 완료** (mnist도 8점수원 × 4위협 × 2파티션 × 3seed) | **추가 런 0** — 단 롤업 `runs/track_h/analysis/p1w_cnn.csv`가 07-25자로 낡았고 `make_p1w_cnn_table.py::CORE`에 mnist 2쌍이 없다(집계기 두 줄 추가 + 재산출) |
| **2-LLM 표준 개입 무해성 (clean do-no-harm)** | std20 clean-IID · 1B/3B/7B · **vanilla + Flirds 가중 + Flirds 선택만** | **● 완료** | ShapleyFL 가중·FedIF 가중 행 삭제(3행 표로 축소) |

### 3.3 Cost `[부록]`

| 실험 | 상태 | 남은 일 |
|---|---|---|
| **5-LLM runtime — 교차-사일로 silo5** | **● 3-seed** | 제외 방법 행 삭제. **(a)oracle 행 신규**(31,137 s = Flirds의 292× · (b)의 58.6×) — §2.1 silo5 (a)-leg의 가격표 |
| **5-LLM runtime — 소형 앵커 anchor5 (1B·3B·7B)** | **● 3-seed** (+(a)oracle 1B) | 〃. loss-heur C6 교정본 병기 유지 |
| **5-LLM runtime — 표준 부분참여 std20 (1B·3B·7B)** | **● 3-seed** | 〃. 소-cohort 역전(Flirds 1.61×) 서술 유지 |
| **5-CNN runtime 방법별 상세** | **● 완료** | 없음 — [[flirds-results-cost]] §5-CNN |

> **추가 실험 없이 기존 rundir에서 만든 표다.** 방법별 CNN runtime의 출처:
> - `runs/track_c/c2fid/rundirs/*/metrics.json`의 **`methods.<m>.runtime`** — 9방법 × 168셀 전량(`analysis/fidelity.csv`엔 fidelity 지표만 있고 runtime 열은 없다). 예: cifar10/dir1 오염 3종(lf@0.70·frzero·gn) 평균(n=9) → Flirds-1st 4.21±0.10 · FedIF 5.35±0.17 · loss-heur 9.22±0.12 · **Flirds 10.64±0.37** · ComFedSV 23.65±0.58 · FedSV 293.98±4.90 · **(b)oracle 836.58±14.07** · GTG 1079.92±131.64 · ShapleyFL 1468.47±16.34.
> - `runs/track_c/c1/*/metrics.json`의 `methods.<m>.runtime` + `traj_time`(학습) — N=10 full 무대, 3-seed.
>
> **학습(client-training) wall-clock은 싣지 않는다**(valuation-only) → CNN rundir `timing.json` 배선 불필요. 다만 N=10 무대엔 학습 궤적 `traj_time`(mnist 136.3 s / cifar10 104.5 s)과 **(a) 2¹⁰ 재학습 `t_a`(41,168 s / 32,912 s = Flirds의 64,730× / 28,177×)** 를 비교 대상 아닌 _참조 행_으로 병기했다 — (a)-무대를 N=10에 묶어둘 수밖에 없는 이유의 가격표다.

### 3.4 Ablation `[부록]`

| 실험 | 상태 | 남은 일 |
|---|---|---|
| **A축 용량 lever probe — CNN** | **● 완료** (폭 {0.5,1,2,4}× × 참여 {0.2,0.5,1.0}, 22 프리픽스 3-seed + c2 8 프리픽스 3-seed) | 기준칸 w=1·k=1.0은 C1 rundir 재사용 — provenance 각주 |
| **φ 부호 감사 — 게이팅의 작동 전제** | **● 완료** (LLM 레그 파생 · CNN 레그 = C1 축 그리드 48/48 파생) | 없음 — [[flirds-results-ablation]] §4-공통 표 A·B + 표 A/B/C-CNN((a)oracle 행 포함) |

---

## 4. 결손 목록 — 실험 설계 세션 입력

> **런 단위 규약**: CNN track_h는 `<ds>_<part>_<threat>_<source>_seed<N>` 한 rundir가 **온라인 4정책 arm**(P1 `gate_v2` · P1w `gatew_v2` · P3 `mult` · P4 `zgate_v2`)을 함께 낳고, 관측자 rundir `..._obs(f)_seed<N>` 하나가 **vanilla + 전 점수원의 retrain T2 arm**(`t2_sign_*` · `t2_signw_*`)을 낳는다. 그래서 "필요 런"은 **arm 수가 아니라 rundir 수**다. 상세 §7.2.

| ID | 작업 | 무대·러너 | 필요 런(잔여) | 코드 변경 | 산출 축 | 우선도 |
|---|---|---|---|---|---|---|
| **G1** | **R4-L2: 주무대 (b) 오라클 부착** — 현재 **7/9** | `phase2_matrix.py` `REGIME=gsm50k5` (1B · N=50 · 5/50 · R=200) | **2** (= frzero s0 · s2) | **불필요**(regime 구현 완료; `MIN_METHODS`로 seed1·2는 (b)+Flirds+1st만 = 3.2× 절감, 실측 9.37~9.88 h/셀) | 본문 fidelity(T2 frzero 열) | **P0 · 유일 잔여 실험** |
| **G7** | op-count를 N·R·K 파라메트릭으로 재작성 | `runs/measured_2026-07/op_counts.py` | **0**(실행 없음) | 출력 포맷만 | 본문 cost | **P0(문서)** |
| **G13** | *(선택)* loss-heur C6 교정 3B/7B 재측정 | `runs/track_d/rundirs`의 3B·7B 12셀 | **12** | 불필요 | 부록 cost 각주 | **P3** |

> **완료·취소된 결손 ID**(다른 문서의 상호참조용 인덱스): G2 · G3 · G4 · G6 · G8 · G9 · G10 · G14 = **종료** · G5 · G12 = **실행 취소**(3-seed 미달 → 미수록, §5).

### 4.1 총량 요약

| 그룹 | 잔여 런 수 (2026-07-28) | 코드 변경 |
|---|---|---|
| **P0 (본문 필수)** | **G1 2셀** ≈ **20 GPU-h** | 없음 |
| **P1 (본문 필수)** | **0** | 없음 |
| **P2 (부록)** | **0** | 없음 |
| **문서 전용** | G7 = **0** | 없음 |
| **P3 (선택)** | G13 = 12 | 없음 |
| **결정 대기** | §4.2의 clean 보강 3 + 3 | 없음 |

> **필수 실험은 G1 frzero 2셀이 전부다.** 나머지는 전부 표 정리·문서 재작성·집계기 재산출이며 **코드 변경은 한 건도 남지 않았다**.

### 4.2 clean 대조가 없고, 채울 계획도 없는 무대

§0.1은 **"clean은 오염이 아니라 대조 앵커이므로 전 무대에서 열/행으로 유지한다"**고 정했다. 그 규칙과 현재 실행 계획이 어긋나는 지점은 아래 3곳이 전부다(rundir 전수 대조).

| 무대 (수록 위치) | 남는 오염 열 | clean 열 | 필요 런 | 비용 | 판정 |
|---|---|---|---|---|---|
| **1A-LLM 대규모 교차-디바이스 앵커** `[부록 fidelity]` — 본문 cost(device100 runtime)와 **무대를 공유** | noisy · frzero | **전무** — α 전 구간(0.0/0.01/0.1/0.5/5.0) 어디에도 clean 셀이 없다 | **3** (clean × 3seed · α=0.5 앵커 + `ORACLE_B`) | (b) per-round valuation만 셀당 24,975 s ≈ 6.9 h → **~21 GPU-h**(학습시간 별도·미계측) | **결정 필요** — 이 무대의 부록 fidelity 표가 오염 2행만 남는다 |
| **Removal-curve — LLM** `[본문 ablation]` | noisy · frzero | **전무**(silo5 removal은 noisy·frzero·frrand·poison만) | **3** (silo5 clean × 3seed) | removal 재학습 **317.7 s/런** + silo5 학습·valuation → **1 GPU-h 미만** | **채우기 권고** — 거의 공짜인데 CNN 짝과의 비대칭이 사라진다 |
| **Removal-curve — CNN** `[본문 ablation]` | lf@0.70 · frzero · gn (오염축 정렬 완료) | **rundir는 있다** — `cifar10_iid_seed{0,1,2}`(`rates` 전부 0.0 = 완전 무오염) | **0** | — | **표에 행만 추가**(파생) |

> **그 외 수록 무대는 clean 대조가 있다**: 1A-CNN(c2fid clean 3-seed) · 1A-LLM silo5 (b)-leg · 1B-LLM silo5 (a)-leg · 2-CNN cifar10/mnist {dir1, iid} · 2-LLM R4. G1의 잔여 2셀은 frzero라 clean은 이미 3-seed다.
> **iid(저-이질성) 축은 결손 없다**: CNN은 전 무대가 {dir1, iid} 쌍으로 차 있고, LLM은 **본문 주무대 R4(gsm50k5)가 N=50 IID**이고 부록 silo5가 비IID라 IID↔비IID 대조가 수록분만으로 성립한다.

---

## 5. 표에서 빠지는 실험

> 전부 rundir·러너 산출은 **존속**한다. 표에서만 뺀다. 설계 세션은 **이 목록을 다시 계획하지 않는다.**

| 축 | 빠지는 실험 | 사유 |
|---|---|---|
| Fidelity | c2fid **qskew · shard** 파티션 | 스코프 |
| Fidelity | c2fid **fmnist {dir1, iid}** | 부록의 mnist가 대체 |
| Fidelity | 3B silo5 (b)-leg · N=10 완전열거(2¹⁰) | 스코프 |
| Fidelity | **신호 실재성: 오염×비IID 매트릭스**(iid5) | 진단 전용 — 단 §1C 서술 근거로는 인용 가능 |
| Downstream | **개입 정책 축 P3·P4·P5h·P5s** — P1·P1w만 남음 | 스코프 |
| Downstream | strmain 경쟁 · **clean 오발화 분해** · 완전참여 100/100 · 동적 재추첨 · std50k5 mixed | 스코프 |
| Downstream | fmnist·iid 8점수원 경쟁(seed0 파일럿) · 확장 파티션 flirds 단독 표 | 3-seed 미달 |
| Downstream | silo5/iid5 φ-게이팅 회수 표 | R4로 흡수 |
| Downstream | 무해성에서 **ShapleyFL 가중·FedIF 가중** 행 | 비교군 컷(§0.2) |
| Detection | **탐지 축 전량** — CNN 부분참여·전원참여 φ-AUROC · mnist φ-AUROC · LLM 주무대(R4) 탐지 · α-sweep 탐지 · silo5 탐지 · 전용 탐지기 4종 | **축 자체를 논문에서 제외**(2026-07-28) |
| Cost | 지수-비용 스케일링 표 · 학습↔가치평가 위상분리 계측 | 스코프 |
| Cost | Fed-LOO · Banzhaf · Ripple 행 | 비교군 컷(§0.2) |
| Ablation | **2차항(HVP) LLM 레그**(std50k5 rank sweep) · **A축 용량 lever probe LLM 레그** | **3-seed 미달 → 미수록**. 2차항 논지는 **CNN 레그만으로 간다** |
| Ablation | Taylor 물리잔차(P3) · AdamW 브리지 · TF32 A/B · β 통일 provenance | 3-seed 미달 / 검증 전용 |
| Ablation | **Dose-response (CNN·LLM 모두)** | 스코프. LLM noisy nr 스윕 3-seed 실측은 존치 |
| 위협축 | frrand · lf@{0.15,0.35} · lf-strmain · frdelta · mixed · poison · LLM gnoise | 오염축 컷(§0.1) |

> ⚠ **clean 오발화를 빼면 "Flirds가 clean parity를 P1에서 −0.74pt 위반"의 인과 설명(발화율 0.39~0.43)이 사라진다.** 수치 자체는 개입 주 표의 clean 열에 남으므로, 논문에서도 **인과절만 지우고 수치는 남기는** 편이 정직 보고를 지킨다(§6).

---

## 6. 논문 본문(`paper/paper-ko.md`) 수정 대상

> 제외 결정(§0.2·§5)을 논문 쪽에 반영할 때 손대야 하는 곳. **이 문서는 목록만 제시하고 논문 파일은 수정하지 않는다.**

| 제외 항목 | 논문 내 위치 | 할 일 |
|---|---|---|
| **clean 오발화** | §5.3 읽기 ④ "**clean 오발화의 정직 보고**: online에서 Flirds −0.7pt·individual utility −1.3pt(누적 부호의 0-교차 노이즈), Flirds-1st·FedIF는 무발화" | **인과절만 삭제.** −0.7pt 수치는 개입 주 표 clean 열에서 나오고 그 표는 남는다 — 잃는 건 "왜 발화했나"의 설명뿐이라, 수치는 남기는 편이 정직 보고를 지킨다 |
| **Fed-LOO** | 제외 사유 문단에 Banzhaf·Ripple만 명시 | 같은 문단에 Fed-LOO 추가 |

> 부록 D.2의 `LLM-Main ⬚` · `CNN-Main ⬚` 두 빈칸은 각각 **G1(R4-L2)** 과 c2fid rundir 파생으로 채워진다 → §3.1의 1C 항목과 같은 작업이다.

## 7. 실행 규약 (설계 세션 참고)

### 7.1 무대별 러너·환경변수

| 무대 | 러너 | 핵심 env |
|---|---|---|
| CNN 부분참여 fidelity/detection (1A-CNN) | `codes/experiments/track_c2_fid.py` | `C2_DATASET`(cifar10\|fmnist\|mnist) · `C2_PARTITION`(iid\|dir1\|shard\|qskew) · `C2_THREAT`(clean\|label_flip\|free_rider\|frrand\|grad_noise) · `C2_FLIP_RATE` · `C2_SEED` |
| CNN 부분참여 downstream (2-CNN) | `codes/experiments/track_c2.py` (+ track_h arm env) | 위와 동일 + arm 지정 |
| CNN vs (a) (1B-CNN) | `codes/experiments/track_c1.py` | **축 그리드**: `C1_DATASET`(mnist\|cifar10) · `C1_PARTITION`(iid\|dir1) · `C1_THREAT`(clean\|label_flip\|free_rider\|grad_noise) · `C1_FLIP_RATE`(기본 0.70) · `C1_SEED` · `C1_MODE`. **레거시**: `C1_SCENARIO` — 둘 다 미설정 시 레거시 경로가 비트동일 유지 |
| CNN removal-curve | `codes/experiments/track_c1.py` + `runs/removal_dose/sbatch_cnn_removal_axis.sh` | 축 그리드 env + `removal: true` |
| LLM 매트릭스 fidelity/detection | `codes/experiments/phase2_matrix.py` | `REGIME`(silo5\|iid5\|device100\|gsm50k5\|gsm5) · `THREAT` · `SEED` · `NOISY_RATE` · `ORACLE_B` · `COALITION` |
| LLM 개입 (2-LLM) | `codes/experiments/track_g.py` → track_h rundirs_llm | `REGIME` · `THREAT` · arm 지정 |
| LLM 표준무대 (std20·anchor5) | `codes/experiments/track_d.py` | 스케일·레짐 |

> **G1 실행 예**(코드 변경 없음): `REGIME=gsm50k5 THREAT=frzero SEED=0 python -u experiments/phase2_matrix.py` — `NOISY_RATE`는 gsm50k5에서 **0.7이 기본값**이라 생략 가능.

### 7.2 track_h CNN rundir ↔ arm 대응 (런 수 계산의 근거)

| rundir 접미사 | 낳는 arm | 정책 |
|---|---|---|
| `_<source>_seed<N>` | `<src>_gate_v2` · `<src>_gatew_v2` · `<src>_mult` · `<src>_zgate_v2` | **P1 · P1w** · P3 · P4 (온라인) |
| `_<source>p5_seed<N>` | `<src>_cgate` · `<src>_pweight` | P5h · P5s (온라인) |
| `_obs_seed<N>` | `observer`(=vanilla) + **전 8점수원의** `t2_sign_*` · `t2_signw_*` + `t2_random_k*` | **retrain P1 · P1w** |
| `_obsf_seed<N>` | `observer` + **flirds만** `t2_sign_flirds` · `t2_signw_flirds` | 〃 (축소판; `obs`가 있으면 집계기가 `obs`를 우선) |
| `track_g/rundirs_cnn/<ds>_<part>_<threat>_g_seed<N>` | `vanilla` · **`oracle_excl`(천장)** · **`random_excl`(통제)** + flirds 온라인 6종 | recovery의 분모 |

> ⇒ **P1과 P1w는 같은 rundir에서 동시에 나온다** → 부록의 P1w 항목은 **추가 런 0**.
> ⇒ `track_g/rundirs_cnn` = **162셀**(cifar10 96 + fmnist 48 + mnist 18)이고 track_h 집계기가 두 경로를 함께 읽는다. clean 셀은 오염 클라가 0이라 `oracle_excl`·`random_excl`이 정의되지 않는다.
> ⇒ **스택 고정**: recovery의 분모(track_g)와 분자(track_h 소스 arm)는 같은 torch 스택이어야 한다(현행 2.11).

### 7.3 op-count 파라메트릭 식 (G7 재작성 입력)

`runs/measured_2026-07/op_counts.py::per_round(method, K, N)`에서 그대로 읽은 **라운드당** 지배연산. R=라운드 수, K=라운드당 참여 클라 수(=cohort, full 참여면 K=N), N=총 클라 수.

| 방법 | 라운드당 | 전체(× R) | K 의존성 |
|---|---|---|---|
| **Flirds** | 1 HVP + K dot | **R HVP** | **없음 (cohort 무관 상수)** |
| Flirds-1st | 1 grad + K dot | R grad | 없음 |
| FedIF | 1 grad | R grad | 없음 |
| loss-heur | (1+K) fwd | R(1+K) fwd | **선형** |
| ShapleyFL | 2^K fwd | R·2^K fwd | **지수** |
| **(b) exact** | 2^K fwd | **R·2^K fwd** | **지수** |
| FedSV | ≤ min(2^K, max(30,2K)·K) fwd | ≤ R·min(…) | 지수 상한(캐시·TMC 절단으로 하락) |
| GTG | ≤ min(2^K, max(30,⌈0.8·2^K⌉)·K) fwd | ≤ R·min(…) | 〃 |
| ComFedSV | ≤ 1 + min(2^K, M·K) fwd, M=max(10,⌈N ln N⌉) | ≤ R(1+min(…)) fwd + CPU ALS | 지수 상한 **+ 유일하게 N 의존** |

> **논문에서 말할 것**: Flirds의 비용은 **R에만 비례하고 K·N에 무관**(라운드당 HVP 1회, cohort 크기와 독립)인 반면 exact (b)·ShapleyFL은 **K에 지수적**, loss-heur는 K에 선형, ComFedSV만 N까지 들어온다. 시간 환산은 microbench 대입(fp32·B200: forward 1.60 s · HVP 10.36 s, HVP/fwd=6.47) — 하드웨어가 바뀌면 이 두 수만 갈아끼운다.
> ⇒ 이 구조가 **소-cohort 역전**(K=2인 std20에서 Flirds가 (b)보다 1.61× 비쌈: 1 HVP≈6.47 fwd > 2²=4 fwd)과 **대-cohort 압승**(K=10인 device100에서 159×)을 **하나의 식으로 동시에** 설명한다.

---

## 상호 링크

- 분류·이름 정본: [[flirds-experiment-axis-map]]
- 수치 정본: [[flirds-results-fidelity]] · [[flirds-results-downstream]] · [[flirds-results-detection]] · [[flirds-results-ablation]] · [[flirds-results-cost]]
- 승패 메커니즘 해석: [[flirds-principle-analysis]]
- 비용 방법론: [[cost-comparison-methodology-2026-07/cost-comparison-methodology]]
- 결정 근거 메모: [[all-experiments-3-seed]] · [[paper-threat-stage-scope]] · [[banzhaf-ripple-excluded-from-baselines]]
