---
type: survey
title: "Flirds 논문 수록 실험 재편성 — 본문/부록 확정 목록 + 결손 실험"
created: 2026-07-25
updated: 2026-07-25
tags: [flirds, paper, experiment-plan, scope, gap-analysis]
---

# Flirds 논문 수록 실험 재편성 (본문/부록 확정 + 결손)

> **무엇**: Yonghee가 2026-07-25에 확정한 **논문 본문/부록 수록 실험 목록**을 [[flirds-experiment-axis-map]]의 분류·이름 체계로 옮기고, 각 항목이 **지금 실측으로 채워져 있는지**를 rundir 대조로 판정한 문서.
> **왜 두 용도인가**: ① 논문 집필 세션은 "무엇을 어느 표로 싣나"를 여기서 읽는다 ② **실험 설계 세션은 §4(결손 목록)를 입력으로 받아** 부족한 실험을 설계한다. 그래서 "완료"만이 아니라 **무엇이 비었고 몇 런이 필요하며 코드 변경이 필요한지**까지 적는다.
> **수치는 여기 없다** — 실제 값은 축별 결과 페이지가 담당한다: [[flirds-results-fidelity]] · [[flirds-results-downstream]] · [[flirds-results-detection]] · [[flirds-results-ablation]] · [[flirds-results-cost]]
> **판정 근거**: 2026-07-25 기준 `runs/` rundir 전수 대조(c2fid `fidelity.csv` · track_h `cnn_competition.csv`/`llm_competition.csv` · track_h/track_g rundir 이름 파싱 · track_c/c1 · track_d · phase2_matrix · removal_dose · probe_signal). 마커는 실측이며 추정이 아니다.

---

## 0. 이번 재편성의 스코프 규칙 (Yonghee 2026-07-25 확정)

### 0.1 오염축 — 이 다섯 개만 쓴다

| 트랙 | 오염 위협 | 코드 토큰 | 비고 |
|---|---|---|---|
| **CNN** | label-flip **@0.70** | `label_flip` + `C2_FLIP_RATE=0.70` | dose 0.15·0.35는 축 밖 |
| **CNN** | free-rider-zero | `free_rider` | Δ=0 업데이트, φ exact-0 |
| **CNN** | gradient-noise | `grad_noise` | 2차항 존재 이유의 결정 칸 |
| **LLM** | answer-swap **@0.7** | `noisy` (`NOISY_RATE=0.7`) | = 코드의 `answer_swap_graded` |
| **LLM** | free-rider-zero | `frzero` | 〃 |

> **clean은 오염이 아니라 대조 앵커**다 — 제거하지 않고 전 무대에서 열/행으로 유지한다(무해성 parity·오발화 판정의 근거). 아래 "오염축 커버" 표기는 **오염 위협만** 센다.
> **축 밖으로 빠지는 위협**: `frrand`(랜덤-델타 free-rider) · `label_flip@{0.15,0.35}` · `lf-strmain`(강한-주류 변종) · `frdelta` · `mixed` · `poison`(이미 07-22 제외) · LLM `gnoise`.

### 0.2 비교군 — 전용 탐지기와 Fed-LOO는 논문 전체에서 제외

| 상태 | 방법 |
|---|---|
| **수록 (9행)** | **(b)oracle** + same-game 3종(**Flirds · Flirds-1st · loss-heur**) + cross-game 5종(**GTG · FedSV · ComFedSV · ShapleyFL β=0.3 · FedIF**) |
| **제외 (신규)** | **전용 탐지기 4종**(FLDetector · FLTrust · STD-DAGMM · FedDQC) · **Fed-LOO** |
| 제외 (기존) | Banzhaf(07-22) · Ripple(07-19) · Fed-LOO(07-23 → 이번에 재확인) |

> 러너는 계속 이 열들을 산출한다 — **집계·표 단계에서만 뺀다**(rundir·CSV는 존속). 기존 결과 페이지의 _기울임_ 탐지기 행과 Fed-LOO·Banzhaf 행은 전부 삭제 대상이다.
> ⚠ 파급: [[flirds-results-detection]] §3-LLM의 전용탐지기 대조가 사라지면서 "φ-as-detector의 noisy 약세를 전용 탐지기가 이긴다"는 서술의 근거도 함께 빠진다 — **각주로도 살리지 않는다**. 논문 본문엔 해당 서술이 애초에 없다(§6).

### 0.3 seed — 예외 없이 3-seed

수록 실험은 **seed{0,1,2} 전부**. seed 부족분은 채우고, 미실행은 새로 돌린다. 근거: [[all-experiments-3-seed]](2026-07-24 Yonghee).

### 0.4 마커

● 3-seed 실측 · ◐ 부분/1-seed(수록 불가) · ⬚ 미실행 · ⟐ 파생/재분석(재실행 0)

---

## 1. 한눈에 — 배치 × 축 × 상태

| 배치      | 축          | 실험 (축-지도 이름)                                         | 상태                                |
| ------- | ---------- | ---------------------------------------------------- | --------------------------------- |
| **본문**  | Fidelity   | 1A-CNN 대규모 교차-디바이스 부분참여 — cifar10 {dir1, iid}        | ● 완료                              |
| **본문**  | Fidelity   | 1A-LLM 주무대 정확도-무대 충실도 (R4)                           | ⬚ **미실행**                         |
| **본문**  | Fidelity   | 1A-LLM 표준 부분참여 충실도 (1B·3B·7B)                        | ● 완료 (clean-IID 전용 무대)            |
| **본문**  | Fidelity   | 1B-CNN vs (a) — cifar10 {dir1, iid}                  | ◐ **무대 불일치**(오염축·파티션 재실행)         |
| **부록**  | Fidelity   | 1B-LLM 소형 앵커 듀얼오라클 vs (a) — **부록 강등**(07-25) | ● 완료(1B) (clean-IID 참조·폴백)        |
| **본문**  | Fidelity   | 1B-LLM 교차-사일로 (a)-leg 듀얼오라클 — silo5 3위협              | ● 완료 · **본문 수록 확정**(07-25, §2.1 권고 채택)     |
| **본문**  | Downstream | 2-CNN P1 부호-게이트 online/retrain — cifar10 {dir1, iid} | dir1 ● / **iid ⬚(flirds 단독)**     |
| **본문**  | Downstream | 2-LLM 주무대 정확도 개입 (R4 GSM8K EM) — **arm = vanilla·oracle·random·Flirds·Flirds-1st만**(07-25 축소) | ◐ **잔여 = online 1st·clean 시드** |
| **본문**  | Cost       | 5-공통 op-count 모델 — **N·R·K 파라메트릭**                   | ⟐ **재작성**(실행 0)                   |
| **본문**  | Cost       | 5-LLM 실측 runtime — device100 앵커                      | ● 완료                              |
| **본문**  | Ablation   | 2차항(HVP)의 기여 — CNN 레그                                | ● 완료                              |
| **본문**  | Ablation   | 2차항(HVP)의 기여 — LLM 레그                                | ◐ **seed 부족**                     |
| **본문**  | Ablation   | Removal-curve — CNN                                  | ◐ **오염축 불일치**                     |
| **본문**  | Ablation   | Removal-curve — LLM                                  | ● 완료                              |
| **부록**  | Fidelity   | 1A-CNN mnist {dir1, iid}                             | ⬚ **전량 미실행**                      |
| **부록**  | Fidelity   | 1A-LLM 교차-사일로 (b)-leg                                | ● 완료                              |
| **부록**  | Fidelity   | 1A-LLM 대규모 교차-디바이스 앵커                                | ● 완료                              |
| **부록**  | Fidelity   | 1B-CNN mnist {dir1, iid} vs (a)                      | ⬚ **전량 미실행**                      |
| **부록**  | Fidelity   | 1C 재현성·안정성                                           | ⟐ **재산출**(스코프 축소분)                |
| **부록**  | Downstream | 2-CNN mnist {dir1, iid} P1 online/retrain            | ⬚ **전량 미실행**                      |
| **부록**  | Downstream | 2-CNN P1w — cifar10/mnist {dir1, iid}                | cifar10/dir1 ● / 나머지 **⬚(동반 산출)** |
| **부록**  | Downstream | 2-LLM 표준 개입 무해성 (vanilla·Flirds 가중/선택만)              | ● 완료                              |
| **부록**  | Cost       | 5-LLM runtime — silo5 · anchor5 · std20              | ● 완료                              |
| **부록**  | Cost       | 5-CNN runtime 상세(방법별)                                | ● 완료 (실행 0 · 기존 rundir 파생)        |
| **부록**  | Ablation   | A축 용량 lever probe — CNN                              | ● 완료                              |
| **부록**  | Ablation   | A축 용량 lever probe — LLM                              | ◐ **seed 부족**                     |
| **부록**  | Ablation   | **φ 부호 감사 — 게이팅의 작동 전제**                             | ● 완료 (실행 0 · 기존 rundir 파생)        |
| **부록**  | Detection  | 본문/부록 무대에서 발췌                                        | 무대 상태에 종속(§3.5)                   |

> **결손의 무게중심 3곳**: ① **LLM 주무대 (b) 오라클(R4-L2)** — 본문 fidelity·부록 detection이 동시에 걸려 있다 ② **mnist 무대 전량** — 부록 fidelity·downstream 4항목이 여기 걸린다 ③ **cifar10/iid 비-flirds 점수원 7종** — 본문 downstream의 절반.

---

## 2. 본문 수록 예정

### 2.1 Fidelity `[본문]`

| 실험 (축-지도) | 무대·세팅 | 오염축 커버 | 상태 | 남은 일 |
|---|---|---|---|---|
| **1A-CNN 대규모 교차-디바이스 부분참여** | FedSVCNN · cifar10 {dir1, iid} · N=100 · 10/100 · R=120 · (b) per-round | lf@0.70 ● · frzero ● · gn ● (+clean ●) | **● 완료** | 없음. 기존 표에서 qskew·shard·fmnist 파티션 및 축 밖 위협 열 삭제만 |
| **1A-LLM 주무대 정확도-무대 충실도 (R4)** | Llama-3.2-1B · gsm50k5 N=50 · 5/50 · R=200 · (b) per-round · GSM8K | swap@0.7 ⬚ · frzero ⬚ | **⬚ 미실행** | **G1** — 6 셀(+clean 3) |
| **1A-LLM 표준 부분참여 충실도 (1B·3B·7B)** | alpaca IID · N=20 · 2/round · R=200 · (b) per-round | 오염 없음(clean-IID 전용) | **● 완료** | **없음.** 표 캡션에 "clean-IID 전용 무대" 명시만 |
| **1B-CNN 소형 교차-사일로 vs (a)** | 현: LeNet5/FedSVCNN · N=10 · full · R=10 · (a) 2¹⁰ + (b) 2¹⁰ | 현 시나리오 = {feature-noise, label-flip(dose 사다리), quantity-skew, label-skew, iid} → **축 3종 중 0개 일치** | **◐ 무대 불일치** | **G2** — 러너 확장 + cifar10 18 셀 |
| **1B-LLM 소형 앵커 듀얼오라클 vs (a)** | Llama-3.2-1B · anchor5 N=5 · full · R=30 · (a) 2⁵ | 오염 없음(clean-IID 전용) | **● 완료(1B)** | **부록 강등(07-25)** — 본문 자리는 silo5 (a)-leg가 대체. 3B/7B (a)는 ⬚(수록 대상 아님) |
| **1B-LLM 교차-사일로 (a)-leg** | Llama-3.2-1B · silo5 N=5 5도메인 비IID · full · R=10 · (a) 2⁵ = 32 재학습 | swap ● · frzero ● (+clean ●) — **오염축 3/3 정렬** | **● 3-seed** | **본문 수록 확정(07-25 — 권고 채택)**. 남은 일 = 값 전사(전 방법 확장의 ComFedSV clean 열 ⬚) |

> ✅ **07-25 확정 — 권고 채택: silo5 본문, anchor5 부록(참조·폴백).** (아래는 결정 당시 논거 보존.) 07-25 확정 목록엔 `[1B-LLM 소형 앵커]`만 있었는데, 그 뒤 **silo5 (a)-leg 9런(3위협×3seed)이 착지**했다. 두 후보의 성질이 다르다:
> - **anchor5**(현 수록): IID-clean 전용 · (b) 타깃이 seed-불안정(xseed −0.367) · **듀얼오라클 일치도 0.933이 천장** — same-game 방법들의 0.933은 방법 한계가 아니라 무대 한계다.
> - **silo5**(신규): 오염축 {clean, swap, frzero} 3/3 정렬 · (b) 타깃 seed-안정(xseed +0.87~+0.93) · **듀얼오라클 일치도 1.000** — "in-run (b)를 같은-게임 정답으로 쓴다"는 설계 선택을 방법-중립 참값이 **직접 승인**한다. 단 answer-swap 비율이 이 무대 canonical **nr=1.0**(R4의 0.7과 다름)이고 ComFedSV clean 열이 ⬚다.
> **권고: silo5를 본문, anchor5를 부록(스케일·폴백)으로.** 논거 강도·오염축 정렬·§0.3 3-seed 모두 silo5가 앞선다. 수치는 [[flirds-results-fidelity]] §1B-LLM. → **채택됨(07-25)** — `paper-ko.md` 표 F4 = silo5, Anchor는 부록 C.6로 이동 완료.

> **1B-CNN 주의**: (a) 재학습 오라클은 2^N 재학습이라 **N=100에서 원리적으로 불가** → 1A-CNN(N=100·부분참여)과 **N·참여율은 맞출 수 없다**. 정렬 가능한 건 **오염축과 파티션**뿐이다(N=10 유지). 이 비교불가성은 논문에도 명시해야 한다(현 track_c2_fid 헤더의 CAVEAT와 같은 취지).

### 2.2 Downstream `[본문]`

| 실험 | 무대·세팅 | 오염축 커버 | 상태 | 남은 일 |
|---|---|---|---|---|
| **2-CNN P1 부호-게이트 — cifar10/dir1** | 1A-CNN과 동일 무대 · 8점수원 × {online 배포게이팅, retrain T2} · 절대 test acc · 앵커 vanilla/oracle_excl/random_excl | lf@0.70 ● · frzero ● · gn ● (+clean ●) | **● 완료** | 없음 |
| **2-CNN P1 부호-게이트 — cifar10/iid** | 〃 파티션만 iid | **flirds만 ●**, 나머지 7종 ⬚ | **⬚ 미완** | **G3** — 96 rundir |
| **2-LLM 주무대 정확도 개입 (R4 GSM8K EM)** | gsm50k5 · online gate_v2 + retrain t2_sign · 절대 EM(test 1,119) · **수록 arm = {vanilla, oracle-제외, random, Flirds, Flirds-1st}만(Yonghee 07-25 수정)** | retrain: swap ●·frzero ● (Flirds·Flirds-1st ●; loss-heur·FedIF도 기산출이나 **표 미수록**) / online: **Flirds 1종만** ● / clean ◐ seed0 | **◐** | **G4(축소)** — online Flirds-1st(2위협×3seed=**6**) + clean 열(vanilla·Flirds·Flirds-1st) 3-seed화 ≈ **~15 런**. renorm 4종·loss-heur·FedIF 확장(구 ~111런)은 **스코프 아웃** |

> ~~G4의 clean 열 범위는 설계 선택~~ → **07-25 해소**: 수록 점수원이 Flirds 계열로 좁혀지며 clean 열도 vanilla·Flirds·Flirds-1st만 3-seed화하면 된다.

> 앵커(vanilla·oracle_excl·random_excl)는 **cifar10 4파티션 · fmnist 2파티션 전 위협 3-seed 확보** — iid 결손은 **점수원 런에만** 있다.

### 2.3 Cost `[본문]`

| 실험 | 요구 | 상태 | 남은 일 |
|---|---|---|---|
| **5-공통 연산수(op-count) 모델** | 고정 숫자가 아니라 **N(총 클라)·R(라운드)·K(라운드당 참여)의 함수**로 제시 | ⟐ **재작성** | **G7** — 실행 0. `op_counts.py::per_round(method,K,N)`이 이미 파라메트릭이라 **식만 뽑아 쓰면 된다**(§7.3) |
| **5-LLM 실측 runtime — device100 앵커** | N=100 · 10/100 · α=0.5 · 1B · 방법별 valuation wall-clock | **● 3-seed 완료** | 표에서 탐지기 4종 행 삭제 |

### 2.4 Ablation `[본문]`

| 실험 | 무대 | 상태 | 남은 일 |
|---|---|---|---|
| **2차항(HVP)의 기여 — CNN 레그** | c2fid grad-noise 셀(Flirds ρ vs Flirds-1st ρ) + probe_signal k-sweep | **● 완료** | 표를 cifar10 {dir1, iid} 2행으로 축소 |
| **2차항(HVP)의 기여 — LLM 레그** | std50k5(N=50·5/50) LoRA rank {16,32,64} × Spearman vs (b) | **◐** r16 ● / r32·r64 seed0 | **G5** — 4 셀 |
| **Removal-curve — CNN** | worst-first vs best-first 제거 후 재학습 acc 분리 | **◐** 현 시나리오 {feature-noise, label-flip, iid} = 축 밖 | **G6** — 6~9 셀 |
| **Removal-curve — LLM** | silo5 N=5 · worst/best-first 제거 후 val-loss | **● 완료**(noisy·frzero 3-seed) | frrand 행 삭제 |

---

## 3. 부록 수록 예정

### 3.1 Fidelity `[부록]`

| 실험 | 무대·세팅 | 상태 | 남은 일 |
|---|---|---|---|
| **1A-CNN mnist {dir1, iid}** | 1A-CNN(cifar10)과 **동일 세팅**, 데이터셋만 mnist | **⬚ 전량 미실행** — c2fid는 cifar10·fmnist만 지원 | **G8** — 러너에 mnist 추가 + 24 셀 |
| **1A-LLM 교차-사일로 도메인 충실도 ((b)-leg)** | 1B · silo5 N=5 5도메인 · full · R=10 · (b) 2⁵ | **● 완료**(swap·frzero 3-seed) | clean·frrand 행 정리 |
| **1A-LLM 대규모 교차-디바이스 앵커 충실도** | 1B · N=100 α=0.5 · (b) per-round | **● 완료** | frrand 행 삭제 |
| **1B-CNN mnist {dir1, iid} vs (a)** | 1B-CNN과 동일, mnist | **⬚ 전량 미실행** | **G9** — G2와 같은 러너 확장 + 18 셀 |
| **1B-LLM 소형 앵커 vs (a)** (본문→부록 강등 07-25) | anchor5 N=5 full R=30 · (a) 2⁵ | **● 3-seed(1B)** | 없음 — IID-clean 참조·천장(두 참값 일치도 0.933) 전표 |
| **1C 재현성·안정성** | (CNN) 방법 cross-seed 안정성 · (LLM) (b) 타깃 자기-안정성 | **⟐ 재산출** | 스코프 축소(iid5·std20 3B/7B·poison 행 제거) + R4·신규 CNN 무대 행 추가 → **G1·G2·G8 착지 후** |

### 3.2 Downstream `[부록]`

| 실험 | 무대·세팅 | 상태 | 남은 일 |
|---|---|---|---|
| **2-CNN P1 — mnist {dir1, iid}** | 1A-CNN mnist와 동일 무대 · 8점수원 × online/retrain | **⬚ 전량 미실행** | **G10** — 216 rundir |
| **2-CNN P1w — cifar10/mnist {dir1, iid}** | 위와 같은 rundir의 `gatew_v2`/`t2_signw` arm | cifar10/dir1 **●** · 나머지 ⬚ | **추가 런 0** — G3·G10 rundir에 **동반 산출**(§7.2) |
| **2-LLM 표준 개입 무해성 (clean do-no-harm)** | std20 clean-IID · 1B/3B/7B · **vanilla + Flirds 가중 + Flirds 선택만** | **● 완료** | ShapleyFL 가중·FedIF 가중 행 삭제(3행 표로 축소) |

### 3.3 Cost `[부록]`

| 실험 | 상태 | 남은 일 |
|---|---|---|
| **5-LLM runtime — 교차-사일로 silo5** | **● 3-seed** | 탐지기 4종·Fed-LOO·Banzhaf 행 삭제. **(a)oracle 행 신규**(31,137 s = Flirds의 292× · (b)의 58.6×) — §2.1 silo5 (a)-leg를 수록하면 그 가격표 |
| **5-LLM runtime — 소형 앵커 anchor5 (1B·3B·7B)** | **● 3-seed** (+(a)oracle 1B) | 〃. loss-heur C6 교정본 병기 유지 |
| **5-LLM runtime — 표준 부분참여 std20 (1B·3B·7B)** | **● 3-seed** | 〃. 소-cohort 역전(Flirds 1.61×) 서술 유지 |
| **5-CNN runtime 방법별 상세** | **● 완료** | 없음 — [[flirds-results-cost]] §5-CNN. 아래 참조 |

> **추가 실험 없이 기존 rundir에서 만든 표다.** 방법별 CNN runtime의 출처:
> - `runs/track_c/c2fid/analysis/fidelity.csv`의 **`runtime_s` 열** — 9방법 × 144셀(6 데이터셋·파티션 × 8위협 × 3seed) 전량. 예: cifar10/dir1 오염 3종 평균(n=9) → Flirds-1st 4.21±0.10 · FedIF 5.35±0.17 · loss-heur 9.22±0.12 · **Flirds 10.64±0.37** · ComFedSV 23.65±0.58 · FedSV 293.98±4.90 · **(b)oracle 836.58±14.07** · GTG 1079.92±131.64 · ShapleyFL 1468.47±16.34.
> - `runs/track_c/c1/*/metrics.json`의 `methods.<m>.runtime` + `traj_time`(학습) — N=10 full 무대, 3-seed.
>
> **학습(client-training) wall-clock은 싣지 않는다**(valuation-only) → CNN rundir `timing.json` 배선 불필요. 다만 N=10 무대엔 학습 궤적 `traj_time`(mnist 136.3 s / cifar10 104.5 s)과 **(a) 2¹⁰ 재학습 `t_a`(41,168 s / 32,912 s = Flirds의 64,730× / 28,177×)** 를 비교 대상 아닌 _참조 행_으로 병기했다 — (a)-무대를 N=10에 묶어둘 수밖에 없는 이유의 가격표다.

### 3.4 Ablation `[부록]`

| 실험 | 상태 | 남은 일 |
|---|---|---|
| **A축 용량 lever probe — CNN** | **● 완료** (폭 {0.5,1,2,4}× × 참여 {0.2,0.5,1.0}, 22 프리픽스 3-seed + c2 8 프리픽스 3-seed) | 기준칸 w=1·k=1.0은 C1 rundir 재사용 — provenance 각주 |
| **A축 용량 lever probe — LLM** | **◐** 핵심축만 3-seed | **G12** — 23 셀 |
| **φ 부호 감사 — 게이팅의 작동 전제** | **● 완료** (실행 0 · 파생) | 없음 — [[flirds-results-ablation]] §4-공통에 표 2개(**A** clean 클라 φ≤0 = 오배제 위험 · **B** 오염 클라 φ≤0 + exact-0 병기). ⚠ **CNN 레그는 미커버**(감사 스냅샷에 frzero·grad-noise 없음) → G2·G8 후 재감사 |

### 3.5 Detection `[부록 — 발췌]`

> 규칙: **본문/부록에 이미 들어가는 무대의 rundir에서만** 뽑는다(새 무대 금지). 전용 탐지기 4종이 빠지므로 표는 **(b)oracle + 8 추정량** 9행.

| 후보 | 출처 무대 | 왜 뽑나 | 상태 |
|---|---|---|---|
| **CNN 부분참여 φ-AUROC — cifar10 {dir1, iid}** | 1A-CNN과 **같은 rundir** | ① frzero/frrand서 renorm 완전 붕괴(0.00~0.29) ② grad-noise서 Flirds-1st 실명(0.49) = 2차항 논지의 탐지축 재현 ③ Flirds의 (b)-동행(Δ≤0.05) | **● 완료** |
| **CNN 부분참여 φ-AUROC — mnist {dir1, iid}** | 1A-CNN mnist | 데이터셋 강건성 | **⬚**(G8 동반) |
| **LLM 주무대 탐지 (R4)** | 1A-LLM R4-L2와 **같은 rundir** | §2 약속 이행 + LLM 규모의 (b)-동행 판정 | **⬚**(G1 동반) |
| **LLM 교차-디바이스 α-sweep 탐지** | 부록 fidelity device100 | frzero는 전 α 1.000(배경 무관) / noisy는 α 비단조 0.57~0.77 — **(b)도 0.604** = "기여도≠탐지"의 정직한 근거 | **● 완료** |
| *(제외 권고)* LLM 교차-사일로 탐지 | silo5 | φ 9종 전부 1.000(N=5 coarse 천장) → 탐지기가 빠지면 **변별력 0인 표만 남는다** | ● 완료지만 무정보 |

---

## 4. 결손 목록 — 실험 설계 세션 입력

> **런 단위 규약**: CNN track_h는 `<ds>_<part>_<threat>_<source>_seed<N>` 한 rundir가 **온라인 4정책 arm**(P1 `gate_v2` · P1w `gatew_v2` · P3 `mult` · P4 `zgate_v2`)을 함께 낳고, 관측자 rundir `..._obs(f)_seed<N>` 하나가 **vanilla + 전 점수원의 retrain T2 arm**(`t2_sign_*` · `t2_signw_*`)을 낳는다. 그래서 아래 "필요 런"은 **arm 수가 아니라 rundir 수**다. 상세 §7.2.
> **G1–G13은 전부 그대로 열려 있다.** rundir 재대조에서 확인된 현 상태: gsm50k5 (b) rundir **0개**(G1) · cifar10/iid 점수원은 **flirds 1종뿐**(G3) · R4 online은 flirds P1만 3-seed·retrain은 4종 P1만 3-seed·clean 열은 전 arm seed0(G4) · `track_c1` 시나리오는 여전히 {iid, label-skew, quantity-skew, label-flip, feature-noise}(G2·G9) · c2fid 데이터셋은 {cifar10, fmnist}(G8) · removal_dose CNN은 {iid, label-flip, feature-noise}(G6) · probe seed 부족 23셀 그대로(G12). 신규 착지(silo5 (a)-leg 9런)는 이 중 **어느 항목도 닫지 않는다** — §2.1의 수록 결정 항목이다.

| ID | 작업 | 무대·러너 | 필요 런 | 코드 변경 | 산출 축 | 우선도 |
|---|---|---|---|---|---|---|
| **G1** | **R4-L2: 주무대 (b) 오라클 부착** | `phase2_matrix.py` `REGIME=gsm50k5` (1B · N=50 · 5/50 · R=200) | **6** (=2위협 × 3seed) **+clean 3** | **불필요**(regime 구현 완료) | 본문 fidelity **+** 부록 detection | **P0** |
| **G3** | cifar10/iid 비-flirds 점수원 7종 | `track_c2.py`+track_h arm | **84** 점수원 + **12** 관측자 재실행 = **96** | 불필요 | 본문 downstream **+** 부록 P1w | **P0** |
| **G4** | R4 개입 — **arm 축소판(07-25)**: online Flirds-1st + clean 열(vanilla·Flirds·Flirds-1st) 3-seed화 | track_h `rundirs_llm` (gsm50k5) | online 1st **6**(2위협×3seed) + clean **~9** ≈ **~15** *(구안 renorm 4종+online 7종 ~111런은 스코프 아웃)* | 불필요 | 본문 downstream | **P0** |
| **G2** | 1B-CNN (a)-오라클을 제한 오염축·{dir1,iid}로 정렬 | `track_c1.py` (N=10 full · (a) 2¹⁰ + (b) 2¹⁰) | **18** (=cifar10 × 2파티션 × 3위협 × 3seed) **+clean 6** | **필요** — `C1_SCENARIO`에 `free_rider`·`grad_noise` 추가, 파티션 축(`dir1`) 도입 | 본문 fidelity | **P1** |
| **G5** | 2차항 LLM 레그 seed 보강 | `probe_signal` std50k5 rank sweep | **4** (r32·r64 × seed{1,2}) | 불필요 | 본문 ablation | **P1** |
| **G6** | Removal-curve CNN 오염축 정렬 | `removal_dose` CNN | **6**(frzero·gn × 3seed) **+3**(lf@0.70 재실행 시) | G2와 같은 확장 공유 | 본문 ablation | **P1** |
| **G8** | 1A-CNN mnist 무대 신설 | `track_c2.py` / `track_c2_fid.py` | **24** (=2파티션 × 4위협(clean 포함) × 3seed) | **필요(소)** — `track_c2.py:157` `MODEL_FN` 맵에 `"mnist": LeNet5` 추가(`flirds/data/cnn.py`는 이미 MNIST 로더 보유; 파티션·정규화 경로는 확인 필요) | 부록 fidelity **+** 부록 detection | **P2** |
| **G9** | 1B-CNN mnist vs (a) | `track_c1.py` | **18** **+clean 6** | G2와 동일 확장 | 부록 fidelity | **P2** |
| **G10** | 2-CNN mnist downstream | track_h CNN | 관측자 **24** + 점수원 **192** = **216** | G8 확장 공유 | 부록 downstream(P1·P1w 동시) | **P2** |
| **G12** | A축 lever probe LLM seed 보강 | `probe_signal/rundirs`·`noise_probe` | **23** | 불필요 | 부록 ablation | **P2** |
| **G7** | op-count를 N·R·K 파라메트릭으로 재작성 | `runs/measured_2026-07/op_counts.py` | **0**(실행 없음) | 출력 포맷만 | 본문 cost | **P0(문서)** |
| **G13** | *(선택)* loss-heur C6 교정 3B/7B 재측정 | `track_d` rundirs_e4 | **12** | 불필요 | 부록 cost 각주 | **P3** |

### 4.1 G12 내역 (seed 부족 셀)

| 셀 프리픽스 | 현 seed | 부족 |
|---|---|---|
| `1B_anchor5_lr1e-3_st10` | 2 | 1 |
| `1B_anchor5_lr1e-3_st20` · `_st30` | 1 · 1 | 2 · 2 |
| `1B_anchor5_lr2e-3_st20` · `_st30` | 1 · 1 | 2 · 2 |
| `1B_anchor5_lr3e-3_st20` · `_st30` | 1 · 1 | 2 · 2 |
| `1B_anchor5_r32` · `_r64` | 1 · 1 | 2 · 2 |
| `1B_std50k5_r32` · `_r64` | 1 · 1 | 2 · 2 |
| `noise_1B_r64` | 1 | 2 |
| **합** | | **23** |

> `lr2e-3_st10` · `lr3e-3_st10` · `std50k5_r16` · `noise_1B_r16`은 이미 3-seed. **핵심 미확인 질문**은 "lr로 커진 φ가 cross-seed 실재 신호인가"(현 예측 ρ≈0)이고, 그 검증에 필요한 건 `lr{2,3}e-3` 계열 seed1·2다.

### 4.2 총량 요약

| 그룹 | 런 수 | 코드 변경 |
|---|---|---|
| **P0 (본문 필수)** | G1 6~9 · G3 96 · G4 ~15 = **~120** | 없음 |
| **P1 (본문 필수, 러너 확장)** | G2 18~24 · G5 4 · G6 6~9 = **~30** | `track_c1.py` 위협·파티션 확장 |
| **P2 (부록)** | G8 24 · G9 18~24 · G10 216 · G12 23 = **~285** | `track_c2.py` mnist 1줄 + G2 확장 공유 |
| **문서 전용** | G7 = **0** | 없음 |

> **mnist 축(G8·G9·G10 = ~264 런)이 전체의 절반 이상**이다 — **전량 실행**(단계적 게이트 없음). 코드 변경은 `track_c2.py` mnist 1건 + `track_c1.py` 위협·파티션 확장 1건뿐이고 나머지는 셀 수 문제다.

### 4.3 clean 대조가 없고, 채울 계획도 없는 무대

§0.1은 **"clean은 오염이 아니라 대조 앵커이므로 전 무대에서 열/행으로 유지한다"**고 정했다. 그 규칙과 현재 실행 계획이 어긋나는 지점은 아래 4곳이 전부다(rundir 전수 대조).

| 무대 (수록 위치) | 남는 오염 열 | clean 열 | 필요 런 | 비용 | 판정 |
|---|---|---|---|---|---|
| **1A-LLM 대규모 교차-디바이스 앵커** `[부록 fidelity]` — 본문 cost(device100 runtime)·부록 detection이 **공유하는 무대** | noisy · frzero (frrand는 축 밖 삭제) | **전무** — α 전 구간(0.0/0.01/0.1/0.5/5.0) 어디에도 clean 셀이 없다 | **3** (clean × 3seed · α=0.5 앵커 + `ORACLE_B`) | (b) per-round valuation만 셀당 24,975 s ≈ 6.9 h → **~21 GPU-h**(학습시간 별도·미계측) | **결정 필요** — 이 무대의 부록 fidelity 표가 오염 2행만 남는다 |
| **Removal-curve — LLM** `[본문 ablation]` | noisy · frzero (frrand 삭제) | **전무** | **3** (silo5 clean × 3seed) | removal 재학습 **317.7 s/런** + silo5 학습·valuation → **1 GPU-h 미만** | **채우기 권고** — 거의 공짜인데 CNN 짝과의 비대칭이 사라진다 |
| **Removal-curve — CNN** `[본문 ablation]` | label_flip · feature_noise (→ G6로 frzero·gn·lf@0.70 정렬) | 표엔 없으나 **rundir는 있다** — `{cifar10,mnist}_iid_seed{0,1,2}` (`rates` 전부 0.0 = 완전 무오염) | **0** | — | **표에 행만 추가**(파생) |
| **2-LLM R4 downstream · renorm 4종의 clean 열** `[본문 downstream]` | noisy · frzero | ~~문서 불일치~~ → **07-25 해소**: 수록 arm이 {vanilla, oracle, random, Flirds, Flirds-1st}로 축소되며 renorm 4종·loss-heur·FedIF가 표 자체에서 빠짐 — clean 열 대상도 vanilla·Flirds·Flirds-1st뿐 | **0**(컷 확정) | — | **해소** |

> **그 외 수록 무대는 clean 대조가 있다**: 1A-CNN(c2fid clean 3-seed) · 1A-LLM silo5 (b)-leg · 1B-LLM silo5 (a)-leg · 2-CNN cifar10/dir1. 결손 목록도 clean을 명시적으로 포함한다 — G1 **+clean 3** · G2 **+clean 6** · G3의 84에 clean 포함 · G8의 4위협에 clean 포함 · G9 **+clean 6** · G10의 4위협에 clean 포함.
> **iid(저-이질성) 축은 결손 없다**: CNN은 전 무대가 {dir1, iid} 쌍으로 계획돼 있고(G3·G8·G9·G10), LLM은 **본문 주무대 R4(gsm50k5)가 N=50 IID**이고 부록 silo5가 비IID라 IID↔비IID 대조가 수록분만으로 성립한다(제외된 iid5 진단 매트릭스 없이도).

---

## 5. 이번 목록에 없는 기존 실험 (= 표에서 빠짐)

> 전부 rundir·러너 산출은 **존속**한다. 표에서만 뺀다. 설계 세션은 **이 목록을 다시 계획하지 않는다.**

| 축 | 빠지는 실험 | 현 위치였던 곳 |
|---|---|---|
| Fidelity | c2fid **qskew · shard** 파티션 | 본문 주무대 표의 일부 |
| Fidelity | c2fid **fmnist {dir1, iid}** (부록의 mnist가 대체) | 본문 주무대 표의 일부 |
| Fidelity | 3B silo5 (b)-leg · N=10 완전열거(2¹⁰) | `[보류]` · `[부록·비용]` |
| Fidelity | **신호 실재성: 오염×비IID 매트릭스**(iid5) | `[제외·진단]` — 단 §1C 서술 근거로는 인용 가능(**Q6**) |
| Downstream | **개입 정책 축 P1~P5 종합**(P3·P4·P5h·P5s) — P1·P1w만 남음 | `[본문·주무대 보조]` |
| Downstream | strmain 경쟁 · **clean 오발화 분해** · 완전참여 100/100 · 동적 재추첨 · std50k5 mixed | `[후보]`·`[근거]`·`[부록E]` |
| Downstream | fmnist·iid 8점수원 경쟁(seed0 파일럿) · 확장 파티션 flirds 단독 표 | `[후보]` |
| Downstream | silo5/iid5 φ-게이팅 회수 표 | 이미 `[제외 표 / R4로 흡수]` |
| Downstream | 무해성에서 **ShapleyFL 가중·FedIF 가중** 행 | 본문 근거 표 |
| Detection | 전용 탐지기 4종 **전량** · silo5 탐지 표 · iid5 탐지 · frdelta 한계 · 3B 탐지 · 첫 clean-run 탐지 | §3 대부분 |
| Cost | 지수-비용 스케일링 표 · 학습↔가치평가 위상분리 계측 | `[부록E]`·`[본문/부록]` |
| Cost | Fed-LOO · Banzhaf 행 | 전 runtime 표 |
| Ablation | Taylor 물리잔차(P3) · AdamW 브리지 · TF32 A/B · β 통일 provenance | `[보류·부록A]`·`[제외]`·`[검증-전용]`·`[각주]` |
| Ablation | **Dose-response (CNN·LLM 모두)** — 07-25 Yonghee 제외 결정(Q3) | `[보조]` → `[제외]`. LLM noisy nr 스윕 3-seed 실측은 존치 |
| 위협축 | frrand · lf@{0.15,0.35} · lf-strmain · frdelta · mixed · poison · LLM gnoise | 전 축 |

> ⚠ **clean 오발화를 빼면 "Flirds가 clean parity를 P1에서 −0.74pt 위반"의 인과 설명(발화율 0.39~0.43)이 사라진다.** 수치 자체는 개입 주 표의 clean 열에 남으므로, 논문에서도 **인과절만 지우고 수치는 남기는** 편이 정직 보고를 지킨다(§6).

---

## 6. 논문 본문(`paper/paper-ko.md`) 삭제·수정 대상

> 제외 결정(§0.2·§5)을 논문 쪽에 반영할 때 손대야 하는 곳. `paper-ko.md`(1,099행) 전수 검색 결과이며, **이 문서는 목록만 제시하고 논문 파일은 수정하지 않았다**(해당 파일에 미커밋 변경 존재).

| 제외 항목 | 논문 내 위치 | 할 일 |
|---|---|---|
| **clean 오발화** | **L578–579** — §5.3 읽기 ④ "**clean 오발화의 정직 보고**: online에서 Flirds −0.7pt·individual utility −1.3pt(누적 부호의 0-교차 노이즈), Flirds-1st·FedIF는 무발화" | **인과절만 삭제.** −0.7pt 수치는 개입 주 표 clean 열에서 나오고 그 표는 남는다 — 잃는 건 "왜 발화했나"(0-교차 노이즈)의 설명뿐이라, 수치는 남기는 편이 정직 보고를 지킨다 |
| **Fed-LOO** | **L383** 제외 사유 문단에 Banzhaf·Ripple만 명시 | 같은 문단에 Fed-LOO 추가 |
| **전용 탐지기 4종** | **없음** — `탐지기`/`detector`/`FLDetector`/`FLTrust`/`STD-DAGMM`/`FedDQC` 검색 0건. 부록 F(탐지) 표 F.1도 in-run GT · Flirds · Flirds-1st · individual utility 4행뿐 | **할 일 없음** |
| **신호 실재성 매트릭스(iid5)** | **없음** — 부록 D.2 표(L1026–1034) 행은 `Anchor` / `Silo clean` / `Silo answer-swap` / `Silo free-rider(zero)` / `CNN-Grid` / `LLM-Main ⬚` / `CNN-Main ⬚`. 본문 L459·L465가 인용하는 +0.87 / +0.93 / −0.37은 전부 **silo5·anchor5**(수록 무대) 값 | **할 일 없음.** IID↔비IID 대조는 `Anchor −0.367` vs `Silo clean +0.867`로 자립 |

> 부록 D.2의 `LLM-Main ⬚` · `CNN-Main ⬚` 두 빈칸은 각각 **G1(R4-L2)** 과 c2fid rundir 파생으로 채워진다 → §3.1의 1C 항목과 같은 작업이다.

## 7. 실행 규약 (설계 세션 참고)

### 7.1 무대별 러너·환경변수

| 무대 | 러너 | 핵심 env |
|---|---|---|
| CNN 부분참여 fidelity/detection (1A-CNN) | `codes/experiments/track_c2_fid.py` | `C2_DATASET`(cifar10\|fmnist) · `C2_PARTITION`(iid\|dir1\|shard\|qskew) · `C2_THREAT`(clean\|label_flip\|free_rider\|frrand\|grad_noise) · `C2_FLIP_RATE` · `C2_SEED` |
| CNN 부분참여 downstream (2-CNN) | `codes/experiments/track_c2.py` (+ track_h arm env) | 위와 동일 + arm 지정 |
| CNN vs (a) (1B-CNN) | `codes/experiments/track_c1.py` | `C1_DATASET`(mnist\|cifar10) · `C1_SCENARIO`(iid\|label_skew\|quantity_skew\|label_flip\|feature_noise) · `C1_SEED` · `C1_MODE` |
| LLM 매트릭스 fidelity/detection | `codes/experiments/phase2_matrix.py` | `REGIME`(silo5\|iid5\|device100\|gsm50k5\|gsm5) · `THREAT` · `SEED` · `NOISY_RATE` · `ORACLE_B` · `COALITION` |
| LLM 개입 (2-LLM) | `codes/experiments/track_g.py` → track_h rundirs_llm | `REGIME` · `THREAT` · arm 지정 |
| LLM 표준무대 (std20·anchor5) | `codes/experiments/track_d.py` | 스케일·레짐 |

> **G1 실행 예**(코드 변경 없음): `REGIME=gsm50k5 THREAT=noisy SEED=0 python -u experiments/phase2_matrix.py` — `NOISY_RATE`는 gsm50k5에서 **0.7이 기본값**이라 생략 가능.

### 7.2 track_h CNN rundir ↔ arm 대응 (런 수 계산의 근거)

| rundir 접미사 | 낳는 arm | 정책 |
|---|---|---|
| `_<source>_seed<N>` | `<src>_gate_v2` · `<src>_gatew_v2` · `<src>_mult` · `<src>_zgate_v2` | **P1 · P1w** · P3 · P4 (온라인) |
| `_<source>p5_seed<N>` | `<src>_cgate` · `<src>_pweight` | P5h · P5s (온라인) |
| `_obs_seed<N>` (dir1) | `observer`(=vanilla) + **전 8점수원의** `t2_sign_*` · `t2_signw_*` + `t2_random_k*` | **retrain P1 · P1w** |
| `_obsf_seed<N>` (비-dir1) | `observer` + **flirds만** `t2_sign_flirds` · `t2_signw_flirds` | 〃 (축소판) |

> ⇒ **P1과 P1w는 같은 rundir에서 동시에 나온다** → 부록의 P1w 항목은 **추가 런 0**.
> ⇒ **cifar10/iid의 retrain 결손은 `obsf`가 flirds만 담고 있어서** 생긴다. 채우려면 dir1의 `obs`처럼 8점수원 T2 arm을 담은 관측자 런이 필요(**12 rundir · 각 14 재학습 추가**; dir1 `obs`는 `t2_sign_*`×8 + `t2_signw_*`×8 + `t2_random_k*`×3을 담고 있고 `obsf`엔 flirds 2개만 있음).
> ⇒ flirds 온라인 arm은 `track_g/rundirs_cnn`(144셀)에서 오고, track_h 집계기가 두 경로를 함께 읽는다.

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
