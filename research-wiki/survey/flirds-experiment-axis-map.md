---
type: survey
title: "Flirds 실험 축-지도 — 논문 수록 선택용 (개념/축 분류)"
created: 2026-07-25
updated: 2026-07-25
tags: [flirds, experiments, axis-map, paper-selection, fidelity, downstream, detection, ablation, cost]
---

# Flirds 실험 축-지도 (개념/축 분류)

> **무엇**: 지금까지 **설계·수행한 모든 실험**을 run-code(track_d·c2fid·phase2…)가 아니라 **개념/축**으로 재편성한 **"논문 수록 선택용 메뉴판"**. 실험 이름은 코드명이 아니라 **자연어**로 붙였다.
> **왜**: 어떤 실험을 논문에 넣을지 고르려면, 수록 확정분뿐 아니라 **제외·보류·미실행까지 한 자리에서** 축별로 비교할 수 있어야 한다. 이 페이지가 그 전체 지형도다.
> **수치는 여기 없다** — 이 페이지는 *구조·상태·추천*만 담는다. 실제 수치는 축별 결과 페이지가 담당한다:
> - [[flirds-results-fidelity]] · [[flirds-results-downstream]] · [[flirds-results-detection]] · [[flirds-results-ablation]] · [[flirds-results-cost]]
> - 이 축-지도 + 5개 결과 페이지가 구 overview 2종(수록분 미러·전량 카탈로그)을 **대체함**(구 문서·caveats·종합판정은 git 이력).
>
> 각 행 끝의 `출처`는 추적·재생성용으로 **작게** 병기했을 뿐, 분류·이름은 모두 개념/자연어 기준이다.

## 범례

**데이터 상태**: ● 3-seed 실측 · ◐ 부분/1-seed(정본 아님) · ⬚ 설계했으나 미실행 · ⟐ 파생/재분석(재실행 0) · – 해당없음

> **실측 상태 감사 (2026-07-25)**: 아래 전 행의 마커를 **실제 rundir와 1:1 대조**했다(`phi.parquet` seed 열 / `metrics.json` 키 / rundir 존재 여부). **결과: 마커 오류 0** — ⬚ 2건(R4-L2 gsm50k5 (b), silo5 (a)-leg)은 rundir 부재 확인, anchor5의 "1B만 (a)"는 `(a)oracle` 키가 1B 3-seed에만 존재함을 확인, N=10은 seed0 단독 확인, C1·c2fid·device100·silo5·iid5·std20 전부 3-seed 확인. 같은 감사에서 **누락 2건을 발견해 아래에 반영**했다: ① 1A-LLM에 3B silo5 (b)-leg 행 추가(◐ 1-seed) ② 2-CNN 확장 파티션은 "⬚ 미실행"이 아니라 **flirds 단독 3-seed 실측 ◐** 였음(누락된 건 비-flirds 점수원 7종).
**현 논문 위치**: `[본문]` 수록 확정 · `[부록]` 부록 배치 · `[후보]` 미정(넣을 수도) · `[보류]` 일시 보류 · `[제외]` 현재 뺌(사유 병기) · `[기록/전제/근거/각주]` 본문을 뒷받침하나 독립 표는 아님

**상위 5축 ↔ 프로젝트 기존 E1–E7 분류**(자매 문서 `prior-work-taxonomy/validation-experiments.md`와 1:1): Fidelity=E1 · Downstream=E2 · Detection=E3 · **비용·규모=E6** · Ablation ⊇ {E4 fairness(미설계)·E5 stability·E7 aggregation} + 방법-내부 검증. — 상위 4축은 Yonghee 지정, 비용은 독립 5번째 축으로 승격(2026-07-25).

## 한눈에 — 축별 주무대

| 축 | CNN 주무대 | LLM 주무대 |
|---|---|---|
| **1. Fidelity** | 부분참여 충실도 ● | 정확도-무대 충실도 ⬚ |
| **2. Downstream** | 점수원 경쟁 ● | 정확도 개입(GSM8K) ● |
| **3. Detection** | 부분참여 φ-AUROC ●(=fidelity와 같은 rundir) | 주무대 탐지 ⬚ |
| **4. Ablation** | 2차항·lever·removal (CNN 레그) ● | 〃 (LLM 레그) ● |
| **5. 비용·규모** | op-count·runtime ● | 지수-비용 스케일링 ● |

> ⚠ LLM 주무대(정확도 fidelity·탐지)는 같은 캠페인(R4 GSM8K)의 **exact (b) 오라클 부착분(L2)이 아직 미실행**이라 ⬚. downstream(개입)만 착지. CNN은 fidelity·탐지가 **한 rundir(부분참여 캠페인)에서 동시 산출**.

---

## 1. Fidelity — 기여도 측정이 정답 oracle 순위를 얼마나 재현하나 (1차 핵심)

> 비교 대상(oracle)에 따라 둘로 나눈다 — **in-run (b)**: 한 학습 궤적의 exact 분해(같은-게임 정답; 본문 채점은 same-game **Flirds·Flirds-1st**만) · **retrain (a)**: 조합마다 재학습한 exact Shapley(방법-중립 참값 → 전 방법 채점). 듀얼오라클 실험은 (a)-leg와 (b)-leg가 양쪽으로 나뉜다.

### 1A. in-run (b) 오라클 대비

#### 1A-CNN

| 실험 | 무엇을 보여주나 · 세팅 | 데이터 | 위치 | 출처 |
|---|---|---|---|---|
| **대규모 교차-디바이스 부분참여 충실도** | 부분참여 N=100(10/100)에서 (b) per-round 대비 순위·값·거리. cifar10 4파티션(iid·dir1·shard·qskew)+fmnist 2파티션 × 8위협. **탐지와 동일 rundir** | ● 3-seed(141/144; 소수 ◐2) | `[본문·주무대]` (§5.2) | `runs/track_c/c2fid` |

#### 1A-LLM

| 실험                           | 무엇을 보여주나 · 세팅                                                                           | 데이터      | 위치                        | 출처                                             |
| ---------------------------- | --------------------------------------------------------------------------------------- | -------- | ------------------------- | ---------------------------------------------- |
| **주무대 정확도-무대 충실도**           | R4 GSM8K(N=50, 5/50)에 exact (b) per-round 부착. **R4 개입·탐지의 유일 fidelity 대조축**             | ⬚ 미실행    | `[본문·주무대]` (§5.2 R4-L2)   | `runs/phase2_matrix`(gsm50k5)                  |
| **교차-사일로 도메인 충실도 ((b)-leg)** | N=5 5도메인(의료·법률·금융·수리·일반) 비IID에서 (b) 2⁵ 대비. (a)-leg는 §1B                                 | ● 3-seed | `[본문·보조]`                 | `runs/phase2_matrix`(1B_silo5)                 |
| **교차-사일로 스케일 레그 (3B)**       | 위 무대에 3B. near-additive 포화 재확인 + poison서 1B(0.60)보다 더 붕괴(0.00)                              | ◐ 1-seed | `[보류]`                    | `runs/phase2_matrix`(3B_silo5)                 |
| **표준 부분참여 충실도 (1B·3B·7B)**   | N=20(2/round) alpaca, (b) per-round 대비. "스케일 무관 ρ≥0.999" 근거였음                           | ● 3-seed | `[제외]` (07-23 삭제; 되살림 후보) | `runs/track_d`(std20)                          |
| **대규모 교차-디바이스 앵커 충실도**       | N=100 α=0.5 anchor 셀에 exact (b) per-round 부착                                            | ● 3-seed | `[부록]`                    | `runs/phase2_matrix`(device100 anchor)         |
| **N=10 완전열거(2¹⁰) 충실도**       | N=10 전원 exact 2¹⁰. 고-power 확장 + 지수-비용 실측                                                | ◐ 1-seed | `[부록·비용]`                 | `runs/track_d/rundirs_e5_n10`                  |
| **신호 실재성: 오염×비IID 매트릭스**     | iid vs 비IID(도메인)×오염 2×2. (b) cross-seed 신호가 *어디서* 생기나(비IID clean ρ 0.87 vs IID 0.13) 진단 | ● 3-seed | `[제외]` (진단용)              | `runs/matrix_cxni`→`phase2_matrix`(iid5/silo5) |

### 1B. retrain (a) 오라클 대비 — 방법-중립 참값(전 방법 채점)

#### 1B-CNN

| 실험 | 무엇을 보여주나 · 세팅 | 데이터 | 위치 | 출처 |
|---|---|---|---|---|
| **소형 교차-사일로 시나리오별 vs (a)** | 전원참여 N=10에서 재학습(a) 2¹⁰ 대비 순위·값. mnist·cifar10 × 시나리오(feature-noise/label-flip/quantity-skew/iid). (b)-leg·cross-seed 안정성도 같은 셀서 산출 | ● 3-seed | `[본문·보조]` (§5.2 vs (a)) | `runs/track_c/c1`+`c1_oracle` |

#### 1B-LLM

| 실험 | 무엇을 보여주나 · 세팅 | 데이터 | 위치 | 출처 |
|---|---|---|---|---|
| **교차-사일로 도메인 (a)-leg** | silo5(비IID) 재학습(a) 오라클. 실재 cross-seed 신호를 갖는 **유일 (a)-무대** | ⬚ 미실행 | `[본문·보조]` (§5.2 sub) | silo5 (a)-leg |
| **소형 앵커 듀얼오라클 vs (a)** | N=5 전원, (a) 2⁵ retrain 대비 전 방법. (b)-leg도 있으나 near-additive라 무정보 → 폴백 참조 | ● 3-seed(1B; 3B/7B (a)⬚) | `[보류·폴백]` | `runs/track_d`(anchor5) |

### 1C. 재현성·안정성 (fidelity의 짝 — 순위의 seed 재현성)

| 실험 | 무엇을 보여주나 · 세팅 | 데이터 | 위치 | 출처 |
|---|---|---|---|---|
| **(CNN) 방법 cross-seed 안정성** | C1에서 φ 순위의 seed 재현성. Flirds ≈ (b) 내재 안정성, recon-MC 계열은 하락 | ⟐ 파생(3-seed) | `[부록D]` | `runs/track_c/c1` |
| **(LLM) (b) 타깃 자기-안정성** | 매칭 대상 (b) 자체가 seed 재현되나. IID-clean 불안정(−0.37) vs 비IID 안정(+0.87~0.93) — 헤드라인 +1.000의 해석 조건 | ⟐ 파생 | `[보류·부록D]` | 파생: `track_d`+`phase2_matrix` target_stability |

---

## 2. Downstream — 측정한 φ로 클라를 선택/가중해 학습하면 성능이 오르나 (2차 ①)

### 2-CNN

| 실험 | 무엇을 보여주나 · 세팅 | 데이터 | 위치 | 출처 |
|---|---|---|---|---|
| **점수원 경쟁 — 개입 정확도** | 같은 개입 정책에서 **8점수원 중 어느 φ 정의가 학습을 잘 만드나**. {온라인 배포게이팅, 재학습 부호게이트}×위협, cifar10/dir1, 절대 acc. vanilla(바닥)·oracle_excl(천장)·random_excl 앵커 | ● 3-seed | `[본문·주무대]` (§5.3) | `runs/track_h/rundirs_cnn` |
| **확장 파티션·데이터셋 개입** | 위 경쟁을 cifar10{iid,qskew,shard}·fmnist{dir1,iid}로. **정정(07-25 감사)**: flirds 단독 점수원 × P1~P4 online은 **3-seed 실측 존재**(이전 "⬚ 미실행" 표기는 오류). 미실행은 비-flirds 7종(W-D)·비-dir1 retrain | ◐ 3-seed·flirds only | `[후보]` | `runs/track_h`(p1w_cnn) |
| **개입 정책 축 (P1~P5)** | 같은 무대서 7정책(sign/sign-weight/soft-mult/z-gate/신뢰 hard·soft) × 8점수원 종합 경쟁점수 + clean parity | ● 3-seed | `[본문·주무대 보조]` | `runs/track_h`(competition_score) |
| **강한-주류 label-flip(strmain) 경쟁** | lf-strmain 3셀서 8점수원 × P1~P4. renorm도 살아남는 유일 오염 | ● 3-seed | `[후보]` | `runs/track_h`(stage_cell=strmain) |
| **clean 오발화(부호 게이트 발화율)** | 관찰자 런서 점수원별 clean 오발화·오염 발화율 — parity 위반의 원인 분해 | ● 3-seed | `[근거]` | `runs/track_h`(observer_zero_semantics) |
| **완전참여·동적재추첨·신뢰게이트 확증** | Scale 100/100 완전참여(비용선형)·Dyn 매라운드 오염 재추첨(신호파괴 한계)·P5 신뢰기반 sign 정책 | ● 3-seed | `[부록E]` | `runs/track_h/{scale,dyn,p5}` |
| **φ 부호-게이팅 그리드** | sign/z/V2w/V3 게이트를 skew 레짐서 완주(144셀). 점수원 경쟁과 내용 중복이라 **표 미게재** | ● 3-seed(완주) | `[제외]` (경쟁과 중복) | `runs/track_g/rundirs_cnn` |

### 2-LLM

| 실험 | 무엇을 보여주나 · 세팅 | 데이터 | 위치 | 출처 |
|---|---|---|---|---|
| **표준 개입 무해성 (clean do-no-harm)** | clean-IID에서 φ-가중/선택이 성능을 안 깎나(MMLU·ROUGE parity) | ● 3-seed | `[본문·근거]` | `runs/track_d`(arms) |
| **온라인 φ-게이팅 + 재학습 회수** | silo5·iid5서 부호-게이팅 배제 recovery(frzero 1.000·clean 무발화·**noisy는 (b)로 채점해도 recall 0 = 게이트 작동영역 밖**). **독립 표는 제외**, 결과는 R4 online leg로 흡수 | ● 3-seed | `[제외 표 / R4로 흡수]` | `runs/track_g/rundirs`(LLM)→`track_h`(llm_competition) |
| **대규모 부분참여 혼합오염 게이팅** | std50k5(N=50, 5/50) mixed 오염서 Flirds P1 recovery 1.203(천장 초과)·오배제 ShapleyFL의 절반 | ◐ arm별 seed 1~3 | `[후보]` (정본 아님) | `runs/track_h`(regime=std50k5) |
| **주무대 정확도 개입 (GSM8K EM)** | R4에서 **재학습 부호게이트 + 온라인 배포게이팅**의 절대 EM 회수. 순위정보의 가치(vs 무작위·vs 1차) 실측 | ● noisy·frzero 3-seed(retrain+online) / clean ◐ / frrand·strmain ⬚ | `[본문·주무대]` (§5.3) | `runs/track_h/rundirs_llm`(gsm50k5) |

---

## 3. Detection — φ(또는 전용 탐지기)로 오염 클라를 분리하는 AUROC (2차 ③ · 마지막)

### 3-CNN

| 실험 | 무엇을 보여주나 · 세팅 | 데이터 | 위치 | 출처 |
|---|---|---|---|---|
| **부분참여 φ-AUROC (파티션 강건성)** | c2fid rundir로 φ-AUROC. dir1/iid/shard/qskew × 위협, (b)-동행. **fidelity(1A-CNN)와 같은 rundir** | ● 3-seed | `[본문·주무대]` (§5.4) | `runs/track_c/c2fid` |
| **교차-사일로 φ-AUROC** | C1 셀의 오염 클라 탐지 AUROC (arm-수준 주석) | ● 3-seed | `[주석]` | `runs/track_c/c1` |

### 3-LLM

| 실험 | 무엇을 보여주나 · 세팅 | 데이터 | 위치 | 출처 |
|---|---|---|---|---|
| **교차-사일로 탐지 + 전용탐지기 4종** | silo5 noisy/frzero/frrand서 φ-AUROC + FLDetector·FLTrust·STD-DAGMM·FedDQC | ● 3-seed | `[제외 표]` (§3.3.1; R4가 탐지 본무대) | `runs/phase2_matrix`(1B_silo5) |
| **교차-디바이스 α-sweep 탐지 (규모)** | N=100 Dirichlet α∈{0.0,0.01,0.1,0.5,5.0} × φ·전용탐지기 AUROC·runtime. **α=0.5 앵커엔 exact (b)가 있어 H-13 오라클-동행을 N=100서 직접 판정**(noisy Δ=0.000) | ● 3-seed | `[부록E]` | `runs/phase2_matrix`(device100) |
| **IID 소형 무대 탐지 (매트릭스 탐지 레그)** | iid5 3위협 φ-AUROC + 전용탐지기 4종. silo5와 짝 — φ는 양쪽 다 1.000이고 전용탐지기만 갈림(FLDetector IID noisy 0.000) | ● 3-seed | `[제외·진단]` | `runs/phase2_matrix`(1B_iid5) |
| **delta-재활용 free-rider 한계** | frdelta서 (b)와 **동일하게 실패**(0.33) = "기여도≠탐지"의 정직한 한계(게임 공통) | ● 3-seed | `[제외]` (§6 한계 1문장 후보) | `runs/phase2_matrix`(frdelta) |
| **3B 탐지** | 3B silo5 탐지(스케일) | ◐ 1-seed | `[보류]` | `runs/phase2_matrix`(3B_silo5) |
| **주무대 탐지 (R4 φ + 4탐지기)** | gsm50k5서 φ-AUROC + 전용 4종(§2 약속 이행) | ⬚ 미실행 | `[본문·주무대]` (§5.4 L2) | `runs/phase2_matrix`(gsm50k5) |
| **첫 clean-run 탐지 (foundational)** | 첫 clean run의 noisy/free-rider AUROC + selection. lr 의존 반전 | ● 3-seed | `[기록·부록]` | `runs/phase1` |

---

## 4. Ablation — 구성요소·lever·프로토콜 검증

> Yonghee 결정(2026-07-25): **쪼갤 수 있는 건 쪼개서** 보여준다. 같은 주장이라도 CNN 레그/LLM 레그를 분리 등재하고, 진짜 모델-무관인 것만 `공통`에 둔다. (2차항·lever·removal은 각 CNN·LLM 짝이 있음.)

### 4-CNN

| 실험 | 무엇을 보여주나 · 세팅 | 데이터 | 위치 | 출처 |
|---|---|---|---|---|
| **2차항(HVP)의 기여 — CNN 레그** | 부분참여 k-sweep(Flirds 0.891 vs 1차만 0.305)·grad-noise 개입서 1차계열 실명 vs Flirds만 회복. *↔ LLM 짝* | ● 3-seed | `[본문]` (§5.6①) | `probe_signal/cnn_c1` + `track_h`·`c2fid` GN |
| **A축 용량 lever probe — CNN** | 폭×참여 스윕: lever가 cross-seed 신호를 못 만들고 fidelity는 전반 1.000(Taylor tradeoff 없음). *↔ LLM 짝* | ● 3-seed | `[본문]` (§5.6②) | `runs/probe_signal/cnn_c1`·`cnn_c2` |
| **Removal-curve — CNN** | worst-first 제거가 acc 분리(+0.045=(b)동급)·저순위 방법 ≈0 = 순위→성능 인과. *↔ LLM 짝* | ● 3-seed | `[본문]` (§5.6③) | `runs/removal_dose/rundirs_cnn` |
| **정밀도(TF32) A/B** | cuDNN conv TF32 on/off가 CNN 결론(final_acc·φ 순위)을 안 바꿈 실측 | ● seed0 | `[검증-전용]` | `runs/measured_2026-07/tf32_ab` |

### 4-LLM

| 실험 | 무엇을 보여주나 · 세팅 | 데이터 | 위치 | 출처 |
|---|---|---|---|---|
| **2차항(HVP)의 기여 — LLM 레그** | std50k5 부분참여(5/50)서 Flirds +1.000 vs 1차계열(ComFedSV/ShapleyFL/FedIF) 음수 붕괴. *↔ CNN 짝* | ● 3-seed(경량) | `[본문]` (§5.6①) | `runs/probe_signal`(std50k5) |
| **A축 용량 lever probe — LLM** | rank·lr·steps·참여·noise lever가 cross-seed 신호를 못 만듦(lr는 공통 shift만). fidelity lever 전반 1.000. *↔ CNN 짝* | ● 핵심축 3-seed(나머지 seed0) | `[본문]` (§5.6②) | `runs/probe_signal/rundirs`+`noise_probe` |
| **Removal-curve — LLM** | silo5서 worst-first 제거가 val-loss↓·best-first↑ = 게임-무관 인과 검증. *↔ CNN 짝* | ● 3-seed | `[본문]` (§5.6③) | `runs/removal_dose/rundirs` |
| **Dose-response** | φ 탐지 문턱 vs 오염강도(silo5 noisy nr·frrand dm 스윕) | ● 3-seed | `[보조]` | `runs/removal_dose/rundirs`(B) |
| **AdamW 브리지 — external validity** | SGD→AdamW optimizer 갭서 fidelity(+0.77; (a)↔(b) 자체 괴리 caveat) | ● 3-seed | `[제외]` | `runs/removal_dose/rundirs_trackd`(A1/D) |
| **Taylor 물리잔차 (명제 P3)** | 2차 근사가 1차보다 물리잔차 ~3× 작음(1B 실측). 2차항 추가의 물리적 정당화 | ● 3-seed | `[보류·부록A]` | `runs/measured_2026-07/taylor` |

### 4-공통 (모델-무관)

| 실험 | 무엇을 보여주나 · 세팅 | 데이터 | 위치 | 출처 |
|---|---|---|---|---|
| **φ 부호 감사 (게이팅 전제)** | 309 rundir 전수 φ 부호 감사. clean 오배제-0·frzero exact-0 확정, 게이팅 예측표 확정/수정 | ⟐ 파생 | `[전제]` | `runs/track_g/audit` |
| **β 통일 재실행 provenance** | ShapleyFL EMA β 0.5→0.3 통일 재실행·대조. **현재 폐기**(수록 대상 전부 제외됨) | ⟐ 파생/폐기 | `[각주]` | `runs/rerun_beta03` |

---

## 5. 비용·규모 — 방법별 wall-clock과 스케일링 (E6, 독립 5번째 축)

### 5-공통 (모델·하드웨어·정밀도 독립)

| 실험 | 무엇을 보여주나 · 세팅 | 데이터 | 위치 | 출처 |
|---|---|---|---|---|
| **연산수(op-count) 모델** | 지배연산(fwd/grad/HVP) per-round 해석적 카운트 = 하드웨어·정밀도 독립 비용축. 측정 wall-clock 재현 | ● | `[본문]` (§5.5) | `runs/measured_2026-07/op_counts.py` |
| **microbench (fp32/bf16 배율)** | per-op fp32 forward 1.60s·HVP 10.36s(비율 6.47) + fp32/bf16 배율 = op-count→시간 환산 입력 | ● | `[본문]` (§5.5) | `runs/measured_2026-07/microbench` |

### 5-LLM

| 실험 | 무엇을 보여주나 · 세팅 | 데이터 | 위치 | 출처 |
|---|---|---|---|---|
| **표준무대 실측 runtime** | 방법별 wall-clock(초). 소-cohort(std20) 역전 구조 포함 | ● 3-seed | `[본문]` (§5.5) | `runs/track_d`(+`rundirs_e4_fedloo`) |
| **지수-비용 스케일링 (5~160×)** | N=10 2¹⁰ = 1/160 · device100 = 1/159. cohort 큰 무대서 Flirds 압승 실측 | ●/◐ | `[부록E]` | `runs/track_d/rundirs_e5_n10` + `phase2_matrix`(device100) |
| **학습↔가치평가 위상분리 계측** | client-training vs valuation wall-clock·peak-mem 분리(timing.json·gpu-hours) | ● | `[본문/부록]` | `runs/measured_2026-07/timing_device100` |

### 5-CNN

| 실험 | 무엇을 보여주나 · 세팅 | 데이터 | 위치 | 출처 |
|---|---|---|---|---|
| **CNN runtime 요약** | Track C 방법별 runtime(학습 자체보다 2~3자릿수 저렴) | ● | `[부록]` | `runs/track_c`(RESULTS) |
| **자기-궤적 재실행 비용 (제외 baseline)** | 제외 비교군의 자기-궤적 재실행 비용을 별도 위상으로 분리 실측 | ● seed0 | `[검증-전용]` | `runs/measured_2026-07/e3_cost_smoke` |

---

## 참고 — 제외된 비교군·위협축 (데이터는 rundir 존속)

> 실험 "축"은 아니지만, 골라내는 과정에서 되살릴지 판단할 수 있게 함께 남긴다. 전부 rundir·러너 산출은 살아 있고 표에서만 뺐다.

- **비교군 제외**: Banzhaf(2026-07-22) · Ripple(2026-07-19; 감사 근거 [[ripple-audit-2026-07/ripple-baseline-exclusion]]) · Fed-LOO(2026-07-23). — 러너는 계속 열을 생성하므로 집계·표 단계에서만 제외.
- **위협축 제외**: poison(clean-preserving backdoor, 2026-07-22) — (b) oracle은 AUROC 1.0으로 잡으나 1차 Taylor 추정 쪽 실패라 스코프 밖.
- **비교군 9종(수록)** = same-game(Flirds·Flirds-1st·loss-heur) + cross-game(GTG·FedSV·ComFedSV·ShapleyFL β=0.3·FedIF). 전용 탐지기 4종은 §3(Detection).

---

## 현재 논문 스파인 (수록 확정 `[본문]` 만 추림 — 선택의 출발점)

| 축 | CNN | LLM |
|---|---|---|
| **Fidelity** | 부분참여 충실도(in-run) ● · (retrain)C1 vs (a) ● | 정확도-무대 충실도(in-run) ⬚ · (retrain)silo5 (a)-leg ⬚ |
| **Downstream** | 점수원 경쟁 ● | 정확도 개입(GSM8K) ● · (근거)무해성 ● |
| **Detection** | 부분참여 φ-AUROC ● | 주무대 탐지 ⬚ |
| **Ablation** | 2차항·lever·removal (CNN 레그) ● | 2차항·lever·removal (LLM 레그) ● |
| **비용·규모** | op-count·microbench·runtime ● | 지수-스케일링 ●/◐ · 위상분리 ● |

> **선택 관점 힌트**: LLM 쪽 fidelity·detection 주무대(R4-L2)가 **미실행 ⬚**라 현재 본문 LLM 축은 downstream만 실측 완결 — 여기가 최우선 착지 후보. CNN은 fidelity·detection이 한 rundir로 완결돼 있고, downstream도 dir1 완결(fmnist·iid는 `[후보]`). 제외군(std20 fidelity·silo5 탐지 표·Track G 게이팅 표)은 "되살릴 수 있는 예비 카드".

## 상호 링크
- 축별 결과 페이지(수치 정본): [[flirds-results-fidelity]] · [[flirds-results-downstream]] · [[flirds-results-detection]] · [[flirds-results-ablation]] · [[flirds-results-cost]]
- 구 overview 2종(수록분 미러·전량 카탈로그) = git 이력으로 대체됨
- 승패 메커니즘 해석: [[flirds-principle-analysis]]
- 선행연구 E1–E7 분류(CNN/LLM 트랙): [[prior-work-taxonomy/validation-experiments]] · [[prior-work-taxonomy/README]]
- 논문 구조·수록/제외 결정 정본이던 `paper/workplan/00-INDEX.md`는 **2026-07-25 삭제됨**(커밋 `71ad73e`) — 결정 근거가 필요하면 git 이력 참조. 현행 수록/제외 상태는 이 축-지도의 `[본문]/[부록]/[후보]/[보류]/[제외]` 표기가 대신한다.
