---
type: survey
title: "논문 실험 배치안 — 본문 / ablation / appendix"
created: 2026-07-22
updated: 2026-07-22
sources: [flirds-experiment-results-overview]
tags: [survey, paper, placement, experiments]
---

# 논문 실험 배치안 (본문 실험 절 / ablation / appendix)

> **근거 스코프**: `paper/paper-ko.md`의 **§4 이전만** — 초록·§1 서론·§2 관련연구·§3 배경이 세운
> 논증(두 한계·기여 3종·FL 조건 ①–⑤·검증 2겹·관련연구 공백)에 비추어, 재구조화된
> [[flirds-experiment-results-overview]](2026-07-22판, 이하 overview)의 각 실험을 배치한다.
> **paper-ko.md의 §4 이후 현 초안(방법·실험 절)은 배치 근거로 사용하지 않았고, paper-ko.md는
> 무수정**(대규모 미커밋 수정 진행 중). 이 문서는 제안이며 채택 여부는 Yonghee 결정.

## 0. §1–§3이 실험 절에 지우는 의무 (배치의 원칙)

paper §1–§3에서 실험이 반드시 실증해야 하는 주장:

- **L1(한계 1 해소·기여 1)**: "라운드당 필요한 무거운 계산은 HVP 한 번으로 고정, 참여자당 내적 하나" —
  조합 재평가·재학습·추가 통신 없음. → **비용 실측 절 필수**.
- **L2(한계 2 해소·기여 2)**: "같은 목표값을 전수 열거로 직접 계산한 참값(exact in-run Shapley)의
  순위를 1B–7B 전 스케일에서 재현(ρ≥0.999)" — 근사 참조·다운스트림 간접 증거보다 강한 기준. →
  **exact-oracle fidelity 절이 헤드라인**.
- **기여 2(검증 프로토콜)의 실효성 층**: §1이 명시 열거 — "온라인 부호-게이팅 / removal(제거·실제
  재학습으로 순위의 인과적 타당성) / selection / 탐지" + "방법별 계산 비용을 같은 고정 궤적 위에서
  실측". → 네 실효성 실험은 **본문 소속이 §1의 약속**(removal 포함 — overview에선 Ablation §4.4에
  두었지만 paper 관점에선 본문 요약이 필요).
- **기여 3(수학)**: "무기여 클라이언트는 대수적으로 정확히 0 … 라운드 단위 온라인 정산과 결합" →
  frzero exact-0과 그 배포형(부호-게이팅 recovery 1.000)이 이 주장의 실증.
- **§1 명시 유보**: "고전적 exact retrain Shapley와의 실증적 관계는 **특성화 실험으로만** 보고" →
  (a)-oracle 관계는 별도 특성화 절(주장 아님, 관찰).
- **§2 관련연구의 대조점**: ① 계보의 검증이 "소형 CNN 규모 다운스트림 간접 확인"에 머묾(SPACE
  2ⁿ-재학습도 N≤10 CNN) → 우리의 CNN 듀얼 오라클은 그 계보와 같은 무대에서의 직접 채점 + LLM로
  확장. ② 탐지·강건집계는 기여도 평가의 대체재가 아님 — "각자가 설계된 위협에서 비교" → 탐지기
  비교는 탐지 절에 한정, fidelity 표엔 불포함.
- **정직-검증 기조**(§1 "검증을 두 겹으로" + 프로젝트 위계): fidelity 천장 효과("왜 전부 만점인가")에
  대한 선제 방어로 **신호 존재 조건(B축)과 타깃 자기-안정성**을 본문에서 다뤄야 L2 해소 주장이
  버틴다.

**전역 결정 반영(2026-07-22 Yonghee)**:
- Banzhaf·Ripple 비교군 제외(overview §6.2-12) — 논문 전 표에서도 제외.
- 수렴(rounds-to-target) 축 스코프 제외(overview §5.6) — 전용 절 없음.
- **poison(clean-preserving backdoor) 위협축 제외**(overview §6.2-8) — 논문 위협 스코프 = noise류·free-rider류(zero/random/delta)·label-flip류만.
- **LLM selection→성능 실험의 심판 = GSM8K(exact-match)로 확정** — alpaca val-loss·MMLU·ROUGE는 성능 차이 변별력 부족 판단; **클라이언트 소규모 참여 세팅은 이 축에서 진행하지 않음**; std50k5(alpaca 부분참여) selection 계열(Track H Tier 3·R2·track_g 파일럿) = 미사용(overview §3.2 서두 ③).

## 1. 본문 실험 절 구성안 (순서 포함)

| # | 절(가칭) | 내용 | overview 출처 | 배치 근거 (§1–§3) | 진행 중 의존성 |
|---|---|---|---|---|---|
| E1 | 실험 세팅과 측정 프로토콜 | 무대(std20/anchor5/robustness silo5·device100/CNN C1·C2), 듀얼 오라클 (a)/(b) 정의, 비교군·지표, seed 규약 | §3.1.1(a)·§3.1.2(a)·§3.3.1(a)·§3.3.2(a)·§1 범례 | 기여 2가 "검증 프로토콜" 자체를 주장 — 프로토콜을 명시 절로 세워야 주장이 성립 | β0.3 잔여 18+9셀·재현성 P0 재실행(caveat 13)이 절대값 표기·재현성 서술에 영향 |
| E2 | Exact in-run oracle 대비 추정 정확도 (헤드라인) | LLM 1B–7B × std20/anchor5 fidelity 표(순위·값·거리) + 확장: N=10 exact 2¹⁰·Fed-LOO 동률 | §3.1.1(b) 표 3종 · §3.1.3(b1) | 초록 "ρ≥0.999 전 스케일" 직접 실증 = L2 해소; Fed-LOO는 분야표준 앵커 대비(§2 계보 공백) | 없음(완료) |
| E3 | 기여 신호의 존재 조건 + 오염·비IID·부분참여 fidelity | B축 2×2(오염×비IID) cross-seed ρ · (b) self-stability 요지(IID-clean 불안정 vs non-IID 안정) · silo5/device100 오염 무대 Spearman · std50k5 부분참여(uniform-subset 계열 붕괴 vs Flirds 1.000) | §3.1.5(b1) · §5.4(요지) · §3.1.4/§3.3.1–2(Sp 열) · §4.2(b2) | "왜 IID-clean에서 전원 만점인가"의 선제 답(정직-검증 기조) — fidelity 주장의 해석 조건을 규정해야 L2 주장이 리뷰에 버팀; 부분참여는 FL 조건 ③(간헐 참여)에서의 차별화 | silo5 값은 β0.3 재실행판(ce0b454) 정본 — 3B silo5 4셀 재실행(REMAINING §1.2) 후 3B 값 재확인 |
| E4 | Retrain oracle 특성화 + 게임-무관 척도 | (a)vs(b) 0.933(1B anchor5) + 전 방법 vs (a) · CNN 듀얼 오라클(vs (a) 0.35 전 방법 공통 = 두 게임 괴리) · removal-curve 요약(worst-first 인과) | §3.1.1 (a)행·vs(a) 표 · §3.1.2(b1) · §4.4.1–2(요약) | §1 명시 "retrain 관계는 특성화 실험으로만 보고"; removal은 §1이 열거한 실효성 실험이자 게임-무관 공통 자 — 본문 요약+appendix 상세 분할 | P2/P3((a) 3B/7B retrain) ⬚ — 채우면 표 확장; 미완이면 1B 한정 명기 |
| E5 | 비용: 평가 단위와 참여자 수 | op-count 축(하드웨어 독립) + 전 무대 runtime 표 + (b) 지수 비용의 N=10 실측(160×) + "cohort 작으면 (b)가 더 쌈" 정직 보고 | §3.4.1 · §3.4.2 · §3.4.3 · §3.1.3(b2) | L1 해소·기여 1("HVP 1회 고정, 내적 하나")의 실증; FL 조건 ④(재학습·추가연산 불가) | loss-heur 3B/7B 재측정·β deferred 9셀(7B runtime) 대기 — pre-fix 값 caveat 유지 |
| E6 | 기여도의 가치: 부호-게이팅·경쟁·selection·탐지 | ① frzero 온라인 자동배제 recovery 1.000(오배제 0)+clean 무발화 = null-player 공리의 배포 실증 ② 점수원 경쟁(같은 정책·8종): exact-0 계열 생존 vs renorm free-rider 붕괴, zero-semantics 트레이드오프 — **LLM 축 심판은 R4 GSM8K(확정 무대)** ③ CNN C2 오염 회복 ④ top-k selection(phase1) ⑤ 탐지: noisy/FR 1.0 vs 전용 탐지기, frdelta 정직 한계 | §3.2.3–4 · §3.2.6 · §3.2.2 · §3.2.5 · §3.3.1–5(+§3.3.4) | §1 실효성 층의 명시 열거(부호-게이팅·removal·selection·탐지) + 기여 3(정확 0 → 온라인 정산) + §2 "각자가 설계된 위협에서" 탐지기 비교; "다른 정의로 똑같은 실험" 경쟁이 기여도 정의의 실효 우열 증명. §3.2.1의 MMLU/ROUGE parity는 심판 변별력 한계 판단(07-22)에 따라 본문 축이 아니라 appendix 후보(do-no-harm 보조) | **R4 gsm50k5 Tier A 실행 중** — LLM selection 본문 축이 R4 결과에 직접 의존(그 전 LLM 서술은 R3 silo5-noisy 한정; R2 std50k5는 미사용) |

> 순서 논리: 프로토콜(E1) → 1차 주장(E2) → 그 주장의 해석 조건(E3) → 별도 참값과의 관계(E4) →
> 비용(E5) → 실효성(E6). 프로젝트 위계(1차 fidelity → 2차 성능 → 탐지)와 §1 검증 2겹(추정기 층
> E2–E4 / 실효성 층 E6)을 모두 만족하는 순서.

## 2. Ablation 목록 (본문 뒤 또는 실험 절 내 ablation 소절)

| 항목 | 내용 | overview 출처 | 배치 근거 | 의존성 |
|---|---|---|---|---|
| 2차항(HVP)의 기여 | Flirds-1st 대비: 부분참여 붕괴(k=0.2 0.305 vs 0.891)·grad-noise 개입 실명(.244 vs .567~.607)·Taylor 잔차 ~3× | §4.1(종합)·§4.3(b3)·§3.2.6·§5.5 | 방법의 유일한 구조적 선택지(2차항 포함 여부)가 언제 값을 하나 — 기여 1의 구성요소 정당화 | 없음 |
| 게이트 정책 변형(P5 hard/soft) | 신뢰-기반 게이트가 clean 오발화를 회수(flirds retrain 오염-평균 .6207 ≈ 천장); 사전등록 예측 대조 | §4.8.1 | E6-①(sign-게이트)의 정책 강건성 — "τ=0 하나로 되는가"에 대한 답 | R4 P5-leg(REMAINING §1.1-P5) 대기 — CNN만으로 서술 가능 |
| β(EMA) 민감도 | ShapleyFL β0.5↔0.3 전후 대조 = 재실행 노이즈 플로어 수준 | §4.7 | baseline 공정성(자기 논문 β 사용) 방어 | β 잔여 18+9셀은 라벨 통일용(결론 불변 예상) |
| A축 lever(rank·lr·steps·noise·폭·참여) | 어느 lever도 cross-seed 실재 신호를 못 만듦; fidelity는 lever 전반 1.000(Taylor tradeoff 없음) | §4.2·§4.3 | E3(신호 조건)의 반증 축 — "무대를 키우면 되지 않냐"는 반론 차단 | lr·steps intervention 2차 분석(REMAINING §1.4)은 보강용(선택) |
| dose-response | 탐지 문턱: noisy nr0.25↑ 1.0 · FR 전 배율 1.0 | §4.5 | E6-⑤ 탐지 문턱의 정량화(어디부터 잡히나) | 없음 |

## 3. Appendix 목록

| 항목 | 내용 | overview 출처 | 배치 근거 | 의존성 |
|---|---|---|---|---|
| 전체 정확도·거리 표 | std20/anchor5 거리 표, vs (a) 전 방법 표, CNN 시나리오별·Kendall·거리, device100 전 α | §3.1.1·§3.1.2·§3.3.2 | 본문은 대표 표만 — 전량은 appendix(관행) | β deferred 후 7B 열 갱신 |
| removal·dose 상세 | A2 곡선·방법별 일치 카운트, A3 CNN 18셀(acc 분리), dose ladder 전체(noisy·FR) | §4.4·§4.5 | E4·E6의 요약을 뒷받침하는 상세 | 없음 |
| (b) target self-stability 전표 | 전 무대 xseed ρ 표(Exp C) | §5.4 **[overview 보류 항목]** | E3 요지의 정본 표 — appendix 후보로 보류 중(Yonghee 확인) | silo5 행 = β0.3 재실행판 |
| Taylor 물리잔차(P3) | 1차/2차 잔차·3차 무이득·φ 정합 1e-10 | §5.5 **[overview 보류 항목]** | 기여 3(bound)의 물리 실증 — 수학 appendix와 연결(보류 중) | 없음 |
| 신호-크기 진단 확장 | A/B축 종합 판정·noise probe·CNN probe 전표 | §5.3·§4.2(b4)·§4.3 | E3·ablation A축의 배경 진단 | 선택 |
| 탐지 보조 표 | CNN C1 ladder AUROC·C2 arm AUROC·frdelta 상세·3B 1-seed 전표 | §3.3.3–5 | E6-⑤의 보조(본문은 silo5·device100 중심) | 3B 3-seed(P5 계획) 후 승격 가능 |
| 게이팅·경쟁 상세 | Track G 전 arm val-loss·recovery 표, V2w 불승격 판정, Track H 정책×시점 전 표, 예측표(H·HP·HS·DP) 대조 | §3.2.3–4·§3.2.6·§4.8 | E6은 대표 수치만 — 사전등록 예측 대조표는 appendix가 자연스러움 | R4 결과 나오면 LLM 표 추가 |
| Scale 완전참여·Dyn 재추첨 | 100/100 무대(coalition arm 부재 증명 포함)·라운드-오염 null-무대 do-no-harm | §4.8.2–3 | E5(비용: k-선형 주장)와 E6(게이트 한계 정직 보고)의 극한 검증 — appendix | 없음 |
| 프로토콜 상세·재현성 | seed·환경·git 증빙, H1 재현성 정정과 P0/P1 재실행 계획, β provenance | §6.2 caveat 9·13·§4.7 | 기여 2(프로토콜)의 재현성 조항; RERUN 문서와 연동 | **P0 재실행 완료 여부가 "재현 가능" 문구를 결정** |
| 검증-전용(비게재) | LR sweep·TF32 A/B·계측 세부(timing·microbench·acct)·E3 스모크 | §6.3 | 논문 비게재 — 내부 기록만(게재 불필요 판단) | – |

## 4. 미배치·명시 제외

- **수렴(rounds-to-target)**: 스코프 제외(overview §5.6) — 본문·appendix 모두 전용 절 없음.
- **Banzhaf·Ripple**: 비교군 제외(overview §6.2-12) — 전 표 제외; Ripple 제외 근거는
  [[ripple-audit-2026-07/ripple-baseline-exclusion]]을 프로토콜 각주로 인용 가능.
- **poison 위협축**: 논문 비게재(overview §6.2-8) — 관련 실험(silo5/device100/3B poison 셀·
  removal poison ASR·dose pf ladder) 전부 미배치. 데이터는 rundir 존속.
- **std50k5 selection 계열**(Track H Tier 3·R2·track_g std50k5-mixed 파일럿): 미사용(overview
  §3.2 서두 ③) — LLM selection 무대는 R4 gsm8k로 대체. ⚠ std50k5의 *fidelity* 결과(§4.2
  부분참여 probe, E3 소속)는 별개 축이라 유지.
- **alpaca/MMLU/ROUGE 심판의 LLM selection 표**(§3.2.1): 성능 차이 변별력 부족 판단(07-22) —
  본문 축 아님, do-no-harm 보조로 appendix 후보.
- **Fairness·reward(P6)**: 전용 실험 미설계 — §6(논의·한계)에서 향후 과제로만.
- **frdelta의 위치 주의**: "기여도≠탐지"의 게임-공통 사례라 E6-⑤(정직 한계)에 본문 1문단 +
  appendix 상세를 제안 — 숨기면 리뷰 위험, 본문 전면 배치는 위계상 과함.

## 5. 진행 중 실험 의존성 요약 (배치안에 걸리는 것만)

| 실험 | 상태(2026-07-22) | 걸리는 배치 |
|---|---|---|
| R4 gsm50k5 Tier A(seed0) | **서버 실행 중**(07-20 23:29~; rundir 미착지 — 루트 REMAINING §1.1) | E6-② LLM selection 본문 축 전체(확정 무대; 그 전엔 R3 noisy 한정 서술). ⚠ R4 현 스펙은 N=50·5/50 부분참여 — "클라이언트 소규모 참여 세팅 지양"(07-22) 결정과의 정합(스펙 유지/개정)은 Yonghee 확인 필요 |
| β0.3 잔여 18셀 + deferred 9셀 | 큐 대기(REMAINING §1.2–1.3) | E1 절대값 표기·E5 7B runtime·E3 3B silo5·appendix 전표 |
| P2/P3 (a) retrain 3B/7B | ⬚ | E4 (a)-특성화의 스케일 확장(미완이면 1B 한정 명기) |
| P1 (a) 2¹⁰·E5 seeds1-2 | ⬚(장기 대기, REMAINING §1.4) | E2 확장 각주 |
| 재현성 P0/P1 재실행 | 계획 수립(루트 `RERUN_AFTER_REPRO_FIX_2026-07-21.md`) | 전 절의 절대값 재현성 문구(appendix 재현성 조항) |
| lr·steps intervention 분석 | 데이터 있음·분석 대기(REMAINING §1.4) | ablation A축 보강(선택) |

## 6. 매핑 요약 (overview 신 구조 → 논문)

- **본문**: §3.1.1·§3.1.3(→E2) / §3.1.5·§5.4요지·§3.1.4·§4.2(b2)(→E3) / §3.1.1(a)·§3.1.2·§4.4요약(→E4) / §3.4(→E5) / §3.2 전부·§3.3(→E6) / §3.1.1(a)세팅류(→E1)
- **ablation**: §4.1·§4.8.1·§4.7·§4.2–4.3·§4.5
- **appendix**: §4.4–4.5 상세·§5.4·§5.5·§5.3·§3.3.3–5·§3.2.3–4/§3.2.6/§4.8 상세·§4.8.2–3·§6.2(9·13)
- **비게재**: §6.3
