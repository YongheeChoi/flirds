# Paper Workplan — 인덱스 (2026-07-23 확정)

> Yonghee + Claude 설계 세션(07-23)에서 확정한 **논문 실험·작성 계획의 정본**.
> 각 T-문서는 독립 세션에서 수행 가능하도록 자기완결로 작성됨. 완료 시 아래 상태 체크 갱신.
> 이 확정판은 `research-wiki/survey/paper-experiment-placement-plan.md`(구 배치안)를 **대체**한다.

## 0. 확정된 논문 구조 (paper-ko §5 + 부록)

| 절 | 내용 | 수치 상태 |
|---|---|---|
| 5.1 세팅 | 주무대 쌍(LLM R4 gsm50k5 · CNN C2 캠페인 그리드) + sub 무대 · 비교군 9종 · 지표 · 고정-궤적 채점 프로토콜 (상세→부록 B) | 작성 가능 |
| 5.2 Fidelity | **메인**: c2fid + R4-L2, **same-game만**(Flirds·Flirds-1st·loss-heur) vs (b) ⬚ · **sub**: retrain-(a) 특성화 = LLM(gsm5 신설 ⬚ 주 표 / silo5 (a) ⬚ 비IID 보조 / anchor5 기존 0.933 IID 참조) + CNN C1 시나리오별 vs (a)(기존; overview §3.1.2 신규 표) | 쌍·(a)신설 ⬚ |
| 5.3 개입 | **메인**: R4 P1 T1/T2 절대 EM ⬚(L1) + CNN 8점수원 P1 절대 acc(기존 §3.2.3; restack 드리프트 확인 후 기입) · P1w(크기-가중)는 결과 규칙부 수록 | R4 ⬚ |
| 5.4 탐지 | R4 φ + 전용 탐지기 4종 ⬚(L2) + c2fid φ-AUROC ⬚ | ⬚ |
| 5.5 비용 | op-count 모델 + 주무대 실측 runtime ⬚ · 소-cohort 조건부는 op-count 축 서술(std20 실측 삭제 여파) | 부분 ⬚ |
| 5.6 Ablation | ① 2차항(HVP) 기여 ② A축 lever probe ③ removal-curve | 작성 가능 |
| 부록 A | 증명(기존; **A.6·A.11의 Taylor 잔차-실측 참조 삭제**) | 수정만 |
| 부록 B | 프로토콜 상세(하이퍼·위협 정의·분배·환경 1줄·ComFedSV per-round 대용 caveat·β=0.3 각주·LLM에 grad-noise 없는 이유 2문장) | 작성 가능 |
| 부록 C | fidelity 확장(cross-game 비교 vs (b)·(a) 전표 · Kendall/거리 · C1 전표 · std50k5 부분참여) | 작성 가능 |
| 부록 D | stability(C1 방법 안정성 + (b) target 안정성 — 수록 무대 한정) | 작성 가능 |
| 부록 E | 비용·규모 보조(**Scale 100/100 P1** + N=10 2¹⁰ 1/160 + device100 1/159) — 07-23 결정으로 본문→부록 | 작성 가능 |

**전역 결정(07-22~23)**: 본문 fidelity=same-game만(vs (a)는 전 방법 허용 — 중립 참값) · 정책=P1 확정(P2~P5 미수록) · **std20·anchor5-vs(b)·신호실재성(B축)·3B/7B 삭제**(초록 "1B–7B ρ≥0.999" 문구 삭제; 3B/7B는 여유 시 재추가 후보) · **Fed-LOO 논문 전면 제외** · 기존 제외 유지(poison·Banzhaf·Ripple·수렴·std50k5 selection·MMLU/ROUGE·AdamW·frdelta[§6 한계 1문장 후보]·dose·Track G 게이팅 표·silo5 §3.3.1 표·device100 표·예측표 HIT/MISS).

## 1. 신규 실험 (승인 완료)

| 코드 | 내용 | 비용(B200-등가) | 실행처 | 문서 |
|---|---|---|---|---|
| **L7** | R4 P1w(크기-가중) flirds-only {clean,noisy,frzero}×3-seed×{T1,T2} | ~80 | B200(L2 뒤·L4 앞) | T3 |
| **W-B** | CNN 캠페인 그리드 P1w flirds twin leg (T1·T2) + W-A(dir1 기존 P2 재사용) | ~25–35(CNN측) | CNN 서버 | T4 |
| **L8** | retrain-(a) 스위트: gsm5 신설(dual (a)+(b), clean·noisy) + silo5 (a)-leg 3셀 | gsm5 ~60 + silo5 ~26 | **RTX3090×8**(B200 비점유) | T5 |
| W-D | P1w 비-flirds 점수원(CNN 확장무대·R4) | 조건부 | 게이트 | T3/T4 |

P1w 수록 규칙(사전 고정): 전 범위(W-A·W-B·L7)에서 이기면 본문 승격 / 동률 "부호가 가치의 대부분" ablation 1문장 / 열세·타 소스 역전 시 미수록(P1만; rundir 존속).

## 2. 작업 문서와 순서

| # | 문서 | 내용 | 의존 | 상태 |
|---|---|---|---|---|
| T3 | `T3-p1w-llm-impl.md` | R4 P1w 구현+테스트+스모크+큐 등재(H-14 사전등록 포함) | 없음(코드 세션) | ☐ |
| T4 | `T4-p1w-cnn-relay.md` | CNN 서버 세션 전달 스펙(W-A 재사용 판정 + W-B twin leg) | 없음(전달) | ☐ |
| T5 | `T5-retrain-a-suite.md` | gsm5 무대 신설 + (a) 러너 배선 + silo5 (a)-leg + 3090 배치 | 없음(코드→3090) | ☐ |
| T1 | `T1-paper-section5.md` | paper-ko §5·부록 B–E 실작성(기존값 기입 + ⬚ 골격) + 초록/§1/부록A 수정 | 없음(기존값부) | ☐ |
| T2 | `T2-results-overview-page.md` | 논문-순서 결과 overview 위키 페이지 + 시각화(figures) | T1 구조(권장) | ☐ |

권장 순서: **T3·T4·T5 먼저**(GPU가 크리티컬 패스 — 서버/3090 가동) ∥ T1 병행 → T2.

## 3. 공통 규약 (전 세션)

- 수치는 rundir/analysis 재생성 값만(수기 금지). **R4 Tier A seed0(pre-fix `fa5fc6e`) 인용 금지** — L1 3-seed가 정본.
- CNN dir1 기존 수치(§3.2.3·P2)는 **캠페인 restack 드리프트 표 확인 후** 기입/귀속.
- 사전등록(예측 행)은 실행 전 커밋. 결과는 overview(§3.x)에 먼저, paper·T2 페이지는 그로부터.
- push는 Yonghee 직접. 실행 큐 정본 = `REMAINING.md` §1.6 + `runs/track_h/QUEUE_L1L2_2026-07-23.md`.
