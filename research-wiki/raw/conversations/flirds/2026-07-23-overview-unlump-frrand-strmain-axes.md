---
type: conversation
date: 2026-07-23
topic: flirds
participants: [Yonghee, Claude]
tags: [paper, overview, frrand, strmain, threat-symmetry, un-lumping, decision-reversal]
---

# overview 후속 — 뭉친 baseline 개별화 + R4 위협-대칭(frrand·strmain) ⬚ 축 등재

T2 결과 overview(`survey/flirds-paper-results-overview.md`) 검토 중 Yonghee 지적 2건.

## Yonghee 지적/요청

1. **"(메인) CNN 8 점수원 × P1 절대 acc 부분에 왜 각 baseline 값을 다 뭉뚱그려 놓았나 — 하나하나
   개별적으로 써라. 다른 부분에도 이런 식이면 없애라. 그리고 frrand 축은 왜 어디에도 없나?"**
2. (frrand/per-client dose 논의 후) **"이 부분(frrand + per-client dose 설계 공백)을 지금
   paper-overview에 빈 칸으로 넣어라."**
3. **"strmain도 paper-overview에 빈 축으로 추가하고, 두 가지(frrand + strmain) 모두 remaining.md에
   추가로 실험해야 할 내용으로 추가해라."**

Yonghee가 명확히 정정한 프레이밍(초판 내 "CNN이 값싸서 dose 촘촘" = 오류):
- **필연적 배제는 grad-noise 하나뿐**(LLM LoRA 기하에서 등방 노이즈 무반응 — 모달리티 문제).
- **frrand는 LLM에 값싸게 넣을 수 있다**; R4가 free-rider 대표로 frzero만 쓰는 건 계산량/모달리티가
  아니라 **그냥 설계 선택**.
- **"오염 정도를 클라마다 다르게" = 계산량 무관 fidelity 도구**(dose-스프레드로 순위 축퇴 완화).
  CNN fidelity 무대는 이미 사용 — C1 `_pair_ladder`(0/0/5/5/…/20/20%), C2 `strmain`. LLM 무대
  (R4·anchor5·silo5)만 전원 균일-강도 → **설계 공백**.
- **strmain 정의**: FedCorr 기본 label-flip, 오염 클라마다 flip rate ~ U(0.5,1)(클라별 강도 랜덤).
  고정-dose 셀 {.15,.35,.70}(전원 한 rate)의 반대. = "per-client 강도 변조" 그 자체.

## 확인된 사실(데이터 재소싱; runs/ 읽기전용)

- **frrand는 실재하나 CNN flirds 단독 leg**: `track_h/analysis/cnn_competition.csv` threat에
  `frrand` 존재하지만 `source=flirds`만·`timing=online`만(다른 7 방법·retrain 미실행). flirds
  frrand(gate_v2) .5895 > vanilla .5876·random_excl .5839, oracle_excl .6195 근처 → frzero(.6148)와
  같은 exact-0 생존 계열. **R4(LLM) frrand는 전혀 미실행**.
- **뭉친 값 개별화 원천**: CNN online/retrain 전 방법 = cnn_competition(cifar10/dir1/P1). anchor5
  same-game 3종은 `track_d/fidelity.csv`에서 (b)-완전일치(spearman 1.0) 확인 → vs (a) provably
  동일 0.933±.047. 부록 D stability = `track_c/RESULTS.txt` 10칸 재집계, (b) 0.518·Flirds 0.547
  기존값과 일치(방법론 검증) → loss-heur/Flirds-1st/recon 5종 개별값 확정.
- **REMAINING.md에 "R4 frrand 재제안 금지"가 실제로 있었음**(§1.6 제외목록) — 이번 요청이 그 결정의
  번복임을 명시 처리.

## 결정(Yonghee)

- **R4에 frrand + strmain류 per-client dose를 추가한다**(종전 "R4 frrand 재제안 금지" 번복).
  값싸고 모달리티 제약 없음. fidelity 변별에 이로우나 **caveat = R4 포화 주원인은 near-additive
  레짐**이라 dose 변조로 부분 개선만 될 수 있음.

## 반영(로컬; paper/ 무수정·figure 재생성 불요=표-only·로컬 커밋만)

- overview: §5.3 CNN online/retrain 뭉치 → 방법별 개별 행 + 앵커(vanilla/oracle/random_excl) 명시,
  online에 frrand 열(flirds+앵커, 나머지 "–"+각주) · §5.2 anchor5 same-game 3행 분리 · 부록 D
  stability 9행 개별화 · 서술 클레임 방법명 명시형 · §5.1(C) 위협표·설계노트·F-4 각주·§5.3 R4
  개입표에 R4 frrand/strmain-dose ⬚ 축 등재.
- REMAINING.md: 제외목록에서 "R4 frrand" 번복 + **§1.6a**(L9 R4 frrand ~50–70 GPU-h · L10 R4
  strmain-dose ~30–45; 구현 = L6 graded-noisy 승격 + track_c2 frrand 훅 LLM 대응).

## 후속(Yonghee 몫)

- push(로컬 커밋만). L9/L10 우선순위·seed 수 판단(§1.6a 상태 = 판단 대기).
