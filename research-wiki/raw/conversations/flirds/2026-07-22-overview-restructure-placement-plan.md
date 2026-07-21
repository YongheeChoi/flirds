---
type: conversation
date: 2026-07-22
topic: flirds
participants: [Yonghee, Claude]
tags: [overview, restructure, placement-plan, banzhaf-ripple-exclusion, convergence-scope, beta03, verification]
---

# 2026-07-22 — overview 최신화·재구조화 + 논문 실험 배치안

## Yonghee 지시 (요지)

1. 대상 = `research-wiki/survey/flirds-experiment-results-overview.md`. 순서 고정: Phase 1(최신화·runs 정합성 검증, 수집만) → Phase 2(재구조화, 반영은 여기서 한 번에) → Phase 3(논문 배치안).
2. 확정 결정(재논의 없이 적용): **① Banzhaf·Ripple 비교군 제외**(전 표·범례·노트에서 제거, rundir 데이터는 존속, Caveats 1줄) **② 수렴(§3.3) 절 삭제**(데이터 포인터만; 위계 정의의 Conv 항목은 "(overview 스코프 제외, 2026-07-22)" 주석 처리 — 위계 자체는 헌장이라 불변) **③ 검증-전용 실험은 본문 제외 가능**(애매하면 보류 목록) **④ Main/Ablation 분리**.
3. 재구조화 원칙: 실행 세트명 기준 분류 탈피 → 검증 목적 기준; 옛↔새 매핑 표 1개; 전 실험 섹션 = (a)세팅/(b)결과/(c)출처·baseline-set 3블록; 3층 = Main/Ablation/기타 분석 모음 + 검증-전용 축약.
4. 수치는 파일에서만(계통 대조, 스팟체크 아님); rundir 무수정(재생성은 gitignored 파생만); GPU 금지; paper-ko.md 무수정(§4 이후는 배치 근거 사용 금지); 커밋 로컬만·push 금지.
5. 산출물: 재작성 overview(수용 기준: banzhaf/ripple grep = Caveats 1줄뿐 · 수렴 절 부재+포인터 · 3블록 포맷 · §커버리지↔runs 1:1·깨진 참조 0 · 헤더 최신) + `paper-experiment-placement-plan.md` 신규 + 세션 보고 + 로컬 커밋.

## Claude 수행 내역

### Phase 1 — 검증 (인벤토리 1 + 영역별 검증 에이전트 5, 계통 대조)

- **runs/ 인벤토리**: 11개 세트 전수. 미커밋 변경 0(디스크=HEAD `f5d40a7`). track_h가 최신(P5 108런·Scale 21런·Dyn 9런, 07-20~21).
- **수치 계통 대조 결과**: track_d(§3.1.1 3표·vs(a)·MMLU/ROUGE·runtime·E4/E5·target-stability) / track_c(pool·시나리오·Kendall거리·stability·C2·runtime 전 표) / probe·removal·계측·phase1·rerun_beta03 — **전부 파일과 일치**(반올림 경계 수 건 제외). track_g/h — 전 표 일치, 유일 실질 = **§3.2.6 총평 flirds .570→.568**.
- **실질 stale 1덩어리 = β0.3 재실행(ce0b454, 07-20)이 `1B_silo5` 오염 4셀을 제자리 교체**: §3.4.1 9칸 + §3.6.4 5칸 + §3.1.5 frzero 1행이 구판 기준. 최대 변화 = **Flirds poison AUROC 0.917±.118 → 0.500±.354**(per-seed .25/.25/1.0)·Sp 0.967→0.600; frzero xseed ρ 1.000→0.933; STD-DAGMM 3칸; 재실행 스위트에 **Fed-LOO·ComFedSV 신규 추가**(15종); loss-heur runtime rundir에 post-fix 96.6/100.1/99.9s 영속.
- **깨진 참조·stale 사실**: `REMAINING.md`는 **저장소 루트**에 존재(`REMAINING_after_e_session_2026-07-19.md`에서 개명; overview의 "§1.1=Tier3 std50k5" 참조는 현행과 불일치 — 현행 §1.1=R4) · `[[review-claude]]`·`[[2026-07-verification-overview]]`·`overview-figures-2026-07/`·`removal-dose-2026-07/`는 정리 커밋들에서 삭제 · track_g `rundirs_llm/`은 "빈 폴더"가 아니라 부재(LLM은 `rundirs/` 218) · §3.4.5 뒤 "CNN 트랙 탐지 AUROC" dangling 조각(데이터는 c1 metrics.json에 실재 확인).
- **overview에 없던 신규 사실**: ① 루트 `RERUN_AFTER_REPRO_FIX_2026-07-21.md` = **재현성 정정(H1: LLM LoRA adapter init unseeded → 전 LLM rundir 절대값 비재현**, 순위 결론 강건 예상; TF32 부분은 fa2c167 원복; M1 스택 혼재) ② **R4 gsm50k5 Tier A seed0 서버 실행 중**(07-20 23:29~, rundir 미착지) ③ track_g std50k5-mixed = seed0 파일럿 **동결**(1-seed caveat; LLM 참여축 주장은 R4가 대체 — REMAINING §2) ④ `measured_2026-07/loss_heur_acct/` 신규 영속 ⑤ β 캠페인 잔여 재편(18셀 + deferred 9셀).

### Phase 2 — 재작성 (1603→1609줄; 배포 완료)

새 구조: 헤더 → §1 범례 → §2 마스터표(+§링크 열·P11 R4 행·행26 확증런) + §2.1 매핑표 → **§3 Main**(3.1 Fidelity[LLM·CNN+stability·E4/E5·오염요지·B축] / 3.2 Selection[arms·C2·게이팅·top-k·점수원 경쟁] / 3.3 Detection[silo5 β0.3 정본·device100·3B·frdelta·**CNN AUROC 신설**] / 3.4 Cost[op-count·runtime]) → **§4 Ablation**(2차항 종합·A축 LLM/CNN·removal·dose·AdamW·**β 신설**·P5/Scale/Dyn) → **§5 기타 분석**(종합판정[구 2.5+3.2.7 통합]·부호감사·신호진단 종합·Exp C[보류]·Taylor[보류]·수렴 포인터·E매핑) → **§6 부록**(phase1·Caveats 13종[신규: 12 Banzhaf/Ripple 제외·13 재현성 H1]·**검증-전용 기록 신설**·상호링크) → §7 유지보수 → §8 커버리지(runs 1:1, matrix_cxni·rerun_beta03 귀속 명시).
- 수용 기준 전부 충족 확인: banzhaf/ripple grep 1줄 · 수렴 절 부재+§5.6 포인터+Conv 주석 · 3블록 포맷 · 깨진 참조 0 · 헤더 07-22/f5d40a7.

### Phase 3 — `survey/paper-experiment-placement-plan.md` 신규

paper-ko §초록–§3만 근거(§4 이후 비사용, paper-ko 무수정 유지 — `git diff paper/paper-ko.md`는 기존 미커밋분 그대로). 본문 실험 절 6개(E1 프로토콜→E2 exact fidelity→E3 신호 조건→E4 retrain 특성화→E5 비용→E6 실효성) + ablation 5항 + appendix 10항 + 미배치(수렴·Banzhaf/Ripple·P6) + 진행 중 의존성 표(R4·β 잔여·P2/P3·P0 재실행). removal은 overview에선 Ablation이나 paper §1이 명시 열거하므로 "본문 요약+appendix 상세" 분할 제안.

## 보류 목록 (Yonghee 확인 필요)

1. **(b) target self-stability(Exp C, §5.4)·Taylor 물리잔차(E2, §5.5)** — appendix 후보로 보류 표기.
2. **Flirds poison 0.917→0.500 서사** — β0.3 재실행 정본 채택 + 구값 각주 처리로 반영했으나, "2차항 부분 방어" 주장 약화(간헐 방어)의 논문 서술 방향은 Yonghee 판단.
3. **Tier 3 std50k5 12런** — 현행 REMAINING 큐 미등재를 "R4 대체·동결"로 기술; 공식 폐기 여부 확인.
4. removal A3 잔여 옵션(label/quantity_skew·pixel-backdoor ASR) — 기존 "Yonghee 결정 대기" 유지.

## 산출물

- 재작성: `research-wiki/survey/flirds-experiment-results-overview.md`
- 신규: `research-wiki/survey/paper-experiment-placement-plan.md`
- 본 세션 로그 + `wiki/log.md` append. 커밋 = 로컬만(push 금지, Yonghee 검토 대기).
