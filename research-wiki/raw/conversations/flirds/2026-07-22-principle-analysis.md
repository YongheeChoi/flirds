---
type: conversation
date: 2026-07-22
topic: flirds
participants: [Yonghee, Claude]
tags: [principle-analysis, mechanism, win-loss-ledger, paper-settings, verification]
---

# 2026-07-22 — Flirds 원리 분석 세션 (승패 메커니즘 → 논문용 성공 세팅)

## Yonghee의 지시 (원문 = 루트 `PRINCIPLE_ANALYSIS_PROMPT_2026-07-22.md`)

flirds가 잘 할 수 있는 일의 **원리**를 이해·파악해 성공적으로 실험할 수 있는 세팅을 찾을 것.
회귀분석을 할 만큼 실험이 많지 않으므로 승패의 **메커니즘 해석**이 핵심 가치. 3단계 —
① 실험 요소 전수 인벤토리(누락이 과잉보다 나쁨) ② 승패 원장+원리 분석(이론 먼저, 데이터로
검증; 동률·패배 포함; 교락 점검; 고유성 구분) ③ 축(fidelity/downstream/detection)×모델
(CNN/LLM) 매트릭스로 논문용 세팅 제안. **독립성 규칙**: overview의 표 수치는 데이터, "읽기"
·"판정"·§5.1은 의견 — 자기 분석 완료 후에만 대조(불일치 발견이 특히 가치). 제약: runs
read-only·GPU 금지(무GPU 파생 계산 권장)·paper/ 무수정·수치는 파일에서만·로컬 커밋만.

## 수행 내역

1. **입력 정독**: overview 1610행 전체(데이터 블록 위주 추출), irds-fl-math-rigor P1–P8
   (특히 P5 고정가중 vs 재정규화·P6 path-dependence), track_g/track_h README 사전등록
   예측표 원문, REMAINING.md, rundir config.yaml 샘플링(7개 스테이지).
2. **메커니즘 코드 정본화**: `codes/flirds/`에서 (b) oracle(D1·D2)·estimator(닫힌형)·
   GTG/FedSV(renorm-재구성 MC+plain-sum)·ShapleyFL(uniform+min-max+EMA β0.3=신규 0.7)·
   FedIF(단위정규화 1차+min-max+EMA γ0.3)·ComFedSV(저랭크 완성)·탐지기 4종을 (게임, 추정기,
   후처리) 3요소로 그룹핑(G1–G7).
3. **수치 재검증**: 산출물 폴더에 `verify_numbers.py` 작성 — overview의 load-bearing 수치를
   rundir metrics.json/phi.parquet/분석 CSV에서 재계산 대조, **237 PASS / 0 FAIL**
   (`verification_report.txt`). A2 removal Δ의 정의를 곡선에서 역동정(L(k=0)−L(k=4)).
4. **무GPU 파생 계산(신규)**: ① 같은-1차-신호 쌍(FedIF↔Flirds-1st) 순위상관을 무대별 비교
   — silo5 +0.9~1.0 vs std20 +0.16 vs anchor5 산포 = **부분참여 붕괴가 1차 신호가 아니라
   min-max+EMA 후처리의 소행**임을 직접 판별. ② frzero bit-exact 0(6종) vs renorm 유령값
   \|φ\|≈0.004~0.005(clean φ와 동급 크기) 직접 확인. ③ noisy dose ladder φ(오염 클라) 전
   구간 기여-양수(−0.00246→−0.00186) — sign-게이트 작동영역 부재의 직접 확인. ④ Track H
   경쟁 총평 자체 재계산(flirds .5682 1위·flirds1st .4712 최하 — overview와 정합).
5. **산출물 작성**: `research-wiki/survey/flirds-principle-analysis.md` —
   §1 인벤토리(9개 축·요소별 1행 표·암묵 고정 발굴), §2 승패 원장(포화/진짜 승 구분·패배
   전수)+이론 유도 T1–T10+원리 주장 10건(각 뒷받침≥2·반증·교락·신뢰도·서사)+고유성 구분+
   교락 목록 D-1~8+**§5.1 대조표(불일치·보완 6건)**, §3 제안 매트릭스+피해야 할 세팅 10+
   기존/신규 구분+CNN→LLM 전이 리스크+shortlist 5.

## 핵심 결론 (요지)

- **원리 골격**: 신호는 B축(클라 간 실재 차이)이 만들고(1) LoRA 소스텝 무대는 가산 축퇴로
  same-game 동률이 필연(2); 2차항의 가치 영역 = 0-평균 방향 위협 ∪ 저참여·대스텝(3);
  경쟁자 부분참여 붕괴는 min-max+EMA 후처리의 소행(4); zero-semantics(null-player exact-0
  vs renorm 파괴)가 개입 실효성을 가름(5); sign-게이트 작동영역은 φ-부호 궤적으로 사전 판정
  가능(6); 기여도≠탐지는 공리적 갭(7); **Taylor 품질은 optimizer-기하 의존(AdamW서 0차
  실평가에 패배 — 진짜 패배 셀)**(8); 결정론 상속 안정성(9); 비용 우위는 K_r≥3 조건부(10).
- **§5.1과의 불일치 6건**(요구사항): ① CNN vs(b) 1위 서술에 심판-이점 명시 필요(removal
  우선 제안) ② AdamW를 "게임 괴리"가 아니라 **vs (b) 기준 same-game 내 패배**로 재규정
  ③ silo5 탐지 1.0은 "승"이 아니라 포화 ④ "GN 잡는 유일한 estimator"는 시점-강건 기준으로
  한정(online은 loss-heur .598>flirds .567) ⑤ **CNN vs(a) 1위=ShapleyFL .453**(renorm-족
  심판엔 renorm-족이 유리 — T10 예측 적중)이 문서에 누락된 발견 ⑥ 7B xseed +0.733 예외의
  매트릭스 미반영.
- **shortlist**: ① gn_full 완주(H-10; 큐) ② R4 Tier C 3-seed(~95–130 GPU-h; 헤드라인
  방어) ③ 3B silo5 3-seed(큐) ④ **비등n silo5 1셀(신규 제안; ~10–15 GPU-h; P5c 이론
  반례의 실증)** ⑤ Tier B renorm(T2-only 축소 검토; 승인 게이트).

## 산출물·커밋

- `research-wiki/survey/flirds-principle-analysis.md` (본 분석)
- `research-wiki/survey/flirds-principle-analysis/{verify_numbers.py,verification_report.txt}`
- 본 세션 로그 + `wiki/log.md` append. 로컬 커밋만(push 안 함 — Yonghee 검토 대기).
- runs/·paper/ 무수정, GPU 실행 0.
