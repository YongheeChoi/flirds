# 미뤄둔 엄밀 검증 항목 인벤토리 — 통합본 (검증 캠페인 항목 4)

- 작성: 2026-07-04, 검증·감사 캠페인(`PROMPT_VERIFICATION_SURVEY_2026-07.md` 항목 4) 병합 에이전트
- 성격: **읽기 전용 감사 산출물** — 코드·위키·rundir 무수정, 본 문서만 신규 생성
- 소스: 정찰 스윕 노트 3종(스크래치 작업 파일 — 세션 종료 후 소멸 가능; 본 문서는 1차 출처만으로 자립하도록 인용을 원문 file:line으로 옮겨 적음)
  - 위키 스윕(`…/scratchpad/recon/deferred_wiki.md`) — 번호 항목 **35건** + 스코프 메모 6건
  - raw 대화록 스윕(`…/scratchpad/recon/deferred_raw.md`) — 번호 항목 **31건**(경미 표시 3건 포함) + 참고 표 11행
  - 코드·memory·CLAUDE.md 스윕(`…/scratchpad/recon/deferred_code.md`) — 번호 항목 **19건** + 순수-엔지니어링 제외표 9행 + 절차성 5건
- **병합 결과: 메인 표 54건**(I-01~I-54) + 참고·제외 21건(§2.7 = raw-F 설계종결 7 + 코드-B 엔지니어링 9 + 위키-F 엔지니어링 4 + bracket 정정 1). 세 노트의 번호 항목 85건과 부속(메모·표·절차성) 전부가 메인 표 병합 또는 §2.7에 귀속됨 — 매핑은 §1.3.

표기·축약: plan=`research-wiki/wiki/flirds-implementation-plan.md`, flirds=`wiki/flirds.md`, proto=`wiki/flirds-protocol.md`, diag=`wiki/flirds-signal-size-diagnosis.md`, log=`wiki/log.md`, ov=`research-wiki/survey/flirds-experiment-results-overview-2026-06-25.md`. raw 파일은 `research-wiki/raw/conversations/flirds/` 생략. 코드는 `codes/` 기준. **라인 번호는 2026-07-04 스윕 시점 기준 — 인용 전 재확인 필요**(캠페인 공통 규약). ⚠=스윕 노트가 미확정으로 표시한 사실, [추정]=원문 근거 없는 합리적 추정(스윕 노트 표기 승계).

---

## 1. 방법론

### 1.1 병합 기준
1. **동일 근원 병합**: 같은 설계 결정 또는 같은 rundir 공백에서 파생된 항목은 1건으로 통합하고 세 노트의 출처를 전부 병기 (예: "N=10 oracle deferral"은 위키 A1+E6, raw A1+B1, 코드 A1의 5개 항목을 1건으로).
2. **누락 금지**: 세 노트의 모든 번호 항목·참고 표 행·절차성 항목을 메인 표(54건) 또는 §2.7 참고·제외(21건)에 귀속. 매핑 표(§1.3)로 전수 검증 가능.
3. **타 과업 이관 항목도 등재**: 정밀도(항목 3)·Ripple(항목 2)·cost 회계(항목 6)·IRDS 수학(항목 1)의 본체는 해당 과업 문서가 다루되, 본 인벤토리에 교차 표시로 1건씩 등재(이중 작업 방지 + 전수성 유지).

### 1.2 우선순위 기준
**핵심 질문 위계**(루트 CLAUDE.md, Yonghee 2026-06-12 명시) 우선:
- **P1** = 1차 질문(기여도 측정의 oracle 대비 정확도 = fidelity)에 직접 닿는 항목 — 최상위
- **P2** = 무대·프로토콜의 외적 타당성(1차·2차 결과 공통 기반)
- **P3** = 2차-① 일반 성능 / ② 수렴
- **P4** = 2차-③ 탐지·위협모델 (위계상 마지막)
- **P5** = 특성화 잔여·타 과업 이관·경미·파킹

같은 급 안에서는 "결론이 흔들릴 위험 × 엄밀화 비용의 역수"로 정렬 — 표의 행 순서가 곧 급내 순위다. 예외적으로 P4 상단 3건(I-27/28/29)은 위계상 탐지 축이지만 **프로젝트 스스로 위험을 자인해 둔 채 미실행**이라(§4 상세) 상위 10에 포함했다.

### 1.3 노트→병합 ID 매핑 (전수)
- **위키 노트**: A1→I-02, A2→I-06, A3→I-01, A4→I-05, A5→I-17, A6→I-03, A7→I-08 / B1→I-07, B2→I-16, B3→I-10, B4→I-20, B5→I-21, B6→I-14, B7→I-34, B8→I-22, B9→I-15, B10→I-23 / C1→I-30, C2→I-31, C3→I-28, C4→I-27, C5→I-38, C6→I-40, C7→I-29 / D1→I-13, D2→I-49·I-50, D3→I-12, D4→I-37, D5→I-41 / E1→I-04, E2→I-42·I-43·I-44, E3→I-46, E4→I-08, E5→I-51, E6→I-02 / F절 메모 6건: noise-vs-OOD→I-45, fp32→I-48, IRDS 수학→항목 1 이관(I-04 연계), cost→I-49, 순수 엔지니어링→§2.7, seed schedule→I-53.
- **raw 노트**: A1→I-02, A2→I-06, A3→I-05, A4→I-19 / B1→I-01·I-02, B2→I-14, B3→I-06, B4→I-48, B5→I-12 / C1→I-27, C2→I-30, C3→I-32, C4→I-29, C5→I-33, C6→I-35, C7→I-37, C8→I-13, C9→I-29, C10→I-49 / D1→I-04, D2→I-42, D3→I-43, D4→I-44, D5→I-45, D6→I-39, D7→I-07 / E1→I-20, E2→I-24, E3→I-03, E4→I-10, E5→I-52 / F표 11행: filter q-sweep→**I-54 승격**, STD-DAGMM 원류 delta→I-27 병합, drift-residual "측정만"→I-04 병합, Ripple 코드중복→I-13 병합, 나머지 7행(ghost·참여횟수 정규화·13B+/70B·SPACE류 제외·DBA·GGN·removal curve)→§2.7.
- **코드 노트**: A1→I-02·I-06, A2→I-09, A3→I-05, A4→I-11, A5→I-49, A6→I-12, A7→I-27, A8→I-13, A9→I-36, A10→I-47, A11→I-20, A12→I-25, A13→I-03, A14→I-10, A15→I-01, A16→I-33, A17→I-24, A18→I-18, A19→I-40 / B 제외표 9행→§2.7(원노트 판정 유지) / C절 5건: Track D 결정 3건+real run→I-26, clean-oracle bracket arm→§2.7(정정: 의도적 제외), FedSV 이름충돌·sight-unseen 3건·findings.md 빈 템플릿→I-53.

---

## 2. 메인 표

### 2.1 P1 — 1차 질문(fidelity) 직결 (I-01 ~ I-19)

| ID | 항목 | 간소화·미룸 내용 | 당시 사유 (출처) | 흔들릴 수 있는 결론 | 엄밀화 방법·예상 비용 | 우선순위 |
|---|---|---|---|---|---|---|
| I-01 | N=5 근사-가산 무대·coarse 분해능 | 주력 fidelity·AUROC 수치가 전부 N=5 산출 — Spearman은 5점 순위(가능 순위 120개), AUROC는 1-positive coarse. 사후 진단으로 anchor 게임 가산 갭 ≤0.9%·전 방법 +1.000 붕괴 확인 | coarse 자인(plan:357; ov:180) + 비용 계층. 가산성은 07-02 사후 판명(diag §1.5:170–185; raw 2026-07-02-signal-size-diagnosis-probe-plan.md:32-33) | "+1.000 프런티어" 주장에서 fidelity 축이 사실상 무정보(근거는 비용 축뿐); N=5 AUROC 0.75/1.0의 표면 신뢰도; 방법 간 차별화는 부분참여에서만 발생 | 서술 수정 비용 0 — 헤드라인을 부분참여 fidelity(1B std20: ComFedSV +0.09/ShapleyFL +0.19 vs Flirds +1.000; log:545)로 승격. 실험은 I-02·I-10이 해소 | P1-1 |
| I-02 | LLM N=10 (a)/(b) oracle + N=10 detection headline 부재 | N=10 (a) 1024-retrain·(b) 2¹⁰ enum 모두 미실행(ov P1 ⬚); "headline detection = N=10/N=100" 예고(plan:357)의 N=10도 부재 | (a) retrain SV = N=5(2⁵)만 코드 확정(`flirds/oracle/exact_sv_llm.py:21-22`; per-coalition timing은 외삽용); (b) 비용 32×(proto:90); (a)=2–5일 단일GPU, 샤딩 미구축(plan:143; raw 2026-06-07-phase2-task6-a-retrain-oracle.md:105-106); lock "deferred (last, costly)"(flirds:194–198; plan:340,347) | "+1.000 = 방법 검증"의 고-power 버전 부재; proto §9 N=10 primary(proto:141-148)와 실측 괴리; "CNN N=10+LLM N=5만으론 unconvincing"은 Yonghee 자신이 제기(raw 2026-06-04-phase1-data-layer.md:21) | (b) N=10 1B ≈3.5h×3seed≈10h(proto:77; raw 2026-05-27-section-23-lock.md:127); (a) N=10 샤딩 ~11–22h(plan:143) 또는 ~67 GPU-h fp32(raw task6:84-94)+오케스트레이션 코드 신규; detection은 같은 run에서 나옴(추가 비용 0) | P1-2 |
| I-03 | IID-clean 신호 실재성 부재 + clean×non-IID 칸 공백 | Track D 무대에서 (b) oracle 자기 순위 cross-seed ρ≈0(anchor −0.37/std20 −0.11; diag:110-113); 분리 검증용 §2.4 매트릭스는 6셀 스모크만 green·본실행 대기(diag:313)[07-07 완료: 3-seed 본실행 = diag §3.3, non-IID clean cross-seed ρ +0.87 vs IID clean +0.13]; LLM clean×non-IID 칸 빈칸(루트 CLAUDE.md next) | 06-13 재정의가 IID·clean을 문헌 표준무대로 채택(plan:436-437); 신호 부재는 07-02 사후 진단(diag:20-28,116-118); silo5는 오염·비IID 결합으로 분리 불가(diag:138-139) | Track D "+1.000"의 순위 대상 자체가 노이즈(과대해석 위험); do-no-harm parity가 무정보; "non-IID면 신호 실재"가 오염 신호와 미분리 | §2.4 신설 6셀×3seed, silo5급(diag:274-286) [추정 수십 GPU-h]; FedHDS 무대 설계 세션 별도(raw 2026-06-13-track-d-redesign-iid-clean.md:66-70) | P1-3 |
| I-04 | Prop 1/Prop 2 실증(E-sweep) 부재 | #5 E{1,3,5,10}×α{0.01,0.1,0.5,5.0}×3seed×3scale 미실행(plan:155; flirds:141) — E축은 어느 트랙에도 없음. pilot U-shape 후 canonical 재검증 미이행; "drift residual measured, not corrected"의 measured 미이행(flirds:62; raw conversation4.md:44-50); #9 non-IID bias 분해도 연동 공백(flirds:147) | Phase 3 배정 후 그리드가 detection 중심 재편(plan:150-155); N1 lock "canonical target 유지, pilot set-aside"(raw 2026-05-19-…:280-303) | 논문 수학 명제 2건이 empirical validation 없이 제출됨; 05-22 "established" 주장 전부가 Yonghee 정정으로 "planned" 롤백된 후 재확증 기록 없음(raw 2026-05-19:292-301) | 최소판 1B E 4점×α 2점×3seed = 십수 셀 [추정 수십~100+ GPU-h]; 검증 캠페인 항목 1(수학)과 병합 처리; U-shape 재현 시 contingency 분기(flirds:141) | P1-4 |
| I-05 | off-anchor "Flirds proxy-truth" 순환 | device100 α-sweep 12셀의 진리 기준 = 검증된 Flirds 자신(코드 확정: `experiments/phase2_matrix.py:20-23,370` `truth_method="(b)oracle" if … else "Flirds"`); exact (b)는 α=0.5 anchor 1점만 | (b)-perround 칸당 ~25,000s(≈7h; ov:526) — 단 raw는 771ms/fwd×R200×2^K로 ≈44h/GPU 산정(raw 2026-06-08-phase2-task7-crossdevice-detection-redesign.md:33-36)이라 두 출처 ~6× 불일치(원 문서 간 미정합 승계); 실행전략 lock(plan:341, Yonghee 06-09) | α 극단(0, 0.01)에서 Flirds-oracle 정합 무보증(검증점 α=0.5 하나 — 하필 drift-residual 최대 지점); 타 방법의 off-anchor +1.000은 "vs Flirds"이지 "vs oracle" 아님; checkpoint 07 자인 한계(log:515) | α=0(±0.01) 1–2셀에 (b)-perround anchor 추가 = ~11h/4-GPU/셀(plan:19); 표에서 proxy-truth 셀 명시 분리 = 비용 0 | P1-5 |
| I-06 | 스케일 방향 (a)-oracle 공백 (3B 1-run·rundir 부재 / 7B 제외) | 3B (a)vs(b)=+0.900은 1-run(lr3e-3 fp32 9483s; plan:143), "one clean-client swap = retrain noise" 해석 무검증(raw crossdevice:46-49), rundir 부재로 ov 미수록(ov:782 Caveat 2); 7B (a) 전면 제외(proto:63-71) ↔ ov P3는 "채울 행"(ov:102-103) — 문서 간 지위 불일치 ⚠ | 7B (a) 미실행 코드 확정(`experiments/phase2_matrix.py:34-35` "bf16 is the deferred (a) retrain oracle, which 7B does not run"); 7B retrain 42일 → infeasible lock(raw 2026-05-27:126,131-132); 선례 문구 준비됨("LESS·Grosse·DataInf 모두 ≥7B exact retrain 생략"; proto §4.1) | "듀얼 오라클 방법 검증"이 실질 1B N=5 1셀 의존; +0.900이 noise가 아니라 스케일 괴리의 앞머리일 가능성 배제 안 됨(1B는 +1.000) | 3B (a) N=5 3-seed 재실행+rundir 영속화(~2.6h/run fp32 → 반나절~1일; ov:807에 커맨드 존재); 7B는 준비된 문구로 스코프 한정 서술 | P1-6 |
| I-07 | val 크기 lock-실측 괴리 + #16 val-sensitivity 미실행 | lock=도메인당 200/총 1000(plan:40,252; proto:135) vs 실측 100(baseline; log:467)/200(Track D; diag:43)/20(silo5)/10(device100; ov:787) — plan:344엔 "tiny val=4"도 등장, 최종값 rundir 확인 필요 ⚠; #16 ablation 미실행(flirds:161) | lock 근거 "few-shot(~50) noisy"(plan:252); 축소 사유 셀별 미기록 [추정: (b) 비용이 \|val\|에 선형이라 그리드 완주 절충]; caveat 관리만(plan:344) | 25셀 robustness 그리드의 AUROC/Spearman 전부 — val=10–20은 lock이 기각했던 noisy 영역; proto:13 "silent deviation 금지" 위반 상태; "uniform coverage" lock의 강건성 무실증(MATES 경고; flirds:161) | #16은 post-hoc 재평가 전용 설계(flirds:161 "re-eval only") — forward-only [추정 수 GPU-h/셀]; probe C(per-chunk val bootstrap SE, ≈4 GPU-h)는 설계 완료·가동 중(diag:219,229; raw 2026-07-02:37-45)[07-07 완료: diag §3.4 — φ spread ≈ val 노이즈 하한 ~1.1배] | P1-7 |
| I-08 | 듀얼 오라클 (a)/(b) 괴리 특성화 부재 (#8 stress 미실행) | (a)vs(b)=0.933±.047(1B anchor; ov:180)·CNN fine-rank ρ≈0.66(flirds:44)을 "다른 utility"로 종결; 언제/얼마나 갈라지는지 체계 특성화 없음; #8 adversarial stress(극단 α·flip×OOD·큰 N)의 (a)/(b) 병행 보고 미실행(flirds:146; plan:158) | "expected, different utilities"(flirds:44); N=5 coarse+retrain noise(ov:180); Phase 3 배정 | "(b)가 (a)의 싼 대리" 서사의 적용 한계 미기술; 0.66(CNN)↔0.933(LLM) 간극 원인 미규명; "(b) holds, (a) degrades" 가설(N3 앵커) 미검 | α=0 셀 (b) anchor는 I-05와 공유(부분 충족); (a) 병행은 I-02/I-06과 동일 비용 구조 | P1-8 |
| I-09 | Fed-LOO 수치 부재 | Federated-LOO 구현·합성검증 완료(`flirds/oracle/in_run_sv.py:71-98`; brute-force max-diff 0.0)이나 기존 rundir logs 미영속 → 백필 불가, 모든 fidelity 표에 숫자 0건 | 07-02에야 gap으로 식별(memory `baseline-selection-audit-loo-gap.md:12`; loss-heur는 singleton U({k})라 LOO 아님) | 분야표준 앵커 baseline(LOO) 대비 비교가 논문 표에 없음 — 리뷰어 1순위 요구 예상 | fidelity 셀 재실행(logs 재생성): CNN track_c1 시간 단위/셀, LLM track_d·phase2는 트랙 재실행 비용; `make_fidelity`가 동적 수집이라 재실행만으로 자동 반영 | P1-9 |
| I-10 | LoRA rank·참여 probe 미완 + rank sweep lock 미이행 | lock "rank sweep {16,32,64,128}"(plan:38) 중 본실험 전부 r16; probe가 r{32,64} 개방했으나 seed0 파일럿만·r128 미계획(diag §2.1,§3.1); CNN w{0.5,1,2,4}×참여 grid 제출 대기(루트 CLAUDE.md next)[정정: 07-03에 이미 3-seed 완주·커밋(d2e7ed6, 본 문서 작성 전) — runs/probe_signal/cnn_c1·c2; 진단 기입은 diag §3.6(07-07)]; CNN 폭↔rank disanalogy caveat(raw 2026-07-02:48) | r16/α32=FedDQC 선례 매칭(plan:222-228); sweep 이월 사유 미기록 [추정: Phase 밀림+비용]; probe에서 rank가 A축 lever로 재부상(diag:34-37) | 2차항의 rank 의존성(HVP가 2r-차원)이 1-seed 근거뿐(seed0 +1.000 유지; diag:298-306); "신호 구조적 부재" 결론의 probe 재확인 미완; Track D r16 caveat 정량화 부재 | probe A그룹 seeds1–2 완주 ≈40 GPU-h 잔여(diag:226-227); r128 [추정 +30 GPU-h] 또는 lock을 {16,32,64}로 개정 선언; CNN grid는 제출만 하면 됨 | P1-10 |
| I-11 | GTG/FedSV truncation 임계 ≈ 신호 크기 | FedSV `trunc_eps=0.001`(`baselines/fedsv.py:18,26-27,40,46`), GTG `round_trunc=0.001`+`eps=0.001`(`baselines/gtg.py:95,121,146,152`) — val-loss 절대 단위 임계가 측정된 클라 간 갭(≤0.9%; CLAUDE.md)과 같은 자리수 | 원 논문 기본값 이식 [추정]; 신호 크기는 07-02에야 실측 | truncation이 노이즈가 아니라 신호를 자를 수 있음 → GTG/FedSV fidelity(renorm≠0 포함)·runtime 모두 이 하이퍼 의존; cost 비교(항목 6)에도 직결 | trunc=0 대조 1셀(silo5 1-seed, ≤1h급 [추정]) + 논문에 임계값·단위 명기 — 저비용 대비 방어력 큼 | P1-11 |
| I-12 | ComFedSV 평가 조건 3건 | ① C1 partial=False로 ALS completion 우회(plan:407) ② from-logs 경로에서 Assumption 1(round0 전원 참여) 미보장·관측-0 클라 비율 미로깅(`baselines/comfedsv.py:14,97,127-166`) ③ CNN Phase0 재현치 {1.0,0.96,0.85,0.84}가 seed0/디바이스 민감 soft-CHECK 방치(raw crossdevice:44-45); 저R completion-starved는 기지(plan:342) | ① 수치 안정성(φ~1e-219 붕괴 회피)+full 관측엔 수학적 불요 ② 주석 자체 방어("exactly ComFedSV's setting") ③ "관찰만, pre-existing" | std20 +0.093이 방법 한계인지 관측-굶김인지 미분리(baseline 공정성 시비); Phase0 ±5% 재현 게이트 신뢰도; device100 completion 품질 미관측 | 관측율·mask 채움율 로깅(코드 몇 줄)+1셀 재계산; verify 3–5seed×CPU/GPU(값싼 CNN, 수 시간); std20 R 연장 1셀 [추정 수 GPU-h]; "R≥30 정상" 기지 사실 병기=비용 0 | P1-12 |
| I-13 | Ripple — 이월 3건·제외 상시화·구현 미완 | 이월 3건(streaming/eigsh 수렴/task-driven 검증; plan:70); real grid 제외 "dominated+flaky" 한 줄(plan:340); LA/LM TODO가 LLM 포트에서도 미해소(`baselines/ripple.py:120-137`; `ripple_llm.py:110-119` LA 그대로)·미수렴 시 부분 sketch/zero-pad; 자체-궤적 실행으로 표 내 회계 유일 불일치; "교정 별도 세션"(plan:389 ⑥) 미개최였음 | 1B N=5 실측 최저 성능(~4515s[노트 전용 — raw 06-06 기록, rundir 미영속], AUROC 0.50±0.20; flirds:126은 '~42×'로 기재) + ArpackNoConvergence 실측 | "closest competitor"(flirds:244) 비교가 CNN+1회 N=5 예비뿐; 42×·chance-AUROC이 우리-포트(부호·α 버그 수정본; plan:64) 산물일 가능성 — baseline 공정성 시비 | **검증 캠페인 항목 2로 이관**(충실도 감사·속도 분해·eigsh 진단·분리 계측). [실측 대기 — 항목 2: valuation-only 환산 Ripple 수치]. 재실측 3-seed ~4h(실측 4515s/run 기준) | P1-13 |
| I-14 | 3B robustness 1-seed / 7B robustness 미실행 | 그리드 05_scale_3b 4셀 = seed0 단독(ov:781 Caveat 1; raw 2026-06-15-…:85) ↔ proto §2 "single-seed 표 금지"(proto:33; raw 2026-05-19:308-309); 7B silo5/device100 전체 미실행(ov P4) | 비용 계층 lock(plan:341); 3B=스케일 확인 Tier; 재결정 기록 raw에 없음 [추정: 06-09 그리드 설계 시 암묵 수용] | "3B에서도 동일" 스케일 일반화 문장 전부; 하필 poison Flirds-2nd의 run간 분산 기관측({0,0.25,1.0} vs [0.75,1.0,1.0]; log:515,530) | 3B 4셀×seed1,2 추가(셀당 수 시간); 7B는 [실측 대기 — 7B silo5 셀당 wall-clock 추산(fp32 소배치 강제; plan:342)] 후 실행/한정 결정 | P1-14 |
| I-15 | 95% bootstrap CI 미이행 ⚠ | proto §3.3 전 헤드라인 지표 CI(B=1000) 요구(proto:51-57) + §12 "CI bands on every figure"(proto:210) vs 결과 문서 전반 mean±std만; make_analysis 산출물에 CI 존재 여부 미확인 ⚠ | 미기록 [추정: 분석 툴링 후순위 + 3-seed에서 정보량 한계] | "A>B" 주장 규칙(CI 비중첩; proto:57) 적용 수단 부재 — 고분산 수치(poison 0.917±0.118)의 주장 강도 | `make_analysis`/`make_fidelity`에 bootstrap 추가 — rundir-only, GPU 0 | P1-15 |
| I-16 | per-domain macro-average normalization ablation 미실행 | 도메인 공정성 장치(1/D macro vs token-prop)를 flag로만 구현(smoke 수치 log:417), ON/OFF ablation 미실행 — 어느 결과 문서에도 없음 | 06-02/06-04 결정 자체가 "option + ablation"(plan:40,195; proto:138; flirds:118) — novelty hook이라며 실험 이월 | 5-domain 공유-val Shapley의 도메인 공정성 주장; silo5 φ 도메인 편향(finance IRA 0.17≈noisy 0.067 사례; plan:338)이 선택에 좌우 가능 | 로그 궤적 위 post-hoc(val 집계 가중만 교체): estimator-only 분 오더, (b) 포함 [추정 수 GPU-h/셀] | P1-16 |
| I-17 | (b) MC 계획 사문화 | 원 프로토콜 MC(M=5000–10000, 분산 게이트; proto:45-49,109)가 exact per-round 분해(Δφ≈3e-16)로 대체된 뒤 문서 미갱신; MC-vs-exact 비교는 "MC 자체가 어디에도 없음" | Yonghee 06-08 "MC는 (a)용, in-run은 exact"(plan:19 7c) — exact가 더 강함 | 수치 위험 없음; 논문에 "MC 5000–10000" 문구 잔존 시 허위; (a) N=10 MC-halving 옵션이 무검증 잔존 | proto §3.2/§4.2/§5 갱신 = 비용 0; (a)를 MC로 갈 경우에만 CNN 선검증 [추정 <1 GPU-h] | P1-17 |
| I-18 | fidelity 거리·Pearson 해석 제약 | GTG 거리 3종(cosine/Euclid/max-diff)은 same-units 벡터끼리만 유의미(`eval/metrics.py:81-84`) — min-max+EMA 계열(ShapleyFL/FedIF/ComFedSV) 거리값 해석 불가; Pearson N=5 고분산(metrics.py:114); track_c1.py:27-29도 자인 | 로깅은 전 방법 일괄(해석 주의 주석으로 관리) | 표에 거리 열 일괄 기재 시 오독 유발; 값-수준 fidelity 결론의 주 근거는 N=10 CNN이어야 함 | 표 각주·해당 열 마스킹 = 비용 0 | P1-18 |
| I-19 | cross-device 표본 정의(참여 클라만) | N=100 Spearman은 선택된 71명 대상(raw crossdevice:33-34); AUROC는 corrupt-seen P≈0.96·NaN 처리(raw 2026-06-09-phase2-step5-matrix-orchestrator-task8.md:78-80) | 미선택 클라는 로그 부재 → φ=0, 구조적 | 표의 표본 정의(전 클라 vs 참여 클라) 미표기 시 오독 | 표 각주 = 비용 0 | P1-19 |

### 2.2 P2 — 무대·외적 타당성 (I-20 ~ I-23)

| ID | 항목 | 간소화·미룸 내용 | 당시 사유 (출처) | 흔들릴 수 있는 결론 | 엄밀화 방법·예상 비용 | 우선순위 |
|---|---|---|---|---|---|---|
| I-20 | plain SGD(mom=0)·상수 lr 고정 + bridge arm 결정 대기 | 전 트랙 momentum=0 SGD 통일(`fl/client.py:12-14`; `fl/llm_server.py:5-7,45`; plan:389 06-12 결정 ①); 문헌 표준(AdamW 5e-5 cosine)과의 갭은 Track D caveat 3으로만(plan:437); 갭 정량화 bridge arm(vanilla-AdamW 1회) 미구현·승인 대기(plan:438; 루트 CLAUDE.md) | 실증 lock: momentum 0.9에서 2차항이 1차보다 나빠짐(0.73<0.81), plain SGD에선 이김(0.96>0.92)(flirds:46) + IRDS Eq1 가정 일치; Yonghee "mom=0 변함 없어"(raw 2026-06-12-…:97); Adam 이론 확장은 deprioritized(log:346) | 모든 LLM 수치의 외적 타당성 — Adam 세계에서 Taylor 대응이 깨질 수 있음은 자체 momentum 실험이 시사; OpenFedLLM/FlowerTune 비교가능성("caveat-free 직접 비교 없음" 자인; plan:438) | bridge arm 1-seed×2레짐(밸류에이션 없이 학습+평가만 [추정 수 GPU-h]; AdamW 경로 신규 작성 필요 [추정]); 미실행 시 limitation 명기(FedIT-SGD 선례 인용 준비됨; plan:437) | P2-1 |
| I-21 | lr 민감도(lr-반전) + poison 셀만 lr fork | lr 체계 sweep은 phase1 1-seed 부록뿐(ov:97); checkpoint 07이 "lr-반전"을 한계 등재(log:515; 상세는 checkpoint 문서·rundir 대조 필요 ⚠); poison threat만 LR=2e-3/BATCH=8/EPOCHS=5 별도 config(plan:342-344) | backdoor 설치가 lr1e-3·batch16에서 실패(ASR=0) → 설치 가능성 위한 실용 fork(plan:342-343) | poison 셀 vs 타 threat 셀 비교가 lr/batch 효과와 교락; "Flirds evaded(AUROC 0)"가 lr=2e-3 특이 조건일 가능성; "#7 selection 양 lr 작동" 주장의 경계 조건 | poison lr 2점 교차(silo5 셀 재실행 [추정 수 GPU-h ×2]); lr sweep 3-seed 승격 [추정 10–20 GPU-h] | P2-2 |
| I-22 | seq512 절단 영향 미검토 | Track D=OpenFedLLM verbatim seq512(plan:437); 긴 completion 도메인(math AQUA·legal) val-loss 절단 왜곡 여부 미검토; cross-silo 트랙 seq 값은 wiki 미명시 — 코드 확인 필요 ⚠ | 문헌 무대 재현(비교가능성; plan:437) | 도메인 간 φ 공정성(I-16과 결합)에 2차 효과 [추측 — 위키 근거 없음] | val 절단율 측정(GPU 0) → 높으면 seq1024 대조 1셀 [추정 수 GPU-h] | P2-3 |
| I-23 | FedProx ablation row 미실행 | proto §7 "Ablation row: FedProx"(proto:129) — 어느 결과 문서에도 없음 | 미기록 [추정: Phase 3 밀림] | vanilla FedAvg 전제의 aggregator 강건성 — proximal 항 하 Δw_k Taylor 성립 여부 | 1B 1셀 FedProx + estimator/oracle 비교 [추정 수 GPU-h] | P2-4 |

### 2.3 P3 — 2차-① 일반 성능 / ② 수렴 (I-24 ~ I-26)

| ID | 항목 | 간소화·미룸 내용 | 당시 사유 (출처) | 흔들릴 수 있는 결론 | 엄밀화 방법·예상 비용 | 우선순위 |
|---|---|---|---|---|---|---|
| I-24 | MMLU 축 무력화 + 축② 지표 재선정 미결 | MMLU=GPT-judge의 API-free 대체(`eval/mmlu.py:1-15`)였으나 진단 결과 학습 0(포맷만; std20 −0.8pp/7B −1.4pp), 개입 효과는 paired val-loss에서만 검출(raw 2026-07-02:26-36); FlowerTune-채점 모드 결정 대기 | 축② 채택 시 "clean-IID 기대=parity"로 보수적 명시(raw 2026-06-13:87) — 검정력 0은 사후 발견 | 핵심 질문 2차-①(일반 성능) 축의 증거 부재 — 탐지보다 앞서는 축이 빈칸; parity가 '진짜 동률'인지 '지표 둔감'인지 구분 불가 | probe 판정 후 축② 지표 재선정(paired val-loss 승격 등); Alpaca-test ROUGE-L 병행 지표 기존(track_d.py:14); MMLU 단독 해석 금지 명기=비용 0 | P3-1 |
| I-25 | 곱셈 개입 규칙 = replacement와 동일(IID) | `multiplicative w∝n_i·s_i`는 n_i 균등이면 replacement와 수학 동일(`fl/intervene.py:11-18` 주석 자인); Track D=alpaca 20k IID/N=20 [추측: 균등 분할이면 완전 동일 — `build_alpaca_iid` 확인 필요] | 주석에 interpretation caveat로만 관리(track_d.py:32-36) | "곱셈 규칙"(Yonghee's rule, no precedent) novelty의 실증 무대가 Track D에 없음 — arm 차이가 점수·β 차이만 반영 | 분할 균등성 확인 = 비용 0; 곱셈 규칙 주장을 size-skew 셀(track_c/quantity_skew)로 한정 | P3-2 |
| I-26 | Track D 결정 3건 대기(real run은 완료) | [정정] Track D real run(1-seed 파일럿→3-seed 2-레짐→3B/7B)은 06-15~06-26 완료·커밋(runs/track_d/rundirs 18셀) — '미완'은 루트 CLAUDE.md 06-13 문구 승계 오류; 결정 3건(bridge arm=I-20/FlowerTune 채점=I-24/ShapleyFL β=I-50) 대기 | Yonghee 검토 대기(TRACK_D_REVIEW; 루트 CLAUDE.md next) | [정정] 2차-①·② 축 결과는 산출됨(track_d rundir arms + ov §3.2.1 clean-IID do-no-harm parity·§3.3.1 rounds-to-target) — 잔여는 결정 3건 반영(bridge arm 등)뿐 | 1B ≈40–60 GPU-h(루트 CLAUDE.md); 신호크기 probe 판정 후 착수가 합리적(I-03/I-24 연동) | P3-3 |

### 2.4 P4 — 2차-③ 탐지·위협모델 (I-27 ~ I-41)

| ID | 항목 | 간소화·미룸 내용 | 당시 사유 (출처) | 흔들릴 수 있는 결론 | 엄밀화 방법·예상 비용 | 우선순위 |
|---|---|---|---|---|---|---|
| I-27 | delta/advanced-delta free-rider 미구현 — **자인 위험** | free-rider는 zero/random 2모드만; Lin 2019의 delta/advanced-delta(직전 aggregate 재활용±noise)는 Phase1→Phase2→task9로 2회 이월 후 미구현(docstring 명문: `data/corruptors.py:122-125`; phase2_matrix threat 목록에도 없음; raw 2026-06-04-phase1-corruptor-and-7-design.md:23; raw crossdevice:105-106); STD-DAGMM 원류(Lin) 시나리오 head-to-head도 동반 미실행 | prev-aggregate를 FL loop에 스레딩하는 코드 필요 + "RICHEST comparison"이라 본선으로 미룸(plan:374) | **"free-rider φ exact-0" 헤드라인**(루트 CLAUDE.md·plan:19)이 쉬운 2모드에만 성립; raw 자인 "genuinely AT RISK — recycled delta는 ∇ℓ_val과 정렬 → 1차항≠0"(raw crossdevice:106). 일반화 서술 시 즉시 공격 지점 | 루프에 prev-aggregate 전달(소규모 코드)+silo5/device100 각 1 threat-row [추정 수~수십 GPU-h]; 미실행 시 위협모델을 zero/random으로 명시 한정=비용 0(단 자인 위험이라 실험 강권) | P4-1 |
| I-28 | PGD/direction-aligned poison(#13) 미실행 — 2차항 차별화 flagship | "1차로 못 잡는 direction-aligned poison을 2차가 잡는가" = 2차항>1차항 robustness 직접 증명 flagship-candidate(flirds:158) — Phase 3 잠금 후 미실행 | Phase 3 배정 + "only non-trivial compute" 자인(flirds:158,185; plan:163) | **2차항의 존재 가치** — 현 근거는 CNN 부분참여 방어(Flirds-1st k=0.2 붕괴 +0.305 vs 2차 +0.904; diag §3.4)와 poison seed-분산 사례뿐; FedIF(1st-order) 대비 "why isn't 1st-order enough?" 정면 실험 부재(flirds:254) | corruptor 구현(grad_noise 패턴 재사용; plan:415)+silo5 3-seed [추정 수십 GPU-h] | P4-2 |
| I-29 | poison orientation 판정 미결 + FLDetector off-threat 서술 미반영 | matrix "Flirds AUROC 0.0(회피)" vs D2b "1.0(분리)" = 부호 관례(−φ vs φ) 차이 — "Yonghee's call + real config·full val 확인 = verification-session item #1"(raw matrix-orchestrator:103-124; plan:343)이 **판정 기록 없이 잔존**(06-19 문서는 회피-프레이밍 사용; raw 2026-06-19-…:40); FLDetector 0.50 해석도 "non-IID erosion"→"off-threat" 재진단(raw crossdevice:71-75)이 루트 CLAUDE.md에 미반영 | 위계 lock(탐지=마지막, "valuation의 정직한 답"; flirds:20-27)이 프레이밍은 정리했으나 수치 재확정·명시 판정은 미이행 | poison 축 헤드라인 문장 자체가 갈림("detects all three" vs "evaded + matched detector 보완"); 두 문서 상충 상태로 논문 서술 시 사고 위험; detector 비교의 공정성(off-threat를 on-threat처럼) | 실험 거의 불요 — Yonghee 판정 1건 + D2b/상위 문서 정정 + val=200 재평가(I-07 post-hoc 기계와 동일, forward-only [추정 수 GPU-h]); detection 표를 threat-matched 쌍으로 재편 | P4-3 |
| I-30 | stealthy/norm-bound backdoor "불가" 판정 → 별도 연구 이월 | poison arm은 plain-scaled(γ=n/η) full-replacement만; constrain-and-scale arm은 "attacker ‖Δ‖=40× → 불가" 판정 후 제외(plan:337,339; raw 2026-06-09-phase2-task7e-backdoor-install-feddqc.md:52-54,86-87) | D2 실측(all-or-nothing). 단 자인: "we ran neither paper's setting … Bagdasaryan stabilizer 전부 생략"(log:497) — 우리 세팅 안에서의 불가 | poison AUROC 1.0이 40×-outlier(최저 난이도) 대비 과대 성능일 수 있음; "clean-preserving backdoor가 Flirds 회피" 판정도 이 공격형 조건부; 원 논문엔 norm-bound 하 유지 레시피 존재(log:497) → 판정 뒤집힐 여지 | faithful-repro 레시피로 설치 스모크부터 [추정 수 GPU-h + silo5 1셀]; 또는 한정 서술(불가능성 실측이 있어 방어 가능 — raw 판단은 비권장) | P4-4 |
| I-31 | single-token target·약한 trigger backdoor (Xu 원 설정 아님) | corruptor = Xu의 약한 변형("tq" trigger)+single-token target+text-exact-match(plan:337; log:497 — Xu 헤드라인 99%는 classification+3epoch+lr5e-5 조건) | 생성 태스크 각색 + 설치 창 확보(poison_frac 0.5–0.8; plan:337) | poisoning 결론의 위협 대표성 — 자체 제작 변형이라 외부 비교·재현 주장 불가; checkpoint 07 "poison 인위성" 등재(log:515) | multi-token·Induced-Instruction 변형 1종 [추정 수 GPU-h]; 또는 "우리-변형" 명시 서술(GTG 선례 관례; plan:402) | P4-5 |
| I-32 | noisy corruption = answer_swap 단일·MILD | 같은-도메인 유효 텍스트 셔플이라 MILD — noisy AUROC 0.75의 지목 원인(raw corruptor-and-7-design:143-145); FedDQC appendix 변형 메뉴 인지했으나 미스윕(raw 2026-06-12:59); LLM엔 severity sweep 없음(CNN엔 graded ladder 있음) | task9 corruptor extensions가 그리드 미포함 | "noisy 탐지 가능/불가" 결론이 강도-특이적; epistemics 방어 논리("sweep types/severities"; raw crossdevice:61)의 LLM 쪽 공백 | LLM noisy type/severity 1축 스윕(silo5 noisy 셀 비용 [추정 수 GPU-h/셀]) | P4-6 |
| I-33 | poison 설치·ASR 측정의 거칠기 | ① D2 합성이 수렴 가정(benign delta ~cancel) 위(`experiments/phase2_backdoor_d2_smoke.py:8-9`) ② greedy exact-match ASR은 부분 설치에 둔감 — soft ASR(NLL·first-token) 별도 존재(`eval/generate.py:62-67`), 최종 표 사용 지표 rundir 확인 필요 ⚠ ③ device100 ASR=0.75는 tiny val=4 확인치(raw matrix-orchestrator:151) — real-config 수치는 rundir에만 | matrix는 R=60 converged G로 대응(phase2_matrix.py:42-46) | poison 위협 강도 해석; ASR=0 판정이 부분-설치 놓칠 가능성; tiny-val 확인치의 잔존 | rundir에서 real-config ASR 확인(재실험 불요 가능성 높음) + soft-ASR 병기(offline 재계산이면 저비용) | P4-7 |
| I-34 | per_client 40→300 통일 | cross-device 기본 per_client를 poison 설치 임계(~200) 때문에 300으로 올려 전 threat 통일(plan:344, Yonghee 승인; noisy/FR는 "size-independent" 근거) | poison ASR=0 원인이 설치 데이터 부족(40→75)으로 판명(plan:344) | cross-device 외적 타당성 — 소량-데이터 클라(원래 40이 모사)에선 backdoor 불성립 자체가 발견인데 무대를 공격 성립하게 조정; "N=100 poison 탐지"는 "클라당 300+" 조건부 | 조건 명시 서술 = 0; 또는 per_client 2점(40/300) 대조 1셀 [추정 수 GPU-h] | P4-8 |
| I-35 | STD-DAGMM 하이퍼 = 임의 디폴트·1-seed untuned | per-dim standardize/n_gmm=2/latent=4/epochs=200 = "Claude's non-specified defaults, Yonghee did not veto"(raw 2026-06-08-phase2-task7e-detector-suite-steps1-3.md:28-31); real 0.63 AUROC 1-seed untuned(:42-43) | headline은 matrix라며 관리 | "gradient-사용 방법이 model-free를 압도" 비교가 언더튜닝 baseline 대비 — 비교 공정성 관례 위배 소지 | 소규모 하이퍼 스윕(CPU AE 85–110s/fit → 수 시간) 후 최고치로 표 갱신 | P4-9 |
| I-36 | FLDetector score-only 적응 | 원 논문의 클러스터링→제거→재시작 대신 연속 suspicious score만; 윈도 게이트 완화(iteration>N 대신 secant 1쌍부터)(`baselines/fldetector.py:33-41`) | AUROC 비교엔 연속 점수 필요 + 2-means는 N=5 퇴화(주석 논리 타당) | "cheapest but weakest" 판정이 변형 점수 기준; threat mismatch도 주석 자인(:28-31) | "score-only adaptation" 논문 명기 = 비용 0; N=10/100 headline은 grid 기존 | P4-10 |
| I-37 | FedDQC — matched-arm 미실시 + IRA 도메인 confound 미확인 | 전체 파이프라인(IRA+hierarchical training) 비교가 아닌 IRA 점수만 이식(plan:37); D-옵1 제외(plan:436); "per-domain IRA 분산 큼(finance 0.17≈noisy 0.067) → matrix에서 noisy 도메인 변주 확인" 이행 불확실 ⚠(raw backdoor-install-feddqc:73-74) | 06-02 비교 스코프 통제; D-옵1 제외는 IID-clean 재정의와 충돌 | "유일 FL+LLM 선행과 직접 비교" 서사가 점수-수준으로 축소; FedDQC noisy AUROC=1.0이 "noisy=medical" 배치 특이적일 가능성 | rundir config 확인 → 미변주면 noisy 도메인 변주 2–3셀(저비용); §2.4 noisy 칸에 FedDQC arm 1개 [추정 수 GPU-h]; 또는 비교 범위 명시=0 | P4-11 |
| I-38 | maverick(#15)·duplicate(#14) corruptor 이월 | partition-level corruptor 2종 미구현·미실행(plan:366,374; flirds:159-160); #14는 "no participation normalization" lock의 검증 실험, #15는 OOD-good 한계의 sharp form | "실제 쓰는 시점에 확장"(plan:372; 무추상화 원칙) | participation-normalization 없음 lock의 공정성 무실증(#14=ComFedSV 시그니처 테스트); Mavericks 논문의 rare-domain 과소평가 증명(flirds:160)에 대한 우리-무대 정량 부재 | partition 헬퍼 2개+device100 1–2셀 [추정 구현 반나절+수 GPU-h] | P4-12 |
| I-39 | 참여 이질성(빈도↔품질 분리) 실험 미실행 | 참여 정규화 없음 결정 시 설계된 검증(80%/10% 참여 tier 내 ranking이 quality를 따르는지; raw conversation3.md:17) — device100은 uniform K=10뿐, 실행 기록 없음 [추측] | 미기록; 06-03 smoke에서 uneven tier가 AUROC 흐리는 현상 우연 관측(raw 2026-06-03-phase1-backend-abstraction.md:79-81) | "참여 빈도와 무관하게 quality 분리" 주장 불가 상태 — cross-device 일반화 한계 | device100 1셀에 참여율 tier 추가(기존 셀 비용 수준 [추정 수 GPU-h]) | P4-13 |
| I-40 | label_flip 재라벨 K−1 균등 vs FedCorr 전체-K | CNN label_flip이 정답-제외 K−1 균등(`data/corruptors.py:24-33`) — FedCorr convention과 유효 오염률 (K−1)/K 차이, caveat 한 줄 처리(plan:396,418) | Yonghee 결정: rate=실제 오염률·GT 명확(plan:396) | FedCorr (ρ,τ) 숫자와 직접 비교 시 5–10% 정의 차; 자체 결과 내부 무영향 — 경미 | 논문 각주 = 비용 0; 원하면 전체-K 모드 fork 기존 검토(plan:419) + 대표 1–2셀 [추정 <10 GPU-h] | P4-14 |
| I-41 | GTG 5-시나리오 재현의 자체-선택 | GTG 원문 스펙 부재 구간(flip 메커니즘·noise 정의·arch·optimizer)을 우리 선택으로 충전: K−1 flip, pixel-σ, ladder 베이스=same-size IID, quantity-skew=disjoint 정규화(plan:402-406) | 원문 미명시 — "선택 무제약, 논문 명시 서술"(plan:402); 교락 제거는 방법론적 우월 | "GTG 5-시나리오 재현" 라벨의 엄밀성 — 재현이 아닌 우리-변형; 숫자-옆-숫자 비교 불가(원문 figure-only 기지; plan:54) | 서술 정확화("GTG-스타일, 스펙 부재 구간 명시 선택") = 비용 0; literal fork 옵션 기록 존재(plan:403) | P4-15 |

### 2.5 P4~P5 — 특성화·이론 의무 잔여 (I-42 ~ I-47)

| ID | 항목 | 간소화·미룸 내용 | 당시 사유 (출처) | 흔들릴 수 있는 결론 | 엄밀화 방법·예상 비용 | 우선순위 |
|---|---|---|---|---|---|---|
| I-42 | ② cancellation per-layer 특성화 미실행 | lock 약속 "per-layer 분해로 마지막-레이어 부호 요동 도시 + 제외 진단 ≈ 공짜"(raw 2026-05-19:79,392); seam(`per_layer=`)은 구현·기본 off, 실사용은 smoke invariant뿐(raw 2026-06-03-…:57-59; grep: phase1_llm_smoke만) | Phase 3 배정; seam을 미리 판 이유가 정확히 이 실험("retrofit costly"; log:366) | "②를 특성화된 한계로 정직 보고" 전략의 증거 0건 — 리뷰어 질문 시 pilot의 set-aside 수치뿐 | 기존/신규 run 1개에 per_layer=True + post-hoc 분석 — 추가 학습 불요, near-free(flirds:162) | P4-16 |
| I-43 | ③ late-joiner(magnitude/alignment) 특성화 미실행 | lock 약속 "late-joiner test ≈ 비용 0"(raw 2026-05-19:94,393; 원 설계 conversation3.md:18) → D7에서 "구조 유지, 텍스트 정당화"로 축소(raw 2026-06-02-phase0-implementation.md:19); 코드 grep 0건 | 축소 결정만 기록 | ③(합류 시점-가치 혼동)의 크기 정량화 약속이 텍스트로 대체; std20(부분참여)에선 효과 실재 가능 — 미측정 | std20 로그로 "첫 참여 round vs φ" 산점 post-hoc(재학습 0) + 필요시 late-join 강제 1셀 [추정 수 GPU-h] | P4-17 |
| I-44 | Q2 변형 비교(layer-wise/phase-norm) 9-cell 조용히 소멸 | N4 lock의 {default, layer-wise, phase-normalized}×3지표 매트릭스(raw 2026-05-19:229-231,390) — 구현 0건, 이후 어느 raw에도 재등장 없음 | 미기록(무언 취소) | locked Section-3 item과 실험 세트의 정합성; IID-clean 재설계로 무의미해졌다면 그 판단이 미기록 | 명시적 재결정(drop 사유 기록) = 비용 0; 또는 1B 소규모 9-cell [추정 수십 GPU-h] | P5-1 |
| I-45 | noise-vs-OOD-good 보류의 부대 손실 | ① 보류 자체는 characterized-limitation 정식 결정(flirds:79,109)이나, conv3의 제2 기여 후보 일체(temporal-consistency·cross-client-agreement 신호, 3-그룹 분리 실험, OOD-good 제외 control; raw conversation3.md:49-66,161; conversation4.md:52-56)가 동반 소멸; 파생 의무(α-sweep)는 이행됐으나 기준점이 proxy(I-05)라 bias 정량화론 약함 | 전용 스레드의 설계 결정(2026-05-18) | "val-loss 반대 방향=나쁜 데이터" 동일시가 특성화 없이 서술에 스며들 위험(위계 원칙과 표리) | OOD-good 클라 1종 포함 분리 실험 1셀 [추정 중간 비용]; 또는 한계 명문화만 = 0 | P5-2 |
| I-46 | #17 qualitative case study·#18 clean skyline 미실행 | #17(domain-val→클라 credit; per-domain val 256 설계까지 존재; plan:256) 미실행; #18(탐지 상한 clean 통제)은 §2.4 silo5-clean 칸이 흡수 예정(diag:272) | Phase 3 post-hoc 배정(flirds:161-163) | #17=설득 축(약함); #18 없으면 detection 표 상한 참조점 부재 | #17 = 로그 φ post-hoc GPU 0(flirds:162); #18 = I-03 실행에 포함(추가 비용 0) | P5-3 |
| I-47 | phase05 Taylor-잔차 게이트의 fp32 노이즈 플로어 | lr→0에서 (b) loss-diff가 fp32 상쇄 소거로 relL2 플라토(~3e-3); 2차 이점은 moderate lr에서만 관측(`experiments/phase05_sanity.py:4-9`) | "broad trend는 regime sweep에" 위임 | "2차항이 정확도 개선"의 실증이 측정한계와 얽힘 — 항목 1(수학)·항목 3(fp32)과 같은 뿌리 | CNN 소규모 fp64 재게이트(수십 분 [추정]) — O(lr³) 기울기 회복 확인, 논문 부록감 | P5-4 |

### 2.6 P5 — 타 과업 이관·경미·파킹 (I-48 ~ I-54)

| ID | 항목 | 간소화·미룸 내용 | 당시 사유 (출처) | 흔들릴 수 있는 결론 | 엄밀화 방법·예상 비용 | 우선순위 |
|---|---|---|---|---|---|---|
| I-48 | fp32/bf16 정밀도 일관성 → **항목 3 이관** | (a)-val-loss 검증 fp32 강제 규약(bf16 ulp ~9e-3 > coalition diff ~5e-3–2e-2; raw task6:66-73,107-109)의 전 run 일관 적용 여부; proto §1 "학습 bf16" ↔ 실구현 전부 fp32 문서-코드 불일치 | 정밀도 함정은 06-07 발견·규약화 | (a)/(b) 일치 수치의 재현 조건; oracle cost의 fp32-패널티(×3.1) 명시 여부 | 항목 3(정밀도 감사) 스코프 — dtype 전수 감사 + 옵션 비교 문서 | P5-5 |
| I-49 | cost 회계(exact형 baseline·회계 불일치) → **항목 6 이관** | ShapleyFL DMC·Banzhaf MSR 추정기 미구현(`baselines/shapleyfl.py:21-23`; `banzhaf.py:17`) — 비교가 exact 2^N/2^K형으로 수행(fidelity엔 보수적, cost엔 왜곡); Ripple만 자체-궤적 포함 회계(I-13); 로그 생성 시간 미측정 | N=5/anchor에서 exact가 faithful + 근사 confound 제거(강화 평가로 논문 명시 방침; plan:416) | "Flirds 5–15× cheaper"의 분모가 원 방법의 실전 비용과 다름; wall-clock 표의 방법 간 비교 공정성 | 항목 6 스코프 — [실측 대기 — 항목 6: 로그 생성 시간·valuation-only 통일 회계·방법별 비용 프로파일]; 표 각주("exact형 실측") = 비용 0 | P5-6 |
| I-50 | ShapleyFL β0.3 재실행 잔여 | 우리 β=0.5 vs 논문 β=0.3 발견(log:526) → 재실행 캠페인 진행 중; 7B track_d·phase2_matrix 반영 대기(ov:789 Caveat 9)[07-07 PAUSED(서버 이전): 7B×6+phase2×25=31셀 — runs/rerun_beta03/RESUME_AFTER_MIGRATION.md] | 원문 대조에서 발견된 수치 오류 교정(진행 중) | β 미통일 상태의 7B·robustness 표 수치는 곧 바뀔 값 | 캠페인 완료 대기(재집계는 rundir-only) | P5-7 |
| I-51 | P6 fairness·reward 축 미설계 | 전용 실험 없음(ov:106,116); ComFedSV류 fairness-ECDF 지표 없음 | Q1 lock: use case=research-side attribution, incentive=secondary(flirds:69) — 정합적 미설계 | 논문이 incentive 서사를 쓰지 않는 한 없음 | 스코프 밖 유지 권장(비용 0) | P5-8 |
| I-52 | #7 본런 ORACLE_B 기본 OFF | (b)가 지배 비용이라 #7 full run은 estimator-only(est≈oracle은 R=10 smoke 검증; raw corruptor-and-7-design:130-134) | 비용 | 이후 그리드 silo5에 (b) exact 포함 → 대체 검증 존재, 잔여 위험 낮음 | 불요(기록만) | P5-9 |
| I-53 | 서지·명명·선언 잔여 | FedSV 이름충돌(우리=Wang2020 ≠ 2502.17526 Byzantine-FedSV); baseline sight-unseen 재확인 3건(DPVS-Shapley·FedAttr·KFCA); `findings.md`/`MANIFEST.md` 빈 템플릿(감사 근거로 사용 금지); seed schedule [42,123,2024](proto:35) vs 실측 0/1/2 편차 선언 (출처: memory `baseline-selection-audit-loo-gap.md:16`; proto:35) | 각 절차성 잔여 | 논문 표기 실수·baseline 완결성 잔여 | 서술·확인만 = 비용 0 | P5-10 |
| I-54 | filter q-sweep 파킹 | 개입 filter의 q-sweep — Yonghee "나중에 sweep 추가 실험 고려해보자, 기억해놔줘"(raw 2026-06-13:20-21,114) | 오염 실험용으로 파킹 | 없음(파킹 항목의 망각 방지 등재) | 오염 트랙 재개 시 sweep 1축 [추정 수 GPU-h] | P5-11 |

### 2.7 참고·제외 항목 (메인 표 비산입 — 스윕 노트 판정 승계)

**설계 종결(보류 아님, 결론 위험 낮음)** — raw F표 잔여 7건: ghost technique(FL 부적합 판정; conversation1.md:13 등), 참여 횟수 정규화 안 함(Shapley-집계축 lock; conversation3.md:3), 13B+/70B 제외·무언급 lock(2026-05-27:176-184), SPACE/S-FedAvg/FedCorr/Power-of-choice 제외(2026-05-27:101-119), DBA 제외(다중 공모≠단일 corrupt; crossdevice:113-114), GGN 기각(데이터 기반 종결; 2026-06-03-phase05-estimator.md:48-49), removal curve 제외(2026-06-13:87).

**순수 엔지니어링 판정** — 코드 노트 B표 9건(BD_TARGET 스모크 전용, skip_special_tokens, std_dagmm eps, intervene fallback, `_OUT=/tmp`, `_chunked` 정확성(fp32 합산 순서는 항목 3 소관), eval_every skip, ComFedSV ALS 자체구현(관측 로깅은 I-12로 커버), 방향성 노트) + 위키 노트 F의 엔지니어링 4건(driver race·golden env·`_guard` 수정 완료·merge_oracle_a 버그 수정 완료).

**정정으로 종결** — clean-oracle/random-q% bracket arm: 누락이 아니라 의도적 제외(Yonghee 2026-07-02 정정; memory `baseline-selection-audit-loo-gap.md:16`).

---

## 3. 해소 경로 분류

한 항목이 두 경로를 가질 수 있음(실험이 정공법, 서술이 차선). 표기: ID(방법 요약).

### (A) 실험 필요 — 비용 오름차순

**A-0. GPU 0 (post-hoc·CPU·rundir 확인만)**
- I-15 bootstrap CI 툴링(make_analysis 확장) · I-42 per-layer ② post-hoc(near-free) · I-43 late-joiner 산점(std20 로그) · I-46 #17 qualitative(로그 φ) · I-25 분할 균등성 확인 · I-22 val 절단율 측정 · I-33 real-config ASR rundir 확인+soft-ASR 병기 · I-37 noisy 도메인 변주 rundir 확인 · I-12 관측율 로깅 추가+재계산

**A-1. 시간 단위 (≤수 GPU-h)**
- I-11 trunc=0 대조 1셀 · I-47 CNN fp64 재게이트(수십 분) · I-35 STD-DAGMM 스윕(CPU 수 시간) · I-12 ComFedSV verify 3–5seed(CNN 수 시간) · I-07 probe C 완주(≈4 GPU-h, 가동 중) · I-09 CNN Fed-LOO 셀(시간 단위/셀)

**A-2. 수 GPU-h ~ 수십 GPU-h**
- I-07 #16 post-hoc 재평가(forward-only, 셀당 수 GPU-h) · I-16 normalization 재계산 · I-29 val=200 재평가 · I-21 poison lr 2점 교차 · I-31 multi-token backdoor 1종 · I-32 noisy severity 1축 · I-34 per_client 대조 1셀 · I-38 #14/#15 셀 1–2개 · I-39 참여 tier 1셀 · I-23 FedProx 1셀 · I-20 bridge arm 1-seed · I-13 Ripple 재실측 3-seed(~4h; 항목 2 결과 따라) · I-27 **delta free-rider**(코드 소규모+2셀) · I-54 filter q-sweep

**A-3. 하루 급 (10~25 GPU-h; 비용 실측 근거 있음)**
- I-02 **(b) N=10 exact 3-seed ≈10h**(proto:77) · I-05 **(b)-perround α=0 anchor ~11h/4-GPU/셀**(plan:19) · I-06 **3B (a) 3-seed 재실행+rundir**(~2.6h/run fp32×3)

**A-4. 수십 GPU-h 이상**
- I-03 §2.4 clean×non-IID 매트릭스 6셀×3seed(스모크 green) · I-10 probe 잔여 ≈40 GPU-h(+r128 [추정 +30]) · I-14 3B 4셀×2seed · I-28 PGD poison silo5 3-seed · I-04 E-sweep 최소판(십수 셀) · I-26 Track D real run(1B 40–60 GPU-h) · I-09 LLM Fed-LOO(트랙 재실행)

**A-5. 대형·설계 필요 (스코프 결정 후)**
- I-02 (a) N=10 샤딩(~11–22h + 오케스트레이션 코드 신규) · I-14 7B robustness([실측 대기 — 셀당 비용 추산]) · I-30 stealthy 재시도(비권장이 기존 판단) · I-03 FedHDS 무대 설계 세션

### (B) 판정·서술만으로 해소 가능

- **판정 필요(Yonghee)**: I-29 poison orientation(최고 긴급 — 실험 불요·문서 상충 해소) · I-44 Q2 variants drop 명시 재결정 · I-10 rank lock 개정 여부
- **서술·각주(비용 0)**: I-01 헤드라인을 부분참여 fidelity로 승격 · I-05 proxy-truth 셀 명시 분리 · I-17 proto MC 조항 갱신 · I-18 거리·Pearson 열 마스킹 · I-19 표본 정의 각주 · I-34 per_client 조건 명시 · I-36 score-only 명기 · I-40 label_flip 각주 · I-41 GTG "스타일" 정확화 · I-49 exact형 실측 각주 · I-52 기록만 · I-53 서지·명명·seed 선언 · I-07/I-14 프로토콜 편차 일괄 선언(proto:13 조항 이행)

### (C) 논문 limitation 명기로 충분

- I-06 7B (a) 생략(준비된 선례 문구; proto §4.1) — 단 ov P3와 지위 통일 선행
- I-20 Adam 갭(bridge arm 미실행 시; FedIT-SGD 선례 인용) · I-30 stealthy(불가능성 실측 근거로 한정 서술) · I-31/I-32 위협 변형의 자체-제작 명시 · I-08 (b)-대리 적용 한계 · I-22 seq512 caveat · I-45 noise-vs-OOD 한계 명문화 · I-47 fp32 측정한계 부록 언급 · I-51 fairness 스코프 밖 선언

---

## 4. 상위 10개 항목 상세

### 1. I-01 — N=5 근사-가산 무대: "+1.000" 헤드라인의 증거 지위
**왜 위험한가**: 프로젝트의 대표 수치(전 방법 Spearman +1.000 vs oracle, Flirds 5–15× cheaper 프런티어)가 N=5 anchor에서 나왔는데, 07-02 진단이 이 무대의 coalition 게임이 사실상 가산적(갭 ≤0.9%)이고 "모든 제대로 계산된 semivalue가 같은 순위로 붕괴"함을 실측했다(diag §1.5:170–185). 즉 +1.000은 방법의 우수성이 아니라 게임 퇴화의 산물일 수 있고, frontier 주장의 fidelity 축은 무정보가 된다. 이대로 논문에 실리면 리뷰어가 진단 문서 없이도 "왜 전부 만점인가"를 물을 것이고, 그 답이 우리 진단과 같다면 헤드라인이 무너진다.
**무엇을 하면 되는가**: (i) 비용 0 — 이미 실측된 부분참여 fidelity(1B std20: Flirds +1.000 vs ComFedSV +0.093/ShapleyFL +0.19/FedIF +0.16; log:545)를 차별화 헤드라인으로 승격하고, N=5 anchor 수치는 "포화 무대의 일치 검증"으로 재프레이밍. (ii) I-02(N=10 oracle)와 가동 중인 std50k5 참여 probe가 실험적 해소책 — 신규 설계 불요.

### 2. I-02 — LLM N=10 (a)/(b) oracle: fidelity 고-power 검증의 부재
**왜 위험한가**: proto §9는 cross-silo N=10을 primary로 잠갔지만(proto:141-148) 실측 primary는 전부 N=5다. N=5는 가능 순위가 120개뿐이라 Spearman 분해능이 낮고(I-01), 근사 오차가 처음 보일 수 있는 N=10 검증이 비어 있다. "N=10 CNN + N=5 LLM만으론 unconvincing"은 Yonghee가 06-04에 직접 제기한 우려다(raw 2026-06-04-phase1-data-layer.md:21). 예고했던 "headline detection N=10"(plan:357)도 부재해 서술 일관성 문제까지 겹친다.
**무엇을 하면 되는가**: 먼저 (b) N=10 exact — 같은 게임·저비용(1B ~3.5h×3seed ≈10h; proto:77). 이 run 하나로 fidelity와 detection이 동시에 나온다. (a) N=10은 coalition-sharding 오케스트레이션 신규 작성 후 ~11–22h/4-GPU(plan:143) — 실행 여부는 스코프 결정 사안("Yonghee 결정 필요" 참조). 미실행 시 논문 스코프를 "N=5 dual-GT 검증 + 대규모는 (b)-fidelity"로 명시 한정.

### 3. I-03 — IID-clean 신호 부재와 clean×non-IID 빈칸
**왜 위험한가**: Track D의 fidelity·개입 결과가 "(b) oracle 자기 순위조차 cross-seed ρ≈0"인 무대에서 나왔다(diag:110-113). 순위의 대상 자체가 추첨 노이즈라 "IID에서도 정확히 순위를 잰다"는 읽기는 과대해석이고, do-no-harm parity도 정답이 parity인 무대의 자연 결과다. 더 깊은 문제: "non-IID면 신호가 산다(silo5 ρ +0.93–1.00)"는 관찰이 오염 신호와 미분리 상태 — clean×non-IID 칸이 빈 한, "도메인 이질성만으로 신호 실재"는 미확정이며 fidelity 결과의 원인 해석(오염산 vs 비IID산)이 전부 조건부다.
**무엇을 하면 되는가**: §2.4 분리 매트릭스(신설 6셀×3seed, silo5급) — 스모크 6/6 green이라 착수 장벽이 낮다(diag:286,313). LLM 쪽 clean×non-IID 정식 무대(FedHDS 후보)는 별도 설계 세션(Yonghee 06-13 지시 잔존).

### 4. I-04 — Proposition 1/2의 실증 공백
**왜 위험한가**: 논문의 수학 기여(Prop 1 drift-residual 분해·Prop 2 cubic bound)가 pilot에서 U-shape로 falsified된 후 canonical 재검증 없이 남아 있다. 05-22에 "established"로 서술됐다가 Yonghee 정정으로 전부 "planned"로 롤백된 목록(raw 2026-05-19:292-301)이 이후 재확증된 기록이 raw 25편 어디에도 없다. "drift residual은 measured, not corrected"(flirds:62) lock의 "measured"가 미이행 상태로, 이론-실증 결합의 핵심 고리가 끊겨 있다.
**무엇을 하면 되는가**: E∈{1,3,5,10}×α 2–3점×3seed 최소판(1B, 십수 셀)으로 residual 크기 vs cubic bound 기울기를 대조. U-shape 재현 시 N1 contingency(Drop+empirical 전환) 분기 실행. 검증 캠페인 항목 1(IRDS 수학)의 수치 실측과 같은 세션에서 처리하는 것이 중복을 없앤다 — 항목 1이 이론을, 본 항목이 그 이론의 sweep 실증을 맡는 분업.

### 5. I-05 — off-anchor proxy-truth 순환
**왜 위험한가**: α-sweep 12셀의 "Spearman +1.000"은 vs oracle이 아니라 vs Flirds다(코드 확정: phase2_matrix.py:370). Flirds 자신의 off-anchor fidelity는 원리적으로 측정되지 않았고(자기 대비 +1.0 자명), 검증점은 α=0.5 하나뿐이다. 극단 α(0, 0.01)는 drift-residual이 가장 커지는 지점 — 가정이 가장 약한 곳에서 가정에 기대고 있다. checkpoint 07도 이를 한계로 등재했다(log:515).
**무엇을 하면 되는가**: α=0(domain-disjoint) 1셀에 (b)-perround anchor 추가(~11h/4-GPU; plan:19) — 그 1점이 proxy 신뢰 구간을 양 끝에서 고정한다. 비용 0 조치로는 표에서 proxy-truth 셀을 oracle-fidelity 셀과 시각적으로 분리(각주 아닌 열 분리 권장). #8 stress의 (a)/(b) 병행 보고(I-08)와 같은 run을 공유할 수 있다.

### 6. I-06 — 스케일 방향 (a)-oracle: 3B +0.900의 지위
**왜 위험한가**: "듀얼 오라클로 방법을 검증했다"가 rundir 수준에서는 1B N=5 한 셀에 얹혀 있다. 3B (a)vs(b)=+0.900은 1-run이고 "retrain noise" 해석은 무검증이며(raw crossdevice:46-49), 파일-only 원칙 때문에 결과 overview에도 못 실린다(ov:782). 1B +1.000 → 3B +0.900이 스케일 괴리의 앞머리라면 "estimator≈oracle" 일반화 주장 범위가 직접 걸린다. 부수적으로 proto §4.1(7B ❌)과 ov P3("채울 행")의 문서 간 지위 불일치도 방치돼 있다.
**무엇을 하면 되는가**: 3B (a) N=5를 3-seed로 재실행해 rundir 영속화(fp32 ~2.6h/run — 반나절~1일; 커맨드는 ov:807에 준비됨). swap된 clean-pair의 φ 격차가 retrain 분산 이내인지 CI로 보고하면 "+0.900=noise" 해석이 검증되거나 기각된다. 7B는 준비된 선례 문구로 한정 서술하되 ov P3 행의 지위를 proto와 통일.

### 7. I-07 — val 크기 괴리와 #16
**왜 위험한가**: lock은 "few-shot(~50) val은 noisy해서 기각"이었는데(plan:252) 실측 robustness 그리드는 val=10–20으로 돌았다 — lock이 스스로 기각한 영역이다. 25셀의 AUROC/Spearman 전부가 이 val 위에 있고, proto:13("모든 편차는 선언")의 silent-deviation 상태다. near-tie 무대에서 φ 차이가 val-표본 노이즈 이내라면 fidelity 표 해석 자체가 바뀐다(raw conversation1.md:83의 원 우려).
**무엇을 하면 되는가**: 3단 — (i) probe C(per-chunk val bootstrap SE, ≈4 GPU-h) 완주 대기: φ 신호가 val 노이즈 위인지 판정. (ii) #16 post-hoc 재평가(설계상 re-eval only; flirds:161): 로그 궤적 위에서 val만 바꿔 (b)·estimator 재계산, 재학습 0. (iii) 편차 일괄 선언 또는 proto 개정(Yonghee 소관). device100 val=4↔10 최종값의 rundir 확인도 선행.

### 8. I-27 — delta free-rider: 자인된 위험의 방치
**왜 위험한가**: raw가 "여기가 Flirds가 **진짜 위험한**(genuinely AT RISK) 지점 — recycled-aggregate delta는 ∇ℓ_val과 정렬돼 1차항≠0 → 속을 수 있음"이라고 스스로 명시한(raw crossdevice:106) 유일한 항목이 2회 이월 끝에 미구현으로 남아 있다(corruptors.py:122-125 docstring). 현재 "free-rider φ exact-0" 헤드라인은 가장 쉬운 zero/random 모드의 산물이다. 리뷰어가 위키를 못 봐도 Lin 2019 분류를 아는 사람이면 같은 질문을 한다.
**무엇을 하면 되는가**: FL 루프에 직전 aggregate 스레딩(소규모 코드) + silo5/device100 각 1 threat-row(수~수십 GPU-h). φ≠0이 나오면 그것이 오답인지, "정렬된 진짜 신호에 대한 정직한 답"인지(위계 원칙의 적용) 서술 방향까지 정리. 실험을 안 하는 선택지는 위협모델을 zero/random으로 명시 한정하는 것이나, 자인 기록이 있는 만큼 논문 전 실행을 강권.

### 9. I-28 — PGD/direction-aligned poison: 2차항 flagship의 공백
**왜 위험한가**: "2차항>1차항"을 robustness 축에서 직접 증명할 flagship-candidate로 스스로 지정한 실험(#13; flirds:158)이 미실행이다. 현재 2차항의 존재 근거는 CNN 부분참여 방어(Flirds-1st k=0.2 붕괴 vs 2차 +0.904; diag §3.4)와 poison seed-분산 사례뿐 — FedIF(1st-order) baseline을 "why isn't 1st-order enough?" 용으로 들여놓고(flirds:254) 정면 실험이 없다. 2차항이 Flirds의 방법적 정체성인 만큼, 이 공백은 novelty 방어선의 공백이다.
**무엇을 하면 되는가**: update-level corruptor(기존 grad_noise 패턴 재사용 가능; plan:415) 구현 + silo5 규모 3-seed(수십 GPU-h — "only non-trivial compute" 자인 그대로). 1차(Flirds-1st·FedIF)가 놓치고 2차가 잡는 셀 하나가 나오면 2차항 서사가 완성되고, 안 나오면 그것대로 스코프 정리에 필수 정보다.

### 10. I-29 — poison orientation 판정: 실험 없이 해소 가능한 최고-긴급
**왜 위험한가**: 같은 결과를 두고 D2b 증류본은 "evades REFUTED(AUROC 1.0)", matrix는 "EVADED(AUROC 0.0)" — 순전히 부호 관례(−φ vs φ) 차이인데, "Yonghee's call + real config·full val 확인 = verification-session item #1"(raw matrix-orchestrator:103-124)이라는 꼬리표가 달린 채 명시 판정 기록이 없다. 06-19 문서는 회피-프레이밍을 쓰고 있어(raw 2026-06-19:40) 문서 간 상충이 잔존한다. 논문 서술 단계에서 두 문서를 각각 참조하면 정반대 문장이 나온다. FLDetector 0.50의 "non-IID erosion"→"off-threat" 재진단 미반영(루트 CLAUDE.md baseline 문단)도 같은 계열 — off-threat 비교를 on-threat처럼 쓰면 detector 비교 공정성이 공격받는다.
**무엇을 하면 되는가**: 실험 거의 불요. (i) Yonghee 판정 1건(어느 orientation이 §3.9 스토리인가 — 위계 원칙상 "clean-val-loss 낮추는 공격자는 φ 높음이 정직한 답" 프레이밍과 정합적으로). (ii) D2b 증류본·루트 CLAUDE.md 정정. (iii) val=200 재평가(I-07의 post-hoc 기계 공유, forward-only 수 GPU-h)로 tiny-val caveat 해소. (iv) detection 표를 threat-matched 쌍(FedDQC↔noisy 등)으로 재편.

---

## Yonghee 결정 필요

1. **I-29 poison orientation 판정** (실험 불요·최고 긴급): matrix(φ 채점, "evaded") vs D2b(−φ 채점, "detects") 중 §3.9 헤드라인 프레이밍 확정 + 상충 문서 정정 승인.
2. **I-02 (a) N=10 실행 여부**: 샤딩 오케스트레이션 신규 작성 + ~11–22h/4-GPU를 쓸지, "N=5 dual-GT + 대규모 (b)-fidelity"로 스코프 한정할지. ((b) N=10 ≈10h는 비용·효과상 실행 권고 — 승인만 필요.)
3. **I-06 7B (a) 지위 통일**: proto §4.1 "infeasible ❌" lock과 ov P3 "채울 행" 중 어느 쪽으로 문서를 통일할지.
4. **I-20/I-24 Track D 기존 대기 결정**: bridge arm(vanilla-AdamW 1회) 실행 여부, FlowerTune-채점 모드 여부. (ShapleyFL β는 β0.3 캠페인으로 사실상 결정 — 잔여는 7B·phase2 반영뿐.)
5. **I-10 rank lock 처리**: r128을 추가 실행([추정 +30 GPU-h])할지, lock을 {16,32,64}로 개정 선언할지.
6. **I-44 Q2 variants(N4 lock) 처리**: 무언 소멸 상태를 명시적 drop(사유 기록)으로 종결할지, 소규모 9-cell을 실행할지.
7. **I-07 val 편차 처리**: 논문 텍스트 일괄 선언으로 갈지, proto §3.4 자체를 개정할지(프로토콜 개정은 Yonghee 소관 사안).
8. **I-27/I-28 실행 승인**: 자인 위험 2건(delta free-rider·PGD poison)을 논문 전 필수로 볼지 — 본 문서는 실행 강권 입장(합계 [추정 수십 GPU-h]).

(사소한 결정으로 본 문서가 자체 처리한 것: 항목 분류·우선순위 부여 기준(§1.2), raw F표의 filter q-sweep 승격과 나머지 참고 강등(§1.3 — Yonghee "기억해놔줘" 발언 유무 기준), 타 과업 이관 4건의 교차 등재 방식(§1.1-3). 근거는 각 절에 기재.)

## 후속 실험 제안

우선 실행 패키지(의존성·비용 기준 묶음; 비용 수치는 위키·raw 실측 기반, [추정] 표기는 스윕 노트 승계):

- **패키지 0 — GPU 0, 즉시 가능**: CI 툴링(I-15), per-layer ② post-hoc(I-42), late-joiner 산점(I-43), soft-ASR 병기·real-config ASR rundir 확인(I-33), 분할 균등성(I-25), val 절단율(I-22), noisy 도메인 변주 rundir 확인(I-37), ComFedSV 관측율 로깅(I-12), #17(I-46). 서술·판정 일괄(§3-B)도 여기에.
- **패키지 1 — 하루 급 oracle 회수 (합계 ~25–35h GPU)**: (b) N=10 exact 3-seed(≈10h; I-02) + 3B (a) 3-seed 재실행·rundir 영속화(≈8h; I-06) + α=0 (b)-perround anchor 1셀(~11h/4-GPU; I-05, I-08 공유). — P1 최상위 4건을 실측으로 동시 해소하는 최고 효율 묶음.
- **패키지 2 — 자인 위험 회수 ([추정 수십 GPU-h])**: delta free-rider 2셀(I-27) + PGD poison silo5 3-seed(I-28). 코드 작업 각 소규모~반나절 선행.
- **패키지 3 — 진행 중 완주 (신규 비용 0)**: probe 스위트 seeds1–2(≈40 GPU-h 기계획; I-10)[잔여], probe C(I-07)[07-07 완료], CNN width×참여 grid 제출(I-10)[07-03 3-seed 완주·커밋 d2e7ed6 — 작성 전], β0.3 재집계(I-50)[07-07 PAUSED — 31셀, RESUME_AFTER_MIGRATION.md].
- **패키지 4 — 중기**: §2.4 clean×non-IID 매트릭스 본실행(I-03)[07-07 완료 — diag §3.3], E-sweep 최소판(I-04; 항목 1 세션과 병합), Track D real run(I-26)[완료 — 06-15~06-26 rundir 18셀; 결정 3건만 잔여], 3B seed 완성(I-14).
- **저비용 낱개**: trunc=0 대조(I-11), fp64 재게이트(I-47), STD-DAGMM 스윕(I-35), ComFedSV verify 다시드(I-12), #16 post-hoc(I-07), normalization 재계산(I-16), poison lr 교차(I-21).

[실측 대기 — 항목 2: Ripple valuation-only 환산 수치·eigsh 진단] / [실측 대기 — 항목 6: 로그 생성(학습) 시간·통일 회계 기준·방법별 비용 프로파일] / [실측 대기 — 7B silo5 셀당 wall-clock 추산(I-14의 7B 실행/한정 결정 재료)].

---

## 검증 처리 로그 (2026-07-04 사실검증 캠페인)

외부 사실검증 이슈 4건 원자료 대조 후 처리. critical·major 0건, minor 4건(수용 3 / 기각 1).

| # | severity | 위치 | 판정 | 처리 근거 |
|---|---|---|---|---|
| 1 | minor | §소스 요약 line 9 "참고·제외 17건" | **수용** | §2.7 프로세는 21항목을 명시 열거(raw-F 7 + 코드-B 9 + 위키-F 엔지니어링 4 + bracket 1). 17=7+9+1은 위키-F 엔지니어링 4건(driver race·golden env·`_guard`·merge_oracle_a)을 누락. line 9·§1.1#2 두 곳을 21로 정정하고 내역 병기. 소스 항목 실제 탈락은 없음(요약 숫자만 오류). |
| 2 | minor | I-02(line 46)·I-06(line 50) 출처 열 | **수용** | 코드 노트 A1(→I-02·I-06 매핑)의 1차-코드 앵커 2개가 병기 누락. `exact_sv_llm.py:21-22`("(a) retrain SV N=5만"; 재확인 [확인])→I-02, `phase2_matrix.py:34-35`("bf16=deferred (a) oracle, 7B는 미실행"; [확인])→I-06 7B-제외 주장에 추가. §1.1#1 "세 노트 출처 전부 병기" exemplar 규칙 준수 강화. |
| 3 | minor | I-05(line 49) "~25,000s/≈44h" | **수용** | 같은 양(device100 off-anchor 칸당 exact (b) 비용)의 두 인용이 ~6× 불일치: ov:526=~25,000s(≈7h), raw crossdevice:33-36=771ms/fwd×R200×2^K로 ≈44h/GPU. `/`로 병기하면 25,000s≈44h로 오독. 두 출처가 원 문서 수준에서 미정합임을 명시하도록 재서술(각 인용은 개별적으로 정확 — 병합이 미조정한 상속 불일치). |
| 4 | minor | §4 상위 10 vs §1.2 위계 | **기각** | 이슈 자체가 "No change required — 예외는 §1.2에 공개·정당화됨"으로 종결. §4가 P4 탐지 3건(I-27/28/29)을 P1 fidelity 3건(I-08/09/10)보다 앞세운 것은 §1.2 말미의 "프로젝트 자인 위험" 예외로 명시 공개된 사항 — 미공개 위반 아님. 리뷰어도 준수(compliant)로 확인. 서술 무변경(원하면 §4 제목을 "핵심 회수 대상 상세"로 개칭 가능하나 필수 아님). |
