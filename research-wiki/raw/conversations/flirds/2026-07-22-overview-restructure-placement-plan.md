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

---

## 추가 지시 (같은 세션 2차; Yonghee 원문 요지)

> "poison은 논문에 싣지 않을거라 필요 없어. 다 제외시켜줘. noise, freerider, zero, label flip류만
> 남기면 돼. 그리고 tier 3 std50k5는 gsm8k가 아니라 아마도 사용되지 않을 실험 결과야.
> Selection → downstream performance / Aggregation quality에서 llm 실험은 alpaca, mmlu, rouge
> 등이 성능 차이를 보여주기 어렵다고 판단해서 gsm8k 대상 실험으로 하려고해. 클라이언트
> 소규모로 참여하는 세팅도 안하려고 하고."

### 반영 내역

1. **poison 위협축 전면 제거**(overview): 범례에 위협 스코프 명시(noise류·FR류·label-flip류) + caveat 8을 제외-결정으로 교체. 제거 = 마스터표 행 12(device100 poison, 이후 행 renumber)·§3.1.4 요지·§3.1.5 B축 poison 열/행·§3.3.1 poison 2열+각주·§3.3.2(b4) 전체·§3.3.3 poison 열·§4.4.2(removal poison ASR) 전체(A3가 §4.4.2로 승계)·§4.5 poison ladder·§4.7 부수관찰·§5.1(총평·판정 매트릭스·서사)·§5.4 poison 행. 잔존 poison 언급 = caveat 8 + 커버리지 비고 2곳(데이터-존속 표기)뿐. ASR 지표도 범례에서 제거.
2. **std50k5 selection 계열 = 미사용**: §3.2 서두에 [07-22 결정] 블록 신설(GSM8K 심판 확정·소규모 참여 세팅 지양·std50k5 미사용) — §3.2.3 세팅·비용, §3.2.6 R2 블록(표 제거→미사용 1줄)·H-3(판정 없음), P10, 커버리지 갱신. **std50k5 fidelity probe(§4.2)는 별개 축이라 유지**(세션 판단 — Yonghee 문장이 Selection 절 스코프였음).
3. **placement plan**: 전역 결정 4종으로 확장, E4/E6/ablation-dose/appendix에서 poison 제거, §4 미배치에 poison·std50k5·MMLU/ROUGE 심판 추가, §5 의존성에 **R4 스펙 정합 경고**(현 스펙 N=50·5/50 부분참여 ↔ "소규모 참여 지양" — Yonghee 확인 필요) 추가. §3.2.1 MMLU/ROUGE parity는 본문 축에서 appendix 후보로 강등.
4. 메모리 신규: `paper-threat-stage-scope.md`.

### 추가 보류 (Yonghee 확인)

- **R4 gsm50k5 스펙 정합**: 현 스펙(N=50, 5/50)이 "클라이언트 소규모 참여 세팅 안 함"과 어떻게 정합되는지(스펙 유지? Tier B에서 개정?) — placement plan §5에 경고로 표기.
- std50k5 **fidelity** probe(§4.2 부분참여 우위 +1.000)의 유지 판단이 맞는지 — "소규모 참여 세팅 지양"이 fidelity 축까지 포함하면 §4.2(b2)·E3의 부분참여 서사도 축소 필요.

---

## 추가 지시 (같은 세션 3차; Yonghee 원문 요지)

> "CNN 실험 중에 3.2.6에서 보여주는 수치보다 3.2.2나 3.2.4에서 보여주는 수치가 더 좋아보이는데
> 이건 무슨 차이지? 그리고 3.2.4 표도 상대적 수치 말고 그냥 절대적인 accuracy로 변경해줘.
> 그리고 top-k selection도 안쓸거야."

### 답변 요지 (수치 차이의 정체 — 모순 아님)

1. **§3.2.4 ↔ §3.2.6**: 같은 무대·**같은 rundir**(§3.2.6 R1의 flirds/vanilla/oracle/random = §3.2.4 재사용) — dir1 flirds_gate_v2 절대값(.6315/.6148/.5668/.5712)이 §3.2.6 P1-online flirds 행과 소수 4자리까지 동일함을 파일로 확인. 차이는 **표기**(구 §3.2.4 = vanilla-대비 dAcc라 "+.3232"가 절대 ".5668"보다 커 *보였던* 것) + §3.2.4엔 더 쉬운 iid 파티션 포함.
2. **§3.2.2 ↔ §3.2.6**: §3.2.2는 threat-그룹 **pool**(cifar10+fmnist·iid/dir1/shard·강도 변형 str0.05 포함)이라 절대 수준이 높아 보임(clean .686~.734, GN .609~.645) — §3.2.6은 최난도 단일 조합(cifar10·dir1·strmain·lf@0.70)만. 같은 셀로 좁히면 정합(셀별 = RESULTS.txt).

### 반영 내역

1. **§3.2.4 절대 acc 전환**: `cnn_summary.csv` per-seed `final_acc`에서 직접 재집계(anaconda python; flip_rate 열이 NaN이라 cell명에서 dose 파싱) — iid/dir1 × 9 arm × 6 위협 절대값 표로 교체, oracle/random도 절대값. 읽기 문단 절대값 기준 재서술 + "§3.2.6과 rundir 동일" 각주. V2w 판정 불릿은 기준이 delta 정의라 유지(문구 명시).
2. **§3.2.2에 pool-구성 주의 1줄 추가**(세 절 수준 차이의 원인 명문화).
3. **phase1 retrain×top-k selection = 미사용**(3차 결정): §3.2.5 스텁화(제외 표기; rundir 존속), §3.2 매트릭스 칸 ○ 미사용, §5.1 최고 세팅 표 행 삭제, §6.1·마스터표 행13·§2.1 매핑·§8 커버리지 갱신. **softmax-선택 arm(flirds_sel/flirds_select·sfedavg)은 top-k가 아니므로 유지**(세션 해석 — 확인 대상). placement plan: E6에서 ④ top-k 제거 + "§1의 'selection' 열거는 게이팅/softmax-선택이 커버, §1 문구 조정 권고" + §4 미배치 추가.
4. 메모리 `paper-threat-stage-scope` 4항(top-k 미사용·절대 acc 표기 기본) 추가.

### 추가 보류 (3차)

- **softmax-선택 arm 유지 여부**: "top-k selection 안 씀"을 phase1 retrain×top-k로 해석 — §3.2.1 `flirds_sel`·§3.2.2 `flirds_select`/`sfedavg`(확률적 soft 선택)까지 제외 의도였으면 추가 정리 필요.

---

## 추가 지시 (같은 세션 4차; Yonghee 원문 요지)

> "새로운 실험 결과가 도착했어. overview에 반영해줘."

### 도착분 (커밋 4건, f5d40a7 이후)

1. **4c40e30** — R4 gsm50k5 Tier A seed0 4셀 완주(rundir 32 + `gsm50k5_tier_a.csv`·`llm_competition.csv`; `T2_CSIGN`·`GN_ABS` 스위치).
2. **b45d8c6** — γ*=5 확정(abs-probe r32 판단)·gn_full 활성 (REMAINING 문서).
3. **560a2fd** — gnoise 구정의(γ1.0 상대-dose) 폐기: rundir 7개 삭제·`oracle_excl`만 보존(γ-무관), `gsm50k5_tier_a.csv` 재생성(25행), REMAINING §1.0 인수인계 재작성.
4. **c05a951** — β0.3 재실행 device100 a0.1 noisy·frrand 2셀 완주(스위트 10종 = Fed-LOO 추가; timing.json 신설) — 잔여 16셀 큐.

### 반영 내역 (overview)

1. **§3.2.7 신설** — R4 Tier A seed0 3블록: (b1) 무대 판정(noisy +3.6pt 성립 / frzero +0.9 / gnoise −0.3 불성립→신정의 / clean 게이트 비용 −1.0pt), (b2) EM 절대값 표(vanilla/oracle/random/T1/T2 4점수원/t2_random; recovery·kept 병기 — noisy T2 flirds .3584 > lossheur .3548 > 1차 .3432, 동일예산 무작위 .3110 = +4.7pt; frzero 4점수원 kept=30 recovery 1.000), H-8~11 대조표(H-9 완전 적중·H-11 MISS 정직 보고·H-10 판정 불가), gnoise 주(GN_ABS γ*=5·gn_full 큐·llm_competition.csv 구정의 행 잔존 경고), (c) GPU-h(timing.json 합산 clean 8.5/noisy 24.1/frzero 22.0[dedupe 중복 기록, 실소요 ~15.8]/gnoise 잔존 2.4) + 1-seed·스펙-정합 caveat.
2. **§3.3.2 α=0.1 열 = 재실행판 정본으로 교체** — rundir metrics.json 직접 집계(3-seed ddof=0): Flirds .606±.056·Flirds-1st .607±.057·loss-heur .609±.055·**Fed-LOO .607±.056(행 신설)**·FedIF .692±.126·ComFedSV .430±.027·FLDetector .528±.068·STD-DAGMM .615±.103·FLTrust .726±.129·FedDQC 1.000; (b2)에 frrand α=0.1 요약 1줄·(b3)에 Fed-LOO 행·(c)에 집계 출처 주.
3. **전파**: 헤더 HEAD c05a951 · 출처 블록(rundirs_llm 37·gsm50k5_tier_a.csv·device100 canonical 확장) · 마스터표 행10/P11(§3.2.7·상태 갱신) · §2.1 매핑 · §3.2 매트릭스(LLM·IID에 gsm50k5) · §3.2 서두/§3.2.6/(b3)/H-3 포인터 · §4.8.1 P5-leg 문구 · §5.1(2차-① R4 불릿·판정 매트릭스·최고 세팅 표·clean 무해성 예외에 R4 T1 −1.0pt 추가) · caveat 9(잔여 18→16셀) · §6.3 timing · §8 커버리지 3행.
4. **placement plan**: §5 R4 행(완주·seed0 방향 확보·본문 확정치는 Tier C 대기; 스펙-정합 경고 유지) · appendix 게이팅·경쟁 상세 행(§3.2.6–7).

### 검증 노트

- 수치 전부 파일에서: `gsm50k5_tier_a.csv`(25행)·rundir `metrics.json`·`timing.json` 직접 집계(anaconda python). GPU-h는 커밋 메시지(25.2/15.8/14.4)와 timing.json 합산(24.1/22.0[중복]/2.4[잔존])이 다름 — 문서엔 timing.json 합산을 정본으로, dedupe 중복·폐기분은 주석으로 병기.
- §3.2.7 'noisy'=answer-swap(label-flip류) ≠ silo5 alpaca-noisy — §5.2 판정 3('0-교차 도달불가')을 끌어오지 않도록 위협 구분 주석.
- device100 재실행 전후 차: φ-method ≤.002, 탐지기 최대 .044(STD-DAGMM .659→.615) — 결론 불변, 표 주석에 명시.

### 보류(기존 유지)

- R4 스펙(N=50·5/50) vs 소규모-참여-지양 정합 — Yonghee 확인 대기(§3.2.7 caveat ③).
- Tier B(전 8종, ~300–350 GPU-h) 진입 = Yonghee 승인 게이트(REMAINING §3).

---

## 추가 지시 (같은 세션 5차; Yonghee 원문 요지 — 보류 8건 일괄 답변)

> 1. (Exp C·Taylor appendix) "차후 생각해볼게, 일단 그렇게 남겨놔." 2. (A3 label/quantity_skew) "추가 안 할거야."
> 3. (R4 스펙 정합) "R4는 쓸거야. 아까 그건 silo5 같은 범위를 말한거였어." 4. (std50k5 fidelity probe) "남겨놔야해.
> 아까 그건 selection 축 한정이었어." 5. (softmax-선택 arm) "수치가 잘 나오니까 남겨놓자." 6. "여기까지 커밋해주면
> 내가 푸시할게." 7. (R4 Tier B) "응 알겠어. 이거 말고도 지금 seed 실험 추가 돌릴거 있나?" 8. (E5 N=10 seeds)
> "10개 oracle은 돌릴 시간이 없어." (+ 위키 flirds-signal-size-diagnosis 삭제 = 본인 의도 확인)

### 결정 반영

- **해소 5건**: ② A3 옵션 폐기(overview §4.4.2 잔여 없음) ③ '소규모 참여 지양' = **silo5류 소규모 cohort 지칭**으로 확정 — R4(N=50·5/50) 사용 확정(§3.2 서두 ②·§3.2.7 caveat ③·plan §0/§5 ⚠ 제거) ④ std50k5 fidelity probe 유지 확정(지양은 selection 축 한정) ⑤ softmax-선택 arm 유지 확정 ⑧ E5 확장(seeds1·2·(a) 2¹⁰) 미진행 확정(마스터 P1·§3.1.3·§8·plan §5·REMAINING §1.4/§3).
- **존속 2건**: ① Exp C(§5.4)·Taylor(§5.5) appendix 배치 = 보류 표기 유지 ⑦ R4 Tier B 진입 = "응 알겠어"를 승인으로 간주하지 않음 — 게이트 유지(REMAINING §3).
- **링크 정리**(위키 진단 문서 삭제 확인 후): overview 4곳·wiki index·math-rigor 주석 = 커밋 2601229. 삭제 자체(D)와 flirds-protocol.md EOL 터치는 Yonghee 변경분이라 미스테이징.
- 메모리 `paper-threat-stage-scope` 갱신(항목 3~5 확정·신규 항목 5).

### 질문 답변(7) — 남은 seed-추가 실험 인벤토리

① probe A축 seeds 1-2(rank r32/64·st20/30 — 현 seed0; lr격자·noise·std50k5-r16은 이미 3-seed) = ablation 보강(선택) ② 3B silo5 robustness seeds 1-2(마스터 P5; 현 1-seed caveat 1 — β0.3 재실행 잔여 4셀은 seed0 재실행이라 별개) ③ R4 Tier C 3-seed(Tier B 뒤; 논문 E6-② 확정치에 직결). E5 seeds = 이번에 제외. lr·steps intervention 2차검증은 seed 추가가 아니라 기존 데이터 재분석(무GPU).

### 후속(같은 세션): seed-잔여 3건 REMAINING 등재

Yonghee 지시 — "남은 세 개(R4 Tier C·3B silo5 seeds1-2·probe A축 seeds1-2)를 REMAINING.md에 등재하되,
**논문에 실릴지 미정이므로 실리는 것으로 확정될 때만 실행**한다는 조건을 명기." → 루트 `REMAINING.md`
**§1.5 신설**(조건부 3건 + 실행 조건 경고; §1.4의 probe A축 항목은 §1.5로 이동해 중복 제거). 커밋 후 push는 Yonghee.

---

## 추가 지시 (같은 세션 6차; Yonghee 원문 요지)

> "새로 완료된 실험들 overview에 업데이트 해줘. 그리고 3.2.2, 3.2.4, 3.2.5는 미사용할거라 아예 지워도 될 것 같아."

### 도착분 (커밋, 1be6bcf 이후)

- **77ba500** — gnoise γ축 종결(γ=5 gn_full 체크포인트 FAIL·observer 완주 EM .3753[부호 반대] / γ=20 gn20_probe 완주 EM .3718[밴드 밖]; 진단 `gnoise_diag/` = gradient-방향만 2,400~38,000배 민감·등방 LoRA-delta 노이즈 무해; Fang'20 음성대조군·LoRASC 정합; γ*=5 선정근거 무효[주입이 train() 후라 train_loss 무반영]) + P5-soft 진행분 14 rundir(online 3 + t2_pw 8 + dup observer 3).
- **ae1c063** — 원리 분석 프롬프트 수행 완료(다른 세션): `survey/flirds-principle-analysis/`.
- 3a3d139·262f67c — paper §1–4 개정·AAAI 이관(Yonghee 작업; 워킹트리에 paper-ko·citations-table 진행 중).

### 반영 내역

1. **§3.2.7 갱신**: (b1) gnoise 3행(γ1.0/γ5/γ20 전부 불성립 → γ축 종결) · gnoise 주 전면 재작성(negative result 서술 — 방향-문제 진단·문헌 앵커·γ*=5 근거 무효·LIE/sign-flip 미결·H-10 CNN-only) · H-10 행 "판정 불가 확정" · **(b3) P5-soft 신설**(EM 표: noisy T2 순위 역전 fedif .3584 최고·flirds .3467 < lossheur .3503; online pweight 무효과 .3324; frzero pweight 회수 1.50[EM 노이즈 감안]·t2_pw = oracle 동값; clean 구조적 −0.4pt; dup observer 비트비재현 주의) · (c) 출처 40 rundir+gn20 root+gnoise_diag·잔여 4런.
2. **§3.2.2(C2 개입)·§3.2.4(CNN 게이트 그리드+V2w·V3)·§3.2.5(top-k 스텁) 절 삭제** — 번호 결번 유지, **caveat 14 신설**로 기록. 참조 정리: §3.2 매트릭스(CNN 칸 ○ 미사용)·매트릭스 주(CNN 증거 §3.2.6·§4.8 일원화)·마스터 행8/13/24·§2.1 3행·§3.1.2(c)·§3.2.1(flirds_sel 문구·pool 문구)·§3.2.6(R1 재사용 문구·판정 ④ V2w→0-교차 원인으로)·§3.3.5 헤더·§4.3.2 4곳·§5.1 4곳(최대 절대 이득 = §3.2.6 기반 .5668/.6065·회수 0.86/0.96으로 재구성)·§6.1·caveat 5 재스코프·§8 3행.
3. 전파: 헤더 HEAD 77ba500·출처 블록(rundirs_llm 52)·P11·§8 track_h/계획 행·§6.4에 원리분석 링크.
4. placement plan: E6(③ C2 삭제→②에 흡수·softmax-C2 문구 정리·의존성 갱신)·appendix 게이팅 행(V2w 제거·gnoise negative result 추가)·§4 미배치에 C2·CNN그리드 신설·§6 매핑 2곳.
5. 메모리 paper-threat-stage-scope 항목 6(CNN 개입 축 정리)·7(gnoise γ축 종결) 추가, 4번 softmax 잔존 범위 정정(std20 flirds_sel만).

### 검증

- §3.2.2/4/5 잔존 참조 0(구-표기·§6.2-14 제외) · 절 헤더 = 3.2.1/3/6/7 결번 구조 · banzhaf/ripple 1줄·poison 3회 유지.
- 신규 수치 전부 rundir metrics.json·timing·config 직접 확인(gn20 R=200 완주 EM .37176·γ5 observer R=200 EM .37534·P5s EM 12런).

---

## 추가 지시 (같은 세션 7차; Yonghee 원문 요지)

> "3.2.7도 3.2.6처럼 retrain/online 구분하고 다른 baseline의 수치까지 포함시켜서 accuracy 비교
> 가능한 테이블로 바꿔줘. gnoise는 3.2.7에서는 우선은 안 쓸 거라 다 지워도 돼."
> (이어서) "3.2.7에 recovery 정보도 필요 없으니까 지워줘."

### 반영 내역

1. **§3.2.7 (b2) 재구성** — §3.2.6 스타일로: **T1 online 표**(vanilla/oracle/random + flirds P1 게이트·P5-soft 가중 — Tier A 설계상 online은 flirds만, 타 점수원은 Tier B 명시) + **T2 retrain 표**(vanilla/oracle/t2_random + 점수원 4종 × {P1 sign, P5-soft 가중} 행 — P5s clean·frzero의 lossheur/fedif는 ⬚ 진행 중). 구 (b2) recovery/kept 괄호 표기·(b3) P5-soft 별도 블록은 폐지(표에 통합), kept·발화 정보는 읽기 불릿으로.
2. **recovery 전면 제거**(§3.2.7): 표 괄호·읽기 불릿·H-8(EM pt로 환산 +2.4/+2.1/+0.9)·H-9("EM oracle 동값"으로) — §5.1 R4 불릿·배치안 2곳도 "frzero 회수 1.000"→"kept=oracle-동일(EM 동값)"로 정합.
3. **gnoise 전부 §3.2.7에서 제거** — (b1) 3행·gnoise 주·H-10 행·(c) GPU-h/출처/caveat의 gnoise 항목 삭제. 기록은 **§6.3 검증-전용으로 이관**(negative result 요약 + rundir·진단 출처 + H-10 CNN-only + LIE/sign-flip 미결); §3.2.7 서두·(c)에 §6.3 포인터만. 마스터 P11·§8·배치안 §5/E6도 §6.3 포인터로 정합.

### 검증

- §3.2.7 내 "recovery/회수/gnoise" 잔존 = §6.3 포인터 2곳뿐(내용 0) · (b2) 두 표 = 절대 EM만.

---

## 추가 지시 (8차, 2026-07-23; Yonghee 원문 요지)

> "새로 완료된 실험들 overview에 업데이트 해줘. 그리고 절 숫자 안 맞는 것들 맞춰줘."

착지 커밋(HEAD `97316ac`): 49c402a(드레인 정지 + P5-soft clean_t2 완주·frzero_t2 3/4 + device100
a0.1_frzero β0.3 canonical) · 48734fc(track_g CNN skew 31 rundir 중간 스냅샷 — **인용 금지**) ·
409ce0e~5a8b99f(c2fid CNN fidelity leg 구현·144셀 sbatch·파일럿 제출) · 3e9e3cf/0ae7307/525adac/
43027e7(CNN 확장 leg 제출) · 0c8b9aa(R4 L1·L2 큐 + 사전등록 H-12·H-13) · 4399be3(Yonghee: 구식
문서 3건 삭제 + overview 열정렬만).

### 반영 내역 — (1) 신규 결과

1. **§3.2.4(구 3.2.7) P5-soft 채움**: T2 clean 4점수원(.3735/.3735/.3735/fedif .3727)·frzero
   3점수원(전부 .3691 = oracle 동값, kept=30) 착지 → ⬚ 1칸(frzero×fedif)만 남김. **.3736 → .3735
   반올림 정정**(3칸).
2. **기준선 오배정 정정(중요)**: P5s arm의 `vanilla_target`이 **dup observer**를 가리킴을 rundir로
   확인(clean .3727 / noisy .3351 / frzero .3655) → 두 표에 "observer 재실행" 행 신설. 07-22판의
   "P5-soft clean −0.4pt 구조적 비용"은 **기준선 오배정**이었고 자기 계열 대비 +0.1pt 이내(=검출 안 됨).
3. **EM 노이즈 바닥 ±0.5pt 신설**: 같은 구성 observer 2회 EM 차 = 5/6/1문항 → (b1) 주 + 읽기 불릿
   + §5.1 R4 불릿에 "안전 = vs-random +4.7pt·vs-1차 1.5pt / **미분리 = flirds↔lossheur 0.36pt**" 명시.
   frzero 무대 갭(+0.9pt)이 바닥의 2배뿐이라는 점도 (b1)에 표기.
4. **(c) 갱신**: 41 rundir(Tier A 24 + P5s 17) · GPU-h Tier A 54.6 + P5s 57.8 = 112.3(정책 1개 추가
   ≈ ×2 = Tier B 견적 입력) · caveat ④ dup observer 3개 + 기준선 규약 · 잔여 1런 + 차기 큐 L1·L2.
5. **§3.3.2 device100 a0.1_frzero(β0.3)**: (b2) 밑에 frzero α=0.1 주 신설(Flirds/1st/loss-heur/
   Fed-LOO/FLTrust 1.000 · FedIF 0.991±.003 · STD-DAGMM 0.945±.031 · FedDQC 0.532±.220 ·
   FLDetector 0.530±.067 · ComFedSV 0.442±.052) · (b3) Fed-LOO·FedIF 열 확장(frzero 0.818±.006) ·
   출처에 timing 3셀(4.22/4.52/4.78 GPU-h).
6. **캠페인 상태**: caveat 9 전면 개정(완주 3 / poison 영구제외 3 / 보류 10 ~36 GPU-h + poison 열
   β 혼재 기록) · §4.7 (b) 한 줄 추가.
7. **신규 계획 3행**: **P12** R4 (b)-fidelity 셀(H-13) · **P13** CNN c2fid 144셀(F-1~F-4) ·
   **P14** CNN downstream 확장(31/195 착지, 인용 금지) — §3.1 서두에 P12·P13 안내 블록(포화 문제의 답),
   §8 커버리지 3행 갱신(track_g rundirs_cnn 67·track_h rundirs_llm 55·track_c c2fid).

### 반영 내역 — (2) 절 번호 정리

- **§3.2 재번호**(삭제 3절로 생긴 결번 제거): 구 §3.2.3→**§3.2.2** · 구 §3.2.6→**§3.2.3** ·
  구 §3.2.7→**§3.2.4**. 스크립트 일괄 치환(overview 53줄 + placement plan·principle-analysis·
  REMAINING·track_h README/RUN_P5/RUN_DYN/RUN_SCALE) 후 수동 보정:
  - "구 §3.2.x"(=삭제된 절 표기)는 보호 — §2.1에 **표기 규약 주** 신설("구 §3.2.4"=삭제된 CNN 그리드
    ≠ 현 §3.2.4=R4), caveat 14에 매핑 기록.
  - 범위 표기 수동 교정: 구 §3.2.3–4(Track G LLM+CNN) → **§3.2.2**(CNN 절 삭제로 LLM만) ·
    구 §3.2.6–7 → **§3.2.3–4** · placement plan 축약형 "§3.2(§3.2.1·3·6·7)" → "§3.2.1–4 전체".
- **삭제 문서 참조 정리**(4399be3로 사라진 파일): `RERUN_AFTER_REPRO_FIX_2026-07-21.md`(caveat 13·
  §6.4·placement plan §5 → REMAINING §1.4 + P1 DROP) · `baseline-original-paper-verification`(§0 서두·
  §6.4) · deferred-rigor-inventory. **깨진 위키링크 수정**: [[flirds-principle-analysis/flirds-principle-analysis]]
  → [[flirds-principle-analysis]](폴더엔 검증 스크립트만 존재).

### 검증

- 절 헤더 = §3.2.1/2/3/4 연속 · 전 문서 §3.2.5+ 잔존 0(구-표기 제외) · 범위 표기 잔존 오류 0.
- 신규 수치 전부 rundir `metrics.json`/`timing.json` 직접 집계(anaconda python; ddof=0):
  gsm50k5 43 rundir EM·kept·gate·GPU-h 표 + device100 a0.1_frzero 3-seed mean±std.
- runs/ 읽기 전용 · GPU 0 · paper/ 무수정 · 로컬 커밋만(push = Yonghee).

---

## 추가 지시 (9차, 2026-07-23)

> Yonghee: "새로 완료된 실험들 overview에 업데이트 해줘."
> Yonghee(중간 추가): "그리고 fed-loo는 논문에서 제외하기로 결정했기 때문에 overview에서도 그냥 다 지워줘."

착지 커밋: `4fb02fb`(드레인 완주 — device100 a0.01 2셀 + P5-soft frzero×fedif) ·
`2d3f482`(a0.01 2셀 canonical 통합) · `b4f238a`·`194c2e4`·`23659af`·`914b922`(REMAINING 개정, 타 세션).

### 반영 내역 — (1) 신규 결과

1. **R4 P5-soft 12/12 완주**: 마지막 `gsm50k5_frzero_t2_pw_fedif_seed0` 착지 → §3.2.4 (b2) T2 표의
   마지막 ⬚ = **.3700**. kept=30(=oracle 집합)이나 dedupe 불발로 **독립 재학습** → 타 3점수원 .3691과
   **1문항** 차. 이 1문항이 "init 고정 후 run-to-run 잔차"의 눈금이 됨.
2. **[신규 발견] EM ±0.5pt 노이즈 바닥의 정체 = H1 재현성-fix 경계**(커널 비결정성 아님):
   전 gsm50k5 rundir `meta.json:git_sha`를 fix 커밋 `8598cea`(track_g.py:270 `seed_everything(0)`)
   기준으로 가른 결과 — canonical observer 3개 = pre-fix(`fa5fc6e`, LoRA-A init entropy-seeded),
   dup observer 3개 = post-fix(`262f67c`). 즉 ±0.5pt = **init 재추첨 2회의 실측 스프레드**
   (§4.7 'run-인스턴스 재추첨'·caveat 13① 과 같은 현상).
   - **판본 분포(42런)**: pre-fix 21 = observer 3·oracle_excl 2·random_excl 2·clean t2_sign 4·
     frzero t2_sign 4·frzero t2_random·**noisy t2_sign_flirds**·flirds_gate_v2 3 /
     post-fix 21 = noisy t2_sign 3종·noisy t2_random·flirds_pweight 3·P5s t2 12·dup observer 3.
   - **함의**: (b2) noisy T2 P1 행에서 **flirds만 pre-fix**, 경쟁 3종+t2_random은 post-fix로 init 공유
     → flirds가 낀 비교엔 재추첨 1회가 섞임. 1.5pt(vs 1차)·4.7pt(vs random)는 3~9배로 생존,
     0.36pt(vs lossheur)는 이중 미분리. Tier C(L1)는 전부 post-fix라 해소.
   - **부수**: P5s의 `vanilla_target`이 dup을 가리키는 것은 단순 기록이 아니라 **판본 정합**.
   - 반영: §3.2.4 (b1) 주·읽기 1·2·3 불릿·caveat ④ 전면 개정 · §5.1 R4 불릿.
3. **device100 a0.01 noisy·frrand = β0.3 재실행판 canonical**(2d3f482): §3.3.2 (b1) α=0.01 열 갱신
   (Flirds .575→.572·1st .575→.571·loss-heur .574→.571·FedIF .568→.570·ComFedSV .419→.438·
   FLDetector .482→.499·STD-DAGMM .652→.600·FLTrust .602→.592) + frrand 표 밖 열 신설 +
   (b3) FedIF 재실행 a0.01 Spearman(noisy .616±.094·frrand .756±.030) + timing 5.18/5.01 GPU-h.
   ⚠ **a0.01 frzero는 미재실행**(구 실행 = 판본 혼재 명기).
4. **β 캠페인 종료 반영**: §4.7·caveat 9 — 완주 5셀 전부 커밋 완료 / poison 3 영구제외 /
   보류 10(device100 7[a0.0·a5.0·a0.01 frzero] + 3B silo5 3, ~36 GPU-h).
5. GPU-h 재집계: R4 Tier A 54.6 + P5s **60.4** = **115.0**(rundir timing.json 합산).

### 반영 내역 — (2) Fed-LOO 전면 제외

- **caveat 15 신설**(Banzhaf·Ripple caveat 12와 동형): 논문 비게재 → 전 표·범례·baseline-set에서
  행·열·언급 삭제, **rundir 데이터는 존속**, 러너는 계속 산출하므로 집계 단계에서만 제외.
- 삭제 지점: §3.1.3 (b1)·(b2) 행 + 절 제목·(a)·읽기(E4 = '경량 3-seed 재확인 + loss-heur post-fix
  runtime 정본'으로 재정의) · §3.1.4 · §3.3.1 표 행·읽기·baseline-set(14종→13종) · §3.3.2 (b1)·(b3)
  행 + 표 밖 열 3곳 + runtime · §3.3.4 표 행(레이아웃 재배치) · §3.4.1 op-count 열 · §3.4.3 ·
  §4.2 (c) · §4.4.1(5종→4종) · §4.4.2(cifar10 최고 분리 = 익명 처리) · §4.6 AdamW 열거 ·
  §5.1 2곳 · §5.2 감사 2곳 · §6.3 E3 · §2 행 10·19·21·P12·P13 · §2.1 매핑.
- **placement plan도 동반 수정**(논문 계획 문서라 live 인용이 남으면 모순): §전역결정에 Fed-LOO 행 추가 ·
  E2 확장 문구에서 "Fed-LOO 동률"·"분야표준 앵커" 제거(→ §2 계보 공백 앵커 논거 공석으로 명기) ·
  §제외 목록에 항목 추가.

### 검증

- 잔존 "Fed-LOO" 10곳 = 전부 **제외 사실을 기록하는 주석**(caveat 15·폴더명 유래·rundir 존속 안내).
- 표 셀 수 정합 자동검사 통과(escape된 `\|` 1건은 기존 §4.8 P5 표).
- 신규 수치 전부 rundir `metrics.json`/`timing.json` 직접 집계(anaconda python; ddof=0):
  gsm50k5 44 rundir(EM·kept·dedup_shared·git_sha·GPU-h) + device100 a0.01 2셀 3-seed 전후 대조.
- runs/ 읽기 전용 · GPU 0 · paper/ 무수정 · 로컬 커밋만(push = Yonghee).
