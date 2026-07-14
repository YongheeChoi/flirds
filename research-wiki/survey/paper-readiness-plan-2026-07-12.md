# 논문화 준비 계획 (Paper-Readiness Plan) — 2026-07-12

작성: Claude 세션 (survey 폴더 전체 + wiki 핵심 문서 + 코드·rundir 실사 기반).
**시간 제약 없이 "논문이 되기 위해 필요한 것" 전수 목록.** 2주 제출용 우선순위 배분은 별도
대화 기록 참조 (Tier 0/1/2 — 본 문서 §2의 각 항목에 [T0]/[T1]/[T2]/[skip-2wk] 태그로 병기).

정본 원칙 (review-claude.md): **rundir·CSV(file-canon) > survey 문서 > 위키 > dossier.**
본 문서의 모든 수치 인용은 survey 문서 기준이며, 논문 수록 전 rundir 재확인 대상.

---

## 0. 목표 서사 — "측정이 먼저다" (measurement-first)

리뷰(07-02, review-claude.md)와 결과분석(06-26)이 수렴한 결론:

- **팔 것**: ① federated × LLM × client-level valuation을 **exact-2^N 듀얼 오라클**로 채점한
  최초의 측정 표준, ② O(1)/round (1 HVP) 추정기가 그 오라클을 충실 재현, ③ 경계 지도
  (near-additivity·부분참여·poison·비용 crossover)의 경험 법칙들.
- **팔지 않을 것**: "Shapley로 더 나은 학습" free-lunch 서사 (선행이 점유 + 우리 데이터가
  약함 + exact oracle조차 못 이기는 축 — device100 noisy에서 (b) oracle AUROC 0.604±0.050(seed0 0.660)).
- **결과 배치**: hardness ladder (쉬운 무대→어려운 무대) 순. 헤드라인 = 부분참여 fidelity
  (std20/std50k5에서 경쟁자 붕괴) + 듀얼 오라클 + cohort-조건화 비용. clean-anchor "+1.000"은
  포화 표(사다리 1단)로 강등.
- **질문 위계 준수** (Yonghee 2026-06-12 lock): 1차 fidelity → 2차 성능→수렴→탐지.

---

## 1. 논문이 되기 위한 게이트 (필수 — 이것 없이는 reject 위험이 명시적으로 문서화된 항목)

### G1. GT 순환성 봉합 — own-game 심판 셀 ★최우선 [T0 코드 + T1 실행]

리뷰 C-1: (b)는 Flirds가 Taylor 전개하는 바로 그 게임 → fidelity 표가 "추정 오차"와
"게임 불일치"를 분리 못함. **어느 기존 계획에도 없던 실험** — 07-12 설계 완료:

- **설계**: 1B silo5-config (N=5, R=10, fp32, 5-도메인=non-IID) × 2변형(등n=E / 비등n=U,
  예: {600,300,150,150,100}) × 3-seed. 한 frozen 궤적 위에서 동시 산출:
  (b)-고정가중 exact Shapley + **재정규화-게임 exact Shapley**(`measure_taylor_residual.py
  --renorm` 기제) + **(a) retrain oracle** + 전 방법 점수(Flirds/1st/GTG/FedSV/Banzhaf/
  ShapleyFL/ComFedSV/loss-heur/Fed-LOO/FedIF) + ShapleyFL own-game raw(min-max·EMA 전) +
  **U(S) 전체 parquet 영속**(재발 방지).
- **결과 표**: method별 [ρ(m, 자기게임) | ρ(자기게임, (b)) | ρ(m, (b)) | ρ(m, (a))] +
  심판 행 [ρ((b)-고정, (a)) vs ρ(renorm-게임, (a)) vs ρ(ShapleyFL-게임, (a))].
- **코드 사실** (07-12 실사): GTG/FedSV=renorm 게임 확정(`gtg.py:18-27`, `fedsv.py:14`);
  U(S) 미영속(오프라인 불가); 비등n 미지원(빌더 등분 하드코딩, ~30–50줄; 게임 수학은 실제
  n_k 사용이라 데이터만 비대칭이면 작동); measure_taylor_residual.py가 두 게임 exact를 이미
  계산. Fed-LOO는 track_d/track_c1 네이티브, phase2는 method 목록 1줄 추가 필요.
- **비용**: 구현 1–2일 + gpt2 CPU 스모크(로컬) / GPU seed당 ~3–4h((a) 지배) × 6 ≈ ~20h.
- **리스크**: (a)가 renorm 편을 들 수 있음 → fallback = 두 게임 병기·특성화 서사. 사전에
  서사 분기 합의 필요 (§4 결정 D-10).

### G2. 수치 정본화 (R1–R13) [T0]

- **R1**: "3B (a)vs(b)=+0.900" 오귀속 — 3B에 (a) 없음. 전 문서·논문에서 제거/재작성.
  (3B (a) 실측은 G7-2로 별도.)
- **R2**: task6 "+1.000" rundir 미커밋 — 논문 인용 금지. G1 변형 E가 정본 재실행을 겸함.
- **R9**: CNN ComFedSV 구수치 {1.0,0.96,0.85,0.84} 인용 금지.
- **R6**: Ripple ~4515s 회계 주석 필수 (자체 궤적 포함 유일 방법).
- **R11**: ShapleyFL β 프로비넌스 통일 — G9.
- 07-04 검증 산출물(수학/Ripple/정밀도/deferred/cost 5폴더)은 커밋·push 완료(9dacbad, 07-04) — Yonghee 검토 잔여.
- (b) xseed·가산갭·SNR 수치의 CSV 정본화 (K-6; 현재 진단 문서 산문에만 존재).

### G3. 이론 정식화 [T0 집필 + T1 1B 실측]

- P1–P8을 본문 명제 + appendix 증명으로 포팅 (irds-fl-math-rigor.md가 논문 수록급으로 준비
  완료 — 반박 패널(18) 교정 반영본(수용 36건)).
- gpt2 스모크 수치(P1·P2 대수 1e-12 확증) 수록.
- **P3 물리 잔차 1B 실측** (~55–75분, RUN_1B.md 준비 완료): 2차>1차 우위, O(‖Δ‖³) 스케일링,
  P5 순위. §7.2 placeholder 교체.
- 한계 11건(§8) appendix 수록: sequence-function 공리화 미해결 승계, (a)↔(b) 경험적 한정,
  Taylor 상수 미상, LLM token-mean 브리지 미성립, 고정가중=정당화된 선택(유일해 아님) 등.
- 부호 규약 명시 선언 (§4 결정 D-3).

### G4. Fed-LOO 수치 확보 [T1]

구현 완료(`in_run_sv.py:70-98`, 합성 검증 max-diff 0.0), 수치 0건 — "cherry-picking 1순위"
리뷰 예상 지적. 로그 미영속이라 백필 불가 → 재실행 필요:
- track_d 1B 두 레짐 경량 재실행 (coalition off, (b)+Fed-LOO+Flirds만; ~6–9h), 또는
- G1 심판 셀에 포함(설계에 이미 반영) + G9 재실행 셀에 phase2 method 1줄 추가로 동승.

### G5. 통계 위생 (C-7) [T0~T2]

- 모든 headline metric에 95% bootstrap CI(B=1000) — protocol §3이 요구하나 미이행.
- 3-seed 미완 칸 완성: **3B poison seeds 1–2** [T2] ("3B에서 2차도 붕괴" 주장이 1-seed),
  3B robustness 전반(P5), A축 probe seeds 1–2 [skip-2wk 가능 — 논문 주장에 lr probe 안 쓰면].
- N=5 순위 우연성(1/120/seed) 명기 + N=10로 보강(G7-1).
- tiny-val caveat (silo5 val=20, device100 val=10) 명기; val≥200 재실행은 G7-4.
- git_dirty rundir 목록 정리, 델타 로그 영속화는 향후 run부터.

### G6. 자인(self-acknowledged) 위험 2건 [T2 / 일부 skip-2wk]

- **I-27 delta free-rider**: recycled-delta는 ∇ℓ_val 정렬 → 1차항≠0 가능 — "free-rider φ
  exact-0" 헤드라인이 쉬운 zero/random에만 성립. 구현 ~20줄(직전 글로벌 상태 threading,
  `llm_server.py:37` 시그니처) + 1B silo5 셀 (~3h). **문서 강권 — 논문 전 실행 권장.**
- **I-28 PGD/direction-aligned poison**: "2차>1차" flagship 실험 공백. 신규 구현 필요(현재
  Xu trigger + Bagdasaryan scaled만, `corruptors.py:81-105`). [skip-2wk → limitation +
  future work; 시간 나면 CNN 파일럿]

### G7. Oracle 커버리지 확장 (축퇴 레짐 탈출) [T2~]

현행 (a)/(b) 일치가 N=5·near-additive·등n에 갇힘 (I-01/I-02) — 판별력은 축퇴 밖에서만:
1. **(b) N=10 exact 1B 3-seed** (~10h) — 실행 권고 상태, 승인만 필요.
2. **3B (a) silo5 3-seed** (~18–20h) — (a) 앵커 2번째 스케일, R1의 실측 해소.
3. (a) N=10 (샤딩 신규 + 11–22h/4-GPU) — [skip-2wk] 미실행 시 논문 스코프를 "N=5 dual-GT +
   대규모 (b)-fidelity"로 한정 선언.
4. device100 anchor val≥200 재실행 (P2), α=0 (b)-perround anchor (~11h/셀) — proxy-truth
   순환(§5.4) 해소는 서술 한정 가능 [skip-2wk].
5. CNN (a)-발산(전 방법 ≤0.45) 원인 규명 (P5-analysis) — 최소 서술로 해소 가능.

### G8. 비용 표 정비 [T1 스모크 + T0 집필]

- 통일 회계 스모크 (~30분): 로그 생성 시간 명시 측정 + Ripple 분리계측 + fp32-vs-bf16
  ×3.1 확정(현재 **미검증 placeholder — 인용 금지**) + peak-mem.
- 표 구조: valuation-only / end-to-end 2단 분리 (권고안; 결정 D-7).
- caveat 7건 반영 (C1 Ripple 혼합, C6 loss-heur ~2배 과대측정, C7 GPU-hours 미기록 등).
- cohort-조건화 서술 필수 (std20 역전: (b) 2917s < Flirds 4697s).

### G9. ShapleyFL β=0.3 통일 (R11) [T2]

- **발견 (07-12)**: 대기 25 phase2 셀 중 device100 sweep 14셀은 COALITION=0이라 ShapleyFL
  미계산 → β 영향 0, 재실행 불요 (검증 후 큐 제외 권장). 실제 필요 = **7셀**(1B silo5 4 +
  device100-a0.5 anchor 3; 3B silo5 4도 COALITION=0 — ShapleyFL 미계산이라 동일 제외) ≈ 1–1.5 GPU-일.
- 7B track_d 6셀 = 최중량 꼬리 → **각주 처리 권장**("7B ShapleyFL은 β=0.5"); 재실행은 여유 시.
- 재실행 시 phase2 method 목록에 Fed-LOO 1줄 추가 → 백필 공짜 동승.
- 재개 절차: `runs/rerun_beta03/RESUME_AFTER_MIGRATION.md`.

### G10. 2차 축(개입) 처리 방침 [T0 결정 + 선택 구현]

- 현행: clean-IID do-no-harm parity(설계상 정답) + CNN 오염 회복 + 7B r2t 14–18%.
  **B축 무대의 개입 결과를 2차 축 대표로 승격** (신호가 실재하는 곳).
- 선택지: 기존 4 arm 서술로 충분 vs **TRAC**(contribution-weighted-training.md 제안 1순위;
  비용-0 폐형 가중, dead-zone=구성적 do-no-harm) 구현 추가. TRAC는 novelty 델타가 있으나
  이번 논문 스코프 밖으로 미루는 것도 방어 가능 (결정 D-14).
- MMLU 학습-0 문제 (I-24: SFT가 MMLU −0.8~−1.4pp): 축② 지표 재선정 또는 "MMLU power≈0
  명기" 서술 (결정 D-13과 연동).

### G11. 재현성 패키지 [T0~]

- 코드·rundir 공개 계획 (익명 repo?); make_analysis/make_fidelity 재생성 경로 문서화.
- make_analysis.py cp949 버그 수정, 스테일 RESULTS.md 참조 정리 — make_analysis.py는 RESULTS.md 미생성(레거시 make_report.py는 미영속 로그 기반), 실산출물=analysis/00_overview (codex 지적).
- precision_guard 구현 여부 (protocol이 스펙만 존재) — 결정 D-5와 연동.
- protocol §1 문서-코드 불일치 해소 (silent-deviation 자기 규칙 위반 상태).

---

## 2. 실험 전수 목록 (시간 무관; 필수도 순)

| # | 실험 | 상태 | GPU 비용(실측 기반) | 게이트 | 2주 태그 |
|---|---|---|---|---|---|
| E1 | own-game 심판 셀 E+U ×3seed | 설계 완료, 코드 필요 | ~20h | G1·G2(R2)·G3(P5) | T0+T1 |
| E2 | Taylor 잔차 1B | **준비 완료** (RUN_1B.md) | ~1h | G3(P3) | T1 |
| E3 | 통일 회계 스모크 | 스크립트 미작성 | ~0.5h | G8 | T1 |
| E4 | Fed-LOO 경량 백필 (track_d 1B) | 러너 네이티브 | ~6–9h | G4 | T1 |
| E5 | (b) N=10 exact 1B 3-seed | 승인 대기 | ~10h | G7-1 | T2 |
| E6 | 3B (a) silo5 3-seed | 미실행 | ~18–20h | G7-2 | T2 |
| E7 | delta free-rider 1B silo5 | 구현 ~20줄 | ~3h+반나절 | G6(I-27) | T2 |
| E8 | 3B poison seeds 1–2 | 미실행 | ~4h | G5 | T2 |
| E9 | β0.3 재실행 7셀 (+Fed-LOO 동승) | 큐 준비됨 | ~1–1.5일 | G9·G4 | T2 |
| E10 | CNN TF32 A/B (yonsei) | 스크립트 미작성 | ~1–2.5h | 정밀도 각주 | T2 |
| E11 | A축 probe seeds 1–2 | 미실행 | ~수 h | G5(진단 문서) | skip-2wk |
| E12 | (a) N=10 (샤딩 신규) | 미설계 | 11–22h/4-GPU | G7-3 | skip-2wk |
| E13 | PGD poison | 미구현 | 구현+실험 | G6(I-28) | skip-2wk |
| E14 | α=0 (b)-perround anchor | 미실행 | ~11h/셀 | G7-4 | skip-2wk |
| E15 | anchor val≥200 재실행 | 미실행 | ~수 h | G5 | skip-2wk |
| E16 | 7B β0.3 track_d 6셀 | 큐 준비됨 | ~2–3일 | G9 | skip-2wk(각주) |
| E17 | E-sweep×α-sweep (Prop1/2 실증) | 원계획 #5 | 대형 | G3 보강 | skip-2wk |
| E18 | 7B robustness (P4) | 미실행 | 대형 | 커버리지 | skip-2wk |
| E19 | TRAC 개입 arm | 제안만 | 구현+실험 | G10 | skip-2wk |

GPU-0 분석 작업: lr·steps intervention 재분석(track_d arm 데이터 기존), (b) xseed CSV
정본화, B축 자명성 검사(φ vs n_k 부분상관), 전면 재집계+CI 산출.

---

## 3. 집필 체크리스트 (섹션 ↔ 소스 매핑)

| 논문 절 | 소스 | 상태 |
|---|---|---|
| Abstract/Intro | review-claude.md 서사 권고 + taxonomy 빈칸 관찰 | 초안 가능 (paper/ 작성됨) |
| Related work | prior-work-taxonomy 4문서 + flirds.md 비교표 | 초안 가능 |
| Setup (게임 정의·듀얼 오라클) | irds-fl-math-rigor.md §2–3 + flirds-protocol.md | 초안 가능 |
| Method (추정기) + Theory | irds-fl-math-rigor.md P1–P8 + gpt2 스모크 | 초안 가능; P3 1B 대기 |
| Experimental design | flirds-protocol.md + baseline-selection-audit | 초안 가능 |
| Results: fidelity ladder | results-overview §3.1 + probe §3.6 | 초안 가능; N=10·Fed-LOO 대기 |
| Results: own-game | E1 | **실행 대기 — 서사 관문** |
| Results: boundary map | signal-size-diagnosis + B축 매트릭스 | 초안 가능 |
| Results: cost | results-overview §3.5 + cost-methodology | 초안 가능; 스모크 대기 |
| Results: 2차 축 | results-overview §3.2–3.4 | 초안 가능 |
| Limitations | rigor §8 + deferred-inventory + review C-5 | 초안 가능 |
| Appendix: proofs | irds-fl-math-rigor.md | 포팅 작업 |
| Appendix: 프로토콜/재현성 | flirds-protocol.md + rundir 스키마 | 초안 가능 |

**논문 드래프트 위치**: 루트 `paper/` (2026-07-12 생성; main.tex + sections/ + references.bib
+ README.md — placeholder 인벤토리는 paper/README.md 참조).

---

## 4. Yonghee 결정 대기 — 전 문서 통합 취합

집필/실험을 막는 순으로:

| # | 결정 | 출처 | 막는 것 |
|---|---|---|---|
| D-1 | **poison orientation 헤드라인** (matrix φ vs D2b −φ) | I-29 (최고 긴급, 실험 불요) | Results 탐지 절, 상충 문서 정정 |
| D-2 | **타겟 venue/포맷** (NeurIPS/ICML/ICLR, 페이지 수) | 신규 | paper 스타일·분량 |
| D-3 | **부호 규약 논문 표기** (φ>0=유익 반전 권장 vs 내부 유지) | rigor §9-① | 전 수식·표 |
| D-4 | own-game 결과 서사 분기 사전 합의 ((a)가 renorm 편이면?) | G1 리스크 | Results 구조 |
| D-5 | 정밀도 정책 옵션 ①(fp32 유지+protocol 개정) vs ②(학습 bf16) | precision-policy | protocol 절, 비용 서술 |
| D-6 | Ripple 처리 수위 (fidelity 제외+runtime 각주 [권고] vs 부록) | ripple-audit | baseline 표 |
| D-7 | 비용 표 2단 분리 [권고] vs 단일+각주 | cost-methodology | 비용 절 |
| D-8 | (a)/(b) N=10 실행 스코프 ((b)만 [권고] vs 둘 다 vs 안 함) | I-02 | fidelity 표 범위 |
| D-9 | 7B ShapleyFL β: 각주 [권고] vs 재실행 | R11/G9 | 7B 열 |
| D-10 | delta FR·PGD 논문 전 필수 여부 (delta FR만 [권고]) | I-27/28 | 탐지 절 강도 |
| D-11 | CNN 트랙 배치 (본문 압축 vs appendix) | 신규 | 분량 배분 |
| D-12 | 코드·데이터 공개 계획 (익명 repo 등) | review K | 재현성 절 |
| D-13 | Track D 잔여 3건 (bridge arm / FlowerTune 채점 / — β는 D-9) | TRACK_D_REVIEW | 개입 절 |
| D-14 | TRAC 구현 포함 vs 차기 논문 | contribution-weighted | 2차 축 스코프 |
| D-15 | "first" 주장 수위 ("관찰된 빈칸" 표현 유지 [권고]) | taxonomy 자기 caveat | Intro/RW 문구 |
| D-16 | (a) 러너 bf16 기본값·loss-heur 2배·timing.json — 수정 vs 각주 | precision/cost | 재현성 |

---

## 5. 리스크와 fallback

1. **own-game에서 (a)가 renorm 편**: fallback = "두 게임 병기 + 게임 선택은 응용 의존"
   특성화. 논문은 죽지 않음 — 측정학 기여는 유지, 단 "(b)가 옳은 과녁" 주장 철회.
2. **N=10에서 fidelity 하락**: 오히려 정보 — near-additivity가 N에 어떻게 깨지는지가 경계
   지도의 일부. 정직 보고.
3. **delta FR에서 φ≠0**: "exact-0은 zero/random 한정" 경계 명시로 전환 (자인 리스크의
   정직한 해소 — 문서들이 이미 이 프레이밍 준비).
4. **GPU 복구 지연**: T0(집필·정본화·이론)만으로도 골격 제출 가능하나 C-1 미봉합 → 주장
   수위를 "측정 프로토콜 + 관찰" 수준으로 낮추는 축소판 fallback.
5. **3B poison 1-seed 유지 시**: "스케일 의존 회피" 주장을 관찰 수준으로 강등.

---

## 6. 유지보수

- 본 문서는 게이트 완료 시마다 갱신 (완료 표시 + rundir/커밋 링크).
- paper/ 드래프트의 placeholder 목록은 paper/README.md가 단일 소스.
- 새 실험 결과는 results-overview(파일-only 원칙) 경유 후 논문 반영.
