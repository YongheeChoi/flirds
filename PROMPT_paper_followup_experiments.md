# PROMPT — 논문 잔여 실험 실행 (paper-ko 마커 해소용, 2026-07-17)

> **이 문서는 GPU 서버 세션에 넘기는 실행 프롬프트다.** 목적: `paper/paper-ko.md`에 남은
> 실험-대기 마커(🔴TODO/🟣VERIFY)를 해소하는 실험 7종의 실행. Yonghee 결정(2026-07-17) 반영.
> 계획 원본: `research-wiki/survey/paper-readiness-plan-2026-07-12.md` §2 (E-번호 동일).

## 0. 전제 — 확정된 결정 (변경 금지)

- **E1(game-adjudication) 미실행** — future work로 논문에 명시됨. 이 목록에 없다. 실행하지 말 것.
- **bf16 관련 실험 전부 불필요** — bf16/정밀도 논의는 논문에서 제외 확정(실험 세팅에 "fp32 사용"만 언급). bf16 정본 재실행·TF32 A/B(E10) 하지 말 것.
- **poison 실험 전부 불필요** — 논문에서 제외 확정. poison 셀 재실행 금지.
- **Banzhaf 계산 불필요** — baseline에서 제외 확정(전수 열거 = 참값과 동일 비용). 러너에서 꺼도 됨.
- **부호 규약(D-3)**: 논문 표기 φ>0=유익 확정(내부 규약은 음수, 표시 단계 반전).

## 1. 환경·충돌 주의

- python: `/home/korea_bupj/miniconda3/envs/flirds/bin/python`, `codes/`에서 `PYTHONPATH=.`
- **이미 실행/예약 중 — 중복 실행 금지**: ① CIFAR-10 removal 9셀(Exp A3 확장,
  `runs/removal_dose/run_cnn_removal.sh`) ② AdamW seeds 1–2(chain2 예약) ③ 이월 9셀
  (7B β0.3 ×6 + device100 anchor β0.3 ×3, `flirds_batch/scripts/run_deferred.sh`).
- **정본화 규율**: 모든 수치는 rundir(config+meta+phi.parquet+metrics)/CSV로 영속.
  .log·노트에만 남기지 말 것(과거 R1/R2/R9 재발 방지). 기존 canonical rundir 덮어쓰기 금지
  (신규 셀명/별도 root 사용).
- **완료 후 문서 갱신 경로**: rundir → `research-wiki/survey/flirds-experiment-results-overview-2026-06-25.md`
  (파일-canon 원칙) → `paper/paper-ko.md` 해당 마커 해소.

## 2. 실행 목록 (우선순위 순; 총 ~25–35 GPU-h)

| # | 실험 | 비용 | 논문 랜딩 지점 (paper-ko) |
|---|---|---|---|
| 1 | E2 Taylor 잔차 1B | ~1h | §4.2 명제 3의 🔴TODO(2차-우위 비율·스케일링 지수) |
| 2 | E3 비용 회계 통일 스모크 | ~0.5h(+스크립트 작성) | §6.4 granularity 🔴TODO(Ripple 비용 병기) + D-6 종결 |
| 3 | probe seeds 1–2 (N=50·lr·noise) | ~수 h | §6.1 N=50 🟣VERIFY, §6.3(ii) lr 🟣VERIFY, §6.3 노이즈-분리 🟣VERIFY |
| 4 | E7 delta free-rider | 구현 ~20줄 + ~3h | §6.5 탐지 🔴TODO(E7), §7 적대 스코프 |
| 5 | E4 Fed-LOO 본표 | ~6–9h | 표 1 Fed-LOO 행 2칸 + 초록 🔴TODO |
| 6 | E5 (b) N=10 exact | ~10h | §6.1 포화 문단(N=5 우연성 1/120 보강), §7 참값 커버리지 |

### 2.1 E2 — Taylor 잔차 1B 실측 (준비 완료, 그대로 실행)

- **목적**: 명제 3(잔차 bound)의 실측 검증 — 2차 근사가 1차를 이기는 비율, 잔차의
  $O(\|\Delta\|^3)$ log–log 스케일링 기울기. (GPT-2 스모크는 노이즈 바닥이라 지수 인증 불가.)
- **실행**: `research-wiki/survey/irds-fl-math-rigor-2026-07/RUN_1B.md`를 **그대로** 따를 것
  (`measure_taylor_residual.py` 사용; ~55–75분).
- **산출물**: 잔차 CSV + 요약(2차-우위 비율, 기울기) rundir 영속.

### 2.2 E3 — 비용 회계 통일 스모크 (스크립트 작성 필요)

- **목적**: §6.4 비용 표의 회계 통일 — 특히 **Ripple의 비용을 공정한 회계로 실측**해 논문에
  병기 가능하게 만들기(현재 Ripple 비용은 회계 불일치로 논문에서 인용 보류 상태).
- **스펙**: `research-wiki/survey/cost-comparison-methodology-2026-07/cost-comparison-methodology.md`
  — ① 로그(궤적) 생성 시간과 valuation 시간 분리 계측 ② Ripple의 자기-궤적 요구는 별도
  행으로 분리 ③ peak memory 병기 ④ individual-utility(loss-heur) 구현의 ~2× 과대측정 교정.
- **실행**: CNN 트랙(track_c1) 1–2셀에서 전 방법 통일 계측 스모크(~30분).
- **산출물**: `runs/` 신규 timing rundir + 방법별 (traj_s, valuation_s, peak_mem) CSV.

### 2.3 probe seeds 1–2 — 단일-seed 인용 수치의 3-seed화

- **목적**: 논문이 seed 0 단독으로 인용 중인 세 수치의 확정.
- **실행** (`runs/probe_signal/` 러너 재사용, 신규 seed만):
  1. **std50k5 r16, seeds 1–2** (N=50, 5/round, R=200): §6.1의 "Flirds/first-order +1.00"
     3-seed화. 셀당 수 시간 ×2.
  2. **anchor5 lr 격자 중 인용분**(lr∈{1e-3, 2e-3, 3e-3}, steps=10), **seeds 1–2**:
     §6.3(ii) "lr은 φ 크기만 키우고 cross-seed 신호는 못 만든다" 확정(예측: xseed ρ≈0 유지).
     anchor5 R=30이라 저렴.
  3. **noise probe(val bootstrap), seeds 1–2**: §6.3 "노이즈 원인의 분리" 문단(재표집 ρ 0.93
     등) 보강. 학습 재사용 + bootstrap 재계산이라 저렴.
- **산출물**: 기존 probe rundir 스키마 그대로, seed 차원만 추가. cross-seed ρ 재계산 포함.

### 2.4 E7 — delta free-rider 스트레스 (구현 ~20줄 + 1셀)

- **목적**: "free-rider φ=0 exact"가 zero/random-update에서만 성립하는지, **직전 글로벌
  집계를 재활용해 보내는(delta) free-rider**도 잡는지. 결과가 어느 쪽이든 논문에 정직 기재
  (실패 시 §6.5·§7의 zero/random 스코프 문장을 실측 근거로 확정).
- **구현**: `codes/flirds/fl/llm_server.py:37` 시그니처에 직전 글로벌 상태 threading(~20줄)
  + `corruptors.py`에 delta 모드 추가.
- **실행**: 1B cross-silo non-IID(silo5) 1셀, 3-seed (~3h). poison 아님 — free-rider 변형임.
- **산출물**: phase2_matrix 스타일 rundir(신규 threat 이름 `frdelta`), raw φ 포함.

### 2.5 E4 — Fed-LOO 본표 백필 (표 1의 마지막 빈칸)

- **목적**: 표 1(같은-게임 비교: Flirds/first-order/individual utility/**Fed-LOO**)의 두 칸.
  Fed-LOO 구현은 `codes/flirds/oracle/in_run_sv.py:70–98`에 있고 합성 검증 완료(brute force
  대비 max-diff 0.0) — 학습 로그(델타) 미영속이라 백필 불가, 재실행 필요.
- **실행**: track_d 1B 두 레짐(std20, anchor5) **경량 재실행** — coalition baseline 전부 off,
  methods = (b) exact + Fed-LOO + Flirds(+first-order)만, 3-seed. ~6–9h.
- **산출물**: 신규 rundir root(기존 track_d canonical 덮어쓰기 금지) + `make_fidelity.py`
  재집계로 Fed-LOO 행 산출.

### 2.6 E5 — (b) N=10 exact in-run, 1B 3-seed

- **목적**: N=5 순위 일치의 우연성(1/120/seed) 보강 — LLM에서 $2^{10}$ 전수 열거 fidelity.
- **실행**: N=10 전원 참여 스테이지(anchor5 빌더의 N=10 확장; IID Alpaca 동일 레시피),
  (b) exact $2^{10}$ + 같은-게임 방법들(Flirds/first-order/individual utility/Fed-LOO),
  3-seed, ~10h. paper-readiness G7-1 "실행 권고" 항목.
- **산출물**: 신규 rundir + fidelity CSV (+ target self-stability 열).

## 3. (선택) 같은 세션에서 가능한 무GPU 부수 작업

- **oracle noisy AUROC 0.604 vs 0.660 불일치 조정** — `runs/phase2_matrix/analysis/00_overview/master_metrics.csv`에서 확정, overview §3.4.2 각주 정리.
- **bootstrap CI(B=1000)** — 기존 rundir 재집계로 헤드라인 표 CI 산출(§5 평가 지표 TODO).
- **momentum 열화 수치(CNN 0.73 vs 0.81)의 정본 rundir 위치 확인** — §5 🟣VERIFY 해소.

## 4. 완료 보고 형식

실험별로: rundir 경로 + 핵심 수치 1–2줄 + overview 반영 여부 + paper-ko의 어느 마커가
해소되는지. 전부 완료 시 `paper/paper-ko.md`의 🔴TODO는 E-번호 기준으로
{E2, E3, E4, E5, E7, E11(probe seeds)}가 닫힌다.
