# Ripple eigsh CPU 진단 — 실측 결과 (2026-07-04)

> 항목 2 과업 3·4의 실측 파트. **사전 가설(tol=0 → fp32에서 기계정밀 수렴 도달 불가 →
> maxiter 소진 → ARPACK/BLAS CPU-spin으로 "멈춘 듯" 보임)은 이 실측으로 반박됐다.**
> eigsh는 전 호출 정상 수렴하며, Ripple 비용의 실체는 stall이 아니라 "정상 수렴하는 다수
> eigsh 호출(클라×라운드) × matvec 비용 + per-step val-grad"이다.

## 0. 측정 셋업

- **계측 사본** (원본 미수정): `research-wiki/survey/ripple-audit-2026-07/instrumented_ripple_cnn.py`,
  `stall_probe.py`. 원본 `codes/flirds/baselines/ripple.py`의 eigsh 스펙(tol 미전달=0,
  which=LA, v0=고정, ncv=2k+1, maxiter, 비수렴 시 ncv=4k+1 재시도)을 그대로 재현하되
  matvec 카운터·구간 타이머·tol/maxiter 민감도 프로브를 덧붙인 것.
- **환경**: 원격 dongjun-dongha 노드, **CPU-only** (`CUDA_VISIBLE_DEVICES=`, GPU 캠페인과
  비간섭), OMP_NUM_THREADS=8, torch 2.12.0+cu130 / scipy 1.17.1 / numpy 2.4.6.
- **모델**: 원 CNN 트랙 모델 (depth=5, width=1.0), 학습 파라미터 **P=61,706**, k=8, m=16.
  합성 데이터(랜덤 텐서) — stall 메커니즘은 데이터 실물과 무관.
- **config**: n_clients=4, rounds=3, epochs=1, batch=64, lr=0.05, n_val=256.
  `traj_maxiter=300` (포트 기본 1000에서 벽시계 예산 위해 축소; tol=0/which=LA/v0/ncv-retry
  의미는 보존). 원자료: `run1/traj_sections.json`, `run1/eig_results.json`,
  `run1/eigsh_calls.jsonl`, `spin8/spin_omp8.json` (스크래치 사본 `scratchpad/meas/`; 원격 노드·세션 스크래치 전용 — 저장소 미커밋).
- **한계 (문서에 명시)**: (i) CPU·CNN 소형(P=6.2e4) 스케일 — LLM 포트는 GPU·P≈1e9,
  matvec이 전모델 HVP라 per-matvec 비용이 10²배↑. 여기서 확정한 것은 *메커니즘*(수렴 여부·
  matvec 카운트·tol 거동)이지 4515s LLM 절대치가 아니다. (ii) GPU HVP의 H2D/D2H 왕복은 이
  CPU 실측으로 관측 불가(LLM 포트 전용 비용). (iii) OMP=1 대조는 한도 도달로 미완 —
  OMP=8만 확보.

## 1. 핵심 판정: eigsh는 stall하지 않는다

`run1`의 궤적 12개 eigsh 호출 전부 **정상 수렴**:

| 지표 | 값 | 함의 |
|---|---|---|
| eigsh 호출 status | 12/12 `converged` | 비수렴·maxiter 소진 **0건** |
| 호출당 n_matvec | 106–129 (평균 ~115) | maxiter=300 cap(=ARPACK 재시작 수)의 근처도 아님 |
| 연산자 스펙트럼 반경 \|λ\|max | **1.0188** (power iteration) | well-conditioned — ARPACK가 빨리 수렴하는 이유 |
| matvec fp32 평균 | 6.86 ms (min 6.72 / max 7.11) | P=6.2e4 CPU 기준 |
| fp32 vs fp64 상대오차 | ~3.5e-7 | fp32 노이즈 바닥 (연산자 정상) |
| fp32 반복 결정성 | absdiff 0.0 | matvec 결정적 (비수렴 원인 아님) |

**maxiter 의미 확정** (프로브 `maxiter_semantics`): maxiter=3 → 57 matvec에서 `no_convergence`,
maxiter=10 → 109 matvec에서 `converged`. 즉 scipy `maxiter`는 **ARPACK 재시작(restart)
반복 수**이고 재시작 1회당 ≈ ncv(=17) matvec. 따라서 포트의 maxiter=1000(CNN)·300(LLM)은
사실상 ~1.7만~5천 matvec 상한 — 이 well-conditioned 연산자에서는 **결코 도달하지 않는다.**
사전 가설이 상정한 "tol=0이라 수렴 판정을 못 받아 maxiter까지 감"은 이 config에서 발생하지
않았다.

## 2. tol=0의 실제 대가: 폭주가 아니라 ~2배 matvec

같은 연산자에 tol만 바꿔 반복 (전부 `converged`):

| tol | n_matvec | wall(s) | vs tol=1e-3 |
|---|---|---|---|
| **0 (현행 포트)** | 109 | 0.85 | ×1.95 |
| 1e-8 | 71 | 0.55 | ×1.27 |
| 1e-6 | 62 | 0.48 | ×1.11 |
| 1e-3 | 56 | 0.43 | ×1.00 (기준) |

tol=0은 tol=1e-3 대비 matvec을 **약 2배**로 늘리지만 **여전히 수렴**한다. 사전 가설이
예측한 "maxiter 소진(수천 matvec)"과는 자릿수가 다르다. 즉 tol=0은 *실재하는 비효율*이되
*병리적 stall은 아니다* — 후속 개선(§5)에서 tol 완화로 eigsh 비용을 절반 가까이 줄일
여지가 있음을 뜻한다.

부가 프로브:
- `tol0_ncv4k1` (비수렴 시 재시도 경로의 ncv=4k+1=33): 104 matvec — 수렴 시엔 오히려
  비슷하거나 약간 적음. 재시도 fallback이 "일을 배가"하려면 **먼저 비수렴이 나야** 하는데
  이 연산자에선 그 트리거가 안 걸린다.
- `tol0_k20` (k=8→20): 184 matvec — 요청 고유쌍 수에 선형 증가(LLM 포트 k 관련).
- `tol0_fp64op` (연산자 dtype fp64 선언): 동일 109 matvec이지만 wall **2.64s vs fp32 0.85s
  (×3.1)**. 우리 포트의 LinearOperator는 dtype=fp64로 *선언*되지만 실제 matvec은 fp32라
  (ripple_code 노트 확인) fp32 속도로 도는 것 — 만약 선언대로 fp64 matvec이었으면 3배
  느렸을 것.

## 3. 구간 분리 계측 — 회계 통일용 (과업 4)

`run1/traj_sections.json` (n=4, R=3, 자체 FedAvg 궤적 포함):

| 구간 | 시간(s) | 전체 대비 |
|---|---|---|
| **A. 로그 생성** (local_train 0.048 + agg 0.002) | **0.050** | 0.4% |
| **B. valuation** (아래 합) | **12.592** | **99.4%** |
| — eigsh wall | 11.701 | 92.4% |
| —— (그중 matvec_sum) | (10.103) | (79.7%) |
| — round val-grad | 0.612 | 4.8% |
| — drop val-grad | 0.103 | 0.8% |
| — QR | 0.147 | 1.2% |
| — dW/HVP prep + Eq16–19 chain | 0.028 | 0.2% |
| 미계측 잔차 | 0.028 | 0.2% |
| **총계** | **12.67** | 100% |

**회계 함의**: 다른 방법과 통일된 "valuation-only" 회계로 Ripple을 재면, 이 CNN 스모크에서
**로그 생성(A)은 전체의 0.4%에 불과**하다 — 즉 러너(`track_c1._timed`)가 Ripple의 자체
궤적까지 타이머 안에 넣어 생기는 회계 비대칭의 크기는 이 config에서 미미하다. Ripple이
비싼 진짜 이유는 자체 궤적 재실행이 아니라 **valuation 자체가 비싸기 때문**이고, 그 92%가
eigsh(=클라×라운드 만큼의 정상 수렴 고유분해), 나머지 대부분이 per-step/round val-grad다.
(주의: 이 비율은 CNN P=6.2e4·CPU 것. LLM 포트에서는 자체 궤적의 로컬 학습이 훨씬 무거워
A 비중이 커지지만, ripple_code 노트가 확인했듯 LLM 자체 궤적조차 공유 로그 궤적보다 짧아
4515s의 지배 비용은 여전히 eigsh+val-grad로 남는다.)

## 4. 스레드 거동 (spin8, OMP=8, 162 연속 호출)

- 162/162 `converged`, 전부 n_matvec=109 (연산자 고정 → 결정적).
- 총 17,658 matvec / 169.35s wall (그중 matvec 157.73s = 93%), **104 matvec/s, 평균 8.93ms/matvec**.
- 호출별 wall은 0.78–1.46s로 요동 — matvec_sum과 강상관(wall≈matvec_sum+arpack_resid),
  즉 시간은 ARPACK 오케스트레이션이 아니라 **matvec 자체**에 쓰인다.
- OMP=1 대조는 미완(한도) — "BLAS 스레드 spin-wait가 겉보기 stall을 만든다"는 가설의
  스레드 파트는 **미검증**으로 남는다. 다만 §1–3에서 애초에 stall이 없으므로(전 호출 <1.5s
  수렴) spin-wait가 설명해야 할 현상 자체가 이 config엔 없다.

## 5. 결론과 후속(설계만; 수정은 범위 밖)

**판정**: 사전 CPU-spin 가설은 이 스케일에서 **반박**. Ripple의 비용 구조는
"stall/무한반복"이 아니라 **정상 수렴하는 eigsh를 클라×라운드 횟수만큼 반복 + 매 로컬
스텝 val-grad**의 곱이다. tol=0은 matvec을 ~2배로 늘리는 실재 비효율이나 병리는 아니다.
LLM 4515s는 (P≈1e9로) per-matvec이 10²배 비싼 같은 메커니즘의 스케일업으로 설명되며,
"방법 고유 비용"(고유분해 volume)이지 "우리 포트의 버그성 stall"이 아니다.

**후속(설계만)**:
1. **tol 완화** (tol=1e-3): §2에서 matvec ~1.95→1.0로 eigsh(=지배 비용) 절반 근접 절감
   기대. fidelity 영향은 top-k 고유쌍 정밀도에 달렸으므로 별도 검증 필요.
2. **LLM 스케일 재현**: 이 계측 사본을 GPU·1B에 이식해 per-matvec HVP 비용·수렴 반복수를
   실측하면 4515s의 항별 분해가 가능(GPU 필요 → GPU 캠페인 종료 후).
3. **OMP=1 대조**: spin-wait 가설을 완결하려면 OMP∈{1,8,36} × 같은 연산자 반복(수 분) —
   현재는 stall 부재로 우선순위 낮음.
