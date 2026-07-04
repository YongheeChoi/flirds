# 논문화 전 검증·감사 5건 — 통합 요약 (2026-07)

작성 2026-07-04. 근거 코드 스냅샷 main @ 004d076. 실측은 GPU 캠페인(GPU0-2 probe)과
비간섭하도록 원격 **CPU-only**로 수행. **코드·rundir·기존 위키 미수정, git commit/push 없음**
(Yonghee 검토 후 직접). 이 세션은 실험 그리드 재실행이 아니라 감사·수학 검증·소형 재현 측정.

## 항목별 산출물과 한 줄 결론

| 항목 | 폴더 | 한 줄 결론 |
|---|---|---|
| **1. IRDS→Flirds 수학 엄밀성** | [`irds-fl-math-rigor-2026-07/`](irds-fl-math-rigor-2026-07/irds-fl-math-rigor.md) | 명제 P1–P8을 가정·증명·코드대응·IRDS대응·성립경계까지 논문 수록급으로 정리. **estimator와 (b) oracle은 정확히 같은 고정가중 게임**(2차 Taylor 절단만 차이). 적대적 반박 패널(18)로 major 18건 교정. gpt2 스모크가 P1·P2의 **대수적 정확성을 1e-12로 확증**(closed=flirds_values 5.83e-12, per-round=2^N 3.93e-7). 물리적 잔차(P3 2차 우위)는 1B 필요(스크립트 준비, GPU 대기). |
| **2. Ripple 감사·속도** | [`ripple-audit-2026-07/`](ripple-audit-2026-07/ripple-audit.md) + [측정](ripple-audit-2026-07/measurements-eigsh-cpu.md) | 논문 62×/49×는 **소형 모델·"FL 학습 포함 누적시간"을 느린 coalition-평가형 베이스라인 대비**로 잰 것(N 미명시, Ripple 절대 오버헤드는 plain의 2.05×) — 우리 측정과 스케일·베이스라인·회계축이 전부 달라 수백배 격차 설명됨. **사전 CPU-spin 가설은 CPU 실측으로 반박**(eigsh 12/12 정상 수렴, well-conditioned |λ|max≈1.02, maxiter 미도달; tol=0은 ×2 비효율이나 폭주 아님). from-logs 재구성 **불가**(클라 로컬 Hessian·per-sample grad 필요). |
| **3. 정밀도(fp32)** | [`precision-policy-2026-07/`](precision-policy-2026-07/precision-audit-and-policy.md) | 런타임 실측으로 **matmul 진짜 fp32 확정**(allow_tf32=False, f32mp=highest), **cuDNN conv는 TF32 기본 on** → CNN 트랙 노출 가능(B200 확인; yonsei 박스 미확인). (a) oracle 러너 **기본값 bf16**(A_DTYPE=fp32 opt-in) 발견 — headline 런은 fp32(task6 교훈). 옵션 ①(fp32 유지)/②(학습만 bf16)를 대칭 서술. protocol §1 문서-코드 불일치 타임라인 정리. |
| **4. 미뤄둔 엄밀 검증 인벤토리** | [`deferred-rigor-inventory-2026-07/`](deferred-rigor-inventory-2026-07/deferred-rigor-inventory.md) | 위키·raw·코드 3소스 스윕으로 deferral **총 인벤토리화**(카테고리 A–E), 핵심 질문 위계(1차 fidelity > 2차 성능→수렴→탐지)로 우선순위. 최상위 리스크: (a)/(b) 검증이 N=5·near-additive·등n 축퇴 레짐에 갇힘, Fed-LOO baseline 수치 부재, delta free-rider·PGD poison 자인 위험 미실행. |
| **6. cost 비교 방법론** | [`cost-comparison-methodology-2026-07/`](cost-comparison-methodology-2026-07/cost-comparison-methodology.md) | 우리 회계 = "공유 로그 위 valuation-only wall-clock"; **유일 예외 Ripple**(자체 궤적 포함)만 회계 비대칭. 선행 7편 원문 조사: **valuation-only wall-clock 표준 관행 사실상 부재**(FedSV·ShapleyFL·FLDetector 시간 실측 0건, IRDS는 throughput만, ComFedSV만 wall-clock figure). 16-방법 비용 프로파일 표 + 논문 caveat 목록 + 보고 프로토콜 권고. |

## 세션 중 수행한 소형 실측 (CPU-only, 원격)

- **Taylor 잔차 gpt2 스모크** (항목 1 §7): `measure_taylor_residual.py`로 FL 궤적 위 전 2^5 부분집합
  실제 u vs 1차/2차 Taylor + 닫힌형 φ 비교. **기계 검증 통과**(P1·P2), 물리 검증은 1B 대기
  (`RUN_1B.md`에 커맨드·비용 준비 — GPU 해방 시 즉시 실행 가능).
- **Ripple eigsh 진단** (항목 2 §3·§4): 계측 사본으로 궤적 구간 분리(B=valuation 99.4%/A=로그생성
  0.4%) + tol/maxiter 민감도 + matvec 카운트. 사전 가설 반박이 이 실측의 핵심 산물.

## Yonghee 결정 필요 — 취합 (항목별 상세는 각 문서 말미)

**A. 실행 스코프 (실측 필요)**
1. **(a)/(b) N=10 oracle 실행 여부** (항목 4·1): (b) N=10 exact ≈10h는 비용·효과상 실행 권고(승인만),
   (a) N=10은 샤딩 신규 작성+11–22h/4-GPU라 스코프 결정 사안. 미실행 시 "N=5 dual-GT + 대규모
   (b)-fidelity"로 논문 스코프 한정. **(a) vs (b) 판별력은 축퇴 레짐 밖(비등n·비가산·비IID) 셀에서만
   유효 검증됨** — 현행 +1.000/+0.900은 무정보(항목 1 §9, 항목 4).
2. **§7 Taylor 1B 실측 승인** (항목 1): fresh run 필요(로그 미영속), 비용 낮음. GPU 해방 후 즉시.
3. **delta free-rider·PGD poison 실행 여부** (항목 4): 자인 위험 2건 — 논문 전 필수로 볼지(문서는 강권).
4. **CNN TF32 A/B + bf16-train 검증** (항목 3): 옵션 ② 채택 가능 여부를 가르는 실험. yonsei 박스
   런타임 판별(~1분) 선행.

**B. 문서·프레이밍 (실험 불요)**
5. **정밀도 정책** (항목 3): 옵션 ①(fp32 유지 + protocol §1을 코드에 맞게 개정) vs ②(검증 후 학습 bf16).
   문서는 판단 재료만 — 비용/확장성 vs 검증부담/비교단절.
6. **부호 규약 논문 표기** (항목 1): φ<0=유익(내부) 유지 vs 반전 — 어느 쪽이든 명시적 선언 필요.
7. **Ripple 논문 처리 수위** (항목 2): fidelity 표 제외 + runtime 표에 회계 각주 유지(문서 권고) vs 부록.
8. **비용 표 구조** (항목 6): valuation-only + end-to-end 2단 분리(권고) vs 단일표+각주.
9. **poison orientation 헤드라인** (항목 4 I-29): matrix(φ) vs D2b(−φ) 프레이밍 확정 + 상충 문서 정정.
10. **(a) oracle 기본 bf16 / loss-heur 2배 과대측정 / timing.json 미구현** 등 국소 코드 이슈 처리
    방향(항목 3·6): 각주 vs 본런 전 수정. 코드 수정 금지 규약상 문서는 제안만.

## 후속 실험 — 공유 실측 패키지 (여러 항목이 한 run으로 해소)

- **패키지 1 (하루 급 oracle 회수, ~25–35h GPU)**: (b) N=10 exact 3-seed + 3B (a) 3-seed rundir
  영속화 + α=0 (b)-perround anchor — 항목 1·4의 P1 최상위 다건 동시 해소.
- **2026-06-06 표 조건 스모크 재측정** (항목 6·2·3 공유): 1B N=5 R=10에서 로그 생성 시간 명시 측정
  + Ripple 분리 계측 valuation-only 환산 + fp32-vs-bf16 마이크로벤치(×3.1 확정) + peak-mem. **1셀 ~30분급.**
- **§7 Taylor 1B** (항목 1): P3 물리 잔차·P5 순위. 스크립트 준비 완료.

## 미해소 실측 (GPU 대기 — 서버 세션에서 실행)

인수인계 문서: [`PENDING_MEASUREMENTS_2026-07.md`](PENDING_MEASUREMENTS_2026-07.md) — 실측 3건의
커맨드·산출물 회수·결과 반영 지점까지 정리. 요약:
1. **Taylor 잔차 1B** (항목 1 §7): 준비 완료, 커맨드만 붙여넣으면 됨(~55–75분). P3 물리 잔차·2차 우위·P5 순위.
2. **통일 회계 스모크** (항목 6·2·3 공유): 1B N=5 R=10에서 로그생성 시간·Ripple 분리계측·×3.1 확정·peak-mem(~30분). 스크립트 서버에서 마저 작성.
3. **CNN TF32 A/B** (항목 3): yonsei 박스 런타임 판별(~1분) + cifar10 A/B(~1–2.5h). 스크립트 yonsei 세션에서.

GPU0-2는 probe std50k5 완주(ETA 07-05 저녁~06) 후 해방, GPU3는 공유 계정 외부 잡 간헐 점유,
GPU4-7은 프로젝트 규약상 미사용. 안 돌려도 각 문서는 placeholder로 완결돼 있음.
