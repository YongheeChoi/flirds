---
type: conversation
date: 2026-07-04
topic: flirds
participants: [Yonghee, Claude]
tags: [verification, audit, math-rigor, ripple, precision, cost, deferred-inventory, pre-paper]
---

# 논문화 전 검증·감사 5건 (항목 1·2·3·4·6)

Yonghee가 `PROMPT_VERIFICATION_SURVEY_2026-07.md`(직전 세션이 작성한 통합 프롬프트)를
새 세션에 전달, ultracode로 실행. 실험 그리드 재실행이 아니라 **감사·수학 검증·소형 재현
측정** 중심. 산출물은 `research-wiki/survey/` 아래 5개 폴더 + 통합 overview. **코드·rundir·
git-modified 위키 미수정, git commit/push 없음** (Yonghee 검토 후 직접). 원 6건 중 항목 5(ΔW
계산 방식)는 직전 세션 코드 조사로 해소되어 제외.

## 실행 방식

- **정찰 16 에이전트 병렬**(워크플로): IRDS 이론 추출, estimator/oracle/프로토콜 코드 감사,
  Ripple 논문·코드 해부, dtype 전수 감사, cost 타이머 감사, deferred 3-소스 스윕(위키·raw·코드),
  원격 환경 정찰, 선행 6편 비용 절 원문 조사. 결과는 스크래치 노트로.
- **문서화 2 워크플로**(작성→적대적 검증→수정 파이프라인): 항목 1은 명제별 반박 패널(8명제
  ×2렌즈 + IRDS/코드 감사 2 = 18 에이전트)→개정. 항목 2/3/4/6은 작성→사실검증→fix.
- **소형 실측 2**(원격 CPU-only, GPU 캠페인 비간섭): Taylor 잔차 gpt2 스모크, Ripple eigsh 진단.
- 세션 중 Fable 5 한도 도달로 여러 하위단계 중단 → **Opus 4.8로 전환 후 반박 패널·fix·개정
  재실행**해 완주. 원격 스모크 2건은 한도 전 완주(데이터 회수 완료).

## 핵심 발견 (항목별)

**항목 1 (수학)**: estimator φ_k = Σ_r p_k^r[⟨g,δ_k⟩+½⟨δ_k,H·ΔW⟩]와 (b) in-run oracle이
**정확히 같은 고정가중 게임**(둘 다 p_k^r=n_k/Σ_{P_r}n 이 S-비의존)임을 코드로 확정. S-의존
재정규화 게임은 (a) retrain oracle에서만 실현. 명제 P1–P8 정리:
- P1 per-round 분해(null-player 보조정리 완전증명) — gpt2 스모크 per-round=2^N 3.93e-7 확인.
- P2 quadratic 닫힌형 = ½p_k⟨δ_k,HΔW⟩가 1-HVP 구현과 항별 일치 — closed=flirds_values 5.83e-12.
- P3 Taylor 잔차 O(‖Δ‖³) — **IRDS 원문은 비형식(O(η²) 한 문장, smoothness 가정 전무)**이라
  우리가 형식화. 물리적 크기·2차 우위는 gpt2 노이즈 바닥이라 미검증(1B 필요).
- P4 per-sample→per-client: 2-additive merge-consistency 보조정리로 1-step 극한 정확 가산 +
  어긋남 3개(K>1 granularity=게임 선택, minibatch, 3차 이상). 반박 패널이 P4b 분모상쇄가
  CNN mean-CE 인공물이고 LLM token-mean에선 미성립임을 지적 → caveat 추가.
- P5 고정가중 vs 재정규화: 등n·비가산 반례로 순위 뒤집힘 실증(δ₁=(1,0),δ₂=(0,2)).
- P6 path-dependence: (a)/(b) 괴리 3축 분해; **현행 +1.000/+0.900은 축퇴 레짐(near-additive·
  등n)+N=5 저검정력 산물이라 path-independence에 무정보** (반박 패널 최중요 지적).
- P7 momentum=0: 클라 stateless라 클라-수준 게임은 momentum 무관; load-bearing은 1-step 브리지
  + server 무상태성. P8 LoRA 부분공간 무해.

**항목 2 (Ripple)**: 논문 62×/49×는 소형 모델(MNIST MLP/CIFAR CNN)·"FL 학습 포함 누적시간"을
느린 coalition-평가형 베이스라인(AFedSV+/FedSV) 대비로 잰 것. **Ripple 절대 오버헤드는 plain의
2.05×**, N 미명시. 우리 1B N=5 측정(FedSV 대비 ~8.5× 느림)과는 스케일·베이스라인·회계축이
전부 다름. **사전 CPU-spin 가설(tol=0→fp32 기계정밀 미달→maxiter 소진→spin)은 CPU 실측으로
반박**: eigsh 12/12 정상 수렴(n_matvec~115, maxiter=300 미도달), well-conditioned(|λ|max≈1.019),
tol=0은 tol=1e-3 대비 matvec ×1.95(비효율이나 폭주 아님). 비용 실체 = "정상 수렴 eigsh를
클라×라운드만큼 반복 × matvec + per-step val-grad". from-logs 재구성 **불가**(클라 로컬
Hessian·per-sample grad 필요). 구간 분리: valuation 99.4% / 로그생성 0.4% (CNN 스케일).

**항목 3 (정밀도)**: 런타임 실측 — matmul **진짜 fp32**(allow_tf32=False, f32mp=highest),
cuDNN conv **TF32 기본 on** → CNN 트랙 노출 가능(B200 확인, yonsei 박스 미확인). (a) oracle
러너 **기본 bf16**(A_DTYPE=fp32 opt-in) 발견 — headline은 fp32(06-07 task6 교훈). protocol §1
"학습 bf16"은 코드(전부 fp32)와 불일치 — 타임라인 추적. 옵션 ①/② 대칭 서술. ×3.1 배수는
bf16 벤치 부재라 "미검증 placeholder"로 정직화.

**항목 4 (deferred 인벤토리)**: 3-소스 스윕으로 총 인벤토리화(A–E 카테고리, 총 21건+참고).
핵심 질문 위계로 우선순위. 최상위: (a)/(b) N=5·축퇴 갇힘, Fed-LOO 수치 부재, delta free-rider·
PGD poison 자인 위험 미실행, poison orientation 문서 상충(I-29).

**항목 6 (cost)**: 우리 회계="공유 로그 위 valuation-only wall-clock", 유일 예외 Ripple.
선행 7편 원문 조사 결론 — **valuation-only wall-clock 표준 관행 사실상 부재**(FedSV·ShapleyFL·
FLDetector 시간실측 0, IRDS throughput만, ComFedSV만 wall-clock figure, 대부분 asymptotic).
→ Flirds의 wall-clock 보고가 오히려 선행보다 엄격. 16-방법 프로파일 표 + caveat + 보고
프로토콜 권고. 부수 발견: loss-heur ~2배 과대측정(base-loss 캐시 없음, in_run_sv.py:64).

## Yonghee 결정 대기 (취합은 overview 참조)

실행 스코프 4건((a)/(b) N=10, Taylor 1B, free-rider/PGD, TF32/bf16 검증) + 문서·프레이밍 6건
(정밀도 정책, 부호 규약, Ripple 수위, 비용표 구조, poison orientation, 국소 코드 이슈).

## 미해소 (GPU 대기; 이 세션 CPU-only 규약)

항목 1 §7 물리 잔차(1B), 항목 3 CNN TF32 A/B·bf16 검증, 항목 6 로그생성 시간·×3.1·peak-mem.
전부 소형, 커맨드·스크립트 준비됨(measure_taylor_residual.py + RUN_1B.md, instrumented_ripple_cnn.py).
GPU0-2는 probe std50k5 완주(ETA 07-05 저녁~06) 후 해방. 원격 작업물은 신규
`flirds_verify_scratch/`(원본 저장소 무수정).

## 산출물

`research-wiki/survey/` 아래 5폴더 + `2026-07-verification-overview.md`. 각 문서 말미에
반박/검증 처리 로그 표. 스크립트·측정 원자료는 각 폴더 내. 프롬프트 파일은 완료 후 삭제.
