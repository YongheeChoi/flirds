# Flirds 논문화 전 검증·감사 5건 — 통합 프롬프트 (2026-07)

> 이 프롬프트를 새 세션에 그대로 전달. ultracode로 실행 예정 — 항목 병렬화, 수학 명제의
> 적대적 독립 검증(반박 패스), 문헌 fan-out 조사에 워크플로를 적극 활용할 것.

## 역할과 목적

너는 flirds 저장소에서 **논문화 전 엄밀성 검증 감사** 5건을 수행한다. 실험 그리드 재실행이
아니라 **감사·진단·수학 검증·문서화**가 중심이다 (예외: 항목 1의 수치 실측, 항목 2·6의
소형 재현 측정 — 스모크 급만). CLAUDE.md와 memory가 로드되어 있지만 범위·규약은 이
프롬프트가 우선한다.

항목 번호는 1·2·3·4·6 — 원 검증 목록 6건 중 **항목 5(ΔW 계산 방식 확인)는 사전 세션의
코드 조사로 해결되어 제외**됐고, 그때 확정된 FL 프로토콜 사실은 항목 1의 재료 절에
수록되어 있다 (항목 1 문서의 시스템모델 절로 명문화하면 됨).

배경 한 줄: Flirds = IRDS("Data Shapley in One Training Run", Wang et al., arXiv:2406.11011)의
1차+2차 Taylor 기반 data Shapley를 FL **클라이언트 단위·라운드 단위**로 변형한 기여도 측정
방법. Phase 2 real grid 25셀 완료(2026-06-15), 현재 신호크기 진단 probe 캠페인 진행 중.

## 공통 규약

- **산출물 위치**: 항목별로 `research-wiki/survey/` 아래 개별 폴더 생성 (각 항목에 제안
  이름 명시). 메인 문서는 **한국어**, 담백한 팩트 전달 톤(세일즈 톤 금지), **모든 수치에
  설정·출처 명기** (file:line, rundir 경로, 논문 절 번호).
- 각 문서 말미: "Yonghee 결정 필요" 절(해당 시) + "후속 실험 제안" 절. 사소한 결정은
  스스로 내리고 근거를 남길 것; 막히는 결정만 모아서 보고.
- **코드 수정 금지** (발견한 버그·개선점은 문서에 제안으로만). 예외: 계측(타이밍 분리
  측정) 목적이면 **원본은 그대로 두고 사본/래퍼 스크립트**를 survey 폴더나 스크래치에
  만들어 사용. **git commit/push 금지**
  (Yonghee가 검토 후 직접). 기존 rundir·`runs/` 산출물·git-modified 상태인 위키 파일
  (`survey/flirds-experiment-results-overview-2026-06-25.md`,
  `wiki/flirds-signal-size-diagnosis.md`)을 건드리지 말 것. 신규 파일은 survey 폴더나
  스크래치에만.
- **GPU**: 물리 0-3만 사용(4-7 절대 금지; CUDA_VISIBLE_DEVICES=0,1,2,3 설정됨). 현재
  `runs/probe_signal/` 파일럿(GPU3)과 β0.3→probe 캠페인(GPU0-2 자동 인계)이 돌고 있을 수
  있다 — `nvidia-smi`와 `runs/probe_signal/` 상태를 확인하고 **빈 GPU에서 스모크~소형
  config만**. B200 실제 OOM 라인 ~160GB (`memory.used`는 캐시 포함이라 OOM 기준 아님).
- 환경: `/home/korea_bupj/miniconda3/envs/flirds/bin/python`, 실행은 `codes/`에서
  `PYTHONPATH=.`.
- 아래 "사전 확인 사실"은 직전 세션의 조사 결과다. **라인 번호는 어긋날 수 있으니 인용
  전 재확인**하되, 같은 탐색을 처음부터 반복하지는 말 것.
- 마지막에 세션 기록 워크플로 수행: `research-wiki/raw/conversations/flirds/<date>-<slug>.md`
  + `wiki/log.md` conv 항목 (+ 필요시 plan 반영).

## 우선순위와 완료 기준

**1 (최우선·최대 분량) → 2·6 (측정·비용 클러스터, 서로 겹침) → 3 → 4.** 항목 간 독립이라
병렬 가능. 항목이 끝날 때마다 해당 폴더에 문서를 확정해라(중단돼도 부분 산출물이 남게).
전체 완료 시 `research-wiki/survey/2026-07-verification-overview.md`에 요약(폴더 링크 +
"Yonghee 결정 필요" 취합)을 작성.

---

## 항목 1 — IRDS→Flirds 수학적 엄밀성 (이론 + 수치 실측)

폴더: `research-wiki/survey/irds-fl-math-rigor-2026-07/`

**목표**: IRDS 논문이 검증한 사실들이 FL 변형(per-sample→per-client, per-step→per-round)
하에서도 성립하는지 수학적으로 검증하고, **차후 논문에 수록 가능한 수준**(명제·가정·
오차항·증명 스케치, 성립하지 않는 지점은 반례 또는 추가 가정)으로 정리. 이론 주장을
estimator 코드와 항별 대조하고, 잔차·근사 오차는 **수치로 실측**한다.

**재료**:
- 논문 본문 md: `research-wiki/raw/papers/flirds/Data Shapley in One Training Run.md`
- 소스노트: `research-wiki/wiki/sources/in-run-data-shapley.md`
- estimator: `codes/core/flirds_estimator.py` (fp32, true Hessian, 라운드당 HVP 1회)
- (b) in-run oracle: `codes/oracle/in_run_sv.py` (exact 2^N) + per-round exact
  분해(`in_run_shapley_perround`, task 7c에서 Δφ≈3e-16으로 2^N 오라클과 일치 수치 확인됨)
- (a) retrain oracle: `codes/oracle/exact_sv_llm.py`
- 기록된 변형점 4개: per-step→per-round, true Hessian(GGN 검증 후 기각), momentum=0,
  server-side held-out val.
- **확정된 FL 프로토콜** (사전 세션 코드 확인 완료 — 게임 정의의 기초; 문서에 시스템모델
  절로 명문화할 것): 동기식 FedAvg — 라운드 r에 `sample_frac` 균등 비복원 선발
  (`codes/fl/server.py`의 `_fedavg_core`), 참여자 전원 **현재 라운드 global w_r에서
  시작**, **delta = (K-스텝 로컬 SGD 후 파라미터 − w_r)** 즉 gradient 전송이 아닌 모델 차
  (`codes/fl/client.py`, `llm_server.py`), 참여자만의 n_c 가중 FedAvg(`w=n_c/Σ_{참여}n`,
  server lr 없음), 비참여자는 그 라운드 `deltas_map`에 부재(stale 업데이트 구조적 불가),
  FL 상태는 LoRA 파라미터만(requires_grad 필터), `logs=[(w_r, deltas_map)]`,
  `deltas_map[c]=(delta, n_c)`. 게임 정의에 필요한 범위에서 코드 재확인.

**검증 포인트 시드** (전수 아님 — 논문 md를 읽고 추가 발굴할 것):
1. **평가 단위 변경(per-sample → per-client) 자체의 정합성** (Yonghee 명시 요청):
   IRDS의 플레이어는 개별 학습 샘플이고, 기여가 "배치 gradient = 샘플별 gradient의
   평균"이라는 **정확한 선형 구조**로 스텝 utility에 들어온다. Flirds의 플레이어는
   클라이언트이고, 기여는 자기 데이터로 K-스텝 로컬 학습을 돌린 **비선형 산출물
   (delta)**이다. 이 대응에서 (i) 샘플-수준의 정확한 선형성이 클라-수준에서 무엇으로
   대체되는지(라운드 집계의 δ 가중합 구조), (ii) 클라 delta의 라운드 내 상호 독립성
   (전원 같은 w_r에서 시작, 로컬 학습은 타 클라와 무관)이 IRDS의 어떤 전제에
   대응하는지, (iii) "클라 = 샘플 집합"으로 IRDS를 액면 그대로 적용한 것과 우리
   정식화가 언제 일치하고(예: 로컬 1-step SGD 극한) 언제 어긋나는지를 형식화 —
   **단위 변경에서 오는 이론의 어긋남·추가 오차가 있으면 명시하고, 없으면 없음을
   근거와 함께 서술.**
2. **큰 스텝 Taylor 잔차**: IRDS는 per-sample 1 gradient step(이동량 ~lr) 전개, 우리는
   K-스텝 로컬 학습 누적 delta(라운드 이동량)에 전개. 2차 사용 시 잔차 O(‖Δ‖³)의 형식화
   + **실측**: 라운드별 실제 Δval-loss vs 1차/2차 Taylor 예측(클라 부분집합별).
3. **FedAvg 재정규화와 게임 정의**: 부분집합 S의 집계가 `Σ_{c∈S} n_c δ_c / Σ_{c∈S} n_c`
   (분모가 S-의존)이면 utility 게임이 δ에 비선형. estimator가 가정하는 게임과 (b) oracle이
   계산하는 게임이 **정확히 같은 게임인지** 코드로 확정하고, 경험적 near-additivity(신호
   진단 문서에 가산 갭 ≤0.9% 실측 있음 — 재계산 말고 인용)의 이론적 조건을 도출.
4. **per-round Shapley 합산의 정당화**: 라운드 r 참여자가 이후 궤적을 바꾸는
   path-dependence 하에서, IRDS의 per-step 합산 논리가 그대로 이전되는가.
5. **부분 참여**: 비참여 라운드 기여 0 처리(라운드별 2^{|P_r|} 게임 합산)의 공리적 정합성
   (수치 확인은 이미 있음 — 이론만 채우면 됨).
6. **momentum=0 가정의 형식화**: IRDS 유도의 SGD 가정 대응 관계, momentum 시 무엇이
   깨지는지 — limitation 서술용.
7. **LoRA 부분공간 전개**: utility는 전체 모델 함수지만 변화는 LoRA 좌표만(base 고정)
   — 무해함을 명시적으로.
8. **2차 닫힌형 ↔ 1-HVP 구현의 일치**: quadratic game의 Shapley 닫힌형(φᵢ = gᵀδᵢ +
   ½δᵢᵀH(Σδ) 꼴이 되는지부터 유도)과 `flirds_estimator.py` 코드의 항별 대조.
   null-player 공리(free-rider δ=0 → φ=0 정확 성립; 실험으로는 확인됨) 증명 포함.

**실측 규칙**: 기존 산출물 최대 재활용 (`runs/phase2_matrix/` rundir의 phi.parquet·metrics,
신호진단 문서의 실측치). raw logs `[(w_r, deltas_map)]`가 저장돼 있지 않으면 **1B N=5
R=10 스모크 급** fresh run으로 재생성 (1 GPU, 수십 분 내 규모로 제한 — 더 큰 게 필요하면
실행하지 말고 "후속 실험 제안"에 설계만). 실측 스크립트·표·그림은 survey 폴더 안에.

**품질 규약**: 수학 명제마다 독립 검증 패스(반박 시도)를 별도 에이전트로 돌릴 것.
"성립함"으로 끝내지 말고 성립 조건·깨지는 경계를 명시.

---

## 항목 2 — Ripple Shapley 구현 감사·속도 진단 (수정은 범위 밖)

폴더: `research-wiki/survey/ripple-audit-2026-07/`

**배경**: 논문 = "Ripple Shapley: Data Influence Attribution in One Federated Training Run"
(AAAI 2026). PDF: `research-wiki/raw/papers/flirds/40034-Article Text-44125-1-2-20260314.pdf`,
소스노트: `research-wiki/wiki/sources/ripple-shapley.md`. 논문 주장: **AFedSV+ 대비 62×,
FedSV 대비 49× 빠름**. 우리 측정(1B N=5 R=10 val=100, 2026-06-06 세션): Ripple ~4515s =
FedSV(~532s)보다 **~8.5× 느림** → 설명해야 할 격차가 수백 배 규모.

**사전 확인 사실**:
- 우리 구현 중 **Ripple만 자체 FedAvg 궤적을 처음부터 실행**(공유 로그 미사용) — 다른
  모든 방법은 공유 로그 소비 시간만 측정되므로 표 안에서 회계 기준이 유일하게 다름.
- 구조: 라운드마다 전 클라이언트의 **로컬 Hessian top-k를 scipy `eigsh`(ARPACK)로 분해**.
  matvec = GPU HVP(jvp∘grad) 콜백, Lanczos 반복은 CPU Fortran이 orchestrate. **tol=0**
  (기계 정밀도 수렴 강제), maxiter 1000 cap, ArpackNoConvergence 잡아서 ncv 키워 재시도.
  LLM은 k=8. 파일: `ripple.py`(CNN) / `ripple_llm.py`(LLM) — `grep -ri ripple codes
  --include=*.py -l`로 경로 확인. RIPPLE 게이팅은 track_c1의 `C1_RIPPLE` env에만 존재.
- 기록상 LLM 포트는 **미완 선언 상태** ("streaming projection, LA/LM 선택 미결정" —
  `research-wiki/raw/conversations/flirds/2026-06-06-sv-baseline-port-and-results.md` 및
  06-07 세션 노트). 현재 제외 근거 기록은 "dominated + flaky" 한 줄(wiki plan).

**과업**:
1. **충실도 감사**: 논문 pseudocode ↔ 우리 CNN/LLM 포트 대조표 — 어디까지가 원 알고리즘
   그대로고, 어디부터가 우리 포트의 미완/변형/단순화인지.
2. **속도 격차 규명**: 비용 구조 분해(rounds × clients × Lanczos 반복 × HVP 비용)와 논문
   62×/49× 주장의 측정 조건(그들의 모델 스케일, 베이스라인 구현 방식, 학습 포함 여부)을
   대조해 격차 원인을 항목화. "우리 포트가 느린 것"과 "방법 고유 비용"을 구분.
3. **eigsh CPU-spin 구체 설명**: tiny config로 stall 재현 + 프로파일(py-spy 또는 타이밍
   계측) → "무슨 일이 벌어지는가"를 구체적으로. 검증할 가설: tol=0 기계정밀 수렴이 fp32
   연산자에서 사실상 도달 불가 → maxiter 소진까지 반복, ARPACK CPU 구간 + BLAS 스레드
   spin-wait가 겉보기 "CPU만 돌며 멈춤"을 만들고, ncv 재시도 fallback이 일을 배가 —
   맞는지 확정하고 아니면 실제 원인 제시.
4. **회계 통일을 위한 분리 계측** (Yonghee 명시 요청 — "로그 찍는 시간을 제외한 순수
   valuation 시간" 산출): Ripple 루프의 **계측용 사본/래퍼**에 구간 타이머를 넣어
   (a) 로컬 학습·집계(다른 방법에서는 로그 생성에 해당해 표에서 제외되는 부분)와
   (b) valuation 연산(drop/delta 계산, eigsh 스케치 등)을 분리 측정하고, 다른 방법과
   동일한 **valuation-only 회계로 환산한 Ripple 수치**를 스모크 급에서 산출 (항목 6의
   통일 회계 기준과 결과 공유). 아울러 Ripple을 공유 로그 소비형(from-logs)으로
   재구성하는 것이 알고리즘 구조상 가능한지(로컬 Hessian 접근 등 로그 밖 정보 필요
   여부) **가능/불가 판정과 근거만** 문서화 — 구현은 범위 밖.
5. **제외 정당화 문서**: fidelity 비교에서 Ripple을 제외한 사유를 논문/리뷰어 방어
   수준으로 서술 (구현 충실도 판정 결과를 반영해 "우리 포트 한계" vs "방법 고유 비용"
   중 무엇인지 명시).

수정·복귀 시도(tol 완화, torch.lobpcg 대체, from-logs화 등)는 **범위 밖** — 가치가 있어
보이면 "후속 실험 제안" 절에 설계만.

---

## 항목 3 — 정밀도(fp32) 감사 + 정책 문서 (실험은 설계만)

폴더: `research-wiki/survey/precision-policy-2026-07/`

**사전 확인 사실**:
- 코드는 전 경로 fp32로 보임: 모델 로드 `dtype=torch.float32`
  (`codes/experiments/phase2_matrix.py:145`, `track_d.py:122`), SFTConfig
  `bf16=False, fp16=False`(`llm_server.py:46`, `phase2_matrix.py:206`), estimator가
  파라미터·delta `.float()` 캐스팅(`flirds_estimator.py:101,103`), Ripple HVP fp32,
  CNN 기본 fp32. **tf32 플래그를 만지는 코드 0곳.**
- 발견 2건: (i) `research-wiki/wiki/flirds-protocol.md` §1은 "클라 학습 bf16 + 평가·HVP
  fp32"로 명문화 ↔ 실제 구현은 학습까지 전부 fp32 = **문서-코드 불일치**. (ii) torch
  기본값상 matmul tf32=off(LLM이 진짜 fp32로 도는 것 — 기존 "771ms/fwd, no-tensor-core"
  관측과 정합)이지만 **cuDNN conv tf32=on이 기본** → CNN 트랙은 진짜 fp32가 아니었을
  가능성.
- 참고: 신호크기 진단(2026-07-02, `wiki/flirds-signal-size-diagnosis.md`)에서 fp32 floor는
  신호의 10²~10⁴배 아래 → **정밀도는 병목 아님** 판정 완료. bf16 기각 근거 = utility
  diff ~1e-3 < bf16 정밀도 ~8e-3.

**과업**:
1. **dtype 감사 확정**: 위 사실 재검증 + 누락 경로 확인 — CNN 러너 전체, (a)/(b) oracle,
   `eval/` (mmlu·generate), autocast 부재 확인. 실제 런타임에서
   `torch.backends.cuda.matmul.allow_tf32` / `torch.backends.cudnn.allow_tf32` 값을 출력해
   확정 (torch 2.12의 fp32_precision 신 API 포함). 경로별 정밀도 표로 정리.
2. **CNN tf32 노출 평가**: cuDNN tf32가 실제 켜져 있었는지 확인하고, 기존 CNN rundir의
   coalition diff 크기 vs tf32 오차 스케일 비교로 결과 영향 가능성을 **문서로 판정**
   (A/B 재실행은 제안만 — 몇 분짜리 CNN 스모크로 가능하다는 설계 포함).
3. **옵션 비교 문서** (Yonghee가 보고 판단할 판단 재료 — 권고 결론을 강요하지 말고
   양쪽의 문제점을 대칭적으로 정리):
   - **옵션 ① 전부 fp32 유지(현행)** 의 문제점·리스크: oracle ×3.1 비용과 7B·N=10
     oracle 등 대형 칸의 실행 가능성 제약, 선행연구(대개 bf16/fp16)와 정밀도가 달라
     외부 수치와 비교할 때 필요한 caveat, tf32-off로 인한 학습 속도 손해 등.
   - **옵션 ② 학습만 bf16/tf32 허용 + 평가·HVP·oracle은 fp32 유지**(protocol.md §1
     원안)의 문제점·리스크: 학습 궤적 자체가 바뀌는 것이 신호 크기·fidelity에 미칠 수
     있는 영향과 그 논리(estimator와 oracle이 같은 로그를 보므로 내적 일관성은 유지되는
     이유 포함), 기존 fp32 결과와의 비교 가능성 단절, 검증 없이는 채택 불가 — 필요한
     검증 실험 설계(실행하지 않음)와 기대 비용 절감치 산정(7B·N=10 oracle 연결).
   - 부속: `flirds-protocol.md` §1 문서-코드 불일치의 사실 관계 정리(어느 쪽이 언제
     결정됐는지 위키 기록 추적) — 정정 방향은 옵션 비교의 일부로만 제시, **결정은
     Yonghee가 문서를 보고 내린다.**

---

## 항목 4 — 미뤄둔 엄밀 검증 항목 인벤토리

폴더: `research-wiki/survey/deferred-rigor-inventory-2026-07/`

**범위**: 실험 설계 단계에서 "우선 쉬운 걸로 두고 나중에 엄밀 검증"으로 미룬 결정 +
결론에 영향 줄 수 있는 코드 간소화 **포함**. 순수 엔지니어링 shortcut(결과 무관)은 제외.

**소스**: `research-wiki/wiki/flirds-implementation-plan.md`, `wiki/log.md`,
`wiki/flirds*.md`, `raw/conversations/flirds/*`, memory 토픽 파일, 루트 CLAUDE.md next
블록, 코드 주석(TODO/deferred/caveat).

**시드 목록** (전수 아님 — 스윕으로 추가 발굴): N=10 (a)/(b) oracle deferral(비용 2–5일),
7B (a) 제외, N=5 coarse Spearman(5점 순위), val 크기(100–200)·분포 ablation 미실행,
per-domain normalization ablation 보류, Ripple LLM 포트 미완, momentum=0 고정(Adam bridge
arm 등 Track D 보류 결정 3건), lr 민감도(lr3e-3에서 AUROC 역전 관측), MC vs exact 비교
부재, stealthy/norm-bound backdoor 불가 판정, single-token backdoor target, maverick·
duplicate corruptor 이월, delta/advanced free-rider(task9), tiny-val caveat(poison 판정
val=20, device100 val=4), anchor rank16 기준점(현재 probe로 검증 중), seq512,
per_client=300 통일 결정.

**산출**: 표 — 항목 / 간소화 내용 / 당시 사유(출처 링크) / 흔들릴 수 있는 결론 /
엄밀화 방법·예상 비용 / 우선순위 제안. 우선순위는 **핵심 질문 위계**(1차 fidelity >
2차 성능→수렴→탐지) 기준으로 매길 것.

---

## 항목 6 — 시간·컴퓨팅 cost 비교 방법론 검증

폴더: `research-wiki/survey/cost-comparison-methodology-2026-07/`

**사전 확인 사실**:
- 우리 회계 = FL 학습으로 공유 로그를 한 번 생성(이 시간은 **미측정·표에 미포함**) 후,
  각 방법의 **valuation 함수 호출만** GPU-sync + `perf_counter`로 측정 (`_timed`;
  `phase2_matrix.py:280-287`, `track_c1.py:108-115`, `track_d.py:130-137`).
- **유일 예외 = Ripple** (자체 궤적 실행이 측정에 포함) → 표 내 회계 불일치.
- `wiki/flirds-protocol.md` §15에 방법론 일부가 명문화되어 있으나(timing.json,
  aggregate_runs.py) 일부는 "예정" 상태 — 실코드와의 정합 확인 필요.
- runtime 표 출처: `raw/conversations/flirds/2026-06-06-sv-baseline-port-and-results.md`
  (1B N=5 R=10 val=100: Flirds-1st 35s / Flirds 107s / loss-heur 164s / GTG 537s /
  FedSV 532s / Banzhaf·ShapleyFL·(b)oracle ~531s / Ripple 4515s).

**과업**:
1. **회계 감사 확정**: 타이머 범위 재검증 + 로그 생성(=FL 학습) 시간을 스모크에서 측정해
   "학습 대비 overhead 비율"로 병기할 기준 마련 + Ripple 분리 측정(항목 2와 공유).
   방법별 로깅 요구량 차이도 확인 (모두 `(w_r, δ)` 공유 로그로 충분한가, per-step 정보가
   필요한 방법은 없는가, 로그 저장 용량).
2. **선행연구 비용 보고 관행 조사 (디테일하게 — Yonghee 강조)**: GTG-Shapley / FedSV /
   ShapleyFL / ComFedSV / Ripple / IRDS 6편 (+가능하면 FLDetector) — **원문 비용 절을
   직접 확인** (`research-wiki/raw/papers/flirds/`에 있으면 그것, 없으면 WebFetch·arXiv).
   논문별로: 비용 실험의 정확한 셋업(모델·데이터·클라 수·라운드 수), **비용 표가 재는
   대상**(전체 학습 파이프라인 vs valuation 단계만 vs 라운드당 서버 오버헤드), 보고 단위
   (wall-clock / 연산수 / utility-eval 횟수 / 속도 배수), 학습 시간 포함 여부, 하드웨어,
   **비교 베이스라인을 누가 어떻게 구현했는지**(저자 재구현이면 최적화 수준). 소스노트
   사전 파악(원문으로 확정할 것): GTG "fewer utility evaluations"(정성), FedSV O(Tm²),
   ShapleyFL 정성, ComFedSV asymptotic, Ripple 62×/49×.
3. **우리 비교 baseline 전수 회계 점검 (Yonghee 명시 요청)**: 현재 비교표에 등장하는
   방법 전부 — Flirds / Flirds-1st / GTG / FedSV / Banzhaf / ShapleyFL / ComFedSV /
   loss-heur / Ripple / (b) oracle / FedIF(track_d) + 탐지기(FLDetector·STD-DAGMM·
   FLTrust·FedDQC) — 방법별 비용 프로파일 표: 계산 시점(학습 중 in-run vs 사후
   from-logs), 필요한 입력(공유 로그만으로 충분 / 모델+val 접근 / 클라 로컬 데이터
   접근[FedDQC] / 추가 로깅 요구), 연산 장치(GPU vs CPU[FLDetector]), 그리고 우리 단일
   회계("공유 로그 위 valuation-only wall-clock")가 그 방법을 공정하게 재는지 **방법별
   판정**.
4. **종합 판정**: 현행 비교 방식이 문제 없는지 — 유리/불리하게 왜곡하는 지점(예: fp32
   강제 ×3.1은 전 방법 공통이라 내부 공정하지만 외부 수치와 비교 시 명시 필요, CPU-only
   FLDetector 24s와의 비대칭, Ripple 회계 불일치), 논문에 명시해야 할 caveat 목록.
5. **보고 프로토콜 제안**: 논문용 비용 표 표준 — wall-clock + utility-eval 횟수 + FLOPs
   추정 + peak memory + 로그 저장량 중 무엇을 병기할지 권고안 (선행 관행 조사 결과에
   근거해서).

소형 재측정은 허용 (스모크 급, GPU 규약 준수). 대형 재측정은 설계만.
