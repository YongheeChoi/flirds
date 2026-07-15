# 정밀도(fp32) 감사 + 정책 문서 — 항목 3

- 작성: 2026-07-04 (논문화 전 검증·감사 5건 중 항목 3; `PROMPT_VERIFICATION_SURVEY_2026-07.md`)
- 성격: **Yonghee 판단용 재료.** 권고 결론 없음 — §3에서 옵션 ①(전부 fp32 유지)과
  옵션 ②(학습만 bf16 + 평가·HVP·oracle fp32)의 문제점을 대칭적으로 정리.
- 재료: dtype 전수 감사 노트(정찰 세션, codes/ 전체 Grep+정독), 원격 B200 런타임 플래그
  실측(2026-07-04), `wiki/flirds-signal-size-diagnosis.md`(2026-07-02), 로컬
  `runs/track_c` rundir 직접 계산(이 세션, 부록 A 스크립트), wiki·raw 기록 추적.
- 표기: [확인] = file:line/실측 근거. [추측] = 정황 판단(근거 병기). [재계산] = 이 세션에서
  기존 실측치로부터 산술 재유도. [실측 대기 — X] = 실측 미실행 — 서버 세션 인수인계(`../PENDING_MEASUREMENTS_2026-07.md` 실측 3).

## 0. 요약

1. **LLM 경로는 학습·평가·estimator·(b) oracle 전부 fp32이고, B200 런타임 실측으로
   matmul TF32 off까지 확정** — LLM 수치는 "진짜 IEEE fp32"가 맞다. [확인]
2. **예외 1곳 발견**: `phase2_llm_a_oracle.py`의 (a) retrain oracle은 **기본값이 bf16**
   (fp32는 `A_DTYPE=fp32` opt-in). headline (a) 검증 런(06-07)은 fp32로 실행된 기록이
   있으나, 레포 내 스크립트에 `A_DTYPE` 참조가 0건이라 수동 env였던 것으로 보인다. [확인+추측]
3. **CNN 트랙은 cuDNN conv TF32 기본 on에 노출** — B200 박스에서 conv 기본값 tf32 실측
   확정, CNN이 실제 도는 yonsei SLURM 박스(RTX 3090, Ampere)는 미확인.
   [실측 대기 — yonsei 박스 tf32 플래그 + conv fp32/fp64 오차 판별]
4. CNN 노출의 결과 영향(§2 판정): **오염/스큐 셀의 headline 신호(φ spread ~0.05–0.39;
   강-오염 headline 셀 기준. mnist feature-noise/label-skew 등 약-오염·스큐 셀은
   ~0.008–0.05까지 낮으나 모두 TF32 ~1e-3의 1자릿수+ 위 — 부록 A)는 TF32 섭동 추정
   스케일(~1e-3)의 1–2자릿수 위 → 결론이 뒤집힐 가능성 낮음.** iid 셀의
   미세 순위(인접 φ 갭 4e-5–4.4e-4)는 TF32 스케일 이내지만 그 셀들은 이미 "신호
   없음(추첨 노이즈)" 판정이라 서사 불변. 확정은 몇 십 분짜리 A/B 스모크로 가능(설계만 §2.4).
5. **protocol.md §1(bf16 train / fp32 eval)은 코드와 불일치** — bf16 학습은 2026-05-27
   계획이었고, 06-04 첫 구현부터 코드는 fp32-throughout, 06-07(정량 근거)·06-09(7B 결정)로
   사실상 정책이 대체됐는데 문서가 갱신되지 않았다(§3.3 타임라인). 정정 방향(문서를 코드에
   맞출지, 코드를 문서에 맞출지)이 곧 §3의 옵션 선택이다. **결정은 Yonghee.**

---

## 1. dtype 감사 확정

### 1.1 경로별 정밀도 표

정찰 세션 전수 스윕(`dtype|float32|bfloat16|float16|autocast|GradScaler|half()|allow_tf32|
matmul_precision|fp32_precision` — codes/ 전체) + 이 세션 인용 라인 재확인. [확인]

**LLM 트랙** (Llama = Linear/matmul + eager attention, conv-free → cuDNN conv TF32 무관):

| 경로 | dtype | 근거 (file:line) |
|---|---|---|
| 모델 로드 (전 러너 공통) | fp32 + eager attn | `codes/experiments/phase2_matrix.py:145`, `track_d.py:122`, `phase2_llm_a_oracle.py:98`((b)/estimator 쪽), smoke 계열 13개 파일 동일 |
| 클라 로컬 학습 (SFTTrainer) | fp32 — `bf16=False, fp16=False`, 명시 SGD | `codes/flirds/fl/llm_server.py:46,52`; poison 학습도 동일 `phase2_matrix.py:206` |
| FedAvg 집계 | 로그 dtype 상속(=fp32); 가중치 np.float64 | `codes/flirds/fl/server.py:55-63` |
| estimator (1차+2차, HVP) | fp32 — params/Δw `.float()` **명시 캐스팅** | `codes/flirds/core/flirds_estimator.py:101,103` (docstring :34 "fp32 (protocol 1)") |
| val loss_fn (평가 forward) | 캐스팅 없음 — 모델 dtype 상속(호출자 fp32) | `codes/flirds/backends/llm.py:11-12` |
| (b) in-run oracle (utility/LOO/exact/per-round) | 캐스팅 없음 — 로그 dtype 상속 | `codes/flirds/oracle/in_run_sv.py:19(docstring),38-39(_split)` |
| (a) retrain oracle 커널 | **정밀도-무관(호출자 선택)** — docstring이 명시 | `codes/flirds/oracle/exact_sv_llm.py:16-19` |
| (a) 러너 — phase2_llm_a_oracle | **기본 bf16**; `A_DTYPE=fp32` env일 때만 fp32 (기본 모드에선 (a)-val-loss 평가까지 bf16 모델 위) | `codes/experiments/phase2_llm_a_oracle.py:118-122` |
| (a) 러너 — track_d | fp32 (공유 fp32 모델 재사용; "task-6 same-game lesson") | `codes/experiments/track_d.py:11-12,153-171` |
| baselines (GTG/ComFedSV/ShapleyFL/FedIF/FLTrust/STD-DAGMM) | 재구성 params `.float()` 명시 → fp32 | `gtg.py:36,43`, `comfedsv.py:46,52`, `shapleyfl.py:48,54`, `fedif.py:68,78`, `fltrust.py:57,67`, `std_dagmm.py:69,86` |
| baselines (FedSV/FedDQC/Banzhaf/Fed-LOO/loss-heur) | dtype 연산 없음 — loss_fn 경유 fp32 상속 | 각 파일 스윕 매치 0; Banzhaf는 `_coalition_utilities` 재사용 |
| generate/ROUGE | 모델 dtype 상속 (track_d·matrix에선 fp32) | `codes/flirds/eval/generate.py:29,36-43` |
| MMLU 0-shot | 모델 상속 (fp32); argmax 판정이라 민감도 낮음 | `codes/flirds/eval/mmlu.py:35-75` |
| metric 후처리 (Spearman/Pearson 등) | numpy float64 | `codes/flirds/eval/metrics.py:89-115` |
| fp64 특수 사용처 2곳 | FLDetector L-BFGS solve fp64; Ripple eigsh LinearOperator **선언만** fp64(matvec은 fp32) | `fldetector.py:70,90`; `ripple_llm.py:105,109` |

**CNN 트랙** (dtype 캐스팅·autocast·half 일절 없음 → torch 기본 fp32; 단 conv는 cuDNN TF32 노출):

| 경로 | dtype | tf32 노출 | 근거 |
|---|---|---|---|
| 데이터/모델 정의 | fp32 (`ToTensor()`; 기본 dtype) | — | `codes/flirds/data/cnn.py:20-22`, `models/cnn.py:15-16(LeNet5 conv),36-37(FedSVCNN conv)` |
| 로컬 학습 (fwd+bwd) | fp32, autocast 없음 | **노출** (conv fwd/bwd cuDNN) | `codes/flirds/fl/client.py:23` |
| 서버 평가 | fp32 | **노출** | `codes/flirds/fl/server.py:78` |
| estimator/(b) oracle loss_fn | fp32 (functional_call → conv2d) | **노출** | `codes/flirds/backends/cnn.py:25` |
| estimator HVP (jvp∘grad) | fp32 | **노출 가능** [추측: forward-AD conv도 aten conv 커널 경유 — 커널 선택은 런타임 판별 필요] | `flirds_estimator.py:101-113` |
| Ripple (CNN) 학습+HVP | fp32 (fp64는 eigsh 선언뿐) | **노출 가능** | `codes/flirds/baselines/ripple.py:80,103-119` |
| `cudnn_deterministic=True` | 결정론 플래그일 뿐 — **TF32를 끄지 않음**(직교) | — | `codes/flirds/repro.py:14-21`; CNN 전 진입점에서 사용 |

### 1.2 런타임 플래그 실측 (B200 박스, 2026-07-04, verbatim) [확인]

`/home/korea_bupj/miniconda3/envs/flirds/bin/python` (원격 정찰 노트 §3):

```
2.12.0+cu130
matmul_tf32 False
cudnn_tf32 True
matmul_prec none
cudnn_conv_prec tf32
f32mp highest
```

- **matmul = 진짜 IEEE fp32 확정**: `torch.backends.cuda.matmul.allow_tf32=False`,
  `float32_matmul_precision='highest'`, 신 API `matmul.fp32_precision='none'`(미지정=highest).
  기존 관측 "771ms/fwd, fp32-B200 no-tensor-core"(루트 CLAUDE.md baseline)와 정합.
- **cuDNN conv 기본 = TF32** (`cudnn.allow_tf32=True`, `cudnn.conv.fp32_precision='tf32'`).
  LLM은 conv-free라 무관. **이 실측은 B200 박스의 값이다 — CNN 트랙이 실제로 도는
  yonsei SLURM 박스(RTX 3090, torch 2.11.0+cu130; `runs/track_c/RESULTS.txt` 헤더,
  `runs/probe_signal/cnn_c1/pc1_*/meta.json`)는 별도 확인 필요.** torch 2.x 기본값이
  동일하므로 같을 것으로 추정하나 [추측], 플래그 값과 "cuDNN이 이 소형 conv에 실제 TF32
  커널을 선택했는가"는 다른 질문이다(§2).
  [실측 대기 — yonsei 박스: 플래그 덤프 + conv fp32-vs-fp64 오차 판별 스니펫(부록 B)]

### 1.3 부재 확인 [확인]

- **autocast/GradScaler/`.half()`/`torch.set_float32_matmul_precision` 사용 0곳** (전수 Grep 매치 0).
- **tf32 플래그를 만지는 코드 0곳** → 전부 torch 기본값 (위 실측).
- **정밀도 강제 장치 부재**: `assert dtype` 0건; protocol이 명시한 `precision_guard.py`
  (`wiki/flirds-protocol.md:198` "fp32-eval enforcement")는 **미구현** (Glob 매치 0).
  (b) oracle `_split`(`in_run_sv.py:38-39`)은 캐스팅 없이 로그 dtype을 상속 — bf16 로그가
  들어오면 **조용히 bf16 oracle**이 된다. fp32는 "관례 + 명시 로드"로만 유지되는 상태.

### 1.4 사전 확인 사실 대비 달라진 것

| 프롬프트의 사전 확인 사실 | 감사 결과 |
|---|---|
| "코드는 전 경로 fp32로 보임" | **예외 1곳**: (a) oracle 러너 `phase2_llm_a_oracle.py:119` 기본값 bf16 (fp32는 opt-in). headline 검증 런(06-07)은 fp32 실행 기록 있음(raw "1B fp32 126 min"; 루트 CLAUDE.md "N=5@1B fp32") — 단 runs/·slurm/ 스크립트에 `A_DTYPE` 참조 0건이라 세션 수동 env로 추정 [추측]. track_d의 (a)는 fp32 [확인]. |
| "matmul tf32=off (torch 기본값 추정)" | **B200 실측으로 확정** (§1.2 verbatim). |
| "cuDNN conv tf32=on 기본 → CNN이 진짜 fp32가 아니었을 가능성" | **B200에서 conv 기본 tf32 실측 확정**; yonsei 박스는 미확인 → 노출 여부의 최종 판별은 [실측 대기]. 정량 영향 평가는 §2. |
| "protocol §1 문서-코드 불일치" | 타임라인 확정(§3.3): bf16은 05-27 계획, 06-04 구현부터 fp32, 06-07·06-09에 사실상 대체, 문서 미갱신. 같은 불일치가 `flirds.md:130`, `flirds-implementation-plan.md:78,112,242,301`, protocol §13(:224 "7B = bf16 train / fp32 eval")에도 전파. |
| (신규 발견) | ① precision_guard 미구현 + (b) oracle dtype 가드 부재(§1.3). ② 06-07 raw 비용 표의 fp32 N=10 외삽 "~67 GPU-hr"는 자체 선형 규칙과 불일치 — 재계산 ~128 GPU-hr (§3.1, [재계산]). |

---

## 2. CNN tf32 노출 평가

### 2.1 노출 사실관계

- 노출 지점: conv2d 정의는 `models/cnn.py` 단 1개 파일(LeNet5·FedSVCNN 각 conv1/conv2)이고,
  소비자는 CNN 학습·평가·estimator/(b) oracle loss_fn·HVP·Ripple 전부 (§1.1 표). [확인]
- B200 박스는 conv 기본 tf32 확정(§1.2) — 단 **CNN 그리드(track_c 150셀, probe cnn_c1/c2
  96셀)는 yonsei SLURM RTX 3090에서 실행**됐다. 3090은 Ampere(TF32 지원)이고 torch
  2.11 기본값도 cudnn.allow_tf32=True이므로 노출 전제는 성립하나 [추측], 실제 TF32 커널
  선택 여부는 cuDNN heuristics·`cudnn_deterministic=True` 제약에 따라 달라질 수 있어
  실측으로만 확정된다. [실측 대기 — yonsei 박스 부록 B 스니펫]
- `cudnn_deterministic=True`(CNN 전 진입점)는 재현성 주장(protocol §5 "bitwise-identical
  at fp32", `flirds-protocol.md:110`)과 모순되지 않는다 — 같은 HW·같은 플래그면 TF32여도
  bitwise 재현 가능. 문제는 "CNN=fp32" **서술의 엄밀성**(IEEE fp32가 아닐 수 있음)이다.

### 2.2 오차 스케일 vs 신호 스케일

**TF32 오차 스케일** [추측 문헌 일반론 → **B200 A/B가 실측 확증, §2.4**; 부록 B는 여전히
yonsei-박스 노출 판별용]: TF32는 곱셈 입력을 10-bit mantissa로 반올림(상대 ~2⁻¹¹≈4.9e-4/원소),
누산은 fp32. conv 출력의 fp32 대비 상대 편차는 통상 ~1e-3 차수. CE val-loss(utility, CIFAR
~1.1–2.3)로 전파되는 절대 섭동을 **~1e-3 차수**로 잡는다. 섭동은 결정론 커널에서 가중치의
결정적 함수이므로 coalition 간 utility **차이**에서는 부분 상쇄될 수 있다(방향 불확실).
→ **B200 A/B(§2.4) 실측이 이 추정을 확증**: TF32 on/off φ 이동 max|Δφ|가 fidelity-핵심
방법에서 Flirds 8.8e-4·(b)oracle 5.6e-4·Banzhaf 5.7e-4(iid) / 8.4e-4–1.2e-3(label_flip)로
정확히 ~1e-3 차수(`outputs/tf32_ab/*/metrics.json`). ShapleyFL만 φ 절대크기(~0.9)에 비례해
~2.3e-2로 예외.

**신호 스케일** — 로컬 `runs/track_c/c1/*/phi.parquet`의 (b)oracle φ에서 직접 계산
([재계산], 이 세션; 부록 A 스크립트, 30셀 전체 수치 포함):

| 셀 (3-seed 범위) | φ spread (max−min) | 인접 φ 갭 최소 | 인접 φ 갭 중앙값 |
|---|---|---|---|
| cifar10 **iid** | 0.021–0.037 | **6e-5–4.4e-4** | 1.5e-3–1.7e-3 |
| mnist **iid** | 0.009–0.013 | **4e-5–1.0e-4** | 0.8e-3–1.2e-3 |
| cifar10 label-flip | 0.093–0.120 | 2.2e-4–1.3e-3 | 1.0e-2–1.5e-2 |
| cifar10 quantity-skew | 0.176–0.220 | 1.1e-4–4.2e-3 | 1.1e-2–2.6e-2 |
| mnist label-flip | 0.326–0.386 | 3.7e-4–8.6e-4 | 1.0e-2–1.4e-2 |
| cifar10 feature-noise / label-skew | 0.048–0.134 | 2.6e-4–2.4e-3 | 2.5e-3–1.8e-2 |

### 2.3 판정 (문서 판정 — **A/B 스모크로 확정됨, §2.4**)

1. **오염/스큐 셀의 headline 결론은 안전할 가능성이 높다.** corrupt-vs-clean 분리와
   cross-seed 안정성(label_flip ρ=0.968, quantity_skew 0.968; `runs/track_c/RESULTS.txt`)을
   떠받치는 신호(spread ~1e-1, 인접 갭 중앙값 ~1e-2)는 TF32 섭동 추정치(~1e-3)의
   1–2자릿수 위. 탐지 AUROC 0.93–0.99(진단 문서 §3.6)도 같은 크기의 신호에 기반.
2. **iid 셀의 미세 순위 구조는 TF32 스케일 이내다** (인접 갭 4e-5–4.4e-4 < ~1e-3).
   즉 iid 셀의 세부 순위·fidelity 소수점(예: cifar10 iid Flirds-vs-(b) Spearman
   0.939/0.952/0.964, mnist iid 0.636–0.964; metrics.json 직접 확인)은 TF32 유무에 따라
   ±수 순위 흔들릴 수 있음을 배제 못 한다. 단 **이 셀들은 이미 "클라 간 진짜 신호
   부재(추첨 노이즈), ρ_xseed≈0" 판정**(진단 문서 §1.4·§3.6)이라, TF32가 결과를 바꿔도
   서사("iid는 신호 없음")는 불변 — 오히려 그 판정과 정합적인 노이즈원 하나가 추가될 뿐.
3. **내적 일관성**: estimator·oracle·baselines가 같은 박스·같은 커널·같은 로그를 공유하므로
   비교는 자기-일관적 게임 위에서 이뤄졌다. 단 estimator의 HVP(forward-AD)와 oracle의
   replay-forward는 **다른 커널 경로**라 TF32 반올림이 경로별로 달라질 수 있고, 이것이
   estimator-vs-oracle fidelity에 ~1e-3급 경로 의존 노이즈로 들어갔을 가능성은 A/B 없이
   배제 불가 [추측]. (CNN pooled Flirds fidelity +0.953±0.080의 결손 일부가 여기서 왔을
   가능성 — 크기상 iid 셀에 국한될 것.) **[A/B 후속 §2.4: 이 추측은 순위 기준 반박됨** —
   iid·label_flip 모두 Flirds·Banzhaf의 estimator-vs-(b) `spearman_b`가 on/off **비트 동일**
   (Flirds iid 0.98788/0.98788, label_flip 1.0/1.0)이라 결손은 TF32 경로 노이즈가 아닌 실제
   신호 구조. 단 값-수준 `pearson_b`는 ~7e-4 움직이고(Flirds iid 0.99485→0.99416), iid
   약-fidelity 방법의 `spearman_b`는 순위 재편(Flirds1st 0.588→0.539, loss-heur 0.636→0.564)
   — §2.3-2 예측과 정합.]
4. **논문 서술 각주 필요**: 확정 전까지 CNN 수치에 "conv는 cuDNN 기본 설정(TF32 가능성,
   Ampere)"의 각주가 정직하다. LLM 수치는 각주 불요(matmul fp32 실측 확정, conv-free).

### 2.4 확정용 A/B 스모크 — 실측 완료 (B200, 2026-07 서버 이전 후)

설계대로 **cifar10 {iid, label_flip} × {TF32-on, TF32-off} × seed0 = 4런**을 실행했다. 단
무대는 (계획했던 yonsei 3090이 아니라) **이전 후 B200**(venv, torch 2.12.0+cu130;
`outputs/tf32_ab/{cifar10_iid,cifar10_label-flip}_tf32{on,off}_seed0/metrics.json`,
staging flirds_batch). TF32-off는 `cudnn.allow_tf32=False`로 강제, on은 현행 기본값. 방법
11종(estimator·oracle·baselines)을 같은 로그·같은 loss_fn으로 재측정.

**결과 — 헤드라인 지표는 TF32-불변** (on−off; 전 방법 파싱):

| 방법 | ΔAUROC (LF) | Δ spearman_vs_rate (LF) | max\|Δφ\| iid | max\|Δφ\| LF |
|---|---|---|---|---|
| (b)oracle | 0.000 | 0.000 | 5.6e-4 | 8.4e-4 |
| **Flirds** | 0.000 | 0.000 | 8.8e-4 | 1.2e-3 |
| Flirds1st | 0.000 | 0.000 | 1.4e-3 | 1.3e-3 |
| Banzhaf | 0.000 | 0.000 | 5.7e-4 | 8.4e-4 |
| loss-heur | 0.000 | 0.000 | 1.4e-3 | 1.5e-3 |
| Fed-LOO | 0.000 | 0.000 | 1.3e-3 | 1.1e-3 |
| FedIF | 0.000 | 0.000 | 1.3e-2 | 1.4e-2 |
| GTG | 0.000 | 0.000 | 2.9e-3 | 5.3e-3 |
| ComFedSV | 0.000 | 0.000 | 4.8e-3 | 6.1e-3 |
| **FedSV** | 0.000 | **+0.025** | 4.3e-3 | 5.2e-3 |
| **ShapleyFL** | 0.000 | **+0.148** | 2.3e-2 | 2.3e-2 |

- **AUROC**: 11개 방법 전부 on=off (비트 동일). label_flip 탐지 헤드라인
  ((b)oracle·Flirds·Flirds1st·Banzhaf·loss-heur·FedIF = 1.000) 완전 불변.
- **spearman_vs_rate**: 11개 중 9개 소수 15자리까지 on=off (Flirds·Banzhaf·(b)oracle·loss-heur
  0.98473, GTG 0.78779, ComFedSV 0.49237, FedIF 0.93550, Fed-LOO 0.96011). **예외 2개**:
  ShapleyFL 0.443→0.295(Δ0.148), FedSV 0.886→0.862(Δ0.025).
- **φ 이동(max|Δφ|)**: fidelity-핵심 방법은 ~1e-3(§2.2 추정 확증). 재구성-MC·대진폭 방법이 더 큼
  (FedIF ~1.4e-2, ComFedSV ~6e-3). **최대 = ShapleyFL 2.3e-2**(φ 절대크기 ~0.9 = 전 방법 최대,
  min-max+EMA 증폭).
- **궤적**: iid final_acc on 0.65425 vs off 0.654375 (Δ1.3e-4), label_flip 0.627125 vs 0.626
  (Δ1.1e-3) — 사실상 동일.

**판정 (§2.3 확정):**
1. **헤드라인 안전 확정.** 오염 셀(label_flip)의 탐지 AUROC·spearman_vs_rate가 헤드라인
   방법(Flirds·Banzhaf·(b)oracle)에서 TF32에 완전 불변 → 원래 그리드가 TF32-노출이었든 아니든
   헤드라인 결론 불변. **yonsei 박스 노출 판별(부록 B/P1)은 이로써 저-스테이크**가 된다(노출이어도
   결과가 안 바뀜을 B200이 보임). LLM은 여전히 각주 불요(matmul fp32 실측, conv-free).
2. **ShapleyFL·FedSV의 spearman_vs_rate 흔들림은 헤드라인 밖.** 둘 다 φ가 near-tie
   (ShapleyFL 대진폭+저-fidelity 0.30–0.44 = near-random / FedSV clean 클라 φ≈0)라 ~1e-2급
   섭동에 순위가 재편된다. 그러나 둘 다 fidelity-열위 방법(헤드라인 = Flirds·Banzhaf)이라 서사에
   영향 없음. **"ShapleyFL만 민감"은 과언** — FedSV도 작게 움직인다(0.025 vs 0.148). (fp32 유지가
   ShapleyFL 통일 재실행에 β와 무관하게 안전한 근거 하나 추가.)
- **후속 결정 재료(불변)**: 노출 확정 시 "CNN 전 진입점에 `cudnn.allow_tf32=False` 1줄 추가
  (재현성 플래그 옆, `repro.py`)" 여부 — 기존 150셀과의 비교 단절이 생기나, 위 1의 저-스테이크
  판정으로 **긴급도는 낮다**(각주로 충분). 부록 B 스니펫(수 초)은 여전히 yonsei 박스에서 "conv가
  실제 TF32 커널로 돌았는가"의 사전 판별용으로 유효 — (a) 각주 vs (b) 플래그+재실행 선택은 §3
  옵션 결정과 함께 Yonghee 판단.

---

## 3. 옵션 비교 — 정밀도 정책

두 옵션 모두 "평가·HVP·oracle = fp32"는 공유한다(06-07 정량 근거가 확립: (a)-val-loss
coalition diff ~0.005–0.02 < bf16 절대 정밀도 ~0.009 → bf16 평가는 +0.3, fp32 재실행 시
+1.000; raw `2026-06-07-phase2-task6-a-retrain-oracle.md:66-73`; protocol §4.4:90 동일 논리).
쟁점은 **학습(로그 생성 + (a) retrain)의 정밀도**다.

### 3.1 옵션 ① — 전부 fp32 유지 (현행)의 문제점·리스크

**(1) oracle 비용 — 실측 근거** (raw 06-07:85-93, 1B N=5 R=10):

| | (a) N=5 총 | retrain/클라 | eval/coalition |
|---|---|---|---|
| 1B bf16 | 47 min | 27.5 s | 19.7 s |
| 1B fp32 | **126 min** | **85.2 s (×3.1 — no tensor cores)** | 23 s (×1.17) |
| 3B bf16 | 90 min | 53.2 s | 35.6 s (3B fp32 미실측) |

- retrain이 (a)의 78–90%이고 per-|S| 선형. N=5 총비용 fp32/bf16 = ×2.7.
- **N=10 외삽 [재계산]**: retrain 64×(Σ C(10,s)·s=5120 클라-retrain) + eval 32×(1024
  coalition) 규칙(raw 06-07:91-92)을 fp32 실측에 적용하면 85.2s×5120 + 23s×1024 ≈
  **~128 GPU-hr (단일GPU ~5.3일, 4-GPU 샤딩 ~32h)**. raw 표의 "~67 GPU-hr"는 126min×32
  (eval 배수만 적용)와 정확히 일치해 **과소 기재로 보임** [추측] — 항목 6(cost 감사)과
  공유할 발견. bf16이면 ~45 GPU-hr(표 기재와 정합).
- (b) oracle: N=100 cross-device 771ms/fwd(fp32-B200) → **~11h/4-GPU, α 1점만** (루트
  CLAUDE.md baseline; plan:19). N=5↔N=10 = 32× 규칙(protocol §4.4:90). 7B·N=10 oracle이
  "설계상 deferred"(루트 CLAUDE.md next)인 배경에 이 fp32 비용 구조가 있다.

**(2) 대형 칸 실행 가능성 제약**: N=10 (a)는 "2–5일 단일GPU → real experiment로 deferred"
(raw 06-07:93-94; plan:143 — 3B N=10 retrain 하한 84.3h). fp32 유지는 이 칸들을 계속
비싸게 묶어둔다. 단, **(b) oracle·평가는 옵션 ②로도 fp32 유지라 이 축의 절감은 없다**
(§3.2-(5)) — fp32 유지가 잠그는 것은 주로 (a) retrain과 로그 생성(FL 학습) 시간이다.
예: 현재 probe 캠페인 std50k5 셀은 vanilla 학습만 ~3.1h/셀(원격 실측; anchor 셀의
vanilla 학습은 R=30이라 ~0.5h — noise probe vanilla 1885s 급) — 학습 부분이 bf16이면
retrain 실측 배수(×3.1)에 준해 줄었을 몫이다 [추측: 학습=matmul 지배 전제]. **주의**:
anchor 셀의 *총* 벽시계는 ~10.4h지만(r32 실측) 그 대부분(~6.75h)은 (b)oracle+coalition
방법 fidelity 계산이라 **fp32-lock — bf16 학습으로 줄지 않는다**(§3.2-(5)와 정합); anchor는
oracle_a=false라 (a) retrain도 없다(remote_recon §1.4·§1.6). 즉 '학습만' 시간(std)과 '총
셀' 시간(anchor)은 같은 축이 아니다 — bf16으로 줄어드는 몫은 학습 부분(std ~3.1h / anchor
~0.5h)에 국한된다.

**(3) 선행연구와의 외부 비교 caveat**: 선행 FL-SV 연구·LLM SFT 관행은 대개 bf16/fp16
(protocol §1:19의 원 근거이기도 함 — LESS, Grosse 2023, MATES, FedDQC). 우리 runtime
수치(Flirds 35s/GTG 537s/oracle 531s 등)는 fp32 기준이라 **외부 논문의 wall-clock과 직접
비교할 때 정밀도 차이를 명시해야 한다**. 방법 간 상대 배수는 전 방법 공통 fp32라 내부
공정 [확인: 전 방법이 같은 로그·같은 loss_fn 소비, §1.1]. 리뷰어 관점에서 "왜 관행(bf16)을
안 따랐나"는 질문 포인트 — 답변 근거는 있음(신호 ~1e-3 < bf16 ~8e-3)이나, "학습만 bf16으로
하고 평가만 fp32면 되지 않나"(=옵션 ②)가 자연스러운 재반박이 된다.

**(4) tf32/bf16 대비 fp32 연산 벌점 — B200 마이크로벤치로 실측 확정** [실측]
(2026-07 서버 이전 후; Llama-3.2-1B, val=100; `outputs/microbench/summary.json`, torch
2.12.0+cu130, staging flirds_batch). matmul TF32는 코드가 안 만져 기본 off(§1.2)라 B200
tensor core를 fp32 경로 어디에도 안 쓴다 — 켰을 때의 벌점을 이제 직접 측정했다:

| 연산 | fp32 | tf32 | bf16 | fp32÷tf32 | fp32÷bf16 |
|---|---|---|---|---|---|
| forward (s/pass) | 1.601 | 0.311 | 0.300 | **×5.16** | **×5.33** |
| HVP (2차, s/pass) | 10.36 | 2.848 | 2.535 | **×3.64** | **×4.09** |
| raw GEMM (TFLOPS) | 65.97 | 793.3 | 1495.8 | **×12.0** | **×22.7** |

(HVP peak-mem fp32/tf32 90.5 GiB vs bf16 66.9 — 대형 칸 HVP는 메모리도 bind.)

- **§3.1-(1) 표 "×3.1"의 해명 [재계산]**: 그 ×3.1은 06-07 **retrain 벽시계** fp32÷bf16
  (85.2s/27.5s=3.10)으로, 데이터로딩·옵티마이저·비-matmul 오버헤드가 섞인 **워크로드** 값이라
  raw-op 배수보다 희석돼 있다. 순수 연산 벌점의 진짜 상한은 GEMM ×22.7(bf16)/×12.0(tf32),
  forward·HVP는 ×4–5다 — 즉 retrain ×3.1은 **상한이 아니라** 오버헤드 지배 워크로드의 실측
  하한이다(이전 문안 "×3.1/retrain 상한 참고치"는 라벨 오류였고 마이크로벤치가 정정;
  cost-comparison §4 C3의 "×3.1 = 미검증 placeholder, 인용 금지" 지적도 이 실측으로 해소).
  학습(matmul 지배)만 bf16이었다면 §3.1-(2)의 std 학습 ~3.1h/셀은 GEMM 배수 근처까지 줄
  여지가 있으나, 실제 학습 루프는 오버헤드 때문에 retrain ×3.1급에 가까울 것 [추측: 학습=matmul 지배 전제].
- **caveat**: 배수는 B200 고유값(하드웨어 의존) — 이전 전 박스(yonsei RTX 3090 등)엔 그대로
  옮겨지지 않고, 정성적 결론("fp32는 tensor-core 대비 크게 느림; 평가 fp32는 §3.1-(5)상
  필요조건, 줄일 여지는 학습 정밀도뿐")만 전이된다.

**(5) 요약 리스크**: 비용·확장성·외부 비교 — 수치의 신뢰성 리스크는 없음(오히려 가장
보수적). 신호크기 진단(§1.1)은 fp32 floor가 신호의 10²–10⁴배 아래임을 확인 — fp32가
"필요 최소"보다 과잉인지는 bf16 평가 정밀도(~8e-3)가 신호(~1e-3)보다 **크다**는 사실이
답한다: 평가 fp32는 과잉이 아니라 필요조건. 과잉 가능성이 있는 부분은 학습 정밀도뿐이고,
그것이 옵션 ②의 존재 이유다.

### 3.2 옵션 ② — 학습만 bf16(+tf32), 평가·HVP·oracle fp32 (protocol §1 원안)의 문제점·리스크

**(1) 검증 없이 채택 불가 — 기존 근거가 옵션 ②를 커버하지 않는다.** 06-07의 bf16 실패
데이터포인트는 **학습·평가 모두 bf16**인 런이었다(phase2_llm_a_oracle 기본 모드: bf16 모델
위에 val_loss_fn까지 빌드, `:120-122`). fp32 재실행은 둘을 동시에 바꿨으므로, "bf16 학습 +
fp32 평가"라는 옵션 ② 조합은 **한 번도 실험된 적이 없다** [확인: raw 06-07 + 코드 구조].
채택하려면 §3.2-(6)의 A/B 검증이 선행되어야 한다.

**(2) 학습 궤적 변경의 영향 — 논리 정리.**
- *내적 일관성은 유지된다*: Flirds의 게임은 "실현된 로그 `[(w_r, deltas_map)]` 위에서
  정의"되고, estimator·(b) oracle·전 baselines가 **같은 로그를 소비**하며(§1.1) 평가는
  fp32다. 학습이 bf16이면 궤적 자체가 다른 궤적이 될 뿐, 그 궤적 위의 fidelity 비교
  (estimator vs (b))는 여전히 잘 정의된 같은 게임의 비교다. bf16 mixed(fp32 master
  weight — SFTConfig `bf16=True`의 HF 표준 동작)라면 로그에 저장되는 w_r·delta도 fp32
  master 값이라 (b) oracle 입력의 표현 정밀도도 유지된다.
- *그럼에도 남는 리스크*: ① bf16 fwd/bwd 반올림(상대 ~4e-3/텐서연산)이 gradient에 실려
  per-round delta에 들어간다 — IID-clean 무대의 클라 간 φ 차이(~1e-4–1e-3, 진단 §1.3d)는
  이미 추첨 노이즈 수준이라, 궤적 노이즈가 하나 더 얹히는 것은 "seed가 하나 더 늘어난 것"과
  비슷한 효과일 것 [추측 — 진단 §1.4의 신호원=B축 판정과 정합]. 오염 무대의 신호(poison
  분리 9–18×)는 흔들릴 크기가 아님 [추측]. 그러나 **(a)-vs-(b) 검증(+0.900~1.000)이나
  cross-seed ρ 같은 미세 지표가 bf16 궤적에서 어느 쪽으로 움직이는지는 실측 전 미지**.
  ② 신호크기 진단의 절대 수치(Δval-loss, φ 크기)가 bf16 궤적에서 재현되는지 재확인 필요.
- *구현 함정*: "bf16 mixed(fp32 master)"와 "bf16 모델 로드"는 다르다 — 후자(현 (a) 러너
  기본 모드의 방식)는 가중치 자체가 bf16 그리드라 delta(파라미터당 ~1e-5–1e-4)가 양자화에
  눌릴 수 있다. 옵션 ② 구현은 반드시 전자(SFTConfig `bf16=True`)여야 하고, (b) oracle
  `_split`이 로그 dtype을 상속하는 현 구조(§1.3)에서는 **precision_guard(로그 fp32 assert)
  구현이 사실상 전제 조건**이 된다.

**(3) 기존 결과와의 비교 단절**: phase2 25셀 + track_d 18셀 + probe 캠페인(진행 중 포함,
seed0 파일럿 3셀이 지금도 fp32로 돌고 있음)이 전부 fp32-throughout이다. 지금 전환하면
(i) 전 그리드 재실행(수백 GPU-h) 또는 (ii) 논문 표가 정밀도 혼합이 되는 것(비교 불가) 중
하나를 감수해야 한다. B축 매트릭스(matrix_cxni) 등 예정 실험과의 정밀도 축 엇갈림도 동일.

**(4) 논문 서술 부담의 이동**: 옵션 ①의 "왜 fp32?" 질문이 사라지는 대신, "학습 bf16이
valuation에 영향 없다는 것을 어떻게 아는가"를 검증 실험으로 답해야 한다(=(6)). 검증이
성공하면 오히려 방법론 강건성 주장(정밀도-불변)이 하나 생기는 부수 효과 [추측].

**(5) 절감이 없는 곳 — 기대치 관리**: fp32로 남는 (b) oracle·평가·HVP는 그대로다.
N=100 (b) ~11h/4-GPU, N=10 (b) 32× 규칙, 7B (b) 비용은 옵션 ②로 **줄지 않는다**
(bf16 평가 이득 실측치도 eval/coalition ×1.17에 불과 — 단 이는 generation 지배 측정이라
(b)-류 batch forward의 이득을 과소평가할 수 있음 [추측]). 즉 "7B·N=10 oracle"을 여는
열쇠의 절반만 옵션 ②가 쥔다: **(a) 쪽은 ~128→~46 GPU-hr(×2.8)로 열리지만(1B N=10,
[재계산]: bf16 retrain 27.5s×5120 + fp32 eval 23s×1024), (b) 쪽 비용은 남는다.**
로그 생성(FL 학습) wall-clock 절감(×~2.7 추정)은 전 그리드에 적용되는 실질 이득.

**(6) 필요한 검증 실험 설계 (실행 안 함 — 채택의 전제 조건)**:
- **무대**: 1B anchor5(N=5 full, R=30) — (a)+(b) 듀얼 GT가 있는 유일 레짐. 3-arm:
  A=fp32-throughout(기존 track_d rundir 재사용, 0원), B=bf16-mixed 학습+fp32 평가·oracle,
  C=(선택) bf16-mixed 학습+bf16 평가(음성 대조 — 06-07 실패 재현으로 sanity).
- **측정(위계 순)**: ① fidelity — B에서 estimator vs (b) Spearman·Pearson이 A와 같이
  1.000/0.9999+ 유지되는가; (a)-fp32-eval vs (b) — 06-07의 +1.000이 bf16 궤적에서
  유지되는가(이게 "학습 vs 평가" confound를 처음으로 분리). ② 신호 크기 — φ spread,
  per-round Δ, Σφ가 A 대비 몇 % 이동하는가(진단 §1.3d와 같은 표). ③ AUROC(부차).
- **비용**: B arm = 학습 bf16이라 A(10.4h/셀 급 probe 실측)보다 싸다 — 방법 스위트를
  Flirds/Flirds-1st/(b)/(a)로 트리밍하면 셀당 ~4–6h 추정 [추측]; seed0 파일럿 B+C ≈
  **1 GPU-day 미만**, 3-seed 확장 ≈ 2–3 GPU-day.
- **판정선**: fidelity 1.000 유지 + (a)-vs-(b) ≥ +0.9 + 신호 크기 이동 ≤ seed 분산이면
  "옵션 ② 안전" — 그때 비로소 전환/신규 실험 적용 여부를 다시 결정.

### 3.3 부속 — protocol.md §1 문서-코드 불일치 사실관계 타임라인

(정찰 세션 git log + raw Grep 추적; 인용 라인 이 세션 재확인)

| 시점 | 사건 | 근거 |
|---|---|---|
| 2026-05-19→22 | 파일럿 bf16 평가 artefact 의심 → "**fp32 eval 강제**, training precision은 별도 결정" | raw `2026-05-19-section23-walkthrough.md:134,147,307,399` |
| 2026-05-27 | Yonghee "training부터 모든 걸 fp32로 하지 않을 이유가 있어?" → bf16 mixed가 학습 표준이라는 답변 → "fp는 관행대로" → **protocol §1 = bf16 train / fp32 eval로 lock** | raw `2026-05-27-section-23-lock.md:84-88`; `flirds-protocol.md:19-25` |
| 2026-06-04 | 첫 LLM FL 구현이 **실제로는 전부 fp32** (`bf16=False` 포함, 커밋 f25daa4) — 같은 날 raw에는 여전히 bf16 계획 문구 | git f25daa4; `llm_server.py:46`; raw `2026-06-04-phase1-llm-stage2.md:79` |
| 2026-06-07 | task6: bf16 (a)-val-loss 실패(+0.3) → 원인=bf16 정밀도(diff 0.005–0.02 < 0.009) → fp32 재실행 +1.000. 교훈 lock: "(a)-val-loss 검증은 항상 fp32; ROUGE-only만 bf16" | raw 06-07:66-73,107-108 (커밋 aa70a98); `exact_sv_llm.py:16-19` |
| 2026-06-09 | task8: "**7B = fp32 + small batch, no bf16**" → §13(:224)의 "7B bf16 train" 사실상 폐기 | raw `2026-06-09-…-task8.md:48-49`; `phase2_matrix.py:34` 주석(현행 정책의 실질 선언) |
| 2026-06-13 | Track D: "fp32 no-quant = oracle/estimator precision floor"를 deviation caveat으로 명문화 | `track_d.py:20-23` |
| 2026-07-02 | 신호크기 진단: fp32는 병목 아님(신호가 floor의 10²–10⁴배); "bf16이었다면 φ 차이 ~1e-3이 ~8e-3에 묻힘 — fp32 관례가 맞았음" | `wiki/flirds-signal-size-diagnosis.md` §0-1·§1.1 |

**요지**: "bf16 학습"은 05-27 시점의 **계획**이고, 코드는 첫 구현(06-04)부터
fp32-throughout이었으며, 06-07·06-09를 거쳐 fp32가 사실상의 정책이 됐다. protocol §1 표와
`flirds.md:130` / `flirds-implementation-plan.md:78,112,242,301` / protocol §13:224의 bf16
서술은 갱신되지 않은 잔재다. **정정 방향은 옵션 선택과 동치**: 옵션 ① 채택 시 → 문서를
코드에 맞춰 개정(§1 표를 fp32-throughout으로 + "bf16-train은 미검증 옵션" 각주 + 전파된
5곳 수정 + (a) 러너 bf16 기본값 정리); 옵션 ② 채택 시 → 코드를 문서로 이행(§3.2-(6) 검증
통과가 전제 + precision_guard 구현 + §1에 06-07 교훈("(a)-val-loss는 fp32 평가") 반영 +
§13의 7B 서술은 06-09 결정과 재조정). 어느 쪽이든 **현 상태(문서≠코드, deviation 선언
없음)는 protocol 자신의 scope rule(:13 "silent deviations are not allowed") 위반 상태**다.

---

## Yonghee 결정 필요

1. **정밀도 정책 (§3)**: 옵션 ①(fp32-throughout 유지 + protocol §1을 코드에 맞게 개정)
   vs 옵션 ②(검증 실험 §3.2-(6) 선행 후 학습 bf16 이행). 문서는 판단 재료만 제공 —
   비용·확장성(①의 약점) vs 검증 부담·비교 단절(②의 약점).
2. **CNN tf32 (§2)**: (i) yonsei 박스 실측(부록 B) + A/B 스모크(§2.4, ~1–2.5h)를 돌릴지,
   (ii) 결과에 따라 `cudnn.allow_tf32=False` 1줄을 CNN 진입점에 넣을지 vs 논문 각주로
   처리할지. (스모크 전이라도 논문에 CNN 각주는 필요 — §2.3-4.)
3. **(a) oracle 러너 기본값 (§1.4)**: `phase2_llm_a_oracle.py:119`의 bf16 기본값을 fp32로
   뒤집을지(06-07 교훈상 val-loss 용도 기본값으로는 위험한 방향), 아니면 run 스크립트에
   `A_DTYPE=fp32`를 명시할지. 코드 수정 금지 규약상 이 문서에서는 제안만.
4. **precision_guard (§1.3)**: 어느 옵션이든 (b) oracle의 "로그 dtype 조용히 상속" 구조는
   가드 없이 남아 있음 — protocol이 명시한 `precision_guard.py`(:198) 구현 여부.

## 후속 실험 제안

| # | 실험 | 비용 | 답하는 질문 |
|---|---|---|---|
| P1 | yonsei 박스 런타임 판별 (부록 B 스니펫) | ~1분 | CNN 그리드가 실제 TF32 커널로 돌았는가 (전제 확정) |
| P2 | CNN TF32 A/B 스모크 (§2.4; cifar10 iid+label_flip seed0 × on/off) | ~1–2.5h (3090) | iid 세부 순위·fidelity의 TF32 민감도; headline 불변 확인; `allow_tf32=False` 채택 근거 |
| P3 | bf16-train 검증 A/B (§3.2-(6); 1B anchor5 3-arm, seed0 파일럿) | <1 GPU-day (파일럿), 2–3 GPU-day (3-seed) | 옵션 ② 채택 가능 여부 — 학습 정밀도가 fidelity·신호크기·(a)-vs-(b)에 미치는 영향의 최초 분리 측정 |
| P4 | (선택) TF32-matmul-on fp32 학습 속도 실측 (1B 스모크, `matmul.allow_tf32=True`) | ~1h | 옵션 ①을 유지하면서 학습만 TF32로 여는 중간안의 이득 크기 (§3.1-(4) 미실측 해소) — 단 이 중간안도 궤적 변경이므로 P3와 같은 검증 필요 |

작성 중 스스로 내린 소결정(근거): ① 문서 파일명·수치 표는 rundir 재계산 포함(부록 A) —
"모든 수치에 출처" 규약상 계산 과정 재현 가능해야 하므로. ② raw 06-07 표의 fp32 N=10
"~67 GPU-hr"를 본문에서 [재계산] ~128 GPU-hr로 교정 병기 — 산술 불일치가 명백하고(126min×32와
정확 일치) 옵션 비교의 핵심 수치라 그대로 인용하면 옵션 ①의 비용을 과소평가하게 되므로.
③ P4를 "옵션 ③"으로 승격하지 않음 — 궤적 변경이라는 점에서 옵션 ②와 같은 검증 부담을
지므로 별도 옵션이 아니라 ②의 변형으로 취급.

---

## 부록 A — CNN φ 신호 스케일 재계산 (§2.2 표의 출처)

`runs/track_c/c1/<cell>/phi.parquet`의 `phi_(b)oracle` 열을 셀별로 정렬해
spread(max−min)·인접 갭(min/median)을 계산 (이 세션, pandas):

```python
import json, glob, os, numpy as np, pandas as pd
root = "runs/track_c/c1"
for d in sorted(glob.glob(os.path.join(root, "*_seed*"))):
    phi = pd.read_parquet(os.path.join(d, "phi.parquet"))["phi_(b)oracle"].to_numpy()
    s = np.sort(phi); g = np.diff(s)
    print(os.path.basename(d), phi.max()-phi.min(), g.min(), np.median(g))
```

30셀 전체 수치 (spread / min-gap / med-gap):

```
cifar10_feature-noise seed0-2: 0.106/9.6e-4/1.0e-2 · 0.078/2.4e-3/8.8e-3 · 0.134/1.3e-3/1.8e-2
cifar10_iid           seed0-2: 0.029/6e-5/1.6e-3 · 0.021/1.7e-4/1.7e-3 · 0.037/4.4e-4/1.5e-3
cifar10_label-flip    seed0-2: 0.119/1.3e-3/1.5e-2 · 0.093/1.3e-3/9.9e-3 · 0.120/2.2e-4/1.2e-2
cifar10_label-skew    seed0-2: 0.071/7.7e-4/5.5e-3 · 0.048/2.6e-4/2.5e-3 · 0.068/4.8e-4/4.8e-3
cifar10_quantity-skew seed0-2: 0.220/3.2e-3/2.6e-2 · 0.194/4.2e-3/1.1e-2 · 0.176/1.1e-4/1.6e-2
mnist_feature-noise   seed0-2: 0.008/2.1e-4/5.8e-4 · 0.011/2.5e-4/1.0e-3 · 0.013/4.0e-5/7.7e-4
mnist_iid             seed0-2: 0.013/8.0e-5/1.2e-3 · 0.009/1.0e-4/9.6e-4 · 0.009/4.0e-5/8.1e-4
mnist_label-flip      seed0-2: 0.386/8.6e-4/1.4e-2 · 0.326/3.7e-4/1.0e-2 · 0.360/7.6e-4/1.1e-2
mnist_label-skew      seed0-2: 0.042/8.0e-5/3.8e-3 · 0.023/1.4e-4/1.2e-3 · 0.036/2.8e-4/2.3e-3
mnist_quantity-skew   seed0-2: 0.338/3.1e-3/1.6e-2 · 0.347/2.2e-4/6.5e-3 · 0.391/3.8e-4/1.3e-2
```

iid fidelity 참고치(metrics.json `spearman_b` 직접 확인): cifar10 iid Flirds
{0.939, 0.952, 0.964} / Banzhaf {0.976, 0.988, 0.988}; mnist iid Flirds {0.842, 0.636, 0.964};
cifar10 label-flip Flirds {1.0, 0.988, 1.0}.

## 부록 B — 런타임 판별 스니펫 (yonsei 박스용; 실행 안 함)

정찰 세션 감사 노트 §6의 초안. 핵심: 플래그 값 + **실측 오차 판별**(fp64 참조 대비 conv
오차 ~1e-4↓=IEEE fp32 커널 / ~1e-2↑=TF32 커널), `cudnn.deterministic=True` 조건 재현 포함.

```python
import torch
print(torch.__version__, torch.cuda.get_device_name(0))
print("matmul_tf32", torch.backends.cuda.matmul.allow_tf32)      # 기대 False
print("cudnn_tf32 ", torch.backends.cudnn.allow_tf32)            # 기대 True(기본)
print("f32mp      ", torch.get_float32_matmul_precision())
for lbl, obj in [("matmul", torch.backends.cuda.matmul), ("conv", torch.backends.cudnn.conv)]:
    print(f"fp32_precision[{lbl}]", getattr(obj, "fp32_precision", "(API 없음)"))
torch.manual_seed(0)
x = torch.randn(8, 3, 32, 32, device="cuda"); w = torch.randn(16, 3, 5, 5, device="cuda")
def err():
    return (torch.nn.functional.conv2d(x, w).double()
            - torch.nn.functional.conv2d(x.double(), w.double())).abs().max().item()
print(f"conv err (default)      = {err():.3e}")
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
print(f"conv err (deterministic) = {err():.3e}   # ~1e-4↓=IEEE fp32, ~1e-2↑=TF32")
```
