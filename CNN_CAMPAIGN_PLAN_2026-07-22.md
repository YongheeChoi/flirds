# CNN 캠페인 종합 계획 — downstream · fidelity · detection (2026-07-22)

> **용도**: 타 세션 교차 검증용 핸드오프. 이 세션(07-22)에서 결정·구현·제출된 것과 설계
> 확정분 전부를 담는다. 스코프 = **CNN 트랙만** (LLM legs: R4/P5/β0.3 등은 별도 문서).
> 수치 canon: rundir → overview → paper. 이 문서의 수치는 전부 rundir/로그/코드 실측.
> 검증 후 이 문서는 삭제 가능(내용은 runs/track_g/README.md·REMAINING.md·커밋에 분산 존재).

## 0. 세 축과 질문 위계

핵심 질문 위계(Yonghee 2026-06-12, 루트 CLAUDE.md): **1차 = fidelity**(oracle 대비 기여도
정확도) → **2차 = ①성능 ②수렴 ③탐지(마지막)**. 이 캠페인의 세 축:

| 축 | 실험 | 위계 |
|---|---|---|
| **downstream** | Track G CNN 게이트 개입 그리드(+Track H 점수원 경쟁) | 2차 ①② |
| **fidelity** | 신규 leg: C2 무대 동결 궤적 위 9방법 vs (b) oracle | 1차 |
| **detection** | 별도 실험 아님 — 위 두 leg에서 읽는 지표 | 2차 ③ |

## 1. 공통 무대 (전 leg 공유 — `codes/experiments/track_c2.py` `build()`+`CFG["full"]`)

### 1.1 FL 상수 (변경 금지; 기존 그리드 verbatim)

| 항목 | 값 |
|---|---|
| N / 참여 | 100 / 라운드당 균등 10명 (frac=0.1) |
| R / E / lr / batch / opt | 120 / 5 / 0.01 / 64 / **SGD momentum=0** (프로젝트 고정 규약) |
| val / test | test 10k을 **split seed 0 고정** perm으로 val 2,000 + test 8,000 분할 (seed 무관 동일) |
| r2t target | acc 0.6 |
| 게이트 HP (전 셀 고정, 셀별 튜닝 금지) | burn_in=10, τ=0.0, min_obs=2, probation_every=5, decay=1.0, z_c=1.5, α_w=1.0, conf_z=1.645 |

### 1.2 데이터셋·모델

| 데이터셋 | 모델 | 파라미터 | Δw 크기 | 클라당 데이터(iid) |
|---|---|---|---|---|
| cifar10 | FedSVCNN | 2,156,490 | 8.63 MB | 500 |
| fmnist | LeNet5 | 61,706 | 0.25 MB | 600 |

- **mnist는 C2 무대 미지원**(MODEL_FN 맵에 없음). mnist = fidelity(N=10, track_c1) canon,
  fmnist = N=100 개입 무대(C2) canon — Yonghee 07-22 "fmnist 유지" 확정.

### 1.3 파티션 4종 = 2×2 skew 분해 (실측: cifar10 N=100 seed0)

| 파티션 | 정의 | 크기 min/med/max | 클래스/클라 | 순수 축 |
|---|---|---|---|---|
| `iid` | 균등 셔플 분할 | 500/500/500 | 10.0 | 없음 |
| `shard` | McMahan 2-shard | 500/500/500 | **1.95** | label만 |
| `qskew` | GTG quantity-skew (**07-22 신규**, C1 규칙 재사용) | **40/510/960 (24×)** | 10.0 (클라 내 균형) | size만 |
| `dir1` | Dirichlet α=1 | 187/488/1168 (6.2×; seed1 5.1× seed2 3.8×) | 9.87 | label+size 혼합 |

- `qskew` 크기 규칙 = `fl.partition.gtg_quantity_ratios(n)` = pair p → 10+5p (단일 정의;
  track_c1 `_quantity_ratios`는 이 함수에 위임, 값 불변 — tests/test_partition_qskew.py).
- ⚠️ **가법 분해 아님**: shard의 label-skew(1.95 cls)는 dir1(9.87)보다 세고, qskew의
  size-skew(24×)도 dir1(6.2×)보다 세다. "dir1 = shard+qskew" 검산 금지, **축 귀속만** 읽는다.

### 1.4 위협 8종 (오염 클라 규약 포함)

| 위협 | 기전 | 상수 | 오염 클라 |
|---|---|---|---|
| `clean` | — | — | 0 |
| `free_rider` | Δw = 0 (update-level) | — | 정확히 40 |
| `frrand` (**07-22 신규**) | Δw를 **순수 랜덤** U(−s,s)로 교체; s = √3·std(해당 클라의 정직 would-be Δw)·FRRAND_MULT(=1.0) → benign per-entry std 정합(norm 필터 무력) | FRRAND_MULT=1.0 | 정확히 40 |
| `grad_noise` | 정직 Δw + N(0, σ²) | σ=0.1 (GAMMA_GRADNOISE) | 정확히 40 |
| `label_flip@{0.15,0.35,0.70}` | 고정 dose per-client flip (Track G; Stage 0 감사가 부호-교차 span ~0.13–0.55 관통하도록 선정) | C2_FLIP_RATE | ~Binomial(100, 0.4) |
| `label_flip strmain` (**07-22 신규 셀**) | FedCorr (ρ,τ): rate ~ **U(0.5, 1.0)** per-client | TAU=0.5 | ~Binomial(100, 0.4) |

- MAL_FRAC=0.4. **fr·frrand·gn 3종은 같은 seed에서 같은 40명**(`default_rng(1000+seed)`의
  동일 choice draw) → 직접 대조 가능. label_flip은 Bernoulli 경로라 집합 다름.
- update-level 3종 = **신호/노이즈 사다리**: free_rider(신호0·노이즈0) / frrand(신호0·노이즈全)
  / grad_noise(신호+노이즈). frrand의 per-(client,round) generator seed = `seed+1000·c+r`.
- **strmain은 C2 소프트 그리드(runs/track_c/c2)의 원래 표준**(FLIP_RATE env 미설정 경로 =
  U(TAU,1) draw). Track G가 고정 dose를 도입하며 축이 갈라졌던 것을 다시 잇는 것 — 재실행
  아님, 셀 추가. `str0.6/0.8` 셀의 str은 rate가 아니라 **오염 비율 ρ** 노브임에 주의.
- poison 제외·top-k 없음·Banzhaf/Ripple 제외(§7 금지 + 07-22 결정).

### 1.5 재현성·환경

- `seed_everything(seed, cudnn_deterministic=True)`; **fedavg가 진입 시 재시드** → arm
  순서·셀 분리와 무관하게 같은 (config, seed) 궤적은 **비트 동일**. TF32-on 유지(fa2c167).
- 실행: yonsei Slurm `base_suma_rtx3090`(RTX3090), PY=`~/anaconda3/envs/lora4cl/bin/python`
  (torch 2.11.0+cu130, numpy 2.4.4). **QOS 유저당 8 GPU**. slurmdbd 죽음 → 성공 판정 =
  로그 `TRACK-C2 RUN OK` + rundir 존재. 로그는 NFS(리포 내) 필수. cifar10 `~/data` 실사본.
- **스택 경계**(감사 M1): 구세대 rundir = torch 2.12.0/B200(git 69cb6bf·a92a2d6, dirty),
  신규 = 2.11.0/RTX3090(**커밋 후 제출 = git clean**). 동일-config 18셀 실측 drift:
  clean·fr **|Δacc| ≤ 0.0020**, grad_noise ≤ 0.0241(vanilla 0.23–0.28 붕괴 레짐 = seed 산포급).
  Track G는 restack 재실행으로 표를 단일 스택화(§2.1); Track H는 P1 leg(2.12)↔P5 leg(2.11)가
  이미 경계를 갖고 있으나 strmain 신규 위협에서는 전 셀 단일 스택(2.11).

## 2. Downstream — Track G CNN 게이트 그리드

### 2.1 셀 매트릭스 (완성 시 144셀 = 6콤보 × 8위협 × 3seed; 셀당 1런)

(ds, part) 콤보 6 = cifar10×{iid, dir1, shard, qskew} + fmnist×{iid, dir1}.
fmnist×{shard, qskew}는 **옵션(기본 제외, Yonghee 지시 시만)** — 원 지시 §1.

| 잡 | 내용 | 런수 | 상태(07-22 22시) |
|---|---|---|---|
| (기존) | cifar10×{iid,dir1}×{clean,fr,gn,lf@3점}×3s — torch 2.12/B200/69cb6bf(dirty) | 36 | 완료, **read-only 동결** |
| 1860256 `gskew` | A: {c10-shard, c10-qskew, fm-iid, fm-dir1}×7위협×3s(84) + B: c10×{iid,dir1}×frrand 백필(6) | 90 | 큐/실행 중 |
| 1860257 `grestack` | 기존 36셀을 현 스택 재실행 → `rundirs_cnn_restack/` (원본 보존; drift 표 자동) | 36 | 큐 |
| 1860471 `gstrmain` | {c10×4part, fm×2part}×strmain×3s | 18 | 큐 |

- rundir: `runs/track_g/rundirs_cnn/<ds>_<part>_<threat-tag>_g_seed<N>` (threat-tag:
  `clean`·`free-rider`·`frrand`·`grad-noise`·`label-flip_fr0.15/0.35/0.70`·`label-flip_strmain`);
  restack만 `rundirs_cnn_restack/` 동명. **기존 rundir 무수정 원칙.**
- 인덱스 규약: 전 sbatch **seed-major** (`--array` 앞 절단 = seed0 파일럿).

### 2.2 arm (오염 9종 / clean 7종 — excl 2종 제외)

| arm | 기계 |
|---|---|
| `vanilla` | 무개입 n-가중 FedAvg |
| `oracle_excl` / `random_excl` | 진짜 오염 / 동수 무작위 **고정 제외** (상한/통제) |
| `flirds_gate_v1` | 라운드 집계-게이트: 該라운드 raw≤τ 델타 집계 제외 |
| `flirds_gate_v2` | **참여-게이트**: 누적 cum≤τ 학습 제외 (burn-in·min_obs·probation) + V1 스크린 |
| `flirds_zgate_v2` | 코호트-상대 z-게이트 (cum z<−1.5 제외) |
| `flirds_gatew_v2` | V2 선택 + w∝n·max(cum,0)^α (α=1) |
| `flirds_gatew_v1` | per-round raw 크기 가중 (CNN ablation) |
| `flirds_mult` | 기존 min-max EMA mult β=0.5 (소프트 대조) |

부호 규약(리포 관례): `phi.parquet` 저장 φ = **suspicion**(도움=음수) / 게이트가 읽는
raw·cum(`phi_rounds.parquet`) = **contribution**(도움=양수 = −φ). 게이트 규칙 = strict
`cum > τ`(τ=0) 포함.

### 2.3 지표·분석 (runs/track_g/make_analysis.py — 07-22 확장판)

- **절대 acc 표**(고정 결정; 상대 dAcc 표기 금지) + Δacc + recovery=(arm−van)/(oracle−van).
- **recovery 분모 가드(07-22 신규)**: |oracle_excl−vanilla| < **0.02** → recovery 공란,
  Δacc만. 실측 근거: lf@0.15 분모 iid 0.0033/dir1 0.0064 → recovery seed-std 3.14/1.58 폭주.
  실측 분모: fr 0.027/0.032 · gn 0.379/0.377 · lf@0.35 0.039 · lf@0.70 0.114/0.099.
- dose 백필: 기존 36셀 config에 flip_rate 키가 없어(07-21 이후 추가됨) 셀 **이름**에서 추출.
- **게이트 제외집합 사후 재구성**(CNN 셀은 gate 메트릭 블록이 없음): phi_rounds에서
  V2 = "r≥burn_in에서 round-(r−1) 종료 스냅샷 기준 n_obs≥min_obs ∧ cum≤τ" / V1 = "참여자
  raw≤τ" → micro pair P/R + clean 오발화(pairs·clients). probation 복귀는 판정에 불포함.
  기존 36셀은 `CNN_GATE_DEFAULT`(burn_in10/τ0/min_obs2) 폴백; 신규 셀은 config `gate` 블록
  자기기술(07-22 track_c2 수정).
- restack_merge: 표는 restack 값 사용(단일 스택), orig-vs-restack **drift 표** 자동 출력.
- 산출: `analysis/{cnn_summary.csv, cnn_cellmean.csv, README.md}` — 2×2 분해표(절대 acc +
  recovery), 사전등록 자동 대조, C2 같은-셀 대조.

### 2.4 사전등록 예측 (runs/track_g/README.md "확장 ②" — 실행 전 등록 완료, MISS 그대로 보고)

| id | 예측(요약) | crisp check |
|---|---|---|
| H-K1 | free_rider 회복 파티션-불변 (기존 iid .808/dir1 .838) | shard·qskew recovery(V2) ≥ 0.6, 4파티션 spread < 0.35 |
| H-K2 | **frrand도 잡힘** — 1차항 평균0이나 2차항 ½ΔwᵀHΔw ⪰ 0이 d~10⁵서 압도 | recovery(V2) ≥ 0.7; 반증 시 "CNN도 2차항 약함=LLM 감사 코인플립과 일치"로 보고 |
| H-K3 | clean 오발화 shard 최대(기존 pairs iid 561/dir1 3808) + shard Δacc<−0.006, qskew는 parity 유지 | 표기됨 |
| H-K4 | qskew서 seed 분산 최대(오염 추첨이 크기 무관 → 제외 mass 24× 요동) | sd(qskew) > 1.5·sd(iid) |
| H-K5 | lf@0.15 분모 4파티션 전부 < 0.02 (**기존 데이터로 이미 HIT**) | gap < 0.02 |
| H-K6 | fmnist 효과 크기만 축소, recovery는 cifar10 ±0.15 내 | 분모≥0.02 셀만 |

### 2.5 C2 소프트-arm 같은-셀 대조 (runs/track_c/c2, read-only 90 rundir)

- 대응 존재: {cifar10, fmnist}×{iid, dir1, shard} × {clean, free-rider, grad-noise} strmain
  + **label-flip strmain**(07-22 셀 추가로 처음 대조 가능). C2 쪽 비교 arm = vanilla·flirds_mult.
- 대응 없음(표에 명시): **qskew · frrand** 전체, label-flip 고정 dose 3점.

### 2.6 비용 실측

cifar10 셀: clean(7-arm) ~33분 / corrupt(9-arm) 41–43분 → **arm당 ~4.7분**(자기 궤적 포함).
fmnist 미실측(LeNet5, 대략 0.3–0.6×). 큐 총량(Track G 3잡 144런) ≈ **75–95 GPU-h**, 8-GPU
wall ~10h. ⚠️ 사전 추정이 1.6× 빗나간 전례(23–26분 추정 → 33–43 실측) — 이후 산정은 실측 앵커만.

## 3. Downstream — Track H 점수원 경쟁 strmain 확장 (+51런, job 1860727)

### 3.1 기존 구조 (runs/track_h/rundirs_cnn, 204 rundir — read-only)

무대: **cifar10 dir1 고정** × {clean, lf@**0.70**, free-rider, grad-noise} × 3seed.
셀타입 17종 × 4위협 × 3seed:

| 셀타입 | arm 구성 | 개수 |
|---|---|---|
| `<src>` (P1; **7종 — flirds 없음**) | `<src>_{gate_v2, gatew_v2, mult, zgate_v2}` (4 arm) | 84 |
| `<src>p5` (8종) | `<src>_{cgate, pweight}` (P5-hard/soft, 2 arm) | 96 |
| `obs` | observer(8소스 동시 채점) + T2 legacy 재학습(t2_sign/t2_signw/t2_random) | 12 |
| `obsp5` | observer + T2 P5 재학습(t2_csign/t2_pw; T2_LEGACY=0) | 12 |

- **flirds P1이 없는 이유**: flirds 게이트 arm들이 곧 Track G 그리드의 arm — Track G 셀이
  그 역할. strmain의 flirds P1도 job 1860471(cifar10_dir1_label-flip_strmain_g_*)이 담당.
- 점수원 8종 = `fl.score_providers.SOURCES`: flirds, flirds1st, lossheur, gtg, fedsv,
  comfedsv, shapleyfl, fedif (per-round raw provider).

### 3.2 strmain 확장 (5번째 위협; **추가일 뿐 기존 lf@0.70 결과 유효 — 재실행 아님**)

- 17 셀타입 × 3seed = **51런** 제출됨(1860727, seed-major, `runs/track_h/sbatch_strmain.sh`).
  명명: `cifar10_dir1_label-flip_strmain_<tag>_seed<N>` (tag = `<src>`/`<src>p5`/`obs`/`obsp5`).
- **과학적 동기**: 고정 dose 0.70은 오염 클라가 전부 동일 강도 → 경계 사례 부재. strmain은
  rate~U(0.5,1) 연속이라 **경계 클라(rate~0.5)가 처음 생김** = P5(신뢰 게이트: "경계 분산을
  확신적 해악과 다르게 취급")의 설계 의도가 **처음으로 시험되는 무대**.
- T2(사후 재학습, deployment semantics=위협 유지·고정 제외) 포함 제출 — obs/obsp5 셀.
- 비용: obs 셀 실측 앵커 ~2.2h(fr0.70), P5 2-arm ~12분급 → 총 **25–40 GPU-h** 추정.
- 분석: `runs/track_h/make_analysis.py`가 THREATS 튜플에 strmain 미포함 — **완료 후 확장 필요**
  (현재 하드코딩 `("clean","label_flip","free_rider","grad_noise")` + fr0.70 태그 가정).

## 4. Fidelity leg — C2 무대 동결 궤적 fidelity (신규 설계; 구현 대기)

### 4.1 확정 설계 (Yonghee 07-22; "전면 대조표에서 전부 오른쪽(C2), 예외 3개는 왼쪽(C1)")

| 축 | 값 | 출처 |
|---|---|---|
| 무대 전부 (N·참여·R·E·lr·데이터셋·파티션·위협) | §1 그대로 (C2) | 오른쪽 |
| **궤적** | **동결 vanilla 1개**(selection·개입 없음), on_round 로깅 | 왼쪽(C1 방식) |
| **정답** | **(b) in-run oracle만** — `in_run_shapley_perround`. **(a) 2^N 재학습 포기**(N=100 불가) | 결정 |
| **방법 9종** | Flirds, Flirds1st, GTG, FedSV, ComFedSV, ShapleyFL, FedIF, loss-heur, **Fed-LOO**. **Ripple·Banzhaf 제외** | 결정 |
| **지표** | C1 세트(§4.4) | 왼쪽 |

### 4.2 (b) oracle

`in_run_shapley_perround` (`flirds/oracle/in_run_sv.py:183`): 라운드별 2^{|P_r|} 부분게임
분해 — **2^N 전열거와 정확히 같은 값**(동치 검증 = `experiments/phase2_crossdevice_oracle_smoke.py`;
LLM device100·task7 기실전 사용). 비용/셀 = Σ_r 2^{|P_r|} = 120 × 2^10 = **122,880 utility 평가**
(평가 1회 = perturbed param 합성 + val 2,000 forward). 라운드 독립이라 필요시 라운드 샤딩 가능.

### 4.3 방법 9종 — from-logs 경로·부호 (track_c1.run_seed 방법 블록 미러)

| 방법 | 함수 | 부호 처리(good→**low** 통일) | 참여 10/100 주의 |
|---|---|---|---|
| (b)oracle | in_run_shapley_perround | 그대로(손실변화 게임: 도움=음수) | 기준 |
| Flirds / Flirds1st | flirds_values(second_order=T/F) | 그대로 | — |
| GTG | gtg_from_logs(round_trunc=0, eps=0) | 그대로 | 라운드 코호트 위 MC |
| FedSV | fedsv_from_logs(trunc_eps=0) | 그대로 | 〃 |
| ComFedSV | comfedsv_from_logs(**partial=True**) | **−negate** | 부분관측 행렬 → 저랭크 완성 **활성**(C1 전참여선 무의미했음) |
| ShapleyFL | shapleyfl_from_logs(β=0.3) | **−negate** | — |
| FedIF | fedif_from_logs | **−negate** | — |
| loss-heur | in_run_singletons (C6 캐시 경로) | 그대로 | — |
| **Fed-LOO** | in_run_loo | 그대로 | — |

개수 검산: C1의 11 = 위 9 + Banzhaf + Ripple. C2 `SOURCES` 8 = 위 9 − Fed-LOO(온라인
게이팅 레지스트리에 from-logs 전용인 Fed-LOO가 없었을 뿐, 구현은 기존재).

### 4.4 지표 (C1 세트 그대로)

- 값/순위 fidelity vs (b): **Spearman · Kendall · Pearson** + 거리 3종(cos/euc/maxdiff).
- **탐지 AUROC**(오염 셀): detection_auroc(φ, corrupt-이진) — good→low라 오염=고점.
- **spearman_vs_rate**(strmain 셀만): φ vs per-client 실현 flip rate 순위상관 = **oracle-무관
  강도응답 ruler**("더러울수록 φ가 단조 하락하나" — AUROC보다 강한 검사; 고정 dose 셀은
  rate가 2값뿐이라 AUROC와 동일 정보 → 정의 불가). 변형 2개 산출 권고: ①전 클라(rate 0
  포함; C1 호환) ②corrupt-only 40클라(용량 해상도).
- 방법별 wall-clock (+PhaseTimer timing.json).

### 4.5 게임 의미 캐비엇 (서술 필수)

10/100 참여라 클라당 기대 참여 **~12/120 라운드** — φ = "참여 라운드 부분게임 기여 합".
C1(N=10 전원참여 R=10)과 **다른 게임** → C1 표와 수치 직접 비교 금지, **"배포(cross-device)
레짐 fidelity"**로 별도 절 서술. (b)가 같은 게임의 exact 값이므로 fidelity 정의는 자기완결.

### 4.6 구현 계획 (안 A = from-logs 포팅) — ✅ 구현 완료 2026-07-23 (하단 상태 블록)

1. 신규 러너 (제안: `codes/experiments/track_c2_fid.py`) — `track_c2.build()` **재사용**
   (파티션·위협·마스크 규약 자동 상속), `fedavg(..., on_round=logs.append)`로 동결 궤적
   + 로그 캡처, 이후 §4.3 방법 배터리 + 지표 = track_c1 미러.
2. **로그 CPU 강제**: cifar10 로그 ≈ 120r × (10Δ+1 w_r) × 8.63MB ≈ **11.4 GB** — GPU(24GB)
   불가, CPU(64G 노드) OK. 델타 캡처 시 `.cpu()` 확인 필수. fmnist ≈ 0.33 GB.
3. **rates 영속(신규 요구)**: spearman_vs_rate에 per-client 실현 rate 필요한데 track_c2는
   corrupt 이진만 저장. build() 내 rate draw는 파티션 경로와 rng를 공유(dir1 빈 클라 보정
   draw가 끼어듦)하므로 **build 시점 캡처**가 안전(모듈-레벨 사이드채널 등 추가 방식은
   비트-중립으로; build 반환 시그니처 변경은 기존 콜러 회귀 확인 필수).
4. rundir: 제안 `runs/track_c/c2fid/<ds>_<part>_<threat-tag>_fid_seed<N>` (**열린 점** §7-2).
5. 스모크: smoke 모드(n=20, R=4) 1셀 — (b) 대수검증(perround vs 전열거 2^N smoke 규모 대조),
   부호 방향(오염 φ>clean φ), 지표 산출 확인.

**→ 구현 완료(2026-07-23, 이 세션)**: `codes/experiments/track_c2_fid.py` — build() 재사용
+rate 캡처(corruptor 랩, 비트중립)+로그 CPU 강제+**(b) per-round 청킹**(라운드별
`in_run_shapley_perround` 1콜 = GPU 1회 스테이징+phi_b_rounds.parquet 영속+라운드 샤딩
`C2FID_B_ROUNDS` 일체 해결)+지표 C1 세트+svr 양변형. `codes/tests/test_c2fid.py` **5개
green**((b)청크=2^N 동치 / 샤드 병합+커버리지 / efficiency=독립 U(N) / rate 캡처 비트중립 /
궤적 조인 비트동일). fmnist smoke e2e green(eff-gap **0.0**, 9방법 전부 부분참여 산출,
phi.parquet 열-호환 확인). `runs/track_c/c2fid/{README.md(1행 캐비엇+**F-1~F-4 사전등록**),
sbatch_fid.sh(144셀 seed-major; **파일럿 = `sbatch --array=11`** = cifar10 dir1 grad_noise
seed0)}. 파일럿 제출은 현 큐(195런) 종료 후.

### 4.7 범위 — ✅ 확정 144셀 (교차검증 세션 회신 07-23)

- **144 = downstream 동일 6콤보 × 8위협 × 3seed** 확정(192 반대). 근거: 궤적 조인(§4.9)이
  핵심 자산인데 downstream 쌍 없는 48셀은 조인 불가 + fmnist 는 효과크기 축소 무대(H-K6).
- 여유 시 fmnist×{shard,qskew} +48 은 **downstream 쌍으로 동반** 추가(fidelity 단독 금지).

### 4.8 순서·비용 (Yonghee 확정: **구현 대기 + 1셀 파일럿**)

현 큐(195런) 종료 → **cifar10 dir1 grad_noise seed0 1셀 파일럿** → 실측 GPU-h 보고 →
본런 144셀. **파일럿 제출됨 = job 1861067**(`--array=11`,
`--dependency=afterany:1860256:1860257:1860471:1860727` → 현 큐 4잡 전부 끝난 뒤 자동 기동;
슬롯 선점 없음. 취소는 `scancel 1861067`). 산술 추정(검증 전): 궤적 ~5분(42분/9arm 실측서 역산) + from-logs 9방법 +
(b) 123k 평가 → cifar10 셀 **0.75–1.5h**, fmnist 0.3–0.6h → 144셀 ≈ **80–150 GPU-h**.
⚠️ 이 추정은 §2.6의 1.6× 전례 때문에 파일럿 실측으로만 확정.

### 4.9 궤적 조인 (보너스; 같은-스택 셀만)

fidelity 궤적 = 같은 (config, seed)의 **Track G vanilla arm 궤적과 비트 동일**(fedavg 재시드
규약). → 방법 φ(1차) ↔ 같은 궤적의 downstream 게이트 성과(2차)를 **동일 궤적 위에서** 조인
가능. Track H obs 셀(dir1)과도 (threat, seed) 일치 시 동일 궤적 — observer 온라인 cum과
from-logs φ 교차검증 가능. 단 게이트 arm들의 궤적은 개입으로 갈라지므로 조인은 "vanilla
궤적 위 φ ↔ 개입 성과 상관"이지 인과 동일성 아님.

## 5. Detection — 두 leg에서 읽는 지표 (별도 실험 없음)

| 출처 | 지표 | 정의 |
|---|---|---|
| fidelity leg | 방법별 φ-AUROC | detection_auroc(φ, corrupt); (b) 자기-AUROC = 천장 참조 |
| fidelity leg (strmain) | spearman_vs_rate | §4.4 |
| downstream | arm별 온라인 AUROC | roc_auc_score(corrupt, −score); 게이트 arm=−cum, mult=−s. vanilla/excl arm은 NaN |
| downstream | 게이트 micro P/R + clean 오발화 | phi_rounds 재구성(§2.3 규칙) |

- 전용 탐지기 베이스라인(FLDetector·FLTrust·FedDQC·STD-DAGMM)은 **CNN 스코프 밖**(LLM
  task5 축) — 교차검증 세션이 이견 있으면 제기.
- 서술 순서 = 위계상 **마지막**(fidelity → 성능 → 수렴 → 탐지). "clean-val-loss를 낮추는
  공격자 φ 상위 = valuation의 정직한 답" 각주 관례 유지(단 2026-07-14 감사 각주 참조).

## 6. 결정 로그 (이 세션, 2026-07-22, Yonghee)

1. skew-축 분해 완성(shard·qskew) + fmnist + 24셀/72런 — 원 지시.
2. 두 번째 데이터셋 = **fmnist 유지**(mnist 아님; mnist는 fidelity N=10 canon·C2 셀 0개).
3. **frrand 위협 신설**("gradient+random 말고 그냥 random noise 클라") — LLM frrand의 CNN 이식.
4. 제출 = 90런 전체 한 번에(파일럿 생략).
5. **스택 통일**: cifar10 iid·dir1 12셀 현 스택 재실행(restack) — drift 실측 보고 후 결정.
6. **strmain 셀 추가**(fidelity 필수 ruler) + downstream에도 동일 적용 — "시간 더 걸릴 이유
   없지 않나" = 맞음(셀당 비용 동일; 셀 수만 +).
7. 기완료 실험 재실행 **불요** 확인(가법성; C2 소프트는 이미 strmain이 표준).
8. **Track H strmain 확장 전부**(P1+P5+obs+obsp5, 51런) 즉시 제출.
9. **Fidelity leg**: 오염축 C2 통일, (a) 포기, Fed-LOO 포함, Ripple·Banzhaf 제외, 동결
   궤적(selection 없음), GT=(b), 지표=C1 세트. 순서 = 구현 대기 + 1셀 파일럿.

**(07-23, 교차검증 세션 회신 — §7 전건 해소)**

10. **fidelity 범위 = 144**(§4.7): 궤적 조인이 핵심 자산 — downstream 쌍 없는 48셀 조인
    불가 + fmnist=효과크기 축소 무대(H-K6); +48 은 쌍 동반일 때만.
11. 러너/rundir = `experiments/track_c2_fid.py` → `runs/track_c/c2fid/` 확정 + **§4.5
    게임 캐비엇을 c2fid README 1행으로 명기**(반영됨).
12. spearman_vs_rate **양변형 산출**(전클라=C1·LLM 호환 / corrupt-only=용량 해상도; 비용 0).
13. fidelity 분석 도구 = 신규 스크립트 가능하되 **산출 스키마 = c1 fidelity.csv 열-호환 +
    `stage` 컬럼**(LLM fidelity 표와 병합·논문 표 생성기 재사용).
14. **(b) 라운드 샤딩 채택**(라운드 독립=손실 0; 샤드 병합 커버리지 assert 1개 —
    `test_c2fid` 로 규약 고정).
15. Track H strmain **T2 leg 유지**(R4 T2 와의 CNN↔LLM 대칭 자산 — 사후-부호 재학습).
16. `runs/track_h/make_analysis.py` strmain 확장 동의(**완주 후** 수정).
17. **fidelity-leg 사전등록 추가**(H-K1~6 은 downstream 예측뿐이었음): **F-1~F-4** 를
    `runs/track_c/c2fid/README.md` 에 본런 전 등록 완료 — qskew 균등-합성 계열(코드 근거
    ShapleyFL·ComFedSV; 릴레이 표기 "G4·G6"는 매핑 미교환이라 FedSV 별도 행 병기) 하락 /
    fr·frrand exact-0 계열 정상 vs renorm 유령값 / 참여 10/100 std50k5 붕괴 서열 재현 /
    strmain svr **Flirds≈(b)>1st**. qskew fidelity 착지 시 **F-L2(LLM 비등n silo5)와
    cross-track 쌍** → F-L2 우선순위 근거 강화(P5c 양 트랙 실증).

## 7. 열린 질문 (교차검증 요청) — ✅ 전건 회신 완료 07-23; 답 = §6 결정 10–16

1. fidelity 범위 → **144**(결정 10). 2. 러너/rundir → **확정**(결정 11). 3. svr 변형 →
**둘 다**(결정 12). 4. 분석 도구 → **신규+열-호환 스키마**(결정 13). 5. (b) 샤딩 →
**채택**(결정 14). 6. Track H T2 → **유지**(결정 15). 7. hstrmain make_analysis →
**완주 후 확장**(결정 16).

## 8. 인벤토리

**커밋(로컬만, push 금지 — Yonghee 직접)**: `43027e7`(qskew·frrand·fmnist 구현+스모크+사전등록)
→ `525adac`(restack leg) → `0ae7307`(strmain 18런) → (이 문서+Track H sbatch 커밋).
**잡**: 1860256(90) · 1860257(36) · 1860471(18) · 1860727(51) = **195런**; 07-22 22시 기준
15셀 완료·8 실행·172 대기. 완료 감시 모니터 가동 중(실패·15셀 단위·전체 완료 알림).
**sbatch**: `runs/track_g/sbatch_cnn_{skew,restack,strmain}.sh`, `runs/track_h/sbatch_strmain.sh`.
**테스트**: `codes/tests/test_partition_qskew.py`(3)·`test_frrand_cnn.py`(4) 신규, 전 54개 green.
**07-23 fidelity leg 자산(구현 완료)**: `codes/experiments/track_c2_fid.py` ·
`codes/tests/test_c2fid.py`(5 green) · `runs/track_c/c2fid/{README.md(캐비엇+F-1~F-4
사전등록), sbatch_fid.sh(144셀; 파일럿 `--array=11`)}` — 파일럿은 현 큐 종료 후 제출.
**레거시 회귀**: fmnist×{iid,dir1,shard} HEAD-트리 vs 작업-트리 metrics.json **완전 동일**(3/3).
**인접 자산(이번 캠페인 밖)**: `rundirs_cnn_v3/`(C1_V3 12런), `runs/track_h/rundirs_cnn_{dyn,scale}/`,
`runs/track_c/c1`(N=10 fidelity 30런; mnist+cifar10 — 이 캠페인의 fidelity leg와 **별개 무대**).
**예측 원문**: `runs/track_g/README.md` "확장 ②" / 위협·게이트 정의: `codes/flirds/fl/intervene.py`
· `codes/experiments/track_c2.py` / drift 실측·분모 실측: §1.5·§2.3.
