# removal_dose — C-1~C-5 완화 실험 (removal-curve · dose-response · target-stability · AdamW)

계획·셀 매트릭스·GPU예산·caveat = **이 README 에 인라인**(원 계획 프롬프트 `PROMPT_removal_dose_c2_adamw.md`
는 실행 스테이징 완료 후 제거 — 필요 시 git `525bb75` 에서 복구). Yonghee 승인 결정(2026-07-14) 반영.
이 폴더 = 4종 완화 실험의 **신규 산출물 전용 rundir root**(기존 phase2_matrix/track_d canonical
rundir 를 덮어쓰지 않음 — `RUNDIR_ROOT` 로 분리). 리뷰어 공격 C-1~C-5(`research-wiki/survey/
review/review-claude.md`)에 대한 **상호보완 비교축**이며, (b) oracle fidelity 표는 건드리지 않는다.

## 방어 매핑 (무엇을 방어하나)
- **C-1** self-referential GT → Exp A(removal, 게임-무관 재학습 척도)로 **부분 방어** (own-game 대조 G-1 은 별개).
- **C-2** (b) target seed-안정성 미검증 → **Exp C 정면 대응, 재실행 0** (아래 결과 참조).
- **C-3** LLM downstream 실효성 → Exp A(오염 레짐 이득)로 **절반**.
- **C-4** baseline 공정성 → Exp A(게임-무관 공통 자) + Fed-LOO/ComFedSV 커버리지 보강.
- **C-5** 레짐-실무 괴리(plain SGD) → **Exp D**(AdamW-fidelity) 저비용 겨냥.

## 현재 상태 (2026-07-14)

### ✅ DONE — 코드 + 로컬 검증 (GPU 서버 불필요분)
- **코드 변경 6파일**(전부 하위호환·기본값=현행 동작):
  - `flirds/oracle/exact_sv.py` — `exact_shapley(…, return_u=True)` → `(phi, u)` 2^N 캐시 반환.
  - `flirds/fl/llm_server.py` — `client_optimizer(lr)` = `CLIENT_OPT`(sgd 기본|adamw) 스위치 (Exp D).
  - `flirds/data/corruptors.py` — `answer_swap_graded(recs, cid, rate)` (rate≥1.0 = 기존 answer_swap **비트동일**).
  - `flirds/data/llm.py` — `noisy_rate` 관통(build/build_alpaca_iid/build_crossdevice; `_noisy` 헬퍼).
  - `experiments/track_d.py` — Exp A1: (a) `u` 캐시 포획 + `removal_curves()` worst/best-first + persist; `client_opt` provenance.
  - `experiments/phase2_matrix.py` — Exp A2 `removal_retrain_curves`+`poison_removal_asr`, `build_trajectory(exclude=)`, `compute_methods`에 **Fed-LOO + ComFedSV(silo)** 추가, `NOISY_RATE`/`DOSE_MULT`/`REMOVAL`/`REMOVAL_METHODS` env, dose 토큰 네이밍·config provenance.
- **Exp C (target-stability) 실측 완료** — `runs/track_d/make_target_stability.py` (신규) + `make_fidelity.py`(xseed 열 추가). 출력:
  - `runs/track_d/target_stability.csv`, `runs/phase2_matrix/target_stability.csv` (gitignore = 파생·재생성).
- **로컬 검증**(Windows anaconda, 동일 스택 torch2.11/transformers4.57/trl1.2/peft0.19; `PYTHONUTF8=1` 필수):
  - 순수로직 단위테스트 20/20 PASS (corruptor 비트동일·경계, `return_u`, `removal_curves` 방향성).
  - gpt2(tiny, random-init) **GPU 통합 스모크 2종 PASS**: phase2_matrix(15-method set + removal 캐시 + poison exclude + poison_removal_asr), track_d Exp A1((methods,u_a) 반환 + curve).
- **Exp A3 (CNN removal-curve) 코드 + 로컬 검증 완료 (2026-07-16)** — 아래 §Exp A3 참조; 실행 대기.

### ⏳ TODO — DGX 서버 (서버 이전 완료 후)
- **파일럿**(`run_pilot.sh`, §7.1): noisy×silo5×seed0×축소방법셋. **목표 = silo5 단일 FL 재학습 실측시간**(`removal_retrain_s`) → 풀스윕 GPU예산 산정. → Yonghee 승인.
- **풀스윕**(`run_full_sweep.sh`, §7.2): 4 threat × 전방법 × 3 seed × 2 runner + dose 스윕.
- **Exp D**(AdamW) 마지막.
- **Exp A3 CNN removal 스윕**(`run_cnn_removal.sh`, mnist 9셀 — LLM 풀스윕과 독립, 유휴 GPU 지정): 아래 §Exp A3.

## Exp C 결과 (이미 산출됨 — C-2 정본화)
(b) oracle per-client φ 의 seed 간 자기-Spearman. **핵심: IID-clean 무대에서 매칭 대상 (b) 자신이 seed-불안정**
→ 그 위의 per-seed +1.000 fidelity 는 불안정한 GT 를 좇는 것. 비-IID(silo5)에선 (b)가 안정 → +1.000 이 의미 있음.
리뷰의 "저자가 잰 −0.37~−0.11(노트-only)"을 **재현·정본화**:

| 무대 | cell | mean xseed ρ |
|---|---|---|
| track_d IID-clean | 1B_anchor5 | **−0.367** (min −0.90) |
| track_d IID-clean | 1B_std20 | −0.114 |
| track_d IID-clean | 3B_std20 | −0.243 |
| track_d IID-clean | 7B_anchor5 | +0.733 (스케일↑ 안정↑) |
| phase2 IID | 1B_iid5_clean | +0.133 |
| phase2 비-IID | 1B_silo5_clean | **+0.867** |
| phase2 비-IID | 1B_silo5_noisy/poison | +0.933 / +1.000 |

→ CLAUDE.md 신호크기 진단("non-IID clean +0.87 vs IID clean +0.13")과 정합. `make_fidelity.py` 실행 시
fidelity 표 아래 target-안정성 열이 함께 출력됨(리뷰 §4/§5.1 프로토콜 격상).

## 실행 순서
1. `bash run_pilot.sh`  (GPU 0; `REPO=` 서버경로·`GPU=` 오버라이드) → `_logs/`·`rundirs/` 생성.
2. 파일럿 산출 확인 → **`removal_retrain_s` 보고** → Yonghee 승인.
3. `bash run_full_sweep.sh`  (`SEEDS="0 1 2"`, `DO="1 2 3 4"` 섹션 선택, GPU 샤딩은 섹션별 병렬 셸).
4. review-claude.md §5.4/G-1/G-4/G-7 대응 문단 초안(파일럿 실측치 반영).

## 풀스윕 셀 매트릭스 (GPU 예산 = 파일럿 후 확정)
| 섹션 | runner | 셀 | 셀당 비용 |
|---|---|---|---|
| [1] silo5 removal | phase2_matrix | 4 threat × 3 seed = **12** | traj + n_retrains×T_retrain (noisy/fr ≤2^5=32 distinct, 캐시공유; poison k=1 = 1~3 rebuild) |
| [2] dose noisy | phase2_matrix | 6 rate × 3 = **18** | traj + methods (removal 없음) |
| [2] dose frrand | phase2_matrix | 5 mult × 3 = **15** | traj(+warmup) + methods |
| [2] dose poison | phase2_matrix | 10 frac × 3 = **30** | poison install traj + methods (셀당 최고가) |
| [3] Exp A1 anchor5 | track_d | 3 seed = **3** | (a) oracle 2^5 재학습(1B ~시간) — 논문 anchor5 재실행에 removal 무료 편승 |
| [4] Exp D adamw | track_d | **1** | (a) oracle 포함 1셀 |

예산 공식: `T ≈ Σ_[1] (T_traj + n_retr·T_retrain) + Σ_[2] (T_traj + T_methods) + 3·T_(a)oracle + T_D`.
`T_retrain`·`T_traj`·`T_methods`·`n_retr(noisy)` = **파일럿에서 측정**(현재 미측정). `T_(a)oracle` ≈ 기존 anchor5 (a) 시간.

## Exp A3 — CNN removal-curve (코드 완료 2026-07-16; 실행 대기)

LLM removal(Exp A2 silo5, 2026-07-16 완료)의 CNN 확장 — 게임-무관 재학습 척도의 **크로스-스테이지
일반화**(C-1/C-4 방어 보강) + **accuracy 축**: LLM 무대는 생성형이라 val_loss 뿐이지만 CNN 은 재학습된
global 하나에서 **val_loss(게임 지표) + test acc(8000 disjoint; 배치 지표)** 를 동시 기록(Yonghee 명시
요청; 추가 비용 ≈0). 서술 위계: removal = 2차-①(일반 성능) 실효성 검증 — headline 은 1차 fidelity.

- **코드**: `codes/experiments/track_c1.py` 확장 — `C1_REMOVAL=1` env 게이트(기본 0 = **비트동일**, 아래
  검증), `removal_retrain_curves()` = phase2_matrix A2 패턴 이식. **옵션 1**(독립 재학습 경로) 채택:
  (a) u-캐시 유도(옵션 2, track_d A1 패턴)는 acc 가 u 에 없고 `C1_ORACLE_A=1` 셀 한정이라 기각 — A2 와
  균일한 단일 경로. worst/best-first 실제 재학습, frozenset 캐시 방법·방향 공유. **Ripple 기본 제외**
  (자기-궤적 방법 — 공유 frozen 궤적의 순위와 비가환; `C1_REMOVAL_METHODS` 로 강제 가능), 대상 = C1 val
  전 방법 + (b)oracle + Fed-LOO (11종).
- **스키마**(LLM 집계 도구 호환): `removal_curve` = A2 동일 구조·키(`{method:{worst_first:[[k,val_loss],…],
  best_first:[…]}}`) + **신규 병렬 키 `removal_curve_acc`**(`[[k,test_acc],…]`) + `removal_retrain_s`
  + `removal_orient`/`removal_acc_orient`. config.yaml 에 `removal` provenance 키.
- **셀 매트릭스**(실행 세션 최종 확정): core = `mnist × {label_flip, feature_noise, iid} × seed{0,1,2}`
  = **9셀** — ladder 2종 = 오염 클라 제거→acc↑ 기대, iid = 통제군(제거 = 실데이터 손실 → 중립~해로움
  기대). optional = cifar10 동일 9셀(`DATASETS="mnist cifar10"`).
- **비용 추정**(canonical c1 실측 B200): 재학습 1회 ≈ traj ≈ 1.5분(mnist 92s / cifar 78s), methods
  (Ripple 제외) 2–8분. distinct 재학습 수 = 캐시 공유로 순위합의도에 좌우 — 스모크(n=6, 11방법) 실측
  21회, full n=10 예상 ~25–80회(상한 11방법×2방향×10=220) → **셀당 ~0.7–2.5h, mnist 9셀 5-GPU ~1.5–5h**.
- **실행(1줄)**: `bash runs/removal_dose/run_cnn_removal.sh` (`GPUS="…"` 유휴 GPU 지정, `DRYRUN=1` = 큐만
  생성·출력, done-마커 재개; 결과 = `rundirs_cnn/` — canonical `runs/track_c/c1` **불변**). 큐 셀 =
  `C1_ORACLE_A=0`(removal 은 (a) 2^N 불필요) + `C1_RIPPLE=0`(removal 미사용 + fidelity 는 canonical c1
  에 기영속 — 중복 재계산 회피).
- **로컬 검증(2026-07-16, CPU)**: ① 순수로직 단위테스트 **5/5 PASS**(`codes/tests/test_removal_cnn.py`
  — 방향성 worst≤best·캐시 1회/kept-set(2n−1)·Ripple 기본제외·A2 스키마·게이트 기본 off); ②
  **`C1_REMOVAL=0` 비트동일** — HEAD 스모크 vs 신규 코드 스모크(mnist·label_flip·seed0),
  metrics.json(wall-clock 필드 `traj_time`/`runtime` 제외 정규화) sha256 `8d90922f…` **일치** +
  phi.parquet 값 완전일치; ③ `C1_MODE=smoke C1_REMOVAL=1` 통합 스모크 **green**(21 distinct retrains,
  스키마 검증 통과). 스모크 수치 자체는 무학습 수준(코드패스 검증용).
- **Caveats**: 재학습 = **clean FedAvg**(A2 와 동일); iid 통제군 곡선 차이 = 순수 데이터양 손실 효과로
  읽을 것; ComFedSV full-participation 축퇴 caveat 은 A2 와 동일 적용.
- **옵션(미구현; Yonghee 결정 대기, 구현 시 별도 env 게이트)**: CNN 픽셀-트리거 backdoor +
  poison_removal ASR(LLM poison 사각지대의 CNN 재현 — clean-preserving 설계 필요),
  label/quantity_skew removal(이질성 무대 '낮은 φ ≠ 제거해도 됨' 대조).

## 산출물 / 스키마
- rundir(RunLogger §6): `config.yaml`(+`noisy_rate`/`dose_mult`/`removal`/`client_opt`) · `meta.json`(git/env) · `phi.parquet` · `metrics.json`.
- `metrics.json` 신규 키:
  - `<threat>_seed<s>.removal_curve` = `{method: {worst_first:[[k,val_loss],…], best_first:[…]}}`, `removal_orient="val_loss (lower=better); phi good->low"`, `removal_retrain_s`(평균 재학습 초).
  - poison: `<threat>_seed<s>.poison_removal` = `{baseline_asr, by_method:{m:{excluded, asr}}}`.
  - track_d: `seed<s>.removal_curve`(util=−val_loss, higher=better) + `removal_orient`.
- **정본화 규율**: 어떤 수치도 .log/노트에만 남기지 말 것(리뷰 R1/R2/R9 재발 금지). 전부 rundir/CSV.

## Caveats (결과 캡션에 병기)
- **ComFedSV @ silo5**: full participation → low-rank/partial 근사가 자명 성립 = **축퇴 레짐**(리뷰 C-4). 승리로 읽지 말 것.
- **removal 재학습 = clean 재학습**(fabricated update 미재현): free-rider 는 데이터 clean → 제거해도 ~중립(정직한 nuance); noisy 는 데이터 오염 → 제거 시 이득.
- **AdamW = "브리지 설정"**(현 constant lr 유지). 논문 AdamW 5e-5 cosine 과의 갭은 deviation caveat.
- **freerider_zero**: dose 제외(Δw=0 극값), removal 만 포함(locked).

## 환경
- python `/home/korea_bupj/miniconda3/envs/flirds/bin/python`, codes/ 에서 `PYTHONPATH=.`, GPU 0-3(0 free).
- 서버 이전 중 → 스크립트 상단 `REPO=` 경로 이전 후 갱신 필요.
- 로컬(코드/분석 검증): Windows anaconda + `PYTHONUTF8=1`(cp949 locale 회피).
