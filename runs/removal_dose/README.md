# removal_dose — C-1~C-5 완화 실험 (removal-curve · dose-response · target-stability · AdamW)

계획서: `PROMPT_removal_dose_c2_adamw.md` (루트). Yonghee 승인 결정(2026-07-14) 반영.
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

### ⏳ TODO — DGX 서버 (서버 이전 완료 후)
- **파일럿**(`run_pilot.sh`, §7.1): noisy×silo5×seed0×축소방법셋. **목표 = silo5 단일 FL 재학습 실측시간**(`removal_retrain_s`) → 풀스윕 GPU예산 산정. → Yonghee 승인.
- **풀스윕**(`run_full_sweep.sh`, §7.2): 4 threat × 전방법 × 3 seed × 2 runner + dose 스윕.
- **Exp D**(AdamW) 마지막.

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
