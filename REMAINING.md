# REMAINING — 남은 작업 (상시 현행; 완료·폐기 항목은 지우고 git 히스토리로만 남김)

> 갱신 2026-07-23(2차 정리 — T1 완료·CNN 그리드 144 완주·L1 가동·논문 제외 결정 반영).
> 파일-canon: rundir → overview → paper. **논문 작업 분할 정본 = `paper/workplan/00-INDEX.md`**(T1 ☑ / T2–T5 ☐).
> 실행 큐 정본 = §1.6 + `runs/track_h/QUEUE_L1L2_2026-07-23.md`(서버 큐 기록 = `runs/track_h/QUEUE_L1_2026-07-23.txt`).

## 1. 실험 (GPU) — 순서대로

- **환경(컨테이너 공통; 2026-07-20 재구축)**:
  `BATCH=/NHNHOME/26msit001_A/BASE/edge_ai_lab/yonghee/flirds_batch` 기준
  `PY=$BATCH/venv/bin/python`, `HOME=$BATCH/home`,
  `HF_HOME=$BATCH/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`, `codes/`에서 `PYTHONPATH=.`.
  현 컨테이너=B200 4장(0–3). venv는 기존 rundir meta.json과 **동일 버전 고정**(torch 2.12.0+cu130,
  transformers 5.9.0, trl 1.5.1, peft 0.19.1, accelerate 1.13.0, datasets 4.8.5, numpy 2.4.6).
  meta-llama gated 재취득 불가(유효 토큰 무) → 해시 교차검증된 공개 미러로 캐시 재구성 —
  검증 체인·근거는 `$BATCH/PROVENANCE.md`.

### 1.0 드라이버·큐 운용 교훈 (차기 캠페인 공통; QUEUE 문서들이 참조)

- **큐 정지는 줄 삭제가 아니라 주석 처리.** 드라이버는 매 루프 큐를 다시 읽고 `consumed`
  **인덱스**로 위치를 추적하므로, 줄을 지우면 인덱스가 밀려 오배치가 난다. 줄 수를 보존해야 안전.
- **러너는 셀 단위 원자적** — 중도 kill = 전손. 정지는 드레인(실행 중 셀 완주 후 드라이버 종료).
- **마지막 셀은 `done[ok]` 줄이 안 남는다**(루프가 리핑 전에 종료 조건에서 빠져나감) → 완주 판정을
  `grep 'done\[ok\]'` 단독으로 하지 말 것 — 셀 로그의 `TRACK G DONE`/`MATRIX DONE` + rundir mtime 교차 확인.
- **내부 직렬화가 긴 arm(t2 다점수원 등)은 큐 레벨로 분할 노출**(셀 하나 17h 점유 방지) — L1 큐에 반영 완료.

### 1.0b 진행 중 — B200 L1·L2 캠페인 (2026-07-23 기동, 마감 07-25 03:27)

L1 R4 Tier C **P1-only**(Yonghee 07-23: P5s 전면 중단) 가동 중. 실행 큐 정본 =
`$BATCH/runlogs/queue_L1.txt`(리포 기록 = `runs/track_h/QUEUE_L1_2026-07-23.txt`).
- **순서**: seed0 패치(진행) → seed1 → **L2((b)-fidelity 2셀, seed2보다 앞으로 이동** =
  fidelity 1차·마감 방어) → seed2. 드라이버 = `run_multi_driver.sh`(pid 20798, 단일).
- **사전절차 완료**: pre-fix Tier A(git_sha==`fa5fc6e`) 20 rundir → `runs/track_h/rundirs_llm_prefixh1/`
  아카이브·커밋 — make_analysis의 pre-fix 오채택 위험 해소.
- **seed0 패치 = 옛코드 포크 이슈**: 패치 셀이 §1.7 가드 커밋 **직전** 디스패치돼 옛 레거시
  가드로 돈다. `noisy_obs_t2`가 `t2_sign_{flirds1st,lossheur,fedif}`·`t2_random_k37`을
  persist 시 해시로 갈라짐(숫자는 가드-후와 동일 — FL/스코어링/T2 경로 불변). **정리 =
  `runs/track_h/consolidate_hash_dirs.py --apply`**(canonical별 최신본 유지·해시본 제거).
  워처가 `noisy_obs_t2`(pid 20802) 완주 감지 후 자동 실행(`$BATCH/runlogs/consolidate_watch.log`).
- **완료 후**: rundir 커밋 + `make_analysis.py` 재생성 + H-12/H-13 대조 → paper I1·F2·D1 ⬚ 채움.

### 1.2 β0.3 재실행 잔여 **10셀** (device100 7 + 3B silo5 3) — L-캠페인 종료 후 재개

**잔여 10셀 = `$BATCH/runlogs/queue_postswap.txt`의 `#PAUSED-0723` 줄**(접두어 제거로 재개,
~36 GPU-h ≈ 4-GPU 9h). poison 3셀은 **영구 제외**(Yonghee 07-23, 논문 축 제외).
§1.7 정체성 가드로 **재개 시 셀당 `RUNDIR_REPLACE=1` 필요**(의도된 동작 — β0.5 원본 명시 교체).
드라이버 유실 시 수동 재개:
```bash
sed -i 's/^#phase2/phase2/' runs/rerun_beta03/logs/resume36h.txt   # 유실 시 RESUME_AFTER_MIGRATION.md 31줄에서
                                                                   # 완료분 1B_silo5 4셀 제외하고 재생성
PY=$PY PP=<repo>/codes HOME=… HF_HOME=… HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  QUEUE=<abs>/runs/rerun_beta03/logs/resume36h.txt GPUS_FILE=<abs>/logs/gpus36h.txt GPUS="<g...>" \
  LOGDIR=<abs>/runs/rerun_beta03/logs bash runs/rerun_beta03/run_multi_driver.sh
```
완료 후: rundir 커밋 + overview §3.4 phase2 ShapleyFL 행 갱신. (3B silo5 3셀은 논문-제외 축이나
overview β-canon 통일용 — 재개 시 범위 Yonghee 재확인.)

### 1.3 β0.3 deferred 9셀 — **무기한 보류** (07-23 논문 제외 여파)

7B_std20×3(70–90h) + device100-a0.5 anchor×3(63h) + 7B_anchor5×3(35–45h) —
**3B/7B·std20 논문 제외로 실행 근거 상실**; "3B/7B 여유 시 재추가" 결정 시에만 부활
(목록 = `runs/rerun_beta03/RESUME_AFTER_MIGRATION.md`).

### 1.4 장기·조건부 (실행 조건 = 해당 결과의 수록 확정)

- lr·steps intervention 2차검증(무GPU 재분석; 데이터는 기존 track_d arm rundir에 있음) ·
  1B·CNN β-불변 canon 확인(Yonghee 확인 필요) · **P0(H1) 소급 재실행 스코프**(논문 인용 셀 한정;
  그룹 카탈로그 = git 히스토리의 `RERUN_AFTER_REPRO_FIX_2026-07-21.md`).
- 3B silo5 robustness seeds 1·2 — 3B 논문 제외로 보류(overview §3.3.3 1-seed caveat 해소용으로만).
- probe A축 seeds 1·2(r32/64·st20/30) — 신호실재성(B축) 절 삭제로 논문 목표 상실; ablation 보강 필요 시만.

### 1.4b CNN Slurm 잔여 — c2fid 본런 GO 대기 (07-22 캠페인은 완주)

07-22 캠페인 완주 기록: skew/fmnist 90 + frrand 백필 + strmain 18 = track_g 그리드
**144/144 완주·분석 재생성 커밋(`570a93f`)** · restack 36런 커밋(`373a1d8` — drift 행 산출,
† 확정 판정은 T4) · track_h strmain 24런 착지(P1 7점수원+obs ×3s) · `make_analysis` strmain
인식 확장 완료 · 오염집합 규약(FedCorr (ρ,τ) 병기) = paper B.2 반영(`a865738`).

- **잔여 ① c2fid 본런 143셀**: 파일럿 `cifar10_dir1_grad-noise_fid_seed0` 완주·커밋 —
  **실측 1.05 GPU-h/셀**(3,777s = 궤적 재생 284 + (b) 2¹⁰×120 오라클 824 + 8방법 2,669;
  peak 3.3 GiB) → 본런 **≈150 GPU-h**(RTX3090 기준; fmnist 셀은 더 낮음, 8-GPU wall ~19h).
  **Yonghee GO 후** `sbatch runs/track_c/c2fid/sbatch_fid.sh` → `make_analysis` →
  F-1~F-4 사전등록 대조(MISS 포함 보고) → paper F1·c2fid AUROC ⬚ 채움.
- 잔여 ② overview 결과 블록 반영: §3.2.4 skew·fmnist·strmain + restack drift 표(570a93f 분석 기준).

### 1.6 LLM L1–L8 캠페인 (Yonghee 07-23: β0.3 잔여보다 선행, β는 그 후 재개)

**실행 절차·명령 정본 = `runs/track_h/QUEUE_L1L2_2026-07-23.md`**. 사전등록 =
`runs/track_h/README.md` H-12·H-13(+H-14는 T3에서 선커밋).

| # | 셀 | 비용(GPU-h) | 상태 |
|---|---|---|---|
| **L1** | R4 Tier C: {noisy nr0.7, frzero} × **seeds 0·1·2** 전체 재실행(pre-fix seed0 비-canonical; T2 4점수원 = 소스별 프로세스 분할) | ~120–138 | **실행 중**(§1.0b) |
| **L2** | R4 (b)-fidelity: `phase2_matrix.py REGIME=gsm50k5`(nr 0.7·(b) per-round 2⁵·9방법·탐지기 4종·timing.json) — noisy→clean 각 seed0; 산출은 c2fid 열-호환 스키마 롤업 | ~10–15/셀 | 큐 등재(§1.0b — seed1 뒤·seed2 앞) |
| **L4** | R4 Tier B **T2-only**(renorm 4점수원 × noisy·frzero) | ~150 | **Yonghee 승인 게이트**(§3) — L1·L2 순항 후 |
| L5 | 비등n silo5 1셀(4:2:1:1:1, clean+noisy, 3-seed) — CNN qskew fidelity와 P5c 쌍 | ~10–15 | 여유 시 |
| L6 | silo5 graded-noisy(nr~U(0.5,1), `answer_swap_graded`) — spearman_vs_rate LLM 대응 | ~6–12 | 여유 시(L4 우선) |
| **L7** | **R4 P1w**(w∝max(cum,0)·합-1 재정규화; flirds-only) × {clean,noisy,frzero} × 3-seed × {T1,T2} = 18런 — 스펙·H-14 = `paper/workplan/T3-p1w-llm-impl.md` | ~80 | **확정** — 순서 L2 뒤·**L4 앞** |
| **L8** | **retrain-(a) 스위트**: gsm5 신설(dual (a)+(b), clean·noisy×3-seed) + silo5 (a)-leg 3셀 — 스펙 = `paper/workplan/T5-retrain-a-suite.md` | gsm5 ~60 + silo5 ~26 | **확정** — **RTX3090×8 몫**(B200 비점유) |

- **하지 않는 것**: gnoise 재개·등방 노이즈류 LLM 재시도(종결 근거 = `runs/track_h/gnoise_diag/README.md`) ·
  LIE/sign-flip(위협 스코프 게이트) · (a) retrain 3B/7B · R4 frrand(frzero와 실질 동일) ·
  P0 전면 소급(§1.4) · **Fed-LOO**(논문 전면 제외; L2 러너 산출은 무해·미게재) ·
  **poison**(영구 제외) · **P5h/P5s 비교**(전면 제외 — 완주 6런·`runs/track_h/p5/` rundir는 보존) ·
  std20/anchor5-vs(b) 스위트 재실행(논문 삭제) · E5 N=10 확장(seeds1·2·(a) 2¹⁰ — 미진행 확정 07-22).
- 예산: 필수(L1+L2+L7) 220–248 → +L4 370–398 / B200×4 5일 ≈480 GPU-h(명목); L8은 3090 별도 풀.
  vast.ai는 B200 초과 시 **비-timing seed 복제만** 예외(timing/canonical 셀 이관 금지).

### 1.7 rundir 정체성 — 잔여 배선

처방 1+2 구현 완료(07-23, Yonghee 승인): 정체성 allow-list(`check_identity`/`precheck` —
같으면 덮어쓰기·다르면 `RunDirIdentityError`·정체성 키 부재=불일치; 우회 `RUNDIR_REPLACE=1`) +
β 단일화(`shapleyfl.BETA = env SFL_BETA, 기본 0.3` — 하드코딩 7곳 정리). 배선 완료 =
`track_g`·`phase2_matrix`, 테스트 `tests/test_rundir_identity.py` 6개. 상세 진단 = git 히스토리.
- **잔여**: ① `track_c1`·`track_c2`·`track_c2_fid`·`track_d`·`phase1_*`는 아직 `identity=None`
  (레거시 통짜 비교) — 이들 config에 `sfl_beta`를 추가하려면 **identity 배선을 함께** 해야 한다
  (레거시 가드에선 키 추가 자체가 포크 유발). ② 처방 3(`superseded.json`) 미착수.
- β0.3 재개 시 셀당 `RUNDIR_REPLACE=1` 필요(§1.2).

## 2. 문서·분석 (무GPU)

1. **paper ⬚·† 해소 원장**(T1 산출 `paper/paper-ko.md` §5·부록 B–E):
   ⬚ F1←c2fid 본런(§1.4b) · F2/D1←L2 · I1←L1 · F3+silo5(a)←L8 · P1w←L7+W-B ·
   주무대 비용 실측←L1/L2 timing+c2fid runtime · F-4←c2fid /
   **† = CNN dir1 수치 전량 ← W-A drift 판정(T4; drift 행 소스는 착지 완료 `570a93f`·`373a1d8`)**.
2. **T2–T5 워크플랜**(상태 정본 = `paper/workplan/00-INDEX.md`): T2 결과 overview 페이지+figures —
   **진행 흔적 있음**(`research-wiki/survey/flirds-paper-results-overview-figs/` figs 6종 +
   make_figures.py, 미커밋 — 해당 세션 몫) · T3 P1w LLM 구현(H-14 선커밋) · T4 CNN 릴레이(W-A 판정+W-B twin) ·
   T5 retrain-(a) 스위트.
3. overview 반영 잔여: **device 학습시간**(원격 계측/재구성 — cost-accounting 잔여) +
   §1.4b-② CNN 결과 블록. (E4·E5·frdelta §3.3.4·loss-heur 99s·AUROC 0.604 정본화는 반영 확인됨.)
4. **STD-DAGMM 인용 캐비엇**: 단일 seed 값은 재현 안 됨(같은 seed 재실행 AUROC 평균 |Δ|=0.177;
   코드 무변 — AE+GMM 비볼록 + 순위 여유 0). 3-seed 평균은 안정(noisy 0.652→0.600,
   frrand 0.925→0.892) → 논문 D1 기입 시 **3-seed mean±std + 단일-seed 재현 불가 caveat 필수**.
5. 영문 tex(`paper/sections/*.tex`) 반영 = paper-ko §5 확정 후 일괄 — tab:cost 갱신
   (loss-heur ~99s·device overhead%·end-to-end/overhead% 2블록) 포함.
6. **§6(한계) 실작성 재료**: T1 스텁 주석 7건 + gnoise negative(`runs/track_h/gnoise_diag/README.md`
   그대로 인용 가능 — **Krum σ=200·arXiv 3편 = 인용 금지 목록 동봉**) + frdelta "기여도≠탐지"
   1문장(overview §3.3.4; (b)oracle 동일 실패 = 게임 공통).

## 3. Yonghee 결정 대기

- **push**: 로컬 커밋 다수(워크플랜 `b66b926`·T1 `25f3093`·B.2 `a865738`·CNN 144 `570a93f` 등) — Yonghee 직접.
- **c2fid 본런 GO**(§1.4b): 파일럿 실측 **1.05 GPU-h/셀** → 143셀 ≈ **150 GPU-h**(8-GPU wall ~19h).
- **R4 Tier B(L4) 게이트**(§1.6): L1·L2 순항 확인 후(~150 GPU-h).
