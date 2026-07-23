# REMAINING — 남은 실험 인수인계 (상시 현행; 완료·폐기 항목은 지우고 git 히스토리로만 남김)

> 갱신 2026-07-23(3차 — 스코프 원복: **이 파일 = 덜 돌린 실험을 다른 세션에 전달하는 용도 전용**).
> 논문·문서 작업의 정본 = `paper/workplan/00-INDEX.md`(⬚·† 해소 원장 = T1 행; push는 Yonghee 직접).
> 파일-canon: rundir → overview → paper. 실행 큐 정본 = §1.6 + `runs/track_h/QUEUE_L1L2_2026-07-23.md`.

## 1. 실험 (GPU)

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

### 1.4 논문-인용 셀 provenance 감사 2건 (무GPU~저비용; 장기)

- **P0(H1) 소급 재실행 스코프**: 논문 인용 셀 한정으로 판단(그룹 카탈로그 = git 히스토리의
  `RERUN_AFTER_REPRO_FIX_2026-07-21.md`).
- **ShapleyFL β-출처 확인**: 부록 B.5가 β=0.3을 명기하는데, 논문이 인용하는 기존 셀
  (CNN C1·anchor5)의 값이 β0.3 canon인지 미확보("값 동일/β-불변" 주장 검증 안 됨) —
  rundir meta git_sha로 확인, 어긋나면 해당 셀만 재계산/재실행 판단.

### 1.4b CNN Slurm 잔여 — c2fid 본런 GO 대기 (07-22 캠페인은 완주 — `570a93f`·`373a1d8`)

- **잔여 ① c2fid 본런 143셀**: 파일럿 `cifar10_dir1_grad-noise_fid_seed0` 완주·커밋 —
  **실측 1.05 GPU-h/셀**(3,777s = 궤적 재생 284 + (b) 2¹⁰×120 오라클 824 + 8방법 2,669;
  peak 3.3 GiB) → 본런 **≈150 GPU-h**(RTX3090 기준; fmnist 셀은 더 낮음, 8-GPU wall ~19h).
  **Yonghee GO 후** `sbatch runs/track_c/c2fid/sbatch_fid.sh` → `make_analysis` →
  F-1~F-4 사전등록 대조(MISS 포함 보고) → paper F1·c2fid AUROC ⬚ 채움.
- 잔여 ② 완주분 overview 결과 블록 반영: §3.2.4 skew·fmnist·strmain + restack drift 표
  (570a93f 분석 기준; W-A drift **판정**은 T4 세션 몫).

### 1.6 LLM L1–L8 캠페인 (Yonghee 07-23: 최우선)

**실행 절차·명령 정본 = `runs/track_h/QUEUE_L1L2_2026-07-23.md`**. 사전등록 =
`runs/track_h/README.md` H-12·H-13(+H-14는 T3에서 선커밋).

| # | 셀 | 비용(GPU-h) | 상태 |
|---|---|---|---|
| **L1** | R4 Tier C: {noisy nr0.7, frzero} × **seeds 0·1·2** 전체 재실행(pre-fix seed0 비-canonical; T2 4점수원 = 소스별 프로세스 분할) | ~120–138 | **실행 중**(§1.0b) |
| **L2** | R4 (b)-fidelity: `phase2_matrix.py REGIME=gsm50k5`(nr 0.7·(b) per-round 2⁵·9방법·탐지기 4종·timing.json) — noisy→clean 각 seed0; 산출은 c2fid 열-호환 스키마 롤업 | ~10–15/셀 | 큐 등재(§1.0b — seed1 뒤·seed2 앞) |
| **L4** | R4 Tier B **T2-only**(renorm 4점수원 × noisy·frzero) | ~150 | **Yonghee 승인 게이트** — L1·L2 순항 후 |
| L5 | 비등n silo5 1셀(4:2:1:1:1, clean+noisy, 3-seed) — CNN qskew fidelity와 P5c 쌍 | ~10–15 | 여유 시 |
| L6 | silo5 graded-noisy(nr~U(0.5,1), `answer_swap_graded`) — spearman_vs_rate LLM 대응 | ~6–12 | 여유 시(L4 우선) |
| **L7** | **R4 P1w**(w∝max(cum,0)·합-1 재정규화; flirds-only) × {clean,noisy,frzero} × 3-seed × {T1,T2} = 18런 — 스펙·H-14 = `paper/workplan/T3-p1w-llm-impl.md` | ~80 | **확정** — 순서 L2 뒤·**L4 앞** |
| **L8** | **retrain-(a) 스위트**: gsm5 신설(dual (a)+(b), clean·noisy×3-seed) + silo5 (a)-leg 3셀 — 스펙 = `paper/workplan/T5-retrain-a-suite.md` | gsm5 ~60 + silo5 ~26 | **확정** — **RTX3090×8 몫**(B200 비점유) |

- **하지 않는 것(재제안 금지)**: gnoise류 LLM 재시도(종결 = `runs/track_h/gnoise_diag/README.md`) ·
  LIE/sign-flip · R4 frrand · (a) 3B/7B · P0 전면 소급 · Fed-LOO · poison · P5h/P5s(rundir는 보존) ·
  std20/anchor5-vs(b) 재실행 · E5 N=10 확장 · **β0.3 잔여 재실행(10셀+deferred 9셀 — 대상 표 전부
  논문 제외로 폐기 07-23; 부활 시 목록 = `runs/rerun_beta03/RESUME_AFTER_MIGRATION.md`)**.
- 예산: 필수(L1+L2+L7) 220–248 → +L4 370–398 / B200×4 5일 ≈480 GPU-h(명목); L8은 3090 별도 풀.
  vast.ai는 B200 초과 시 **비-timing seed 복제만** 예외(timing/canonical 셀 이관 금지).

### 1.7 rundir 정체성 — 잔여 배선

처방 1+2 구현 완료(07-23): 정체성 allow-list(`check_identity`/`precheck`; 우회 `RUNDIR_REPLACE=1`) +
β 단일화(`shapleyfl.BETA = env SFL_BETA, 기본 0.3`). 배선 완료 = `track_g`·`phase2_matrix`,
테스트 6개. 상세 진단 = git 히스토리.
- **잔여**: ① `track_c1`·`track_c2`·`track_c2_fid`·`track_d`·`phase1_*`는 아직 `identity=None`
  (레거시 통짜 비교) — 이들 config에 `sfl_beta`를 추가하려면 identity 배선을 함께 해야 함.
  ② 처방 3(`superseded.json`) 미착수.
