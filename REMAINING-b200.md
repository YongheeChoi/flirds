# REMAINING (B200 풀) — LLM 주무대 R4 (gsm50k5, 1B)

> 실행처별 인수인계 중 **B200 컨테이너** 몫. 짝 파일 = `REMAINING-slurm.md`(CNN + 작은-N LLM).
> **현재: R4 L1·L2 진행 중(§1).** 마감 07-25 03:27. push는 Yonghee 직접. 수치 = rundir/analysis 재생성 값만.
> 논문·문서 정본 = `paper/workplan/00-INDEX.md`. 실행 절차·명령 정본 = `runs/track_h/QUEUE_L1L2_2026-07-23.md`.
> 사전등록 = `runs/track_h/README.md` H-12·H-13(+H-14는 T3에서 선커밋).

## 0. 환경 (B200 컨테이너; 2026-07-20 재구축)

`BATCH=/NHNHOME/26msit001_A/BASE/edge_ai_lab/yonghee/flirds_batch` 기준
`PY=$BATCH/venv/bin/python`, `HOME=$BATCH/home`,
`HF_HOME=$BATCH/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`, `codes/`에서 `PYTHONPATH=.`.
현 컨테이너=B200 4장(0–3). venv는 기존 rundir meta.json과 **동일 버전 고정**(torch 2.12.0+cu130,
transformers 5.9.0, trl 1.5.1, peft 0.19.1, accelerate 1.13.0, datasets 4.8.5, numpy 2.4.6).
meta-llama gated 재취득 불가(유효 토큰 무) → 해시 교차검증된 공개 미러로 캐시 재구성 —
검증 체인·근거는 `$BATCH/PROVENANCE.md`.

### 드라이버·큐 운용 교훈 (B200 드라이버 특유)

- **큐 정지는 줄 삭제가 아니라 주석 처리.** 드라이버는 매 루프 큐를 다시 읽고 `consumed`
  **인덱스**로 위치를 추적하므로, 줄을 지우면 인덱스가 밀려 오배치가 난다. 줄 수를 보존해야 안전.
- **러너는 셀 단위 원자적** — 중도 kill = 전손. 정지는 드레인(실행 중 셀 완주 후 드라이버 종료).
- **마지막 셀은 `done[ok]` 줄이 안 남는다**(루프가 리핑 전에 종료 조건에서 빠져나감) → 완주 판정을
  `grep 'done\[ok\]'` 단독으로 하지 말 것 — 셀 로그의 `MATRIX DONE`/`TRACK G DONE` + rundir mtime 교차 확인.
- **내부 직렬화가 긴 arm(t2 다점수원 등)은 큐 레벨로 분할 노출**(셀 하나 17h 점유 방지) — L1 큐에 반영 완료.

## 1. 진행 중 — L1·L2 캠페인 (2026-07-23 기동)

L1 R4 Tier C **P1-only**(Yonghee 07-23: P5s 전면 중단) 가동 중. 실행 큐 정본 =
`$BATCH/runlogs/queue_L1.txt`(리포 기록 = `runs/track_h/QUEUE_L1_2026-07-23.txt`).
- **순서**: seed0 패치(진행) → seed1 → **L2((b)-fidelity 2셀, seed2보다 앞으로 이동** =
  fidelity 1차·마감 방어) → seed2. 드라이버 = `run_multi_driver.sh`(pid 20798, 단일).
- **사전절차 완료**: pre-fix Tier A(git_sha==`fa5fc6e`) 20 rundir → `runs/track_h/rundirs_llm_prefixh1/`
  아카이브·커밋 — make_analysis의 pre-fix 오채택 위험 해소.
- **seed0 패치 = 옛코드 포크 이슈**: 패치 셀이 rundir 정체성 가드 커밋 **직전** 디스패치돼 옛 레거시
  가드로 돈다. `noisy_obs_t2`가 `t2_sign_{flirds1st,lossheur,fedif}`·`t2_random_k37`을
  persist 시 해시로 갈라짐(숫자는 가드-후와 동일 — FL/스코어링/T2 경로 불변). **정리 =
  `runs/track_h/consolidate_hash_dirs.py --apply`**(canonical별 최신본 유지·해시본 제거).
  워처가 `noisy_obs_t2`(pid 20802) 완주 감지 후 자동 실행(`$BATCH/runlogs/consolidate_watch.log`).
- **완료 후**: rundir 커밋 + `make_analysis.py` 재생성 + H-12/H-13 대조 → paper I1·F2·D1 ⬚ 채움.

| L2 셀 | 명령 |
|---|---|
| noisy (nr 기본 0.7) → `1B_gsm50k5_noisy_nr0.7_s0` | `REGIME=gsm50k5 THREAT=noisy SEED=0 CUDA_VISIBLE_DEVICES=<g> PYTHONPATH=. $PY -u experiments/phase2_matrix.py` |
| clean (포화 대조) → `1B_gsm50k5_clean_s0` | `REGIME=gsm50k5 THREAT=clean SEED=0 CUDA_VISIBLE_DEVICES=<g> PYTHONPATH=. $PY -u experiments/phase2_matrix.py` |

L2 = `phase2_matrix.py REGIME=gsm50k5`(nr 0.7·(b) per-round 2⁵·9방법[Fed-LOO 제외]·탐지기 4종·timing.json);
산출은 c2fid 열-호환 스키마 롤업(CNN fidelity leg와 공용 표). frzero 셀은 예산 여유 시만((b)-frzero exact-0 자명).

## 2. 대기 큐 — L4·L5·L6·L7

| # | 셀 | 비용(GPU-h) | 상태 |
|---|---|---|---|
| **L4** | R4 Tier B **T2-only**(renorm 4점수원 × noisy·frzero) | ~150 | **Yonghee 승인 게이트** — L1·L2 순항 후 |
| L5 | 비등n silo5 1셀(4:2:1:1:1, clean+noisy, 3-seed) — CNN qskew fidelity와 P5c 쌍 | ~10–15 | 여유 시 |
| L6 | silo5 graded-noisy(nr~U(0.5,1), `answer_swap_graded`) — spearman_vs_rate LLM 대응 | ~6–12 | 여유 시(L4 우선) |
| **L7** | **R4 P1w**(w∝max(cum,0)·합-1 재정규화; flirds-only) × {clean,noisy,frzero} × 3-seed × {T1,T2} = 18런 — 스펙·H-14 = `paper/workplan/T3-p1w-llm-impl.md` | ~80 | **확정** — 순서 L2 뒤·**L4 앞** |

- **하지 않는 것(재제안 금지)**: gnoise류 LLM 재시도(종결 = `runs/track_h/gnoise_diag/README.md`) ·
  LIE/sign-flip · (a) 3B/7B · P0 전면 소급 · **Fed-LOO**(양 무대 공통 제외; 러너 산출은 무해·미게재) ·
  **poison**(양 무대 공통, 영구 제외) · P5h/P5s(rundir는 보존) · std20/anchor5-vs(b) 재실행 ·
  E5 N=10 확장 · **β0.3 잔여 재실행(device100·3B·7B — 대상 표 전부 논문 제외로 폐기 07-23;
  부활 시 목록 = `runs/rerun_beta03/RESUME_AFTER_MIGRATION.md`)**.
  (※ R4 frrand는 07-23 번복 → §3 L9로 부활.)
- 예산: 필수(L1+L2+L7) 220–248 → +L4 370–398 / B200×4 5일 ≈480 GPU-h(명목).
  vast.ai는 B200 초과 시 **비-timing seed 복제만** 예외(timing/canonical 셀 이관 금지).

## 3. 주무대 위협-대칭 확장 — R4 frrand + strmain-dose (2026-07-23 Yonghee 신규)

> **종전 "R4 frrand 재제안 금지"를 번복.** 주무대 쌍 모두 free-rider-random 축이 불완전:
> R4(gsm50k5)는 **frrand 전무**(frzero만). 값싸고 모달리티 제약 없음 → frrand를 full-method 축으로
> 완성 + strmain-dose 추가 = 대칭 복원 + fidelity 변별(순위 축퇴 완화)이 목적. (CNN 대응 = `REMAINING-slurm.md` C-fr.)

| # | 셀 | 내용 | 비용(GPU-h, 추정) | 상태 |
|---|---|---|---|---|
| **L9** | R4 frrand | free-rider-random(zero 대신 무작위 업데이트) × {T1,T2} × 3-seed — L1 frzero-leg에 위협 훅만 교체 | ~50–70(L1 frzero-leg 동형) | **신규**(Yonghee: 우선순위·seed 수 판단 대기) |
| **L10** | R4 strmain류 per-client noisy-dose | answer-swap rate $\sim U(0.5,1)$ 클라별 변조(=FedCorr `strmain` 기본 draw) × 3-seed — **지표 = spearman_vs_rate(φ↔per-client dose) + vs (b) fidelity** | ~30–45(L2형 per-seed ~10–15) | **신규**(Yonghee 판단 대기) |

- **기대**: (i) **L9** → free-rider가 zero/random 양쪽서 exact-0 계열 생존·renorm 붕괴 재현.
  (ii) **L10** → 오염 클라마다 dose를 $U(0.5,1)$로 벌려 순위 축퇴 완화 → F-4(strmain dose-해상도)의
  LLM 대응 = Flirds ≳ Flirds-1st 변별 기대. **caveat**: R4 fidelity 포화의 주원인은 near-additive
  레짐이라 dose 변조로 **부분 개선**만 될 수 있음(완전 해소 아님).
- **구현 재사용**: L10 = silo5 graded-noisy(L6, `answer_swap_graded`)를 주무대로 승격 · L9 = CNN
  `track_c2` frrand 위협 훅의 LLM 대응. 신규 러너 최소.
- **overview 반영 완료**: `survey/flirds-paper-results-overview.md` §5.1(C)·§5.2 F-4·§5.3 R4 개입표(L9·L10 ⬚ 축).

## 4. anchor5 β0.3 재실행 3셀 (ShapleyFL β 감사 — C1은 `REMAINING-slurm.md` §4)

논문 인용 ShapleyFL 값이 실제 **β=0.5** rundir 산출로 판명(감사 07-23): 1B_anchor5 3셀
(git_sha `39a0a97`, 06-15)이 β0.5→0.3 변경(`e89af94`, 06-25) **이전** — 재실행 계획 미반영.
paper B.5 "β=0.3" 서술과 불일치 → **β0.3 재실행 확정(Yonghee 07-23)**.
- **셀 = 1B_anchor5 3셀**(track_d; seeds 0·1·2). 오케스트레이터 = `rerun_beta03/` 재사용.
- 실행 = `SFL_BETA=0.3`(현 소스 기본값 이미 0.3) + 셀당 `RUNDIR_REPLACE=1`(β0.5 원본 명시 교체; 정체성 가드).
- 완료 후: rundir 교체 커밋 → overview §3.1.1(anchor5) ShapleyFL 행 갱신 → paper §5.2 F4·부록 C 갱신 +
  **B.5 재실행-대기 주석 삭제**(C1 3090분 함께 착지해야 완결). 영향 = cross-game 비교표만(same-game 무영향).
