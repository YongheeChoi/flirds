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
> **⚠ seed 정책 (2026-07-24 Yonghee): 전 실험 예외 없이 3-seed.** 이 결정이 아래 및
> `QUEUE_L1L2_2026-07-23.md` §2·§3의 seed 범위를 **대체**한다(그 문서의 "L2 seed0 2셀"·"L1 noisy·frzero"
> 표기는 이 노트로 무효화). 두 가지 변경:
> - **L2 → noisy·clean × seed{0,1,2} = 6셀**(종전 seed0 2셀). frzero-(b)는 free-rider φ=0 정의상
>   **해석적 exact-0**이라 측정 seed 셀이 아님 — seed 미달이 아니라 정의상 상수(§5.4 frzero = 해석값 + L1 3-seed 개입으로 충족).
> - **L1에 clean threat 추가 → clean·noisy·frzero × seed{0,1,2}.** 이유 = §5.3 clean 열(오발화 대조·
>   "R4 clean T1 −1.0pt")의 3-seed 정본 소스가 **전무**하기 때문. clean 개입 EM은 `track_g THREAT=clean`
>   에서만 나오는데(L2=phase2_matrix는 fidelity·탐지만 산출, 게이트 arm 미생성) L1이 noisy·frzero만 돌아
>   고아 상태 — 현존 clean 개입 = 제외된 seed0 파일럿뿐. clean 셀 = L1 레시피에서 THREAT=clean,
>   **oracle_excl/random_excl 없음**(제외 대상 부재) + **T2 clean은 kept=전원→`equals_vanilla`로 스킵**
>   → 실질 신규 = T1(online) 게이트 arm뿐 = 저비용.
- **순서**: seed0 패치(진행) → seed1 → **L2((b)-fidelity 6셀, seed2보다 앞으로 이동** =
  fidelity 1차·마감 방어) → seed2. clean threat은 L1 각 seed에 동반. 드라이버 = `run_multi_driver.sh`(pid 20798, 단일).
- **사전절차 완료**: pre-fix Tier A(git_sha==`fa5fc6e`) 20 rundir → `runs/track_h/rundirs_llm_prefixh1/`
  아카이브·커밋 — make_analysis의 pre-fix 오채택 위험 해소.
- **seed0 패치 = 옛코드 포크 이슈**: 패치 셀이 rundir 정체성 가드 커밋 **직전** 디스패치돼 옛 레거시
  가드로 돈다. `noisy_obs_t2`가 `t2_sign_{flirds1st,lossheur,fedif}`·`t2_random_k37`을
  persist 시 해시로 갈라짐(숫자는 가드-후와 동일 — FL/스코어링/T2 경로 불변). **정리 =
  `runs/track_h/consolidate_hash_dirs.py --apply`**(canonical별 최신본 유지·해시본 제거).
  워처가 `noisy_obs_t2`(pid 20802) 완주 감지 후 자동 실행(`$BATCH/runlogs/consolidate_watch.log`).
- **완료 후**: rundir 커밋 + `make_analysis.py` 재생성 + H-12/H-13 대조 → paper I1·F2·D1 ⬚ 채움.

| L2 셀 (각 × seed s∈{0,1,2}) | 명령 |
|---|---|
| noisy (nr 기본 0.7) → `1B_gsm50k5_noisy_nr0.7_s{s}` | `REGIME=gsm50k5 THREAT=noisy SEED=<s> CUDA_VISIBLE_DEVICES=<g> PYTHONPATH=. $PY -u experiments/phase2_matrix.py` |
| clean (포화 대조) → `1B_gsm50k5_clean_s{s}` | `REGIME=gsm50k5 THREAT=clean SEED=<s> CUDA_VISIBLE_DEVICES=<g> PYTHONPATH=. $PY -u experiments/phase2_matrix.py` |

L2 = `phase2_matrix.py REGIME=gsm50k5`(nr 0.7·(b) per-round 2⁵·9방법[Fed-LOO 제외]·탐지기 4종·timing.json);
산출은 c2fid 열-호환 스키마 롤업(CNN fidelity leg와 공용 표). **frzero-(b) fidelity/탐지는 해석적 exact-0**
(free-rider φ=0)이라 측정 seed 셀 불요 — 3-seed 의무의 예외가 아니라 정의상 상수(§5.4 frzero 행 = 이 해석값 + L1 3-seed 개입).

## 2. 대기 큐 — L4·L5·L6·L7

| # | 셀 | 비용(GPU-h) | 상태 |
|---|---|---|---|
| **L4** | R4 Tier B **T2-only**(renorm 4점수원 × **clean·noisy·frzero** × 3-seed) | ~200–230 | **Yonghee 승인 게이트** — L1·L2 순항 후 |
| **L11** | R4 §5.3 online 완성 | 7 non-flirds(flirds1st·lossheur·fedif·gtg·fedsv·comfedsv·shapleyfl) T1 부호-게이트 × clean·noisy·frzero × 3-seed = **63 run** — 기존 observer cum φ 재사용(신규 FL run만·관찰자 재실행 0) | ~250–300 | **신규**(seed=3; CNN 8방법 parity, 2026-07-24 확정·GO 대기) |
| L5 | 비등n silo5 1셀(4:2:1:1:1, clean+noisy, 3-seed) — CNN qskew fidelity와 P5c 쌍 | ~10–15 | 여유 시 |
| L6 | silo5 graded-noisy(nr~U(0.5,1), `answer_swap_graded`) — spearman_vs_rate LLM 대응 | ~6–12 | 여유 시(L4 우선) |
| **L7** | **R4 P1w**(w∝max(cum,0)·합-1 재정규화; flirds-only) × {clean,noisy,frzero} × 3-seed × {T1,T2} = 18런 — 스펙·H-14 = `paper/workplan/T3-p1w-llm-impl.md`; **실행 런북 = §2a** | ~80 | **코드 구현 완료**(2026-07-23·이 세션; 커밋·push 후 실행) — 순서 L2 뒤·**L4 앞** |

- **L4 clean (2026-07-24 Yonghee)**: renorm-4도 §5.3 **clean 오발화 열**이 필요 → L4 = clean·noisy·frzero
  × 3-seed(종전 noisy·frzero만). flirds와 달리 renorm은 clean에서도 음수 φ로 **오발화(false-firing)** 가능
  → clean T2가 `equals_vanilla` 스킵 안 되고 실제 재학습 발생(비용 산입). §5.3 clean 열의 renorm-4 칸 = 이걸로 채움.
  **online 표 8방법 = 확정(2026-07-24 Yonghee "당연히")**: R4 §5.3은 CNN처럼 **online·retrain 두 표 각 8방법**.
  현재 online = flirds만(L1) → 나머지 **7방법**(flirds1st·lossheur·fedif·gtg·fedsv·comfedsv·shapleyfl) T1-online = **신규 L11**(아래).
  (retrain 8방법 = L1 4 exact-0 + L4 renorm-4로 이미 완성.)
- **하지 않는 것(재제안 금지)**: gnoise류 LLM 재시도(종결 = `runs/track_h/gnoise_diag/README.md`) ·
  LIE/sign-flip · (a) 3B/7B · P0 전면 소급 · **Fed-LOO**(양 무대 공통 제외; 러너 산출은 무해·미게재) ·
  **poison**(양 무대 공통, 영구 제외) · P5h/P5s(rundir는 보존) · std20/anchor5-vs(b) 재실행 ·
  E5 N=10 확장 · **β0.3 잔여 재실행(device100·3B·7B — 대상 표 전부 논문 제외로 폐기 07-23;
  부활 시 목록 = `runs/rerun_beta03/RESUME_AFTER_MIGRATION.md`)**.
  (※ R4 frrand는 07-23 번복 → §3 L9로 부활.)
- 예산(3-seed·full-8 반영, 2026-07-24): 필수(L1[+clean]+L2 3-seed+L7) ≈ **275–335** → +L4(~200–230)
  +**L11 online 완성(~250–300)** ≈ **725–865**(≈ B200×4 **7.5–9일**) / 5일 명목 480 **대폭 초과**.
  ⟹ **§5.3 clean·noisy·frzero full-8(online+retrain)·3-seed는 5일 예산 밖** — Yonghee 스케줄 결정 필요(연장 or vast.ai seed-복제 예외).
  ※ **frrand(L9)=full-8 확정(2026-07-24)**: renorm 붕괴를 random-FR서도 시연(8방법×{T1,T2}×3seed, ~200 = +~150 vs flirds-족).
  **strmain(L10)=fidelity-leg 유지**(downstream full-8 아님; spearman_vs_rate·vs(b)). ⟹ **전체 ≈ 960–1100 GPU-h ≈ 10–11.5일**(B200×4) → vast.ai 병행 필수(아래).
  **vast.ai 활성(2026-07-24 Yonghee)**: 5일 초과분 vast.ai로 — 단 **seed0(canonical 앵커)+전 timing 셀은
  B200 고정**(스택-일관·hw 측정), **seeds 1-2의 non-timing fidelity/downstream 복제만 vast.ai**(Spearman·recovery
  스택-강건 = W-A). **timing.json은 vast.ai 산출 사용 금지**(§5.5 cost는 B200 실측만).

### 2a. L7 실행 런북 — R4 P1w (코드 구현 완료 2026-07-23·이 세션·로컬 Windows; 커밋·push·서버 pull 후 실행)

> 스펙·사전등록 = `paper/workplan/T3-p1w-llm-impl.md` + `runs/track_h/README.md` **H-14**(실행 전 커밋).
> **P1w ≡ Track H P2**(sign+크기가중) — 신규 게이트 없음. arm 라벨: **T1 `flirds_gatew_v2`**(온라인;
> `build_arm`의 `_gatew_v2` 분기 이미 존재 = 신규 arm 코드 0) / **T2 `t2_signw_flirds`**(`T2_W=1` 신규 블록).
> "P1w"는 분석 표기만(CNN W-B와 동일 규약; `make_analysis.parse_arm`이 `gatew_v2`/`t2_signw`→policy P2로 매핑
> → LLM·CNN P1w가 같은 정책으로 묶여 '전 범위 승격' 규칙 성립).

**구현 파일(이 세션·미커밋)**: `flirds/fl/intervene.py`(`signw_retrain_wvec`=max(cum,0)^α 가중벡터 +
`make_static_weights_fn`=t2_pw/t2_signw 공용 정적-가중 함수) · `experiments/track_g.py`(`T2_W` 플래그 +
`t2_signw_<src>` 방출 블록[관찰자 cum>0 kept, w∝n·max(cum,0)^α] + 재학습 wf를 `make_static_weights_fn`으로
교체[t2_pw 비트동일] + `flirds_gatew_v2` 승격 주석 + config `t2_w` provenance) · `tests/test_p1w.py`(6종 신규).
**테스트 전부 green**(로컬 fl_shapley env): `test_p1w`(6)·`test_signgate`(15 회귀)·`test_track_h`(7 회귀)·
`test_r4`(4; **Windows는 `PYTHONUTF8=1` 필수** — trl import cp949 이슈, 리눅스 서버는 불요).
**로컬 스모크 green**(GPU tiny-gpt2+SYNTH_DATA offline, `CI=1`로 trl telemetry 우회): noisy·clean 두 셀 오류 0 —
T1 gatedweight 가중 로그 + T2 `t2_signw_flirds` 비어있지 않은 kept서 **재학습 실행 확인**(P1은 kept=전원→
`equals_vanilla`, **P1w는 가중 불균등→재학습** = T3 의도된 P1↔P1w 차이).

**셀**: {clean, noisy nr0.7, frzero} × seeds {0,1,2} = 9셀. 셀당 신규 = **T1 + T2 = 2런 → 18런**(~80 GPU-h, 런당 ~4.5).

**서버 스모크(GPU 1장, 수 분; 실모델 배선 확인)** — `PERSIST=0`이라 rundir 미생성:
```
REGIME=gsm50k5 THREAT=noisy SEED=0 PERSIST=0 DOWNSTREAM=0 ROUNDS=3 VAL=20 MAX_STEPS=2 \
  ARMS=observer,flirds_gatew_v2 OBS_SOURCES=flirds T2=1 T2_W=1 T2_LEGACY=0 T2_P5=0 \
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. $PY -u experiments/track_g.py
```
확인: `flirds_gatew_v2` 가중 로그 + `t2_signw_flirds` 산출 + 오류 0.

**본 셀(GPU 1장/셀; seed-major 4-GPU 분산)** — L1 canonical(`rundirs_llm`) 무수정 위해 **별도 root 착지**:
```
RUNDIR_ROOT=<repo>/runs/track_h/rundirs_llm_p1w \
REGIME=gsm50k5 THREAT=<clean|noisy|frzero> SEED=<0|1|2> \
  ARMS=observer,flirds_gatew_v2 OBS_SOURCES=flirds T2=1 T2_W=1 T2_LEGACY=0 T2_P5=0 \
  CUDA_VISIBLE_DEVICES=<g> PYTHONPATH=. $PY -u experiments/track_g.py
```
- noisy는 `NOISY_RATE=0.7` 기본, frzero는 THREAT=frzero(nr 무관). 이 명령은 **가장 단순·정확**(오늘 코드 그대로 동작).

**⚠ 관찰자 재사용 vs 재실행 (Yonghee 실행 결정 — 예산에 직결)**:
- T2 `t2_signw`는 **관찰자(vanilla 궤적)의 최종 cum**이 필요 → 위 명령은 셀마다 `observer`를 **동반 재실행**
  (~4.5 GPU-h/셀 = estimator arm 등가). 총비용 ≈ 80(P1w 18런) + ~40(관찰자 9회) ≈ **~120 GPU-h**.
- **T3 §3 예산 ~80은 "관찰자·통제 L1 재사용(재실행 0)" 전제.** noisy·frzero 관찰자는 L1 Tier C가 이미 산출
  (`rundirs_llm/gsm50k5_{noisy_nr0.7,frzero}_observer_seed{0,1,2}`의 `metrics.json → observer_cum[flirds]`).
  재사용하려면 **T2 블록이 그 cum을 읽어 obs_accs 구성하는 소형 로더가 필요**(현 코드 미구현 — cum-only라
  `T2_W`/`T2_LEGACY`만 안전, `T2_P5` 병용 금지; 실 rundir로 스키마·경로 검증 필수). **clean 관찰자는 canonical
  부재 가능**(L1 Tier C=noisy+frzero만; Tier A clean seed0은 pre-fix 비-canonical) → clean은 관찰자 신규 실행 불가피.
- **권장**: (a) 예산 여유 → 위 명령대로 셀별 관찰자 동반(단순·정확, ~120) · (b) 예산 빠듯 → noisy·frzero만
  L1 cum 재사용 로더 추가 + clean만 관찰자 실행(~80~90). **로더 추가 시 T1은 `ARMS=flirds_gatew_v2`만(관찰자 불요)로
  분리 실행 → `rundirs_llm`에 직접 착지 가능**(P1w arm명 신규라 충돌 0, 별도 root 불필요).

**분석 후속(실행 후)**: `runs/track_h/make_analysis.py` 확장 — ① 별도 root 착지 시 LLM 로더에
`_load(ROOT/"rundirs_llm_p1w")` 추가(track_h dup-win 유지) ② `competition_score` LLM stage 필터에 gsm50k5
편입(P1w 행) → **H-14 자동 대조**(pre-registered) → overview §3.2.4 이웃 신규 소절 기입 → paper §5.3·T2 페이지는 그로부터.

**완료 조건(T3 §4)**: 18런 rundir(fix-후 git_sha) + 분석 CSV + overview §3.2.4 결과 +
**수록 규칙 판정 1줄(승/동률/미수록)을 `paper/workplan/00-INDEX.md` §1에 기록**. push는 Yonghee 직접.

## 3. 주무대 위협-대칭 확장 — R4 frrand + strmain-dose (2026-07-23 Yonghee 신규)

> **종전 "R4 frrand 재제안 금지"를 번복.** 주무대 쌍 모두 free-rider-random 축이 불완전:
> R4(gsm50k5)는 **frrand 전무**(frzero만). 값싸고 모달리티 제약 없음 → frrand를 full-method 축으로
> 완성 + strmain-dose 추가 = 대칭 복원 + fidelity 변별(순위 축퇴 완화)이 목적. (CNN 대응 = `REMAINING-slurm.md` C-fr.)

| # | 셀 | 내용 | 비용(GPU-h, 추정) | 상태 |
|---|---|---|---|---|
| **L9** | R4 frrand **full-8** | free-rider-random(zero 대신 무작위) × **8방법**(same-game 3+FedIF+renorm-4) × {T1 online, T2 retrain} × 3-seed — clean·noisy·frzero 열과 동일 구성(L1+L4+L11 machinery에 frrand threat 추가); renorm 붕괴를 random-FR서도 시연 | ~200 | **신규**(seed=3·full-8 확정 2026-07-24; 우선순위·GO만 대기) |
| **L10** | R4 strmain류 per-client noisy-dose | answer-swap rate $\sim U(0.5,1)$ 클라별 변조(=FedCorr `strmain` 기본 draw) × 3-seed — **지표 = spearman_vs_rate(φ↔per-client dose) + vs (b) fidelity** | ~30–45(L2형 per-seed ~10–15) | **신규**(seed=3; GO 대기) |

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
