# REMAINING — 남은 작업 (상시 현행; 완료·폐기 항목은 지우고 git 히스토리로만 남김)

> 갱신 2026-07-23. 완료·폐기 결정 기록은 커밋 메시지·git 히스토리 참조.
> 파일-canon: rundir → overview → paper.

## 1. 실험 (GPU) — 순서대로

- **환경(컨테이너 공통; 2026-07-20 재구축)**:
  `BATCH=/NHNHOME/26msit001_A/BASE/edge_ai_lab/yonghee/flirds_batch` 기준
  `PY=$BATCH/venv/bin/python`, `HOME=$BATCH/home`,
  `HF_HOME=$BATCH/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`, `codes/`에서 `PYTHONPATH=.`.
  현 컨테이너=B200 4장(0–3). venv는 기존 rundir meta.json과 **동일 버전 고정**(torch 2.12.0+cu130,
  transformers 5.9.0, trl 1.5.1, peft 0.19.1, accelerate 1.13.0, datasets 4.8.5, numpy 2.4.6).
  meta-llama gated 재취득 불가(유효 토큰 무) → 해시 교차검증된 공개 미러로 캐시 재구성 —
  검증 체인·근거는 `$BATCH/PROVENANCE.md`.

### 1.0 직전 세션 인수인계 (07-22 02:45 기동 → 07-23 02:46 드레인 종료)

**컨테이너 잔여**: 48h 제약 마감 **07-24 02:2x** — 드라이버 종료 시점 기준 **~23h** 남음
(GPU 4장 유휴, 러너 0). 재기동 = `bash $BATCH/tools/launch_driver.sh`.
**정지 사유**: Yonghee 07-23 "β 작업 이전에 다른 실험을 해야 하니 지금 돌리는 것까지만" —
다음에 무엇을 넣을지는 **미지정**. 큐(`$BATCH/runlogs/queue_postswap.txt`)는 48줄 전부 소비됨.

**gnoise 축 = 종결.** γ=5·20 모두 무대 미성립, dose가 아니라 **방향** 문제(진단·문헌·실무대
3경로 일치). 근거·수치·재현법 전부 `runs/track_h/gnoise_diag/README.md`. 재실험 계획 없음
(§1.6 "하지 않는 것"). 남은 것은 서술뿐 → §2-9.

#### 드라이버·큐 운용 교훈 (다음 캠페인에 반영할 것)

- **큐 정지는 줄 삭제가 아니라 주석 처리.** 드라이버는 매 루프 큐를 다시 읽고 `consumed`
  **인덱스**로 위치를 추적하므로, 줄을 지우면 인덱스가 밀려 오배치가 난다. 줄 수를 보존해야 안전.
- **러너는 셀 단위 원자적** — 중도 kill = 전손. 정지는 드레인(실행 중 셀 완주 후 드라이버 자체 종료).
- **마지막 셀은 `done[ok]` 줄이 안 남는다**: 루프가 매 회 앞에서 `consumed>=N && !running`을
  GPU 리핑보다 먼저 검사해 그대로 빠져나간다. 완주 판정을 `grep 'done\[ok\]'` 단독으로 하지 말 것
  (큐 헤더의 재개 절차 1)이 이 함정에 걸린다) — 셀 로그의 `TRACK G DONE`/`MATRIX DONE` +
  rundir mtime으로 교차 확인.
- **진짜 비효율은 t2 셀의 내부 직렬화**: 한 프로세스가 observer(~6h) 후 t2_pw 4소스를 **순차**
  재학습(소스당 ~3h) → 셀 하나가 17h 점유. 큐 재배열(3B 먼저·LPT)은 이득 0이었음(셀 수 ≫ GPU
  수라 makespan이 총량에 지배됨). **쪼갤 수 있는 축(소스·arm)을 큐 레벨로 노출**할 것 —
  §1.6 L1에 반영 완료(T2 4점수원 = 소스별 프로세스 분할).

#### rundir 충돌 가드 — 처리 완료(2d3f482) + 설계 결함(§1.7)

`RunLogger`는 같은 이름에 **다른 config**면 `<name>_<cfg-sha8>`로 비켜 쓰고, **같으면 덮어쓴다**
(`flirds/run_logger.py:70-82`). config 스키마 증가(`poison`·`dose_mult`·`removal`·`client_opt`·
`noisy_rate` 신규 키 = 해당 셀엔 전부 기본값/무관)로 이번 세션 β 재실행분이 해시 접미사에 착지.
**phase2_matrix 3셀은 canonical로 통합 완료**(a0.1_frzero=49c402a, a0.01_noisy·a0.01_frrand=2d3f482).
- 검증: 신규 = 구 canonical 대비 **Fed-LOO + cos_d/euc_d 추가된 상위집합**(단순 β 재실행 아님).
  겹치는 수치 미세차(Flirds 0.6344→0.6323)는 LLM 트랙 비결정성 범위. 구본은 git 히스토리(4be6f48).
- **track_h observer 해시 3건은 덮어쓰지 않음 — 반대 상황**: canonical=`fa5fc6e`(**pre-fix,
  H1 미적용**), 해시=`262f67c`(post-fix). 해시 쪽이 옳은 판본이나 §1.5-1 "pre-fix 동결 보존"에
  따라 §1.6 사전절차(`rundirs_llm_prefixh1/` 아카이브)로 처리할 것. **그 전에 분석 돌리면
  make_analysis가 pre-fix를 집을 수 있다** — arm별 dup 중 하나만 채택하므로 채택본 확인 필수.

- ⚠ **β는 `config.yaml`에 기록되지 않는다**(소스 리터럴 `beta=0.3` 하드코딩 4곳: `phase2_matrix.py:359`,
  `track_c1.py:302`, `track_d.py:226`, `track_c2_fid.py:218`) → 가드가 β0.5 원본을 지켜주지 못함.
  β 출처 근거는 `meta.json:git_sha`뿐(a0.1_noisy·a0.1_frrand·silo5 4셀은 config가 같아 자동 덮어써짐).
- ⚠ **poison 열은 β 혼재**: poison 3셀 영구 제외(Yonghee 07-23 "오염축 poison 미사용")로
  device100 a0.5/a0.0 + 3B_silo5는 **β0.5 원본 그대로**(mtime 06-12), 1B_silo5(07-20)·
  1B_iid5(07-06)만 β0.3. poison 열을 다시 쓰게 되면 β 일관성부터 확인(§1.4 'β-불변 canon'과 동일 안건).
- ⚠ **P5-soft t2 런의 observer 중복**: t2 런이 observer를 재실행하며 해시 rundir을 **새로 만든다**
  (덮어쓰지 않음 = 원본 보존). 값도 비트동일 아님(val 0.6088232 vs 0.6091607, EM 1문항 차) —
  LLM 트랙은 conv-free라 `cudnn_deterministic` 미사용 = 비트재현 미보장(알려진 특성, 타당성 문제
  아님). `make_analysis`는 arm별 dup 중 하나만 채택하므로 중복집계는 없으나,
  **분석 시 어느 rundir이 채택됐는지 확인**할 것.

### 1.0b 진행 중 캠페인 (2026-07-23 세션 — B200×4 컨테이너, 마감 07-25 03:27)

L1 R4 Tier C **P1-only**(Yonghee 07-23: P5s 전면 중단) 가동 중. 실행 큐 정본 =
`$BATCH/runlogs/queue_L1.txt`(리포 기록 = `runs/track_h/QUEUE_L1_2026-07-23.txt`).
- **순서**: seed0 패치(진행) → seed1 → **L2((b)-fidelity 2셀, seed2보다 앞으로 이동** =
  fidelity 1차·마감 방어) → seed2. 드라이버 = `run_multi_driver.sh`(pid 20798, 단일).
- **seed0 패치 = 옛코드 포크 이슈**: 패치 셀이 §1.7 가드 커밋 **직전** 디스패치돼 옛 레거시
  가드로 돈다. `noisy_obs_t2`가 `t2_sign_{flirds1st,lossheur,fedif}`·`t2_random_k37`을
  persist 시 해시로 갈라짐(숫자는 가드-후와 동일 — FL/스코어링/T2 경로 불변). **정리 =
  `runs/track_h/consolidate_hash_dirs.py --apply`**(canonical별 최신본 유지·해시본 제거).
  워처가 `noisy_obs_t2`(pid 20802) 완주 감지 후 자동 실행(`$BATCH/runlogs/consolidate_watch.log`).
- **완료 후**: rundir 커밋 + `make_analysis.py` 재생성 + H-12/H-13 대조.

### 1.1 P5-soft 분석 (무GPU; ~~즉시 실행 대상~~ → **P5s 중단으로 폐기**, 2026-07-23 Yonghee)

기존 seed0 P5s rundir(flirds_pweight·t2_pw_*)는 보존만(삭제 안 함). P1-vs-P5s EM 표는
목표에서 제외. 아래 원문은 히스토리 참조용.

6런 완주(07-23 02:46; rundir 커밋 완료). `python runs/track_h/make_analysis.py` →
**P1 vs P5s EM 표**(vanilla/oracle_excl 앵커 포함) + `runs/track_h/p5/RUN_P5.md` §4 HP 대조 →
분석 커밋. LLM 몫은 **HP-3·5·6**만 — hard-측 HP-1·2·4는 P5-hard(cgate/csign) 전면 제외로 **N/A**
(CNN 본실험이 별도 Slurm 서버서 커버, 실행 정본 = RUN_P5.md). **MISS는 그대로 보고**
(RUN_P5 §2 공정성 조항: z=1.645 고정, 학습-중 관측 통계만·사전 정보 금지).

### 1.2 β0.3 재실행 잔여 **10셀** (device100 7 + 3B silo5 3) — 07-23 **보류**(§1.0)

진행: 완주 2(a0.1_noisy·a0.1_frrand) + 07-22~23 세션 완주 3(a0.1_frzero·a0.01_noisy·a0.01_frrand)
→ **잔여 10셀 = `queue_postswap.txt`의 `#PAUSED-0723` 줄**(접두어 제거로 재개, ~36 GPU-h ≈ 4-GPU 9h).
poison 3셀은 **영구 제외**(Yonghee 07-23; 부작용은 §1.0 참조). 드라이버 유실 시 수동 재개:
```bash
sed -i 's/^#phase2/phase2/' runs/rerun_beta03/logs/resume36h.txt   # 유실 시 RESUME_AFTER_MIGRATION.md 31줄에서
                                                                   # 완료분 1B_silo5 4셀 제외하고 재생성
PY=$PY PP=<repo>/codes HOME=… HF_HOME=… HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  QUEUE=<abs>/runs/rerun_beta03/logs/resume36h.txt GPUS_FILE=<abs>/logs/gpus36h.txt GPUS="<g...>" \
  LOGDIR=<abs>/runs/rerun_beta03/logs bash runs/rerun_beta03/run_multi_driver.sh
```
완료 후: rundir 커밋 + overview §3.4 phase2 ShapleyFL 행 갱신(**poison 행은 β0.3 아님** — §1.0).

### 1.3 β0.3 deferred 9셀 (최중량 꼬리; 별도 캠페인)
7B_std20×3(70–90h) + device100-a0.5 anchor×3(63h) + 7B_anchor5×3(35–45h) — `RESUME_AFTER_MIGRATION.md`.
완료 후 overview 7B 열(§3.1.1·§3.5.1) 갱신.

### 1.4 장기 대기 (우선순위 낮음)
lr·steps intervention 2차검증(무GPU 재분석) · 1B·CNN β-불변 canon 확인 ·
**P0(H1) 소급 재실행 스코프 결정**(논문 인용 셀 한정으로 이후 진행; 그룹 카탈로그 = git 히스토리의
`RERUN_AFTER_REPRO_FIX_2026-07-21.md` — 파일 삭제 07-23, TF32-on=CNN canon 확정으로 P1은 DROP).
(E5 N=10 oracle 확장(seeds1·2·(a) 2¹⁰) = 미진행 확정, Yonghee 07-22 — 시간 제약.)

### 1.4b Track G CNN skew-축 확장 + fmnist + frrand (**즉시 실행 대상**; Slurm 서버, 2026-07-22)

컨테이너 48h 큐와 **독립**(yonsei Slurm `base_suma_rtx3090`, torch 2.11.0 env). 지시 = Yonghee
2026-07-22. 구현·스모크·사전등록 **완료(로컬 커밋, push 안 함)** — 남은 것은 제출·분석·문서.

- **그리드 90런/30셀** = ① cifar10×{shard(label만),qskew(size만)} ② fmnist×{iid,dir1}
  각각 × {clean, free_rider, **frrand**, grad_noise, label_flip@{0.15,0.35,0.70}} × 3-seed
  + ③ cifar10×{iid,dir1}×frrand 백필 6런. 기존 36런 rundir는 **read-only**.
- **스택 통일 재실행 36런 동반**(Yonghee 07-22 결정): cifar10 {iid,dir1} 12셀을 현 스택에서
  재실행해 `rundirs_cnn_restack/`에 착지 → 2×2 표 단일 스택화(`RERUN_AFTER_REPRO_FIX` P1은
  07-23 TF32-on canon 확정으로 **DROPPED** — restack은 스택-통일 역할만 담당).
  기존 36 rundir는 무수정 보존(→ 두 스택 재현성 drift 표 자동 생성).
- **제출**: `sbatch runs/track_g/sbatch_cnn_skew.sh`(90런) + `sbatch runs/track_g/sbatch_cnn_restack.sh`(36런).
  인덱스는 둘 다 seed-major(필요시 `--array` 절단으로 파일럿 가능).
- **예상 비용**: 126런 총 **48–56 GPU-h**(cifar10 실측 앵커 3.5분/scoring-arm, fmnist 미측정
  0.5–1.0×), QOS 8-GPU 동시 → wall 6–7h.
- **사전등록 H-K1~H-K6** = `runs/track_g/README.md` "확장 ②". 완료 후
  `python runs/track_g/make_analysis.py` → 2×2 분해표·예측 대조·C2 같은-셀 대조 자동 생성.
- **Yonghee 결정 대기 2건**: ① cifar10 {iid,dir1} 12셀 동일-스택 재실행(+18 GPU-h)로
  2×2 표의 스택 경계(감사 M1: 기존=torch 2.12/B200, 신규=2.11/RTX3090) 제거할지
  ② 예산 압박 시 5-arm 축소안 사용 여부(현재는 9-arm 대칭 유지).
- 완료 후: overview §3.2.4 skew-분해·fmnist 블록 + §3.2 커버리지 매트릭스 + §8 갱신.
- **07-22 확장 확정(제출됨)**: ① label_flip **strmain** 셀(rate~U(0.5,1)) 18런(1860471) —
  fidelity 강도응답 ruler + C2 lf 같은-셀 대조 확보 ② **Track H strmain** 51런(1860727;
  17셀타입×3s, P5 경계-클라 첫 시험 무대; `runs/track_h/sbatch_strmain.sh`) ③ **CNN fidelity
  leg 설계 확정**(C2 무대 동결 궤적 × 8방법 vs (b)-perround oracle; (a) 포기·Ripple/Banzhaf/
  **Fed-LOO**[07-23 비교 제외] 제외) → 구현 완료(아래). **종합 계획·교차검증 핸드오프 =
  루트 `CNN_CAMPAIGN_PLAN_2026-07-22.md`**.
- **07-23 fidelity leg 구현 완료 + 교차검증 회신 7건 전건 해소**(plan §6 결정 10–17):
  범위 **144셀 확정**, `codes/experiments/track_c2_fid.py` + `tests/test_c2fid.py`(5 green)
  + fmnist smoke e2e green(eff-gap 0.0) + `runs/track_c/c2fid/{README.md(1행 게임 캐비엇 +
  **사전등록 F-1~F-4**), sbatch_fid.sh}`. (b) 라운드 샤딩 채택(병합 커버리지 assert).
  분석 도구 `runs/track_c/c2fid/make_analysis.py`도 작성 완료(c1 열-호환+stage, F-1~F-4
  자동 판정, 샤드 병합+커버리지 assert; 스모크 rundir로 배선 검증).
  **잔여**: ① **파일럿 제출됨 = job 1861067**(현 큐 4잡 `afterany` 의존 → 자동 기동) →
  완주 시 GPU-h 실측 보고 → Yonghee GO 후 `sbatch runs/track_c/c2fid/sbatch_fid.sh`(144셀)
  ② `runs/track_h/make_analysis.py` strmain 인식 확장(THREATS 4→5종; 완주 후)
  ③ 본런 완료 후 F-1~F-4 대조(MISS 포함 보고).

### 1.5 seed-추가 잔여 3건 (**조건부** — Yonghee 2026-07-22)

> ⚠ **실행 조건: 해당 결과가 논문에 실리는 것으로 확정될 때만 돌린다.** 현재는 수록
> 여부 자체가 미정이라 즉시 실행 대상이 아님 — 논문 구성(배치안 E6-②·§3.3.3·ablation
> A축)이 확정되는 시점에 개별 판단.

1. **R4 Tier C 3-seed** — E6-②(LLM selection 본문) 확정 수치용. **[07-23 조건 충족·실행
   확정]** R4=LLM 주 무대 확정(Yonghee)으로 승격 — 스펙·예산·순서 = **§1.6(L1)**.
   **seeds 0·1·2 전체 재실행**(Tier A seed0 = fix-전 코드 `fa5fc6e`로 판정 — H1 미적용이라
   canonical 아님; pre-fix rundir는 동결 보존). Tier B(L4)와 독립으로 선행.
2. **3B silo5 robustness seeds 1·2**(마스터 P5) — overview §3.3.3·caveat 1(현 1-seed)
   해소용. §1.2의 β0.3 재실행 3B 4셀(seed0 재실행·라벨 통일)과는 별개.
3. **probe A축 seeds 1·2**(rank r32/64·st20/30 셀; lr격자·noise·std50k5-r16은 3-seed
   완료) — ablation A축 보강(선택; overview §4.2 "커진 φ의 cross-seed 실재" 확인).

### 1.6 LLM L1–L6 캠페인 (**차기 B200 컨테이너 최우선** — Yonghee 07-23: β0.3 잔여보다 선행, β는 그 후 재개)

**실행 절차·명령 정본 = `runs/track_h/QUEUE_L1L2_2026-07-23.md`**(사전절차: git-clean 확인 →
hf_pin을 `$BATCH/hf_home` snapshot-SHA로 고정 → pre-fix Tier A 아카이브[git_sha==`fa5fc6e`만
`rundirs_llm_prefixh1/`로] → L2 스모크 → L1 → L2). 사전등록 = `runs/track_h/README.md` H-12·H-13.

| # | 셀 | 비용(GPU-h) | 상태 |
|---|---|---|---|
| **L1** | R4 Tier C: {noisy nr0.7, frzero} × **seeds 0·1·2**(§1.5-1; Tier A arm 세트 동일, T2 4점수원은 소스별 프로세스 분할) | ~120–138 | 확정 — 큐 문서 §2 |
| **L2** | R4 (b)-fidelity: `phase2_matrix.py REGIME=gsm50k5`(**이식 완료 07-23** — nr 기본 0.7·(b) per-round 2⁵·9방법[**Fed-LOO 제외 = Yonghee 07-23**]·탐지기 4종·timing.json) — noisy→clean 각 seed0; 산출은 c2fid 열-호환 스키마로 롤업(CNN fidelity leg와 공용 표) | ~10–15/셀 | 코드 완료·서버 스모크 대기 |
| **L4** | R4 Tier B **T2-only**(renorm 4점수원 × noisy·frzero) | ~150 | **Yonghee 승인 게이트**(§3) — L1·L2 순항 확인 후 |
| L5 | 비등n silo5 1셀(4:2:1:1:1, clean+noisy, 3-seed; phase2_matrix 분배만 변경) — CNN qskew fidelity와 P5c 쌍 | ~10–15 | 여유 시 |
| L6 | silo5 graded-noisy(per-client nr~U(0.5,1), `answer_swap_graded`) — spearman_vs_rate의 LLM 대응(CNN strmain 거울) | ~6–12 | 여유 시(L4와 경합 시 L4 우선) |

- (L3 = 3B silo5 β0.3 4셀은 §1.2 β 재개 큐 몫 — 착지 시 git_sha만 확인.)
- **하지 않는 것**: gnoise 재개·등방 노이즈류 LLM 재시도·LIE/sign-flip(위협 스코프 게이트)·
  (a) retrain 3B/7B(P2/P3)·R4 frrand 추가(frzero와 실질 동일)·P0 전면 소급(§1.4).
- 예산: 필수(L1+L2) 140–168 → +L4 290–318 → 최대 ~345 / B200×4 5일 ≈480 GPU-h(명목).
  vast.ai는 B200 초과 시 **비-timing seed 복제만** 예외(timing/canonical 셀 이관 금지).

### 1.7 rundir 정체성 설계 수정 (**처방 1+2 구현 완료 07-23**; 잔여 배선은 아래 "잔여")

**진단**: `run_logger.py:77`이 `config`를 **문자열 통째로** 비교한다. 그런데 `config`는 상반된 두
역할을 겸한다 — **정체성**("어떤 실험인가", 좁고 안정적이어야 함) vs **출처**("정확히 뭐가 돌았나",
넓고 계속 늘어남). 한 딕셔너리에 섞어 통째 비교하니 **출처를 개선할 때마다 정체성이 바뀐 것처럼
보인다.** 가드가 목적과 **역상관**: 무해한 스키마 증가엔 오발화(5건), 진짜 의미 변경인 β0.5→β0.3엔
침묵(소스 리터럴 → config 바이트 동일 → 조용히 덮어씀). 덤으로 LLM 트랙은 conv-free라 같은 config로도
같은 숫자가 안 나오므로 "config 같음 ⇒ 같은 런"이 애초에 참이 아니다.

**처방 1+2 구현 완료(07-23, Yonghee 승인)**. 원칙 = **정체성 ≡ 이름이 인코딩하는 것** →
"이름 같은데 정체성 다름"은 정의상 이름 생성 버그.
1. **정체성 allow-list** — `run_logger.check_identity()` + `RunLogger(..., identity=(...))`.
   같으면 **덮어쓴다**(이전 판본은 git). 다르면 `RunDirIdentityError`로 **크게 실패**(팬텀 미생성).
   **저장된 config에 정체성 키가 아예 없으면 불일치로 센다** — β0.5 시절 rundir엔 `sfl_beta`
   키가 없고, 그 부재가 바로 조용한 덮어쓰기를 허용했던 구멍이다. 우회 = `RUNDIR_REPLACE=1`.
   ⚠ **`precheck` 추가 이유**: `RunLogger`는 **arm 종료 후 `persist()` 안에서** 생성된다
   (`track_g.py:589` ← `:678`). 그대로 두면 16h 돌고 마지막에 죽으므로,
   `RunLogger.precheck(...)`를 **프로세스 시작 시점**(`_load` 전)에 호출한다.
2. **β 단일화** — `flirds/baselines/shapleyfl.BETA = float(env SFL_BETA, 기본 0.3)`.
   ⚠ **정정: 하드코딩은 4곳이 아니라 7곳이었다** — `shapleyfl_from_logs(beta=)` 4곳
   (phase2_matrix·track_c1·track_c2_fid·track_d) + `OnlineScorer(beta=)` 2곳(track_c2·track_d)
   + `sfl_beta=` 1곳(phase1_baseline_compare). 4곳만 고쳤으면 나머지 3곳이 조용히 갈라졌다.
   phase2_matrix는 `sfl_beta`를 config+정체성에 실었다(+`removal`·`client_opt`).
   → **β0.3 잔여 10셀 재개 시 `RUNDIR_REPLACE=1`이 셀당 1회 필요**(의도된 동작).
- 배선 완료 = `track_g`(IDENTITY: track/regime/threat/noisy_rate/arm/seed/scale/model — 이름에
  없는 scale·model 포함 → 3B가 1B 셀을 덮어쓰면 실패) · `phase2_matrix`.
  테스트 `tests/test_rundir_identity.py` 6개(출처 증가 무포크 / 정체성 변경 차단 / β-키-부재 /
  RUNDIR_REPLACE / precheck 무부작용 / 레거시 경로 불변).

**잔여**: ① `track_c1`·`track_c2`·`track_c2_fid`·`track_d`·`phase1_*`는 아직 `identity=None`
(레거시 통짜 비교) — 이들 config에 `sfl_beta`를 추가하려면 **identity 배선을 함께** 해야 한다
(레거시 가드에선 키 추가 자체가 포크를 유발). ② 처방 3(`superseded.json`) 미착수.

## 2. 문서·부수분석 (무GPU)

1. **overview 반영**(`research-wiki/survey/flirds-experiment-results-overview.md`): E4 Fed-LOO·E5 N=10·
   E7 frdelta·AdamW 3-seed(−0.53±0.33)·probe seeds1-2·loss-heur runtime(96.6/100.1/100.2s)·device 학습시간.
2. **표1 Fed-LOO 재집계**: `python runs/track_d/make_fidelity.py`(root `rundirs_e4_fedloo` 인자 확인).
3. **tab:cost**(`paper/sections/results.tex`): loss-heur 170→~99s·device overhead%·E3 CNN cost·end-to-end/overhead% 2블록.
4. **paper-ko 마커 해소**: E2·E3·E4·E5·E7·E11 🔴TODO/🟣VERIFY + §3.7.4 AdamW 갱신.
5. **Track G 서술**: 잔여 = paper-ko §6.5 — silo5/iid5 3-seed + CNN 그리드 + std50k5-mixed
   **s0 1-seed 파일럿(동결; 1-seed caveat 필수)** 구성. 확정치 = V2w 불승격·frzero 회수 1.0·
   noisy 게이트 침묵·clean parity max|Δ|=0.00056. LLM 참여축 성능 주장은 R4가 담당.
6. **Track H 서술**: overview §3.2.3 반영 완료 — **정본 수치 = §3.2.3**(P1-T1 동률 .707·
   P1-T2 flirds .839 1위·renorm 붕괴는 FR 국한 −5.9~−6.6·GN은 renorm도 0.9+;
   07-19 커밋 메시지의 정정-전 수치 인용 금지). 잔여 = paper 반영. LLM 경쟁 무대 = R4.
7. **부수분석**: 3.1 loss-heur 정본화(CSV/rundir) · oracle noisy AUROC 0.604/0.660 불일치 확정 ·
   bootstrap CI(B=1000) · momentum 열화(0.73 vs 0.81) 정본 rundir 위치.
8. **STD-DAGMM 보고 캐비엇(07-23 확정, 조치 없음 = Yonghee "평균이 크게 안 변했으면 됐다")**:
   **단일 seed 값은 재현되지 않는다.** device100-a0.01 재실행에서 같은 seed인데 AUROC가
   noisy s0 0.918→0.505 등 **평균 |Δ|=0.177**(2위 ComFedSV 0.021의 8배). 원인 = 코드 변경 아님
   (`std_dagmm.py` 무수정, run seed 정상 전달) — ①같은 입력 교란을 7배 증폭(φ median 변화
   Flirds 3.9% vs STD-DAGMM 28%; AE+GMM 200-epoch 비볼록 + energy=음의 로그밀도) ②오염/정상
   점수 분포가 겹쳐 **순위 여유 0**(Flirds는 AUROC 변화가 눈금 1/(5·93)의 정확히 1칸 = 순위
   1건, STD-DAGMM은 ~192칸). **3-seed 평균은 안정**(noisy 0.652→0.600, frrand 0.925→0.892)
   → 논문 인용 시 **3-seed mean±std 병기 필수 + 단일 seed 재현 불가 caveat**.
   (미분리: φ가 애초에 4% 움직인 원인 = LLM 비결정성 vs 188커밋 중 수치 변경. 확인하려면
   현 코드로 같은 seed 2회 = device100 1셀 ~5h.)
9. **NEW gnoise negative result 서술**: "CNN 표준 grad-noise 위협이 LLM LoRA에서는 무대 미성립"
   — 근거·수치·문헌대조 전부 `runs/track_h/gnoise_diag/README.md`에 정리됨(그대로 인용 가능).
   배치 후보 = paper §5 한계/negative 또는 부록 1문단. **인용 금지 항목 주의**(Krum σ=200,
   arXiv 2509.09097·2602.19926·2605.07961 = 검증 실패). H-10은 CNN 결과로만 서술.

## 3. Yonghee 결정 대기

- **push**: 로컬 커밋 다수 — push 여부/시점(07-22: Yonghee가 직접 push 예정).
- **R4 Tier B 진입**(§1.6 L4; Tier A seed0 보고 완료 — overview §3.2.7).
- **§1.7 rundir 정체성 수정 범위**: 처방 1+2(β를 정체성에 태움 — 재개 시 명시 플래그 필요) vs
  1만(β는 문서 관리). 미착수.
- **잔여 컨테이너 ~23h(마감 07-24 02:2x)에 무엇을 넣을지** — "β 이전에 할 다른 실험" 미지정.
