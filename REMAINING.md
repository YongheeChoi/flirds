# REMAINING — 남은 작업 (상시 현행; 완료·폐기 항목은 지우고 git 히스토리로만 남김)

> 갱신 2026-07-21. 완료·폐기 결정 기록은 커밋 메시지·git 히스토리 참조.
> 파일-canon: rundir → overview → paper.

## 1. 실험 (GPU; 새 컨테이너) — 순서대로

- **환경(컨테이너 공통; 2026-07-20 재구축)**:
  `BATCH=/NHNHOME/26msit001_A/BASE/edge_ai_lab/yonghee/flirds_batch` 기준
  `PY=$BATCH/venv/bin/python`, `HOME=$BATCH/home`,
  `HF_HOME=$BATCH/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`, `codes/`에서 `PYTHONPATH=.`.
  현 컨테이너=B200 4장(0–3). venv는 기존 rundir meta.json과 **동일 버전 고정**(torch 2.12.0+cu130,
  transformers 5.9.0, trl 1.5.1, peft 0.19.1, accelerate 1.13.0, datasets 4.8.5, numpy 2.4.6).
  meta-llama gated 재취득 불가(유효 토큰 무) → 해시 교차검증된 공개 미러로 캐시 재구성 —
  검증 체인·근거는 `$BATCH/PROVENANCE.md`.

### 1.0 현 세션 진행 상황 (컨테이너 07-22 02:2x 생성, 드라이버 02:45 기동; **48h 제약 → 마감 07-24 02:2x**)

재개 절차(venv 확인 → `launch_driver.sh` → 4-GPU 디스패치)는 **02:45 완료**. `queue_postswap.txt`
체크리스트대로 `done[ok]: 1B_device100-a0.1_frrand`(07-21 17:33) 확인 → IFKILLED 줄 주석 유지.

#### gnoise 축 — **종결**(Yonghee 07-22 결정: "gnoise 제외하고 나머지 먼저")

γ=5(`gn_full`, 08:56 킬)·γ=20(`gn20_probe`, 16:01 완주) 모두 밴드 미달. **γ 축을 닫는다.**

| | val | gsm8k_em | |
|---|---|---|---|
| clean observer | 0.60219 | 0.3771 | 기준 |
| **gnoise γ=5** | 0.60246 | **0.3753** | oracle보다 높음(부호 반대) |
| **gnoise γ=20** | 0.60247 | **0.3718** | 부호는 정상화, 여전히 밴드 밖 |
| oracle_excl(γ-무관) | 0.60239 | 0.3735 | |

**판정 = dose가 아니라 방향 문제**(진단·문헌·실무대 3경로 일치). 상세·재현법 =
`runs/track_h/gnoise_diag/README.md`(진단 스크립트 + diag30.json 동봉). 요점만:
- **진단**(forward-only 섭동, γ 1~1000 × 3방식 × 2스냅샷): (a)A·B노이즈≈(b)ΔW등방≈0,
  **(c)gradient방향만 2,400~38,000배**. γ=1000의 ‖Ξ‖/‖W0‖=1.30%인데 Δval +0.0083뿐.
  (a)>(b)이므로 "ΔW 공간으로 dose 재지정" 처방은 **역효과**. ‖A‖는 학습 내내 불변(B만 움직임).
- **문헌**: Fang(USENIX Sec'20)이 가우시안 공격 σ를 benign 클라간 분포에 moment-matching
  (**canonical γ≈1**)하고, 이를 *"randomly crafted models can not effectively attack"*를 보이는
  **음성 대조군**으로 명시 → 우리 결과는 실패가 아니라 **예측된 결과**. LoRASC(EMNLP'24)는
  같은 축(std of BA)에서 λ=10≈**γ2.9**를 **성능 향상** 레시피로 씀. γ=1000도 INT8 양자화 수준.
  **LLM/LoRA FL 노이즈 dose 선행 0건** — OpenFedLLM §5.4가 남긴 open problem 그 자체.
- **γ*=5 선정근거 무효**: `_add_gnoise`는 `trainer.train()` **이후** 주입
  (`flirds/fl/llm_server.py:91-97`) → train_loss는 노이즈를 구조적으로 미반영. "abs-probe
  train_loss 3자리 동일 = γ-포화"는 근거가 아니었음.
- **남은 것**: negative result 서술(§2에 신설) + 위협 교체 여부. 문헌 표준 대안 = **LIE**
  (`μ_j+z·σ_j`, z≈1σ, 조정된 mean-shift) 또는 **sign-flip**(크기 파라미터 없음). 프로젝트
  인벤토리 **I-28**(direction-aligned poison = 2차항 flagship 후보)과 동일 안건.
  **H-10은 CNN에서 이미 성립**(vanilla .244 vs Flirds .567~.607 vs 1차계열 .244~.248)이라
  서사는 유지되고 LLM leg만 공백.

#### P5-soft 6런 (진행 중)

online 2런 완주(`noisy` 06:41, `clean` 06:43). 잔여 = `noisy_t2`·`clean_t2`·`frzero_online`·
`frzero_t2`. **주의**: t2 런이 observer를 재실행하며 `gsm50k5_noisy_nr0.7_observer_seed0_4a55c78a`
처럼 **해시 접미사 rundir을 새로 생성**(큐 주석의 "결정론적 동일값 덮어쓰기"는 부정확 —
덮어쓰지 않고 별도 저장 = 원본 보존). 값도 비트동일 아님(val 0.6088232 vs 0.6091607,
EM 0.3342270 vs 0.3351206 = 1문항): LLM 트랙은 conv-free라 `cudnn_deterministic` 미사용
= 비트재현 미보장(알려진 특성, 타당성 문제 아님). `make_analysis`는 arm별 dup 중 하나만
채택하므로 중복집계는 없으나, **분석 시 어느 rundir이 채택됐는지 확인**할 것.

#### 세션 종료 — **드레인 정지**(Yonghee 07-23 00:2x: "β 작업 이전에 다른 실험을 해야 해서
우선 지금 돌리는 것까지 하고 멈추자" + "poison 오염축은 안 쓸 거라 빼도 된다")

**큐 편집 완료**(`$BATCH/runlogs/queue_postswap.txt`, 00:25). 실행 중 4셀만 완주하고 드라이버가
스스로 `MULTI-DRIVER DONE`으로 종료 — **kill 없음**(러너는 셀 단위 원자적이라 중도 킬이 곧 손실).
- **줄 삭제 대신 주석 처리**: 드라이버는 매 루프 큐를 다시 읽고 `consumed` **인덱스**로 위치를
  추적하므로 줄을 지우면 인덱스가 밀려 오배치가 난다. 줄 수(48)를 보존해야 안전.
- `#DROPPED-poison` **3셀 영구 제외**: device100 a0.5/a0.0 + 3B_silo5. `#PAUSED-0723` **10셀 보류**
  (device100 7 + 3B silo5 3, ~36 GPU-h) — 접두어만 지우면 그대로 재개.

| GPU | 셀 | 시작 | 완료(예상) |
|---|---|---|---|
| 1 | `p5s_gsm_clean_t2` | 07-22 06:41 | ~00:20 (fedif 1소스 잔여) |
| 2 | `1B_device100-a0.01_noisy` | 07-22 20:32 | ~01:00 |
| 3 | `1B_device100-a0.01_frrand` | 07-22 20:49 | ~01:20 |
| 0 | `p5s_gsm_frzero_t2` | 07-22 08:56 | **~03:00** ← 드라이버 종료 시각 |

**poison 제외의 부작용(기록 필수)**: `runs/phase2_matrix/rundirs/`의 poison 5셀 중 device100 2 +
3B_silo5 1은 mtime 06-12 = **β0.5 원본 그대로**(재실행 안 됨), 1B_silo5(07-20)·1B_iid5(07-06)만
재실행분. **β는 config.yaml에 기록되지 않아** mtime·커밋 이력이 유일한 근거 — 매트릭스에 β 혼재.
poison 열을 다시 쓰게 되면 β 일관성부터 확인할 것(§1.4 '1B·CNN β-불변 canon 확인'과 같은 안건).

**rundir 이름 규칙(알아둘 것)**: `RunLogger`는 같은 이름에 **다른 config**가 있으면
`<name>_<cfg-sha8>`로 비켜 쓰고, **같으면 덮어쓴다**(`flirds/run_logger.py:70-82` collision guard).
이번에 `1B_device100-a0.1_frzero`가 config 스키마 증가(attacker_*·poison·dose_mult 등 신규 키)로
`..._fd887e1b`에 비켜 썼으나 **Yonghee 07-23 지시로 canonical에 덮어씀**(해시 디렉터리 제거,
β0.5 원본은 git 히스토리에만). a0.1_noisy·a0.1_frrand·silo5 4셀은 애초에 config가 같아 자동 덮어씀 —
**β가 config에 없어 가드가 β0.5 원본을 지켜주지 못한다**는 점은 그대로 유효(위 poison 항과 같은 안건).

**남은 후처리(무GPU, 순서대로)**: ①P5-soft 6런 완주 → `make_analysis` → P1 vs P5s EM 표
(vanilla/oracle 앵커) + RUN_P5.md §4 HP 대조(LLM 몫: HP-3·5·6 재판정, hard-측 HP-1·2·4는
N/A) → rundir+분석 커밋 ②이번 세션 β 완주 3셀(a0.1_frzero·a0.01_noisy·a0.01_frrand) rundir 커밋
③gnoise negative result 서술(§2-8) ④push는 Yonghee 결정 대기.

**이번 세션 스케줄 회고(다음 캠페인 반영)**: 큐 재배열(3B 먼저·LPT)은 이득 0이었음 — 셀 수 ≫ GPU
수라 GPU가 쉬지 않고 makespan이 총량에 지배됨. **진짜 비효율은 t2 셀의 내부 직렬화**: 한 프로세스가
observer(~6h) 후 t2_pw 4소스를 **순차** 재학습(소스당 ~3h) → 셀 하나가 17h를 점유. 소스는 서로
독립이므로 **다음 캠페인은 쪼갤 수 있는 축(소스·arm)을 큐 레벨로 노출**할 것(4-GPU 분할 시 8h).

### 1.1 R4 Tier A — gsm50k5 accuracy 파일럿 seed0 (**완주 07-22 01:31, 커밋 4c40e30**;
gnoise 셀은 §1.0에서 **종결** — γ=5·20 모두 무대 미성립, 재실험 계획 없음)

4셀 = {clean, noisy(answer-swap@0.7), gnoise(γ=1.0), frzero} × seed0 — observer+통제+
flirds P1-T1+T2, 심판 = GSM8K test 1,119 exact-match. 스펙·예측(H-8~11) =
`runs/track_h/README.md` §1.6. 드라이버·큐 = `$BATCH/runlogs/`(R4 4셀 → 1.2 순).
- 종료 후: `python runs/track_h/make_analysis.py`(gsm8k_em·delta_em·recovery_em) →
  **acc 갭 보고**(vanilla↔oracle_excl EM — answer-swap·gnoise서 수 pt 이상=무대 성립) +
  **R-플래토 확인**(R≤100 수렴 시 Tier B/C는 R=100) → GPU-h 보고 → H-8~11 대조 → rundir 커밋.
- **Tier B(+7점수원 P1, 전 8종 관찰자 재실행, ~300–350 GPU-h) = Yonghee 승인 게이트.**
- 금지: 게이트 하이퍼·GN_GAMMA(=1.0) 셀별 튜닝, poison, P2/P3/P4 arm(P1만).

#### 1.1-P5 — R4에 P5 정책 leg 추가 (Yonghee 07-21: "R4도 동일 적용"; **Tier A 종료 후 실행**)

> **[07-21 밤 스코프 확정 — Yonghee]** ① 정책은 **P1 sign-게이트 + P5-soft(pweight) 두 가지만**
> (P5-hard cgate/csign 전면 제외; `track_g.py`에 `T2_CSIGN=0` 스위치 신설, 기본 1=비트동일,
> tiny-gpt2 스모크 green — t2_pw만 생성 확인). T1(온라인)+T2(사후) × 오염축 전부.
> P1 몫은 Tier A가 이미 확보(gate_v2 online + t2_sign). csign의 UCB-보수성 분석(noisy 오염
> 11–13명 재포함)은 observer parquet 오프라인 선계산으로 확보됨 — GPU 불요.
> ② **gnoise 재주입**: γ=1.0 무대 불성립 확정(oracle 갭 −0.3pt) 후 Yonghee 지시로 dose 증강.
> 07-21 밤 유휴 GPU 3장에 **γ-probe {5,10,20}** observer 가동(23:29~, oracle_excl은 γ-무관
> [오염클라 전원배제 = 노이즈 미유입] → 0.3735 재사용). **r50 중간판독(gn_trend.py, 07-22
> 00:58): 상대-dose 자기감쇠 발견** — σ=γ·RMS(delta)가 수렴하며 delta와 함께 줄어 γ=5/10/20
> train_loss가 소수 3자리까지 동일 + clean 갭 r20 +0.017→r50 +0.002 소멸 중. 어떤 γ든 종반
> 무해 예상 → **`GN_ABS` 모드 확정**(Yonghee 07-22 01시: "클라 간 noise 크기 동일해야" —
> **런의 첫 오염 업데이트에서 σ 한 번 동결, 전 오염클라·전 라운드 공통 적용** = CNN
> 고정-σ(0.1, FedIF main) 관례와 정합; llm_server `_add_gnoise` shared frozen dict +
> track_g `GN_ABS` env, 기본 0 = 기존 비트동일; 단위 로직검증 + tiny-gpt2 스모크 green,
> config.yaml에 gn_abs 기록). 상대-dose probe 3개는 r~60서 kill(자기감쇠 근거는 r50 추이로
> 문서화, EM 확증은 미완 — caveat 유지), **abs-probe {5,10,20} 01:23 재기동**(observer만,
> root `rundirs_llm_gnabs{5,10,20}`). **probe는 r32에서 중단**(Yonghee 07-22 02시: 교체
> 우선, "지금껄로 판단") — **γ*=5 확정**: 근거 = 3점(5/10/20) train_loss 소수3자리 동일
> 32라운드(조기피해 γ-포화 → γ↑는 종반 붕괴위험만↑) + "적절히 망가짐"(밴드 vanilla EM
> 0.29~0.34 = oracle −3~−8pt)엔 최소 γ + CNN 중강도 고정σ 정합. `gn_full` 활성(γ=5).
> **체크포인트**: observer가 첫 arm·arm별 영속 → 셀 ~5h 시점 vanilla@γ5 EM을 밴드와 대조,
> 이탈 시 셀 킬 후 γ 조정 재기동(손실 상한 ~5h). EM-미확증 caveat: 자기감쇠는 rel r50
> 추이로만, abs 유효성은 gn_full observer가 첫 실측. γ=1.0 rundir 보존. §1.1 "GN_GAMMA 튜닝
> 금지"는 사전등록 게이트 FAIL 후 무대-수리로 해제(Yonghee) — dose 선택 과정 전체를 보고.
> ③ 비용 ≈ **80 GPU-h**(gn_full ~29 + soft 6런 ~50) + probe 15.6(별도, 유휴 GPU 소진).
> ④ RUN_P5.md hard-측 예측(HP-1·2·4)은 LLM leg N/A 처리(CNN 본실험이 별도 서버서 커버).

(이하 07-21 낮 원계획 — 스코프는 위 블록이 우선)

배선 완료(07-21 로컬 커밋: track_g.py에 `<src>_cgate`[P5-hard 신뢰게이트]/
`<src>_pweight`[P5-soft Φ(t)-가중] arm + `T2_P5=1` → `t2_csign_*`/`t2_pw_*` 재학습 +
`T2_LEGACY=0` = Tier A의 t2_sign 중복 재실행 스킵; 테스트 10 + tiny-gpt2 R4형 스모크
green). **스펙·공정성 조항(z=1.645 고정, 학습-중 관측 통계만·사전 정보 금지)·예측 =
`runs/track_h/p5/RUN_P5.md` §2–4** — 위반 금지, MISS 그대로 보고.

**§1.0 `queue_postswap.txt`에 등재 완료(β보다 앞 순번) — 교체 후 자동 실행.** 수동 스펙(참고):
셀당 2 프로세스, env 베이스는 Tier A와 동일:
```bash
# ① online P5 arm 2종 (arm별 기본 명명 -> gsm50k5_<t>_flirds_cgate_seed0 등; RUN_NAME 설정 금지)
REGIME=gsm50k5 THREAT=<t> SEED=0 ARMS=flirds_cgate,flirds_pweight \
  RUNDIR_ROOT=<repo>/runs/track_h/rundirs_llm CUDA_VISIBLE_DEVICES=<g> $PY -u experiments/track_g.py
# ② T2 P5 재학습 (observer 재실행은 결정론적 동일값 덮어쓰기 = 무해; RUN_NAME 설정 금지)
REGIME=gsm50k5 THREAT=<t> SEED=0 ARMS=observer T2=1 T2_P5=1 T2_LEGACY=0 \
  RUNDIR_ROOT=<repo>/runs/track_h/rundirs_llm CUDA_VISIBLE_DEVICES=<g> $PY -u experiments/track_g.py
```
- `<t>` ∈ {clean, noisy, gnoise, frzero} — 4셀 × 2프로세스 = 8런. 비용 ≈ 셀당
  (online 2 + observer 1 + 재학습 dedupe 후 소량) × R=200 — **Tier A 실측 GPU-h로
  산정해 착수 전 보고**(Tier A 셀당 실측의 대략 2~3배/셀 예상).
- 종료 후: `python runs/track_h/make_analysis.py` → **P1 vs P5h vs P5s EM 표**(vanilla/
  oracle_excl 앵커 포함) + RUN_P5.md §4 HP-1~6 대조 → rundir+analysis 커밋.
- CNN 쪽 P5 본실험은 **별도 Slurm 서버** 담당(이 컨테이너 아님) — `runs/track_h/p5/`
  sbatch 2종, 실행 정본 = RUN_P5.md.

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
lr·steps intervention 2차검증(무GPU 재분석) · 1B·CNN β-불변 canon 확인.
(E5 N=10 oracle 확장(seeds1·2·(a) 2¹⁰) = 미진행 확정, Yonghee 07-22 — 시간 제약.)

### 1.4b Track G CNN skew-축 확장 + fmnist + frrand (**즉시 실행 대상**; Slurm 서버, 2026-07-22)

컨테이너 48h 큐와 **독립**(yonsei Slurm `base_suma_rtx3090`, torch 2.11.0 env). 지시 = Yonghee
2026-07-22. 구현·스모크·사전등록 **완료(로컬 커밋, push 안 함)** — 남은 것은 제출·분석·문서.

- **그리드 90런/30셀** = ① cifar10×{shard(label만),qskew(size만)} ② fmnist×{iid,dir1}
  각각 × {clean, free_rider, **frrand**, grad_noise, label_flip@{0.15,0.35,0.70}} × 3-seed
  + ③ cifar10×{iid,dir1}×frrand 백필 6런. 기존 36런 rundir는 **read-only**.
- **스택 통일 재실행 36런 동반**(Yonghee 07-22 결정): cifar10 {iid,dir1} 12셀을 현 스택에서
  재실행해 `rundirs_cnn_restack/`에 착지 → 2×2 표 단일 스택화 + `RERUN_AFTER_REPRO_FIX` P1 해소.
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
  leg 설계 확정**(C2 무대 동결 궤적 × 9방법 vs (b)-perround oracle; (a) 포기·Ripple/Banzhaf
  제외·Fed-LOO 포함) — 구현 대기 + 1셀 파일럿 게이트. **종합 계획·교차검증 핸드오프 =
  루트 `CNN_CAMPAIGN_PLAN_2026-07-22.md`** (열린 질문 7건 포함).
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

1. **R4 Tier C 3-seed** — E6-②(LLM selection 본문) 확정 수치용. 순서상 Tier B 뒤,
   비용 ≈ (Tier A+B)×2 (`runs/track_h/README.md` §1.6). 셋 중 논문 의존성 최상위.
2. **3B silo5 robustness seeds 1·2**(마스터 P5) — overview §3.3.3·caveat 1(현 1-seed)
   해소용. §1.2의 β0.3 재실행 3B 4셀(seed0 재실행·라벨 통일)과는 별개.
3. **probe A축 seeds 1·2**(rank r32/64·st20/30 셀; lr격자·noise·std50k5-r16은 3-seed
   완료) — ablation A축 보강(선택; overview §4.2 "커진 φ의 cross-seed 실재" 확인).

## 2. 문서·부수분석 (무GPU)

1. **overview 반영**(`research-wiki/survey/flirds-experiment-results-overview.md`): E4 Fed-LOO·E5 N=10·
   E7 frdelta·AdamW 3-seed(−0.53±0.33)·probe seeds1-2·loss-heur runtime(96.6/100.1/100.2s)·device 학습시간.
2. **표1 Fed-LOO 재집계**: `python runs/track_d/make_fidelity.py`(root `rundirs_e4_fedloo` 인자 확인).
3. **tab:cost**(`paper/sections/results.tex`): loss-heur 170→~99s·device overhead%·E3 CNN cost·end-to-end/overhead% 2블록.
4. **paper-ko 마커 해소**: E2·E3·E4·E5·E7·E11 🔴TODO/🟣VERIFY + §3.7.4 AdamW 갱신.
5. **Track G 서술**: 잔여 = paper-ko §6.5 — silo5/iid5 3-seed + CNN 그리드 + std50k5-mixed
   **s0 1-seed 파일럿(동결; 1-seed caveat 필수)** 구성. 확정치 = V2w 불승격·frzero 회수 1.0·
   noisy 게이트 침묵·clean parity max|Δ|=0.00056. LLM 참여축 성능 주장은 R4가 담당.
6. **Track H 서술**: overview §3.2.6 반영 완료 — **정본 수치 = §3.2.6**(P1-T1 동률 .707·
   P1-T2 flirds .839 1위·renorm 붕괴는 FR 국한 −5.9~−6.6·GN은 renorm도 0.9+;
   07-19 커밋 메시지의 정정-전 수치 인용 금지). 잔여 = paper 반영. LLM 경쟁 무대 = R4.
7. **부수분석**: 3.1 loss-heur 정본화(CSV/rundir) · oracle noisy AUROC 0.604/0.660 불일치 확정 ·
   bootstrap CI(B=1000) · momentum 열화(0.73 vs 0.81) 정본 rundir 위치.
8. **NEW gnoise negative result 서술**: "CNN 표준 grad-noise 위협이 LLM LoRA에서는 무대 미성립"
   — 근거·수치·문헌대조 전부 `runs/track_h/gnoise_diag/README.md`에 정리됨(그대로 인용 가능).
   배치 후보 = paper §5 한계/negative 또는 부록 1문단. **인용 금지 항목 주의**(Krum σ=200,
   arXiv 2509.09097·2602.19926·2605.07961 = 검증 실패). H-10은 CNN 결과로만 서술.

## 3. Yonghee 결정 대기

- **push**: 로컬 커밋 다수 — push 여부/시점(07-22: Yonghee가 직접 push 예정).
- **R4 Tier B 진입**(1.1 seed0 보고 완료 — overview §3.2.7; 승인 대기).
