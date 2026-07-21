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

### 1.0 다음 세션 재개 절차 (컨테이너 재생성 07-22 02:2x; **새 컨테이너도 48h 제약**)

직전 컨테이너 마감 상태: Tier A 4/4 완주·커밋(4c40e30) + β 2셀(a0.1 noisy·frrand) 완주,
시스템 전체 유휴로 교체 = 진행분 손실 0. gnoise는 **신정의(GN_ABS 공통고정σ, γ*=5)로 재실험
확정** — 구정의(γ1.0 상대) rundir 7개 폐기, `gsm50k5_gnoise_oracle_excl_seed0`만 보존
(oracle은 오염클라 전원배제라 γ·모드 무관 = 신무대 oracle로 재사용).

**재개(컨테이너 생성 직후 즉시 — 48h 예산이 빠듯함):**

1. venv 확인: `$BATCH/venv/bin/python -c "import torch;print(torch.cuda.is_available())"`
   (깨졌으면 `$BATCH/tools/` 재구축 스크립트 + PROVENANCE.md; BATCH=…/flirds_batch).
2. `bash $BATCH/tools/launch_driver.sh` → `tail -f $BATCH/runlogs/_driver.log`로 4-GPU
   디스패치 확인. QUEUE = `queue_postswap.txt`(완비): **gn_full(신정의 gnoise, ~29h 최장
   셀) → P5-soft 6런(p5s_*; online=pweight, t2=T2_CSIGN=0) → β0.3 16셀**(이관 2 포함,
   3B silo5 4셀 맨 뒤). 구 큐 `queue_2026-07-20.txt` 재사용 금지(완료 줄 미주석).
3. **gn_full 체크포인트(의무)**: 시작 ~5h 시점 observer 영속 → vanilla@γ5 EM을 밴드
   **0.29~0.34**(oracle 0.3735 − 3~8pt)와 대조. 이탈 시 셀 킬 → GN_GAMMA 조정(미달↑/붕괴↓)
   → gn_full만 재기동(뒤 큐는 계속 돎).

**48h 검산(07-22 02시 산정)**: gn_full ~29 + P5-soft ~51(online 9.9 + t2 41: pw 가중이
소스별로 달라 dedupe 불가 전제) + β0.3 ~83(device100 10×4.2 + poison 2×4.5 + 3B 4×8)
= **~163 GPU-h → 4-GPU wall ~43–45h = 48h 내 가능(마진 3–5h)**. 전제 = 즉시 기동·무사고.
**spill 규칙**: 초과 위험 시 **3B silo5 4셀(~32 GPU-h, 큐 맨 뒤)을 다음 컨테이너로 이월**
(캠페인상 독립; 자르면 wall ~35h로 여유). gn_full 재도스 발생 시(+5h) 3B 이월을 조기 결정.

**실험 완료 후 후처리(순서대로)**: ①P5-soft 완료 시 `make_analysis` → P1 vs P5s EM 표
(vanilla/oracle 앵커) + RUN_P5.md §4 HP 대조(LLM 몫: HP-3·5·6 재판정, hard-측 HP-1·2·4는
N/A) → rundir+분석 커밋 ②gn_full 완료 시 H-10 재판정(신무대) ③β 완주 시 rundir 커밋 +
overview §3.4 phase2 ShapleyFL 행 갱신 ④push는 Yonghee 결정 대기.

### 1.1 R4 Tier A — gsm50k5 accuracy 파일럿 seed0 (**완주 07-22 01:31, 커밋 4c40e30**;
gnoise만 구정의 폐기→§1.0 신정의 재실험)

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

### 1.2 β0.3 재실행 잔여 (device100 + 3B silo5) — P5 뒤 큐 자동 재개

진행: 완주 1(a0.1_noisy) + 이번 세션 완주 예정 1(a0.1_frrand; §1.0 IFKILLED 체크) +
`queue_postswap.txt` 등재 16(이관 a0.1_frzero·a0.01_noisy 포함; P5 다음 순번).
드라이버 유실 시 수동 재개:
```bash
sed -i 's/^#phase2/phase2/' runs/rerun_beta03/logs/resume36h.txt   # 유실 시 RESUME_AFTER_MIGRATION.md 31줄에서
                                                                   # 완료분 1B_silo5 4셀 제외하고 재생성
PY=$PY PP=<repo>/codes HOME=… HF_HOME=… HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  QUEUE=<abs>/runs/rerun_beta03/logs/resume36h.txt GPUS_FILE=<abs>/logs/gpus36h.txt GPUS="<g...>" \
  LOGDIR=<abs>/runs/rerun_beta03/logs bash runs/rerun_beta03/run_multi_driver.sh
```
완료 후: 18셀 rundir 커밋 + overview §3.4 phase2 ShapleyFL 행 갱신.

### 1.3 β0.3 deferred 9셀 (최중량 꼬리; 별도 캠페인)
7B_std20×3(70–90h) + device100-a0.5 anchor×3(63h) + 7B_anchor5×3(35–45h) — `RESUME_AFTER_MIGRATION.md`.
완료 후 overview 7B 열(§3.1.1·§3.5.1) 갱신.

### 1.4 장기 대기 (우선순위 낮음)
lr·steps intervention 2차검증 · 1B·CNN β-불변 canon 확인 · probe A축 seeds 1-2.
(E5 N=10 oracle 확장(seeds1·2·(a) 2¹⁰) = 미진행 확정, Yonghee 07-22 — 시간 제약.)

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

## 3. Yonghee 결정 대기

- **push**: 로컬 커밋 다수 — push 여부/시점(07-22: Yonghee가 직접 push 예정).
- **R4 Tier B 진입**(1.1 seed0 보고 완료 — overview §3.2.7; 승인 대기).
