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

### 1.1 R4 Tier A — gsm50k5 accuracy 파일럿 seed0 (**실행 중** 2026-07-20 23:29~, 4-GPU)

4셀 = {clean, noisy(answer-swap@0.7), gnoise(γ=1.0), frzero} × seed0 — observer+통제+
flirds P1-T1+T2, 심판 = GSM8K test 1,119 exact-match. 스펙·예측(H-8~11) =
`runs/track_h/README.md` §1.6. 드라이버·큐 = `$BATCH/runlogs/`(R4 4셀 → 1.2 순).
- 종료 후: `python runs/track_h/make_analysis.py`(gsm8k_em·delta_em·recovery_em) →
  **acc 갭 보고**(vanilla↔oracle_excl EM — answer-swap·gnoise서 수 pt 이상=무대 성립) +
  **R-플래토 확인**(R≤100 수렴 시 Tier B/C는 R=100) → GPU-h 보고 → H-8~11 대조 → rundir 커밋.
- **Tier B(+7점수원 P1, 전 8종 관찰자 재실행, ~300–350 GPU-h) = Yonghee 승인 게이트.**
- 금지: 게이트 하이퍼·GN_GAMMA(=1.0) 셀별 튜닝, poison, P2/P3/P4 arm(P1만).

#### 1.1-P5 — R4에 P5 정책 leg 추가 (Yonghee 07-21: "R4도 동일 적용"; **Tier A 종료 후 실행**)

배선 완료(07-21 로컬 커밋: track_g.py에 `<src>_cgate`[P5-hard 신뢰게이트]/
`<src>_pweight`[P5-soft Φ(t)-가중] arm + `T2_P5=1` → `t2_csign_*`/`t2_pw_*` 재학습 +
`T2_LEGACY=0` = Tier A의 t2_sign 중복 재실행 스킵; 테스트 10 + tiny-gpt2 R4형 스모크
green). **스펙·공정성 조항(z=1.645 고정, 학습-중 관측 통계만·사전 정보 금지)·예측 =
`runs/track_h/p5/RUN_P5.md` §2–4** — 위반 금지, MISS 그대로 보고.

Tier A 4셀 완주 확인 후, 셀당 2 프로세스 (env 베이스는 Tier A와 동일):
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

### 1.2 β0.3 재실행 잔여 18셀 (device100 14 + 3B silo5 4) — R4 뒤 큐 자동 재개

현행 드라이버 큐에 등재되어 있음(R4 다음 순번). 드라이버 유실 시 수동 재개:
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
E5 seed1·2(2¹⁰, 33h/셀) · lr·steps intervention 2차검증 · 1B·CNN β-불변 canon 확인 · probe A축 seeds 1-2.

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

- **push**: 로컬 커밋 다수 — push 여부/시점.
- **R4 Tier B 진입**(1.1 seed0 보고 후).
- E5 N=10 3-seed 여부(1.4).
