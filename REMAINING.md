# REMAINING — 남은 작업 (상시 현행; 완료 항목은 지우고 git 히스토리로만 남김)

> 갱신 2026-07-20 09:3x (F-세션 컷 시점). 완료 기록은 커밋 메시지·git 히스토리
> (`REMAINING_after_e_session_2026-07-19.md` 이력 포함) 참조. 파일-canon: rundir → overview → paper.

## 1. 실험 (GPU; 새 컨테이너) — 권장 순서대로

- **환경(컨테이너 공통; 2026-07-20 새 컨테이너에 재구축)**:
  `BATCH=/NHNHOME/26msit001_A/BASE/edge_ai_lab/yonghee/flirds_batch` 기준
  `PY=$BATCH/venv/bin/python`, `HOME=$BATCH/home`(~/data=CNN 데이터),
  `HF_HOME=$BATCH/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`, `codes/`에서 `PYTHONPATH=.`.
  현 컨테이너=B200 4장(0–3). venv는 기존 rundir meta.json과 **동일 버전 고정**(torch 2.12.0+cu130,
  transformers 5.9.0, trl 1.5.1, peft 0.19.1, accelerate 1.13.0, datasets 4.8.5, numpy 2.4.6).
  meta-llama gated 재취득 불가(토큰 무) → 해시 교차검증된 공개 미러로 캐시 재구성 —
  검증 체인·근거는 `$BATCH/PROVENANCE.md`.

### 1.1 Track H Tier 3 — std50k5 mixed seed0 12런 (~40–60 GPU-h)
```bash
for ARM in flirds_gatew_v2 flirds1st_gate_v2 flirds1st_gatew_v2 lossheur_gate_v2 lossheur_gatew_v2 \
           gtg_gate_v2 gtg_gatew_v2 fedsv_gate_v2 fedsv_gatew_v2 comfedsv_gate_v2 comfedsv_gatew_v2 \
           shapleyfl_gatew_v2; do
  REGIME=std50k5 THREAT=mixed SEED=0 ARMS=$ARM V3=0 \
    RUNDIR_ROOT=<repo>/runs/track_h/rundirs_llm CUDA_VISIBLE_DEVICES=<g> \
    $PY -u experiments/track_g.py    # arm당 ~3.1–9.9 GPU-h (std50k5 seed0 실측 준용)
done
```
- `shapleyfl_gate_v2` seed0은 **재실행 금지**(track_g 완주분 재사용 — make_analysis가 자동 병합).
- 종료 후: `python runs/track_h/make_analysis.py` → GPU-h 보고 → **3-seed 확장 = Yonghee 승인 게이트**.
- 보고 프로토콜(스펙 §5·§6): competition_score.csv + H-1~7 대조(MISS 포함) + rundir/analysis 커밋.

### 1.2 Track G — std50k5 3-seed 잔여 7셀 (~30 GPU-h)
s1 {vanilla, oracle_excl, random_excl} + s2 {flirds_gate_v2, vanilla, oracle_excl, random_excl}:
```bash
REGIME=std50k5 THREAT=mixed SEED=<s> ARMS=<arm> V3=0 CUDA_VISIBLE_DEVICES=<g> \
  $PY -u experiments/track_g.py     # rundir 루트 기본값=runs/track_g/rundirs (single-arm 이름 규약 동일)
```
완주 후: `python runs/track_g/make_analysis.py` 최종 재생성 → rundir+analysis 커밋.

### 1.3 β0.3 재실행 잔여 18셀 (device100 14 + 3B silo5 4)
```bash
sed -i 's/^#phase2/phase2/' runs/rerun_beta03/logs/resume36h.txt   # 유실 시 RESUME_AFTER_MIGRATION.md 31줄에서
                                                                   # 완료분 1B_silo5 4셀 제외하고 재생성
PY=$PY PP=<repo>/codes HOME=… HF_HOME=… HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  QUEUE=<abs>/runs/rerun_beta03/logs/resume36h.txt GPUS_FILE=<abs>/logs/gpus36h.txt GPUS="<g...>" \
  LOGDIR=<abs>/runs/rerun_beta03/logs bash runs/rerun_beta03/run_multi_driver.sh
```
완료 후: 18셀 rundir 커밋 + overview §3.4 phase2 ShapleyFL 행 갱신.

### 1.4 β0.3 deferred 9셀 (최중량 꼬리; 별도 캠페인)
7B_std20×3(70–90h) + device100-a0.5 anchor×3(63h) + 7B_anchor5×3(35–45h) — `RESUME_AFTER_MIGRATION.md`.
완료 후 overview 7B 열(§3.1.1·§3.5.1) 갱신.

### 1.5 Track H R4 Tier A — gsm50k5 accuracy 파일럿 seed0 (~85–110 GPU-h; 우선순위 = Yonghee 지정 대기, 1.1과 독립·병행 가능)

스펙·예측(H-8~11) = `runs/track_h/README.md` §1.6. 구현·스모크 = 로컬 3090에서 07-20 완료
(합성 tiny-gpt2 gnoise/frzero + 실 gpt2·실 GSM8K 마이크로런 green; 로더 불변식 검증
— 50×149 균등, val 200/test 1,119 전 분할 disjoint, swap dose 정확 104/149).

```bash
# pre-flight ①: GSM8K를 오프라인 캐시에 확보 (서버 HF_DATASETS_OFFLINE=1이므로 1회 선행)
HF_HOME=$BATCH/hf_home HF_DATASETS_OFFLINE=0 $PY -c \
  "from datasets import load_dataset; load_dataset('openai/gsm8k','main')"
# pre-flight ②: 스모크 (tiny-gpt2 합성 → 실 gpt2 마이크로; 둘 다 ~1분/GPU)
SMOKE_MODEL=tiny-gpt2 SYNTH_DATA=1 REGIME=gsm50k5 THREAT=frzero N_CLIENTS=8 K_ABS=4 \
  VAL=10 TEST=10 ROUNDS=5 MAX_STEPS=2 BURN_IN=2 CORRUPT_IDS=0,1,2 T2=1 \
  ARMS=observer,flirds_gate_v2 PERSIST=0 $PY -u experiments/track_g.py
SMOKE_MODEL=gpt2 REGIME=gsm50k5 THREAT=noisy N_CLIENTS=10 K_ABS=5 VAL=20 TEST=20 \
  ROUNDS=2 MAX_STEPS=2 BURN_IN=1 CORRUPT_IDS=0,1,2,3 T2=1 \
  ARMS=observer,flirds_gate_v2 PERSIST=0 $PY -u experiments/track_g.py
# 본런: 셀당 1프로세스 (observer=vanilla 겸용 + 통제 + flirds P1-T1; T2=1이 관찰자
# 누적으로 t2_sign_{flirds,flirds1st,lossheur,fedif}+매치드 random 재학습을 자동 수행)
for THREAT in clean noisy frzero gnoise; do   # noisy=answer-swap@0.7(기본 dose)
  REGIME=gsm50k5 THREAT=$THREAT SEED=0 T2=1 \
    ARMS=observer,oracle_excl,random_excl,flirds_gate_v2 \
    RUNDIR_ROOT=<repo>/runs/track_h/rundirs_llm CUDA_VISIBLE_DEVICES=<g> \
    $PY -u experiments/track_g.py   # corrupt 셀 ~24 GPU-h(arm 4 + T2 재학습 2~5) / clean ~9–13
done
```
- 종료 후: `python runs/track_h/make_analysis.py`(gsm8k_em·delta_em·recovery_em 열 자동;
  observer=vanilla 앵커 매핑됨) → **acc 갭 보고**(vanilla↔oracle_excl EM — answer-swap·gnoise서
  수 pt 이상이어야 경쟁 무대 성립) + **R-플래토 확인**(val-curve R≤100 수렴 시 Tier B/C는
  R=100 = 비용 반감, 스펙 사전등록 룰) → GPU-h 보고 → H-8~11 대조(MISS 포함) → rundir 커밋.
- **Tier B(경쟁: +7점수원 P1, 전 8종 관찰자 재실행 OBS_SOURCES=전체, ~300–350 GPU-h) = Yonghee 승인 게이트.**
- 금지: 게이트 하이퍼·GN_GAMMA(=1.0 고정) 셀별 튜닝, poison, P2/P3/P4 arm(Yonghee 07-20: P1만).

### 1.6 장기 대기 (우선순위 낮음)
E5 seed1·2(2¹⁰, 33h/셀) · lr·steps intervention 2차검증 · 1B·CNN β-불변 canon 확인 · probe A축 seeds 1-2.

## 2. 문서·부수분석 (무GPU)

1. **overview 반영**(`research-wiki/survey/flirds-experiment-results-overview.md`): E4 Fed-LOO·E5 N=10·
   E7 frdelta·AdamW 3-seed(−0.53±0.33)·probe seeds1-2·loss-heur runtime(96.6/100.1/100.2s)·device 학습시간.
2. **표1 Fed-LOO 재집계**: `python runs/track_d/make_fidelity.py`(root `rundirs_e4_fedloo` 인자 확인).
3. **tab:cost**(`paper/sections/results.tex`): loss-heur 170→~99s·device overhead%·E3 CNN cost·end-to-end/overhead% 2블록.
4. **paper-ko 마커 해소**: E2·E3·E4·E5·E7·E11 🔴TODO/🟣VERIFY + §3.7.4 AdamW 갱신.
5. **Track G 서술**: overview는 §3.2.3–4 반영됨(std50k5 진행분 07-20 로컬 포함); 잔여 = paper-ko §6.5
   (1.2 완주 후 analysis 최종본 인용; 현재 확정치 = V2w 불승격·frzero 회수 1.0·noisy 게이트 침묵·
   clean parity max|Δ|=0.00056).
6. **Track H 서술**: **overview §3.2.6 반영 완료(07-20 로컬)** — ⚠ make_analysis 집계 정정 포함
   (lf-dose join 실패·equals_vanilla 결측 → dir1 공통 9셀 재집계; 커밋 메시지의 "lossheur .849 >
   flirds .762"/"fedif=flirds1st 1.17 T2 최고"는 정정 전 수치, 정본 = §3.2.6: P1-T1 동률 .707·
   P1-T2 flirds .839 1위·renorm 붕괴는 FR 국한 −5.9~−6.6·GN은 renorm도 0.9+). 잔여 = paper 반영 +
   Tier3 완주 후 §3.2.6 R2 확정 갱신.
7. **부수분석**: 3.1 loss-heur 정본화(CSV/rundir) · oracle noisy AUROC 0.604/0.660 불일치 확정 ·
   bootstrap CI(B=1000) · momentum 열화(0.73 vs 0.81) 정본 rundir 위치.

## 3. Yonghee 결정 대기

- **push**: 로컬 커밋 다수(69cb6bf~95fca5c + 이 커밋) — push 여부/시점.
- **Track H Tier3 3-seed** (1.1 seed0 보고 후) · **Track G std50k5는 3-seed 이미 승인·잔여만**(1.2).
- **Track H R4 우선순위**(1.5 Tier A를 1.1보다 먼저 돌릴지) · Tier B 진입(1.5 보고 후).
- E5 N=10 3-seed 여부(1.6).
