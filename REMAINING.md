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

### 1.5 장기 대기 (우선순위 낮음)
E5 seed1·2(2¹⁰, 33h/셀) · lr·steps intervention 2차검증 · 1B·CNN β-불변 canon 확인 · probe A축 seeds 1-2.

## 2. 문서·부수분석 (무GPU)

1. **overview 반영**(`research-wiki/survey/flirds-experiment-results-overview.md`): E4 Fed-LOO·E5 N=10·
   E7 frdelta·AdamW 3-seed(−0.53±0.33)·probe seeds1-2·loss-heur runtime(96.6/100.1/100.2s)·device 학습시간.
2. **표1 Fed-LOO 재집계**: `python runs/track_d/make_fidelity.py`(root `rundirs_e4_fedloo` 인자 확인).
3. **tab:cost**(`paper/sections/results.tex`): loss-heur 170→~99s·device overhead%·E3 CNN cost·end-to-end/overhead% 2블록.
4. **paper-ko 마커 해소**: E2·E3·E4·E5·E7·E11 🔴TODO/🟣VERIFY + §3.7.4 AdamW 갱신.
5. **Track G 서술**: overview 신규 절 + paper-ko §6.5 (1.2 완주 후 analysis 최종본 인용; 현재 확정치 =
   V2w 불승격·frzero 회수 1.0·noisy 게이트 침묵·clean parity max|Δ|=0.00056).
6. **Track H 서술**: Tier1+2 확정 스토리(절대-0점 vs 상대-0점의 위협-의존 트레이드오프: CNN corrupt서
   renorm −2.4~−3.0 붕괴 ↔ LLM noisy서 renorm 회수 1.66–1.91 + 오배제 리스크; exact-0 계열 게이트
   회수 0.4–0.95) — Tier3 후 overview·paper 반영.
7. **부수분석**: 3.1 loss-heur 정본화(CSV/rundir) · oracle noisy AUROC 0.604/0.660 불일치 확정 ·
   bootstrap CI(B=1000) · momentum 열화(0.73 vs 0.81) 정본 rundir 위치.

## 3. Yonghee 결정 대기

- **push**: 로컬 커밋 다수(69cb6bf~95fca5c + 이 커밋) — push 여부/시점.
- **Track H Tier3 3-seed** (1.1 seed0 보고 후) · **Track G std50k5는 3-seed 이미 승인·잔여만**(1.2).
- E5 N=10 3-seed 여부(1.5).
