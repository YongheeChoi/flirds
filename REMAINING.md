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

> **폐기 기록(2026-07-20 밤, Yonghee)**: LLM 성능 심판을 val-loss→**downstream accuracy**로
> 전환함에 따라 ①구 1.1 Track H Tier3(std50k5 mixed 12런; 실행 중 4런 kill·rundir 무생성)
> ②구 1.2 Track G std50k5 잔여 7셀 **폐기** — std50k5-mixed는 s0 1-seed 파일럿으로 동결
> (overview §3.2.3–4 반영분 유지). LLM 참여축 성능 주장은 R4(gsm50k5·EM)가 승계.
> **CNN 축은 폐기 대상 아님**(심판이 원래 test-acc; Tier1 96런·track_g CNN 그리드 전부 유효) — 1.2 참조.

### 1.1 R4 Tier A — gsm50k5 accuracy 파일럿 seed0 (**최우선 재개 07-20 23:2x**)

> **정정(Yonghee 07-20 23:2x)**: "β0.3 말고 방금 가져온 실험 먼저" — R4(gsm50k5)는
> 새 방법 라인의 실험로서 **큐 최우선 복귀**(β0.3 앞). 07-20 심야의 "Track H·G 보류"는
> **legacy 축**(H = 점수원 경쟁 잔여·CNN 확장, G = 게이트 트랙)에만 적용되는 것으로 정정.
> 실행: 현행 β0.3 4셀이 끝나는 셀 경계부터 R4 4셀이 GPU를 이어받음(무손실 전환).

본런 4셀(clean/noisy/frzero/gnoise; observer+통제+flirds P1-T1+T2, seed0).
스펙·예측(H-8~11)=`runs/track_h/README.md` §1.6.
- 종료 후: `python runs/track_h/make_analysis.py`(gsm8k_em·delta_em·recovery_em) →
  **acc 갭 보고**(vanilla↔oracle_excl EM — answer-swap·gnoise서 수 pt 이상=무대 성립) +
  **R-플래토 확인**(R≤100 수렴 시 Tier B/C는 R=100) → GPU-h 보고 → H-8~11 대조 → rundir 커밋.
- **Tier B(+7점수원 P1, 전 8종 관찰자 재실행, ~300–350 GPU-h) = Yonghee 승인 게이트.**
- 금지: 게이트 하이퍼·GN_GAMMA(=1.0) 셀별 튜닝, poison, P2/P3/P4 arm(P1만).

### 1.2 Track H CNN 확장 — R1 잔여 경쟁 (**보류 07-20 심야**; ~30–45 GPU-h 추정)

> **보류(1.1과 동일 사유)** — 단 CNN 결과·rundir는 전부 유효 보존(폐기 아님), 새 방법
> 확정 후 스코프(P2–P4 범위·T2×P2·iid 파티션) 재논의.

후보 잔여 = Tier1 완주분(96런 = P1-T1×비Flirds 7종 + 관찰자) 대비 티어표(§4)의
**T1×{P2,P3,P4}×S7(252런) + T2×P2×S8(dedupe 실효↓)**.
- ~~선행 블로커: track_c2 행(hang)~~ → **원인 확정·수정(07-20 심야)**: 코드 무결 —
  cs.toronto.edu CIFAR-10 다운로드 스트림이 ~20MiB서 stall(부분 tar 잔존→매 실행 재시도).
  faulthandler 스택으로 확정(`$BATCH/runlogs/cnn_hang_probe.log`), 데이터 파일 복구로 해결.

### 1.3 β0.3 재실행 잔여 18셀 (device100 14 + 3B silo5 4) — **진행 중**(07-20 22:34~, 4-GPU 드라이버)
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
5. **Track G 서술**: overview는 §3.2.3–4 반영됨; 잔여 = paper-ko §6.5 —
   **std50k5-mixed는 s0 1-seed 파일럿 동결(폐기 07-20)이므로 1-seed caveat 명기**하고
   silo5/iid5 3-seed + CNN 그리드 + s0 파일럿으로 재구성(확정치 = V2w 불승격·frzero 회수 1.0·
   noisy 게이트 침묵·clean parity max|Δ|=0.00056; LLM 참여축 성능은 R4가 승계).
6. **Track H 서술**: **overview §3.2.6 반영 완료(07-20 로컬)** — ⚠ make_analysis 집계 정정 포함
   (lf-dose join 실패·equals_vanilla 결측 → dir1 공통 9셀 재집계; 커밋 메시지의 "lossheur .849 >
   flirds .762"/"fedif=flirds1st 1.17 T2 최고"는 정정 전 수치, 정본 = §3.2.6: P1-T1 동률 .707·
   P1-T2 flirds .839 1위·renorm 붕괴는 FR 국한 −5.9~−6.6·GN은 renorm도 0.9+). 잔여 = paper 반영.
   **Tier3(R2 std50k5) 폐기(07-20)** — §3.2.6 R2 확정-갱신 항목 소멸, LLM 경쟁 무대는 R4로 대체.
7. **부수분석**: 3.1 loss-heur 정본화(CSV/rundir) · oracle noisy AUROC 0.604/0.660 불일치 확정 ·
   bootstrap CI(B=1000) · momentum 열화(0.73 vs 0.81) 정본 rundir 위치.

## 3. Yonghee 결정 대기

- **push**: 로컬 커밋 다수 — push 여부/시점.
- **legacy Track H·G 재개 여부** — 새 방법 확정 후: CNN 확장 스코프(1.2)·G 후속. (R4는 1.1로 재개됨)
- E5 N=10 3-seed 여부(1.5).
