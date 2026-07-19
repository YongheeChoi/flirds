# Track H — GPU 서버 실행 인수인계 (2026-07-19)

> **스펙 정본** = `runs/track_h/README.md` (질문·축·예측표 H-1~7·판정·비용·§1.5 재사용 규칙).
> 이 문서 = 실행 절차만. 코드는 구현·단위테스트 green·로컬 CNN 스모크까지 완료 상태로 커밋됨.
> 우열 기준은 **학습 성능(recovery)만** — 탐지 AUROC류는 판정에 쓰지 않는다(spec §3).

## 0. 코드 상태 (이 커밋에 포함)

| 파일 | 내용 |
|---|---|
| `codes/flirds/fl/score_providers.py` | **신규** — 8 점수원 per-round provider(공통 시그니처, contribution orientation; GTG/FedSV canonical exact 설정; **ComFedSV = per-round 대용치**(completion 생략, Yonghee 07-19 결정 — README §1 caveat) |
| `codes/experiments/track_g.py` | `raw_by_arm`에 flirds1st/gtg/fedsv/comfedsv 추가(기존 arm 무변경) → `<src>_gate_v2`/`_gatew_v2`/`_zgate_v2` 전 점수원 동작 |
| `codes/experiments/track_c2.py` | `<src>_<policy>` 경쟁 arm 파싱(P1 gate_v2/P2 gatew_v2/P3 mult/P4 zgate_v2) + **`observer`** arm(전 점수원 동시 채점, phi_rounds에 `method` 열) + **`C2_T2=1`** retrain 레그(t2_sign/t2_signw/t2_random, kept-set dedupe). 레거시 arm 분기 무변경(additive-only) |
| `codes/tests/test_track_h.py` | 7 테스트(부호 방향·MC 결정론·파싱·T2 dedupe·static weights) — green |
| `runs/track_h/make_analysis.py` | rundir-only 집계(신규+§1.5 재사용 루트 병합) → `analysis/{llm,cnn}_competition.csv`+`competition_score.csv`+`observer_zero_semantics.csv` — 재사용 rundir만으로 검증 완료(LLM 214·CNN 204행) |

## 1. 서버 사전 확인 (티어 착수 전, 1회)

```bash
cd <REPO>/codes
# ① 단위테스트 (신규 + 회귀)
PYTHONPATH=. python tests/test_track_h.py     # 7/7
PYTHONPATH=. python tests/test_signgate.py    # 15/15
# ② LLM 오프라인 배선 스모크 (다운로드 없음, ~2분)
SMOKE_MODEL=tiny-gpt2 SYNTH_DATA=1 REGIME=silo5 THREAT=frzero ROUNDS=5 MAX_STEPS=2 \
  BURN_IN=2 SEED=0 PERSIST=0 \
  ARMS=vanilla,oracle_excl,gtg_gate_v2,fedsv_gate_v2,comfedsv_gate_v2,flirds1st_gate_v2 \
  PYTHONPATH=. python -u experiments/track_g.py
# ③ CNN 스모크 (관찰자+T2 배선; cifar10 필요, ~수분)
C2_DATASET=cifar10 C2_PARTITION=dir1 C2_THREAT=free_rider C2_SEED=0 C2_MODE=smoke \
  C2_FRAC=0.2 C2_BURN_IN=2 C2_T2=1 C2_PERSIST=0 \
  C2_ARMS=vanilla,observer,gtg_gate_v2,comfedsv_mult \
  PYTHONPATH=. python -u experiments/track_c2.py
```

기대: 테스트 all-pass · ②에서 frzero 클라(1)가 gtg/fedsv/comfedsv 게이트에선 **배제 안 될 수도 있음**(renorm≠exact-0 — 그게 측정 대상) · ③에서 t2_* arm 출력 + phi_rounds `method` 열.

## 2. Tier 1 — CNN R1 (cifar10 dir1, 4 threat × 3 seed; ~50–80 GPU-h)

셀 = `dir1 × {clean, grad_noise, free_rider, label_flip@fr0.70}` × seed{0,1,2}.
**재사용(재실행 금지)**: vanilla/oracle_excl/random_excl + Flirds 4정책 = `runs/track_g/rundirs_cnn`에 이미 있음(§1.5). label-flip은 반드시 `C2_FLIP_RATE=0.70`(track_g 셀과 동일 dose).

**프로세스 A — 관찰자+T2** (셀·seed당 1회; T2 arm ~17개 내장):
```bash
C2_DATASET=cifar10 C2_PARTITION=dir1 C2_THREAT=grad_noise C2_SEED=0 C2_MODE=full \
  C2_T2=1 C2_ARMS=observer \
  C2_RUN_ROOT=<REPO>/runs/track_h/rundirs_cnn \
  C2_RUN_NAME=cifar10_dir1_grad-noise_obs_seed0 \
  PYTHONPATH=. python -u experiments/track_c2.py
# label_flip 셀은 + C2_FLIP_RATE=0.70, RUN_NAME에 _fr0.70 표기
```

**프로세스 B — T1 경쟁 arm** (점수원당 1프로세스 = 4 arm; GPU 샤딩 자유):
```bash
for SRC in flirds1st lossheur gtg fedsv comfedsv shapleyfl fedif; do
  C2_DATASET=cifar10 C2_PARTITION=dir1 C2_THREAT=grad_noise C2_SEED=0 C2_MODE=full \
    C2_ARMS=${SRC}_gate_v2,${SRC}_gatew_v2,${SRC}_mult,${SRC}_zgate_v2 \
    C2_RUN_ROOT=<REPO>/runs/track_h/rundirs_cnn \
    C2_RUN_NAME=cifar10_dir1_grad-noise_${SRC}_seed0 \
    PYTHONPATH=. python -u experiments/track_c2.py
done
```

⚠ **`C2_RUN_NAME`은 프로세스마다 반드시 유니크**(RunLogger가 같은 이름 디렉토리에 덮어씀).
⚠ clean 셀도 T1/T2 전부 실행(무해성 축) — oracle/random_excl만 자동 생략됨.
비용 유의: 코호트 10에서 gtg/fedsv/comfedsv/shapleyfl은 라운드당 최대 2^10 val-forward(기존 C2 shapleyfl arm과 동급). 관찰자 프로세스는 8 점수원 동시라 단일 arm보다 무겁다.

## 3. Tier 2 — LLM R3 (silo5 noisy nr1.0, 3 seed; 신규 4 arm × 3 = 12 run, ~24 GPU-h)

재사용: flirds/lossheur/oracleb 게이트·zgate·flirds_w·통제·v3 = `runs/track_g/rundirs`.
```bash
for S in 0 1 2; do
  REGIME=silo5 THREAT=noisy SEED=$S \
    ARMS=gtg_gate_v2,fedsv_gate_v2,comfedsv_gate_v2,shapleyfl_gate_v2 \
    RUNDIR_ROOT=<REPO>/runs/track_h/rundirs_llm \
    PYTHONPATH=. python -u experiments/track_g.py
done
```
예측 H-4: GTG/FedSV 게이트는 **발화할 것**(renorm 0-교차) — clean 오배제 동반 여부와 최종 val-loss(vanilla 2.3340±.047 / oracle_excl 2.3323±.047 기준)가 판정. 효과 크기 작음(사전 명시됨).

## 4. Tier 3 — LLM R2 (std50k5 mixed seed0; ~12–13 run × ~4.4 GPU-h ≈ 53–57 GPU-h)

**착수 전 필수 게이트(§1.5)**: 원격 track_g 진행분과의 교차 확인 —
```bash
ls <REPO>/runs/track_g/rundirs | grep std50k5     # shapleyfl_gate_v2·seeds1-2 신착?
ls /NHNHOME/WORKSPACE/26msit001_A/flirds_batch/state/ 2>/dev/null | grep -i std50
```
신착 완료분(특히 `std50k5_mixed_shapleyfl_gate_v2_seed0`)은 **재사용 귀속·재실행 금지**; 진행 중이면 해당 arm 대기.
```bash
ARMS=flirds_gatew_v2,flirds1st_gate_v2,flirds1st_gatew_v2,lossheur_gatew_v2,\
gtg_gate_v2,gtg_gatew_v2,fedsv_gate_v2,fedsv_gatew_v2,comfedsv_gate_v2,comfedsv_gatew_v2,\
shapleyfl_gate_v2,shapleyfl_gatew_v2
REGIME=std50k5 THREAT=mixed SEED=0 ARMS=$ARMS \
  RUNDIR_ROOT=<REPO>/runs/track_h/rundirs_llm \
  PYTHONPATH=. python -u experiments/track_g.py     # arm별 프로세스 분할 가능(ARMS 쪼개기)
```
(lossheur_gate_v2@std50k5는 track_g DEFAULT에 있었으나 rundir 미영속 — 목록에 포함해 실행.)
seed0 결과·GPU-h 보고 → **3-seed 확장은 Yonghee 승인 게이트**.

## 5. 티어 종료마다 (보고 프로토콜)

1. `python runs/track_h/make_analysis.py` → `analysis/competition_score.csv` 갱신
   (재사용 루트 자동 병합; 실행 중 셀은 그냥 안 보임).
2. GPU-h 실측 보고(rundir `metrics.json`의 `train_s` 합 / CNN은 wall-clock) — 다음 티어는 Yonghee 게이트.
3. rundir + analysis 커밋(결과 커밋 관례; 코드 수정 없음 확인).
4. 예측표 H-1~7 대조 결과를 MISS 포함 그대로 보고(spec §6).

## 6. 금지·주의 (spec §6 요약)

- **poison 무대 전면 제외 · Banzhaf 제외**(Yonghee 07-19).
- 게이트 하이퍼 셀별 튜닝 금지(τ=0·c=1.5·α=1·burn-in CNN 10/silo 3 고정).
- ShapleyFL β=0.3 · 재사용 arm 재실행 금지(§1.5 표) · rundir 루트는 반드시 `runs/track_h/`(track_g 루트에 쓰지 말 것).
- 대기열 충돌: β0.3 deferred 9셀(7B·device100)과 GPU 배분만 조율(무대 불교차).
