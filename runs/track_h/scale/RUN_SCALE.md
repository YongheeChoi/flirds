# Scale 실험 — CNN 완전 참여 100/100 (Slurm 서버 실행 지시서)

> **이 문서 = Scale 실험(Slurm 서버)의 실행 정본.** B200 컨테이너와 무관(REMAINING.md
> 항목 아님). Slurm 세션은 §5 pre-flight → §6 sbatch 제출(파일럿 먼저) → §7 보고·커밋
> 순서로 진행. 예측(§4)은 사전 등록이며 **실행 후 수정 금지**(MISS 그대로 보고).
> 결정(Yonghee 2026-07-21): **P5 arm 포함, audit(대체-oracle 계측) 제외, 비교 대상 =
> vanilla 학습만**(oracle_excl/random_excl/T2/removal-curve 등 완전-참여가 아닌
> 훈련 비교군 전부 제외).

## 1. 배경 — 왜 완전 참여 100인가

Flirds의 핵심 비용 주장: valuation이 라운드 cohort 크기 k에 **선형**(k HVP + 1
val-grad). 반면 per-round exact (b)는 2^k, coalition-MC 계열은 O(k²) eval — k=100
완전 참여에서 우리 구현 기준:

| 방법 | 라운드당 비용 (k=100) | 판정 |
|---|---|---|
| ShapleyFL | `exact_shapley` = 2^k 전수 (`baselines/shapleyfl.py:91`) | 2^100 — 불가(종료 안 함) |
| GTG | ≥max(30,k)=100 순열 × ≤k evals (`baselines/gtg.py:103`) | ≥10⁴ val-fwd/라운드 — 사실상 불가 |
| FedSV·ComFedSV | max(30,2k)=200 순열 × k evals (`fl/score_providers.py`) | 2×10⁴ val-fwd/라운드 — 사실상 불가 |
| **flirds** | k HVP + 1 val-grad | **선형 — 이 실험의 주장** |

따라서 이 무대에는 baseline 개입 arm이 존재할 수 없고, **비교 대상은 vanilla 학습**
(= observer 궤적)이다. 기여도의 가치는 "같은 φ로 게이트한 훈련이 vanilla를 이기는가"
(오염 셀) + "clean에서 해가 없는가"(parity)로 판정한다.

## 2. 설계 — R1과 동일, 참여 frac만 0.1 → 1.0

**훈련/무대** (Track H CNN R1 = overview §3.2.6 스테이지와 동일; 변경점은 굵게):

| 항목 | 값 |
|---|---|
| 데이터/모델 | cifar10, FedSVCNN(width 1), dir1(Dirichlet α=1) |
| FL | N=100, **frac 1.0 (100/100 매 라운드)**, R=120, E=5, lr 0.01(SGD mom=0), batch 64 |
| 평가 | val 2000 / test 8000 (test-split seed 0 고정), 절대 test acc |
| 위협 | clean / label_flip(고정 dose 0.70) / free_rider(zero) / grad_noise(σ=0.1), MAL_FRAC=0.4 |
| seeds | 0, 1, 2 |
| 게이트 | burn_in=10, tau=0, min_obs=2, probation=5, conf_z=1.645 — **R1/P5와 동일, 셀별 튜닝 금지** |

**Arm 4종 (셀당 1 프로세스에 순차; 전부 100-완전-참여 훈련)**:

| arm | 역할 |
|---|---|
| `observer` | **= vanilla** (n-가중; 채점은 집계에 무개입 + cudnn_deterministic → 궤적이 vanilla와 결정론적으로 동일) + flirds φ per-round 영속 |
| `flirds_gate_v2` | P1 strict sign 게이트 (배제형; Track H P1 규약) |
| `flirds_cgate` | P5-hard 신뢰 게이트 (UCB ≤ 0일 때만 배제) |
| `flirds_pweight` | P5-soft w ∝ n·Φ(t) 확률 가중 |

- 정책 정의·엣지 규칙·공정성 조항(z=1.645 고정, 그 런의 학습-중 관측 통계만, 사전
  정보 금지)은 **`runs/track_h/p5/RUN_P5.md` §2–3이 정본** — 여기서도 그대로 적용.
- **observer 소스 = flirds만** (`C2_OBS_SRCS=flirds`): coalition 소스는 §1 표대로
  k=100에서 불가라 관찰자에 넣으면 job이 죽거나 며칠 점유한다. 이 env가 이번에
  추가된 유일한 코드 변경(기본값 = 8종 전부 = 레거시 비트동일).
- T2 없음(`C2_T2` 미설정), oracle_excl/random_excl 없음.

## 3. 산출물 (rundir당)

`config.yaml`(cfg.frac=1.0 + gate + track_h.obs_srcs 기록) + `meta.json` +
`metrics.json`(arm별 final_acc/acc_curve/AUROC/rounds_to_target + corrupt 마스크 +
observer_cum) + `phi_rounds.parquet`(observer: method=flirds 전 라운드×전 클라
raw/cum/n_obs; 게이트 arm: raw/cum/weight/participated).

## 4. 사전 등록 예측 (HS-1~5; MISS 그대로 보고)

- **HS-1** 오염 3셀(lf/fr/gn): 세 정책 arm 모두 vanilla를 수 pt 이상 상회.
  frzero는 exact-0 즉시 배제로 천장 근접(참여-희소였던 R1보다 빠른 회복 — 매 라운드
  전 FR이 관측되므로).
- **HS-2** clean 셀: P1은 경계선 분산 과금으로 vanilla 하회 위험. **P5h는 clean
  parity 회복** — 완전 참여는 n_obs≈R(≈110)의 증거-풍부 레짐이라 P5의 설계 의도가
  가장 잘 드러나야 하는 무대.
- **HS-3** 오염 셀에서 P5h → P1 수렴(같은 배제 집합): t = √n·(mean/σ̂)가 n과 함께
  커져 corrupt의 음수 증거가 조기 포화.
- **HS-4** P5s: clean parity + 오염 셀 회복(Φ(t) 양극화). 가중 정규화 때문에 전
  클라가 애매해도 학습 강도는 vanilla와 동일(균일 shrink는 약분됨).
- **HS-5** 비용: 셀(4-arm)당 wall-clock ≈ R1 셀의 ~10배 이내(클라 학습 10배가 지배,
  valuation은 선형 동반 증가). 파일럿 실측이 이 배수를 확정 — 논문 op-count 표의
  k=100 열과 함께 보고.

## 5. Pre-flight (서버에서 먼저; venv·데이터 = RUN_P5.md §5와 동일 서버 셋업 재사용)

```bash
cd <REPO>/codes
# 1) 테스트 (기존 + P5; 전부 green이어야 함)
PYTHONPATH=. <VENV_PY> -m pytest tests/test_p5.py tests/test_track_h.py -q
# 2) 완전-참여 e2e 스모크 (~2-3분, GPU 1장; fmnist 자동 다운로드 OK)
C2_MODE=smoke C2_DATASET=fmnist C2_PARTITION=dir1 C2_THREAT=free_rider C2_SEED=0 \
  C2_FRAC=1.0 C2_BURN_IN=1 C2_OBS_SRCS=flirds \
  C2_ARMS=observer,flirds_gate_v2,flirds_cgate,flirds_pweight \
  C2_RUN_ROOT=/tmp/c2_scale_smoke PYTHONPATH=. <VENV_PY> -u experiments/track_c2.py
# 확인: 4 arm 라인 + 게이트 arm AUROC 1.000 (FR exact-0) + TRACK-C2 RUN OK
# (burn_in=1은 스모크 한정 — R=4 안에서 게이트 경로를 밟기 위함. 본 실험은 기본 10.)
```

cifar10 데이터 검증(170,498,071 bytes·scp 권장)은 RUN_P5.md §5 그대로.

## 6. 실행 (sbatch 1개; 파일럿 → 잔여 2단 제출)

```bash
cd <REPO>/runs/track_h/scale
# ① 파일럿 = seed0 4셀 (idx 0,3,6,9) — 완주 후 태스크당 wall-clock을 보고하고 잔여 진행
REPO=<REPO> PY=<VENV_PY> sbatch --array=0,3,6,9 sbatch_scale.sh
# ② 잔여 8셀 (seed 1,2)
REPO=<REPO> PY=<VENV_PY> sbatch --array=1,2,4,5,7,8,10,11 sbatch_scale.sh
```

- 태스크 = 1셀(위협×seed) = 4 arm 순차, **추정 4–8h/태스크**(R1 2-arm 10–16분의
  ~10배 × 2 + 관찰자 채점; 파일럿으로 확정) → 전체 12태스크 ≈ 60–90 GPU-h.
- `--time=12:00:00` 기본. 파일럿이 8h를 넘기면 잔여 제출 전에 time만 상향(다른
  설정 변경 금지).

## 7. 종료 후 (보고 프로토콜)

1. `PYTHONPATH=codes <VENV_PY> runs/track_h/scale/make_analysis.py`
   → 절대 acc 표(vanilla/P1/P5h/P5s × clean·lf·fr·gn + 오염-평균; 3-seed mean±sd)
   + AUROC 표 + 게이트 행동 요약(corrupt vs clean 참여율·상대가중).
2. §4 HS-1~5 대조(HIT/MISS만; 수치는 analysis 산출물에).
3. rundir(`runs/track_h/rundirs_cnn_scale/`) + `runs/track_h/scale/analysis/` 커밋.
   결과 수치를 이 문서에 쓰지 말 것(파일-canon: rundir → overview → paper).

## 8. 금지

- 게이트·신뢰 하이퍼 변경/셀별 튜닝(z=1.645, burn_in=10 등 §2 값 고정; 스모크의
  burn_in=1은 스모크 한정).
- `C2_OBS_SRCS`에 coalition 소스(gtg/fedsv/comfedsv/shapleyfl) 추가 — §1 표 근거.
- `C2_T2`, oracle_excl/random_excl, audit 계측(LOO/그룹-Shapley 등) 추가 — 07-21
  결정으로 이 무대에서 제외.
- R/E/lr/MAL_FRAC 등 R1 세팅 변경(참여 frac만 1.0), poison 위협 추가.
- 이 문서·예측의 사후 수정.
