# runs/plots — 본문 그림 (rundir-only, 재실행 0)

스크립트 둘. 둘 다 **rundir/analysis CSV만 읽는다** — GPU·재학습·재평가 전부 불필요.

- `make_paper_curves.py` — downstream 2무대의 per-round 학습곡선 (아래 본문)
- `make_fig_fidelity.py` — fidelity 축 3그림: 본문 1 + 부록 2 (§"fidelity 그림" 절)

`make_paper_curves.py`는 논문 본문에 들어가는 downstream 2무대의
per-round 학습곡선을 그린다.

```bash
python runs/plots/make_paper_curves.py              # 두 무대 전부 -> runs/plots/figs/
python runs/plots/make_paper_curves.py --stage llm --dpi 300
python runs/plots/make_paper_curves.py --smooth 5   # 가독용 중심 이동평균(그림에 명시됨)
```

## 무엇을 그리나 — 본문 배치 기준

계획서 §1 배치표에서 **본문**이면서 per-round 궤적이 rundir에 남아 있는 실험은 2개다.
(fidelity·cost·ablation 축은 라운드축 곡선이 아니라 값/순위 지표라 대상이 아니다.)

| 그림 | 무대 | 곡선 출처 | arm |
|---|---|---|---|
| `fig_r4_llm_convergence` | **2-LLM 주무대 정확도 개입 (R4)** · gsm50k5 · Llama-3.2-1B · N=50 5/50 · R=200 | `metrics.json["val_curve"]` (val loss) | 07-25 5-arm 수록 범위: observer·oracle_excl·random_excl·Flirds·Flirds-1st |
| `fig_cnn_dir1_online` / `_retrain` | **2-CNN 점수원 경쟁** · FedSVCNN · cifar10/dir1 · N=100 10/100 · R=120 | `metrics.json["arms"][arm]["acc_curve"]` = `[[round, acc], ...]` | 본문 표 로스터: 앵커 + 8 점수원 × {online `<src>_gate_v2`, retrain `t2_sign_<src>`} |

셀 묶기(`_load`)·arm 파싱(`parse_arm`)·`flip_rate` 폴백은 **`runs/track_h/make_analysis.py`에서
그대로 import**한다 — 그림과 발표된 표가 "어느 rundir가 한 셀인가"에서 어긋날 수 없게.

## 규약

- 밴드 = seed 간 **mean ± std (ddof=0)**. 3-seed 미만 패널은 제목에 `◐ n=…`(§0.3 위반이 눈에 보이도록).
- 선 굵기는 그리는 순서대로 미세하게 가늘어진다 — **완전히 겹치는 곡선**(kept=전원인 T2 arm은
  vanilla와 동일)이 서로를 가리지 않게 하는 장치.
- `skipped=equals_vanilla` T2 arm은 해당 셀의 vanilla 곡선을 복제해 그린다(`equals_vanilla` 열로 표시).
  make_analysis가 이를 delta=0으로 채점하는 것과 같은 처리 — 빼버리면 grad-noise에서
  **1차-계열이 아예 발화하지 않았다는 결과 자체가 그림에서 사라진다**.
- `--smooth N`은 **집계된 곡선에만** 걸리는 중심 이동평균이고, 걸리면 그림 하단에 창 크기가 찍힌다.
  seed 통계에는 손대지 않는다. 기본값 1 = 원본.

## 산출물 (`figs/`, gitignore — 스크립트만 추적)

- `fig_r4_llm_convergence.{png,pdf}` — 3 위협 × (전구간 / round≥100 확대) 2행
- `fig_cnn_dir1_online.{png,pdf}`, `fig_cnn_dir1_retrain.{png,pdf}` — 5 위협 패널
- `curves_*.csv` — 그림에 들어간 long-format 원자료
- `coverage_*.csv` — (패널, arm) → seed 수. **0이면 미실행**이라는 뜻이고 그림에서도 빈다.

## 검증 (2026-07-25)

곡선 마지막 점 = 발표된 표의 최종값인지 대조: **CNN 1,560행 · LLM 31행 전부 max |diff| = 0.000e+00**.
`equals_vanilla` 복원분도 grad-noise retrain에서 `0.2436±0.0181`(Flirds-1st·FedIF)로 표와 정확히 일치.

## 알려진 공백 (데이터가 없어서 비는 것 — 필터가 아니다)

- **R4 retrain(T2) 곡선 없음**: `track_g.py`가 V3/T2 레그에서 `curve[-1]`만 캐시한다
  (`final_val_loss` O, `val_curve` X). 그래서 LLM 그림은 online 레그만이다.
- **R4 online Flirds-1st ⬚ 미실행** → 범례에 나오지 않는다(G4 잔여 6런).
- **R4 clean = seed0 1런** → `◐ n=1`.
- **GSM8K EM 곡선 없음**: EM은 배포 모델 1회 평가 → 라운드축 정확도는 재실행 대상.
- CNN clean 패널에 oracle/random 앵커 없음 — 오염이 없어 제외할 대상이 없다(표에서도 `–`).

---

# fidelity 그림 (`make_fig_fidelity.py`)

표로는 못 하는 두 종류의 주장 — **분포**(여러 세팅에 걸친 견고성)와 **관계**(두 Shapley 값
사이의 조건부 일치) — 을 그림으로 옮긴 것. 그림 셋: 본문용 1개(`body`) + 부록용 2개.

```bash
python runs/plots/make_fig_fidelity.py                  # 셋 다 -> figs/
python runs/plots/make_fig_fidelity.py --which body --dpi 300
python runs/plots/make_fig_fidelity.py --which retrain --dpi 300
python runs/plots/make_fig_fidelity.py --dose           # 축 밖 lf dose 사다리 + frrand 포함
```

## `fig_fidelity_body` — §5.2 본문 그림 (2패널 1행, 7.05×2.45in)

산출물은 `figs/`와 함께 **`paper/AAAI/Figures/`에도 복사**한다(Yonghee 07-28: 논문 그림
폴더 계속 갱신). 두 패널 모두 막대 = Spearman ρ, **잉크 눈금 = 같은 셀의 Pearson r**
(연결선 없음 — 눈금 소속은 행 위치가 결정). **축 눈금 라벨은 % 단위**(07-28;
(a) 0–100%, (b) 99.2–100%).

| 패널 | 형태 | 인코딩 |
|---|---|---|
| (a) CNN CIFAR-10, 한 패널 | 얇은 가로 짝막대 | 행 = 위협(대) × {Dir(1), IID}(소); 막대 = ρ(0 기준; xlim −0.15로 음수 눈금 여백 확보), 눈금 = r(음수 1회: gn Dir(1) 1st −.05가 기준선 왼쪽); 직접 라벨·음영·연결선 없음 |
| (b) LLM 2세팅 6셀 | 확대 짝막대 | 축 99.2–100%(시작점 = 데이터 최솟값에서 .004 격자 내림, 자동); **막대가 확대 축 시작(99.2%)에서 출발** — 잘린 baseline은 Yonghee 07-28 명시 결정이고 % 눈금 라벨이 공개한다. **cross-device 2셀·5-domain 3셀 제외**(전 셀 1.0000; 부록 존속). 1-seed 표시 없음(아래) |

**이 그림만의 규약 예외(07-28 Yonghee)**: 3-seed 미만 속 빈 마커·`*` 미적용(GSM8K
free-rider 현재 1-seed; 잔여 seed 착지 시 값만 갱신, n_seeds는 CSV에 존속), 그림 내
"3-seed mean"·셀 개수 문구 없음(캡션이 보유).

색 = 파랑 Flirds · 주황 Flirds-1st. 점(위치 부호화)의 확대는 정당, 막대(길이 부호화)는 0
기준 유지 — baseline 절단 금기 준수.

출처: (a) `track_c/c2fid/analysis/fidelity.csv`(cifar10 × {dir1,iid} × 오염축 4위협,
label-flip @0.70), (b) `phase2_matrix/rundirs/*/metrics.json`(GSM8K·5-domain)
+ `track_d/fidelity.csv`(1B/3B/7B). 그린 값 전량 = `figs/fig_fidelity_body_points_{cnn,llm}.csv`.

**설계 이력(2026-07-28 Yonghee 왕복 3회)**
1. dumbbell → 세로 막대(+Pearson bullet): 점-쌍은 크기 비교가 어렵다.
2. → 가로 막대 2-facet + LLM lollipop 공통 0–1 축(Pearson 제거·표 [F2] 삭제): 위협 라벨
   가독·'천장 그 자체' 노출.
3. → (a) facet 병합(위협×파티션 행)·막대 얇게·직접 라벨 제거·**Pearson 눈금+연결선
   복원**, (b) **cross-device 제외 + 확대 점판**("방법들 간 차이가 눈에 보이도록")·세팅
   머리글 여유 확대.
4. → (b) 리더선을 왼쪽 축 시작에서 출발시키고 **5-domain도 제외**(차이 없음),
   1-seed 표시·그림 내 "3-seed mean"/셀 수 문구 제거.
5. → (a) 음영 박스 제거, (b) 리더선을 점(ρ)까지로 단축, 그림 폰트 전체 확대
   (`BODY_FONTS` rc_context — 본문 그림만; 부록 그림은 전역 style() 유지).
6. → **최종**: 축 라벨 % 단위, (a) 0 기준선 우측 이동(xlim −0.15) + Pearson 연결선 제거,
   (b) 점판 → **확대 짝막대**(리더선이 이미 막대로 읽힘 → 실제 막대로 통일; baseline
   99.2%는 명시 결정), (b) 눈금 길이 80%, 하단 라벨 "(bars)".
- retraining 레그 → **본문 표 [F2]**(구 [F3]; CIFAR-10 dir1, `track_c/c1`, singleton utility
  행은 Yonghee가 본문 로스터에서 제외). **평균 열 = clean 포함 전 위협 평균**(07-28 지시;
  seed별 4위협 평균 → seed 간 mean±std(ddof=1), `methods_long.csv`에서 직접). 그림 형태는
  부록용 `fig_fidelity_retrain`에 남는다.

값 검증(2026-07-28): CNN 8셀 전부 T1과 일치 · LLM 6셀 전부 T2/T3와 일치(제외한 cross-device
2셀·5-domain 3셀 포함 시 11셀) · 산문 인용치는 stdout이 재현(CNN min ρ .847 / min r .937 ·
seed std ρ ≤ .076 · LLM min ρ .9946 / min r .9980 · zoom 시작 .992) · [F2]는
`methods_long.csv` dir1 12셀 ddof=1 재집계(T4의 ddof=0 판과 동일 값) · 평균 열 = in-run SV
+0.797±0.082 / Flirds +0.779±0.053 / 1st +0.478±0.114.

## 부록 그림 둘

| 그림 | 패널 | 셀 수 | 출처 |
|---|---|---|---|
| `fig_fidelity_inrun` | (a) 판정 셀의 per-client 값(표준화) | 100 client | `track_c/c2fid/rundirs/cifar10_dir1_grad-noise_fid_seed0/phi.parquet` |
| 〃 | (b) CNN 8세팅 × 4위협 | 32셀 ×3seed | `track_c/c2fid/analysis/fidelity.csv` |
| 〃 | (c) LLM 4무대(주무대·5도메인·규모·교차디바이스) | 11셀 | `phase2_matrix/rundirs/*/metrics.json` + `track_d/fidelity.csv` |
| `fig_fidelity_retrain` | (a) x=두 Shapley 값의 일치도, y=Flirds vs retrain | 60셀 | `track_c/c1/analysis/methods_long.csv` · `phase2_matrix/silo5_a_fidelity_1B.csv` · `track_d/rundirs/1B_anchor5_seed*/phi.parquet` |
| 〃 | (b) 방법별 paired Δ | 8방법 ×48셀 | `track_c/c1/analysis/methods_long.csv` |

## 규약 (fidelity 그림 공통)

- 점 = seed 평균(per-client 패널 제외). **3-seed 미만 셀은 속 빈 마커 + 눈금 `*`** — 미완주가
  눈에 보이도록(§0.3). 현재 해당은 LLM 주 세팅 free-rider 1셀뿐이다.
- 위협 축은 확정 5종만. `--dose`를 줘야 lf@{0.15,0.35}·strmain·frrand가 들어온다(스코프 밖).
- 패널 (b)/(c)는 **y축 공유**. LLM 무대가 천장에 붙는 것은 결과 자체이므로 y를 확대하지 않는다.
- 색은 dataviz 레퍼런스 팔레트 slot 1·2(#2a78d6 / #eb6834) 그대로. 재스텝 없음 → 그 팔레트가
  이미 통과시킨 all-pairs CVD 게이트를 그대로 상속한다(로컬에 node가 없어 validator 미실행).
- 그린 점은 전부 `figs/fig_fidelity_*_points*.csv`로 함께 떨어진다 — 그림과 표가 어긋날 수 없게.

## 판독 (2026-07-28 산출 기준)

- (b): gradient noise에서 **8/8 세팅이 분리**된다 — Flirds .763~.972 vs Flirds-1st .218~.362.
  위협별 전체 평균(각 n=24): clean .986/.700 · free-rider .986/.770 · gradient noise
  .901/.302 · label-flip .981/.929.
- retrain (a): x가 −0.564~+1.000으로 퍼진다(= 두 게임의 일치는 조건부). paired Δ는
  CNN −0.017±0.052(38/48이 ±0.05 이내) · LLM +0.000±0.000(12/12). LLM 5도메인 9셀은
  (1.00, 1.00)에 겹쳐 찍힌다.
