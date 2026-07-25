# runs/plots — 본문 수렴곡선 그림 (rundir-only, 재실행 0)

`make_paper_curves.py` 하나. **rundir만 읽어** 논문 본문에 들어가는 downstream 2무대의
per-round 학습곡선을 그린다. GPU·재학습·재평가 전부 불필요.

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
