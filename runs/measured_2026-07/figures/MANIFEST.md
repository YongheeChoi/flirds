# measured_2026-07/figures — 검증 실측 4종 figure (rundir-only)

Generator: `runs/measured_2026-07/make_figures.py`. 7월 검증 캠페인(서버 이전 후 B200) 실측 4종.

**Coverage** (2026-07-16): taylor 3/3 seed, tf32_ab 4/4 셀, microbench 1/1, acct 3/3 seed — MISSING 0.

| figure | 내용 (1줄) | 데이터 출처 |
|---|---|---|
| `01_taylor_residuals_p3.png` | P3 물리 잔차 — resid1(1차) vs resid2(2차) pooled 크기(로그, fp32 ulp 기준선) + log-log slope(이론 2/3 대비; **slope2≈1.5 = cubic 미확증**) | `taylor/llama1b_r10_seed*/summary.json` |
| `02_tf32_ab_contrast.png` | TF32 on/off A/B — method별 spearman_b·spearman_vs_rate·AUROC 짝 대조(cifar10 iid·label-flip; iid서 GTG/FedSV/ShapleyFL **음수** 그대로 표시) | `tf32_ab/*/metrics.json` |
| `03_precision_microbench.png` | 정밀도 마이크로벤치 — forward/HVP/GEMM × fp32/tf32/bf16 s/op(로그)+배율 | `microbench/summary.json` |
| `04_cost_accounting.png` | 정직 비용 회계 — method wall-clock vs **FL 학습 자체**(coalition류 ~130%, Flirds 26%, Flirds1st 9%) | `acct/acct_seed*.summary.txt` (이 캠페인의 영속 아티팩트) |
| `taylor_summary.csv` / `acct_runtime.csv` | figure 정확 입력 | — |

재생성: `python runs/measured_2026-07/make_figures.py`.
스팟체크(2026-07-16): taylor seed0 slope_r2 = 1.6349(summary.json) = figure 1.63;
acct seed0 GTG 526.6s/434.6s = 121%(3-seed 평균 131% = figure 주석) 일치.
발견·수정: tf32 iid Spearman ylim이 음수값을 가리던 버그 → (-1.09,1.09)로 교정(데이터 무손실 표시).
