# phase2_matrix/figures — committed headline figures (rundir-only)

Generator: `runs/phase2_matrix/make_figures.py` (rerunnable; regenerates everything here
from rundir artifacts only — no document numbers). Deeper per-category tables/charts:
`make_analysis.py` → `analysis/` (gitignored, regenerable, June campaign only).

**Data sources**: `rundirs/<cell>/{config.yaml,metrics.json,phi.parquet}` (June 2026 campaign)
+ `rundirs_2026-07/<cell>/…` (July 2026 post-server-migration re-run; auto-included on rerun).

**Coverage** (2026-07-16 실행 기준): June 25/25 grid cells (silo5 4, sweep 12, poison 2,
anchor 3, 3B 4; 1B=3-seed, 3B=seed0). July 22/25 — `1B_device100-a0.5_{noisy,frrand,frzero}_anchor`
3셀 MISSING(이월 배치 대기; 재실행 시 자동 포함). June `rundirs/`의 B축 매트릭스 6셀
(`1B_iid5_*`, `1B_silo5_clean`)은 여기서 제외 — `runs/matrix_cxni/figures/` 담당.
Pearson: July/B축 셀은 metrics 네이티브, June 구셀은 phi.parquet 백필(스크립트가 재계산; 로그에 건수).

| figure | 내용 (1줄) | 비고 |
|---|---|---|
| `01_fidelity_spearman_vs_oracle.png` | 1차 fidelity — oracle 보유 셀(silo5/anchor/3B)의 method×cell Spearman 히트맵, June\|July 패널 | anchor 열의 ref=(b) per-round, 나머지=(b) exact 2^N |
| `02_fidelity_pearson_vs_oracle.png` | 값-수준 fidelity — 동일 레이아웃의 Pearson(순위 포화 아래 값 격차) | June 구셀은 phi 백필 |
| `03_fidelity_spearman_vs_proxy_sweep.png` | sweep/poison 셀(oracle 없음)의 Spearman — **ref=Flirds proxy, GT fidelity 아님** | 라벨에 명기 |
| `04_migration_june_vs_july.png` | 서버 이전 재현성 — 동일 (cell,threat,seed,method) 매칭 산점 3패널(Spearman/AUROC/runtime) | median\|Δ\|·max\|Δ\| 주석 |
| `05_cost_runtime_by_method.png` | method별 valuation wall-clock(로그 s), June vs July | 3B 포함으로 June 꼬리 김 |
| `06_detection_auroc_heatmap.png` | 3차(탐지) — 전 method×cell AUROC 히트맵, val/det 구분선, June\|July | 0.5=chance 중심 diverging |
| `07_detection_auroc_vs_alpha_sweep.png` | device100 sweep의 AUROC vs Dir(α), val/det 2행×threat 3열, June(실선) vs July(점선) | poison sweep은 06에서만 |
| `summary_by_cell_method.csv` | 위 figure들의 정확한 입력(campaign×cell×method 집계) — overview 세션용 | 467 rows |

재생성: `python runs/phase2_matrix/make_figures.py` (stdout에 커버리지 리포트).
스팟체크(2026-07-16): `1B_silo5_noisy` seed0 Flirds Spearman — phi.parquet 손계산 = metrics.json
= summary CSV = +1.000000 일치; `3B_silo5_poison`(July) Flirds AUROC 0.0 = metrics 일치.
