# matrix_cxni/figures — B축(오염 × 비IID) 매트릭스 figure (rundir-only)

Generator: `runs/matrix_cxni/make_figures.py`. 이 실험의 rundir은 설계상
`../phase2_matrix/rundirs/`에 있음(README 참조): 신규 6셀(`1B_iid5_{clean,noisy,frrand,frzero,poison}`
+ `1B_silo5_clean`) + June silo5 오염 4셀 재사용(`1B_silo5_{noisy,frrand,frzero,poison}`).

**Coverage** (2026-07-16): 10/10 셀, 전부 3-seed, (b)=exact 2^5 oracle.

| figure | 내용 (1줄) | 데이터 출처 |
|---|---|---|
| `01_fidelity_oracle_crossseed_rho.png` | **진단 결정타** — (b)oracle 자기순위 cross-seed ρ 바(stage×threat; clean 열=오염 0에서 도메인 이질성만의 신호) | `phi.parquet` method=(b)oracle, 3 seed-pair Spearman |
| `02_crossseed_rho_by_method.png` | method별 자기 φ 순위의 seed 간 안정성 히트맵 | `phi.parquet` 전 val method |
| `03_fidelity_vs_oracle_heatmap.png` | 추정기 fidelity — Spearman(metrics 네이티브)·Pearson(phi 재계산, 네이티브와 144/144 일치 검증) 2패널 | `metrics.json` + `phi.parquet` |
| `04_detection_auroc_matrix.png` | 탐지 AUROC IID vs non-IID 블록(3차 축; clean 제외 — 오염 클라 없음) | `metrics.json` auroc |
| `crossseed_rho.csv` | 01·02의 정확한 입력(cell×method cross-seed ρ) | — |

재생성: `python runs/matrix_cxni/make_figures.py` (stdout에 커버리지+self-check).
스팟체크(2026-07-16): oracle cross-seed ρ 손계산 — IID clean pairs {-0.3,+0.6,+0.1} mean **+0.133**,
non-IID clean pairs {+1.0,+0.8,+0.8} mean **+0.867** = figure/CSV 일치(진단 문서 canon +0.13/+0.87 재현).
