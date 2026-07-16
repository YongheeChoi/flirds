# track_d/figures — IID-clean LLM 표준무대 figure (rundir-only)

Generator: `runs/track_d/make_figures.py` (rerunnable; `fidelity.csv`/`target_stability.csv`
파생물에 의존하지 않고 phi.parquet에서 재계산 — 존재 시 교차검증만 수행, 6/6 일치 확인).

**Coverage** (2026-07-16): 18/18 rundir ({1B,3B,7B} × {std20,anchor5} × 3-seed).
(a)-retrain oracle은 `1B_anchor5_seed{0,1,2}`에만 존재(설계).

**Provenance caveat (git ancestry로 실측)**: ShapleyFL β 통일(e89af94, 0.5→0.3) 기준 —
3B rundir(git `1c02fcd`, 07-03)=β0.3 **이후**; **1B(git `39a0a97`, 06-30)·7B(git `a5f5893`/`f677427`,
06-24~26)=β0.3 커밋 이전 코드 = β0.5 시절**. 7B는 서버-이전 후 재실행 대기(이월 배치;
`runs/rerun_beta03/RESUME_AFTER_MIGRATION.md`). ShapleyFL 행·shapleyfl_w arm 해석 시 주의.

| figure | 내용 (1줄) | 데이터 출처 |
|---|---|---|
| `01_fidelity_by_scale_regime.png` | 1차 fidelity — method×(scale×regime) Spearman(metrics)+Pearson(phi 재계산) 2패널, (a)oracle 행 포함 | `metrics.json` spearman + `phi.parquet` |
| `02_target_stability_oracle_crossseed.png` | Exp C — (b)oracle 자기순위 cross-seed ρ 바(레짐×스케일; IID-clean 무신호 진단) | `phi.parquet` method=(b)oracle |
| `03_arms_mmlu_rouge.png` | 2차 성능 — arm별 MMLU·ROUGE-L (do-no-harm parity 검증; base 대비) | `metrics.json` arms.*.{mmlu,rouge_l} |
| `04_convergence_val_curves.png` | 2차 수렴 — arm별 val-loss 곡선(스케일×레짐 6패널; base는 곡선 미영속=설계) | `metrics.json` arms.*.val_curve |
| `05_cost_runtime_by_method.png` | 비용 — method별 wall-clock(로그 s, marker=스케일) | `metrics.json` runtime |
| `fidelity_summary.csv` / `arms_summary.csv` / `target_stability_recomputed.csv` | figure 정확 입력(overview 세션용) | — |

재생성: `python runs/track_d/make_figures.py` (stdout에 커버리지+교차검증).
스팟체크(2026-07-16): `1B_anchor5_seed0` (a)oracle Spearman — metrics.json=+0.900=CSV 일치
(3-seed mean +0.933 = 히트맵 셀); `1B_std20` flirds_w MMLU 3-seed 손평균 0.474481 = CSV 일치;
target-stability 재계산 = 기존 `target_stability.csv`(별도 코드 경로) 6/6 일치.
