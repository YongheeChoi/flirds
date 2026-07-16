# track_c/figures — CNN 표준세팅 figure (rundir-only)

Generator: `runs/track_c/make_figures.py`. C1=fidelity(N=10 full, 11 methods incl Ripple),
C1_oracle=(a) 2^10 retrain, C2=개입(N=100 C=0.1 T=120).

**Coverage** (2026-07-16): c1 30/30, c1_oracle 30/30, c2 90/90 (30그룹×3seed) — MISSING 0.
dismissal q-sweep = cifar10 dir1 strmain 4-threat 12셀(설계; fmnist엔 미영속).
str sweep = dir1 label_flip{0.6,0.8,main}·grad_noise{0.05,main}; flirds_repl/add arm은 dir1 셀 전용.

**(a) 부호 규약**: `gt_a = -phi_a` (retrain oracle은 good→low; 러너·merge_oracle_a.py와 동일) —
vs-(a) Spearman은 phi 리스트에서 재계산, **기존 `fidelity.csv`(별도 코드 경로)와 300/300 일치 검증**.

| figure | 내용 (1줄) | 데이터 출처 |
|---|---|---|
| `01_c1_fidelity_dual_oracle.png` | 1차 fidelity — 듀얼 오라클: method×(ds×scenario) Spearman vs (b) / vs (a) 2패널((b)oracle 자신의 vs-(a) 행 포함 = 게임 괴리 가시화) | `c1/*/metrics.json` + `c1_oracle/*/metrics.json` phi_a |
| `02_c1_semantic_ladder.png` | 의미 검증 — −Spearman(φ, 오염율): 오염 사다리에서 φ가 단조 감소하는가 | `c1` spearman_vs_rate (native) |
| `03_c1_cost_runtime.png` | 비용 — method wall-clock vs 공유 traj vs (a) 2^10 retrain(로그) | `c1` runtime·traj_time + `c1_oracle` t_a |
| `04_c1_detection_auroc.png` | 3차 탐지 — method×(ds×scenario) AUROC(iid 제외) | `c1` auroc |
| `05_c2_outcome_delta_grid.png` | 2차 성능 — arm별 final-acc Δ vs vanilla(24그룹 히트맵, strmain) | `c2/*_strmain_*` arms |
| `06_c2_strength_ladder.png` | 강도 사다리 — dir1 label-flip·grad-noise 강도별 arm outcome(repl/add 포함) | `c2` str{0.6,0.8,0.05,main} |
| `07_c2_dismissal_qsweep.png` | dismissal q-sweep — 하위 φ 클라 drop 시 정확도(clean 무해·위협 개선) | `c2/cifar10_dir1_*` dismissal |
| `08_c2_detection_auroc.png` | 3차 탐지 — arm×(ds×part×threat) AUROC | `c2` arms.auroc |
| `c1_fidelity_vs_a.csv` | vs-(a) 재계산 롱테이블(figure 정확 입력) | — |

재생성: `python runs/track_c/make_figures.py` (stdout에 커버리지+fidelity.csv 교차검증).
스팟체크(2026-07-16): vs-(a) 300/300 = merge_oracle_a.py 산출과 일치(부호 규약 포함);
vs-(b)는 metrics 네이티브 그대로.
