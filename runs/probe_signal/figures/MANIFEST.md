# probe_signal/figures — A축 신호크기 probe figure (rundir-only)

Generator: `runs/probe_signal/make_figures.py`. 질문(진단 위키 §2): 학습 lever(lr·steps·
LoRA rank·참여·CNN width)로 신호를 **키울** 수 있나 — 키워지면 **실재**(cross-seed 안정)인가.

**신호 크기 지표 = 클라 간 spread(std of φ)**: mean|φ|는 전 클라 공통 학습 shift가 지배해
순위와 무관(공통항은 랭킹에서 소거). 진단 문서 §3.4의 "~3배" 주장도 spread 기준임을 rundir
재계산으로 확인(spread lr3e-3에서 2.50–2.95x vs mean|φ| 1.31–1.46x).

**Coverage** (2026-07-16):
- LLM lr×steps 9/9(기준점 lr1e-3/st10 = `runs/track_d/rundirs/1B_anchor5_seed0` 재사용) +
  anchor rank 3/3(r16 = 동일 재사용) + std50k5 rank 3/3 — **전부 seed0 전용**
  (A축 seeds 1-2 미실행 = cross-seed 실재성은 LLM 축에서 미검증, 제목에 명기)
- noise_probe 2/2 (r16·r64)
- CNN C1 72/72 (2 scen × 4w × 3k × 3seed; (w=1,k=1) = `runs/track_c/c1` 재사용) — **3-seed = 실재성 축 보유**
- CNN C2 30/36 — **MISSING 6**: `pc2_cifar10_{clean,label-flip}_w1_f0.2_seed{0,1,2}`
  (제출 30 중 24 완료; (w=1,f=0.1) = `runs/track_c/c2` strmain 재사용)

| figure | 내용 (1줄) | 데이터 출처 |
|---|---|---|
| `01_llm_phi_magnitude_levers.png` | lr×steps 그리드의 (b)oracle φ spread 배율(lr↑=~3배·steps 무영향) + rank lever 절대값 | `phi.parquet` (probe+track_d 기준점) |
| `02_llm_fidelity_across_levers.png` | 전 lever에서 method별 Spearman vs (b) 유지 여부(seed0) | `metrics.json` spearman |
| `03_llm_participation_std50k5.png` | 참여 스트레스(N=50, 5/round): method별 생존 대조(Flirds +1.00 vs ComFedSV·ShapleyFL 붕괴) | `metrics.json` spearman |
| `04_llm_noise_probe_se.png` | val-chunk bootstrap: φ spread가 측정노이즈 대비 실재인가(spread/max-SE≈1.1) | `noise_probe/*/metrics.json` |
| `05_cnn_c1_realness_vs_magnitude.png` | **반쪽 판정 시각화** — w×k 그리드에서 SIZE(spread 배율)와 REALNESS(cross-seed ρ)가 따로 노는 것 | `cnn_c1/*/metrics.json` phi |
| `06_cnn_c1_fidelity_grid.png` | w×k에서 Flirds/GTG/FedSV/ComFedSV fidelity(3-seed 평균) | `cnn_c1/*/metrics.json` spearman_b |
| `07_cnn_c2_arms_outcome.png` | C2 개입 outcome: arm별 final-acc(2차)+탐지 AUROC(3차), w/f sweep | `cnn_c2/*/metrics.json` arms |
| `cnn_c1_realness.csv` / `llm_probe_summary.csv` | figure 정확 입력 | — |

재생성: `python runs/probe_signal/make_figures.py` (stdout에 커버리지+MISSING).
스팟체크(2026-07-16): CNN label-flip w2 k1.0 cross-seed ρ 손계산 +0.8586 = CSV 일치;
std50k5 r16 Spearman(Flirds +1.00/ComFedSV −0.109/ShapleyFL −0.064/FedSV +0.91/GTG +0.983)
= canon 서술과 일치; lr×steps spread 배율표 손계산 = 히트맵 일치.
