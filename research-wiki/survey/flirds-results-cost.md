---
type: survey
title: "Flirds 결과 — 비용·규모 (Cost / Scalability)"
created: 2026-07-25
updated: 2026-07-25
tags: [flirds, results, cost]
---

# Flirds 결과 — 5. 비용·규모

> **축**: 방법별 wall-clock과 스케일링(E6). 분류·순서는 [[flirds-experiment-axis-map]] §5.
> **읽는 법**: op-count는 **해석적**(지배연산 수 = seed-무관·하드웨어/정밀도 독립)이라 std 없음 — 이게 정본 비용축. wall-clock은 계측 캠페인의 단일/소수 측정(정밀도·CPU·재구현 caveat)이라 op-count의 **교차검증**으로만 병기. 런타임 표는 **낮을수록 좋음** → 최저=볼드·2위=<u>밑줄</u>.
> 자매 페이지: [[flirds-results-fidelity]] · [[flirds-results-downstream]] · [[flirds-results-detection]] · [[flirds-results-ablation]]

---

## 5-공통 · 연산수(op-count) 모델 `[본문]`

> **세팅**: 라운드당 지배연산(forward/gradient/HVP)의 해석적 카운트. Flirds(2차)=1 HVP/round(cohort 무관) · Flirds-1st=1 grad/round · loss-heur=1+|P_r| fwd/round · (b) exact=2^|P_r| fwd/round(cohort에 지수적). **출처**: `runs/measured_2026-07/op_counts.py`.

| 무대 (regime) | Flirds | Flirds-1st | loss-heur | (b) exact |
|---|---|---|---|---|
| silo (N=5·R10·full) | 10 HVP | 10 grad | 60 fwd | 320 fwd |
| anchor (N=5·R30·full) | 30 HVP | 30 grad | 180 fwd | 960 fwd |
| device (N=100·R30·10/100) | 30 HVP | 30 grad | 330 fwd | **30,720 fwd** |

> 단위가 달라(HVP/grad/fwd) 표 안에서 직접 볼드 비교는 안 한다 — 시간 환산은 아래 microbench. **핵심**: (b)는 cohort에 지수적(2^k)이라 cohort 큰 무대(device)서 30,720 fwd로 폭증, Flirds는 30 HVP로 평평.

## 5-공통 · microbench (per-op, fp32·B200) `[본문]`

| 연산 | 시간(s) | 비고 |
|---|---|---|
| forward | 1.60 | 기준 |
| HVP | 10.36 | HVP/fwd = 6.47 |

> fp32→bf16 배율: forward ×5.33 · HVP ×4.09 · GEMM ×22.68. **출처**: `runs/measured_2026-07/microbench/summary.json`.

---

## 5-LLM · 실측 runtime (op-count 교차검증) `[본문]`

> 런타임(초)은 op-count×microbench 예측과 측정의 교차검증. 낮을수록 좋음. loss-heur는 C6 회계-교정 후 정본.

| 무대 | Flirds | loss-heur | (b) exact | Flirds vs (b) |
|---|---|---|---|---|
| silo (N=5·R10) | <u>104</u> | **96** | 530 | 5.1× |
| anchor (N=5·R30) | <u>707</u> | **657** | 3,528 | 5.0× |
| device (N=100·R30·10/100) | **157** | <u>330</u> | **24,975** | **159×** |

> Flirds-1st의 정밀 wall-clock은 소스에 없어 표에서 뺐다 — op-count상 **전 무대 최저**(1 grad/round < HVP < fwd)라 항상 가장 싸다(위 op-count 표). **읽기**: cohort 큰 무대(device 10/100)서 Flirds가 (b) 대비 **159× 저렴**(op-count 30,720 vs 30 예측과 정합). 소-cohort(silo full)선 loss-heur가 Flirds보다 약간 싸다(near-additive 무대의 정직한 긴장점). **출처**: `runs/track_d`·`runs/phase2_matrix`(device100) rundir `runtime` + `runs/measured_2026-07/loss_heur_acct/`(silo 96.6/100.1/100.2 s).

## 5-LLM · 지수-비용 스케일링 `[부록E]`

| 무대 | (b) exact | Flirds | 배율 | seed |
|---|---|---|---|---|
| N=10 완전열거 2¹⁰ | 117,649 s (32.7 h) | 733 s | **1/160** | ◐ 1-seed |
| device100 anchor (per-round) | ~24,975 s | 157 s | **1/159** | ● 3-seed |

> **출처**: `runs/track_d/rundirs_e5_n10/1B_anchor10_seed0/metrics.json` · `runs/phase2_matrix/rundirs/1B_device100-a0.5_*` + op-count 모델.

## 5-LLM · 학습↔가치평가 위상분리 `[본문/부록]`

> device100 clean: client-training **2,249 s** vs valuation **2,704 s**(peak 33.5/99.1 GiB). 3-seed 셀 = client-training 6,811~7,277 s + valuation 8,397~8,990 s = **셀당 4.2~4.5 GPU-h**. **출처**: `runs/measured_2026-07/timing_device100/` · rundir `timing.json`(`flirds/timing.py`).

---

## 5-CNN · runtime `[부록]`

> Track C CNN: 방법별 valuation runtime이 **학습 자체보다 2~3자릿수 저렴**(Flirds 0.27~0.36 s vs 공유 valuation 65.8/163.9 s vs 자기-궤적 재학습 2,151/4,275 s). 제외 baseline의 자기-궤적 재실행 비용은 셀 total의 ~95%. **출처**: `runs/track_c`(RESULTS) · `runs/measured_2026-07/e3_cost_smoke/`.

| 항목 (mnist / cifar10, iid seed0) | mnist | cifar10 |
|---|---|---|
| Flirds valuation | **0.27** | **0.36** |
| 전 방법 공유 valuation | <u>65.8</u> | <u>163.9</u> |
| 자기-궤적 재학습(제외 baseline) | 2,151 | 4,275 |

---

## 출처·재생성

- op-count: `python runs/measured_2026-07/op_counts.py`(해석적, seed-무관).
- microbench: `runs/measured_2026-07/microbench/summary.json`.
- runtime: 각 rundir `runtime`/`timing.json` + `runs/measured_2026-07/{loss_heur_acct,timing_device100,e3_cost_smoke}/`.
- ⚠ 런타임은 fp32·CPU·재구현 caveat 있는 단일/소수 측정 → **op-count가 하드웨어-독립 정본**. 상세 방법론 [[cost-comparison-methodology-2026-07/cost-comparison-methodology]].
- 축 지도: [[flirds-experiment-axis-map]] (구 카탈로그 §3.4 = git 이력)
