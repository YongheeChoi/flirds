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

## 5-LLM · 실측 runtime — 무대별 전 방법 `[본문]`

> **세팅**: valuation-only wall-clock(초). 학습 시간은 별도(위상분리 절). **낮을수록 좋음** → 최저=볼드·2위=<u>밑줄</u>. 전용 탐지기 4종은 _기울임_(가치평가가 아니라 탐지 전용이라 서열 비교 대상 밖이지만 같은 무대 비용으로 병기). 무대마다 별도 표.

**교차-사일로 silo5 (N=5 · full · R=10 · 1B)** (● 3-seed · noisy 셀)

| 방법 | runtime(s) |
|---|---|
| **Flirds-1st** | <u>34.85±0.85</u> |
| Flirds | 106.59±1.96 |
| loss-heur | 98.86±1.95 |
| FedIF | 35.43±0.67 |
| GTG | 537.28±9.56 |
| FedSV | 532.42±15.49 |
| ComFedSV | 386.96±54.76 |
| ShapleyFL | 530.01±10.34 |
| Fed-LOO | 118.28±2.37 |
| Banzhaf | 532.11±10.48 |
| **(b)oracle (2⁵)** | 531.45±11.60 |
| _FLDetector_ | 90.79±6.47 |
| _FLTrust_ | 35.70±0.81 |
| _STD-DAGMM_ | 341.49±57.55 |
| _FedDQC_ | **21.81±0.41** |

**표준 부분참여 std20 (N=20 · 2/round · R=200 · alpaca)** (● 3-seed)

| 방법 | 1B | 3B | 7B |
|---|---|---|---|
| Flirds | 4696.7±111.7 | 11163.0±238.1 | 20180.4±250.3 |
| **Flirds-1st** | **1530.8±37.1** | **3632.9±78.9** | **6484.5±86.8** |
| loss-heur | 2913.0±72.2 | 6912.5±156.5 | 12298.8±166.8 |
| FedIF | <u>1533.6±36.9</u> | <u>3638.5±77.0</u> | <u>6494.7±88.0</u> |
| GTG | 3646.5±90.2 | 8653.1±196.9 | 15393.1±206.3 |
| FedSV | 3646.4±90.1 | 8653.9±199.1 | 15393.2±207.5 |
| ComFedSV | 2330.0±21.5 | 5530.6±37.8 | 9839.1±110.3 |
| ShapleyFL | 2916.5±72.1 | 6920.6±158.5 | 12312.4±166.5 |
| Fed-LOO | 2925.2±72.0 | – | – |
| **(b)oracle** | 2917.3±71.8 | 6922.9±153.6 | 12309.7±164.5 |

> ⚠ **std20은 Flirds가 (b)보다 비싸다**(1B 4697 vs 2917 = 1.61×). cohort=2뿐이라 (b) exact가 2²=4 fwd/round로 값싸고, HVP 1회(≈6.5 fwd)가 그보다 비싸기 때문 — **op-count 모델이 예측하는 소-cohort 역전**이고 방법의 결함이 아니다. cohort가 커지면 즉시 뒤집힌다(아래 device100).

**소형 앵커 anchor5 (N=5 · full · R=30)** (● 3-seed)

| 방법 | 1B | 3B | 7B |
|---|---|---|---|
| Flirds | 707.4±15.7 | 1679.1±36.8 | 3027.3±35.7 |
| **Flirds-1st** | **231.2±5.0** | **548.3±12.4** | **974.8±13.0** |
| loss-heur | 1093.3±26.0 | 2591.2±59.9 | 4612.6±62.1 |
| FedIF | <u>232.4±5.3</u> | <u>550.3±11.8</u> | <u>977.5±12.8</u> |
| GTG | 3551.7±82.2 | 8414.8±191.9 | 14972.0±193.1 |
| FedSV | 3535.6±86.1 | 8377.7±198.5 | 14906.5±270.4 |
| ComFedSV | 2556.5±215.3 | 6056.8±508.9 | 10791.5±1044.3 |
| ShapleyFL | 3512.7±82.9 | 8324.5±193.1 | 14812.2±198.5 |
| Fed-LOO | 772.7±18.6 | – | – |
| Banzhaf | 3527.3±82.7 | 8352.3±193.1 | 14844.0±198.2 |
| **(b)oracle (2⁵ in-run)** | 3527.7±82.7 | 8350.1±191.6 | 14838.5±196.4 |
| **(a)oracle (2⁵ 재학습)** | **30816.8±244.1** | ⬚ | ⬚ |

> **(a) retrain 오라클의 실가격**: anchor5 1B에서 **30,817 s = Flirds의 43.6×, (b) in-run의 8.7×**. 이게 "(a)-무대를 늘릴 수 없는" 근본 이유이자 (b)를 같은-게임 정답으로 쓰는 비용 근거. 3B/7B는 (a) 미실행(⬚).
>
> ⚠ **C6 회계 교정 (loss-heur 전용)**: `in_run_sv.py`가 singleton utility에서 base U(P_r)를 클라마다 중복 평가하던 버그를 base-캐시로 교정(라운드당 forward 2|P_r| → 1+|P_r|). **φ는 비트동일 → fidelity 무영향, runtime만 과대**였다. 교정 후 재측정 정본: **anchor5 1B = 657.3±15.2**(위 표 1093.3의 0.60배; `rundirs_e4_fedloo`) · **silo5 = 96.6/100.1/100.2 → 평균 99.0**(위 silo5 표 98.86과 일치 = silo5는 이미 교정판). std20 1B은 같은 재측정 런에서 **2199.0±48.9**(위 표 2913.0의 0.75배)로 나오나 README가 정본으로 지명한 건 anchor5뿐이라 **표 본문은 원 rundir 값을 유지하고 여기 병기**한다. 3B/7B loss-heur 재측정은 미실행.

**교차-디바이스 device100 앵커 (N=100 · 10/100 · α=0.5 · 1B)** (● 3-seed · noisy 셀)

| 방법 | runtime(s) |
|---|---|
| Flirds | 157.28±5.37 |
| **Flirds-1st** | <u>53.80±2.34</u> |
| loss-heur | 467.22±21.93 |
| FedIF | **53.65±2.34** |
| GTG | 18148.61±1592.19 |
| FedSV | 4968.95±199.63 |
| ComFedSV | 357.89±18.17 |
| ShapleyFL | 24934.65±1123.28 |
| **(b)oracle (per-round exact)** | 24974.94±1114.92 |
| _FLDetector_ | 238.09±34.63 |
| _FLTrust_ | 55.03±0.75 |
| _STD-DAGMM_ | 388.56±141.90 |
| _FedDQC_ | 443.87±6.22 |

> **여기가 본론**: cohort=10이 되자 (b)가 24,975 s로 폭증하고 **Flirds는 157 s로 평평** = **159× 저렴**(op-count 30,720 vs 30 fwd-equivalent 예측과 정합). ShapleyFL(24,935)·GTG(18,149)도 (b)급으로 폭증 — **cohort-지수 비용을 피하는 건 same-game Taylor 계열뿐**. 비-앵커 α 셀(0.0/0.01/0.1/5.0)도 Flirds 155~158 s로 동일(α 무관). **3B 스케일**(silo5): Flirds 250.9 · Flirds-1st 82.1 · loss-heur 384.3 · (b) 1243.8 (◐ 1-seed).

## 5-LLM · 지수-비용 스케일링 `[부록E]`

| 무대 | (b) exact | Flirds | 배율 | seed |
|---|---|---|---|---|
| N=10 완전열거 2¹⁰ | 117,649 s (32.7 h) | 733 s | **1/160** | ◐ 1-seed |
| device100 anchor (per-round) | 24,975 s | 157 s | **1/159** | ● 3-seed |
| anchor5 (a) 재학습 2⁵ | 30,817 s | 707 s | **1/44** | ● 3-seed |
| std20 (b) per-round (cohort=2) | 2,917 s | 4,697 s | 1.61× (역전) | ● 3-seed |

> N=10 셀 전체 runtime: (b) 117,648.9 · Flirds 732.8 · Flirds-1st 239.8 · loss-heur 1,239.7 · Fed-LOO 1,372.7 s. **cohort가 커질수록 배율이 커지는 단조 구조**(cohort 2 → 1.61× 손해 · cohort 5 → 5.0× 이득 · cohort 10 → 159~160× 이득). **출처**: `runs/track_d/rundirs_e5_n10/1B_anchor10_seed0/metrics.json` · `runs/phase2_matrix/analysis/04_device100_anchor/csv/runtime_table.csv` + op-count 모델.

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
- runtime(LLM): `runs/phase2_matrix/analysis/{01_silo5,04_device100_anchor,02_device100_sweep,05_scale_3b}/csv/runtime_table.csv`(`make_analysis.py`) · `runs/track_d/rundirs*/*/metrics.json` → `runtime` dict.
- loss-heur C6 교정본: `runs/measured_2026-07/loss_heur_acct/`(silo5) · `runs/track_d/rundirs_e4_fedloo/`(anchor5·std20).
- 위상분리·CNN: `runs/measured_2026-07/{timing_device100,e3_cost_smoke}/` · `runs/track_c/RESULTS.txt`.
- **⬚ 미실행**: anchor5 3B/7B의 (a) 재학습 오라클 · loss-heur 3B/7B 재측정 · CNN 방법별 3-seed runtime 표.
- ⚠ 런타임은 fp32·CPU·재구현 caveat 있는 단일/소수 측정 → **op-count가 하드웨어-독립 정본**. 상세 방법론 [[cost-comparison-methodology-2026-07/cost-comparison-methodology]].
- 축 지도: [[flirds-experiment-axis-map]] (구 카탈로그 §3.4 = git 이력)
