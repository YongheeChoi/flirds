---
type: survey
title: "Flirds 결과 — Ablation (구성요소·검증)"
created: 2026-07-25
updated: 2026-07-25
tags: [flirds, results, ablation]
---

# Flirds 결과 — 4. Ablation

> **축**: 구성요소·lever·프로토콜 검증. 분류·순서는 [[flirds-experiment-axis-map]] §4. **쪼갤 수 있는 건 CNN/LLM 레그로 분리**, 모델-무관만 §4-공통.
> **읽는 법**(공통 규약 [[flirds-results-fidelity]] §읽는 법): mean±std(ddof0) · metric별 최고=볼드·2위=<u>밑줄</u> · ● 3-seed/◐ 부분.
> 자매 페이지: [[flirds-results-fidelity]] · [[flirds-results-downstream]] · [[flirds-results-detection]] · [[flirds-results-cost]]

---

## 4-CNN

### 2차항(HVP)의 기여 — CNN 레그 `[본문 §5.6①]` ● 3-seed

> **세팅**: c2fid grad-noise 셀에서 Flirds(2차) vs Flirds-1st(1차만) fidelity ρ vs (b). 등방 grad-noise는 1차 gradient-정렬로 안 보이고 2차 곡률로만 잡힌다. *↔ LLM 짝(§4-LLM)*

| 파티션 | Flirds ρ | Flirds-1st ρ |
|---|---|---|
| cifar10/dir1 | **0.847±0.027** | <u>0.218±0.053</u> |
| cifar10/iid | **0.870±0.024** | <u>0.313±0.062</u> |
| cifar10/qskew | **0.763±0.053** | <u>0.305±0.059</u> |
| cifar10/shard | **0.941±0.005** | <u>0.285±0.052</u> |
| fmnist/dir1 | **0.967±0.016** | <u>0.310±0.067</u> |
| fmnist/iid | **0.972±0.007** | <u>0.358±0.027</u> |

> **grad-noise서 Flirds-1st 전 파티션 붕괴(0.22~0.36)** vs Flirds 0.76~0.97 = 2차항 존재 이유. **다운스트림 재현**([[flirds-results-downstream]]): grad-noise 개입 acc Flirds .5668(online)/.6065(retrain) vs 1차계열 .244~.248 실명. **부분참여 k-sweep**(C1 label-flip, 파일럿): Flirds 0.891 vs Flirds-1st 0.305 @k=0.2 (k=0.5 .979/.765·full .993/.940) — 1차항은 클라당 참여 적으면 붕괴, 2차항이 방어(상세는 구 카탈로그 §4.1 = git 이력). **출처**: `runs/track_c/c2fid`(grad-noise) + `runs/probe_signal/cnn_c1`(k-sweep).

### A축 용량 lever probe — CNN `[본문 §5.6②]` ● 3-seed

> **세팅**: cifar10 · N=10 · R=10 · **모델 폭 {0.5,1,2,4}× × 클라당 참여율 {0.2,0.5,1.0}** × {iid(오염0), label-flip(오염)} × seed{0,1,2}. LLM 짝과 같은 질문 — "신호 크기 lever가 실재 신호를 만드나". *↔ LLM 짝*

**(b) 오라클 순위의 cross-seed ρ — 신호 실재성** (● 3-seed)

| 폭 | iid k=0.2 | iid k=0.5 | iid k=1.0 | lf k=0.2 | lf k=0.5 | lf k=1.0 |
|---|---|---|---|---|---|---|
| w=0.5× | 0.022 | -0.022 | 0.034 | -0.147 | 0.038 | **0.976** |
| w=1.0× | 0.515 | -0.285 | -0.042 | 0.160 | 0.095 | <u>0.968</u> |
| w=2.0× | -0.188 | -0.131 | 0.038 | -0.051 | 0.111 | 0.859 |
| w=4.0× | 0.083 | -0.228 | 0.123 | 0.038 | -0.042 | 0.923 |

**(b) φ 절대크기(spread)** (● 3-seed)

| 폭 | iid k=0.2 | iid k=0.5 | iid k=1.0 | lf k=0.2 | lf k=0.5 | lf k=1.0 |
|---|---|---|---|---|---|---|
| w=0.5× | 0.093 | 0.060 | 0.015 | 0.076 | 0.061 | 0.035 |
| w=1.0× | 0.170 | 0.062 | 0.010 | 0.117 | 0.064 | 0.035 |
| w=2.0× | 0.120 | 0.075 | 0.019 | 0.112 | 0.068 | 0.042 |
| w=4.0× | 0.149 | 0.072 | 0.020 | 0.125 | 0.086 | 0.042 |

> **판정**: ① **iid 12칸 전부 xseed ρ ≈ 0**(−0.29~+0.52, 부호 무작위) — **폭을 8배(0.5×→4×) 늘려도 IID엔 실재 신호가 안 생긴다**. LLM의 lr lever와 같은 결론. ② **label-flip에서도 k=0.2·0.5는 ρ≈0이고 k=1.0(전원참여)에서만 0.86~0.98로 살아난다** — 즉 CNN R=10 무대에선 **참여율이 오염 신호를 재현 가능하게 만드는 실질 lever**다(폭은 아님). ③ φ 절대크기는 **참여율이 낮을수록 커지지만**(k=0.2에서 0.076~0.170 vs k=1.0에서 0.010~0.042) 그 큰 φ가 곧 실재 신호는 아니다(iid k=0.2 spread 0.17인데 ρ 0.52 → 다른 셀은 음수). **크기 ≠ 실재성**이 CNN·LLM 양쪽에서 같은 형태로 확인된다. fidelity(Flirds vs (b))는 lever 전반 유지 = **Taylor tradeoff 없음**. **출처**: `runs/probe_signal/figures/cnn_c1_realness.csv`.

### Removal-curve — CNN `[본문 §5.6③]` ● 3-seed

> **세팅**: cifar10/mnist N=10, φ 순위대로 worst-first(나쁜 클라부터) vs best-first 제거 후 재학습 → acc 분리. 게임-무관 인과 검증. *↔ LLM 짝*

| 시나리오 | Flirds ρ(vs b) | worst-first Δacc | best-first Δacc | 판정 |
|---|---|---|---|---|
| cifar10 label_flip | +1.00 | **+0.0445** | ≈ (b) +0.045 | ✅ 순위→acc 인과 (mnist의 ~13×) |
| cifar10 feature_noise | +1.00 | +0.0385 | 저순위 방법 ≈0 | ✅ 순위→분리 재확인 |

> worst-first 제거가 acc를 (b)와 동급으로 올리고, 저순위 방법은 분리 ≈0 = 순위정보가 인과적. **출처**: `runs/removal_dose/rundirs_cnn`.

### 정밀도(TF32) A/B `[검증-전용]` ◐ seed0

> cuDNN conv TF32 on/off의 final_acc 차 **≤0.001**·Flirds spearman_b 비트동일 — 정밀도 축이 CNN 결론을 안 바꿈. **출처**: `runs/measured_2026-07/tf32_ab/`.

---

## 4-LLM

### 2차항(HVP)의 기여 — LLM 레그 `[본문 §5.6①]` ● 3-seed(경량)

> **세팅**: Llama-3.2-1B · **std50k5 부분참여(N=50, 5/50 = 클라당 참여 희박)** · (b) per-round 대비 Spearman · LoRA rank {16,32,64} 3점 = 용량 lever와의 교차 확인. seed0 파일럿(rank16만 3-seed). *↔ CNN 짝*

| 방법 | r=16 | r=32 | r=64 |
|---|---|---|---|
| **Flirds** | **1.000** | **1.000** | **1.000** |
| Flirds-1st | <u>1.000</u> | 1.000 | 0.999 |
| loss-heur | 1.000 | <u>1.000</u> | 0.999 |
| Fed-LOO | 1.000 | 1.000 | <u>1.000</u> |
| GTG | 0.983 | 0.983 | 0.981 |
| FedSV | 0.910 | 0.899 | 0.909 |
| **ComFedSV** | **-0.109** | **-0.125** | **-0.081** |
| **ShapleyFL** | **-0.064** | **-0.093** | **-0.078** |
| **FedIF** | **-0.040** | **-0.076** | **-0.052** |

> 부분참여 5/50에서 **same-game 계열(Flirds·Flirds-1st·loss-heur·Fed-LOO)만 ≥0.999 유지**하고 **uniform-subset(ComFedSV·ShapleyFL)과 FedIF는 −0.04~−0.13으로 부호까지 뒤집힌다** — 참여가 희박하면 subset-샘플링 추정량이 붕괴한다는 뜻. GTG/FedSV는 0.90~0.98로 중간. **rank를 4배 키워도 서열 불변** = 붕괴 원인은 용량이 아니라 참여 구조. LLM은 R=200이라 클라당 참여 횟수가 충분해 **Flirds-1st도 버틴다**(CNN R=10 짧은 지평선에서 1차항이 붕괴하는 것과 대비 — 위 §4-CNN k-sweep). **출처**: `runs/probe_signal/figures/llm_probe_summary.csv`.

### A축 용량 lever probe — LLM `[본문 §5.6②]` ● 핵심축 3-seed(나머지 seed0)

> **세팅**: 1B anchor5(N=5 · IID · clean) · lever = LoRA rank{16,32,64} × lr{1e-3,2e-3,3e-3} × local steps{10,20,30}. **묻는 것**: "용량/학습률을 키우면 클라 간 신호가 커지나, 그리고 Taylor 근사가 깨지나". *↔ CNN 짝*

**lever가 fidelity를 깨나 — Spearman vs (b)** (◐ seed0; lr1e-3·10st·r16만 3-seed)

| 방법 | r=16 | r=32 | r=64 | lr1e-3·10st | lr1e-3·20st | lr1e-3·30st | lr2e-3·10st | lr2e-3·20st | lr2e-3·30st | lr3e-3·10st | lr3e-3·20st | lr3e-3·30st |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Flirds** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| Flirds-1st | <u>1.000</u> | <u>1.000</u> | <u>1.000</u> | <u>1.000</u> | <u>1.000</u> | <u>1.000</u> | <u>1.000</u> | <u>1.000</u> | <u>1.000</u> | <u>1.000</u> | <u>1.000</u> | <u>1.000</u> |
| loss-heur | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 |
| GTG | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| FedSV | 0.700 | 0.700 | 0.500 | 0.700 | 0.800 | 0.800 | 0.900 | 0.800 | 0.500 | 0.900 | 0.900 | 0.500 |
| ComFedSV | 0.900 | 0.900 | 0.900 | 0.900 | 1.000 | 0.900 | 1.000 | 1.000 | 0.600 | 0.900 | 0.900 | 0.600 |
| ShapleyFL | 0.900 | 0.900 | 0.700 | 0.900 | 1.000 | 0.600 | 0.800 | 0.800 | 1.000 | 0.500 | 0.900 | 0.900 |
| FedIF | 0.700 | 0.500 | 0.700 | 0.700 | 0.900 | 0.700 | 0.800 | 0.800 | 1.000 | 0.500 | 0.900 | 1.000 |

**lever가 φ 크기를 키우나 — (b) φ spread** (◐ seed0)

| lever | φ spread |
|---|---|
| r=16 (기준) | 0.000446 |
| r=32 | 0.000340 |
| r=64 | 0.000376 |
| lr1e-3 · 10/20/30 steps | 0.000446 / 0.000608 / 0.000442 |
| lr2e-3 · 10/20/30 steps | 0.000823 / 0.001021 / 0.000832 |
| **lr3e-3 · 10/20/30 steps** | **0.001125 / 0.001315 / 0.001115** |
| std50k5 r=16/32/64 (참여축) | 0.000948 / 0.001130 / 0.001330 |

> **판정 2줄**: ① **Flirds·Flirds-1st는 12개 lever 셀 전부에서 1.000** — rank 4배·lr 3배·steps 3배 어디서도 Taylor 근사가 안 깨진다(HVP 강건성). 반대로 **FedSV(0.50~0.90)·FedIF(0.50~1.00)는 lever마다 출렁**여 lever 자체가 renorm 추정량의 잡음을 키운다. ② **lr만 φ 절대크기를 키운다**(1e-3 0.00045 → 3e-3 0.00113 ≈ **2.5배**), rank와 steps는 무영향. 그러나 **커진 φ가 cross-seed 실재 신호는 아니다** — 같은 IID-clean 무대의 (b) 자기-안정성은 여전히 −0.37(anchor5, [[flirds-results-fidelity]] §1C). 즉 **A축(용량)은 φ를 키울 뿐 신호를 만들지 않고, 신호는 B축(비IID·오염)이 만든다**(non-IID clean ρ 0.87 vs IID 0.13). **출처**: `runs/probe_signal/figures/llm_probe_summary.csv`.
> ⚠ **미확인**: lr·steps 셀은 seed0만 — "커진 φ가 cross-seed 실재냐"의 직접 검증(예측 ρ≈0)은 seed1·2 미실행.

### Removal-curve — LLM `[본문 §5.6③]` ● 3-seed

> **세팅**: silo5 N=5, worst-first vs best-first 제거 후 val-loss. *↔ CNN 짝*

| 위협 | Flirds ρ(vs b) | worst-first Δval-loss | best-first Δval-loss | 판정 |
|---|---|---|---|---|
| noisy | +1.00 | **+0.0076** | −0.0084 | ✅ worst-first 제거가 loss 내림 |
| frrand | +1.00 | +0.0071 | −0.0015 | ✅ |
| frzero | +1.00 | +0.0067 | −0.0016 | ✅ |

> **출처**: `runs/removal_dose/rundirs`(A2).

### Dose-response · AdamW 브리지 · Taylor 물리잔차

### Taylor 물리잔차 (명제 P3) `[보류·부록A]` ● 3-seed

> **세팅**: Llama-3.2-1B · N=5 · R=10 · val=100 · 실제 학습 궤적 위에서 **1차 전개 잔차 r1 = |ΔL − g·ΔW|** vs **2차 전개 잔차 r2 = |ΔL − g·ΔW − ½ΔWᵀHΔW|** 를 라운드×연합마다 측정(pooled). log-log 기울기는 ‖ΔW‖에 대한 잔차의 수렴 차수. **낮을수록 좋음** → 최저=볼드.

| seed | r1 mean | r1 median | r1 max | r2 mean | r2 median | r2 max | r2/r1 (mean) | 기울기 r1 | 기울기 r2 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2.642e-06 | 1.753e-06 | 1.545e-05 | **6.695e-07** | 6.380e-07 | 2.391e-06 | 0.253 | 2.719 | 1.635 |
| 1 | 2.307e-06 | 1.549e-06 | 1.307e-05 | 5.573e-07 | 4.513e-07 | 2.157e-06 | 0.242 | 2.263 | 1.565 |
| 2 | 1.814e-06 | 1.277e-06 | 1.004e-05 | **4.738e-07** | **4.411e-07** | **1.372e-06** | 0.261 | 1.818 | 1.466 |
| **평균** | 2.254e-06 | 1.526e-06 | 1.285e-05 | **5.669e-07** | 5.101e-07 | 1.973e-06 | **0.252** | 2.267 | 1.555 |

> **2차항 추가의 물리적 정당화**: 2차 잔차가 1차 잔차의 **0.25배**(≈ 4× 작음; median 기준 3.0×, max 기준 6.5×) — 즉 HVP 항이 실제 손실 변화를 유의하게 더 잘 설명한다. 단 **log-log 기울기는 r1 2.27 / r2 1.56**으로 이론값(각각 2·3)과 어긋나는데, ‖ΔW‖ 범위가 좁고(라운드 간 4e-3 부근) 잔차가 fp32 ULP(2.4e-07) 근방이라 **수치 바닥에 걸린 것**으로 보인다 — 배율 주장은 유효하고 차수 주장은 이 데이터로 못 한다. closed-form↔Flirds(2차) max|Δφ| ~1e-10(대수적 동치 확인). **출처**: `runs/measured_2026-07/taylor/llama1b_r10_seed{0,1,2}/summary.json`.

### Dose-response — 오염강도 vs 탐지 문턱 `[제외]` (07-25 Yonghee: 논문 미수록) ● 3-seed

> **세팅**: silo5 · N=5 · **오염 강도를 연속으로 낮춰가며** AUROC가 언제 무너지나. noisy는 답 교체율 nr∈{0, 0.1, 0.25, 0.5, 0.75, 1.0}, frrand는 델타 배율 dm∈{0.25, 0.5, 1, 2, 4}.

**noisy — 답 교체율(nr) 스윕 · AUROC** (● 3-seed)

| 방법 | nr=0 | nr=0.1 | nr=0.25 | nr=0.5 | nr=0.75 | nr=1.0 |
|---|---|---|---|---|---|---|
| **(b)oracle** | 0.833 | 0.750 | **1.000** | **1.000** | **1.000** | **1.000** |
| Flirds | 0.833 | 0.750 | <u>1.000</u> | <u>1.000</u> | <u>1.000</u> | <u>1.000</u> |
| Flirds-1st | 0.833 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 |
| loss-heur | 0.833 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 |
| GTG · ShapleyFL · Banzhaf | 0.833 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 |
| FedSV | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ComFedSV | **0.917** | <u>0.917</u> | 1.000 | 1.000 | 1.000 | 1.000 |
| FedIF | 1.000 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |
| _FLTrust_ | **1.000** | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |
| _FLDetector_ | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 |
| _STD-DAGMM_ | 0.500 | 0.083 | 0.250 | 0.500 | 0.500 | 0.583 |
| _FedDQC_ | 0.083 | 0.250 | 0.250 | 0.500 | 0.750 | 0.917 |

**frrand — 델타 배율(dm) 스윕 · AUROC** (● 3-seed)

| 방법 | dm=0.25 | dm=0.5 | dm=1.0 | dm=2.0 | dm=4.0 |
|---|---|---|---|---|---|
| **(b)oracle · Flirds · Flirds-1st · loss-heur · GTG · FedSV · ComFedSV · ShapleyFL · FedIF · Banzhaf · Fed-LOO** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| _FLTrust_ · _FLDetector_ | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| _STD-DAGMM_ | 0.833 | 1.000 | 1.000 | 1.000 | 1.000 |
| _FedDQC_ | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 |

> **문턱**: noisy는 **nr≥0.25에서 φ 계열 전부 1.000**, nr≤0.1에선 0.75~0.83으로 내려간다 — 단 그 하락은 **(b) oracle도 똑같이 겪는다**(0.833/0.750) = 게임의 신호 한계이지 추정 실패가 아니다. free-rider(frrand)는 **델타 배율 0.25~4 전 구간에서 φ 계열 1.000** = 배율과 무관하게 잡힌다. 전용 탐지기 중 FedDQC만 dose에 단조 반응(0.083→0.917), STD-DAGMM은 전 구간 약함. *(제외 위협 poison의 pf 스윕도 같은 rundir에 있다 — Flirds가 pf 0.1~0.2에서 0.000이고 pf 0.9 이상에서야 1.000, Flirds-1st는 전 구간 0.000.)* **출처**: `runs/removal_dose/rundirs/1B_silo5_{noisy,frrand,poison}_dose_*`.

### AdamW 브리지 — external validity `[제외]` ● 3-seed

> **세팅**: 같은 anchor5 무대에서 **optimizer만 SGD(mom=0) → AdamW로 교체**. 우리 실험 전체가 SGD 상수-lr이라, 문헌 표준(AdamW)에서 결론이 유지되는지 보는 외적 타당도 점검. 지표 = Spearman vs (b).

| 방법 | SGD (기준) | AdamW | Δ |
|---|---|---|---|
| **Flirds** | **1.000±0.000** | <u>0.767±0.189</u> | −0.233 |
| Flirds-1st | <u>1.000±0.000</u> | 0.700±0.216 | −0.300 |
| loss-heur | 1.000±0.000 | 0.967±0.047 | −0.033 |
| GTG | 1.000±0.000 | **0.900±0.082** | −0.100 |
| Banzhaf | 1.000±0.000 | 1.000±0.000 | 0.000 |
| Fed-LOO | 1.000±0.000 | 0.933±0.094 | −0.067 |
| FedSV | 0.700±0.163 | 0.700±0.163 | 0.000 |
| ComFedSV | 0.500±0.436 | 0.567±0.125 | +0.067 |
| ShapleyFL | 0.600±0.424 | 0.267±0.236 | −0.333 |
| FedIF | 0.000±0.454 | 0.233±0.287 | +0.233 |
| **(a)oracle vs (b)** | 0.933±0.047 | **−0.533±0.330** | **−1.467** |

> **왜 제외인가**: AdamW에서 Flirds가 1.000 → 0.767로 떨어지는 건 사실이지만, **같은 셀에서 (a) 재학습 오라클과 (b) in-run 오라클의 일치도 자체가 0.933 → −0.533으로 붕괴**한다. 즉 AdamW에선 *두 정답이 서로 다른 답을 주는* 상태라 "Flirds가 틀렸다"고 읽을 수 없다 — 모멘텀·적응적 스케일링이 in-run 분해의 전제(가중 고정 게임)를 깨는 쪽에 가깝다. 이 갭을 제대로 다루려면 AdamW용 게임 정의부터 다시 세워야 해서 **스코프 밖으로 뺐다**(데이터는 존속). **출처**: `runs/removal_dose/rundirs_trackd/1B_anchor5_{removal,adamw}_seed{0,1,2}`.

---

## 4-공통 (모델-무관)

### φ 부호 감사 — 게이팅의 작동 전제 `[부록]` ⟐ 파생(3-seed rundir 전수)

> **무엇**: 309 rundir · 73,288행 전수에서 **클라별 누적 φ의 부호**를 세었다. 부호 게이트(P1: 누적 φ>0만 참여)의 두 전제가 실제로 성립하는지 — ① **clean 클라를 오배제하지 않는가**(clean 쪽 φ≤0 비율 = 오발화율의 상한) ② **오염 클라를 발화시키는가**(오염 쪽 φ≤0 비율 = 회수 가능성의 상한) — 를 개입 실험과 **독립적으로** 검증하는 표다. 개입 결과(recovery·parity)의 상한을 여기서 미리 읽을 수 있다.
> **세팅**: canonical variant · Llama-3.2-1B · 오염축 {answer-swap@0.7, free-rider-zero} + clean 대조 · seed{0,1,2}. `φ≤0` = exact-0 또는 음수 = **τ=0 게이트가 배제 판정**하는 조건. exact-0은 별도 병기(명제 P2의 0-공리 성립 여부).
> ⚠ ComFedSV는 silo5·iid5(N=5 전원참여)에 미산출 → 해당 칸 `–`. device100에서 **(b)·GTG·FedSV·ShapleyFL은 앵커 α=0.5 셀에만 존재**(나머지 α는 exact (b) 불가·해당 baseline 미실행)라 클라-행 수가 나머지의 1/9이다(275 vs 2,475).

**표 A — clean 클라이언트 φ≤0 비율 (%) — 오배제 위험, 낮을수록 좋음** (● 3-seed)

| 방법 | silo5 clean | silo5 swap | silo5 frzero | device100 swap | device100 frzero |
|---|---|---|---|---|---|
| **(b)oracle** | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |
| Flirds | <u>0.0</u> | <u>0.0</u> | <u>0.0</u> | <u>0.0</u> | <u>0.0</u> |
| Flirds-1st | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| loss-heur | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| FedIF | 20.0 | 0.0 | 0.0 | 2.5 | 1.7 |
| GTG | 0.0 | 0.0 | 0.0 | 3.3 | 1.8 |
| FedSV | 0.0 | 0.0 | 0.0 | 5.1 | 4.7 |
| ComFedSV | – | – | – | **51.4** | **51.4** |
| ShapleyFL | 13.3 | 0.0 | 0.0 | 4.0 | 2.9 |
| *(클라-행 수 n)* | *15* | *24* | *24* | *275~2,475* | *275~2,475* |

**표 B — 오염 클라이언트 φ≤0 비율 (%) — 게이트 발화, 높을수록 좋음. 괄호=그중 exact-0 비율** (● 3-seed)

| 방법 | silo5 swap | silo5 frzero | device100 swap | device100 frzero |
|---|---|---|---|---|
| **(b)oracle** | **0.0** | **100.0** (100.0) | **0.0** | **100.0** (100.0) |
| Flirds | <u>0.0</u> | <u>100.0</u> (100.0) | <u>0.0</u> | <u>100.0</u> (100.0) |
| Flirds-1st | 0.0 | 100.0 (100.0) | 0.0 | 100.0 (100.0) |
| loss-heur | 0.0 | 100.0 (100.0) | 0.0 | 100.0 (100.0) |
| FedIF | 100.0 (100.0) | 100.0 (100.0) | 36.3 (36.3) | 100.0 (100.0) |
| GTG | 100.0 (0.0) | 100.0 (0.0) | 26.7 (0.0) | 100.0 (0.0) |
| FedSV | 100.0 (0.0) | 100.0 (0.0) | 33.3 (0.0) | 100.0 (0.0) |
| ComFedSV | – | – | 38.5 (0.0) | 34.8 (0.0) |
| ShapleyFL | 100.0 (0.0) | 100.0 (100.0) | 26.7 (0.0) | 73.3 (73.3) |
| *(클라-행 수 n)* | *6* | *6* | *15~135* | *15~135* |

> **읽기 세 줄**.
> ① **free-rider(zero)는 부호 게이트의 이상적 작동 칸**: same-game 계열 + FedIF가 **100% exact-0**(bit-exact 0.0 — 명제 P2의 0-공리) → 게이트가 100% 발화하고 clean 오배제는 0.0%다. 개입 실험의 "frzero recovery 1.000·precision 1.000"([[flirds-results-downstream]])이 여기서 이미 예고된다. renorm 계열도 발화는 하지만 **exact-0이 아니라 음수**(GTG·FedSV 0.0% exact-0)라 0-공리는 성립하지 않는다. ⚠ **정정**: 종전 감사 요약의 "renorm은 exact-0 아님"은 GTG·FedSV·ComFedSV엔 맞지만 **ShapleyFL엔 부분적으로 틀리다** — silo5 frzero에선 100% exact-0이고 device100에선 73.3%로 갈린다(나머지 26.7%는 양수). EMA 평활이 0-업데이트를 그대로 통과시키는 셀이 있다는 뜻이라, ShapleyFL은 "조건부 0-공리"로 읽어야 한다.
> ② **answer-swap은 부호 게이트의 작동영역 밖**: (b)oracle·Flirds·Flirds-1st·loss-heur 모두 오염 클라에게 **φ>0을 준다**(발화 0.0%). 이건 추정 오차가 아니라 **게임의 답** — (b)로 채점해도 같다. 그래서 noisy 회수는 게이트가 아니라 연속 가중(P3)으로 가야 한다. dose 감사에서 0-교차 지점은 nr≈3.44로 **도달 불가**(nr 정의역이 (0,1])라 이 결론은 강도를 올려도 안 바뀐다.
> ③ **ComFedSV의 clean 51.4%가 renorm 파국의 근원**: 오염이 무엇이든 clean 클라의 **절반을 음수로 찍는다** → 게이트를 씌우면 clean을 절반 내쫓는다. [[flirds-results-downstream]] 정책 축의 renorm online −2.6~−3.3 파국과 정확히 대응하는 수치다. GTG·FedSV·ShapleyFL도 device100에서 clean 오배제가 1.8~5.1%로 same-game(0.0%)과 갈린다.
>
> ⚠ **CNN 레그는 이 감사에 없다**: 감사 스냅샷의 CNN 슬라이스(`scale=cnn`)는 재편성 전 C1 시나리오 집합(iid·label_flip·feature_noise·label_skew·quantity_skew)이라 **free-rider-zero·gradient-noise 칸이 존재하지 않는다**. CNN 오염축 정렬 재실행([[flirds-paper-experiment-plan]] G2·G8) 후 재감사 대상.
> **출처**: `runs/track_g/audit/sign_table.csv`(`variant==canon` 필터 → `contribution` 부호 집계) · `SIGN_AUDIT.md`.

### β 통일 재실행 provenance `[각주]` ⟐ 파생/폐기

> ShapleyFL EMA β 0.5→0.3 통일 재실행·대조. **현재 폐기**(수록 대상 전부 제외). CNN 120셀 = β0.5-era 산출(β-불변 canon 미확보 실측). β는 config에 미기록 → mtime·커밋이 유일 근거. **출처**: `runs/rerun_beta03/figures/{beta_provenance,beta_contrast_3b}.csv`.

---

## 출처·재생성

- 2차항: `runs/track_c/c2fid`(grad-noise) · `runs/probe_signal/cnn_c1`(k-sweep) · `runs/probe_signal/figures/llm_probe_summary.csv`(std50k5 rank sweep).
- lever: `runs/probe_signal/figures/{cnn_c1_realness,llm_probe_summary}.csv`.
- removal·dose·AdamW: `runs/removal_dose/{rundirs,rundirs_cnn,rundirs_trackd}`.
- Taylor: `runs/measured_2026-07/taylor/llama1b_r10_seed{0,1,2}/summary.json`(pooled resid1/resid2).
- **φ 부호 감사**: `runs/track_g/audit/sign_table.csv` → `variant=="canon"` 필터 후 (scale, regime, threat, method, corrupt)별 `contribution` 부호 집계(양수 / exact-0 / 음수). β: `runs/rerun_beta03`. TF32: `runs/measured_2026-07/tf32_ab`.
- **◐ seed0만**: LLM lever의 lr·steps 셀(rank16·lr1e-3·10st만 3-seed) · std50k5 rank32/64.
- 축 지도: [[flirds-experiment-axis-map]] (구 카탈로그 §4·§5 = git 이력)
