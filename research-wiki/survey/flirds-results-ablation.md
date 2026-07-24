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

> **세팅**: cifar10 × {iid, label-flip} × 폭{0.5,1,2,4} × 참여{0.2,0.5,1.0} × 3seed. "신호 크기 lever가 fidelity를 만드나" 검증. *↔ LLM 짝*
>
> **판정**: 폭·참여 lever는 IID서 **cross-seed 실재 신호를 못 만든다**(xseed ρ ≈ 0, 예: iid w0.5 k{0.2/0.5/1.0} = 0.022/−0.022/0.034). fidelity(Flirds vs (b))는 lever 전반 유지(**Taylor tradeoff 없음**). 참여는 별도 역할 — 짧은 지평서 1차항을 흐리고 방법 구별을 만든다(§4①). **출처**: `runs/probe_signal/figures/cnn_c1_realness.csv`(xseed_rho_mean).

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

> **세팅**: std50k5 부분참여(N=50, 5/50 = 클라당 참여 희박). Flirds(2차) vs 1차-계열 fidelity vs (b). *↔ CNN 짝*

| 방법 | Spearman vs (b) |
|---|---|
| **Flirds** | **+1.000** |
| Flirds-1st | +1.000 |
| ComFedSV | 음수 붕괴 |
| ShapleyFL | 음수 붕괴 |
| FedIF | 음수 붕괴 |

> 부분참여 5/50서 **Flirds/Flirds-1st만 +1.000 유지**, uniform-subset(ComFedSV/ShapleyFL)·FedIF는 음수로 붕괴 = 부분참여 스트레스에서 Taylor 계열 생존. LLM은 R=200으로 클라당 참여 횟수가 충분해 Flirds-1st도 버팀(CNN R=10 짧은 지평선 붕괴와 대비). 상세·수치는 구 카탈로그 §4.2(git 이력).

### A축 용량 lever probe — LLM `[본문 §5.6②]` ● 핵심축 3-seed(나머지 seed0)

> **세팅**: 1B, rank{16,32,64}·lr·steps·참여·noise lever. *↔ CNN 짝*
>
> **판정**: 어느 lever도 IID-clean cross-seed 실재 신호를 못 만든다(lr는 공통 shift ~1.3×만, 클라 간 분리는 seed 분산에 묻힘). fidelity(Flirds vs (b))는 lever 전반 **1.000**(HVP가 rank·lr↑에 강건). 신호는 A축(용량)이 아니라 **B축(비IID·오염)**이 만든다(non-IID clean ρ 0.87 vs IID 0.13). **출처**: `runs/probe_signal/figures/llm_probe_summary.csv`.

### Removal-curve — LLM `[본문 §5.6③]` ● 3-seed

> **세팅**: silo5 N=5, worst-first vs best-first 제거 후 val-loss. *↔ CNN 짝*

| 위협 | Flirds ρ(vs b) | worst-first Δval-loss | best-first Δval-loss | 판정 |
|---|---|---|---|---|
| noisy | +1.00 | **+0.0076** | −0.0084 | ✅ worst-first 제거가 loss 내림 |
| frrand | +1.00 | +0.0071 | −0.0015 | ✅ |
| frzero | +1.00 | +0.0067 | −0.0016 | ✅ |

> **출처**: `runs/removal_dose/rundirs`(A2).

### Dose-response · AdamW 브리지 · Taylor 물리잔차

- **Dose-response** `[보조]` ● 3-seed: φ 탐지 문턱 vs 오염강도(silo5 noisy nr·frrand dm 스윕) — noisy nr≥0.25↑·FR 전 배율 문턱 통과. `runs/removal_dose/rundirs`(B).
- **AdamW 브리지** `[제외]` ● 3-seed: SGD→AdamW optimizer 갭서 fidelity **+0.77**(SGD 1.000 대비 저하; (a)↔(b) 자체 −0.53 괴리 caveat = external-validity 한계). `runs/removal_dose/rundirs_trackd`(A1/D).
- **Taylor 물리잔차 (명제 P3)** `[보류·부록A]` ● 3-seed: 2차 잔차 ≈ 1.3~1.8e-6 → **2차 근사가 1차보다 ~2.7~3.4× 작음**(2차항 추가의 물리적 정당화). closed↔Flirds(2차) max|Δφ| ~1e-10. `runs/measured_2026-07/taylor/`.

---

## 4-공통 (모델-무관)

### φ 부호 감사 (게이팅 전제) `[전제]` ⟐ 파생

> 309 rundir 전수 φ 부호 감사(73,288행). **판정**: ①clean 오배제-0 전제 성립(canonical clean 전 method·클라 누적 φ 양수 → τ=0 게이트 무발화) ②frzero **exact-0**(Flirds·(b)·Flirds-1st·loss-heur·FedIF bit-exact 0.0; renorm은 exact-0 아님) ③noisy엔 sign-게이트 작동영역 없음(0-교차 nr≈3.44 도달불가) ④frrand 누적부호 = seed-코인플립. **출처**: `runs/track_g/audit/{SIGN_AUDIT.md,sign_table.csv}`.

### β 통일 재실행 provenance `[각주]` ⟐ 파생/폐기

> ShapleyFL EMA β 0.5→0.3 통일 재실행·대조. **현재 폐기**(수록 대상 전부 제외). CNN 120셀 = β0.5-era 산출(β-불변 canon 미확보 실측). β는 config에 미기록 → mtime·커밋이 유일 근거. **출처**: `runs/rerun_beta03/figures/{beta_provenance,beta_contrast_3b}.csv`.

---

## 출처·재생성

- 2차항: `runs/track_c/c2fid`(grad-noise) · `runs/probe_signal/cnn_c1`(k-sweep) · `runs/probe_signal`(std50k5).
- lever: `runs/probe_signal/figures/{cnn_c1_realness,llm_probe_summary}.csv`.
- removal: `runs/removal_dose/{rundirs,rundirs_cnn}`. Taylor: `runs/measured_2026-07/taylor`.
- 감사: `runs/track_g/audit`. β: `runs/rerun_beta03`.
- 축 지도: [[flirds-experiment-axis-map]] (구 카탈로그 §4·§5 = git 이력)
