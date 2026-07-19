---
type: survey
title: "Baseline ↔ 원 논문 결과·세팅 대조 검증"
created: 2026-06-22
updated: 2026-07-19
tags: [survey, baselines, verification, reproduction]
---

# Baseline ↔ 원 논문 결과·세팅 대조 검증

> 우리 실험에서 나온 각 baseline의 수치가 그 방법의 **원 논문이 보고한 결과**와 얼마나
> 일치하는지, 그리고 **실험 세팅**이 얼마나 맞물렸는지를 baseline별 테이블로 정리한 문서.
> 최초 2026-06-22 작성 → **2026-07-19 전면 갱신**: 이후 완료된 실험(7B 3-seed · Fed-LOO/N=10
> oracle E4·E5 · frdelta E7 · B축 오염×비IID 매트릭스 · std50k5 참여 probe · removal-dose ·
> AdamW 브리지 · Taylor 물리잔차 · φ-부호 감사 Stage 0 · Ripple 정밀 감사 · C6 비용버그 수정)을
> 전 baseline 표에 반영.
>
> 출처: 우리 결과 = [[flirds-experiment-results-overview]](전 트랙 rundir 전수 집계 정본; 수치
> 재생성 경로 포함) + 개별 rundir(`runs/track_d` · `runs/track_c/RESULTS.txt` ·
> `runs/phase2_matrix/rundirs*`(frdelta 포함) · `runs/probe_signal` · `runs/removal_dose` ·
> `runs/measured_2026-07` · `runs/track_g/audit`). Ripple 심층 = [[ripple-audit]].
> 값-수준 fidelity.csv·master_metrics.csv는 gitignored 파생물(각 `make_fidelity.py` /
> `make_analysis.py`로 재생성). 원 논문 = `research-wiki/wiki/sources/*.md` (ingest 노트).
>
> **자매 문서**: [[baseline-selection-audit-2026-07-02]] — *다른 질문*을 본다: 애초에 baseline을
> **제대로 골랐는가**(선정의 정당성·완결성; 후보를 A/B/C로 판정). 이 문서(06-22 작성, 07-19 갱신)는
> *이미 고른* baseline의 **수치·거동**이 원 논문과 맞는지. 즉 07-02="맞게 골랐나", 이 문서="고른 게
> 맞게 굴러가나".

---

## 0. 이 검증의 성격 (먼저)

세 가지 구조적 이유로 **"원 논문 숫자 = 우리 숫자" 직접 재현은 대부분 성립하지 않는다.** 그래서
아래 테이블의 "판정"은 *수치 재현*이 아니라 대부분 *이론 주장·정성 거동의 일치* 여부다.

1. **무대가 다르다.** 우리는 *LLM FL에서 φ가 exact Shapley oracle에 얼마나 충실한가(fidelity)*를
   잰다. 원 논문 대부분은 *CV 이미지분류/표/중앙집중*에서 *정확도·탐지정확도·런타임*을 잰다.
2. **재현이 아니라 port.** baseline의 원 실험을 다시 돌린 게 아니라 알고리즘을 우리 무대로 옮겼다.
3. **노트에 원 논문 hard number가 거의 없다.** valuation 9종 중 6종(GTG·FedSV·S-FedAvg·Banzhaf·
   ShapleyFL·Ripple)은 노트에 **정성 서술만** 있고 수치가 없다 → 이론·정성 대조까지만 가능.
   (정량 대조를 원하면 `research-wiki/raw/papers/`의 원본 PDF 재추출 필요.)

판정 범례: **✓ 일치**(주장/거동 재현) · **◐ 부분**(맞는 레짐선 재현, 다른 데선 어긋나되 설명됨) ·
**✗ 불일치**(주장이 우리 무대로 전이 안 됨). 세팅 일치도: **높음 / 중간 / 낮음 / 불명**.

---

## 1. 요약 표

| 방법          | 계열        |     세팅 일치     |  결과 판정  | 한 줄                                                                       |
| ----------- | --------- | :-----------: | :-----: | ------------------------------------------------------------------------ |
| Banzhaf     | valuation |  해당없음(중앙↔FL) |  **✓**  | 값-수준 fidelity 최강 유지(CNN Pearson 0.998·probe 0.996·AdamW +1.000 유일)        |
| GTG-Shapley | valuation |      낮음       |  **✓**  | 가법 영역서 exact 수렴(≈1.0), 대규모·저참여서 우아한 열화(0.78~0.98)                         |
| FedSV       | valuation |      낮음       |  **◐**  | 오염 랭킹은 맞으나 rank-fidelity 낮음(MC 분산 xseed 0.29 실측); renorm이 null-player 0 깨뜨림 |
| ShapleyFL   | valuation |      낮음       |  **◐**  | exact fidelity 낮은 게 정상(surrogate); 탐지·poison 무력화는 작동, 저참여(5/50)선 음수 붕괴   |
| FedIF       | valuation |  **중간**(최근접)  |  **◐**  | Shapley 아님(설계), 속도·robust 일치; 단 delta-재활용 FR엔 완전 오정렬(0.0)                 |
| ComFedSV    | valuation |      낮음       |  **✗**  | low-rank 가정 위배 → fidelity 붕괴(홈 레짐인 대N 부분참여 std50k5서도 음수)                  |
| Ripple      | valuation |      불명       |  **✗**  | 우리 valuation-only 무대선 최고비용(Flirds의 ~33×); 단 "62×"는 조건부 주장이라 모순 아님(감사 확정)  |
| FLDetector  | detector  |      낮음       | **✓✓**  | 강점(crafted 1.0)+한계(noisy/non-IID 침식·off-threat frdelta 0.0) 둘 다 재현        |
| FLTrust     | detector  |      낮음       |  **✓**  | cosine-to-root가 scaled/free-rider 잡음(1.0); 정렬된 delta-재활용은 통과(메커니즘 정합 한계) |
| STD-DAGMM   | detector  | **device=높음** | **✓✓**  | 정량 일치 2건 — N=100 FR 0.87–0.96 ≈ 원 0.91–0.96 + 자체 위협 delta 1.0 유일 탐지       |
| FedDQC      | detector  |  **높음**(동종)   |  **✓**  | 유일한 LLM-LoRA 무대, noisy-품질서 강함(0.92–1.0); IID 배경 유리까지 실측 재현                |
| Bagdasaryan | attack    |      중간       |  **✓**  | model-replacement 재현(silo5 ASR≈1.0, clean 보존); 공격자-제거 시 무력화(결속성 정합)       |
| Xu          | attack    |      낮음       |  **✗**  | 다른 메커니즘(generative+scaled) → 재현이라 부를 수 없음                                 |

> **세팅이 맞물린 칸에서만 정량 대조가 정당하다**: FedIF(C2)=중간, STD-DAGMM(device100)=높음
> (+frdelta는 위협까지 동종), FedDQC=높음. 거기서 결과도 잘 맞는다(§4).

---

## 2. 우리 실험 무대 세팅 (참조용 — 아래 각 표의 "우리" 열이 가리키는 무대)

| 트랙                        | 모델                     | N (참여)      | rounds        | local   | batch        | lr / opt          | 데이터                 | 분할                          | seeds | oracle                                       |
| ------------------------- | ---------------------- | ----------- | ------------- | ------- | ------------ | ----------------- | ------------------- | --------------------------- | ----- | -------------------------------------------- |
| **C1** (CNN fidelity)     | LeNet5 / FedSVCNN      | 10 (full)   | 10            | E=5     | 64           | 0.01 / SGD m=0    | MNIST, CIFAR-10     | iid+4 non-IID 시나리오          | 3     | (a)&(b) exact 2¹⁰                            |
| **C2** (CNN intervention) | FedSVCNN / LeNet5      | 100 (C=0.1) | 120           | E=5     | 64           | 0.01 / SGD m=0    | CIFAR-10, FMNIST    | iid / dir1(α=1) / shard     | 3     | — (성능/AUROC)                                 |
| **D std20** (LLM 표준)      | Llama-3.2 1B·3B / Llama-2 7B | 20 (2/rd)   | 200           | 10 step | 16/8/4       | 1e-3 / SGD m=0    | alpaca-gpt4 20k     | **clean·IID**               | 3     | (b) per-round exact                          |
| **D anchor5**             | Llama-3.2 1B·3B / Llama-2 7B | 5 (full)    | 30            | 10 step | 16/8/4       | 1e-3 / SGD m=0    | alpaca-gpt4         | clean·IID                   | 3     | (b) 2⁵ + **(a) retrain val-loss fp32(1B만)**  |
| **D-E5 anchor10** (N=10)  | Llama-3.2 1B           | 10 (full)   | 30            | 10 step | 16           | 1e-3 / SGD m=0    | alpaca-gpt4         | clean·IID                   | 1     | **(b) exact 2¹⁰**(1024 열거)                   |
| **probe std50k5** (저참여)   | Llama-3.2 1B           | 50 (5/rd)   | 200           | 10 step | 16           | 1e-3 / SGD m=0    | alpaca-gpt4         | clean·IID                   | 1–3*  | (b) per-round exact 2⁵                       |
| **P2 silo5** (robustness) | Llama-3.2 1B           | 5 (full)    | 10            | —       | 16(poison 8) | 1e-3(poison 2e-3) | 5-domain cross-silo | non-IID(domain-disjoint)    | 3     | (b) exact 2⁵                                 |
| **P2 iid5** (B축 매트릭스)     | Llama-3.2 1B           | 5 (full)    | 10            | —       | 16(poison 8) | 1e-3(poison 2e-3) | alpaca IID          | **clean·IID ↔ 오염**          | 3     | (b) exact 2⁵                                 |
| **P2 frdelta** (E7)       | Llama-3.2 1B           | 5 (full)    | 10            | —       | 16           | 1e-3              | 5-domain cross-silo | non-IID + delta-재활용 FR      | 3     | (b) exact 2⁵                                 |
| **P2 device100**          | Llama-3.2 1B           | 100 (K=10)  | 30(poison 60) | —       | 16(poison 8) | 1e-3(poison 2e-3) | 5-domain pool       | Dirichlet α∈{0,.01,.1,.5,5} | 3     | α=0.5만 (b) per-round; 그 외 Flirds-proxy       |
| **P2 3B**                 | Llama-3.2 3B           | 5 (full)    | 10            | —       | 16           | 1e-3(poison 2e-3) | 5-domain            | non-IID                     | 1     | (b) exact 2⁵, coalition off                  |

\* std50k5는 r16 셀만 3-seed(seeds 1-2는 Flirds·Flirds-1st만 채점하는 경량 스위트; coalition류는 seed0).
LoRA r16/α32, fp32, momentum=0 공통. **β0.3 provenance 주의**: ShapleyFL EMA β는 논문값 0.3으로
통일 재실행 중이나 **7B 6셀 + device100 anchor 3셀은 아직 β0.5-era**(deferred; overview §4.2-9),
1B track_d·CNN도 재실행 커밋 없음(단 3B 전후 대조에서 β 변경 효과 = 재실행 노이즈 플로어 수준).

---

## 3. 기여도 추정 (valuation) 계열

### 3.1 Banzhaf — 판정 ✓
Data Banzhaf (Wang & Jia, arXiv 2205.15466). **중앙집중** 데이터 가치(비FL).

**세팅 비교**

| 항목     | 원 논문                                      | 우리                          |
| ------ | ----------------------------------------- | --------------------------- |
| task   | 중앙집중 bad-data detection / weighted-sample | FL 클라 기여도(exact 2ᴺ Banzhaf) |
| 데이터/모델 | 노트 미기록                                    | CNN N=10, LLM N=5           |
| 추정기    | **MSR estimator**(불편추정)                   | **exact**(추정기 미사용)          |
| 지표     | ranking stability, bad-data detection     | Spearman/Pearson vs oracle  |

**결과 비교**

| 지표 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| 값 품질 순위 | Banzhaf > Beta-Shapley > Shapley > LOO (수치 미기록) | CNN vs (b) Pearson **0.998**(전 방법 최강; probe 72셀 Spearman 0.996±.009, **저참여 k=0.2에서도 1.000**), LLM anchor5 1B/3B/7B 전부 **+1.000**, silo5 +1.000 | ✓ |
| 학습 확률성 하 안정성 | "Shapley보다 훨씬 높음"(MSR 기준) | **AdamW 브리지 vs (b) +1.000±.000 = 전 방법 중 유일 무손상**(Flirds +0.77, GTG +0.90); C1 cross-seed 순위 안정성 0.508 ≈ (b) Shapley-oracle 0.518(동급 — 단 seed가 데이터 분할까지 바꿔 원 논문의 "학습 노이즈만" 설정과 다름) | ✓(부분) |

> 단서: 논문 핵심 기여인 *MSR 불편추정*은 우리가 exact를 써서 테스트한 게 아니다. 값 자체의
> 품질만 확인 — 그 한도 내에서 "Banzhaf가 가장 깨끗한 값-수준 proxy"는 전 무대 일치. 새 경계 1건:
> **IID-poison(clean-보존 backdoor)에선 Banzhaf도 (b)·loss-heur와 같이 AUROC 0.00**(iid5 B축) —
> 값 품질 문제가 아니라 val-loss 게임 공통 속성(§6).

### 3.2 GTG-Shapley — 판정 ✓
GTG-Shapley (Liu et al., ACM TIST 2022, arXiv 2109.02053).

**세팅 비교**

| 항목 | 원 논문 | 우리 |
|---|---|---|
| task/데이터 | 이미지분류(데이터셋 미기록) | CNN MNIST/CIFAR + LLM 1B/3B/7B |
| N / rounds | 미기록 | CNN N=10, LLM N=5/10/20/50/100 |
| 분할 | 미기록 | iid + non-IID |
| 지표 | utility-eval 수 대비 추정 정확도 | Spearman/Pearson vs exact oracle |

**결과 비교**

| 무대 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| 가법 영역 | "경쟁 federated-Shapley보다 정확"(수치 없음) | std20 Spearman 0.975(1B)/0.990(3B)/0.977(7B), Pearson 0.989–0.997; anchor5 1B·7B **+1.000**(3B 0.967); silo5 1.000(poison Sp만 0.867) | ✓ |
| 대규모·저참여 비가법 | "guided-MC 초기 고분산"(정성) | device100 anchor Spearman 0.784~0.843(위협별; noisy Pearson 0.892) · **std50k5(5/50) 0.983**(저참여 생존) · CNN full-2¹⁰ 0.50~0.57(큰 게임 근사 한계) — 우아한 열화 | ✓ |
| 실용 순위 품질 | (해당 측정 없음) | removal-curve 엄밀 일치 8/9 셀(이탈 시에도 곡선 차 ≤0.002) | ✓ |

> 새 단서(φ-부호 감사): noisy dose에서 GTG 누적 φ의 0-교차(~nr0.76)는 **per-round renorm 오차의
> 부산물**(진짜 게임값은 nr≤1에서 0-교차 없음) — GTG 값을 부호-기반 정책에 쓸 때 주의.

### 3.3 FedSV — 판정 ◐
FedSV (Wang et al. 2020, *A Principled Approach…*, arXiv 2009.06192).

**세팅 비교**

| 항목 | 원 논문 | 우리 |
|---|---|---|
| 데이터/모델 | MNIST/CIFAR-10, full-model | CNN N=10, LLM 1B/3B/7B |
| 분할 | IID + non-IID | iid + non-IID |
| 지표 | noisy/backdoor 랭킹 vs Federated-LOO | Spearman/Pearson + 탐지 AUROC |
| 보장 | Theorem 1: per-round 공리 유일 만족 | (동일 알고리즘 port) |

**결과 비교**

| 지표 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| 오염 랭킹 | "noisy/backdoor를 낮게, LOO보다 우수"(수치 없음) | silo5 전 위협 탐지 AUROC **1.000**; frdelta도 순위 추종(Sp 0.933) | ✓ |
| rank-fidelity | — | std20 0.910(1B; 3B 0.966·7B 0.968 = 스케일↑ 개선)/Pearson 0.959; anchor5 0.700(1B)/0.667(3B)/0.933(7B); **poison Spearman +0.367**; std50k5 0.910(저참여 생존); CNN full 0.302 | ◐ |
| MC 분산(원인) | (per-round MC 근사) | C1 cross-seed 안정성 **0.289**(vs (b) 자체 0.518, Flirds 0.547) = "per-round MC 분산" 서술의 정량 근거 | ◐(정합) |
| non-IID 약점 | "rare 클라 음수 SV" 단서 | device100 α=0.5 +0.752 (Flirds +1.0 대비 낮음) | ◐(정합) |
| null-player 공리 | Theorem 1(공리 만족 주장) | φ-부호 감사: free-rider(zero) φ가 **exact-0 아님**(작은 음수; per-round renorm 탓 — Flirds/(b)/Banzhaf/loss-heur는 bit-exact 0) = port에서 공리의 수치 위반 실측 | ◐ |

> 참고(LOO 앵커): 원 논문의 비교 대상 Federated-LOO를 E4에서 실측 — near-additive 무대(std20·
> anchor5·N=10)에선 Fed-LOO도 +1.000이라 "LOO보다 우수" 주장을 가릴 수 있는 무대가 아니다(구별은
> 비가법·저참여 무대에서만 생김).

### 3.4 ShapleyFL — 판정 ◐ (설계상)
ShapleyFL (Sun et al., KDD 2023). EMA β=0.3(논문값)으로 통일 재실행; 7B·device100 anchor 셀은
아직 β0.5-era(§2 각주).

**세팅 비교**

| 항목 | 원 논문 | 우리 |
|---|---|---|
| 데이터 | MNIST/FMNIST/CIFAR/Fed-ISIC2019 | CNN + LLM 1B/3B/7B |
| 성격 | **surrogate** Shapley(joint multi-round 아님) | 동일 surrogate port |
| 지표 | noisy/adversarial 하 최종정확도 | Spearman/Pearson vs exact + AUROC |

**결과 비교**

| 지표 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| exact 대비 fidelity | (해당 측정 없음) | std20 0.194(1B)/0.211(3B)/0.406(7B, β0.5-era); anchor5 0.700(1B)/0.167(3B)/0.833(7B) — **낮은 게 설계상 정상** | ◐ |
| 오염 하 효용 | "FedAvg+random보다 정확도↑" | silo5 탐지 AUROC **1.000**; removal poison서 worst-1 제거 → **ASR 1.0→0.0 무력화 성공**; CNN C2 grad_noise acc 0.645(전 arm 최고 칸 존재) | ✓ |
| 저참여·변별력(새 경계) | (주장 없음) | **std50k5(5/50) Spearman −0.06~−0.09 음수 붕괴**; CNN removal 변별 최저(ρ +0.07~0.26, acc 분리 ≈0); xseed 안정성 0.124(전 방법 최하 — EMA surrogate 분산) | ✗(새 관측) |

### 3.5 FedIF — 판정 ◐ (설계상) · 세팅 최근접
FedIF (Tang et al., arXiv 2509.25560, 2025). 영향도→가중(Shapley 아님).

**세팅 비교** — 전 baseline 중 우리 C2와 가장 근접

| 항목 | 원 논문 | 우리 (C2) |
|---|---|---|
| 데이터/모델 | CIFAR-10+FMNIST, **CNN** | CIFAR-10/FMNIST, **CNN** ✓ |
| N / C / E | **100 / 0.1 / 5** | **100 / 0.1 / 5** ✓ |
| batch / lr / mom / T | 16 / 1e-3 / **0.9** / 100 | 64 / 1e-2 / **0** / 120 |
| 분할 | Dirichlet **α=1** | dir1 = **α=1** ✓ |
| 지표 | label/grad-noise 하 정확도 + agg wall-clock | (C2) 개입 정확도/AUROC · (fidelity) Spearman |

**결과 비교**

| 지표 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| agg 속도 | **0.2s/rd vs AFedSV 70–92s (450×)** | 전 무대 최저가군(Flirds-1st급): silo5 ~37s · 1B anchor5 232s·std20 1534s · device100 ~51s | ✓(빠름 재현) |
| exact-Shapley fidelity | (해당 없음; Shapley 아님) | std20 0.157(1B)/0.203(3B)/0.480(7B); anchor5 0.067~0.200 — **낮은 게 설계상 정상** | ◐ |
| robust-under-noise | "comparable-better than AFedSV" | device100 noisy AUROC 0.57–0.97; silo5 전 위협 AUROC 1.000 | ✓ |
| 방향-정렬의 한계(새 경계) | (주장 없음) | **frdelta AUROC 0.000** — 재활용 delta를 최고 가치로 오정렬(1차 influence가 완전히 속음); **std50k5 −0.04~−0.08 붕괴**; removal은 스테이지 의존(LLM frzero 유일 낙오 ↔ CNN mnist acc 분리 최고 +0.0042) | ✗(새 관측) |

### 3.6 ComFedSV — 판정 ✗ (설명됨)
ComFedSV (Fan et al., ICDE 2022, arXiv 2109.09046).

**세팅 비교**

| 항목    | 원 논문                                            | 우리                           |
| ----- | ----------------------------------------------- | ---------------------------- |
| 데이터   | synth/MNIST/FMNIST/CIFAR-10 non-IID             | CNN + LLM 1B/3B/7B           |
| 규모    | 100 클라, 10개 noisy@30% flip, **partial**         | N=5/20/**50 partial**/100    |
| 핵심 가정 | utility 행렬 **low-rank**(convex 이론; VGG16까지 경험적) | **LoRA/transformer — 가정 위배** |
| 지표    | Spearman vs ground truth, Jaccard               | Spearman/Pearson vs oracle   |

**결과 비교**

| 지표 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| Spearman | "4개 데이터셋서 FedSV 근접·상회"(개별값 미기록) | CNN 0.348; LLM std20 **0.093**(1B)/**−0.137**(3B, β0.3 3-seed)/0.039(7B); anchor5 0.500~0.600; device100 모든 α ≈0(anchor −0.023); **std50k5(5/50) −0.08~−0.13** | ✗ |
| 홈 레짐 반론 차단 | (설계 중심 = 대N partial 참여) | std50k5 = 원 논문 참여형태에 가장 가까운 LLM 무대 — **거기서도 음수 붕괴** → "참여형태 탓" 반론 소거, low-rank 위배가 원인 | — |
| 원인 | (Theorem: (4δ/N)-fair, δ=완성오차) | low-rank 위배 → δ 폭증 → fidelity 붕괴(노트 사전 경고); xseed 안정성 0.198 | — |

> **판정의 축 구분(07-19 정밀화)**: 논문의 "Spearman vs ground truth"에서 GT = **주입 오염(노이즈
> 수준) 랭킹**이지 exact Shapley oracle이 아니다 — 우리 헤드라인 0.348은 논문이 잰 적 없는 더 엄한
> oracle-fidelity 축. **논문 자체 평가축(오염 랭킹·noisy 탐지)으로 보면 CNN에선 재현된다**: mnist
> label-flip에서 ComFedSV AUROC 0.98·오염랭킹 ρ 0.96 = FedSV(1.0/0.985) 근접("근접·상회" 중 근접 ✓;
> cifar10은 R=10 짧은 지평이라 둘 다 약함). 붕괴는 **LLM에서 두 축 모두**: oracle-fidelity 0.09~음수 +
> 오염축도 device100 noisy/FR AUROC 0.37–0.45(chance 이하). GT 정의 차이(retrain 등)로는 설명 안 됨 —
> ComFedSV의 자기 게임(u_t(S)=per-round recon test-loss 감소)이 곧 우리 (b) 계열이고, (a) retrain
> 기준으로도 0.338로 동일하게 낮다.
>
> 정직한 발견: **ComFedSV의 "FedSV 상회" 주장은 LLM 스케일로 전이되지 않는다** — 참여형태를
> 맞춰줘도(std50k5) 마찬가지.

### 3.7 Ripple — 판정 ✗ (감사로 원인 확정 — 모순은 아님)
Ripple Shapley (Zeng et al., AAAI 2026). sample-level, single-run. 심층 = [[ripple-audit]].
**[2026-07-19 결정] baseline 스위트에서 제외**(Yonghee) — 게임 불일치·from-logs 불가·코드 부재
(전면 자체 구현)·최고비용·실측 성능 저조 종합; 근거·리뷰어 Q&A = [[ripple-baseline-exclusion]].
아래 표는 원 논문 대조 기록으로 유지.

**세팅 비교**

| 항목 | 원 논문 | 우리 |
|---|---|---|
| 데이터 | 미기록(real-time pricing 응용) | CNN N=10 (+감사 캠페인서 LLM silo5 3-seed 실측) |
| 수준 | **sample-level** | **client-level** 적응 |
| 핵심 가정 | low-rank Jacobian-subspace | 동일 |
| 회계 | **학습 포함 누적 시간**, 느린 coalition-형 대비 | **valuation-only** wall-clock, from-logs 소비형 대비 |
| 지표 | 속도 at comparable accuracy | Spearman vs oracle + 런타임 |

**결과 비교**

| 지표 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| 속도 | **62×/49× speedup** — 감사로 측정 조건 확정: 소형 모델·학습 포함 누적·느린 coalition-형 베이스라인 대비(절대 주장은 plain training 대비 2.05×뿐) | 우리 회계(valuation-only)선 **전 무대 최고비용**: LLM silo5 3-seed 3,536s(2,366–4,363) = coalition-sweep의 ~6.6×·**Flirds의 ~33×**; CNN C1 MNIST ~1.1–2.3k s/CIFAR ~7.5–11.1k s(FL 학습 자체의 10–130×); E3 위상분리 — **Ripple 위상(자체 궤적+융합 valuation)** 이 셀 총시간의 ~95%인데 그중 순수 궤적(로그 생성) 몫은 **~0.4%**(CNN 계측) = 지배 비용은 valuation 자체 | ✗(우리 무대 전이 실패; 단 분모·무대가 달라 **양립** — "반박"이 아님) |
| 정확도 | "comparable"(미정량) | CNN pool Spearman 0.373(E3 iid 0.345/0.406 — 셀 최고 비용으로 same-game 최상위권 미달); LLM silo5 noisy AUROC 0.25–0.75 seed-불안정(free-rider는 1.0) | ✗ |
| 원인 | — | **(구 서술 정정)** "eigsh CPU-spin 불안정" 가설은 실측 반박 — eigsh는 정상·빠르게 수렴(tol=0의 ~2× matvec 비효율만 port 개선 여지). 실체 = ① 정상 수렴 eigsh를 클라×라운드만큼 반복하는 **방법 고유 고유분해 volume** ② **자체 궤적 필요**(온라인·클라 참여형 프로토콜이라 from-logs 재구성 불가 = 공유-로그 그리드 편입 불가; 회계 비대칭은 논문 충실성의 결과) | — |

> 구서버 수치 "~4,515s"(06-06)는 서버 이전으로 신규 B200 실측과 직접 비교 불가 — 정본 축은 동일
> run 내 방법 간 **비율**(6.6× coalition · 33× Flirds).
>
> **회계 공정성(비대칭 교정 실측, 07-19)**: Ripple만 자기 궤적이 타이머에 포함되는 비대칭이 실재하나,
> 어느 방향으로 통일해도 결론 불변 — ① Ripple에서 궤적을 빼면(valuation-only 통일) CNN 계측상 −0.4%뿐
> (LLM도 자체 궤적이 공유 궤적보다 작은 4r×4step이라 관대하게 공유 궤적치 410s를 통째로 빼도 ~29×
> Flirds); ② 반대로 전 방법에 공유 FL 궤적(~410s, acct 실측)을 더하면(논문식 end-to-end) 33×→**~6.8×**
> (vs Flirds)·6.6×→~3.7×(vs coalition)로 줄지만 **여전히 전 방법 중 최고비용**. 논문 자체 셋업에선
> 역방향 — valuation-only 환산 시 Ripple 우위가 94.8×/120.9×로 오히려 커짐. 즉 회계 정의 선택은
> 어느 쪽 결론도 뒤집지 못한다(감사 §2.3 요인 ③).

### 3.8 (참고) IRDS — Flirds의 기반, baseline 아님
In-Run Data Shapley (Wang et al., ICML 2024, arXiv 2406.11011). 중앙집중 GPT-2/Pythia, the Pile.

| 지표 | 원 논문 보고 | 우리(FL 확장) | 비고 |
|---|---|---|---|
| 2차항 효용 | "중앙 per-step선 1차 대비 이득 미미"(Appx E.2.2) | FL per-round선 2차항이 실측 이득: ① silo5 1B poison Flirds-1st AUROC **0.000** → 2차 포함 **0.917** ② CNN 저참여 k=0.2 Flirds-1st 0.305 vs Flirds **0.891** ③ Taylor 물리잔차 — 2차 근사가 1차보다 **~2.7–3.4× 작음**(1B 3-seed 실측) | 모순 아님: FL per-round multi-step ≠ 중앙 per-step (우리 thesis 확인·정량화) |
| 2차 방어의 한계(새) | — | **3B poison은 Flirds(2차)도 AUROC 0.000**(1-seed) = poison 방어는 스케일 취약; dose서 pf≤0.2 완전 회피·pf0.3–0.7 seed-불안정 전이대; removal서 Flirds top-1 제거 실패(ASR 유지) | 2차항 이득은 "부분 방어"로 서술해야 정확 |
| 음수 가치 | "Pile의 ~16% 음수 Shapley" | (해당 응용 없음; 부호 감사상 clean 무대 누적 φ는 전 클라 양수) | — |

---

## 4. 탐지기 (detector) 계열

### 4.1 FLDetector — 판정 ✓✓ (강점·한계 둘 다 재현, 정합 최고)
FLDetector (Zhang et al., KDD 2022, arXiv 2207.09209).

**세팅 비교**

| 항목 | 원 논문 | 우리 |
|---|---|---|
| 데이터/모델 | MNIST/CIFAR-10(ResNet20)/FEMNIST | LLM 1B/3B LoRA |
| 악성 비율 / non-IID | 28% / degree 0.5 | poison 1/5, noisy 1/5 ~ 1/100 |
| 위협모델 | **crafted-update**(noisy-honest 아님) | poison(crafted) + noisy/free-rider(zero·random·delta) |
| 지표 | DACC / FNR | AUROC |

**결과 비교**

| 위협 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| crafted(poison) | **DACC 0.85–1.0, FNR~0** (FEMNIST) | silo5 poison **1.000**, device100 poison 0.983–0.987, 3B 전 위협 1.000(1-seed) | ✓ |
| noisy-honest | "crafted만 탐지"(한계 명시) | silo5 noisy **0.75**, frzero 0.75 | ✓(한계 재현) |
| non-IID 침식 | "IID 가정서만 보장, 이질성서 침식" | device100 noisy **0.48–0.54** | ✓(침식 재현) |
| off-threat(새) | (위협모델 밖) | **frdelta AUROC 0.000** — 복제된 직전 글로벌 delta는 일관성 검사를 통과(예측-업데이트 정합) → crafted 스코프 밖 위협엔 무력 | ✓(스코프 한계 정합) |

### 4.2 FLTrust — 판정 ✓
FLTrust (Cao et al., NDSS 2021, arXiv 2012.13995). robust aggregation(가치 부여 아님).

**세팅 비교**

| 항목 | 원 논문 | 우리 |
|---|---|---|
| 데이터 | 6종(MNIST/FMNIST/CIFAR/CH-MNIST/HAR) | LLM 1B/3B |
| root | clean root 데이터 ~100 예시 | **root = val-gradient**(별도 데이터셋 없음) |
| 유사도 | ReLU-clipped cosine | **signed cosine** |
| 지표 | test-error / backdoor success | AUROC |

**결과 비교**

| 위협 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| scaled/backdoor | under-attack ≤0.04 err, backdoor ≤0.03, **90% malicious 견딤** | silo5 poison **1.000**, 3B 1.000; 단 **device100 poison 0.650(α0)/0.498(α0.5)** = cross-device 희석·tiny-val 무대선 약세 | ✓(silo)·◐(device) |
| free-rider | (cosine 정렬로 misaligned 차단) | zero/random 전 레짐 **1.000**; **frdelta 0.000** — 재활용 delta는 root 방향과 실제 정렬 → cosine 통과(메커니즘상 예측되는, 위협모델 밖 한계) | ✓(메커니즘 정합) |
| non-IID noisy 약점 | "benign-misaligned도 0으로 깎임" | device100 noisy 0.60(α0.01)–1.0 | ✓(한계 정합) |

### 4.3 STD-DAGMM — 판정 ✓✓ (정량 일치 2건 — 탐지기 중 대조 정합 최고)
STD-DAGMM (Lin et al., arXiv 1911.12560, 2019).

**세팅 비교**

| 항목 | 원 논문 | 우리 (device100 / frdelta) |
|---|---|---|
| 모델 | MNIST + 2-layer MLP | Llama-3.2-1B LoRA(5.6M→256 feature-hash) |
| **N / free-rider** | **100 / 20** | device100 = **100 / ~20%** ✓ · frdelta = 5 / 1 |
| 위협 | free-rider(zero/random/**delta/advanced-delta**) | freerider_zero / random / **delta(E7)** ✓ 위협 동종 |
| 지표 | AUC | AUROC |

**결과 비교**

| 무대 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| **N=100 free-rider** | **AUC 0.96(rnd5) / 0.91(rnd80)** (advanced-delta) | **frzero AUROC 0.87–0.96**, frrand 0.51–0.96; device100 poison도 0.983–1.000(update-이상으로 잡음) | ✓(수치 근접) |
| **delta-재활용(자체 위협)** | delta/advanced-delta 탐지가 논문의 존재 이유 | **frdelta AUROC 1.000±.000 (3-seed)** — 15개 방법 중 **유일 탐지**(FLDetector/FLTrust/FedIF 0.0; φ-게임 계열은 구조상 0.33) — 원 논문 약세 조건인 소N(N=5)에서 달성 | ✓✓(가장 직접적 재현) |
| 소N 열화 | "ratio↑·threshold 의존서 degrade" | silo5(N=5) frzero **0.25**, 3B frzero **0.0** — 단 열화는 zero/random 한정(delta는 N=5서도 1.0) | ◐(위협-의존으로 재정밀화) |
| LLM 증거 | "PEFT/LLM 증거 없음" | 첫 LLM-scale 테스트(pure-evasion zero/random엔 약, 패턴 있는 delta엔 강) | — |

### 4.4 FedDQC — 판정 ✓ · 세팅 동종(유일)
FedDQC (Du et al., ACL 2025 Findings, arXiv 2410.11540).

**세팅 비교** — 원 무대도 LLM-LoRA instruction tuning

| 항목 | 원 논문 | 우리 |
|---|---|---|
| 모델 | **LLaMA-2-7B + LoRA** | **Llama-3.2-1B/3B + LoRA** ✓ |
| 데이터 | PubMedQA/FiQA/AQUA/Mol + Fed-WildChat | 5-domain instruction(의료/법률/금융/수학/일반) + alpaca IID ✓ |
| 위협 | 50% 노이즈(데이터품질; crafted 아님) | noisy(answer_swap) + off-threat 대조 |
| 지표 | downstream perf(AUROC 미보고) | AUROC |

**결과 비교**

| 위협 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| noisy 품질 | "전 baseline 상회, 때때로 clean-oracle도 상회" | silo5 noisy **0.917**, device100 noisy **0.96–1.0**; **B축 신규 — IID 배경선 1.00 vs non-IID 0.92**(균질 배경일수록 품질 대비 선명 = 원 논문 지향 무대에 유리) | ✓ |
| poison(데이터 지시문) | (주요 축 아님) | silo5·device100·iid5 poison 전부 **1.00** — val-loss 게임((b)·Banzhaf 0.00)이 놓치는 backdoor 지시문을 data-quality 축이 잡음 | ✓ |
| off-threat | (free-rider 탐지기 아님) | free-rider 0.14–0.57(device100; silo5 0.75)·frdelta 0.750 (낮은 게 정상) | ✓(범위 정합) |
| 비용 | "IRA ~1% train-time, DataInf의 1/150" | 21–50s(silo5 1B/3B; device100은 ~435–475s) | ✓ |

---

## 5. 공격 (attack) 계열

### 5.1 Bagdasaryan (model-replacement) — 판정 ✓
*How To Backdoor FL* (Bagdasaryan et al., AISTATS 2020, arXiv 1807.00459).

**세팅 비교**

| 항목 | 원 논문 | 우리 |
|---|---|---|
| 데이터 | CIFAR-10 / Reddit | 5-domain LLM instruction |
| 공격 | single-shot model-replacement, γ=n/η (실험 γ=100) | silo5 γ=n/η=**5**(N=5,η=1) |
| 공격자 학습 | E_adv 6–10 (benign 2) | EPOCHS=5, poison_frac=0.8 |

**결과 비교**

| 지표 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| backdoor 설치 | **single-shot ~100% backdoor, main-task drop<1%** | silo5 **ASR≈1.00, clean-val 보존(+0.027)** | ✓ |
| 지속/희석 | "20+ rounds 지속; n/η 미만선 점진 degrade" | device100선 per_client≥300+R=60 필요, ASR α0.5≈0.50(희석; α0≈1.00) | ✓(precondition 단서 정합) |
| 공격자 결속성(새) | (단일 공격자 주입 구조) | removal 실측 — 공격자(worst-1) 제거 후 재학습 시 **ASR 1.0→0.0**(coalition류 방법 전부 성공; Flirds top-1 오지목 seed는 ASR 유지 = 공격이 아니라 valuation 쪽 한계) | ✓(메커니즘 정합) |

### 5.2 Xu (instructions-as-backdoors) — 판정 ✗ (다른 메커니즘)
*Instructions as Backdoors* (Xu et al., NAACL 2024, arXiv 2305.14710).

**세팅 비교**

| 항목 | 원 논문 | 우리 |
|---|---|---|
| task | **classification**(SST-2/HateSpeech/Tweet/TREC) | **generative** free-form |
| 공격 | **1% 데이터-poison만**(gradient 접근 없음) | 약한 token-level "tq" + **scaled-update**(Bagdasaryan) |
| ASR 정의 | 분류 label-match | text-exact-match |

**결과 비교**

| 지표 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| ASR | **1% poison → >90%**(Induced-Instruction avg 95.36%) | 우리 데이터-poison-only ASR=0 (≠ Xu 실험) | ✗(재현 아님) |

> 노트가 명시: 우리는 Xu의 헤드라인 실험(classification·1% 데이터-poison)을 돌린 게 아니라
> 약한 트리거를 generative 무대에 빌려온 것 → **"Xu 반박"도 "Xu 재현"도 아니다.**

---

## 6. 종합

**진짜 정량 일치 (세팅·위협이 맞물린 칸):**
1. **STD-DAGMM ×2** — ① device100 free-rider AUROC **0.87–0.96 ≈ 원 논문 0.91–0.96**(N≈100·~20%
   레짐 일치) ② **frdelta(원 논문 자체 위협인 delta-재활용) 1.000±.000 = 15개 방법 중 유일 탐지**.
   탐지기 대조에서 가장 깨끗한 두 칸.
2. **수렴-to-exact 보장류**(Banzhaf·GTG·FedSV·Fed-LOO의 가법 영역) — N=5 전 스케일(1B/3B/7B)
   Spearman/Pearson ≈1.0에 더해 **N=10 exact 2¹⁰(E5)에서도 same-game 방법 전원 +1.000** —
   "exact Shapley/semivalue 수렴" 이론이 규모 축으로 확장 확인.
3. **FedDQC** noisy 0.92–1.0(유일한 동종 LLM-LoRA 무대) + IID-배경 유리(1.00 vs 0.92) 실측.

**강점·한계 동시 재현:** FLDetector(crafted 1.0 + non-IID 침식 0.48 + off-threat frdelta 0.0),
FLTrust(scaled 1.0 + 정렬된 delta 통과 0.0 = 메커니즘 정합 + device100 poison 약세),
FedDQC(noisy 1.0 + off-threat 약), STD-DAGMM(N=100·delta 강 vs 소N zero/random 약).

**전이 안 됨(설명 가능):** ComFedSV(low-rank 가정 위배 — **홈 레짐인 대N 부분참여 std50k5에서도
음수 붕괴**라 "참여형태 탓" 반론까지 소거), Ripple(감사 확정 — "62×"는 소형·학습포함·느린-베이스라인
조건부 주장이라 **모순은 아니나** 우리 valuation-only 무대선 최고비용 ~33×; 구 "eigsh 불안정" 서술은
실측 반박되어 방법 고유 고유분해 volume + 자체-궤적 필요(from-logs 불가)로 정정).

**설계상 낮은 게 정상:** ShapleyFL(surrogate)·FedIF(영향도-가중) — exact-fidelity 낮음을 "나쁘다"로
읽으면 오독. 단 **새 관측 2건은 설계 밖 약점**: 저참여 std50k5에서 두 방법 모두 음수 붕괴, FedIF는
frdelta 완전 오정렬(0.0).

**게임-공통 실패의 분리(해석 지침, 신규):** frdelta와 IID-poison에선 **(b) exact oracle 자신이
실패한다**(AUROC 0.33 / 0.00) — 이 칸에서 φ-계열 baseline의 낮은 AUROC는 방법 실패가 아니라
val-loss 게임의 속성이고(기여도≠탐지), 진짜 방법 실패는 FLDetector/FLTrust/FedIF의 frdelta 0.0처럼
**oracle과 무관하게 자기 메커니즘이 속은 칸**이다. baseline 약점을 읽을 때 이 두 층을 구분할 것.

---

## 7. 이 검증의 한계

- **노트에 hard number 없는 6종**(GTG·FedSV·S-FedAvg·Banzhaf·ShapleyFL·Ripple)은 정성·이론
  대조까지만. 정량 대조는 `research-wiki/raw/papers/` 원본 PDF 재추출 필요(STD-DAGMM·FedIF·
  FLDetector·FLTrust만 노트에 수치 있음).
- **CNN(Track C) 평균은 pool값**(iid·label_skew 등 전 방법 저조 칸 포함)이라 깎임 — iid 제외 시
  Flirds vs (b) 0.93·Banzhaf 0.99.
- **β0.3 통일 재실행 잔여**: 7B 6셀 + device100 anchor 3셀의 ShapleyFL은 아직 β0.5-era
  (deferred 9셀; overview §4.2-9). 1B track_d·CNN도 재실행 커밋 없음 — 단 3B 전후 대조에서 β 변경
  효과가 재실행 노이즈 플로어 수준이라 표의 결론엔 영향 없음.
- **device100 off-anchor Spearman은 Flirds-proxy 기준**(진짜 oracle 아님). α=0.5 anchor만 (b)
  per-round exact.
- **집계 파이프라인 미반영 셀**: frdelta·B축(iid5·silo5_clean)·std50k5는 `make_analysis.py`/
  `make_fidelity.py` 자동 집계에 아직 없음 → 본 문서 수치는 rundir(metrics.json/phi.parquet)
  직접 집계(overview와 동일 규약).
- **Ripple LLM 3-seed 실측**은 감사 캠페인 산출([[ripple-audit]] §4.3)로 공유 그리드 rundir가
  아님; 서버 이전 전 수치(~4,515s)와는 비율 축으로만 비교 가능.
- **3B robustness = 1 seed**(poison 스케일 취약 판정 포함) — 3-seed 확정 전까지 단서 유지.

**가장 단단한 후속**: STD-DAGMM 1:1 정량 대조는 frdelta로 이미 확보. 남은 것은 ① FedDQC 칸의
"원 논문 수치 vs 우리 수치" 1:1 표(원 논문이 AUROC 미보고라 downstream-perf 축 환산 필요)와
② 6종 정성-only baseline의 원본 PDF 수치 재추출 — 여기가 정량 재현 주장을 넓힐 유일한 지점.
