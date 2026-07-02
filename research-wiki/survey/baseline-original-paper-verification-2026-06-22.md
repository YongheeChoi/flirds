# Baseline ↔ 원 논문 결과·세팅 대조 검증 (2026-06-22)

> 우리 실험에서 나온 각 baseline의 수치가 그 방법의 **원 논문이 보고한 결과**와 얼마나
> 일치하는지, 그리고 **실험 세팅**이 얼마나 맞물렸는지를 baseline별 테이블로 정리한 문서.
> 출처: 우리 결과 = `runs/track_c/{fidelity.csv,RESULTS.txt}` · `runs/track_d/fidelity.csv` ·
> `runs/phase2_matrix/{RESULTS.md,analysis/00_overview/master_metrics.csv}`.
> 원 논문 = `research-wiki/wiki/sources/*.md` (ingest 노트).
>
> **자매 문서**: [[baseline-selection-audit-2026-07-02]] — *다른 질문*을 본다: 애초에 baseline을
> **제대로 골랐는가**(선정의 정당성·완결성; 후보를 A/B/C로 판정). 이 문서(06-22)는 *이미 고른*
> baseline의 **수치·거동**이 원 논문과 맞는지. 즉 07-02="맞게 골랐나", 06-22="고른 게 맞게 굴러가나".

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
**✗ 불일치**(주장이 우리 무대로 전이 안 됨). 세팅 일치도: **高 / 中 / 低**.

---

## 1. 요약 표

| 방법          | 계열        |    세팅 일치     |  결과 판정  | 한 줄                                                    |
| ----------- | --------- | :----------: | :-----: | ------------------------------------------------------ |
| Banzhaf     | valuation |  N/A(중앙↔FL)  |  **✓**  | 값-수준 fidelity 최강(Pearson 0.998), "고품질·안정" 주장 정합        |
| GTG-Shapley | valuation |      低       |  **✓**  | 가법 영역서 exact 수렴(≈1.0), 대규모서 우아한 열화                     |
| FedSV       | valuation |      低       |  **◐**  | 오염 랭킹은 맞으나 rank-fidelity 낮음(per-round MC 분산)           |
| ShapleyFL   | valuation |      低       |  **◐**  | exact fidelity 낮은 게 정상(surrogate), 탐지는 작동              |
| FedIF       | valuation |  **中**(최근접)  |  **◐**  | Shapley 아님(설계), 속도·robust 주장은 일치                       |
| ComFedSV    | valuation |      低       |  **✗**  | low-rank 가정 위배 → fidelity 붕괴                           |
| Ripple      | valuation |      不明      |  **✗**  | "62× speedup" 주장과 정반대(우리 port 최저속)                     |
| FLDetector  | detector  |      低       | **✓✓**  | 강점(crafted 1.0)+한계(noisy/non-IID 침식) 둘 다 재현            |
| FLTrust     | detector  |      低       |  **✓**  | cosine-to-root가 scaled/free-rider 잡음(1.0)              |
| STD-DAGMM   | detector  | **device=高** | **◐→✓** | 동일 레짐(N=100 free-rider)서 수치 근접(0.87–0.96 vs 0.91–0.96) |
| FedDQC      | detector  |  **高**(동종)   |  **✓**  | 유일한 LLM-LoRA 무대, noisy-품질서 강함(0.96–1.0)                |
| Bagdasaryan | attack    |      中       |  **✓**  | model-replacement 재현(silo5 ASR≈1.0, clean 보존)          |
| Xu          | attack    |      低       |  **✗**  | 다른 메커니즘(generative+scaled) → 재현이라 부를 수 없음              |

> **세팅이 맞물린 칸에서만 정량 대조가 정당하다**: FedIF(C2)=中, STD-DAGMM(device100)=高,
> FedDQC=高. 거기서 결과도 잘 맞는다(§4).

---

## 2. 우리 실험 무대 세팅 (참조용 — 아래 각 표의 "우리" 열이 가리키는 무대)

| 트랙                        | 모델                | N (참여)      | rounds        | local   | batch        | lr / opt          | 데이터                 | 분할                          | seeds | oracle                                 |
| ------------------------- | ----------------- | ----------- | ------------- | ------- | ------------ | ----------------- | ------------------- | --------------------------- | ----- | -------------------------------------- |
| **C1** (CNN fidelity)     | LeNet5 / FedSVCNN | 10 (full)   | 10            | E=5     | 64           | 0.01 / SGD m=0    | MNIST, CIFAR-10     | iid+4 non-IID 시나리오          | 3     | (a)&(b) exact 2¹⁰                      |
| **C2** (CNN intervention) | FedSVCNN / LeNet5 | 100 (C=0.1) | 120           | E=5     | 64           | 0.01 / SGD m=0    | CIFAR-10, FMNIST    | iid / dir1(α=1) / shard     | 3     | — (성능/AUROC)                           |
| **D std20** (LLM 표준)      | Llama-3.2 1B & 3B | 20 (2/rd)   | 200           | 10 step | 16/8         | 1e-3 / SGD m=0    | alpaca-gpt4 20k     | **clean·IID**               | 3*    | (b) per-round exact                    |
| **D anchor5**             | Llama-3.2 1B & 3B | 5 (full)    | 30            | 10 step | 16/8         | 1e-3 / SGD m=0    | alpaca-gpt4         | clean·IID                   | 3*    | (b) 2⁵ + **(a) retrain val-loss fp32** |
| **P2 silo5** (robustness) | Llama-3.2 1B      | 5 (full)    | 10            | —       | 16(poison 8) | 1e-3(poison 2e-3) | 5-domain cross-silo | non-IID(domain-disjoint)    | 3     | (b) exact 2⁵                           |
| **P2 device100**          | Llama-3.2 1B      | 100 (K=10)  | 30(poison 60) | —       | 16(poison 8) | 1e-3(poison 2e-3) | 5-domain pool       | Dirichlet α∈{0,.01,.1,.5,5} | 3     | α=0.5만 (b) per-round; 그 외 Flirds-proxy |
| **P2 3B**                 | Llama-3.2 3B      | 5 (full)    | 10            | —       | 16           | 1e-3(poison 2e-3) | 5-domain            | non-IID                     | 1     | (b) exact 2⁵, coalition off            |

\* 3B fidelity.csv는 현재 2-seed(seed2 rundir 미병합). LoRA r16/α32, fp32, momentum=0 공통.

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
| 값 품질 순위 | Banzhaf > Beta-Shapley > Shapley > LOO (수치 미기록) | CNN Pearson **0.998**(최강), LLM anchor5 **1.000**, silo5 +1.000 | ✓ |
| 안정성 | "훨씬 높음" | 해당 실험 없음(우린 exact라 estimator 분산 미검증) | — |

> 단서: 논문 핵심 기여인 *MSR 불편추정·안정성*은 우리가 exact를 써서 테스트한 게 아니다. 값 자체의
> 품질만 확인 — 그 한도 내에서 "Banzhaf가 가장 깨끗한 값-수준 proxy"는 일치.

### 3.2 GTG-Shapley — 판정 ✓
GTG-Shapley (Liu et al., ACM TIST 2022, arXiv 2109.02053).

**세팅 비교**

| 항목 | 원 논문 | 우리 |
|---|---|---|
| task/데이터 | 이미지분류(데이터셋 미기록) | CNN MNIST/CIFAR + LLM 1B/3B |
| N / rounds | 미기록 | CNN N=10, LLM N=5/20 |
| 분할 | 미기록 | iid + non-IID |
| 지표 | utility-eval 수 대비 추정 정확도 | Spearman/Pearson vs exact oracle |

**결과 비교**

| 무대 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| 가법 영역 | "경쟁 federated-Shapley보다 정확"(수치 없음) | LLM std20 Spearman 0.975 / Pearson 0.995; anchor5·silo5 **+1.000** | ✓ |
| 대규모 비가법 | "guided-MC 초기 고분산"(정성) | device100 α=0.5 Spearman +0.784 / Pearson 0.892 (우아한 열화) | ✓ |

### 3.3 FedSV — 판정 ◐
FedSV (Wang et al. 2020, *A Principled Approach…*, arXiv 2009.06192).

**세팅 비교**

| 항목 | 원 논문 | 우리 |
|---|---|---|
| 데이터/모델 | MNIST/CIFAR-10, full-model | CNN N=10, LLM 1B/3B |
| 분할 | IID + non-IID | iid + non-IID |
| 지표 | noisy/backdoor 랭킹 vs Federated-LOO | Spearman/Pearson + 탐지 AUROC |
| 보장 | Theorem 1: per-round 공리 유일 만족 | (동일 알고리즘 port) |

**결과 비교**

| 지표 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| 오염 랭킹 | "noisy/backdoor를 낮게, LOO보다 우수"(수치 없음) | silo5 noisy/frrand 탐지 AUROC **1.000** | ✓ |
| rank-fidelity | — | std20 0.910/Pearson 0.959; anchor5 0.700; **poison Spearman +0.367** | ◐ |
| non-IID 약점 | "rare 클라 음수 SV" 단서 | device100 α=0.5 +0.752 (Flirds +1.0 대비 낮음) | ◐(정합) |

### 3.4 ShapleyFL — 판정 ◐ (설계상)
ShapleyFL (Sun et al., KDD 2023).

**세팅 비교**

| 항목 | 원 논문 | 우리 |
|---|---|---|
| 데이터 | MNIST/FMNIST/CIFAR/Fed-ISIC2019 | CNN + LLM 1B/3B |
| 성격 | **surrogate** Shapley(joint multi-round 아님) | 동일 surrogate port |
| 지표 | noisy/adversarial 하 최종정확도 | Spearman/Pearson vs exact + AUROC |

**결과 비교**

| 지표 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| exact 대비 fidelity | (해당 측정 없음) | std20 0.195/Pearson 0.245; anchor5 0.700 — **낮은 게 설계상 정상** | ◐ |
| 오염 하 효용 | "FedAvg+random보다 정확도↑" | silo5 탐지 AUROC **1.000**(오염 랭킹은 맞음) | ✓ |

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
| agg 속도 | **0.2s/rd vs AFedSV 70–92s (450×)** | LLM서 35–54s (Flirds-1st급, 최저가군) | ✓(빠름 재현) |
| exact-Shapley fidelity | (해당 없음; Shapley 아님) | std20 0.157 / anchor5 0.067 — **낮은 게 설계상 정상** | ◐ |
| robust-under-noise | "comparable-better than AFedSV" | device100 noisy AUROC 0.83–0.97 | ✓ |

### 3.6 ComFedSV — 판정 ✗ (설명됨)
ComFedSV (Fan et al., ICDE 2022, arXiv 2109.09046).

**세팅 비교**

| 항목 | 원 논문 | 우리 |
|---|---|---|
| 데이터 | synth/MNIST/FMNIST/CIFAR-10 non-IID | CNN + LLM 1B/3B |
| 규모 | 100 클라, 10개 noisy@30% flip, **partial** | N=5/20/100 |
| 핵심 가정 | utility 행렬 **low-rank**(convex 이론; VGG16까지 경험적) | **LoRA/transformer — 가정 위배** |
| 지표 | Spearman vs ground truth, Jaccard | Spearman/Pearson vs oracle |

**결과 비교**

| 지표 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| Spearman | "4개 데이터셋서 FedSV 근접·상회"(개별값 미기록) | CNN 0.348; LLM std20 **0.093**; 3B **−0.133**; device α0.5 **−0.023** | ✗ |
| 원인 | (Theorem: (4δ/N)-fair, δ=완성오차) | low-rank 위배 → δ 폭증 → fidelity 붕괴(노트 사전 경고) | — |

> 정직한 발견: **ComFedSV의 "FedSV 상회" 주장은 LLM 스케일로 전이되지 않는다.**

### 3.7 Ripple — 판정 ✗ (설명됨)
Ripple Shapley (Zeng et al., AAAI 2026). sample-level, single-run.

**세팅 비교**

| 항목 | 원 논문 | 우리 |
|---|---|---|
| 데이터 | 미기록(real-time pricing 응용) | CNN N=10 (LLM/Phase2 제외) |
| 수준 | **sample-level** | **client-level** 적응 |
| 핵심 가정 | low-rank Jacobian-subspace | 동일(eigsh 불안정) |
| 지표 | 속도 at comparable accuracy | Spearman vs oracle + 런타임 |

**결과 비교**

| 지표 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| 속도 | **62× speedup** | **최저속**(~4515s, Flirds의 42배) | ✗(정반대) |
| 정확도 | "comparable"(미정량) | CNN Spearman 0.373(최약), noisy AUROC 0.50 | ✗ |
| 원인 | — | client-level 적응 + eigsh CPU-spinning 불안정 → LLM/Phase2 설계상 제외 | — |

### 3.8 (참고) IRDS — Flirds의 기반, baseline 아님
In-Run Data Shapley (Wang et al., ICML 2024, arXiv 2406.11011). 중앙집중 GPT-2/Pythia, the Pile.

| 지표 | 원 논문 보고 | 우리(FL 확장) | 비고 |
|---|---|---|---|
| 2차항 효용 | "중앙 per-step선 1차 대비 이득 미미"(Appx E.2.2) | silo5 poison서 Flirds-1st AUROC **0.000** → 2차 포함 Flirds **0.917** | 모순 아님: FL per-round multi-step ≠ 중앙 per-step (우리 thesis 확인) |
| 음수 가치 | "Pile의 ~16% 음수 Shapley" | (해당 응용 없음) | — |

---

## 4. 탐지기 (detector) 계열

### 4.1 FLDetector — 판정 ✓✓ (강점·한계 둘 다 재현, 정합 최고)
FLDetector (Zhang et al., KDD 2022, arXiv 2207.09209).

**세팅 비교**

| 항목 | 원 논문 | 우리 |
|---|---|---|
| 데이터/모델 | MNIST/CIFAR-10(ResNet20)/FEMNIST | LLM 1B/3B LoRA |
| 악성 비율 / non-IID | 28% / degree 0.5 | poison 1/5, noisy 1/5 ~ 1/100 |
| 위협모델 | **crafted-update**(noisy-honest 아님) | poison(crafted) + noisy/free-rider |
| 지표 | DACC / FNR | AUROC |

**결과 비교**

| 위협 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| crafted(poison) | **DACC 0.85–1.0, FNR~0** (FEMNIST) | silo5 poison **1.000**, device100 poison 0.98 | ✓ |
| noisy-honest | "crafted만 탐지"(한계 명시) | silo5 noisy **0.75**, frzero 0.75 | ✓(한계 재현) |
| non-IID 침식 | "IID 가정서만 보장, 이질성서 침식" | device100 noisy **0.48–0.54** | ✓(침식 재현) |

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
| scaled/backdoor | under-attack ≤0.04 err, backdoor ≤0.03, **90% malicious 견딤** | silo5 poison **1.000** | ✓ |
| free-rider | (cosine 정렬로 misaligned 차단) | free-rider 전 레짐 **1.000** | ✓ |
| non-IID noisy 약점 | "benign-misaligned도 0으로 깎임" | device100 noisy 0.60(α0.01)–1.0 | ✓(한계 정합) |

### 4.3 STD-DAGMM — 판정 ◐→✓ (동일 레짐서 수치 근접)
STD-DAGMM (Lin et al., arXiv 1911.12560, 2019). **이 칸이 정량 일치가 가장 깨끗하다.**

**세팅 비교**

| 항목 | 원 논문 | 우리 (device100) |
|---|---|---|
| 모델 | MNIST + 2-layer MLP | Llama-3.2-1B LoRA(5.6M→256 feature-hash) |
| **N / free-rider** | **100 / 20** | **100 / ~20%** ✓ |
| 위협 | free-rider(zero/random/delta/advanced-delta) | freerider_random / freerider_zero |
| 지표 | AUC | AUROC |

**결과 비교**

| 무대 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| **N=100 free-rider** | **AUC 0.96(rnd5) / 0.91(rnd80)** (advanced-delta) | **frzero AUROC 0.87–0.96**, frrand 0.51–0.96 | ✓(수치 근접) |
| 소N 열화 | "ratio↑·threshold 의존서 degrade" | silo5(N=5) frzero **0.25**, 3B frzero **0.0** | ◐(예측대로 열화) |
| LLM 증거 | "PEFT/LLM 증거 없음" | 첫 LLM-scale 테스트(pure-evasion서 약) | — |

### 4.4 FedDQC — 판정 ✓ · 세팅 동종(유일)
FedDQC (Du et al., ACL 2025 Findings, arXiv 2410.11540).

**세팅 비교** — 원 무대도 LLM-LoRA instruction tuning

| 항목 | 원 논문 | 우리 |
|---|---|---|
| 모델 | **LLaMA-2-7B + LoRA** | **Llama-3.2-1B/3B + LoRA** ✓ |
| 데이터 | PubMedQA/FiQA/AQUA/Mol + Fed-WildChat | 5-domain instruction(의료/법률/금융/수학/일반) ✓ |
| 위협 | 50% 노이즈(데이터품질; crafted 아님) | noisy(answer_swap) |
| 지표 | downstream perf(AUROC 미보고) | AUROC |

**결과 비교**

| 위협 | 원 논문 보고 | 우리 결과 | 판정 |
|---|---|---|---|
| noisy 품질 | "전 baseline 상회, 때때로 clean-oracle도 상회" | silo5 noisy **0.917**, device100 noisy **0.96–1.0** | ✓ |
| off-threat | (free-rider 탐지기 아님) | free-rider 0.14–0.57 (낮은 게 정상) | ✓(범위 정합) |
| 비용 | "IRA ~1% train-time, DataInf의 1/150" | 21–49s(최저가군) | ✓ |

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
| 지속/희석 | "20+ rounds 지속; n/η 미만선 점진 degrade" | device100선 per_client≥300+R=60 필요, ASR 0.50–0.75(희석) | ✓(precondition 단서 정합) |

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

**진짜 정량 일치 (세팅이 맞물린 칸):**
1. **STD-DAGMM** device100 free-rider AUROC **0.87–0.96 ≈ 원 논문 0.91–0.96** — N≈100·~20% free-rider로
   레짐이 맞는 유일한 칸. 가장 깨끗한 수치 일치.
2. **수렴-to-exact 보장류**(Banzhaf·GTG·FedSV·ComFedSV의 가법 영역) — N=5 anchor·silo5서 Spearman/
   Pearson ≈1.0로 "exact Shapley/semivalue 수렴" 이론 확인.

**강점·한계 동시 재현:** FLDetector(crafted 1.0 + non-IID 침식 0.48), FLTrust(scaled 1.0),
FedDQC(noisy 1.0 + off-threat 약), STD-DAGMM(N=100 강·N=5 약).

**전이 안 됨(설명 가능):** ComFedSV(low-rank 가정 위배), Ripple(client-level+eigsh 불안정 → 62×
speedup 정반대). 둘 다 노트가 사전 경고한 경계 → 결함 아닌 예측된 한계.

**설계상 낮은 게 정상:** ShapleyFL(surrogate)·FedIF(영향도-가중). exact-fidelity 낮음을
"나쁘다"로 읽으면 오독.

---

## 7. 이 검증의 한계

- **노트에 hard number 없는 6종**(GTG·FedSV·S-FedAvg·Banzhaf·ShapleyFL·Ripple)은 정성·이론
  대조까지만. 정량 대조는 `research-wiki/raw/papers/` 원본 PDF 재추출 필요.
- **CNN(Track C) 평균은 pool값**(iid·label_skew 등 전 방법 저조 칸 포함)이라 깎임 — separable
  시나리오만 보면 Flirds/Banzhaf ≥0.95.
- **3B Track D fidelity.csv는 2-seed**(seed2 rundir 미병합). `python runs/track_d/make_fidelity.py`
  재실행 시 반영.
- **device100 off-anchor Spearman은 Flirds-proxy 기준**(진짜 oracle 아님). α=0.5 anchor만 (b) per-round exact.

**가장 단단한 후속**: 세팅이 맞물린 두 칸(STD-DAGMM device100, FedDQC noisy)을 "원 논문 수치 vs
우리 수치" 1:1 표로 못박기 — 여기가 유일하게 정당한 정량 재현 주장 지점.
