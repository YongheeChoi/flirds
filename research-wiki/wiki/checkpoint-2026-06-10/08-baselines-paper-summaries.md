---
type: checkpoint-doc
title: "직접 비교 Baseline 논문 정리 — valuation 7편 + detection 4편"
created: 2026-06-12
updated: 2026-06-12
note: "Flirds 실험에서 직접 비교군으로 선택한 baseline 11편의 논문 단위 정리. 각 논문은 개요 → 제안 방법론 → 실험 및 검증 → 결론 및 한계 순. 출처: 원 논문 PDF(raw/papers/flirds/) 직접 대조 + wiki sources/ 노트 + 우리 실측(codes/runs). 인용 수는 OpenAlex/Semantic Scholar 게재판 기준, 2026-06-12 조회 — 시점에 따라 변동."
---

# 비교군 구성 한눈에

Flirds의 직접 비교군은 두 축이다. **(A) valuation 7편** — 클라이언트 기여도 점수를 내는 방법들로, Spearman(vs oracle) + AUROC(vs 주입 라벨)로 비교한다. 계보 원조(FedSV) → 비용 절감 계열(GTG·ComFedSV·ShapleyFL) → semivalue 통제 변인(Banzhaf) → 동류 경쟁자(FedIF=1st-order, Ripple=in-run+curvature) 순으로 선정했다. **(B) detection 4편** — 기여도가 아닌 이상/품질 탐지 전용 방법들로, 위협별 매칭(threat-matched)으로 선정했고 AUROC만 비교한다(부호 있는 가치가 없으므로 Spearman 불가).

| 방법 (A=valuation, B=detection) | 게재처 / 연도 | 인용 수* | 비교에서의 위치 |
|---|---|---|---|
| **A1** FedSV | Springer *FL: Privacy and Incentive* (LNCS 12500), 2020 | ~168 | per-round federated Shapley의 원조 |
| **A2** GTG-Shapley | ACM TIST 13(4), 2022 | ~133 | gradient 재구성 + guided MC의 실용 표준 |
| **A3** ComFedSV | IEEE ICDE 2022 | ~72 | partial participation 공정성(행렬 완성) |
| **A4** ShapleyFL | ACM KDD 2023 | ~70 | surrogate SV → 가중·선택 (별칭 AFedSV) |
| **A5** Data Banzhaf | AISTATS 2023 | ~177 | semivalue 축 통제(노이즈 강건성) |
| **A6** FedIF | arXiv 2509.25560 (preprint), 2025 | 0 | 최근접 경쟁자 — 1st-order TracIn |
| **A7** Ripple Shapley | AAAI 2026 | 1 | 유일한 in-run + curvature FL 경쟁자 |
| **B1** STD-DAGMM | arXiv 1911.12560 (preprint), 2019 | ~145 | free-rider 위협 매칭 (model-free) |
| **B2** FLTrust | NDSS 2021 | ~788 | 신뢰 cosine — free-rider/poisoning |
| **B3** FLDetector | ACM KDD 2022 | ~296 | poisoning 위협 매칭 (시간 일관성) |
| **B4** FedDQC | ACL 2025 Findings | 2 | data-quality 위협 매칭 (IRA) |

\* 게재판 기준, OpenAlex/Semantic Scholar 2026-06-12 조회. arXiv 판 인용은 별도(예: FLTrust arXiv판 +35).

비고: 비교표의 `loss-heur`(클라이언트 단독 utility 휴리스틱)는 논문이 아닌 자체 하한 baseline이라 이 문서에서 제외. (b)/(a) oracle 역시 비교 기준점이지 baseline이 아니므로 제외.

---

# Part A — Valuation baselines

## A1. FedSV — A Principled Approach to Data Valuation for Federated Learning

> **Flirds 비교에서의 역할**: federated Shapley 계보의 원조 — "정통 SV 계열" 대표. 구현 `flirds/baselines/fedsv.py`(논문 Alg.2 permutation-MC + TMC truncation self-build; 공식 코드 없음). 우리 실측(1B N=5, 3-seed, lr 1e-3): Spearman vs (b) oracle **+1.000**, runtime ~532s. free-rider φ는 within-subset 재정규화 때문에 exact-0이 아님(Flirds/Banzhaf와 대조). tier1 poison(2026-06-10 실측)에서 Spearman **+0.367**로 추락 — near-additive 동률 구조가 처음 깨진 지점.

### 개요 (Overview)

- **논문 정보**: *A Principled Approach to Data Valuation for Federated Learning*. Tianhao Wang (Harvard), Johannes Rausch, Ce Zhang (ETH Zürich), Ruoxi Jia (Virginia Tech), Dawn Song (UC Berkeley). Springer 단행본 *Federated Learning: Privacy and Incentive* (LNCS 12500), pp. 153–167, 2020. arXiv:2009.06192. 인용 ~168(게재판) + 16(arXiv판).
- **연구 분야 (Keywords)**: federated learning, data valuation, Shapley value, incentive design, order-aware utility.
- **문제 정의 (Problem Definition)**: FL 참여자(클라이언트)별 기여도를 정량화하는 task. canonical Shapley를 FL에 직접 적용하면 (1) 모든 데이터-부분집합 재학습이 필요해 데이터가 사일로화된 FL에서 불가능하고, (2) symmetry 공리가 순서를 무시하는데 FL은 본질적으로 순차적(라운드 진행, lr decay로 초기 라운드 영향이 더 큼)이라는 두 격차가 생긴다.
- **연구 배경 및 필요성 (Motivation)**: 데이터 시장·인센티브 설계에서 SV가 표준 기여 측정이지만, FL 환경에 맞는 "추가 통신 없는 order-aware 기여 측정"이 부재했다. 기존 LOO류는 non-IID에서 변별력이 떨어진다.
- **핵심 기여 (Main Contribution)**: ① **federated SV 정의** — 라운드별 참여 cohort 내부의 canonical Shapley를 라운드 합산하는 per-round 분해, ② 공리 보존 정리(instantaneous group rationality·fairness·additivity; Theorem 1), ③ permutation sampling / group testing 두 추정기, ④ noisy-label·backdoor 클라이언트 검출 실증과 함께 **실패 사례(failure cases)를 명시 보고**해 후속 연구 방향을 연 점.

### 제안 방법론 (Proposed Method)

- **핵심 아이디어**: 서버가 이미 수신한 라운드-$t$ 업데이트들의 부분집합 $S$를 재집계해 글로벌 모델 위에 적용하고 서버 validation 성능으로 utility를 평가 — **재학습도 추가 통신도 없이** 라운드 내부의 SV를 계산하고 라운드에 걸쳐 합산한다.
- **모델 구조 및 알고리즘**: utility $\nu(\cdot)$는 ordered sequence를 인자로 받는다(이전 라운드 누적 $I_{1:t-1}$ 위에 부분집합 추가). 라운드 $t$ 참여자 $i$의 값:

$$s_t(i)=\tfrac{1}{|I_t|}\sum_{S\subseteq I_t\setminus\{i\}}\tbinom{|I_t|-1}{|S|}^{-1}\big[\nu(I_{1:t-1}+(S\cup\{i\}))-\nu(I_{1:t-1}+S)\big],\qquad s(i)=\sum_{t=1}^{T}s_t(i)$$

  미참여 라운드는 $s_t(i)=0$. exact 비용 $O(T\,2^m)$($m$=라운드 cohort 크기). **permutation sampling**(Alg.2): 라운드별 무작위 순열의 누적 marginal 평균, $(\epsilon,\delta)$-근사에 $Tm\frac{2r^2}{\epsilon^2}\log\frac{2m}{\delta}$회 평가. **group testing**(Alg.3-4): SV *차이* $C_{ij}$를 추정 후 pivot 1명만 MC로 복원, $O((\log m)^2)$ — cohort가 크면 우위(실험은 cohort 10이라 전부 permutation 사용).
- **사전 지식 (Preliminaries)**: Shapley 공리(efficiency/symmetry/null player/additivity), FedAvg 집계, Monte-Carlo SV 추정(Ghorbani-Zou), Hoeffding bound.

### 실험 및 검증 (Experiments & Evaluation)

- **실험 환경 (Setup)**: MNIST + CIFAR10. **N=100, 매 라운드 10명** 무작위 참여. IID = 균등 셔플(100명×600개), non-IID = label 정렬 후 2-shard/클라이언트(McMahan 방식). 모델: MLP(2×200)+simple CNN(MNIST), CNN 2-conv(CIFAR10). MNIST 25라운드(정확도 IID 97%/non-IID 92%), CIFAR10 50–200라운드(IID 77%/non-IID 70% — valuation 평가가 목적이라 SOTA 정확도는 비목표라고 명시). baseline: Federated-LOO, Random. 평가: 검사-비율 vs 검출-비율 **곡선**(AUROC/F1 같은 스칼라 없음).
- **실험 결과 (Main Results)**: ① noisy-label 검출(100명 중 20명 flip; 비율 MNIST 10%/CIFAR10 3%): IID에선 Fed-LOO ≈ Fed-SV, **non-IID에선 Fed-SV > Fed-LOO**. ② backdoor 검출(30% adversary, model-replacement; CIFAR10 striped-wall 특징/MNIST pixel-pattern): Fed-SV가 Fed-LOO보다 일관 우위. ③ 라운드별 총 기여가 학습 진행에 따라 감소(수렴 둔화) → **per-round norm 정규화 변형**을 도입하면 noisy/backdoor 분리가 크게 개선되나 group rationality·additivity를 상실. ④ data summarization은 non-IID에서 random에 지고 일부 구간 LOO에도 짐.
- **분석 (Analysis)**: 정규화 변형의 공리-검출력 트레이드오프가 사실상 핵심 ablation. 드물게 샘플링된 non-IID 클라이언트가 **음수 SV를 받기 쉬운 bias**를 스스로 진단. IID/non-IID 전 task 교차 비교, 추정기 간 비교는 이론 복잡도만.

### 결론 및 한계 (Conclusion & Discussion)

- **한계점 (Limitations)**: (논문이 "failure cases"로 명시) ① 초기 라운드 기여 inflate — 구조적 편향이며 정규화로 고치면 공리 상실, ② 드물게 선택된 참여자의 음수 SV bias(non-IID에서 심함), ③ non-IID backdoor 검출에서 benign 저기여자와 혼동. 그 외: 미선택 클라이언트 라운드값 0이 만드는 비대칭(후속 ComFedSV가 공격한 지점), 평가가 곡선 보고뿐, 라운드당 다수의 validation forward 비용.
- **향후 과제 (Future Work)**: 다양한 data-value 변형의 상세 분석을 명시 제안 — "실패 사례 보고로 후속 연구를 자극"하는 포지셔닝. (우리 관점: early-round inflation·정규화-공리 긴장은 round-summed federated SV에 내재 — Flirds의 라운드 집계 설계에도 재등장하는 trade-off.)

---

## A2. GTG-Shapley — Efficient and Accurate Participant Contribution Evaluation in FL

> **Flirds 비교에서의 역할**: 비용 절감 계열의 실용 표준 baseline. 구현 `flirds/baselines/gtg.py`(cyyever/torch_algorithm 참조 self-build — 공식 repo는 빈 껍데기). 우리 실측(1B N=5, 3-seed): Spearman vs (b) **+1.000**, ~537s. free-rider φ ≠ 0(within-subset 재정규화). 논문 스스로 인정한 **label-skew(Scenario 2)에서만 오차 > 1e-2** — Track C1의 GTG 5-시나리오(graded label-flip ladder) 공략점.

### 개요 (Overview)

- **논문 정보**: *GTG-Shapley: Efficient and Accurate Participant Contribution Evaluation in Federated Learning*. Zelei Liu, Yuanyuan Chen, Han Yu (NTU Singapore), Yang Liu (Tsinghua AIR), Lizhen Cui (Shandong Univ.). ACM Transactions on Intelligent Systems and Technology 13(4), pp. 1–21, 2022. arXiv:2109.02053 (2021-09). 인용 ~133(TIST판) + 9(arXiv판).
- **연구 분야 (Keywords)**: FL participant valuation, Shapley value, gradient-based sub-model reconstruction, guided Monte Carlo, truncation.
- **문제 정의 (Problem Definition)**: FL 참여자 SV 계산의 비용 병목 제거. 자체 측정으로 gradient 기반 SV 시간의 **98.5%가 sub-model 평가(evaluation)**(training 1.4%, reconstruction 0.1%) — 평가 횟수 자체를 줄여야 한다는 task 정의.
- **연구 배경 및 필요성 (Motivation)**: dataset 기반 SV(coalition 재학습)는 FL에서 비현실적. gradient 기반(MR/TMR 등)도 MC 순열 비용이 크고, 순열 위치 편향(늦게 합류하는 참가자의 marginal 과소평가) 문제가 있었다.
- **핵심 기여 (Main Contribution)**: **G**uided **T**runcation **G**radient Shapley — ① 코얼리션 sub-model을 이미 수신한 per-client gradient의 재합산으로 재구성(재학습 0), ② 2단 truncation(within-round + between-round), ③ guided sampling(순열 첫 $m$비트를 고정 순서로 순환 배치해 위치 편향 제거 + 수렴 가속). 복잡도 $O(T\log N)\sim O(TN\log N)$.

### 제안 방법론 (Proposed Method)

- **핵심 아이디어**: FedAvg에서 $\theta^{(t+1)}=\theta^{(t)}+\sum_{c\in S}w_c\,\Delta\theta_c^{(t)}$이므로, 임의 코얼리션 $S$의 sub-model은 수신된 업데이트의 재합산만으로 구성 가능 — 평가 1회 = forward pass들 뿐. 여기에 "평가할 가치가 없는" 순열 구간과 라운드를 잘라낸다.
- **모델 구조 및 알고리즘**: ① **between-round truncation** — 라운드 $t$의 전체 marginal gain $|v_N-v_0|\le\epsilon_b$이면 라운드 skip(전원 해당 라운드 φ=0), ② **within-round truncation** — 순열 위치 $j$에서 잔여 gain $|v_N-v_{j-1}|<\epsilon_i$이면 이후 평가 생략, ③ **guided sampling** — 순열의 첫 $m$비트($m\ll n$)를 고정 순서 순환으로 채우고 나머지는 random. 수렴 판정: 직전 10회 추정 대비 평균 상대 변화 < 0.05. (주의: 실험에 쓴 $\epsilon_b,\epsilon_i$ 수치는 논문에 미기재.)
- **사전 지식 (Preliminaries)**: TMC-Shapley(Ghorbani-Zou)의 truncation 아이디어, FedAvg 선형 집계(재구성의 전제), MC 순열 추정과 Hoeffding류 분석, bounded Pareto 분포(valuable 위치/라운드 모델링).

### 실험 및 검증 (Experiments & Evaluation)

- **실험 환경 (Setup)**: **MNIST 단일**(train 54,210 / test 8,920), **N=10 고정**, 5개 시나리오 — ① IID(참가자당 1,084장 균등), ② label-skew(쌍별로 특정 digit의 80% 집중), ③ quantity skew(10%~30% 차등), ④ noisy labels(쌍별 flip 0/5/10/15/20%), ⑤ noisy features(Gaussian 0~20%). baseline 6종: Original Shapley(exact 재학습) / TMC / GroupTesting / MR / Fed-SV / TMR. metric: 총 계산시간(log10) + exact SV와의 거리 3종(cosine/Euclidean/max). 모델 아키텍처·lr·batch·하드웨어는 본문 미기재(재현성 약점).
- **실험 결과 (Main Results)**: **전 시나리오에서 efficiency·accuracy 동시 1위**. Scenario 1에서 best baseline TMR 대비 **7.4× 빠름**. SV 추정 오차는 3개 distance metric 모두 **< 1×10⁻²** — 단 **Scenario 2(label-skew)는 예외로 gap 증가**(이때 TMC가 2위로 부상). 일반 관찰: gradient 기반 ≫ dataset 기반 효율, truncation은 non-IID에 민감하나 GTG만 guided partial-permutation 덕에 일관. 수치는 전부 log-log scatter figure로 보고(표 없음).
- **분석 (Analysis)**: ablation 3변형 — GTG-Ti(within만)/GTG-Tib(+between)/GTG-OTi(종료 후 1회 one-shot). between-round truncation은 **+17%(최대 3×) 효율**. **guided sampling이 정확도의 핵심**(전 세팅 최고 accuracy, 대부분 "orders of magnitude" 우위). one-shot은 2–3× 빠르지만 accuracy 최악(누적 gradient 재구성이 utility를 ground truth에서 이탈시킴) — SV 계산 빈도의 효율-정확도 트레이드오프.

### 결론 및 한계 (Conclusion & Discussion)

- **한계점 (Limitations)**: 명시적 limitations 섹션 없음. 드러나는 한계 — MNIST 단일 + N=10 고정(스케일 실험 없음), truncation threshold·모델·하이퍼파라미터 미보고, 결과가 figure-only, label-skew 오차 증가는 본문 인정, "평가가 비용의 98.5%"라는 구조적 병목은 횟수만 줄였을 뿐 평가 단가 자체는 그대로.
- **향후 과제 (Future Work)**: GTG 기반 **FL participant auditing 도구**(규제 기관의 FL 데이터 거래 감독 지원)를 명시 제안. (우리 관점: 평가 단가까지 없애는 closed-form 방향 — Flirds의 1 HVP/round — 이 자연 후속이며, label-skew 약점은 비교 실험의 차별화 지점.)

---

## A3. ComFedSV — Improving Fairness for Data Valuation in Horizontal FL

> **Flirds 비교에서의 역할**: cross-device(N=100, partial participation) 전용 대표 baseline. 구현 `flirds/baselines/comfedsv.py`(numpy ALS로 행렬 완성 self-build — LIBMF 의존 제거; uniform-subset 모델 + per-round test-loss 감소 utility). 우리 실측: N=100 α=0.5, R=30에서 Spearman **+1.000**(task7d; uniform-subset 평가는 exact uniform-Shapley와 일치 확인, CNN 경로 bit-identical). CNN Phase0 Spearman {1.0, 0.96, 0.85, 0.84}. tiny-R에서 낮은 Spearman은 R≈30 필요(real-config 재검증 항목).

### 개요 (Overview)

- **논문 정보**: *Improving Fairness for Data Valuation in Horizontal Federated Learning*. Zhenan Fan (UBC), Huang Fang, Zirui Zhou (Huawei Canada), Jian Pei (SFU), Michael P. Friedlander (UBC), Changxin Liu (KTH), Yong Zhang (Huawei Canada). IEEE ICDE 2022. arXiv:2109.09046. 인용 ~72.
- **연구 분야 (Keywords)**: federated Shapley, fairness, partial participation, low-rank matrix completion, utility matrix.
- **문제 정의 (Problem Definition)**: partial participation($|I_t|=m<N$)에서 FedSV의 "미선택 라운드 = 기여 0" 규칙이 깨뜨리는 **Shapley symmetry의 복원**. 동일 데이터($D_i=D_j$)를 가진 두 클라이언트의 실현값 $s_i,s_j$가 높은 확률로 크게 발산(MNIST: 동일-데이터 쌍의 상대차 > 0.5가 65% run).
- **연구 배경 및 필요성 (Motivation)**: exact Shapley는 코얼리션별 재학습이라 불가능 → $\epsilon$-Shapley-fairness($\epsilon$-symmetry/zero-element/additivity)로 완화하고, 관측 못한 코얼리션 utility를 통계적으로 채울 방법이 필요했다.
- **핵심 기여 (Main Contribution)**: ① 라운드×코얼리션 **utility matrix** $\mathcal{U}\in\mathbb{R}^{T\times 2^N}$ 관점 도입, ② smooth/strongly-convex 가정에서 $\epsilon$-rank가 $O(\log T/\epsilon)$(클라이언트 수 무관)임을 증명, ③ 저랭크 행렬 완성으로 미관측 entry를 채워 SV 계산, ④ 완성 오차 $\delta$에 대해 $(4\delta/N)$-Shapley-fair 보장(Theorem 1).

### 제안 방법론 (Proposed Method)

- **핵심 아이디어**: "관측 못한 코얼리션의 utility는 비슷한 클라이언트(비슷한 열)·느린 라운드 변화(비슷한 행)가 만드는 저랭크 구조에서 **보간**할 수 있다" — FedSV가 0으로 버리는 정보를 imputation으로 복원.
- **모델 구조 및 알고리즘**: $\mathcal{U}_{t,S}$ = 라운드 $t$에 코얼리션 $S$의 업데이트만 집계한 모델의 test-loss 감소. 관측 = $S\subseteq I_t$인 entry만. 정규화 행렬분해($\mathcal{U}\approx WH^\top$, LIBMF)로 완성 후 완성된 entry로 Shapley 계산. MC permutation으로 지수 행렬을 $T\times MN$($M=O(N\log N)$)으로 축소, 총 $O(TN^2\log N)$. **Assumption 1 (Everyone Being Heard)**: 모든 클라이언트가 ≥1회 선택되어야 함(≈$\lceil N/m\rceil$ 라운드-상당의 추가 비용).
- **사전 지식 (Preliminaries)**: Shapley 공리와 $\epsilon$-완화, 저랭크 행렬 완성(정규화 MF), FedAvg + client sampling, strong convexity/smoothness 가정 하의 궤적 해석.

### 실험 및 검증 (Experiments & Evaluation)

- **실험 환경 (Setup)**: Synthetic / MNIST / Fashion-MNIST / CIFAR-10, non-IID 파티션. 공정성 평가 = 동일-데이터 클라이언트 쌍의 값 차이 CDF. 검출 평가 = 100 clients 중 10명에 30% label flip, Jaccard + Spearman(vs ground truth).
- **실험 결과 (Main Results)**: ① 공정성: 동일-클라이언트 쌍 점수 차의 CDF가 FedSV를 stochastic dominance로 지배(4개 데이터셋 전부), ② noisy 검출: Spearman이 ground truth에 근접하며 FedSV 능가, label-flip 검출 Jaccard도 FedSV보다 높음, ③ 비용: FedSV/ComFedSV 런타임 비율은 참여율 $K/N$에 수렴 — utility 호출 $O(TNK\log N)$ vs FedSV $O(TK^2\log K)$.
- **분석 (Analysis)**: 신경망에서 utility matrix의 특이값 감쇠를 실증(저랭크 가정의 경험적 뒷받침 — 이론은 convex까지만). 완성 rank는 명제로부터 유도해 설정. 저자들이 직접 "convex 이론 vs 신경망 실증"의 간극을 인지하고 실험으로 보완하는 구조.

### 결론 및 한계 (Conclusion & Discussion)

- **한계점 (Limitations)**: ① 저랭크 이론이 convex 가정(LoRA-PEFT LLM의 비凸 궤적에서 utility matrix가 저랭크인지는 미검증 — VGG16까지만 실증), ② Everyone-Being-Heard가 cross-device에서 부담(우리 비교의 핵심 regime), ③ 행렬을 만들려면 결국 **전 클라이언트 업데이트가 매 라운드 필요** — client sampling의 통신 절감 논리와 충돌, ④ 완성 오차 $\delta$가 실제로 통제 불가(비凸 모델에 대한 a priori bound 없음).
- **향후 과제 (Future Work)**: (우리 관점) 완성 비용을 내고 symmetry를 복원하는 노선 vs 비대칭을 수용하고 참여-tier 내 ranking으로 읽는 노선(Flirds의 잠금 결정)의 head-to-head가 자연 실험 — cross-device non-IID에서 ranking 품질·noisy 검출·통신 비용 비교.

---

## A4. ShapleyFL — Robust Federated Learning Based on Shapley Value

> **Flirds 비교에서의 역할**: "AFedSV"라는 별칭으로 후속 논문들(FedIF 등)이 SV 비교군 삼는 그 방법 — 별도 AFedSV 논문은 없음. 구현 `flirds/baselines/shapleyfl.py`(균등평균 서브모델 + per-round exact SV + min-max + EMA; 우리 실행은 β=0.5). 우리 실측(1B N=5): Spearman **+1.000**, ~531s. 핵심 통찰: min-max+EMA+균등평균 가공이 없으면 plain-sum surrogate는 Shapley linearity에 의해 (b) oracle과 수학적으로 동치 — ShapleyFL을 "다른 방법"으로 만드는 건 정확히 그 세 가공. Track C2 개입 설계에서 가중집계 대체형(w←s)의 관례 출처.

### 개요 (Overview)

- **논문 정보**: *ShapleyFL: Robust Federated Learning Based on Shapley Value*. Qiheng Sun, Xiang Li, Jiayao Zhang, Zhan Qin, Kui Ren (Zhejiang Univ.), Li Xiong (Emory), Weiran Liu (Alibaba), Jinfei Liu (교신; ZJU + ZJU-Hangzhou Global STIC). ACM KDD 2023, pp. 2096–2108. DOI 10.1145/3580305.3599500. 인용 ~70.
- **연구 분야 (Keywords)**: robust FL, surrogate federated Shapley, adaptive aggregation weighting, importance-sampling client selection.
- **문제 정의 (Problem Definition)**: 데이터 이질성·오염(irrelevant/noisy/malicious 데이터) 하에서 FL 최종 모델의 성능 방어. SV로 클라이언트 가치를 추정해 ① 집계 가중과 ② 클라이언트 선택 확률에 반영하는 robust-FL task — 가치 측정 자체가 아니라 가치-기반 **개입**이 목적.
- **연구 배경 및 필요성 (Motivation)**: per-round Shapley는 $O(2^n)$ 비용 + 라운드별 스케일 불균형(학습 후반 utility 변화 감소). 무작위 선택은 가치 있는 클라이언트를 놓침. 기존 robust 집계(Krum/RFA류)는 기여 "가치"가 아닌 통계 outlier 관점.
- **핵심 기여 (Main Contribution)**: ① **surrogate federated SV** — 라운드 참여자 한정 partial SV → min-max 정규화 → EMA의 3단 가공, ② SSV-비례 가중집계 + 분산 최소화 client-selection 확률(Thm 5.1), ③ 차이-기반 DMC 추정기, ④ 수렴(Thm 4.4)·안정성(Thm 4.5) 이론, ⑤ 5개 오염 시나리오 + 실제 의료 FL(Fed-ISIC2019) 검증.

### 제안 방법론 (Proposed Method)

- **핵심 아이디어**: FL을 라운드별 협력 게임의 **수열**로 보고, 라운드마다 "그 라운드 참여자 내부의" Shapley를 구해 시간축으로 평활(EMA)한 값을 가중·선택 신호로 쓴다.
- **모델 구조 및 알고리즘**: ① partial SV(Def 4.1): $U_F(S)=\Phi(D_v,\Psi(x^t,S))$ — 라운드 업데이트의 부분집합을 적용한 글로벌 모델의 server-validation 성능(재학습 없음), 참여자 $C^t$ 내부에서 exact SV, ② min-max(Def 4.2): $NSV_i^t=\frac{SV_i^t-\min SV^t}{\max SV^t-\min SV^t}$, ③ EMA(Def 4.3): $SSV_i^t=\beta\,SSV_i^{t-1}+(1-\beta)NSV_i^t$(미참여자는 carry-forward; **β=0.3 채택**), ④ 가중 $w_i\propto SSV_i$, ⑤ 선택 확률 $p_i^t=(m-|L^t|)\,w_i^{t-1}/\sum_{j\notin L^t}w_j^{t-1}$(상위 $l$명은 1로 clip; 분산 최소화 해), ⑥ **DMC 추정기**: SV 차이 $\Delta SV_{k,i}$의 분산이 SV 자체보다 작음을 이용한 코얼리션-크기 층화 차이-샘플링.
- **사전 지식 (Preliminaries)**: Shapley 정의·MC/TMC 추정, FedAvg + client sampling, importance sampling 분산 분석, validation 기반 utility 설계.

### 실험 및 검증 (Experiments & Evaluation)

- **실험 환경 (Setup)**: CIFAR-10(150R)·Fashion-MNIST(100R), **100 clients, 참여율 0.1**, 2-shard non-IID(200 shards), simple CNN; server validation = 원 test set의 20%(2,000장). + **Fed-ISIC2019**(피부암 8-class, 6 clients 자연 크기 skew 9,930~351장, EfficientNet, 25R, balanced accuracy, 매 라운드 malicious 2명, 10회 반복). **오염 시나리오 5종**: long-tail imbalance / open-set label noise(무관 이미지에 기존 label, 해당 클라이언트 데이터의 절반) / closed-set label flip($L\to(L{+}1)\bmod C$) / 입력 Gaussian noise($\sigma=1$) / gradient poisoning($a\to a(1+b)$, $b\sim U[-0.5,0.5]$). baseline: FedAvg / FedProx / FedSV / S-FedAvg / RFA. lr·batch 등은 코드 repo로 위임. **free-rider 시나리오는 없음**.
- **실험 결과 (Main Results)**: 전 세팅에서 AFedSV/AFedSV+ 우위 — CIFAR-10 최대 개선 **vs FedAvg +8.1% / FedProx +20.7% / RFA +11.0% / S-FedAvg +7.9% / FedSV +7.3%**; open-set 대표치 58.22%(150R). **Fed-ISIC**: AFedSV 64.24% = vs FedAvg **+25.1%** / FedProx +13.7% / RFA +13.4% / FedSV +4.4%, error-bar도 최소(가중 메커니즘의 안정성). AFedSV+ > AFedSV(importance sampling 효과, FMNIST에서 뚜렷). FedProx는 오염 세팅에서 FedAvg보다도 나쁨(proximal이 오염 gradient를 동등 취급).
- **분석 (Analysis)**: ① β sweep(CIFAR open-set): 0.1→0.5482 / **0.3→0.5822** / 0.5→0.5784 / 0.9→0.5588 — 양극단 모두 손해, ② DMC: MC-2000 permutation 기준 MSE가 MC/TMC 대비 전 샘플 구간 최저(샘플 적을수록 우위; 예: CIFAR 80-sample에서 DMC 1.52e-2 vs MC 5.66e-2), ③ FMNIST closed-set에서 **전 알고리즘 급락** — 가중 하향만으로는 오염 gradient 영향을 제거 못함을 본문 인정.

### 결론 및 한계 (Conclusion & Discussion)

- **한계점 (Limitations)**: 명시 섹션 없음(결론 4문장). 드러나는 한계 — ① **clean server validation set 필수**(오염 시 분석 없음), ② partial SV도 라운드당 $2^m$ utility 평가(m=10이면 1,024회 validation forward; DMC는 보완책이고 본 실험은 exact), ③ image classification 한정 + 평가가 acc-vs-round뿐(**exact SV 대비 ranking 충실도 미측정** — DMC MSE는 MC-2000 벤치마크 대비), ④ 독립 malicious 가정(collusion 비판하면서 자체 실험 없음), ⑤ free-rider류 비용-회피 위협 미실험.
- **향후 과제 (Future Work)**: 저자 명시 없음. (우리 관점: ranking 충실도 평가의 공백이 Flirds 비교 설계의 차별점 — 우리는 (b) oracle 대비 Spearman을 직접 잰다. 또한 min-max+EMA 가공이 검출 신호에 미치는 영향이 흥미로운 비교축.)

---

## A5. Data Banzhaf — A Robust Data Valuation Framework for Machine Learning

> **Flirds 비교에서의 역할**: semivalue 축의 통제 변인. 구현 `flirds/baselines/banzhaf.py` — **(b) oracle의 exact $2^N$ 코얼리션 utility를 그대로 재가중**(Shapley 계수 → 균일 $1/2^{n-1}$)해 "utility 고정, semivalue만 교체"를 격리. (b) utility는 deterministic이라 ranking이 거의 안 움직여야 한다는 게 포인트 — 실측(1B N=5) Spearman **+1.000** 확인, ~531s. free-rider(zero-delta)는 모든 코얼리션 marginal 0 → **φ exact-0**(Shapley/Flirds와 동일; GTG/FedSV의 재정규화와 대조). N≤10 exact, N=100에서는 $2^N$이라 비교군에서 드롭(matrix gating). cross-seed stability(Track C3)의 이론 근거이기도 함.

### 개요 (Overview)

- **논문 정보**: *Data Banzhaf: A Robust Data Valuation Framework for Machine Learning*. Jiachen T. Wang (Princeton), Ruoxi Jia (Virginia Tech). AISTATS 2023 (PMLR). arXiv:2205.15466. 인용 ~177. (읽은 md는 arXiv v7; venue 연도는 외부 서지 확인.)
- **연구 분야 (Keywords)**: data valuation, semivalue, Banzhaf value, robustness to stochastic training, sample-efficient estimator.
- **문제 정의 (Problem Definition)**: SGD의 확률성으로 utility $U(S)$(부분집합 학습 모델의 validation 성능)가 노이즈를 가질 때도 **순위가 뒤집히지 않는** data value notion 찾기. 실제로 같은 valuation 알고리즘을 두 번 돌리면 ranking이 달라진다는 관찰에서 출발.
- **연구 배경 및 필요성 (Motivation)**: Shapley/LOO 기반 가치는 deterministic utility를 가정하지만 현대 학습은 그렇지 않다. ranking 불안정은 data pricing·저품질 데이터 검출 같은 다운스트림을 직접 깨뜨린다.
- **핵심 기여 (Main Contribution)**: ① **safety margin** 개념화(순위 보존을 깨는 데 필요한 최소 utility 교란; Def 4.1-4.2), ② **Banzhaf가 모든 semivalue 중 최대 safety margin $\tau\cdot 2^{n/2-1}$** 달성 증명(Thm 4.6; Shapley·LOO 대비 지수적 우위), ③ **MSR(Maximum Sample Reuse) 추정기** — $\ell_\infty$ 기준 $O(\frac{1}{\epsilon^2}\log\frac{n}{\delta})$로 simple MC 대비 factor $n$ 절약(Thm 4.9), 하한 대비 near-optimal(Thm 4.10), 그리고 이런 추정기의 존재가 semivalue 중 Banzhaf 고유함을 증명.

### 제안 방법론 (Proposed Method)

- **핵심 아이디어**: Shapley는 코얼리션 "크기별" 균등 가중인데, worst-case 노이즈 방어에는 "모든 부분집합" 균등 가중(Banzhaf)이 최적 — 균일 가중이 최대 안전 여유를 만든다.
- **모델 구조 및 알고리즘**: Banzhaf 값 $\phi_i^{\mathrm{Bz}}=\frac{1}{2^{n-1}}\sum_{S\subseteq D\setminus\{i\}}[U(S\cup\{i\})-U(S)]$. **MSR**: $S\sim\mathrm{Unif}(2^N)$를 $m$개 뽑아 $\widehat\phi(i)=\mathrm{mean}_{S\ni i}U(S)-\mathrm{mean}_{S\not\ni i}U(S)$ — 모든 utility 평가가 **모든** 점의 추정에 재사용된다(Shapley에는 이런 unbiased 재사용 분포가 존재하지 않음; Appendix C.2). safety margin 일반식과 함께 LOO=$\tau$, Shapley≈$\tau(n-1)/\sqrt{\smash{\sum_k\binom{n-2}{k-1}^{-1}}}$ 수준 vs Banzhaf $\tau 2^{n/2-1}$.
- **사전 지식 (Preliminaries)**: semivalue 가족(가중 $w(k)$로 모수화된 marginal 평균; Shapley/Banzhaf/Beta 모두 포함), cooperative game theory의 Banzhaf index, Hoeffding/이항 집중 부등식.

### 실험 및 검증 (Experiments & Evaluation)

- **실험 환경 (Setup)**: 분류 13개 데이터셋(MNIST/FMNIST/CIFAR10은 2,000점 평가, Click 1,000, Phoneme 500, 나머지 200). NN + **Adam**(utility가 본질적으로 noisy하도록 의도). 추정 예산 100,000회 평가. baseline: Shapley(permutation), LOO, Beta-Shapley 4종(>500점에서 수치 불안정으로 대형 셋 결과 생략), weighted-samples task에는 Uniform 추가. metric: noisy-label 검출 F1(하위 10퍼센타일 컷), weighted-training acc, run 간 Spearman. centralized per-datum valuation(FL 아님).
- **실험 결과 (Main Results)**: ① **run 간 ranking 안정성**(CIFAR10 20점/5 mislabeled/5 runs): 평균 Spearman **LOO 0.001 / Shapley 0.038 / Banzhaf 0.856**, ② noisy-label F1(10% flip): 대형 셋 우위 — MNIST 0.193(Shapley 0.135/LOO 0.165), CIFAR10 0.220(0.152/0.086), Click 0.206; 소형 셋은 혼전(Fraud는 Shapley 0.65 > Banzhaf 0.47 등) — 본문 표현은 "best overall", ③ weighted samples acc 대부분 1위, ④ MSR ≫ simple MC(수렴 속도·분산), ⑤ 노이즈 클수록 Banzhaf 우위 확대.
- **분석 (Analysis)**: 노이즈 원천 교체 실험(SGD → randomized-smoothing GD)에도 우위 유지 — 구조 불문 강건성. tiny-data exact 검증(10점 + 명시적 Gaussian 노이즈)에서도 동일 결론. Banzhaf = Datamodel($p=0.5$, 무정규화)의 특수형과 동치라는 관찰.

### 결론 및 한계 (Conclusion & Discussion)

- **한계점 (Limitations)**: (§6 명시) 보장이 **worst-case**(임의·적대 교란) — 노이즈 원천을 아는 상황에선 threat model이 과도. 또한 Banzhaf는 efficiency 공리 포기(값 합 ≠ 총 utility) — 예산-균형이 필요한 데이터 마켓에는 별도 고려 필요.
- **향후 과제 (Future Work)**: 학습 확률성이 만드는 **특정 utility-noise 분포를 반영한 robustness notion** — 1-D 선형회귀+GD+Gaussian init조차 val loss가 generalized $\chi^2$(intractable)임을 부록에서 시연하며 노이즈 구조 이해가 선행 과제라고 명시. (우리 관점: "추정 노이즈 하 ranking 안정성"이라는 평가축 자체가 Track C3 cross-seed stability 실험의 직접 근거.)

---

## A6. FedIF — Lightweight and Robust Federated Data Valuation

> **Flirds 비교에서의 역할**: **가장 가까운 published 경쟁자** — "1st-order로 충분한가?"를 묻는 직접 baseline. 구현 `flirds/baselines/fedif.py`(Eq.6-8 재현; influence는 good→HIGH라 비교 시 부호 반전). FedIF의 $\Phi=\langle\Delta w/\|\Delta w\|,\nabla\ell_{val}\rangle$는 FLTrust cosine ≈ **normalized Flirds-1st**와 같은 신호 계열 — Flirds의 2nd-order HVP 항이 차별점 그 자체. phase2 matrix에 상시 포함(regime 불문; real grid 집계 진행 중). FedIF의 future-work 문장("gradient의 더 많은 정보 활용")이 정확히 Flirds 방향을 가리킴.

### 개요 (Overview)

- **논문 정보**: *Lightweight and Robust Federated Data Valuation*. Guojun Tang, Steve Drew (Univ. of Calgary), Jiayu Zhou (Univ. of Michigan), Mohammad Mamun (NRC Canada). arXiv:2509.25560 (2025-09-29, preprint — 미게재). 인용 0. 코드 공개(github.com/guojuntang/FedIF).
- **연구 분야 (Keywords)**: federated data valuation, TracIn, trajectory influence, robust aggregation, first-order.
- **문제 정의 (Problem Definition)**: SV 기반 robust FL(ShapleyFL/AFedSV류)의 라운드당 부분집합 재가중+추론 비용을 제거하면서 비슷한 강건성을 내는 클라이언트 가치 추정 task.
- **연구 배경 및 필요성 (Motivation)**: SV 기반은 정확하나 라운드당 70–92s급 집계 오버헤드로 스케일 한계. 클래식 influence function은 per-client Hessian inverse가 필요해 "FL에 부적합"하다고 명시적으로 기각 → Hessian-free 1st-order 설계로 방향을 잡음. TracIn(Pruthi et al. 2020)의 gradient-trace를 FL 클라이언트 업데이트에 가져온 **최초** 시도라고 자기 위치 규정.
- **핵심 기여 (Main Contribution)**: ① TracIn의 FL 이식 — L2-정규화 클라이언트 업데이트와 validation gradient의 내적을 라운드 영향으로 정의, ② min-max + EMA + 적응 가중의 경량 파이프라인, ③ noisy 환경에서 FedAvg보다 빠듯한 1-step loss 상계(Thm 1; clean이면 FedAvg로 환원), ④ AFedSV 동등~우위의 robustness를 **집계 오버헤드 450×↓**로 달성.

### 제안 방법론 (Proposed Method)

- **핵심 아이디어**: "노이즈 업데이트는 validation gradient와 어긋난다" — 방향 정렬도 하나로 가치를 측정하고, 그 값을 그대로 집계 가중으로 쓴다.
- **모델 구조 및 알고리즘**: ① 라운드 영향(Eq.6): $\Phi_i^t=\frac{\Delta w_t^i}{\|\Delta w_t^i\|}\cdot\nabla\ell(w_{t-1},D')$ — 정규화로 스케일 제거(방향 지배), ② min-max(Eq.7): $\Psi_i^t$, ③ EMA(Eq.8): $\Omega_i^t=(1-\gamma)\Omega_i^{t-1}+\gamma\Psi_i^t$($\gamma\in\{0.3,0.4\}$; 미참여자 carry-forward), ④ 가중(Eq.9): $p_i^t=\Omega_i^t/\sum_j\Omega_j^t$로 집계. TracIn의 단일-step gradient 자리에 multi-local-epoch 파라미터 delta($E=5$)를 대입하는 게 핵심 근사.
- **사전 지식 (Preliminaries)**: TracIn(트레이닝 궤적 따라 $\sum_t\eta_t\nabla\ell(w_t,d)\cdot\nabla\ell(w_t,d')$), influence function의 Hessian-inverse 형태(기각 대상으로 인용), FedAvg, $L$-smoothness 하의 1-step 해석.

### 실험 및 검증 (Experiments & Evaluation)

- **실험 환경 (Setup)**: CIFAR-10 + Fashion-MNIST, **CNN**, 100 clients, Dirichlet $\alpha=1$, 참여율 $C=0.1$, $E=5$, $B=16$, SGD lr 0.001 + **momentum 0.9**, $T=100$. validation = test split의 20%. RTX 4080. baseline: FedAvg, AFedSV(=ShapleyFL 계열 가중), 그 외 robust 집계.
- **실험 결과 (Main Results)**: ① label noise·gradient noise에서 AFedSV 동등~우위(CIFAR-10에서 상회, FMNIST 근접), ② clean에선 FedAvg와 동등(Remark 1의 이론 예측과 일치), ③ **집계 ~0.2s/round vs AFedSV 70–92s → 최대 450×**(집계 시간 한정 — 학습 시간은 전 방법 동일), ④ 실패 모드: **PGD adversarial 샘플에 무력**(PGD는 클린과 유사한 업데이트 방향 보존 → 방향-만 신호의 구조적 사각), 극단 노이즈($n=0.7$)에서 붕괴.
- **분석 (Analysis)**: ablation — 로컬 가중 정규화(WN)/라운드 정규화(RN)/EMA(SU) 각각 기여하며 **WN이 gradient-noise 케이스에 가장 중요**. Thm 1의 noise-항 축소 메커니즘($p\propto 1/\|\delta\|$)으로 견고성을 설명.

### 결론 및 한계 (Conclusion & Discussion)

- **한계점 (Limitations)**: ① **strictly 1st-order** — 2차항·클라이언트 상호작용(HVP) 항이 없고, 저자 스스로 Hessian을 FL에서 비현실적이라 단정(Flirds가 반박하는 지점), ② Shapley가 아님 — min-max/EMA 가공 점수로 공리 없음, ③ CNN-only, LoRA/LLM 없음, ④ 집계를 바꾸는 robust-FL 알고리즘(post-hoc valuation이 아님), ⑤ PGD류 방향-보존 오염에 사각.
- **향후 과제 (Future Work)**: 저자 명시 — "gradient에서 더 많은 정보를 활용한 업데이트 평가". (우리 관점: 그 문장이 가리키는 곳이 2nd-order — Flirds의 차별 주장과 정확히 맞물리는, 가장 중요한 "1st-order 상한" baseline. 또한 FedIF가 쓰는 multi-step delta는 Flirds가 Taylor 전개하는 그 객체와 동일 — FedIF는 1차 내적까지만 취한다.)

---

## A7. Ripple Shapley — Data Influence Attribution in One Federated Training Run

> **Flirds 비교에서의 역할**: 유일한 **in-run + curvature** FL 경쟁자(IRDS 직계 — Flirds와 같은 뿌리에서 출발). 구현 `flirds/baselines/ripple.py`/`ripple_llm.py`(Alg.1 + Eq.5-19 self-build; 공개 코드 없음). 우리 실측(1B N=5): noisy AUROC **0.50±0.20**(비교군 최약) + ~4,515s(Flirds의 ~42×, 비교군 최고가) = **fully dominated**. 클라이언트별 local Hessian eigsh가 지배 비용이며 CPU-spin stall이 잦아 기본 비교에서 RIPPLE=0으로 제외, Track C1에 eigsh guard를 달아 재포함 예정. Flirds의 1 HVP/round와 비용 구조가 정확히 대비됨.

### 개요 (Overview)

- **논문 정보**: *Ripple Shapley: Data Influence Attribution in One Federated Training Run*. Dewen Zeng, Wenlong Tian* (Univ. of South China), Haozhao Wang (HUST), Jianfeng Lu (WUST), Weijun Xiao (VCU), Zhiyong Xu (Suffolk Univ.). AAAI 2026, Proceedings v40, pp. 28085–28093. DOI 10.1609/aaai.v40i33.40034. 인용 1(2026-06 시점 — 출판 직후).
- **연구 분야 (Keywords)**: federated Shapley, in-run/single-run attribution, sample-level valuation, influence propagation, low-rank Jacobian chain, data pricing.
- **문제 정의 (Problem Definition)**: 한 번의 FL 학습 run에서 **샘플 수준** 기여를 추적하는 task — 특히 어떤 샘플의 영향이 자신이 참여한 라운드 이후의 글로벌 업데이트들을 타고 **전파(cascade)**되는 부분까지 포함.
- **연구 배경 및 필요성 (Motivation)**: 기존 federated SV는 multi-round 재학습 계열(비쌈) 아니면 per-round 집계 계열(cross-round 전파 무시)이었고, 샘플 수준 측정이 없었다. centralized 쪽 In-Run Data Shapley(IRDS)의 FL 확장이 공백.
- **핵심 기여 (Main Contribution)**: ① 기여 = **drop term**(자기 라운드 내 즉시 효과) + **ripple term**(이후 라운드로의 재귀 전파) 분해, ② Jacobian chain의 저랭크 spectral 근사(rank $k=20$, 누적 subspace $m=50$)로 tractable화, ③ 저랭크 근사 하에서도 Shapley 공리(symmetry/dummy/linearity) 보존 주장, ④ AFedSV+ 대비 **62× 속도**(100R 누적), ⑤ dynamic participation·real-time data pricing 시나리오 제시.

### 제안 방법론 (Proposed Method)

- **핵심 아이디어**: 샘플의 영향을 "출생 라운드의 즉시 기여"와 "그 기여가 만든 모델 변화가 이후 라운드를 타고 흐르는 물결"로 나눠, 후자를 글로벌 곡률의 Jacobian 곱으로 명시 계산한다.
- **모델 구조 및 알고리즘**: ① drop(Eq.5-7): 로컬 $T$-step 누적 val-loss 개선 $\mathcal{U}_{local}$에 FedAvg 가중 $\alpha_k=n_k/n_s$를 곱함(IRDS식 1st-order를 realized 로컬 궤적 따라 누적), ② ripple(Eq.8-13): influence path의 chain rule에서 라운드 Jacobian을 $J^{(t)}\approx I-\eta H^{(t)}$로 선형화($H^{(t)}$=클라이언트 가중 글로벌 Hessian; $\beta$-smooth 가정 하 오차 $\le\frac{\beta}{2}\eta^2$), $\mathcal{U}_{ripple}=\sum_{r=2}^{R}\frac{\partial\mathcal{L}}{\partial w^{(t_0+r)}}\prod_l(I-\eta H^{(t_0+l)})\frac{\partial w^{(t_0+1)}}{\partial z}$, ③ 저랭크화(Eq.14-19): 각 클라이언트가 local Hessian top-$k$ eigenpair를 업로드("Hessian sketch"), progressive orthonormal subspace $Q$($m\le rk$)에서 chain을 $m$-차원 재귀로 계산.
- **사전 지식 (Preliminaries)**: In-Run Data Shapley(per-step utility의 1st-order Taylor), Hessian spectrum 급감(Ghorbani et al. 2019 — 저랭크 근사의 근거), chain rule을 통한 영향 전파, FedAvg.

### 실험 및 검증 (Experiments & Evaluation)

- **실험 환경 (Setup)**: MNIST(2-layer MLP) + CIFAR-10(standard CNN). 시나리오 4종: non-IID(label skew) / long-tail imbalance / open-set noise / data noise. 5 local epochs, 100 rounds, B=10, lr 0.01, 5회 반복, **Tesla V100×2**. **클라이언트 수 N은 본문 미기재**("Sun et al. 2023[=ShapleyFL] 프로토콜 추종"이라고만 기술). 가중집계는 $w_i=\lambda u_i+(1-\lambda)\frac{n_i}{\sum n_j}$, **λ=0.5 고정**(additive 혼합형). baseline: FedAvg / FedProx / S-FedAvg / FedSV / AFedSV+. 평가는 누적 계산시간 + acc-vs-round — **수치오차(MSE) 기반 attribution 평가를 명시적으로 거부**하고 task-driven(robustness 개선)으로만 평가.
- **실험 결과 (Main Results)**: ① 100R 누적 계산시간 985s = plain training의 2.05× = **4.6× / 49.1× / 62.4× faster** vs S-FedAvg / FedSV / AFedSV+(62×의 비교 대상이 AFedSV+), ② accuracy는 거의 전 세팅에서 동등 이상, **open-set noise에서 +10%p 이상**, ③ 초기 라운드 우위가 최종 우위로 이어진다고 해석.
- **분석 (Analysis)**: ① propagation depth: ripple 값이 **depth 20 이후 수렴** → $R=20$ 권장, ② dynamic FL case study: 참여율 0.5–1.0 변동·dropout에도 전파 추적 유지, 같은 데이터도 참여 시점에 따라 가치가 달라짐(temporal 모델링) → real-time pricing 논의, ③ 반복/복제 데이터의 marginal 가치는 ripple 감쇠로 자연 하락. 명시적 ablation 표는 없음(λ=0.5는 tuning 배제 목적).

### 결론 및 한계 (Conclusion & Discussion)

- **한계점 (Limitations)**: **명시적 limitations 섹션 없음**. 근사 보장은 "smoothness 가정 하 bounded error"로만 언급(증명은 부록, proceedings 본문 미수록). 드러나는 한계 — 클라이언트 수·분포 모수 미보고, attribution 충실도의 ground-truth 정량 비교 부재(의도적 배제), 클라이언트가 local Hessian eigenpair를 계산·업로드해야 하는 추가 부담(통신·계산 모두 vanilla FedAvg 초과), CNN/MLP 스케일까지만 검증.
- **향후 과제 (Future Work)**: real-time data pricing 활용 가능성 언급 수준. (우리 관점: 우리 LLM-scale 실측에서 eigsh 비용·불안정과 약한 검출력이 확인됨 — "curvature를 쓰되 어디서 쓰는가"(per-client local Hessian sketch vs Flirds의 server-side validation HVP 1회/라운드)가 갈림길임을 보여주는 대조군.)

---

# Part B — Detection baselines (threat-matched)

## B1. STD-DAGMM — Free-riders in Federated Learning: Attacks and Defenses

> **Flirds 비교에서의 역할**: **free-rider 위협 매칭** 검출 baseline이자 유일한 model-free(=gradient 미사용) 독립 비교군(FLTrust는 cosine ≈ Flirds-1st라 auxiliary). 구현 `flirds/baselines/std_dagmm.py` — per-(client,round) pooling(N=5 퇴화 해결) + signed feature-hash 5.6M→256차원(std는 full vector에서 계산, 축소 무관). 우리 실측: synthetic AUROC 1.0; **real 1B N=100 free-rider(random) 0.628** — model-free 검출기는 std-매칭 random 방향과 진짜 LoRA 업데이트를 gradient 없이 분리 못함(FLTrust/Flirds-1st는 같은 위협에서 1.0); tier1 silo5(06-10) FR-zero 0.083 / FR-random 1.0. 이 위협 모델의 **첫 PEFT-scale 시험**이 우리 실험.

### 개요 (Overview)

- **논문 정보**: *Free-riders in Federated Learning: Attacks and Defenses*. Jierui Lin, Min Du, Jian Liu (UC Berkeley). arXiv:1911.12560 (2019-11, preprint — 미게재). 인용 ~145.
- **연구 분야 (Keywords)**: federated learning, free-rider attack, anomaly detection, DAGMM, incentive gaming.
- **문제 정의 (Problem Definition)**: **free-rider 공격의 최초 정의** — 데이터/컴퓨트 없이 그럴듯한 가짜 업데이트를 제출해 글로벌 모델·보상을 받는 클라이언트를, per-client history 없는 서버가 한 라운드 업데이트만으로 검출하는 task.
- **연구 배경 및 필요성 (Motivation)**: FL 인센티브 모델은 "기여하면 보상"을 전제하는데, holdout-validation 검사는 영리한 가짜(utility를 거의 안 바꿈)에 회피된다. 가짜 업데이트의 통계적 시그니처를 잡는 고차원 이상탐지가 필요.
- **핵심 기여 (Main Contribution)**: ① 공격 분류 체계 — zero-weights / random-weights($[-R,R]$, $R$ 튜닝으로 benign std 모사) / delta-weights(이전 라운드 글로벌 차분 재활용 — 수렴 근처에선 진짜와 구별 불가) / advanced delta(+$\mathcal{N}(0,10^{-3})$), ② **STD-DAGMM** — DAGMM(deep AE 임베딩 + 재구성 거리 → GMM energy)에 업데이트 벡터의 **표준편차 1개 스칼라**를 결합해 lr 전 구간에서 동작하는 검출기.

### 제안 방법론 (Proposed Method)

- **핵심 아이디어**: 0으로 스케일되거나 남의 평균을 재활용한 가짜 업데이트는 **분산이 비정상적으로 작다** — AE/GMM이 놓치는 케이스를 std 피처가 받치고, std만으로 안 되는 케이스($R$ 튜닝)는 DAGMM의 cosine/재구성 항이 받친다.
- **모델 구조 및 알고리즘**: flatten한 업데이트 → deep AE 저차원 임베딩 $z_c$ + 재구성 거리 2종 + **std 스칼라**를 스택 → estimation network가 GMM 파라미터 추정 → energy 높은 클라이언트 = 이상. 학습은 클라이언트 업데이트 풀에서 비지도.
- **사전 지식 (Preliminaries)**: DAGMM(Zong et al. 2018, 고차원 비지도 이상탐지), FedAvg($M_{j+1}=M_j-\eta\cdot\mathrm{avg}\,G$ — delta-weights 공격이 이 식에서 직접 유도됨), GMM energy 기반 스코어링.

### 실험 및 검증 (Experiments & Evaluation)

- **실험 환경 (Setup)**: **MNIST + 2-layer MLP(~0.2M 파라미터) 단일**. IID + pathological non-IID. 단일/다수 free-rider, 학습 초기(round 5)와 수렴기(round 80) 모두 평가.
- **실험 결과 (Main Results)**: ① zero는 자명하게 검출, random은 plain AE 회피하나 DAGMM cosine 항이 잡음, delta는 DAGMM이 $\eta\approx1$에선 잡고 작은 $\eta$에서 실패, advanced delta는 plain DAGMM **완전 실패** — STD-DAGMM만 전 구간 wide margin으로 검출, ② 20/100 free-rider advanced-delta: **AUC 0.96(r5)/0.91(r80) vs DAGMM 0.89/0.85**, 전 비율에서 우위(비율 상승 시 둘 다 저하), ③ DP privacy-amplification(클라이언트가 $1/q$ 라운드마다 참여)은 delta 누적을 키워 오히려 검출을 도움.
- **분석 (Analysis)**: std 단독으론 불충분($R$/$\sigma$ 튜닝으로 benign std 모사 가능), DAGMM 단독도 불충분(advanced delta) — **결합이 요점**. secure aggregation 환경에선 서버 측 검출 자체가 불가함을 인정.

### 결론 및 한계 (Conclusion & Discussion)

- **한계점 (Limitations)**: MNIST + 2-layer MLP뿐(PEFT/LLM·대형 모델 증거 없음), threshold 의존(energy 컷), free-rider 비율 상승 시 저하, no-collusion·전-라운드-가짜 가정, secure aggregation이면 무력.
- **향후 과제 (Future Work)**: delta/advanced-delta급 지능형 free-rider에 대한 robust 검출(우리 task9로 이연된 정확히 그 지점). (우리 관점: Flirds는 free-rider를 **부호 있는 가치의 부산물**로 강등시킴 — $\langle-\nabla\ell^{val},\Delta w_k\rangle\approx0$ → φ≈0 — 이라 AE/GMM 기계 없이 같은 위협을 처리하며, unsigned anomaly 점수의 non-IID 오탐 페널티도 없다는 게 대조점.)

---

## B2. FLTrust — Byzantine-robust Federated Learning via Trust Bootstrapping

> **Flirds 비교에서의 역할**: gradient-사용 검출기 대표이자 Flirds-1st와의 **구조 동일성**을 드러내는 거울. 구현 `flirds/baselines/fltrust.py` — root = 서버 val set, $g_0=-\nabla_{val}(w_r)$(cosine은 scale-free라 1-step이면 충분), 검출 점수는 **signed cosine**(ReLU 제거 — ReLU는 benign(<0)과 free-rider(~0)를 모두 0으로 뭉개 부호를 지움; ReLU+크기 정규화는 집계 게이트라 per-client ranking 불변). 따라서 $\cos(\Delta w_i,-\nabla_{val})$ = **normalized Flirds-1st와 정확히 동일 신호** → 독립이 아닌 auxiliary baseline. 우리 실측: free-rider AUROC **1.0**(N=100; free-rider 5명이 의심 top-5), poison(scaled) 1.0.

### 개요 (Overview)

- **논문 정보**: *FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping*. Xiaoyu Cao (Duke), Minghong Fang, Jia Liu (Ohio State), Neil Zhenqiang Gong (Duke). NDSS 2021. DOI 10.14722/ndss.2021.24434. arXiv:2012.13995. 인용 ~788(+arXiv 35) — 비교군 중 최다 인용.
- **연구 분야 (Keywords)**: Byzantine-robust aggregation, model poisoning defense, root of trust, cosine similarity, trust score.
- **문제 정의 (Problem Definition)**: 클라이언트 다수가 악성일 수 있는 FL에서 글로벌 모델을 방어하는 robust 집계. 기존 Krum/Trimmed-mean/Median은 클라이언트 간 상호 비교만 하므로 **신뢰 기준점(root of trust)이 없어** 정교한 local-model-poisoning에 뚫린다는 갭.
- **연구 배경 및 필요성 (Motivation)**: FedAvg는 악성 1명에도 붕괴 가능; 통계 robust 집계는 악성 비율이 크거나 non-IID면 실패. "서버가 아주 작은 clean 데이터를 직접 갖는다"는 현실적 가정 하나로 신뢰의 뿌리를 만들 수 있다는 발상.
- **핵심 기여 (Main Contribution)**: ① 서버 root dataset(~100개)로 server update $g_0$를 만들고, ② ReLU-clipped cosine trust score + 크기 정규화 + 신뢰 가중 평균의 집계 규칙, ③ strongly-convex 가정 하 임의 악성 수에 대한 $\|w_t-w^*\|$ bound, ④ 6개 데이터셋 × 다양한 공격에서 **40–90% 악성 비율까지 견디며** 무공격 FedAvg 정확도 유지.

### 제안 방법론 (Proposed Method)

- **핵심 아이디어**: "서버 자신의 작은 clean 데이터로 만든 업데이트 방향과 얼마나 정렬되는가"를 신뢰로 정의 — 반대 방향은 0점, 크기는 서버 업데이트 크기로 강제 통일.
- **모델 구조 및 알고리즘**: 매 라운드 서버가 root $D_0$로 $g_0$ 계산. 클라이언트 $i$: ① $TS_i=\mathrm{ReLU}(\cos(g_i,g_0))$, ② $\bar g_i=\frac{\|g_0\|}{\|g_i\|}g_i$(scaled poison 무력화 + 과소 업데이트 확대), ③ $g=\frac{1}{\sum_j TS_j}\sum_i TS_i\bar g_i$, $w\leftarrow w+\alpha g$. 방향이 게이트, 크기가 정규화 — 둘 다 ablation으로 필요성 입증.
- **사전 지식 (Preliminaries)**: Byzantine-robust 집계 계열(Krum/Trim/Median)과 그 한계, local model poisoning 공격(Fang et al.), cosine 유사도 기반 신뢰, strongly-convex 수렴 해석.

### 실험 및 검증 (Experiments & Evaluation)

- **실험 환경 (Setup)**: MNIST(IID/non-IID), Fashion-MNIST, CIFAR-10, CH-MNIST, HAR 등 6개. 공격: label-flipping / Krum 공격 / Trim 공격 / Scaling(backdoor) / adaptive(집계 규칙을 아는 공격자). root 크기·분포 bias·악성 비율 sweep.
- **실험 결과 (Main Results)**: ① 공격 하에서도 무공격 FedAvg 대비 test error 증가 ≤ 0.04, backdoor 성공률 ≤ 0.03 — Krum/Trim/Median은 같은 조건에서 붕괴, ② **root ~100개면 충분**(<50이면 저하), MNIST-0.5에서 악성 90%(backdoor 95%)까지 견딤, ③ root 분포 bias ≤ ~0.4까지 강건, 강한 class-bias root에선 실패.
- **분석 (Analysis)**: ReLU 게이트와 크기 정규화 각각 제거 시 무너짐(ablation). adaptive 공격에도 bound 유지. root 분포 민감도가 실질적 운영 조건.

### 결론 및 한계 (Conclusion & Discussion)

- **한계점 (Limitations)**: ① **clean root 가정** — 오염된 root에 대한 무보장(서버 validation set을 쓰는 모든 방법이 공유하는 리스크; Flirds도 동일 구조), ② 강한 class-bias root에서 실패(validation 대표성이 신호 품질의 상한), ③ 이론이 strong convexity + 1 local step, ④ hard gate라 "benign인데 방향이 다른"(non-IID) 클라이언트를 0으로 죽일 수 있음.
- **향후 과제 (Future Work)**: zeroth-order adaptive 공격 분석의 확장. (우리 관점: 같은 서버-측 신뢰 데이터를 ① 필터링(Fang), ② 신뢰 가중(FLTrust), ③ **부호 있는 가치 평가(Flirds)** 로 쓰는 세 갈래 중 마지막이 우리 — cosine이라는 coarse 방향 필터 vs 2차항까지 읽는 fine-grained 기여 추정의 대비가 비교 포인트.)

---

## B3. FLDetector — Defending FL Against Model Poisoning via Detecting Malicious Clients

> **Flirds 비교에서의 역할**: **poisoning 위협 매칭** 검출기(threat-matched suite). 구현 `flirds/baselines/fldetector.py` — model-free from-logs 포트(Byrd-Nocedal compact L-BFGS HVP + Cauchy MVT 1-step 예측; R<10 적응; cross-device는 per-client **gap-HVP** — full participation에서 per-round 예측과 bit-identical). 우리 실측: **최저가 ~24s**(model-free CPU linalg)지만 noisy 0.50/FR 0.75(06-07) — 이는 위협 불일치였음(`answer_swap`은 정직-but-noisy지 crafted update가 아님; 06-08 poisoning으로 재배치). poison(scaled γ)에선 AUROC **1.0**. 매 seed에서 *clean* math 클라이언트가 최고 의심점수 = 논문이 자인한 non-IID erosion의 LLM-scale 재현 — "separator는 진짜 어렵다"는 우리 프레이밍의 근거.

### 개요 (Overview)

- **논문 정보**: *FLDetector: Defending Federated Learning Against Model Poisoning Attacks via Detecting Malicious Clients*. Zaixi Zhang (USTC), Xiaoyu Cao, Jinyuan Jia, Neil Zhenqiang Gong (Duke). ACM KDD 2022. arXiv:2207.09209. 인용 ~296(+10).
- **연구 분야 (Keywords)**: model poisoning, malicious-client detection, temporal consistency, L-BFGS Hessian approximation, unsupervised detection.
- **문제 정의 (Problem Definition)**: 다수(기본 28%)의 공격자-통제 클라이언트가 보내는 crafted 업데이트(untargeted + backdoor)를 **서버 validation 데이터 없이** 식별·제거하는 task — 제거 후 Byzantine-robust 집계가 작동할 수 있는 조건을 만드는 전처리.
- **연구 배경 및 필요성 (Motivation)**: Krum/Trim/Median은 소수 악성만 견디고, FLTrust는 clean root가 필요하며, VAE 검출기도 clean validation 필요 + 통계적으로 비슷한 업데이트엔 실패. "validation-free + 대규모 악성" 조합이 공백.
- **핵심 기여 (Main Contribution)**: ① **모델-업데이트 시간 일관성** 신호 — benign 클라이언트의 업데이트는 자신의 과거로부터 예측 가능하다, ② Cauchy mean-value theorem + L-BFGS 적분 Hessian 근사로 업데이트 예측 $\hat g_i^t=g_i^{t-1}+\hat{\mathbf{H}}^t(w^t-w^{t-1})$, ③ 예측 오차의 정규화 거리 N-라운드 평균 = suspicious score, Gap statistic + 2-means로 비지도 판정, ④ 검출-제거 후 학습 재시작으로 높은 악성 비율에서도 downstream 방어 성립.

### 제안 방법론 (Proposed Method)

- **핵심 아이디어**: 정직한 클라이언트는 자기 데이터로 일관되게 학습하므로 라운드 간 업데이트가 곡률로 설명되는 궤적을 그리지만, crafted 업데이트는 그 예측에서 반복적으로 이탈한다.
- **모델 구조 및 알고리즘**: 글로벌 모델·업데이트 차분의 윈도(N=10)로 L-BFGS 컴팩트 Hessian $\hat{\mathbf{H}}^t$를 만들고(클라이언트 공유), 클라이언트별 예측 오차 $\|\hat g_i^t-g_i^t\|_2$를 라운드 내 $\ell_1$ 정규화 → 과거 N라운드 평균이 점수. Gap statistic으로 클러스터 수 판정 후 2-means 상위 클러스터 = 악성. **Theorem 1**($\mathbb{E}(s^{benign})<\mathbb{E}(s^{malicious})$)은 **명시적 IID 가정 하에서만** 성립. 클라이언트 추가 비용 0, 서버 비용 ~선형.
- **사전 지식 (Preliminaries)**: Cauchy MVT(적분 Hessian), L-BFGS 2-loop/compact form, model poisoning 공격 계열(Fang/Scaling/DBA/A-Little-Is-Enough), Gap statistic·k-means.

### 실험 및 검증 (Experiments & Evaluation)

- **실험 환경 (Setup)**: MNIST / CIFAR-10(ResNet20) / FEMNIST. 공격: Fang(untargeted), Scaling·DBA·A-Little-Is-Enough(backdoor). 악성 28% 기본, non-IID degree 0.5 기본. 비교: VAE 검출기, FLD 변형(ablation).
- **실험 결과 (Main Results)**: ① DACC ≈ 0.85–1.00, FEMNIST에서 FNR ≈ 0 — VAE/FLD-Norm/FLD-NoHVP 능가(**HVP 항이 중요**하다는 ablation 포함), ② 검출-제거 후 Median 집계가 near-clean 정확도 회복, backdoor 성공률 ~50–95% → **~2%**, ③ adaptive 공격(예측에 맞춰 정규화한 crafted 업데이트)에는 DACC가 내려가도 공격 효력 자체가 낮게 유지 — 회피와 공격력의 트레이드오프.
- **분석 (Analysis)**: **non-IID 민감도를 스스로 보고** — Thm 1은 IID 한정이고, 실험적으로도(Fig.2) non-IID degree가 공격별 임계 넘으면 정확도 하락. 하이퍼파라미터(N, B)에는 둔감.

### 결론 및 한계 (Conclusion & Discussion)

- **한계점 (Limitations)**: ① **IID-only 이론 보장 + non-IID 성능 침식**(논문 자인) — benign 이질성과 악성 이탈의 분리가 본질적으로 어려움, ② 검출 대상이 *crafted-update* 공격자뿐 — 정직-but-noisy/OOD 클라이언트는 표적이 아니며 오히려 non-IID에서 오탐 원천, ③ 이론은 1-step full-batch GD인데 실제 FL은 multi-step local SGD(우리 $\Delta w_k$ regime — 이 간극은 논문에서 미실험), ④ 공유 단일 Hessian 근사가 클라이언트 곡률 발산 시 깨질 가능성.
- **향후 과제 (Future Work)**: adaptive 공격 내성의 형식화. (우리 관점: 우리 실측이 ③의 간극을 처음 채움 — multi-step LoRA 궤적에서 clean non-IID 클라이언트가 최고 의심점수를 받는 erosion을 재현했고, 위협을 맞췄을 때(scaled backdoor)만 1.0이 나옴. "어느 위협에 어느 신호인가"의 정확한 사례.)

---

## B4. FedDQC — Data Quality Control in Federated Instruction-tuning of LLMs

> **Flirds 비교에서의 역할**: **data-quality(`answer_swap`) 위협 매칭** 검출기 — 비교군 중 유일하게 LLM-native + 유일하게 클라이언트 **데이터 접근**을 요구(우리 다른 방법들은 전부 server-side logs/loss_fn에서 동작). 구현 `flirds/baselines/feddqc.py` — IRA를 per-sample 계산 → 클라이언트 평균 → suspicion = −IRA(오염 클라이언트가 높음). 우리 실측(smoke): noisy(answer_swap, medical) AUROC **1.0**; caveat — 도메인 간 IRA 분산이 큼(clean finance 0.17 ≈ noisy medical 0.067) → matrix에서 noisy 도메인·seed 변주 필요. FedDQC의 50% response-swap 오염 설계는 우리 `answer_swap` corruptor의 정당화 인용원이고, Table-1 미러(FiQA·AQUA)가 Track D-옵1.

### 개요 (Overview)

- **논문 정보**: *FedDQC: Data Quality Control in Federated Instruction-tuning of Large Language Models*. Yaxin Du, Rui Ye, Fengting Yuchi, Yanfeng Wang, Siheng Chen (SJTU/Shanghai AI Lab), Wanru Zhao (Cambridge), Jingjing Qu (Shanghai AI Lab). **ACL 2025 Findings** (DOI 10.18653/v1/2025.findings-acl.791). arXiv:2410.11540 (2024-10). 인용 2(게재 직후).
- **연구 분야 (Keywords)**: federated instruction tuning, LLM, data quality, IRA(instruction-response alignment), curriculum/hierarchical training, LoRA.
- **문제 정의 (Problem Definition)**: federated instruction tuning에서 서버가 클라이언트 데이터를 보지 못한 채 **저품질/오염 샘플을 거르고 학습 순서를 제어**하는 task. 클라이언트는 자기 로컬만 보므로 글로벌 품질 기준이 없다.
- **연구 배경 및 필요성 (Motivation)**: 휴리스틱 품질 지표(PPL/IFD/NUGGETS)는 중앙집중·무노이즈 가정, attribution 계열(Shapley/IF/DataInf)은 자원 제약 클라이언트에 과중, 분류용 FL 품질 기법(label-noise 정정 등)은 생성 태스크에 전이 안 됨 — "on-device·저비용·프라이버시 보존·생성-인지" 4박자가 공백.
- **핵심 기여 (Main Contribution)**: ① **IRA** — $f_{IRA}((q,a);\theta)=L(a;\theta)-L(a|q;\theta)$: instruction 조건부/무조건부 응답 loss 차이로 "instruction이 answer를 설명하는 정도"를 측정하는 on-device 1-pass 지표(학습 시간의 ~1%, DataInf 스코어링의 ~1/150), ② IRA 내림차순 **계층(hierarchical) 학습** — 쉬운 것부터, 계층마다 현재 글로벌 모델로 재채점(adaptive curriculum), ③ LLaMA-2-7B + LoRA, 임의 집계 규칙과 호환.

### 제안 방법론 (Proposed Method)

- **핵심 아이디어**: "좋은 instruction-response 쌍이라면 instruction을 주는 것이 response 생성을 크게 쉽게 만든다" — 그 loss 격차(상호정보 류 신호)를 품질로 읽고, 모델이 자라면서 품질 기준도 같이 갱신한다.
- **모델 구조 및 알고리즘**: ① 서버가 글로벌 모델 배포 → ② 각 클라이언트가 로컬에서 IRA 계산·정렬·임계 $\lambda$ 이상 선별·$K$개 계층 분할 → ③ 최고-IRA 계층부터 federated 학습 → ④ 다음 계층 전 현재 글로벌 모델로 재채점 → 반복. 데이터는 클라이언트 밖으로 나가지 않음(프라이버시 보존). FedAvg/FedAvgM/FedAdagrad/FedYOGI/FedAdam과 호환 검증.
- **사전 지식 (Preliminaries)**: instruction tuning과 completion loss, 품질 지표 계열(PPL/IFD/NUGGETS)·attribution 계열(DataInf)의 위치, curriculum learning(easy-to-hard), LoRA.

### 실험 및 검증 (Experiments & Evaluation)

- **실험 환경 (Setup)**: synthetic 오염 — PubMedQA / FiQA / AQUA-RAT / Mol-Instructions에 **50% 노이즈**(response swap), IID + non-IID; real-world — Fed-WildChat(70% subset). 모델 LLaMA-2-7B + LoRA. baseline: PPL, IFD, NUGGETS, DataInf + full-data FedAvg.
- **실험 결과 (Main Results)**: ① synthetic 전 셋에서 baseline 전부 능가, 일부 세팅에선 **full-clean-data oracle도 상회**, ② Fed-WildChat에서 FedDQC > 전 baseline > full-data FedAvg, ③ 비용: IRA 스코어링 ≈ 학습 시간의 1%, ④ **DataInf 기반 선별이 real-world FL에서 random보다 못함** — "이질적 클라이언트 분포에서 gradient 기반 attribution이 깨진다"는 저자 프레이밍.
- **분석 (Analysis)**: 계층 순서 ablation — 내림차순(high→low IRA) > random > 오름차순; **계층 학습의 이득은 IRA에만 존재**(PPL/IFD/NUGGETS/DataInf로 계층화하면 무익) → IRA가 training-difficulty 신호를 담는다는 해석. 재채점(adaptive)이 정적 커리큘럼과의 차별점.

### 결론 및 한계 (Conclusion & Discussion)

- **한계점 (Limitations)**: ① IRA는 **quality 지표지 contribution 지표가 아님** — "쉬운" 샘플과 "기여하는" 샘플은 다르며, 쉬운 중복 샘플을 고평가할 가능성(quality vs value 혼동은 이 문헌 전반의 문제), ② curriculum learning의 일반 caveat(쉬운 데이터 조기 정체 등) 상속, ③ 노이즈 50% 단일 수준 외 민감도 미탐구, ④ 클라이언트 on-device 계산·데이터 접근 전제 — 서버-측 from-logs 방법과 비용 구조가 다름.
- **향후 과제 (Future Work)**: (우리 관점 포함) IRA와 influence function의 형식적 관계 규명(특정 근사 하 동치인지), DataInf가 FL 이질성에서 깨지는 메커니즘 분석 — 후자는 "gradient 기반이 FL에서 어렵다"는 주장의 반례를 Flirds가 제공하는지가 우리 실험의 흥미로운 부속 질문. 검출 비교에선 유일한 data-access 방법이라는 비용 비대칭을 명시하고 읽어야 함.

---

# 부록 — 비교군 전체에서 본 위치 요약

**계보**: FedSV(2020, per-round SV 정의) → GTG(2022, 평가 횟수 절감) · ComFedSV(2022, 미관측 보간) → ShapleyFL(2023, 가공-SV로 개입) → [semivalue 통제: Banzhaf(2023)] → FedIF(2025, SV 포기·1st-order) · Ripple(2026, in-run+curvature·sample-level). detection 축은 위협별 매칭: free-rider→STD-DAGMM(2019), poisoning→FLTrust(2021)·FLDetector(2022), data-quality→FedDQC(2025).

**우리 실측 종합**(1B N=5 3-seed, lr 1e-3 기준; 상세 §02·§03 문서): valuation 전 방법 Spearman vs (b) oracle +1.000(N=5 near-additive 구간) — 차이는 **runtime**(Flirds-1st 35s / Flirds 107s vs SV-계열 ~530s / Ripple ~4,515s)과 **free-rider φ의 exact-0 여부**, 그리고 **poison 위협에서의 분기**(FedSV +0.367 추락, Flirds-1st AUROC 0.0 회피 vs loss-heur/(b)는 포착). detection은 위협 매칭이 결정적: 매칭 시 1.0, 불일치 시 0.5 부근(FLDetector-noisy, STD-DAGMM의 silo5 FR-zero 0.083).

> 갱신 메모: 인용 수는 2026-06-12 OpenAlex/Semantic Scholar 조회값 — 발표자료에 옮길 때 재조회 권장. FedIF는 preprint(인용 0)라 "최근접 경쟁자" 표현은 게재 여부 추적 필요.
