---
type: conversation
date: 2026-06-12
topic: flirds
participants: [Yonghee, Claude]
tags: [experiment-design, track-c, track-d, cnn-standard-setting, llm-standard-setting, baselines, comparability]
---

# Track C/D 추가 실험 설계 — CNN 표준-세팅 + LLM 표준-세팅 비교

## Yonghee의 요청 (세션 시작)

> "기존 연구들 대부분이 컴퓨팅 파워의 제약 때문에 CNN 모델에서 이루어졌고, 우리 데이터셋과 좀 다르게 일반적인 학습 데이터(client 별 도메인이 다르지 않은)에서 shapley value를 비교하거나 성능의 수렴 속도, 정확도 등을 평가하면서 이루어졌어. … 기존 연구들과의 직접적인 비교가 어려운 것이 나중에 약점이 될 수 있을 것 같아서 선행 연구들이 검증 차원에서 했던 실험들과 같은 세팅의 CNN 모델 실험도 같이 돌리기로 했고, client 별 도메인이 다른 경우 말고 일반적인 학습 세팅에 대한 실험이 더 main이 되어야 해."

작업 구조: (1) 선행연구 실험 조사 + 설계 제안 → (2) 논의로 확정 → (3) 구현은 이후 세션에서 하나하나.

## 조사 (병렬 agent 8개: 선행 13편 실험 섹션 정밀 분석 + CNN 트랙 코드 인벤토리 + FL-LLM/중앙화-LLM 평가 관례 + 경쟁자 sweep)

### CNN — 그룹 A: SV 근사 fidelity 계열

| 논문 | 설정 | GT | metric | downstream |
|---|---|---|---|---|
| GTG-Shapley (TIST'22) | MNIST만(arch 미공개), N=10 full part., 5 시나리오: IID / label-skew(2클래스 80%) / quantity-skew(10~30%) / **graded label-flip 0/0/5/5/10/10/15/15/20/20%** / graded feature-noise | **exact retrain SV(2¹⁰ 재학습)** = 우리 (a) | cosine/Euclid/max-diff (log₁₀; **rank corr 없음**) + wall-clock | 없음. label-skew(시나리오②)에서만 오차>1e-2 |
| ComFedSV (ICDE'22) | MNIST(MLP)/FMNIST(CNN)/CIFAR-10(VGG16), N=10(30% part.)+N=100, IID+McMahan 2-class shard | full-matrix 자기자신(=(b)형) | Spearman(노이즈 랭킹), Jaccard(noisy-label), fairness ECDF(동일 클라 쌍) | noisy 식별. graded Gaussian 5·i% |
| FedSV (Wang'20) | MNIST(MLP/2conv)/CIFAR-10(2conv=우리 FedSVCNN), N=100 @10% part., McMahan shard | **없음** | 없음 | noisy(20/100@10%flip)/backdoor inspection-curve + **bottom-q% dismissal→최종 acc** |

### CNN — 그룹 B: valuation→학습 개선 계열

| 논문 | 설정 | 위협 | 평가 | 개입 메커니즘 |
|---|---|---|---|---|
| ShapleyFL (KDD'23) | CIFAR-10/FMNIST, **N=100 C=0.1**, R=100–150, E=5, 2-shard | long-tail/open-set 50%/label-flip(L+1)%C 25%/feature-noise/grad-poison 25% | acc-vs-round(5 seeds)+round-1 MSE vs 2000-perm MC | SV→min-max→EMA→**가중 대체** + importance-sampling 선택 |
| S-FedAvg (AAAI'21) | MNIST 5-class, K=10/m=5 | irrelevant 4/10 | acc 곡선(수치표 없음) | φ→softmax **선택**(집계는 균등) |
| FedIF (arXiv'25) | CIFAR-10/FMNIST, N=100 C=0.1 T=100, **Dir(α=1)** | label-flip(ρ·u그리드)/grad-noise σ/PGD | 최종 acc±5seed + 집계시간(0.18s vs AFedSV 91.8s) | influence→min-max→EMA→**가중 대체**(≈normalized Flirds-1st) |
| Ripple (AAAI'26) | MNIST(MLP)/CIFAR-10(CNN), ShapleyFL 프로토콜 | non-IID/long-tail/open-set/data-noise | **fidelity 명시 거부**; acc-vs-round(λ=0.5 가중)+누적 runtime 표(62×) | **w=λ·u+(1−λ)·n/Σn additive 혼합, λ=0.5** |

### CNN — 그룹 C: valuation 원류 + 평가방법론

- Ghorbani-Zou'19 canonical: discovery curve(검수율→발견율) / removal·addition curves / 소규모 exact 대비 Pearson 98%+.
- Data Banzhaf(AISTATS'23): 10% flip→F1@10th-pct; **학습 무작위성 랭킹 안정성**(k-회 평균 utility, Spearman vs k=50 기준; cross-run Spearman Banzhaf 0.856 vs Shapley 0.038).
- IRDS(ICLR'25): 1-step MC-1000-perm GT, RMSE+Spearman; CIFAR-1000pt sanity AUROC 0.68.
- **Volatility(Geimer, 2만+ run)**: 집계전략만 바꿔도 기여도 share 30–50% 요동, seed 분산이 전략차 압도 → "안정성을 GT-유사도와 함께 보고하라"(분야 유일 평가 처방). 우리 frozen-trajectory exact-(b) 프레이밍의 정당화 인용원.
- FedCorr(CVPR'22) label-noise convention: 확률 ρ로 noisy 클라, 각자 u~U(τ,1), uniform-random 재라벨, 5 trials.

### 공백 = 기회 (4가지)

1. 그룹 A+B를 한 방법으로 모두 커버한 논문 없음 (fidelity 계열은 학습개선 안 보고, 학습개선 계열은 GT 대비 fidelity 안 봄).
2. (a) retrain GT + (b) in-run GT 동시 사용 논문 전무 → CNN N=10에서 (a)=(b)=estimator 3-way는 우리만 가능.
3. Ripple(최근접 경쟁자)은 fidelity·detection·stability 실험 0개.
4. 하이퍼파라미터 미공개(GTG는 arch조차) → 숫자 직접 비교가 아니라 "파티션+corruption+metric 레이어 매칭"이 현실적 목표.

### 우리 CNN 트랙 인벤토리 (~70% 준비)

- 있음: MNIST/FMNIST/CIFAR-10 로더, IID/Dirichlet/per-client-Dir 파티션, LeNet5(66K)/FedSVCNN(614K=FedSV arch), partial-part. FedAvg+로그, estimator, (a)+(b) oracle, baseline 전 CNN 경로(FedDQC 제외; Ripple은 phase0 검증됨·러너 wiring만), AUROC/Spearman, RunLogger.
- 신규 필요: **label_flip corruptor**(현 label_shuffle은 100% 셔플만), **개입 루프**(가중/선택/dismissal), CNN matrix 러너, 수렴곡선 다시드 집계.

### LLM — FL-LLM 표준 스택 (OpenFedLLM 중심)

- 표준: **OpenFedLLM 프레임워크**(KDD'24; FedDQC가 그 위 구현, 로컬 참조 클론 보유) — Llama-2-7B+LoRA(r8–64)+8bit+Alpaca 템플릿, cross-silo **N=5–20, 2/round, 10 steps×batch16, 100–200R**, AdamW cosine. cross-device는 FedKSeed/FedHDS 계열(200–738 클라, 5%, 40–60R).
- **FedDQC corruption = 50% response-swap = 우리 `answer_swap`과 본질 동일**(그들=데이터셋 전역 쌍 교환, 우리=클라 내 순열) → 정당화 인용원 확보. 변형 메뉴(delete/cut/substitute/token-noise) appendix에 존재.
- 데이터 겹침: 우리 medical(`medical_meadow_medical_flashcards`)=OpenFedLLM·FlowerTune medical 학습셋 그 자체; FiQA·AQUA-RAT=FedDQC 도메인; Dolly=FedHDS 파티션 소스. legal만 무전례.
- 평가 2문화: GPT-4-judge(MT-Bench/Vicuna/win-rate — **API 없어 불가**) vs close-ended 로컬(MMLU/FPB·FiQA-SA·TFNS/MedQA·PubMedQA·MedMCQA/HumanEval/Rouge-L — 전부 가능). FlowerTune이 Alpaca-GPT4 학습+MMLU 로컬 평가 전례.
- 중앙화-LLM 관례: detection canonical=DataInf(GLUE 20% flip→AUROC, RoBERTa 규모; LLM-scale에선 proxy로 후퇴); selection canonical=LESS(Tülu 270k pool, top-5%, MMLU/TydiQA/BBH, random-q%+full-100% 대조). **LLM-scale noisy-instruction AUROC 벤치마크 부재** = 인용 가능한 공백.
- 경쟁자 sweep: **LLM-scale FL valuation 직접 경쟁자 없음**(2026-06 기준). 인용·대조용 인접 3편: LM-arithmetic DPO Shapley(arXiv:2512.15765, 중앙화 n=4), TraceFL(ICSE'25, per-prediction attribution), CLAIR(toy). 제출 전 Ripple/FedDQC/GTG 인용그래프 패스 권장.

## 확정된 설계

### Track C1 — CNN fidelity & cost (cross-silo; GTG 무대 + 듀얼 oracle)
- MNIST+LeNet5 / CIFAR-10+FedSVCNN, **N=10 full participation**, R=10–30, E=5, SGD mom=0, 3–5 seeds.
- 시나리오: GTG 5종(IID/label-skew80%/quantity-skew/graded label-flip 0–20% ladder/graded feature-noise) + free-rider 옵션.
- GT: **(a) exact retrain SV(2¹⁰, val-loss utility, fp32) + (b) exact in-run** (acc-utility는 (b) 부록 후보).
- metric: Spearman/Kendall + cosine/Euclid/max-diff(GTG 표 호환) + wall-clock.
- 비교: Flirds-1st/2nd, GTG, FedSV, ComFedSV, Banzhaf, ShapleyFL, FedIF, loss-heur, **Ripple(포함; eigsh iteration-cap/timeout guard)**.

### Track C2 — CNN 일반 성능 (cross-device; 추가분의 메인)
- CIFAR-10+FMNIST, **N=100, C=0.1, T=100–150, E=5**, 5 seeds; 파티션 {IID, Dir(α=1), 2-shard}.
- 위협: clean / label-flip(FedCorr (ρ,τ) convention) / free-rider / grad-noise(FedIF σ).
- 개입 3종 모두: ① **가중집계**(규칙 3종: **곱셈형 w∝n_i·s_i 메인** + 대체형 w∝s_i[FedIF/ShapleyFL 관례] + additive λ=0.5[Ripple 관례]) ② selection(S-FedAvg식 softmax) ③ bottom-q% dismissal(FedSV식). baseline 각자는 자기 논문 메커니즘으로.
- 평가(전 메커니즘 공통): detection AUROC+discovery curve / 최종 acc±seed / acc-vs-round / rounds-to-target. (b)-perround는 anchor 1–2 config만.

### Track C3 — stability (보고 축; 추가 비용 0)
- C1/C2 다시드 데이터에서 cross-seed Spearman + top/bottom-k% 일관성 method별 추출 (Banzhaf 프로토콜 + Volatility 처방 응답).

### Track D — LLM 표준 세팅 (전부 API-free)
- **D-메인**: Alpaca-GPT4 20k **IID** 분할(OpenFedLLM·FlowerTune general 표준), N=5 full part.(+N=20 2/round 확장 옵션), answer_swap 50%(FedDQC convention). 평가: AUROC+Spearman vs (b) / **φ-bottom 필터링 후 재학습→MMLU**(FedAvg-mixed 하한·clean-oracle 상한·random-q% 대조) / 비용.
- **D-옵1 (FedDQC Table-1 미러)**: FiQA 또는 AQUA-RAT(파이프라인 기존재 데이터), N=5, 8k, 50% swap — published 숫자 옆 비교. GPT-judge 컬럼(FiQA win-rate 등)은 로컬 메트릭 대체.
- **D-옵2 (FedHDS 미러, cross-device)**: Dolly-15k category-Dirichlet(α∈{0.5,5}) 200 클라 → held-out task Rouge-L (FedIT/Random/Perplexity 4-seed 앵커 published).
- 모델: 1B/3B=Llama-3.2-Instruct + **7B=meta-llama/Llama-2-7b-hf**(plan task8 원결정; FL-LLM 문헌 표준 모델과 일치 → 절대 수치 비교 가능, 남는 차이=optimizer). HF 토큰 Llama-2 접근 확인(Yonghee).

## 결정 로그 (Yonghee, 06-12)

1. **momentum**: "momentum은 0으로 통일하고 유지해야해." — 전 트랙, baseline 포함 통일.
2. **모델**: "그냥 우리가 기존에 구현해놓은 모델로 하자." — LeNet5/FedSVCNN 유지, CNNCifar 재현 안 함.
3. **noise**: "image classification task에서는 보통 label flip을 쓰는거 아니었나?" → 확인 맞음(baseline 전부 flip 계열) → **label_flip corruptor 신규**(샘플별 uniform-random 재라벨; per-client rate 배열로 GTG ladder·(ρ,τ) 모두 표현). label_shuffle은 LLM-정합 비교용으로 존치.
4. **개입**: "1번을 기본으로 하는데 2번, 3번도 같이 돌려보자." 혼합 가중의 의미 정정: "원래 fedavg에서 데이터 개수로 곱해지는 가중치에다가 추가로 기여도를 곱하자" = **곱셈형 w∝n_i·s_i**. 조사 결과 비교군 8종 중 곱셈형 전례 없음(ShapleyFL/FedIF/FedTSV=대체, Ripple=additive 혼합, S-FedAvg=균등+선택, FLTrust=trust×정규화업데이트) → **Flirds 고유 가중 규칙로 채택**, 대체형·additive는 통제 비교용. 수렴속도 평가는 모든 메커니즘 공통(①에도 acc-vs-round 적용 — ShapleyFL/FedIF/Ripple이 ①로 보고한 그 평가).
5. **순서**: "순차로 하자" — C1→C2 stage-gate.
6. **Ripple**: "cnn 논문이니까 cnn 비교에는 당연히 포함돼야하고 구현은 논문 그대로." LLM eigsh(Lanczos 수렴 실패로 CPU 스핀; RIPPLE=0 제외 이력) 교정/제외는 별도 세션.
7. **LLM**: "전부 다 돌려 보자"(D-메인+옵1+옵2). Alpaca-GPT4=정적 데이터셋, API 불필요 확인. "optimizer는 mom=0이야 이건 변함 없어."(문헌 AdamW와의 차이는 caveat 명시; FedIT-SGD 참조 숫자 FedHDS Table 2 존재).

## 코드 변경 (이 세션, 미커밋)

- `experiments/phase2_matrix.py`: SCALE 파싱에 `"7B" if "Llama-2-7b" in MODEL` 분기 추가 + docstring에 7B=meta-llama/Llama-2-7b-hf 명시 (기존 파싱은 Llama-3.2 외 모델에서 MODEL_CFG 키 불일치=KeyError 예정이었음). 3개 모델 ID 파싱 검증 완료.

## 다음 세션 (구현)

- 순서: label_flip corruptor → C1 러너 → 개입 루프(가중 3규칙/selection/dismissal) → C2 러너 → D 러너.
- 구현 메모: φ 음수 가능 → 가중 사용 전 min-max 정규화+EMA 필수(FedIF/ShapleyFL 그대로). **곱셈형은 equal-n_k 세팅에선 대체형과 동일** — 차이는 size-skew(Dir(α), quantity-skew)에서만 드러남(해석 주의). 7B 첫 smoke에서 Llama-2 토크나이저(pad token 부재) 확인. C1 (a)-oracle은 CNN이라 시드당 수 시간 오더(B200 1장). FedDQC 미러 시 judge 컬럼 로컬 대체 명시.
