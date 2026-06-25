---
type: survey
title: "metric · benchmark 카탈로그 + Flirds 채택 매핑"
created: 2026-06-25
updated: 2026-06-25
tags: [survey, metrics, benchmarks, ground-truth, flirds]
---

# metric · benchmark 카탈로그 + Flirds 채택 매핑

[[validation-experiments]]의 각 셀엔 이미 *데이터셋·지표·규모*가 들어가 있다. 이 페이지는 그걸 두 차원으로 **한눈에 모으고**(§1 metric / §2 E1 ground-truth / §3 benchmark), 마지막에 **Flirds가 그중 무엇을 채택했는지**(§4)를 정리한다. 약어·E# 정의 → [[prior-work-taxonomy/README#범례 (legend)|README 범례]].

---

## §1 — 검증 유형(E#)별 대표 metric

| E# | 대표 metric | 쓰는 연구(예) |
|---|---|---|
| **E1** 충실도 | **Spearman ρ · Kendall τ · Pearson r** (순위/값-수준) · MAE/RMSE · **Jaccard@k**(top-k 집합 겹침) · **LDS**(Linear Datamodeling Score) · Cover@c / top-1 Acc(retrieval) | IF-Fragile(ρ·r) · ACC-SGD(RMSE·τ·Jaccard@10) · DataInf(r vs exact-IF) · ComFedSV(ρ·Jaccard) · MATES(ρ vs live oracle) · TRAK·LoGra·LoRIF(**LDS**) · Do-IF(top-1 Acc·Cover@c) |
| **E2** 선별 | downstream **accuracy** · **ROUGE-L** · **perplexity/loss** · **EM**(exact-match) · **rounds-to-target / R@99**(수렴) | LESS(MMLU·TyDiQA·BBH acc) · MATES·DsDm(9-task avg acc) · FedHDS(ROUGE-L) · Mavericks(R@99) · IRDS·DVEmb(수렴·loss) |
| **E3** 검출 | **AUROC/AUC** · **DACC**(detection acc) · **FNR/FPR** · **Jaccard**(검출 vs 진짜) · **ASR / backdoor-acc** · attack-rate↓ / accuracy-under-attack | FLDetector(DACC·FNR) · STD-DAGMM(AUC) · FLTrust(backdoor-succ·test-err) · FoolsGold(attack-rate) · FedCorr(acc-under-noise) · ComFedSV(Jaccard) · Do-IF(Acc) |
| **E4** 공정 | 공리 만족(efficiency/symmetry/null/additivity) · value↔양/질 상관 · reward-share gap·**CDF dominance** · IR/IC/social-welfare · reward-share 변동% | FedSV(공리) · ComFedSV(CDF) · Mavericks(under-credit Prop) · iPFL(IR/IC/SW) · Shapley-Volatility(reward swing 30–50%) · ADS(중복 non-free-ride) |
| **E5** robust | seed간 **랭킹 안정성** · **safety-margin** · val-set-size/distribution 불변 · cross-epoch fidelity · ordering 민감도 | Data Banzhaf(SGD-run 안정·safety-margin) · SPACE(val-size 불변) · IF-Fragile(depth/width 침식) · ACC-SGD(cross-epoch) · Shapley-Volatility(strategy-invariance) |
| **E6** 비용 | **wall-clock** · #model-eval / #utility-eval · #round · **throughput**(tok/s·pairs/s) · **storage**(GB/TB) · FLOPs % · compute-multiplier · 모델규모(≤52B/70B) | 거의 전부; LoGra(throughput·3.5TB) · MATES(11.5% FLOPs) · Ripple(62×) · FedIF(450×) · DsDm(2× compute) |
| **E7** 집계 | value-가중 집계의 **test accuracy** vs vanilla FedAvg (+ corruption fraction별 강건) | S-FedAvg · FedTSV · FedIF · ShapFed(class-imbalance) · FLTrust·FedCorr(detector-집계) |

> **관찰**: LLM은 exact-SV가 사실상 불가 → E1을 **LDS**(counterfactual 선형회귀) 또는 **선별 E2**로 우회한다. CNN-FL은 소N서 exact/retrain GT를 직접 쓸 수 있어 **Spearman·Jaccard 기반 E1**이 흔하다.

---

## §2 — E1의 ground-truth(정답 SV) 출처

E1(충실도)은 "무엇을 정답으로 두는가"가 핵심이다. 이 축이 비교의 강도를 가른다:

| GT 출처 | 정의 | 쓰는 연구 |
|---|---|---|
| **exact 2ᴺ Shapley** (소N) | 모든 coalition 열거(재학습 또는 in-run) — 근사 없음 | Data Shapley(정의) · Banzhaf(2ⁿ⁻¹) · **Flirds (b) in-run oracle + (a) retrain oracle** |
| **TMC / permutation-MC** | truncated/permutation Monte-Carlo 근사 SV | Data Shapley estimator · FedSV · GTG · ShapleyFL |
| **LOO-retrain** | leave-one-out 재학습 차이 | IF-Fragile · ACC-SGD · DVEmb |
| **LDS / Datamodels** | 무작위 subset 재학습 → 선형회귀(counterfactual) | TRAK · LoGra · LoRIF · DsDm |
| **PBRF** (proximal Bregman) | 재학습 대신 근접-응답 함수(=현대 IF가 실제로 계산하는 것) | Grosse/EK-FAC 검증 대상 |
| **GT 없이 안정성** | 정답 대신 *분포/크기 불변성*으로 평가 | DRDV(Wasserstein) · SPACE(val-size) |

대부분이 **근사 GT**(TMC·LOO·LDS·PBRF)를 쓴다. **exact 2ᴺ**를 정답으로 쓰는 건 소N(≤10) 정의류와 Flirds의 oracle뿐 — 소규모에서 가능한 가장 강한 기준이다(아래 §4).

---

## §3 — benchmark / dataset 카탈로그

### CNN / MLP 트랙

| 분류 | 데이터셋 | 모델 |
|---|---|---|
| 일반 이미지 분류 | MNIST · Fashion-MNIST · CIFAR-10 · CIFAR-100 · STL-10 · ImageNet · FEMNIST(연합) | logistic · 2-layer MLP · LeNet-5 · small-CNN · VGG13/14 · ResNet-18/20/50 |
| noisy-label / 강건 | Clothing1M(실세계 noise) · synthetic label-flip | (상동) |
| 연합 cross-silo 의료 | Fed-ISIC2019 · Chest X-Ray | small-CNN |
| sybil/poison 평가 | VGGFace2 · KDDCup · Amazon | (상동) |
| 비-이미지(tabular/text) | Iris · Adult · 20News | FFN |

### LLM 트랙

| 분류 | 데이터셋 / 벤치마크 | 모델 |
|---|---|---|
| eval 벤치마크 | **MMLU · TyDiQA · BBH · LAMBADA · ARC-E · SQuAD · Jeopardy · CS-Algorithms · 9-task avg · GLUE/MRPC · Emotion/Grammar/MathQA** | — |
| 사전학습/perplexity corpus | Pile · OpenWebText · WikiText-103 · C4 | GPT-2 · Pythia-410M/1B · OLMo-7B · Apertus-70B |
| instruction-tuning 데이터 | FLAN · **Natural Instructions · Dolly-15K · Alpaca/Alpaca-GPT4** · 270K pool(LESS) · Fed-WildChat · PubMedQA/FiQA/AQUA-RAT/Mol-Instructions(FedDQC) | RoBERTa · BERT · mT5 · FLAN-T5(80M–11B) · SmolLM-135M · TinyLlama · DataJuicer-1.3B · **Llama-2-7B/13B/70B · Llama-3-8B · Mistral-7B** |
| 검출 위협 데이터(공격) | SST-2 · HateSpeech · Tweet-Emotion · TREC(Xu backdoor) · Reddit-LM(Bagdasaryan) | (상동) |

> **federated × LLM에서 *공통 벤치마크가 없다*.** centralized LLM은 MMLU/LDS-corpus로 수렴해 있으나, federated-LLM valuation은 표준 무대 자체가 미정립(FedDQC=PubMedQA류, FedHDS=NI/Dolly, iPFL=자체) → Flirds도 *자체 5-domain*을 구성해야 했다(아래).

---

## §4 — Flirds가 채택한 metric · benchmark

| E# | Flirds 채택 metric | Flirds benchmark / GT | 분야 대비 비고 |
|---|---|---|---|
| **E1** | **Spearman + Kendall + Pearson**(값-수준) + **GTG 거리 trio(cosine/euclidean/max)** | **GT = exact 2ᴺ in-run oracle (b) + exact retrain oracle (a)**; utility=**val-loss**(미분가능·같은 게임); N=5 exact · N=100 exact per-round 분해 · CNN N∈{5,10}; **fp32** | 대다수는 근사 GT(TMC·LOO·LDS) — Flirds는 **exact 2ᴺ**(소N 최강 기준). Pearson은 N5 near-additive서 Spearman이 +1 포화될 때 값 격차 노출(예: silo5 poison Spearman 0.0 → Pearson −0.95) |
| **E2** | val-loss(≤random-K) · per-domain **ROUGE-L** · AQUA **EM** · (arm) **MMLU 0-shot** · **Alpaca-test ROUGE-L** | 자체 5-domain + MMLU + Alpaca | 다운스트림 지표를 **utility(val-loss)와 분리**(순환성 회피); MMLU·ROUGE는 분야 표준 채택 |
| **E3** | **AUROC**(noisy=answer_swap · free-rider=zero/random) · **ASR / soft-ASR**(poison=Xu trigger + Bagdasaryan scaled γ) · free-rider φ **exact-0** 확인 | 자체 5-domain + 위협 주입; baseline FLDetector/FLTrust/STD-DAGMM/FedDQC 비교 | 분야 표준(AUROC·ASR) 채택; LLM-FL client-level 검출은 선례 희소 → 첫 PEFT-scale 비교 |
| **E4** | (전용 보상지표 미채택) — partial-participation 동일가치 #14 · maverick 과소평가 #15 *특성화* | — | 공정성 *보장*은 비목표; 한계로 특성화([[taxonomy]] straddle) |
| **E5** | seed-determinism(CNN cudnn) · noise-vs-OOD 분리 **deferred**(한계 명시) · 2차 PGD 검증 #13 | — | replication/ordering 전용 실험 미채택; model-level/in-run 프레이밍으로 흡수 |
| **E6** | **wall-clock** + **#HVP/round = 1** vs 2ᴺ coalition sweep + estimator/round ms | **동일 frozen trajectory**(공정 비교) | 분야의 #util-eval/throughput 대신 wall-clock+HVP-count; 모든 baseline이 같은 궤적 위(5–15× 저렴, ~42× vs Ripple) |
| **E7** | value-가중 집계 arm **flirds_w**(β=0.5 곱셈) · **flirds_sel** → MMLU 0-shot + Alpaca ROUGE-L; CNN **w∝n·s** | clean-IID = **do-no-harm parity** 기대 | 집계는 부차 arm(FedIF/FedTSV처럼 집계가 *목적*은 아님 — valuation이 본령) |

### Flirds의 3가지 의도적 선택 (방법론)

1. **utility = val-loss, ROUGE 아님.** Shapley 계산을 검증하려면 estimator와 oracle이 *같은 미분가능 게임*이어야 한다. ROUGE는 argmax-text라 비미분 → estimator-ROUGE 불가, (b)-ROUGE 비현실적. (a)-ROUGE는 answer_swap의 도메인-포맷 학습에 속아 발산(1B +0.4 / 3B −0.9) → val-loss가 옳다는 방증으로만 보관.
2. **GT = exact 2ᴺ, TMC/LOO 아님.** 소N(≤10)에선 exact가 싸고 sampling noise 0 → 분야 표준(근사 GT)보다 강한 기준. estimator≠oracle(추정은 1 HVP/round 근사, oracle이 정답)를 명확히 분리해 검증.
3. **fp32 + 동일 frozen trajectory.** utility=loss-diff(~1e-3)가 bf16 정밀도(~8e-3)보다 작아 fp32 필수(bf16은 +1.000을 ~0.4로 떨어뜨림). 모든 비교방법을 *같은 궤적* 위에서 돌려 backend·estimator-vs-exact·utility정의 차이만 변수로 남김.

> 요컨대 Flirds는 분야 표준 지표(Spearman/Kendall·AUROC·ROUGE·MMLU·wall-clock)를 채택하되, **(1) 값-수준 Pearson을 추가**하고 **(2) GT를 근사가 아닌 exact 2ᴺ oracle로** 둔 점이 fidelity(1차 질문) 검증을 분야 평균보다 강하게 만든다. 핵심 질문 위계(1차 fidelity → 2차 perf/conv/detection)는 [[flirds]] 참조.
