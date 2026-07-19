---
type: survey
title: "데이터·기여도 평가/귀속 선행연구 분류 서베이 — README + 마스터 표"
created: 2026-06-25
updated: 2026-07-02
sources: [koh-liang-influence-functions, ghorbani-zou-data-shapley, data-banzhaf, in-run-data-shapley, asymmetric-data-shapley, du-shapley, datainf, trak, logix, grosse-llm-influence, less, mates, dsdm, data-value-embedding, lorif, accumulative-sgd-influence, dpo-shapley-lm-arithmetic, do-influence-functions-work-on-llms, influence-functions-fragile, principled-federated-data-valuation, comfedsv, gtg-shapley, game-of-gradients-sfedavg, shapleyfl, space-participant-amalgamation, ripple-shapley, fedif, fedtsv, shapfed, shapley-volatility-fl, mavericks-shapley-fl, feddqc, fedhds, dice, rfedlr, fldetector, fedcorr, fltrust, foolsgold, free-riders-fl-std-dagmm, distributionally-robust-data-valuation, ipfl-model-market, instructions-as-backdoors-xu, how-to-backdoor-fl-bagdasaryan]
tags: [survey, taxonomy, data-valuation, attribution, federated, landscape]
---

# 데이터·기여도 평가/귀속 선행연구 분류 서베이

데이터/기여도 평가·귀속(data valuation & attribution) 연구를 **두 관점**으로 분류한다:

1. **무엇을 타겟하는가** — 분류축(taxonomy): federation setting, valuation 기반, 모델 클래스, 평가 단위, 푸는 문제, 보조 축.
2. **목적을 어떻게 검증했는가** — 검증 실험(E1–E7), CNN 트랙과 LLM 트랙을 분리.

목적은 서로 다른 연구가 *지형의 어느 부분집합을 점유하고*, *어느 칸이 비어 있는지*를 한눈에 보는 것이다. 연합학습(FL)은 모집단의 경계가 아니라 **Federation 축의 한 값**으로 다룬다. 관찰된 핵심 빈칸은 **federated × LLM × client-level *valuation* (fidelity 검증 포함)** — 이 칸을 우리 방법 [[flirds|Flirds]]가 겨냥하며, 지형 안 그 위치를 한 행(=ours)으로 넣어 보인다.

## 범위 (scope)

- **모집단**: 데이터/기여도 평가·귀속 방법 ~40편 (centralized·federated·decentralized 모두 1급 멤버) + FL 강건성/검출 baseline 5종(E3 검증축 연결) + 데이터 마켓 2편.
- **In**: semivalue(Shapley/Banzhaf/LOO/Beta/class-specific), influence/IF·TracIn, in-run/single-run, sub-model 재구성, datamodels, KD/low-rank/DPO-algebra, dedup/quality-probe.
- **참조만(행 아님)**: 순수 공격 논문 2편([[sources/instructions-as-backdoors-xu]], [[sources/how-to-backdoor-fl-bagdasaryan]])은 위협 정의(E3)용 참조.
- **근거**: 1차 자료는 이 wiki의 source 노트 ~45편(작성 시점 기준). 최신성(2024–2026)·빈칸만 표적 웹으로 보강. 핵심 수치는 원문 재확인 권장(아래 근거 태그 참조).

## 읽는 법 (how to read)

- **이 파일(README)** — 범례 + **마스터 한눈에 표**(연구당 1행) + 핵심 관찰 + 전체 인용(arXiv).
- **[[taxonomy|taxonomy.md]]** — 6개 축별 부분집합 분류표 + **2D 교차표**(Federation×모델, Federation×valuation-기반) + 지형 서술. straddle(경계)을 정직하게 표기.
- **[[validation-experiments|validation-experiments.md]]** — **CNN 트랙 표 / LLM 트랙 표**를 분리해 E1–E7 검증 실험을 데이터셋·지표·규모와 함께 정리.
- **[[metrics-and-benchmarks|metrics-and-benchmarks.md]]** — E#별 **metric 카탈로그** + E1의 **ground-truth(정답 SV) 출처** + **benchmark/dataset 카탈로그**(CNN/LLM) + **Flirds가 채택한 metric·benchmark 매핑**(§4).

> **자매 서베이**(taxonomy 밖, 이 지형을 우리 실험에 적용): [[flirds-experiment-results-overview]](실험 결과 전체) · [[baseline-selection-audit-2026-07-02]](**baseline 선정 정당성 감사** — 이 taxonomy를 근거로 실험별 baseline을 A/B/C 판정; aggregation-first 분류가 C-제외 논거) · [[baseline-original-paper-verification]](baseline 수치 ↔ 원 논문 대조).

## 범례 (legend)

**Federation**: `C` centralized · `F` federated(star, server) · `D` decentralized(peer-to-peer graph).

**Basis (valuation 기반)**:
- `retrain` — 재학습 counterfactual (S로 처음부터/coalition 재학습)
- `in-run` — 단일 궤적(재학습 X; per-step/per-round Taylor·trajectory)
- `IF` — gradient·influence (IF / TracIn / iHVP / 투영)
- `recon` — sub-model 재구성 (저장 gradient로 coalition 모델 재합, 재학습 아님 — FL-Shapley 계열)
- `other` — 기타 (datamodels-regression, 1-step probe, DPO-algebra, KD-1round, dedup, quality-probe, utility-design, market, detector)

**Model**: `CNN/MLP`(소 이미지 포함) · `LLM`(LoRA/대형) · `kernel/reg`(RKHS·회귀) · `tabular`.

**Unit (평가단위)**: `sample` · `client` · `class` · `dataset`(contributor) · `node`(graph).

**Purpose (주목적)**: `eff` 효율 · `fid` 정확도/충실도 · `fair` 공정·보상 · `agg` 집계·모델품질 · `det` 강건성·악성검출 · `sel` 데이터 선별 · `mkt` 데이터 마켓 · `attr` 귀속·해석. (`quality` = 품질평가 — value/contribution과 구분)

**E/A**: Shapley/value 계산이 `exact`(2ᴺ 열거) / `approx`(MC·Taylor·근사) / `both`(소N exact + 대N 근사) / `n/a`(semivalue 아님).

**검증실험(E#)**: `E1` exact/GT-SV 대비 충실도(Spearman·Kendall·Pearson·LDS·MAE) · `E2` 선별→다운스트림 · `E3` 악성/이상 검출(noise·free-rider·poison·backdoor; AUROC·검출율·ASR) · `E4` 공정성·보상 · `E5` stochastic robustness·replication·ordering · `E6` 비용·확장성(wall-clock·#eval·통신) · `E7` 집계 품질(Shapley-weighted vs FedAvg). 자세히 → [[validation-experiments]].

**근거 태그**: 별도 표기 없으면 **내부 노트 기반**(작성 시점). `✓web` = 최신/인용을 웹으로 확인. `⚠crit` = 비판/음성결과(critique) 논문. `(detector)` = valuation 아닌 검출 baseline. `(ours)` = Flirds.

---

## 마스터 한눈에 표

> 연구당 1행. **Federation → valuation-기반** 순으로 정렬해 부분집합이 인접하도록 했다. 큰 두 블록(centralized 18편 / federated 16편)과, federated 블록 안에서 **LLM 칸이 valuation에서 비어 있는 것**(아래 ▣ 표시)이 한눈에 보인다. 전체 인용(저자·venue·arXiv)은 맨 아래 [References](#references).

### Centralized — valuation / attribution

| Method | Fed | Basis | Model | Unit | Purpose | E/A | 핵심 E# | 출처 |
|---|---|---|---|---|---|---|---|---|
| Data Shapley | C | retrain | CNN/MLP+의료img | sample | fid·sel·det·mkt | both | E2·E3 | [[sources/ghorbani-zou-data-shapley]] (ICML'19) |
| Data Banzhaf | C | retrain | CNN/MLP(분류기) | sample | det·fid·sel | both | E2·E3·E5 | [[sources/data-banzhaf]] ('22) |
| DU-Shapley | C | other(scalar-util) | kernel/reg | dataset | eff·fid·mkt | approx | E1·E6 | [[sources/du-shapley]] ('23) |
| ADS (Asymmetric) | C | retrain(state-cond) | kernel(KNN)+synth | sample/group | fair·mkt·attr | both | E3·E4·E5 | [[sources/asymmetric-data-shapley]] ('25) |
| DRDV / DRGE | C | other(utility-design) | kernel/reg(NTK) | sample | mkt | n/a | E1·E2·E6 | [[sources/distributionally-robust-data-valuation]] (ICML'24) |
| IRDS (In-Run) | C | in-run | LLM(GPT-2,Pythia) | sample | eff·attr·sel | approx | E2·E6 | [[sources/in-run-data-shapley]] (ICML'24) |
| DVEmb | C | in-run | MLP+LLM(Pythia,GPT-2) | sample | attr·sel·eff | approx | E1·E2·E6 | [[sources/data-value-embedding]] ('24) |
| ACC-SGD-IE | C | in-run | tabular·MNIST·CIFAR | sample | fid·sel·attr | approx | E1·E3·E5 | [[sources/accumulative-sgd-influence]] ('25) |
| IF (Koh-Liang) | C | IF | CNN/MLP(logistic) | sample | attr·det | approx | E3·E5 | [[sources/koh-liang-influence-functions]] (ICML'17) |
| DataInf | C | IF | LLM-LoRA(RoBERTa,Llama2-13B)+diffusion | sample | eff·attr | approx | E1·E3·E6 | [[sources/datainf]] (ICLR'24) |
| TRAK | C | IF | CNN/CLIP(ImageNet)·BERT·mT5 | sample | eff·fid·attr | approx | E1·E5·E6 | [[sources/trak]] (ICML'23) |
| LoGra / Logix | C | IF | LLM(Llama3-8B,GPT-2)+ResNet-9 | sample | eff·fid·mkt | approx | E1·E6 | [[sources/logix]] ('24) |
| EK-FAC (Grosse) | C | IF | LLM ≤52B | sample | attr·fid | approx | E1·E5·E6 | [[sources/grosse-llm-influence]] ('23) |
| LESS | C | IF / in-run | LLM-LoRA(Llama2-7B/13B,Mistral) | sample | sel·eff | approx | E2·E5·E6 | [[sources/less]] (ICML'24) |
| LoRIF | C | IF | LLM(GPT-2,OLMo-7B,Apertus-70B) | sample | eff·fid·attr | approx | E1·E6 | [[sources/lorif]] ('26) |
| DsDm | C | other(datamodels) | LLM(125M–1.3B) | sample | sel·eff | approx | E2·E5·E6 | [[sources/dsdm]] (ICML'24) |
| MATES | C | other(1-step probe) | LLM(Pythia 410M,1B) | sample | sel·eff | approx | E1·E2·E5·E6 | [[sources/mates]] (NeurIPS'24) |
| DPO-Shapley | C | other(DPO-algebra)+retrain | LLM-LoRA(SmolLM-135M) | dataset | sel·attr·mkt | both | E4·E6 | [[sources/dpo-shapley-lm-arithmetic]] ('26) |

### Centralized — critiques (음성결과·caveat 앵커)

| Method | Fed | Basis | Model | Unit | Purpose | E/A | 핵심 E# | 출처 |
|---|---|---|---|---|---|---|---|---|
| ⚠crit IF-Fragile | C | IF(eval) | CNN/MLP·ResNet·ImageNet | sample | fid(critique) | both | E1·E5 | [[sources/influence-functions-fragile]] (ICLR'21) |
| ⚠crit Do-IF-work-on-LLMs | C | IF(eval) | LLM-LoRA(Llama2-7B,Mistral) | sample | fid·det(critique) | approx | E1·E3·E5 | [[sources/do-influence-functions-work-on-llms]] (EMNLP'25) |

### Federated — valuation / attribution

| Method | Fed | Basis | Model | Unit | Purpose | E/A | 핵심 E# | 출처 |
|---|---|---|---|---|---|---|---|---|
| **Flirds (ours)** | **F** | **in-run(1st+2nd, Hessian)** | **CNN + LLM-LoRA(1B/3B/7B)** | **client** | **fid·sel·det** | **approx (vs exact ⓐ/ⓑ)** | **E1·E2·E3·E6** | [[flirds]] (ours '26) |
| Ripple Shapley | F | in-run | CNN(MNIST/CIFAR)▣ | **sample** | attr·eff·mkt | approx | E6 | [[sources/ripple-shapley]] (AAAI'26) |
| FedTSV | F | in-run | MLP·CNN(ResNet-20) | client | fair·agg·det | approx | E3·E4·E5·E7 | [[sources/fedtsv]] (ECC'26) |
| FedIF | F | IF/in-run(1차) | CNN(CIFAR,FMNIST) | client | agg·det | n/a | E3·E5·E6·E7 | [[sources/fedif]] ('25) |
| FedSV | F | recon | CNN(MNIST,CIFAR) | client | fid·fair·det | both | E3·E4·E6 | [[sources/principled-federated-data-valuation]] ('20) |
| GTG-Shapley | F | recon | CNN(image cls) | client | eff·fid | approx | E1·E6 | [[sources/gtg-shapley]] (TIST'22) |
| ComFedSV | F | recon+completion | CNN(MNIST,FMNIST,CIFAR) | client | fair·fid·det | approx | E1·E3·E4·E6 | [[sources/comfedsv]] (ICDE'22) |
| S-FedAvg (GoG) | F | recon | CNN(FL bench) | client | det·agg·sel | ? | E2·E3·E7 | [[sources/game-of-gradients-sfedavg]] (AAAI'21) |
| ShapleyFL (AFedSV) | F | recon | CNN(MNIST,CIFAR)+Fed-ISIC | client | eff·det·sel | approx | E3·E6 | [[sources/shapleyfl]] (KDD'23) |
| SPACE | F | other(KD-1round) | CNN(분류·prototype) | client | eff·sel·fair | ? | E5·E6 | [[sources/space-participant-amalgamation]] (NeurIPS'23) |
| ShapFed | F | other(class-cosine) | CNN(CIFAR,X-ray,ISIC) | **class** | agg·fair·attr | n/a | E1·E3·E7 | [[sources/shapfed]] (IJCAI'24) |
| Mavericks / FedEMD | F | other(LOO/Inf-index) | CNN(MNIST,CIFAR,STL) | client | fair·sel·attr | both | E2·E4·E5·E6 | [[sources/mavericks-shapley-fl]] ('21) |
| FedDQC ▣LLM | F | other(IRA quality) | **LLM-LoRA(Llama2-7B)** | sample | sel·`quality` | n/a | E2·E3·E5·E6 | [[sources/feddqc]] (ACL'25) |
| FedHDS ▣LLM | F | other(dedup) | **LLM(1.3B,3B)** | sample | sel·eff | n/a | E2·E5·E6 | [[sources/fedhds]] (ACL'25) |
| iPFL ▣LLM | F | other(market) | **LLM-LoRA(Mistral,Llama2)** | dataset | mkt·fair | n/a | E4·E6 | [[sources/ipfl-model-market]] (Nat.Commun.'25) |
| RFedLR | F | other(robust-PEFT) | ?(FL bench, LoRA) | client | det·agg | n/a | E3·E7 | [[sources/rfedlr]] (NeurIPS'25) |

### Federated — critique

| Method | Fed | Basis | Model | Unit | Purpose | E/A | 핵심 E# | 출처 |
|---|---|---|---|---|---|---|---|---|
| ⚠crit Shapley-Volatility-FL | F | in-run(eval) | CNN(CIFAR/MNIST) | client | det·fair(critique) | approx | E4·E5·E6 | [[sources/shapley-volatility-fl]] ('25) |

### Federated — detection baselines (valuation 아님; E3 참조축)

| Method | Fed | Basis | Model | Unit | Purpose | E/A | 핵심 E# | 출처 |
|---|---|---|---|---|---|---|---|---|
| FLDetector (detector) | F | other(update-consistency) | CNN(MNIST,CIFAR,FEMNIST)▣ | client | det | n/a | E3 | [[sources/fldetector]] (KDD'22) |
| FLTrust (detector) | F | other(cosine-to-root) | CNN/shallow▣ | client | det·agg | n/a | E3·E7 | [[sources/fltrust]] (NDSS'21) |
| FoolsGold (detector) | F | other(cosine-sim) | CNN/shallow | client | det·agg | n/a | E3·E7 | [[sources/foolsgold]] (RAID'20) |
| FedCorr (detector) | F | other(LID-GMM)+correct | CNN(CIFAR,Clothing1M) | client/sample | det·agg | n/a | E3·E7 | [[sources/fedcorr]] (CVPR'22) |
| STD-DAGMM (detector) | F | other(DAGMM-AE) | CNN(MNIST,MLP)▣ | client | det | n/a | E3 | [[sources/free-riders-fl-std-dagmm]] ('19) |

### Decentralized

| Method | Fed | Basis | Model | Unit | Purpose | E/A | 핵심 E# | 출처 |
|---|---|---|---|---|---|---|---|---|
| DICE / DICE-E | D | IF(cascade) | CNN(ResNet-18, 16-node) | node/dataset | attr·fair·det | approx | (개념·경량 empirics) | [[sources/dice]] ('25) |

> `▣` = 그 모델 칸을 우리가 LLM으로 포팅했거나(FLDetector·FLTrust·STD-DAGMM = Flirds baseline 포팅), 또는 federated×LLM 셀의 straddle(FedDQC=quality·FedHDS=selection·iPFL=market — valuation 아님)임을 표시. Ripple▣=원논문은 CNN-scale, sample-level.

---

## 핵심 관찰 (landscape, 담백하게)

1. **두 모집단이 거의 분리돼 있다.** FL-Shapley/valuation 계열(FedSV·GTG·ComFedSV·S-FedAvg·ShapleyFL·SPACE·ShapFed·FedTSV·FedIF·Ripple)은 **전부 CNN-분류** 무대다. LLM-scale 귀속/선별 계열(IRDS·DataInf·LoGra·EK-FAC·LESS·MATES·DsDm·LoRIF·DPO-Shapley·DVEmb)은 **전부 centralized**다. ([[validation-experiments]] CNN/LLM 표가 이 분리를 그대로 보여준다.)

2. **federated × LLM × client-level *valuation* 칸은 관찰상 비어 있다.** 그 칸의 현재 점유자는 *valuation이 아닌* 인접 문제뿐 — FedDQC(per-sample **quality**), FedHDS(**selection**), iPFL(model-**market**). client별 기여도 값을 **exact-SV oracle 대비 충실도(E1)** 로 검증한 federated-LLM 연구는 내부 노트·표적 웹(2024–2026) 범위에서 확인되지 않았다(`✓web`: "FL + LoRA + Shapley valuation" 조합은 아직 문헌이 두텁지 않음). 이 칸을 [[flirds|Flirds]]가 겨냥한다 — fidelity(1차)부터 검증. *단정적 노벨티가 아니라 관찰된 빈칸으로 읽을 것.*

3. **valuation은 점점 aggregation으로 흡수된다.** 최근 federated 계열 다수(FedIF·FedTSV·ShapFed·S-FedAvg)는 기여도를 *집계 가중치*로 쓰는 robust-aggregation 방법이지, 사후 valuation accountant가 아니다 — fidelity(E1) 검증을 생략하고 E7(집계 품질)로 평가한다. 순수 valuation-fidelity를 1차로 두는 연구는 소수(FedSV·GTG·ComFedSV·Flirds).

4. **충실도 자체가 흔들린다는 경고가 양쪽에 있다.** centralized: IF-Fragile·Do-IF-work-on-LLMs(IF가 deep/LLM서 GT 대비 무너짐). federated: Shapley-Volatility-FL(근사 FL-Shapley는 집계전략만 바꿔도 보상 30–50% 출렁). → exact oracle 대비 검증(Flirds의 (a)/(b) 듀얼 oracle)의 동기.

5. **straddle은 정직하게.** LESS=LLM·LoRA·val-gradient지만 centralized·per-example(=Flirds의 가장 가까운 centralized 사촌). Ripple=federated·in-run이지만 CNN-scale·sample-level·2차항 없음. DICE=유일한 decentralized지만 node-level·CNN·경량 empirics. 자세한 경계 → [[taxonomy]].

---

## References

> 저자·venue·arXiv. **2026년 arXiv id 및 "venue 미상" 항목은 내부 노트 기준(원문 재확인 권장)**. `✓web` = 이번 작성 중 웹으로 인용 확인.

**Centralized — valuation/attribution**
- [[sources/ghorbani-zou-data-shapley]] — Ghorbani & Zou 2019, *Data Shapley: Equitable Valuation of Data for ML*, ICML 2019, arXiv:1904.02868.
- [[sources/data-banzhaf]] — Wang & Jia 2022, *Data Banzhaf: A Robust Data Valuation Framework*, arXiv:2205.15466.
- [[sources/du-shapley]] — Garrido-Lucero et al. 2023, *DU-Shapley: A Shapley Value Proxy for Efficient Dataset Valuation*, arXiv:2306.02071.
- [[sources/asymmetric-data-shapley]] — Zheng et al. 2025, *Rethinking Data Value: Asymmetric Data Shapley*, arXiv:2511.12863 `✓web`.
- [[sources/distributionally-robust-data-valuation]] — Lin et al. 2024, *Distributionally Robust Data Valuation*, ICML 2024.
- [[sources/in-run-data-shapley]] — Wang et al. 2024, *Data Shapley in One Training Run* (IRDS), ICML 2024 / ICLR 2025, arXiv:2406.11011.
- [[sources/data-value-embedding]] — Wang et al. 2024, *Capturing the Temporal Dependence of Training Data Influence* (DVEmb), arXiv:2412.09538 `✓web`.
- [[sources/accumulative-sgd-influence]] — Shi et al. 2025, *Accumulative SGD Influence Estimation* (ACC-SGD-IE), arXiv:2510.26185.
- [[sources/koh-liang-influence-functions]] — Koh & Liang 2017, *Understanding Black-box Predictions via Influence Functions*, ICML 2017, arXiv:1703.04730.
- [[sources/datainf]] — Kwon et al. 2024, *DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models*, ICLR 2024, arXiv:2310.00902.
- [[sources/trak]] — Park et al. 2023, *TRAK: Attributing Model Behavior at Scale*, ICML 2023, arXiv:2303.14186.
- [[sources/logix]] — Choe et al. 2024, *What is Your Data Worth to GPT?* (LoGra/Logix), arXiv:2405.13954.
- [[sources/grosse-llm-influence]] — Grosse et al. 2023, *Studying LLM Generalization with Influence Functions* (EK-FAC), arXiv:2308.03296.
- [[sources/less]] — Xia et al. 2024, *LESS: Selecting Influential Data for Targeted Instruction Tuning*, ICML 2024, arXiv:2402.04333.
- [[sources/lorif]] — Li et al. 2026, *LoRIF: Low-Rank Influence Functions for Scalable TDA*, arXiv:2601.21929 (note 기준).
- [[sources/dsdm]] — Engstrom et al. 2024, *DsDm: Model-Aware Dataset Selection with Datamodels*, ICML 2024, arXiv:2401.12926.
- [[sources/mates]] — Yu et al. 2024, *MATES: Model-Aware Data Selection ... with Data Influence Models*, NeurIPS 2024, arXiv:2406.06046.
- [[sources/dpo-shapley-lm-arithmetic]] — Tamine et al. 2026, *Data Valuation for LLM Fine-Tuning via Language Model Arithmetic*, arXiv:2512.15765 `✓web`.

**Centralized — critiques**
- [[sources/influence-functions-fragile]] — Basu et al. 2021, *Influence Functions in Deep Learning Are Fragile*, ICLR 2021, arXiv:2006.14651.
- [[sources/do-influence-functions-work-on-llms]] — Li et al. 2024, *Do Influence Functions Work on Large Language Models?*, EMNLP 2025 Findings, arXiv:2409.19998.

**Federated — valuation/attribution**
- [[flirds|Flirds]] — (ours) 2026, *Federated Learning + In-Run Data Shapley*, in progress.
- [[sources/ripple-shapley]] — Zeng et al. 2026, *Ripple Shapley: Data Influence Attribution in One Federated Training Run*, AAAI 2026 `✓web`.
- [[sources/fedtsv]] — Kuznetsov & Wang 2026, *Fairness-Aware FL with Trajectory Shapley Value* (FedTSV), ECC 2026, arXiv:2605.30336 (note 기준).
- [[sources/fedif]] — Tang et al. 2025, *Lightweight and Robust Federated Data Valuation* (FedIF), arXiv:2509.25560.
- [[sources/principled-federated-data-valuation]] — Wang et al. 2020, *A Principled Approach to Data Valuation for FL* (FedSV), Springer LNCS 12500, arXiv:2009.06192.
- [[sources/gtg-shapley]] — Liu et al. 2022, *GTG-Shapley: Efficient and Accurate Participant Contribution Evaluation in FL*, ACM TIST 2022, arXiv:2109.02053.
- [[sources/comfedsv]] — Fan et al. 2022, *Improving Fairness for Data Valuation in Horizontal FL* (ComFedSV), ICDE 2022, arXiv:2109.09046.
- [[sources/game-of-gradients-sfedavg]] — Nagalapatti & Narayanam 2021, *Game of Gradients: Mitigating Irrelevant Clients in FL* (S-FedAvg), AAAI 2021.
- [[sources/shapleyfl]] — Sun et al. 2023, *ShapleyFL: Robust FL Based on Shapley Value*, KDD 2023.
- [[sources/space-participant-amalgamation]] — Chen et al. 2023, *SPACE: Single-round Participant Amalgamation for Contribution Evaluation in FL*, NeurIPS 2023.
- [[sources/shapfed]] — Tastan et al. 2024, *Redefining Contributions: Shapley-Driven FL* (ShapFed), IJCAI 2024, arXiv:2406.00569.
- [[sources/mavericks-shapley-fl]] — Huang et al. 2021, *Is Shapley Value Fair? Improving Client Selection for Mavericks in FL* (FedEMD), arXiv:2106.10734.
- [[sources/feddqc]] — Du et al. 2025, *FedDQC: Data Quality Control in Federated Instruction-tuning of LLMs*, ACL 2025 Findings, arXiv:2410.11540.
- [[sources/fedhds]] — Qin et al. 2025, *Federated Data-Efficient Instruction Tuning for LLMs* (FedHDS), ACL 2025 Findings, arXiv:2410.10926.
- [[sources/ipfl-model-market]] — Zhang et al. 2025, *Incentivizing Inclusive Contributions in Model Sharing Markets* (iPFL), Nature Communications 16:7923.
- [[sources/rfedlr]] — Fang & Ye 2025, *Towards Robust PEFT for FL* (RFedLR), NeurIPS 2025.

**Federated — critique**
- [[sources/shapley-volatility-fl]] — Geimer et al. 2025, *On the Volatility of Shapley-Based Contribution Metrics in FL*, arXiv:2405.08044.

**Federated — detection baselines (E3 참조)**
- [[sources/fldetector]] — Zhang et al. 2022, *FLDetector*, KDD 2022, arXiv:2207.09209.
- [[sources/fltrust]] — Cao et al. 2021, *FLTrust*, NDSS 2021, arXiv:2012.13995.
- [[sources/foolsgold]] — Fung et al. 2020, *Mitigating Sybils in FL* (FoolsGold), RAID 2020, arXiv:1808.04866.
- [[sources/fedcorr]] — Xu et al. 2022, *FedCorr: Multi-Stage FL for Label Noise Correction*, CVPR 2022, arXiv:2204.04677.
- [[sources/free-riders-fl-std-dagmm]] — Lin et al. 2019, *Free-riders in FL: Attacks and Defenses* (STD-DAGMM), arXiv:1911.12560.

**Decentralized**
- [[sources/dice]] — Zhu et al. 2025, *DICE: Data Influence Cascade in Decentralized Learning*, arXiv:2507.06931.

**위협 정의 참조 (행 아님)**
- [[sources/instructions-as-backdoors-xu]] — Xu et al. 2024, *Instructions as Backdoors*, NAACL 2024, arXiv:2305.14710.
- [[sources/how-to-backdoor-fl-bagdasaryan]] — Bagdasaryan et al. 2020, *How To Backdoor Federated Learning*, AISTATS 2020, arXiv:1807.00459.

---

## 커버 못한 것 / 불확실 / 추가로 찾을 논문

- **웹 표적검색서 떠올랐으나 내부 노트 없음(미수집)** — 모두 *추가 확인 필요*, 본 표에는 미반영:
  - Owen Sampling for FL Contribution Estimation (arXiv:2508.21261, 2025) — FL 기여도 추정 가속(추정 CNN-scale).
  - Maverick-Aware Shapley Valuation / "Rewarding the Rare" (arXiv:2405.12590, 2024) — [[sources/mavericks-shapley-fl]]와 별개 논문(class-wise SV).
  - Secure Shapley Value for Cross-Silo FL (VLDB) — secure-aggregation 호환 SV.
  - Towards Fair, Robust and Efficient Client Contribution Evaluation in FL (arXiv:2402.04409, 2024).
  - FedKDShap (WWW 2025) — Shapley + KD, non-IID(=SPACE 계열).
  - In-Run Data Shapley for Adam Optimizer (arXiv:2602.00329, 2026) — IRDS의 Adam 확장(centralized; DVEmb의 "Adam이 Eq.2를 깬다" 한계와 직결).
  - 서베이 2편: *Shapley-value-based Contribution Evaluation in FL: A Survey* (IEEE); *A Comprehensive Study of Shapley Value in Data Analytics* (arXiv:2412.01460) — 본 표의 비교 대조용.
  - → 공통 패턴: 모두 **CNN-scale FL-Shapley(효율/공정/선별/secure)** 또는 **centralized LLM-Shapley**. federated×LLM×valuation-fidelity 빈칸을 메우는 것은 미발견 → 관찰 결론 유지.
- **불확실(원문 재확인 권장)**: 2026 arXiv id(LoRIF 2601.21929 / FedTSV 2605.30336 / DPO-Shapley 2512.15765[✓web] / ACC-SGD-IE 2510.26185); "venue 미상" 다수; S-FedAvg·SPACE의 exact/approx 구분; RFedLR 모델클래스(노트 stub).
- **스코프 밖(의도)**: Beta-Shapley·CS-Shapley 전용 노트 없음(개념 페이지만; [[concepts/semivalue]] 참조). 순수 공격(Xu/Bagdasaryan)은 위협정의만. interpretability/alignment/RL은 별도 wiki.
- **검증 권장**: 내부 노트는 작성 시점 기준 — E1–E7 셀의 정량치(특히 fidelity Spearman, AUROC, wall-clock)는 발표/논문 인용 전 원문 1회 재확인.
