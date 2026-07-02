# Flirds 논문화 대비 리뷰: 강점, 약점, novelty, 논문 구조

작성 원칙: 이 문서의 Pass 1은 Appendix A(`flirds-signal-size-diagnosis.md`)를 읽기 전에, 사용자 dossier §2-§4만으로 고정한 독립 비평이다. Pass 2와 레포 검증은 그 이후에 추가한다.

## Pass 1: Appendix A 이전 독립 비평

### A. 한 문단 요약

Flirds가 현재 가장 강하게 증명한 것은 "FL 학습 로그와 validation loss game이 주어졌을 때, LLM/LoRA FedAvg 세팅에서 exact in-run Shapley oracle (b)에 매우 낮은 추가 비용으로 거의 완전한 fidelity를 보인다"는 주장이다: anchor N=5에서는 전 방법이 Spearman +1.000이지만 Flirds는 Pearson 0.99999+까지 맞추고, std20 부분 참여에서는 Flirds +1.000, Flirds-1st +0.999, GTG +0.975±0.02, FedSV +0.91±0.09, ShapleyFL +0.19, ComFedSV +0.09로 갈린다(§3.5). 또한 retrain-based oracle (a)와 in-run oracle (b)가 N=5 1B fp32에서 +1.000으로 맞는 점은 valuation paper에서 드문 강한 검증 설계다(§3.1, §3.5). 그러나 최상위 메인트랙 기준으로 아직 못 증명한 것은 더 중요하다: 이 완벽한 fidelity가 near-additive/low-resolution game의 산물인지, retrain oracle에서도 넓은 N/scale/regime에 유지되는지, 그리고 정확한 valuation이 실제 FL 성능/수렴/운영 의사결정 이득으로 이어지는지다. 현 상태는 "강한 실험 설계와 유망한 LLM-scale valuation estimator"이지만, 현재 수치만으로는 NeurIPS/ICML 메인트랙 accept보다 reject 또는 major-revision 성향이 더 자연스럽다. 특히 loss-heuristic이 std20에서 거의 +1.000인 점과 anchor N=5에서 모든 방법이 동률인 점은 Reviewer 2가 "problem instance가 너무 쉽고 Flirds의 기술적 필요성이 과장됐다"고 공격할 수 있는 핵심 근거다(§3.5).

### B. 강점

1. **진짜 기여: fidelity를 1차 headline으로 둔 설계가 맞다.** 근거: dossier가 핵심 지표를 estimator vs oracle Spearman/Kendall, Pearson, 거리 metric으로 두고, downstream 성능/수렴/탐지는 2차로 둔다(§3.4). data valuation에서는 detector AUROC나 downstream gain만으로는 "기여도를 정확히 측정했는가"를 증명할 수 없다. Flirds의 핵심 claim은 Shapley-like contribution estimator이므로 oracle 대비 순위와 값 수준 fidelity가 본체다.

2. **진짜 기여: 이중 oracle 설계는 분야 난제를 정면으로 다룬다.** 근거: (b) 동일 학습 로그 utility에서 exact 2^N coalition Shapley를 계산하고, (a) 각 subset을 FedAvg 재학습해 배포 모델을 채점한다(§3.1). N=5 1B fp32에서 (a)=(b)=estimator가 Spearman +1.000인 결과는 "estimator가 self-referential in-run oracle만 맞춘다"는 공격을 일부 차단한다(§3.5). Data valuation 논문에서 가장 취약한 지점은 참값 부재인데, 이 설계는 최소한 작은 N에서는 참 retraining game과 in-run reconstruction game의 일치를 직접 점검한다.

3. **진짜 기여: LLM-scale에서 비용 우위가 명확하다.** 근거: N=5 1B 1 GPU에서 Flirds-1st ~35s, Flirds ~107s이고, GTG/FedSV/Banzhaf/ShapleyFL/(b) oracle은 ~530s, Ripple은 ~4515s다(§3.5). 또한 oracle (b)는 2^N·R·val·seq라 N=5에서 N=10으로 32배 증가하지만, estimator는 라운드당 HVP 1회에 가깝다(§3.5). FL valuation에서 "정확하지만 못 돌린다"는 방법은 운영 가치가 낮으므로, 5-15배 wall-clock 절감과 지수적 coalition 평가 회피는 단순 engineering convenience가 아니라 method의 실질적 의미다.

4. **진짜 기여: LLM/PEFT 구현상의 failure mode를 실제로 처리했다는 신호가 있다.** 근거: forward-mode AD HVP와 eager attention, fp32 master weight, LoRA parameter keying, functorch hook hygiene, plain SGD momentum=0 조건이 명시돼 있다(§2.2). 이는 "작은 CNN toy에서만 되는 Shapley 근사"와 구별되는 실험 엄밀성 신호다. 특히 utility가 loss 차이 ~1e-3이고 bf16 정밀도 ~8e-3보다 작다는 지적은 valuation numerical precision의 실질 문제를 잡고 있다(§2.2).

5. **진짜 기여: anchor, std20, cross-device로 쉬운 동률 레짐과 갈리는 레짐을 분리하려는 설계가 있다.** 근거: anchor N=5는 5 domain full participation exact 2^N, std20은 N=20 부분 참여 2/20 R=200, cross-device N=100은 Dirichlet α-sweep 및 per-round exact decomposition이다(§3.2). anchor에서 전 방법 Spearman +1.000이라는 결과를 그대로 headline으로 삼으면 약하지만, std20에서 방법 간 차이가 크게 벌어지는 구조는 "어디서 estimator가 필요한가"를 분석할 여지를 만든다(§3.5).

6. **진짜 기여: 값 수준 Pearson을 보고한다.** 근거: anchor N=5에서 전 방법이 (b) 대비 Spearman +1.000이지만 Flirds는 Pearson 0.99999+라고 보고한다(§3.5). N=5 순위 상관은 coarse해서 쉽게 포화되므로 값 수준 fidelity가 있어야 "동률 순위 맞추기" 이상의 주장이 가능하다. 다만 std20/cross-device에서도 Pearson과 거리 metric을 같은 비중으로 공개해야 이 강점이 완성된다.

7. **진짜 기여: valuation과 detection을 분리하는 정직성이 있다.** 근거: clean-preserving backdoor ASR 1.0에서 공격자의 업데이트가 clean val-loss를 낮추면 Flirds가 그 client를 "기여 높음"으로 rank한다고 보고한다(§3.5). 이것은 Flirds의 실패라기보다 clean validation-loss game에 대한 정직한 valuation 결과다. data attribution과 threat detection을 혼동하지 않는 서술은 construct validity를 높인다.

8. **진짜 기여에 가까운 설계: client 간 game fairness를 위해 format과 size confound를 통제했다.** 근거: 5개 도메인을 모두 free-form instruction-response로 통일하고, 도메인당 train 12k/val 200/test 2k, train size를 균등화했다(§2.3). Shared validation loss 기반 Shapley는 출력 포맷이 다르면 client value가 data quality가 아니라 formatting advantage를 반영할 수 있다. 이 통제는 novelty hook 자체라기보다는 oracle game을 방어하기 위한 필수 조건이다.

9. **있으면 좋은 것: detector baseline 폭은 넓지만 주 기여는 아니다.** 근거: FedDQC, STD-DAGMM, FLTrust, FLDetector 등 threat-matched detector를 포함한다(§3.3). 이는 2차-③ 탐지 질문에는 유용하지만, valuation fidelity 논문에서 중심 근거가 되면 오히려 backdoor 회피 결과 때문에 논지가 흐려진다(§3.5).

10. **있으면 좋은 것: CNN Track C는 보조 sanity check로 가치가 있다.** 근거: CIFAR/MNIST FedSVCNN N=10에서 ComFedSV Spearman {1.0, 0.96, 0.85, 0.84}, oracle rank stability가 비IID/오염 셀에서 0.51-0.97, IID 셀에서 ~0이라고 보고한다(§3.5). 이는 "신호가 real하게 생기는 조건" 분석에는 좋지만, LLM-scale Flirds contribution의 본체는 아니다.

### C. 약점

1. **[수정가능, reject 가능] 완벽한 fidelity가 near-additive/easy game의 산물일 수 있다.** 근거: anchor N=5에서 valuation 전 방법이 (b) 대비 Spearman +1.000이고, std20에서도 loss-heuristic이 거의 +1.000이며, cross-device N=100은 per-round exact decomposition이라고 설명된다(§3.2, §3.5). 파급: Reviewer 2는 "Flirds가 Shapley를 잘 근사한 것이 아니라, 이 실험의 client utilities가 거의 additive라서 어떤 reasonable score도 같은 순위를 낸다"고 주장할 수 있다. 이 공격은 현재 상태에서 치명적이다. 특히 N=5 Spearman은 가능한 순위 공간이 작고 포화되기 쉬우므로, "모든 방법 +1.000"은 강점이 아니라 hardness 부족의 증거로 읽힐 수 있다.

2. **[수정가능, reject 가능] retrain oracle 검증이 너무 좁고 3B에서 이미 균열이 보인다.** 근거: N=5 1B fp32에서는 (a)-valloss, (b), estimator가 +1.000이지만, 3B에서 (a)-valloss vs (b)는 +0.900이고 estimator vs (b)는 +1.000이다(§3.5). 파급: estimator가 in-run oracle에는 완벽하지만 true retrain game과의 fidelity는 scale이 커질수록 깨질 수 있다는 공격이 가능하다. 최상위 리뷰어는 "Flirds는 exact Shapley of the log-reconstructed game이지, FL training contribution의 Shapley가 아니다"라고 쓸 수 있다.

3. **[수정가능] downstream actionability가 아직 핵심 claim을 지탱하지 못한다.** 근거: 성능 향상, 수렴 속도, 탐지의 2차 지표 중 intervention arm 6종, MMLU/ROUGE, rounds-to-target는 §4의 Track D 본런에서 계획됐으나 아직 미완이다. 현재 확정된 2차 결과는 noisy AUROC 0.75, free-rider 1.0, clean-preserving backdoor 회피 정도다(§3.5). 파급: "정확한 기여도 측정이 그래서 FL training/incentive/client selection에 무엇을 개선하는가"라는 significance 질문에 아직 답이 약하다.

4. **[완화가능] 방법론 novelty가 IRDS의 비교적 직접적인 FL 확장으로 보일 위험이 있다.** 근거: Flirds의 뿌리는 중앙집중 per-step sample-level IRDS이고, 핵심 계산도 validation loss의 1차+2차 Taylor 전개 및 HVP다(§2.2). 파급: 논문이 derivation만 제시하면 "IRDS를 client delta에 적용한 것"으로 읽힐 수 있다. 이를 substantial contribution으로 만들려면 FL multi-step local update, client aggregation, partial participation, LLM LoRA parameterization, exact oracle fidelity가 결합될 때 기존 IRDS가 직접 처리하지 못한 문제가 무엇인지 수식과 실험으로 분명히 해야 한다.

5. **[완화가능] Taylor approximation의 이론적 오차와 assumptions가 아직 약하다.** 근거: Flirds는 라운드 모델 주변 1차+2차 Taylor 근사이며, 모든 valuation run은 plain SGD momentum=0으로 제한된다(§2.2). 파급: 실제 FL은 local multi-step, optimizer state, adaptive optimizer, clipping/compression/DP, heterogeneous client drift를 포함한다. 이론이 없으면 "works only in a carefully sanitized training loop"라는 비판이 가능하다.

6. **[인정필수] Shapley game의 utility가 clean validation loss라는 construct choice에 강하게 묶인다.** 근거: val-loss oracle이 "같은 game"이라 ROUGE oracle보다 corruption에 속지 않는다고 설명하지만, clean-preserving backdoor는 clean val-loss를 낮추기 때문에 Flirds가 공격자를 high contribution으로 평가한다(§3.1, §3.5). 파급: 이 논문은 "client의 전반적 사회적 가치"를 측정한다고 쓰면 틀린다. 정확한 claim은 "선택한 validation-loss utility game에 대한 contribution"이어야 한다.

7. **[완화가능] loss-heuristic이 강한 baseline으로 등장해 method의 필요성을 압박한다.** 근거: std20에서 loss-heuristic이 +1.000 수준이고, anchor에서는 전 방법이 +1.000이다(§3.5). 파급: Flirds가 HVP까지 쓰는 비용을 정당화하려면 loss-heuristic이 실패하는 조건, 값 수준 calibration, interaction-sensitive cases, intervention gain을 보여야 한다. 그렇지 않으면 "cheap heuristic is enough"가 강한 reject 논거가 된다.

8. **[수정가능] N=5 anchor는 너무 coarse하고, N=20도 per-round 참여 2/20이면 interaction을 충분히 자극하지 못할 수 있다.** 근거: anchor는 5 client=5 domains full participation, std20은 round당 2 client 참여다(§3.2). 파급: Shapley의 핵심은 coalition interaction인데, per-round 2-client subgame은 고차 상호작용을 거의 보여주지 않는다. exact decomposition이 가능하다는 장점이 있지만, 방법의 차별성을 보여주는 무대로는 약할 수 있다.

9. **[완화가능] 7B와 N=10 oracle가 deferred인 것은 scale claim의 상한을 만든다.** 근거: 7B (a)-oracle과 N=10 oracle은 deferred로 남아 있다(§4). 파급: LLM-scale이라고 주장할 수는 있지만, true retrain oracle fidelity가 7B에서 확인되지 않았고, exact oracle complexity wall을 N=10에서 실증하지 못했다는 공격을 받을 수 있다.

10. **[인정필수] 탐지 claim은 headline이 되면 위험하다.** 근거: Flirds φ detection AUROC는 noisy 0.75, free-rider 1.0(N=5 coarse), clean-preserving backdoor는 회피된다(§3.5). 파급: "valuation detects bad clients"라는 문장은 거짓에 가깝다. free-rider에는 잘 맞지만, clean validation objective에 맞춰 조작된 poison/backdoor는 contribution estimator가 탐지기가 아님을 드러낸다.

11. **[완화가능] compute/reproducibility 서술이 아직 review artifact 수준이어야 한다.** 근거: DGX B200 x4, 1B Track D 40-60 GPU-h, rank/participation probe ~450 GPU-h가 제시된다(§3.5, §4). 파급: 최상위 리뷰어는 wall-clock뿐 아니라 peak memory, validation set size sensitivity, seed variance, implementation overhead, hardware transferability를 요구할 수 있다.

### D. Novelty 판정

**판정:** 방법론만 보면 "IRDS의 FL client-level/per-round 확장"은 incremental로 공격받을 수 있다. 그러나 전체 패키지, 즉 LLM/LoRA FL에서 라운드당 HVP 1회로 in-run Shapley를 추정하고, exact in-run oracle과 retrain-based oracle을 함께 구축해 fidelity를 검증하며, 기존 FL Shapley 계열 대비 wall-clock/scaling을 계량한다는 조합은 substantial로 만들 여지가 있다. 현 상태에서는 "substantial but not yet secured"이다. Top-tier 메인트랙 충분성은 §4 실험이 hard-regime과 actionability gap을 실제로 메우는지에 달려 있다.

선행 대비 좌표는 다음과 같다.

1. **Data Shapley (Ghorbani & Zou) 및 exact/MC Shapley 계열 대비:** Flirds는 Shapley 공리 자체를 새로 제안하지 않는다. 차별점은 FL training log와 validation loss Taylor expansion을 이용해 client-level contribution을 재학습 없이 추정하는 계산 경로다(§2.1, §2.2). 따라서 novelty는 "새 가치 개념"이 아니라 "LLM-scale FL에서 실행 가능한 Shapley estimator와 oracle 검증"에 있다.

2. **KNN-Shapley, Beta-Shapley, Data-Banzhaf 대비:** 이들은 value definition 또는 approximation target을 바꾸거나 특정 model/data setting에서 효율화한다. Flirds의 target은 여전히 Shapley-like FL client contribution이며, Banzhaf도 baseline으로 포함된다(§3.3). Flirds가 더 강하려면 "Shapley target fidelity"와 "cost"를 동시에 보여야 한다. 단순히 Banzhaf보다 좋은 detector라는 식의 서사는 부적절하다.

3. **Influence Functions, TracIn, DataInf, LoGra 대비:** 이 계열은 gradient/influence 기반 attribution으로 재학습을 피한다는 점에서 정신적으로 가깝다. Flirds의 차별점은 FL round/client aggregation과 Shapley coalition utility를 explicit target으로 둔다는 점이다(§2.2, §3.1). 다만 Taylor/HVP 기반이라는 사실만으로는 novelty가 충분하지 않으므로, "influence score"가 아니라 "exact Shapley oracle에 대한 fidelity"가 논문의 방어선이어야 한다.

4. **IRDS 대비:** 가장 위험한 비교다. IRDS가 이미 중앙집중 per-step sample-level contribution을 in-run Taylor 전개로 계산한다면, Flirds는 FL setting, client-level, per-round local multi-step, LoRA backend, partial participation으로 옮긴 것이다(§2.2). 이것이 incremental인지 substantial인지는 다음 두 가지에 달려 있다. 첫째, FL local update delta가 per-sample step과 달리 aggregation, client sample count n_c, partial participation, multi-step optimizer trajectory를 포함하므로 기존 IRDS formula가 그대로 작동하지 않는다는 derivation을 명확히 해야 한다. 둘째, LLM-scale exact oracle 검증이 기존 IRDS가 보여주지 못한 empirical regime임을 설득해야 한다. 현재 dossier 기준으로는 substantial package가 가능하지만, method theorem이 약하면 "engineering extension"으로 낮아질 수 있다.

5. **GTG-Shapley, FedSV, ShapleyFL, ComFedSV, Ripple 대비:** 이들이 FL Shapley approximation의 직접 경쟁군이다. Flirds의 novelty는 다수 coalition model evaluation 없이 라운드당 HVP 1회에 가까운 비용으로 값을 산출한다는 점이며, std20에서 기존 방법과 fidelity가 갈린다는 결과가 핵심 근거다(§2.1, §3.5). 다만 anchor에서 전 방법이 +1.000이고 loss-heuristic도 강하므로, "기존 방법보다 정확하다"보다 "동급 또는 더 높은 fidelity를 훨씬 낮은 비용으로 얻는다"가 더 방어 가능한 claim이다.

6. **FLDetector, FLTrust, FedDQC 대비:** 이들은 detector이지 valuation estimator가 아니다(§3.3). Flirds를 detector로 포지셔닝하면 손해다. 정직한 novelty는 detector 대체가 아니라 "valuation score가 detector와 언제 일치하고 언제 분리되는가를 보여주는 분석"이다.

**최종 novelty verdict:** 메인트랙에서 살아남으려면 headline을 "IRDS extended to FL"로 두면 안 된다. Headline은 "LLM-scale FL contribution valuation with dual-oracle fidelity validation and near-zero per-round overhead"이어야 한다. 이 경우 novelty는 방법+검증+스케일의 결합으로 설득 가능하지만, hard-regime과 downstream actionability가 비어 있으면 borderline를 넘기 어렵다.

### E. 필수요소·분야요건 스코어카드

| 차원 | 판정 | 근거 |
|---|---|---|
| 문제·gap 명확성 | 충족 | exact Shapley의 2^N 재학습/평가 비용과 LLM-scale FL valuation 공백이 명확하다(§2.1). |
| Novelty | 부분 | IRDS 확장 자체는 incremental 위험이 있으나, LLM FL + HVP 1회 + dual oracle 검증 조합은 substantial 가능성이 있다(§2.2, §3.1). |
| 기술적 엄밀성 | 부분 | HVP/fp32/LoRA hygiene 등 구현 엄밀성은 강하지만 Taylor error, optimizer restriction, retrain oracle 폭이 약하다(§2.2, §3.5). |
| Significance / so what | 부분 | 비용 절감은 의미 있지만, downstream 성능/수렴 개선이 아직 계획 단계다(§3.5, §4). |
| 정직성·한계 | 충족에 가까움 | clean-preserving backdoor 회피와 detector와 valuation의 분리를 인정한다(§3.5). 단 논문 본문에서도 이 톤을 유지해야 한다. |
| 재현성 | 부분 | backend-agnostic API와 implementation caveat는 좋지만 DGX B200 의존, seed/variance/memory 정보가 부족하다(§2.2, §3.5, §4). |
| Ground-truth 검증 신뢰성 | 부분 | dual oracle은 강하지만 N=5 1B 중심이고 3B에서 (a)-(b) 0.900 균열이 있다(§3.1, §3.5). |
| Scalability | 충족 | estimator 비용은 Flirds ~107s vs coalition/eval 계열 ~530s, Ripple ~4515s이고 oracle은 N 증가에 지수적이다(§3.5). |
| 이론적 근거 | 부분 | Shapley target과 Taylor expansion은 명확하나 approximation bound와 FL-specific assumptions가 약하다(§2.2). |
| Robustness | 부분/미흡 | non-IID, α-sweep, corruption plan은 있으나 clean-preserving backdoor는 회피되고 hard-regime 실험이 미완이다(§3.2, §3.5, §4). |
| Actionability | 미흡/부분 | contribution-weighted 학습, MMLU/ROUGE, rounds-to-target가 아직 §4 계획이다. 현재 탐지 AUROC만으로는 부족하다(§3.5, §4). |
| Baseline 포괄성 | 충족에 가까움 | FL Shapley 계열, Banzhaf, loss heuristic, detector 계열을 포함한다(§3.3). 단 loss heuristic의 강세를 정면 분석해야 한다. |

### F. 논문 구조 설계

**권장 구조:** Method paper처럼 보이되, 실제 설득축은 "fidelity-first empirical validation paper"로 설계해야 한다. Abstract와 Results의 순서는 반드시 1차 fidelity → 2차 성능/수렴 → 3차 탐지여야 한다. 탐지를 앞에 놓으면 backdoor 회피 결과 때문에 논문 중심이 흔들린다.

#### Abstract

담을 내용: FL client contribution valuation의 Shapley target, exact Shapley의 지수 비용, Flirds의 1차+2차 Taylor/HVP estimator, dual oracle 검증, 핵심 결과 3개만. 숫자 우선순위는 std20 fidelity, cost, retrain oracle check 순서가 좋다.

포함할 수치: std20 Flirds Spearman +1.000 vs GTG +0.975±0.02, FedSV +0.91±0.09, ShapleyFL +0.19, ComFedSV +0.09; N=5 1B Flirds ~107s vs ~530s 계열; N=5 1B retrain/in-run/estimator +1.000(§3.5).

피해야 할 것: "detects poisoned clients" 같은 문장. 탐지는 secondary로 한 문장만, clean validation objective와 분리해 써야 한다.

#### 1. Introduction

담을 내용: Shapley가 FL incentive/client selection에 자연스럽지만 2^N coalition 재학습 때문에 LLM-scale에서 불가능하다는 문제(§2.1). 기존 FL Shapley approximations는 재학습은 피하나 다수 model evaluation이 필요하다는 gap(§2.1). Flirds의 핵심 아이디어: 학습 중 client delta와 validation Taylor expansion으로 contribution을 귀속한다(§2.2).

기여 bullet 권장:

1. LLM/LoRA FL을 위한 client-level in-run Shapley estimator: 1차+2차 Taylor, round당 HVP 1회, backend-agnostic logs(§2.2).
2. Exact in-run oracle와 retrain-based oracle을 함께 쓰는 fidelity protocol(§3.1).
3. 1B/3B/7B, anchor/std20/cross-device에서 11개 valuation baseline과 비교하는 fidelity-first evaluation(§3.2, §3.3).
4. Contribution-weighted training, convergence, detection은 secondary validation으로 배치하되, valuation과 detection의 차이를 명시(§3.4, §3.5).

그림: Figure 1 conceptual diagram. FedAvg rounds에서 server model w_r, client deltas Δ_{r,c}, validation gradient/HVP, φ_c 산출, exact oracle/retrain oracle 비교 흐름.

#### 2. Problem Setup and Valuation Game

담을 내용: FL protocol, client utility game, validation loss utility, Shapley definition, efficiency/symmetry/null-player/linearity를 왜 reference로 삼는지. 이 절에서 "우리는 clean validation-loss game의 contribution을 측정한다"를 못박아야 한다.

표/그림: Table 1 notation and games. Columns: target, utility, cost, role. Rows: retrain oracle (a), in-run exact oracle (b), Flirds estimator, baselines.

필요 실험/근거: §3.1 dual oracle 설계. 아직 필요한 것: validation set composition sensitivity를 보여주는 appendix 또는 analysis.

#### 3. Method: Flirds

담을 내용: IRDS로부터의 출발점, FL round-level local multi-step delta로의 변환, 1차 term(-∇_val alignment), 2차 curvature correction, HVP 1회 계산, aggregation over rounds/clients, sample count n_c 처리. Flirds-1st ablation도 여기서 정의한다.

필수 수식:

- Utility = validation loss decrease 또는 negative validation loss change로 정의.
- Δ_{r,c}에 대한 first-order term.
- Hessian quadratic term과 HVP 구현.
- Complexity: O(R·C_part·|Δ|) logging plus O(R·HVP_val) estimator vs O(2^N·R·val·seq) oracle.

그림/표: Algorithm 1 Flirds. Figure 2 complexity scaling schematic. Table 2 assumptions: fp32 master, eager attention, SGD momentum=0, LoRA keying, no model mutation.

필요 실험/근거: Flirds vs Flirds-1st ablation(§3.5), rank/LoRA probe(§4).

#### 4. Experimental Setup

담을 내용: 핵심 질문 위계 그대로 구성한다.

1. Primary: oracle fidelity metrics: Spearman, Kendall, Pearson, distance metrics(§3.4).
2. Secondary: actionability in order: general performance MMLU/ROUGE, convergence rounds-to-target, detection AUROC(§3.4).
3. Regimes: anchor N=5, std20 N=20 2/round R=200, cross-device N=100 Dirichlet α, CNN Track C(§3.2).
4. Models: Llama-3.2 1B/3B, Llama-2-7B, LoRA r16; CNN track separately(§3.2).
5. Baselines: 11 valuation methods and detector baselines(§3.3).

표: Table 3 experimental matrix. Rows = regimes; columns = N, participation, model, oracle available, metrics, status(existing/planned).

주의: CNN Track C는 main LLM evidence와 분리된 "additional controlled study"로 둔다.

#### 5. Results I: Fidelity to Exact In-Run Shapley

담을 내용: 논문의 핵심 결과. anchor는 "sanity check, not decisive"로 쓰고, std20과 cross-device를 먼저 해석해야 한다.

Claim mapping:

- Claim 1: Flirds matches exact in-run Shapley across scales/regimes. Evidence: std20 Flirds +1.000, Flirds-1st +0.999, GTG +0.975±0.02, FedSV +0.91±0.09, ShapleyFL +0.19, ComFedSV +0.09, FedIF +0.16(§3.5).
- Claim 2: Anchor N=5 is saturated. Evidence: all methods Spearman +1.000; Flirds Pearson 0.99999+(§3.5). 이 결과는 강점이 아니라 calibration/scatter evidence로 제한.
- Claim 3: cross-device α=0.5에서도 per-round exact oracle에 +1.000. Evidence: N=100 α=0.5(§3.5).

그림/표: Figure 3 rank/scatter plots vs oracle for anchor/std20/N100. Table 4 full fidelity metrics including Spearman/Kendall/Pearson/distances. 반드시 loss-heuristic을 같은 표에 넣어야 한다.

아직 필요한 것: std20/cross-device Pearson 및 distance metrics, seed variance, hard α/participation sweeps.

#### 6. Results II: Retrain-Based Oracle Validation

담을 내용: "in-run exact Shapley가 실제 retraining Shapley를 대체할 수 있는가"만 다룬다. 이 절은 작아도 매우 중요하다.

Claim mapping:

- Claim 4: N=5 1B fp32에서 retrain oracle (a), in-run oracle (b), estimator가 같은 ranking을 낸다. Evidence: +1.000(§3.5).
- Caveat claim: 3B에서는 (a)-valloss vs (b)가 +0.900으로 떨어진다. Evidence: §3.5.

그림/표: Figure 4 oracle triangle: (a) vs (b), Flirds vs (b), Flirds vs (a). Table 5 by model scale.

아직 필요한 것: std20 subset retrain check 또는 N=10 oracle 일부, 3B discrepancy analysis, deferred 7B (a)-oracle는 appendix라도 있으면 강해진다(§4).

#### 7. Results III: Cost and Scalability

담을 내용: 정확도와 비용을 joint plot으로 보여야 한다. 단순 wall-clock 표보다 x축 cost, y축 fidelity의 Pareto frontier가 더 설득력 있다.

Claim mapping:

- Claim 5: Flirds gives oracle-level fidelity at much lower cost. Evidence: Flirds-1st ~35s, Flirds ~107s vs GTG/FedSV/Banzhaf/ShapleyFL/(b) ~530s, Ripple ~4515s, loss-heur ~164s(§3.5).
- Claim 6: exact oracle scales exponentially in N, estimator does not. Evidence: N=5↔N=10에서 32배, cross-device (b) 771ms/fwd and R200 ~11h/4-GPU(§3.5).

그림/표: Figure 5 Pareto curve, Figure 6 theoretical and measured scaling, Table 6 hardware and memory.

아직 필요한 것: peak memory, validation size sensitivity, hardware-normalized throughput.

#### 8. Results IV: Actionability of Valuation

담을 내용: 2차 질문 중 ① 일반 성능 향상 → ② 수렴 속도 순서. 탐지는 아직 넣지 않는다.

Claim mapping:

- Claim 7: Contribution-weighted training improves or preserves downstream performance. 현재는 §4 Track D 계획으로 미완.
- Claim 8: Valuation-based weighting reaches target performance in fewer rounds. 현재는 §4 Track D 계획으로 미완.

그림/표: Figure 7 MMLU/ROUGE vs intervention arm, Figure 8 rounds-to-target. Table 7 intervention arms.

아직 필요한 것: Track D 본런의 6 intervention arms, MMLU/ROUGE, convergence curves(§4). 결과가 약하면 정직하게 "fidelity without consistent downstream gain"으로 써야 한다.

#### 9. Results V: Robustness, Corruption, and Detection

담을 내용: 2차 질문 중 마지막인 탐지. threat-matched detector와 Flirds φ를 비교하되, valuation과 detection의 target mismatch를 명확히 한다.

Claim mapping:

- Claim 9: Flirds φ can identify some low-quality/free-rider clients but is not a universal detector. Evidence: noisy AUROC 0.75, free-rider 1.0, N=5 coarse(§3.5).
- Claim 10: Clean-preserving backdoor can have high contribution under clean val-loss. Evidence: ASR 1.0 공격자가 clean val-loss를 낮춰 high φ rank(§3.5).
- Claim 11: CNN Track C suggests oracle rank stability emerges in non-IID/corruption cells, not IID cells. Evidence: rank stability 0.51-0.97 vs IID ~0(§3.5).

그림/표: Figure 9 detection AUROC by threat, Figure 10 backdoor scatter of clean val-loss contribution vs ASR, Table 8 detector costs and AUROC.

아직 필요한 것: 오염축 × 비IID축 2×2 matrix(§4), adversary-aware utility discussion.

#### 10. Analysis: When Is the Shapley Signal Real?

담을 내용: near-additivity, signal size, participation, LoRA rank, α sweep, IID vs non-IID, corruption이 contribution signal을 어떻게 만드는지. 이 절이 없으면 "all methods tie" 공격을 막기 어렵다.

그림/표: Figure 11 signal size by rank/participation; Figure 12 method gap vs interaction strength; Table 9 regimes where loss heuristic fails/succeeds.

필요 실험: LoRA rank {16,32,64}, participation N=50/5, IID/non-IID corruption matrix(§4).

#### 11. Limitations and Threats to Validity

담을 내용: clean validation-loss game의 제한, N=5 retrain oracle 한계, 3B (a)-(b) gap, SGD momentum=0, LoRA-only, DP/secure aggregation/compression 부재, detector가 아님. 이 절은 방어적으로 숨기지 말고 본문 Results와 연결해야 한다.

#### 12. Conclusion

담을 내용: "Flirds makes Shapley-style client valuation measurable during LLM FL training" 정도로 제한된 결론. "solves bad-client detection" 또는 "fair incentives generally"는 피한다.

### G. 예상 리뷰어 반론 + 반박

1. **공격: 실험 game이 너무 쉬워서 모든 방법이 맞는다.** 현재 근거로 가능한 반박: std20에서 GTG, FedSV, ShapleyFL, ComFedSV가 갈리고 Flirds가 +1.000을 유지한다(§3.5). 부족한 점: loss-heuristic도 거의 +1.000이라 이 반박은 절반만 유효하다. 필요한 추가 실험: loss-heuristic이 실패하는 hard interaction regime, LoRA rank/participation lever, non-IID×corruption matrix, Pearson/distance metric 공개(§4).

2. **공격: Flirds는 true retrain Shapley가 아니라 in-run surrogate oracle만 맞춘다.** 현재 반박: N=5 1B fp32에서 (a)=(b)=estimator +1.000(§3.5). 부족한 점: 3B에서 (a)-(b) +0.900으로 이미 gap이 있다. 필요한 추가 실험: 3B discrepancy decomposition, N=10 또는 std20 sampled retrain oracle, 7B retrain oracle 일부(§4 deferred).

3. **공격: 방법은 IRDS를 FL client delta에 적용한 incremental extension이다.** 현재 반박: FL에서는 local multi-step client delta, partial participation, aggregation, LoRA backend, HVP constraints가 결합돼 중앙집중 per-step sample-level IRDS와 다르다(§2.2). 부족한 점: dossier만으로는 수식적 novelty가 충분히 분명하지 않다. 필요한 서술: IRDS formula가 직접 적용되지 않는 지점과 Flirds derivation의 새 항/assumption을 명시.

4. **공격: clean validation loss contribution은 fair reward나 safety에 맞지 않는다.** 현재 반박: 논문은 clean val-loss game의 Shapley를 측정한다고 정의하고, clean-preserving backdoor가 high φ가 되는 것을 정직하게 보고한다(§3.1, §3.5). 부족한 점: incentive/safety use-case까지 확장하는 문장은 위험하다. 필요한 추가 실험/서술: val set mixture sensitivity, safety-aware utility 또는 ASR-penalized utility를 별도 game으로 제시.

5. **공격: 정확한 valuation이 실제 성능이나 수렴을 개선한다는 증거가 없다.** 현재 반박: contribution-weighted learning, MMLU/ROUGE, rounds-to-target가 Track D에 계획돼 있다(§4). 부족한 점: 현재 확정 결과에는 없다. 필요한 추가 실험: Track D 본런의 intervention arms를 핵심 Results IV로 완성하고, gain이 없을 경우 왜 valuation fidelity 자체가 여전히 의미 있는지 scope를 낮춰야 한다.

6. **공격: SGD momentum=0, fp32, eager attention 같은 제한 때문에 실제 FL training과 다르다.** 현재 반박: IRDS per-step assumption과 numerical precision 때문에 필요한 통제이며, valuation run의 엄밀성을 위한 선택이다(§2.2). 부족한 점: production FL/LLM training은 AdamW, momentum, bf16, fused attention을 쓴다. 필요한 추가 실험/서술: training backend는 현실적으로 돌리되 valuation replay만 fp32/eager로 하는지, optimizer-state correction 가능성, 제한을 Limitations에 명확히 기재.

7. **공격: detector comparison은 threat-matched가 아니거나 Flirds에 불리/유리하게 구성됐다.** 현재 반박: threat별 detector FedDQC/STD-DAGMM/FLTrust/FLDetector를 배치했다(§3.3). 부족한 점: Flirds는 detector가 아니므로 detector AUROC로 주 claim을 세우면 안 된다. 필요한 서술: detector section은 secondary이며, bad-client detection은 utility definition과 threat model에 의존한다고 명시.

8. **공격: N=5 anchor와 N=20 2/round는 Shapley high-order coalition interaction을 검증하지 못한다.** 현재 반박: exact oracle tractability를 위해 선택한 anchor이고, cross-device N=100 α=0.5도 있다(§3.2, §3.5). 부족한 점: per-round exact decomposition이 high-order interaction을 충분히 만들지 의문이다. 필요한 실험: N=10 exact/retrain 일부, participation 5/50, higher client-per-round subgames(§4).

### H. §4 계획 실험 완료 후에도 남는 갭

1. **Full retrain oracle의 scale gap:** Track D와 rank/participation probe를 해도 N=10 oracle과 7B (a)-oracle은 deferred로 남는다(§4). true retrain Shapley fidelity claim의 상한이 남는다.

2. **Modern optimizer gap:** valuation run이 plain SGD momentum=0에 묶여 있다(§2.2). AdamW, momentum, client optimizer state가 있는 실제 SFT/FL에서 Taylor attribution이 얼마나 유지되는지는 별도 문제다.

3. **Secure/realistic FL systems gap:** DP, secure aggregation, update compression, client dropout, stragglers, heterogeneous compute가 없다. LLM FL production relevance를 주장하려면 한계로 인정해야 한다.

4. **Utility construct gap:** clean validation loss game만으로 fair reward, safety, robustness, user utility를 대표할 수 없다. §4 matrix가 contamination/non-IID를 분리해도 utility definition 자체의 normativity는 해결하지 못한다.

5. **Adversarial adaptation gap:** clean-preserving backdoor가 이미 Flirds를 회피한다(§3.5). §4 poison/noisy/free-rider matrix를 해도 valuation-aware attacker가 φ를 조작하는 문제는 남는다.

6. **Actionability mechanism gap:** Contribution-weighted training이 성능을 올려도, fair compensation, client selection policy, incentive compatibility까지 증명되는 것은 아니다.

7. **Generalization beyond instruction-response formatting:** format 균일화는 fairness를 높이지만, real cross-silo FL에서는 output format/task가 client마다 다를 수 있다(§2.3). 논문은 이 통제가 필요한 범위와 외삽 한계를 인정해야 한다.

8. **Theoretical bound gap:** rank/participation probe는 empirical diagnosis다. Taylor residual과 coalition interaction error에 대한 bound 또는 diagnostic이 없으면 method generality는 empirical로만 남는다.

9. **Validation set sensitivity gap:** domain mixture, val size 200/domain, loss metric choice가 φ에 미치는 영향이 별도 분석되지 않으면 contribution은 "this validation set"에 종속된다(§2.3).

10. **Cost portability gap:** DGX B200 x4에서의 wall-clock은 인상적이지만, commodity GPU나 memory-limited setting에서 HVP cost와 fp32 replay가 어떻게 변하는지는 남는다(§3.5).

### I. 포지셔닝·서사 추천

**최적 스토리:** "Flirds is a fidelity-first, dual-oracle validated estimator for Shapley-style client contribution in LLM federated learning." Headline은 탐지나 보상이 아니라 "oracle-level contribution valuation at LLM scale without coalition evaluation"이어야 한다. Abstract/Intro/Results의 순서는 1차 fidelity → 2차 성능/수렴 → 마지막 탐지로 고정한다. 정직한 한계는 "clean validation-loss game에 대한 valuation이며, detector나 universal fairness metric이 아니다"이다. 이 스토리는 §3.5의 std20 fidelity와 cost 결과를 가장 잘 살리고, backdoor 회피 결과를 약점이 아니라 construct validity 분석으로 흡수할 수 있다.

**대안 스토리:** "From IRDS to FL: a second-order Taylor account of client contribution under FedAvg." 이 스토리는 method novelty를 전면에 둔다. 장점은 IRDS와의 이론적 연결을 명확히 할 수 있다는 점이고, 단점은 Reviewer가 "direct extension"이라고 공격하기 쉽다는 점이다. 이 대안을 쓰려면 theorem/derivation이 강해야 하고, 실험은 method validation의 보조가 된다. 현재 dossier 기준으로는 최적 스토리보다 위험하다.

**Headline 배치:** std20에서 Flirds +1.000, 기존 FL Shapley baselines의 하락, Flirds ~107s vs ~530s/Ripple ~4515s를 전면에 둔다(§3.5). Anchor N=5 전 방법 +1.000은 headline이 아니라 sanity/calibration으로 낮춘다. 탐지 AUROC는 마지막 secondary result로 둔다.

### J. Threats to Validity + Related-Work 지도

#### Internal validity

1. Exact oracle (b)가 동일 로그의 utility를 재구성하므로 estimator와 shared assumptions를 가진다(§3.1). retrain oracle (a)가 이를 보완하지만 범위가 좁다(§3.5).
2. fp32/eager attention/functorch/LoRA keying 같은 numerical implementation detail이 결과에 큰 영향을 줄 수 있다(§2.2).
3. Baseline implementations와 budget parity가 중요하다. 특히 loss-heuristic이 강하므로, 모든 method의 input information과 evaluation budget을 명확히 해야 한다(§3.3, §3.5).
4. N=5 Spearman 포화는 metric artifact일 수 있다(§3.5). Kendall/Pearson/distance와 scatter가 필수다.
5. 3B에서 (a)-(b) +0.900 gap은 oracle construction 또는 training stochasticity 문제일 수 있다(§3.5).

#### External validity

1. LoRA r16, SGD momentum=0, selected Llama scales가 전체 LLM FL을 대표하지 않는다(§2.2, §3.2).
2. 5-domain instruction-response formatting은 공정성을 위해 필요하지만 real client heterogeneity를 줄인다(§2.3).
3. Cross-silo N=5와 std20 2/round는 high-order coalition interaction이 약할 수 있다(§3.2).
4. DGX B200 x4 결과가 lower-end hardware에 그대로 이식된다고 볼 수 없다(§3.5).
5. CNN Track C는 domain이 다르고 LLM 결론의 직접 증거가 아니다(§3.5).

#### Construct validity

1. φ는 clean validation-loss utility에 대한 contribution이지 "도덕적 공정성", "안전성", "보상 받을 자격" 자체가 아니다(§3.1, §3.5).
2. Detection AUROC는 valuation fidelity의 proxy가 아니다. 특히 clean-preserving backdoor는 high contribution이 정직한 답일 수 있다(§3.5).
3. MMLU/ROUGE downstream gain이 있더라도 Shapley fidelity와 incentive compatibility는 별개다(§3.4).
4. Equal train size와 format control은 confound를 줄이지만, size/value tradeoff를 연구하지는 않는다(§2.3).

#### Related-work 지도

1. **Classical data valuation:** Data Shapley, KNN/Beta-Shapley, Data-Banzhaf는 value target과 approximation의 모태다. Flirds는 새 공리보다 FL/LLM 계산 가능성을 기여한다.
2. **Gradient/influence attribution:** Influence Functions, TracIn, DataInf, LoGra는 retraining-free attribution의 인접 축이다. Flirds는 Shapley oracle fidelity와 FL client aggregation을 target으로 삼는 점에서 갈라진다.
3. **IRDS:** 직계 부모다. Flirds의 novelty는 FL client-level/per-round/multi-step/LoRA setting과 dual oracle 검증으로 방어해야 한다.
4. **FL Shapley approximations:** GTG-Shapley, FedSV, ShapleyFL, ComFedSV, Ripple이 직접 baseline이다. Flirds는 "less evaluation, similar or higher fidelity"가 차별점이다.
5. **FL anomaly/detector:** FLDetector, FLTrust, FedDQC는 다른 문제를 푼다. Flirds와 비교하되, valuation score가 detector가 아니라는 경계선을 유지해야 한다.

## Pass 2: Appendix A 및 레포 검증 이후 대조

### Repo 검증 메모

1. **Track D는 더 이상 "계획만 있고 미완"이 아니다.** dossier §4는 Track D 본런을 planned로 적었지만, 현재 레포에는 `runs/track_d/rundirs` 18개 셀(1B/3B/7B × std20/anchor5 × 3 seed)이 존재하고, `python runs/track_d/make_fidelity.py`로 `runs/track_d/fidelity.csv`를 재생성했다. 따라서 Pass 1의 "downstream/actionability가 아직 계획 단계"라는 판단은 dossier 기준으로는 맞지만, current repo 기준으로는 "clean-IID Track D 결과는 존재하며, 결과는 do-no-harm parity/미세한 val-loss 이득"으로 갱신해야 한다.

2. **현재 checkout에는 파생 결과 파일 일부가 빠져 있었다.** `runs/track_d/fidelity.csv`는 없어서 재생성했고, `runs/phase2_matrix/RESULTS.md`는 `make_analysis.py`를 돌려도 현재 코드가 생성하지 않는다. 대신 `runs/phase2_matrix/analysis/00_overview/master_metrics.csv`와 chart/CSV 34개가 생성된다. 최신 overview 문서의 "RESULTS.md 재생성" 설명은 현재 코드와 약간 stale하다. 결론 영향은 없다. 수치는 `fidelity.csv`와 `master_metrics.csv`에서 직접 대조했다.

3. **phase2 분석 스크립트는 Windows 기본 cp949에서 실패하고 UTF-8 모드가 필요했다.** `python runs/phase2_matrix/make_analysis.py`는 `master_queue.txt`의 UTF-8 문자를 cp949로 읽다가 실패했고, `python -X utf8 runs/phase2_matrix/make_analysis.py`는 성공했다. 재현성 섹션에는 Windows/locale caveat 또는 explicit UTF-8 read_text가 필요하다.

4. **Track D fidelity 수치는 dossier의 핵심 패턴을 유지한다.** 재생성한 `runs/track_d/fidelity.csv` 기준 std20 Spearman mean은 1B: Flirds +1.000, loss-heur +1.000, Flirds-1st +0.999, GTG +0.975, FedSV +0.910, ShapleyFL +0.194, ComFedSV +0.093, FedIF +0.157이다. 3B/7B도 Flirds가 +1.000/+0.999이고 loss-heur도 +0.999/+0.999로 거의 동률이다. 즉 "정확도 우위"보다 "near-additive game에서 exact in-run oracle을 저비용으로 재현"이 더 정직한 headline이다.

5. **Track D anchor5는 dossier보다 더 미묘하다.** 1B anchor5에서는 Flirds/Flirds-1st/GTG/loss-heur/Banzhaf가 모두 vs (b) +1.000이고, (a)oracle vs (b)는 +0.933±0.058이다. 3B anchor5에서는 Flirds full이 +0.967±0.058이고 Flirds-1st/loss-heur/GTG가 +1.000이다. N=5에서는 한 client swap만으로 Spearman이 크게 변하므로 full 2차항이 약간 떨어지는 것 자체를 과대해석하면 안 되지만, "2차가 항상 더 좋다"는 문장은 쓸 수 없다.

6. **3B retrain oracle은 canonical source가 갈린다.** dossier와 `CLAUDE.md`는 별도 task6에서 3B (a)-valloss vs (b)=+0.900을 언급하지만, current `runs/track_d/rundirs`와 재생성된 `fidelity.csv`에는 3B/7B (a) oracle이 없다. 논문 표의 canonical을 "rundir-only reproducible result"로 잡으면 3B (a)=+0.900은 별도 provenance를 찾아 부록으로만 쓰거나, 재실행 후 편입해야 한다. 결론은 더 엄격해진다: LLM retrain-oracle 검증은 현재 파일 기준 1B anchor5에만 단단하다.

7. **비용 claim은 regime-specific로 바꿔야 한다.** phase2 N=5 robustness에서는 Flirds ~107s vs coalition/oracle ~530s가 맞지만, Track D 1B std20에서는 라운드당 2명 참여라 (b) oracle이 2917s이고 Flirds full은 4697s로 더 느리다. 반대로 device100 anchor K=10에서는 Flirds 157s vs (b) 24,975s로 약 160× 차이가 난다. 따라서 "Flirds is cheaper"가 아니라 "cohort size가 커질수록 exact 2^K 대비 cohort-independent HVP 비용이 압도적으로 유리하다"가 정확한 claim이다.

8. **Fed-LOO baseline은 코드에는 추가됐지만 결과에는 아직 없다.** `codes/flirds/oracle/in_run_sv.py`에 `in_run_loo`가 있고 `codes/experiments/track_d.py`는 `Fed-LOO`를 compute_fidelity에 와이어링했다. 그러나 기존 rundir의 `phi.parquet`는 Fed-LOO 추가 전 생성되어 `fidelity.csv`에는 Fed-LOO 행이 없다. Baseline audit의 A-gap은 "구현 완료, 수치 재실행 대기"로 해석해야 한다.

9. **poison/backdoor 해석은 Pass 1보다 수정되어야 한다.** dossier만 보면 clean-preserving backdoor가 clean val-loss를 낮춰 Flirds가 high contribution으로 평가하는 "valuation의 정직한 답"처럼 보였지만, current phase2 CSV에서는 1B silo5 poison에서 exact (b), loss-heur, Banzhaf, detector들이 AUROC 1.0이고 Flirds-1st만 AUROC 0.0, Flirds full은 0.917이다. 즉 이 셀은 "utility game이 공격자를 도움으로 평가했다"보다 "1차 Taylor가 큰 scaled update에서 부호를 틀렸고 2차항이 상당 부분 복원했다"가 맞다. 단 3B silo5 poison은 1 seed에서 Flirds/Flirds-1st 둘 다 AUROC 0.0이고, device100 poison은 Flirds AUROC 1.0이라 설정 의존이다.

10. **코드 현실은 method description과 잘 맞는다.** `flirds_estimator.py`는 `loss_fn(params,buffers)`와 logs만 받고, true Hessian HVP를 `torch.func.jvp(grad(vloss), dW)`로 라운드당 1회 계산한다. `in_run_sv.py`는 exact (b) oracle과 per-round decomposition을 분리하고, `exact_sv_llm.py`는 (a) retrain oracle을 별도 code path로 둔다. `llm.py`는 5-domain free-form loader와 Alpaca IID loader를 모두 갖고 있으며 answer_swap/backdoor corruptor hook이 있다. 방법 설명과 구현 사이의 큰 불일치는 없다.

### K. 저자 자기진단과의 대조

#### 겹치는 것

1. **near-additivity/easy-game 우려는 완전히 겹친다.** Pass 1의 최강 reject 논거는 "완벽한 fidelity가 near-additive/easy game의 산물일 수 있다"였고, Appendix A는 이를 더 정량화한다: 가산성 갭이 anchor에서 Σφ의 0.1-0.9% 수준, std20에서도 0.0-0.5%이며, singleton 순위 vs Shapley 순위가 전 셀 ρ=+1.00이라고 진단한다. 이건 단순한 약점이 아니라 논문 서사의 중심 제약이다.

2. **IID-clean에서는 클라이언트 간 real signal이 구조적으로 없다는 판단이 겹친다.** Pass 1은 anchor N=5와 std20 2/round가 interaction을 충분히 만들지 못한다고 봤다. Appendix A는 더 직접적으로, Track D (b) oracle 자기 순위의 cross-seed Spearman이 1B anchor -0.37, 1B std20 -0.11, 3B std20 -0.24 등 거의 0이라고 보여준다. 통제 대조군도 같은 방향이다: CNN cifar10 IID는 −0.04지만 label_flip/quantity_skew는 0.97이고, LLM IID-alpaca는 −0.37~-0.11인 반면 silo5 5-domain non-IID는 +0.93~1.00이다. 즉 estimator가 noisy한 것이 아니라 oracle 자체가 IID shard lottery를 랭킹한다.

3. **loss-heuristic 강세가 method 필요성을 압박한다는 판단이 겹친다.** Pass 1에서 loss-heur가 std20 +1.000인 점을 reject-grade 리스크로 봤고, Appendix A도 semivalue가 사실상 같은 순위로 붕괴한다고 본다. 이 때문에 Flirds의 clean-IID headline은 "accuracy superiority"가 아니라 "fidelity at lower or scaling-favorable cost"여야 한다.

4. **actionability가 clean-IID에서는 parity라는 판단이 겹친다.** Pass 1은 Track D가 끝나도 clean-IID에서는 downstream gain이 작을 가능성을 우려했다. Appendix A와 current Track D는 base→vanilla가 val-loss/ROUGE는 움직이지만 MMLU는 0 또는 하락이며, intervention arm은 paired val-loss에서만 −0.001~−0.004 정도로 보이고 MMLU/ROUGE는 분해능 밖이라고 진단한다. 이것은 "성능 향상" headline을 약화시키지만 "do-no-harm under accurate valuation"으로는 쓸 수 있다.

5. **detector와 valuation을 분리해야 한다는 판단이 겹친다.** Pass 1은 탐지를 마지막으로 둬야 한다고 했고, Appendix A와 phase2 결과는 device100 noisy에서 exact (b) oracle AUROC 자체가 0.604±0.050이고 FedDQC는 1.0이라는 점을 보여준다. 이는 Flirds 근사 실패가 아니라 selected utility game의 탐지 한계다.

6. **fp32/bf16 precision 문제에 대한 결론은 방향이 겹치되 severity는 Appendix A가 더 낮춘다.** Pass 1은 fp32 master weight가 엄밀성 신호라고 봤다. Appendix A는 더 나아가 현재 fp32가 병목은 아니며, 관측 φ 차이 1e-4~1e-3은 fp32 ulp ~1.7e-7보다 2-4자릿수 크다고 말한다. 즉 precision은 "올바르게 처리한 engineering hazard"이지 남은 핵심 약점은 아니다.

#### 저자가 놓쳤는데 Pass 1/레포 대조가 잡은 것

1. **최상위 novelty의 가장 약한 고리는 IRDS-in-FL incremental 공격이다.** Appendix A는 signal diagnosis에 집중하지만, top-tier review에서는 "IRDS를 client delta에 적용한 engineering extension"이라는 공격이 더 직접적이다. 이를 막으려면 FL multi-step delta, participant-normalized FedAvg weights, partial participation, LoRA HVP, dual oracle protocol이 기존 IRDS에서 왜 nontrivial한지 Related/Method에서 명시해야 한다.

2. **retrain oracle의 canonical status가 아직 불안정하다.** Appendix A는 1B anchor5 (a) vs (b)=0.933과 기존 노트의 3B 0.900을 언급하지만, current Track D rundir에는 3B/7B (a)가 없다. 논문 제출용 artifact 기준으로는 "LLM (a)-oracle은 1B anchor5 only"가 안전하다. 3B 0.900을 쓰려면 raw provenance와 재현 스크립트를 찾아야 한다.

3. **cost advantage를 한 문장으로 말하면 틀린다.** Appendix A는 신호 진단 중심이라 비용의 regime-dependence를 덜 강조한다. 레포 수치상 1B std20에서는 Flirds full 4697s가 (b) 2917s보다 느리다. Flirds의 비용 우위는 K=5 anchor나 K=10 device100처럼 cohort가 큰 곳에서 강하다. Abstract의 cost sentence는 반드시 "cohort-independent, favorable when K is not tiny"로 써야 한다.

4. **Fed-LOO baseline은 paper table에 들어갈 준비가 아직 끝나지 않았다.** Baseline audit는 A-gap을 정확히 잡았고, 코드 추가도 됐지만 기존 result tables에는 없다. 이 상태로 논문에 "baseline comprehensive"라고 쓰면 Reviewer가 LOO 누락을 지적할 수 있다. 최소 Track D/C1 fidelity 재실행으로 Fed-LOO 한 행을 채워야 한다.

5. **Windows/encoding 재현성 caveat가 있다.** `make_analysis.py`가 기본 cp949에서 실패한 것은 논문 본문 이슈는 아니지만 artifact/reproducibility checklist에는 들어가야 한다. `Path.read_text(encoding="utf-8")` 같은 작은 수정으로 해결 가능하다.

6. **poison 해석은 dossier 문장 그대로 쓰면 위험하다.** current phase2에서 1B silo5 poison은 "clean val-loss game이 공격자를 도움으로 본다"가 아니라 "exact game은 잡는데 1차 Taylor가 실패한다"에 가깝다. 이 차이는 construct validity의 핵심이다. backdoor section은 threat별로 `(b)oracle`, loss-heur, Flirds-1st, Flirds full을 나란히 보여줘야 한다.

#### Pass 1이 놓쳤는데 저자 자기진단이 잡은 것

1. **MMLU가 표본 SE 아래라 검출 불가능하다는 정량화.** Pass 1은 downstream actionability가 약하다고만 봤지만, Appendix A는 MMLU 효과 크기 ~0.001이 SE ±0.004보다 작다고 명시한다. 이것은 "결과가 약하다"가 아니라 "이 무대에서 MMLU로 intervention 차이를 검출하는 실험 설계 자체가 under-powered"라는 더 정확한 진단이다.

2. **paired vs unpaired 비교의 차이.** Pass 1은 seed variance를 일반적 리스크로 봤다. Appendix A는 unpaired final val-loss seed std 0.021-0.027이 intervention 효과보다 20배 크지만, paired val-loss Δ는 SNR 2.4-4.5로 보인다고 분리한다. 논문은 intervention 결과를 반드시 paired design으로 보고해야 한다.

3. **신호 크기와 신호 실재성의 구분.** Appendix A는 rank↑가 φ 규모를 키울 수 있어도 IID-clean 클라이언트 순위의 실재성을 만들지 못하고, 참여수↑가 per-round subgame 크기와 method 구별력을 키운다고 구분한다. Pass 1은 hard-regime 필요성을 말했지만, 이 두 축을 충분히 분해하지 않았다.

4. **CNN Track C를 단순 보조가 아니라 diagnostic control로 쓰는 관점.** Pass 1은 CNN을 별도 track으로 낮췄지만, Appendix A는 label_flip oracle stability 0.968 vs IID -0.042를 들어 "심긴 신호가 있으면 oracle 순위가 안정된다"는 대조군으로 쓴다. 이 역할은 논문 Analysis 섹션에서 유용하다.

#### 심각도 판단이 갈리는 것

1. **clean-IID null result의 의미.** 저자 진단은 clean-IID parity를 설계상 정직한 결과로 받아들인다. 나는 top-tier 심사 관점에서 이것을 여전히 significance 리스크로 본다. "정확히 측정했지만 쓸 곳이 없다"는 반론을 막으려면 오염/비IID/품질격차가 있는 무대에서 valuation-based intervention이 더 큰 실효성을 보여야 한다.

2. **near-additivity의 위치.** 저자 진단은 near-additivity를 무대 특성으로 설명한다. 나는 이것을 paper-acceptance의 핵심 취약점으로 본다. 이유는 top-tier reviewer가 method novelty를 실험 난이도와 함께 본다는 점이다. easy game의 정밀한 분석은 좋지만, method paper의 main evidence가 easy game에 묶이면 accept bar에 못 미친다.

3. **3B poison failure의 severity.** 저자 진단은 n=1 caveat를 강하게 둔다. 나는 이것을 단순 robustness caveat가 아니라 Taylor approximation의 trust-region 한계로 해석한다. 3B 3-seed와 attack-strength sweep 없이 "2차항이 poison에서 결정적"이라고 쓰면 과장이다.

4. **(a) vs (b) oracle gap의 severity.** 저자 문서는 CNN에서 (a)와 (b)가 갈리는 것을 discovery로 본다. 나는 이것이 "true contribution"이라는 표현을 제한해야 하는 심각한 construct risk라고 본다. Flirds는 정확히는 "frozen-trajectory in-run Shapley" estimator이며, retrain counterfactual Shapley는 별도 target이다.

#### 저자 진단을 넘어서는 통찰

1. **논문은 "hardness ladder"를 명시해야 한다.** Clean IID는 calibration/parity, std20 partial은 estimator strategy gap, device100 K=10은 cost scaling, poison/non-IID/CNN은 non-additivity/hardness, retrain oracle은 construct validation이다. 이 ladder 없이 수치를 나열하면 +1.000이 너무 쉬워 보인다.

2. **Flirds full vs Flirds-1st의 역할은 benign과 adversarial에서 다르다.** Clean LLM에서는 거의 동률이고, CNN benign에서는 full이 도움되며, 1B silo5 poison에서는 full이 1st의 부호 실패를 복원하지만, 3B poison에서는 full도 실패한다. 따라서 2차항 claim은 "always improves"가 아니라 "curvature term is load-bearing in non-additive or large-update regimes, but has a trust-region boundary"로 써야 한다.

3. **loss-heuristic을 적으로 숨기지 말고 diagnostic baseline으로 승격해야 한다.** loss-heur가 강한 것은 논문 약점이지만, 동시에 game additivity를 드러내는 probe다. Table에는 loss-heur를 반드시 포함하고, "when singleton/LOO/Shapley agree, the game has little interaction"이라는 분석을 넣어야 Reviewer 2의 공격을 선제적으로 흡수할 수 있다.

4. **"LLM-scale" claim은 model size보다 oracle protocol과 implementation constraints에서 나온다.** 단순히 1B/3B/7B LoRA를 돌렸다는 것보다, fp32/eager/HVP/chunked loss/dual oracle/per-round exact decomposition을 실제로 구성했다는 점이 분야 기여다. Abstract는 model size 숫자보다 "dual-oracle fidelity validation under LLM FL logs"를 앞세워야 한다.

5. **proxy-truth off-anchor 수치는 paper figure에서 명확히 분리해야 한다.** device100 α≠0.5의 Spearman은 Flirds proxy 기준이고, exact (b)는 α=0.5 anchor에만 있다. proxy 수치를 fidelity evidence로 쓰면 순환논증이 된다. off-anchor α-sweep은 detection/robustness trend로만 쓰고, exact-fidelity claim은 anchor cell에 제한해야 한다.

6. **Top-tier용 final story는 "method + measurement science"여야 한다.** Flirds 자체만으로는 IRDS extension 공격이 가능하고, downstream gain만으로는 clean-IID null 때문에 약하다. 가장 강한 기여는 "federated LLM contribution valuation을 exact in-run/retrain oracle로 검증 가능한 measurement problem으로 만든 것"이다.

#### 뒤집을 반론: near-additivity와 do-no-harm을 강점으로 쓸 수 있는가

1. **near-additivity는 약점만은 아니다.** 제대로 프레이밍하면 이것은 "Flirds가 쉬운 게임을 만든다"가 아니라 "LLM LoRA FedAvg의 clean-IID valuation game이 실제로 어떤 구조를 갖는지 측정했다"는 empirical finding이다. Shapley/Banzhaf/singleton/LOO가 동률로 붕괴한다는 결과는 semivalue 이론의 선형성 sanity check이기도 하다. 따라서 논문에는 "clean-IID calibration regime"으로 넣을 수 있다.

2. **하지만 near-additivity는 main novelty evidence가 될 수 없다.** Reviewer가 요구하는 것은 "어려운 interaction이 있을 때 Flirds가 왜 필요한가"이다. 그러므로 clean-IID +1.000은 Figure 1/2의 headline이 아니라, hardness ladder의 첫 칸이어야 한다. Main claim은 std20 partial participation, device100 K=10 cost scaling, poison/non-IID/CNN hard-regime에서 세워야 한다.

3. **do-no-harm parity도 강점으로 쓸 여지는 있다.** Contribution-weighted learning이 clean-IID에서 성능을 망치지 않는다는 것은 operational safety sanity check다. 특히 정확한 valuation이 없는 상황에서 weighting/selection이 clean clients를 임의로 흔들 수 있다는 점을 생각하면, parity는 "무의미"가 아니라 "intervention machinery가 균질한 무대에서 과잉 반응하지 않는다"는 evidence가 된다.

4. **그러나 do-no-harm은 significance를 대신하지 못한다.** Top-tier 메인트랙에서는 "해치지 않는다"만으로는 부족하다. 논문은 parity를 낮은 순위의 안정성 결과로 두고, 실제 이득은 비IID·오염·품질격차가 있는 B축에서 보여야 한다. Appendix A가 말한 오염축×비IID축 2×2 matrix는 이 약점을 메우는 필수 실험이지 optional ablation이 아니다.

5. **가장 강한 반론은 오히려 저자에게 유리하게 바꿀 수 있다.** "IID-clean에서 client 간 진짜 신호가 없다"는 것은 방법 실패가 아니라 valuation problem의 identifiability statement다. 이걸 명시하면 논문이 더 정직해진다: Flirds는 존재하지 않는 client-quality ordering을 만들어내지 않고, oracle이 seed-불안정하면 estimator도 그 불안정성을 재현한다. 단 이 주장은 "우리는 oracle을 충실히 측정한다"는 주장만 방어하고, "실용적 client selection을 개선한다"는 주장은 방어하지 않는다.

### L. 한 줄 종합 판정

**현재 상태:** 최상위 메인트랙 기준은 **weak reject / major-revision 성향**이다. exact in-run fidelity와 implementation은 강하지만, clean-IID near-additivity, loss-heuristic 동률, LLM retrain-oracle 폭 부족, regime-dependent cost, actionability의 미세한 효과가 아직 accept-grade story를 막는다. Appendix A는 이 판단을 약화시키기보다 더 정밀하게 만든다: 병목은 fp32가 아니라 client 간 실재 신호 부재와 near-additive game이다.

**§4/추가 계획 완료 시 예상:** Fed-LOO 포함 재실행, rank/참여 probe, 오염축×비IID축 2×2 hard-regime, 큰-val device100 anchor, 3B/7B 또는 N=10 retrain-oracle 일부가 채워지면 **borderline to weak accept**까지 갈 수 있다. 단 그때도 headline은 "universal downstream improvement"가 아니라 "dual-oracle validated in-run Shapley measurement for LLM FL, with clear limits"여야 한다.
