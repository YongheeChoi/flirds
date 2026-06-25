---
type: survey
title: "검증 실험 정리 — CNN 트랙 / LLM 트랙 (E1–E7)"
created: 2026-06-25
updated: 2026-06-25
tags: [survey, validation, experiments, cnn, llm]
---

# 검증 실험 (E1–E7) — CNN 트랙 / LLM 트랙

각 연구가 "목적을 어떤 실험으로 검증했는지"를 실험 유형별로 정리. **CNN과 LLM은 경향이 크게 달라 표를 분리**한다(한 연구가 둘 다 했으면 양쪽에 등장). 셀엔 가능한 한 데이터셋·지표·규모(N clients 또는 #points, 모델 크기)를 적었다. 빈 실험 = `—`, 불확실 = `?`. 약어·E# 정의 → [[prior-work-taxonomy/README#범례 (legend)|README 범례]].

**E# 열**: `E1` exact/GT-SV 충실도 · `E2` 선별→다운스트림 · `E3` 악성/이상 검출 · `E4` 공정·보상 · `E5` stochastic robust·replication·ordering · `E6` 비용·확장성 · `E7` 집계 품질.

> metric·benchmark·GT 출처를 **유형별로 모은 카탈로그**와 **Flirds가 그중 무엇을 채택했는지**(§4)는 → [[metrics-and-benchmarks]]. 아래 표는 *연구별* 셀에 그 둘을 함께 담는다.

> **경향 요약**: CNN 트랙은 **federated valuation·detection이 가득**(E1/E3/E4/E7 중심), LLM 트랙은 **centralized 귀속·선별이 가득**(E1-LDS / E2-선별 / E6-확장성 중심). LLM 트랙의 federated 행은 valuation이 아니라 selection/quality/market(FedDQC·FedHDS·iPFL)뿐이고, **federated×LLM에서 E1(oracle 대비 충실도)을 한 연구는 [[flirds|Flirds]] 외에 관찰되지 않는다** — 두 표를 나란히 보면 빈칸이 드러난다.

---

## CNN / MLP 트랙

| Method | Fed | E1 충실도 | E2 선별 | E3 검출 | E4 공정 | E5 robust | E6 비용 | E7 집계 |
|---|---|---|---|---|---|---|---|---|
| **Flirds** | **F** | **estimator vs (b) Spearman ~1.0 in-regime; N∈{5,10}; 2차항 도움(0.96>0.92)** | (→LLM #7) | (→LLM) | — | seed-determinism(cudnn) | (b) 2ᴺ vs estimator 1 HVP/round | CNN 곱셈 가중 w∝n·s(arm) |
| IF (Koh-Liang) | C | — | — | mislabel self-influence 랭킹(상위=오라벨) | — | train-set 적대공격(1-img→test flip) | LiSSA params-linear | — |
| Data Shapley | C | (exact 정의 자체) | 저가치 drop 재학습→동등/우수(synth+의료img) | outlier/mislabel > LOO·leverage | 4공리 | — | TMC ~수천 점 | — |
| Data Banzhaf | C | — | reweight: Banzhaf>Beta>Shapley>LOO | bad-data: 동일 순위 | (efficiency 포기) | **SGD-run 랭킹 안정성 ≫; safety-margin 증명** | MSR log 샘플복잡도 | — |
| ACC-SGD-IE | C | Adult/20News/MNIST vs LOO-retrain(20seed): RMSE−17%, Kendall+7% | — | cleansing MNIST MCR 0.875→0.72% | — | cross-epoch fidelity(ep38 RMSE 86%↑) | Θ(Nn) HVP(vmap/critical-layer) | — |
| IF-Fragile ⚠crit | C | **Pearson/Spearman vs LOO-retrain: depth/width로 붕괴(ResNet-50 MNIST 0.24, ImageNet<0.15)** | — | (언급만) | — | weight-decay 필수·test-point 의존 | — | — |
| TRAK | C | **Datamodels LDS 일치 @CIFAR/ImageNet, 100–1000× 저비용** | — | — | — | multi-ckpt 평균(5–10) | 100–1000× ↓; ImageNet/CLIP/BERT 최초 | — |
| LoGra/Logix | C | LDS @MLP-FMNIST·ResNet-9-CIFAR(소규모) | — | — | — | — | (→LLM 대규모) | — |
| DVEmb | C | MNIST-MLP Spearman vs exact-LOO(IF 약함) | (→LLM) | — | — | trajectory-order 의존 | (→LLM) | — |
| FedSV | F | — | — | **noisy-label·backdoor client 저랭킹(MNIST/CIFAR, IID+non-IID) > Fed-LOO** | 공리(group rationality/fairness/additivity) | 초기라운드 inflation; rare non-IID 음수 SV | exact O(Tm²) / approx O(Tm log m) | — |
| ComFedSV | F | **noisy detection Spearman vs GT > FedSV(Synth/MNIST/FMNIST/CIFAR)** | — | noisy-label 100 clients(10@30%flip) Jaccard>FedSV | identical-client gap CDF가 FedSV 지배 | completion δ 미보장 | O(TN²logN) | — |
| GTG-Shapley | F | **경쟁 FL-Shapley 추정기보다 정확(image cls)** | — | — | — | guided-MC 초기 분산 | uniform-MC보다 util-eval ↓↓ | — |
| S-FedAvg (GoG) | F | — | FRCS: 음수-Shapley client prune | noisy/irrelevant/label-flip/malicious 강건 | — | "음수=나쁨" heuristic, 고노이즈서 flip 위험 | 매 라운드 Shapley = 비쌈 | **S-FedAvg Shapley-가중→test acc↑** |
| ShapleyFL | F | — | importance-sampling client 선택 | noisy/adversarial서 최종 acc↑ vs FedAvg+random(MNIST/FMNIST/CIFAR,Fed-ISIC) | — | adaptive-selection 수렴보장 | naive 2ⁿ보다 comm+compute↓ | — |
| SPACE | F | prototype-eval가 val-set 크기에 안정 | client reweight/select | — | contribution 기반 인센티브 | **val-set 크기 불변 랭킹** | single-round, GTG보다 comm/compute↓ | — |
| Ripple | F | (62× 속도 "comparable accuracy") | — | — | — | sample-level 4공리 보존 | **62× vs 선행 FL-Shapley** | — |
| FedIF | F | — | — | label+gradient noise 강건 vs AFedSV; PGD·극단노이즈 실패 | — | clean ≈ FedAvg(CIFAR/FMNIST,100,Dir α=1) | **agg ~0.2s/r vs AFedSV 70–92s(450×)** | adaptive weighted FedAvg |
| FedTSV | F | — | — | 20 malicious(label-shuffle) 분리 | **3그룹 분리 > LOO/CGSV** | acc 가장 안정 | client 1패스+서버 val 1패스/r | **truncated adaptive FedAvg > FedAvg/LOO/CGSV**(MLP-MNIST,CNN-CIFAR ResNet-20,100) |
| ShapFed | F | CSSV가 true val-acc Shapley 추종(synth, Fig3) | — | vulnerable서 AFedSV 동급/상회 | per-class per-client readout | — | last-layer cosine만→저렴 | **ShapFed-WA > FedAvg(class imbalance; CIFAR/X-ray/ISIC)** |
| Mavericks/FedEMD | F | (SV 오랭킹 *증명*, Fig2) | **FedEMD R@99 최속(MNIST/FMNIST/CIFAR/STL) ≥26.9%** | — | SV가 high-quantity·skewed under-credit(Prop3.1/3.2) | bias 라운드 의존(초기 심함) | O(K log(N/K)) | — |
| Shapley-Volatility ⚠crit | F | (size-기반 GT) | — | (Krum이 추정 최악) | per-client 불공정 큼·zero-sum 이동 | **집계전략만 바꿔도 보상 30–50% 이동, α-불변** | **>20,000 FL run, 8 aggregator** | 8전략 무승부 |
| FLDetector (det) | F | — | — | **poison/backdoor MNIST/CIFAR-ResNet20/FEMNIST, 28% malicious non-IID0.5, DACC 0.85–1.0 FNR~0, backdoor→~2%** | — | **IID-only(Thm1)**; adaptive 강건 | params-linear, client 0 | detect→remove→Median 회복 |
| FLTrust (det) | F | — | — | **poison/backdoor 6 datasets, 40–90% malicious(95% backdoor), test-err≤0.04** | — | Thm1 bound; root-분포 민감 | 서버 root fine-tune/r | **trust-weighted = 집계규칙** |
| FoolsGold (det) | F | — | — | **sybil/poison MNIST/VGGFace2/KDDCup/Amazon, A-99(990 sybil) 공격율≈0** | — | **IID서 honest 오탐(한계)** | — | adaptive-LR 가중(α→0) |
| FedCorr (det) | F | — | — | **label-noise CIFAR-10/100/Clothing1M ρ≤0.8; CIFAR-10 IID 90.6% vs FedAvg 72%** | — | non-IID 설계 대응; 동적참여 X(한계) | comm 1.3–1.9× 효율 | correct+relabel+FedAvg |
| STD-DAGMM (det) | F | — | — | **free-rider MNIST+2-layer MLP(IID+non-IID), 20/100 advanced-delta AUC≈0.96(r5)/0.91(r80)** | — | 비율↑→저하, threshold 의존 | — | — |
| DICE | D | (GT cascade 정의; exact-SV 비교 X) | collaborator 선택(제안·미벤치) | free-rider 검출(제안·미벤치) | 분산마켓 인센티브 | curvature 항 비쌈 | — | — |

> **위협 정의(공격, 행 아님)**: [[sources/how-to-backdoor-fl-bagdasaryan]] — CIFAR semantic backdoor, single-shot γ=n/η=100 → backdoor ~100% / main-task drop<1% / 20+라운드 지속(FLDetector·FLTrust의 매칭 위협). E3 baseline들의 공격 대상.

---

## LLM 트랙

| Method | Fed | E1 충실도 | E2 선별 | E3 검출 | E4 공정 | E5 robust | E6 비용 | E7 집계 |
|---|---|---|---|---|---|---|---|---|
| **Flirds** | **F** | **estimator=(a)valloss=(b)exact Spearman +1.000 @1B N=5(fp32,양 lr); 3B N=5 (a)vs(b)+0.900[estimator vs (b)+1.000]; cross-device N=100 α=.5 vs (b)-perround +1.000; +Kendall/Pearson** | **#7 clean 1B(양lr,3seed): flirds_topk val-loss≤randomK & ROUGE≥randomK; keep=[2,3,4]=clean** | **noisy AUROC 0.75 / free-rider 1.0 @1B N=5 lr1e-3(lr3e-3서 반전); poison: working backdoor에 Flirds-1st AUROC 0.0 EVADED([b]·loss-heur catch)** | (partial-participation 공정 #14) | noise-vs-OOD 분리 deferred(한계 명시); 2차 PGD 검증(#13) | **Flirds-1st~35s/Flirds~107s vs (b)·baseline~530s/Ripple~4515s = 5–15× 저렴(~42× vs Ripple)** | intervention arms flirds_w/sel → MMLU+Alpaca ROUGE |
| IRDS | C | — | **Pile ~16% 음수 Shapley, 제거→수렴 가속+최종 성능↑(GPT-2/Pythia-410M)** | — | copyright 귀속 case study | stage-dependent 기여 | **"정규학습만큼 빠름"(1–2 backward)** | — |
| DVEmb | C | (MNIST-MLP는 CNN트랙) | early+late window<half가 full 추종(>5× ↓) | — | — | trajectory-order 의존; T-무관 오차한계 | Pythia-410M ~170GB·peak 0.84 vs 63.6GB, >15× throughput | — |
| DataInf | C | **noisy GLUE/MRPC LoRA r=1 exact-IF와 Pearson ~0.64(LiSSA 0.45)** | — | **mislabel 검출 RoBERTa-large LoRA noisy-GLUE > LiSSA·Hessian-free** | — | rank↑서 저하 | O(nDL) closed-form, O(D) mem | — |
| LoGra/Logix | C | **LDS GPT-2/WikiText(EKFAC 약간 하회, TRAK 상회)** | — | — | — | — | **Llama3-8B+1B OWT A100: ~6500× throughput, 5× mem↓, 3.5TB 저장** | — |
| EK-FAC (Grosse) | C | **LiSSA(22M Transformer)·PBRF 대비 검증(§5.1)** | — | — | — | word-ordering 민감(영향≈0) | **52B로 확장(이전 ~300M의 2자릿수); TF-IDF+query batching** | — |
| LESS | C | — | **inst-tuning 5%/270K vs full: Llama2-7B MMLU 51.6/50.2·TyDiQA 54.0/56.2; Mistral은 full 상회; BM25/DSIR/RDS 상회; 7B→13B/Mistral 전이** | — | — | **N=4 warmup ckpt>N=1; off-shelf grad<random; SGD-warmup 악화** | single A100: warmup 6h+grad 48h/17.7GB; 선택<1min | — |
| MATES | C | locally-probed oracle Spearman 0.32; static infl-model<0.5(Fig5) | **25B-token 선별 9-task avg: 1B 47.5 vs random 46.4/DsDm 46.7; 410M 45.8 vs LESS 44.6; 2.3× 빠름** | — | — | dynamic refit load-bearing | oracle 2.5s/pt; 160k 14GPU-h; 전체 11.5% FLOPs | — |
| DsDm | C | (TRAK/LDS 의존) | **사전학습 선별: 125M가 10×-compute random 일치(SQuAD/Jeopardy/LAMBADA); 1.3B=1.8B Chinchilla@2× random; SemDeDup/DSIR 상회** | — | — | target-task 선택 중요(LAMBADA-only는 해침) | 2× compute multiplier; TRAK on 125M proxy | — |
| LoRIF | C | **LDS GPT2/WikiText Pareto > LoGra/TrackStar; OLMo-7B/Apertus-70B tail-patch 1.5–2.2×** | — | — | — | — | **20× 저장+query 속도; OLMo-7B 20.3×/22×; 70B 확장** | — |
| DPO-Shapley | C | (수렴서 exact; 충실도 연구 없음) | — | — | 음수 Shapley=undesirable source flag | 순차-DPO 가환성 수렴서만 exact | **exp→linear: n source = n DPO run** | — |
| Do-IF-work-on-LLMs ⚠crit | C | **3 GT-task top-1 Acc/Cover@c: IF 급락(backdoor DataInf 26% vs RepSim 100%)** | — | **harmful-data/class-attr/backdoor: RepSim≈100% ≫ IF(Llama2-7B/Mistral LoRA r=4)** | — | **convergence 불안정; param≠behavior(ASR ±90% w/o ‖Δθ‖ 차)** | single H100; LiSSA 10 iter | — |
| FedDQC ▣ | F | — | **synth 50%noise(IID+non-IID) > PPL/IFD/NUGGETS/DataInf; real Fed-WildChat 70% > all > full-FedAvg** | **noisy 필터; DataInf-선별<random on real FL(음성결과)** | — | descending hierarchical > random | **IRA ~1% train time, DataInf의 1/150** | (FedAvg 등과 호환, 불변) |
| FedHDS ▣ | F | — | **NI+Dolly-15K(1.3B&3B non-IID): full 대비 +10.72% Rouge-L(<1.5% 샘플); FedIT 대비 +19.5%(NI)** | — | — | subset이 local overfit 완화 | **최대 48.8× 속도(Turbo)**; comm 무시가능 | (downsample 위 FedAvg) |
| iPFL ▣ | F | — | — | (attacker 영향 bounded; 위협모델 제한) | **IR+IC+social-welfare 보장; per-participant 품질↑** | attacker 무이득(coalition 연기) | **billion-param LoRA 확장** | (personalized, 단일집계 아님) |

> **위협 정의(공격, 행 아님)**: [[sources/instructions-as-backdoors-xu]] — instruction-trigger backdoor, ~1% poison → Induced-Instruction avg ASR 95.36%(FLAN-T5 80M–11B, Llama2-7B/70B); 큰 모델일수록 취약. Flirds poison arm의 trigger 출처(약한 token-level "tq"만 차용). · Bagdasaryan(상동, Reddit LM single-target-word)도 LLM 측 위협.

---

## 두 표에서 읽히는 것 (담백하게)

- **E1(충실도)의 비대칭**: CNN 트랙은 exact-SV/GT 대비 E1이 흔하다(ComFedSV·GTG·FedSV·IF-Fragile·DataInf-on-small). LLM 트랙의 E1은 **전부 centralized**(LoGra·EK-FAC·DataInf·LoRIF의 LDS, MATES의 oracle-Spearman). **federated×LLM의 E1**은 [[flirds|Flirds]]의 (a)/(b) 듀얼 oracle뿐 — 이것이 빈칸을 채우는 지점.
- **E2(선별)는 LLM의 주력**: LESS·MATES·DsDm·FedHDS·DVEmb·IRDS 모두 "선별→다운스트림"으로 가치를 입증한다(LLM에선 exact-SV가 불가능에 가까워 E1 대신 E2로 우회). Flirds도 #7에서 E2를 통과(`random`이라는 hard bar 상회).
- **E3(검출)의 무대 차이**: CNN엔 전용 detector가 두텁다(FLDetector·FLTrust·FoolsGold·FedCorr·STD-DAGMM). LLM엔 client-level 검출 선례가 사실상 없어, Flirds가 포팅한 CNN-detector들이 첫 PEFT-scale 테스트가 된다([[sources/free-riders-fl-std-dagmm]] "first PEFT-scale", [[sources/feddqc]] DataInf 실세계 FL 실패).
- **E6(비용)의 단위 차이**: CNN-FL은 #util-eval·#round(2ⁿ 회피), LLM은 throughput·저장(TB)·FLOPs-%·wall-clock. Flirds는 양쪽에서 "1 HVP/round로 (b) 2ᴺ oracle 순위를 5–15× 싸게 재현"으로 보고.
- **E4/E7(공정·집계)는 federated 고유**: centralized엔 거의 없고(ADS·DPO-Shapley 정도), federated에 몰린다 — 단 E7(집계)로 평가하는 연구(FedIF·FedTSV·ShapFed·S-FedAvg)는 valuation-fidelity(E1)를 생략한다는 점에서 Flirds(E1 우선)와 목적이 다르다([[taxonomy#축 5 — 푸는 문제 (목적)|taxonomy 축5]]).
