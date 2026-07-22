# Flirds 논문 인용 선행연구 표

> **범위**: `paper/AAAI/references.bib`(31) + `paper/paper-ko.md` 본문·부록에서 이름이 등장하는 선행연구 + 실험 재현에 필요한 데이터셋·모델·지표 + 문헌 표준상 빠지면 안 되는 추가 인용.
> **근거**: 저자·venue는 `research-wiki/wiki/sources/*.md`의 Citation 절과 대조. 데이터셋·모델 ID는 `codes/flirds/data/llm.py`·`eval/mmlu.py`·`hf_pin.py`의 실제 로드 경로에서 확인.
> 제목은 **Google Scholar 검색용 full title**로 적었다(시스템명은 괄호 병기).

**상태 코드** — `인용` / `메타` 두 축.

| 코드 | 뜻 |
|---|---|
| `tex` | `main.tex` §1–§4에 `\citep` 됨 |
| `ko` | `paper-ko.md`에만 등장 — 영문 이관 시 배선 필요 |
| `–` | bib에만 존재, 미인용 |
| `NEW` | bib 엔트리 자체가 없음 — 신규 작성 필요 |
| `✔` | wiki 노트 대조 완료 |
| `⚠` | bib에 오류/`[CHECK]` 잔존 — wiki에 확정 메타 있음 |
| `❓` | 원문 재확인 필요 |

---

## 표 1 — 선행연구 (40건)

| 분류 | bib key | 저자 (연도) | 제목 (full) | Venue | 위치 | 역할 | 상태 |
|---|---|---|---|---|---|---|---|
| 토대 | `shapley1953` | Shapley (1953) | A Value for *n*-Person Games | *Contributions to the Theory of Games II*, Princeton Univ. Press, 307–317 | §1, §3.2 | 공정 분배 4공리 유일성 | tex ✔ |
| 토대 | `ghorbani2019datashapley` | Ghorbani & Zou (2019) | Data Shapley: Equitable Valuation of Data for Machine Learning | ICML 2019 | §1, §3.2 | 데이터 가치평가 축의 출발점 | tex ✔ |
| 토대 | `mcmahan2017fedavg` | McMahan, Moore, Ramage, Hampson, Agüera y Arcas (2017) | Communication-Efficient Learning of Deep Networks from Decentralized Data | AISTATS 2017 | §1, §3.1 | 식 (1) 집계 규칙 출처 | tex ✔ |
| 토대 | `wang2024irds` | Wang, Mittal, Song, Jia (2024) | Data Shapley in One Training Run (IRDS) | ICML 2024 / ICLR 2025 | §1, §2, §3.3, §4 | **직접 확장 대상** | tex ❓ |
| 토대 | `hu2022lora` | Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Chen (2022) | LoRA: Low-Rank Adaptation of Large Language Models | ICLR 2022 | §1, §D.1 | PEFT 좌표(§A.10 P8의 대상) | tex ✔ |
| 연합 SV 계보 | `wang2020fedsv` | Wang, Rausch, Zhang, Jia, Song (2020) | A Principled Approach to Data Valuation for Federated Learning (FedSV) | *Federated Learning: Privacy and Incentive*, Springer LNCS 12500 | §1, §2, §5 | 연합 Shapley의 기원; 재정규화 게임 대비군 | tex ✔ |
| 연합 SV 계보 | `liu2022gtg` | Liu, Chen, Yu, Liu, Cui (2022) | GTG-Shapley: Efficient and Accurate Participant Contribution Evaluation in Federated Learning | ACM TIST 13(4) 2022 | §1, §2, §5 | guided-truncation MC; free-rider $\phi \neq 0$ 반례 | tex ✔ |
| 연합 SV 계보 | `fan2022comfedsv` | Fan, Fang, Zhou, Pei, Friedlander, Liu, Zhang (2022) | Improving Fairness for Data Valuation in Horizontal Federated Learning (ComFedSV) | ICDE 2022 | §1, §2, §5 | 부분참여 utility 행렬 low-rank 완성 | tex ✔ |
| 연합 SV 계보 | `sun2023shapleyfl` | Sun, Li, Zhang, Xiong, Liu, Liu, Qin, Ren (2023) | ShapleyFL: Robust Federated Learning Based on Shapley Value | KDD 2023 | §1, §2, §5 | 정규화·이동평균 surrogate; $\beta$ 출처 | tex ✔ |
| 연합 SV 계보 | `space2024` | Chen, Chen, Wang, Chen (2023) | SPACE: Single-round Participant Amalgamation for Contribution Evaluation in Federated Learning | NeurIPS 2023 | §2 | **exact 참값 직접 채점의 유일 선례**($2^n$ 재학습, $N{\le}10$ CNN) | tex ⚠ |
| 공리 완화 | `fedif2025` | Tang, Drew, Zhou, Mamun (2025) | Lightweight and Robust Federated Data Valuation (FedIF) | arXiv:2509.25560 | §2, §5.1·5.3·5.6, B.3 | $\Delta w$ 1차 TracIn; removal 질적 이탈 유일 사례 | tex ⚠ |
| 공리 완화 | `fedtsv2026` | Kuznetsov & Wang (2026) | Fairness-Aware Federated Learning with Trajectory Shapley Value (FedTSV) | ECC 2026 (arXiv:2605.30336) | §2 | 궤적 Shapley → 적응 집계 | tex ⚠ |
| 공리 완화 | `shapfed2024` | Tastan, Fares, Aremu, Horváth, Nandakumar (2024) | Redefining Contributions: Shapley-Driven Federated Learning (ShapFed) | IJCAI 2024 | §2 | 클래스별 Shapley 집계 가중치 | tex ⚠ |
| 공리 완화 | `nagalapatti2021sfedavg` | Nagalapatti & Narayanam (2021) | Game of Gradients: Mitigating Irrelevant Clients in Federated Learning (S-FedAvg) | AAAI 2021 | §2 | Shapley 가중 FedAvg + 음수-$\phi$ pruning | tex ✔ |
| 비판 앵커 | `shapleyvolatility` | Geimer, Fiz, State (2025) | On the Volatility of Shapley-Based Contribution Metrics in Federated Learning | arXiv:2405.08044 (venue 없음) | §1 | 집계 전략만 바꿔도 보상 수십 % 출렁임 | tex ⚠ |
| 비판 앵커 | `mavericks2024` | Huang, Hong, Chen, Roos (2021) | Is Shapley Value Fair? Improving Client Selection for Mavericks in Federated Learning | arXiv:2106.10734 | §6 | maverick 과소평가를 고치지 않고 특성화 | ko ⚠ |
| 중앙 LLM 귀속 | `koh2017influence` | Koh & Liang (2017) | Understanding Black-box Predictions via Influence Functions | ICML 2017 | §2 | IF 계열의 뿌리 | tex ✔ |
| 중앙 LLM 귀속 | `grosse2023` | Grosse, Bae, Anil, Elhage, Tamkin, … Bowman (2023) | Studying Large Language Model Generalization with Influence Functions | arXiv:2308.03296 | §2 | LLM 규모 IF 상한 앵커(EK-FAC @52B) | tex ⚠ |
| 중앙 LLM 귀속 | `kwon2024datainf` | Kwon, Wu, Wu, Zou (2024) | DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models | ICLR 2024 | §1, §2 | LoRA 폐형 $H^{-1}$; 중앙집중·샘플단위 대비축 | tex ✔ |
| 중앙 LLM 귀속 | `park2023trak` | Park, Georgiev, Ilyas, Leclerc, Madry (2023) | TRAK: Attributing Model Behavior at Scale | ICML 2023 | §2 | eNTK 선형화 + 랜덤 사영 | tex ✔ |
| 중앙 LLM 귀속 | `choe2024logra` | Choe, Ahn, Bae, Zhao, Kang, … Xing (2024) | What is Your Data Worth to GPT? LLM-Scale Data Valuation with Influence Functions (LoGra) | arXiv:2405.13954 | §2 | LLM-scale IF 사영 | tex ⚠ |
| 중앙 LLM 귀속 | `xia2024less` | Xia, Malladi, Gururangan, Arora, Chen (2024) | LESS: Selecting Influential Data for Targeted Instruction Tuning | ICML 2024 | §1, §2 | **가장 가까운 중앙집중 비교자** | tex ✔ |
| 중앙 LLM 귀속 | `pruthi2020tracin` | Pruthi, Liu, Kale, Sundararajan (2020) | Estimating Training Data Influence by Tracing Gradient Descent (TracIn) | NeurIPS 2020 | §2 | LESS·FedIF가 계승한 궤적 influence | tex ✔ |
| 중앙 LLM 귀속 | `yu2024mates` | Yu, Das, Xiong (2024) | MATES: Model-Aware Data Selection for Efficient Pretraining with Data Influence Models | NeurIPS 2024 | §2 | 증류 소형모델 데이터 선별 | tex ✔ |
| 중앙 LLM 귀속 | `engstrom2024dsdm` | Engstrom, Feldmann, Mądry (2024) | DsDm: Model-Aware Dataset Selection with Datamodels | ICML 2024 | §2 | 선형 datamodel 선별 | tex ✔ |
| 중앙 LLM 귀속 | `dvemb2024` | Wang, Song, Zou, Mittal, Jia (2024) | Capturing the Temporal Dependence of Training Data Influence (Data Value Embedding) | arXiv:2412.09538 | *미인용* | in-run 최근접 형제(IRDS 저자군) | – ⚠ |
| 중앙 LLM 귀속 | `basu2021fragile` | Basu, Pope, Feizi (2021) | Influence Functions in Deep Learning Are Fragile | ICLR 2021 | *미인용* | 1차 근사 취약성 caveat(§5.2·§6 보강) | – ✔ |
| 중앙 LLM 귀속 | `doifwork2025` | Li, Zhao, Li, Sun (2025) | Do Influence Functions Work on Large Language Models? | EMNLP 2025 Findings (arXiv:2409.19998) | *미인용* | LLM IF 음성결과 — $H^{-1}$ 회피의 대비 | – ⚠ |
| 탐지·강건 | `zhang2022fldetector` | Zhang, Cao, Jia, Gong (2022) | FLDetector: Defending Federated Learning Against Model Poisoning Attacks via Detecting Malicious Clients | KDD 2022 | §2, §5.6, §D.4 | 탐지 비교군(L-BFGS 예측잔차) | tex ✔ |
| 탐지·강건 | `cao2021fltrust` | Cao, Fang, Liu, Gong (2021) | FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping | NDSS 2021 | §2, §5.6, §D.4, E.2 | 서버 검증-gradient cosine 비교군 | tex ✔ |
| 탐지·강건 | `lin2019freerider` | Lin, Du, Liu (2019) | Free-riders in Federated Learning: Attacks and Defenses (STD-DAGMM) | arXiv:1911.12560 | §2, §5.6, §D.3–4 | free-rider 위협 정의 + 탐지 비교군 | tex ✔ |
| 탐지·강건 | `feddqc2024` | Du, Ye, Yuchi, Zhao, Qu, Wang, Chen (2025) | FedDQC: Data Quality Control in Federated Instruction-tuning of Large Language Models | ACL 2025 Findings | §2, §5.6, §D.4, E.2 | LLM 연합 품질관리(IRA) 탐지 비교군 | tex ⚠ |
| 탐지·강건 | `blanchard2017krum` | Blanchard, El Mhamdi, Guerraoui, Stainer (2017) | Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent (Krum) | NeurIPS 2017 | §2 | 강건 집계 = 대체재 아님 논거 | tex ✔ |
| 탐지·강건 | `yin2018byzantine` | Yin, Chen, Ramchandran, Bartlett (2018) | Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates | ICML 2018 | §2 | 좌표별 median / trimmed-mean | tex ✔ |
| 무대·위협 | `ye2024openfedllm` | Ye, Wang, Chai, Li, Li, Xu, Du, Wang, Chen (2024) | OpenFedLLM: Training Large Language Models on Decentralized Private Data via Federated Learning | KDD 2024 | §D.2, Track D | 무대·템플릿 verbatim 출처 | ko ✔ |
| 무대·위협 | `bagdasaryan2020` | Bagdasaryan, Veit, Hua, Estrin, Shmatikov (2020) | How To Backdoor Federated Learning | AISTATS 2020 | §6 | 표적 공격 = 스코프 경계 | ko ✔ |
| 무대·위협 | `xu2024backdoor` | Xu, Ma, Wang, Xiao, Chen (2024) | Instructions as Backdoors: Backdoor Vulnerabilities of Instruction Tuning for Large Language Models | NAACL 2024, 3111–3126 | §6 | instruction-tuning backdoor 위협 | ko ✔ |
| 무대·위협 | `fedhds2025` | Qin, Wu, He, Deng (2025) | Federated Data-Efficient Instruction Tuning for Large Language Models (FedHDS) | ACL 2025 Findings | *미인용* | 연합 LLM 데이터 선별(valuation 아님) | – ⚠ |
| 결정 보류 | `wang2023banzhaf` | Wang & Jia (2023) | Data Banzhaf: A Robust Data Valuation Framework for Machine Learning | AISTATS 2023 | *미인용* | 비교군 제외(07-22) — semivalue 대안으로 §2 1줄 권장 | – ✔ |
| 결정 보류 | `ripple2026` | Zeng, Tian, Wang, Lu, Xiao, Xu (2026) | Ripple Shapley: Data Influence Attribution in One Federated Training Run | AAAI 2026 | *미인용* | **in-run × federated 최근접 동시대 연구**(단, 샘플 단위) | – ⚠ |

---

## 표 2 — 데이터셋·모델·지표 (전부 bib 엔트리 없음 = `NEW`)

부록 D가 이름으로만 부르는 것들. HF ID는 `codes/flirds/data/llm.py`·`eval/mmlu.py` 실제 로드 경로.

| 종류 | 코드상 ID | 인용할 논문 (full title) | 저자 (연도) | Venue | 위치 |
|---|---|---|---|---|---|
| 모델 | `meta-llama/Llama-3.2-1B/3B-Instruct` | The Llama 3 Herd of Models | Grattafiori/Dubey et al., Llama Team (2024) | arXiv:2407.21783 | §5.1, §D.1 |
| 모델 | `meta-llama/Llama-2-7b-hf` | Llama 2: Open Foundation and Fine-Tuned Chat Models | Touvron et al. (2023) | arXiv:2307.09288 | §5.1, §D.1 |
| 데이터 (IID) | `vicgalle/alpaca-gpt4` | Instruction Tuning with GPT-4 | Peng, Li, He, Galley, Gao (2023) | arXiv:2304.03277 | §D.2 |
| 데이터 (IID) | ″ (템플릿·명령셋 원본) | Stanford Alpaca: An Instruction-following LLaMA Model | Taori, Gulrajani, Zhang, Dubois, Li, Guestrin, Liang, Hashimoto (2023) | GitHub (기술보고서) | §D.2 |
| 데이터 (medical) | `medalpaca/medical_meadow_medical_flashcards` | MedAlpaca — An Open-Source Collection of Medical Conversational AI Models and Training Data | Han, Adams, Papaioannou, et al. (2023) | arXiv:2304.08247 | §D.2 |
| 데이터 (finance) | `LLukas22/fiqa` | WWW'18 Open Challenge: Financial Opinion Mining and Question Answering (FiQA) | Maia, Handschuh, Freitas, Davis, McDermott, Zarrouk, Balahur (2018) | WWW '18 Companion, 1941–1942 | §D.2 |
| 데이터 (math) | `deepmind/aqua_rat` | Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems (AQuA-RAT) | Ling, Yogatama, Dyer, Blunsom (2017) | ACL 2017 | §D.2 |
| 데이터 (general) | `databricks/databricks-dolly-15k` | Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM | Conover et al., Databricks (2023) | 기술 블로그 (논문 없음) | §D.2 |
| 데이터 (legal) | `ibunescu/qa_legal_dataset_train` | **출처 논문 불명** | — | HF 커뮤니티 데이터셋 | §D.2 |
| 데이터 (selection) | `openai/gsm8k` | Training Verifiers to Solve Math Word Problems (GSM8K) | Cobbe et al. (2021) | arXiv:2110.14168 | selection 축 |
| 벤치마크 | `cais/mmlu` | Measuring Massive Multitask Language Understanding (MMLU) | Hendrycks, Burns, Basart, Zou, Mazeika, Song, Steinhardt (2021) | ICLR 2021 | §E.4, Track D |
| 지표 | `eval/metrics.py: rouge_l` | ROUGE: A Package for Automatic Evaluation of Summaries | Lin (2004) | Text Summarization Branches Out (ACL 워크숍), 74–81 | §E.4, Track D |
| 데이터+모델 (CNN) | MNIST / LeNet-5 | Gradient-Based Learning Applied to Document Recognition | LeCun, Bottou, Bengio, Haffner (1998) | Proc. IEEE 86(11), 2278–2324 | §5.1, §D.1 |
| 데이터 (CNN) | CIFAR-10 | Learning Multiple Layers of Features from Tiny Images | Krizhevsky (2009) | Tech. Report, Univ. of Toronto | §5.1, §D.1 |

---

## 표 3 — 놓친 필수/권장 인용 (선행연구가 표준으로 다는 것들)

FedSV·GTG·ShapleyFL·IRDS·LESS의 인용 관행과 대조해 **우리 논문이 하는 주장에 직접 필요한데 빠진 것**만 골랐다.

| 우선순위 | 논문 (full title) | 저자 (연도) | Venue | 왜 필수인가 |
|---|---|---|---|---|
| **필수** | Fast Exact Multiplication by the Hessian | Pearlmutter (1994) | Neural Computation 6(1), 147–160 | **HVP가 우리 방법의 핵심 연산 단위**(§4.2 "라운드당 HVP 한 번", 코드는 `jvp∘grad`). 이 원전 없이 HVP를 쓰는 IF/Shapley 논문은 없다 |
| **필수** | Practical Secure Aggregation for Privacy-Preserving Machine Learning | Bonawitz, Ivanov, Kreuter, Marcedone, McMahan, Patel, Ramage, Segal, Seth (2017) | ACM CCS 2017 | §6이 "secure aggregation과 비호환"을 한계로 **명시**하는데 정작 그 대상이 인용돼 있지 않다 |
| **필수** | Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification | Hsu, Qi, Brown (2019) | arXiv:1909.06335 | §D.2의 **Dirichlet($\alpha$) 파티션**이 이 논문의 프로토콜이다. cross-device 무대 전체가 여기 근거 |
| **필수** | Advances and Open Problems in Federated Learning | Kairouz, McMahan, et al. (2021) | Found. & Trends in ML 14(1–2) | 논문 전반이 쓰는 **cross-silo / cross-device** 구분의 표준 출처. §1 "실제 FL 세팅의 조건" 열거의 근거 |
| **필수** | Towards Efficient Data Valuation Based on the Shapley Value | Jia, Dao, Wang, Hubis, Hynes, Gürel, Li, Zhang, Song, Spanos (2019) | AISTATS 2019 | Data Shapley의 표준 동반 인용(효율적 SV 추정 계보의 출발). §1 "지수적 비용 우회" 서사의 정본 |
| **필수** | Polynomial Calculation of the Shapley Value Based on Sampling | Castro, Gómez, Tejada (2009) | Computers & Operations Research 36(5), 1726–1730 | §1·§2가 반복하는 "**permutation Monte Carlo 표본 추출**"의 원전. FedSV가 계승한 대상 |
| 권장 | Datamodels: Predicting Predictions from Training Data | Ilyas, Park, Engstrom, Leclerc, Mądry (2022) | ICML 2022 | §2에서 TRAK·DsDm을 인용하면서 그 둘이 딛고 선 datamodel 원전이 빠져 있다 |
| 권장 | Beta Shapley: A Unified and Noise-Reduced Data Valuation Framework for Machine Learning | Kwon & Zou (2022) | AISTATS 2022 | Banzhaf와 함께 semivalue 축을 완성. §4.1 "게임 선택은 하나의 선택" 논거를 넓힘 |
| 권장 | Deep Leakage from Gradients | Zhu, Liu, Han (2019) | NeurIPS 2019 | §6 프라이버시 한계(서버가 개별 $\Delta w$를 본다)의 위험이 실재함을 뒷받침 |
| 권장 | Bootstrap Methods: Another Look at the Jackknife | Efron (1979) | Annals of Statistics 7(1), 1–26 | §5.4·§E.3의 bootstrap 재표집 프로토콜 근거 |
| 권장 | Transformers: State-of-the-Art Natural Language Processing | Wolf et al. (2020) | EMNLP 2020 (System Demos) | 재현성 부록의 소프트웨어 스택 인용(PEFT·TRL은 GitHub 각주로 충분) |
| 선택 | Federated Learning with Non-IID Data | Zhao, Li, Lai, Suda, Civin, Chandra (2018) | arXiv:1806.00582 | non-IID 성능 저하의 표준 인용. §5.4 "신호는 클라 간 실제 차이가 만든다" 논의와 접점 |

---

## 조치 필요 (우선순위 순)

### 1. Ripple 미인용 — 가장 시급
`ripple2026`(Zeng et al., **AAAI 2026**, *Ripple Shapley: Data Influence Attribution in One Federated Training Run*)은 제목부터 "one federated training run"이다. 07-22의 **비교군 제외는 baseline 결정이지 인용 결정이 아니다.** 지금 `main.tex`·`paper-ko` 어디에도 없어 "가장 가까운 선행 누락"으로 읽힐 위험이 크다.
→ §2에 1–2문장: Ripple = **샘플 단위** in-run 귀속 / 우리 = **클라이언트 단위** + exact 참값 대비 채점. 기여 1의 "클라이언트-수준 LLM 규모 최초" 주장이 이 대비 없이는 방어되지 않는다.

### 2. 표 2·표 3이 통째로 미작성 (26 entries `NEW`)
데이터셋·모델·지표 14건 + 필수 6건이 bib에 아예 없다. **Pearlmutter(HVP)·Bonawitz(secure agg)·Hsu(Dirichlet)** 셋은 각각 §4.2·§6·§D.2가 직접 의존하는 대상이라 리뷰어가 바로 짚는다.

### 3. `mavericks2024` 엔트리가 서로 다른 두 논문을 섞고 있음
bib은 *"Mavericks in federated learning: Contribution-based client selection under distribution shift"* + `arXiv:2405.12590`인데, §6이 실제 근거하는 wiki 노트는 **Huang et al. (2021), *Is Shapley Value Fair? …*, arXiv:2106.10734**다. taxonomy README도 2405.12590을 "*Rewarding the Rare*, 별개 논문"으로 명시한다. → 교체.

### 4. `space2024` 메타 오류
2024 → **NeurIPS 2023**, 제목에 *Single-round* 누락. "exact 참값 직접 채점의 유일 선례"로 무게를 싣는 인용이라 정확해야 한다.

### 5. `[CHECK]` 채우기 — wiki에 확정 메타 전부 있음
`ripple2026` `fedif2025` `fedtsv2026` `shapfed2024` `space2024` `dvemb2024` `doifwork2025` `feddqc2024` `fedhds2025` `shapleyvolatility` + `grosse2023`·`choe2024logra`의 `others`.
- `shapfed2024` 실제 제목 = *Redefining Contributions: Shapley-Driven Federated Learning*
- `fedif2025` 실제 제목 = *Lightweight and Robust Federated Data Valuation*
- `feddqc2024` → **ACL 2025 Findings**(key 연도와 불일치), `fedhds2025` → ACL 2025 Findings(venue verify)
- `shapleyvolatility` → 공식 venue 없음(arXiv only) — preprint 인용임을 명시
- `liu2022gtg` `fan2022comfedsv` `sun2023shapleyfl` `nagalapatti2021sfedavg` `xu2024backdoor`의 `[CHECK]`는 **대조 결과 일치** — 주석만 삭제

### 6. `wang2024irds` venue 확정
bib은 `booktitle=ICLR, year=2025`, wiki 노트는 "ICML 2024; ICLR 2025 Outstanding Paper Runner-up". 논문 전체가 이 한 편의 확장이므로 원문으로 확정.

### 7. legal 데이터셋 출처 불명
`ibunescu/qa_legal_dataset_train`에 대응하는 논문을 찾지 못했다. cross-silo 5-도메인 중 하나이므로 (a) 출처 논문을 찾아 인용하거나 (b) HF URL 각주로 처리할지 결정 필요.

### 8. 영문 이관 대기 (`ko` 4건)
`ye2024openfedllm` `bagdasaryan2020` `xu2024backdoor` `mavericks2024`.

### 9. 미인용 3건 처분
`dvemb2024`·`basu2021fragile`·`doifwork2025` — 인용하면 §2·§5.2·§6이 단단해지고, 안 쓸 거면 bib에서 제거해 `[CHECK]` 부채를 줄이는 편이 낫다.
