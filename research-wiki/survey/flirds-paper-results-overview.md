---
type: survey
title: Flirds 논문-수록 실험 결과 overview (paper §5 미러)
created: 2026-07-23
updated: 2026-07-23
tags: [flirds, paper, results, dashboard]
---

# Flirds 논문 결과 한눈에 보기 (paper §5 순서)

> **무엇**: 논문 *"Measure First: Federated LLM 파인튜닝에서 클라이언트-수준 데이터 가치평가의 Exact-Oracle Fidelity"* 에 **넣기로 확정된 실험만**, **paper-ko §5 순서 그대로** 모은 대시보드. 미완 실험은 **⬚** + 채울 소스 경로.
> **자립형(self-contained)**: 모든 수치는 `runs/<track>/…`의 rundir·analysis CSV에서 직접 소싱한다(파생 카탈로그에 의존하지 않음). 전량(미수록 포함) 카탈로그는 별도 문서 `flirds-experiment-results-overview.md`가 담당하며, 이 페이지는 그 문서와 **독립**이다.
> **스코프 규약**: 미수록 실험(poison·Banzhaf·Ripple·Fed-LOO·수렴축·std20-vs(b)·B축 신호실재성·3B/7B·std50k5 selection·Track G 게이팅 표 등)은 **넣지 않는다**. 위협축 = noise·free-rider·label-flip류만.
> **정본 순서 = `paper/workplan/00-INDEX.md` §0** (구조·수록/제외 결정) + `T1-paper-section5.md`(표 스펙).

**마커**: ● 실측 값 기입 · ◐ 부분/1-seed(정본 아님) · ⬚ 미실행(경로만) · – 해당없음. **비교군 9종** 위계 = same-game(Flirds·Flirds-1st·loss-heur) ↔ cross-game(GTG·FedSV·ComFedSV·ShapleyFL·FedIF). fidelity는 **same-game만 vs (b)**(본문·부록 공통 — cross-game은 in-run 오라클과 다른 게임이라 vs (b) 미채점), vs (a)는 전 방법(방법-중립 참값). seed = 3-seed mean±std(std=ddof0), 예외는 ◐ 명시.

---

## §5.1 실험 세팅 (작성 가능)

> 무대는 **질문 위계(1차 fidelity → 2차① downstream → 2차③ detection)별로 목적이 다르고**, 주무대 쌍(LLM `R4` · CNN `C2`)이 세 목적을 모두 잇는다. **오염 축은 주무대끼리 오염-클래스로 통일**((C)), sub 무대는 목적별로 다르다.

### (A) 무대 상세 — 논문 역할별

| 무대 | 논문 역할 | 모델 | N · 참여 · R | oracle · 심판 | 상태 |
|---|---|---|---|---|---|
| **R4 gsm50k5** (주-LLM) | fidelity(§5.2 L2) · **downstream**(§5.3) · detection(§5.4) | Llama-3.2-1B-Instruct · LoRA r16/α32 | 50 · 5/50 · 200 | (b) per-round[L2 ⬚] · **EM** 1,119 | ◐ seed0(정본 L1 ⬚) |
| **C2 캠페인** (주-CNN) | fidelity(§5.2 c2fid) · **downstream**(§5.3) · detection(§5.4) | FedSVCNN(cifar)/LeNet5(fmnist) · 전체학습 | 100 · 10/100 · 120 | (b) per-round · **acc** | ● (dir1 restack 확인) |
| anchor5 | (a) 특성화 **폴백 참조**(§5.2; 보류 2026-07-24) | 1B/3B/7B · LoRA | 5 · full · 30 | (a)+(b) 2⁵ · val-loss | ● (1B (a)) |
| silo5 (a)-leg | retrain-(a) **주 무대**(§5.2; 유일 (a)) | 1B | 5 · full · 10 | (a)+(b) 2⁵ · val-loss | ⬚ (L8) |
| ~~gsm5~~ | ⏸ **보류(2026-07-24; anchor5와 IID 중복)** | 1B | 5 · full · 30 | (a)+(b) dual | 보류 (코드 존치) |
| silo5 | fidelity (a)-leg(§5.2) · removal(§5.6③) | 1B | 5 · full · 10 | (b) 2⁵ · val-loss | ● / (a)-leg ⬚ |
| CNN C1 | fidelity sub · 듀얼오라클(§5.2) · removal(§5.6③) | LeNet5(mnist)/FedSVCNN(cifar) | 10 · full · 10 | (a)+(b) 2¹⁰ | ● |
| device100 · Scale 100/100 | 비용·규모 보조(부록 E) | 1B / CNN | 100 · (10/100 또는 full) · 30/120 | (b) anchor · AUROC/acc | ● |

### (B) 데이터셋

| 데이터셋 | 과제 (모델) | non-IID/분배 축 | 쓰는 무대 | 규모 |
|---|---|---|---|---|
| **GSM8K** | 초등수학 word-problem · 생성 · **EM** (1B-Instruct LoRA) | IID(149문항/클라) | R4(N=50) · gsm5(N=5) | train 7,473 · **test 1,119 EM** · val 200 |
| **Alpaca-GPT4** | 범용 instruction · 생성 · val-loss (1B/3B/7B LoRA) | IID 균등 shard | anchor5(N=5) · std20(N=20) | 20k · val 200 / test 1000 |
| **5-도메인 silo** | 의료·법률·금융·수리·일반 instruction (1B LoRA) | **cross-silo**(1 도메인 = 1 클라) | silo5(N=5) | 도메인당 train 200 / val 20 / test 40 |
| **CIFAR-10** | 이미지 분류 (FedSVCNN · 전체학습) | iid · **Dir dir1(α=1)** · **2-shard**(label-skew) · **qskew**(quantity) | C2 캠페인(N=100) · C1(N=10) | C1 val 2000 / test 8000 |
| **MNIST** | 숫자 분류 (LeNet5 · 전체학습) | 5 시나리오(GTG-Shapley) | C1(N=10) | val 2000 / test 8000 |
| **Fashion-MNIST** | 의류 분류 (LeNet5) | iid · dir1 | C2 캠페인(N=100) | — |

> silo5 도메인 원천: medical=medical_meadow_flashcards · legal=ibunescu legal-QA · finance=FiQA · math=AQUA-RAT · general=Dolly(전부 free-form instruction→response, size 균등=B1 통제). C1 5-시나리오 = iid·label_skew·quantity_skew·label_flip·feature_noise(GTG-Shapley 무대 이식). **출처**: `codes/flirds/data/llm.py` · `codes/flirds/fl/partition.py` · `codes/flirds/models/cnn.py`.

### (C) 오염/위협 축 — 주무대 통일 규약

> **주무대 쌍(R4·C2)은 오염 *클래스*를 통일한다**: 둘 다 {clean · 라벨-오염 · free-rider}를 오염 클라 **~40%** 로 주입, 클래스는 1:1 대응. 라벨-오염의 구현 이름만 모달리티별로 다르다(생성 EM엔 "라벨 뒤집기"가 없어 **answer-swap** = 분류의 label-flip).

| 오염 클래스 | 주-LLM `R4` | 주-CNN `C2` | 대응 · 비고 |
|---|---|---|---|
| (통제) clean | clean | clean | 오발화 대조 |
| **라벨-오염** | answer-swap@0.7 (단일 dose) · **per-client 강도변조(strmain류) ⬚ 계획** | label-flip@{.15,.35,.70} + **strmain** | 같은 클래스. **strmain** = FedCorr 기본 draw = 오염 클라마다 flip rate $\sim U(0.5,1)$ = **per-client 강도 변조**(고정-dose 셀은 전원 한 rate). **R4 strmain-dose = 신규 ⬚**(REMAINING §1.6a L10) |
| **free-rider** | frzero · **frrand ⬚ 계획** | fr(zero) + frrand | zero/random-update free-rider. **R4 frrand = 신규 ⬚**(REMAINING §1.6a L9; CNN은 flirds leg 존재) |
| **grad-noise** | ✗ (무대 미성립) | gn | LLM LoRA 기하선 등방 노이즈 응답이 gradient-방향 대비 수십분의 1 → **LLM 배제**(부록 B); CNN에선 **2차항 판별 핵심 셀**(§5.6①) |

> **설계 노트 — 비대칭의 성격(필연 vs 선택; 07-23 Yonghee 지적 반영, 초판 "grid-depth=비용" 프레이밍 정정)**:
> - **필연 1건 = grad-noise만**: LLM 모달리티상 불가(부록 B). 원리적 배제는 이것뿐.
> - **나머지는 전부 설계 선택**(비용·모달리티 필연 아님): ① **frrand**는 LLM에도 값싸게 넣을 수 있다 — 현재 R4가 free-rider 대표로 frzero만 쓰는 건 선택. ② **per-client 강도 변조**(CNN이 쓰는 strmain $U(0.5,1)$ / C1 사다리 `_pair_ladder` 0–20%)는 오염 dose를 클라마다 달리해 **기여도 스프레드를 만드는 fidelity 도구**이고 **계산량과 무관** — 현 LLM 무대(R4·anchor5·silo5)는 전원 균일-강도라 이 도구를 안 쓴다. ③ 고정-dose를 여러 셀 스윕하는 **grid-depth**만 CNN이 full 2ᴺ이라 값싸서 촘촘한 부분.
> - ⟹ **결정(07-23 Yonghee): R4에 frrand + strmain류 per-client 변조를 추가**한다 — 위 표에 ⬚ 축으로 등재, 실행 큐 = **REMAINING §1.6a**(L9 frrand · L10 strmain-dose; 종전 "R4 frrand 재제안 금지"를 번복). 값싸고 모달리티 제약 없음(원리적 배제는 grad-noise 하나뿐). fidelity 변별에도 이롭다(strmain F-4 dose-해상도의 LLM 대응 = Flirds ≳ 1st 기대). **단 caveat**: LLM fidelity 포화의 주원인은 **near-additive 레짐**(전 방법 ≈ exact라 순위 +1.000 포화)이라 dose 변조가 이를 *완전히* 깨진 못할 수 있음(부분 개선 기대).
> - **오염 집합 추출**(위협별 규약, 부록 B): label-flip = FedCorr $(\rho,\tau)$ 베르누이 → 오염 클라 **수가 시드마다 변동**(표엔 명목 $\rho$ 대신 **실현 수**; rate $\tau\sim U(0.5,1)$, 고정-dose 셀은 $\{.15,.35,.70\}$) · update-level(free-rider·grad-noise) = 정확 $\lfloor\rho N\rceil$ = 40% 비복원. **R4는 오염 클라 0–19 고정**(40%). 한 시드는 데이터·분배·dose에 걸쳐 같은 오염 집합.
> - **sub 무대 위협이 다른 건 목적 차이**(통일 규약은 *주무대끼리*만): silo5 = 탐지·removal(noisy·frrand·frzero) **+ (a)-oracle 주 무대(clean·noisy·frzero; 2026-07-24 유일 (a))** · CNN C1 = GTG-Shapley 5-시나리오 fidelity(iid·label_skew·quantity_skew·label_flip·feature_noise; 오염 사다리는 label_flip·feature_noise만) · anchor5 = (a)-oracle **폴백 참조**(gsm5와 함께 보류).

### (D) 비교군 · 프로토콜 · 지표

- **비교군 9종**: Flirds / Flirds-1st / loss-heur(same-game) · GTG / FedSV / ComFedSV / ShapleyFL(β=0.3) / FedIF(cross-game). 제외 각주 = Banzhaf·Ripple·poison·Fed-LOO. **전용 탐지기 4종**(FLDetector/FLTrust/STD-DAGMM/FedDQC)은 §5.4.
- **고정-궤적 채점**: 한 셀 안 모든 방법이 **같은 동결 궤적·같은 손실 구현**을 소비. 로컬 = plain SGD mom=0·상수 lr. **공정성**: zero-semantics도 채점 대상, 셀별 튜닝 금지(P1 τ=0 parameter-free).
- **지표**: 순위 Spearman/Kendall · 값 Pearson(본문) · 거리 3종(부록 C) · 탐지 AUROC · 절대 EM/acc · recovery. **target self-stability**(exact (b) 자기 cross-seed 일치도) 병기. seed = 3-seed mean±std(ddof0), 예외는 ◐.
- **출처**: `runs/track_h/rundirs_llm/gsm50k5_*`(주-LLM) · `runs/track_h/rundirs_cnn/`(주-CNN) · `runs/track_c/{c1,c2fid}` · `runs/track_d/rundirs/1B_anchor5_*` · `runs/phase2_matrix/rundirs/1B_silo5_*`(sub).

---

## §5.2 Fidelity (1차 핵심)

### (메인) same-game vs (b) exact in-run oracle — ⬚

주무대(c2fid + R4-L2)에 exact (b) per-round oracle을 붙여 same-game 3종을 채점. **현 fidelity 표는 대부분 포화(≈1.000)**라 변별용으로 이 두 셀을 신설.

| 무대                                   | Flirds Sp/Pe | Flirds-1st Sp/Pe | loss-heur Sp/Pe | 상태                                                    |
| ------------------------------------ | ------------ | ---------------- | --------------- | ----------------------------------------------------- |
| c2fid (CNN C2 twin, cifar10/dir1) — **전 시나리오 평균** | 0.974/0.988 | 0.606/0.506 | 0.923/0.947 | ● 3-seed(seed0–2·8시나리오) |
| └ clean | 0.984/0.990 | 0.413/0.294 | 0.834/0.887 | ● 3-seed |
| └ free-rider(frzero) | 0.984/0.997 | 0.511/0.467 | 0.877/0.932 | ● 3-seed |
| └ frrand | 0.994/0.997 | 0.537/0.466 | 0.879/0.932 | ● 3-seed |
| └ grad-noise | 0.847/0.937 | 0.218/−0.051 | 0.986/0.972 | ● 3-seed |
| └ label-flip@0.15 | 0.989/0.990 | 0.454/0.300 | 0.880/0.906 | ● 3-seed |
| └ label-flip@0.35 | 0.995/0.995 | 0.772/0.678 | 0.945/0.961 | ● 3-seed |
| └ label-flip@0.70 | 0.998/0.999 | 0.971/0.945 | 0.991/0.992 | ● 3-seed |
| └ label-flip strmain | 0.999/0.998 | 0.974/0.951 | 0.991/0.992 | ● 3-seed |
| R4-L2 (LLM gsm50k5, (b) per-round)   | ⬚            | ⬚                | ⬚               | 채움 = `runs/phase2_matrix/rundirs/1B_gsm50k5_*` (L2 큐) |

> 각주(c2fid ● 3-seed·cifar10/dir1, seed0–2): **clean 칸은 신호-부재 레짐**(오발화 대조용, fidelity 해석 금지) · std 범위(Sp): Flirds ≤.03 · loss-heur ≤.04 · **Flirds-1st ≤.10**(lf@.15/.35서 큼) · **strmain 셀** Sp Flirds 0.999 / 1st 0.974 / loss-heur 0.991(전 방법 고포화) · **frrand 셀** Flirds 0.994 / 1st 0.537 vs **renorm 붕괴**(GTG 0.019 · FedSV 0.009 · ShapleyFL −0.040) — same-game 3열엔 안 보이나 cross-game 변별 큼 · **F-4(dose 해상도)**: spearman_vs_rate Flirds **0.858 ≈ (b) 0.857**(corrupt-only Flirds 0.519 / (b) 0.515 = 오라클 정확추종) 이나 Flirds-1st **0.877/0.659 ≥** Flirds → 사전등록 "Flirds ≳ 1st" **MISS 유지**(3-seed; 2차항의 dose-우위는 이 셀서 미확인; Flirds 강점은 (b) 정확추종) · **cifar10/iid도 3-seed 완성**(Flirds 평균 Sp 0.982) · **R4 strmain류 per-client dose(⬚ L10)** = 이 dose-변별의 LLM 대응(REMAINING §1.6a).
> ![[flirds-paper-results-overview-figs/f3_main_pair_heatmap.png]] ⬚ *(F3 = 메인 쌍 heatmap — 데이터 착지 시 `make_figures.py`에 f3 함수 추가·생성; 현 미구현)*

### (sub) retrain-(a) 특성화 — 작은-N 별도 무대

> 도입 문구(필수): **(a)는 2ᴺ 재학습이라 주무대(N=50/100)에선 불가 → 부득이 작은-N 별도 무대 + 실험-다양성 목적의 의도적 세팅 차이**. vs (a)는 방법-중립 참값이라 전 방법 허용.

**LLM anchor5 (N=5) — 전 방법 vs (a) retrain oracle** ● (1B, 3-seed mean±std) — ⏸ **보류 참조(2026-07-24; 주 (a)-무대=silo5, anchor5는 폴백)**

| method | Spearman vs (a) ↑ | (참고) vs (b) ↑ |
|---|---|---|
| **Flirds** | **0.933±.047** | 1.000 |
| **Flirds-1st** | **0.933±.047** | 1.000 |
| **loss-heur** | **0.933±.047** | 1.000 |
| GTG | 0.933±.047 | 1.000 |
| FedSV | 0.733±.170 | 0.700 |
| ShapleyFL | 0.767±.330 | 0.700 |
| ComFedSV | 0.467±.450 | 0.500 |
| FedIF | 0.167±.613 | 0.067 |

> 천장 효과: same-game 3종·GTG가 vs (a) **0.933** 동률인 이유 = 이들이 (b)와 거의 완전일치 → vs (a) 점수 = **(b)↔(a) 듀얼오라클 일치도 0.933** 그 자체. **출처**: `runs/track_d/rundirs/1B_anchor5_seed{0,1,2}/phi.parquet`(truth=`(a)oracle`).
> ![[flirds-paper-results-overview-figs/f2_anchor5_vs_a_bar.png]]

**LLM silo5 (a)-leg (★유일 (a)-무대) — {clean,noisy,frzero}×3seed** ⬚ · ~~LLM gsm5 (주표)~~ ⏸ **보류(2026-07-24)**
- 채움 = `runs/track_d/rundirs/1B_silo5_*_aleg`(L8). **silo5**(non-IID)만이 실재 cross-seed 신호를 갖는 (a)-검증 무대 → 헤드라인 = `(b)oracle` 행 `rho_a`(목표 clean +0.87 / noisy +0.93). **gsm5·anchor5 보류**(gsm5=IID 축퇴로 anchor5 0.933과 중복; 코드·캐시 존치·부활 가능). anchor5 0.933은 아래 **폴백 참조**.

**CNN C1 시나리오별 vs (a) retrain oracle** ● (3-seed 평균; 8칸 = cifar10·mnist × 4 시나리오 · 전 8 방법 · **label_skew 제외**)

> **오염 구조 (칸마다 신호가 다른 이유 + 데이터셋별 오염군)**: C1의 오염은 **클라 index 사다리** `_pair_ladder`(pair p → 5p%, N=10 → **0/0/5/5/10/10/15/15/20/20%**)로 **`label_flip`·`feature_noise`에만** 적용 = strmain류 **per-client 강도 변조**(클라마다 dose가 다름). `quantity_skew`=크기 이질성, **`iid`=오염0 신호-부재 대조군**. *(label_skew는 오염0 + (a)-게임 축퇴로 정보량 낮아 07-23 삭제.)* ⚠ **오염 사다리는 데이터셋·seed 무관 동일**(client index로만 결정) → cifar10↔mnist 값 차이는 *오염군이 달라서가 아니라* 과제 난이도(아래 읽기 iii).

**Spearman vs (a)** (신호-강 위 / iid 아래 / 방법별 평균)

| dataset/scenario | Flirds | Flirds-1st | loss-heur | GTG | FedSV | ComFedSV | ShapleyFL | FedIF |
|---|---|---|---|---|---|---|---|---|
| cifar10 / feature_noise | **+0.63** | +0.50 | +0.56 | +0.44 | +0.18 | +0.39 | +0.28 | +0.40 |
| cifar10 / label_flip | +0.52 | +0.59 | +0.58 | +0.45 | +0.41 | +0.32 | +0.29 | +0.36 |
| cifar10 / quantity_skew | +0.57 | +0.56 | +0.57 | +0.70 | +0.70 | +0.72 | **+0.81** | -0.03 |
| mnist / feature_noise | +0.33 | +0.44 | +0.44 | +0.40 | -0.07 | -0.07 | +0.60 | +0.66 |
| mnist / label_flip | **+0.96** | +0.97 | +0.97 | +0.97 | +0.96 | +0.94 | +0.96 | +0.96 |
| mnist / quantity_skew | **+0.85** | +0.77 | +0.84 | +0.56 | +0.68 | +0.65 | +0.51 | -0.09 |
| cifar10 / iid *(오염0)* | -0.23 | -0.13 | -0.18 | -0.20 | -0.18 | +0.30 | +0.00 | +0.07 |
| mnist / iid *(오염0)* | +0.36 | +0.52 | +0.48 | +0.19 | -0.11 | -0.09 | +0.66 | +0.74 |
| **평균 (8칸)** | **+0.50** | **+0.53** | **+0.53** | **+0.44** | **+0.32** | **+0.39** | **+0.52** | **+0.38** |
| _평균 (iid 제외 6칸)_ | _+0.64_ | _+0.64_ | _+0.66_ | _+0.59_ | _+0.48_ | _+0.49_ | _+0.58_ | _+0.38_ |

**Pearson vs (a)** (같은 순서)

| dataset/scenario | Flirds | Flirds-1st | loss-heur | GTG | FedSV | ComFedSV | ShapleyFL | FedIF |
|---|---|---|---|---|---|---|---|---|
| cifar10 / feature_noise | +0.60 | +0.51 | +0.57 | +0.43 | +0.19 | +0.44 | +0.31 | +0.47 |
| cifar10 / label_flip | +0.44 | +0.46 | +0.49 | +0.32 | +0.26 | +0.33 | +0.28 | +0.42 |
| cifar10 / quantity_skew | +0.60 | +0.55 | +0.60 | +0.72 | +0.63 | +0.71 | +0.72 | -0.04 |
| mnist / feature_noise | +0.42 | +0.52 | +0.46 | +0.22 | -0.10 | -0.17 | +0.63 | +0.63 |
| mnist / label_flip | +0.99 | +0.99 | +0.99 | +0.98 | +0.97 | +0.94 | **+1.00** | +0.98 |
| mnist / quantity_skew | +0.59 | +0.60 | +0.61 | +0.45 | +0.55 | +0.63 | +0.42 | -0.15 |
| cifar10 / iid *(오염0)* | -0.18 | -0.06 | -0.02 | -0.14 | -0.20 | +0.46 | +0.05 | +0.09 |
| mnist / iid *(오염0)* | +0.42 | +0.59 | +0.54 | +0.32 | -0.09 | -0.15 | +0.69 | +0.70 |
| **평균 (8칸)** | **+0.48** | **+0.52** | **+0.53** | **+0.41** | **+0.28** | **+0.40** | **+0.51** | **+0.39** |
| _평균 (iid 제외 6칸)_ | _+0.61_ | _+0.60_ | _+0.62_ | _+0.52_ | _+0.42_ | _+0.48_ | _+0.56_ | _+0.39_ |

> 읽기: (i) **신호-강 칸(label_flip·feature_noise·quantity_skew)**서 두 게임 수렴 — same-game 3종 1위 2칸(cifar10/fn·mnist/qskew), mnist/label_flip은 전 방법 Sp +0.94~+0.97·Pe +0.94~+1.00 동수렴. (ii) **평균 읽는 법 — iid 희석 주의**: iid(오염0)은 대부분 0~음수(파티션만 있어 (a) 재학습-게임이 순위 재현 못 함)라 **평균(8칸)에 iid가 섞이면 same-game이 깎여 renorm과 붙어 보인다**(Flirds Sp +0.50 ≈ ShapleyFL +0.52). **실제 서열은 iid 제외 6칸**: same-game **0.64~0.66** > GTG 0.59 > ShapleyFL 0.58 > FedSV/ComFedSV 0.48~0.49 > FedIF 0.38. renorm-족이 신호-부재/크기 칸서 부분 우위(cifar10/qskew ShapleyFL Sp +0.81 = (a) 재정규화-게임 심판이 renorm 유리, 부록 C 각주 T10). (iii) **데이터셋 차 = 난이도**: 같은 오염 사다리인데 mnist/label_flip +0.96 > cifar10/label_flip +0.52 = mnist가 더 분리적(쉬운 과제)이라 오염 신호가 (a)에도 뚜렷 — 오염군이 다른 게 아니다. (iv) **Spearman↔Pearson 정합**(renorm-유리 칸서만 소폭 갈림). **출처**: `runs/track_c/fidelity.csv`(`spearman_a`/`pearson_a`, (dataset,scenario,method) 3-seed 평균, label_skew 제외).
> ![[flirds-paper-results-overview-figs/f1_cnn_c1_vs_a_heatmap.png]] *(좌=Spearman·우=Pearson 2-panel; label_skew 제외 8칸 + 방법별 평균 행)*

---

## §5.3 개입 (downstream)

### (메인) LLM R4 P1 — T1 online / T2 retrain × {clean, noisy, frzero, **frrand ⬚**, **strmain-dose ⬚**}, 절대 EM — ⬚

행 = vanilla·oracle_excl·random_excl·t2_random + estimator 4점수원(renorm 4종은 L4 착지 시 블록 추가). **frrand·strmain-dose 열 = 신규 계획**(REMAINING §1.6a L9·L10; 종전 "R4 frrand 재제안 금지" 번복) — CNN은 frrand가 flirds 단독 leg로 존재(아래 CNN 표), R4는 미실행.

| arm | clean | noisy(swap@.7) | frzero | frrand ⬚ | strmain-dose ⬚ |
|---|---|---|---|---|---|
| vanilla / oracle_excl / random_excl | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| Flirds / Flirds-1st / loss-heur / FedIF (P1) | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |

> ⚠ **pre-fix R4 seed0 값 인용 금지** — 정본 = L1 3-seed. 채움 = `runs/track_h/rundirs_llm/gsm50k5_*`(fix-후) → `runs/track_h/analysis/gsm50k5_*.csv`. (seed0 ◐ 파일럿은 존재하나 H1 판본 혼재로 본 페이지 미기입.)
> ![[flirds-paper-results-overview-figs/f4_r4_intervention_em.png]] ⬚ *(F4 = R4 EM bar — L1 착지 후 생성)*

### (메인) CNN 8 점수원 × P1 절대 acc ● (dir1; 캠페인 restack 드리프트 확인 후 확정)

**P1 sign-게이트 · online** (3-seed mean; vanilla=바닥·oracle_excl=천장·random_excl=무작위-제외 대조)

| arm | clean | frzero | frrand¹ | grad-noise | label-flip@.70 | 평균² |
|---|---|---|---|---|---|---|
| vanilla (바닥) | .6389 | .5879 | .5876 | .2436 | .5247 | .4521 |
| oracle_excl (천장) | – | .6203 | .6195 | .6203 | .6236 | .6214 |
| random_excl (무작위) | – | .5838 | .5839 | .2590 | .5018 | .4482 |
| **flirds** | .6315 | .6148 | .5895 | **.5668** | .5712 | .5843 |
| flirds1st | .6384 | .6216 | ⬚ | .2479 | .5717 | .4804 |
| lossheur | .6264 | .6114 | ⬚ | .5981 | .5670 | .5922 |
| fedif | .6386 | .6143 | ⬚ | .2479 | .5728 | .4783 |
| gtg | .6051 | **.3915** | ⬚ | .5972 | .5479 | .5122 |
| fedsv | .5982 | **.3966** | ⬚ | .5972 | .5164 | .5034 |
| comfedsv | .5963 | **.3918** | ⬚ | .5871 | .5152 | .4981 |
| shapleyfl | .6045 | **.4020** | ⬚ | .6115 | .5278 | .5138 |

> ¹ **frrand = 현재 flirds 단독 leg** — CNN competition에서 free-rider-random은 flirds·앵커만 실행(다른 7 방법·retrain 미실행) → **7 방법 셀 = ⬚**(C-fr 착지 시 채움; 앵커·flirds만 ● 값). flirds .5895(vanilla .5876·random_excl .5839 위, oracle_excl .6195 근처)로 frzero(.6148)와 같은 exact-0 생존 계열. ⚠ **완전한 축(7 방법+retrain)으로 만들려면 CNN도 추가 실험 필요**(REMAINING §1.6a **C-fr**; random free-rider서도 renorm 붕괴가 나는지 = frzero 대칭 확인) · **R4(LLM) frrand는 전무 = 신규 계획**(§1.6a L9).
> ² **평균** = 오염 3종(frzero·grad-noise·label-flip@.70) 산술평균 — clean(do-no-harm 대조)·frrand(대부분 ⬚)은 방법 간 비교가능성 위해 제외. online·retrain 공통. (개별 셀은 위 각 열; 소스 = `cnn_competition.csv`.)

**P1 · retrain** (관찰자 최종 부호로 kept → init부터 재학습; **frrand retrain = C-fr 계획**[⬚], 현재 online만 실행)

| arm | clean | frzero | grad-noise | label-flip@.70 | 평균² |
|---|---|---|---|---|---|
| vanilla (바닥) | .6389 | .5879 | .2436 | .5247 | .4521 |
| oracle_excl (천장) | – | .6203 | .6203 | .6236 | .6214 |
| **flirds** | .6277 | .6063 | **.6065** | .6192 | .6107 |
| flirds1st | .6386 | .6252 | .2436 | .6236 | .4975 |
| fedif | .6417 | .6252 | .2436 | .6217 | .4968 |
| lossheur | .6293 | .6125 | .4518 | .6205 | .5616 |
| gtg | .6265 | .5158 | .6203 | .5991 | .5784 |
| fedsv | .6166 | .5140 | .6203 | .5904 | .5749 |
| comfedsv | .6232 | .5200 | .6203 | .5921 | .5775 |
| shapleyfl | .6223 | .5113 | .6203 | .6028 | .5781 |

> 서술 클레임(정확형): "**전 정책·전 시점 상위권 + grad-noise를 잡는 유일한 estimator(2차항: flirds GN online .5668 / retrain .6065 — vs 1차-계열 실명 flirds1st .2479/.2436·fedif .2479/.2436, lossheur 부분 .5981/.4518) + frzero에서 exact-0 계열 생존(online .61~.62) vs renorm 붕괴(online gtg .3915·fedsv .3966·comfedsv .3918·shapleyfl .4020 / retrain .51~.52)**". 정직 보고: clean 오발화 flirds −0.7pt·R4 clean T1 −1.0pt, 단 **T2 최종-부호는 무해(kept=전원)**. **출처**: `runs/track_h/analysis/cnn_competition.csv`(dataset=cifar10·partition=dir1·policy=P1; frrand=flirds leg). **P1w(크기-가중)** 는 결과 규칙부(00-INDEX §1).
> ![[flirds-paper-results-overview-figs/f5_cnn_competition_p1_online.png]]

---

## §5.4 탐지 (2차 ③ — 마지막) — ⬚

R4 φ-파생(same-game 3종 + (b)) + **전용 탐지기 4종**(FLDetector/FLTrust/STD-DAGMM/FedDQC — §2 약속 이행) · c2fid φ-AUROC.

| 셀 | Flirds AUROC | (b) AUROC | 탐지기 4종 | 상태 |
|---|---|---|---|---|
| R4 gsm50k5 · noisy | ⬚ | ⬚ | ⬚ | 채움 = `runs/phase2_matrix/rundirs/1B_gsm50k5_*`(L2) |
| R4 gsm50k5 · frzero | ⬚ | ⬚ | ⬚ | (L2) |
| R4 gsm50k5 · **frrand** | ⬚ | ⬚ | ⬚ | **신규 축** = L9(§1.6a) — random free-rider 탐지 |

**c2fid (CNN C2, cifar10/dir1) — φ-AUROC 전 방법 × 오염-시나리오** ● 3-seed (seed0–2; clean=오염0 → 탐지 대상 없어 제외)

| 방법 | frzero | frrand | grad-noise | lf@.15 | lf@.35 | lf@.70 | strmain | 평균 |
|---|---|---|---|---|---|---|---|---|
| **(b)oracle** | 0.683 | 0.685 | 1.000 | 0.574 | 0.951 | 0.997 | 0.997 | 0.841 |
| **Flirds** | 0.683 | 0.682 | 0.998 | 0.603 | 0.954 | 0.996 | 0.997 | 0.845 |
| Flirds-1st | 0.978 | 0.978 | 0.494 | 0.602 | 0.939 | 1.000 | 1.000 | 0.856 |
| loss-heur | 0.800 | 0.801 | 0.994 | 0.571 | 0.931 | 0.997 | 0.996 | 0.870 |
| GTG | 0.022 | 0.023 | 1.000 | 0.336 | 0.444 | 0.867 | 0.861 | 0.508 |
| FedSV | 0.008 | 0.008 | 1.000 | 0.370 | 0.482 | 0.872 | 0.870 | 0.516 |
| ComFedSV | 0.219 | 0.219 | 0.738 | 0.463 | 0.462 | 0.570 | 0.590 | 0.466 |
| ShapleyFL | 0.000 | 0.000 | 1.000 | 0.380 | 0.577 | 0.894 | 0.893 | 0.535 |
| FedIF | 0.955 | 0.955 | 1.000 | 0.642 | 0.943 | 1.000 | 1.000 | 0.928 |

> frzero/frrand 데이터 사실(해석 없이 수치만): free-rider φ = **정확히 0**(Flirds·(b)·Flirds-1st 모두 min=max=0·std=0). benign 클라 φ<0 비율 = (b) 68.3% / Flirds 68.3% / Flirds-1st 97.8%(benign 평균 φ = −0.017 / −0.020 / −0.110; cifar10/dir1 3-seed). renorm(GTG/FedSV/ShapleyFL) frzero·frrand AUROC = 0.0~0.02.

> H-13 판정 기준: 주장은 절대값이 아니라 **oracle-동행** $|\mathrm{AUROC(Flirds)}-\mathrm{AUROC((b))}|\le 0.05$. c2fid ● 3-seed에서 frzero(Δ.000)·grad-noise(Δ.002)·frrand(Δ.003) 모두 **동행 성립** — 단 frrand·frzero는 (b) 자체가 0.68(약신호)이라 판정 대상은 "탐지 성능"이 아니라 "오라클 추종". **CNN frrand 탐지는 c2fid로 착지**(C-fr는 개입-acc leg만 남음); **R4 frrand 탐지 = 신규 축 미실행**(L9, §1.6a). (strmain-dose 탐지는 부차 — dose-변조 하 오염 클라 flag; 필요 시 별도 판단.)
> ![[flirds-paper-results-overview-figs/f6_detection_auroc.png]] ⬚ *(F6 = 탐지 AUROC — L2/c2fid 착지 후 생성)*

---

## §5.5 비용 (부분 ⬚)

**op-count 축 — 지배 연산 수 · 하드웨어/정밀도 독립** ●

| regime | Flirds | Flirds-1st | loss-heur | (b) | 교차검증(측정) |
|---|---|---|---|---|---|
| silo (N5·R10) | 10 HVP | 10 grad | 60 fwd | 320 fwd | Flirds 104→107s · (b) 512→530s · loss-heur 96s(post-fix) |
| anchor (N5·R30) | 30 HVP | 30 grad | 180 fwd | 960 fwd | Flirds 707s · (b) 3528s |
| device (N100·R30·K10) | 30 HVP | 30 grad | 330 fwd | **30,720 fwd** | (b) **24,975s** vs Flirds 157s ≈ **159×** |

> microbench(fp32·B200): forward 1.60s · HVP 10.36s → HVP/fwd = 6.47. **Flirds(2차) 비용은 라운드당 cohort와 무관(1 HVP/round)**, (b)는 cohort에 지수적(2^k/round) → cohort 큰 무대(anchor·device)서 Flirds 압승, 작은 cohort(std20)선 Flirds-1st만 우위(소-cohort 역전은 op-count 축 서술). **주무대 실측 runtime(R4 timing.json)** = ⬚(K_r=5 조건부 "vs (b) ~5×"). 지수-비용 실측(N=10 160×·device100 159×) = 부록 E. **출처**: `runs/measured_2026-07/op_counts.py` + rundir `runtime`.
> ![[flirds-paper-results-overview-figs/f7_cost_scaling.png]]

---

## §5.6 Ablation (작성 가능)

**① 2차항(HVP)의 기여 — Flirds vs Flirds-1st** ●
- **부분참여 fidelity**: CNN C1 label-flip k=0.2에서 Flirds **0.891** vs Flirds-1st **0.305**(k=0.5 .979/.765·full .993/.940; 전 72셀 pool Flirds 0.953). → 2차 Hessian 항이 partial 참여에서 값을 함.
- **grad-noise 개입**: Track H에서 Flirds GN acc **.567~.607**(online .5668/retrain .6065) vs **1차-계열 실명** — Flirds-1st .248/.244·FedIF .248/.244(≈vanilla .244), loss-heur 부분 .598/.452 — 1차 정보만으론 noise 클라 불가시. **c2fid fidelity로도 재확인**(● 3-seed): grad-noise vs (b) Spearman Flirds **0.847** vs Flirds-1st **0.218**(1차 붕괴; loss-heur 0.986) — 개입-acc·fidelity 양쪽서 2차항 필요.
- **dose-해상도 변별(F-4)**: CNN c2fid strmain(● 3-seed) — spearman_vs_rate Flirds **0.858 ≈ (b) 0.857**(corrupt-only Flirds 0.519 / (b) 0.515 = 오라클 정확추종)이나 Flirds-1st **0.877/0.659 ≥** Flirds → 사전등록 "Flirds ≳ Flirds-1st" **MISS 확정**(3-seed; 2차항의 dose-해상도 우위는 이 셀서 미확인; Flirds 강점은 (b) 정확추종). **R4 strmain-dose = ⬚**(L10; §5.2 F-4 각주의 LLM 대응).
> ![[flirds-paper-results-overview-figs/f9_second_order_ksweep.png]]

**② A축 lever probe** ●
- rank·lr·steps·폭·참여 lever가 **신호를 못 만들고**(IID-clean cross-seed ρ≈0), fidelity는 lever 전반 **1.000**(Taylor tradeoff 없음). 신호는 A축(용량)이 아니라 B축(클라 간 실제 차이=오염·비IID)이 만든다. **출처**: `runs/probe_signal/`(LLM)·`runs/probe_signal/cnn_c1`·`cnn_c2`(CNN). (표 최소 — §4.2·§4.3 요약.)

**③ removal-curve — 게임-무관 인과 검증** ●

| threat | Flirds ρ(vs b) | worst-first Δ | best-first Δ | 판정 |
|---|---|---|---|---|
| silo5 noisy (val-loss) | +1.00 | **+0.0076** | −0.0084 | worst-first 제거가 loss 내림 = 순위 인과적 ✅ |
| silo5 frrand | +1.00 | +0.0071 | −0.0015 | ✅ |
| silo5 frzero | +1.00 | +0.0067 | −0.0016 | ✅ |
| cifar10 label_flip (acc) | +1.00 | **acc 분리 +0.0445** | (b) 동급 +0.045 | ✅ (mnist의 ~13×) |
| cifar10 feature_noise (acc) | +1.00 | acc 분리 +0.0385 | ShapleyFL(저순위) ≈0 | 순위→분리 인과 재확인 |

> Flirds가 (b)·coalition과 동일 인과-removal 품질을 **5× 싸게**. **출처**: `runs/removal_dose/rundirs/`(LLM A2) · `runs/removal_dose/rundirs_cnn/`(CNN A3).
> ![[flirds-paper-results-overview-figs/f8_removal_curves.png]]

---

## 부록 B–E 대응

**부록 B — 프로토콜 상세** (작성 가능)
- 무대별 하이퍼 표 · 위협 구현 정의(answer_swap/frzero/frrand/gn/lf + 라벨-플립 $(\rho,\tau)$ 규약[§5.1]) · 데이터 분배(GSM8K val=공식 test 카브) · 환경 1줄("fp32; cuDNN conv TF32; 스택 내 결정론") · ComFedSV per-round 대용 caveat · ShapleyFL β=0.3 각주 · **LLM에 grad-noise 없는 이유 2문장**(등방 노이즈는 LoRA 기하에서 gradient-방향 대비 응답 수십분의 일 → 무대 미성립).

**부록 C — fidelity 확장** (부분)
- **same-game 3 vs (b)** 확장(c2fid·R4-L2 ⬚; Kendall·거리 3종) — **cross-game은 vs (b) 미비교**(2026-07-24 기준: in-run은 same-game 오라클이라 동일-게임 방법만 채점; cross-game 판정은 vs (a)에서만) · **vs (a) 전 방법**(C1 시나리오 ● · **silo5 ⬚(활성)** · anchor5 ●=보류 폴백 · **gsm5 보류**) · std50k5 부분참여 probe. **출처**: `runs/track_c/fidelity.csv`·`runs/track_d/rundirs/*`.

**부록 D — stability** ● (수록 무대 한정)

| method | rho_xseed ↑ | topJ ↑ | botJ ↑ |
|---|---|---|---|
| (b)oracle (자체) | 0.518±.453 | 0.522±.395 | 0.555±.453 |
| **Flirds** | **0.547±.394** | 0.544±.324 | 0.500±.419 |
| Flirds-1st | 0.510±.461 | 0.611±.395 | 0.455±.386 |
| loss-heur | 0.474±.448 | 0.500±.287 | 0.467±.384 |
| GTG | 0.311±.441 | 0.467±.358 | 0.378±.345 |
| FedSV | 0.289±.385 | 0.356±.351 | 0.345±.292 |
| ComFedSV | 0.198±.383 | 0.300±.268 | 0.289±.366 |
| ShapleyFL | 0.124±.431 | 0.200±.276 | 0.344±.331 |
| FedIF | 0.243±.413 | 0.322±.296 | 0.244±.293 |

> C1 방법 안정성: **Flirds 0.547 = (b) 0.518**(oracle 내재 안정성을 그대로 추종), recon MC baseline은 0.12~0.31로 하락. (b) target 안정성(수록 무대 한정) = c2fid 착지 후 열 추가. **출처**: `runs/track_c/RESULTS.txt`(C1 stability)·`runs/track_c/c1/*/metrics.json`.

**부록 E — 비용·규모 보조** ● (07-23 본문→부록)
- **N=10 2¹⁰**: (b) exact **117,649s(32.7h)** vs Flirds **733s = 1/160**(1-seed 명기). 출처 `runs/track_d/rundirs_e5_n10/1B_anchor10_seed0/metrics.json`.
- **device100 anchor**: (b) per-round ~25,000s vs Flirds **157s = 1/159**. 출처 `runs/phase2_matrix/rundirs/1B_device100-a0.5_*` + op-count 모델.
- **Scale 100/100 P1**(완전참여): P1 행 + vanilla/oracle/random 앵커. 출처 `runs/track_h/rundirs_cnn_scale/`. (§4.8.2)
> F7 Panel B(=위 §5.5)가 N=10·device100 지수-비용을 함께 보여준다.

---

## 갱신 규칙

1. 실험 착지 → **rundir/analysis 재생성**(수기 금지) → 필요 시 전량 카탈로그(`flirds-experiment-results-overview.md`) 먼저 → **이 페이지 ⬚ 채움**.
2. **figure 재생성**: `python research-wiki/survey/flirds-paper-results-overview-figs/make_figures.py`(anaconda: `C:\Users\chyoy\anaconda3\python.exe`). 입력 = 리포 내 rundir/CSV, 출력 = 같은 폴더 PNG. **현 구현 = F1/F2/F5/F7/F8/F9**(데이터 있음). **F3/F4/F6은 미구현** — c2fid 메인쌍·L1 R4 EM·L2/c2fid 탐지 착지 시 스크립트에 함수 추가 필요(현재는 __main__이 ⬚로 안내·skip).
3. pre-fix R4 seed0·CNN restack-드리프트 미확정 값은 **⬚ 유지**(정본 확정 전 기입 금지).
4. 미수록 실험은 넣지 않는다(스코프 규약) — 전량은 카탈로그 문서가 담당.

**figure 인벤토리**: F1·F2·F5·F7·F8·F9 = ● 생성(6/6) · F3(메인 쌍)·F4(R4 EM)·F6(탐지) = ⬚(데이터 착지 후).
