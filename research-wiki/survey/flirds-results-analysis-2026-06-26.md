---
type: survey
title: "Flirds 실험 결과 해석·종합 (2026-06-26)"
created: 2026-06-26
updated: 2026-07-14
sources: [flirds-experiment-results-overview-2026-06-25, baseline-original-paper-verification-2026-06-22, prior-work-taxonomy/README]
tags: [survey, analysis, synthesis, fidelity, detection, cost, cnn-vs-llm, claims]
---

# Flirds 실험 결과 해석·종합

> **성격**: [[flirds-experiment-results-overview-2026-06-25]]가 "팩트(수치)" 문서라면, 이 문서는 그 수치가
> *무슨 이야기를 하는지*를 핵심 질문 위계(1차 fidelity → 2차 ①성능 ②수렴 ③탐지 → 비용; 루트 CLAUDE.md)
> 순으로 끌어내는 "해석" 문서다. **수치 나열이 아니라 종합**이 목표.
> **CNN과 LLM은 거동이 다르므로 모델 계열을 분리**해 본다(둘이 갈리는 칸은 pool 금지).
>
> **출처 규율**: 모든 수치는 overview(이미 RESULTS↔CSV 교차검증)에서 인용. 결론이 걸리는 핵심 수치 +
> overview가 스스로 플래그한 모순만 raw로 spot-check 했고(아래 §7), 전부 일치/해소됨. git HEAD `c414392`. (주: 이후 07-03 `b1b95d0`가 3B track_d rundir를 β0.3 재실행으로 교체 — 본문 3B 수치는 그 이전 기준.)
> 자매 문서: baseline↔원논문 대조 [[baseline-original-paper-verification-2026-06-22]] · 선행연구 6축 분류
> [[prior-work-taxonomy/README]].
>
> **확립(established)** = 3-seed·실측 oracle·복수 무대에서 일관 / **시사적(suggestive)** = 1-seed·tiny-val·
> 단일 셀·proxy-truth 의존. 본문에서 매 주장에 이 구분을 단다.

---

## 0. 결과 한 문단 (abstract용)

Flirds는 **연합학습 한 학습 궤적**에서 client-level 데이터 기여도를 1차+2차 Taylor(true Hessian)로
추정한다. 네 트랙(LLM 표준 `track_d` 1B/3B/7B · CNN `track_c` · Robustness `phase2_matrix` · Foundational
`phase1`)에서, **clean·near-additive 레짐의 fidelity는 사실상 포화된다** — LLM에서는 Flirds·Flirds-1st·
loss-heur·GTG·Banzhaf가 거의 모두 exact in-run oracle 대비 Spearman 1.000으로 동률이고(track_d std20/
anchor5, 3-seed), 그래서 이 레짐에서 Flirds의 차별점은 *정확도가 아니라 비용*이다: 2차 Flirds의 비용은
**라운드당 cohort 크기와 무관**(1 HVP/round)인 반면 exact oracle은 cohort에 지수적(2^k/round)이라,
참여가 많은 무대에서 Flirds가 oracle을 **160×까지**(device100 anchor 157s vs (b) 25,000s) 앞선다.
**정확도 차별은 near-additivity가 깨지는 곳에서만** 드러난다: (i) poison(clean-보존 backdoor)에서 recon-MC
baseline(FedSV Spearman 0.367)과 Flirds-1st(0.000)가 무너질 때 2차 Flirds가 0.967로 버티고, (ii) device100
non-IID에서 Flirds 1.000 vs GTG 0.78/FedSV 0.75/ComFedSV≈0. 단 **CNN은 거동이 다르다** — clean에서도
fidelity가 큰 spread를 보이고(Flirds 0.919 vs GTG 0.57/FedSV 0.40/ComFedSV 0.35), 2차항이 benign에서도
도움이 되며(0.832→0.919), 무엇보다 **retrain oracle (a)와 in-run oracle (b)가 갈린다**(Flirds vs (b)=0.919
이지만 vs (a)=0.35; 어떤 method도 (a) 대비 0.45를 못 넘김) — LLM에서 두 oracle이 0.933으로 거의 일치하는
것과 대조된다. 2차 검증(성능·수렴·탐지)은 위계대로다: clean-IID에서 성능·수렴은 **do-no-harm parity**(차이
미미), 오염이 있어야 기여도-가중이 정확도/수렴을 회복하며, 탐지는 위계상 마지막 — **valuation φ는 전용
탐지기가 아니다**(device100 non-IID에서 φ 0.57~0.77 vs FedDQC 1.0; 이는 *exact* oracle도 0.604에 그쳐
근사 결함이 아니라 valuation의 내재 특성). 종합하면 가장 단단한 주장은 **"Flirds는 exact in-run Shapley를
충실·저렴하게 근사하며, 그 비용 우위는 cohort에 무관하다"**이고, 정확도 우위·retrain-counterfactual 충실도·
poison 강건성은 무대 의존적인 조건부 주장이다.

---

## 1. 핵심 발견 (위계 순)

### 1.1 Fidelity (1차) — 무엇이 확립됐나

**LLM (track_d, clean·IID, 3-seed).** clean·near-additive 레짐에서 *정직한* valuation 방법은 거의 전부
exact (b) in-run oracle 대비 **Spearman 1.000으로 천장에 붙는다**: std20(N=20)·anchor5(N=5) 모두 Flirds·
Flirds-1st·loss-heur가 1.000, GTG 0.97~1.00, Banzhaf(anchor5) 1.000. **확립**: clean-LLM-FL에서 in-run
fidelity는 "푸는 문제"가 아니라 거의 자동으로 풀린다 — Shapley 선형성 + near-additive 효용이면 exact·근사가
degenerate-equal이 되기 때문. **이 사실의 직접적 함의**: 이 레짐에서 method 간 *정확도* 서열은 거의 무의미
하고(§5.1 긴장), 진짜 갈리는 축은 비용(§1.5)이다.

- LLM에서 *항상 낮은* 방법은 설계상 그렇다(오독 금지, [[baseline-original-paper-verification-2026-06-22]] §3):
  **FedIF**(영향도-가중, Shapley 아님; std20 0.157), **ShapleyFL**(surrogate; std20 0.194), **ComFedSV**
  (low-rank 가정 위배 → 3B std20 **−0.137**(β0.3 재실행 `b1b95d0` 기준)). 이들의 저-fidelity는 "나쁨"이 아니라 "다른 게임"·"가정 위반".
- **FedSV만이 정직한 recon-MC인데도 LLM anchor5에서 0.700으로 떨어진다**(std20 0.910) → per-round MC
  분산. recon-MC가 LLM에서도 exact를 못 따라가는 첫 신호(CNN에서 더 크게 나타남, §2).

**CNN (track_c C1, N=10 full, 10시나리오×3seed).** **확립 + LLM과 결정적으로 다름**: clean에서도 fidelity가
**큰 spread**를 보인다. vs (b) Spearman pool: Flirds **0.919**, Banzhaf 0.989, loss-heur 0.860, Flirds-1st
0.832, FedIF 0.491, FedSV 0.401, ShapleyFL 0.391, Ripple 0.373, ComFedSV 0.348, GTG 0.569. recon-MC 계열
(GTG/FedSV/ComFedSV/Ripple/ShapleyFL)이 CNN에서 훨씬 약하다 — LLM의 "거의 다 1.0"과 정반대. (spot-check
재집계: Flirds 0.9192, Banzhaf 0.9891, Flirds-1st 0.8315 — overview와 일치.)

**method 우열과 이유**:
- **Banzhaf가 값-수준 fidelity 최강**(CNN Pearson 0.998, Kendall 0.967) — exact semivalue라 noise 없이
  깨끗. 단 anchor5(N=5)에서만 실행 가능(2^N exact, N≤10 규칙) → 대규모 무대엔 baseline으로 못 씀.
- **Flirds(2차)가 in-run oracle을 가장 충실히 따르는 *근사* 방법**(CNN 0.919, LLM 1.000): exact가 불가능한
  무대(N=20/100, LLM)에서 유일하게 천장급을 유지하는 1-HVP 방법.
- recon-MC(GTG/FedSV/ComFedSV)의 약점은 **CNN full-model·non-additive에서 sub-model 재구성이 빗나가기**
  때문 — 이론적 수렴은 가법 영역에서만(taxonomy의 GTG "가법 영역서 exact 수렴" 판정과 정합).

**dual-oracle (a) vs (b) — 확립된 cross-model 갈림 (§2.4에서 심화)**: CNN은 (a) retrain·(b) in-run 둘 다
실측했는데 **갈린다** — Flirds vs (b)=0.919이지만 vs (a)=0.352, 그리고 *어떤 method도 vs (a) 0.45를 못
넘는다*(ShapleyFL 0.453 최고, Flirds-1st 0.408 > Flirds 0.352). 즉 (b)≈(a)가 아니라 (b)와 (a)가 서로 다른
순위를 매긴다. 반대로 **LLM(1B anchor5)은 (a) vs (b)=0.933으로 거의 일치**. (3B/7B (a)는 미실행 ⬚.)

### 1.2 Selection→성능 / Aggregation (2차 ①)

**LLM (track_d, clean·IID intervention arms).** **확립**: 모든 개입 arm(flirds_w/shapleyfl_w/fedif_w/
flirds_sel)의 MMLU·ROUGE가 vanilla와 **±0.001~0.003 이내** = **do-no-harm parity**. clean-IID에선
기여도-가중이 성능을 *해치지도 크게 올리지도* 않음 → 이는 기대대로의 **null/parity 결과**이지 성능 *향상*
주장이 아니다. (학습 자체는 base 대비 ROUGE 크게 ↑: 3B std20 0.222→0.302; MMLU는 SFT로 분포-밖 소폭 ↓.)

**CNN (track_c C2, N=100, 오염 존재).** **확립 + 보완적**: 오염이 있으면 기여도-가중이 정확도를 **회복**
한다 — grad_noise vanilla 0.499 → flirds_mult 0.609 / shapleyfl 0.645 / flirds_repl 0.621; label_flip
vanilla 0.583 → flirds_mult 0.626. clean에선 parity~소폭↑(0.686→0.698). **두 트랙이 일관된 이야기**: 신호
(오염/non-IID)가 있어야 selection 이득이 난다 — LLM clean-IID의 null과 CNN clean의 parity가 같은 결을
이루고, CNN 오염 칸에서만 이득. **단 Flirds가 arm 중 유일 최강은 아니다**: grad_noise에선 shapleyfl(0.645)
이 flirds_mult(0.609)를 앞서고, dir1 전용 flirds_repl/add가 더 강함 → **"경쟁적이되 지배적이지 않다"**(정직).

### 1.3 Convergence (2차 ②)

**LLM.** clean-IID라 arm 간 거의 동률. **시사적**(under-powered): 두드러진 칸은 **7B std20** — 개입 arm이
vanilla 184.7 라운드 대비 ~151~158 라운드로 target 도달(~14~18% 빠름). 단 std가 겹친다(184.7±18.3 vs
153±18.8) → 통계적으로 깨끗하지 않음. 3B std20도 ~192 vs 198(미미). 1B·anchor5는 차이 없음.

**CNN.** **확립 방향**(분산 큼): grad_noise에서 기여도-가중이 수렴을 크게 앞당김 — vanilla 27.2 라운드 →
flirds_mult 13.7 / flirds_repl 6.3; free_rider flirds_repl 7.7 vs vanilla 41.2. 성능과 같은 패턴(오염서 이득).
단 target-미달 셀 때문에 std가 커서 수치 자체는 coarse.

### 1.4 Detection (2차 ③, 위계상 마지막)

**LLM silo5 (N=5, 3-seed).** **확립**: noisy·free-rider 위협은 valuation·전용탐지기가 거의 전부 AUROC
1.000(near-additive). **poison(clean-보존 backdoor)이 유일한 분리점**: Flirds-1st AUROC **0.000**/Spearman
**0.000**(3 seed 전부 — spot-check 확인) = 완전 회피. **2차 Flirds가 0.917로 일부 버팀**(seed별 {0.75,1.0,1.0}).
loss-heur·(b) oracle·Banzhaf·FedIF·GTG·FLDetector·FLTrust·FedDQC = 1.0으로 잡음. → **핵심**: exact (b)
in-run oracle은 공격자에 *높은* φ를 줘 잡는데(AUROC 1.0), Flirds-1st는 *낮은* φ를 줘 "기여 높음"으로
오판한다 → 이건 "정직한 valuation 답"이 아니라 **Taylor 근사의 부호 실패**이고, 2차항이 그 부호를 상당부분
복원한다(§5.2에서 심화).

**LLM device100 (N=100, non-IID, 3-seed).** **확립**: valuation φ는 **non-IID에서 침식**된다. noisy:
Flirds 0.57~0.77, **FedDQC 1.0이 최강**. 결정적 spot-check — anchor α=0.5에서 **exact (b) per-round oracle
자체의 noisy AUROC도 0.604에 그친다**(3-seed; seed0=0.660. Flirds도 0.604로 oracle과 동률). 즉 φ-침식은 Flirds *근사*의
결함이 아니라 **valuation 자체의 내재 한계**(non-IID에서 clean 소수 클라가 상위로 보임). free-rider는
gradient 쓰는 방법(Flirds/Flirds-1st/loss-heur/FLTrust = 1.0)이 깔끔, FedDQC는 off-threat이라 0.14~0.57.
→ **위계 명제 "valuation ≠ 탐지기"를 데이터가 직접 지지**: 위협마다 *전용* 탐지기(품질=FedDQC, free-rider=
FLTrust)가 φ를 이긴다.

**LLM device100 poison (3-seed).** silo5와 달리 **회피 안 됨** — α=0 Flirds(2차) AUROC 1.0. cross-device
희석으로 backdoor 설치가 약함(ASR α0≈1.0/α0.5≈0.50). → poison-회피는 **설정 의존**이지 보편 현상 아님.

**CNN.** C2의 "탐지"는 φ-AUROC가 아니라 **개입-정확도 회복**(grad_noise 0.499→0.609)으로 측정 → LLM의
φ-as-detector와 **무대가 달라 직접 pool 불가**(§2.6).

### 1.5 Cost (비용) — 모양은 같고 스케일만 다름

**확립, 양 계열 공통**:
- **Flirds-1st 항상 최저가**(1 val-gradient/round, Hessian 無): LLM 1B std20 1531s, CNN MNIST 0.08s.
- **2차 Flirds 비용 모델 = cohort-독립**(1 HVP/round 고정), **(b) oracle = cohort-지수적**(2^k/round). 이
  비대칭이 우열을 무대마다 가른다:
  - **cohort 크면 Flirds가 oracle 압승**: device100 anchor(K=10) Flirds 157s vs (b) **25,000s** (≈160×);
    anchor5(N=5 full) 707s vs (b) 3528s (≈5×).
  - **cohort 작으면 (b)가 더 쌈**: std20은 라운드당 2명(2²=4 eval)이라 (b) per-round가 이미 저렴 →
    1B std20 Flirds(2차) 4697s **>** (b) 2917s (1 HVP가 4 forward보다 비쌈). 이 레짐에선 **Flirds-1st만 우위**.
- (a) retrain oracle = (b)의 **~9배**(1B anchor5 30,817s) — fidelity 비교군이 아니라 별도 GT.
- **CNN: 절대 regime이 sub-초~초**(Flirds 대개 <2s, 최대 ~14.6s), LLM은 분~시간. **Ripple은 압도적 dominated**(CNN ~1.1~11k s
  = 학습 자체보다 10~130×; "62× speedup" 원논문 주장과 정반대, [[baseline-original-paper-verification-2026-06-22]] §3.7).
- **CNN 특기**: Flirds·Flirds-1st가 traj_time(FL 학습 자체, ~80~94s)보다 1~3 자릿수 싸다 → "기여도 추정이
  학습보다 훨씬 저렴".

**비용 결론**: Flirds의 비용 우위는 **"라운드당 참여가 많아 exact 2^k가 비싼" 무대에서만** 나온다. 이게
near-additivity 긴장(§5.1)의 핵심 — clean에서 정확도가 동률이면 남는 차별점이 비용인데, 비용 우위조차
cohort-큰 무대 한정이다.

---

## 2. CNN vs LLM 모델 계열 대조 (전용 절)

> 결론은 데이터로만 판단(단정 금지). 갈리는 칸은 절대 pool 안 함.

### 2.1 clean fidelity 천장 — **갈린다 (가장 큰 차이)**
- **LLM**: 정직한 방법 거의 전부 1.0(천장). **CNN**: 큰 spread (Flirds 0.919 / loss-heur 0.860 /
  Flirds-1st 0.832 / GTG 0.569 / FedSV 0.401 / ComFedSV 0.348 / Ripple 0.373).
- **무엇이 갈리나**: recon-MC가 CNN에서 훨씬 약함. LLM에선 GTG 0.97~1.0인데 CNN에선 0.57.
- **왜 (가설, 단정 X)**: (1) **near-additivity 강도** — LLM-LoRA는 저차원·짧은 라운드(R=30, 10 steps)라
  효용이 거의 가법 → 모든 Shapley류가 선형성으로 degenerate-equal. CNN은 full-model·E=5 multi-epoch라
  효용이 더 non-additive → recon-MC의 sub-model 재구성이 빗나감. (2) **게임 난이도** — LLM에서 loss-heur
  (라운드별 val-loss 변화량, Shapley조차 아님)가 1.0 동률인 것은 *게임이 쉽다*는 신호일 수 있음(§5.1). CNN
  에선 loss-heur 0.860으로 천장 아래 → CNN 게임이 더 어렵고 그래서 method가 갈린다.

### 2.2 Banzhaf vs Flirds — **갈린다 (천장 효과로 역전·해소)**
- **CNN**: Banzhaf가 약간 앞(vs (b) Spearman 0.989/Pearson 0.998 vs Flirds 0.919/0.934). **LLM**: 동률 1.0
  (anchor5). **왜**: Banzhaf는 exact semivalue라 noise가 없어 CNN의 어려운 게임에서 깨끗하게 앞섬. LLM은
  게임이 near-additive·천장이라 둘 다 1.0으로 붙어 차이가 사라짐. **함의**: Banzhaf 우위는 CNN-고유이고
  *exact라 N≤10에서만* 가능 → 대규모 baseline으론 못 쓰는 우위(Flirds는 N=100에서도 천장 유지).

### 2.3 2차항(Hessian)의 효용 — **갈린다 (benign vs poison으로 역할이 다름)**
- **CNN clean**: 1차→2차가 **benign에서도 도움** (Flirds-1st 0.832 → Flirds 0.919, +0.087). non-additive
  CNN 효용에서 2차 곡률이 fidelity를 끌어올림.
- **LLM clean**: 1차·2차 둘 다 ~1.0 → benign에서 **2차 무의미**(천장).
- **LLM poison**: 여기서 2차가 *가른다* — silo5 1B Flirds-1st 0.000 → 2차 0.917(buy-back), 단 3B는 둘 다
  0.000(2차도 못 구함; 1-seed). device100 poison은 1차도 1.0(설치 약해서).
- **종합 (IRDS 확장 thesis 직결)**: IRDS 원논문은 "중앙 per-step에선 2차항 이득 미미"(Appx E.2.2)라 했는데,
  **우리는 2차가 (i) CNN non-additive benign과 (ii) LLM poison의 큰 scaled-update에서 결정적**임을 보임 —
  FL per-round multi-step은 중앙 per-step과 다르다는 우리 가설을 데이터가 지지([[baseline-original-paper-verification-2026-06-22]] §3.8). 단 3B poison 실패는 한계(큰 γ-점프엔 2차 Taylor도 부족; 1-seed).

### 2.4 dual-oracle (a) vs (b) — **갈린다 (CNN-고유 현상으로 보임)**
- **CNN**: 둘 다 실측, **괴리** — Flirds vs (b)=0.919 but vs (a)=0.352; *모든* method vs (a) ≤0.45
  (spot-check: Flirds 0.352, Flirds-1st 0.408, ShapleyFL 0.453, Banzhaf 0.355). 즉 (b)≈Flirds이고 (b)와 (a)가
  서로 다른 순위. **LLM**: 1B anchor5만 (a) 있고 (a) vs (b)=**0.933**(거의 일치); near-additive 방법들은
  천장 효과로 vs (a)도 0.933으로 동률.
- **왜 (가설)**: CNN full-model·multi-epoch retrain은 coalition마다 학습 *경로*가 달라져 retrain-counterfactual
  (a)이 single-trajectory 분해 (b)과 본질적으로 다른 게임이 됨. LLM-LoRA near-additive에선 두 게임이 수렴.
- **이것이 CNN-고유인지 = 미해결**: 3B/7B (a)가 없어 "LLM에서도 스케일 키우면 (a)≠(b)?"는 확인 불가(P2/P3).
  N=5 coarse + retrain noise도 0.933의 불완전성에 기여. **가장 중요한 개념적 갈림**(§5, §6).

### 2.5 비용 절대 regime — **같다 (모양), 다르다 (스케일)**
- 순서 동일: Flirds-1st < Flirds(2차) ≪ exact-2^N급 (b)≈Banzhaf≈ShapleyFL; GTG/FedSV는 사이; Ripple 압도적
  꼴찌. cohort-독립 vs cohort-지수 비대칭도 양쪽 공통. **스케일만**: CNN sub-초~초, LLM 분~시간.

### 2.6 탐지 무대 — **다르다 (직접 비교 불가)**
- **CNN(C2)**: 개입-정확도 회복(grad_noise 0.499→0.609)으로 "강건성"을 봄. **LLM(phase2)**: φ-as-detector
  AUROC + 전용탐지기. 두 트랙은 *측정 대상*이 달라(정확도 vs AUROC) pool 금지. 단 *방향*은 일관 — 오염이
  있어야 기여도 신호가 쓸모를 보임.

### 2.7 원인 후보 종합 (단정 X)
CNN↔LLM 갈림의 공통 뿌리 후보: **① near-additivity 강도**(LoRA 저차원·짧은 라운드 = 강한 가법성 → LLM
천장; CNN full-model = 약한 가법성 → spread·(a)/(b) 괴리·2차 benign 이득). **② 게임 난이도**(loss-heur가
LLM서 1.0 동률 = 쉬운 게임 신호; CNN 0.86 = 어려움). **③ N·참여**(LLM 5/20/100 vs CNN 10/100). **④ val
크기**(LLM tiny 10~20 vs CNN 2000 — LLM AUROC가 coarse한 이유, §6). near-additivity가 가장 많은 현상을
한 번에 설명(천장·Banzhaf역전·2차역할·(a)(b)괴리 전부).

---

## 3. claims → evidence → caveat 표

> 낼 만한 주장 후보별 (모델계열 | 지지 실험·수치 | 위협/한계). **확립/시사적** 구분 명시.

| # | 주장 | 모델계열 | 지지 (셀·수치) | 위협·한계 | 강도 |
|---|---|---|---|---|---|
| C1 | Flirds는 exact in-run Shapley를 충실히 근사 | LLM | track_d std20/anchor5 Spearman **1.000** (3-seed); device100 anchor vs 진짜 (b)perround **1.000** | clean/near-additive라 다수 method도 1.0 (차별 약) | **확립** |
| C2 | 〃 (CNN, 어려운 게임) | CNN | C1 vs (b) **0.919±.134** (30셀); Banzhaf만 더 높음(0.989) | (a) 대비는 0.35로 낮음(C9); iid 포함 pool | **확립** |
| C3 | 비용이 cohort-독립 → 참여 많을수록 oracle 압승 | 둘 다 | device100 anchor 157s vs (b) 25,000s (**160×**); anchor5 5×; CNN <2s vs (b) ~31s | std20(소cohort)에선 (b)가 더 쌈 → Flirds-1st만 우위 | **확립** |
| C4 | 2차 Hessian이 FL서 의미있다(중앙과 달리) | CNN benign + LLM poison | CNN 0.832→**0.919**; silo5 1B poison Flirds-1st 0.000→2차 **0.917** | 3B poison은 2차도 0.000(1-seed); LLM benign은 무의미 | **확립(CNN benign) / 시사적(LLM poison, n=3 silo만)** |
| C5 | clean-IID 기여도-가중은 do-no-harm | LLM | 전 arm MMLU/ROUGE vanilla ±0.001~0.003 (3-seed, 6셀) | parity일 뿐 *향상* 아님 | **확립(null)** |
| C6 | 오염 하에선 기여도-가중이 성능·수렴 회복 | CNN | grad_noise acc 0.499→0.609; rounds 27.2→13.7 | 그룹 pool std 큼; Flirds가 arm 중 최강은 아님(shapleyfl 0.645) | **확립(방향)** |
| C7 | valuation φ ≠ 탐지기 (전용탐지기가 이김) | LLM | device100 noisy φ 0.57~0.77 < FedDQC 1.0; **exact (b)도 0.604** | tiny val=10; 위계상 후순위라 의도된 결과 | **확립** |
| C8 | clean-보존 poison이 Flirds-1st를 회피 | LLM | silo5 1B Flirds-1st AUROC/Sp **0.000** (3-seed); 3B 둘 다 0.000 | device100 poison은 회피 안 됨(설정 의존); 3B 1-seed | **확립(silo5 1B) / 시사적(3B)** |
| C9 | retrain (a)·in-run (b) oracle은 갈린다 | CNN | Flirds vs (b)=0.919 vs (a)=0.352; 전 method vs (a)≤0.45 | LLM(1B)은 0.933 일치 → CNN-고유? 미확인; N=5 coarse | **확립(CNN) / 미해결(cross-model)** |
| C10 | LLM-scale client-level in-run Shapley fidelity 선행 공백 | LLM | taxonomy 빈칸(federated×LLM×valuation-E1); FedDQC/FedHDS/iPFL은 인접(quality/sel/market) | "관찰된 빈칸"이지 단정 노벨티 X | **시사적(문헌 관찰)** |

---

## 4. Novelty / 기여

verification·taxonomy와 엮은 *진짜* 새로운 것:

1. **LLM-scale client-level in-run Shapley의 fidelity 검증 (선행 공백).** [[prior-work-taxonomy/README]] §핵심관찰
   2: federated × LLM × client-level *valuation*을 **exact-SV oracle 대비 fidelity(E1)**로 검증한 연구가
   내부 노트·표적 웹(2024–26)에서 미확인. 그 칸의 현 점유자는 valuation이 아닌 인접 문제(FedDQC=per-sample
   quality, FedHDS=selection, iPFL=market). FL-Shapley 계열(FedSV·GTG·ComFedSV·ShapleyFL·Ripple)은 *전부
   CNN-분류*, LLM-attribution 계열(IRDS·DataInf·LESS·LoGra)은 *전부 centralized*. Flirds가 이 교차칸을 1B→7B
   3 스케일로 채우고 fidelity 1.000(vs 진짜 oracle)을 보임 → **단단한 기여**. (단정 노벨티 아닌 관찰 빈칸.)

2. **cohort-독립 비용 모델.** 2차 Flirds = 1 HVP/round로 *라운드당 참여 수와 무관*, exact oracle은 2^k/round.
   이 비대칭을 LLM/CNN 양쪽에서 정량화(device100 160×). FedIF가 "agg 450× 빠름"을 주장하지만 그건 Shapley가
   아닌 영향도-가중([[baseline-original-paper-verification-2026-06-22]] §3.5) → *Shapley-충실하면서 cohort-독립
   저비용*은 Flirds 고유.

3. **poison-회피 경계 + 2차항의 FL-특이적 효용.** clean-보존 backdoor에서 Flirds-1st가 회피되고 2차가
   복원하는 경계를 처음 측정. IRDS의 "중앙 per-step 2차 이득 미미"와 표면상 모순이나 무대가 달라 모순 아님
   (§2.3) → **2차항이 FL per-round에서 결정적**이라는 우리 thesis의 직접 증거. 단 회피 자체가 valuation의
   경계를 드러냄(§5.2) → 상보적 탐지기 필요성(loss-heur/(b)/FLDetector가 잡음)을 *데이터로* 보임.

4. **(부수) dual-oracle 괴리 발견.** in-run (b)과 retrain (a) Shapley가 CNN full-model에서 다른 게임이 됨
   (vs (a)≤0.45). 문헌 공백(FL에서 두 oracle을 같은 무대서 비교한 연구 희소) + Shapley 의미론에 대한 관찰.

> **정직 단서**: 1·4는 "관찰된 빈칸"·"발견"이지 정리(theorem)가 아님. 2·3이 가장 방어 가능한 정량 기여.

---

## 5. 놀라운 점·긴장 (§2가 안 가져간 cross-model 개념만)

### 5.1 near-additivity → "Flirds 차별점은 비용인가 정확도인가?"
clean·소규모에서 다수 method가 1.0 동률(LLM 전부, CNN Banzhaf/Flirds). **긴장**: 그러면 fidelity-정확도는
차별점이 아니고 남는 건 *비용*인데(§1.5), 비용 우위조차 cohort-큰 무대 한정. **정확도 차별이 실재하는 곳**:
(i) **poison** — Flirds(2차) Spearman 0.967 vs FedSV **0.367**, Flirds-1st 0.000 (near-additive 동률의 *첫
붕괴*); (ii) **device100 non-IID** — Flirds 1.000 vs GTG 0.78/FedSV 0.75/ComFedSV≈0. → **데이터의 답**: Flirds는
benign에서 *동률·더 쌈*, 비가법(poison/non-IID)에서 *더 정확*. "더 정확"을 헤드라인으로 쓰려면 비가법 무대를
전면에 둬야 함(clean 1.0 자랑은 trivial로 읽힘, §6).

### 5.2 poison 회피 — 한계인가 / 정직한 답인가 / 상보 탐지기 필요인가
옵션을 펼침(pre-position 금지):
- **"정직한 valuation 답" 가설은 데이터가 약화시킨다**: clean val-loss를 낮추는 공격자가 valuation상
  "기여 높음"이면 *같은 val-loss 게임*을 쓰는 exact (b) oracle·loss-heur도 그래야 한다. 그런데 **(b)·loss-heur는
  AUROC 1.0으로 잡는다** → val-loss 게임 자체는 공격자를 *낮은* 기여로 보지 않음. 회피하는 건 **Flirds-1st의
  Taylor 부호 실패**(큰 γ-scaled update에서 1차 선형화가 깨짐)이고, 2차가 부호를 복원(silo5 1B 0.917).
- → **읽기**: 이건 "valuation의 정직한 답"보다 **"1차 Taylor 근사의 한계 + 2차의 부분 복원"**에 가깝다. 단
  **3B에선 2차도 실패**(0.000, 1-seed) → 큰 모델·큰 점프엔 Taylor가 부족 → **상보적 탐지기 필요**(FLDetector/
  FLTrust/(b)/loss-heur가 보완). device100 poison은 아예 회피 안 됨(설치 약함) → **경계는 매트릭스**.
- **결론(미리 못 박지 말 것)**: poison-회피는 보편 한계가 아니라 *큰 scaled-update × 큰 모델*에서 Taylor가
  무너지는 조건부 현상. 실제-config + (b) oracle로 재확인 + 3B 3-seed 필요(§8 P5).

### 5.3 device100 non-IID 탐지 침식 → 위계와 정합
φ 0.57~0.77, FedDQC·FLTrust가 이김. **이게 위계 "valuation≠탐지" 명제와 정합**: 결정적으로, **exact (b)
oracle도 noisy AUROC 0.604**(3-seed; spot-check의 0.660은 seed0)이라 침식은 *근사*가 아닌 *valuation 본질*. 즉 "Flirds가 탐지에서
진다"가 아니라 "기여도 측정과 이상 탐지는 다른 목적이고, non-IID에선 정상 소수 클라가 기여-낮음=이상으로
오인된다"는 *설계상 예측된* 결과. → 비판이 아니라 위계의 실증.

### 5.4 proxy-truth 순환성 주의
off-anchor device100(α∈{0,0.01,0.1,5.0})의 Spearman은 **truth=Flirds 자기참조** → Flirds-1st·loss-heur의
"1.000"은 "*Flirds와 동일 순위*"이지 vs exact oracle 아님. **진짜 oracle 검증은 α=0.5 anchor 한 칸뿐**(거기선
vs (b)perround 1.000으로 genuine — spot-check 확인). 발표/논문에서 off-anchor 1.0을 fidelity 증거로 쓰면
순환 논증 → **anchor만 fidelity 주장, 나머지 α는 "탐지 AUROC의 α-의존"으로만** 쓸 것.

---

## 6. 위협·리뷰어 공격 (red-team)

1. **tiny val (10~20).** device100 φ-침식(0.57~0.77)·silo5 noisy AUROC가 전부 val=10~20에 올라탐 → AUROC
   coarse. 리뷰어: "0.604 vs 0.605 차이가 val=10 noise 안에 있지 않나?" → **방어**: 큰 val로 anchor 재실행
   1칸이라도 필요(§8). fidelity Spearman은 val 크기에 덜 민감(순위라) → fidelity 주장은 상대적으로 안전.
2. **3B robustness = 1-seed.** poison 3B 둘 다 0.000(C8), 3B robustness 전 수치가 n=1 → "3B에서 2차도
   실패"는 단일 관측. **방어 불가, 채워야 함**(P5).
3. **(a) oracle = 1B anchor5만.** dual-oracle 일치 0.933(LLM)·괴리(CNN)이 각각 한 스케일/한 무대 → "LLM에서
   (a)≈(b)"를 일반화 못 함. 3B/7B (a) 미실행(P2/P3). **CNN (a)≤0.45는 "어느 oracle이 GT냐"를 흔든다**(아래 6).
4. **near-additivity가 fidelity 1.0을 trivial하게 만드나?** 리뷰어 최강 공격: "Shapley 선형성 + 가법 효용이면
   exact=근사가 자명. 당신의 1.0은 게임이 쉬워서다 — loss-heur(Shapley조차 아님)도 1.0이지 않나?" →
   **부분 인정**: clean/소cohort fidelity 1.0은 약한 증거. **반론은 (a) 비가법 무대**(poison·device100서 동률
   붕괴: FedSV 0.367, GTG 0.78) **(b) 비용**(cohort-독립). 헤드라인을 clean-1.0이 아니라 *비가법-차별 + 비용*에
   둬야 함(§5.1).
5. **proxy-truth 순환성**(§5.4) — off-anchor 1.0 자기참조. 리뷰어가 바로 짚음 → 선제적 caveat 필수.
6. **"true contribution" 주장의 약점**: Flirds는 *in-run 게임 (b)*의 Taylor 근사이고 (b)는 exact in-run이라,
   vs (b) fidelity는 본질적으로 **"근사가 잘 됐나"** 측정(설계상 가까움). *deployment counterfactual*인 (a)와는
   CNN서 갈림(≤0.45) → **가장 단단한 주장은 "exact in-run Shapley를 충실·저렴 근사", "참 기여도(retrain)
   포착"은 약함**(특히 CNN). 논문 워딩을 이 선에서 보수적으로.

---

## 7. 내부 일관성 점검 (spot-check 결과)

overview가 스스로 플래그한/의심스러운 항목을 raw로 확인:

| 항목 | overview 진술 | spot-check (raw) | 판정 |
|---|---|---|---|
| 3B (a)≈0.900 | "프로젝트 노트엔 있으나 track_d rundir엔 없음 → 미수록"(caveat #2) | `grep (a)oracle … 3B` → **NONE** | ✅ 정확히 부재. 문서 처리 옳음 |
| 1B anchor5 dual-oracle | (a) vs (b) 0.933±.047 | per-seed {0.90, 1.00, 0.90} → mean **0.933** | ✅ 일치 |
| CNN Flirds vs (b) | 0.919±.134 (pool 30) | 재집계 **0.9192** | ✅ |
| CNN 2차 이득 | Flirds-1st 0.832 → Flirds 0.919 | 0.8315 / 0.9192 | ✅ |
| CNN (a) 괴리 | vs (a)≈0.35 | Flirds 0.352, Flirds-1st 0.408, ShapleyFL 0.453, 전부 ≤0.45 | ✅ |
| silo5 poison Flirds-1st | AUROC/Sp 0.000 | 3 seed 전부 0.0/0.0; 2차 {0.75,1.0,1.0}=0.917 | ✅ |
| 3B silo5 poison | Flirds·1st 둘 다 0.000 | Flirds Pearson −0.893, 1st −0.934 (Sp 0.0, AUROC 0.0) | ✅ |
| device100 anchor truth | α=0.5 = 진짜 (b)perround (off-anchor만 proxy) | a0.5 noisy 행 ref=**(b)perround**, Flirds Sp 1.000; (b)oracle AUROC 0.604(3-seed; seed0=0.660) | ✅ (앞선 loose-grep 오인 정정) |
| 3B track_d seed 수 | seeds=3 | fidelity.csv·rundir 모두 seed0/1/2 존재 | ✅ (06-22 verification 문서의 "2-seed"는 **stale** — 그 후 seed2 병합됨) |

**플래그**: 단 한 건 — **[[baseline-original-paper-verification-2026-06-22]] §63·§345의 "3B fidelity.csv는
2-seed(seed2 미병합)" 노트가 이제 stale**(현재 3-seed). 그 문서 갱신 권장(수치 결론엔 영향 없음, 이미
3-seed 기준 서술과 일치). 그 외 overview-내부 모순 없음.

---

## 8. 다음 실험 우선순위

현 결과가 *확립 못 한* 것 기준. (overview P# 재사용 + 신규)

| 순위 | 실험 | 메우는 갭 | 근거 |
|---|---|---|---|
| **P1** | **3B robustness 3-seed 완성** (현 1-seed) | C8(3B poison)·전 3B robust 수치가 n=1 → 통계 無 | red-team #2; 가장 싼 단단화(silo5 N=5) |
| **P2** | **device100 anchor를 큰 val(≥200)로 1칸 재실행** | tiny val=10이 φ-침식·oracle AUROC 0.604를 신뢰 못 하게 함 | red-team #1; "valuation≠탐지"(C7) 방어 |
| **P3** | **3B/7B anchor5 (a) retrain oracle** (P2/P3 in overview) | dual-oracle (a)≈(b)?가 1B 한 스케일뿐 → CNN-괴리가 LLM서도 스케일로 나타나나(§2.4) | red-team #3; C9 cross-model 미해결 핵심 |
| **P4** | **poison 실제-config + (b) oracle 재확인** (silo5·device100) | poison-회피가 Taylor 한계인지 매트릭스 경계인지(§5.2) | C8 caveat(tiny config); 헤드라인 결정 전 필수 |
| **P5** | **CNN (a)-oracle 게임 정의 재검** (왜 전 method ≤0.45?) | (a) retrain이 noise인지 진짜 다른 게임인지 → "참 기여 포착" 주장 가능성 | red-team #6; C9; novelty 4 단단화 |
| **P6** | **비가법 무대 확장**(poison 강도·non-IID α 더 촘촘) | 정확도-차별이 비가법서만 나오므로(§5.1) 그 영역을 두껍게 | C1/C2가 clean서 trivial 비판 방어; 헤드라인 근거 |
| P7(신규) | LLM N=10 (a)/(b) 고-power fidelity (overview P1) | N=5 coarse(0.933 불완전성 원인) → 깨끗한 dual-oracle | 비용 deferred지만 fidelity 1차의 결정적 power |

> **전략**: P1·P2가 가장 싸고(기존 무대 seed/val만 늘림) red-team 최강 공격 2개를 막음 → 먼저. P3·P5는
> 개념적으로 가장 값짐(dual-oracle cross-model)이나 비쌈. P4는 헤드라인(poison framing) 확정 전 게이트.

---

## 9. 그림/플롯 계획 (데크·논문 figure 후보)

1. **비용-fidelity frontier** (산점도, log-x). x=runtime, y=Spearman vs (b). LLM·CNN 각 1장. Flirds·Flirds-1st가
   좌상단(싸고 충실), (b)/Banzhaf/GTG/ShapleyFL 우측(비쌈), Ripple 극우(dominated). **near-additivity 긴장(§5.1)
   을 한 장으로** — clean에선 점들이 y=1.0에 뭉치고 x만 갈림(=비용이 차별). 수치: §3.5 runtime × §3.1 Spearman.
2. **AUROC-vs-α 침식 곡선** (device100 noisy). x=α{0,0.01,0.1,0.5,5.0}, y=AUROC. 선=Flirds/FedIF/FLTrust/
   FedDQC/(b)@anchor. **FedDQC 1.0 평탄 vs φ 0.57~0.77 침식** → "valuation≠탐지"(C7) 시각화. §3.4.2 (b1).
3. **poison 회피 막대** (silo5 1B·3B). method별 poison AUROC. Flirds-1st 0.000 / 2차 0.917(1B)·0.000(3B) vs
   loss-heur·(b)·FLDetector 1.0. **2차의 buy-back과 3B 실패를 나란히**(§2.3, §5.2). §3.4.1·§3.4.4.
4. **CNN vs LLM fidelity 대조** (그룹 막대). 같은 method가 CNN clean spread(0.35~0.99) vs LLM 천장(≈1.0).
   §2.1을 한 장으로 — recon-MC의 CNN 붕괴. §3.1.1 vs §3.1.2.
5. **dual-oracle 괴리** (CNN, vs(b) vs vs(a) 쌍 막대 또는 scatter). 전 method vs(a)≤0.45 천장 → (b)와 (a)가
   다른 게임. LLM 1B 0.933 한 점 병기. §2.4, §3.1.2.
6. **(보조) 2차항 이득** (CNN 시나리오별 Flirds-1st→Flirds Δ). benign 칸서 +0.087, poison 무대선 LLM으로
   교차참조. §3.1.2 시나리오 표.

> **헤드라인 figure 2장**: #1(비용-fidelity frontier)이 *주 기여*(저렴·충실), #2(AUROC-vs-α)가 *위계 명제*.
> #3(poison)·#5(dual-oracle)는 긴장/발견용 보조.

---

## 부록 — 이 분석이 *못* 한 것

- 전면 재집계 안 함(overview 신뢰 + 결론-критical 9개 spot-check만). per-seed 분포·정규성 검정 미실시 →
  "통계적 유의" 주장은 본 문서 범위 밖(특히 §1.3 수렴의 겹치는 std).
- (a)-oracle 게임 정의(왜 CNN서 ≤0.45)의 *원인* 규명은 코드/추가 실험 필요(P5) — 여기선 현상만 보고.
- Fairness·reward(E4)는 전용 실험 미설계(overview P6) → 다루지 않음.
