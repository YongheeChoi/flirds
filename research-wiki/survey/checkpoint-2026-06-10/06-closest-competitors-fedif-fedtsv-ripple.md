---
type: checkpoint
title: "Flirds 체크포인트 06 — 가장 가까운 경쟁자 3종 (FedIF · FedTSV · Ripple)"
created: 2026-06-10
updated: 2026-06-12
note: "내 연구의 직접 경쟁자 3편을 'LLM+FL+기여도평가' 교집합 축에 놓고 근접도+장단점 분석. 위키 노트(sources/*.md) 종합 + Ripple은 우리 구현 코드(ripple.py)를 직접 읽어 '2차항 없음' 주장을 정밀화(§6.5, ⓒ Yonghee 결정 대기)."
---

# 06 · 가장 가까운 경쟁자 3종 (FedIF · FedTSV · Ripple)

> **목적**: "FedIF, FedTSV, Ripple이 가장 큰 경쟁자"라는 인식을, *실제로 LLM+FL+기여도평가 교집합에 얼마나 가까운지* + *Flirds 대비 장단점*으로 정리한 포지셔닝 문서.
> **근거**: [[sources/fedif]] · [[sources/fedtsv]] · [[sources/ripple-shapley]] (노트) + `codes/flirds/baselines/ripple.py` (코드 직접 대조). baseline 전체의 PDF 1:1 대조는 [03-baselines-and-prior-work](03-baselines-and-prior-work.md) 참조 — 이 문서는 그중 **직접 경쟁 3종만** 깊게.
> **3-state**: 근접도/차별화 = ⓒ 설계·포지셔닝 락. §6.5 Ripple-2차 정밀화 = ⓒ **Yonghee 결정 대기**. Ripple 실측 dominated = ⓑ([[flirds.md|flirds]]:117).

---

## 6.1 축별 근접도 한눈에

세 경쟁자 모두 **FL+in-run 공간엔 있으나 셋 다 CNN/이미지 분류** — LLM·LoRA를 건드린 연구는 없음. "LLM" 축에서 Flirds는 단독. 차별화는 *2차 client-interaction + client-level Shapley + post-hoc(aggregation 무변경)*의 **교집합**.

| 축 | **Ripple** (AAAI'26) | **FedIF** (arXiv 2509, '25) | **FedTSV** (ECC'26) | **Flirds** |
|---|---|---|---|---|
| 도메인 | CNN, MNIST/CIFAR | CNN, CIFAR/F-MNIST | MLP/ResNet-20, MNIST/CIFAR | **LoRA-LLM 1B/3B/7B** |
| FL | ✓ | ✓ | ✓ | ✓ |
| 평가 단위 | **sample-level** | client | client | **client** |
| Shapley 공리 | ✓ (sample) | ✗ (TracIn 점수) | ✓ (per-round MC) | ✓ (client) |
| in-run (로그만) | ✓ | ✓ | △ (서버 val-training pass 추가) | ✓ |
| 2차/곡률 항 | **local-Hessian 전파 Jacobian** (있음, 종류 다름) | ✗ 순수 1차 | ✗ 0차 기하근접 | **val-loss HVP client-interaction** |
| closed-form | ✗ (재귀 Jacobian 저랭크) | ✗ (1차 내적) | ✗ (MC 샘플링) | ✓ (1+2차 Taylor) |
| aggregation 변경 | 안 함 (post-hoc) | **함** (robust agg) | **함** (fairness agg) | 안 함 (post-hoc) |
| 추가 통신 | 0 | 0 | **+서버 val-train/round** | 0 |
| 코드 공개 | ✗ | ✓ (`github.com/guojuntang/FedIF`) | (미확인) | — |

**근접도 순위(교집합 기준): Ripple > FedIF > FedTSV.** 위키 prior-work scan 결론과 일치 — "no single paper does the full intersection"([[flirds.md|flirds]]:243). 노벨티는 "first federated in-run"이 아니라(그건 Ripple이 이미 선점) **교집합**.

---

## 6.2 Ripple Shapley — "가장 직접적인 경쟁자"

**무엇**: 단일 run **sample-level** FL 기여도. `drop`(자기 라운드 즉시 marginal) + `ripple`(이후 라운드 재귀 전파, Jacobian chain $J_s\cdots J_{t+1}$). 저랭크 부분공간 근사로 tractable. Shapley 공리 보존. "62× 빠름"은 **vs AFedSV+/FedSV이지 GTG 아님**([[flirds.md|flirds]]:220). 실시간 data pricing 시연.

**근접도**: in-run+FL+Shapley공리 → **"first federated in-run" 자리 선점**. 단 (i) sample-level(client 아님), (ii) CNN 전용.

| Ripple이 가진 것 (장점) | Flirds 대비 단점 |
|---|---|
| cross-round 시간 전파를 **명시 모델링**(Jacobian chain) | granularity mismatch(sample vs client) → **SV 근사축 head-to-head ill-defined**; 비교는 학습성능축만([[2026-05-27-section-23-lock]]:140) |
| sample-level = 더 세밀 | 저랭크 Jacobian-subspace 가정 강함 |
| AAAI'26 강venue, 공간 선점 | **자기 논문에 SV ground-truth 없음**(task-driven만) → Flirds dual-oracle보다 약한 표준 |
| Shapley 4공리 sample-level 보존 | **실측 dominated**: LLM 포팅서 가장 느리고(~4515s≈42×) 가장 약한 noisy(AUROC 0.50±0.20 고분산)([[flirds.md|flirds]]:117); `eigsh`는 Track C1서 guard 적용(maxiter 상한+고정 v0+ncv 재시도+partial fallback, `phase0_verify_ripple` AUROC 1.0 = ⓐ; LLM-scale 교정/제외는 별도 세션, plan §3.11 결정⑥) |

→ Ripple 이기는 그림 = (1) **학습성능축** head-to-head, (2) **2차 종류** 차별화(§6.5), (3) 보너스 reduction(§6.6).

---

## 6.3 FedIF — "1차로 충분한가?"의 핵심 baseline

**무엇**: TracIn 스타일 **순수 1차** FL 데이터평가 + robust aggregation. 매 라운드 클라 **L2-정규화 업데이트** · **validation gradient** 내적 → min-max → EMA → adaptive weight. TracIn을 FL 클라업데이트에 처음 도입. SV 대비 aggregation 비용 **450× 절감**(단, **aggregation-time만**; training-time은 동등 — [[sources/fedif]]:80). CNN 전용, Hessian/2차/LoRA/LLM 전무. **코드 공개**.

**근접도**: **Ripple 다음으로 가깝다.** client-level + in-run + $\Delta w$ 위 + val-gradient anchor → Flirds 1차항과 거의 동일. FedIF 라운드 영향력 $\Phi_i^t=\frac{\Delta w_i}{\lVert\Delta w_i\rVert}\cdot\nabla\ell_\text{val}$ = **Flirds 1차항의 L2-정규화**(≈ FLTrust cosine). "Flirds-1st ≈ FedIF"는 **1차 val-grad 내적이라는 골격에 한정**되는 대응 — FedIF는 **독립 baseline으로 포팅·실측됨**(real grid 22셀 영속화, `runs/phase2_matrix/rundirs`), head-to-head는 ⓑ. 실측에선 L2-정규화 차이가 갈라짐(예: silo5_poison FedIF AUROC 1.000 vs Flirds-1st 0.000·Flirds 2차 0.917±0.118; 1B_device100-a0.0_noisy FedIF 0.973±0.017 vs Flirds-1st 0.772±0.058).

| FedIF이 가진 것 (장점) | Flirds 대비 단점 |
|---|---|
| **코드 공개** → 진짜 head-to-head 가능 | **엄격히 1차** — 2차/HVP interaction 없음; 저자가 "FL서 Hessian infeasible" **명시 회피** → 정확히 Flirds 차별점 |
| 이론 보장(Thm1: noisy서 FedAvg보다 tight한 1-step loss bound) | **Shapley 아님**(EMA+min-max 점수→weight; 공리 버림; min-max=cohort/순서 의존) |
| robust-agg 알고리즘 → **모델 자체 개선**(Flirds는 안 함) | **aggregation을 바꿈**(모델 trajectory 변경); Flirds는 vanilla FedAvg 위 post-hoc |
| | CNN 전용, LoRA/LLM 없음 |
| | **PGD blind spot**: 방향보존 poison은 방향-only 1차로 못 잡음(논문 인정 — [[sources/fedif]]:44) |

→ **차별화 실험 #13(PGD/direction-aligned poison)**: FedIF 공인 blind spot에서 Flirds **2차(곡률) 항**이 분리하는지 + `Flirds-1st-only` 짝지어 "2nd>1st"를 robustness 축에서 직접 증명([[flirds.md|flirds]]:149). FedIF의 future-work("update gradient 정보 더 쓰자")가 문자 그대로 2차 방향을 가리킴([[sources/fedif]]:72).

---

## 6.4 FedTSV — 같은 공간, "다른 종류의 물건"

**무엇**: Trajectory Shapley Value(TSV). 코얼리션 평균 업데이트가 **validation-reference 업데이트**(서버가 held-out val에 같은 $K$ SGD step)와 얼마나 정렬되는지로 per-round 점수. bounded geometric utility $v^t(S)=(1+\text{Dist}(\Delta_S,\Delta_\text{val})^2/\sigma)^{-1}$ → MC Shapley → 누적 → **adaptive FedAvg weight**. fairness/robust-**aggregation** 방법. MLP/ResNet-20.

**근접도**: 개념적으론 인접(client-level per-round SV + val anchor) 하지만 **scoop risk LOW** — 물건의 *종류*가 다름([[sources/fedtsv]]:60).

| FedTSV이 가진 것 (장점) | Flirds 대비 단점 |
|---|---|
| per-round MC Shapley + linearity 보존 | **평가 회계 아니라 aggregation 방법** — 점수 목적이 *다음 aggregation 조종*; 모델 trajectory 변경 |
| fairness framing(late/infrequent 참여 회계) | **0차 기하근접** — val-loss를 파라미터로 **미분 안 함**; gradient/Hessian/closed-form/2차 전무, MC 의존 |
| ECC'26 peer-reviewed | **서버 추가연산**: 매 라운드 **val training pass($K$ SGD)** + MC coalition; Flirds는 1 HVP/round + 받은 $\Delta w$만(통신 0) |
| $\sigma$ 정규화로 CGSV magnitude 민감성 교정 | CNN/MLP 전용; 수렴증명 없음; threat 가벼움(고정 label-shuffle만); oracle GT 없음 |

→ 직접 comparand보다 **대조군**: "valuation이지 weighting 아님 / closed-form이지 MC 아님 / 2차 interaction이지 방향근접 아님 / 서버추가연산 0"을 날카롭게 함.

---

## 6.5 ★ Ripple "2차항 없음" 주장 정밀화 (코드 근거) — ⓒ Yonghee 결정

**문제**: Yonghee의 2026-05-27 포지셔닝 — **"Ripple은 2차 근사항을 사용 안 했다는 점이 FL서 큰 불이익"**([[2026-05-27-section-23-lock]]:137) — 은 **그대로 쓰면 reviewer에게 반박당함.** 우리 자신의 faithful 재구현이 반증:

```python
# codes/flirds/baselines/ripple.py:191  (Eq 18 factor)
Ms = [np.eye(m) - lr * (Bs[t] * Ls[t]) @ Bs[t].T for t in range(rounds)]   # = I − η·H_local (저랭크)
```

Ripple의 ripple term Jacobian은 정확히 $M_t = I - \eta H_\text{local}$ — 각 클라 **로컬 데이터 Hessian** top-k eigen-sketch(`ripple.py:84` `local_hessian_topk`, `eigsh`). **Ripple은 2차/곡률을 쓴다.** "2차 없음"은 Eq 18을 본 reviewer가 바로 칠 약점.

**방어 가능한 정밀 차별화 = "2차의 *종류*가 다르다"**:

| | **Ripple의 2차** | **Flirds의 2차** |
|---|---|---|
| 어떤 Hessian? | per-client **local-data** loss | **validation** loss |
| 역할 | $I-\eta H$ Jacobian = **cross-round 시간 전파** | **within-round client-interaction**(coalition 곡률) |
| 클라 결합? | **없음** — 각 클라 1차 drop을 자기 $\Delta w_c$에 *선형* 전파 (`ripple.py:205`: `Δw_c·(전파된 val-grad)`, $j\neq k$ 교차항 없음) | **있음** — val-loss Taylor의 $\Delta w_j^\top H_\text{val}\Delta w_k$ pairwise 교차항 |

→ Ripple 곡률 = "**내 업데이트를 모델 곡률로 미래 전파**". Flirds 2차 = "**같은 라운드서 클라들 업데이트가 val-loss 곡률로 *서로 상호작용***". **다른 물건.** "non-IID FL서 client interaction 중요" 주장과 정확히 맞물리고(Ripple은 within-round 코얼리션 상호작용을 *구조적으로* 못 담음), 실측 dominated에 **메커니즘 설명**까지 붙음.

> **정직 단서**: Ripple의 전파 연산자 $\Pi(I-\eta H_\text{local})$는 *코호트의* local Hessian으로 구성되므로 약한 결합은 있음. 하지만 (a) val-loss Hessian이 아니라 local-data Hessian, (b) 시간 전파지 within-round 분해 아님, (c) 클라 기여가 자기 $\Delta w_c$에 *선형*(pairwise $\Delta w_j\cdot H\cdot\Delta w_k$ 없음) — 세 가지로 객체가 다름. 과장 없이 이 3점으로 주장.

**추천(ⓒ)**: related-work/§4에서 "Ripple lacks 2nd-order" → **"Ripple's curvature is a per-client local-Hessian propagation Jacobian with no within-round client-interaction term"**으로 교체. 위키 [[sources/ripple-shapley]] · [[flirds.md|flirds]] differentiator도 이 구분 미명시 → 같은 함정 잔존. **Yonghee 승인 시 위키 반영.**

---

## 6.6 종합 — 어디서 이기고 무엇을 보여야 하나

- **Ripple** = 가장 직접 경쟁자지만 sample-level+CNN+자기-SV-GT 없음 + **이미 실측 dominated**. 비교 경로(plan §3.11 Track C/D): (1) **C1 CNN fidelity 비교군에 포함**(N=10 full, 듀얼 oracle (a) 2¹⁰ retrain+(b) exact, Spearman/Kendall+GTG 거리; eigsh guard), (2) **C2 개입(학습성능) arm에서는 제외**(drop+ripple는 미래-round 참조=비인과 → 온라인 개입 불가; Ripple 논문 자체가 fidelity 거부+acc-vs-round 메서드라 개입 프레임과 부정합 — Yonghee 결정), (3) **within-round vs cross-round 2차 종류** 차별화(§6.5), (4) 보너스 — LoRA+2-term Taylor 하 Ripple drop+ripple → Flirds 1+2차 reduction(닫히면 Proposition, 안 닫혀도 손해 없음 — [[flirds.md|flirds]]:128). Track C 구현 완료=ⓐ(단위검증 green, 실측 없음), Track D=ⓒ 설계만.
- **FedIF** = **"1차로 충분한가?" 그 baseline.** `Flirds-1st-only`가 사실상 FedIF. 결정 실험 = **#13 PGD**(공인 blind spot서 2차 분리 → "2nd>1st" robustness 축 직접 증명). 코드 공개라 진짜 숫자 가능.
- **FedTSV** = **다른 quadrant(adaptive aggregation/fairness).** 직접 comparand보다 "valuation/closed-form/2차-interaction/통신0"을 날카롭게 하는 **대조군**.

**한 줄**: 셋 다 LLM 축에선 멀고(전부 CNN), FL+in-run 축에선 가깝다. 진짜 경쟁은 **2차 client-interaction × client-level Shapley × post-hoc(LoRA-LLM)** 교집합 — 거기서 Ripple은 "다른 종류의 2차", FedIF는 "1차뿐", FedTSV는 "aggregation 물건"으로 갈라진다.

---

*근거 추적: 노트 = `wiki/sources/{fedif,fedtsv,ripple-shapley}.md`; 코드 = `codes/flirds/baselines/ripple.py:{84,191,205}`; Yonghee 포지셔닝 = `raw/conversations/flirds/2026-05-27-section-23-lock.md:137`; 실측 = [[flirds.md|flirds]]:117. 전체 baseline 대조 = [03-baselines-and-prior-work](03-baselines-and-prior-work.md).*
