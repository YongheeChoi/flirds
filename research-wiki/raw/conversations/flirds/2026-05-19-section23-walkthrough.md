---
type: conversation
date: 2026-05-19
topic: flirds
participants: [Yonghee, Claude]
tags: [section-2, section-3, raw, lock-discussion, restored-from-jsonl]
---

# 2026-05-19 ~ 2026-05-22 — Section 2 / Section 3 walkthrough (raw transcript)

Raw turn-by-turn transcript of the Yonghee ↔ Claude design conversation that drove the Section 2 / Section 3 lock discussion. The session was interrupted between 2026-05-22 and 2026-05-27 (Yonghee's UI froze); transcript restored from the Claude Desktop JSONL session log.

This is the **raw record**. Distilled version (decisions, preferences, what got locked and why): [[2026-05-27-section-23-lock]]. For the resulting project state: [[../../../wiki/flirds]] + [[../../../wiki/flirds-protocol]].

Tool calls (Read / Bash / etc.), tool results, and thinking blocks are stripped — only Yonghee's messages and Claude's text responses are preserved. The 2026-05-27 resumption turn (which directly produced the wiki distill) is **not** in this file — it lives in the current session's transcript and is summarized in the distill.

---

## [2026-05-19 08:08] Yonghee

연구 계획을 이어서 논의해보자. 지금 나는 Flirds 연구 정리 html 파일 기준으로 현황을 파악중이야. 결정이 필요한 부분을 먼저 다 정리하고 보강이 필요한 부분으로 넘어가자.

## [2026-05-19 08:09] Claude

연구 계획 논의를 이어가기 전에, 먼저 현재 상태를 파악하겠습니다. 메모리와 wiki 현황, 그리고 말씀하신 HTML 파일을 찾아보겠습니다.

## [2026-05-19 08:09] Claude

메모리가 아직 없습니다. HTML 파일과 wiki 현황을 함께 읽어보겠습니다.

## [2026-05-19 08:09] Claude

현황 파악을 위해 1차 기록인 design conversation 4편을 읽겠습니다.

## [2026-05-19 08:12] Claude

현황 파악 완료 — `flirds-summary.html`, `wiki/flirds.md`, 그리고 1차 기록 `conversation1–4.md` 모두 읽었습니다. 요청대로 **Section 2(결정이 필요한 부분)를 먼저 전부 정리**하겠습니다.

## Section 2 — 결정 항목 전체 맵

### Group A · 이미 실질적으로 닫힘 (재논의 불필요, lock만)

| 항목 | 상태 | 처리 |
|---|---|---|
| **① Noise vs OOD-good** | 보류 확정 (한계로 재프레이밍, 2026-05-18) | 결정 끝. 단 파생 의무 1건(**non-IID bias 정량화**)이 Section 3로 넘어감 — 이건 결정이 아니라 실험 의무 |
| **⑤ Ground-truth 정의** | 잠정 결정 | conv3 §2에서 논리 완결됨: (b) MC in-run = 主 oracle, (a) retraining = "different utility" 단서 단 sanity 1장. **그대로 lock 권장, 재논의 가치 없음** |
| **⑥ Benchmarks** | 목록 확정 | ① 보류로 OOD 분리 실험만 빠짐. 나머지 생존. 결정 아님 |

### Group B · 진짜 열린 결정

핵심 통찰: **②③④는 독립 3개가 아니라 2개의 전략 레버로 collapse됩니다.**

- **레버 1 = Flirds의 主 use case.** 이게 ④(trajectory dependence)를 *직접* 결정합니다. research-side면 trajectory 의존성은 IRDS처럼 feature(보류 종결), incentive면 bug → multi-run averaging 필요(통신·복잡도↑, "통신 0" claim 약화). 동시에 전체 narrative와 ③의 무게를 결정.
- **레버 2 = contribution scope.** ②(cancellation)·③(magnitude vs alignment)를 *풀 것이냐*, ① 선례처럼 *특성화된 한계로 둘 것이냐*. ①을 보류해 "기여 1개 + 특성화된 한계"로 좁힌 직후라, ②③를 풀면 그 보류 취지와 정면 충돌합니다.
- **⑦ dataset**: use-case가 정해지면 자동으로 좁혀짐 → 지금 hard decision 아님. 권장안: Llama-3.2-1B 主 / 3B scale-check, cross-silo 도메인-귀속 데모는 {medical, legal, code, math, general} 6–8 큐레이션 mix, cross-device 규모는 Super-NaturalInstructions. use-case 확정 후 lock.
- **2.3 Ripple positioning**: 별도 레버(포지셔닝 공격성). reduction 시도 자체는 Section 3 작업이고, 여기서 정할 건 *얼마나 load-bearing하게 주장하느냐*.

**내 권장 (근거+trade-off):**
- 레버 1 → **research-side in-run attribution**. IRDS 계보 그대로 상속, trajectory 의존성을 feature로 명시(IRDS framing 인용), multi-run 불필요 → "통신 0"·scope 유지. trade-off: incentive 응용은 "논의된 application + caveat"으로 격하.
- 레버 2 → **②③ 둘 다 특성화된 한계**. ① 선례와 일관, 서사 = "닫힌형 client-level 추정기 + drift-residual proposition" 1개 + 정직하게 특성화된 IRDS-상속 한계 세트. trade-off: novel-method 포인트는 줄지만 reviewer 리스크·scope 최소.
- Ripple → **다른 설계점이 主 주장, reduction은 성공 시 보너스**. derivation이 안 닫혀도 논문이 인질 안 잡힘.

이 3개만 정하면 Section 2가 완전히 닫히고 Section 3(보강)로 넘어갈 수 있습니다. 레버 1과 2는 상호작용하니(특히 incentive 선택 시 ③의 무게가 달라짐) 함께 답해 주세요.

## [2026-05-19 08:24] Claude

Q1(research-side)·Q3(다른 설계점)는 잠겼습니다. Q2를 정하기 전에 ②③를 구체적으로 풀어 설명하겠습니다 — 그리고 **Q1을 research-side로 잠근 사실이 Q2 답을 거의 강제**한다는 게 핵심입니다.

## ② Cancellation effect (IRDS Appendix A.3)

**메커니즘.** Flirds 1차항 $-\nabla\ell(w^r,z^{val})\cdot\Delta w_k$ 는 전체 LoRA 파라미터 벡터 위의 내적입니다. 이를 레이어별로 쪼개면 보통 **마지막 블록(LM head 인접) 성분의 gradient norm이 압도적**입니다. IRDS A.3가 지적한 건: 마지막 레이어에서 서로 다른 example들의 influence가 *큰 크기로 부호가 엇갈려 상쇄*된다 → 합은 작은데 개별 항은 큰, "큰 수의 작은 차" 구조라 per-unit 귀속이 불안정.

**non-IID에서 증폭.** 클라이언트마다 라벨 분포가 다르므로 $\Delta w_k$의 마지막-레이어 성분이 *클라이언트 고유 방향*으로 크게 튑니다. 그래서 $\phi_k$ ≈ (마지막-레이어 항, 크고 라운드마다 부호 요동) + (앞 레이어들의 작은 기여). 즉 client score가 **데이터 품질이 아니라 라벨-분포 skew**에 의해 지배될 수 있음.

**풀려면 (option 2):** layer-wise 정규화(레이어별 norm 역가중) / 특정 레이어만 사용(LM head 제외) / per-layer Shapley 후 가중합.

**왜 spine을 깨나 (research-side에서 특히):** drift-residual **Proposition은 $\Delta w_k$를 단일 벡터로 두고** $w^r$ 기준 전개해서 유도됩니다(전개 #2). 레이어 가중 $\Delta w_k\mapsto \mathrm{diag}(c_\ell)\Delta w_k$를 넣으면 conv3 §4의 **granularity-불변 lemma**("client-level = data-level을 client로 합산 + drift residual")가 더 이상 성립하지 않아 — *논문의 핵심 수학 기여인 Proposition을 새로 증명하거나 잃습니다*. 게다가 레이어 선택은 reviewer가 cherry-pick으로 공격하는 손잡이. IRDS 본인도 A.3에서 *언급만* 하고 안 풉니다.

**특성화 = 거의 공짜.** $\phi_k^{(r)}$의 per-layer 분해를 그려 마지막-레이어 항의 cross-round 부호 요동(non-IID vs IID)을 보이고, "마지막 레이어 제외"를 *진단용*으로만 써서 ranking이 얼마나 흔들리는지 = 한계의 크기. 이미 돌릴 run으로 측정 가능.

## ③ Magnitude vs alignment 혼동 (IRDS 한계 (b))

**메커니즘.** 1차항은 정렬(코사인)뿐 아니라 **크기 $\|\Delta w_k\|$에도 비례**합니다. 이미 잘 fit된(=학습 후반, 또는 모델이 이미 배운) 데이터는 gradient가 작아 → 완벽히 정렬돼 있어도 낮은 점수. FL에선 **늦게 합류한 클라이언트가 join 시점만으로 저평가**(품질이 아니라 타이밍 때문, conv1 (b)·conv3 late-joiner). 타이밍/참여를 가치와 혼동.

**풀려면 (option 2):** phase-normalization(라운드별 스케일로 나눔) / cumulative-reduction tracking / 정렬만 쓰는 변형.

**왜 spine을 깨나 — ②보다 더 치명적:** Flirds 추정기는 $U_{total}(S)=\sum_r[\ell(w^r+\sum p_k\Delta w_k,z^{val})-\ell(w^r,z^{val})]$의 Taylor 근사이고, **(b) MC oracle은 바로 이 동일한 $U_{total}$의 MC**입니다. 1차항의 크기는 *문자 그대로 loss 변화량*. 라운드 r을 $s_r$로 나누면 추정기는 $\tilde U\ne U_{total}$의 Shapley가 됩니다 →
(i) (b) oracle도 $\tilde U$로 재정의 → "in-run validation loss 감소" 헤드라인 해석 상실, 또는
(ii) oracle은 $U_{total}$ 유지 → Flirds-정규화가 *자기 oracle과 일부러 불일치* → "네 method가 네가 주장하는 양을 추정 안 한다"는 정면 공격.
어느 쪽이든 논문 척추가 갈라집니다.

**research-side에서는 ③가 애초에 bug가 아님.** "이 run에서의 marginal 기여를 귀속한다"가 정직한 질문이면, 거의 평평해진 loss에 기여한 클라이언트가 작은 값을 받는 건 **정직한 답**입니다. phase-normalization은 research-side가 의도적으로 버린 "데이터-내재 품질" 개념을 다시 주입 → 방금 잠근 Q1과 *모순*.

**특성화 = 이미 계획된 실험.** late-joiner 실험(conv3 §1) + 계획된 $E$-sweep×$\alpha$-sweep로 "고정 품질에서 join-round vs score" 곡선을 그리면 그 gap이 한계의 크기. 추가 비용 거의 0.

## 종합 — Q1=research-side가 만드는 결론

| | ② cancellation | ③ magnitude/alignment |
|---|---|---|
| 풀 때 깨는 것 | Proposition 증명·granularity lemma + cherry-pick 손잡이 | **추정기↔(b) oracle 정합성**(논문 척추) |
| research-side에서 | IRDS처럼 "언급된 한계" | **bug 아닌 정직한 답** — 풀면 Q1과 모순 |
| 특성화 비용 | per-layer 분해(기존 run) ≈ 0 | late-joiner/α-sweep(이미 계획) ≈ 0 |

원래 option 3("③만 풀기")은 incentive 분기 전용이었는데 Q1=research-side라 **소멸**했습니다. 남은 건 사실상 "둘 다 특성화"(권장, Q1이 더 강하게 지지) vs "spine 비용 감수하고 푼다"입니다.

## [2026-05-22 07:04] Claude

Q2 잠금: **default = IRDS-form 둘 다 (option 1 특성화), 변형(layer-wise/phase-norm)은 downstream 성능 기준 ablation으로 비교.** spine 안 깨고 변형은 비교군으로 — Q1·Q3와 완전 정합. 

그리고 실험 데이터가 wiki/HTML 시점(05-19) 이후 새로 들어왔으니, **Section 2 결정 맵을 데이터로 재확인**하고, **Section 3 우선순위를 새 사실에 맞게 재구성**하겠습니다.

## 결정 맵 — 실험으로 재확인

| 항목 | wiki 예측 → 실측 | 결정 영향 |
|---|---|---|
| ① noisy 탐지 보류 | "분리 안 됨" 가설 → **AUROC=0.50±0.14 (chance)** | 한계 재프레이밍이 *data-backed*. 더 강해짐 |
| ② cancellation 한계 | "관찰될 것" → **ρ(Flirds, exact)=1.0000 across α∈{0.1,0.5,5.0}** | 한계가 *materialize 안 함*. 프레이밍 = "이론적 가능성, 본 scale·세팅에서 ranking 손상 0" |
| Prop 2 (cubic-in-R_r bound) | 계획만 → **B4 E-sweep slope으로 확인** | spine 강화 |
| Theorem 1 + ρ=1.0 evidence | wiki 미명시 → **9.3/10 spine** | "planned"→"established"로 승격, Section 1 편입 |
| ⑤ ground-truth lock | (b) MC 主 + (a) sanity | ρ=1.0이 (b) sanity 동시 수행 → **lock** |
| ⑦ scale | 1B/3B 후보 | **Llama-3.2-1B + Qwen-3B 동시 진행, 부분 확정** |

**내가 빗나간 예측 — 솔직 인정.** ②의 한계 *크기*를 과대평가했고(ρ=1.0이면 paper에서 "한계로 관찰됨" 어조는 부적절), ③의 E-monotonicity는 틀렸습니다(U-shape: E=5 worst, E=10 회복). 전략 레벨(framing·scope·spine)은 맞았으나 한계 크기와 empirical 패턴 양적 예측은 보수적으로 가야겠습니다.

## 실험이 드러낸 새 결정 4개 (Section 2→3 경계 재조정)

기존 Section 3 일부는 *이미 끝났거나 결과가 다름* — 새로 정해야 할 것들:

- **N1. Prop 1 (drift residual monotonic) 처리.** 실험으로 falsified, U-shape 관측(E=10에서 divergence sign-dominance로 회복). 논문 일관성 직결이라 가장 시급.
- **N2. Scale 주장 한도.** 1B+3B로 마감 vs 7B 1점 추가 vs 13B까지. 비용·timeline 트레이드오프.
- **N3. ρ=1.0 사전 방어.** 결과가 "너무 좋아" reviewer 의심 유발 가능. stress-test로 ρ 점진적 degradation 시도할지.
- **N4. Q2 변형 비교 metric.** layer-wise/phase-norm 변형의 "model 성능 향상 기여"를 무엇으로 측정.

bf16 artefact는 결정 아님 — Section 3에 "fp32 eval 강제 + LR×precision pair 명시" 프로토콜로 박으면 됩니다.

## Section 3 (보강) — 재우선순위 제안

| 우선도 | 항목 | 비고 |
|---|---|---|
| ★★★ | **Prop 1 처리** (N1 결정 후) | 현재 falsified, 논문 spine 일관성 |
| ★★★ | **Ripple head-to-head + reduction (보너스)** | Q3 결정 — scoop 방어 |
| ★★★ | **Baseline run set**: GTG-Shapley, FedSV(=Wang 2020), ComFedSV, Data-Banzhaf-in-FL, Ripple, loss-heuristic | reviewer 필수 |
| ★★ | **Q2 변형 비교**: 기본 Flirds vs layer-wise vs phase-norm (N4 metric으로) | 사용자 명시 |
| ★★ | **②③ 특성화 보고**: ②=ρ=1.0으로 cancellation 무력화 *입증*; ③=U-shape으로 magnitude 혼동 *측정* | 한계 정직 보고 |
| ★★ | **α-sweep × E-sweep 매트릭스 정리** | drift-residual study |
| ★ | **scale extrapolation** (N2 결정 후) | |
| ★ | **bf16/precision 프로토콜 lock** | 재현성 |
| ★ | **stress-test 라인** (N3 결정 후) | reviewer 사전 방어 |

이 4개에 답해주시면 Section 2가 완전히 닫히고 Section 3·실험 실행 계획·wiki 업데이트(flirds.md / log / summary HTML)까지 한 번에 정합됩니다.

## [2026-05-22 07:18] Claude

N3·N4 잠금했고, N2=13B+ 결정 받았습니다 (operational form은 아래에서 정리). **N1만 자세히 풀어드리고 재선택**받겠습니다.

## N1 — Proposition 1 falsified, 무엇을 할 것인가

### 원래 계획 vs 실측

| | 원래 Prop 1 (planned) | 실측 (1B + 3B 동시) |
|---|---|---|
| 진술 | drift residual은 E에 monotone하게 커지고, valuation 품질도 monotone하게 떨어진다 | E=1 정상 → E=3 약화 → **E=5 worst** → **E=10 회복** (U-shape) |
| 메커니즘 | local-trajectory length × Hessian-norm으로 깨끗하게 증가 | **divergence sign-dominance**: E=10에선 residual이 너무 커져 Δw_k의 *방향*이 client-identity로 수렴 → magnitude는 망가졌지만 *순위*는 보존 |

요컨대 **bound 자체(Prop 2 cubic-in-R_r)는 살아있고 empirically 확인됨**. 깨진 건 그 bound 위에 얹은 *해석*(품질이 monotone하게 나빠진다)일 뿐.

### 세 가지 처리법 — 트레이드오프

**(a) Drop + U-shape을 empirical finding으로 보고**
- Prop 1 진술을 약화: "drift residual 크기는 $O(\eta E L_H \cdot \text{traj-length})$ — *상한*만 진술" (현재 확인된 사실만). Prop 2(cubic bound)가 rigorous spine을 담당.
- U-shape은 별도 figure + informal 설명("E=10에서 residual이 magnitude를 dominate하지만 방향은 client-identity로 수렴해 순위 회복") + "이 회복 메커니즘의 엄밀 이론은 future work."
- **장점**: 즉시 paper-ready. 13B+ 추가로 timeline 빠듯한 상황에 안전. U-shape이 오히려 *긍정적 발견*("Flirds는 큰 E에서도 무너지지 않음")이 됨.
- **단점**: 회복의 *이유*가 informal로 남음. 리뷰어가 "이론 왜 없냐" 물을 여지.

**(b) 재유도 — U-shape을 예측하는 새 정리**
- 후보 형태: $\text{error}(E) \le C_1/E + C_2 E^2$ 같이 E에 비단조 bound. minimum이 중간에. 또는 sign-dominance를 정량화하는 별도 lemma + 결합해 ρ recovery 정리.
- **장점**: 성공 시 theory contribution 크게 강화. 헤드라인이 "non-trivial U-shape with closed-form prediction".
- **단점**: 비단조 FL-drift bound는 표준 문헌에 없음. 새 derivation이 깨끗이 닫힌다는 보장 없고, 13B+ 실험·writing과 병행하기엔 timeline 위험. 닫히지 않으면 (a)로 backup.

**(c) Conditional formulation — "특정 가정 하 monotone, 일반에선 U-shape"**
- "Assumption A (예: 작은 Hessian Lipschitz / bounded curvature) 하에선 Prop 1 monotone 성립; 일반에선 sign-dominance로 U-shape" 같은 진술.
- **장점**: 안전, 부분적으로 theory 살림.
- **단점**: 가정이 실험에서 *깨졌으므로* conditional theorem이 자기 실험에 적용 안 됨 — 리뷰어 입장에선 "이 정리는 너희 setup에 안 맞는데 왜 있냐". *worst-of-both* 위험.

### 내 권장: (a) Drop + empirical 보고

이유:
- Theorem 1 + Prop 2가 이미 spine을 짊어짐(ρ=1.0 evidence + cubic bound 확인). 추가 정리 없이도 "method/theory track 9.3/10."
- 13B+가 timeline에 들어왔으니 *theory backlog 추가는 비용 큼*. (b)는 닫히면 좋지만 인질 잡힐 만한 노력은 아님.
- U-shape은 **긍정 시그널로 reframe 가능**: "naïve bound는 monotone degradation 예측하나, 실측은 E=10까지도 ρ 회복 — Flirds는 큰 local epoch에서도 robust." 이건 limitation이 아니라 *결과*.
- (b)를 *bonus track*으로 백그라운드 시도 가능 — 닫히면 paper에 추가, 안 닫혀도 손해 없음.

---

## N2 — 13B+ operational form (잠금 + 비용 정리)

13B+로 가는 건 결정됐고, 어떻게 가는지를 정리하겠습니다 (제 가정안 — 다르면 알려주세요):

| Evidence | 1B (Llama-3.2-1B) | 3B (Qwen-3B) | 13B+ (e.g., Qwen-2.5-14B / Llama-3-70B LoRA) |
|---|---|---|---|
| Flirds 본체 (FedAvg + per-round HVP) | ✓ | ✓ | ✓ — 저비용 (LoRA 차원이 작음) |
| **(b) MC in-run oracle** ($2^N$ enum at N=10) | ✓ | ✓ | ✓ — forward-pass만, B200×4로 며칠 |
| **(a) 실제 exact SV** (1024 retrain at N=10) | ✓ | ✓ | ✗ — 14B에서 1024 retrain은 *주~월* 단위, 13B+에선 *생략* |
| Downstream (client-selection, task acc) | ✓ | ✓ | ✓ |
| Noisy/free-rider AUROC | ✓ | ✓ | ✓ |

**13B+에서 (a) 생략**은 IRDS paper도 동일 — exact Shapley는 LLM scale에서 일반적으로 안 함. 정직하게 "exact Shapley는 1B/3B로 정확성 입증, 13B+는 (b) MC + downstream으로 scale claim" 보고. 이 분할이 N3 dual-oracle 프레이밍과 완벽히 정합.

**70B까지 갈지**는 컴퓨트 가능성 보고 별도 결정 — 일단 14B(Qwen-2.5)을 13B+ headline으로, 70B는 stretch goal.

---

## N3 잠금 — dual oracle reporting + stress + precision

사용자 통찰을 반영해 정리:

**두 가지 Shapley 정의 분리 보고:**
- **(a) 실제 exact SV** (retrain 전체 과정 SV) — "data valuation community standard" 의미. *harder test*, IRDS paper 선례대로 ρ < 1.0 예상, gap을 "different utility" 프레이밍으로 정직 보고.
- **(b) IRDS-정의 SV** (per-batch 누적 SV) — "Flirds lineage" 의미. Flirds가 *바로 이 (b)*의 닫힌형 Taylor 근사이므로 **ρ≈1.0이 expected/sanity**. 이전 ρ=1.0은 이 sense.

**reframe 효과**: ρ=1.0이 "수상하게 좋다"에서 "(b) sense의 sanity check일 뿐, 더 강한 (a) sense에선 IRDS 식 gap 보고"로 바뀜 → reviewer 의심 차단 + paper evidence 구조 풍부화.

**Adversarial stress**: 두 sense 모두에 α≤0.01, label-flip×OOD 조합, larger N 등 deliberately-hard regime 추가. 결과는 "(b) sense ρ는 stress에도 유지, (a) sense는 stress와 함께 점진적 degradation을 보고" 가설.

**Exact Shapley 정밀화**: MC 분산·sample 수·ties 처리·CI band 등을 부록·protocol에 명시.

---

## N4 잠금 — 세 metric 전부 run

Q2 변형(layer-wise / phase-norm / 기본) × 세 metric (client selection convergence / downstream task acc-F1 / noisy & free-rider AUROC) 풀 매트릭스. ①(default Flirds AUROC=0.50) 결과 위에서 변형이 AUROC를 회복시키는지가 *부수적 흥미 결과* — 회복하면 "variant가 default의 한계를 일부 보완"의 작은 contribution, 안 회복하면 "한계가 estimator-form 무관하게 견고"로 더 강한 limitation 진술.

---

N1만 결정되면 Section 2가 완전히 닫힙니다.

## [2026-05-22 07:34] Claude

먼저 N1에 대해 솔직히 한 번 짚고 가야겠습니다 (rubber-stamp 안 하는 게 콜라보 의무라고 봅니다). 그리고 N2 조사는 별도 에이전트에게 백그라운드로 맡기겠습니다.

## N1 — "관측 무시" 제안에 대한 우려

원래 가정 유지 + U-shape 무시에는 두 가지 위험이 있습니다:

1. **재현 강도가 큽니다.** U-shape이 *Llama-3.2-1B와 Qwen-3B 양쪽*에서 동일하게 관측됐다고 하셨습니다. 서로 다른 architecture·tokenizer·pretraining에서 같은 패턴이면 단일 실험 버그일 가능성은 낮습니다. bf16 artefact는 Qwen-3B의 *특정 lr* 조건에 한정됐다 하셨고, 1B는 별개 setup이니 그 메커니즘으로 1B의 U-shape을 설명하기 어렵습니다.

2. **Theorem-data 불일치는 paper integrity 위험.** Method/theory track에서 "예측 vs 실측 불일치"가 보고 안 됐다가 리뷰어 자체 replication에서 드러나면 9.3/10 평가가 통째로 흔들립니다. ICML/NeurIPS의 강리뷰어는 1B 모델은 자기 GPU로 retest 자주 합니다.

**제안 하이브리드 — replication step을 넣고 분기**:
- **(i)** 다른 seed × fp32 강제 × 다른 aggregation 조합으로 *1B만* E∈{1,3,5,10} 빠르게 재실행 (며칠).
- **(ii) 결과 분기**:
  - U-shape이 *사라짐* → 원 setup에 특정 artefact 있었던 것 → **원 가정 (a) 유지 정당화 + 새 데이터로 뒷받침**. 가장 안전.
  - U-shape이 *남음* → real phenomenon → 옵션 (a) Drop+empirical로 가는 게 정직. 이 경우 paper에 *positive*("Flirds가 큰 E에서도 robust")로 reframe.
- **백그라운드로 (b) sign-dominance 재유도** 가능 — 닫히면 bonus.

이 분기 없이 "원 가정 유지 + 관측 무시"는 *paper integrity 도박*이라 일단 N1을 "lock"이라 부르긴 어렵다고 봅니다.

다만 **이미 U-shape의 명확한 실험적 원인(특정 hyperparameter, eval 코드 버그 등)을 식별하셨다면** 그 정보 주세요 — 그 경우엔 원 가정 유지가 정당합니다.

## N2 — 실험 세팅 조사 (백그라운드)

Federated뿐 아니라 plain LLM data valuation 논문들의 실험 세팅(모델 규모·데이터셋·N·평가 metric)을 일관 매트릭스로 정리하고, wiki에 없는 중요 논문이 있는지 gap-fill 점검을 함께 의뢰합니다.

## [2026-05-22 07:34] Claude

여러 단계가 동시에 열려 있어서 task list로 상태를 정리하는 게 좋겠습니다. TaskCreate 스키마를 먼저 가져오겠습니다.

## [2026-05-22 07:35] Claude

진행 중인 작업이 늘었으니 task list로 정리하겠습니다.

## [2026-05-22 07:35] Claude

Task 의존성을 설정하겠습니다.

## [2026-05-22 07:35] Claude

현재 상태 요약 — N1(사용자 응답 대기)·N2(에이전트 백그라운드 실행 중) 두 trail이 열려 있고, 둘이 닫히면 wiki/HTML/log/protocol 갱신이 순차로 이어집니다. 두 답이 들어오면 합성해 Section 2를 완전 close하고 Section 3 실행으로 넘어가겠습니다.

## [2026-05-22 07:36] Yonghee

N1: 아니 우리가 다시 이론을 단단하게 만들어서 새로 실험을 돌릴거니까 테스트용으로 돌린 실험 결과는 무시하고 제대로 구현해서 다시 실험을 해보자는거지 리뷰어를 속여먹자는 말은 아니야.

## [2026-05-22 07:37] Claude

아 — 제가 잘못 읽었습니다. 사과드립니다. "관측을 무시"가 *발표용 결과에서 빼겠다*가 아니라 *테스트용 pilot run의 데이터는 set aside하고 이론 단단하게 + 제대로 구현 + 다시 실험*이라는 뜻이었군요. 그건 완전히 정상적인 research workflow고 integrity 문제 없습니다.

## 재정렬

**N1 = 잠금**: 원래 이론 프레임(Prop 1 monotone + Prop 2 cubic bound)을 *re-run의 target*으로 유지. pilot 데이터(U-shape, ρ=1.0, AUROC=0.50)는 exploratory signal로만 취급, paper claim에 들어가지 않음. **Contingency**: clean re-run에서도 U-shape이 재현되면 그때 option (a) Drop으로 분기. 즉 fallback은 주머니에 있지만 default는 원 이론.

## 결과적으로 바뀌는 것

| 항목 | 직전 응답에서 한 가정 | 정정 |
|---|---|---|
| Theorem 1 + ρ=1.0 evidence | "established → Section 1 승격" | **여전히 planned**. canonical 실험 전 |
| Prop 2 cubic bound | "B4 slope으로 confirmed" | 여전히 planned |
| ① AUROC=0.50 한계 재프레이밍 | "data-backed로 강화" | 여전히 *예측*. canonical run 후 확정 |
| ② cancellation 무력화 (ρ=1.0) | "vindicated" | 여전히 가설 |
| 9.3/10 method/theory submission-strength | "도달" | pilot 예측, canonical run으로 확증 필요 |
| wiki/flirds.md 갱신 분량 | "pilot 데이터 반영해 대폭 재작성" | **2026-05-19 상태와 거의 동일**, 결정 잠금만 distill |

다시 말해 wiki/HTML은 2026-05-19 plan 그대로가 거의 옳고, 갱신은 Section 2 결정(Q1·Q2·Q3·④·⑤·N3·N4) distill + Section 3 우선순위 lock에 집중됩니다. pilot 결과는 raw conversation log에만 남고 paper-bound 페이지에는 안 들어갑니다.

## Section 3 protocol(Task 6)에 들어갈 "제대로 구현" 사양

- **Numerical precision**: fp32 evaluation 강제 (bf16-suspect 사전 차단). training precision은 별도 결정.
- **Seeds**: minimum 3 seeds per cell, 보고는 mean ± std (single-seed 주장 금지).
- **Reproducibility**: per-run config·env hash·git SHA 저장.
- **Oracle implementation**: (a) retrain SV과 (b) IRDS-定 SV을 *별도 코드 경로*로 분리, ties 처리·MC 분산·CI band 명시.
- **Logging**: per-round per-client $\phi_k^{(r)}$ 전부 저장 (post-hoc 분석용).
- **Sanity gates**: $E=1$에서 drift residual ≈ 0, $N=2$에서 1-client subset이 그 client의 SV과 일치, 등 자동 체크.
- **Aggregation**: FedAvg 기본, FedProx ablation 별도 코드 경로.

이 protocol이 잠겨야 canonical re-run의 결과를 reviewer-safe하게 보고할 수 있습니다.

N1은 닫고, N2(lit-norm survey) 결과 들어오면 합성해 Section 2 완전 close하겠습니다.

## [2026-05-22 07:38] Claude

N2 lit-norm survey 결과 도착. 핵심: **현재 plan에서 13B+가 field norm을 명확히 *초과*하며, agent는 "compute-flex로 보여 method-novelty를 오히려 약화시킬 위험" 우려**를 명시했습니다. 즉 N2의 잠정 결정("13B+까지")이 실제 norm 데이터와 충돌해 재고가 필요합니다.

## Lit-norm 요약

| Track | 현재 norm | 13B+ 위치 |
|---|---|---|
| **Centralized LLM data attribution** | Llama-2-7B / Llama-3-8B + LoRA (LESS, LoGra, DataInf) — Pythia 410M–1B for pretraining (IRDS, MATES) | DataInf chat = 13B(qualitative only); Grosse 2023 = 52B(Anthropic resource flex, 외톨이) |
| **FL data valuation** | small CNN, MNIST/CIFAR-10, N≈10 (FedSV, GTG, ShapleyFL, Ripple AAAI'26 포함 거의 전부) | — |
| **FL + LLM data valuation** | **Llama-2-7B + LoRA** — *FedDQC가 wiki 내 유일 precedent* | 13B+ 선례 없음 |
| **Flirds plan** | 1B (primary) + 3B (scale-check) + **13B+ (headline)** | unprecedented in FL data-valuation |

Agent 권장: **1B + 3B + 7B (FedDQC-comparable instruction-tuning benchmark에서 7B 1점 추가)**. 7B는 (i) centralized LLM attribution norm을 맞춤, (ii) FL+LLM의 유일 precedent와 직접 비교 가능. 13B+는 "method/theory track에서 compute-flex" 프레이밍 위험.

## Trade-off — 정직하게

- **13B+로 가면**: scale claim 강화 + "FL data valuation은 최초 13B+" 차별점. 그러나 (a) (a) exact retrain 불가능, (b) 7B norm을 *건너뛰는* 모양새, (c) reviewer가 "compute로 method 약점 가린다" 의심.
- **7B 추가로 가면**: norm 정중앙 + FedDQC와 직접 비교 가능 + (a)/(b) oracle 양쪽 적용 feasible + reviewer 부담 최소.
- **둘 다 가면 (1B+3B+7B+13B)**: scope 확장, 일정·compute 부담. 7B가 evidence carry, 13B는 boldness statement.

## Gap-fill ADD 후보 (별건)

- **LESS (Xia et al., ICML 2024)** — Llama-2-7B/Mistral-7B + LoRA instruction tuning, gradient-similarity selection. **가장 직접적 centralized 비교 대상**. positioning load-bearing.
- **Grosse et al. 2023 (EK-FAC, 52B)** — IF의 *upper-bound anchor* (위쪽 한계 정의용).
- **MATES (Yu et al., NeurIPS 2024)** — Pythia 410M–1B pretraining, *1B primary 결정을 직접 뒷받침*.
- **DSDM (Engstrom et al., ICML 2024)** — light add, Datamodels → LLM bridge.

이 4편은 wiki ingest 가치가 있습니다(N2 결정과 무관하게).

## N2 재선택 — 새 정보 기반

## [2026-05-22 07:47] Claude

Section 2 완전 close. N2 lock(1B+3B+7B), ADD 4편은 다른 에이전트가 처리 중이니 wiki ingest는 그쪽 결과에 합치겠습니다.

## [2026-05-22 07:48] Claude

## Section 2 잠금 완료

| | 결정 |
|---|---|
| Q1 use case | research-side in-run attribution (incentive·detection은 보조 application) |
| Q2 IRDS-inherited 한계 | ②③ 둘 다 *default*=IRDS-form 특성화, layer-wise/phase-norm 변형은 downstream-perf ablation |
| Q3 Ripple 포지셔닝 | 다른 설계점이 主 주장, reduction은 닫히면 보너스 |
| ④ trajectory dependence | feature(IRDS framing), Q1으로 종결 |
| ⑤ ground-truth | N3로 흡수 (dual oracle) |
| N1 Prop 1 처리 | 원 이론 프레임 유지(re-run target), pilot 데이터 set aside, (a) Drop은 contingency |
| N2 scale tier | **1B + 3B + 7B** (FedDQC-comparable 7B 추가, 13B+ 제외) |
| N3 oracle/stress | **(a) exact retrain + (b) IRDS-定 batch-cumulative** 분리 보고 + adversarial stress + MC precision(분산·CI·ties) |
| N4 변형 비교 metric | client-selection convergence + downstream task acc/F1 + noisy/free-rider AUROC 셋 다 |

---

## Section 3 (보강) — 재정리

이제 *보강이 필요한 부분*으로 넘어갑니다. Section 2 잠금에 맞춰 재구성:

### ★★★ — paper spine을 담당, 가장 먼저

1. **Clean implementation protocol 잠금 문서**. fp32 eval 강제, seeds≥3, per-run config·env·git hash, (a)/(b) oracle 별도 코드 경로, ties/MC 분산/CI 명시, sanity gates(E=1 residual≈0, N=2 singleton SV 일치 등). *이게 잠겨야 다른 모든 실험이 reviewer-safe.*
2. **Baseline run set (cost-matched tiers)**:
   - 필수 = GTG-Shapley, Ripple Shapley, loss-heuristic, Flirds-1차-only
   - 권장 = FedSV(=Wang 2020), Federated Banzhaf(=Data Banzhaf in FL)
   - cross-device = ComFedSV
3. **Dual oracle 구현**: (a) retrain exact SV — N=10 cross-silo + 1B/3B only(7B는 N 축소 또는 생략); (b) IRDS-定 batch-cumulative MC SV — 모든 scale에서 enum/MC.
4. **Ripple head-to-head + reduction 시도(bonus track)**. LoRA+Taylor 하에서 Ripple drop+ripple ⊃/≡ Flirds 1차+2차 정리 검증. 닫히면 paper에 추가, 안 닫혀도 손해 없음.

### ★★ — 한계 정직 보고 + 핵심 ablation

5. **α-sweep × E-sweep drift residual matrix**. E∈{1,3,5,10} × Dirichlet α∈{0.01, 0.1, 0.5, 5.0}. Prop 1·2 검증 + N1 contingency 분기점.
6. **변형 비교(N4)**: {default Flirds, layer-wise weighted, phase-normalized} × {client-selection convergence, downstream task acc/F1, noisy & free-rider AUROC}. 9-cell 매트릭스.
7. **②③ 특성화 보고 실험**:
   - ②: per-layer 분해, 마지막-레이어 항의 cross-round 부호 요동 (non-IID vs IID), 레이어 제외 진단의 ranking 변화.
   - ③: late-joiner test (joining round vs score at fixed quality) — magnitude/timing 혼동의 크기 보고.
8. **Adversarial stress regimes (N3)**: α≤0.01, label-flip×OOD 조합, larger N, late-joiner extremes. (b) 사례에서 ρ 유지, (a) 사례에서 점진적 degradation 기대.
9. **Non-IID 가치평가 bias 정량화** (① 보류의 파생 의무): drift-residual study에 α-sweep을 한 축으로 끼워 보고.

### ★ — protocol·재현성·확장

10. **bf16/precision 프로토콜 문서**: training은 자유롭게 두되 *evaluation은 fp32 강제*, 알려진 LR×precision 페어 caveat 명시.
11. **7B 추가 점 (N2)**: FedDQC-comparable instruction-tuning bench (의료/법률 등 도메인 분리 client). client-selection convergence + (b) MC oracle만 (a)는 N=10 cost로 가능하면 시도, 아니면 1B/3B로 충분.
12. **70B stretch**: 명시적 future work로 paper에 1문장.

---

## 다음 단계 선택

이 Section 3 구성으로 가도 될지, 아니면 우선순위/내용을 조정할지 결정해주시면:

- 동의 → Task 3(wiki/flirds.md distill), Task 4(HTML 재구성), Task 5(대화 log 저장), Task 6(protocol 문서)을 *지금 일괄 실행*하겠습니다.
- 조정 필요 → 어느 항목을 어떻게 바꿀지 알려주시면 반영 후 실행.

Section 3 12개 항목 중 *paper-blocking 우선순위*가 다르게 보이는 게 있는지(특히 ★★★ 4개 vs ★★ 5개의 경계)와, 12번(70B stretch) 같은 항목을 *paper에 안 넣고* 빼는 게 나을지가 가장 큰 trade-off 포인트입니다.
