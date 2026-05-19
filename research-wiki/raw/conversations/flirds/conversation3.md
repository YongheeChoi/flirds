우선 LoRA를 사용하는 PEFT 환경을 가정할거야. 그리고 이어서 앞선 대답에 대해 첨언을 할게.

1. 여전히 Cross-silo와 cross-device 둘 다를 목표로 하지만, 2번에서 말한 benchmark 종류에 따라 적절하게 환경을 선택할 필요가 있을 것 같아. 현실적으로 가능한 범위로 세팅해줘. 참여 횟수 정규화가 모든 클라이언트가 동일한 횟수로 참여해야 한다는 뜻이라면, 정규화를 하지 않는 것이 좋을 것 같아. 다만 그런 유동적인 상황을 가정해도 우리 알고리즘이 의도대로 잘 동작한다는 것을 보여줄 수 있는 실험이 corss-device에서도 가능할지 검토해줘. 너무 큰 유동성이 문제가 될 것 같으면 정규화를 한다해도 문제는 없을 것 같아.
2. Exact Shapley in FL을 정의하는 방식은 a, b가 상관없이 똑같다는게 irds에서 검증된 것 아니야? b의 누적이 a가 되는거잖아. 어떤지 좀 더 자세히 얘기해줘. 그리고 현실적 benchmark 제안에 대해서는 제안해준 검증 방식이 너무 좋은 것 같아. 근데 앞서 validation data loss와 일치하지 않는 방향의 업데이트에 대해 진짜 noise인지 validation data의 out of distribution에 해당하는 좋은 sample인지 고민해봐야 한다고 말했는데, 이를 고려해서 알고리즘을 설계하려고 해. 그래서 그런 부분에서 알고리즘이 제대로 동작하는지 검수하는 실험이 추가로 있으면 좋겠고, 또 ablation study 실험에 대해서도 미리 고민해놓으면 좋을 것 같아.
3. 우선 2차 근사항은 항상 사용할거야. 굳이 때마다 바꿀 필요는 없을 것 같아. 그리고 taylor 오차가 커지는것만 감수하면 되는 문제라면 오차를 감수하면 돼. 그리고 fl에서는 client 단위로 기여도를 평가하는게 더 자연스러운 것 같아. 다만 centralized learning 상황을 가정한다고 해도 여러 data에 대한 총 기여도와 각 data sample에 대한 기여도를 별개로 계산해서 더한 것이 차이가 있다면 그건 고려해볼만한 포인트야. 왜냐하면 client가 업데이트하는 gradient 단위로 기여도를 평가하겠다는 생각은 이 전체 gradient가 client의 각 데이터 샘플의 기여도 총합을 대표한다는 가정을 하고 있는거니까. 이 차이가 수학적으로 어떻게 발생하는지 검증하는 것은 논문에 들어가야 할 것 같아.
4. server-side validation distribution은 우리가 검증하고자 하는 benchmark 및 metric 마다 달라져야 하지 않을까? 기본적으로는 균등하게 모든 domain을 커버한다고 가정할거야. 매 실험에 맞게 실험 결과가 validation data에 의해 좌우된 것이 아님을 보여줄 수 있게 공정한 세팅을 해야지.

LoRA 기반 PEFT 가정이면 통신/연산 비용 측면이 훨씬 깔끔해져서 method design이 자유로워질 거야. 답변 이어갈게.

## 1. Cross-device 실험의 현실적 design

Cross-silo는 N=10~20 정도의 안정적 client로 LLM finetuning (예: 각 client가 다른 domain corpus). 여기서는 retraining-based exact Shapley도 LoRA 덕분에 actually feasible — N=10이면 1024번 LoRA finetuning은 GPU 며칠로 돌아감. **이게 sanity check 용으로 가장 강력한 환경**이야.

Cross-device는 N=100~1000, round당 K=10~50 sampling. 여기서는 exact Shapley 자체가 불가능하니까 **task-based 검증**으로 가야 해. 정규화 없이도 의도대로 동작함을 보일 수 있는 실험 셋:

- **Noisy client injection w/ random participation**: N=100 중 10%에 label noise 주입. 참여율은 전 client 동일 (예: round당 uniform sampling). Flirds가 noise client를 하위 ranking으로 정확히 분리하는지.
- **Heterogeneous participation w/ controlled quality**: 일부 client는 자주 참여 (예: 80% rounds), 일부는 드물게 (10% rounds). 양 그룹 안에 noise/clean 섞어서, Flirds가 _참여 빈도와 무관하게_ quality를 separation하는지. 정규화 없이 절대값은 다르겠지만, **각 빈도 그룹 내에서의 ranking이 quality를 따라가는지**를 보면 됨.
- **Late-joiner test**: 어떤 client가 학습 후반(예: round 50 이후)에만 joining. 이 client가 noise/clean에 따라 적절히 평가되는지. 학습 후반은 gradient magnitude가 작아서 (한계점 b 참고) 점수가 깎이는 경향이 있을 텐데, 이 효과를 quantify해서 limitation으로 보고하는 것도 정직한 reporting이야.

너무 큰 유동성이 결과에 noise로 작용하는 상황(예: 5% 미만 참여 client)은 별도 robustness study로 묶고, main result는 reasonable participation rate (≥10%)에서 보고하면 reviewers가 받아들일 거야.

## 2. (a) vs (b)에 대한 명확한 정리 — 이게 중요한 부분

네 질문에 정확히 답하면: **"(b)의 누적 = (a)" 라는 것이 IRDS에서 검증된 게 아니야.** 이걸 잘못 받아들이면 paper에서 잘못된 claim을 할 수 있으니 짚어둘게.

IRDS는 두 가지를 별개로 보여줘:

- **Linearity 공리에 의한 사실**: $\phi(U_{total}) = \sum_r \phi(U^{(r)})$ 즉 per-round in-run shapley 합 = full-training in-run shapley. 이건 axiom으로부터 자동.
- **경험적 관찰**: in-run shapley vs retraining-based shapley는 _상관성이 높지만 동일하지 않다_. IRDS paper의 Appendix B.1이 이 둘을 명시적으로 구분하면서 "different things"라고 해.

내가 말한 (a)와 (b)의 정확한 의미:

- **(a) Retraining-based exact Shapley**: $U(S) = \texttt{Perf}(\mathcal{A}(S))$. $\mathcal{A}$는 $S$만으로 처음부터 학습. 학습 trajectory 자체가 $S$에 따라 바뀜.
- **(b) In-run exact Shapley**: $U_{total}(S) = \sum_r [\ell(w^r + \sum_{k \in S} p_k \Delta w_k^r, z^{val}) - \ell(w^r, z^{val})]$. 한 trajectory를 fix하고, 각 round에서 counterfactual subset을 가정.

(a)와 (b)는 _다른 utility function에 대한 exact Shapley_. 두 utility가 같지 않으니 Shapley 값도 같지 않아. 직관: (a)는 "이 client가 모든 가능한 학습 결과에 평균적으로 얼마나 기여하는가", (b)는 "이 specific run에서 얼마나 기여했는가".

**Flirds의 자리**: Flirds는 (b)에 Taylor 근사를 추가한 것. 그러므로:

- **Honest comparison**: Flirds vs Monte Carlo로 추정한 (b). Spearman 상관성이 매우 높아야 함 (Taylor 오차만큼만 차이). 이게 _진짜_ sanity check.
- **Conventional comparison**: Flirds vs (a). 상관성은 있지만 1.0은 아닐 것. IRDS paper도 이 둘 사이엔 일정 gap이 있음을 보임. Reviewer가 conventional baseline 비교를 요구하니 보고는 해야 하지만, "두 정의는 다른 양을 측정하므로 완벽 일치는 기대하지 않음"을 명시해야 함.

이 framing을 잘 잡으면 "Flirds는 exact Shapley와 정확히 일치하지 않는다"는 비판을 사전에 무력화할 수 있어. **IRDS paper와 같은 narrative를 쓰는 게 안전해**.

## 3. Noise vs OOD-good 구분 알고리즘 + 실험/Ablation

### 알고리즘 design 방향 (간단히 sketch)

핵심 아이디어: 단일 round의 sign이 아니라 **cross-round, cross-client signal**을 모두 보면 구분 가능.

- **Temporal consistency signal**: 진짜 noise는 round마다 gradient 방향이 random (label과 input 사이의 일관된 신호가 없으니까). OOD-good은 일관된 방향을 유지. 측정: client k의 round별 contribution의 분산 vs 부호 일관성.
- **Cross-client agreement signal**: noise는 client k에 unique한 random pattern. OOD-good은 같은 OOD-distribution을 공유하는 다른 client와 gradient 방향이 align될 가능성. 측정: client 간 update direction의 cluster 구조.

이런 보조 metric을 raw Flirds score와 합치면 robust score를 만들 수 있어.

### 검증 실험

1. **Pure noise vs pure OOD-good 분리 실험**:
    
    - 3 그룹: clean clients (validation distribution), noisy clients (label-flipped), OOD-good clients (validation엔 없지만 high-quality 다른 domain).
    - Naive Flirds: noisy와 OOD-good 둘 다 낮은 점수.
    - Augmented Flirds: noisy만 낮고 OOD-good은 중간/높은 점수.
    - 두 그룹 분리 정확도 (예: AUROC)가 main metric.
2. **Validation expansion 검증**: Naive Flirds로 낮게 평가된 client 중 OOD-good 후보를 식별 → 그 client의 distribution을 validation에 포함시켜 다시 학습 → 그 client의 점수가 _positive로 변하는지_. 이게 OOD-good의 정의에 부합. Noise는 그래도 음수 유지.
    
3. **Synthetic distribution shift**: Validation은 domain A, client는 A/B/C 분포. B는 A와 transfer되는 좋은 OOD, C는 random label noise. Flirds가 B와 C를 다르게 평가하는지.
    

### Ablation study

|Ablation 항목|측정하고자 하는 것|
|---|---|
|1차 only vs 1차+2차|2차 항이 client interaction을 잡아내는 효과|
|Δw 사용 vs per-step gradient 사용 (aggregate된 통계로)|Local step 누적 vs 즉각적 정보|
|Temporal consistency 항 on/off|Noise vs OOD-good 분리 contribution|
|Cross-client agreement 항 on/off|동일|
|Validation size (small/medium/large)|Validation noise propagation|
|Validation distribution (uniform/biased)|Distribution choice의 영향|
|LoRA rank (4/8/16/32)|PEFT 차원이 attribution quality에 미치는 영향|
|Local epoch E (1/3/5/10)|Taylor 오차 vs 실용성 trade-off|
|Aggregation method (FedAvg/FedProx)|Client drift 보정이 valuation에 주는 영향|
|Participation rate|Cross-device robustness|
|Noise ratio (10%/30%/50%)|Detection 한계|

이 정도면 강한 실험 섹션이 만들어져.

## 4. Client gradient = sum of data-level contributions 가정의 수학적 검토

이게 paper의 핵심 기여 중 하나로 들어가야 할 부분이라 정확히 풀어 적을게.

### 결론 먼저

**Centralized SGD에서 한 step 안에서는 두 정의가 정확히 일치한다. FL의 multi-step local update에서는 일치하지 않는다 — 일치하지 않는 양이 정확히 FL이 centralized와 다른 점을 quantify한다.**

### Centralized 한 step에서의 일치성

$g_z := \nabla\ell(w_t, z)$ 표기. 한 batch $B$에서 1차 항:

$$U^{(t)}_{(1)}(S) = -\eta, \nabla\ell^{val} \cdot \sum_{z \in S} g_z$$

이건 $S$에 대해 additive. Data-level Shapley: $$\phi_z^{data} = -\eta, \nabla\ell^{val} \cdot g_z$$

Client $k$를 $I_k$라 하면 client에 모은 합: $$\sum_{z \in I_k} \phi_z^{data} = -\eta, \nabla\ell^{val} \cdot \sum_{z \in I_k} g_z = -\eta, \nabla\ell^{val} \cdot G_k$$

여기서 $G_k := \sum_{z \in I_k} g_z$. 이제 client-level utility를 직접 쓰면: $$U_{client}(S) = -\eta, \nabla\ell^{val} \cdot \sum_{k \in S} G_k$$

Additive이므로 $\phi_k^{client} = -\eta, \nabla\ell^{val} \cdot G_k$. **완전히 일치.**

2차 항도 같은 결론이 나와. 2차 utility는

$$U^{(t)}_{(2)}(S) = \eta^2 \sum_{z, z' \in S} g_z^T H g_{z'}$$

quadratic Shapley 공식 $\phi_i = a_{ii} + \sum_{j \neq i} a_{ij}$ (symmetric $a$)을 적용하면:

$$\phi_z^{data}(U_{(2)}) = \eta^2, g_z^T H \cdot G(B_t)$$

Client에 모으면: $$\sum_{z \in I_k} \phi_z^{data}(U_{(2)}) = \eta^2, G_k^T H \cdot G(B_t)$$

Client-level direct 계산도 같은 quadratic 구조에서 $\phi_k^{client}(U_{(2)}) = \eta^2, G_k^T H \cdot G(B_t)$. **여기도 일치.**

즉 **한 step gradient에 대한 Shapley는 data-level vs client-level이 grouping 무관하게 같다**. 이게 첫 번째 lemma야.

### FL multi-step에서 깨지는 지점

문제는 client가 서버에 보내는 $\Delta w_k$가 $-\eta G_k(w^r)$이 아니라는 것. $E$ local step 동안:

$$\Delta w_k = -\eta \sum_{e=0}^{E-1} \sum_{z \in B_{k,e}} \nabla\ell(w_k^{r,e}, z)$$

각 $\nabla\ell(w_k^{r,e}, z)$를 $w^r$ 기준 1차 Taylor 전개:

$$\nabla\ell(w_k^{r,e}, z) \approx \nabla\ell(w^r, z) + H_z(w^r) (w_k^{r,e} - w^r)$$

대입하면

$$\Delta w_k \approx -\eta E \cdot \bar{G}_k(w^r) - \eta \sum_{e,z} H_z(w^r)(w_k^{r,e} - w^r)$$

여기서 첫 항 $-\eta E \bar{G}_k$는 "centralized-equivalent" 부분 (samples를 $E$번 본 효과), 두 번째 항은 **client drift correction** — 이게 $E>1$일 때만 등장하고, $E$와 local trajectory의 길이에 따라 커져.

따라서 FL에서 client gradient $\Delta w_k$를 그대로 IRDS 1-step 취급하면, valuation은

$$\phi_k^{Flirds} = -\nabla\ell^{val} \cdot \Delta w_k + \text{2차 항}$$

가 되는데, 이 안에는 **순수 data 기여 + client drift bias**가 섞여 있어. Client drift bias는 client의 데이터 분포가 global과 다를수록 (non-IID일수록) 커져 — 즉 **non-IID client일수록 평가에 추가 noise가 끼는 구조**.

### Paper에 들어갈 statement (제안)

> **Proposition (informal)**: Flirds의 client-level Shapley는 centralized data-level Shapley의 client-aggregate에 client drift residual을 더한 양과 같다. Drift residual의 크기는 $O(\eta E |\bar{H}| \cdot \text{local-trajectory length})$이며, $E=1$일 때 사라진다.

이걸 정리/증명하고, residual의 크기를 empirical하게 측정하는 실험 (다양한 $E$에 대해 residual 크기 plot)이 method 섹션의 주요 contribution 한 자리 차지할 수 있어. **"우리는 IRDS를 FL로 단순 옮긴 게 아니다, 옮길 때 발생하는 정확한 수학적 차이를 quantify했다"**가 narrative.

### 추가 고려

이 분석은 **algorithm choice에도 영향**을 줘:

- $E$가 작을수록 residual이 작음 → "Flirds는 small $E$ 권장"이라고 할 수 있음
- 또는 residual을 보정하는 control variate를 도입할 수도 있음 (예: SCAFFOLD-style correction term을 빼고 valuation)
- Server가 client에게 $w^r$ 기준 gradient $G_k(w^r)$도 함께 요청하면 (LoRA에선 cheap) residual을 분해할 수 있음 — 이것도 design 옵션

## 4. Validation distribution

너 의견에 동의해. 기본은 uniform domain coverage. 단, **각 실험마다 어떤 distribution을 썼는지 명시 + 그 선택이 결과를 prejudge하지 않음을 보이는 control**을 같이 보고하는 게 안전해. 예를 들어 "OOD-good 검증 실험에선 OOD-good client의 domain을 validation에서 _제외_해서 그들이 처음엔 음수로 평가되어야 함을 보임" 같은 식.

---

다음 단계로 가기 전에 정리할 만한 design decision이 몇 개 남았어:

1. **Δw_k 외에 client에게서 추가로 받을 통계**가 있을지 (예: $w^r$ 기준 sample mean gradient $\bar{G}_k(w^r)$ 같은). 이게 있으면 client drift 분해가 가능해지지만 통신 비용은 약간 늘어. 받느냐 마느냐가 method 형태를 좌우함.
2. **2차 항에 필요한 Hessian-vector product를 서버가 어떻게 계산할지** — validation gradient에 대한 HVP는 서버에서 compute 가능하지만, client gradient 간 cross-term은 서버가 모든 $\Delta w_k$를 다 받아야 함. LoRA면 차원이 작아 OK이지만, 명시적으로 정해두면 좋음.

이 중 결정이 어려운 게 있으면 같이 더 따져보자.