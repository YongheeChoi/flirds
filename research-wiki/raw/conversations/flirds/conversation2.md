우선 질문에 대한 대답부터 확실히 할게.

1. FL setting 두 가지 모두에서 잘 동작하는 알고리즘을 찾는게 목적이야.
2. valuation의 목적은 조금 고민이 되는 부분이고 선행 연구에 대한 조사가 필요한 부분이야. 내가 지금까지 조사한 바에 따르면 cnn 기반의 작은 모델에서 image classification 기반의 모델과 LLM 모델에 대한 기여도 평가가 성능 검증 benchmark 측면에서 거의 별개의 track인 것처럼 다른 방식으로 성능을 검증하는 것 같아. 작은 모델은 exact shapley에 대한 검증이 가능해서 exact shapley와의 비교가 가장 기본적인 지표이고, 추가적으로 모델의 수혐 속도나 정확도 향상 등의 지표를 보는 것 같아. 근데 모델이 클수록 exact shapley 계산이 불가능해지기 때문에 monte carlo 방식으로 대체하거나 아니면 그냥 모델의 수렴 속도나 정확도 향상을 메인 목표로 잡고 평가를 하는 것 같아. 근데 나는 좀 고민이 되는게 exact shapley와 상관 관계를 비교하려니 fl 환경에서 exact shapley를 어떻게 정의하는게 좋을지 모르겠어서 고민이 돼. 앞서 답변해준 내용 중에서도 기여도 평가의 대상을 client의 local update가 끝난 gradient 자체로 봐야 할 지, local에서 일어나는 매 step의 gradient를 대상으로 봐야할지 애매했잖아. 그래서 LLM에 대한 finetuing을 가정하고 있는 상황에서 어떤 benchmark나 metric을 목표로 해야 할 지 조언을 구하고 싶어.
3. Local epoch을 1로 두는건 너무 비현실적이야. 그래서 1보다 크게 둘거야. 근데 그냥 client 단위로 기여도를 계산할거면, local에서 어떤 업데이트가 일어났건 관계 없이 서버가 그냥 받은 gradient를 irds에서의 data에 대한 update 1 step 취급을 해도 되지 않나? 그냥 그렇게 기여도를 정의하면 같은 논리 아니야?
4. validation data는 서버가 가지고 있어야 할 것 같아. 공정성의 문제도 있고, client가 평가 대상인데 답지를 가지고 있으면 악용의 여지도 있으니까.

답변 잘 정리해줬어. 각 포인트 짚어볼게. 특히 2번이 가장 본질적인 질문이고, 3번에는 수학적으로 짚어둘 caveat이 있어.

## 1. Cross-silo와 cross-device 둘 다 — 의미하는 바

이게 제약 조건으로 작용하는 부분이 있어. Cross-silo는 client 수 적고 안정적이라 상대적으로 무거운 valuation도 허용되지만, cross-device는 client 수 많고 dropout/straggler가 일상이야. 둘 다 잘 동작하려면:

- Per-round 비용이 client 수에 linear여야 함 (quadratic이면 cross-device에서 끝남)
- Round-level 통신만 사용 (client에 추가 round trip 요구하면 cross-device에서 비현실적)
- Partial participation에서도 의미 있는 누적이 가능해야 함

이건 method design 단계에서 이미 ghost technique을 보류한 결정과 일관돼. 다만 cross-device에서 같은 client가 100 round 중 5번만 참여하는 case가 흔하니까, "참여 횟수 정규화"를 method 내장으로 둘지 post-processing으로 둘지 미리 정해두는 게 좋아.

## 2. Benchmark/metric — LLM finetuning 가정 시

이게 가장 본질적인 고민이라 좀 길게 적을게.

**Exact Shapley in FL을 정의하는 두 가지 자연스러운 방식**

(a) **Client-level retraining-based Shapley**: $U(S) = $ FL training using only clients in $S$의 최종 성능. 이게 가장 자연스러운 ground truth야. $|N|=10$ 정도면 $2^{10}=1024$번의 FL training으로 가능 (cross-silo 가정 시 feasible). LLM finetuning에서도 LoRA 같은 PEFT를 쓰면 actually 돌릴 만해. 이게 cross-silo에서 네 method의 검증 기준이 될 수 있어.

(b) **Per-round per-client Shapley**: round $r$에서의 utility를 그 round의 model improvement로 정의. 이건 GTG-Shapley, FedSV 같은 기존 FL valuation 연구가 쓰는 정의야. 네가 IRDS의 "training run에 대한 attribution"을 따라간다면 사실 이게 더 일관된 정의야.

**중요한 인식**: (a)와 (b)는 다른 양을 측정해. (a)는 algorithm-level expected value이고 (b)는 specific run trajectory에 대한 attribution. IRDS 본문도 retraining-based vs in-run의 차이를 부각하잖아. **Flirds는 (b)에 해당하고, 그러므로 ground truth는 (b)의 Monte Carlo 추정이어야 일관됨**. (a)와 비교하면 사과랑 오렌지 비교하는 거 같은 결과가 나올 수 있어 — IRDS paper도 retraining-based와는 일부러 다른 이야기를 하고 있다고 명시함.

**현실적 benchmark 제안 (LLM finetuning 기준)**

Exact Shapley 비교는 cross-silo small-N setting에서만 sanity check로 하고, 메인 평가는 IRDS paper의 평가 방식을 FL로 옮긴 task-based benchmark가 좋을 거 같아:

1. **Noisy client detection**: client 한두 개에 label corruption 또는 random noise를 주입. 그 client가 ranking 하위로 가는지. (IRDS의 "mislabel detection"의 FL 버전)
    
2. **Free-rider detection**: client가 random/zero/fake update를 보내는 경우. 이건 FL valuation 연구에서 표준 task야.
    
3. **Domain attribution for LLM**: 각 client가 다른 domain corpus (의료/법률/코드 등). Validation을 특정 domain으로 잡았을 때 그 domain client가 top rank로 오는지. **이게 IRDS paper의 가장 인상적인 실험을 FL로 가져오는 방식이고, LLM에서 직관적인 demo로 강력해.**
    
4. **Client selection convergence**: top-K client만 골라서 학습했을 때 수렴 속도/최종 성능. 이건 monte carlo나 baseline과의 비교를 절대 점수로 만들어줘서 reviewers한테 어필함.
    
5. **Baseline 비교**: GTG-Shapley, FedSV, ComFedSV, Federated Banzhaf 같은 기존 FL valuation method와 동일 task에서 비교. **Exact Shapley correlation은 N=5-10 cross-silo에서만 sanity check로 사용**, 메인은 위의 task 성능.
    

이 조합이면 "exact Shapley를 FL에서 어떻게 정의?"라는 곤란한 질문을 메인 contribution에서 비껴나갈 수 있어. Sanity check용으로만 잠깐 등장하고, 진짜 평가는 task downstream으로.

## 3. Round delta를 1 step gradient로 취급하는 논리 — 절반은 맞음

수학적으로 네 직관이 맞아. Round-level utility를 정의하면

$$U^{(r)}(S) = \ell(w^r + \sum_{k\in S} p_k \Delta w_k, z^{val}) - \ell(w^r, z^{val}) \approx \nabla\ell(w^r, z^{val}) \cdot \sum_{k\in S} p_k \Delta w_k$$

가 되고, 이건 $S$에 대해 additive야. 그러므로 IRDS의 1차 closed-form이 그대로 옮겨와:

$$\phi_k^{(r)} \approx p_k , \nabla\ell(w^r, z^{val}) \cdot \Delta w_k^r$$

Total: $\phi_k = \sum_r \phi_k^{(r)}$. **수식 형태와 additivity는 완전히 보존됨**.

근데 **두 가지 caveat**이 있어:

(i) **Taylor 근사 오차의 크기**: IRDS는 한 batch step에 대한 $O(\eta^2)$ 오차였는데, 너의 경우 한 round delta에 대한 $O(|\Delta w_k|^2)$ 오차야. $|\Delta w_k| \approx \eta \cdot E \cdot |g_{local}|$ 정도이므로 오차가 $E^2$배 정도로 커. $E=5$면 25배. 이게 _치명적_일지는 empirical question인데, IRDS paper가 1차 근사로도 Spearman 0.94 정도 받은 걸 감안하면, FL에서는 이게 0.7-0.8 정도로 떨어질 수 있어. **2차 항이 IRDS보다 더 중요해질 가능성이 높음**.

(ii) **해석의 변화**: 너는 "client-level shapley"를 계산하는 거지 "data point shapley를 client-aggregate한 것"이 아니야. 둘은 다른 양임. 후자가 필요하면 client가 local에서 추가 통계를 누적해서 보내야 해 (그러면 ghost technique-like trick이 필요해지고). **너의 design choice가 전자라면, "client-level이 우리의 정의"라고 명시적으로 declare하는 게 깔끔해.** 이게 IRDS와 Flirds의 unit-of-attribution 차이를 정직하게 드러내는 방식이고.

그러므로 권장사항: **(1) round delta = "1 step"으로 취급 + (2) validation에서 1차 근사 quality를 empirically 확인 + (3) 필요하면 2차 항 추가**. 1차로 충분하면 1차로 가고, 부족하면 2차도 closed-form이 있으니 비용도 감당 가능.

## 4. Server-side validation — 동의

너가 든 두 이유 (공정성, 악용 방지) 외에 method 측면에서도 합리적이야. Validation gradient $\nabla\ell(w^r, z^{val})$를 매 round 서버에서 계산할 수 있어야 하는데, 이게 client 측에 있으면 round마다 server↔client 추가 통신이 필요해져 (cross-device에서 치명). 서버 보유면 추가 통신 0.

다만 명시적으로 짚어둘 점: **server-side validation distribution은 어떤 distribution으로 잡을 거야?** 균등하게 모든 domain을 cover? 특정 deployment target distribution? 이게 결국 "valuable client"의 정의를 결정해. Method 자체와는 독립적이지만, 실험 design할 때 명확히 해야 reviewers가 "validation 선택이 결과를 prejudge한 거 아닌가" 같은 지적을 안 해.

---

다음 step으로 넘어가기 전에 한 가지 더 확인하고 싶은 게 있어: **LLM finetuning에서 PEFT (LoRA 등)를 쓸 거야, 아니면 full finetuning?** PEFT면 $\Delta w_k$의 차원이 작아서 dot product 계산이 cheap하고 통신 비용도 작아 — 1차/2차 항 다 부담 없이 들어가. Full finetuning이면 ghost-like trick 부재가 정말 큰 비용으로 돌아옴. 이게 method의 efficiency claim을 좌우할 듯해서 물어봐.