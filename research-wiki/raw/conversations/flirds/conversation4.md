1. 추가로 아무것도 받지 않을거야. FL 상황을 고려하면 지켜야하는 큰 전제 중 하나가 privacy이고 그걸 지켜야되기도 하고, 통신 비용도 고려해야하니까.
2. 서버가 어차피 모든 Δw_k를 다 받을거고, lora를 쓸 예정이기도 하니까 그냥 hvp면 괜찮을것같아.

좋아, 이걸로 method의 핵심 형태가 거의 fix됐어. 정리하고, 짚어둘 만한 점들 적어볼게.

## 확정된 design 요약

- **Unit of attribution**: Client-level
- **What server receives**: $\Delta w_k$ (LoRA params만)
- **Approximation**: 1차 + 2차 Taylor (항상 사용)
- **Validation**: Server-side, 기본 uniform domain coverage
- **Shapley 형태** (round $r$):

$$\phi_k^{(r)} \approx -\nabla\ell(w^r, z^{val}) \cdot \Delta w_k + \frac{1}{2} \Delta w_k^T H^{(val)}(w^r) \cdot \Delta W^{(r)}$$

여기서 $\Delta W^{(r)} = \sum_{j} \Delta w_j$ (그 round 참여 client 모두의 합). 2차 항은 client interaction을 잡는 부분.

총합: $\phi_k = \sum_r \phi_k^{(r)}$.

## 짚어둘 implication

### (1) HVP 비용은 round당 1회로 충분

$H^{(val)}(w^r)$ 자체를 만들 필요 없이 $H^{(val)} v$ 형태로 계산 가능 ($v$는 vector). 그런데 2차 항을 잘 보면:

$$\phi_k^{(r)}(2nd) = \frac{1}{2} \Delta w_k^T H^{(val)} \Delta W^{(r)}$$

모든 client가 _같은_ $H^{(val)} \Delta W^{(r)}$를 공유. 즉 round당 HVP를 한 번만 계산하고 ($u := H^{(val)} \Delta W^{(r)}$), 각 client에 대해 $\Delta w_k^T u$만 dot product. **Round당 HVP 1회 + N개 dot product**. LoRA 차원이 작아서 cheap.

이게 IRDS의 ghost technique을 굳이 안 가져와도 되는 이유와 일치해 — FL의 unit이 client라 N개 vector만 다루면 되고, ghost는 batch 안 N개 sample을 처리할 때 의미 있던 거니까.

### (2) Privacy에 대한 해석을 명시화

"$\Delta w_k$만 받는다"는 표준 FedAvg 가정이지만, 엄밀히 말하면 **secure aggregation은 깨야** Flirds가 동작해 — 서버가 individual $\Delta w_k$를 보지 못하면 client별 valuation 자체가 불가능. 이건 method의 본질적 제약이고, 솔직히 paper에 명시하는 게 좋아:

> "Flirds는 vanilla FedAvg와 같이 server가 individual update를 보는 setup을 가정한다. Secure-aggregation FL과 호환되지 않는 것은 client-level data valuation의 본질적 한계이다."

이게 reviewer가 가장 흔히 던지는 질문 ("이거 secure aggregation에선 못 쓰지 않냐")을 사전에 받아치는 framing.

추가로 차별화 포인트: data-level이 아니라 client-level이고, 추가 정보 안 받기 때문에 **gradient inversion attack 측면에서는 baseline FedAvg보다 더 나쁠 게 없음**. 즉 "Flirds는 추가적인 privacy cost를 부과하지 않는다"는 claim이 가능. 이것도 명시할 가치가 있어.

### (3) Drift residual은 측정만, 보정은 안 함

(1)에서 가능했던 SCAFFOLD-style correction이나 $G_k(w^r)$ 별도 수신은 모두 제외. 그러면 4번에서 도출한 **drift residual은 그냥 valuation 안에 남아있는 것**으로 가야 해. Paper에서 다룰 방식:

- Method 섹션: drift residual의 수학적 형태 도출
- 실험 섹션: $E$를 변화시켜가며 residual의 크기를 empirical하게 측정 + valuation quality (task-based metric) 영향 측정
- "Flirds는 reasonable $E$ 범위에서 robust"임을 보이고, 극단적 $E$에서 degradation을 limitation으로 명시

이게 **honest reporting**이고 reviewer 신뢰도 올림. $E=1$ 극한에서 정확함을 보이고, $E$ 증가에 따른 degradation curve를 보여주면 method의 적용 범위가 quantify됨.

### (4) Non-IID에서의 systematic bias — limitation으로 명시

Drift residual은 client의 local trajectory 길이에 비례하는데, non-IID client일수록 local에서 global과 다른 방향으로 더 멀리 가. 즉 **Flirds는 non-IID client를 구조적으로 "noisy"하게 평가하는 경향**이 있어 — drift residual이 noise로 들어오니까.

이걸 noise vs OOD-good 분리 알고리즘이 일부 보정해줄 수 있어 (cross-round consistency를 보면 drift는 round마다 다르고 진짜 quality는 일관되니까). 이게 **noise 분리 알고리즘이 단순히 "noise vs good" 구분만 하는 게 아니라 "drift bias correction 효과"도 있다**는 두 번째 motivation이 됨. Paper의 narrative를 강화하는 포인트야.

### (5) 통신 비용 claim 정리

이제 명확히 말할 수 있어:

> "Flirds는 vanilla FedAvg 대비 추가 통신 비용이 0이다. 모든 valuation 계산은 server가 이미 받는 ${\Delta w_k}$만으로 수행된다."

이게 GTG-Shapley/FedSV 같은 baseline 대비 강한 차별화 포인트야. 기존 FL valuation 연구는 보통 추가 round를 돌리거나 (model을 client에 추가로 보내서 evaluation), 추가 통신을 발생시켜. Flirds는 IRDS의 "free side product of training" 정신을 "free side product of standard FL communication"으로 정확히 옮긴 것.

## 정리한 method를 한 문장으로

> **Flirds**는 매 FL round에서 server가 받는 client updates ${\Delta w_k}$로부터 server-side validation gradient 및 HVP와의 closed-form 1차/2차 Taylor 식을 통해 client-level Shapley value를 계산하는, 추가 통신/privacy cost가 없는 federated data valuation method이다.

## 다음 step 제안

여기까지가 method의 형태와 narrative. 다음으로 정해야 할 것들:

1. **Noise vs OOD-good 구분 component의 구체적 form**: temporal consistency / cross-client agreement signal을 어떻게 raw Flirds score와 결합할지. 이게 이 논문의 두 번째 contribution이 될 거고, 별도로 깊게 design해야 해.
2. **실험 protocol의 구체화**: 어떤 LLM (Llama-3.2-1B/3B 정도가 LoRA finetuning에 현실적), 어떤 dataset (multi-domain coverage가 가능한 것 — Flan, T0, dolly subset 조합 등), retraining-based baseline용 N=10 cross-silo + task-based N=100~ cross-device 두 트랙.
3. **Baseline method 선정**: GTG-Shapley, FedSV, ComFedSV, Federated Banzhaf, 그리고 simple loss-based heuristic (예: client validation loss)도 포함하는 게 좋음.

이 중 어느 쪽을 다음에 더 파고 싶어? 1번이 알고리즘적으로 가장 무겁고 paper의 두 번째 main contribution이 될 거라 이쪽을 먼저 잡는 게 자연스러울 것 같아.