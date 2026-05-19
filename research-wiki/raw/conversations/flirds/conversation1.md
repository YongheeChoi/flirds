Flirds를 만들기 위한 첫 번째 스탭을 공유해줄게. 애매하거나 더 궁금한 부분이 있으면 물어봐줘. IRDS와 다르게 Flirds는 Federated learning 환경에서 data valuation을 해야해. 그럼 irds를 이 환경에 잘 적용시키기 위해 어떤 차이가 있는지 먼저 고민을 해봐야해.
- client와 server가 소통하는 환경이며, 이떄 서버는 클라이언트의 데이터를 보지 못한다.
- 기여도 평가 대상의 단위가 data sample이 아니라 client가 된다.
- 서버 기준 업데이트가 되는 단위가 1 batch에 대한 step이 아니라, fl에서 정의하는 1 comunication round에서의 local step 단위로 증가한다.
- 클라이언트와 서버가 소통하는 comunication round는 centralized learning에 비해 통신에 큰 cost가 발생한다.
- 각 client는 서로 non-iid한 data를 가지고 있을 수 있다.
- validation data가 클라이언트와 서버 중 어디에 존재할 것인가?
- FL 환경에서의 변동성: round 마다 참여하는 client가 다른 경우, client 마다 data의 개수가 다른 경우

그리고 내가 irds 연구에서 사용하고자 하는 부분은 아래와 같아.
- data valuation에 발생하는 shapley value를 1,2차 근사항을 활용함으로써 다른 모든 subset에 대해 기여도를 계산할 필요 없이 closed-form으로 효율적으로 계산하는 부분.
- 학습 과정에 사용되는 cost를 활용해 shapley value를 계산하는 부분.
- ghost technique은 fl 환경에서 사용하기 적합하지 않다고 판단하여 사용 보류

추가로 IRDS에서 여전히 존재하고 있던 한계점들에 대해서도 고민을 해봐야해.
- validation data가 기여도를 얼마나 대표할 수 있는가?
- utility function이 validation data에 대한 loss가 되는 것이 합당한가?
- validation data에 대한 loss와 반대되는 방향의 업데이트라고 해서 모델의 성능에 기여하지 않은 데이터라는 확신이 없다. 그것이 진짜 noise인지 validation data의 out of distribution에 해당하는 좋은 sample인지 고민을 해봐야한다. 이에 대해 고민을 같이 해줘.
- FL 환경에서 발생하는 차이에 대해 추가로 고려해야 할 사항이 있는가?
- IRDS 자체가 기존에 가지고 있던 한계에 대해 추가로 고려해야 할 사항이 있는가?

좋은 정리야. IRDS 본문도 다시 훑어봤고, 정리한 차이/한계가 핵심을 잘 짚고 있어. 같이 더 파볼만한 지점들을 적어볼게. 일부는 네가 이미 의식하고 있을 텐데, Flirds 설계할 때 결정 포인트가 될 만한 것들이라 명시적으로 짚어두는 게 도움될 거야.

## 1. FL 환경 차이 — 추가 고려사항

**(1) Aggregation rule이 IRDS의 SGD 가정을 깬다 — 가장 큰 issue라고 봐** IRDS의 closed-form은 $w_{t+1} = w_t - \eta_t \sum_{z \in \mathcal{B}_t} \nabla \ell(w_t, z)$ 라는 깔끔한 SGD step에서 나와. 그런데 FedAvg는 round 단위로 보면

$$w^{r+1} = w^r + \sum_k p_k \Delta w_k^r,\quad \Delta w_k^r = w_k^{local,E} - w^r$$

이고, $\Delta w_k^r$ 자체가 여러 local step의 누적이야. 단순 gradient sum이 아니라 **trajectory의 endpoint**. 즉 IRDS의 1차 근사를 그대로 FL round 단위에 적용하면 utility는 $-\nabla \ell(w^r, z^{val}) \cdot \Delta w_k^r$ 같은 형태가 되는데, 이건 IRDS가 자랑하던 "한 step의 Taylor"가 아니라 "여러 step의 Taylor"가 돼. 근사 오차가 $O(\eta^2)$가 아니라 $O((\eta E)^2)$로 커짐. 네가 #3에서 짚은 "step 단위가 round 단위로 바뀐다"의 진짜 함의가 이거야.

**의사결정 포인트**:

- (a) Server에서 round delta만 가지고 valuation → 단순/저비용/근사 오차 큼
- (b) Client가 local step 단위로 gradient 통계를 누적해서 보냄 → 정확/통신비용 증가
- (c) Hybrid: client가 매 local step마다 $\nabla \ell(w_{t,e}^k, z) \cdot g^{val}$ 같은 scalar만 누적 (gradient 자체는 안 보냄). IRDS의 ghost technique 같은 trick을 FL에 맞게 변형하는 길이 여기서 열림.

**(2) Client drift = validation gradient의 staleness** 서버가 가진 validation gradient는 $\nabla \ell(w^r, z^{val})$ — round 시작 시점. 클라이언트의 진짜 gradient는 $\nabla \ell(w_k^{local,e}, z)$ — local에서 이미 drift한 모델. 두 vector를 내적해서 의미를 부여하려면 $w^r \approx w_k^{local,e}$여야 하는데, non-IID + 큰 E일수록 이 가정이 깨져. 이건 단순 근사 오차가 아니라 **clients with different distributions look "less aligned" purely because their local model drifted away** — non-IID 클라이언트를 구조적으로 과소평가할 위험이 있어.

**(3) Validation data 위치는 "어디 둘까"가 아니라 "어떤 distribution을 정답으로 삼을까"** 네가 #6으로 적은 부분을 좀 더 강하게 보면, non-IID 환경에서 validation 선택은 **누가 valuable한 client인지의 정의 그 자체**가 돼. 가능한 setup들:

- Server-side public validation: clean하지만 어느 client distribution과도 안 맞을 수 있음
- 모든 client에서 일부씩 모은 federated validation: 그럼 그 모음을 만드는 과정이 또 privacy 문제
- 각 client local validation: 서버는 어떻게 종합? 평균? max-min fair?
- Validation도 round마다 sample/rotate?

이건 IRDS에서는 그냥 "있다고 가정"하고 넘어간 부분인데, FL에서는 design choice이자 fairness statement야. 논문 contribution으로도 의미 있는 지점이고.

**(4) Privacy와 valuation의 본질적 긴장** Secure aggregation이 켜져 있으면 서버는 individual $\Delta w_k$를 못 봐 — 합만 봄. 이러면 client-level shapley는 원리적으로 불가능. 그럼 Flirds는:

- Secure aggregation을 쓰지 않는 cross-silo setting을 가정하거나
- DP-noised individual update 위에서 동작하거나
- Client가 valuation에 필요한 **scalar**만 self-report (gradient 자체는 안 보냄)

이걸 미리 결정해야 method의 형태가 정해져. 나는 (c)가 IRDS 철학("학습 중에 부수적으로 계산")과 가장 잘 맞을 거 같아.

**(5) Participation의 변동성 — Shapley 공리가 깨질 수 있음** 네가 #7에 적은 부분. Round마다 다른 client가 참여하면, 어떤 client는 100 round 중 80번, 어떤 client는 10번 참여. 단순히 sum of per-round shapley를 하면 자주 참여한 client가 유리해져. 그런데 그게 "데이터의 가치"인지 "참여 자원의 가치"인지 구분이 안 됨. 정규화 옵션:

- Per-round expected contribution × 총 round 수
- 참여한 round들에서의 평균
- Sampling probability로 inverse-weighting

각각이 다른 fairness 의미를 가져. 이것도 명시적으로 결정해야 함.

**(6) Cost = communication cost** 네가 #4에서 통신 비용을 언급했는데, IRDS에서 "학습 중 비용 활용"이라는 motivation이 FL에선 더 강해져 — **추가 round를 돌면 안 되고, round 안에서 client가 추가로 보내는 통신량도 최소화**해야 해. "free side product of training"이라는 IRDS 정신을 "free side product of communication"으로 옮기는 게 Flirds의 제약 조건이자 강점이야.

## 2. IRDS 자체 한계 — 추가 고려사항

**(a) 네가 짚은 #3 (sign ambiguity)에 대해 더 깊이** "validation loss와 반대 방향이라고 noise라는 보장이 없다"는 정확한 진단이야. 구분해볼만한 signature들:

- **Noise**: gradient 방향이 round/step마다 무작위 — 즉 _시간적 일관성이 낮음_
- **Hard but useful (OOD-good)**: 방향이 일관되게 다른 곳을 향함 — _시간적 일관성은 높지만 validation과 misalign_
- **Validation distribution shift indicator**: 다수 client와 misalign되지만 일부 client와는 align

즉 단일 step의 sign만 보면 안 되고 (i) temporal consistency와 (ii) cross-client agreement까지 봐야 noise vs hard data가 구분돼. FL은 오히려 cross-client agreement라는 **추가 신호**가 있어서 centralized IRDS보다 이 한계를 다루기에 유리한 환경이야. 이게 Flirds의 contribution 포인트가 될 수 있음.

**(b) Magnitude vs alignment 혼동 — 잘 안 짚히는데 중요** $-\eta \nabla\ell(w_t, z^{val}) \cdot \nabla\ell(w_t, z)$는 두 벡터의 alignment뿐 아니라 magnitude에 비례. 학습 후반에 잘 fit된 데이터는 gradient가 작아서 score도 작아져. 즉 **모델의 핵심 능력을 정의하는 "쉬운" 데이터가 구조적으로 undervalued**됨. FL에선 이게 더 심해져 — 학습 후반에 join하는 client는 이미 fit된 영역이라 score가 깎이게 돼. 보정 방법은 학습 phase별 normalization이나 cumulative reduction 추적 등이 있을 수 있어.

**(c) Loss reduction ≠ generalization** IRDS는 validation loss의 매 step 변화를 attribute하는데, 학습 동역학상 어떤 데이터는 단기적으로 loss를 올렸다가 장기적으로 generalization을 돕기도 해 (regularization 효과, 또는 model이 robust feature를 배우게 만드는 경우). IRDS는 이걸 음수로 attribute함. FL에서 noise 정체성이 약한 client (예: rare class를 가진 client)가 정확히 이 시나리오에 들어와.

**(d) Trajectory dependence — 이게 우리가 원하는 건가?** IRDS의 contribution은 _이 specific training run에서의 contribution_이야 (paper도 명시). 같은 데이터셋, 같은 학습이라도 batch 순서가 다르면 valuation 결과가 다름. 이건 paper에선 feature지만, **client compensation/incentive design 관점에선 bug에 가까움** — client는 자기 valuation이 그날 batch ordering에 의해 결정되는 걸 받아들이기 어려움. Flirds가 incentive 쪽으로 갈 거면 multiple run averaging이나 expected-value 정의가 필요할 수 있어.

**(e) Validation set의 size/quality가 모든 valuation의 noise floor** IRDS는 validation gradient를 매 step 활용하므로 validation이 작거나 noisy하면 그 noise가 모든 client의 score에 propagate됨. FL에서 validation이 작은 경우가 흔하니까 (특히 server-side라면) 이 sensitivity를 quantify할 필요 있어.

**(f) IRDS paper도 인정한 cancellation effect** A.3에서 last layer "cancellation effect"가 언급돼 — 다른 examples의 influence가 큰 magnitude로 서로 상쇄. FL의 non-IID에서는 이게 더 심함. Layer-wise weighting이나 specific layer 선택이 valuation에 큰 영향을 줄 수 있어.

**(g) "Free reduction"의 attribution** Optimizer dynamics (momentum, Adam의 second moment 같은 것), warmup scheduling 등으로 "데이터와 무관한" loss 감소가 어느 정도 일어나. IRDS는 이걸 그 step에 있던 데이터에 attribute함. Round 단위로 보면 이게 client에게 "공짜 점수"로 분배되는 효과가 있어.

---

## 내가 명확히 하고 싶은 것 몇 가지

방향 잡기 위해 알면 좋을 것들:

1. **FL setting 어디?** Cross-silo (참여자 수 적고 안정적, 회사 간) vs cross-device (수많은 device, 변동성 큼). 두 경우 design이 꽤 달라짐.
2. **Valuation의 최종 목적?** Compensation/incentive, free-rider 검출, client selection, data quality monitoring — 무엇을 우선하느냐에 따라 위의 한계 중 어떤 게 critical issue인지 달라져.
3. **Local epoch $E$ 가정**: $E=1$이면 IRDS를 거의 그대로 쓸 수 있고, $E$가 크면 (1)번 issue 해결이 method의 핵심이 돼.
4. **Validation data를 누가 가지고 있다고 가정**? 이게 정해지면 (3) FL의 issue가 자연스럽게 좁혀짐.