# [한글 검토판] Flirds: Efficient Client-Level Contribution Evaluation in Federated Learning via In-Run Data Shapley

> **이 문서는** Flirds 논문의 **한글 작업본**입니다. 초록·서론·용어·문체 개정은 한글판에서
> 먼저 진행하고, 확정되면 영문 tex(`paper/`)에 반영합니다.

---

## 초록

연합학습(FL)에서 client 기여도 평가는 어떤 참여자의 업데이트가 공동 모델의 성능
개선에 얼마나 기여했는지를 정량화하는 문제다. FL 서버는 client의 원데이터를 갖지
않으므로 부분집합 재학습에 기초한 고전적 데이터 Shapley 평가를 스스로 수행할 수 없고,
기여도는 실제 라운드에서 수신한 업데이트들로부터 평가되어야 한다. 그러나 업데이트를 평가
단위로 삼더라도 라운드마다 참여자 부분집합을 열거해 모델을 재구성하고 평가해야 하므로,
정확한 계산의 비용은 참여자 수에 지수적으로 증가한다.

이를 위해 우리는 client 수준 기여도 추정기 **Flirds**(Federated Learning In-Run Data
Shapley)를 제안한다. Flirds는 실제로 수신한 client 업데이트를 플레이어로 삼고 해당
라운드의 서버 집계 가중치를 모든 부분집합에서 고정하는 협력 게임을 정의하며, In-Run
Data Shapley(IRDS)의 Taylor 계산을 client 업데이트 단위로 확장해 2차 surrogate game의
Shapley 값을 폐형식(closed-form)으로 계산한다. 이로써 지수적인 coalition 평가를, 추가
통신이나 client-측 연산 없이, 라운드당 한 번의 서버 측 Hessian-vector product와
참여자별 내적으로 효율적으로 계산한다.

---

## 1. 서론

연합학습(federated learning, FL)에서는 서로 다른 client(클라이언트)가 원데이터를 공유하지 않은 채
하나의 모델을 공동으로 학습한다. 서버는 매 라운드 참여 client가 보낸 업데이트를 모아
모델을 개선하지만, 그 개선에 각 client가 얼마나 기여했는지는 집계 결과만으로 바로
드러나지 않는다. 이 값은 참여자 보상뿐 아니라 데이터 품질 관리, 무임승차 탐지, client
선별과 학습 과정 디버깅의 근거가 될 수 있다. 협력 게임 이론의 Shapley 값은 명시된
utility를 참여자에게 일관되게 배분하는 원칙을 제공하므로 데이터 및 client 기여도
평가에 널리 사용되어 왔다.

고전적인 Data Shapley는 각 데이터 부분집합으로 모델을 처음부터 학습한 뒤 최종 성능을 비교한다.
그러나 FL 서버는 client의 원데이터를 갖지 않으므로 부분집합 재학습을
스스로 수행할 수 없고, 재학습은 연합 전체를 다시 가동해 client의 협조를 반복 요구하는
일이 된다. 즉 이 정의는 비용 이전에 정보 구조상 서버의 선택지가 아니며, client가
간헐적으로 참여하는 환경의 온라인 평가 수단은 더욱 될 수 없다. 이에 기존
FL-Shapley 연구들은 실제 라운드에서 서버가 수신한 업데이트들을 평가 단위로 삼아
라운드별 기여도를 계산하고 이를 누적해 왔다. 이 경우에도 정확한 Shapley 값을 얻으려면
라운드마다 참여자 부분집합을 열거해 모델을 재구성하고 검증셋에서 평가해야 하므로, 비용이
참여자 수에 따라 지수적으로 증가한다.

따라서 본 논문이 풀고자 하는 문제는 다음과 같다. 서버가 관측한 개별 업데이트와 검증셋만을
사용하여, 실제 FedAvg 라운드가 만든 검증손실 개선을 참여 client에게 Shapley 방식으로
귀속하되, 부분집합별 모델 평가를 피할 수 있는가? 이를 위해서는 모든 참여자를 포함했을 때
실제 서버 집계와 일치하는 게임이 먼저 정의되어야 하며, 그 게임을 부분 참여 환경에서
온라인으로 계산할 수 있어야 한다. 또한 계산된 값이 어떤 게임의 Shapley 값을 근사하는지와
근사 오차가 어디에서 발생하는지도 분리해 설명할 수 있어야 한다.

우리는 이 문제를 위한 client 수준 기여도 추정기 **Flirds**(Federated Learning In-Run
Data Shapley)를 제안한다. Flirds는 실제 라운드에서 사용된 각 client의 가중 업데이트를
플레이어로 보고, 일부 업데이트만 반영했을 때의 검증손실 감소를 라운드 utility로 정의한다.
이때 각 client의 가중치는 부분집합마다 다시 정규화하지 않고 실제 서버 집계에서 사용한
값으로 고정한다. 따라서 모든 참여자를 포함한 경우의 모델 이동은 실제 FedAvg 업데이트와
정확히 일치한다. 이 라운드 게임을 전수 평가하면 라운드별 Shapley 값을 얻을
수 있지만, 여전히 라운드당 지수적인 검증 평가가 필요하다. Flirds는 In-Run Data
Shapley(IRDS)의 계산 아이디어를 이용해 검증손실의 2차 Taylor surrogate가 갖는 Shapley 값을
폐형식으로 계산함으로써 이 열거를 없앤다.

이 구성은 방법을 FL 환경과 직접 연결한다. Flirds가 라운드마다 추가로 요구하는 주요
서버 계산은 검증셋 위의 Hessian-vector product(HVP) 한 번이며, 참여자별 계산은 내적 하나다.
추가 통신이나 client 측 연산은 필요하지 않고, 부분 참여 라운드의 값을 온라인으로
누적할 수 있다. 이처럼 실제 FL 실행에 맞는 게임을 정의하고, 이를 효율적으로 계산하며, 동일한 게임을 전수 열거해 얻은 Shapley 값과 비교하고, 실제 학습 효과를 측정하여 정확성과 실효성 양 측면에서 검증하는 것이 본 논문의 구성이다.

이에 따른 구체적인 기여는 다음과 같다.

1. **관측된 연합 라운드에 대한 Shapley 게임.** 동기식 FedAvg 라운드에서 관측된
   client 업데이트들을 플레이어로, 실제 집계 가중치를 고정한 검증손실 게임을 정식화해
   추정 대상을 명시한다(§3.2, §4.1).
2. **오차 해석 가능한 폐형식 2차 추정기.** IRDS의 2차 Taylor 유도를 client·라운드 단위로 옮겨 관측된 라운드 게임의 surrogate Shapley 값을 폐형식으로 얻어 지수적인 coalition 평가를 피한다. 원래의 라운드 게임을 전수 열거한 Shapley 값과의 차이는 라운드별 Taylor 잔차로 귀착된다(§4.2–4.3).
3. **정확성 및 실효성 검증.** 전수 열거해 얻은 라운드별 Shapley 값과 비교하여 추정치의 정확도를 평가한다. 이어 동일한 개입 정책에서 실제 학습 개선 효과를 검증하고, 각 방법의 계산 비용을 함께 보고한다(§5).

---

## 2. 관련 연구

**연합학습에서의 Shapley 기반 기여도 평가.** 이 계열의 공통 병목은 Shapley가 요구하는 지수적
coalition 평가다. FedSV는 permutation sampling·group testing으로, 저장된 업데이트로 submodel을
재구성하는 GTG-Shapley(Liu et al., 2022)는 guided Monte Carlo와 절단으로 평가할 coalition 수를
줄이고, ComFedSV와 SPACE(Chen et al., 2023)는 low-rank completion과 prototype 기반 utility로
coalition utility 계산 자체를 대체·보완한다. ShapleyFL·FedIF·ShapFed 계열은 Shapley에서 파생된
값이나 gradient 유사도를 강건 집계·선별에 사용한다. Ripple Shapley(Zeng et al.,
2026)는 한 번의 실행에서 sample-level 기여를 현재 라운드의 직접 효과와 이후 라운드로 전파되는
효과로 나눠 후자를 저차원 Jacobian 근사로 누적하지만, 평가 단위가 sample이고 cross-round 전파를
겨냥한다는 점에서 관측된 라운드 게임의 client-level 기여도를 다루는 본 연구와는 평가 대상이
다르다.

**중앙집중 학습의 데이터 기여도 평가.** 중앙학습 환경에서는 학습 주체가 개별 sample과 그
미분 정보에 직접 접근할 수 있고, 이 접근을 전제로 sample-level 귀속이 발전했다:
DataInf·EK-FAC·TRAK·LoGra는 gradient 또는 curvature 정보로 influence와 attribution을
추정하고, LESS·MATES·DsDm은 그 신호를 targeted data selection에 사용한다. In-Run Data
Shapley(IRDS; Wang et al., 2025)는 이 계보에서 본 논문이 출발점으로 삼는 방법으로, 가상의
재학습 대신 실제 학습 실행 하나를 고정하고 step별 검증손실의 Shapley 값을 Taylor 전개로 폐형식 계산해 누적한다. 
Flirds는 플레이어별 변위가 합으로 결합되면 2차 surrogate 게임의 Shapley가 폐형식으로 닫힌다는 IRDS의 계산 구조를 FL 환경에 접목하였다.

**연합 LLM의 품질 평가·선별·시장 설계.** FedDQC는 instruction–response alignment로
client 내부 데이터 품질을 평가하고 학습 순서를 정하며, FedHDS는 표현 공간의
중복성을 이용해 federated instruction tuning용 coreset을 고른다. iPFL은 개인화 모델
시장에서 협업과 지급 규칙을 함께 설계한다. 이 연구들은 federated LLM/FedFM 규모에서
품질 관리, 선별 또는 incentive mechanism이 가능함을 보인다.

---

## 3. 문제 설정과 배경

### 3.1 연합학습 설정과 표기

FedAvg(McMahan et al., 2017)에서는 크기 $n_k$의 로컬 데이터를 가진 client
$k \in [N] := \{1, \dots, N\}$ 중 라운드 $r$의 참여 집합 $P_r \subseteq [N]$이 현재 모델
$w^r$에서 로컬 학습을 수행한다. client $k$가 그 결과로 얻은 모델 차이를
$\Delta w_k^r$로 보내면, 서버는 $r = 0, \dots, R{-}1$에 대해

$$w^{r+1} \;=\; w^r + \sum_{k \in P_r} p_k^r\, \Delta w_k^r, \qquad
p_k^r \;=\; \frac{n_k}{\sum_{j \in P_r} n_j} \tag{1}$$

로 집계한다. 여기서 $p_k^r$는 실제 참여 집합 $P_r$에 대해 서버가 사용한 가중치다.

이때 서버가 client별로 관측하는 것은 원데이터가 아니라 업데이트
$\Delta w_k^r$이며, 실제 라운드의 모델 이동은 이 업데이트들의 가중합이다. 따라서 본
논문은 client가 제출한 업데이트를 평가 단위로 삼고, 실현된 업데이트 벡터와 서버
검증손실을 이용해 각 라운드의 기여도를 평가한다.

### 3.2 Shapley 값 기반 기여도 정의

Shapley 값은 플레이어 집합과 coalition utility가 주어졌을 때 각 플레이어의 한계 기여를
모든 참여 순서에 대해 평균한다. 플레이어 $k$의 값은

$$\phi_k(U) \;=\; \sum_{S \subseteq [N] \setminus \{k\}}
\frac{|S|!\,(N{-}|S|{-}1)!}{N!}\,
\big(U(S \cup \{k\}) - U(S)\big) \tag{2}$$

로 정의된다. 이 값은 명시된 게임 안에서 efficiency, symmetry, null player, linearity를
만족하지만, 어떤 utility가 올바른지는 Shapley 공식 자체가 결정하지 않는다. 따라서
client 기여도 평가에서는 계산법에 앞서 어떤 반사실적 질문에 답할지를 정해야 한다.

고전적인 Data Shapley는 coalition의 데이터만으로 모델을 처음부터 다시 학습했을 때의 최종
성능을 utility로 삼는다. 이는 “이 client들만 학습에 참여했다면 어떤 모델을 얻었을까”를
묻는다. 반면 실제 FL 실행의 라운드별 기여도는 “현재 모델과 실제로 수신된 업데이트를
고정했을 때, 이 업데이트들의 일부가 해당 라운드의 모델 개선에 얼마나 기여했는가”를 묻는다.
전자는 전체 재학습에 대한 counterfactual이고 후자는 실현된 궤적에 대한 attribution이므로,
두 값은 서로 다른 게임의 답이다. 본 논문은 후자를 직접적인 목표로 삼고 전자와의 관계는
실증적으로 별도 분석한다.

### 3.3 In-Run Data Shapley

In-Run Data Shapley(IRDS; Wang et al., 2025)는 가상의 재학습을 반복하는 대신 실제 학습
실행을 고정하고, 각 gradient step을 작은 협력 게임으로 본다. step $t$에서 미니배치 $B_t$의 샘플 $z$별 손실 $\ell_z$와 학습률 $\eta_t$를 써서
$w^{t+1} = w^t - \eta_t \sum_{z \in B_t}\nabla\ell_z(w^t)$로 모델을 갱신할 때, 검증셋 위의 손실을 $\ell_{\mathrm{val}}$이라 하고 배치의
부분집합 $S \subseteq B_t$에 대한 utility를

$$u_t(S) \;=\; \ell_{\mathrm{val}}(w^t)
\;-\; \ell_{\mathrm{val}}\Big(
w^t - \eta_t \textstyle\sum_{z \in S}\nabla\ell_z(w^t)\Big) \tag{3}$$

로 둔다. 즉 일부 샘플의 gradient만 반영했을 때 얻는 검증손실 감소를 해당 coalition의
가치로 본다. 각 step의 Shapley 값을 실제 학습 실행에 걸쳐 합하면 그 궤적에 대한 샘플별
in-run attribution을 얻는다.

IRDS의 계산상 핵심은 플레이어별 모델 변위가 합으로 결합된다는 점이다. 검증손실을 현재
모델에서 Taylor 전개하면 1차항은 각 플레이어의 변위와 검증 gradient의 정렬을 나타내고,
2차항은 같은 step의 다른 변위와 결합될 때 생기는 곡률 효과를 나타낸다. 검증손실의
gradient와 Hessian을 $g_t := \nabla\ell_{\mathrm{val}}(w^t)$,
$H_t := \nabla^2\ell_{\mathrm{val}}(w^t)$로 두면 2차 surrogate의 Shapley 값은

$$\phi_z(u_t) \;\approx\;
\eta_t\big\langle g_t,\nabla\ell_z(w^t)\big\rangle
\;-\; \tfrac{\eta_t^2}{2}
\Big\langle\nabla\ell_z(w^t),\,
H_t\textstyle\sum_{z' \in B_t}\nabla\ell_{z'}(w^t)\Big\rangle \tag{4}$$

로 정리된다. 중요한 점은 Hessian 전체나 역행렬이 아니라, 전체 step 방향에 대한
Hessian-vector product 하나로 모든 플레이어의 2차항을 계산할 수 있다는 것이다.

FedAvg의 가중 업데이트 합은 이 계산 아이디어를 client 수준에 적용할 수 있는 구조를
제공한다. 다만 샘플 gradient를 client 업데이트로 치환하는 것만으로는 게임이 정해지지
않는다. IRDS의 step 게임에서는 전개 구간이 한 스텝이고, 배치의 모든 샘플이 같은 계수를
받으며, 플레이어 집합이 배치로 주어져 있어 묻혀 있던 물음이 FL에서는 셋 남는다.
① **게임의 단위**: 서버는 스텝이 아니라 여러 로컬 스텝이 누적된 결과만 관측하므로
전개 구간이 스텝보다 훨씬 커지고, 스텝이 작다는 전제에 기대던 근사의 지위가 달라진다.
② **부분집합의 계수**: client마다 $p_k^r$이 다르므로 남은 참여자로 다시 정규화할 수도
있으며, 어느 쪽을 택하느냐에 따라 게임 자체가 갈린다. ③ **참여자 누적**: 라운드마다 플레이어
집합이 달라, 라운드별 값을 더한다는 것이 무엇을 뜻하는지부터 규정해야 한다. 다음 절은 이
셋을 정하고 각 결정이 무엇을 보장하는지 보인다.

---

## 4. 방법

> **그림 1 (자리).** (a) 관측된 라운드 게임(식 5): 실제 집계 가중치로 합성된 coalition
> 이동과 검증손실 감소, grand coalition = 실제 서버 집계. (b) 세 값의 관계: 두 Shapley 값은
> 서로 다른 게임의 답(관계는 실증, §5.2)이고, Flirds는 in-run 쪽 게임의 2차 surrogate가
> 갖는 Shapley 값이다(유일한 오차 = 절단). 하단 = 비용(재학습 $2^N$ vs forward
> $\sum_r 2^{|P_r|}$ vs 라운드당 HVP 1회). §4.1이 (a)의 게임을, §4.2가 폐형식 추정기를,
> §4.3이 (b)의 오차 구조와 공리 성립을 다룬다.
> <!-- 제작: paper/AAAI/Figures/figure1_flirds_concept_prompt.md (유일 최신본, 07-27:
>      §4 명칭 in-run Shapley / retraining-based Shapley · 부분 참여와 로컬 스텝 묶음 명시
>      · Flirds = surrogate가 갖는 Shapley 값)로 생성 -->
> <!-- 캡션이 §4 도입 문단을 대체한다(07-27). §3.3의 세 물음은 §4.1이 단위·계수·누적
>      순서로 그대로 받는다. -->

### 4.1 연합 라운드 게임

FL의 라운드 구조는 IRDS의 step 구조와 형식적으로 대응한다: 식 (3)의 step 게임에서 샘플의
gradient 항 $-\eta_t \nabla \ell_z$ 자리에 라운드의 per-client 가중 업데이트
$p_k^r \Delta w_k^r$를 넣으면 같은 가산 구조를 client 업데이트 단위로 정의할 수 있다.
실현된 궤적 $\{w^r, \{\Delta w_k^r\}_{k \in P_r}\}_{r=0}^{R-1}$을 고정하고, 라운드별 coalition utility를

$$u_r(S) \;=\; \ell_{\mathrm{val}}(w^r) \;-\; \ell_{\mathrm{val}}\Big(w^r + \textstyle\sum_{k \in S} p_k^r\, \Delta w_k^r\Big),
\qquad S \subseteq P_r \tag{5}$$

로 정의한다. 플레이어는 여러 번의 로컬 스텝을 거쳐 제출된 client의 가중 업데이트이고, 계수
$p_k^r$은 부분집합 안에서 다시 정규화하지 않고 실제 집계가 사용한 값을 그대로 쓴다. 이때,
모든 참여자를 포함한 coalition의 이동은 서버가 실제로 수행한 집계와 정확히 일치한다.
스텝이 아니라 라운드를 단위로 삼았으므로 이 게임은 IRDS의 step 게임과 다른 게임이고(부록
A.2), 계수를 남은 참여자로 다시 정규화한 게임 역시 다른 게임이다(부록 A.3).

이 게임의 Shapley 값을 라운드당 $2^{|P_r|}$개 부분집합 **전수 열거**로 계산해 라운드에 걸쳐
누적한 값을 **in-run Shapley** $\phi^{\mathrm{in}}$이라 부른다:
$\phi_k^{\mathrm{in}} = \sum_{r\,:\,k \in P_r} \phi_k(u_r)$. 라운드마다 플레이어 집합이
다르지만, 그 라운드에 참여하지 않은 client는 제출한 업데이트가 없어 coalition에 넣어도
이동을 바꾸지 못한다($u_r(S) = u_r(S \cap P_r)$). 따라서 모든 라운드 게임은 전체 client
집합 위의 게임으로 읽히고, 위 합은 학습 전체를 하나로 본 게임
$U^{\mathrm{in}} := \sum_r u_r$의 Shapley 값과 정확히 일치한다(부록 A.2). 라운드별 efficiency $\sum_k \phi_k(u_r) = u_r(P_r)$에
grand coalition의 이동이 곧 실제 집계라는 사실을 넣으면
$u_r(P_r) = \ell_{\mathrm{val}}(w^r) - \ell_{\mathrm{val}}(w^{r+1})$이므로, 라운드를 더할 때
중간 항이 모두 상쇄되어 전 client 값의 합이 학습 전체의 검증손실 개선과
일치한다:
$\sum_k \phi_k^{\mathrm{in}} = \sum_{r=0}^{R-1} u_r(P_r) = \ell_{\mathrm{val}}(w^0) - \ell_{\mathrm{val}}(w^R)$.

비교를 위해 데이터 가치평가의 고전적 정의를 client 수준으로 옮긴 값도 함께 정의한다.
client 부분집합 $S$만 참여시켜 $R$ 라운드를 처음부터 재학습한 최종 모델
$w_S^R$($w_\emptyset^R = w^0$)의 검증손실 감소 $U^{\mathrm{re}}(S) = \ell_{\mathrm{val}}(w^0) - \ell_{\mathrm{val}}(w_S^R)$에 대한 Shapley
값을 $2^N$개 부분집합 **전수 재학습**으로 계산한 값이며, IRDS의 용어를 따라 이를
**retraining-based Shapley** $\phi^{\mathrm{re}}$라 부른다($\phi_k^{\mathrm{re}} = \phi_k(U^{\mathrm{re}})$).
두 값의 관계는 §5.2에서 실증적으로 특성화한다.

### 4.2 폐형식 추정기

in-run Shapley 전수 열거는 근사가 없는 대신 라운드당 $2^{|P_r|}$번의 검증 평가를 요구한다.
Flirds의 출발점은, 라운드 utility를 $w^r$ 주변에서 2차까지 Taylor 전개한 근사 게임, 곧
**2차 surrogate 게임** $\hat u_r$의 Shapley 값이 폐형식으로 나온다는 것이다(IRDS Thm 4; 부록 A.3).
$g_r := \nabla \ell_{\mathrm{val}}(w^r)$,
$H_r := \nabla^2 \ell_{\mathrm{val}}(w^r)$,
$\delta_k^r := \Delta w_k^r$, coalition $S$의 집계 이동
$\Delta_S^r := \sum_{k \in S} p_k^r\, \delta_k^r$, 그리고 $\Delta W_r := \Delta_{P_r}^r$라 하면:

$$\hat\phi_k^{(r)}
\;=\;
-\,p_k^r\, \big\langle g_r,\, \delta_k^r \big\rangle
\;-\; \tfrac{1}{2}\, p_k^r\, \big\langle \delta_k^r,\; H_r\, \Delta W_r \big\rangle,
\qquad
\hat\phi_k = \sum_{r\,:\,k\in P_r} \hat\phi_k^{(r)}. \tag{6}$$

1차 항은 $k$의 업데이트가 혼자서 검증손실의 하강 방향과 얼마나 정렬돼 있는지를 잰다. **2차 곡률항**은 $k$의 업데이트가 라운드의 집계 이동 전체와 곡률을 통해 결합되는 몫을 배분한다. 여기서 2차 항을 제거한 변형을 **Flirds-1st**라 부르고 2차 곡률항의 ablation으로 사용한다. 식 (4)와 1차 항의 부호가 다른 것은 여기서는 $\delta_k^r$가 변위이기 때문이다.

이 폐형식이 성립하는 것은 coalition의 이동 $\Delta_S^r$가 coalition-독립 벡터
$p_k^r\,\delta_k^r$들의 합으로 남기 때문이고, 이는 §4.1에서
계수를 부분집합과 무관하게 고정했기에 생기는 구조다. 계수를 재정규화하면 이동이 어떤
벡터 배정으로도 이런 합으로 표현되지 않아 이 유도 자체가 시작되지 않는다(부록 A.3).

**비용.** 라운드당 무거운 연산은 검증 gradient $g_r$와 HVP $H_r \Delta W_r$뿐이고, 둘은 gradient 함수 위에
forward-mode 미분을 겹친(forward-over-reverse) 한 번의 Jacobian-vector product(JVP)
호출로 함께 얻는다. 참여자별 계산은 $v_r := g_r + \tfrac12 H_r\Delta W_r$와의 내적 하나다(Algorithm 1;
부수 비용은 파라미터 차원 $d$에 대해 $O(|P_r|\,d)$ 내적과 라운드 내 업데이트 보관). Hessian을 만들거나 역행렬하지
않으며, 비용은 부분집합 수 $2^{|P_r|}$과 무관하다.

```text
Algorithm 1 — Flirds (서버 측; 라운드 r 한 번)
입력: 현재 모델 w^r, 참여 집합 P_r, 업데이트 {δ_k^r}, 데이터 수 {n_k}, 검증셋 D_val
1: p_k^r ← n_k / Σ_{j∈P_r} n_j                    ▷ 실제 집계 가중치 (식 1)
2: ΔW_r ← Σ_{k∈P_r} p_k^r δ_k^r                   ▷ 서버가 어차피 계산하는 집계 이동
3: (g_r, h_r) ← jvp(∇ℓ_val, w^r; ΔW_r)            ▷ 한 호출: primal = g_r, tangent = H_rΔW_r
4: v_r ← g_r + ½ h_r
5: for k ∈ P_r:
6:     φ̂_k ← φ̂_k − p_k^r ⟨δ_k^r, v_r⟩             ▷ 식 (6)과 동치; 참여자당 내적 1회
7: w^{r+1} ← w^r + ΔW_r                           ▷ 표준 FedAvg 집계 그대로 (식 1)
출력: 누적 기여도 φ̂ (client 추가 연산·통신 없음; 동결 로그에서 사후 재계산도 동일)
```

### 4.3 Shapley 공리와 근사 오차

Flirds는 식 (5)의 비선형 게임 자체를 정확히 계산하지 않는다. 정확히 계산하는 것은 같은
고정 궤적·같은 집계 가중치·같은 전개점을 쓰는 **2차 Taylor surrogate의 Shapley 값**이며,
따라서 in-run Shapley와의 차이는 라운드별 Taylor 절단뿐이다.
이 근사 게임 위에서 Shapley 공리는 정확히 성립하므로 공리 위반이 생기지
않고(IRDS Remark 3), 공리 성립은 FL 운영에 직접 쓰이는 세 성질로 구체화된다.
① client는 참여하지 않은 라운드의 게임에서 null player이므로 받는 값이 0이고, 서버는
매 라운드 그 라운드 참여자에게만 값을 주고 더해 가면 된다. ② 전 참여 라운드에서
$\delta_k^r = 0$인 zero-update free-rider도 null player이며, 식 (6)에서는 두 항 모두
0-벡터와의 내적이라 **대수적으로 정확히** $\hat\phi_k = 0$을 받는다. ③ 값의 합은
surrogate 게임의 전체 개선과 일치한다(efficiency).

그렇다면 그 오차는 얼마나 큰가. 식 (6)은 검증손실의 Taylor 전개를 2차항까지 취한 것이므로, 오차는 절단된 3차항부터의 꼬리다(1차 surrogate는 2차항부터). IRDS의 step 게임에서는 전개 변위가 $-\eta_t \sum_{z \in B_t}\nabla\ell_z$여서 이 꼬리가 학습률의 3제곱 $O(\eta_t^3)$으로 떨어지고, 학습률이 작다는 사실이 절단을 정당화한다. 여기서는 전개 변위가 여러 로컬 스텝이 누적된 coalition의 라운드 이동 $\Delta_S^r$이므로 같은 논리가 단위만 바꿔 적용된다: 잔차는 $O(\|\Delta_S^r\|^3)$(1차 surrogate는 $O(\|\Delta_S^r\|^2)$)이고, Shapley 값은 coalition marginal들의 가중평균이므로 이 차수는 player 수준 값과 라운드 누적에 그대로 전파된다. 다만 라운드 이동은 한 스텝만큼 작다는 보장이 없으므로 절단이 자동으로 정당화되지는 않는다. 조건·상수를 명시한 상계와 그 한계는 부록 A.4에 두고, 실제 오차의 크기는 같은 게임을 전수 열거한 값과의 비교로 잰다(§5.2).

---

## 5. 실험

> **작업본 표기 규약.** ⬚ = 실행 중·대기 실험의 자리(채울 분석 파일은 각 위치의 HTML 주석에
> 명시; 수치는 rundir/분석 스크립트 재생성 값만 기입). ◐ = 1-seed. ● = 3-seed 확정. 별도 표기 없으면 3-seed mean±sample
> std이다. 표가 평균만 싣는 경우 표 제목에 이를 명시하고 같은 표의 seed별 값 또는
> mean±std 보조표를 가리킨다. paired arm은 같은 seed의 차이를 먼저 계산한 뒤
> $\mathrm{mean}\pm\mathrm{std}$를 보고한다. 결과표의 데이터 셀은 선택적 굵은 표시를 쓰지
> 않으며, 비교의 핵심은 본문에서 서술한다. 여러 위협 열을 가진 표에는 평균 열을 병기한다.
> ±가 없는 평균 열은 표시된 열 평균값들의 단순 평균이고, ±가 있는 평균·오염-평균 열은 그
> 표의 규약대로 seed별 평균을 먼저 계산한 값이다.

실험은 §1의 검증 두 겹을 핵심 질문의 위계 순으로 배치한다. 1차는 fidelity(충실도), 즉
추정값이 채점 기준인 Shapley 값의 순위·값을 재현하는가이다(§5.2). 2차는 실효성, 즉 측정한
기여도로 학습을 실제로 개선할 수 있는가이다(§5.3). 이어서 방법별 계산 비용(§5.4)을 본다. 
본문의 공간 제약으로 인해 프로토콜 상세와 재현성 분석, 추가 실험 결과 등은 부록에 남겼다.

### 5.1 실험 세팅

**세팅.** 주 세팅은 LLM·CNN 한 쌍이며, 두 트랙 모두 "오염 client가 섞인 부분 참여
FL"이라는 같은 구도를 공유한다. LLM 트랙은 Llama 계열 모델을 GSM8K/Alpaca-GPT4/자체 구성 5-domain 데이터 등을 통해 평가했고, CNN 트랙은 FedSVCNN/LeNet5 모델을 CIFAR-10/MNIST/Fashion-MNIST를 통해 평가하였다.
CNN 트랙의 클라이언트 분할은 IID 균등 분할과 Dirichlet($\alpha{=}1$) 비IID 분할 두 가지이며, 그림과 표에서는 후자를 Dir(1)로 줄여 적는다. 오염 축은 다섯이다: **clean**(오염 없는 대조 앵커), **answer-swap**(형식은 정상 데이터이나 문항-응답 대응을 깨는 순열 교체; LLM), **zero-update free-rider**(로컬 학습 없이 0 업데이트 제출), **gradient noise**(정상 학습
후 업데이트에 가우시안 노이즈 주입; CNN), **label-flip**(라벨 무작위 교체; CNN).

**비교 방법.** 비교에는 기여도 방법 7종을 쓴다. Flirds는 본 논문에서 제안하는 식 (6)의 2차
폐형식 추정기이고, Flirds-1st는 같은 식에서 1차 항만 남긴 변형이다. 여기에 §2의 선행 계보를
대표하는 다섯을 함께 둔다. GTG-Shapley와 FedSV는 저장된 업데이트로 재구성한 coalition의
utility를 coalition 안에서 다시 정규화한 가중치로 정의하고, ComFedSV는 일부만 채운 utility
행렬을 low-rank 완성으로 메우며, ShapleyFL과 FedIF는 각각 Shapley에서 파생한 값과 gradient
유사도 기반 influence를 client 점수로 쓴다.

**지표.** 부호 규약부터 적는다. 식 (5)의 $u_r$은 검증손실의 감소량이므로 손실을 낮출수록
기여도가 높다고 여긴다. fidelity는 Spearman $\rho$(순위)·Pearson $r$(값)을 본문에, Kendall
$\tau$·거리 3종(cosine/euclidean/max)을 부록 C에 둔다. 개입은 성능을 직접 읽는다. CNN은
test 정확도, LLM은 생성한 최종 답의 정답 일치율인 exact match(EM)를 쓰고, vanilla(개입
없음)·oracle-제외(오염 client를 정확히 제외한 참조 arm)와 함께 본다.

### 5.2 Fidelity

이 절은 1차 질문에 답한다. Flirds와 Flirds-1st가 식 (5)의
라운드 게임을 전수 열거해 얻은 in-run Shapley를 얼마나 충실히 재현하는가이다.
셀(세팅 × 위협 × seed)마다 학습 궤적 하나를 동결하고 모든 방법과 in-run
Shapley를 같은 로그 위에서 채점하므로, 방법 간 차이는 궤적 분산이 아니라 방법의 몫이다. 순위
재현은 Spearman, 값 재현은 Pearson으로 읽는다.

In-run Shapley 대비 채점에 올리는 것은 같은 게임을 추정하는 Flirds-류 뿐이다. 다른 baseline들은 애초에 다른 게임을 추정 대상으로 삼으므로 공정한 비교가 되지 않는다. 그래서 비교군 전부를 한자리에서 보려면
채점 기준을 바꿔야 하고, 이어지는 retraining-based Shapley 대비 채점이 그 자리다. 이 값은
부분집합만으로 처음부터 재학습해 얻는 고전적 정의의 Shapley 값이며, 어떤 방법의 추정 대상도
아니면서 전 방법에 같은 잣대로 적용된다는 뜻에서 공통의 외부 비교점이 된다.

**두 트랙의 재현.** 두 트랙의 결과가 그림 2다. CNN 트랙(a)에서 Flirds는 두 파티션의 네 위협
모두 $\rho \geq .847$·$r \geq .937$을 유지한다. LLM 트랙(b)은 GSM8K 위협 축과 1B에서 7B로
키우는 모델 규모 축의 여섯 셀 전부에서 두 변형이 $\rho \geq .994$·$r \geq .997$을 지키고
Flirds는 $\rho \geq .999$다. 다만 LLM 트랙은 client 간 실제 차이가 작은 무대라 절대 상관
자체가 판별력을 갖지는 않는다. 이 패널이 지지하는 것은 규모와 세팅을 바꿔도 겨냥한 값을
놓치지 않는다는 데까지다.

> **그림 2.** in-run Shapley 재현. (a) CNN ($N{=}100$·10/100 참여·$R{=}120$·오염 40%), (b) LLM 두 세팅(GSM8K 위협 축·alpaca 모델 규모 축)


**2차 곡률항이 값을 가르는 곳.** 두 변형이 갈리는 칸은 gradient noise 하나다. 여기서 Flirds는
$\rho$ .847(Dir(1))·.870(IID)을 지키는 반면 Flirds-1st는 .218·.313으로 내려가고, 값 수준에서는
$r$가 −.051까지 떨어져 부호마저 뒤집힌다. 노이즈가 실린 업데이트는 검증 gradient와의 정렬만으로는
benign과 구별되지 않고 그 결합 벌점이 2차 항에만 실리기 때문이며, 같은 방향이
개입에서도 재현된다. 즉 2차 항은 모든 레짐에서 필요한 것이 아니라 곡률이 오염과 결합하는 레짐에서 1차
근사의 실패를 막는다.

**retraining-based Shapley 대비.** 채점 기준을 바꾸면 무엇이 달라지는지는 두 Shapley 값을 모두
전수 계산할 수 있는 $N{=}10$ 세팅에서 읽는다(표 [F2]). 먼저 볼 것은 방법이 아니라 첫 행이다.
두 Shapley 값의 일치 자체가 칸에 따라 다르다. 오염이 있는 세 칸에서는 +.848~+.947이지만 clean
칸에서는 +0.515로 내려가고 seed 간 편차가 ±0.409로 커진다. Flirds는 그 값을 거의 그대로 따라가
네 위협 평균이 in-run Shapley 자신의 +0.797 대비 +0.779다. 즉 Flirds가 retraining-based
Shapley와 벌어지는 폭은 두 게임 자체의 간극과 구별되지 않는다. Flirds-1st는 여기서도 gradient
noise에서만 −0.184로 무너져 앞 문단의 판정이 채점 기준을 바꿔도 유지된다. 타 게임을 겨냥하는
계열은 zero-update free-rider에서 갈린다(FedSV +0.131, ShapleyFL −0.211). coalition을 다시
정규화하면 0 업데이트를 낸 client에게도 몫이 배분되기 때문이다. 다만 이 표를 방법 우열의
판정으로 읽지는 않는다. retraining 게임 자체가 부분집합 크기로 가중치를 다시 정규화하는
구조여서 재정규화 계열과 구조적으로 가까운 칸이 있고(clean 칸의 GTG +0.766), 어느 방법도 이
값을 추정 대상으로 삼지 않기 때문이다.

표 [F2] — retraining-based Shapley 대비(CIFAR-10 Dirichlet($\alpha{=}1$); $N{=}10$
전원참여·$R{=}10$·오염 4/10; 두 Shapley 값 모두 $2^{10}$ 전수 계산). 전 칸이
retraining-based Shapley 대비 Spearman $\rho$의 3-seed mean±std이고, 첫 행은 방법이 아니라
in-run Shapley 자신의 값이다

| 방법               | clean *(대조)* | zero-update  | gradient noise | label-flip   | 평균           |
| ---------------- | ------------ | ------------ | -------------- | ------------ | ------------ |
| in-run SV *(앵커)* | +0.515±0.409 | +0.848±0.116 | +0.947±0.019   | +0.879±0.032 | +0.797±0.082 |
| Flirds           | +0.515±0.401 | +0.848±0.116 | +0.883±0.120   | +0.871±0.043 | +0.779±0.053 |
| Flirds-1st       | +0.438±0.335 | +0.802±0.194 | −0.184±0.303   | +0.855±0.085 | +0.478±0.114 |
| GTG-Shapley      | +0.766±0.266 | +0.596±0.214 | +0.822±0.067   | +0.911±0.050 | +0.774±0.121 |
| FedSV            | +0.648±0.430 | +0.131±0.201 | +0.830±0.021   | +0.907±0.043 | +0.629±0.163 |
| ComFedSV         | +0.616±0.303 | +0.265±0.378 | +0.600±0.378   | +0.907±0.007 | +0.597±0.091 |
| ShapleyFL        | +0.596±0.337 | −0.211±0.044 | +0.838±0.031   | +0.883±0.046 | +0.527±0.114 |
| FedIF            | +0.531±0.410 | +0.565±0.094 | +0.640±0.043   | +0.701±0.172 | +0.609±0.130 |

### 5.3 기여도 기반 개입

실효성 검증을 위해 측정한 기여도를 실제 학습 결정에 개입시키면 성능이 나아지는지를
본다. 기여도 $\hat\phi_k \le 0$인 client를 학습에서 배제하는 sign-gating 규칙을 따라서, 
§5.2와 같은 실험 환경에서 측정된 fidelity가 다운스트림 우열로 이어지는지 직접
시험한다. 본문에서는 최초 1회 학습 과정에서 기여도를 계산한 후, 누적 기여도의 부호로 
남길 client 집합을 정한 뒤 동일한 초기값부터 다시 학습하는 selection-retrain 실험을 보고한다. 매 라운드
학습에 참여할 클라이언트를 선별하는 online 방식은 부록 F.1에 둔다. 결과적으로 전체 방법론을 Vanilla(observer), 오염 client를 정확히 제외한 oracle-제외, 같은 크기의 집합을 무작위로 남긴 selection-random과 함께 비교한다.

표 [I2] — CNN 트랙(CIFAR-10·Dirichlet($\alpha{=}1$); $N{=}100$·10/100 참여·$R{=}120$·
오염 40%), test 정확도(%), sign-gating selection-retrain

|                    | clean       | zero-update  | gradient noise | label-flip   | 평균    |
| ------------------ | ----------- | ------------ | -------------- | ------------ | ----- |
| vanilla (observer) | 63.89±0.52  | 58.79±0.29   | 24.36±2.22     | 52.47±2.89   | 49.88 |
| oracle-제외          | –           | 62.03±0.28   | 62.03±0.28     | 62.36±0.31   | –     |
| selection-random   | –           | 58.38±1.65   | 25.90±1.70     | 50.18±4.97   | –     |
| Flirds             | 62.77±0.96  | 60.63±0.72   | 60.65±0.33     | 61.92±0.80   | 61.49 |
| Flirds-1st         | 63.86±0.53  | 62.52±0.24   | 24.36±2.22     | 62.36±0.31   | 53.28 |
| FedIF              | 64.17±0.31  | 62.52±0.24   | 24.36±2.22     | 62.17±0.58   | 53.31 |
| GTG                | 62.65±0.86  | 51.58±0.47   | 62.03±0.28     | 59.91±1.47   | 59.04 |
| FedSV              | 61.66±0.96  | 51.40±0.81   | 62.03±0.28     | 59.04±1.01   | 58.53 |
| ComFedSV           | 62.32±1.22  | 52.00±1.53   | 62.03±0.28     | 59.21±1.02   | 58.89 |
| ShapleyFL          | 62.23±1.18  | 51.13±1.64   | 62.03±0.28     | 60.28±0.68   | 58.92 |

표 [I2]는 fidelity의 CNN 트랙 중 CIFAR-10·Dirichlet($\alpha{=}1$)와 같은 세팅에서 기여도 기반 개입 학습을 한 결과이다.
Flirds는 세 오염 위협 모두에서 60.6% 이상을 지켜 오염 client를 정확히 아는
oracle-제외(62.0~62.4%)에 가장 근접하다. 위협별로는
서로 다른 성질이 드러난다. zero-update free-rider에서는 0을 정확히 주는지가 갈림길이다.
$\hat\phi_k = 0$이 대수적으로 보장되는 계열(Flirds·Flirds-1st·FedIF)이
60.6~62.5%인 반면, coalition 안에서 가중치를 다시 정규화하는 4종은 51.1~52.0%로
vanilla(58.8%)보다도 낮다. 재정규화가 zero-update client에게도 몫을 배분해, 게이트가 오염
client는 남기고 정상 client를 내보내기 때문이다. gradient noise는 2차 곡률항의 판별
칸이다. 도함수를 쓰는 저비용 방법 가운데 Flirds만 60.65%로 회복하고, 1차 정보만 쓰는
Flirds-1st와 FedIF는 게이트가 발화하지 못해 vanilla와 같은 24.36%에 머문다.
label-flip에서는 전 방법이 59.0~62.4%로 몰려 방법 간 차이가 작다. clean 칸에서는 Flirds가
정상 client를 일부 배제해 vanilla보다 낮다(62.77% vs 63.89%).
요컨대 Flirds의 강점은 모든 칸의 단독 1위가 아니라, 2차 곡률이 필요한 칸에서 같은 비용대의
비교군과 갈린다는 데 있다.

표 [I1] — LLM 주 세팅(GSM8K·Llama-3.2-1B-Instruct; $N{=}50$·5/50 참여·$R{=}200$·오염 40%),
test 1,119문항 EM(%), sign-gating selection-retrain, 3-seed mean±std ●

|                    | clean *(대조)* | answer-swap | zero-update free-rider | 평균    |
| ------------------ | ------------ | ----------- | ---------------------- | ----- |
| vanilla (observer) | 36.52±1.29   | 32.74±0.69  | 35.60±1.57             | 34.95 |
| oracle-제외          | –            | 36.25±1.21  | 36.25±1.21             | –     |
| selection-random   | –            | 32.80±0.59  | 34.76±1.79             | –     |
| Flirds             | 36.52±1.29   | 34.79±0.37  | 36.25±1.21             | 35.85 |
| Flirds-1st         | 36.52±1.29   | 34.58±0.18  | 36.25±1.21             | 35.78 |

표 [I1]는 fidelity의 LLM 트랙 중 GSM8K와 같은 세팅에서 기여도 기반 개입 학습을 한 결과이다.
answer-swap에서 sign-gating은 EM 34.79%로 vanilla(32.74%)를 2.1%p 올리고, 같은 수의 client를
무작위로 남긴 selection-random(32.80%)과 뚜렷이 갈린다. 이득이 동수 배제 자체의 효과가
아니라 기여도 부호가 고른 집합의 효과라는 뜻이다. 다만 oracle-제외(36.25%)에는 못 미치므로
게이트가 오염 client를 모두 걸러내지는 못한다. zero-update free-rider에서는 두 방법
모두 kept가 정확히 정상 client 30명이 되어 oracle-제외와 같은 재학습이 되고 EM도 같다. §4.3
②의 대수적 0이 정책 결정으로 그대로 번역된 칸이다. clean에서는 세 seed 모두 kept가 50명
전원이라 오발화 없이 vanilla와 값이 같다.

### 5.4 비용

이 절에서는 기여도 평가를 위한 계산 비용에 대해 이야기한다. 측정 대상은 서버가 기여도 평가에
추가로 쓰는 연산이고, client의 로컬 학습 시간은 분리해 비교의 기준선으로만 쓴다. 축은 둘이다.
하나는 라운드당 지배 연산(HVP·gradient·forward)의 횟수를 방법별로 닫아 두는 하드웨어 독립
연산수 모델이고, 다른 하나는 같은 궤적·같은 스택에서 잰 wall-clock이다. 둘을 함께 보는 이유는
실측만으로는 재구현·정밀도·하드웨어에 따라 값이 달라지고, 연산수만으로는 실제 부담이 보이지
않기 때문이다. 이 절이 세우려는 것은 어느 방법이 언제나 싸다는 주장이 아니라, 방법 간 배율이
라운드 참여자 수의 함수로 예측된다는 것이다.

**연산수와 실측.** 폐형식이 전 client에 공유하는 벡터 $v_r = g_r + \tfrac12 H_r \Delta W_r$를 한 번
만들면 client별 계산은 내적뿐이므로(§4.2), Flirds의 무거운 연산은 라운드당 HVP 1회이고 라운드
참여자 수 $K$와 무관하다. 반면 coalition마다 모델을 재구성해 검증셋에서 평가해야 하는 in-run
Shapley는 라운드당 $2^K$ forward를 요구한다. 차수 차이는 $K$가 클 때 드러나므로, 표 [C1]은
연산수 옆에 라운드 참여가 10인 세팅의 실측을 나란히 놓았다.

표 [C1] — 라운드당 지배 연산과 그것이 실제로 든 시간. 연산수는 해석적 카운트이고($K$ = 라운드
참여자 수, $N$ = 전체 client 수, $c$ = 절단 상수, $M = \max(10, \lceil N \ln N\rceil)$),
wall-clock 세팅은 $N{=}100$·10/100 참여·$R{=}30$·Llama-3.2-1B

| 방법                | 라운드당 지배 연산                        | wall-clock(초) |
| ----------------- | --------------------------------- | ------------- |
| Flirds            | 1 HVP $+\,K$ 내적                   | 157.3±5.4     |
| Flirds-1st        | 1 gradient                        | 53.8±2.3      |
| FedIF             | 1 gradient                        | 53.7±2.3      |
| ComFedSV          | $\le 1 + \min(2^K,\; MK)$ forward | 357.9±18.2    |
| FedSV             | $\le \min(2^K,\; cK)$ forward     | 4,969±200     |
| GTG-Shapley       | $\le \min(2^K,\; cK)$ forward     | 18,149±1,592  |
| ShapleyFL         | $2^K$ forward                     | 24,935±1,123  |
| in-run SV         | $2^K$ forward                     | 24,975±1,115  |

Flirds는 in-run Shapley의 1/159이고, 이는 연산수 모델의 예측($2^{10}/6.5 \approx 158$)과 일치한다.
ShapleyFL·GTG-Shapley·FedSV가 같은 자릿수의 비용을 무는 것은 세 방법도 라운드마다 coalition
모델을 재구성하기 때문이며, 표본 수를 줄이는 절단은 상수를 낮출 뿐 $K$ 의존성의 차수를 바꾸지
않는다. 2차 곡률항의 가격은 Flirds와 Flirds-1st의 차이(157 s 대 54 s, 2.9배)로 읽힌다.

---

## 6. 한계

가장 근본적인 전제는 서버 검증셋이 가치의 심판이라는 것이다.
게임의 utility가 검증손실이므로 모든 값의 의미를 검증셋의 구성이 결정한다. 검증셋이 모든
client에게 공정하다는 전제가 깨지면(특정 집단·task의 과소대표) 그들에게 유익한 client가
구조적으로 저평가되고, 검증셋이 커버하지 않는 분포 밖(out-of-distribution) 기여는 게임에
아예 보이지 않아 평가 자체가 불가능하다.

두번째로 Flirds가 주는 수는 데이터의 내재 가치가 아니라 실현된 한
궤적의 검증손실 감소에 대한 조건부 귀속이다. 어느 client가 어느 라운드에 뽑혔는지의
lottery는 보정하지 않고, Shapley 공리는 frozen 라운드 게임 안의 성질이며 궤적-특이
utility의 공리화는 열린 문제다. 폐형식의 합은 2차 surrogate 게임 안의
efficiency라 실현 개선과의 간극(Taylor 잔차)이 라운드마다 남는다.

마지막으로 폐형식과 실현-궤적 의미론은 개별 업데이트가
관측되고 가산 집계되는 동기식 FedAvg, momentum 없는 무상태 서버, stateless client에
의존한다. server momentum·Adam은 이 의미론을 깨고, Secure Aggregation처럼
합산만 노출되는 환경에서는 client별 내적을 계산할 수 없다.

---

## 7. 결론

Flirds는 관측된 FedAvg 라운드의 검증손실 게임을 2차 Taylor 근사해, 라운드당 HVP 1회와 client별 내적만으로 in-run Shapley를 계산한다. 이를 통해 coalition 재평가 없이도 평가 대상과 근사 오차가 명확한 client 기여도 평가가 가능하다.

실험에서 Flirds는 CNN·LLM 두 트랙의 전 셀에서 전수 열거한 in-run Shapley의 순위와 값을
재현했고, 1차 근사가 무너지는 gradient noise 레짐을 2차 곡률항이 지켰다. 같은 기여도의
sign-gating 개입은 전 오염 위협에 걸쳐 oracle-제외에 가장 근접한 회복을 보였으며, 비용은
라운드 참여자 수와 무관해 라운드 참여 10인 실측에서 전수 열거의 1/159에 그쳤다. 관측된
라운드에 대한 Shapley 귀속이 지수적 coalition 평가 없이도 정확성과 실효성을 잃지 않음을
보인 것이다.