# [한글 검토판] Measure First: Federated LLM 파인튜닝에서 클라이언트-수준 데이터 가치평가의 Exact Ground-Truth Fidelity

> **이 문서는** Flirds 논문의 **한글 작업본**입니다. 초록·서론·용어·문체 개정은 한글판에서
> 먼저 진행하고, 확정되면 영문 tex(`paper/`)에 반영합니다.

---

## 초록

데이터 기여도 평가(data valuation)는 학습된 모델의 성능 중 얼마가 어느 데이터 제공자의
몫인지를 정량화하는 문제이며, 공정한 분배의 공리를 만족하는 Shapley 값이 그 표준 도구다.
이 질문은 보상 분배·데이터 마켓·품질 관리가 전부 클라이언트별 기여도 수치를 요구하는
연합학습(FL)에서 가장 절실해진다. 하지만 기존 연합 Shapley 연구들은 FL이 부여하는 제약들로
인해 다음 두 가지 한계를 지닌다: 목표로 삼는 라운드별 Shapley 값의 정확한 계산은 참여자
수에 지수적이어서 실제로는 무작위
샘플링·절단 근사로 우회해 왔고, 그 과정에서 저마다 새로 정의한 대체 기여도가 원래의 Shapley
값과 얼마나 어긋나는지는 엄밀한 bound로도 exact 참값 대비로도 검증되지 않았다.

우리가 제안하는 클라이언트-수준 기여도 평가 방법 **Flirds**는 이 두 한계에 하나씩 답한다.
Flirds는 In-Run Data Shapley(Data Shapley in One Training Run)의 closed-form 계산을
FL의 라운드 구조로 확장해, 서버가 이미 수신하는 업데이트만으로 조합 재평가 없이 라운드별
Shapley 기여도를 계산한다.
목표인 라운드별 Shapley 값은 명시적으로 정의된 라운드 게임의 Shapley 값이어서, 공정한
분배의 공리를 그 게임 안에서 그대로 만족하며, 이 성질들은 라운드 단위 온라인 정산 요건과
결합되어 측정한 기여도를 그대로 보상 분배의 기준으로 쓸 수 있게 한다.
Flirds는 이 목표값의 근사이지만 유일한 오차인 Taylor 절단이 엄밀하게 bound되고, 같은
목표값을 전수 열거로 근사 없이 직접 계산한 참값(exact in-run Shapley)의 순위를 오염
클라이언트가 섞인 LLM 파인튜닝·CNN 연합학습 무대에서 높은 충실도로
재현한다(⬚<!-- 수치: 메인 무대(c2fid·R4-L2) 착지 후 기입 — 초록 fidelity 클레임은 주무대 기준으로 확정 -->).
나아가 removal·selection·탐지 같은 실험 환경에서 측정한 기여도가 실제로 가치 있는 값인지까지 검증한다.

---

## 1. 서론

**데이터 기여도 평가, 그리고 연합학습.** 현대 기계학습 모델의 품질은 데이터가 결정한다.
그래서 "이 모델의 성능 중 얼마가 누구의 데이터 덕인가"를 정량화하는 **데이터 기여도
평가(data valuation)** 가 독립된 연구 축으로 자리 잡았고, 협력 게임 이론의 **Shapley 값**이
그 표준 도구가 되었다. 공정한 분배를 특징짓는 공리(efficiency·symmetry·null-player·linearity)를
유일하게 만족하기 때문이다. 이 질문이 가장 절실한 무대가 **연합학습(FL)** 이다: 기여의 주체가
애초에 클라이언트 단위로 분리되어 있고, 서버는 그들의 데이터를 보지 못한 채 하나의
모델을 공동 학습하며, 보상 분배, 데이터 마켓, 무임승차 방지,
품질 관리와 디버깅 등 참여를 지속시키는 유인 구조가 전부 "누가 얼마나 기여했는가"라는 수치에 걸려 있다.

**실제 FL 세팅의 조건.** 배포된 FL에서 기여도 평가는 다음 조건들 아래에서 동작해야 한다:

- **정보 조건**: ① 서버는 클라이언트의 원데이터를 개별적으로 볼 수 없다: 서버가 보는 것은
  각 클라이언트의 모델 업데이트가 전부다. ② 공정한 평가의 기준이 되는 검증 데이터는 서버만
  보유한다: 클라이언트의 자기 신고는 기준이 될 수 없다.
- **운영 조건**: ③ 기여도는 매 라운드 온라인으로 계산되어야 한다: 클라이언트가 간헐적으로
  참여하고(라운드마다 cohort가 다름) 학습이 계속 이어지는(continual) 상황에서 사후 일괄
  평가는 정산·운영에 쓸 수 없다. ④ 부분집합 재학습은 감당 불가능하고, 평가를 위해 추가 통신
  라운드나 클라이언트-측 추가 연산을 부과할 수도 없다. ⑤ 평가 단위는 클라이언트다: 보상과 책임의 주체가 클라이언트이기
  때문이다.

**측정에서 정산으로.** 다만 위 조건들은 기여도를 *계산할 수 있기* 위한 조건이다. 측정한
기여도를 보상 분배에 실제로 소비하려면 값 자체에 대한 요구가 더 붙는다: 지급의 근거가 되는
값이 명시적으로 정의되어야 하고(무엇에 대한 대가인가), 같은 학습 로그에서 같은 값이
재계산되어야 하며(감사 가능성), 기여도의 총합이 실현된 개선과 정합해야 하고(efficiency),
아무것도 제출하지 않은 참여자는 정확히 0을 받아야 한다(null player). 이 넷은 Shapley 공리가
약속하는 바로 그 성질들이지만, 근사가 끼어드는 순간 가장 먼저 희생되는 성질들이기도 하다(§2).

**왜 retrain 기반 Shapley 값이 아닌가.** 데이터 가치평가의 고전적 참값은 retrain 기반
Shapley 값, 곧 클라이언트 부분집합별로 학습을 처음부터 다시 수행해 얻는 utility에 대한
Shapley 값이다(§4.1). 그러나 FL에서는 매 라운드 서로 다른 클라이언트가 간헐적으로 참여하고,
그때마다 공정한 기여도 평가가 요구된다. 이런 환경에서 부분집합별 재학습을 전제하는 위
정의는 실질적으로 계산 불가능하다. 그래서 기존
연합 Shapley 연구들은 대부분 평가 대상을 라운드 단위로 옮겨 왔다: 매 라운드 서버가 수신한
업데이트들로 정의되는 라운드별 Shapley 값을 계산하고, 이를 라운드에 걸쳐 누적한다. 본
논문도 평가 대상을 같은 라운드 단위에 두며, 그 목표값이 §4에서 정의할 **exact in-run
Shapley**다.

**남은 두 한계.** 이 라운드 단위
계보(FedSV, GTG-Shapley, ComFedSV, ShapleyFL 등)는 대부분 위 조건들 안에서 동작하도록
설계되어 있다. 그러나 두 가지 한계가 남는다.
첫째는 **계산**이다: 목표로 삼는 라운드별 Shapley 값을
정확히 계산하려면 라운드마다 클라이언트 조합(coalition)별로 모델을 재구성하고 검증 데이터로
재평가해야 하므로 비용이 라운드 참여자 수에 지수적으로 자라고, 실제로는 무작위 샘플링과
절단(truncation) 근사로 이를 우회해 왔으며, 이 계보 전체가 소형 CNN 규모의 검증에 머물러
왔다. 둘째는 **그 우회가 낳은, 검증되지 않은 대체 정의**다: 비용을 피하는 과정에서 방법마다
서로 다른 대체 utility를 겨냥하게 되었고,
일부는 Shapley 공리를 명시적으로 포기하며, 그 대체값이 원래의 Shapley 값과 얼마나
어긋나는지는 엄밀한 bound로도 exact 참값 대비로도 검증되지 않았다. 측정한 기여도의 타당성은
대부분 탐지·수렴·최종 정확도 같은 다운스트림 증거로만 간접 확인되었고, 추정값 자체를 exact
참값 대비로 채점한 경우는 소규모 synthetic이나 근사 참조에 그친다. 근사 FL-Shapley 보상이
서버 집계 전략만 바꿔도 수십 %씩 출렁인다는 보고는 이 상태의 실무적 얼굴이다. 요컨대 이
계보에서 Shapley 값을 겨냥한 방법들은 그 대가로 표본 분산과 조합 비용을 치렀고, 비용과
결정론을 확보한 방법들은 그 대가로 공리를 내려놓았다 — 두 축을 동시에 만족한 방법은
없었다. 우리의 접근이 겨냥하는 것이 정확히 이 두 한계다. In-Run Data Shapley가 중앙집중 학습에
대해 제안한 closed-form Shapley 계산을 연합학습의
라운드 구조로 확장하면 두 문제가 한꺼번에 풀린다. 조합 재평가가 통째로 사라져 지수적 계산을
우회할 샘플링·절단이 애초에 필요 없어지고(계산), 목표와 근사가 모두 명시적이라 오차를
엄밀하게 특성화할 수 있으며 참값을 직접 계산해 추정기를 채점할 수 있다(정의와
검증).

**제안 방법.** 우리는 In-Run Data Shapley를 연합학습으로 확장한 클라이언트-수준 기여도 평가
방법 **Flirds**(Federated Learning In-Run Data Shapley)를 제안한다. 출발점은 FedAvg 계열의
학습이 기여도 평가에 필요한 재료를 이미 서버에 전달하고 있다는 관찰이다: 매 라운드 서버는
참여 클라이언트들의 모델 업데이트를 받아 가중 평균으로 집계한다. Flirds는 이 라운드별 집계를 "어떤
클라이언트 부분집합의 업데이트만 반영했다면 검증 손실이 얼마나 줄었을 것인가"라는 하나의
협력 게임으로 보고, 이 게임의 exact Shapley 값을 검증 손실의 1차+2차 Taylor 전개로 근사한
closed-form으로 계산한다. 즉 Flirds 역시 참값 자체가 아니라 그 근사를 계산하는 추정기다. 다만
근사의 원천이 Taylor 절단 하나뿐이고 그 오차가 표본 추출 없이 결정론적으로 bound된다는
점에서, 무작위 샘플링의 분산과 검증되지 않은 대체 정의가 겹겹이 끼는 선행 근사들과
다르다. 그 결과 라운드당 필요한 무거운 계산은 검증셋 위의 Hessian-vector product 한
번으로 고정되고, 참여자당 추가 비용은 내적 하나가 전부다: 재학습도, 부분집합별 모델
재구성·재평가도, 추가 통신이나 클라이언트-측 연산도 필요 없다. 내적은 업데이트가 놓인
파라미터 공간의 차원에만 비례하므로, 전체 모델을 학습하는 소형 네트워크부터 PEFT(LoRA)
어댑터만 교환하는 LLM 파인튜닝까지 같은 식이 그대로 적용된다.

**기여.** 이 논문의 기여는 다음과 같다.

1. **방법.** In-Run Data Shapley의 closed-form 계산을 연합학습의 라운드 구조로 확장한
   클라이언트-수준 기여도 추정기 Flirds를 제안한다. 연합 클라이언트-수준 기여도 평가를 LLM
   규모(PEFT 파인튜닝)에서 수행한 것은 우리가 아는 한 이 논문이 처음이다.
2. **검증 프로토콜.** 추정 정확도는 전수 열거로 계산한 exact in-run Shapley 대비로, 기여도의
   실효성은 같은 개입 정책 아래에서의 실제 학습 개선으로 채점한다. exact 참값 대비 채점을
   LLM 규모까지 끌어올린 것은 우리가 아는 한 처음이다.
3. **수학적 정당성.** Flirds가 계산하는 값이 라운드별 집계 게임의 2차 Taylor 근사가 갖는
   정확한 Shapley 값임을 증명하고, 유일한 오차인 Taylor 절단을 bound한다. 오차원에
   무작위성이 없다는 점이 선행 근사와 갈리는 지점이다: 값이 동결된 학습 로그의 결정론적
   함수이므로 같은 로그는 언제 다시 계산해도 같은 값을 주고, 절단 오차는 표본 분산이 아니라
   결정론적 bound로 특성화된다. 공정성 공리는 라운드 게임 수준에서 성립해 무기여
   클라이언트가 대수적으로 정확히 0을 받는다. 예산 정합(efficiency), 무기여자 exact-0,
   로그-결정론 이 셋이 라운드 단위의 온라인 정산과 결합되어, 측정한 기여도를 실현된 런에
   대한 보상 분배의 기준으로 그대로 소비할 수 있게 한다.

---

## 2. 관련 연구

**연합학습에서의 Shapley 기반 기여도 평가.** 연합 Shapley 값의 계보는 클라이언트 기여도를
라운드 단위의 Shapley 값으로 분해하고 permutation Monte Carlo 표본 추출로 추정하는
FedSV에서 시작한다. 이후의 흐름은 지수적 비용을 낮추는 근사의 연쇄다. GTG-Shapley는
부분집합 모델 재구성과 유도-절단 Monte Carlo를, ComFedSV는 부분 참여로 관측되지 않는
조합을 메우는 utility 행렬의 low-rank 완성을, ShapleyFL은 정규화와 이동평균으로 가공한
라운드별 대체(surrogate) 값을 쓰고, 이후의 FedIF·FedTSV·ShapFed·S-FedAvg 계열은 Shapley
공리를 완화하거나 포기한 채 기여도 신호를 강건-집계 가중치로 소비하는 쪽으로 이동한다. **이
계보가 무엇을 내주었는지는 두 갈래로 갈린다.** Shapley 값을 겨냥한
쪽(FedSV·GTG-Shapley·ComFedSV)은 지수 비용을 표본 추출·절단·보간으로 갚으면서 결정론과 오차
규명을 내주었고, 비용과 결정론을 확보한 쪽(ShapleyFL·FedIF 계열)은 공리를 완화·포기하면서
값의 정의를 내주었다 — 같은 이름 아래 재정규화 게임·보간된 손실 행렬·가공된 surrogate라는
서로 다른 값이 추정되고 있다. 어느 쪽이든 그 값의 타당성은 대부분 다운스트림 결과로만 간접
확인되었고, exact 참값 대비 직접 채점은 SPACE의 $2^n$-재학습 비교($N \le 10$의 CNN
분류)가 유일한 선례다.

**중앙집중 LLM-규모 attribution.** 중앙집중 학습에서는 개별 학습 예제 단위의 데이터
귀속·선별이 LLM 규모까지 활발히 발전해 왔으며, 크게 세 줄기다. 첫째, **influence function
계열**은 각 예제가 검증 손실에 미치는 영향을 gradient와 Hessian 역행렬($H^{-1}$)로 추정하며,
LLM 규모에서는 $H^{-1}$ 근사를 서로 다르게 처리한다(EK-FAC[Grosse et al., 52B], LoRA용 closed-form
근사 DataInf, TRAK, LoGra 등). 둘째, 2024년의 **Hessian-free 흐름**은 $H^{-1}$을 아예
우회한다: LESS는 TracIn 계열의 궤적 influence를 LoRA gradient 사영으로 계산해 instruction
tuning 예제를 고르고, MATES·DsDm은 각각 증류한 소형 모델과 선형 datamodel로 사전학습
데이터를 고른다.
셋째, **In-Run Data Shapley**(§3.3)는 사후 $H^{-1}$ 대신 실제 학습 궤적을 따라 매 스텝의
Taylor 기여를 누적한다. 이들은 모두 학습 데이터에 직접 접근할 수 있는 중앙집중 세팅을
전제하고 평가 단위도 개별 샘플이어서, §1의 정보·운영 조건 아래에서는 그대로 쓰일 수 없다.
세 줄기 가운데 연합학습의 무대로 가장 자연스럽게
이어지는 것은 In-Run 계열인데, FedAvg 집계 $\sum_k p_k \Delta w_k$가 배치 gradient의
샘플-선형 분해와 같은 구조를 클라이언트 수준에서 이미 드러내기 때문이다. 본 논문은 이
관찰에서 출발해 IRDS의 closed-form 계산을 연합 라운드 게임으로 확장한다(§4).

---

## 3. 배경: 연합학습, Data Shapley, In-Run Data Shapley

### 3.1 연합학습과 FedAvg

FedAvg(McMahan et al., 2017)에서는 크기 $n_k$의 로컬 데이터를 가진 클라이언트
$k \in [N] := \{1, \dots, N\}$ 중 라운드 $r$의 참여 집합 $P_r \subseteq [N]$이 현재 모델
$w^r$에서 로컬 학습한 차를 업데이트 $\Delta w_k^r$로 보내고, 서버는 $w^0$부터 $R$ 라운드
동안($r = 0, \dots, R{-}1$) 데이터 크기에 비례하는 **고정 참여자 가중치**로 집계한다:

$$w^{r+1} \;=\; w^r + \sum_{k \in P_r} p_k^r\, \Delta w_k^r, \qquad
p_k^r \;=\; \frac{n_k}{\sum_{j \in P_r} n_j}. \tag{1}$$

### 3.2 Data Shapley

부분집합(coalition) $S$에 가치를 주는 utility 함수 $U$의 협력 게임에서 **Shapley 값**은
플레이어 $k$의 한계 기여를 모든 참여 순서에 대해 평균한

$$\phi_k(U) \;=\; \sum_{S \subseteq [N] \setminus \{k\}}
\frac{|S|!\,(N{-}|S|{-}1)!}{N!}\, \big( U(S \cup \{k\}) - U(S) \big) \tag{2}$$

이며(임의의 유한 플레이어 집합에 같은 식), 공정한 분배의 네 공리 — efficiency
($\sum_k \phi_k(U) = U([N]) - U(\emptyset)$), symmetry, null player, linearity — 를
유일하게 만족한다(Shapley 1953).
Data Shapley(Ghorbani & Zou 2019)는 학습에 사용되는 데이터를 플레이어로 보고 $U(S)$를 "$S$의
데이터만으로 학습한 모델의 검증 성능"으로 두어 이를 데이터 기여도에 적용한다.

### 3.3 In-Run Data Shapley

**In-Run Data Shapley(IRDS)**(Wang et al., 2024)는 가상의 재학습들 대신 실제로 일어난 학습
런 하나를 고정하고, 매 gradient step을 작은 협력 게임으로 본다. step $t$의 SGD 업데이트가
$w^{t+1} = w^t - \eta_t \sum_{z \in B_t} \nabla \ell_z(w^t)$일 때($B_t$는 배치, $\ell_z$는
샘플 $z$의 학습 손실, $\ell_{\mathrm{val}}$은 검증셋 위 평균 손실), $S \subseteq B_t$의
utility는 "$S$의 gradient만 반영했을 때의 검증
손실 감소"

$$u_t(S) \;=\; \ell_{\mathrm{val}}(w^t) \;-\; \ell_{\mathrm{val}}\Big(w^t - \eta_t \textstyle\sum_{z \in S} \nabla \ell_z(w^t)\Big) \tag{3}$$

이고, $\phi_z = \sum_{t:\, z \in B_t} \phi_z(u_t)$로 합산한다(표기: $\phi_z(u_t)$는 게임
$u_t$의 Shapley 값(식 (2)), 인자 없는 $\phi_z$는 그 누적이다). 핵심 관찰: $u_t$를
$w^t$에서 Taylor 전개하면 근사 게임의 Shapley 값이 closed-form으로 나온다. 검증 손실의
gradient·Hessian을 $g_t := \nabla \ell_{\mathrm{val}}(w^t)$, $H_t := \nabla^2 \ell_{\mathrm{val}}(w^t)$로
두면, 1차 게임은 가법적이라 Shapley 값이 gradient 내적 그 자체이고, 2차까지 취하면

$$\phi_z(u_t) \;\approx\; \eta_t \big\langle g_t,\, \nabla \ell_z(w^t) \big\rangle
\;-\; \tfrac{\eta_t^2}{2} \Big\langle \nabla \ell_z(w^t),\; H_t\, \textstyle\sum_{z' \in B_t} \nabla \ell_{z'}(w^t) \Big\rangle \tag{4}$$

이다(부호는 식 (3)의 손실-감소 규약을 따른다). Shapley 공리는 step 게임 수준에서 계승되고, efficiency가
telescoping과 결합해 $\sum_z \phi_z = \sum_{t=0}^{T-1} u_t(B_t) = \ell_{\mathrm{val}}(w^0) -
\ell_{\mathrm{val}}(w^T)$($T$는 총 스텝 수)가 성립하며, gradient가 0인 샘플은 null player로 정확히 0을
받는다(형식적 서술·증명은 원 논문 참조). retrain 정의와는 묻는 질문이
다르고(counterfactual 대 realized 귀속) 값이 궤적-특이적(trajectory-specific)이다.

---

## 4. Flirds: 연합 in-run 게임과 그 closed-form 추정

### 4.1 연합 게임과 두 참값

데이터 가치평가의 고전적 참값 **exact retrain Shapley**는 클라이언트 부분집합 $S$만
참여시켜 $R$ 라운드를 처음부터 재학습한 최종 모델 $w_S^R$($w_\emptyset^R = w^0$)의 검증 손실
감소

$$U^{\mathrm{re}}(S) \;=\; \ell_{\mathrm{val}}(w^0) - \ell_{\mathrm{val}}\big(w_S^R\big) \tag{5}$$

에 대한 Shapley 값 $\phi_k^{\mathrm{re}} = \phi_k(U^{\mathrm{re}})$를 $2^N$개 부분집합
전부의 재학습으로 근사 없이 계산한 것이다(방법-중립적 참값; 묻는 질문은 counterfactual "$S$만
참여했다면"). 한편 FL의 라운드 구조는 IRDS의 step 구조와 정확히 대응한다: 식 (3)의 step 게임에서
샘플의 gradient 항 $-\eta_t \nabla \ell_z$ 자리에 라운드의 per-client 가중
업데이트 $p_k^r \Delta w_k^r$를 넣으면 게임이 클라이언트 단위로 이식되고, 재료는 서버가 표준
프로토콜에서 어차피 수신하는 것뿐이다.
실현된 궤적 $\{w^r, \{\Delta w_k^r\}_{k \in P_r}\}_{r=0}^{R-1}$을 고정하고(**고정 궤적**;
이후 모든 방법이 이 동일한 학습 로그를 공유한다), 라운드별 coalition utility를

$$u_r(S) \;=\; \ell_{\mathrm{val}}(w^r) \;-\; \ell_{\mathrm{val}}\Big(w^r + \textstyle\sum_{k \in S} p_k^r\, \Delta w_k^r\Big),
\qquad S \subseteq P_r \tag{6}$$

로 정의한다. 가중치는 런 자신이 사용한 **고정 가중치** $p_k^r$ 그대로이며 $S$ 안에서
재정규화하지 않는다. 일부 baseline(GTG-Shapley, FedSV)처럼 $S$ 안에서 재정규화하면 이는
구현 차이가 아니라 **다른 협력 게임**이다: zero-update free-rider가 0 아닌 값을 받게 되고,
비등크기·부분 참여·2차 곡률에서 순위가 뒤집히는 반례가 있으며(부록 A), 라운드를 HVP 한
번으로 접는 합-형태 구조가 깨져 closed-form 계산이 성립하지 않는다. **exact in-run Shapley**
$\phi^{\mathrm{in}}$은 각 라운드 게임의 Shapley
값을 라운드당 $2^{|P_r|}$개 부분집합 전수 열거로 근사 없이 계산해 합한 값이다:
$\phi_k^{\mathrm{in}} = \sum_{r: k \in P_r} \phi_k(u_r)$. IRDS의 efficiency가
telescoping과 결합해 그대로 성립한다:
$\sum_k \phi_k^{\mathrm{in}} = \sum_{r=0}^{R-1} u_r(P_r) = \ell_{\mathrm{val}}(w^0) - \ell_{\mathrm{val}}(w^R)$.

**용어 규약.** 이하 참값(ground truth, GT)은 위의 **exact retrain Shapley**(부분집합별 전수
재학습)와 방금 정의한 **exact in-run Shapley**(라운드별 전수 열거) 둘뿐이며, 각각
**retrain GT**·**in-run GT**로 줄여 쓴다. 어느 게임의 참값인지가 항상 수식어로 붙으므로
수식어 없는 "GT"나 "exact"는 쓰지 않는다. "oracle"은 오염 클라이언트 집합을 아는 개입
arm(§5.3의 oracle-제외)에만 쓰고 참값에는 쓰지 않으며, 탐지·개입의 정답 라벨로 쓰는 실제
오염 집합은 "true corrupt set"으로 따로 부른다. 두 참값 모두 값이 클수록 유익하다(검증
손실을 낮추면 양수).

### 4.2 closed-form 추정기

exact in-run Shapley는 근사가 없지만 라운드당 $2^{|P_r|}$번의 검증 평가를 요구한다. Flirds의
출발점은, 라운드 utility를 $w^r$ 주변에서 2차까지 Taylor 전개한 근사 게임 $\hat u_r$의
Shapley 값이 closed-form으로 나온다는 것이다(명제 2). $g_r := \nabla \ell_{\mathrm{val}}(w^r)$,
$H_r := \nabla^2 \ell_{\mathrm{val}}(w^r)$(Gauss–Newton 근사가 아닌 true Hessian),
$\delta_k^r := \Delta w_k^r$, coalition $S$의 집계 이동
$\Delta_S^r := \sum_{k \in S} p_k^r\, \delta_k^r$, 그리고 $\Delta W_r := \Delta_{P_r}$라 하면:

$$\hat\phi_k^{(r)}
\;=\;
-\,p_k^r\, \big\langle g_r,\, \delta_k^r \big\rangle
\;-\; \tfrac{1}{2}\, p_k^r\, \big\langle \delta_k^r,\; H_r\, \Delta W_r \big\rangle,
\qquad
\hat\phi_k = \sum_{r\,:\,k\in P_r} \hat\phi_k^{(r)}. \tag{7}$$

첫 항은 $k$의 업데이트가 검증 손실의 하강 방향과 정렬된 정도를, 둘째 항인
**클라이언트-상호작용 항**은 $k$의 업데이트가 다른 참여자들의 집계 업데이트와 곡률을 통해
합성되는 정도를 잰다(식 (4)와 첫 항의 부호가 다른 것은 그곳의 변위
$-\eta_t \nabla \ell_z$에는 음부호가 명시돼 있고 여기서는 $\delta_k^r$ 자체가
변위이기 때문이다). 라운드당 필요한 것은 forward-mode 자동미분으로 계산하는
Hessian-vector product(HVP) $H_r \Delta W_r$ **한 번**과 참여자당 내적 하나가 전부다.
Hessian을 만들거나 역행렬하지 않으며, 비용은 부분집합 수 $2^{|P_r|}$과 무관하다. 2차 항을
제거한 변형을 **Flirds-1st** 라 부르고 상호작용 항의 ablation으로 사용한다.

### 4.3 이론: 추정기와 exact in-run Shapley는 같은 게임을 계산한다

이 절의 요지는 한 문장이다: Flirds는 임의의 점수가 아니라 식 (6)이 정의하는 바로 그 게임의
Shapley 값을 계산하며, exact in-run Shapley와의 유일한 차이는 Taylor 절단뿐이다. 형식적
서술·증명·가정·반례·수치 검증은 부록 A에 있다.

**명제 1 (라운드별 분해).** 고정 궤적 위에서 전체 게임 $\sum_r u_r$(각 $u_r$은
$u_r(S) := u_r(S \cap P_r)$로 $[N]$ 위 게임으로 확장해 합산)의 Shapley 값은 라운드별
$|P_r|$-인 게임의 Shapley 값의 합으로 정확히 분해된다(비참여 라운드의 기여 0은 정의가
아니라 정리다).

**명제 2 (closed-form; free-rider 0; efficiency).** 라운드 utility의 2차 Taylor 근사 게임 $\hat u_r$에
대해 exact Shapley 값은 식 (7)과 정확히 같다: $\hat\phi_k^{(r)} = \phi_k(\hat u_r)$(증명은 부록
A). 따름 성질 둘: 전 참여 라운드에서 $\delta_k^r = 0$인 zero-update free-rider는 대수적으로
**정확히** $\hat\phi_k = 0$을 받고, 값의 합은 근사 게임의 전체 utility와 일치한다(efficiency).

**명제 3 (Taylor 잔차).** 근사의 대가는 절단 오차다. 라운드당 utility 오차는 2차에서
$\frac{M_3^r}{6}\|\Delta_S^r\|^3$, 1차에서 $\frac{M_2^r}{2}\|\Delta_S^r\|^2$로 bound되고 라운드 수에
선형 누적된다(라운드별 국소 곡률 상수 $M_2^r, M_3^r$의 형식적 정의는 부록 A). 이 상수들은
추정 가능한 양이 아니므로, 이 bound가 주는 것은
잔차의 **차수**(스케일링 법칙)까지다.

**비고 (적용 범위).** exact in-run Shapley는 실현된 궤적의 함수이므로, 궤적-특이적 값의
공리화 미해결이라는 IRDS의 한계를 물려받는다(exact retrain Shapley와의 관계는 이론이 아니라
실증 질문이고 §5.2에서 그렇게 보고한다). 게임은 FL 프로토콜이 서버에 노출하는 단위인
**클라이언트** granularity에서 정의되며, per-sample에서 per-client로의 형식적 연결은
mean-loss·단일 로컬 스텝에서 성립하고(부록 A) token-mean LLM 손실에는 그대로 이전되지
않는다. LLM 트랙에서 $w$·$\delta_k^r$·$g_r$·$H_r$은 전부 서버가 실제로 교환·평균하는
LoRA 인자 좌표의 양이다(CNN 트랙은 전체 파라미터를 학습하므로 구분이 없다; 좌표 의존성의
형식적 논의는 부록 A.10).

---

## 5. 실험

> **작업본 표기 규약.** ⬚ = 실행 중·대기 실험의 자리(채울 분석 파일은 각 위치의 HTML 주석에
> 명시; 수치는 rundir/분석 스크립트 재생성 값만 기입). † = CNN Dirichlet(α=1) 기존 실측 — 캠페인
> restack 드리프트 대조(W-A) 확인 후 확정. ◐ = 1-seed. 별도 표기 없으면 3-seed mean±std.

실험은 §1의 검증 두 겹을 핵심 질문의 위계 순으로 배치한다. **1차 — fidelity**: 추정값이
exact 참값의 순위·값을 재현하는가(§5.2). **2차 — 실효성**: 측정한 기여도로 학습을 실제로
개선할 수 있는가(§5.3). 이어서 방법별 계산 비용(§5.4)과 구성요소별 ablation(§5.5)을 본다.
프로토콜 상세와 위협 구현은 부록 B, fidelity 확장(비교 방법 전종 × 두 참값, 추가 지표, 보조
무대)은 부록 C, 재현성(안정성)은 부록 D, 개입 확장(주무대 경쟁의 MNIST 짝·크기-가중
변형·무해성·φ 부호 감사)은 부록 E, 비용 상세는 부록 F, 오염 클라이언트 탐지는 부록 G에 있다.

### 5.1 실험 세팅

**무대.** 주무대는 LLM·CNN 한 쌍이며, 두 트랙 모두 "오염 클라이언트가 섞인 부분 참여
FL"이라는 같은 구도를 공유한다. retrain GT가 필요한 비교는 $2^N$ 재학습이
가능한 작은-$N$ 보조 무대가 담당한다(§5.2).

| 무대 | 트랙 / 데이터 | $N$ · 참여 | $R$ | 위협 축 | GT | 쓰임 |
|---|---|---|---|---|---|---|
| **주** `LLM-Main` | LLM 1B(LoRA) · GSM8K | 50 · 5/50 | 200 | clean / answer-swap@0.7 / free-rider(zero) — 오염 40% | in-run GT (per-round $2^5$) | §5.2–5.4 · 부록 G |
| **주** `CNN-Main` | CNN · CIFAR-10 × {Dirichlet(α=1), iid} | 100 · 10/100 | 120 | clean / free-rider(zero) / gradient noise / label-flip@0.70 — 오염 40% | in-run GT (per-round $2^{10}$) | §5.2·§5.3 · 부록 G |
| 보조 `LLM-Scale` | LLM 1B·3B·7B · alpaca IID clean | 20 · 2/20 | 200 | – (clean-IID 전용) | in-run GT (per-round $2^2$) | §5.2 |
| 보조 `Silo` | LLM 1B · 5-도메인 비IID | 5 · 전원 | 10 | clean / answer-swap / free-rider(zero) | **retrain GT ($2^5$)** + in-run GT ($2^5$) | §5.2·§5.5 · 부록 C·D |
| 보조 `Anchor` | LLM 1B · alpaca IID clean | 5 · 전원 | 30 | – | retrain GT ($2^5$) + in-run GT ($2^5$) | 부록 C·F |
| 보조 `CNN-Small` | CNN · CIFAR-10 × {Dirichlet(α=1), iid} | 10 · 전원 | 10 | `CNN-Main`과 동일 | **retrain GT ($2^{10}$)** + in-run GT ($2^{10}$) | §5.2 · 부록 C·D |
| 보조 `LLM-Device` | LLM 1B · alpaca · Dirichlet($\alpha$) | 100 · 10/100 | 30 | clean / answer-swap / free-rider(zero) | in-run GT (per-round $2^{10}$) | §5.4 · 부록 C·G |

**오염축은 트랙당 고정한다**: CNN = {label-flip@0.70, free-rider(zero), gradient noise},
LLM = {answer-swap@0.7, free-rider(zero)}. clean은 위협이 아니라 전 무대 공통의 대조
앵커다(무해성 parity·오발화 판정의 기준). `CNN-Main`·`CNN-Small`의 MNIST 짝(동일 세팅,
데이터셋만 교체)은 데이터셋 강건성 확인 전용으로 부록 C·G에 둔다.

성능 심판은 LLM-Main = GSM8K 공식 test의 잔여 1,119문항 exact-match(EM; greedy 디코딩),
CNN = held-out test 정확도, `LLM-Scale`의 개입 무해성 심판은 MMLU·Alpaca-test ROUGE-L(부록
E)이다. 학습은 두 트랙 모두 momentum 없는 plain SGD·상수
학습률·stateless 클라이언트(부록 A.9의 가정 그대로)이고, LLM 트랙은 LoRA(r=16, α=32) 인자만
교환한다(부록 A.10). 하이퍼파라미터 전량·데이터 분배 규칙은 부록 B.

**비교 방법: 겨냥하는 게임으로 나눈다.** 기여도 방법 8종을 두 계열로 구분한다.

- **같은-게임 계열(3)** — 식 (6)의 고정-가중 라운드 게임을 겨냥한다: **Flirds**(2차 closed-form),
  **Flirds-1st**(1차 항만), **individual utility**(라운드 게임의 singleton utility
  $u_r(\{k\})$를 forward 평가로 직접 계산해 합산하는 가산 근사 — Shapley 값 대신 singleton
  값). 이들과 in-run GT의 차이는 순수한 근사 오차다.
- **cross-game 계열(5)** — GTG-Shapley, FedSV, ComFedSV, ShapleyFL, FedIF: §2의 계보
  그대로 재정규화 게임·보간 행렬·가공 surrogate·influence 등 저마다 다른 값을 겨냥한다.
  in-run GT와의 불일치에는 근사 오차와 "다른 게임" 성분이 섞여 있다.

**본문 fidelity 표는 Flirds와 Flirds-1st만 싣는다.** §5.2가 답하는 것은 추정기 층의 질문 —
제안 방법의 유일한 근사인 Taylor 절단이 실제로 얼마나 큰가 — 이고, 같은 게임을 겨냥한 채
절단 차수만 다른 두 변형의 대조가 그 질문의 최소 충분 단위이기 때문이다. 나머지 6종과의
비교는 두 자리에서 한다: 측정한 기여도의 다운스트림 우열(§5.3의 CNN 8 점수원 경쟁)과
fidelity 전표(부록 C — 전 방법 × 두 참값, Kendall·거리 포함).

제외: Banzhaf(다른 semivalue 축), Fed-LOO(Shapley가 아닌 leave-one-out 축),
Ripple(방법이 자체 재학습 궤적을 요구해 아래 고정-궤적
채점과 비호환). clean-preserving poisoning 위협은 본 논문의 스코프 밖이다(§6). baseline
재구현·파라미터 주석(ShapleyFL EMA β, ComFedSV per-round 대용 등)은 부록 B.5.

**지표.** fidelity = Spearman $\rho$(순위)·Pearson $r$(값)을 본문에, Kendall
$\tau$·거리 3종(cosine/euclidean/max)을 부록 C에 둔다. 탐지 = AUROC(오염 클라이언트를
양성으로 두고 기여도 순위 하위 = 의심 규약; 부록 B·G). 개입 = **절대 성능**(EM/acc)을
vanilla(개입 없음 = 바닥)와 oracle-제외(오염 클라 정확 제외 = 천장) 사이에서 직접 읽는다 —
정규화 점수가 셀 간 기준선 차이를 가리는 것을 피한다. 비용 = 같은 궤적 위 valuation 단독
wall-clock과 하드웨어-독립 연산수 모델(§5.4).

**고정-궤적 채점.** 셀(무대 × 위협 × seed)마다 학습 궤적 하나를 실측해 동결하고, **모든
방법과 in-run GT을 같은 로그 위에서 채점한다**(§4.1의 프로토콜). 방법 간 차이는 전부 방법
자신의 몫이며 궤적 분산이 아니다. retrain GT만 정의상 부분집합별 재학습을 따로
수행한다. 개입 실험(§5.3)은 관찰자 런 — 개입 없는 vanilla와 비트동일 궤적에 점수원들을 동시
부착 — 으로 점수를 얻은 뒤, 정책·시점이 같고 점수원만 다른 arm들을 같은 seed에서 비교한다.

**공정성 원칙.** ① 각 방법은 자기 정의 그대로 채점된다. 예컨대 재정규화 계열이 zero-update
free-rider에 0이 아닌 값을 주는 것은 구현 오류가 아니라 그 방법이 겨냥한 게임의
귀결이며(§4.1), 그대로 표에 남긴다(그 0-의미론의 실효성 귀결이 §5.3의 판정 재료다). ② 개입
정책은 sign-gating(누적 기여도 $\le 0$ 배제, 문턱 $\tau{=}0$의 parameter-free) 하나로 고정하고
무대·셀별 튜닝을 하지 않는다. ③ 예측이 어긋난 칸·오발화도 그대로 보고한다.

### 5.2 Fidelity — exact in-run Shapley 재현 (1차)

**주무대: Flirds vs Flirds-1st, in-run GT 대비.** 표 [F1]은 CNN 주무대 24셀(파티션 2 × 위협
4(clean 포함) × 3-seed)을 위협별로 풀링한 Spearman, 표 [F2]는 LLM 주무대의 위협별
Spearman·Pearson, 표 [F3]은 스케일 축(`LLM-Scale`)이다.

표 [F1] — CNN 주무대(`CNN-Main`): Flirds vs Flirds-1st, in-run GT 대비 Spearman ↑ (위협별; 파티션 × seed 풀 = 위협당 6셀, mean±std) ●
<!-- 출처: runs/track_c/c2fid/analysis/fidelity.csv 재풀링(dataset=cifar10, partition∈{dir1,iid},
     scenario∈{clean, free_rider, grad_noise, label_flip@'0.70'}, 값 = spearman_b = vs (b)oracle).
     24셀(2 파티션 × 4 위협 × 3-seed) 전량 실측 ●. qskew·shard·fmnist 파티션과
     frrand·lf@{.15,.35}·strmain 열은 위협축 스코프 밖 — rundir 존속, 표 미수록. -->

| method | clean | free-rider(zero) | gradient noise | label-flip@0.70 |
|---|---|---|---|---|
| **Flirds** | .991±.008 | .990±.009 | **.858±.031** | .998±.001 |
| Flirds-1st | .640±.251 | .702±.212 | .266±.082 | .979±.011 |

표 [F2] — LLM 주무대(`LLM-Main`): Flirds vs Flirds-1st, in-run GT 대비 (per-round $2^5$) ⬚
<!-- 채움: G1(R4-L2) — REGIME=gsm50k5 × {answer-swap, free-rider(zero), clean} × seed{0,1,2} = 9셀.
     runs/phase2_matrix/rundirs/1B_gsm50k5_*/metrics.json -->

| method | answer-swap Sp | answer-swap Pe | free-rider(zero) Sp | free-rider(zero) Pe | clean Sp | clean Pe |
|---|---|---|---|---|---|---|
| Flirds | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| Flirds-1st | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |

표 [F3] — `LLM-Scale`(1B·3B·7B; clean-IID 전용 무대): Flirds vs Flirds-1st, in-run GT 대비 Spearman ↑ (3-seed) ●
<!-- 출처: runs/track_d/fidelity.csv (cell=*_std20_seed{0,1,2}, 진리값 TRUTH="(b)oracle" —
     make_fidelity.py가 각 rundir의 phi.parquet에서 재계산; 18셀 중 std20 9셀). anchor5 9셀은 부록 C.6. -->

| scale | Flirds | Flirds-1st |
|---|---|---|
| 1B | 1.000±.000 | .999±.001 |
| 3B | 1.000±.000 | .997±.003 |
| 7B | .999±.001 | .998±.002 |

(`LLM-Scale`은 라운드당 참여가 2라 in-run GT 직접 열거($2^2$)가 오히려 싼 무대이고 clean-IID라
클라이언트 간 실재 신호도 없다 — 이 표의 질문은 우위가 아니라 **모델 규모가 1B→7B로 커져도
Taylor 절단 fidelity가 유지되는가** 하나다.)

(free-rider(zero) 칸의 일치는 오염 클라이언트 쪽에서는 부분적으로 자명하다 — in-run GT와
같은-게임 계열 모두 해당 클라이언트에 대수적으로 정확히 0을 주므로(명제 2), 이 칸의 변별
정보는 잔여 clean 클라이언트 순위의 재현에 있다. exact-0 성질 자체는 개입(§5.3)과
탐지(부록 G)에서 실효성으로 검증된다.)

읽기(CNN·스케일 실측 기준). ① **2차항이 fidelity를 가르는 칸은 gradient noise다**: 표 [F1]에서
Flirds .858 vs Flirds-1st .266 — 1차 근사는 순위를 거의 잃는다. 같은 칸의 개입 결과(§5.3의
.5668/.6065 vs vanilla 수준)와 부호가 정확히 맞는다. ② 나머지 오염 칸(label-flip)은 두 변형
모두 .98 이상으로, 절단 차수의 효과는 위협의 성격에 달려 있다 — 클라이언트 업데이트가
등방적 잡음으로 흐트러질수록 1차 항만으로는 부족해진다. ③ **모델 규모는 fidelity를 낮추지
않는다**: 표 [F3]에서 1B→7B 내내 두 변형 모두 .997 이상이다(이 무대는 $K{=}2$·clean-IID라
근사 부담 자체가 작다 — 스케일 불변성의 확인이지 우위의 증거는 아니다).

읽기 각주. **clean 칸은 신호-부재 레짐이다**: 클라이언트 간 실제 차이가 없어 in-run GT
자신의 순위조차 seed를 넘어 재현되지 않는다(부록 D). 다만 fidelity는 셀 안에서 같은 궤적을
두고 재는 양이라 이 칸에서도 정의된다 — 표 [F1]에서 Flirds가 clean .991을 얻는 것은 재현할
신호가 있다는 뜻이 아니라 잡음 수준의 값 차이까지 따라간다는 뜻이다. 이 칸은 방법 간 우열이
아니라 오발화 대조용으로만 읽는다.

**sub — retrain GT 특성화.** retrain GT는 $2^N$번의 전체 재학습을 요구하므로
주무대($N{=}50/100$)에서는 계산이 원리적으로 불가능하다 — 주무대와 $N$·참여율을 공유하는
retrain GT 무대는 존재할 수 없고, 정렬 가능한 축은 데이터·파티션·위협뿐이다. 그래서 트랙당
작은-$N$ 무대 하나가 주 표를 담당한다: **Silo**(LLM) = 5-도메인 비IID 실재-신호 무대,
**CNN-Small**(CNN) = 주무대와 데이터(CIFAR-10)·파티션({Dirichlet(α=1), iid})·위협 축을 전부
공유하는 $N{=}10$ 축소판. LLM 쪽 IID-clean 참조(`Anchor`)는 부록 C.6에 둔다.

표 [F4] — `Silo`(5-도메인 비IID; 실재-신호 무대): Flirds vs Flirds-1st, retrain GT ($2^5$) 대비 Spearman ↑ (1B, 3-seed) ●
<!-- 출처: runs/phase2_matrix/silo5_a_fidelity_1B.csv (열 spearman_a = vs (a)retrain GT;
     (b)oracle 행의 spearman_a = retrain GT↔in-run GT 일치도). rundir = 1B_silo5_{clean,noisy,frzero}_aonly_s{0,1,2}
     (3위협 × 3-seed = 9런, 셀당 2^5=32 재학습). 값 수준 Pearson(pearson_a): clean .9999±.0001 /
     answer-swap .9997±.0001 / free-rider .9998±.0000 (Flirds; Flirds-1st는 answer-swap만 .9996).
     전 방법(8종) = 부록 C.6(ComFedSV clean 열 ⬚ 주의). 본문 승격 = Yonghee 07-25(권고 채택). -->

| threat | Flirds | Flirds-1st | retrain GT ↔ in-run GT |
|---|---|---|---|
| clean | 1.000±.000 | 1.000±.000 | 1.000±.000 |
| answer-swap | 1.000±.000 | 1.000±.000 | 1.000±.000 |
| free-rider(zero) | 1.000±.000 | 1.000±.000 | 1.000±.000 |

읽기. Silo가 LLM 쪽 주 표인 이유는 둘이다: ① $N{=}5$ 무대 가운데 유일하게 in-run GT 타깃
순위가 seed를 넘어 재현되는 실재-신호 무대이고(부록 D: clean +0.87, 오염 +0.93), ② LLM
오염축 {clean, answer-swap, free-rider(zero)}이 전부 정렬된다. 헤드라인은 마지막 열이다 — 이
무대에서 **retrain GT↔in-run GT 일치도는 세 위협 모두 1.000±.000**(9셀 전량): in-run GT를
같은-게임 정답으로 쓰는 본 논문의 설계 선택을, 어느 방법의 목표값도 아닌 방법-중립 참값이
실재-신호 무대에서 직접 승인한다. Flirds·Flirds-1st도 같은 값에 도달하며, 순위가 $N{=}5$에서
포화하는 만큼 값 수준 Pearson을 함께 본다 — 두 변형 모두 .9996–.9999로, 일치는 순위뿐
아니라 값에서도 성립한다. **이 표는 방법 간 우열을 가리는 자리가 아니다**: 같은-게임 계열은
전 칸 포화이고(구별은 cross-game 계열에서만 생긴다 — ComFedSV .833–.867, FedIF .867–.933;
부록 C.6), 이 무대가 답하는 것은 "두 참값이 같은 것을 재는가" 하나다. caveat: 이 무대의
answer-swap은 클라 내 교체 비율 $nr{=}1.0$(무대 canonical)로 주무대의 0.7과 다르다(부록 B.2).
반면 IID-clean 참조 무대 `Anchor`에서는 두 참값 사이의 일치도 자체가 0.933으로 떨어져
같은-게임 계열의 점수가 그 천장에 갇힌다 — 신호-부재 무대의 한계이지 방법의 한계가 아니며,
전표와 읽기는 부록 C.6에 둔다.

표 [F5] — `CNN-Small`: Flirds vs Flirds-1st, retrain GT ($2^{10}$) 대비 Spearman ↑ (3-seed) ⬚
<!-- 채움: G2 — track_c1 위협·파티션 확장(free_rider·grad_noise 위협 + dir1 파티션) 후
     cifar10 {dir1,iid} × {clean, frzero, gn, lf@0.70} × 3-seed = 24셀 재실행.
     전 방법·Pearson·vs in-run GT = 부록 C.2·C.3. 구 시나리오 격자(feature-noise·label/quantity-skew·
     dose 사다리) 결과는 위협축 스코프 밖 — rundir 존속(runs/track_c/c1), 표 미수록. -->

| partition / threat | Flirds | Flirds-1st | retrain GT ↔ in-run GT |
|---|---|---|---|
| Dirichlet(α=1) / label-flip@0.70 | ⬚ | ⬚ | ⬚ |
| Dirichlet(α=1) / free-rider(zero) | ⬚ | ⬚ | ⬚ |
| Dirichlet(α=1) / gradient noise | ⬚ | ⬚ | ⬚ |
| Dirichlet(α=1) / clean | ⬚ | ⬚ | ⬚ |
| iid / label-flip@0.70 | ⬚ | ⬚ | ⬚ |
| iid / free-rider(zero) | ⬚ | ⬚ | ⬚ |
| iid / gradient noise | ⬚ | ⬚ | ⬚ |
| iid / clean | ⬚ | ⬚ | ⬚ |

이 표가 재는 것은 두 겹이다: ① 실재 신호가 있는 오염 칸에서 두 게임(realized 귀속 vs
counterfactual 재학습)이 수렴하는가 — 마지막 열(retrain GT↔in-run GT)이 그 게임-간 천장이고,
Flirds가 그 천장에 얼마나 붙는가가 방법의 몫이다. ② clean·iid 칸은 신호-부재 통제(위 각주와
같은 현상의 retrain GT-버전 — retrain GT 게임 값 자체가 재학습 노이즈 수준의 차이를 순위화)다.

### 5.3 개입 — 측정한 기여도로 학습을 개선하는가 (2차 ①)

**정책과 시점.** §1의 실효성 검증 가운데 "학습 개선" 축이다. 정책은 sign-gating 하나:
누적 기여도 $\hat\phi_k \le 0$인 클라이언트를 배제한다($\tau{=}0$, parameter-free — 명제 2의
free-rider exact-0과 clean 클라이언트의 누적 양수 성질이 문턱 0을 자연스러운 기준으로
만든다). 시점 2종: **online gating** — 매 라운드, 그 시점까지의 누적 부호로 즉시 게이트.
**retrain** — 관찰자 런의 최종 누적 부호로 남길 집합(kept)을 정하고 초기값부터
재학습(§1의 selection 실험). 통제 = vanilla(바닥) / oracle-제외(천장) / random-제외(동수
무작위) / retrain-random(retrain arm과 같은 kept 크기의 무작위 재학습 — 순위 정보의 가치를 크기 효과와
분리). 같은 정책에 점수원만 갈아끼우며 비교하므로, 이 절은 fidelity(§5.2)가 다운스트림
우열로 이어지는지의 직접 시험이기도 하다.

**LLM 주무대(`LLM-Main`), 절대 EM.** 표 [I1] ⬚ — 점수원은 **Flirds·Flirds-1st**로 좁힌다.
이 무대의 질문은 두 개다: 순위 정보의 가치(vs 무작위 통제)와 2차항의 기여(vs 1차). 전
점수원 경쟁은 CNN 표 [I2]가 담당한다.
<!-- 점수원 축소 = Yonghee 2026-07-25 결정: I1 = {vanilla, oracle-제외, random, Flirds, Flirds-1st}만.
     채움: runs/track_h/analysis/gsm50k5_*.csv (fix-후 rundir만; pre-fix seed0 git_sha fa5fc6e 인용 금지).
     현황: retrain {Flirds, Flirds-1st} × {answer-swap, frzero} 3-seed ●, online Flirds ●,
     clean 열 ◐(seed0). 남은 런 = online Flirds-1st(2위협 × 3-seed) + clean 열(vanilla·Flirds·
     Flirds-1st) 3-seed화. 구 G4의 renorm 4종·individual utility·FedIF 확장은 스코프 아웃
     (기산출 rundir 존속, 표 미수록). -->

**online gating**(정책 = sign-gating):

| arm | clean ◐ | answer-swap | free-rider |
|---|---|---|---|
| vanilla(관찰자) | .3727 | .3274±.0069 | .3560±.0157 |
| oracle-제외 (천장) | – | .3625±.0121 | .3625±.0121 |
| random-제외 (통제) | – | .3280±.0059 | .3476±.0179 |
| **Flirds** | .3664 | **.3479±.0054** | .3566±.0108 |
| Flirds-1st | ⬚ | ⬚ | ⬚ |

**retrain**(정책 = sign-gating):

| arm | clean ◐ | answer-swap | free-rider |
|---|---|---|---|
| vanilla(관찰자) | .3727 | .3274±.0069 | .3560±.0157 |
| oracle-제외 (천장) | – | .3625±.0121 | .3625±.0121 |
| retrain-random (통제) | – | .3220±.0104 | .3482±.0117 |
| **Flirds** | .3727 | **.3479±.0037** | **.3625±.0121** |
| Flirds-1st | .3727 | .3458±.0018 | **.3625±.0121** |

읽기(LLM 실측 기준). ① **answer-swap이 이 무대의 판별 칸이다**: vanilla .3274 → Flirds .3479로
천장(.3625) 대비 격차의 58%를 회수하는 반면, kept 크기를 맞춘 무작위 통제는 online .3280 ·
retrain .3220으로 vanilla 근처이거나 그 아래다 — 회수는 클라이언트를 덜어낸 효과가 아니라
**순위 정보** 자체에서 온다. Flirds-1st(retrain)는 .3458로 2차항 없이도 대부분을 따라오지만
같은 방향으로 조금 뒤진다(회수 52% vs 58%). ② **free-rider 칸은 retrain에서 정확 회수, online에서
판별력 없음**: retrain의 Flirds·Flirds-1st는 세 seed 모두 oracle-제외와 **비트 동일한 값**을
낸다 — kept 집합이 정확히 clean 30인과 일치하기 때문이다(명제 2의 exact-0이 그대로 정확
선별로 발현). 반면 online은 .3566으로 vanilla(.3560)와 사실상 같은데, 이 칸은 vanilla와 천장의
간격 자체가 0.65pt로 seed 분산(±1.1–1.6pt)보다 작아 **애초에 개입 이득을 잴 수 없는 칸**이다
(zero-update 클라이언트는 성능을 깎지도 않는다). 정직한 보고는 "online이 실패했다"가 아니라
"이 칸은 정확 선별(retrain)로만 검증된다"이다. ③ **clean 오발화**: online Flirds는 −0.63pt(.3664 vs
.3727), retrain은 아무도 배제하지 않아 vanilla와 완전히 같다(kept = 50) — CNN 표 [I2]의 ④와
같은 패턴이다. ◐ = clean 열은 seed0 단독.

**CNN 주무대, 절대 test acc.** 같은 정책·같은 무대·같은 seed에서 **점수원만 8종으로 교체**한
경쟁이다. 표 [I2]는 Dirichlet(α=1) 파티션†(오염-평균 = free-rider·gradient noise·label-flip
3위협 평균), iid 파티션 블록(동일 구성)은 ⬚.
<!-- 출처: runs/track_h/analysis/{competition_score,cnn_competition}.csv (07-20 집계 정정판).
     † = W-A(캠페인 restack 드리프트 표) 확인 후 확정 — paper/workplan/T4 참조.
     iid 블록 = G3: 비-Flirds 점수원 7종 84 rundir + 8점수원 T2 관측자(obs) 12 재실행 = 96 rundir;
     Flirds 온라인 arm은 track_g rundirs_cnn 144셀에서 기산출. -->

sign-gating · online:

| arm | clean | free-rider | gradient noise | label-flip@0.70 | **오염-평균** |
|---|---|---|---|---|---|
| vanilla (바닥) | .6389 | .5879 | .2436 | .5247 | .4521 |
| oracle-제외 (천장) | – | .6203 | .6203 | .6236 | .6214 |
| random-제외 (통제) | – | .5838 | .2590 | .5018 | .4482 |
| **Flirds** | .6315 | .6148 | **.5668** | .5712 | **.5843** |
| Flirds-1st | .6384 | **.6216** | .2479 | .5717 | .4804 |
| individual utility | .6264 | .6114 | .5981 | .5670 | .5922 |
| FedIF | .6386 | .6143 | .2479 | .5728 | .4783 |
| GTG | .6051 | .3915 | .5972 | .5479 | .5122 |
| FedSV | .5982 | .3966 | .5972 | .5164 | .5034 |
| ComFedSV | .5963 | .3918 | .5871 | .5152 | .4981 |
| ShapleyFL | .6045 | .4020 | .6115 | .5278 | .5138 |

sign-gating · retrain:

| arm | clean | free-rider | gradient noise | label-flip@0.70 | **오염-평균** |
|---|---|---|---|---|---|
| vanilla (바닥) | .6389 | .5879 | .2436 | .5247 | .4521 |
| oracle-제외 (천장) | – | .6203 | .6203 | .6236 | .6214 |
| **Flirds** | .6277 | .6063 | **.6065** | .6192 | **.6107** |
| Flirds-1st | .6386 | .6252 | .2436 | .6236 | .4975 |
| individual utility | .6293 | .6125 | .4518 | .6205 | .5616 |
| FedIF | .6417 | .6252 | .2436 | .6217 | .4968 |
| GTG | .6265 | .5158 | .6203 | .5991 | .5784 |
| FedSV | .6166 | .5140 | .6203 | .5904 | .5749 |
| ComFedSV | .6232 | .5200 | .6203 | .5921 | .5775 |
| ShapleyFL | .6223 | .5113 | .6203 | .6028 | .5781 |

읽기(CNN 실측 기준)†. ① **0-의미론이 게이트의 생사를 가른다**: free-rider 무대에서 exact-0
계열(Flirds·Flirds-1st·individual utility·FedIF)은 .61~.62로 천장(.620)에 근접하는 반면, 재정규화
계열 4종은 online에서 .39~.40으로 vanilla(.588)보다도 한참 낮다 — coalition 재정규화가
zero-update free-rider에게 몫을 계속 배분해, 게이트가 free-rider는 남기고 멀쩡한
클라이언트를 내쫓기 때문이다(§4.1의 게임 선택 논거가 성능으로 발현; retrain에선 온라인
복리 악화가 없어 .51~.52로 완화). ② **gradient noise는 2차항 판별 칸이다**: estimator 계열 중
Flirds만 회복하고(.5668 online / .6065 retrain), 1차 정보만 쓰는 Flirds-1st·FedIF는
.2479/.2436 ≈ vanilla(.2436) — noise 클라이언트를 아예 보지 못한다(§5.5-①). ③ label-flip은
재학습(retrain) 우위: 전 estimator가 online .57대 → retrain .62대(천장 근접)로 상승한다.
④ **clean 오발화의 정직 보고**: online에서 Flirds −0.7pt·individual utility −1.3pt,
Flirds-1st·FedIF는 무발화. 종합하면 정확한 클레임은 "Flirds 단독 1위"가 아니라 — **전
정책·전 시점 상위권 + gradient noise를 잡는 유일한 estimator**이며, 개별 칸 최고는
FedIF·individual utility도 차지한다. LLM 주무대(표 [I1])는 이 경쟁을 Flirds·Flirds-1st로
좁혀, 순위 정보의 가치(vs 무작위)와 절단 차수의 효과(vs 1차)만 절대 EM으로 확인한다 — 판별
칸(answer-swap)에서 천장 대비 회수 58%(Flirds) vs 52%(Flirds-1st), 무작위 통제는 0 이하다.

크기-가중 변형(sign-gating 배제 + 양수 누적 기여도 비례 가중)과 clean-IID 무해성 parity는
부록 E에 둔다.

### 5.4 비용

**연산수 모델 — $N$·$R$·$K$의 함수.** 라운드당 참여 수를 $K := |P_r|$로 두면, 각 방법의
valuation 비용은 라운드당 지배 연산수로 해석적으로 닫힌다(표 [O1]; 전 방법이 같은 동결 로그
위의 후처리라 학습 비용은 공통이므로 제외). 구조가 말하는 것: **Flirds의 비용은 $R$에만
비례하고 $K$·$N$과 무관**(라운드당 HVP 1회 — 참여자당 내적은 지배항이 아니다), in-run
GT·ShapleyFL은 $K$에 지수, individual utility는 $K$에 선형, Monte-Carlo 계열(FedSV·GTG)은
절단으로 지수 상한이 내려오며, ComFedSV만 $N$이 들어온다.

표 [O1] — 라운드당 지배 연산수(전체 = × $R$)
<!-- 출처: runs/measured_2026-07/op_counts.py::per_round(method, K, N) — G7 파라메트릭 재작성 -->

| method | 라운드당 | $K$·$N$ 의존성 |
|---|---|---|
| **Flirds** | **1 HVP** (+ $K$ dot) | **없음 — cohort 무관 상수** |
| Flirds-1st | 1 grad | 없음 |
| FedIF | 1 grad | 없음 |
| individual utility | $1{+}K$ fwd | $K$ 선형 |
| in-run GT | $2^K$ fwd | $K$ 지수 |
| ShapleyFL | $2^K$ fwd | $K$ 지수 |
| FedSV | $\le \min\!\big(2^K,\ \max(30, 2K)\cdot K\big)$ fwd | $K$ 지수 상한(절단으로 하락) |
| GTG | $\le \min\!\big(2^K,\ \max(30, \lceil 0.8\cdot 2^K\rceil)\cdot K\big)$ fwd | 〃 |
| ComFedSV | $\le 1+\min\!\big(2^K,\ M\cdot K\big)$ fwd, $M{=}\max(10,\lceil N\ln N\rceil)$ | $K$ 지수 상한 + **유일한 $N$ 의존** |

시간 환산은 microbench per-op 단가와의 곱이다(fp32·B200: forward 1.60s·HVP 10.36s,
HVP/forward ≈ 6.5) — 하드웨어가 바뀌면 이 두 수만 갈아끼운다. 이 하나의 식이 두 극단을 동시에
설명한다: 라운드당 참여가 아주 작은 무대(`LLM-Scale`, $K{=}2$)에서는 $2^2{=}4$ forward < 1
HVP ≈ 6.5 forward라 in-run GT 직접 열거가 오히려 싸고(실측 Flirds 1.61× — 부록 F),
$K{=}10$이면 지수 항이 폭발해 159×로 역전된다(아래). 즉 **Flirds의 비용 우위는 "참여가 많아
전수 열거가 지수적으로 비싼" 무대의 것**이고, 그렇지 않은 무대에서도 Flirds-1st(라운드당
gradient 1회)는 항상 최저가다. 모델 예측이 무대별 실측 wall-clock을 재현하는 검증 라인과 보조
무대 전표(retrain GT ~9× 포함)는 부록 F.

**주무대급 실측(`LLM-Device`, $K{=}10$·$R{=}30$).** 방법별 valuation 단독 wall-clock(1B,
3-seed):

| method | wall-clock (s) | Flirds 대비 |
|---|---|---|
| Flirds-1st | 53.8±2.3 | 0.34× |
| FedIF | 53.7±2.3 | 0.34× |
| **Flirds** | **157.3±5.4** | **1×** |
| ComFedSV | 357.9±18.2 | 2.3× |
| individual utility | ⬚ | – |
| FedSV | 4,969±200 | 31.6× |
| GTG | 18,149±1,592 | 115× |
| ShapleyFL | 24,935±1,123 | 159× |
| in-run GT | 24,975±1,115 | **159×** |

<!-- 출처: runs/phase2_matrix/analysis/04_device100_anchor/csv/runtime_table.csv, threat=noisy(answer-swap)
     열의 3-seed mean±std (rundir 1B_device100-a0.5_noisy_anchor). free-rider(zero) 열도 오차범위 내 동일
     (Flirds 157.156±7.073 / in-run GT 24975.698±1182.093) — wall-clock은 동결 로그 위 후처리라 위협 무관.
     individual utility(loss-heur)는 측정값 467.220±21.928s가 존재하나 **C6 이전(pre-fix) 측정**이라 미수록:
     당시 구현이 싱글턴마다 base loss를 재계산해 라운드당 2K fwd(=20)를 썼고, 2026-07-17 수정으로
     1+K(=11)로 정정됨(phi 비트동일, runtime만 변함 — cost-comparison-methodology §C6). 표 [O1]의
     1+K 모델과 어긋나는 값을 실측으로 실을 수 없으므로 교정본 재측정 후 기입. 재측정 시 예상 ~260s.
     탐지기 4종(FLDetector·FLTrust·STD-DAGMM·FedDQC) 행은 같은 CSV에 있으나 스코프 아웃. -->

in-run GT 대비 **159×**(표 [O1]의 30,720 forward 예측과 정합), coalition-열거
계열 대비로도 32×(FedSV)~159×(ShapleyFL)다. 이 표는 표 [O1]의 모델을 두 지점에서 확인해
준다: ① 라운드당 gradient 1회로 같은 자리에 있는 Flirds-1st와 FedIF가 53.8 vs 53.7로 사실상
겹치고, ② $2^K$ 열거를 그대로 수행하는 ShapleyFL(24,935)이 in-run GT(24,975)와 만난다 —
ShapleyFL의 비용이 exact 열거와 같다는 것은 그 방법이 겨냥하는 값이 다를 뿐 계산량은
줄지 않는다는 뜻이다. `LLM-Main`($K{=}5$)의 실측 wall-clock은 G1 rundir의 timing과 함께 ⬚.

### 5.5 Ablation

**① 2차항(클라이언트-상호작용 항)의 기여 — CNN 레그.** §4.2의 Flirds-1st와의 대조다.
fidelity 축: $N{=}10$ 참여 k-sweep probe(label-flip)에서 클라이언트당 참여 횟수가 적으면 1차
근사가 붕괴하고 2차항이 방어한다 — 라운드 참여 2/10에서 Flirds 0.891±.147 vs Flirds-1st
0.305±.434(전원 참여에선 0.993±.008 vs 0.940±.039; 전 72셀 풀 Flirds 0.953±.080).
다운스트림 축: 주무대 gradient noise 칸에서 2차항 유무가 회복과 실명을 가른다(§5.3: Flirds
.5668/.6065 vs 1차 계열 .2479/.2436 ≈ vanilla)†; 같은 칸의 fidelity 대조는 표 [F1]과 같은
rundir에서 나온다.

**② 2차항의 기여 — LLM 레그.** 주무대와 같은 참여 구조(5/50, $N{=}50$·$R{=}200$)의
alpaca-IID 무대에서 같은-게임 계열은 in-run GT 재현 0.999–1.000을 유지하지만(Flirds 3-seed
+1.000±.000) 1차-influence·재정규화 계열은 무너진다 — ComFedSV·ShapleyFL·FedIF 음수 붕괴,
GTG 0.98·FedSV 0.91. 이 대조는 LoRA rank {16,32,64}에 걸쳐 불변이다(전표 부록 C.5; r32·r64는
3-seed 보강 ⬚ <!-- G5: seed{1,2} × {r32,r64} = 4런 -->). 한편 클라이언트당 참여 횟수가
충분하면(~20회) Flirds-1st도 1.000을 유지한다 — 2차항이 필요한 레짐은 ①의 k-sweep과
합쳐 읽으면 "참여 분수"가 아니라 **클라이언트당 참여 횟수가 희소한** 레짐이다.

**③ Removal-curve — 게임-무관 인과 검증.** 각 방법의 기여도 순위대로 클라이언트를 실제로
제거하고 처음부터 재학습해, 순위의 인과적 타당성을 게임 정의와 무관한 공통 자(ruler)로
확인한다. LLM `Silo`(answer-swap·free-rider(zero) × 3-seed + clean 통제 ⬚): worst-first
제거가 val-loss를 내리고 best-first는 올린다 ⬚ <!-- 채움: runs/removal_dose 재풀링(frrand 행
제외; 구 3위협 풀 수치 +0.0067~+0.0076 / −0.0084~−0.0015는 재풀링 후 갱신). clean 통제 =
silo5 clean × 3seed 신규 3런(§5.1 clean-앵커 규칙; 플랜 §4.3 권고, ~1 GPU-h 미만) -->;
Flirds·Flirds-1st·individual utility·in-run GT의 removal 곡선은 전 셀에서 엄밀히 일치하고(같은
순위 → 같은 곡선), FedIF만 free-rider(zero)에서 질적으로 얕다. CNN 레그는 주무대 위협
축({label-flip@0.70, free-rider(zero), gradient noise} + iid/clean 통제)으로 정렬해 **정확도
축**의 같은 구조 — 순위 품질이 worst−best acc 분리로 이어지고 통제군 분리 ≈ 0 — 를 확인한다
⬚ <!-- G6: removal_dose CNN — frzero·gn × 3-seed(6런) + lf@0.70 재실행(3런); G2 러너 확장 공유 -->.

---

## 6. 논의와 한계

<!-- 스텁 — 실험 완주 후 작성. 확정 재료 목록(T1 스펙):
 - 궤적-특이 utility의 공리화 미해결 — IRDS로부터의 승계(§3.3·부록 A.8); 공리 성립 주장은 frozen 게임 한정.
 - per-sample→per-client 브리지의 LLM 한계: token-mean CE에서 분모 불일치로 비성립(부록 A.7) — LLM valuation을 per-sample 합으로 근거짓지 않음.
 - LLM 위협 축의 스코프: gradient noise는 LoRA 기하에서 무대 미성립(부록 B.6) — LLM 쪽 update-공격은 free-rider 계열로 한정. 서술 근거·수치 = runs/track_h/gnoise_diag/README.md(그대로 인용 가능; 단 그 안의 Krum σ=200·arXiv 3편은 검증 실패 = 인용 금지 목록 동봉).
 - "기여도≠탐지"의 게임-공통 사례 1문장: delta-재활용 free-rider(frdelta)는 val-loss를 실제로 낮춰 in-run GT 자신이 "기여함"으로 답함 — 추정 실패가 아니라 val-loss 게임의 정직한 답이며, 업데이트 패턴 자체를 보는 별도 축의 문제다.
 - 공정 분배·보상 스킴(fairness/reward)으로의 연결은 향후 과제.
 - 3-seed 규칙(전 수록 실험 seed{0,1,2}) 준수 명시; 미착지 ⬚ 칸의 해소 계획은 G-목록(연구노트)과 대응.
 - IID-clean 신호-부재 무대의 해석(부록 D): 방법이 아니라 무대의 결함 — retrain GT 특성화(CNN-Small·Anchor)의 clean·iid 칸과 LLM-Scale의 낮은/불안정 ρ 해석 포함.
 - 참여 lottery: 어느 라운드·어느 cohort에 뽑혔는지가 지급액에 영향 — symmetry는 라운드 게임 안에서만 성립하고, 라운드 배정 자체는 운영자 정책의 몫이다(정산 프레임의 스코프 한계).
-->

---

## 7. 결론

<!-- 골격 — 실험 완주 후 작성:
 - 헤드라인: §1 "측정에서 정산으로"가 열거한 요구들이 서로 배타적이지 않음을 보였다 —
   배타성의 원인이던 조합 재평가를 closed-form으로 없앤 것이 열쇠(§2 두 갈래 → §4).
 - LLM-scale 최초 2건 회수(기여 1·2): 클라이언트-수준 연합 valuation의 LLM 규모 수행 +
   exact GT 대비 채점의 LLM 규모 확장. 성질별 "최초" 주장은 하지 않는다.
 - 결과를 위계 순으로 한 문장씩: fidelity(§5.2) → 개입(§5.3) → 비용(§5.4); 탐지는 부록 G.
 - 향후: 궤적-특이 게임의 공리화, incentive·보상 스킴 설계로의 연결(§6).
-->


---

## 부록

### 부록 A. 증명과 형식 서술

*(전체 형식화의 출처는 내부 math-rigor dossier(2026-07-04, 반박 패널 36건 반영본)이며, 여기서는
논문 부록 분량으로 정리한다. 표기: retrain GT = exact retrain Shapley, in-run GT = exact in-run Shapley.
본문 명제와의 대응: 명제 1 = P1(+L1), 명제 2 = P2(+따름 P2-1·P2-2), 명제 3 = P3(+L2),
§4.1의 고정-가중 논거와 §5.1의 공정성 원칙 = P4(i)+P5, 비고(적용 범위) = P4(ii–iv)+P6+P8.)*

**A.1 시스템 모델과 가정.** 동기식 FedAvg를 가정한다: 라운드 $r$의 참여자 $P_r$은 같은 시작
상태 $w^r$에서 로컬 학습을 시작하고(모델차 $\delta_k^r$ 제출), 서버는 무상태 가중합
$w^{r+1} = w^r + \sum_{k \in P_r} p_k^r \delta_k^r$, $p_k^r = n_k / \sum_{j \in P_r} n_j$로
집계한다(가중 분모는 $P_r$ 전체로, coalition $S$에 비의존인 **고정 가중**이다). 로컬 optimizer는
momentum 없는 plain SGD·상수 학습률이며 라운드마다 새로 생성된다(클라이언트 stateless).
frozen log는 $[(w^r, \{(\delta_k^r, n_k)\}_{k \in P_r})]_{r<R}$이다. 표기:
$g_r = \nabla\ell_{\mathrm{val}}(w^r)$, $H_r = \nabla^2\ell_{\mathrm{val}}(w^r)$(true Hessian; GGN/Fisher 아님),
$a_k^r = p_k^r \delta_k^r$, $\Delta_S^r = \sum_{k \in S \cap P_r} a_k^r$,
$\Delta W_r = \Delta_{P_r}$. 부호 규약은 본문과 동일하다: $\phi$는 검증 손실 **감소**의
귀속(유익한 클라이언트 $\Rightarrow \phi_k > 0$)이다.

**A.2 게임 정의.** 라운드 부분게임(고정 가중)은
$u_r(S) := \ell_{\mathrm{val}}(w^r) - \ell_{\mathrm{val}}(w^r + \Delta_S^r)$, 전체 게임은 $U_b(S) := \sum_r u_r(S)$이며, in-run GT는
이 게임의 exact Shapley다. 라운드 surrogate는
$\hat u_r(S) := -\langle g_r, \Delta_S^r \rangle - \tfrac12 \langle \Delta_S^r, H_r \Delta_S^r \rangle$(1차 전용은
$\hat u_r^{(1)}$). **추정기가 근사하는 게임이 정확히 $U_b$다**. 같은 로그·같은 $\ell_{\mathrm{val}}$·같은
$S$-비의존 가중·같은 전개점이고, 유일한 차이는 라운드별 2차 Taylor 절단이다. retrain GT는 $S$만으로
처음부터 재학습한 게임으로, 분모가 $\sum_{j\in S} n_j$로 재정규화되고 궤적 전체가 $S$에
의존하는 **다른 게임**이다(retrain GT↔in-run GT 관계는 이론이 아니라 실증 질문; A.8).

**A.3 명제 G1 (telescoping/efficiency).** $S \supseteq P_r$이면 $u_r(S) = \ell_{\mathrm{val}}(w^r) -
\ell_{\mathrm{val}}(w^{r+1})$ (정확). 라운드 합산의 telescoping으로 $U_b([N]) = \ell_{\mathrm{val}}(w^0) - \ell_{\mathrm{val}}(w^R)$이고
Shapley efficiency에 의해 $\sum_k \phi_k(U_b) = \ell_{\mathrm{val}}(w^0) - \ell_{\mathrm{val}}(w^R)$이다. 즉 전 클라이언트
기여도의 합은 근사 없이 전체 run의 검증 손실 감소와 같다. *증명*: 서버 무상태성으로 grand
coalition 섭동이 실현 스텝과 일치. $\square$

**A.4 명제 P1 (per-round 분해; 보조정리 L1).** **L1 (null-player 제거)**: $v(S) = v(S \cap
T)$ 꼴 게임에서 $i \notin T$는 $\phi_i = 0$이고, $i \in T$의 Shapley는 $T$-부분게임의
Shapley와 같다. *증명 스케치*: 비참여자의 모든 marginal이 0; 균등 랜덤 순열이 $T$에 유도하는
상대순서가 균등함을 세어 순열형 기대값을 환원. $\square$ 이를 $T = P_r$로 적용하면
$\phi_k(U_b) = \sum_{r: k \in P_r} \phi_k^{P_r}(u_r|_{P_r})$이다. 즉 **비참여 라운드의 기여 0은
정의가 아니라 정리**이고, $N$-인 게임이 라운드별 $|P_r|$-인 부분게임으로 정확히 축약된다.
(IRDS는 이 성질을 외부 정리 인용으로 처리했으나 여기서는 직접 증명한다.)

**A.5 명제 P2 (closed-form).** $k \in P_r$에 대해 추정기는 본문 식 (7)과 같은 closed-form
$$\phi_k(\hat u_r) = -p_k^r \langle g_r, \delta_k^r \rangle - \tfrac12\, p_k^r \langle
\delta_k^r,\, H_r \Delta W_r \rangle$$
을 갖는다. *증명 스케치 (unanimity 분해)*: $\hat u_r$을 가산부($-\langle g_r, a_k^r\rangle - \tfrac12
q_{kk}$)와 쌍대부($q_{ij} = \langle a_i^r, H_r a_j^r\rangle$)로 나누면, 2-인 unanimity 게임의 Shapley가
당사자 절반씩이므로 $\phi_k = -\langle g_r, a_k^r \rangle - \tfrac12 \sum_j q_{kj} = -\langle g_r,
a_k^r\rangle - \tfrac12 \langle a_k^r, H_r \textstyle\sum_j a_j^r\rangle$. $\square$ 우측 공통 벡터 $H_r \Delta W_r$이 전
클라이언트에 공유되므로 라운드당 **HVP 1회**로 닫힌다. **따름 P2-1 (free-rider exact-0)**:
$\delta_k^r = 0$이면 0-텐서와의 내적이라 추정기 값은 **대수적으로 정확히 0**(수치 결정성 무관);
in-run GT 게임 쪽 0은 forward의 bit-결정성이 전제다(CNN cudnn-deterministic 확립, LLM은 조건부).
**따름 P2-2 (efficiency)**: $\sum_k \phi_k(\hat u_r) = \hat u_r(P_r)$.

**A.6 명제 P3 (Taylor 잔차; 보조정리 L2).** $\ell_{\mathrm{val}} \in C^3$과 국소 상수 $M_2^r, M_3^r$ 하에
$|u_r(S) - \hat u_r(S)| \le \tfrac{M_3^r}{6}\|\Delta_S^r\|^3$, 1차는
$\tfrac{M_2^r}{2}\|\Delta_S^r\|^2$. **L2 (오차 전파)**: $|\phi_i(v) - \phi_i(v')| \le 2\|v -
v'\|_\infty$. 결합하면 $|\phi_k(U_b) - \hat\phi_k| \le \sum_r \tfrac{M_3^r}{3} \max_S
\|\Delta_S^r\|^3$ (1차 추정기는 선행계수가 $M_2^r$이며, $M_2/3$을 쓰면 무효 상계다). **지위**:
이는 상계이고 상수는 추정 가능한 양이 아니며 $R$에 선형 누적된다 — 본 논문에서 이 bound의
역할은 잔차의 차수(스케일링 법칙)를 밝히는 데서 그친다(본문 명제 3과 동일한 이론-한정
지위). per-round 이동이 $K$-스텝 누적이라 IRDS per-step보다 구조적으로 약한 보증이라는
점도 명시한다.

**A.7 명제 P4 (per-sample→per-client 브리지; 보조정리 L3).** **(i) 고정 가중의 필수성**:
재정규화 이동 $\widetilde\Delta_S$는 coalition-독립 벡터족의 합으로 표현 불가(등$n$ 2-클라
반례: singleton이 $b_k = \delta_k$를 강제하나 pair가 $\tfrac12(\delta_1+\delta_2)$). 즉 고정
가중은 편의가 아니라 IRDS closed-form 기계 전체를 이식하기 위한 필수 가정이다. **(ii) L3
(merge-consistency)**: 2-additive 게임에서 플레이어 블록 병합 후의 Shapley는 블록 내 Shapley
합과 정확히 일치한다(쌍별 상호작용이 항상 당사자 절반씩 분배되므로). **(iii) 따름 P4b**:
1-step·full-batch·mean-CE 극한에서 per-client 게임은 per-sample 게임의 블록 병합이고, 클라
값 = 소속 샘플 값의 합(정확). **스코프**: 이 상쇄는 "로컬-손실 reduction 분모 = FedAvg 집계
가중"일 때만 성립한다. CNN mean-CE(이미지 수)는 성립하지만, **LLM token-mean CE는 분모가 토큰
수·가중이 시퀀스 수라 성립하지 않는다**(LLM valuation을 per-sample 값의 합으로 근거짓지
않음). **(iv) 3차 실패 반례**: 3-인 unanimity 게임을 $\{1,2\},\{3\}$으로 병합하면 블록 합
$\tfrac23 \ne$ 병합 Shapley $\tfrac12$이다. 즉 2차 surrogate를 넘어서면 두 단위는 원리적으로
다르다(크기는 P4c bound).

**A.8 명제 P5–P6 (게임 선택과 경로-의존).** 등$n$·1차 극한에서 고정-가중과 재정규화 게임은
라운드별 **순위 동일**(P5b; 재정규화 Shapley가 $b_k$의 공통 양-기울기 affine)이나, 이
안심은 좁다: ① **비등$n$ 반례**($n_1{:}n_2 = 3{:}1$, $b = (1,2)$)에서 1차인데도 순위가
반전되고, ② **부분참여**에서는 라운드-합 리프트가 깨지며(3-클라 2-라운드 반례), ③ 등$n$·정확
가산 게임에서도 singleton 곡률의 $c_S^2$ 증폭으로 반전한다. near-additivity 항등식
$\hat u_r(P_r) - \sum_k \hat u_r(\{k\}) = \sum_{i<j} q_{ij}$는 "같은 게임의 semivalue들이
붕괴"함을 말할 뿐 "다른 두 게임이 같은 답을 줌"을 말하지 않는다. **P6**: 라운드 합산의 대수는
정확하지만(선형성), 전개점 $w^r$이 실현 궤적의 함수라는 의미의 경로-의존은 in-run GT 게임에
내재한다. $R{=}1$(후속 궤적이 없어 경로-의존이 정확히 0)에서도 위 ①의 반례로 retrain GT와 in-run GT의
순위가 반전되므로, retrain GT≈in-run GT는 궤적 안정성만으로는 성립하지 않고 등$n$/near-additive 조건이
추가로 필요하다. 궤적-특이 utility의 공리화는 IRDS가 명시한 열린 문제이며 본 논문에
승계된다(공리 성립 주장은 frozen 게임 한정).

**A.9 명제 P7 (momentum 스코핑).** 클라이언트가 stateless인 한 로컬 optimizer는 게임·closed-form
정의에 영향이 없다(realized $\delta$만 소비). 진짜 load-bearing한 것은 **서버 무상태성**이다:
서버 momentum(FedAvgM)이 있으면 grand coalition 섭동이 실현 스텝과 달라져 G1(telescoping)이
깨지고, 게임이 "실현 run의 가산 분해"라는 의미론을 잃는다. 라운드 간 상태를 갖는 로컬
optimizer는 P4 브리지를 상실시킨다. 둘 다 가정으로 명시한다(§5.1, §6).

**A.10 명제 P8 (LoRA 좌표).** 게임·추정기·in-run GT가 전부 같은 LoRA 인자 좌표 $\theta$에서
정의·연산되므로 P1–P7은 $\ell_{\mathrm{val}} \to \tilde\ell_{\mathrm{val}}(\theta)$ 치환 하에 문자 그대로 성립하고, 부분공간
제한이 추가 오차를 만들지 않는다(LoRA 사상이 다항이라 $\tilde\ell_{\mathrm{val}} \in C^3$ 자동). 게임 값
자체는 모델 평가만으로 정의되므로 좌표 무관이다. full-weight 재해석을 유보하는 실질 이유는
둘이다: ① 집계가 factor에 선형이라 가법 구조 $\Delta_S^r = \sum_k a_k^r$는 $\theta$-공간의
사실이다 — $w = W_0 + BA$의 쌍선형성 탓에 coalition의 full-weight 이동은 클라이언트 간
교차곱 $p_j^r p_k^r\, \Delta B_j \Delta A_k$를 포함해, 일반적으로 coalition-독립 per-client
벡터족의 합으로 표현되지 않는다(P4(i)와 같은 유형의 표현 불가). ② Hessian 연쇄법칙
$\nabla_\theta^2 \tilde\ell_{\mathrm{val}} = J^\top \nabla_w^2 \ell_{\mathrm{val}}\, J
+ \sum_i \partial_i \ell_{\mathrm{val}}\, \nabla_\theta^2 w_i$($J = \partial w / \partial\theta$)의
둘째 항(사상 곡률)이 바로 위 교차곱을 계상하므로 2차 surrogate는 좌표 의존이고,
$\theta$-Hessian을 weight-Hessian의 당김($J^\top \nabla_w^2 \ell_{\mathrm{val}} J$)으로 읽어서는 안 된다.

**A.11 대수 검증 (GPT-2 스모크).** GPT-2($N{=}5$, $R{=}3$, fp32)에서: closed-form =
추정기 최대 절대차 $5.8 \times 10^{-12}$; $\hat u^{(2)}$ 게임의 $2^5$ 전수 Shapley = closed-form
$7.6 \times 10^{-12}$; per-round 분해 = $2^N$ 전수 in-run GT $3.9 \times 10^{-7}$(fp32 forward
노이즈 바닥); telescoping 잔차 $\le 9.5 \times 10^{-7}$; Hessian 대칭성 $\le 1.8 \times
10^{-11}$.

### 부록 B. 실험 프로토콜 상세

**B.1 무대 하이퍼파라미터.** (전량; 셀별 원본은 각 rundir의 `config`)

| 무대 | 모델 | 데이터·분배 | $N$ · 참여 | $R$ · local | optimizer | 기타 |
|---|---|---|---|---|---|---|
| `LLM-Main` | Llama-3.2-1B-Instruct + LoRA r16/α32 | GSM8K train 7,473 → 클라당 149문항 IID 균등 | 50 · 5/50 | 200 · 10 steps × batch 16 | SGD mom 0, lr 1e-3 | maxlen 512; val 200 = 공식 test에서 카브, 심판 = 잔여 1,119 EM |
| `LLM-Scale` | Llama 1B·3B·7B + LoRA r16/α32 | alpaca-gpt4 20k IID | 20 · 2/20 | 200 · 10 steps × batch 16 | SGD mom 0, lr 1e-3 | seq 512; clean-IID 전용 |
| `Silo` | Llama-3.2-1B + LoRA r16/α32 | 5-도메인 비IID(클라 = 도메인); train 200 / val 20 / test 40 | 5 · 전원 | 10 · 10 steps × batch 16 | SGD mom 0, lr 1e-3 | maxlen 768, warmup 2; 이중 GT(retrain+in-run) $2^5$ |
| `Anchor` | Llama-3.2-1B-Instruct + LoRA r16/α32 | alpaca-gpt4 20k IID; val 200 / test 1000 | 5 · 전원 | 30 · 10 steps × batch 16 | SGD mom 0, lr 1e-3 | seq 512; 이중 GT(retrain+in-run) $2^5$ |
| `CNN-Main` | CIFAR-10 = FedSVCNN | 파티션 2종(B.3); MNIST 짝 = LeNet5(부록) | 100 · 10/100 | 120 · 5 epochs × batch 64 | SGD mom 0, lr 0.01 | 분배 seed 고정 |
| `CNN-Small` | CIFAR-10 = FedSVCNN | `CNN-Main`과 동일 파티션·위협; val 2000 / test 8000; MNIST 짝(부록) | 10 · 전원 | 10 · 5 epochs × batch 64 | SGD mom 0, lr 0.01 | 이중 GT(retrain+in-run) $2^{10}$ |
| `LLM-Device` | Llama-3.2-1B + LoRA r16/α32 | alpaca, Dirichlet $\alpha{=}0.5$(탐지는 $\alpha$ 스윕), 클라당 300; val 10 / test 40 | 100 · 10/100 | 30 · 5 steps | SGD mom 0, lr 1e-3 | – |

전 무대 fp32 학습·채점.

**B.2 위협 구현.** 모든 오염은 클라이언트-재현적 seed로 고정된다(같은 셀 재실행 시 동일
오염 실현).

- **answer-swap@rate**(LLM): 클라이언트 데이터의 rate 비율에서 응답(풀이 + 최종
  답)을 같은 클라이언트의 다른 문항 것으로 순열 교체 — 형식은 완전한 정상 데이터이나
  문항-응답 대응이 깨진 현실적 mislabel. `LLM-Main`은 rate 0.7; `Silo`의
  answer-swap($nr$)도 같은 순열 교체($nr$ = 클라 내 교체 비율; 이 무대 canonical은
  $nr{=}1.0$).
- **free-rider(zero)**: 로컬 학습 없이 $\Delta = 0$ 제출.
- **gradient noise**(CNN): 정상 로컬 학습 후 업데이트에 가우시안 노이즈 주입.
- **label-flip@0.70**(CNN): 오염 클라이언트 데이터의 70%에서 라벨을 **참 라벨을 제외한**
  $K{-}1$개 오답 중 균일 무작위로 교체 — 참 라벨을 배제하므로 명목 rate가 곧 실효 오염률이다.

**강도는 상수로 고정한다.** 어느 무대에서도 위협 강도를 분포에서 뽑지 않는다(label-flip 도즈
0.70 · answer-swap rate 0.7, `Silo`는 $nr{=}1.0$ · gradient noise $\sigma{=}0.1$ ·
free-rider(zero)는 $\Delta{=}0$이라 강도 자유도가 없다). 도즈-반응을 따로 보지 않는 표에서
강도가 시드마다 흔들리면 방법 간 차이와 강도의 흔들림이 분리되지 않기 때문이다.

**오염 클라이언트 집합은 무대별로 명목 비율과 실현 방식이 다르다.** 표에는 명목 비율 대신
**실현 수**를 병기한다:

| 무대 | 실현 방식 | 실현 오염 수 (seed 0/1/2) |
|---|---|---|
| `CNN-Main` ($N{=}100$) | label-flip: 클라이언트별 독립 Bernoulli($\rho{=}0.4$) → 시드마다 변동 | 39 / 48 / 47 (평균 44.7%) |
| 〃 | free-rider · gradient noise: $\lfloor \rho N \rceil$ 비복원 추출 | 40 / 40 / 40 (40%) |
| `CNN-Small` ($N{=}10$) | 전 위협 $\lfloor \rho N \rceil$ 비복원 추출 | 4 / 4 / 4 (40%) |
| `LLM-Main` ($N{=}50$) | 전 위협 고정 인덱스 $\{0,\dots,19\}$ | 20 / 20 / 20 (40%) |
| `LLM-Device` ($N{=}100$) | 전 위협 고정 인덱스 $\{10,30,50,70,90\}$ | 5 / 5 / 5 (5%) |
| `Silo` ($N{=}5$) | answer-swap = 클라 $\{0\}$ · free-rider = 클라 $\{1\}$ | 1 / 1 / 1 (20%) |

세 가지를 덧붙인다. ① **고정 인덱스는 일반성을 잃지 않는다** — LLM 무대의 클라이언트 샤드는
공식 split을 시드별로 셔플해 등분한 것이므로 같은 인덱스가 쥔 데이터가 시드마다 바뀐다.
② **$N{=}10$에서 Bernoulli를 쓰지 않는 이유는 상대 변동**이다 — $N{=}100$에서 39–48은 명목값의
$\pm12\%$지만 $N{=}10$이면 3/7/6, 즉 $\pm39\%$로 흔들려 label-flip 열만 실효 도즈가 달라지고 네
위협이 한 fidelity 표를 공유할 수 없다. ③ **모든 대조는 위협과 무대를 고정한 채 이뤄지므로 이
차이가 한 비교 안에서 만나지 않는다** — 무대 간 수치를 직접 겹쳐 읽지 않는다는 뜻이고, 이는
$N$·참여율·$R$이 이미 무대마다 다르다는 제약(§5.1)과 같은 성격이다.

**B.3 데이터 분배.** GSM8K: 공식 test 1,319문항에서 200을 카브해 서버 검증셋으로, 잔여
1,119문항이 성능 심판 — 학습 데이터와의 분리는 공식 split 그대로다. CNN 파티션 2종:
`iid` / `Dirichlet(α=1)` = 라벨+크기 동시 skew. `CNN-Small`은 별도 시나리오 없이 주무대의
파티션·위협 축을 $N{=}10$으로 그대로 축소한다.

**B.4 연산 환경.** 학습·채점 전량 fp32(단 cuDNN convolution의 TF32 기본 활성은 CNN 트랙에
노출됨을 명시); 스택 내 결정론 옵션 고정(같은 seed 재실행의 궤적 재현 — free-rider exact-0의
in-run GT-쪽 전제). 실행 환경(GPU·라이브러리 버전)은 셀별 rundir meta에 기록된다.

**B.5 baseline 재구현 주석.** ① 부호 규약은 전 방법 contribution orientation(도움 =
양수)으로 통일했다(원 정의가 반대 방향이면 부호 반전만). ② ShapleyFL의 EMA는
$\beta{=}0.3$이다($\beta{=}0.5$ 대비 차이가 같은 셀 재실행 노이즈 수준임을 확인).
<!-- ⚠ 재실행 대기(2026-07-23 Yonghee 결정 = β0.3 재실행): 현재 이 "β=0.3" 서술은 논문 인용
     ShapleyFL 값과 잠정 불일치다 — 인용 값(부록 C.6 Anchor vs retrain GT의 .767)은
     아직 β=0.5 rundir 산출(1B_anchor5 3셀 = git_sha 39a0a97, 2026-06-15, beta=0.5).
     구 C1 30셀도 β=0.5였으나 CNN-Small 재편(G2)으로 전량 재실행되므로 자동 해소.
     Anchor 해소 = β0.3 재실행(B200) 착지 후 rundir 교체 + 부록 C.6 값 갱신 + 이 주석 삭제.
     same-game 본문 주장엔 무영향(ShapleyFL=cross-game 비교군). -->
③
ComFedSV의 utility 행렬 low-rank 완성은 사후 일괄 계산이 원형이므로, per-round 점수가
필요한 개입 무대(§5.3)에서는 균등평균 submodel + 손실-감소 효용의 per-round 대용치(원 논문
Eq. 6 기반)를 쓴다 — fidelity 무대(§5.2)는 원형 그대로다.

**B.6 LLM 위협 축에 gradient noise가 없는 이유.** 업데이트에 주입하는 등방(isotropic) 노이즈는
LoRA 인자 공간의 기하에서 검증-gradient 방향 성분이 미미해, CNN에서와 달리 성능·φ 어느
쪽에도 유의한 오염 효과를 만들지 못한다(무대 미성립). LLM 쪽 update-공격 축은 free-rider
계열이 담당한다.

### 부록 C. Fidelity 확장

**C.1 주무대 전 방법 vs in-run GT.** ⬚ — §5.2 표 F1·F2의 전 방법 확장판이다. 같은 표에서
읽히도록 본문 2종(Flirds·Flirds-1st) 행을 함께 싣는다. cross-game 계열의 in-run GT 대비
불일치에는 근사 오차와 "다른 게임" 성분이 합산돼 있음을 §5.1의 구분과 함께 다시 밝힌다.
CNN 열의 MNIST 짝(동일 세팅, 데이터셋만 교체)은 ⬚ <!-- G8: track_c2에 mnist 추가(MODEL_FN
LeNet5) 후 2파티션 × 4위협(clean 포함) × 3-seed = 24셀 -->.
<!-- 채움: c2fid analysis(cifar10 {dir1,iid} 재풀링) + LLM-Main G1(8방법 × answer-swap·clean). -->

| method | `CNN-Main` clean | free-rider(zero) | gradient noise | label-flip@0.70 | `LLM-Main` answer-swap | free-rider(zero) | clean |
|---|---|---|---|---|---|---|---|
| Flirds | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| Flirds-1st | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| individual utility | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| GTG | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| FedSV | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| ComFedSV | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| ShapleyFL | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| FedIF | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |

**C.2 `CNN-Small` 파티션·위협별 vs in-run GT, 전 방법 — Spearman ↑** (3-seed) ⬚
<!-- 채움: G2 재실행 rundir(track_c1 확장) — cifar10 {dir1,iid} × {clean, frzero, gn, lf@0.70}.
     구 시나리오 격자(feature-noise·label/quantity-skew·dose 사다리) 표는 위협축 스코프 밖으로
     폐기(rundir 존속 runs/track_c/c1). MNIST 짝 = G9(동일 확장) ⬚. -->

| partition / threat | Flirds | Flirds-1st | individual utility | GTG | FedSV | ComFedSV | ShapleyFL | FedIF |
|---|---|---|---|---|---|---|---|---|
| dir1 / label-flip@0.70 | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| dir1 / free-rider(zero) | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| dir1 / gradient noise | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| dir1 / clean | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| iid / label-flip@0.70 | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| iid / free-rider(zero) | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| iid / gradient noise | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| iid / clean | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |

**C.3 `CNN-Small` 파티션·위협별 vs retrain GT, 전 방법 — Spearman ↑** (§5.2 표 F5의 확장) ⬚
<!-- 채움: G2 — C.2와 같은 rundir의 retrain GT leg; Pearson도 동일 분석 파일에서 재생성 -->

| partition / threat | Flirds | Flirds-1st | individual utility | GTG | FedSV | ComFedSV | ShapleyFL | FedIF |
|---|---|---|---|---|---|---|---|---|
| dir1 / label-flip@0.70 | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| dir1 / free-rider(zero) | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| dir1 / gradient noise | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| dir1 / clean | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| iid / (4위협 동일 구성) | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |

**C.4 Kendall·거리 지표 풀** (vs in-run GT) ⬚
<!-- 채움: 주무대(c2fid 재풀링·G1)와 CNN-Small(G2)에서 Kendall τ·cosine/euclid/max 거리 —
     구 시나리오 격자 풀 값은 무대 폐기로 미수록 -->

| method | Kendall ↑ | cosine_d ↓ | euclid_d ↓ | max_diff ↓ |
|---|---|---|---|---|
| Flirds | ⬚ | ⬚ | ⬚ | ⬚ |
| Flirds-1st | ⬚ | ⬚ | ⬚ | ⬚ |
| individual utility | ⬚ | ⬚ | ⬚ | ⬚ |
| GTG | ⬚ | ⬚ | ⬚ | ⬚ |
| FedSV | ⬚ | ⬚ | ⬚ | ⬚ |
| ComFedSV | ⬚ | ⬚ | ⬚ | ⬚ |
| ShapleyFL | ⬚ | ⬚ | ⬚ | ⬚ |
| FedIF | ⬚ | ⬚ | ⬚ | ⬚ |

**C.5 부분참여 probe(`Partial-Probe`: $N{=}50$, 5/50, $R{=}200$, IID-clean) — vs in-run GT Spearman ↑**
(LoRA rank별; r16 3-seed ●, r32·r64는 3-seed 보강 ⬚ <!-- G5: seed{1,2} × {r32,r64} -->)

| method | r16 | r32 | r64 |
|---|---|---|---|
| **Flirds** | **1.000** | **1.000** | **1.000** |
| Flirds-1st | 1.000 | 1.000 | 0.999 |
| individual utility | 1.000 | 1.000 | 0.999 |
| GTG | 0.983 | 0.983 | 0.981 |
| FedSV | 0.910 | 0.899 | 0.909 |
| FedIF | −0.040 | −0.076 | −0.052 |
| ShapleyFL | −0.064 | −0.093 | −0.078 |
| ComFedSV | −0.109 | −0.125 | −0.081 |

전원 참여 무대(anchor류)에서는 대부분 방법이 1.000으로 붕괴-동률이지만, 부분참여가 방법을
가른다 — uniform-subset/1차-influence 계열(ComFedSV·ShapleyFL·FedIF)은 in-run GT 반대
순위(음수)로 붕괴하고, 같은-게임 계열은 rank와 무관하게 1.000을 유지한다(Flirds r16 3-seed
+1.000±.000). §5.5-②(2차항 LLM 레그)의 전표다.

**C.6 LLM 보조 무대 전 방법 vs retrain GT ($2^5$)** — §5.2 표 F4(`Silo`)의 전 방법 확장판과
IID-clean 참조 전표(`Anchor`; 본문에서 내린 폴백). retrain GT는 어느 방법의 목표값도 아닌
중립 참값이므로 전 방법을 같은 표에 둔다.

`Silo`(실재-신호 무대; 1B, 3-seed) ⬚
<!-- 채움: silo5 (a)-leg rundir + canonical phi.parquet 조인 — 실측 완료(●), 값 전사만 남음.
     ComFedSV clean 열은 ⬚(미산출). -->

| method | clean | answer-swap | free-rider(zero) |
|---|---|---|---|
| Flirds | ⬚ | ⬚ | ⬚ |
| Flirds-1st | ⬚ | ⬚ | ⬚ |
| individual utility | ⬚ | ⬚ | ⬚ |
| GTG | ⬚ | ⬚ | ⬚ |
| FedSV | ⬚ | ⬚ | ⬚ |
| ComFedSV | ⬚ | ⬚ | ⬚ |
| ShapleyFL | ⬚ | ⬚ | ⬚ |
| FedIF | ⬚ | ⬚ | ⬚ |
| **retrain GT ↔ in-run GT 일치도** | ⬚ | ⬚ | ⬚ |

`Anchor`(IID-clean 참조; 1B, 3-seed)
<!-- 출처: runs/track_d/rundirs/1B_anchor5_seed{0,1,2}/phi.parquet — truth=retrain GT로 피벗 재계산 -->

| method | Spearman vs retrain GT ↑ | Pearson vs retrain GT ↑ | (참고) Spearman vs in-run GT |
|---|---|---|---|
| Flirds | 0.933±.047 | 0.933±.055 | 1.000 |
| Flirds-1st | 0.933±.047 | 0.929±.060 | 1.000 |
| individual utility | 0.933±.047 | 0.931±.057 | 1.000 |
| GTG | 0.933±.047 | 0.937±.052 | 1.000 |
| FedSV | 0.733±.170 | 0.685±.249 | 0.700 |
| ShapleyFL | 0.767±.330 | 0.916±.084 | 0.700 |
| ComFedSV | 0.467±.450 | 0.598±.280 | 0.500 |
| FedIF | 0.167±.613 | 0.048±.585 | 0.067 |

Anchor에서 갈리는 지점은 **천장(0.933 = 두 참값 일치도)에 도달하는가**다: 같은-게임 3종과
GTG는 in-run GT를 사실상 완전 재현하므로(vs in-run GT ≈ 1.000) retrain GT 점수가 게임 간
간극으로 수렴해 동률이 되고, in-run GT 재현이 무너지는 방법들(FedSV·ShapleyFL 0.700,
ComFedSV 0.500, FedIF 0.067)은 그만큼 retrain GT 대비로도 내려앉는다. 즉 이 무대에서
retrain GT 점수는 대체로 in-run GT 재현도의 함수이며, 두 참값을 각각 겨냥해 얻는 별도의
서열이 아니다. 남은 간극 0.067은 추정 오차가 아니라 두 참값이 묻는 질문의 차이(realized
귀속 대 counterfactual 재학습)이며, IID-clean이라 in-run GT 타깃 자체가 seed-불안정한(부록
D: −0.37) 무대 특성의 산물이다 — 실재-신호 무대(`Silo`)에서 같은 일치도가 1.000으로 닫히는
것(표 F4)과 대조된다.

**C.7 `Silo` 도메인 비IID 전 방법 vs in-run GT ($2^5$)** (1B, answer-swap·free-rider(zero)·clean × 3-seed) ⬚
<!-- 채움: runs/phase2_matrix 1B_silo5 rundir — 3-seed 실측 완료(●), 값 전사만 남음(frrand 행 미수록).
     의의: N=5 무대 중 유일하게 in-run GT 타깃이 seed를 넘어 재현되는(부록 D.2) 실재-신호 무대의 전표 -->

| method | clean | answer-swap | free-rider(zero) |
|---|---|---|---|
| Flirds | ⬚ | ⬚ | ⬚ |
| Flirds-1st | ⬚ | ⬚ | ⬚ |
| individual utility | ⬚ | ⬚ | ⬚ |
| GTG | ⬚ | ⬚ | ⬚ |
| FedSV | ⬚ | ⬚ | ⬚ |
| ComFedSV | ⬚ | ⬚ | ⬚ |
| ShapleyFL | ⬚ | ⬚ | ⬚ |
| FedIF | ⬚ | ⬚ | ⬚ |

**C.8 `LLM-Device` 대규모 교차-디바이스 vs in-run GT (per-round $2^{10}$)** (answer-swap·free-rider(zero); $N{=}100$·10/100, $\alpha{=}0.5$, 1B, 3-seed) ⬚
<!-- 채움: phase2_matrix device100 anchor 셀 — 3-seed 실측 완료(●), 값 전사만 남음(frrand 행 미수록).
     §5.4 비용 실측과 같은 rundir: 규모 무대에서의 fidelity·비용 동시 근거.
     ⚠ 이 무대엔 clean 셀이 없다(전 α 미실측) — §5.1 clean-앵커 규칙의 유일한 예외; 채움 여부
     결정 계류(플랜 §4.3: (b) per-round 셀당 ~6.9h → clean × 3seed ≈ 21 GPU-h). -->

**C.9 학습-강도 lever는 fidelity를 흔들지 않는다.** LoRA rank {16,32,64}(용량), lr
{1e-3,2e-3,3e-3} × local steps {10,20,30}(강도), CNN 폭 {0.5,1,2,4}(8×)를 각각 sweep해도:
Flirds vs in-run GT는 **전 칸 1.000**(lr·steps로 per-round 이동이 커져도 라운드당 1 HVP가
in-run GT를 정확 재현 — Taylor 절단의 실무적 트레이드오프가 이 범위에선 관측되지 않는다),
클라이언트 간 φ 분리도 lever로는 거의 변하지 않는다(CNN 폭 8×에 φ range 평평;
오염(label-flip)이 iid의 2–4×를 만든다). CNN 레그는 3-seed ●(기준칸 폭 1·참여 1.0은 구 C1
rundir 재사용 — provenance 각주), LLM 레그는 핵심축만 3-seed이고 나머지는 보강 ⬚
<!-- G12: lr{1e-3}st{20,30}·lr{2,3}e-3 st{20,30}·anchor r{32,64}·std50k5 r{32,64}·noise r64
     seed 보강 23런 — 핵심 미확인 질문 "lr로 커진 φ가 cross-seed 실재 신호인가"는 lr{2,3}e-3
     계열 seed{1,2}가 판정 -->.

### 부록 D. 안정성(재현성)

**D.1 방법 순위의 seed 간 안정성(CNN).** 같은 방법의 φ 순위를 seed 간
상관($\rho_{xseed}$, 3-seed 쌍별 평균)으로 잰다 ⬚:
<!-- 채움: CNN-Small(G2) 착지 후 신-무대 풀로 재산출. 구 시나리오 격자 풀 값(Flirds 0.547 =
     in-run GT 자체 0.518 추종, MC 재구성 계열 0.12~0.31 하락)은 무대 폐기로 미수록 —
     같은 구조의 재현이 사전 기대. -->

| method | $\rho_{xseed}$ ↑ |
|---|---|
| in-run GT (자체) | ⬚ |
| Flirds | ⬚ |
| Flirds-1st | ⬚ |
| individual utility | ⬚ |
| GTG | ⬚ |
| FedSV | ⬚ |
| ComFedSV | ⬚ |
| ShapleyFL | ⬚ |
| FedIF | ⬚ |

**D.2 in-run GT 타깃 자기-안정성(수록 무대).** fidelity의 매칭 대상인 in-run GT 자신이 seed를 넘어
재현되는가(in-run GT φ 순위의 seed 간 쌍별 Spearman 평균):

| 무대 | cross-seed $\rho$ ↑ |
|---|---|
| `Anchor` (IID-clean) | **−0.367** |
| `Silo` clean (비IID) | **+0.867** |
| `Silo` answer-swap | +0.933 |
| `Silo` free-rider(zero) | +0.933 |
| `LLM-Main` | ⬚ <!-- 채움: G1 rundir in-run GT φ 피벗 --> |
| `CNN-Main` | ⬚ <!-- 채움: c2fid rundir in-run GT φ 피벗(⟐ 파생, 실행 0) --> |
| `CNN-Small` | ⬚ <!-- 채움: G2 rundir --> |

판정: IID-clean 무대의 in-run GT 타깃은 seed-불안정하다(−0.37) — 그 위의 per-seed fidelity
1.000은 "불안정한 참값을 그때그때 정확히 좇는 것"이다. 반면 비IID·오염 무대에서는 타깃이
안정하고(+0.87~+0.93) 그곳의 fidelity가 실재 신호에 대한 재현이다. 본 논문의 fidelity 표는
이 구분과 함께 읽어야 하며(§5.2 각주), 이것이 retrain GT 특성화의 주 표를 실재-신호·주무대-정합
무대(`Silo`·`CNN-Small`)에 두고 Anchor를 참조로만 쓰는 이유다.

### 부록 E. 개입 확장

**E.1 CNN sign-gating 경쟁의 MNIST 짝.** §5.3 표 [I2]와 동일 무대·정책·점수원 8종
(online·retrain, {Dirichlet(α=1), iid})을 데이터셋만 MNIST로 교체해 개입 실효성의 데이터셋
강건성을 확인한다 ⬚.
<!-- G10: 관측자 24 + 점수원 192 = 216 rundir(G8 러너 확장 공유). 크기-가중(P1w) arm은 같은
     rundir에서 동반 산출(E.2). -->

**E.2 크기-가중 게이팅(P1w).** sign-gating과 같은 배제에 더해, 남는 클라이언트를 양수 누적
기여도 크기에 비례해 가중($w_k \propto n_k \max(\hat\phi_k, 0)$, 합-1 재정규화)하는 변형이다.
cifar10/Dirichlet(α=1)은 실측 완료 ●, 나머지 무대는 §5.3의 점수원 확장 rundir에서 동반
산출된다(추가 런 0) ⬚.
<!-- 채움: track_h rundir의 gatew_v2/t2_signw arm — cifar10/dir1 ● 값 전사; iid = G3 동반,
     mnist = G10 동반. 수록 규칙(사전 고정 — workplan 00-INDEX §1): CNN·LLM 전 범위에서
     sign-gating 상회 시 본문 승격 / 동률 시 "부호가 가치의 대부분" 1문장 / 열세·타 점수원
     역전 시 본 표까지만. -->

**E.3 clean-IID 무해성(do-no-harm) parity.** `LLM-Scale`(1B·3B·7B, clean-IID)에서 φ-기반
개입이 성능을 깎지 않는지 — vanilla / Flirds-가중 / Flirds-선택 3 arm의 MMLU·Alpaca-test
ROUGE-L parity ⬚.
<!-- 채움: runs/track_d arms — 3-seed 실측 완료(●), 값 전사만 남음. ShapleyFL-가중·FedIF-가중
     행은 스코프 재편으로 미수록(3행 표). clean 무대 기대 = 개입이 해를 끼치지 않음(parity). -->

**E.4 φ 부호 감사 — sign-gating의 작동 전제.** 게이트 문턱 $\tau{=}0$이 성립하려면 clean
클라이언트의 누적 기여도가 양수여야 하고(오배제 0), 오염 클라이언트가 음수/0이어야
한다(회수). 전 수록 rundir에 대한 부호 전수 감사 ⬚:
<!-- 채움: runs/track_g/audit 파생(⟐ 실행 0) — 표 A: clean 클라 누적 φ≤0 비율(점수원별;
     오배제 위험) / 표 B: 오염 클라 누적 φ≤0 비율 + free-rider(zero) exact-0 병기.
     ⚠ CNN 레그(frzero·gn 무대)는 현 감사 스냅샷 미커버 — G2·G8 착지 후 재감사. -->

### 부록 F. 비용 상세

**F.1 LLM 보조 무대 실측 — 연산수 모델의 검증 라인.** §5.4 표 [O1]의 per-op
곱(microbench)이 무대별 실측 wall-clock을 재현한다(무대마다 검증셋 크기·시퀀스 길이가 달라
per-op 절대 단가는 다르지만, 연산수 비가 wall-clock 비를 지배한다):

| 무대 | Flirds | Flirds-1st | individual utility | in-run GT |
|---|---|---|---|---|
| `Silo` ($K{=}5$·$R{=}10$) | 10 HVP → 예측 104s / 실측 ~107s | 10 grad (~35s) | 60 fwd → 96s / 96.6–100.1s | 320 fwd → 512s / ~530s |
| `Anchor` ($K{=}5$·$R{=}30$) | 30 HVP → 실측 707±16s | 30 grad (231±5s) | 180 fwd (657±19s) | 960 fwd (3,528±83s) |
| `LLM-Scale` ($K{=}2$·$R{=}200$) | ⬚ (**in-run GT의 1.61×** — 소-cohort 역전) | ⬚ | ⬚ | ⬚ |

cross-game 5종의 무대별 wall-clock 전량(같은 분석 파일) ⬚:

| method | `Silo` | `Anchor` | `LLM-Scale` |
|---|---|---|---|
| GTG | ⬚ | ⬚ | ⬚ |
| FedSV | ⬚ | ⬚ | ⬚ |
| ComFedSV | ⬚ | ⬚ | ⬚ |
| ShapleyFL | ⬚ | ⬚ | ⬚ |
| FedIF | ⬚ | ⬚ | ⬚ |

<!-- 채움: 두 표 모두 3-seed ●, 1B 기준 — anchor5·std20의 3B·7B 실측(● 3-seed)도 동일 소스에서
     값 전사 시 병기. LLM-Scale 행 = runs/track_d std20 timing(individual utility는 C6 교정본
     runtime만 인용 — in_run_singletons 캐시 fix 후 재측정값). Fed-LOO·Banzhaf·탐지기 행 미수록. -->

$K{=}2$ 행이 표 [O1]의 소-cohort 역전 실증이다: $2^2{=}4$ forward < 1 HVP ≈ 6.5 forward.
retrain GT의 가격표(둘 다 $2^5$개 부분집합 전체 재학습): `Silo`($R{=}10$)에서 31,137s —
Flirds(~107s)의 292×, in-run GT(~530s)의 58.6×; `Anchor`($R{=}30$)에서 30,817±244s —
in-run GT(3,528±83s)의 ~9배.

**F.2 CNN 방법별 실측.** `CNN-Main`(cifar10/Dirichlet(α=1), 오염 3위협 평균, $n{=}9$)의
방법별 valuation wall-clock(초):

| method | wall-clock (s) |
|---|---|
| Flirds-1st | 4.21±0.10 |
| FedIF | 5.35±0.17 |
| individual utility | 9.22±0.12 |
| **Flirds** | **10.64±0.37** |
| ComFedSV | 23.65±0.58 |
| FedSV | 293.98±4.90 |
| in-run GT | 836.58±14.07 |
| GTG | 1,079.92±131.64 |
| ShapleyFL | 1,468.47±16.34 |

<!-- 출처: runs/track_c/c2fid/analysis/fidelity.csv runtime_s 열 -->

CNN에서 valuation은 학습 자체보다 자릿수로 싸다. 참조 가격표: `CNN-Small`($N{=}10$)의
retrain GT는 $2^{10}$ 전체 재학습이라 valuation과 자릿수 4~5개 차이가 난다 ⬚
<!-- 채움: G2 rundir의 t_a(재학습 총시간)·traj_time — 구 무대 실측(cifar10 t_a 32,912s =
     Flirds의 ~28,000×)은 G2 착지 후 신-무대 값으로 교체 -->
— retrain GT 무대를 $N{=}10$에 묶어둘 수밖에 없는 이유의 가격표다.

### 부록 G. 탐지

탐지를 핵심 질문 위계의 마지막에 두고 부록으로 보내는 이유는 기여도와 탐지가 직결이 아니기
때문이다: 검증 손실을 실제로 낮추는 오염(예: 직전 라운드의 글로벌 업데이트를 재제출하는
free-rider 변형)에 대해 val-loss 게임의 정직한 답은 "기여함"이고, 이는 추정 실패가 아니라
in-run GT 자신의 답이기도 하다(§6). 따라서 여기서의 주장 형식은 절대 AUROC가 아니라
**in-run GT 일치**다 — 추정기가 게임의 답과 같은 답을 주는가(사전 등록 기준:
$|\mathrm{AUROC}(\text{Flirds}) - \mathrm{AUROC}(\phi^{\mathrm{in}})| \le 0.05$). φ-파생 탐지는
기여도 순위 하위 = 의심 규약의 AUROC를 쓴다(부록 B). 표는 본문·부록에 이미 등장한 무대의
rundir에서만 뽑는다(새 무대 없음).

**G.1 LLM 주무대(`LLM-Main`) AUROC** ⬚
<!-- 채움: G1 rundir metrics.json auroc — in-run GT 일치 기준 대조 포함 -->

| method | answer-swap | free-rider |
|---|---|---|
| in-run GT (게임의 답) | ⬚ | ⬚ |
| Flirds | ⬚ | ⬚ |
| Flirds-1st | ⬚ | ⬚ |
| individual utility | ⬚ | ⬚ |

**G.2 CNN 주무대(`CNN-Main`) φ-AUROC** (오염 셀, 위협별 풀) ⬚
<!-- 채움: c2fid analysis auroc 열(cifar10 {dir1,iid} 재풀링; ● 실측 — free-rider 셀 exact-0
     계열 1.0 vs renorm 붕괴 0.00~0.29, gradient-noise 셀 Flirds-1st 실명 0.49, Flirds는
     in-run GT-동행 Δ≤0.05). MNIST 짝 = G8 동반 ⬚. -->

free-rider(zero) 셀은 0-의미론이 그대로 탐지 성능으로 드러나는 칸이다: 고정-가중 게임을
겨냥하는 계열은 명제 2에 의해 해당 클라이언트에 정확히 0을 주어 분리가 자명한 반면,
재정규화 게임을 겨냥하는 계열은 0 아닌 몫을 배분해 순위가 흐려진다(같은 구조가 개입
성능으로 나타나는 것이 §5.3의 판정 재료다). gradient-noise 셀은 §5.5-①의 2차항 논지가
탐지 축에서 재현되는 칸이다.

**G.3 `LLM-Device` $\alpha$-스윕 탐지 — "기여도 ≠ 탐지"의 정직한 근거.** 비IID 강도
$\alpha$를 스윕하며 φ-AUROC를 잰다 ⬚
<!-- 채움: phase2_matrix device100 rundir(3-seed ●) — free-rider(zero)는 전 α에서 1.000
     (배경 무관), answer-swap은 α 비단조 0.57~0.77이며 in-run GT 자신도 0.604 -->.
free-rider(zero)는 배경 분포와 무관하게 전 $\alpha$에서 완전 분리되는 반면, answer-swap의
AUROC는 $\alpha$에 비단조적이고 **in-run GT 자신조차 낮다** — 탐지가 약한 곳에서 추정기는
게임의 답을 그대로 좇고 있으며(in-run GT 일치), 낮은 절대 AUROC는 방법의 실패가 아니라
val-loss 게임이 탐지기가 아니라는 사실의 정직한 표현이다(§6).