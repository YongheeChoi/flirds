# [한글 검토판] Measure First: Federated LLM 파인튜닝에서 클라이언트-수준 데이터 가치평가의 Exact-Oracle Fidelity

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
Flirds는 In-Run Data Shapley(Data Shapley in One Training Run)의 폐형(closed-form) 계산을
FL의 라운드 구조로 확장해, 서버가 이미 수신하는 업데이트만으로 조합 재평가 없이 라운드별
Shapley 기여도를 계산한다.
목표인 라운드별 Shapley 값은 명시적으로 정의된 라운드 게임의 Shapley 값이어서, 공정한
분배의 공리를 그 게임 안에서 그대로 만족한다.
Flirds는 이 목표값의 근사이지만 유일한 오차인 Taylor 절단이 엄밀하게 bound되고, 같은
목표값을 전수 열거로 근사 없이 직접 계산한 참값(exact in-run Shapley)의 순위를 1B–7B LLM
전 스케일에서 사실상 완벽히 재현한다(Spearman $\rho \ge 0.999$).
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

**왜 retrain 기반 Shapley 값이 아닌가.** 데이터 가치평가의 고전적 참값은 retrain 기반
Shapley 값, 곧 클라이언트 부분집합별로 학습을 처음부터 다시 수행해 얻는 utility에 대한
Shapley 값이다(§4.1). 그러나 FL에서는 매 라운드 서로 다른 클라이언트가 간헐적으로 참여하고,
그때마다 공정한 기여도 평가가 요구된다. 이런 환경에서 부분집합별 재학습을 전제하는 위
정의는 실질적으로 계산 불가능하다. 그래서 기존
연합 Shapley 연구들은 대부분 평가 대상을 라운드 단위로 옮겨 왔다: 매 라운드 서버가 수신한
업데이트들로 정의되는 라운드별 Shapley 값을 계산하고, 이를 라운드에 걸쳐 누적한다. 본
논문도 평가 대상을 같은 라운드 단위에 두며, 그 목표값이 §4에서 정의할 **exact in-run
Shapley**다.

**남은 두 한계: 지수적 계산의 우회, 그리고 검증되지 않은 대체 정의.** 이 라운드 단위
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
서버 집계 전략만 바꿔도 수십 %씩 출렁인다는 보고는 이 상태의 실무적 얼굴이다. 반대편에서
LLM 규모의 데이터 기여도 평가(In-Run Data Shapley(IRDS), DataInf, LESS 등)는 활발히 발전했지만 전부 데이터에 직접
접근할 수 있는 중앙집중 세팅·샘플 단위의 방법이고, 이는 위에서 기술한 FL의 제약 안에서 이루어질 수 없는 방법들이다.

우리의 접근이 겨냥하는 것이 정확히 이 두 한계다. In-Run Data Shapley가 중앙집중 학습에
대해 제안한 폐형(closed-form) Shapley 계산을 연합학습의
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
닫힌 식으로 계산한다. 즉 Flirds 역시 참값 자체가 아니라 그 근사를 계산하는 추정기다. 다만
근사의 원천이 Taylor 절단 하나뿐이고 그 오차가 표본 추출 없이 결정론적으로 bound된다는
점에서, 무작위 샘플링의 분산과 검증되지 않은 대체 정의가 겹겹이 끼는 선행 근사들과
다르다(§4). 그 결과 라운드당 필요한 무거운 계산은 검증셋 위의 Hessian-vector product 한
번으로 고정되고, 참여자당 추가 비용은 내적 하나가 전부다: 재학습도, 부분집합별 모델
재구성·재평가도, 추가 통신이나 클라이언트-측 연산도 필요 없다. 내적은 업데이트가 놓인
파라미터 공간의 차원에만 비례하므로, 전체 모델을 학습하는 소형 네트워크부터 PEFT(LoRA)
어댑터만 교환하는 LLM 파인튜닝까지 같은 식이 그대로 적용된다.

**검증.** 선행 연구들은 기여도 점수의 타당성을 주로 다운스트림 효과로 확인해 왔다:
기여도-선택 학습이 수렴이나 성능 향상을 돕는지, 저기여 클라이언트가 실제로 오염돼
있었는지를 보는 식이다.
우리는 검증을 두 겹으로 나눈다. **추정기 층**에서는 Flirds가 겨냥하는 게임의 exact Shapley 값, 곧
in-run 정의를 근사 없이 라운드당 $2^{|P_r|}$ 전수 열거로 계산한 **exact in-run Shapley**를
같은 학습 궤적 위에서 직접 구해, 추정값이 이를 순위·값 수준에서 재현하는지 잰다. 근사
참조나 다운스트림 간접 증거에 의존해 온 선행 관행보다 강한 기준이다. **실효성 층**에서는 측정한 기여도가 실제로 학습에
도움이 되는 값인지를 잰다. 각 방법이 매긴 기여도를 기반으로 학습 과정을 개선하고, 그
학습 성과로 방법들을 비교한다: 학습 도중 순기여 $\le 0$인 클라이언트를
자동 배제하는 온라인 부호-게이팅, 기여도 순서대로 클라이언트를 제거하고 실제로
재학습해 순위의 인과적 타당성을 확인하는 removal 실험, 기여도 상위 클라이언트만 참여시켜
재학습하는 selection 실험, 그리고 오염·무임승차 클라이언트 탐지다. 여기에 방법별 계산 비용을
같은 고정 궤적 위에서 실측한다. 고전적 exact retrain Shapley와의 실증적 관계는 특성화 실험으로만 보고한다(§5.3).

**기여.** 이 논문의 기여는 다음과 같다.

1. **방법.** In-Run Data Shapley의 폐형 계산을 연합학습의 라운드 구조로 확장한
   클라이언트-수준 기여도 추정기 Flirds를 제안한다. 연합 클라이언트-수준 기여도 평가를 LLM
   규모(PEFT 파인튜닝)에서 수행한 것은 우리가 아는 한 이 논문이 처음이다.
2. **검증 프로토콜.** 추정 정확도는 전수 열거로 계산한 exact in-run Shapley 대비로, 기여도의
   실효성은 같은 개입 정책 아래에서의 실제 학습 개선으로 채점한다. exact 참값 대비 채점을
   LLM 규모까지 끌어올린 것은 우리가 아는 한 처음이다.
3. **수학적 정당성.** Flirds가 계산하는 값이 라운드별 집계 게임의 2차 Taylor 근사가 갖는
   정확한 Shapley 값임을 증명하고, 유일한 오차인 Taylor 절단을 bound한다. 공정성 공리가
   라운드 게임 수준에서 성립해 무기여 클라이언트는 대수적으로 정확히 0을 받으며, 이 성질들은
   라운드 단위의 온라인 정산과 결합되어 측정한 기여도를 실현된 런에 대한 보상 분배의
   기준으로 그대로 소비할 수 있게 한다.

---

## 2. 관련 연구

**연합학습에서의 Shapley 기반 기여도 평가.** 연합 Shapley 값의 계보는 클라이언트 기여도를
라운드 단위의 Shapley 값으로 분해하고 permutation Monte Carlo 표본 추출로 추정하는
FedSV에서 시작한다. 이후의 흐름은 지수적 비용을 낮추는 근사의 연쇄다. GTG-Shapley는
부분집합 모델 재구성과 유도-절단 Monte Carlo를, ComFedSV는 부분 참여로 관측되지 않는
조합을 메우는 utility 행렬의 low-rank 완성을, ShapleyFL은 정규화와 이동평균으로 가공한
라운드별 대체(surrogate) 값을 쓰고, 이후의 FedIF·FedTSV·ShapFed·S-FedAvg 계열은 Shapley
공리를 완화하거나 포기한 채 기여도 신호를 강건-집계 가중치로 소비하는 쪽으로 이동한다. 이
계보에서 §1의 두 한계를 그대로 관찰한다. 첫째는 **계산의 우회 그 자체**다: 위 방법 전부가
목표 Shapley 값의 exact 계산 대신 무작위 표본 추출·절단·보간에 의존하며, 검증도 소형 CNN
규모에 머물러 왔다. 둘째는 **우회가 낳은 대체 정의와 검증 공백**이다: 비용을 낮추는
과정에서 같은 이름 아래 서로 다른 값(재정규화 게임, 보간된 손실 행렬, 가공된 surrogate)이
추정되고 있으나, 그 값의 타당성은 대부분 다운스트림 결과로만 간접 확인되었고, exact 참값
대비 직접 채점은 SPACE의 $2^n$-재학습 비교($N \le 10$의 CNN 분류)처럼 소형 스케일에
국한된다.

**중앙집중 LLM-규모 attribution.** 중앙집중 학습에서는 개별 학습 예제 단위의 데이터
귀속·선별이 LLM 규모까지 활발히 발전해 왔으며, 크게 세 줄기다. 첫째, **influence function
계열**은 각 예제가 검증 손실에 미치는 영향을 gradient와 Hessian 역행렬($H^{-1}$)로 추정하며,
LLM 규모에서는 $H^{-1}$ 근사를 서로 다르게 처리한다(EK-FAC[Grosse et al., 52B], LoRA용 폐형
근사 DataInf, TRAK, LoGra 등). 둘째, 2024년의 **Hessian-free 흐름**은 $H^{-1}$을 아예
우회한다: LESS는 TracIn 계열의 궤적 influence를 LoRA gradient 사영으로 계산해 instruction
tuning 예제를 고르고, MATES·DsDm은 각각 증류한 소형 모델과 선형 datamodel로 사전학습
데이터를 고른다.
셋째, **In-Run Data Shapley**(§3.3)는 사후 $H^{-1}$ 대신 실제 학습 궤적을 따라 매 스텝의
Taylor 기여를 누적한다. 세 줄기 가운데 연합학습의 무대로 가장 자연스럽게
이어지는 것은 In-Run 계열인데, FedAvg 집계 $\sum_k p_k \Delta w_k$가 배치 gradient의
샘플-선형 분해와 같은 구조를 클라이언트 수준에서 이미 드러내기 때문이다. 본 논문은 이
관찰에서 출발해 IRDS의 폐형 계산을 연합 라운드 게임으로 확장한다(§4).

**탐지·강건 집계 baseline.** 연합학습에는 오염·악성 클라이언트에 대응하는 별도의 계열이
있다: 업데이트 통계로 이상 클라이언트를 찾아내는 **탐지**(FLDetector, FLTrust, STD-DAGMM,
FedDQC 등)와, 집계 단계에서 outlier 업데이트의 영향을 억제해 강건성을 확보하는 **강건
집계**(Krum, 좌표별 median, trimmed-mean 등)다. 이들은 이진 제거(keep/discard)나 신뢰
가중치를 산출할 뿐 부호 있는 연속 기여도와 분배 공리를 다루지 않으므로, 기여도 평가의
대체재가 아니다. 우리는 두 축을 분리하고, 위 전용 탐지기들과는 각자가 설계된 위협에서
비교한다(§5.6).

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
$w^t$에서 Taylor 전개하면 근사 게임의 Shapley 값이 닫힌 식으로 나온다. 검증 손실의
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

## 4. Flirds: 연합 in-run 게임과 그 폐형 추정

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
번으로 접는 합-형태 구조가 깨져 폐형 계산이 성립하지 않는다. **exact in-run Shapley**
$\phi^{\mathrm{in}}$은 각 라운드 게임의 Shapley
값을 라운드당 $2^{|P_r|}$개 부분집합 전수 열거로 근사 없이 계산해 합한 값이다:
$\phi_k^{\mathrm{in}} = \sum_{r: k \in P_r} \phi_k(u_r)$. IRDS의 efficiency가
telescoping과 결합해 그대로 성립한다:
$\sum_k \phi_k^{\mathrm{in}} = \sum_{r=0}^{R-1} u_r(P_r) = \ell_{\mathrm{val}}(w^0) - \ell_{\mathrm{val}}(w^R)$.

**용어 규약.** 이하 참값(oracle)은 위의 **exact retrain Shapley**(부분집합별 전수 재학습)와
방금 정의한 **exact in-run Shapley**(라운드별 전수 열거) 둘뿐이며, 항상 전체 이름으로
지칭하고 수식어 없는 "exact"는 쓰지 않는다. 두 참값 모두 값이 클수록 유익하다(검증 손실을
낮추면 양수).

### 4.2 폐형 추정기

exact in-run Shapley는 근사가 없지만 라운드당 $2^{|P_r|}$번의 검증 평가를 요구한다. Flirds의
출발점은, 라운드 utility를 $w^r$ 주변에서 2차까지 Taylor 전개한 근사 게임 $\hat u_r$의
Shapley 값이 닫힌 식으로 나온다는 것이다(명제 2). $g_r := \nabla \ell_{\mathrm{val}}(w^r)$,
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
제거한 변형을 **Flirds (first-order)** 라 부르고 상호작용 항의 ablation으로 사용한다.

### 4.3 이론: 추정기와 exact in-run Shapley는 같은 게임을 계산한다

이 절의 요지는 한 문장이다: Flirds는 임의의 점수가 아니라 식 (6)이 정의하는 바로 그 게임의
Shapley 값을 계산하며, exact in-run Shapley와의 유일한 차이는 Taylor 절단뿐이다. 형식적
서술·증명·가정·반례·수치 검증은 부록 A에 있다.

**명제 1 (라운드별 분해).** 고정 궤적 위에서 전체 게임 $\sum_r u_r$(각 $u_r$은
$u_r(S) := u_r(S \cap P_r)$로 $[N]$ 위 게임으로 확장해 합산)의 Shapley 값은 라운드별
$|P_r|$-인 게임의 Shapley 값의 합으로 정확히 분해된다(비참여 라운드의 기여 0은 정의가
아니라 정리다).

**명제 2 (폐형; free-rider 0; efficiency).** 라운드 utility의 2차 Taylor 근사 게임 $\hat u_r$에
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
실증 질문이고 §5.3에서 그렇게 보고한다). 게임은 FL 프로토콜이 서버에 노출하는 단위인
**클라이언트** granularity에서 정의되며, per-sample에서 per-client로의 형식적 연결은
mean-loss·단일 로컬 스텝에서 성립하고(부록 A) token-mean LLM 손실에는 그대로 이전되지
않는다. LLM 트랙에서 $w$·$\delta_k^r$·$g_r$·$H_r$은 전부 서버가 실제로 교환·평균하는
LoRA 인자 좌표의 양이다(CNN 트랙은 전체 파라미터를 학습하므로 구분이 없다; 좌표 의존성의
형식적 논의는 부록 A.10).

---

## 5. 실험

이 절은 세팅과 측정 프로토콜(§5.1)을 소개한 뒤, 추정 정확도를 신호 구조가 단순한 세팅부터
복잡한 세팅 순으로 제시하고(§5.2), 참값(exact retrain Shapley)·게임-무관 척도 대비
검증(§5.3), 기여 신호가 존재하는 조건의 지도(§5.4), 비용(§5.5)을 거쳐, 질문 위계의 두 번째
층위인 "측정한 기여도가 실제로 가치 있는 값인가"를 부호-게이팅을 메인 실험으로
검증한다(§5.6).

### 5.1 실험 세팅과 측정 프로토콜

**세팅.** 클라이언트 간 실제 차이(신호)를 담는 정도가 서로 다르도록 무대를 골랐다. §5.4에서
보듯 세팅의 구조가 측정 가능성 자체를 결정하기 때문이다. LLM 트랙은
Llama-3.2-1B/3B-Instruct·Llama-2-7B(LoRA)로 다섯 무대를 쓴다: **N=5 full-participation
IID**(exact retrain Shapley를 3-seed로 계산할 수 있는 유일한 무대), **N=20**(2/round)과
**N=50**(5/round)의 partial-participation IID, 클라이언트마다 도메인이 다른 **N=5
cross-silo non-IID**(의료·법률·금융·수리추론·일반지시문), 그리고 **N=100
cross-device**(10/round, Dirichlet 파티션)다. 오염 위협(noisy answer-swap,
zero/random-update free-rider)은 cross-silo와 cross-device에 주입하며, $N{=}5$ 세팅이
IID와 non-IID로 둘 있음에 주의한다. **CNN 대조 트랙**(MNIST/CIFAR-10, $N{=}10$ 전원
참여)은 두 exact 참값을 모두 $2^{10}$ 전수 계산으로 값싸게 얻는 고검정력 검증 무대이며,
개입 실험용 $N{=}100$ 구성을 둔다. 데이터 구성·라운드 수·참여율·하이퍼파라미터 전체는
부록 D에 있다.

**측정 규율.** 한 셀 안의 모든 방법은 같은 고정 궤적과 같은 손실 구현을 소비한다. seed는
헤드라인 트랙 전부 셀당 3개이며, 유일한 예외인 LLM $N{=}10$ 전수-열거 셀($2^{10}$ 열거가
seed당 32.7 GPU-시간)은 seed 0 단독임을 인용처마다 명시한다. 셀마다 설정·클라이언트별
$\phi$·방법별 wall-clock을 run 디렉토리로 영속화하며, 모든 표·그림은 여기서 재생성된다.
정밀도·optimizer·attention 구현 등 수치 세부는 부록 D에 있다.

**평가 지표.** 순위는 rank correlation(Spearman $\rho$, Kendall $\tau$), 값 수준은 Pearson
상관으로 잰다. 값 수준이 따로 필요한 이유는 근사-가법 세팅에서 순위 지표가 포화하기
때문이다(§5.2). 헤드라인 수치는 3-seed 평균이고 ±는 seed 간 모표준편차다. 모든 정확도 표에
exact in-run Shapley **자신**의 cross-seed 자기-일치도(**target self-stability**)를
병기한다. 정확도는 기준값이 안정적인 만큼만 의미가 있기 때문이다(§5.4).

**비교 방법 (기여도 평가 9종 + 탐지 4종).** 기여도 평가는 Flirds, Flirds (first-order),
GTG-Shapley, FedSV, ComFedSV, ShapleyFL, individual-utility baseline(단독 utility
$u(\{k\})$를 그대로 점수로 쓰며 가법성 진단 probe를 겸한다), Fed-LOO(brute force 대비 최대
차이 $0.0$으로 검증), FedIF다. 탐지는 FLDetector, FLTrust, STD-DAGMM, FedDQC이고 각자의 홈
위협에서 비교한다. 방법별 하이퍼파라미터는 부록 D에 있다. 다른 게임을 목표로 하는
방법들(GTG-Shapley, FedSV, ShapleyFL, ComFedSV,
FedIF)을 한 게임의 exact Shapley 값으로 채점하는 것은 공정하지 않으므로(§4.1의 게임 차이),
이들은 §5.2의 exact in-run Shapley 대비 정확도 표에서 제외하고 게임-무관 척도(removal
§5.3, 탐지 §5.6)·방법-중립 참값인 exact retrain Shapley 대비(§5.3)·비용(§5.5)에서 비교한다.

### 5.2 Exact in-run Shapley 대비 추정 정확도

**표 1**은 LLM 트랙에서 기여도 추정값이 exact in-run Shapley의 순위를 얼마나 재현하는지를
보여준다(Spearman $\rho$, 3-seed 평균). 비교군은 같은 게임을 겨냥하는 방법들로
한정한다(§5.1의 공정성 원칙). *셀 값의 범위는 1B/3B/7B 세 스케일에 걸친 것이며 스케일별 전체 표는 부록
C에 있다. Fed-LOO는 1B 경량 재실행(3-seed)이다.*

| 방법 | N=5 full (IID) | N=20 (2/round, IID) |
|---|---|---|
| **Flirds** | $1.000$ (전 스케일) | $1.000$/$1.000$/$0.999$ (1B/3B/7B) |
| **Flirds (first-order)** | $1.000$ (전 스케일) | $0.997$–$0.999$ |
| Individual utility | $1.000$ (전 스케일) | $0.999$–$1.000$ |
| Fed-LOO | $1.000$ (1B, 3-seed) | $1.000$ (1B, 3-seed) |

**포화 세팅: 헤드라인이 아니라 발견.** $N{=}5$ 전원 참여의 오염 없는 세팅에서는 세 방법
모두 exact in-run Shapley를 정확히 재현한다($\rho = 1.000$; 표 1 왼쪽). 우리는 이를 추정기
품질의 증거로만 읽지 않는다. 이 세팅은 게임 자체가 근사-가법적이어서(실측 비가법성이
utility의 $0.9\%$ 이하; 전 스케일) 어떤 합리적 지표든 순위가 같아지며, 상호작용을 전혀 보지
않는 individual utility의 일치가 바로 이 가법성의 가장 깨끗한 진단이다(additivity probe).
그래서 포화 결과는 방법의 성질이 아니라 세팅의 성질로 보고한다. 통계적 무게는 $N{=}10$이
진다: CNN 트랙($2^{10}$ 전수, 3-seed)과 LLM $N{=}10$ 전수 열거(seed 0)에서도 같은-게임 방법
전원이 순위 상관 $1.000$을 유지했다. 우연히 나올 확률이 $1/3{,}628{,}800$인 순위다.

**부분 참여, 그리고 2차 항이 제값을 하는 곳.** 부분 참여에서도 Flirds는 정확도를 유지한다:
N=20에서 $0.999$–$1.000$(표 1 오른쪽), 더 벼린 N=50 probe에서도 $+1.00$(3-seed). 중요한
것은 참여 비율이 아니라 **클라이언트당 참여 횟수**다. 참여 횟수가 매우 적은 CNN 트랙에서는
1차 근사가 흔들리고(label-flip, 라운드당 20% 참여에서 Flirds (first-order) $+0.305$), 2차
항을 가진 Flirds는 버틴다($+0.891$; 폭 4종 × 3-seed 풀 평균). 상호작용 항이 제값을 하는
지점이다.

**오염·non-IID 무대에서도 유지된다.** 오염 클라이언트를 주입한 무대 전부에서 Flirds는 exact
in-run Shapley를 그대로 재현한다: cross-silo non-IID의 noisy·free-rider 셀 전부와
오염축×분포축 매트릭스 8셀에서 $1.000$(3-seed), cross-device 위협 셀에서 $\ge 0.9999$. 즉
**추정 정확도는 배경 분포와 오염 유무에 무관하다**. 무대에 따라 갈리는 것은 추정이 아니라
신호의 존재 여부다(§5.4).

**CNN 대조 실험 ($N{=}10$, 3-seed).** 10개 시나리오를 합치면 exact in-run Shapley 대비
Flirds의 순위 상관은 $0.919\pm.134$, Flirds (first-order) $0.832$, individual utility
$0.860$이다. 안정성 구조도 보인다: exact in-run Shapley 자신의 cross-seed 자기-일치도가
$0.518$인데 Flirds는 $0.547$이다. 즉 **추정기는 기준값의 내재적 불안정성 위에 자기 분산을
더하지 않는다.** 반면 Monte Carlo 표본 추출 방법들은 표본 분산이 더해져 자기-재현성이
$0.124$–$0.311$로 떨어진다.

**값 수준의 정확도, 그리고 공리 준수.** 정산처럼 기여도의 **값** 자체를 소비하는 용도를
위해 값 수준도 잰다: 표준 세팅에서 Pearson $1.000$, cosine 거리 $10^{-4}$ 미만(1B/3B/7B
전반)이고, efficiency는 게이트 테스트에서 대수 항등식(값의 합 = 근사 게임의 전체 utility;
명제 2)으로 오차 0이며, 실현된 손실 감소 $\ell_{\mathrm{val}}(w^0) - \ell_{\mathrm{val}}(w^R)$와의 차이는 Taylor 잔차
수준이다(부록 A.11의 telescoping 잔차). 공리 준수는 상관계수가 가르지 못하는 것을 가른다: 아무 업데이트도 보내지 않은
free-rider에게 Flirds는 **정확히** $\hat\phi = 0$을 주지만(명제 2), 부분집합-재정규화 게임
방법들은 양의 raw 값을 지급한다(3-seed 평균 GTG $0.0037$, FedSV $0.0047$). 고정-가중
게임과 재정규화 게임이 다른 게임이라는 사실(§4.1)의 실무적 얼굴이다.

**Taylor 잔차 실측 (명제 3의 보완).** 명제 3의 bound는 잔차의 차수만 주므로 크기는 직접
잰다. 1B 실측($N{=}5$, 3-seed)에서 라운드별 잔차는 1차 평균 $\sim 2\times 10^{-6}$, 2차 평균
$\sim 6\times 10^{-7}$로 coalition utility($10^{-3}$–$10^{-2}$)보다 $10^3$–$10^4$배 작고, 1차
잔차의 스케일링 기울기 $1.8$–$2.7$은 이론 예측(기울기 2)과 정합한다(2차 잔차는 fp32 해상도
수준이라 하한만 관측). 라운드별 열거의 합과 직접 $2^N$ 열거의 구현-수준 일치(명제 1·2의
항등식 검증)는 부록 A.11에 있다.

### 5.3 참값 대비 검증: exact retrain Shapley, 그리고 게임-무관 척도

exact retrain Shapley를 3-seed로 계산할 수 있었던 1B N=5 full-participation 세팅에서 두
exact 참값의 순위 상관은 $\rho(\phi^{\mathrm{re}}, \phi^{\mathrm{in}}) = 0.933 \pm
0.047$이다(seed별 $0.90/1.00/0.90$; 2026-07 재실행에서 재현·정본화). exact in-run Shapley를
정확히 맞추는 방법들은 retrain 참값 대비로도 같은 $0.933$을 물려받고(천장 효과), FedSV는
$0.733$, FedIF는 $0.133$에 그친다. 한편 CNN 트랙에서는 두 참값이 **갈라진다**(10개 시나리오
집계에서 모든 방법이 retrain 대비 $0.45$ 이하). 1차 후보 원인은 retrain 참값 **자신의
재학습 노이즈 바닥**이다: exact retrain Shapley의 cross-seed 자기-일치도는 대부분의 셀에서
$-0.28$–$+0.68$에 머물고, 두 참값이 모두 자기-안정적인 유일한 셀(MNIST label-flip, 양쪽
$+0.97$)에서는 둘이 $0.96$으로 일치한다. 다만 retrain이 비교적 자기-안정적($+0.68$)인데도
어긋나는 셀(CIFAR-10 label-skew)이 남아, 경로-의존·가중 재정규화·counterfactual 궤적이라는
진짜 게임 차이도 후보로 남는다. 열린 특성화 질문으로 보고한다.

**정직한 범위 설정.** 위의 일치는 많은 게임 정의가 어차피 일치하는 레짐(근사-가법, 등크기,
전원 참여)에서 얻은 것이다(부록 A). 계산 파이프라인의 인증이지 게임 정의의 인증이 아니며,
판별력 있는 시험(게임들이 갈라지는 조건에서 exact retrain Shapley를 방법-중립 심판으로
세우는 game-adjudication)은 future work다(§6).

**게임-무관 검증 ①: removal-curve (LLM).** 각 방법의 기여도 순서대로 클라이언트를 하나씩
제거하고 남은 클라이언트만으로 실제 재학습해 검증 손실을 잰다. 순위가 인과적으로 옳다면
worst-first 제거는 성능을 개선하고 best-first 제거는 악화시켜야 한다. N=5 cross-silo 오염
세팅(3-seed)에서 Flirds의 순위는 정확히 그렇게 행동한다: worst-first는 손실을
개선하고(noisy $+0.0076$, random-update $+0.0071$, zero-update $+0.0067$), best-first는
악화시킨다($-0.0084$/$-0.0015$/$-0.0016$). 같은-게임 방법 전원은 **9개 셀(3 위협 × 3 seed)
전부에서** exact in-run Shapley와 제거 순서가 완전히 일치해 곡선까지 동일했고,
GTG-Shapley(8/9)·ShapleyFL(7/9)·FedSV(6/9)도 대부분 같은 곡선으로 수렴한다(이탈은 clean
클라이언트 간 중간 순서, 곡선 차 $\le 0.002$). 유일한 질적 예외는 FedIF로, 순위 오류가
zero-update free-rider의 worst-first 곡선을 눈에 띄게 얕게 만든다.

**게임-무관 검증 ②: CNN removal-curve와 accuracy 축.** 방법 간 순위가 실제로 갈리는 CNN
스테이지에서 반복하면 removal이 방법을 변별하기 시작하고, test 정확도 축의 인과 검증이
가능해진다(LLM은 생성 모델이라 검증 손실뿐). 오염 시나리오에서 Flirds의
worst/best-first 분리는 뚜렷하고(label-flip MNIST $+0.0035$·CIFAR-10 $+0.045$,
feature-noise CIFAR-10 $+0.039$) 세 경우 모두 exact in-run Shapley의 분리 폭과 일치한다.
순위가 부정확할수록 분리가 얕아지고(feature-noise에서 FedSV는 $+0.13$), clean IID
대조군에서는 분리가 예상대로 사라진다($\approx 0$). LLM에서 유일하게 낙오했던 FedIF가
CNN에서 가장 큰 분리를 보인 것은 removal 척도 자체의 스테이지 의존성을 보여 주는 관찰이다.

**게임-무관 검증 ③: dose–response.** 실험자가 직접 통제한 오염 강도도 게임-무관 참조축이다.
Flirds의 탐지 AUROC(3-seed)는 noisy에서 비율 $0.25$ 이상 $1.00\pm.00$으로 올라서는 문턱을
보이고(비율 $0.1$ 이하 $0.75$), free-rider에서는 업데이트 크기 배율 $0.25$–$4.0$배 전 구간
$1.00\pm.00$으로 크기와 무관하다. 오염 없는 대조 셀의 $0.83\pm.12$는 $N{=}5$ 거친 AUROC의
무신호 기준선이 0.5가 아님을 보여 주는 계측 참조점이다.

### 5.4 기여 신호는 어떤 조건에서 존재하는가

참값을 직접 계산하면 간접 검증으로는 물을 수 없는 질문을 물을 수 있다: **이 세팅에
클라이언트 순위 신호가 애초에 존재하는가?** exact in-run Shapley 자신의 cross-seed
자기-일치도, 곧 seed를 바꿔 다시 학습했을 때 순위가 재현되는 정도로 답한다. **표 2**가
세팅별 결과다($\rho$, 1B, 3 seeds; 0 근처면 순위가 사실상 추첨이다).

| 세팅 | clean | noisy | free-rider |
|---|---|---|---|
| IID | $+0.13$ | $+0.60$ | $+0.70$ |
| non-IID | $+0.87$ | $+0.93$ | $+1.00$ |

*표 2의 셀은 오염×분포 매트릭스 캠페인($N{=}5$ full, cross-silo와 같은 $R{=}10$ 구성; 부록
E.2)이고, free-rider 열은 zero-update다(random-update는 부록 E.2). $R{=}30$ 표준 그리드의
IID-clean 셀은 $-0.37$(부록 C3)로 값 자체는 다르나 — 구성 차이에 더해 $N{=}5$ 3-seed
cross-seed $\rho$는 표본 분산이 크다 — '0 근처 = 무신호' 판정은 같다.*

세 가지 귀결이 따른다. **첫째**, 오염 없는 IID 세팅에서는 참값 자신의 순위조차 seed 간에
재현되지 않는다(1B 표준 세팅들에서 $-0.37$–$-0.11$; 스케일을 바꿔도 불안정하고 양의 극단인
7B N=5 full이 명시된 예외다). 이런 세팅의 "순위 상관 $1.000$"은 추정기가 그 런의 실현된
무작위성까지 충실히 재현한다는 뜻이다. 실현된 런의 **정산**에는 정확히 올바른 행동이지만,
어떤 방법도 다운스트림 효용을 보일 수 없는 세팅이라는 뜻이기도 하다. **둘째**, 신호를
만드는 것은 클라이언트 간 실제 차이다. 오염이 전혀 없어도 도메인 이질성만으로 $+0.87$까지
올라간다. 반면 학습 강도는 신호를 만들지 못한다: 학습률을 키우면 $\phi$의 공통 크기만
커지고 cross-seed 신호는 생기지 않는다(3-seed; 부록 E). **셋째**, 오염 없는 IID 세팅에서는
기여도 기반의 어떤 개입도 "아무것도 바꾸지 않는 것(do no harm)"이 올바른 답이다. 우리의
selection 실험도 거기서는 vanilla와 구별되지 않았고, 이를 실패가 아니라 설계대로의 sanity
check로 보고한다.

**노이즈 원인의 분리.** 이 불안정성은 세팅의 성질(데이터 파티션·학습 궤적의 무작위성)이지
검증셋 표본 추출의 노이즈가 아니다. 고정된 런 안에서 검증셋을 bootstrap 재표집해도 $\phi$
순위는 유지된다: 재표집 자기-상관 $0.93$–$0.99$, 검증셋 반분할 $0.90$–$1.00$(상세는 부록
F). 정리하면 오염 없는 IID의 기여도 평가는 **정밀하지만 세팅-제한적**이다: 추정기는 검증
노이즈를 뚫고 실현된 런의 순서를 분해해 내며, 런을 바꿨을 때 재현되지 않는 것은 그 순서
자체다.

### 5.5 비용: 평가 단위(granularity), 그리고 참여자 수

비용은 서로 독립적인 두 축이 결정한다: **평가 단위**(샘플 대 클라이언트)와,
클라이언트-수준 방법들 사이에서 라운드마다 **coalition을 어떻게 처리하는가**다. **표 3**은
같은 고정 궤적 위에서 잰 방법별 기여도 계산의 wall-clock이다(초; 1B fp32, B200 GPU; 3-seed
평균, 가능한 곳은 ±std). individual-utility 런타임은 회계 버그($\approx$1.7×) 교정 후
재측정값이다.

| | N=5 non-IID ($R{=}10$) | N=5 full ($R{=}30$) | N=100 (10/round) |
|---|---|---|---|
| Flirds (first-order) | ~35 | $231\pm5$ | ~53 |
| **Flirds** | ~107 | $707\pm16$ | $157$ |
| Individual utility | ~99 | $657\pm15$ | — |
| ComFedSV | — | $2{,}557\pm215$ | — |
| FedSV | ~530 | $3{,}513$–$3{,}552$ | ~4,970 |
| GTG-Shapley | ~530 | $3{,}513$–$3{,}552$ | ~18,100 |
| ShapleyFL | ~530 | $3{,}513$–$3{,}552$ | ~24,900 |
| Exact in-run Shapley (전수 열거) | ~530 | $3{,}528\pm83$ | $24{,}975\pm911$ |
| Exact retrain Shapley (재학습) | — | $30{,}817\pm244$ | — |

**Coalition 축.** Flirds는 라운드당 검증-규모 계산이 HVP 한 번으로 고정되고 coalition
방법들은 $O(2^{|P_r|})$번의 검증 평가가 필요하므로, 우위는 라운드 참여자 수에 조건부다.
N=100 cross-device(10/round)에서 Flirds는 라운드별 exact in-run Shapley의 순위를 그대로
재현하면서($\rho = 1.000$, $\alpha{=}0.5$) $157$초, 전수 열거는 약 $25{,}000$초로 약
**160배** 차이다. $N{=}10$ 전원 참여($2^{10}$)에서도 $117{,}649$초 대 $733$초로 같은
160배다(seed 0). 반대로 라운드당 2개만 참여하는 N=20에서는 전수 열거가 라운드당 4번의
평가라 HVP보다 싸서 관계가 **역전된다**($2{,}917$초 vs $4{,}697$초; first-order는
$1{,}531$초로 여전히 최저). 그래서 비용 주장은 조건부로 서술한다: Flirds의 우위는 정확히
전수 열거가 불가능해지는 참여자 많은 라운드에서 결정적이며, 그곳이 애초에 추정을 필요로
하는 레짐이다.

**Granularity 축.** 평가 단위가 비용의 **차수**를 결정한다. 샘플-수준 점수는 모든 학습
예제의 per-sample 양을 만져야 하므로 학습 자체와 같은 $O(N)$이고, 클라이언트-수준 Flirds는
학습 데이터를 전혀 만지지 않으므로 $N$과 무관하다(라운드당 검증셋 HVP 1회 + $|P_r|$개
내적). 실측도 이 차수를 반영한다: LLM $N{=}100$ 셀에서 학습 $2{,}249$초 대비 Flirds
$157$초(약 7%), CNN에서 학습 $78$–$96$초 대비 $0.5$–$15$초다. 이 비대칭은 구조적이다:
클라이언트-수준 점수는 서버가 이미 보유한 $|P_r|$개 업데이트만 소비하지만, 샘플-수준 점수는
연합학습에서 클라이언트–서버 신뢰 경계 반대편에 있는 계산을 요구한다. 요컨대 클라이언트
granularity는 샘플 granularity의 열화판이 아니라 **연합학습에서 in-run 기여도 평가가 배포될
수 있는 단위**이며, 동의한 클라이언트가 자기 데이터 안에서 수행하는 per-sample 세분화는
직교하는 future work다(§6).

### 5.6 기여도의 가치: 부호-게이팅, selection, 탐지

질문 위계의 두 번째 층위로, 이 절은 측정된 기여도가 신호가 존재하는 곳에서(§5.4) 실제로
가치 있는 값인지를 검증한다. 메인 실험은 기여도를 배포 루프에서 직접 소비하는
**부호-게이팅**이다. selection과 기여도-가중 집계는 순위 전체를 소비하는 더 일반적인
개입이고, 탐지는 위계의 마지막 질문이다.

**부호-게이팅 (LLM; 온라인 게이트와 배제-재학습).** 학습 도중 기여도가 낮은 클라이언트를
자동 배제하는 온라인 게이트를, 오염 클라이언트를 알고 배제한 뒤 처음부터 재학습하는
**oracle-배제 재학습** 기준과 대조했다($N{=}5$ cross-silo non-IID·IID 두 무대, 3-seed).
**라운드 게이트**(그 라운드의 raw $\phi \le 0$인 업데이트만 해당 라운드 집계에서 제외)는
zero-update free-rider 셀에서 oracle-배제 성능의 $0.90$을 회수했고, **누적 게이트**(누적
$\phi \le 0$이면 참여 제외, 5라운드마다 복귀 재평가)는 **정확히 $1.000$을 회수**했다. 즉
oracle-배제 재학습과 최종 손실이 동일하며, free-rider의 $\hat\phi = 0$ 보장(명제 2)이
온라인 자동 배제로 그대로 이어진다. 오배제는 0건이고, 오염 없는 셀에서는 두 게이트 모두 한
번도 발화하지 않았다(do-no-harm). 스코프는 명확하다: 부호-게이트는 순기여 0 이하만 배제하는
도구이며, 기여가 양수이되 낮은 noisy 클라이언트는 selection(아래)과 탐지의 몫이다.

**Selection(선택 후 재학습).** 기여도 상위-$k$ 클라이언트만 참여시켜 처음부터 재학습한
성능을 무작위 선택·전체-클라이언트 학습과 비교한다. 오염을 주입한 소형 세팅($N{=}5$,
3-seed)에서 Flirds의 선택은 매 seed 오염 없는 집합을 정확히 복원했고, 최종 검증 손실에서
동률 이상으로 이겼다(두 학습률 모두). §5.3의 removal-curve(selection의 일반형)가 같은
결론을 재확인한다. 오염 없는 IID에서는 selection이 이득을 만들지 못하는데(§5.4), 신호
부재의 논리적 귀결이며 설계대로의 결과다.

**기여도-가중 집계 (CNN).** CNN cross-device($N{=}100$, 10% 참여, 3-seed)에서 기여도를
라운드별 **집계 가중치**로 소비하는 개입은 오염 무대에서 뚜렷한 이득을 낸다: label-flip
최종 test 정확도 vanilla 대비 약 $+0.09$(모델 폭 $0.5$–$4$배에서 $+0.087$–$+0.092$로 일정),
gradient-noise $0.499 \to 0.609$($+0.11$). 오염 없는 clean 대조군에서는 아무것도 바꾸지
않는다($|\Delta\mathrm{acc}| < 0.006$). §5.4의 do-no-harm 예측이 개입 축에서도 성립한다.

**탐지.** 위계의 마지막 질문이고 답은 위협과 배경에 따라 다르므로, 위협별로 exact in-run
Shapley 자신의 상한까지 명시해 보고한다. $N{=}5$ cross-silo non-IID에서 기여도 점수는 전용
탐지기 수준의 탐지력을 보인다: Flirds는 noisy·free-rider를 AUROC $1.000$(3-seed)으로
분리하는데, 전용 탐지기들은 각자 홈 그라운드에 구멍이 있다(FLDetector noisy $0.750$, FedDQC
free-rider $0.750$, STD-DAGMM $0.250$–$0.417$). 배경을 바꿔도 Flirds의 탐지는 $1.00$으로
배경-무관이고(FLTrust도 그렇다), FedDQC는 noisy에서 non-IID로 가면 침식되며($1.00 \to
0.92$) cross-device free-rider에서는 완전히 실패한다($0.14$–$0.57$; 기여도-기반 점수는
$1.000$). 정직한 반대면도 있다: cross-device 규모의 noisy에서는 FedDQC가 $1.0$으로
지배하고 모든 loss 기반 점수는 $0.60$–$0.76$에 머물며 **exact in-run Shapley 자신도
$0.604\pm.041$이다**. 깨끗한 검증 손실로 정의된 게임은 오염 탐지기가 아니며, 탐지는 기여도
평가의 부산물로서 평가되어야 한다. 보장의 경계도 스트레스 케이스로 실측했다: 직전 라운드의
글로벌 업데이트를 재활용해 보내는 **delta 방식 free-rider**는 잡히지 않는데(AUROC $0.33$),
exact in-run Shapley 자신도 매 seed 정확히 같은 값이다. 재활용 업데이트가 실제로 검증
손실을 낮추므로 게임이 기여로 계상하는 것이고, Flirds는 그 값을 여전히 $1.000$으로
재현한다(실패는 추정이 아니라 게임 수준이다). $\hat\phi = 0$ 보장은 zero/random-update에
한정되며, 이런 위장의 탐지는 업데이트-통계 탐지기의 몫이다(STD-DAGMM $1.00$).

---

## 6. 논의와 한계

**참값 직접 검증이 사주는 것.** 이 논문의 세 발견은 다운스트림 결과만으로는 보이지 않았을
것들이다. 첫째, 오염 없는 세팅에서 게임이 근사-가법적이라는 사실. coalition 표본 추출
추정기의 비용 전제를 무효화하고, 2차 항의 위치를 "가법적이지 **않은** 세팅을 위한, 1차
대비 약 3배 비용이지만 coalition 열거보다 훨씬 싼 보험"으로 정해 준다. 둘째, 오염 없는 IID
세팅에서는 참값 자체에 cross-seed 순위 신호가 없다는 사실. "다운스트림 이득이 없다"를
실패가 아니라 그 세팅의 올바른 답으로 재해석하게 한다. 셋째, 클라이언트당 참여 횟수가 적은
환경에서 1차 근사가 흔들리고 2차 항이 보완한다는 조건. 우리는 이 프로토콜(고정 공유 궤적,
두 exact 참값, 값 수준 지표, 셀별 영속 산출물)이 재사용 가능한 표준이 되리라
믿는다.

**궤적-특이성과 유스케이스.** exact in-run Shapley는 **실현된** 런을 평가한다. 신호 없는
세팅의 cross-seed 불안정성은, 그 런의 진전을 지불·감사 목적으로 귀속하는 **정산**형
용도에는 버그가 아니라 정의 그 자체다. 정산은 값 수준 정확도가 직접 소비되는 곳이기도
하다: Pearson $\approx 1.000$의 값 일치와 exact efficiency가 결합하면 기여도 비례 지불은
런이 달성한 것을 정확히 분배하고 free-rider는 증명 가능하게 배제된다(명제 2). 다만
균질하고 깨끗한 집단에서 단일 런의 값을 이식 가능한 평판으로 재사용하는 것은 경계해야
한다. 클라이언트가 진짜로 다른 세팅에서는 값이 훨씬 안정적이지만(표 2) 모든 세팅에서
균일하게 그렇지는 않으므로(일부 CNN 오염 셀은 불안정), 이식성은 가정하지 말고 세팅별로
검사해야 한다.

**알려진 한계 (숨기지 않고 명시한다).**

- **게임 정의 의존성.** 고정 가중 의미론은 폐형 계산이 강제하고 런 자신의 집계 방식과도
  일치하지만 여전히 하나의 **선택**이며, 부분집합-재정규화 게임도 옹호 가능한 대안이다.
  게임-무관 removal 검증(§5.3)이 의존성을 부분 완화하고, exact retrain Shapley를 심판으로
  세우는 game-adjudication은 future work다.
- **레짐 제약.** momentum 없는 plain SGD·상수 학습률·LoRA 좌표·eager attention.
  서버 momentum은 telescoping을, 클라이언트 Adam 계열은 Taylor 대응을 깬다. 다른
  optimizer로의 이식은 검증하지 않았다.
- **평가 단위.** 값은 클라이언트 단위다. 동의한 클라이언트가 자기 데이터 위에서 수행하는
  샘플-수준 세분화는 호환 가능한 future work다. 서버 측 게임은 $\Delta w_k$보다 세밀한 것을
  볼 수 없고, 보아서도 안 된다.
- **프라이버시.** 서버가 개별 업데이트를 봐야 하므로 secure aggregation과 비호환이다.
  클라이언트-수준 평가라는 문제 클래스 전체에 내재한 제약이다.
- **적대 시나리오의 범위.** $\hat\phi = 0$은 zero/random-update에 대해 성립하고, delta 방식
  위장은 게임 수준에서 잡히지 않는다(§5.6). 검증 손실을 역이용하는 표적 공격(예: backdoor
  poisoning)의 강건성 평가는 범위 밖이다.
- **참값의 커버리지.** exact 참값 대비 평가는 $N$이 작은 세팅에 제한된다(LLM $N{=}5$–$10$,
  CNN $N{=}10$; LLM $N{=}10$은 seed 0 단독). retrain 참값과의 일치는 1B에서 측정되었고
  CNN에서는 발산한다(열린 질문). 7B retrain 참값은 선행 LLM-규모 연구들과 마찬가지로 계산
  범위 밖이다.
- **이질성 편향.** FL-Shapley 계열의 maverick 과소평가 지적을 고치지 않고 측정해
  특성화하며, 비교한 어떤 방법도 하나의 부호 있는 값 안에서 "noisy"와 "분포는 다르지만
  유용함"을 분리하지 못한다.

---

## 7. 결론

우리는 연합 클라이언트 기여도 평가의 간접 정당화 루프를 직접 루프로 교체했다: 게임을
정의하고, exact Shapley 참값을 실현된 in-run 정의와 counterfactual retrain 정의로 두 번
근사 없이 계산하고, 모든 방법을 하나의 고정-궤적·멀티-seed 프로토콜 아래 그 대비로
측정했다. 그 기준 안에서, 라운드당 HVP 한 번으로 닫히는 폐형 추정기는 서버가 이미 받는
업데이트로부터 클라이언트를 평가하며, 1차 근사만으로는 순위가 흔들리는 세팅을 포함해 모델
스케일과 세팅 전반에서 exact in-run Shapley에 충실했다. 비용은 라운드 참여자 수에
지수적이지 않고(전수 열거는 $O(2^{|P_r|})$회) 학습 데이터 수와도 무관하다(샘플-수준 방법과
학습 자체는 $O(N)$). 그만큼 중요하게, 이 기준은 기여도 평가가 애초에 유의미한 조건의
지도를 그려 준다: 기여 신호는 학습 강도가 아니라 진짜 클라이언트 간 차이가 만들며,
근사-가법성은 오염 없는 소규모 세팅을 모든 방법에게 쉽게 만든다. 평가 프로토콜과 참값
계산을 포함한 전체 코드·실험 산출물은 게재 시점에 공개할 예정이다.

---

## 부록

### 부록 A. 증명과 형식 서술

*(전체 형식화의 출처는 내부 math-rigor dossier(2026-07-04, 반박 패널 36건 반영본)이며, 여기서는
논문 부록 분량으로 정리한다. 표기: (a) = exact retrain Shapley, (b) = exact in-run Shapley.
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
$u_r(S) := \ell_{\mathrm{val}}(w^r) - \ell_{\mathrm{val}}(w^r + \Delta_S^r)$, 전체 게임은 $U_b(S) := \sum_r u_r(S)$이며, (b)는
이 게임의 exact Shapley다. 라운드 surrogate는
$\hat u_r(S) := -\langle g_r, \Delta_S^r \rangle - \tfrac12 \langle \Delta_S^r, H_r \Delta_S^r \rangle$(1차 전용은
$\hat u_r^{(1)}$). **추정기가 근사하는 게임이 정확히 $U_b$다**. 같은 로그·같은 $\ell_{\mathrm{val}}$·같은
$S$-비의존 가중·같은 전개점이고, 유일한 차이는 라운드별 2차 Taylor 절단이다. (a)는 $S$만으로
처음부터 재학습한 게임으로, 분모가 $\sum_{j\in S} n_j$로 재정규화되고 궤적 전체가 $S$에
의존하는 **다른 게임**이다((a)↔(b) 관계는 이론이 아니라 실증 질문; A.8).

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

**A.5 명제 P2 (닫힌형).** $k \in P_r$에 대해 추정기는 본문 식 (7)과 같은 닫힌형
$$\phi_k(\hat u_r) = -p_k^r \langle g_r, \delta_k^r \rangle - \tfrac12\, p_k^r \langle
\delta_k^r,\, H_r \Delta W_r \rangle$$
을 갖는다. *증명 스케치 (unanimity 분해)*: $\hat u_r$을 가산부($-\langle g_r, a_k^r\rangle - \tfrac12
q_{kk}$)와 쌍대부($q_{ij} = \langle a_i^r, H_r a_j^r\rangle$)로 나누면, 2-인 unanimity 게임의 Shapley가
당사자 절반씩이므로 $\phi_k = -\langle g_r, a_k^r \rangle - \tfrac12 \sum_j q_{kj} = -\langle g_r,
a_k^r\rangle - \tfrac12 \langle a_k^r, H_r \textstyle\sum_j a_j^r\rangle$. $\square$ 우측 공통 벡터 $H_r \Delta W_r$이 전
클라이언트에 공유되므로 라운드당 **HVP 1회**로 닫힌다. **따름 P2-1 (free-rider exact-0)**:
$\delta_k^r = 0$이면 0-텐서와의 내적이라 추정기 값은 **대수적으로 정확히 0**(수치 결정성 무관);
(b) 게임 쪽 0은 forward의 bit-결정성이 전제다(CNN cudnn-deterministic 확립, LLM은 조건부).
**따름 P2-2 (efficiency)**: $\sum_k \phi_k(\hat u_r) = \hat u_r(P_r)$.

**A.6 명제 P3 (Taylor 잔차; 보조정리 L2).** $\ell_{\mathrm{val}} \in C^3$과 국소 상수 $M_2^r, M_3^r$ 하에
$|u_r(S) - \hat u_r(S)| \le \tfrac{M_3^r}{6}\|\Delta_S^r\|^3$, 1차는
$\tfrac{M_2^r}{2}\|\Delta_S^r\|^2$. **L2 (오차 전파)**: $|\phi_i(v) - \phi_i(v')| \le 2\|v -
v'\|_\infty$. 결합하면 $|\phi_k(U_b) - \hat\phi_k| \le \sum_r \tfrac{M_3^r}{3} \max_S
\|\Delta_S^r\|^3$ (1차 추정기는 선행계수가 $M_2^r$이며, $M_2/3$을 쓰면 무효 상계다). **지위**:
이는 상계이고 상수는 실측 불가하며 $R$에 선형 누적된다. 그래서 본문(명제 3)의 방침대로
잔차의 실제 크기는 실측으로 보완한다(§5.2). per-round 이동이 $K$-스텝 누적이라 IRDS per-step보다 구조적으로 약한 보증이라는
점도 명시한다.

**A.7 명제 P4 (per-sample→per-client 브리지; 보조정리 L3).** **(i) 고정 가중의 필수성**:
재정규화 이동 $\widetilde\Delta_S$는 coalition-독립 벡터족의 합으로 표현 불가(등$n$ 2-클라
반례: singleton이 $b_k = \delta_k$를 강제하나 pair가 $\tfrac12(\delta_1+\delta_2)$). 즉 고정
가중은 편의가 아니라 IRDS 닫힌형 기계 전체를 이식하기 위한 필수 가정이다. **(ii) L3
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
정확하지만(선형성), 전개점 $w^r$이 실현 궤적의 함수라는 의미의 경로-의존은 (b) 게임에
내재한다. $R{=}1$(후속 궤적이 없어 경로-의존이 정확히 0)에서도 위 ①의 반례로 (a)와 (b)의
순위가 반전되므로, (a)≈(b)는 궤적 안정성만으로는 성립하지 않고 등$n$/near-additive 조건이
추가로 필요하다. 궤적-특이 utility의 공리화는 IRDS가 명시한 열린 문제이며 본 논문에
승계된다(공리 성립 주장은 frozen 게임 한정).

**A.9 명제 P7 (momentum 스코핑).** 클라이언트가 stateless인 한 로컬 optimizer는 게임·닫힌형
정의에 영향이 없다(realized $\delta$만 소비). 진짜 load-bearing한 것은 **서버 무상태성**이다:
서버 momentum(FedAvgM)이 있으면 grand coalition 섭동이 실현 스텝과 달라져 G1(telescoping)이
깨지고, 게임이 "실현 run의 가산 분해"라는 의미론을 잃는다. 라운드 간 상태를 갖는 로컬
optimizer는 P4 브리지를 상실시킨다. 둘 다 가정으로 명시한다(§5.1, §6).

**A.10 명제 P8 (LoRA 좌표).** 게임·추정기·(b)가 전부 같은 LoRA 인자 좌표 $\theta$에서
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

**A.11 대수 검증 (GPT-2 스모크 + 1B 실측).** GPT-2($N{=}5$, $R{=}3$, fp32)에서: 닫힌형 =
추정기 최대 절대차 $5.8 \times 10^{-12}$; $\hat u^{(2)}$ 게임의 $2^5$ 전수 Shapley = 닫힌형
$7.6 \times 10^{-12}$; per-round 분해 = $2^N$ 전수 (b) $3.9 \times 10^{-7}$(fp32 forward
노이즈 바닥); telescoping 잔차 $\le 9.5 \times 10^{-7}$; Hessian 대칭성 $\le 1.8 \times
10^{-11}$. 1B 3-seed 물리 실측(잔차 크기·스케일링)은 본문 §5.2의 잔차 실측 문단과 같다.
### 부록 B. Removal·dose 실험 상세

**B.1 removal-curve 프로토콜.** 각 방법의 $\phi$로 클라이언트를 순위화한 뒤, **worst-first**는
$\phi$ 최하위부터, **best-first**는 최상위부터 1명씩 누적 제거하며, 각 kept 부분집합에 대해
**동일 init·동일 seed·동일 라운드 수의 clean FedAvg를 처음부터 전체 재학습**하고 배포 모델의
검증 손실(CNN은 test 정확도 병기)을 잰다. 곡선 점은 $[k, \ell_{\mathrm{val}}]$ ($k$ = 누적 제거 수; LLM
$k{=}0..4$, CNN $k{=}0..9$)이다. $U(\text{kept})$는 부분집합 키로 캐시되어 방법·방향 간
공유되며, 순위가 합의된 방법들은 같은 재학습 체인으로 붕괴한다(추가 비용 0). **caveat**:
재학습은 clean 재학습(위조 업데이트 미재현)이므로, noisy 클라이언트는 데이터 자체가 오염이라
제거 시 손실 개선이 기대되지만 free-rider는 데이터가 clean이라 제거가 $\approx$중립인 것이
정직한 기대치다(B.3에서 실측 확인). game-adjudication 설계는 future work로
이관되었다(설계·코드 존재).

**B.2 dose 격자와 결과.** noisy dose는 오염 비율 $nr \in \{0, 0.1, 0.25, 0.5, 0.75, 1.0\}$
(오염 부분집합 내 순환 재배정; $nr{=}1.0$은 전체 answer-swap과 비트동일, $nr{=}0$은 음성
대조), free-rider(random) dose는 위조 업데이트의 표준편차 배율 $dm \in \{0.25, 0.5, 1.0,
2.0, 4.0\}$ (benign 업데이트 std 대비; zero-update는 극값이라 dose 축 제외)이다. 결과(3-seed,
Flirds = exact in-run Shapley와 동일): noisy AUROC는 $nr \le 0.1$에서 $0.75$, $nr \ge 0.25$에서
$1.00\pm.00$ 포화로, 문턱은 $nr \in (0.1, 0.25]$이다. free-rider는 $dm$ 전 구간
$1.00\pm.00$으로 진폭에 불감하다. 음성 대조($nr{=}0$)는 $0.83\pm.12$로, "오염 라벨이 붙었지만 실제로는 clean"인
클라이언트가 non-IID 배경에서 도메인 편향으로 저평가될 수 있음을 보여주는 계측
참조점이다(FedIF·FLTrust는 이 셀에서 AUROC $1.0$인데, 검출력이 아니라 도메인 편향으로 읽어야
한다). FedDQC는 $nr$에 단조 반응($0.25 \to 0.92$)하나 저-dose에서 최약이고, free-rider는
데이터가 clean이라 원리적으로 잡지 못한다($0.75$ 고정).

**B.3 LLM silo5 removal (3-seed).** 위협별 분리 $\Delta(k) = \ell_{\text{best-first}} -
\ell_{\text{worst-first}}$ ($k{=}4$): noisy $+0.0161\pm.0015$, fr-random $+0.0086\pm.0005$,
fr-zero $+0.0083\pm.0007$. $k{=}1$(오염 클라 1명 제거)의 손실 변화: noisy $-0.0018$(즉시
개선), free-rider $-0.0002$($\approx$중립; B.1 caveat 그대로). Flirds·first-order는 **9/9
셀 전부에서 exact in-run Shapley와 제거 순서가 완전히 일치**했고, 나머지 방법의 이탈은 clean
클라이언트 간 중간 순서 차이(곡선 차 $\le 0.002$) 수준이다. 유일한 질적 예외는 FedIF(fr-zero
seed 0): 최고 가치로 지목한 클라이언트를 제거했더니 손실이 오히려 하락했다(가치-상위 순위
오류).

**B.4 오염 없는 대조.** IID-clean N=5 full(1B, $2^5$ 재학습 캐시 재사용, 3-seed)에서도
$\Delta(k{=}4) = +0.0042\pm.0012$의 작지만 방향-일관된 분리가 남는다. 실현된 런의 순위가
그 런 안에서는 유효하다는 §5.4의 정산 해석과 정합하며, 크기는 오염 무대의 $1/2$–$1/4$이다.

**B.5 CNN removal 셀별 요약 (seed-mean; $\Delta A$ = worst-first $-$ best-first test 정확도,
곡선 전 구간($k{=}0..9$) 평균 / $k$별 최대; 본문 §5.3와 같은 규약).**

| 데이터셋 | 시나리오 | $\Delta A$ mean | $\Delta A$ max | 비고 |
|---|---|---|---|---|
| MNIST | label-flip | $+0.0035$ | $+0.0107$ | worst-first가 $k{=}7$까지 손실을 baseline 아래로 |
| MNIST | feature-noise | $+0.0000$ | $+0.0017$ | 저강도 오염($\sigma \le 0.2$)이라 MNIST에선 거의 무해 |
| MNIST | iid (통제군) | $+0.0004$ | $+0.0056$ | $\approx 0$ (설계대로) |
| CIFAR-10 | label-flip | $+0.0449$ | $+0.1570$ | 정확도 축 분리 최대 |
| CIFAR-10 | feature-noise | $+0.0385$ | $+0.1145$ | 동일 ladder가 CIFAR에선 뚜렷 |
| CIFAR-10 | iid (통제군) | $-0.0027$ | $+0.0401$ | $\approx 0$; 큰 $k$의 재학습 분산 큼 |

오염 ladder는 클라이언트별 rate/$\sigma$ = $[0,0,.05,.05,.10,.10,.15,.15,.20,.20]$이다.
CIFAR-10은 큰 $k$에서 재학습 분산이 커서 손실 축이 출렁이므로 손실·정확도 두 축을
병기한다(본문 §5.3). Flirds는 18셀 중 7셀에서 exact in-run Shapley와 제거 순서가 완전 동일(오염
CIFAR-10 6셀 중 5셀 포함), 나머지는 인접 등급 스왑 수준이다. 정본:
`runs/removal_dose/rundirs{,_cnn,_trackd}/`.
### 부록 C. 전체 정확도 표

수치는 전부 rundir에서 재생성한 것이다(mean±std, 3-seed; ±는 seed 간 표준편차). 표 C1의
아래 4행(다른-게임 방법)은 §5.1의 공정성 원칙에 따라 **방법 품질 지표가 아니라 참조**다.
exact in-run Shapley는 그들의 목표 게임이 아니기 때문이다. 7B ShapleyFL은 $\beta$ 하이퍼파라미터의 통일
재실행이 완료되지 않아 값을 보고하지 않는다.

**표 C1. LLM: exact in-run Shapley 대비 Spearman $\rho$ (스케일 × 세팅).**

| 방법 | 1B N=5 | 1B N=20 | 3B N=5 | 3B N=20 | 7B N=5 | 7B N=20 |
|---|---|---|---|---|---|---|
| **Flirds** | $1.000\pm.000$ | $1.000\pm.000$ | $1.000\pm.000$ | $1.000\pm.000$ | $1.000\pm.000$ | $0.999\pm.001$ |
| Flirds (first-order) | $1.000\pm.000$ | $0.999\pm.001$ | $1.000\pm.000$ | $0.997\pm.002$ | $1.000\pm.000$ | $0.998\pm.001$ |
| Individual utility | $1.000\pm.000$ | $1.000\pm.000$ | $1.000\pm.000$ | $0.999\pm.001$ | $1.000\pm.000$ | $0.999\pm.001$ |
| Fed-LOO | $1.000\pm.000$ | $1.000\pm.000$ | — | — | — | — |
| *GTG-Shapley (참조)* | $1.000\pm.000$ | $0.975\pm.018$ | $0.967\pm.047$ | $0.990\pm.005$ | $1.000\pm.000$ | $0.977\pm.017$ |
| *FedSV (참조)* | $0.700\pm.163$ | $0.910\pm.073$ | $0.667\pm.205$ | $0.966\pm.006$ | $0.933\pm.047$ | $0.968\pm.010$ |
| *ComFedSV (참조)* | $0.500\pm.432$ | $0.093\pm.146$ | $0.600\pm.327$ | $-0.137\pm.065$ | $0.600\pm.216$ | $0.039\pm.171$ |
| *ShapleyFL (참조)* | $0.700\pm.283$ | $0.194\pm.351$ | $0.167\pm.094$ | $0.211\pm.158$ | — | — |

Kendall $\tau$는 같은 구조를 보인다(전 셀에서 Flirds $\ge 0.996$, first-order $\ge 0.982$,
individual utility $\ge 0.996$). 값 수준: Flirds의 Pearson은 전 셀에서 최소 $0.9999$ 이상이고
cosine 거리는 $3 \times 10^{-6}$ 이하다(§5.2). Fed-LOO는 1B 경량 재실행에서 측정했다(구현은
brute-force 대비 최대 차이 $0.0$으로 검증).

**표 C2. CNN: 시나리오별 exact in-run Shapley 대비 Spearman $\rho$ (같은-게임 3종, 3-seed).**

| 시나리오 | Flirds | Flirds (1st) | Indiv. utility |
|---|---|---|---|
| MNIST iid | $0.814\pm.135$ | $0.778\pm.090$ | $0.842\pm.059$ |
| MNIST label-flip | $0.996\pm.006$ | $0.992\pm.006$ | $0.992\pm.006$ |
| MNIST feature-noise | $0.786\pm.114$ | $0.697\pm.133$ | $0.782\pm.045$ |
| MNIST label-skew | $0.713\pm.212$ | $0.612\pm.266$ | $0.632\pm.262$ |
| MNIST quantity-skew | $0.964\pm.026$ | $0.980\pm.006$ | $0.960\pm.032$ |
| CIFAR-10 iid | $0.952\pm.010$ | $0.535\pm.160$ | $0.689\pm.114$ |
| CIFAR-10 label-flip | $0.996\pm.006$ | $0.952\pm.017$ | $0.947\pm.021$ |
| CIFAR-10 feature-noise | $0.996\pm.006$ | $0.895\pm.082$ | $0.903\pm.069$ |
| CIFAR-10 label-skew | $0.984\pm.006$ | $0.915\pm.026$ | $0.879\pm.069$ |
| CIFAR-10 quantity-skew | $0.992\pm.011$ | $0.960\pm.032$ | $0.976\pm.017$ |

10-시나리오 집계는 본문 §5.2(Flirds $0.919\pm.134$, first-order $0.832$, individual utility
$0.860$)이다.

**표 C3. 기준값 자기-일치도 (cross-seed Spearman; 정확도 표의 동반 열).**

| 셀 | (b) exact in-run | 비고 |
|---|---|---|
| 1B N=5 full | $-0.367$ | IID-clean; §5.4 |
| 1B N=20 | $-0.114$ | |
| 3B N=5 full | $+0.033$ | |
| 3B N=20 | $-0.243$ | |
| 7B N=5 full | $+0.733$ | 명시된 양의 극단 |
| 7B N=20 | $+0.164$ | |
| CNN 10-시나리오 평균 | $+0.518$ | Flirds $+0.547$; 추정기는 분산을 더하지 않음 |

CNN의 (a) exact retrain 자기-일치도(시나리오별 $-0.28$–$+0.97$)는
`runs/track_c/figures/a_oracle_xseed_stability.csv`에 정본화했다(§5.3의 발산 논의).
### 부록 D. 프로토콜 상세와 재현성

**D.1 세팅별 하이퍼파라미터.** 전 LLM 세팅 공통: plain SGD(momentum 0)·상수 lr(warmup 없음),
LoRA target = q/k/v/o/gate/up/down proj, dropout 0, completion-only loss, utility 평가 fp32.

| 세팅 | 모델 | LoRA $r/\alpha$ | lr | 로컬 스텝 | 배치 | seq len | $R$ | 참여 | seeds |
|---|---|---|---|---|---|---|---|---|---|
| N=5 full | 1B/3B/7B | 16/32 | $10^{-3}$ | 10 | 16/8/4 | 512 | 30 | 5/5 | 3 |
| N=20 | 1B/3B/7B | 16/32 | $10^{-3}$ | 10 | 16/8/4 | 512 | 200 | 2/20 | 3 |
| N=50 | 1B | 16/32 | $10^{-3}$ | 10 | 16 | 512 | 200 | 5/50 | 3 |
| N=5 cross-silo | 1B/3B | 16/32 | $10^{-3}$ | 10 | 16/8 | 768 | 10 | 5/5 | 3 (3B는 seed 0) |
| N=100 cross-device | 1B | 16/32 | $10^{-3}$ | 5 | 16 | 768 | 30 | 10/100 | 3 |
| CNN cross-silo | LeNet-5 / CNN(전체 파라미터) | — | $10^{-2}$ | 5 epochs | 64 | — | 10 | 10/10 | 3 |
| CNN cross-device | 동일 | — | $10^{-2}$ | 5 epochs | 64 | — | 120 | 10%/100 | 3 |

모델: Llama-3.2-1B/3B-Instruct, Llama-2-7B; LoRA $r{=}16$, $\alpha{=}32$. 검증 손실 평가는 청크 합산이라 메모리 knob이 값에
영향을 주지 않는다. ShapleyFL은 원 논문 값 $\beta{=}0.3$을 쓰되, 7B 셀은 $\beta$ 통일
재실행이 완료되지 않아 값을 보고하지 않고(부록 C1) robustness 셀은 $\beta{=}0.5$임을 해당
위치에 명시한다.

**D.2 데이터 레이어.** IID 무대는 alpaca-gpt4 52k 중 20k(OpenFedLLM 템플릿 verbatim; 셔플 후
val 200 → test 1,000 → train 20k 상호-disjoint carve, N개 균등 shard). 5-도메인 non-IID는
의료 플래시카드(medical-meadow) / 법률 QA / 금융 QA(FiQA) / 수리 추론(AQuA-RAT) / 일반
지시문(Dolly-15k)을 free-form instruction→response로 통일하고(형식 이질성이 공유 검증-손실
게임을 불공정하게 만들기 때문), 도메인당 train 크기를 동일하게 통제한다. 검증셋은 서버 측
held-out(도메인-층화; cross-silo 셀은 도메인당 20, cross-device는 도메인당 10이며, 소형
val이라 AUROC가 거친 격자임을 명시). cross-device는 도메인 풀을 합친 뒤 클라이언트별
Dirichlet($\alpha \in \{0, 0.01, 0.1, 0.5, 5\}$) 혼합으로 300예제/클라를 배정한다($\alpha{=}0$은 단일-도메인 one-hot).

**D.3 corruptor 정의(코드 기준).** noisy(answer-swap): 클라이언트 내부에서 completion 열만
무작위 순열(프롬프트 불변; 비율형은 오염 부분집합 내 순환 재배정, rate 1.0은 전체 순열과
비트동일). CNN noisy: label shuffle/graded label-flip(정답 제외 균등 재라벨),
feature-noise(픽셀 단위 Gaussian). free-rider: zero($\Delta w = 0$ 제출) /
random($\Delta w \sim U(-s,s)$, $s$는 clean warmup 라운드에서 잰 benign 업데이트 std에
매칭한 탐지-회피형 세팅) / delta(직전 라운드 글로벌 집계 재활용; §5.6의 스트레스 케이스).

**D.4 탐지 baseline 설정.** FLDetector: L-BFGS Hessian 예측과 실제 업데이트의 차 norm 점수,
연속 점수로 AUROC(클러스터링 생략), CPU. STD-DAGMM: 업데이트를 feature-hashing으로 256차원
투영 후 AE+GMM energy, CPU. FLTrust: 라운드별 서버 검증-gradient와의 signed cosine 평균.
FedDQC: per-sample IRA(빈 프롬프트 대비 조건부 손실 차), 클라당 128 서브샘플. 네 탐지기 모두
전 위협에서 실행하되 본문 비교는 각자의 홈 위협 기준이다.

**D.5 하드웨어·정밀도·재현성.** DGX B200 4장(대부분 셀은 1장/셀; 2026-07 비용 재계측은 B200
1장/방법). utility·HVP·내적은 fp32로 계산하며 감사로 검증했다; CNN 트랙의 conv 학습 연산은
cuDNN 기본 TF32다. attention은 eager 모드(forward-mode HVP 호환). 전 러너
seed-고정(torch/numpy/CUDA), CNN은 cudnn-deterministic으로 fp32 비트동일. 셀마다 run
디렉토리에 `config.yaml`(전체 설정), `meta.json`(git SHA·환경 해시·패키지 버전),
`phi.parquet`(방법별×클라이언트별 $\phi$ 원본), `metrics.json`(fidelity·AUROC·런타임),
`timing.json`(phase별 wall-clock·peak 메모리·GPU-h)을 영속화하며, 모든 표·그림은 rundir만으로
재생성된다.

**D.6 비용 회계 caveat.** ① individual-utility 런타임은 singleton utility의 base 중복 평가
버그(라운드당 forward $2|P_r| \to 1{+}|P_r|$)를 교정한 재측정값이다. $\phi$는 비트동일,
런타임만 $\sim$1.7배 과대였다. ② FLDetector·STD-DAGMM은 CPU 실행이라 GPU 방법들과 하드웨어가 다르다(model-free
로 GPU가 불필요하다는 것 자체가 이 방법들의 특성). ③ 보조 축으로 방법별 per-round
연산수(forward/gradient/HVP) 해석적 카운트 × 연산별 microbench 시간이 실측 wall-clock을
재현함을 확인했다(하드웨어-독립 검산). ④ 조사한 선행 7편 중 valuation-only wall-clock 회계를
명시 정의한 논문은 없었다. 본 절의 회계 정의 자체가 보고 관행의 개선이다.
### 부록 E. 신호-크기 진단 (확장)

**E.1 학습-강도 축 probe.** "IID-clean에서 신호가 없는 것은 학습이 약해서인가"를 네 lever로
검사했다. ① **LoRA rank** $r \in \{16, 32, 64\}$ (N=5 full, seed 0): exact in-run Shapley의
클라이언트 간 $\phi$ 범위는 $0.0012 \to 0.0011$로 평평하고 Flirds fidelity는 전 rank
$+1.000$. ② **참여 구조** (N=50, 5/round, $r \in \{16,32,64\}$): $\phi$ 범위는 rank로
$\times 1.2$ 변화뿐; 대신 부분 참여 자체가 방법을 갈랐다(본문 §5.2; Flirds/first-order
$+1.00$ vs uniform-subset 계열 음수 붕괴). ③ **학습률** $\{1,2,3\} \times 10^{-3}$ (3-seed,
st10): seed 0에서는 $\phi$ 범위가 $\sim$3배 커 보이지만 seed 1에서는 오히려 줄어(0.0015 →
0.0009), **전 seed 재현되는 효과는 평균 $|\phi|$의 $\sim$1.3배 공통 이동뿐**이다; exact
in-run Shapley의 자기-일치도는 lr 무관 $\approx 0$($-0.37 / -0.20 / -0.23$)이고 Flirds fidelity는 전 칸
$+1.000$. ④ **로컬 스텝** $\{10,20,30\}$ (seed 0): $\phi$ 범위 무영향. CNN 대응 실험(폭
$w \in \{0.5,1,2,4\}$, 3-seed)도 같다: IID의 자기-일치도는 폭 8배에도 0 근처($+0.03$–$+0.12$)
불변, label-flip은 폭 무관 $\approx 0.9$. **결론**: 어떤 학습-강도 lever도 cross-seed 실재
신호를 만들지 못하며, 신호는 클라이언트 간 실제 차이(비IID·오염·데이터 양)가 만든다.

**E.2 오염 × 분포 매트릭스 (1B, N=5 full, $R{=}10$ 매트릭스 캠페인, 3-seed; exact in-run
Shapley 자기-일치도).**

| 무대 | clean | noisy | fr-random | fr-zero |
|---|---|---|---|---|
| IID | $+0.13$ | $+0.60$ | $+0.80$ | $+0.70$ |
| non-IID (5-도메인) | $+0.87$ | $+0.93$ | $+0.93$ | $+1.00$ |

두 축이 독립적으로 신호를 만든다. 결정적 칸은 **non-IID clean $+0.87$**이다. 오염이 전혀 없어도
도메인 이질성만으로 신호가 생기며, 기존 non-IID 셀의 높은 안정성이 오염 때문이 아님을
분리해서 확정한다. 탐지 축 대조(3-seed AUROC): Flirds·FLTrust는 배경과 무관하게 noisy·FR
$1.00$; FedDQC는 균질 배경에서 noisy가 더 깨끗하고(IID $1.00$ vs non-IID $0.92$) fr-zero에는
약하다(IID $0.58$ / non-IID $0.75$); STD-DAGMM은 전반적으로 약하다. 정본:
`runs/matrix_cxni/figures/crossseed_rho.csv` + `runs/phase2_matrix/rundirs/1B_{iid5,silo5}_*`.

**E.3 검증-노이즈 분리 (bootstrap).** 학습된 모델을 고정하고 검증셋을 청크 단위로 2,000회
bootstrap 재표집하면: $\phi$의 클라이언트 간 산포는 클라이언트별 bootstrap 표준오차의
$1.15/1.37/2.34$배(seed 0/1/2)에 불과하지만, 재표집 자기-순위 상관은 $0.93/0.96/0.99$,
검증셋 반분할 상관은 $0.90/0.90/1.00$이다(rank 64도 동일 구조). 극단 쌍의 분리가 SE의
$\sim$10배라 순위는 검증 노이즈에 강건하다. 즉 **같은 seed 안에서는 순위가 측정 노이즈를
뚫고 재현되고, seed를 넘으면 무너지는 것은 데이터 파티션·궤적의 실현 그 자체다**. 본문
§5.4의 층위 구분 그대로다.

**E.4 MMLU 검정력.** 이 무대의 연합 SFT는 같은-분포 표면 지표(val-loss $-0.03$–$-0.12$,
ROUGE-L $+1.5$–$+12.8$pp)는 실제로 움직이지만 capability 축은 움직이지 않는다: MMLU는 base
대비 $-1.4$–$+0.3$pp(이항 SE $\pm 0.42$pp)로 향상 0 또는 소폭 하락이다. 개입 효과의 기대
크기($\Delta$val-loss $\sim 10^{-3}$)는 MMLU 표본 SE($\pm 4 \times 10^{-3}$)와 vanilla 최종
손실의 seed 분산($0.02$–$0.03$)보다 작아 **비대응(unpaired) 벤치마크 축으로는 원리적으로
검출 불가**하고, 같은 seed·init·데이터로 짝지은 paired val-loss 축에서만 SNR $2.4$–$4.5$로
일관 검출된다. selection·개입 실험을 val-loss 축으로 평가하는 이유다.
