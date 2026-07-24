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

**검증.** 선행 연구들은 기여도 점수의 타당성을 주로 다운스트림 효과로 확인해 왔다:
기여도-선택 학습이 수렴이나 성능 향상을 돕는지, 저기여 클라이언트가 실제로 오염돼
있었는지를 보는 식이다.
우리는 검증을 두 겹으로 나눈다. **추정기 층**에서는 Flirds가 겨냥하는 게임의 exact Shapley 값, 곧
in-run 정의를 근사 없이 라운드당 $2^{|P_r|}$ 전수 열거로 계산한 **exact in-run Shapley**를
같은 학습 궤적 위에서 직접 구해, 추정값이 이를 순위·값 수준에서 재현하는지 잰다. 근사
참조나 다운스트림 간접 증거에 의존해 온 선행 관행보다 강한 기준이다. **실효성 층**에서는 측정한 기여도가 실제로 학습에
도움이 되는 값인지를 잰다. 각 방법이 매긴 기여도를 기반으로 학습 과정을 개선하고, 그
학습 성과로 방법들을 비교한다: 학습 도중 순기여 $\le 0$인 클라이언트를
자동 배제하는 온라인 sign-gating, 기여도 순서대로 클라이언트를 제거하고 실제로
재학습해 순위의 인과적 타당성을 확인하는 removal 실험, 누적 기여도가 양(+)인 클라이언트만 남겨 처음부터
재학습하는 selection 실험, 그리고 오염·무임승차 클라이언트 탐지다. 여기에 방법별 계산 비용을
같은 고정 궤적 위에서 실측한다. 고전적 exact retrain Shapley와의 실증적 관계는 특성화 실험으로만 보고한다(§5.2).

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
공리를 완화하거나 포기한 채 기여도 신호를 강건-집계 가중치로 소비하는 쪽으로 이동한다. 이
계보에서 §1의 두 한계를 그대로 관찰한다. 첫째는 **계산의 우회 그 자체**다: 위 방법 전부가
목표 Shapley 값의 exact 계산 대신 무작위 표본 추출·절단·보간에 의존하며, 검증도 소형 CNN
규모에 머물러 왔다. 둘째는 **우회가 낳은 대체 정의와 검증 공백**이다: 비용을 낮추는
과정에서 같은 이름 아래 서로 다른 값(재정규화 게임, 보간된 손실 행렬, 가공된 surrogate)이
추정되고 있으나, 그 값의 타당성은 대부분 다운스트림 결과로만 간접 확인되었고, exact 참값
대비 직접 채점은 SPACE의 $2^n$-재학습 비교($N \le 10$의 CNN 분류)처럼 소형 스케일에
국한된다.

이 계보를 §1의 정산 관점에서 다시 보면 방법들이 두 갈래로 갈린다. Shapley 값을 겨냥한
쪽(FedSV·GTG-Shapley·ComFedSV)은 지수 비용을 표본 추출·절단·보간으로 갚으면서 결정론과 오차
규명을 내주었고, 비용과 결정론을 확보한 쪽(ShapleyFL·FedIF 계열)은 Shapley 공리를 완화하거나
포기하면서 "이 값이 무엇에 대한 대가인가"에 정의로 답할 근거를 내주었다. §1이 열거한 정산
요구들은 개별로 보면 어느 한쪽이 이미 만족한다 — 문제는 둘을 동시에 만족한 방법이 없다는
것이고, 그 배타성의 원인은 지수 비용이다. 비용을 근사로 갚는 대신 해석적으로 접어 없애면
배타성 자체가 사라진다는 것이 본 논문의 출발점이다(§4).

**중앙집중 LLM-규모 attribution.** 중앙집중 학습에서는 개별 학습 예제 단위의 데이터
귀속·선별이 LLM 규모까지 활발히 발전해 왔으며, 크게 세 줄기다. 첫째, **influence function
계열**은 각 예제가 검증 손실에 미치는 영향을 gradient와 Hessian 역행렬($H^{-1}$)로 추정하며,
LLM 규모에서는 $H^{-1}$ 근사를 서로 다르게 처리한다(EK-FAC[Grosse et al., 52B], LoRA용 closed-form
근사 DataInf, TRAK, LoGra 등). 둘째, 2024년의 **Hessian-free 흐름**은 $H^{-1}$을 아예
우회한다: LESS는 TracIn 계열의 궤적 influence를 LoRA gradient 사영으로 계산해 instruction
tuning 예제를 고르고, MATES·DsDm은 각각 증류한 소형 모델과 선형 datamodel로 사전학습
데이터를 고른다.
셋째, **In-Run Data Shapley**(§3.3)는 사후 $H^{-1}$ 대신 실제 학습 궤적을 따라 매 스텝의
Taylor 기여를 누적한다. 세 줄기 가운데 연합학습의 무대로 가장 자연스럽게
이어지는 것은 In-Run 계열인데, FedAvg 집계 $\sum_k p_k \Delta w_k$가 배치 gradient의
샘플-선형 분해와 같은 구조를 클라이언트 수준에서 이미 드러내기 때문이다. 본 논문은 이
관찰에서 출발해 IRDS의 closed-form 계산을 연합 라운드 게임으로 확장한다(§4).

**탐지·강건 집계 baseline.** 연합학습에는 오염·악성 클라이언트에 대응하는 별도의 계열이
있다: 업데이트 통계로 이상 클라이언트를 찾아내는 **탐지**(FLDetector, FLTrust, STD-DAGMM,
FedDQC 등)와, 집계 단계에서 outlier 업데이트의 영향을 억제해 강건성을 확보하는 **강건
집계**(Krum, 좌표별 median, trimmed-mean 등)다. 이들은 이진 제거(keep/discard)나 신뢰
가중치를 산출할 뿐 부호 있는 연속 기여도와 분배 공리를 다루지 않으므로, 기여도 평가의
대체재가 아니다. 우리는 두 축을 분리하고, 위 전용 탐지기들과는 각자가 설계된 위협에서
비교한다(§5.4).

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
개선할 수 있는가(§5.3), 오염 클라이언트가 가려지는가(§5.4). 이어서 방법별 계산 비용(§5.5)과
구성요소별 ablation(§5.6)을 본다. 프로토콜 상세와 위협 구현은 부록 B, fidelity 확장
표(cross-game 계열 전표 포함)는 부록 C, 재현성(안정성) 분석은 부록 D, 비용·규모 보조 실험은
부록 E에 있다.

### 5.1 실험 세팅

**무대.** 주무대는 LLM·CNN 한 쌍이며, 두 트랙 모두 "오염 클라이언트가 섞인 부분 참여
FL"이라는 같은 구도를 공유한다. retrain GT가 필요한 비교는 $2^N$ 재학습이
가능한 작은-$N$ 보조 무대가 담당한다(§5.2).

| 무대 | 트랙 / 데이터 | $N$ · 참여 | $R$ | 위협 축 | GT | 쓰임 |
|---|---|---|---|---|---|---|
| **주** `LLM-Main` | LLM 1B(LoRA) · GSM8K | 50 · 5/50 | 200 | clean / answer-swap@0.7 / free-rider(zero) — 오염 40% | in-run GT (per-round $2^5$) | §5.2–5.5 |
| **주** `CNN-Main` 캠페인 | CNN · CIFAR-10×{iid,Dirichlet(α=1),shard,quantity-skew} + FMNIST×{iid,Dirichlet(α=1)} | 100 · 10/100 | 120 | clean / free-rider(zero·rand) / gradient noise / label-flip@{.15,.35,.70}·variable-intensity label-flip — 오염 40% | in-run GT (per-round $2^{10}$) | §5.2–5.4 |
| 보조 `LLM-Small` | LLM 1B · GSM8K(주무대 정합) | 5 · 전원 | 30 | clean / answer-swap@0.7 — 오염 40% | **retrain GT ($2^5$)** + in-run GT ($2^5$) | §5.2 |
| 보조 `Silo` | LLM 1B · 5-도메인 비IID | 5 · 전원 | 10 | clean / answer-swap / free-rider | in-run GT ($2^5$) (+ retrain GT ⬚) | §5.2·§5.6 |
| 보조 `Anchor` | LLM 1B · alpaca IID clean | 5 · 전원 | 30 | – | **retrain GT ($2^5$)** + in-run GT ($2^5$) | §5.2 |
| 보조 `CNN-Grid` | CNN · MNIST+CIFAR-10 × 5시나리오 | 10 · 전원 | 10 | iid / label·quantity-skew / label-flip / feature-noise | **retrain GT ($2^{10}$)** + in-run GT ($2^{10}$) | §5.2·§5.6·부록 C·D |

성능 심판은 LLM-Main = GSM8K 공식 test의 잔여 1,119문항 exact-match(EM; greedy 디코딩),
CNN = held-out test 정확도다. 학습은 두 트랙 모두 momentum 없는 plain SGD·상수
학습률·stateless 클라이언트(부록 A.9의 가정 그대로)이고, LLM 트랙은 LoRA(r=16, α=32) 인자만
교환한다(부록 A.10). 하이퍼파라미터 전량·데이터 분배 규칙은 부록 B. 표에서 위협 이름
`lf@r`는 label-flip@r(라벨 오염 비율 $r$)의 축약이다. 부록 E의 보조
무대(완전참여 100/100, $N{=}10$ LLM $2^{10}$, cross-device anchor)는 비용·규모 주장 전용이다.

**비교 방법: 겨냥하는 게임으로 나눈다.** 기여도 방법 8종을 두 계열로 구분하며, 이 구분이
§5.2 표 구성의 원리다.

- **같은-게임 계열(3)** — 식 (6)의 고정-가중 라운드 게임을 겨냥한다: **Flirds**(2차 closed-form),
  **Flirds-1st**(1차 항만), **individual utility**(라운드 게임의 singleton utility
  $u_r(\{k\})$를 forward 평가로 직접 계산해 합산하는 가산 근사 — Shapley 값 대신 singleton
  값). 이들과 in-run GT의 차이는 순수한 근사 오차다.
- **cross-game 계열(5)** — GTG-Shapley, FedSV, ComFedSV, ShapleyFL, FedIF: §2의 계보
  그대로 재정규화 게임·보간 행렬·가공 surrogate·influence 등 저마다 다른 값을 겨냥한다.
  in-run GT와의 불일치에는 근사 오차와 "다른 게임" 성분이 섞여 있으므로 본문 fidelity 표에서는
  같은-게임 계열과 분리하고, 전표를 부록 C에 둔다. 단 **retrain GT 대비 비교는 전 방법을
  같은 표에** 둔다 — retrain GT는 어느 방법의 목표값도 아닌 중립 참값이기 때문이다.

전용 탐지기 4종(FLDetector, FLTrust, STD-DAGMM, FedDQC)은 §2의 약속대로 §5.4에서 비교한다.
제외: Banzhaf(다른 semivalue 축), Ripple(방법이 자체 재학습 궤적을 요구해 아래 고정-궤적
채점과 비호환). clean-preserving poisoning 위협은 본 논문의 스코프 밖이다(§6). baseline
재구현·파라미터 주석(ShapleyFL EMA β, ComFedSV per-round 대용 등)은 부록 B.5.

**지표.** fidelity = Spearman $\rho$(순위)·Pearson $r$(값)을 본문에, Kendall
$\tau$·거리 3종(cosine/euclidean/max)을 부록 C에 둔다. 탐지 = AUROC(오염 클라이언트를
양성으로 두고 기여도 순위 하위 = 의심 규약; 부록 B). 개입 = **절대 성능**(EM/acc)을
vanilla(개입 없음 = 바닥)와 oracle-제외(오염 클라 정확 제외 = 천장) 사이에서 직접 읽는다 —
정규화 점수가 셀 간 기준선 차이를 가리는 것을 피한다. 비용 = 같은 궤적 위 valuation 단독
wall-clock과 하드웨어-독립 연산수 모델(§5.5).

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

**주무대, 같은-게임 계열 vs in-run GT.** 표 [F1]은 CNN 주무대 144셀(파티션 6 × 위협 8 ×
3-seed)을 위협별로 풀링한 Spearman, 표 [F2]는 LLM 주무대의 위협별 Spearman·Pearson이다.

표 [F1] — CNN 주무대(`CNN-Main`): 같은-게임 3종 vs in-run GT, Spearman ↑ (위협별; 파티션 × seed 풀) ⬚
<!-- 채움: runs/track_c/c2fid/analysis/{fidelity,cellmean}.csv (make_analysis.py 재생성) -->

| method | clean | free-rider(zero) | free-rider(rand) | gradient noise | lf@0.15 | lf@0.35 | lf@0.70 | variable-intensity label-flip |
|---|---|---|---|---|---|---|---|---|
| Flirds | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| Flirds-1st | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| individual utility | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |

표 [F2] — LLM 주무대(`LLM-Main`): 같은-게임 3종 vs in-run GT (per-round $2^5$) ⬚(◐)
<!-- 채움: runs/phase2_matrix/rundirs/1B_gsm50k5_{noisy_nr0.7,clean}_s0/metrics.json (L2) -->

| method | answer-swap Spearman | answer-swap Pearson | clean Spearman | clean Pearson |
|---|---|---|---|---|
| Flirds | ⬚ | ⬚ | ⬚ | ⬚ |
| Flirds-1st | ⬚ | ⬚ | ⬚ | ⬚ |
| individual utility | ⬚ | ⬚ | ⬚ | ⬚ |

(free-rider(zero) 셀의 fidelity는 별도 채점하지 않는다 — in-run GT와 같은-게임 계열 모두 해당
클라이언트에 대수적으로 정확히 0을 주어(명제 2) 일치가 자명하며, 이 성질은 탐지(§5.4)와
개입(§5.3)에서 실효성으로 검증된다.)

읽기 각주 두 개. ① **clean 칸은 신호-부재 레짐이다**: 클라이언트 간 실제 차이가 없어 in-run GT
자신의 순위조차 seed를 넘어 재현되지 않는다(부록 D). 이 칸의 낮은 $\rho$는 방법의 실패가
아니라 잴 것이 없음을 뜻하며, 오발화 대조용으로만 둔다. ② variable-intensity label-flip(오염 클라이언트별 강도가
연속 분포) 칸에서는 φ가 오염 강도 자체를 얼마나 해상하는지도 채점한다(φ vs 실현 오염율의
Spearman): ⬚ <!-- F-4 사전등록: Flirds ≈ in-run GT 자기천장 > 1st — c2fid make_analysis 자동 대조 -->.

**sub — exact retrain Shapley retrain GT 특성화.** retrain GT는 $2^N$번의 전체 재학습을 요구하므로
주무대($N{=}50/100$)에서는 계산이 원리적으로 불가능하다. 그래서 부득이하게 작은-$N$ 별도
무대에서 특성화하되, 무대마다 다른 축을 담당하도록 세팅을 의도적으로 다양화했다:
**LLM-Small** = 주무대와 데이터·위협·오염 비율(2/5 = 40%)·검증 규칙·하이퍼를 전부 공유하고
$N{=}5$ 전원 참여·$R{=}30$만 축소한 "라운드-cohort 축소판"(주무대의 라운드당 참여자 수가 곧
5다) — 주무대 정합 주 표. **Silo** = 도메인 이질성(비IID) 신호 무대. **Anchor** = IID-clean
참조. **CNN-Grid** = retrain GT·in-run GT를 동시에 $2^{10}$ 전수로 계산하는 시나리오 격자.

표 [F3] — `LLM-Small`: 전 방법 vs retrain GT ($2^5$) (주 표) ⬚
<!-- 채움: L8 (paper/workplan/T5-retrain-a-suite.md) rundir → 분석 CSV; 3-seed -->

| method | clean Sp vs retrain GT | answer-swap Sp vs retrain GT | (참고) Sp vs in-run GT |
|---|---|---|---|
| Flirds | ⬚ | ⬚ | ⬚ |
| Flirds-1st | ⬚ | ⬚ | ⬚ |
| individual utility | ⬚ | ⬚ | ⬚ |
| GTG | ⬚ | ⬚ | ⬚ |
| FedSV | ⬚ | ⬚ | ⬚ |
| ComFedSV | ⬚ | ⬚ | ⬚ |
| ShapleyFL | ⬚ | ⬚ | ⬚ |
| FedIF | ⬚ | ⬚ | ⬚ |
| **retrain GT↔in-run GT 일치도** | ⬚ | ⬚ | – |

Silo의 retrain GT-leg(clean·answer-swap·free-rider × 3-seed)는 ⬚
<!-- 채움: L8 Silo *_aonly_* rundir + 기존 canonical phi.parquet 조인(merge 패턴) -->
— Silo는 $N{=}5$ 무대 가운데 유일하게 in-run GT 타깃 순위가 seed를 넘어 재현되는 무대라(부록 D:
clean +0.87, 오염 +0.93) "retrain GT가 그 실재 신호를 같게 매기는가"를 처음 재는 칸이다.

표 [F4] — `Anchor`(IID-clean 참조): 전 방법 vs retrain GT ($2^5$) (1B, 3-seed)
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

읽기: 같은-게임 3종과 GTG가 0.933으로 동률인 것은 천장 효과다 — 이들은 in-run GT와 사실상 완전
일치(vs in-run GT ≈ 1.000)하므로, vs retrain GT 점수가 두 참값 사이의 일치도 **retrain GT↔in-run GT = 0.933±.047**
그 자체로 수렴한다. 즉 이 무대에서 같은-게임 계열의 retrain GT 재현은 게임 간 간극이 상한이다.
주의할 점은 Anchor가 IID-clean이라 in-run GT 타깃 자체가 seed-불안정하다는 것(부록 D: cross-seed
$\rho$ −0.37)이며, 이것이 LLM-Small·Silo를 주 표로 두는 이유다.

표 [F5] — CNN `CNN-Grid`: 같은-게임 3종 vs retrain GT ($2^{10}$), Spearman ↑ (시나리오별, 3-seed 평균)
<!-- 출처: runs/track_c/fidelity.csv spearman_a group-mean. 전 방법(8종)·Pearson = 부록 C -->

| dataset / scenario | Flirds | Flirds-1st | individual utility |
|---|---|---|---|
| cifar10 / feature_noise | **+0.63** | +0.50 | +0.56 |
| cifar10 / iid | −0.23 | −0.13 | −0.18 |
| cifar10 / label_flip | +0.52 | +0.59 | +0.58 |
| cifar10 / label_skew | −0.18 | −0.07 | +0.14 |
| cifar10 / quantity_skew | +0.57 | +0.56 | +0.57 |
| mnist / feature_noise | +0.33 | +0.44 | +0.44 |
| mnist / iid | +0.36 | +0.52 | +0.48 |
| mnist / label_flip | **+0.96** | +0.97 | +0.97 |
| mnist / label_skew | −0.28 | −0.06 | −0.16 |
| mnist / quantity_skew | **+0.85** | +0.77 | +0.84 |

읽기: 두 게임(realized 귀속 vs counterfactual 재학습)은 **클라이언트 간 실제 차이가 있는
칸에서 수렴한다** — cifar10의 label-flip·feature-noise·quantity-skew와 mnist의
label-flip·quantity-skew에서 같은-게임 계열 vs retrain GT +0.50~+0.97이고, 전 방법(8종) 가운데
Flirds가 1위인 칸이 둘(cifar10/feature_noise +0.63, mnist/quantity_skew +0.85),
mnist/label_flip은 전 방법이 +0.94~0.97로 동수렴한다(부록 C). mnist/feature_noise(+0.33~
+0.44)는 오염 σ가 약해 게임 수준 신호 자체가 흐릿한 칸이다.
신호-부재 칸(iid·label_skew)에서는 두 게임의 순위가 상관을 잃는데, 이는 위 각주 ①과 같은
현상의 retrain GT-버전이다 — retrain GT 게임 값 자체가 재학습 노이즈 수준의 차이를 순위화한 것이기
때문이다. 한편 retrain GT가 부분집합 크기로 재정규화되는 게임이라는 사실이 재정규화 계열에
유리하게 작동하는 칸도 있다(cifar10/quantity_skew는 ShapleyFL +0.81이 최고 — 부록 C).

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

**LLM 주무대(`LLM-Main`), 절대 EM.** 표 [I1] ⬚
<!-- 채움: L1 Tier C 3-seed — runs/track_h/analysis/gsm50k5_*.csv (fix-후 rundir만).
     pre-fix seed0(git_sha fa5fc6e) 값 인용 금지. renorm 4점수원 블록은 L4 착지 시 추가. -->

**online gating**(정책 = sign-gating; 점수원 = Flirds):

| arm | clean | answer-swap | free-rider |
|---|---|---|---|
| vanilla(관찰자) | ⬚ | ⬚ | ⬚ |
| oracle-제외 (천장) | – | ⬚ | ⬚ |
| random-제외 (통제) | – | ⬚ | ⬚ |
| Flirds · sign-gating | ⬚ | ⬚ | ⬚ |

**retrain**(정책 = sign-gating × 점수원 4종):

| arm | clean | answer-swap | free-rider |
|---|---|---|---|
| vanilla(관찰자) | ⬚ | ⬚ | ⬚ |
| oracle-제외 (천장) | – | ⬚ | ⬚ |
| retrain-random (통제) | – | ⬚ | ⬚ |
| Flirds · sign-gating | ⬚ | ⬚ | ⬚ |
| Flirds-1st · sign-gating | ⬚ | ⬚ | ⬚ |
| individual utility · sign-gating | ⬚ | ⬚ | ⬚ |
| FedIF · sign-gating | ⬚ | ⬚ | ⬚ |

**CNN(Dirichlet α=1)†, 절대 test acc.** 같은 정책·같은 무대·같은 seed에서 **점수원만 8종으로 교체**한
경쟁이다. 표 [I2] (오염-평균 = free-rider·gradient noise·label-flip 3위협 평균).
<!-- 출처: runs/track_h/analysis/{competition_score,cnn_competition}.csv (07-20 집계 정정판).
     † = W-A(캠페인 restack 드리프트 표) 확인 후 확정 — paper/workplan/T4 참조. -->

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
.2479/.2436 ≈ vanilla(.2436) — noise 클라이언트를 아예 보지 못한다(§5.6-①). ③ label-flip은
재학습(retrain) 우위: 전 estimator가 online .57대 → retrain .62대(천장 근접)로 상승한다.
경계-강도 무대(variable-intensity label-flip, 클라별 rate ~ U(.5,1))에서도 같은 서열이 재현된다(sign-gating retrain 회복률
Flirds +0.98·renorm 0.60~0.74; online renorm 파국 최심 FedSV −0.32)†. ④ **clean 오발화의
정직 보고**: online에서 Flirds −0.7pt·individual utility −1.3pt(누적 부호의 0-교차 노이즈),
Flirds-1st·FedIF는 무발화. 종합하면 정확한 클레임은 "Flirds 단독 1위"가 아니라 — **전
정책·전 시점 상위권 + gradient noise를 잡는 유일한 estimator**이며, 개별 칸 최고는
FedIF·individual utility도 차지한다. LLM 주무대에서 같은 구조가 재현되는지는 표 [I1] ⬚.

**magnitude-weighted gating(크기-가중 변형).** [자리] sign-gating과 같은 배제에 더해, 남는 클라이언트를 양수 누적 기여도
크기에 비례해 가중($w_k \propto n_k \max(\hat\phi_k, 0)$, 합-1 재정규화)하는 변형의 전 무대
검증이 진행 중이다 ⬚.
<!-- 수록 규칙(사전 고정 — workplan 00-INDEX §1): CNN·LLM 전 범위에서 sign-gating 상회 시 본문 승격 /
     동률 시 "부호가 가치의 대부분" 1문장 / 열세·타 점수원 역전 시 미수록(본 절은 sign-gating만). -->

### 5.4 탐지 (2차 ③)

탐지를 위계의 마지막에 두는 이유는 기여도와 탐지가 직결이 아니기 때문이다: 검증 손실을
실제로 낮추는 오염(예: 직전 라운드의 글로벌 업데이트를 재제출하는 free-rider 변형)에 대해
val-loss 게임의 정직한 답은 "기여함"이고, 이는 추정 실패가 아니라 in-run GT 자신의 답이기도
하다(§6). 따라서 주장 형식은 절대 AUROC가 아니라 **in-run GT 일치**이다 — 추정기가 in-run GT와 같은
답을 주는가(사전 등록 기준: $|\mathrm{AUROC}(\text{Flirds}) - \mathrm{AUROC}(\phi^{\mathrm{in}})| \le
0.05$). φ-파생 탐지는 기여도 순위 하위 = 의심 규약의 AUROC이고, 전용 탐지기 4종은 각자의
스코어를 그대로 쓴다.

표 [D1] — LLM 주무대(`LLM-Main`) AUROC ⬚(◐)
<!-- 채움: L2 rundir metrics.json auroc — H-13 대조(in-run GT 일치 기준) 포함 -->

| method | answer-swap | free-rider |
|---|---|---|
| in-run GT (게임의 답) | ⬚ | ⬚ |
| Flirds | ⬚ | ⬚ |
| Flirds-1st | ⬚ | ⬚ |
| individual utility | ⬚ | ⬚ |
| FLDetector | ⬚ | ⬚ |
| FLTrust | ⬚ | ⬚ |
| STD-DAGMM | ⬚ | ⬚ |
| FedDQC | ⬚ | ⬚ |

CNN 주무대 φ-AUROC(오염 셀, 위협별 풀) ⬚
<!-- 채움: c2fid analysis auroc 열. F-2 사전등록: free-rider 셀 exact-0 계열 AUROC 1.0 vs
     renorm 유령값 저하; free-rider(rand) 셀 Flirds 계열 ≥ GTG/FedSV -->

### 5.5 비용

**연산수 모델(하드웨어-독립).** 방법별 라운드-누적 지배 연산수는 해석적으로 닫힌다:
Flirds = 라운드당 **HVP 1회**(cohort 크기와 무관), Flirds-1st = 라운드당 val-gradient 1회,
individual utility = 라운드당 $1{+}|P_r|$ forward, in-run GT = 라운드당 $2^{|P_r|}$ forward. per-op
실측(fp32·B200: forward 1.60s, HVP 10.36s — HVP/forward ≈ 6.5)과의 곱이 wall-clock을
재현한다:

| 무대 | Flirds | Flirds-1st | individual utility | in-run GT |
|---|---|---|---|---|
| Silo ($N{=}5$·$R{=}10$) | 10 HVP → 예측 104s / 실측 ~107s | 10 grad (~35s) | 60 fwd → 96s / 96.6–100.1s | 320 fwd → 512s / ~530s |
| Anchor ($N{=}5$ 전원·$R{=}30$) | 30 HVP → 실측 707±16s | 30 grad (231±5s) | 180 fwd (657±19s) | 960 fwd (3,528±83s) |
| cross-device ($K{=}10$·$R{=}30$) | 30 HVP → 실측 157s | 30 grad (53s) | 330 fwd | **30,720 fwd → 실측 24,975s** |

(무대마다 검증셋 크기·시퀀스 길이가 달라 per-op 절대 단가는 다르지만, 연산수 비가 wall-clock
비를 지배한다 — Silo 행은 microbench 곱이 실측을 그대로 재현하는 검증 라인이다.)

**주무대 실측** ⬚ <!-- 채움: LLM-Main L1·L2 timing.json + c2fid runtime_s 열 -->. 구조는 위
모델이 이미 말해 준다. 주무대 LLM은 라운드당 cohort 5라 in-run GT가 라운드당 $2^5{=}32$ forward로
아직 감당 가능하고(같은 구조인 Anchor의 실측 비 ≈ 5×), cohort 10이면 실측 159×
(cross-device), $N{=}10$ 전원 참여의 $2^{10}$ 전 라운드 열거는 32.7h vs Flirds 733s =
160×로 벌어진다(부록 E). 반대로 라운드당 참여가 아주 작은 무대(예: 2명 → $2^2{=}4$ forward
< 1 HVP ≈ 6.5 forward)에서는 in-run GT 직접 열거가 Flirds(2차)보다 싸다 — **Flirds의 비용 우위는
"참여가 많아 전수 열거가 지수적으로 비싼" 무대의 것**이고, 그렇지 않은 무대에서도
Flirds-1st(라운드당 gradient 1회)는 항상 최저가다. retrain GT는 같은 무대에서 in-run GT의
~9배다(Anchor: 30,817±244s vs 3,528±83s — $2^5$개 부분집합 × $R{=}30$ 재학습). CNN에서는
valuation이 학습 자체보다 두 자릿수 싸다(CNN-Grid: Flirds 0.6~1.4s vs FL 학습 80~94s).

### 5.6 Ablation

**① 2차항(클라이언트-상호작용 항)의 기여.** §4.2의 Flirds-1st와의 대조다.
fidelity 축: CNN-Grid 참여 sweep에서 클라이언트당 참여 횟수가 적으면 1차 근사가 붕괴하고
2차항이 방어한다 — label-flip, 라운드 참여 2/10에서 Flirds 0.891±.147 vs Flirds-1st
0.305±.434(전원 참여에선 0.993±.008 vs 0.940±.039; 전 72셀 풀 Flirds 0.953±.080). 참여
횟수가 충분하면(1B 5/50·$R{=}200$, 클라당 ~20회) 1차도 1.000을 유지한다 — 조건은 "참여
분수"가 아니라 **클라이언트당 참여 횟수**다. 다운스트림 축: gradient noise 무대에서 2차항
유무가 회복과 실명을 가른다(§5.3: Flirds .5668/.6065 vs 1차 계열 .2479/.2436 ≈ vanilla)†.
오염-강도 해상도(variable-intensity label-flip, F-4) ⬚.

**② 학습-강도 lever는 fidelity를 흔들지 않는다.** LoRA rank {16,32,64}(용량), lr
{1e-3,2e-3,3e-3} × local steps {10,20,30}(강도), CNN 폭 {0.5,1,2,4}(8×)를 각각 sweep해도:
Flirds vs in-run GT는 **전 칸 1.000**(lr·steps로 per-round 이동이 커져도 라운드당 1 HVP가 in-run GT를
정확 재현 — Taylor 절단의 실무적 트레이드오프가 이 범위에선 관측되지 않는다), 클라이언트 간
φ 분리도 lever로는 거의 변하지 않는다(CNN 폭 8×에 φ range 평평; 오염(label-flip)이 iid의
2–4×를 만든다). 방법 간 구별을 만드는 축은 lever가 아니라 **참여 형태**다: 5/50
부분참여(1B·$R{=}200$)에서 같은-게임 3종은 0.999–1.000을 유지하지만(Flirds 3-seed
+1.000±.000) ComFedSV·ShapleyFL·FedIF는 음수로 붕괴하고, GTG 0.98·FedSV 0.91이다(전표
부록 C.5).

**③ Removal-curve — 게임-무관 인과 검증.** 각 방법의 기여도 순위대로 클라이언트를 실제로
제거하고 처음부터 재학습해, 순위의 인과적 타당성을 게임 정의와 무관한 공통 자(ruler)로
확인한다. LLM Silo(3위협 × 3-seed): worst-first 제거가 val-loss를 내리고(+0.0067~+0.0076)
best-first는 올린다(−0.0084~−0.0015); Flirds·Flirds-1st·individual utility·in-run GT의 removal 곡선은 9/9
셀에서 엄밀히 일치하고(같은 순위 → 같은 곡선), FedIF만 free-rider(zero)에서 질적으로
얕다(worst-first +0.0038). CNN-Grid(6셀 × 3-seed)은 **정확도 축**으로 같은 구조를
보인다: cifar10 label-flip·feature-noise에서 Flirds 순위의 worst−best 정확도 분리
+0.039~+0.045로 in-run GT와 동급(+0.038~+0.045)이고, 순위가 낮은 방법(ShapleyFL $\rho$
+0.07~+0.26)은 분리 ≈ 0 — 순위 품질이 그대로 인과 분리로 이어진다. iid 통제군은 기대대로
분리 ≈ 0이다(cifar10은 소폭 음수 −0.0033이나 in-run GT도 −0.0027 — 데이터량 손실이 지배하는 무대
특성이지 방법 실패가 아니다).

---

## 6. 논의와 한계

<!-- 스텁 — 실험 완주 후 작성. 확정 재료 목록(T1 스펙):
 - 궤적-특이 utility의 공리화 미해결 — IRDS로부터의 승계(§3.3·부록 A.8); 공리 성립 주장은 frozen 게임 한정.
 - per-sample→per-client 브리지의 LLM 한계: token-mean CE에서 분모 불일치로 비성립(부록 A.7) — LLM valuation을 per-sample 합으로 근거짓지 않음.
 - LLM 위협 축의 스코프: gradient noise는 LoRA 기하에서 무대 미성립(부록 B.6) — LLM 쪽 update-공격은 free-rider 계열로 한정. 서술 근거·수치 = runs/track_h/gnoise_diag/README.md(그대로 인용 가능; 단 그 안의 Krum σ=200·arXiv 3편은 검증 실패 = 인용 금지 목록 동봉).
 - "기여도≠탐지"의 게임-공통 사례 1문장: delta-재활용 free-rider(frdelta)는 val-loss를 실제로 낮춰 in-run GT 자신이 "기여함"으로 답함 — 탐지는 update-패턴 탐지기의 몫.
 - 공정 분배·보상 스킴(fairness/reward)으로의 연결은 향후 과제.
 - 1-seed 항목 명시: N=10 2^10(부록 E), LLM-Main in-run GT-fidelity 셀(◐).
 - IID-clean 신호-부재 무대의 해석(부록 D): 방법이 아니라 무대의 결함 — retrain GT 특성화의 iid 칸 낮은 ρ 해석 포함.
 - 참여 lottery: 어느 라운드·어느 cohort에 뽑혔는지가 지급액에 영향 — symmetry는 라운드 게임 안에서만 성립하고, 라운드 배정 자체는 운영자 정책의 몫이다(정산 프레임의 스코프 한계).
-->

---

## 7. 결론

<!-- 골격 — 실험 완주 후 작성:
 - 헤드라인: §1 "측정에서 정산으로"가 열거한 요구들이 서로 배타적이지 않음을 보였다 —
   배타성의 원인이던 조합 재평가를 closed-form으로 없앤 것이 열쇠(§2 두 갈래 → §4).
 - LLM-scale 최초 2건 회수(기여 1·2): 클라이언트-수준 연합 valuation의 LLM 규모 수행 +
   exact GT 대비 채점의 LLM 규모 확장. 성질별 "최초" 주장은 하지 않는다.
 - 결과를 위계 순으로 한 문장씩: fidelity(§5.2) → 개입(§5.3) → 탐지(§5.4) → 비용(§5.5).
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
| `LLM-Small` | 동일 | 동일 규칙(클라당 149문항) | 5 · 전원 | 30 · 동일 | 동일 | 오염 2/5(=40%); 이중 GT(retrain+in-run) |
| `Silo` | Llama-3.2-1B + LoRA r16/α32 | 5-도메인 비IID(클라 = 도메인); train 200 / val 20 / test 40 | 5 · 전원 | 10 · 10 steps × batch 16 | SGD mom 0, lr 1e-3 | maxlen 768, warmup 2 |
| `Anchor` | Llama-3.2-1B-Instruct + LoRA r16/α32 | alpaca-gpt4 20k IID; val 200 / test 1000 | 5 · 전원 | 30 · 10 steps × batch 16 | SGD mom 0, lr 1e-3 | seq 512 |
| `CNN-Main` 캠페인 | CIFAR-10 = FedSVCNN / FMNIST = LeNet5 | 파티션 4종(B.3) | 100 · 10/100 | 120 · 5 epochs × batch 64 | SGD mom 0, lr 0.01 | 분배 seed 고정 |
| `CNN-Grid` | MNIST = LeNet5 / CIFAR-10 = FedSVCNN | 시나리오 5종(B.3); val 2000 / test 8000 | 10 · 전원 | 10 · 5 epochs × batch 64 | SGD mom 0, lr 0.01 | 이중 GT(retrain+in-run) $2^{10}$ |

부록 E 보조 무대: **완전참여** = `CNN-Main`의 Dirichlet(α=1) 무대에서 참여만 100/100; **$N{=}10$ LLM** =
anchor 무대의 $N{=}10$ 확장($R{=}30$, in-run GT = $2^{10}$ 전수); **cross-device** = 1B,
$N{=}100$·10/100, $R{=}30$·5 steps, 클라당 300, Dirichlet $\alpha{=}0.5$, val 10 / test 40.
전 무대 fp32 학습·채점.

**B.2 위협 구현.** 모든 오염은 클라이언트-재현적 seed로 고정된다(같은 셀 재실행 시 동일
오염 실현).

- **answer-swap@rate**(LLM): 클라이언트 데이터의 rate 비율에서 응답(풀이 + 최종
  답)을 같은 클라이언트의 다른 문항 것으로 순열 교체 — 형식은 완전한 정상 데이터이나
  문항-응답 대응이 깨진 현실적 mislabel. `LLM-Main`·`LLM-Small`는 rate 0.7; `Silo`의
  answer-swap($nr$)도 같은 순열 교체($nr$ = 클라 내 교체 비율).
- **free-rider(zero)**: 로컬 학습 없이 $\Delta = 0$ 제출. **free-rider(rand)**: 무학습, benign
  스케일에 맞춘 무작위 delta 제출.
- **gradient noise**(CNN): 정상 로컬 학습 후 업데이트에 가우시안 노이즈 주입.
- **label-flip@rate**(CNN): rate 비율의 라벨을 무작위 교체. **variable-intensity label-flip**: 오염 클라이언트별
  rate를 $U(0.5, 1)$에서 추첨(연속 강도 무대). 이때 label-flip의 오염 클라이언트 집합은
  FedCorr의 $(\rho, \tau)$ 잡음 모델을 공식 구현 그대로 따른다: 오염 여부가 클라이언트별 독립
  Bernoulli($\rho{=}0.4$)로 뽑혀 **오염 수가 시드마다 변동**하며($N{=}100$ 실현값 39/48/47,
  평균 44.7% — 표에는 명목 $\rho$ 대신 실현 수를 병기), 위 variable-intensity label-flip이 곧 FedCorr의 기본 강도
  draw($\tau \sim U(0.5,1)$)이고 고정-dose 셀 $\{0.15, 0.35, 0.70\}$은 그 draw를 지정값으로
  대체한 것이다(update-level 위협 — free-rider(zero)·free-rider(rand)·gradient noise — 는 준거 문헌이 없어
  정확히 $\lfloor \rho N \rceil{=}40$명 비복원 추출; 모든 대조는 위협을 고정한 채 이뤄지므로
  두 규약이 한 비교 안에서 만나지 않는다). **feature-noise**: 입력에 데이터 표준편차
  대비 $\sigma$ 가우시안. **label/quantity-skew**(`CNN-Grid`): 비IID 분배 자체가 시나리오(오염
  없음).

**B.3 데이터 분배.** GSM8K: 공식 test 1,319문항에서 200을 카브해 서버 검증셋으로, 잔여
1,119문항이 성능 심판 — 학습 데이터와의 분리는 공식 split 그대로다. `CNN-Main` 파티션: `iid` /
`Dirichlet(α=1)` = 라벨+크기 동시 skew / `shard` = 클라당 2-shard 라벨
skew(McMahan) / `quantity-skew` = 크기 skew 전용(라벨 IID; GTG 비율). `CNN-Grid` 시나리오 5종은
GTG-Shapley의 5-시나리오 무대 이식이다.

**B.4 연산 환경.** 학습·채점 전량 fp32(단 cuDNN convolution의 TF32 기본 활성은 CNN 트랙에
노출됨을 명시); 스택 내 결정론 옵션 고정(같은 seed 재실행의 궤적 재현 — free-rider exact-0의
in-run GT-쪽 전제). 실행 환경(GPU·라이브러리 버전)은 셀별 rundir meta에 기록된다.

**B.5 baseline 재구현 주석.** ① 부호 규약은 전 방법 contribution orientation(도움 =
양수)으로 통일했다(원 정의가 반대 방향이면 부호 반전만). ② ShapleyFL의 EMA는
$\beta{=}0.3$이다($\beta{=}0.5$ 대비 차이가 같은 셀 재실행 노이즈 수준임을 확인).
<!-- ⚠ 재실행 대기(2026-07-23 Yonghee 결정 = β0.3 재실행): 현재 이 "β=0.3" 서술은 논문 인용
     ShapleyFL 값과 잠정 불일치다 — 인용 값(표 [F4] Anchor vs retrain GT의 .767, 부록 C의 CNN C1)은
     아직 β=0.5 rundir 산출:
       · C1 30셀 = git_sha 5cb927b (2026-06-12), track_c1 shapleyfl_from_logs(beta=0.5)
       · 1B_anchor5 3셀 = git_sha 39a0a97 (2026-06-15), track_d shapleyfl_from_logs(beta=0.5)
     β 0.5→0.3 변경 e89af94(2026-06-25)의 재실행 계획이 이 두 셋엔 미반영이었음(phase2_matrix 일부·3B만).
     해소 = β0.3 재실행 확정(REMAINING §1.4; C1 30=RTX3090, 1B_anchor5 3=B200). 착지 후 rundir 교체 +
     F4·부록 C 값 갱신 + 이 주석 삭제. same-game 본문 주장엔 무영향(ShapleyFL=cross-game 비교군). -->
③
ComFedSV의 utility 행렬 low-rank 완성은 사후 일괄 계산이 원형이므로, per-round 점수가
필요한 개입 무대(§5.3)에서는 균등평균 submodel + 손실-감소 효용의 per-round 대용치(원 논문
Eq. 6 기반)를 쓴다 — fidelity 무대(§5.2)는 원형 그대로다. ④ 탐지기 4종은 논문 스펙
재구현이며 스코어 방향만 통일했다.

**B.6 LLM 위협 축에 gradient noise가 없는 이유.** 업데이트에 주입하는 등방(isotropic) 노이즈는
LoRA 인자 공간의 기하에서 검증-gradient 방향 성분이 미미해, CNN에서와 달리 성능·φ 어느
쪽에도 유의한 오염 효과를 만들지 못한다(무대 미성립). LLM 쪽 update-공격 축은 free-rider
계열이 담당한다.

### 부록 C. Fidelity 확장

**C.1 주무대 전 방법(cross-game 포함) vs in-run GT.** ⬚
<!-- 채움: c2fid analysis(8방법 × 위협 8종) + LLM-Main L2(8방법 × answer-swap·clean) — §5.2 표
     F1·F2의 전 방법 확장판. cross-game 계열의 불일치는 근사 오차 + "다른 게임" 성분 합산임을
     본문 §5.1 구분과 함께 다시 명시. -->

| method | `CNN-Main` clean | free-rider(zero) | free-rider(rand) | gradient noise | lf@.15 | lf@.35 | lf@.70 | variable-intensity label-flip | `LLM-Main` answer-swap | clean |
|---|---|---|---|---|---|---|---|---|---|---|
| GTG | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| FedSV | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| ComFedSV | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| ShapleyFL | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |
| FedIF | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ | ⬚ |

**C.2 CNN `CNN-Grid` 시나리오별 vs in-run GT, 전 방법 — Spearman ↑** (3-seed 평균)
<!-- 출처: runs/track_c/fidelity.csv spearman_b group-mean -->

| dataset / scenario | Flirds | Flirds-1st | GTG | FedSV | ComFedSV | ShapleyFL | FedIF | individual utility |
|---|---|---|---|---|---|---|---|---|
| cifar10 / feature_noise | 1.00 | 0.89 | 0.59 | 0.40 | 0.19 | 0.20 | 0.62 | 0.90 |
| cifar10 / iid | 0.95 | 0.54 | 0.21 | 0.22 | 0.12 | 0.18 | 0.45 | 0.69 |
| cifar10 / label_flip | 1.00 | 0.95 | 0.64 | 0.54 | 0.31 | 0.37 | 0.74 | 0.95 |
| cifar10 / label_skew | 0.98 | 0.92 | 0.49 | 0.53 | 0.31 | 0.29 | 0.68 | 0.88 |
| cifar10 / quantity_skew | 0.99 | 0.96 | 0.78 | 0.56 | 0.67 | 0.44 | −0.20 | 0.98 |
| mnist / feature_noise | 0.79 | 0.70 | 0.41 | 0.13 | 0.21 | 0.48 | 0.57 | 0.78 |
| mnist / iid | 0.81 | 0.78 | 0.47 | 0.04 | 0.10 | 0.47 | 0.73 | 0.84 |
| mnist / label_flip | 1.00 | 0.99 | 0.99 | 0.97 | 0.95 | 0.98 | 0.98 | 0.99 |
| mnist / label_skew | 0.71 | 0.61 | 0.33 | −0.01 | 0.14 | −0.02 | 0.41 | 0.63 |
| mnist / quantity_skew | 0.96 | 0.98 | 0.78 | 0.63 | 0.49 | 0.52 | −0.07 | 0.96 |

풀 평균(10시나리오 × 3-seed): Flirds **0.919±.134** > individual utility 0.860±.154 > Flirds-1st
0.832±.194 > GTG 0.569±.343 > FedIF 0.491 > FedSV 0.401 > ShapleyFL 0.391 > ComFedSV
0.348. iid 셀(신호-부재)을 빼면 Flirds 0.928±.136으로 비교군 내 1위가 유지된다.

**C.3 CNN `CNN-Grid` 시나리오별 vs retrain GT, 전 방법 — Spearman ↑** (§5.2 표 F5의 확장)
<!-- 출처: runs/track_c/fidelity.csv spearman_a group-mean; Pearson도 동일 파일 -->

| dataset / scenario | Flirds | Flirds-1st | individual utility | GTG | FedSV | ComFedSV | ShapleyFL | FedIF |
|---|---|---|---|---|---|---|---|---|
| cifar10 / feature_noise | **+0.63** | +0.50 | +0.56 | +0.44 | +0.18 | +0.39 | +0.28 | +0.40 |
| cifar10 / iid | −0.23 | −0.13 | −0.18 | −0.20 | −0.18 | +0.30 | +0.00 | +0.07 |
| cifar10 / label_flip | +0.52 | +0.59 | +0.58 | +0.45 | +0.41 | +0.32 | +0.29 | +0.36 |
| cifar10 / label_skew | −0.18 | −0.07 | +0.14 | +0.44 | +0.40 | +0.28 | +0.12 | +0.19 |
| cifar10 / quantity_skew | +0.57 | +0.56 | +0.57 | +0.70 | +0.70 | +0.72 | **+0.81** | −0.03 |
| mnist / feature_noise | +0.33 | +0.44 | +0.44 | +0.40 | −0.07 | −0.07 | +0.60 | +0.66 |
| mnist / iid | +0.36 | +0.52 | +0.48 | +0.19 | −0.11 | −0.09 | +0.66 | +0.74 |
| mnist / label_flip | +0.96 | +0.97 | +0.97 | +0.97 | +0.96 | +0.94 | +0.96 | +0.96 |
| mnist / label_skew | −0.28 | −0.06 | −0.16 | −0.22 | −0.14 | −0.04 | +0.28 | +0.54 |
| mnist / quantity_skew | **+0.85** | +0.77 | +0.84 | +0.56 | +0.68 | +0.65 | +0.51 | −0.09 |

각주: cifar10/quantity_skew·label_skew처럼 재정규화 계열이 유리한 칸은 retrain GT 게임 자체가
부분집합 크기로 재정규화되는 게임이라는 구조와 정합한다(§4.1의 게임 구분이 심판 선택에도
작용하는 사례). Pearson 표는 같은 파일에서 재생성.

**C.4 CNN `CNN-Grid` Kendall·거리 풀** (10시나리오 × 3-seed; vs in-run GT)

| method | Kendall ↑ | cosine_d ↓ | euclid_d ↓ | max_diff ↓ |
|---|---|---|---|---|
| Flirds | 0.849±.192 | 0.001±.003 | 0.133±.114 | 0.054±.048 |
| Flirds-1st | 0.733±.223 | 0.009±.020 | 0.207±.186 | 0.085±.068 |
| individual utility | 0.757±.187 | 0.009±.018 | 0.216±.164 | 0.090±.060 |
| GTG | 0.470±.302 | 0.074±.078 | 0.175±.124 | 0.117±.110 |
| FedSV | 0.324±.344 | 0.225±.185 | 0.547±.418 | 0.346±.346 |
| ComFedSV | 0.270±.318 | 0.329±.221 | 0.756±.507 | 0.490±.370 |
| ShapleyFL | 0.307±.327 | 0.057±.030 | 1.456±.305 | 0.697±.088 |
| FedIF | 0.399±.333 | 0.078±.071 | 1.092±.192 | 0.601±.097 |

(주무대 Kendall·거리 열은 c2fid·L2 분석 CSV에서 동일 형식으로 ⬚.)

**C.5 부분참여 probe(`Partial-Probe`: $N{=}50$, 5/50, $R{=}200$, IID-clean) — vs in-run GT Spearman ↑**
(LoRA rank별, seed0; Flirds·Flirds-1st는 r16 3-seed 확정)

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
+1.000±.000). §5.6-②의 "구별을 만드는 축은 참여 형태" 클레임의 전표다.

### 부록 D. 안정성(재현성)

**D.1 방법 순위의 seed 간 안정성(CNN `CNN-Grid`).** 같은 방법의 φ 순위를 seed 간
상관($\rho_{xseed}$, 3-seed 쌍별 평균; 10시나리오 풀)으로 잰다:

| method | $\rho_{xseed}$ ↑ |
|---|---|
| in-run GT (자체) | 0.518±.453 |
| **Flirds** | **0.547±.394** |
| Flirds-1st | 0.510±.461 |
| individual utility | 0.474±.448 |
| GTG | 0.311±.441 |
| FedSV | 0.289±.385 |
| FedIF | 0.243±.413 |
| ComFedSV | 0.198±.383 |
| ShapleyFL | 0.124±.431 |

in-run GT 자체의 안정성이 0.518이다 — CNN 무대는 seed마다 클라이언트 기여가 실제로
달라진다. Flirds(0.547)는 **in-run GT의 내재 안정성을 그대로 추종**하고, Monte-Carlo
재구성 계열은 추가 분산으로 0.12~0.31까지 떨어진다.

**D.2 in-run GT 타깃 자기-안정성(수록 무대).** fidelity의 매칭 대상인 in-run GT 자신이 seed를 넘어
재현되는가(in-run GT φ 순위의 seed 간 쌍별 Spearman 평균):

| 무대 | cross-seed $\rho$ ↑ |
|---|---|
| `Anchor` (IID-clean) | **−0.367** |
| `Silo` clean (비IID) | **+0.867** |
| `Silo` answer-swap | +0.933 |
| `Silo` free-rider(zero) | +0.933 |
| `CNN-Grid` (10시나리오 풀) | +0.518±.453 |
| `LLM-Main` | ⬚ <!-- 채움: L1·L2 rundir in-run GT φ 피벗 --> |
| `CNN-Main` 캠페인 | ⬚ <!-- 채움: c2fid rundir in-run GT φ 피벗 --> |

판정: IID-clean 무대의 in-run GT 타깃은 seed-불안정하다(−0.37) — 그 위의 per-seed fidelity
1.000은 "불안정한 참값을 그때그때 정확히 좇는 것"이다. 반면 비IID·오염 무대에서는 타깃이
안정하고(+0.87~+0.93) 그곳의 fidelity가 실재 신호에 대한 재현이다. 본 논문의 fidelity 표는
이 구분과 함께 읽어야 하며(§5.2 각주 ①), 이것이 retrain GT 특성화의 주 표를
LLM-Small·Silo(신호-실재 무대)에 두고 Anchor를 참조로만 쓰는 이유다.

### 부록 E. 비용·규모 보조 실험

**E.1 완전참여 100/100(CNN, Dirichlet(α=1) 무대).** 라운드당 100/100 참여 — coalition 계열은
라운드당 $2^{100}$ 평가라 개입 arm 자체가 존재할 수 없고(전수·MC 어느 쪽도), Flirds는
라운드당 HVP 1회 그대로다(비용 주장 "cohort 크기에 상수"의 극한 무대). sign-gating의 절대
test acc(3-seed):

| arm | clean | label-flip@0.70 | free-rider | gradient noise | **오염-평균** |
|---|---|---|---|---|---|
| vanilla (바닥) | .6527±.003 | .5550±.022 | .6077±.004 | .5497±.005 | .5708 |
| oracle-제외 (천장) | – | .6301±.004 | .6339±.003 | .6339±.003 | .6326 |
| random-제외 (통제) | – | .5216±.034 | .5953±.009 | .5136±.012 | .5435 |
| Flirds · sign-gating | .6440±.005 | .5862±.010 | .6223±.007 | .6102±.001 | .6062 |

완전참여는 vanilla 자체를 강하게 만들지만(gradient noise .5497 vs 10/100 무대의 .2436 — 오염
업데이트가 100클라 평균에 희석), 그럼에도 sign-gating이 오염 3셀 전부에서 vanilla를 상회한다(회수율
평균 +0.56). random-제외는 전 위협에서 vanilla보다 해롭다(회수율 −0.43~−0.47) — 게이트
이득이 "그냥 동수를 뺀 효과"가 아님의 통제 실증.

**E.2 $N{=}10$ LLM, in-run GT ($2^{10}$) (◐ 1-seed).** 전원 참여 $N{=}10$·$R{=}30$에서 in-run GT를
라운드당 $2^{10}{=}1{,}024$ coalition 완전 열거로 계산: 같은-게임 3종 모두 Spearman
1.000(값 수준 Pearson 잔차만 분리 — Flirds $1{-}r \approx 9{\times}10^{-7}$로 최소).
비용: **in-run GT 117,649s(32.7h) vs Flirds 733s = 1/160**(Flirds-1st 240s, individual utility 1,240s).
$N{=}5$($2^5$)에서 $N{=}10$($2^{10}$)으로 갈 때 in-run GT 비용이 실증적으로 폭발하는 동안
Flirds는 라운드당 HVP 1회로 고정임을 보이는 규모 축 실측이다.

**E.3 cross-device anchor($N{=}100$, 10/100, $R{=}30$).** in-run GT per-round $2^{10}$ ≈
**24,975s vs Flirds 157s = 1/159**(Flirds-1st 53s; 참고 coalition 계열 — GTG ≈ 18,100s,
FedSV ≈ 4,970s, ShapleyFL ≈ 24,900s). §5.5 연산수 모델의 30,720 forward 예측과 정합하며,
E.2와 함께 "cohort가 큰 무대에서의 지수 대 상수" 주장을 두 무대에서 교차 실측한다.