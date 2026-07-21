---
type: survey
title: "IRDS→Flirds 수학적 엄밀성 검증: per-sample·per-step 이론의 per-client·per-round 이식 (2026-07-04)"
created: 2026-07-04
updated: 2026-07-04
tags: [survey, math-rigor, irds, flirds, shapley, taylor, proofs]
---

# IRDS→Flirds 수학적 엄밀성 검증

> **이 문서의 질문**: In-Run Data Shapley(IRDS, arXiv:2406.11011)의 per-sample·per-step data
> Shapley 이론이 Flirds의 FL 변형(per-client·per-round) 하에서도 성립하는가. 논문 수록 가능한
> 수준(가정 명시·명제·완전 증명·오차항·반례)으로 정리한다.
>
> **근거 자료** (인용 표기 규약):
> - IRDS 원문: `research-wiki/raw/papers/flirds/Data Shapley in One Training Run.md` (arXiv HTML v3
>   클리핑). 인용은 "IRDS §N / 원문 M행" 형식. 이 세션에서 74–118행·348–395행은 직접 재확인했고,
>   Appendix 세부 행번호는 이론 추출 노트(2026-07-04 정찰 세션, `scratchpad/recon/irds_theory.md`)의
>   검산을 경유한다(경유 인용은 "[노트 경유]" 표기).
> - 코드: `codes/flirds/**` — `flirds_estimator.py`·`in_run_sv.py`·`server.py`·`client.py`·
>   `llm_server.py`는 이 세션에서 전문을 직접 읽고 인용. `exact_sv_llm.py`·`backends/llm.py`·
>   `corruptors.py`의 세부 라인은 코드 감사 노트(`scratchpad/recon/estimator_audit.md`, 2026-07-04,
>   main @ 004d076) 경유.
> - 수학 골격 초안: `scratchpad/math_skeleton.md` (메인 세션). 본 문서는 이를 완전한 명제·증명으로
>   승격하며, 골격·감사 노트의 서술 중 **부정확했던 2곳을 정정**한다(§4.1 Remark R1-c, §4.5 근거②).

---

## 1. 요약

**성립하는 것 (무조건).**

- **(b) in-run 게임의 per-round 분해** — 라운드 미참여 클라이언트의 기여 0은 정의가 아니라
  null-player 공리의 귀결이고, $N$-인 게임의 Shapley가 라운드별 $\lvert P_r\rvert$-인 부분게임
  Shapley의 합으로 정확히 축약된다 (P1, 완전 증명).
- **quadratic surrogate의 닫힌형 Shapley** — 2차 Taylor 근사 게임의 exact Shapley가
  $\phi_k=p_k^r\langle g^r,\delta_k^r\rangle+\tfrac12 p_k^r\langle\delta_k^r,H^r\Delta W^r\rangle$로
  붕괴하며, 구현(`flirds_estimator.py:128,130`)과 **fp32 재결합 오차(~5.8e-12)까지 항별로 일치**한다
  (bit-identical 아님 — 합산 순서 차; "같은 게임"은 §2.1 fp32-로그 전제 위. 수치 확인: gpt2 스모크 §7).
  null-player(free-rider) exact-0과 efficiency는 따름정리 (P2).
- **estimator와 (b) oracle은 같은 게임을 본다** — 양쪽 다 고정가중($S$-비의존 $p_k^r$) 게임이고,
  차이는 라운드별 Taylor 절단뿐 (코드 감사 §0 + P2).
- **grand coalition telescoping** — $U_b(\mathcal N)=\ell(w_R)-\ell(w_0)$이 근사 없이 정확 (G1).
- **merge-consistency 보조정리 (L3)** — 2-additive 게임을 임의 분할로 병합하면 블록 Shapley = 블록
  내 원소 Shapley 합이 **무조건** 성립한다 (순수 대수; P4의 L3). *단 이를 Flirds에 적용한
  per-sample↔per-client 브리지는 조건부다 — 아래 "조건부" 절 참조.*
- **LoRA 부분공간 전개의 무해성** — 이론 전체를 훈련가능 좌표의 함수 $\tilde\ell$로 진술하면
  P1–P7이 문자 그대로 성립하며 부분공간 제한이 추가 근사를 만들지 않는다 (P8).

**조건부로 성립하는 것.**

- **per-sample↔per-client 브리지** — L3를 Flirds에 적용하려면 (i) 로컬 **1-step full-batch mean-CE
  극한**(K>1 minibatch 운영 레짐 **아님** — 실제 무대는 `llm_server.py:44` `max_steps` 다중스텝)과
  (ii) **2차 surrogate 수준**이 필요하다. 이 두 조건에서 per-sample IRDS surrogate 값의 클라별 합 =
  per-client Flirds 값이 정확하고(P4b), 정확 게임 갭은 3차 잔차로 bound(P4c). 또 분모 상쇄(로컬-손실
  reduction 분모 = FedAvg 집계 가중)는 **CNN mean-CE(샘플=이미지)** 전용이고 **LLM token-mean CE**
  에서는 분모(토큰수)≠가중(시퀀스수)이라 공통-lr sum이 성립하지 않는다 (P4b Remark;
  `backends/llm.py:78` vs `llm_server.py:85`).
- **Taylor 잔차 bound** — $\ell\in C^3$ + 3차 도함수 유계 하에 
  $\lvert u_r(S)-\hat u_r(S)\rvert\le\frac{M_3}{6}\lVert\Delta_S\rVert^3$과 Shapley 수준 전파가
  성립하지만, **P3은 상수 $M_2/M_3$가 실측 불가한 상계**이고 그 타이트함·지수는 아직 미관측(gpt2
  스모크는 노이즈 바닥, loglog 기울기 0.30)이라 §7 1B 실측에 조건부다. IRDS 원문에는 이 잔차의
  형식화가 없어(비형식 $O(\eta^2)$뿐) P3이 이를 명시적으로 형식화한다 (P3, §7).
- **고정가중 vs 재정규화 게임의 순위 동일성** — 등$n$·선형(1차 surrogate) 극한에서 **라운드별**로
  (그리고 **전원참여**일 때 라운드 합산까지) 두 게임의 순위가 증명 가능하게 동일하다(P5b). 그러나
  **비등$n$에서는 1차에서도 순위가 뒤집히는 수치 반례가 존재**하고(P5c), **부분참여**(등$n$·선형이어도
  라운드별 희석상수·가중 불일치)와 **2차 곡률**(등$n$·$H$-직교여도)에서도 어긋날 수 있다 (P5).
- **momentum=0의 역할** — 클라 optimizer가 라운드마다 새로 생성(stateless)되므로 클라-수준 게임은
  로컬 momentum이 있어도 정의 가능하다. **클라** momentum=0(F9)이 실제로 load-bearing한 것은 Flirds가
  채택하지 않는 **fine per-step(K>1) 게임**뿐이다 — P4b가 서는 1-step full-batch 극한에선 첫 SGD
  스텝이 momentum-무관(PyTorch buffer=grad)이라 vacuous. 실현-run telescoping·무상태성은 클라 momentum이
  아니라 **server 무상태성(F6; 별개 knob)**이 보증한다 (P7).

**깨지는 것 / 정의로 우회하는 것.**

- **3차 이상에서 merge-consistency 실패** — 반례 존재(3-샘플 unanimity 병합). 단위 변경(샘플→클라)이
  값을 바꾸는 진짜 어긋남이며, 크기는 3차 잔차로 bound (P4-iv).
- **K>1 로컬 스텝의 granularity** — per-round 전개는 IRDS-faithful per-step 전개와 **다른 게임**이다.
  오차가 아니라 게임 정의 선택이며, IRDS 자신의 per-step 선택과 같은 지위 (P4-iv, P6).
- **path-dependence** — (b) 게임은 frozen trajectory 위의 정의이고 후속 궤적 변화를 반영하지 않는다.
  IRDS의 게임-재정의 논리(근사가 아니라 정의; sequence-function 공리화 미해결 포함)가 라운드
  수준으로 그대로 이전되고, (a) retrain 게임과의 관계는 경험 보고로 한정된다
  (1B Spearman +1.000 / 3B +0.900) (P6).

---

## 2. 시스템 모델과 표기

### 2.1 FL 프로토콜 확정 사실 (코드 명문화)

아래는 전부 이 세션에서 코드로 직접 확인한 사실이다 (코드 감사 노트 §7과 일치).

| # | 사실 | 근거 |
|---|---|---|
| F1 | 동기식 FedAvg: 라운드 루프 내 선택→순차 로컬 학습(전원 같은 $w_r$에서)→일괄 집계 | `fl/server.py:43–67` |
| F2 | 부분 참여: $K=\max(1,\mathrm{round}(\texttt{sample\_frac}\cdot N))$명 균등 비복원 추출 | `server.py:41,46` (Python `round`=banker's rounding — 경계값 유의) |
| F3 | 모든 참여 클라가 라운드 시작 상태 $w_r$에서 로컬 학습 시작 | `client.py:16`, `llm_server.py:38` |
| F4 | 클라 산출물 = 모델차 $\delta_k^r=w_k^{\text{local}}-w_r$ | `client.py:26`, `llm_server.py:55–56` |
| F5 | 집계 가중 $p_k^r=n_k/\sum_{j\in P_r}n_j$ (참여자 정규화; $n_k$= 클라 표본수) | `server.py:54–56`; $n_k$: CNN `server.py:95`, LLM `llm_server.py:85` |
| F6 | server learning rate 없음: $w_{r+1}=w_r+\sum_{k\in P_r}p_k^r\delta_k^r$ 순수 가중합 | `server.py:60–63` |
| F7 | LLM은 LoRA(=requires_grad) 파라미터만 로그·집계; frozen base는 로그 밖 | `llm_server.py:83–84`, docstring :6–14 |
| F8 | 로그 계약 $\texttt{logs}=[(w_r,\{k:(\delta_k^r,n_k)\}_{k\in P_r})]_{r<R}$; $w_r$는 라운드 시작 상태의 clone | `server.py:44–45,64–65`; CNN `:117–120`, LLM `llm_server.py:91–94` |
| F9 | 로컬 optimizer = plain SGD, momentum 0, 상수 lr | `client.py:8,18`; `llm_server.py:45,52` |
| F10 | 클라 stateless: optimizer(·trainer)가 **매 라운드 새로 생성** — 라운드 간 상태 없음 | `client.py:18` (SGD 새 인스턴스), `llm_server.py:49–53` (SFTTrainer 새 인스턴스) |
| F11 | 개입 seam(select/weights/delta_transform)은 기본 None = seam 이전 루프와 bit-identical | `server.py:30–36` |
| F12 | raw logs는 디스크에 영속화되지 않음(파생 phi/metrics만 저장) — 로그-수준 재측정은 fresh run 필요 | 감사 노트 §9 (`run_logger.py` grep) |

정밀도: estimator는 params·deltas를 fp32로 강제 캐스팅(`flirds_estimator.py:101,103`)하고, (b)
oracle은 로그 dtype을 그대로 신뢰한다(`in_run_sv.py:36–49` 캐스팅 없음). 현행 파이프라인은 fp32
훈련(LLM `llm_server.py:46` bf16/fp16 False)이므로 일치하지만, "같은 게임" 주장은 **로그가 fp32라는
프로토콜 전제** 위에 있다 (감사 노트 §8; §6 유의점 목록).

**정칙성 전제 (전 명제 공통)**: 모든 라운드에서 $P_r\ne\emptyset$이고 $\sum_{j\in P_r}n_j>0$ (참여자
표본수 합이 양). F2($K=\max(1,\cdot)$)가 $P_r\ne\emptyset$를, 비어있지 않은 로컬 데이터셋이
$\sum n_j>0$을 보장한다. 이 전제가 깨지면 $p_k^r$(F5·D1)이 정의되지 않고 코드가 NaN을 낸다
(`flirds_estimator.py:100` `nr.sum()==0`; `in_run_sv.py:32` 0-나눗셈). 표본수 0인 참여 클라 개별은
$p_k=0\Rightarrow a_k=0\Rightarrow\phi_k=0$ (P2-1과 정합). 재정규화 foil($\tilde u_r$, §4.5)은
추가로 $S\cap P_r$의 표본수 합이 양이어야 정의된다.

### 2.2 표기

- 클라이언트 집합 $\mathcal N=\{1,\dots,N\}$ (코드는 0-기반 인덱스; 문서는 1-기반).
- 라운드 $r=0,\dots,R-1$; 참여 코호트 $P_r\subseteq\mathcal N$, $K_r:=\lvert P_r\rvert$.
- 검증 손실 $\ell(\cdot)$ = `loss_fn` (고정 val 셋; backends 빌더). $g^r:=\nabla\ell(w_r)$,
  $H^r:=\nabla^2\ell(w_r)$ (**true Hessian**; GGN/Fisher 아님 — `flirds_estimator.py:15` "true
  Hessian, as IRDS"; IRDS 원문 111행과 동일 선택).
- 가중 delta $a_k^r:=p_k^r\delta_k^r$; coalition 이동 $\Delta_S^r:=\sum_{k\in S\cap P_r}a_k^r$;
  실현 라운드 이동 $\Delta W^r:=\Delta_{P_r}^r=\sum_{j\in P_r}a_j^r$, 즉 $w_{r+1}=w_r+\Delta W^r$ (F6).
- $q_{ij}^r:=(a_i^r)^\top H^r a_j^r$ (클라 쌍 상호작용). $H^r$ 대칭($\ell\in C^2$, Schwarz 정리)이므로
  $q_{ij}^r=q_{ji}^r$.
- 게임 $v:2^{\mathcal N}\to\mathbb R$, $v(\emptyset)=0$. $\phi_i(v)$ = Shapley 값 (§2.4).
- 라운드 첨자 $r$은 혼동이 없으면 생략한다.

### 2.3 부호 규약 (고정)

IRDS 원문에는 **암묵적 부호 반전**이 있다: 형식 정의는 utility = val-loss **변화**(유익한 데이터
⇒ loss 감소 ⇒ $U<0$, $\phi<0$; 원문 76–79행의 $U^{(t)}$ 정의와 Thm 3 닫힌형
$-\eta_t\nabla\ell_{\mathrm{val}}\cdot\nabla\ell_z$ 모두 이 방향)인데, 실험 절(§5.3.1, 281행)은
"negatively valued data"를 저품질로 취급해 제거한다 — 즉 응용에서는 부호를 반전한 점수(loss 감소 =
양의 가치)를 쓰면서 그 반전을 명시하지 않는다 (이론 추출 노트 §1.4 [확인+검산]).

**본 문서와 Flirds 코드의 규약**: $\phi$는 val-loss **변화의 귀속**이다 — **유익한 클라이언트
⇒ $\phi_k<0$** (`flirds_estimator.py:27–28` "a client that reduces val loss -> phi_k < 0";
`in_run_sv.py` 게임 정의 동일). 순위·fidelity 비교는 방향만 일치시키면 부호 선택과 무관하다.
논문 제시 시 반전 여부는 §9 결정 사항.

### 2.4 Shapley 예비 정리

**정의** (IRDS Def 1, 원문 56–57행과 동일 형): 유한 게임 $v:2^{\mathcal N}\to\mathbb R$,
$v(\emptyset)=0$에 대해

$$\phi_i(v):=\sum_{S\subseteq\mathcal N\setminus\{i\}}\frac{\lvert S\rvert!\,(N-\lvert S\rvert-1)!}{N!}\big[v(S\cup\{i\})-v(S)\big].$$

**보조정리 L0 (기초 성질).** (a) 순열형 동치: $\pi$를 $\mathcal N$의 순열,
$\mathrm{Pre}_i(\pi):=\{j:\pi\text{에서 }j\text{가 }i\text{보다 앞}\}$이라 하면
$\phi_i(v)=\frac{1}{N!}\sum_\pi[v(\mathrm{Pre}_i(\pi)\cup i)-v(\mathrm{Pre}_i(\pi))]$.
(b) 계수합 1: $\sum_{S\subseteq\mathcal N\setminus i}\frac{\lvert S\rvert!(N-\lvert S\rvert-1)!}{N!}=1$.
(c) 선형성: $\phi_i(\alpha v+\beta w)=\alpha\phi_i(v)+\beta\phi_i(w)$.
(d) Efficiency: $\sum_i\phi_i(v)=v(\mathcal N)$.

*증명.* (a) 고정 $S\subseteq\mathcal N\setminus i$에 대해 $\mathrm{Pre}_i(\pi)=S$인 순열의 수는
$S$의 원소를 $i$ 앞에 배열하는 $\lvert S\rvert!$가지 × 나머지 $N-\lvert S\rvert-1$개를 $i$ 뒤에
배열하는 $(N-\lvert S\rvert-1)!$가지 $=\lvert S\rvert!(N-\lvert S\rvert-1)!$. 이를 $N!$로 나눠
합치면 조합형과 순열형이 일치한다.
(b) 크기 $s$인 $S$는 $\binom{N-1}{s}$개이므로 합은
$\sum_{s=0}^{N-1}\binom{N-1}{s}\frac{s!(N-1-s)!}{N!}=\sum_{s=0}^{N-1}\frac{(N-1)!}{N!}=\sum_{s=0}^{N-1}\frac1N=1$.
(c) 정의식이 $v$에 선형.
(d) 순열형에서 고정 $\pi$에 대해 $\sum_i[v(\mathrm{Pre}_i\cup i)-v(\mathrm{Pre}_i)]$는 $\pi$ 순서대로
플레이어를 추가하는 telescoping 합 $=v(\mathcal N)-v(\emptyset)=v(\mathcal N)$. 모든 $\pi$에 대한
평균도 같다. $\blacksquare$

---

## 3. 게임 정의

### 3.1 (b) in-run 게임 (고정가중)

**정의 D1 (라운드 부분게임).** frozen logs (F8) 위에서, $S\subseteq\mathcal N$에 대해

$$u_r(S):=\ell\Big(w_r+\sum_{k\in S\cap P_r}p_k^r\,\delta_k^r\Big)-\ell(w_r)
=\ell(w_r+\Delta_S^r)-\ell(w_r),\qquad u_r(\emptyset)=0.$$

**정의 D2 (글로벌 (b) 게임).** $U_b(S):=\sum_{r=0}^{R-1}u_r(S)$.

코드 대응: D1·D2는 `oracle/in_run_sv.py:3–9` (docstring 수식), `_round_weight`
(`in_run_sv.py:30–33` — 분모 = $P_r$ 전체의 $\sum n$, **$S$-비의존**), `_perturbed_params`
(`:43–50`), `in_run_utility` (`:53–67`)와 문자 그대로 일치한다. $2^N$ 전수 순회는
`_coalition_utilities`/`in_run_shapley` (`:101–153`), per-round 경로는 `in_run_shapley_perround`
(`:156–196`).

**핵심 설계 선택 (고정가중).** $p_k^r$의 분모는 어떤 $S$에 대해서도 재정규화되지 않는다.
따라서 $S\mapsto\Delta_S^r$은 **coalition-독립 계수의 가산 사상** $\Delta_S=\sum_{k\in S\cap P_r}a_k$
이며, $\{\Delta_S\}_{S\subseteq P_r}$는 실현 이동 $\Delta W^r$의 가산 분해 조각들이다. 이것이 IRDS의
sum-형 update 구조(원문 74행: $w_{t+1}=w_t-\eta_t\sum_{z\in\mathcal B_t}\nabla\ell(w_t,z)$; $\eta_t$가
$\lvert S\rvert$-불변 — 원문 81행 $\widetilde w_{t+1}(S)$ 정의)의 정확한 클라-수준 대응이다.
필수성(이 구조가 없으면 무엇이 깨지는지)은 P4-i, 대안 게임과의 비교는 P5.

estimator(`core/flirds_estimator.py`)가 근사하는 게임도 **정확히 D1·D2다** — 같은 logs, 같은
`loss_fn`, 같은 $S$-비의존 $p_k^r$ (`flirds_estimator.py:98–100` vs `in_run_sv.py:30–33`), 같은 전개
지점 $w_r$. 유일한 차이는 라운드별 2차 Taylor 절단 (P2·P3). — 코드 감사 §0 Q1 판정과 일치.

### 3.2 Grand coalition telescoping

**명제 G1.** 임의 $S\supseteq P_r$에 대해 $u_r(S)=\ell(w_{r+1})-\ell(w_r)$ (정확). 따라서
$U_b(\mathcal N)=\ell(w_R)-\ell(w_0)$이고, L0(d)에 의해
$\sum_{k}\phi_k(U_b)=\ell(w_R)-\ell(w_0)$ — 전 클라 Shapley 합 = 전체 run의 val-loss 변화 (정확).

*증명.* $S\supseteq P_r\Rightarrow S\cap P_r=P_r\Rightarrow\Delta_S^r=\Delta W^r$. F6에 의해
$w_r+\Delta W^r=w_{r+1}$이므로 $u_r(S)=\ell(w_{r+1})-\ell(w_r)$. $r$에 대해 합하면 telescoping으로
$U_b(\mathcal N)=\ell(w_R)-\ell(w_0)$. Efficiency는 L0(d). $\blacksquare$

IRDS 대응: 원문 86행의 telescoping 서술("sum of individual data points' Shapley values equals the
overall loss reduction")과 동일 논리 — 그쪽도 grand coalition에서 $\widetilde w_{t+1}(\mathcal B_t)=w_{t+1}$이므로 정확 (이론 추출 노트 §1.4 [검산]).

주의: `in_run_sv.py:7–8` docstring이 이 성질을 자기 문서화한다("the perturbation equals the
realized update when S ⊇ P_r") — 역으로 $S\not\supseteq P_r$이면 섭동은 "S만 참여한 FedAvg"가
**아니다**. 그 counterfactual은 (a) 게임이다 (§3.3).

### 3.3 (a) retrain 게임과의 구분

(a) 게임: $U_a(S)$ = $S$만으로 처음부터 FedAvg를 **재훈련**한 최종 모델의 성능
(`oracle/exact_sv_llm.py:75–96` [감사 노트 §6 경유]; PRIMARY = $-$val-loss). 재훈련이므로 FedAvg
분모가 자연히 $\sum_{j\in S}n_j$로 재정규화되고(감사 노트 §6.1 [해석]), utility는 궤적 전체가 $S$에
의존하는 **다른 게임**이다. (a)와 (b)는 코드도 분리되어 있다(protocol 4.3; `in_run_sv.py:15–19`).
(a)↔(b) 관계는 이론이 아니라 실증 질문이며 P6에서 다룬다.

---

## 4. 명제와 증명

### 4.1 P1 — per-round 분해와 부분 참여의 공리적 정합성

**가정.** A1 (frozen logs; F8). — 그 외 가정 불필요 (특히 아래 Remark R1-c 참조).

**보조정리 L1 (null-player 제거 / 부분게임 축약).** $v:2^{\mathcal N}\to\mathbb R$,
$v(\emptyset)=0$이고 어떤 $T\subseteq\mathcal N$($\lvert T\rvert=t$)에 대해 모든
$S\subseteq\mathcal N$에서 $v(S)=v(S\cap T)$라 하자. 그러면

(i) 모든 $i\notin T$에 대해 $\phi_i^{\mathcal N}(v)=0$;
(ii) 모든 $i\in T$에 대해 $\phi_i^{\mathcal N}(v)=\phi_i^{T}(v|_T)$ ($v|_T$ = $T$ 위 제한 게임).

*증명.* (i) $i\notin T$이면 $(S\cup i)\cap T=S\cap T$이므로 $v(S\cup i)=v(S\cap T)=v(S)$ — 모든
marginal contribution이 0이고, L0(b)에 의해 Shapley는 marginal들의 볼록결합이므로 $\phi_i=0$.

(ii) $i\in T$이면 $(S\cup i)\cap T=(S\cap T)\cup i$이므로 marginal은
$m(S):=v(S\cup i)-v(S)=v|_T\big((S\cap T)\cup i\big)-v|_T(S\cap T)$ — 즉 $S\cap T$에만 의존한다.

*하위 보조정리:* $\mathcal N$의 균등 랜덤 순열 $\pi$가 $T$에 유도하는 상대순서 $\pi|_T$는 $T$의
$t!$개 순서 위에서 균등분포다. 증명(계수): 고정된 $T$-순서 $\sigma$에 대해 $\pi|_T=\sigma$인 $\pi$의
수는, $T$ 원소들이 차지할 위치 $\binom{N}{t}$가지 × 그 위치들에의 배치는 $\sigma$가 유일하게 강제 ×
나머지 $N-t$개의 자유 배열 $(N-t)!$가지 $=\binom{N}{t}(N-t)!=N!/t!$ — $\sigma$와 무관하므로 균등.
또한 정의상 $\mathrm{Pre}_i(\pi)\cap T=\mathrm{Pre}_i^{T}(\pi|_T)$ ($\pi$에서 $i$보다 앞선 $T$-원소
집합 = 유도순서에서 $i$보다 앞선 집합).

이제 L0(a)의 순열형으로

$$\phi_i^{\mathcal N}(v)=\mathbb E_{\pi}\big[m(\mathrm{Pre}_i(\pi))\big]
=\mathbb E_{\pi}\Big[v|_T\big((\mathrm{Pre}_i(\pi)\cap T)\cup i\big)-v|_T\big(\mathrm{Pre}_i(\pi)\cap T\big)\Big]
=\mathbb E_{\sigma\sim\mathrm{Unif}(S_T)}\Big[v|_T\big(\mathrm{Pre}_i^T(\sigma)\cup i\big)-v|_T\big(\mathrm{Pre}_i^T(\sigma)\big)\Big],$$

마지막 등호는 하위 보조정리(유도순서의 균등성). 우변은 $T$-게임 Shapley의 순열형이므로
$=\phi_i^T(v|_T)$. $\blacksquare$

**명제 P1.** A1 하에서:
(i) 각 $u_r$은 $u_r(S)=u_r(S\cap P_r)$을 만족한다 (D1의 구성상; `in_run_sv.py:59`가 $S\cap P_r$만
섭동에 반영).
(ii) $\phi_k(U_b)=\sum_{r=0}^{R-1}\phi_k^{\mathcal N}(u_r)$ (L0(c) 선형성; $U_b$가 유한 합이므로).
(iii) L1을 $T=P_r$로 적용하면

$$\phi_k(U_b)=\sum_{r:\,k\in P_r}\phi_k^{P_r}\big(u_r|_{P_r}\big),$$

즉 **비참여 라운드의 기여 0은 정의가 아니라 정리**(L1(i))이고, $N$-인 게임의 Shapley는 라운드별
$K_r$-인 부분게임의 exact Shapley 합으로 정확히 축약된다. $\blacksquare$

**코드 대응.** `in_run_shapley_perround` (`in_run_sv.py:156–196`)가 정확히 (iii)을 구현: 라운드마다
$2^{K_r}$ 부분게임 utility(`:185–189`) 후 $K_r$-인 exact Shapley(`:190–195`). estimator의 라운드
합산(참여 라운드만, 참여횟수 정규화 없음 — `flirds_estimator.py:97,119,131` + docstring `:22–24`)도
(iii)의 구조와 일치한다. **수치 확인**: per-round 분해 = $2^N$ 전수 oracle — task 7c(CNN 결정적)
$\Delta\phi\approx3\times10^{-16}$, 그리고 **본 문서 gpt2 스모크(§7)에서 재확인 max_abs $=3.93\times10^{-7}$**
(fp32 forward 노이즈 바닥; `in_run_sv.py:165–166` docstring이 `phase2_crossdevice_oracle_smoke` 언급).

**IRDS 대응.** 원문 83–84행: 배치 밖 샘플의 $\phi=0$과 augmented game의 값 보존을 Wang & Jia
(arXiv:2302.11431) Thm 5 **인용으로** 처리. 우리는 L1로 직접 증명해 외부 인용 의존을 제거했다.
스텝(라운드) 합산의 linearity 논리는 원문 86행과 동일.

**Remarks.**
- **R1-a (성립 경계)**: L1은 $v(S)=v(S\cap T)$ 구조만 요구한다. 이 구조는 server-side가 무상태일 때
  성립한다 — server momentum 등으로 라운드 $r$의 실현 이동이 과거 라운드 delta에 의존해도 D1의
  **정의 자체**는 유지되지만, 게임의 의미론이 훼손된다 (P7-ii).
- **R1-b (부분 참여 계약)**: estimator의 `n_clients` 자동 추론(`flirds_estimator.py:90–91`,
  $1+\max$)은 최고 인덱스 클라가 전 라운드 비참여면 $\phi$ 배열이 짧아진다. docstring `:73–74`가
  부분참여 시 명시 전달을 요구 — P1의 수학과 무관한 **호출부 규율** 사항 (감사 노트 부록 A.3).
- **R1-c (감사 노트 정정)**: 감사 노트 §2.3은 "만약 분모가 $S$-의존이었다면 이 분해는 성립하지
  않았다"고 서술했으나 이는 **과잉 주장**이다. 재정규화 게임
  $\tilde u_r(S)=\ell(w_r+\sum_{k\in S\cap P_r}\frac{n_k}{\sum_{j\in S\cap P_r}n_j}\delta_k)-\ell(w_r)$
  역시 $S\cap P_r$만의 함수이므로 L1·P1은 그대로 적용된다. 고정가중이 load-bearing한 곳은 P1이
  아니라 **P2의 가산 구조와 P5의 게임 의미론**이다.

### 4.2 P2 — quadratic surrogate의 닫힌형 Shapley

**가정.** A1; A2 (고정가중, D1); A3: $\ell$은 $w_r$ 근방에서 $C^2$ ($H^r$ 존재·대칭).

**A3의 조각적-매끄러움 유의 (backend별).** LLM 트랙(attention·softmax·GELU·CE)은 진성 $C^\infty$이라
A3가 무조건 성립한다. 그러나 **CNN 트랙**(LeNet5·FedSVCNN)은 forward 전체가 ReLU+max-pool
(`models/cnn.py:22–26,42–46` `F.relu`/`F.max_pool2d`)이라 합성 val loss가 **전역적으로 $C^0$일 뿐**
이고 활성 영역 내부에서만 $C^\infty$이다. 따라서 CNN에서 A3의 $H^r$은 "$w_r$이 속한 활성 영역 내부의
**region-local Hessian**"으로 읽는다 (generic $w_r$에서 well-defined; `jvp∘grad`가 정확히 이 영역-국소
Hessian을 반환). A3/A4의 근방 $\mathcal U_r$은 하나의 활성 영역 안에 있다고 가정하며, 라운드 이동
$\Delta_S$가 ReLU kink를 가로지르면($K$-스텝 큰 이동에서 발생 가능 — §4.3) 추가 오차가 생긴다.
**P2의 대수적 결론은 $w_r$에서 대칭 $H^r$ 하나만 존재하면 유지되므로 이 조각성에 깨지지 않는다**;
영향받는 것은 P3/A4의 근방형 가정(활성 영역 경계를 넘는 이동)이다. Schwarz 대칭(아래 증명 (2)의
$q_{ij}=q_{ji}$)도 영역 내부에서만 유효하다.

**정의 D3 (라운드 surrogate 게임).** $S\subseteq P_r$에 대해

$$\hat u_r(S):=\langle g^r,\Delta_S\rangle+\tfrac12\,\Delta_S^\top H^r\Delta_S,\qquad
\Delta_S=\sum_{k\in S}a_k,\ a_k=p_k^r\delta_k^r.$$

(1차 전용 변형 $\hat u_r^{(1)}(S):=\langle g^r,\Delta_S\rangle$ — estimator의
`second_order=False` 경로, `flirds_estimator.py:71,114–115`.)

**명제 P2.** $k\in P_r$에 대해

$$\phi_k(\hat u_r)=\langle g^r,a_k\rangle+\tfrac12\,a_k^\top H^r\,\Delta W^r
= p_k^r\langle g^r,\delta_k^r\rangle+\tfrac12\,p_k^r\big\langle\delta_k^r,\,H^r\Delta W^r\big\rangle.$$

*증명.* $\Delta_S=\sum_{k\in S}a_k$를 전개하면

$$\hat u_r(S)=\sum_{k\in S}\langle g^r,a_k\rangle+\tfrac12\sum_{i\in S}\sum_{j\in S}q_{ij}
=\underbrace{\sum_{k\in S}\Big(\langle g^r,a_k\rangle+\tfrac12 q_{kk}\Big)}_{v_{\mathrm{add}}(S)}
+\underbrace{\sum_{\substack{\{i,j\}\subseteq S\\ i\ne j}}q_{ij}}_{v_{\mathrm{quad}}(S)},$$

여기서 비대각 쌍 $\{i,j\}$의 두 순서쌍 항 $\tfrac12(q_{ij}+q_{ji})=q_{ij}$로 접는 데 $H^r$ 대칭(A3)을
사용했다. L0(c)에 의해 $\phi_k(\hat u_r)=\phi_k(v_{\mathrm{add}})+\phi_k(v_{\mathrm{quad}})$.

(1) *가산부*: $v_{\mathrm{add}}$에서 $k$의 모든 marginal은 상수
$c_k:=\langle g^r,a_k\rangle+\tfrac12 q_{kk}$이고, L0(b)에 의해 Shapley는 이 상수들의 볼록결합이므로
$\phi_k(v_{\mathrm{add}})=c_k$.

(2) *쌍대부*: $v_{\mathrm{quad}}=\sum_{i<j}q_{ij}\,\pi_{ij}$, $\pi_{ij}(S):=\mathbf 1[\{i,j\}\subseteq S]$
(2-인 unanimity 게임). $K:=K_r$-인 게임에서 $\phi_i(\pi_{ij})$를 직접 계산한다: $i$의 marginal은
$\pi_{ij}(S\cup i)-\pi_{ij}(S)=\mathbf 1[j\in S]$이므로

$$\phi_i(\pi_{ij})=\sum_{s=1}^{K-1}\binom{K-2}{s-1}\frac{s!(K-s-1)!}{K!}
=\sum_{s=1}^{K-1}\frac{(K-2)!\,s}{K!}=\frac{1}{K(K-1)}\sum_{s=1}^{K-1}s=\frac12,$$

($j$를 포함하는 크기 $s$ 집합의 수 $=\binom{K-2}{s-1}$;
$\binom{K-2}{s-1}s!(K-s-1)!=\frac{(K-2)!}{(s-1)!(K-s-1)!}s!(K-s-1)!=(K-2)!\,s$;
$\sum_{s=1}^{K-1}s=\frac{K(K-1)}2$.) 대칭으로 $\phi_j(\pi_{ij})=\tfrac12$;
$l\notin\{i,j\}$는 marginal이 항상 0이므로 $\phi_l=0$ (L1(i)의 특수형). L0(c)로

$$\phi_k(v_{\mathrm{quad}})=\sum_{j\ne k}q_{kj}\cdot\tfrac12=\tfrac12\sum_{j\ne k}q_{kj}.$$

(3) 합산: $\phi_k(\hat u_r)=\langle g^r,a_k\rangle+\tfrac12 q_{kk}+\tfrac12\sum_{j\ne k}q_{kj}
=\langle g^r,a_k\rangle+\tfrac12\sum_{j\in P_r}q_{kj}
=\langle g^r,a_k\rangle+\tfrac12 a_k^\top H^r\big(\sum_{j\in P_r}a_j\big)$. $\blacksquare$

**경계 $K_r=1$.** 증명 (2)의 쌍대부는 $\phi_i(\pi_{ij})$에서 $K(K-1)$로 나누므로 $K_r\ge2$ 전용이다.
$K_r=1$(단일 참여; F2의 `round`가 $\le1$로 떨어지는 경계)에서는 $i\ne j$ 쌍이 없어
$v_{\mathrm{quad}}\equiv0$이고 $\phi_k=\langle g^r,a_k\rangle+\tfrac12 q_{kk}$인데, 단일 참여자라
$\Delta W^r=a_k$이므로 $\tfrac12 a_k^\top H^r\Delta W^r=\tfrac12 q_{kk}$로 닫힌형과 정확히 일치한다
(경계에서도 결론 유지).

**따름정리 P2-1 (null player; free-rider exact-0).**
(i) $\delta_k=0\Rightarrow a_k=0\Rightarrow\phi_k(\hat u_r)=0$ **정확히** (내적 두 항 모두 0 텐서와의
내적; `flirds_estimator.py:128,130`). 또 $a_k=0$은 $\Delta W^r$을 오염시키지 않는다(`:109`).
(ii) exact 게임에서도: $\delta_k=0$이면 `_perturbed_params`가 $p_k\cdot 0$ 덧셈으로 bit-identical
파라미터를 만들고(`in_run_sv.py:47–49`), forward가 **bitwise-결정적**이면 $u_r(S\cup k)=u_r(S)$ —
모든 marginal이 정확히 0, $\phi_k(u_r)=0$. $u_r(S)$와 $u_r(S\cup k)$는 **별도 forward 호출**로
계산되므로(`in_run_sv.py:119–129` 각 coalition 재계산) 이 상쇄에는 반복 호출의 bit 재현성이 필요하다:
CNN은 `server.py:92` cudnn_deterministic ⇒ **성립**, 그러나 **LLM은 두 별도 GPU forward가 비결정
커널 하에 ~$10^{-7}$ 어긋날 수 있어 미검증**(`llm_server.py:82` "no cudnn-det"; §6-3). free-rider
zero-delta 경로는 `data/corruptors.py:109–128` [감사 노트 §4 경유]; baseline의 "free-rider $\phi$
exact-0: Flirds/oracle" 관측(루트 CLAUDE.md baseline, 2026-06-07)과 부합(CNN). **주의**: (ii)의
"정확히 0"은 **CNN에서만 확립**되며 LLM (b)-oracle exact-0은 현재 미검증 GPU forward 결정성에 조건부다.
견고한 진술은 estimator 측 **대수적** exact-0($0\cdot(\cdot)$; `flirds_estimator.py:128,130`)으로,
결정성과 무관하다.

**따름정리 P2-2 (efficiency).**
$\sum_{k\in P_r}\phi_k(\hat u_r)=\langle g^r,\Delta W^r\rangle+\tfrac12(\Delta W^r)^\top H^r\Delta W^r=\hat u_r(P_r)$.
(계산: $\sum_k\langle g,a_k\rangle=\langle g,\Delta W\rangle$;
$\tfrac12\sum_k\sum_jq_{kj}=\tfrac12\Delta W^\top H\Delta W$.) 라운드 합산 후:
$\sum_k\hat\phi_k=\sum_r\hat u_r(P_r)$, 그리고 P3에 의해
$\big\lvert\sum_r\hat u_r(P_r)-(\ell(w_R)-\ell(w_0))\big\rvert\le\sum_r\frac{M_3^r}{6}\lVert\Delta W^r\rVert^3$ (G1과 결합).

**코드 대응 (항별).**

| 이론 항 | 코드 | 라인 (`core/flirds_estimator.py`) |
|---|---|---|
| $p_k^r$ | `pr` ($P_r$ 전체 분모) | :98–100 |
| $\Delta W^r=\sum_j p_j\delta_j$ | `dW` | :109 |
| $(g^r,\ u^r{=}H^r\Delta W^r)$ | `jvp(grad(vloss),(params,),(dW,))` — HVP 라운드 1회 | :111 |
| $p_k\langle g^r,\delta_k\rangle$ | 1차 항 | :128 |
| $\tfrac12 p_k\langle\delta_k,u^r\rangle$ | 2차 항 | :130 |
| $\sum_{r:\,k\in P_r}$ | 라운드 루프, 참여횟수 정규화 없음 | :97,131; docstring :22–24 |

**수치 확인 (gpt2 스모크, §7)**: 닫힌형 $\phi$ = `flirds_estimator.flirds_values` max_abs
$=5.83\times10^{-12}$ (fp32 재결합; bit-identical 아님); $\hat u^{(2)}$ 게임의 exact $2^5$ Shapley =
닫힌형 max_abs $=7.61\times10^{-12}$ — P2 유도(2차 게임 Shapley $=\tfrac12 p_k\langle\delta_k,H\Delta W\rangle$)의
대수적 정확성을 확인.

닫힌형이 **HVP 1회로 붕괴**하는 이유가 증명의 (3)단계다: $\tfrac12\sum_j q_{kj}=\tfrac12 a_k^\top(H\sum_j a_j)$ — 우측 벡터 $H\Delta W$가 모든 $k$에 공통. 이는 IRDS ghost-HVP의 효율
포인트(우변 = 배치 gradient 합으로 고정 — 원문 977–984행 [노트 경유 §8.2])와 동형 구조다.
LLM chunked 경로(`:42–62`)는 grad·HVP가 loss에 선형이므로 $\sum_c w_c\cdot$grad/HVP$(\mathrm{lf}_c)$
= full-val grad/HVP **정확히** (근사 아님; docstring `:43–47`).

**IRDS 대응.** Thm 4(원문 136–204행) / 증명 Thm 6(444–581행) [노트 경유 §4.2 — 조합 항등식 2개로
완결, 검산 통과]. 대응: $a_k\leftrightarrow -\eta_t\nabla\ell(w_t,z)$,
$\Delta W\leftrightarrow-\eta_t\sum_{z_j\in\mathcal B_t}\nabla\ell(w_t,z_j)$로 두면 그들의
$\phi_z(U^{(t)}_{(1)})+\tfrac12\phi_z(U^{(t)}_{(2)})=-\eta_t\nabla\ell_{\mathrm{val}}\cdot\nabla\ell_z+\frac{\eta_t^2}2\nabla\ell_z^\top H(\sum_j\nabla\ell_j)$와 P2가 동형이다. 증명 경로는 다르다(IRDS:
Shapley 공식에 marginal 대입 후 조합 항등식; 본 문서: 게임의 unanimity 분해) — 결과는 같고, 게임
분해 경로가 P4의 merge-consistency에 재사용된다는 이점이 있다. IRDS의 "근사게임의 exact Shapley"
프레이밍(Remark 3, 원문 354–356행: 공리는 근사 utility에 대해 정확히 성립)도 그대로 이전된다 —
P2의 공리 성립 주장은 $\hat u_r$에 대한 것이고, $u_r$에 대해서는 P3의 잔차만큼 근사다.

### 4.3 P3 — Taylor 잔차의 형식화

**선행 확인**: IRDS 원문의 오차 진술은 111행의 비형식 한 문장이 전부다("with approximation errors
of $O(\eta_t^2)$ and $O(\eta_t^3)$…") — smoothness 가정, 상수, 배치크기 의존성, 스텝 합산 후 누적
bound 전부 부재하고 정리·증명이 없다 (이론 추출 노트 §3.2 [갭]; 원문 직접 재확인). **아래는 이 잔차를
(IRDS·Flirds 기준으로) 명시적으로 형식화한 것이다** — 잔차 정리 자체는 표준 Taylor 적분나머지+연산자노름
bound이고, 신규 기여는 IRDS가 비형식 $O(\eta^2)/O(\eta^3)$로 남긴 smoothness 가정($C^3$·$M_2/M_3$)·상수·
$R$-라운드 누적을 명시한 데 있다.

**가정.** A1–A3; A4: $\ell\in C^3(\mathcal U_r)$, $\mathcal U_r$은 **클라 coalition 이동과 샘플
coalition 이동 양쪽**의 선분
$\{w_r+t\Delta_S:t\in[0,1],\,S\subseteq P_r\}\cup\{w_r+t\!\sum_{i\in S'}x_i:t\in[0,1],\,S'\subseteq B\}$을
포함하는 열린 볼록집합이고 ($x_i,B$는 §4.4.3; 샘플 coalition은 클라 블록을 쪼갤 수 있어 후자가 더 넓은
집합이므로, 아래 $M_2^r/M_3^r$이 P4c의 per-sample 잔차항까지 지배하도록 $\mathcal U_r$을 이렇게 잡는다.
클라만 다루는 P3에는 무해한 확대)

$$M_2^r:=\sup_{w\in\mathcal U_r}\lVert\nabla^2\ell(w)\rVert_{\mathrm{op}},\qquad
M_3^r:=\sup_{w\in\mathcal U_r}\sup_{\lVert v\rVert=1}\big\lvert\nabla^3\ell(w)[v,v,v]\big\rvert\ <\infty.$$

**명제 P3.** 모든 $S\subseteq P_r$에 대해

(i) $\big\lvert u_r(S)-\hat u_r^{(1)}(S)\big\rvert\le\frac{M_2^r}{2}\lVert\Delta_S\rVert^2$;
(ii) $\big\lvert u_r(S)-\hat u_r(S)\big\rvert\le\frac{M_3^r}{6}\lVert\Delta_S\rVert^3$;
(iii) $\lVert\Delta_S\rVert\le\sum_{k\in S}p_k^r\lVert\delta_k^r\rVert\le\max_{k\in P_r}\lVert\delta_k^r\rVert$
(삼각부등식; $\sum_{k\in S}p_k^r\le1$).

*증명.* $h(t):=\ell(w_r+t\Delta_S)$, $t\in[0,1]$로 두면 A4에 의해 $h\in C^3$이고
$h'(t)=\langle\nabla\ell(w_r+t\Delta_S),\Delta_S\rangle$,
$h''(t)=\Delta_S^\top\nabla^2\ell(w_r+t\Delta_S)\Delta_S$,
$h'''(t)=\nabla^3\ell(w_r+t\Delta_S)[\Delta_S,\Delta_S,\Delta_S]$.
적분형 잔차의 Taylor 정리:

$$h(1)=h(0)+h'(0)+\int_0^1(1-t)\,h''(t)\,dt
=h(0)+h'(0)+\tfrac12h''(0)+\int_0^1\frac{(1-t)^2}{2}\,h'''(t)\,dt.$$

$u_r(S)=h(1)-h(0)$, $\hat u_r^{(1)}(S)=h'(0)$, $\hat u_r(S)=h'(0)+\tfrac12h''(0)$이므로:
(i) $\lvert\int_0^1(1-t)h''(t)dt\rvert\le M_2^r\lVert\Delta_S\rVert^2\int_0^1(1-t)dt=\frac{M_2^r}2\lVert\Delta_S\rVert^2$
($\lvert h''(t)\rvert\le M_2^r\lVert\Delta_S\rVert^2$: 연산자 노름 정의).
(ii) $\lvert h'''(t)\rvert=\lVert\Delta_S\rVert^3\lvert\nabla^3\ell[\tfrac{\Delta_S}{\lVert\Delta_S\rVert},\cdot,\cdot]\rvert\le M_3^r\lVert\Delta_S\rVert^3$이고 $\int_0^1\frac{(1-t)^2}2dt=\frac16$. $\blacksquare$

**보조정리 L2 (Shapley 오차 전파).** 두 게임 $v,v'$ ($v(\emptyset)=v'(\emptyset)=0$)에 대해
$\lvert\phi_i(v)-\phi_i(v')\rvert\le 2\,\lVert v-v'\rVert_\infty$
($\lVert d\rVert_\infty:=\max_S\lvert d(S)\rvert$).

*증명.* L0(c)로 $\phi_i(v)-\phi_i(v')=\phi_i(d)$, $d:=v-v'$. 각 marginal은
$\lvert d(S\cup i)-d(S)\rvert\le2\lVert d\rVert_\infty$이고 L0(b)에 의해 Shapley 계수 합이 1이므로
$\lvert\phi_i(d)\rvert\le2\lVert d\rVert_\infty$. $\blacksquare$

**따름정리 P3-1 (라운드·전체 $\phi$ 오차).**

$$\big\lvert\phi_k^{P_r}(u_r|_{P_r})-\phi_k(\hat u_r)\big\rvert
\le\frac{M_3^r}{3}\max_{S\subseteq P_r}\lVert\Delta_S\rVert^3,\qquad
\big\lvert\phi_k(U_b)-\hat\phi_k\big\rvert\le\sum_{r=0}^{R-1}\frac{M_3^r}{3}\max_{S\subseteq P_r}\lVert\Delta_S^r\rVert^3,$$

($\hat\phi_k$ = estimator 출력; 좌측은 P1(iii)+L2+P3(ii), 우측은 라운드 합의 삼각부등식.)

**1차 전용 estimator의 올바른 Shapley bound는 계수가 다르다** (2차식에 단순 대입 금지):

$$\big\lvert\phi_k^{P_r}(u_r|_{P_r})-\phi_k(\hat u_r^{(1)})\big\rvert\le M_2^r\max_{S\subseteq P_r}\lVert\Delta_S\rVert^2,\qquad
\big\lvert\phi_k(U_b)-\hat\phi_k^{(1)}\big\rvert\le\sum_r M_2^r\max_{S\subseteq P_r}\lVert\Delta_S^r\rVert^2.$$

L2가 P3(i)의 $\tfrac12$-계수를 2배해 선행계수가 $M_2$(계수 **1**)이므로, 2차식 $M_3/3$의 자리에
$M_2/3$을 넣는 것은 **틀리다**(3배 과소평가 = 무효 상계).

**P3의 지위 (상계·미검증 타이트함).** P3-1은 **상계**이고 그 타이트함은 미검증이다: $M_2^r/M_3^r$은
실측 불가, per-round $\lVert\Delta\rVert$가 $K$-스텝 누적으로 구조적으로 크고, 우변이 $R$에 선형
누적한다. gpt2 스모크(§7)는 잔차가 fp32 노이즈 바닥에 놓여 1차·2차 모두 loglog 기울기 $\approx0.30$
(예측 2/3 아님)이고 2차>1차 우위도 미관측 — 즉 P3(i)(ii)의 $\lVert\Delta\rVert^2/\lVert\Delta\rVert^3$
지수는 **아직 실증 미확인**이며 §7의 1B 실행에 달려 있다.

**IRDS와의 규모 차 (bound가 말해주는 것과 못하는 것).**
- IRDS per-step 이동: $\eta_t\times$(배치 gradient 합), lr $3\times10^{-4}$·batch 16 규모 (원문
  §5.2, 273행). Flirds per-round 이동 $\delta_k^r$: 로컬 $K$-스텝(예: 10 steps)·epochs 누적 — 같은
  lr이면 $\lVert\Delta_S\rVert$가 구조적으로 크고, bound는 세제곱으로 악화된다. 즉 **P3은 IRDS보다
  실질적으로 약한 보증**이며, 상수 $M_3^r$은 실측 불가하므로 bound 자체의 타이트함은 열려 있다.
- 누적: 우변이 $R$에 선형으로 쌓인다. IRDS는 누적 오차에 대한 어떤 진술도 없고 검증도 단일
  iteration에 국한된다(원문 §5.2·E.2.1; 이론 추출 노트 §3.2·§9.1 [갭]) — 같은 갭이 우리에게도 있으며
  §7의 실측이 이를 직접 측정한다(라운드별 실제 $u_r(S)$ vs $\hat u^{(1)},\hat u$, 전 $S$).
- 참고로 현행 무대의 실측 정황은 잔차가 작은 쪽을 가리킨다: per-round 이동이 작고 게임이 사실상
  가산적 (가산 갭 $\le$0.9% — §4.5에서 인용).

**코드 대응.** estimator·oracle의 유일한 괴리가 정확히 이 잔차다 — 단 **fp32-로그 프로토콜(§2.1)과
forward bitwise-결정성(P2-1) 조건부**다. bf16 로그나 비결정 forward에서는 estimator의 fp32 강제
캐스팅(`flirds_estimator.py:101,103`) vs oracle의 로그-dtype 신뢰(`in_run_sv.py:36–49`), 그리고 반복
forward 비재현성이 **비-Taylor 괴리**로 추가된다 (§2.1·§6-2,3). (§3.1; 감사 노트 §0 Q1·§3.3.)
전개 지점은 양쪽 다 $w_r$ (`flirds_estimator.py:101`; `in_run_sv.py:63–64`).

**IRDS 대응.** 원문 111행 (비형식) → P3 (형식화, 신규). IRDS의 utility-수준 검증(E.2.2: 1차만으로
Spearman>0.94, 2차 추가 이득 미미 — 원문 111행·1019행 [노트 경유 §9.2])은 centralized 소-$\eta$
regime의 결과로, FL per-round(큰 이동) regime에 자동 이전되지 않는다 — Flirds Phase 0.5에서 2차가
유효했던 관측과 모순 아님 (regime 상이; 이론 추출 노트 §11-10).

### 4.4 P4 — per-sample → per-client 단위 변경

IRDS의 플레이어는 개별 샘플 $z_i$(원문 48행), Flirds의 플레이어는 클라이언트 $k$다. 이 절은 단위
변경이 (i) 어떤 구조적 대응 위에 서 있고, (ii) 어떤 극한에서 정확한 환원이 되며, (iii) 어디서
어긋나는지를 분리해 증명한다.

#### 4.4.1 (i) 선형 구조의 대응과 고정가중의 필수성

IRDS additivity의 두 필요조건 (이론 추출 노트 §2 [추론], 원문 74·111–114행에서 재구성):
**(C1)** update가 플레이어별 항의 합이고 계수가 coalition-독립
($\widetilde w_{t+1}(S)-w_t=-\eta_t\sum_{z\in S}\nabla\ell(w_t,z)$; sum-형 규약이라 $1/\lvert S\rvert$
없음), **(C2)** 1차 Taylor가 update에 선형으로 작용. (C1)+(C2) ⇒ 1차 surrogate가 additive 게임
(원문 114행 "This shows that $U^{(t)}_{(1)}$ is an *additive* utility function").

**명제 P4a (대응 + 필수성).**
(1) 고정가중 게임(D1)의 이동 사상은 $\Delta_S=\sum_{k\in S}a_k$ ($a_k$ coalition-독립) — (C1)의
클라-수준 정확 대응이며, 따라서 $\hat u^{(1)}_r$은 additive, $\hat u_r$은 additive+pairwise
(P2의 전제 구조).
(2) 반면 **FedAvg 정통 재정규화 이동** $\widetilde\Delta_S:=\sum_{k\in S}\frac{n_k}{\sum_{j\in S}n_j}\delta_k$는 (C1)을 위반한다: 어떤 coalition-독립 벡터족 $\{b_k\}$에 대해서도
$\widetilde\Delta_S=\sum_{k\in S}b_k$ 꼴로 쓸 수 없다 (퇴화 경우 제외).

*증명 (2).* 등$n$ 2-클라 반례. singleton에서 $\widetilde\Delta_{\{k\}}=\delta_k$이므로 표현이
존재한다면 $b_k=\delta_k$가 강제된다. 그러나 pair에서
$\widetilde\Delta_{\{1,2\}}=\tfrac12(\delta_1+\delta_2)\ne\delta_1+\delta_2=b_1+b_2$
($\delta_1+\delta_2\ne0$인 한). 따라서 표현 불가. 같은 이유로 1차 surrogate의 additivity도 깨진다:
$\langle g,\widetilde\Delta_{\{1,2\}}\rangle=\tfrac12\big(\langle g,\delta_1\rangle+\langle g,\delta_2\rangle\big)\ne\langle g,\delta_1\rangle+\langle g,\delta_2\rangle$ (합이 0이 아닌 한). $\blacksquare$

**결론**: 고정가중은 편의가 아니라 **IRDS 닫힌형 기계 전체(P2)를 이식하기 위한 필수 가정**이다.
명제·논문 서술에서 가정으로 명시해야 한다. (재정규화 게임을 채택하지 않은 것의 정당화는 P5.)

#### 4.4.2 (ii) 라운드 내 독립성의 대응

IRDS: 스텝 $t$의 모든 per-sample gradient가 **같은 $w_t$에서** 평가된다 (원문 74행). Flirds 대응:
라운드 $r$의 모든 $\delta_k^r$이 같은 $w_r$에서 출발한 로컬 학습의 산출물이다 (F3;
`server.py:49–50`이 같은 `global_state`로 순차 호출, `client.py:16`이 매번 재동기화).

정확한 진술: $\delta_k^r$은 $(w_r,\ D_k,\ \text{해당 시점 RNG 상태})$의 결정적 함수이며, **다른
클라의 데이터·delta의 값이 $\delta_k^r$에 들어갈 정보 경로는 없다** — 클라 간 상호작용은 집계(F6)
시점에만 발생한다. 단 $\delta_k^r$이 다른 클라와 **통계적으로 독립인 것은 아니다**: 서버는 공유 global
torch RNG 스트림 위에서 클라를 순차 학습하므로(`server.py:49–50`; `client.py:16`은 가중치만 재동기화,
RNG는 아님), 셔플·dropout이 있으면 클라 $k$ 차례의 RNG 상태가 앞선 클라들이 소비한 난수량에 의존한다
— 처리 순서를 통한 실제 커플링 경로다. 그러나 IRDS의 realization-fixing 논리(원문 81행: 랜덤성
실현값을 utility 정의에 고정 → 결정론적 게임)에 의해 **이 커플링은 실현값에 구워져 게임 정의에는
영향이 없다** — frozen logs의 $\delta_k^r$이 게임의 원자이기 때문이다. (살아남는 것은 frozen-게임
정의뿐이고, 라운드 내부의 통계적 독립성이 아니다.)

#### 4.4.3 (iii) 로컬 1-step full-batch 극한에서의 정확한 환원

**설정 (극한 조건)**: 로컬 학습 = 1 epoch, 1 스텝, full-batch, mean-reduction cross-entropy
(`client.py:23`의 `F.cross_entropy` 기본 reduction), plain SGD lr $\eta$ (F9). 그러면

$$\delta_k=-\frac{\eta}{n_k}\sum_{i\in D_k}\nabla\ell_i(w_r),\qquad
a_k=p_k^r\delta_k=-\frac{\eta}{\sum_{j\in P_r}n_j}\sum_{i\in D_k}\nabla\ell_i(w_r),$$

($\ell_i$ = 샘플 $i$의 훈련 손실). $x_i:=-\eta_{\mathrm{eff}}\nabla\ell_i(w_r)$,
$\eta_{\mathrm{eff}}:=\eta/\sum_{j\in P_r}n_j$로 두면 $a_k=\sum_{i\in D_k}x_i$이고 라운드 이동은

$$\Delta W^r=\sum_{k\in P_r}a_k=-\eta_{\mathrm{eff}}\sum_{i\in B}\nabla\ell_i(w_r),\qquad B:=\bigcup_{k\in P_r}D_k$$

— **IRDS의 sum-형 배치 스텝(원문 74행)과 정확히 동일한 형태** (배치 $B$, lr $\eta_{\mathrm{eff}}$).
즉 이 극한에서 per-sample IRDS 게임(플레이어 = $B$의 샘플, update 벡터 $x_i$)이 잘 정의되고,
per-client 게임은 그 샘플들을 클라 블록으로 **병합(merge)**한 게임이다. 병합이 값을 보존하는지가
다음 보조정리다.

**보조정리 L3 (2-additive 게임의 merge-consistency).** 유한 ground set $B$ 위의 게임

$$v(S)=\sum_{i\in S}c_i+\sum_{\substack{\{i,j\}\subseteq S\\ i\ne j}}q_{ij}\qquad(q_{ij}=q_{ji})$$

와 $B$의 분할 $\{D_k\}_{k\in\mathcal K}$가 주어졌을 때, 병합 게임
$\bar v(T):=v\big(\bigcup_{k\in T}D_k\big)$ ($T\subseteq\mathcal K$)에 대해

$$\phi_k(\bar v)=\sum_{i\in D_k}\phi_i(v)\qquad\forall k\in\mathcal K.$$

*증명.* 먼저 $\bar v$도 같은 클래스임을 보인다:

$$\bar v(T)=\sum_{k\in T}\underbrace{\Big(\sum_{i\in D_k}c_i+\sum_{\{i,j\}\subseteq D_k}q_{ij}\Big)}_{=:C_k}
+\sum_{\substack{\{k,l\}\subseteq T\\ k\ne l}}\underbrace{\sum_{i\in D_k}\sum_{j\in D_l}q_{ij}}_{=:Q_{kl}},$$

(블록 내부 쌍은 블록 자신의 가산항으로 흡수되고, 블록 간 쌍만 상호작용으로 남는다; $Q_{kl}=Q_{lk}$.)
P2의 증명 (1)(2)단계는 임의의 additive+pairwise 게임에 적용되므로:

$$\phi_k(\bar v)=C_k+\tfrac12\sum_{l\ne k}Q_{kl}
=\sum_{i\in D_k}c_i+\sum_{\{i,j\}\subseteq D_k}q_{ij}+\tfrac12\sum_{i\in D_k}\sum_{j\notin D_k}q_{ij}.$$

한편 원 게임에 P2 (1)(2)를 적용하면 $\phi_i(v)=c_i+\tfrac12\sum_{j\ne i}q_{ij}$ (행-선형: $\phi_i$의
상호작용 몫이 $q$의 $i$-행 합의 절반), 따라서

$$\sum_{i\in D_k}\phi_i(v)=\sum_{i\in D_k}c_i+\tfrac12\sum_{i\in D_k}\sum_{j\ne i}q_{ij}
=\sum_{i\in D_k}c_i+\tfrac12\Big(2\!\!\sum_{\{i,j\}\subseteq D_k}\!\!q_{ij}+\sum_{i\in D_k}\sum_{j\notin D_k}q_{ij}\Big),$$

(블록 내부 쌍은 $i$-행 합들에서 두 번, 블록 외부 상대 쌍은 한 번 세어진다.) 두 식이 일치한다.
$\blacksquare$

병합의 직관: 쌍별 상호작용 $q_{ij}$는 항상 두 당사자에 절반씩 분배되는데(P2 (2)), 병합 후에는
블록 내부 쌍의 두 절반이 모두 같은 블록에 귀속되고($\tfrac12q_{ij}+\tfrac12q_{ji}=q_{ij}$ — $C_k$로
흡수된 것과 일치), 외부 쌍의 절반 분배는 그대로다 — 그래서 정확히 가산된다.

**따름정리 P4b (2차 surrogate 내 브리지).** 위 극한 설정에서, $w_r$ 기준 2차 Taylor surrogate에
대해 per-sample IRDS 게임의 Shapley를 클라별로 합한 것 = per-client Flirds 게임(D3)의 Shapley:

$$\sum_{i\in D_k}\phi_i(\hat u^{\mathrm{sample}}_r)=\phi_k(\hat u_r),\qquad
\hat u^{\mathrm{sample}}_r(S):=\Big\langle g^r,\sum_{i\in S}x_i\Big\rangle+\tfrac12\Big(\sum_{i\in S}x_i\Big)^{\!\top}\! H^r\Big(\sum_{i\in S}x_i\Big),\ S\subseteq B.$$

*증명.* $\hat u^{\mathrm{sample}}_r$은 L3의 클래스($c_i=\langle g,x_i\rangle+\tfrac12x_i^\top Hx_i$,
$q_{ij}=x_i^\top Hx_j$)이고, $a_k=\sum_{i\in D_k}x_i$이므로 $\hat u_r(T)=\hat u^{\mathrm{sample}}_r(\bigcup_{k\in T}D_k)$ — 즉 $\hat u_r$이 정확히 병합 게임이다. L3 적용. $\blacksquare$

**Remark P4b-CNN (분모 상쇄는 mean-CE ⊗ n-비례 가중 전용; LLM token-mean은 위반).** 위 극한의 핵심은
로컬-손실 reduction 분모 $n_k$가 FedAvg 가중 분자 $n_k$와 **정확히 상쇄**되어 per-sample 공통 lr
$\eta_{\mathrm{eff}}$가 남는다는 것이다(이것이 "IRDS sum-형 배치 스텝"을 만든다). 이는 **CNN mean-CE
(샘플=이미지)**에서 성립한다: `client.py:23` `F.cross_entropy`가 이미지 평균, 가중 $n_k$=이미지 수(F5).
그러나 **LLM/LoRA 트랙은 token-mean CE**다: 배치 손실이 $\big(\sum_{\text{tok}}\big)/\text{tokens}_k$
(`backends/llm.py:78,80–82,92–93` 토큰-가중 평균)이라 자연 분모는 $\text{tokens}_k=\sum_{i\in D_k}\lvert z_i\rvert$
인데, FedAvg 가중은 $n_k=\text{len(ds)}$=**시퀀스 수**(`llm_server.py:85`)다. 둘은 시퀀스별 completion
길이가 모두 같을 때만 상쇄되고, 일반적으로 per-token 갱신이 $n_k^{\text{seq}}/\text{tokens}_k=1/(\text{클라 }k\text{ 평균 completion 길이})$
로 스케일되어 **공통-lr sum-형이 나타나지 않는다**. 즉 **P4b의 정확 브리지는 CNN mean-CE에 스코프**
되고, LLM per-token IRDS 분해는 공통-lr sum으로 환원되지 않는다. 일반 숨은 가정: "로컬-손실 reduction
분모 = FedAvg 집계 가중". 대응표 1행의 브리지가 LLM valuation을 "per-sample IRDS 값의 합"으로
근거짓는다고 읽어서는 안 된다.

**전제·경계 (P4b 설정).** 추가로: (전1) **per-sample gradient 분리성** — $\nabla\ell_i$가 샘플별로
분리(BatchNorm류 배치-통계 결합 없음; LeNet/LoRA-LLM은 만족, IRDS ghost 분해도 같은 전제 상속);
(전2) **기본 n-비례 집계** — 온라인 개입 arm(`server.py:57–59` `weights_fn`; Track C2/D의
flirds_w/shapleyfl_w)이 가중을 바꾸면 $n_k$ 상쇄가 깨져 브리지가 무효다(D1 게임 자체는 무사). 경계:
(경1) $n_k=0$이면 $\delta_k=-(\eta/n_k)\sum\nabla\ell_i$가 $0/0$로 ill-posed(다만 $a_k=0$은 여전히
well-defined; 전원 $n=0$이면 §2.1 정칙성 위반); (경2) **free-rider**(`llm_server.py:39–41`)는 훈련
없이 delta를 **날조**하므로 그 블록 $a_k\ne\sum_{i\in D_k}x_i$이고 per-sample 분해가 없다 — 1-step
극한에서도 브리지 대상에서 제외된다.

**따름정리 P4c (exact 게임의 병합 갭).** exact 게임 쌍
$u^{\mathrm{sample}}_r(S):=\ell(w_r+\sum_{i\in S}x_i)-\ell(w_r)$,
$u_r$(D1)에 대해 (A4 하에서)

$$\Big\lvert\phi_k(u_r|_{P_r})-\sum_{i\in D_k}\phi_i(u^{\mathrm{sample}}_r)\Big\rvert
\le\frac{M_3^r}{3}(1+n_k)\max_{S\subseteq B}\Big\lVert\sum_{i\in S}x_i\Big\rVert^3.$$

*증명.* 삼각부등식으로 3분해: $\lvert\phi_k(u_r)-\phi_k(\hat u_r)\rvert$ (P3-1) $+\ \lvert\phi_k(\hat u_r)-\sum_i\phi_i(\hat u^{\mathrm{sample}}_r)\rvert$ ($=0$, P4b) $+\ \sum_{i\in D_k}\lvert\phi_i(\hat u^{\mathrm{sample}}_r)-\phi_i(u^{\mathrm{sample}}_r)\rvert$ (L2+P3(ii), $n_k$개 항). 클라-coalition 이동은 샘플-coalition 이동의 부분집합
($\Delta_T=\sum_{i\in\cup_{k\in T}D_k}x_i$)이므로 두 max를 샘플 쪽 하나로 통일. $\blacksquare$

#### 4.4.4 (iv) 어긋나는 지점 — 전수 목록과 분류

단위 변경에서 오는 이론의 어긋남은 다음 **세 가지가 전부**이며, 각각의 성격이 다르다.

**(a) $K>1$ 로컬 스텝 — granularity 선택 (게임 정의 선택; 오차 아님).**
로컬이 $K$ 스텝이면 IRDS-faithful한 세밀(fine) 게임은 로컬 중간점 $w_{r,0}=w_r,\dots,w_{r,K-1}$
각각에서 per-step 전개하는 게임이고, Flirds는 라운드 전체를 $w_r$ 한 점에서 전개하는 거친(coarse)
게임이다. 두 exact 게임은 **서로 다른 게임**이다 — 어느 쪽도 상대의 근사가 아니며, 이는 IRDS 자신이
per-step 게임을 "정의"로 선택한 것(원문 86행)과 **같은 논리적 유형**(게임 정의/granularity 선택)의
결정이다. **단 지위가 동일하다는 뜻은 아니다**: IRDS의 per-step 단위는 centralized SGD의 원자
granularity라 애초에 coarsening 선택지가 없었지만(단일 granularity), Flirds per-round는 원자적
per-local-step 단위 **위의 coarsening**이다 — 클라 원자 $\delta_k^r$이 $K$-스텝 합성물(내부 gradient가
off-$w_r$ 반복점 참조)이고 surrogate가 $K$-스텝 큰 이동을 전개하므로 IRDS per-step의 진짜 대응은 Flirds
**per-local-step**(더 세밀한 게임)이다. coarsening은 '오차 아님'이되 **무비용도 아니다**(Taylor 충실도를
P3의 큰 $\lVert\Delta_S\rVert$ 방향으로 악화). 실질 함의 두 가지:
(1) coarse 게임의 Taylor 잔차가 커진다(P3의 $\lVert\Delta_S\rVert$가 $K$-스텝 누적) — 이는 P3이
계량한다. (2) 1차 수준에서 두 게임의 차이는 gradient drift
$\sum_t\langle\nabla\ell(w_{r,t})-\nabla\ell(w_r),\ \text{step}_t\rangle$ 규모다 — fine 게임을
관측하려면 per-step 로그가 필요한데 현행 프로토콜은 $\delta_k^r$만 로그한다(F8) — 측정하려면
프로토콜 변경 필요 (§9).

**(b) 로컬 minibatch 확률성 — realization-fixing으로 게임은 무사, per-sample 브리지만 제한.**
minibatch·셔플 난수는 실현값이 $\delta_k^r$에 구워져 들어가고, 게임은 frozen delta 위에 정의되므로
클라-수준 게임(D1·P1·P2)에는 아무 문제가 없다 — IRDS가 배치 선택 실현을 utility에 고정하는 것
(원문 81행)과 동일한 처리다. 깨지는 것은 per-sample 해석(4.4.3)이다: 샘플들이 서로 다른 로컬
중간점에서 서로 다른 횟수 등장하므로 "$\delta_k=\sum_{i}x_i$, $x_i$는 $w_r$에서의 샘플 항" 구조가
사라진다. **분류: 게임 정의는 성립; per-sample 브리지는 1-step full-batch 극한 전용.**

**(c) 3차 이상에서 merge-consistency 실패 — 진짜 이론의 어긋남.**
**반례**: ground set $\{1,2,3\}$, 게임 $v=\pi_{123}$ (3-인 unanimity: $v(S)=\mathbf 1[S=\{1,2,3\}]$),
분할 $D_1=\{1,2\}$, $D_2=\{3\}$. per-sample Shapley: 대칭+efficiency로
$\phi_1=\phi_2=\phi_3=\tfrac13$ ⇒ $D_1$ 블록 합 $=\tfrac23$. 병합 게임 $\bar v$ = 2-인 unanimity ⇒
$\phi_{D_1}(\bar v)=\tfrac12\ne\tfrac23$. $\blacksquare$
3-additive 성분은 정확히 Taylor 3차 항이 만드는 구조다: 3차 항은 삼중 상호작용
$\nabla^3\ell(w_r)[x_i,x_j,x_k]$ 계수의 항들을 생성하며 일반적으로 0이 아니다. 따라서 **2차
surrogate를 넘어서면 per-sample 합산과 per-client 값은 원리적으로 다르다** — 단위 변경(집계
수준 선택)이 값 자체를 바꾸는 지점. 크기는 P4c가 3차 잔차 규모로 bound. **분류: 이론의 어긋남**
(다만 유한하고 계량 가능).

**IRDS 대응.** IRDS에는 병합/단위 변경에 대한 어떤 논의도 없다 (FL 관련은 관련연구 1문장, 원문
352행). P4 전체(특히 L3·P4b·반례 (c))가 **신규**다.

### 4.5 P5 — 고정가중 게임 vs 재정규화 게임

**대안 게임 (재정규화 = counterfactual FedAvg).**

$$\tilde u_r(S):=\ell\Big(w_r+\sum_{k\in S\cap P_r}\tfrac{n_k}{\sum_{j\in S\cap P_r}n_j}\,\delta_k^r\Big)-\ell(w_r)
=\ell\big(w_r+c_S\,\Delta_S\big)-\ell(w_r),\qquad
c_S:=\frac{\sum_{j\in P_r}n_j}{\sum_{j\in S\cap P_r}n_j}\ \ge1,$$

(재정규화 가중 $=c_S\,p_k^r$이므로 두 번째 등호; $c_S=1\iff S\supseteq P_r$.) 소규모 coalition일수록
$c_S$가 커진다 — 재정규화 게임은 부분 coalition의 이동을 증폭한 가상 스텝을 평가한다.

**채택 근거 (고정가중을 택한 이유 3개).**
1. **IRDS 관행의 정확한 이식**: IRDS의 $\widetilde w_{t+1}(S)$에서 $\eta_t$는 $\lvert S\rvert$와
   무관하게 고정이다(원문 74·81행; sum-형 규약). 샘플을 빼도 lr을 재조정하지 않는 것 — retrain
   기반의 "subset 크기에 따른 hyperparameter 모호성"(원문 B.1(4), 384행)을 realized run에 고정해
   우회하는 것이 IRDS의 명시적 입장이고, 고정가중이 그 클라-수준 대응이다.
2. **실현 run의 가산 분해라는 의미론**: 고정가중에서 $\Delta_S$는 실현 이동 $\Delta W^r$에 실제로
   들어간 기여의 부분합이다($\Delta_{P_r}=\Delta W^r$). 재정규화의 $c_S\Delta_S$ ($S\subsetneq P_r$)는
   실현 궤적에 존재한 적 없는 가상 이동이다 — "이 run에의 기여" 귀속이라는 IRDS 의미론(원문 81·86행,
   Remark 4)과 정합하는 쪽은 전자다. **[골격 초안 정정]**: 초안은 근거 ②를 "S=P_r에서 realized
   update와 일치(telescoping·efficiency)"로 적었으나, $c_{P_r}=1$이므로 **telescoping·efficiency는
   재정규화 게임에서도 똑같이 성립한다** — 차별점이 아니다. 진짜 차별점은 위의 sub-coalition
   의미론이다.
3. **estimator와의 게임 일치**: P4a에 의해 닫힌형 기계(P2)가 성립하는 쪽은 고정가중뿐이고, (b)
   oracle도 같은 게임이므로(§3.1) fidelity 비교(estimator vs oracle)가 같은 대상을 본다. 재정규화
   게임을 채택하면 estimator가 근사하는 게임과 oracle 게임이 갈라진다.

**명제 P5b (등$n$·선형 극한에서 순위 동일).** $n_1=\dots=n_K$ ($K=K_r$), 1차 surrogate 수준
($\tilde u^{(1)}_r(S):=c_S\langle g,\Delta_S\rangle=\frac1{\lvert S\rvert}\sum_{k\in S}b_k$,
$b_k:=\langle g^r,\delta_k^r\rangle$)에서, 재정규화 게임의 Shapley는

$$\phi_k\big(\tilde u^{(1)}_r\big)=\frac{H_K}{K}\,b_k-\frac{H_K-1}{K(K-1)}\sum_{j\ne k}b_j
=\underbrace{\frac{K\,H_K-1}{K(K-1)}}_{>0}\,b_k\ \underbrace{-\ \frac{H_K-1}{K(K-1)}\sum_{j}b_j}_{k\text{-공통 상수}},
\qquad H_K:=\sum_{m=1}^K\tfrac1m,$$

이며, 이는 $b_k$의 **공통 양-기울기 affine 함수**이므로 고정가중 게임의 1차 Shapley
$\phi_k(\hat u^{(1)}_r)=\tfrac1K b_k$ (P2)와 **순위가 동일**하다.

*증명.* marginal: $S\subseteq P_r\setminus k$, $\lvert S\rvert=s$에 대해
$\tilde u^{(1)}(S\cup k)-\tilde u^{(1)}(S)=\frac{b_k}{s+1}-\frac{\sum_{j\in S}b_j}{s(s+1)}$
($s=0$이면 둘째 항 없음). Shapley 정의에 대입해 두 부분을 따로 합산한다.
(1) 자기항: 크기 $s$ 집합 수 $\binom{K-1}{s}$, 계수
$\binom{K-1}{s}\frac{s!(K-s-1)!}{K!}=\frac1K$이므로
$\sum_{s=0}^{K-1}\frac1K\cdot\frac{1}{s+1}\,b_k=\frac{H_K}{K}b_k$.
(2) 희석항: 고정 $j\ne k$는 크기 $s$ 집합 중 $\binom{K-2}{s-1}$개에 등장;
$\frac{s!(K-s-1)!}{K!}\binom{K-2}{s-1}=\frac{(K-2)!\,s}{K!}=\frac{s}{K(K-1)}$이므로 계수는
$\sum_{s=1}^{K-1}\frac{s}{K(K-1)}\cdot\frac1{s(s+1)}=\frac1{K(K-1)}\sum_{s=1}^{K-1}\frac1{s+1}=\frac{H_K-1}{K(K-1)}$.
따라서 $\phi_k=\frac{H_K}K b_k-\frac{H_K-1}{K(K-1)}\sum_{j\ne k}b_j$. 재배열하면 위의 공통 affine 형.
기울기 $\frac{KH_K-1}{K(K-1)}>0$ ($H_K\ge1\Rightarrow KH_K\ge K>1$). 검산: efficiency
$\sum_k\phi_k=\big(\frac{H_K}K-\frac{H_K-1}{K}\big)\sum b=\frac1K\sum b=\tilde u^{(1)}(P_r)$ ✓;
$K=2$ 직접 계산($\phi_1=\tfrac34b_1-\tfrac14b_2$)과 일치 ✓. $\blacksquare$

**Remark P5b-경계·범위.** (1) **$K_r=1$**: 닫힌형 계수 $(H_K-1)/(K(K-1))$·기울기 $(KH_K-1)/(K(K-1))$이
$0/0$가 되므로 위 형은 $K_r\ge2$ 전용이다. $K_r=1$은 퇴화 케이스 — 단일 참여자면 $c_{\{k\}}=1$이라 두
게임이 **동일**(자명히 순위 일치). (2) **per-round → valuation 리프트는 전원참여 전용이다.** P5b는
**단일 라운드** 진술인데 실제 Flirds 값은 라운드 합 $\Phi_k=\sum_r\phi_{k,r}$이다. **부분참여**에서는
각 라운드 희석상수 $C_r$이 그 라운드 참여자에게만 공통이라 $\sum_{r:k\in P_r}C_r$이 클라 간 공통이 아니고,
두 게임이 라운드를 다르게 가중($s_r$ vs $1/K_r$)해 라운드-합 순위 동일성이 깨진다. 반례(전 $n_k=1$
등$n$, 1차): 3-클라, 라운드 A$=\{1,2,3\}$ 전원 $b=0$, 라운드 B$=\{1,2\}$ $b_1=1,b_2=10$. 고정가중
$\Phi=(0.5,5,0)$ → 순위 2>1>3; 재정규화(라운드 B의 $C_B=-2.75$) $\Phi=(-1.75,7.25,0)$ → 순위 2>3>1
(클라 1·3 스왑). **리프트는 전원참여($K_r=N$ 상수, $\sum_rC_r$ 공통)에서만 성립** — N=5 cross-silo
fidelity는 전원참여라 OK, N=100 cross-device는 부분참여라 실패. §1의 '증명 가능하게 동일'은 이 범위로
한정했다.

(참고: 골격 초안의 "$\phi(\tilde u)=b_k\cdot\mathbb E[K/(\lvert S\rvert+1)]$" 서술은 자기항만 본
것이다 — 희석항 $-\frac{H_K-1}{K(K-1)}\sum_{j\ne k}b_j$이 추가로 존재하며, 이것까지 포함해도 공통
affine이라 순위 결론은 유지된다.)

**명제 P5c (비등$n$ 반례 — 순위 뒤집힘).** 등$n$ 가정을 빼면 P5b는 성립하지 않는다.
반례: $K=2$, $n_1=3,n_2=1$ ($p_1=\tfrac34,p_2=\tfrac14$), $b_1=1,\ b_2=2$ (1차 surrogate 수준).

- 고정가중: $\phi^{\mathrm{fix}}_k=p_kb_k$ ⇒ $\phi_1=\tfrac34,\ \phi_2=\tfrac12$ — **클라 1 > 클라 2**.
- 재정규화: $\tilde u(\{k\})=b_k$ (혼자면 가중 1), $\tilde u(\{1,2\})=p_1b_1+p_2b_2=\tfrac54$.
  2-인 Shapley: $\phi_1=\tfrac12 b_1+\tfrac12(\tilde u(N)-b_2)=\tfrac12+\tfrac12(\tfrac54-2)=\tfrac18$,
  $\phi_2=\tfrac12 b_2+\tfrac12(\tilde u(N)-b_1)=1+\tfrac18=\tfrac98$ — **클라 2 > 클라 1**.
  (efficiency 검산: $\tfrac18+\tfrac98=\tfrac54$ ✓.)

순위가 뒤집힌다 (방향 규약과 무관 — 순서 자체가 반전). 즉 **두 게임은 일반적으로 같은 답을 주지
않으며**, 게임 선택은 실질적 결정이다. $\blacksquare$

비선형(2차 이상) regime에서도 어긋날 수 있다: $\tilde u$의 2차 항에는 $c_S^2$이 붙어
$S\mapsto c_S^2\Delta_S^\top H\Delta_S$가 P2 클래스(가산+쌍별)에 속하지 않으므로 공통 단조 사상
논증이 원리적으로 불가능하다. **등$n$·2차 반례(가산 게임에서도 뒤집힘)**: $K=2$, $p_1=p_2=\tfrac12$,
$H=I$, $\delta_1=(1,0),\delta_2=(0,2),g=(1,0)$ ⇒ $a_1=(\tfrac12,0),a_2=(0,1)$, 교차곡률
$q_{12}=a_1^\top Ha_2=0$이라 **고정가중 게임은 정확히 가산**(가산 갭 0)이다. 그런데 고정가중
$\phi=\{\tfrac58,\tfrac12\}$(클라1>클라2) vs 재정규화 $\phi=\{\tfrac{5}{16},\tfrac{13}{16}\}$
(클라2>클라1)로 **순위가 뒤집힌다**(efficiency 검산: 양쪽 $\sum\phi=\tfrac{9}{8}=$ 각 게임의 $v(\{1,2\})$
✓). 뒤집힘의 동인은 각 singleton 자기곡률의 $c_S^2$ 증폭($\tfrac12 c_S^2\Delta_S^\top H\Delta_S$,
$c_S=K/\lvert S\rvert$)이지 클라 간 상호작용 $\sum q_{ij}$가 **아니다** — 즉 **등$n$이고 가산이어도**
두 게임이 갈릴 수 있다.

**near-additivity의 이론 조건과 실측.** 게임의 가산 갭을 surrogate 수준에서 정확히 계산하면

$$\hat u_r(P_r)-\sum_{k\in P_r}\hat u_r(\{k\})
=\tfrac12\Big(\Delta W^\top H\Delta W-\sum_k a_k^\top Ha_k\Big)=\sum_{i<j}q_{ij},$$

즉 **surrogate의 비가산성은 전부 클라 간 교차 곡률항**이다 (1차 게임은 정확히 가산). exact 게임의
가산 갭은 여기에 Taylor 잔차가 더해진다:
$\lvert u_r(P_r)-\sum_ku_r(\{k\})-\sum_{i<j}q_{ij}\rvert\le(1+K_r)\frac{M_3^r}6\max_S\lVert\Delta_S\rVert^3$
(P3(ii)를 $P_r$와 각 singleton에 적용). 따라서 near-additivity의 충분조건: (i) 클라 delta들의
$H$-준직교($a_i^\top Ha_j\approx0$, $i\ne j$), 또는 (ii) delta 방향의 곡률 자체가 작음
($\lVert Ha_k\rVert$ 작음), 그리고 (iii) 3차 잔차 무시 가능. **실측**: 현행 무대의 가산 갭
$v(N)-\sum v(\{i\})$은 $\sum\phi$의 $\le$0.9% — 1B anchor +0.6~0.8%, 3B +0.1~0.3%, 7B $-$0.9%,
std20 0.0~0.5% (`research-wiki/wiki/flirds-signal-size-diagnosis.md` §1.5, 175–177행 — 문서는 2026-07-22 삭제, git 이력에서 조회; **인용만,
재계산하지 않음**). 이 **작은 가산 갭**이 함의하는 것은 정확히 (A) **같은 게임의 서로 다른 semivalue**
(Shapley/Banzhaf/…)가 같은 순위로 붕괴한다는 것이다(§1.5→§1.6 판정과 정합). 그러나 이는 (B) **고정가중
게임과 재정규화 게임(서로 다른 두 게임)**이 같은 순위를 준다는 것과 **다른 진술**이다: 위 등$n$·가산
반례는 고정가중 게임의 가산 갭이 0이라 (A)가 성립(자기 semivalue 순위 일치)하는데도 재정규화 게임과
순위가 뒤집힌다. 즉 고정≈재정규화 실무 근사를 정당화하는 것은 가산 갭 $\sum q_{ij}$가 아니라 조건 (ii)
**작은 곡률**($\lVert Ha_k\rVert$ 작음 → 2차 보정이 작아 P5b의 1차 지배 유지)이다.

**규모 조건 요약(수정).** 두 게임의 순위 차이는 다음 중 **어느 하나**로도 생긴다 (합집합 조건 — 논리곱
아님): (i) **비등$n$**(1차·정확 가산 게임에서도 — P5c), (ii) **부분참여**(등$n$·선형이어도 라운드-합
가중·희석상수 불일치 — P5b Remark), (iii) **2차 곡률**의 $c_S^2$ 증폭(등$n$·$H$-직교여도 — 위 반례).
"순위 차이 $=$ 비등$n$ $+$ 비선형 $+$ 비가산"이라는 **논리곱 요약은 틀렸다**(각 축이 독립 트리거이고,
가산성은 보호막이 아니다; P5c 반례는 정확히 가산인데도 뒤집힌다). 특히 배포 estimator는
`second_order=True` 기본(`flirds_estimator.py:65,108`)이라, P5b(1차)의 "등$n$이면 게임 선택 무관"
안심은 **실제 돌아가는 2차 estimator를 커버하지 않는다** — 등$n$·전원참여여도 (iii)이 남는다.

### 4.6 P6 — per-round 합산과 path-dependence

**성립하는 것 (라운드 합산의 대수).** frozen 게임 틀 안에서 라운드 합산 $\phi_k(U_b)=\sum_r\phi_k(u_r)$은
L0(c)의 선형성으로 **정확**하며(P1(ii)), 로그를 고정하면 $U_b$가 well-defined set function으로 환원되어
그 위의 Shapley 계산에 **계산상 모호함이 없다**. **전제**(이 무모호성의): (전1) server-side 무상태성
(F6 — 순수 가중합; FedAvgM이면 P7-ii로 깨짐), (전2) 각 섭동점에서 손실 유한(각 $u_r$이 실수값 게임이라
선형성 적용), (전3) $\sum_{j\in P_r}n_j>0$(§2.1 — 아니면 `_round_weight` NaN).

**단, path-dependence는 (b) 게임에 내재한다 — "계산 무모호"와 혼동 금지.** 두 종류를 구분한다. (A)
**counterfactual 비전파**($S$에서 $k$를 빼도 후속 $w$가 안 바뀜): 정의상 frozen out(진짜 없음). (B)
**게임 객체가 경로/이력-조건부 sequence function**: 전개점 $w_r$가 실현 궤적(과거 라운드에서 어떤 클라가
어떤 순서로 참여했는지)의 함수이므로 각 항 $g^r,H^r,w_r$이 훈련 경로에 의존하고, 또 참여횟수 정규화가
없어(`flirds_estimator.py:131`, docstring `:22–24`) $\phi_k$가 **클라 $k$가 표집된 라운드 수에 비례**
한다(한 번도 선택 안 된 클라는 데이터 품질과 무관하게 $\phi=0$; `server.py:41,46` 랜덤 선택) — 이것은
(b) 게임 **내부에 존재**한다. IRDS 원문도 이 순서 의존을 (b) 내부 feature로 규정한다(Remark 4·원문 86행·
이론노트 E.4: "기여가 데이터 사용 순서에 의존"). 따라서 "게임 내부에 path-dependence가 없다 / 오직 (a)
해석에서만 등장한다"는 **틀린 진술**이다. 올바른 진술: 라운드 합산의 **대수**는 무모호(계산 well-defined)
하되, path-dependence는 (b) 게임에 **내재**(sense B; sequence function)하며 고정 로그 조건부로 set
function 환원되어 계산만 무모호할 뿐이다 — 아래 sequence-function 승계와 일관된다.

**IRDS 논리의 이전 (정의로 우회).** IRDS는 per-step counterfactual을 스텝 내부에 한정하고
($\widetilde w_{t+1}(S)$는 실현된 $w_t$에서 출발; $S$ 선택이 이후 스텝을 바꾸는 연쇄는 utility에
미반영), 글로벌 utility를 $U(S)=\sum_tU^{(t)}(S)$로 **정의**한다 (원문 86행) — 근사 주장이 아니라
valuation 대상(게임)의 재정의다. 같은 구조가 라운드 수준에 그대로 이전된다: $u_r(S)$ ($S\subsetneq
\mathcal N$)는 " $S$만 참여했다면 $w_{r+1}$ 이후 궤적이 어떻게 달라졌을지"를 반영하지 않는 국소
counterfactual이고, $U_b$는 그런 국소 항들의 합으로 정의된 게임이다.

**미해결 공리화의 승계.** IRDS Remark 4(원문 388–392행): 특정 run의 utility는 순수 set function이
아니라 **sequence function**이며 "Extending Shapley axioms to sequence functions is not
straightforward" — model-specific attribution의 공리적 기초는 열린 문제로 명시된다. 이 미해결성은
Flirds에 그대로 승계된다 (클라 수준이라고 해소되지 않음). 논문 서술 시 P1·P2의 공리 성립 주장은
"고정된 frozen 게임(set function으로 환원된 대상)에 대한 것"으로 한정해야 한다.

**(a)/(b) 관계는 경험 보고로 한정 — 괴리 원인은 path-dependence 하나가 아니다.** (a)-retrain-val-loss
vs (b) in-run: 1B N=5 Spearman **+1.000**, 3B **+0.900** (estimator vs (b)는 양쪽 다 +1.000) — task6
dual-oracle, 2026-06-07, 루트 CLAUDE.md baseline 인용(노트 전용 — rundir 미영속). (a)-ROUGE의 괴리는 utility 함수 자체가 다른
별개 게임이므로 이론적 모순이 아니다("different game"). **괴리 원인 분해(최소 3축)**: (원1) **경로
의존성**(궤적 안정성), (원2) **가중 재정규화** — (a) retrain은 FedAvg 분모를 $\sum_{j\in S}n_j$로
재정규화(§3.3; `exact_sv_llm.py`)하는 반면 (b)는 고정가중 $\sum_{j\in P_r}n_j$이고, P5c는 비등$n$에서
이 두 가중이 1차에서도 순위를 뒤집음을 증명한다, (원3) **참여 샘플링**(realized partial vs retrain
`sample_frac=1.0`). **$R=1$ 반례(경로 의존성 0인데도 순위 반전)**: 단일 라운드면 후속 궤적이 없어
path-dependence가 정확히 0인데도, $K=2,\,n_1{=}3,n_2{=}1,\,b_1{=}1,b_2{=}2$에서 (b) 고정가중
$\phi_1{=}\tfrac34>\phi_2{=}\tfrac12$ vs (a) 단일라운드 재정규화 $\phi_1{=}\tfrac18<\phi_2{=}\tfrac98$
(P5c 값)로 순위가 뒤집힌다 — 궤적 안정성 가정만으로는 (a)≈(b)를 세울 수 없고 **등$n$ 또는 near-additive
조건이 추가로 필요**하다.

**경험 근거의 판별력 caveat.** 위 +1.000/+0.900은 path-independence의 증거가 **아니다**: (판1) 현행
무대는 near-additive·무신호(가산 갭 ≤0.9%, 자기순위 xseed $\rho\approx0$)라 **P5b가 순위 동일을 예측하는
등$n$ 축퇴 영역**이고 — 이 영역에선 path-dependence 유무와 무관하게 (a)·(b)가 같은 순위로 붕괴한다;
(판2) N=5 Spearman은 저검정력이라 $+0.900=1-\tfrac{12}{120}$ = **인접 1쌍 스왑** 딱 그 값이다(3B는 5개
중 한 쌍만 어긋남). 이론은 (a)≈(b)를 **보장하지 않으며**, 판별력 있는 검증은 비등$n$·비가산·비IID
셀(P5c 무대)에서 (a) vs (b) 순위 비교로 해야 한다(§9 후속 실험). 부가: (a)/(b) Spearman은 부호 규약이
반대(감사 §6.3: (a)=$-$val loss, (b)=loss 변화)라 부호 정렬이 선행돼야 하고, grand coalition 자체도
다르다((a)=전원참여 retrain 수준, (b)=realized partial run 변화 합) — 순위(scale-invariant)만 대응한다.
근사 정리에는 궤적 안정성 등 강한 추가 가정이 필요하며 본 문서 범위 밖 미해결로 남긴다 (§8).

### 4.7 P7 — momentum=0의 정확한 역할

momentum=0(F9)이 "클라-수준 게임 정의에 필요하다"는 서술은 부정확하다. 역할을 세 층위로 분리한다.

**(i) 클라-수준 게임: momentum 불필요 (stateless가 핵심).** F10에 의해 로컬 optimizer는 매 라운드
새로 생성된다 (`client.py:18`; `llm_server.py:49–53` — momentum buffer·Adam moment가 있어도 라운드
안에서 태어나 라운드 안에서 죽는다). 따라서 임의의 로컬 optimizer $O$에 대해
$\delta_k^r=f_O(w_r,D_k,\text{rng})$ — 라운드-국소 함수다. frozen 게임(D1·D2)과 그 닫힌형(P2)은
realized delta만 소비하므로 **로컬 momentum이 있어도 잘 정의되고 P1·P2·P3의 증명은 한 글자도
바뀌지 않는다** (P3은 $\ell$의 성질이지 $\delta$ 생성 과정의 성질이 아님).

**(ii) 진짜 load-bearing한 곳 (두 개의 서로 다른 knob — 정밀화).**
1. **fine per-step(K>1) 게임 — 클라 momentum=0(F9)은 여기서만 load-bearing.** 4.4.3의 극한 계산은
   $\delta_k=-\eta\cdot(\text{gradient})$ 구조를 요구하는데, **P4b는 $K=1$ full-batch 극한에서 진술
   된다**. 이 극한에선 PyTorch SGD가 첫 스텝에서 momentum buffer를 raw gradient로 초기화하므로
   ($v_0=g$, step$=\eta g$) momentum 유무와 **bit-identical**이고, 따라서 **momentum=0은 P4b 브리지에
   대해 vacuous**하다 — load-bearing은 '$K=1$ single-step full-batch'이지 momentum=0이 아니다. $K>1$
   에서는 per-sample 선형 구조가 **momentum과 무관하게 이미** granularity로 깨진다(P4-iv(a); coarse
   게임=다른 게임). 클라 momentum=0이 실제로 지키는 유일한 대상은 Flirds가 쓰지 않는 **fine per-step
   게임**(로컬 중간점마다 per-step 전개)이다 — 거기서 momentum·preconditioning이 스텝 이동을 과거
   gradient 누적 가중합으로 만들어 per-step 선형 구조를 파괴한다. `client.py:12–14` docstring("matches
   the per-step plain-SGD assumption of IRDS / Ripple Shapley … faithful to the realized per-step
   displacement")과 `codes/CLAUDE.md` §5 "momentum 하 2차 항 저하"의 의미가 이것이다: **(b) 게임·
   estimator 내적 정합성은 유지**되지만, (가) fine per-step/per-sample 해석이 깨지고, (나) gradient에서
   update를 재구성하는 방법(Ripple·GTG류)의 전제가 깨진다.
2. **server-side 무상태성 (F6 — 클라 momentum F9와 독립 knob).** 이것은 **server** momentum 문제이지
   클라 momentum=0(F9)이 보증하는 것이 **아니다** — 둘은 직교한다(클라 momentum≠0 & server momentum=0,
   또는 그 역도 가능). 우리 서버는 momentum이 없다 (F6 — 순수 가중합). 만약 server
   momentum(FedAvgM: $w_{r+1}=w_r+\Delta W^r+\mu m_{r-1}$)이 있다면, D1의 정의 자체는 유지되지만
   ($u_r$은 여전히 $S\cap P_r$만의 함수 — L1·P1 형식적으로 생존) **G1이 깨진다**: grand coalition
   섭동 $w_r+\Delta W^r$이 실현 스텝 $w_{r+1}$과 달라져 telescoping이 실패하고, 게임이 "실현 run의
   분해"라는 의미론(P5 근거 ②)을 잃는다. 복원하려면 게임 정의에 momentum 항을 넣어야 하는데 그러면
   $m_{r-1}$이 과거 라운드 delta의 함수라 라운드 간 오염이 생긴다 — IRDS의 per-step 문제가 라운드
   수준에서 재현되는 지점.

**(iii) IRDS 원 논문과의 정합성 비교 (관찰).** IRDS 이론은 vanilla SGD 전용이다 — momentum은 논문
전체에서 언급 자체가 없다 (이론 추출 노트 §7, grep 0건 [노트 경유]). 반면 IRDS의 Pile 실험은
**AdamW 궤적** 위에서 SGD-형 공식을 proxy로 적용했다 (원문 E.1 995행; Remark 7 1025–1027행 "using
SGD as a proxy for Adam" [노트 경유]). Flirds 무대는 진짜 plain SGD(mom=0, 상수 lr; F9) 궤적 위에서
SGD-이론을 적용한다 — **우리 무대가 원 논문 실험보다 이론-정합적**이다.

**Limitation.** stateful 로컬 optimizer(SCAFFOLD의 control variate, 라운드 간 유지되는 Adam
moment)는 $\delta_k^r$을 과거 라운드 참여 이력의 함수로 만든다. frozen 게임은 여전히 형식적으로
정의되지만(로그만 있으면 됨), "라운드 $r$의 값 = 라운드 $r$ 데이터 기여의 격리"라는 해석이
훼손되고 P4 브리지는 완전히 상실된다. 현행 프로토콜은 F10으로 이를 배제한다.

### 4.8 P8 — LoRA 부분공간 전개의 무해성

**설정.** LLM 트랙에서 훈련가능 파라미터는 LoRA 인자 좌표 $z$ (= `pkeys`, requires_grad 파라미터;
F7)이고, frozen base는 상수다. $\tilde\ell(z):=\ell(w_{\mathrm{base}};z)$를 **$z$-좌표 공간 위의
함수**로 정의한다 (backends의 `loss_fn(params, buffers)` — `flirds_estimator.py:3–8`).

**명제 P8.** 게임·estimator·oracle이 전부 같은 $z$-좌표에서 정의·연산되므로, P1–P7의 모든 진술은
$\ell\to\tilde\ell$, 파라미터 공간 $\to\mathbb R^{d_{\mathrm{LoRA}}}$ 치환 하에 **문자 그대로
성립**하며, 부분공간 제한이 추가 근사·오차항을 만들지 않는다.

*증명.* 확인할 것은 세 가지다. (1) 게임: D1의 섭동은 $z$-좌표의 delta 덧셈이다 — 로그가 LoRA-only
(F7·F8; `llm_server.py:83–84`), oracle의 `_split`이 `pkeys`로 params/buffers를 나눈다
(`in_run_sv.py:36–40`). (2) estimator: $g^r=\nabla_z\tilde\ell(z_r)$,
$u^r=\nabla^2_z\tilde\ell(z_r)\,\Delta W^r$ — `jvp(grad(vloss))` (`flirds_estimator.py:111`)는
정확히 $z$-좌표의 true Hessian-vector product다 (`vloss`가 $z\mapsto\tilde\ell(z)$ closure).
(3) P3의 가정: A4를 $\tilde\ell$에 부과하면 된다 — $\ell$이 $w$에서 $C^3$이고 LoRA 사상
$z\mapsto w_{\mathrm{base}}+B(z)A(z)$가 다항(각 성분 bilinear)이므로 합성 $\tilde\ell\in C^3$은
자동. 나머지 명제들은 파라미터 공간의 구체적 정체를 전혀 사용하지 않는다(내적·노름·Taylor만 사용).
$\blacksquare$

**[골격 초안 정밀화]** 초안의 "$\langle g_{\mathrm{full}},\Delta\rangle=\langle\nabla\tilde\ell,\Delta\rangle$ 자명" 서술은 파라미터화가 **선형일 때만** 자명하다. LoRA 사상은 $(A,B)$에 bilinear —
선형이 아니므로 full-weight 공간의 $g,H$와 $z$-공간의 $\nabla\tilde\ell,\nabla^2\tilde\ell$은 chain
rule의 Jacobian·2차 항으로 연결될 뿐 동일시되지 않는다. **올바른 진술은 "이론 전체를 $z$-공간에서
닫는다"**이며(P8), full-weight 공간 해석을 주장하지 않는 한 아무 문제가 없다. 우리는 그 주장을
하지 않는다.

**Remark (FedAvg-on-factors; 내적 일관성).** 서버는 LoRA 인자 $A,B$를 **인자 좌표에서 선형
집계**한다 (F6). 인자 곱으로 유도되는 full-weight 이동은
$\sum_kp_kB_kA_k\ne(\sum_kp_kB_k)(\sum_kp_kA_k)$이므로 "클라 full-weight 이동의 가중평균"이 아니다
— FedAvg+LoRA 문헌에서 알려진 집계 이슈다. 그러나 **게임(D1)·estimator(P2)·oracle 모두 실제로
돌아가는 프로토콜(인자-공간 FedAvg) 그 자체를 valuation**하므로 세 구성요소 사이의 내적 일관성은
완전하다. 인자-공간 집계의 채택 여부는 프로토콜 설계 선택이고, valuation fidelity(estimator↔oracle
일치)와 직교한다. — 부가 구현 제약: forward-mode HVP는 eager attention 필요
(`backends/llm.py:14–20` [감사 노트 §10 경유]) — 이론이 아니라 자동미분 구현 사항.

---

## 5. IRDS ↔ Flirds 대응표

| # | IRDS (원문 근거) | Flirds (본 문서) | 지위 |
|---|---|---|---|
| 1 | 플레이어 = 샘플 $z_i$ (48행) | 플레이어 = 클라이언트 $k$ | **신규** (P4 브리지; 단 정확 연결은 **CNN mean-CE 1-step 극한 전용** — LLM token-mean은 P4b Remark로 미성립) |
| 2 | sum-형 SGD update, $\eta_t$ coalition-독립 (74·81행) | 고정가중 FedAvg delta 합 $\Delta_S=\sum a_k$ (D1) | 승계 — 단 **가정으로 명시 필수** (P4a: 재정규화는 위반) |
| 3 | per-step local utility $U^{(t)}$ (76–79행) | per-round $u_r$ (D1) | 승계 |
| 4 | global utility 정의 + linearity 합산 (86행) | $U_b=\sum u_r$ + P1(ii) | 승계 |
| 5 | 배치 밖 샘플 $\phi=0$ — 외부 정리 인용 (83–84행, Wang&Jia Thm 5) | 비참여 라운드 기여 0 — L1 **직접 증명** | 승계+자체 증명으로 대체 |
| 6 | Thm 3 (1차 닫힌형; 116–130행) | P2의 가산부 | 승계 |
| 7 | Thm 4/6 (2차 닫힌형; 조합 항등식 증명) [노트 경유 §4.2] | P2 (unanimity 게임 분해 증명) | 승계 (증명 경로 상이, 결과 동형) |
| 8 | 잔차 $O(\eta^2)/O(\eta^3)$ — **비형식, 가정·정리 없음** (111행) | P3 (C³·$M_2,M_3$·적분형 잔차·L2 전파·$R$-누적) | **부재→신규 형식화** |
| 9 | 누적(스텝 합산) 오차 — 무진술; 검증도 단일 iteration (§5.2, E.2.1) | P3-1 우변 ($R$-합) + §7 실측 설계 | **부재→신규** (실측 완료 §7.2; verdict=CHECK) |
| 10 | realization-fixing·게임 재정의·Remark 4 공리화 미해결 (81·86·388–392행) | P6 — 라운드 수준으로 그대로 이전 | 승계 (미해결성도 승계) |
| 11 | 이론 = vanilla SGD 전용; 실험은 AdamW proxy (74행; E.1 995행·Remark 7 [노트 경유]) | P7 — 무대 자체가 SGD mom=0 (F9·F10) | 승계 + **정합성 개선** |
| 12 | ghost dot-product/HVP — 우변(배치 합) 고정으로 backprop 1회 (977–984행 [노트 경유]) | 클라 delta가 로그에 존재 → ghost 불필요; HVP 1회 붕괴 구조 동형 (P2 (3)단계, `flirds_estimator.py:109–111`) | 대응 구조만 승계 (per-sample 실체화 문제 자체가 소멸) |
| 13 | (병합/단위 변경 논의 없음; FL 언급은 352행 1문장) | L3 merge-consistency + P4b/P4c + 3차 반례 | **신규** |
| 14 | (재정규화 게임 쟁점 약함 — lr 고정 관행만) | P5 (근거 3 + P5b 순위 동일 + P5c 반례) | **신규** |
| 15 | (LoRA/부분공간 논의 없음) | P8 | **신규** |
| 16 | Hessian = val-loss true Hessian (111행) | 동일 (`flirds_estimator.py:15`, jvp∘grad) | 승계 |
| 17 | 부호: 형식 정의(loss 변화)와 응용 해석 사이 암묵 반전 (§1.4 논의) | §2.3에서 규약 명시 고정 ($\phi<0$=유익) | 승계 + 명시화 |

---

## 6. 코드 감사 발견 사항 인덱스 (관련 명제의 remark로 수록된 것들)

1. **estimator `n_clients` 자동 추론 함정** — 부분참여에서 최고 인덱스 클라 미참여 시 $\phi$ 배열
   축소; 명시 전달 필요 (`flirds_estimator.py:73–74,90–91`) → P1 Remark R1-b.
2. **fp32 전제의 비대칭** — estimator만 `.float()` 안전망, oracle은 로그 dtype 신뢰
   (`flirds_estimator.py:101,103` vs `in_run_sv.py:36–49`); "같은 게임" 주장의 정밀도 전제 → §2.1.
3. **null-player exact-0의 결정론 전제** — bit-identical forward 필요 (CNN cudnn_deterministic
   `server.py:92`; LLM 미검증 커널 비결정성 가능) → P2-1.
4. **banker's rounding** — `round(sample_frac·n)` 경계값 (`server.py:41`); 수학 무관, 재현성 각주감.
5. **감사 노트 §2.3의 과잉 주장 정정** — per-round 분해는 고정가중 없이도 성립 → P1 Remark R1-c.
6. **골격 초안 정정 2건** — P5 근거 ② (telescoping은 비차별적), P8 "자명" 서술 (bilinear 비선형성)
   → §4.5·§4.8.
7. **per-round 분해 ↔ $2^N$ 동치의 수치 확인은 문서 신뢰** — $\Delta\phi\approx3\times10^{-16}$
   (task 7c; 본 문서 재실측 안 함) → P1.

---

## 7. Taylor 잔차 실측 (기계 검증 완료 + 물리 검증 1B 3-seed 실측 완료: verdict=CHECK)

P3의 bound는 상수 $M_2^r,M_3^r$이 실측 불가하므로, (a) 코드↔수식의 **대수적** 정확성(P1·P2)과 (b) 잔차의
**물리적** 크기·2차 우위·$\lVert\Delta_S\rVert$ 스케일링(P3)을 나눠 측정한다. **(a)는 gpt2 CPU 스모크로
완료**, **(b)는 realistic $\lVert\Delta W\rVert$가 필요해 1B 본실행 대기**다. raw logs는 영속화되지
않으므로(F12) 둘 다 fresh FL run이 필요하다 (동일 seed 재실행으로 재현 — CNN 결정적, LLM은 GPU 커널
비결정성 미검증).

- 스크립트: `measure_taylor_residual.py` (본 문서 폴더). 1B 커맨드·비용: `RUN_1B.md`.
- gpt2 스모크 산출물: `gpt2_smoke_weakdelta_summary.json`, `gpt2_smoke_weakdelta_phi.csv`.

### 7.1 기계(대수) 검증 — 완료 ✅ (gpt2 CPU 스모크)

**셋업**: gpt2, CPU, $N=5$, $R=3$, train=40, val=10, max_steps=2, lr=1e-3, batch=4, maxlen=256,
lora_r=16, seed=0, renorm=True (`gpt2_smoke_weakdelta_summary.json` `config`).

| 검증 대상 | 지표 | 측정값 | 판정 |
|---|---|---|---|
| **P2** 닫힌형 = estimator | max_abs(closed − `flirds_values`) | **5.83e-12** (allclose rtol 1e-4 True; bit-identical **False**) | ✅ 항별 일치(fp32 재결합) |
| **P2** 2차 게임 Shapley = 닫힌형 | max_abs(û² 게임 exact $2^5$ Shapley − closed) | **7.61e-12** | ✅ 유도($\tfrac12 p_k\langle\delta_k,H\Delta W\rangle$) 정확 |
| **P1** 라운드 분해 = $2^N$ oracle | max_abs(perround φ − `in_run_shapley` $2^5$) | **3.93e-7** | ✅ (fp32 forward 노이즈 바닥) |
| **P2** 1차항 = flirds1 | max_abs(t1 − `flirds_values` 1차) | 3.87e-11 | ✅ |
| **P3** 잔차 부호 | resid_positive | True | ✅ |
| **G1** telescoping | grand-coalition gap | 2.4e-7 ~ 9.5e-7 | ✅ (fp32 바닥) |

부가: Spearman(exact vs t1/t2/closed/flirds2/renorm) 전부 **0.90** — 5점 중 인접 1쌍 스왑(작은 $N$ 포화;
판별력 아님). summary의 `verdict:"CHECK"`는 아래 물리 파트 미검증 때문이지 기계 검증 실패가 아니다
(sanity의 `closed_matches_flirds_values`·`t2_shapley_matches_closed_form` 등 대수 항목 전부 True).

### 7.2 물리(잔차 크기·스케일링) 검증 — 완료 (1B 3-seed 본실행; verdict=CHECK, 노이즈 바닥)

**왜 gpt2 스모크로는 물리 검증이 안 되는가.** max_steps=2·lr=1e-3라 $\lVert\Delta W\rVert\approx6.1\times10^{-4}$
(미미), $u_{\text{grand}}\approx-1\times10^{-5}$, 잔차 median $\approx6.5\times10^{-7}$ = **fp32 평가
노이즈 바닥** 근처다. 2차 비대칭항 $Q_{\text{asym}}\sim10^{-11}$이 노이즈보다 작아 순위가 노이즈에 묻힌다:
`t2_median_lt_t1_median = False`, frac($t2\le t1$) overall 0.33 → **2차>1차 우위 미관측**. loglog 기울기
(잔차 vs $\lVert\Delta_S\rVert$) $r_1\!\cdot\!r_2\approx0.30$(기대 2/3 아님) → **노이즈 지배 regime 확증**
— P3(i)(ii)의 $\lVert\Delta\rVert^2/\lVert\Delta\rVert^3$ 스케일링은 이 무대에서 관측 불가. realistic
$\lVert\Delta W\rVert$(1B, max_steps=10)의 본실행이 필요하다.

**측정 설계 (1B 본실행; `measure_taylor_residual.py`·`RUN_1B.md`)**:

- **무대**: Llama-3.2-1B-Instruct fp32+eager, LoRA r16, silo5 $N=5$, $R=10$, max_steps=10, lr=1e-3,
  batch16 — `RUN_1B.md` 기본값(2026-06-06 valuation-baseline 무대). 여력이 되면 3-seed.
- **라운드별 수집** (FedAvg 궤적 로그 훅):
  1. base $\ell(w_r)$ — forward 1회.
  2. exact $u_r(S)$ — 비어있지 않은 전 $S\subseteq P_r$ ($2^5-1=31$) forward.
  3. $g^r$ — grad 1회.
  4. **클라별 HVP** $h_k:=H^r\delta_k^r$ — 5회/라운드. (estimator 자체는 $H\Delta W$ 1회면 되지만
     — P2 — 전 $S$의 $\hat u^{(2)}$ 재구성에는 쌍별 $q_{ij}=p_ip_j\langle\delta_i,h_j\rangle$가
     필요하므로 측정 전용으로 추가.)
  5. $q_{ij}$ 캐시 후 전 $S$에 대해 $\hat u^{(1)}(S),\hat u(S)$를 행렬 연산만으로 재구성
     (추가 autodiff 없음).
  6. 재정규화 게임 $\tilde u_r(S)$ — 같은 로그에서 forward 31회 (P5 순위 비교 실측).
- **산출물**:
  - $(r,S)$별 표: $[u_r(S),\ \hat u^{(1)},\ \hat u,\ \lvert u-\hat u^{(1)}\rvert,\ \lvert u-\hat u\rvert,\ \lVert\Delta_S\rVert]$.
  - 스케일링 검증: $\log\lvert u-\hat u^{(1)}\rvert$ vs $\log\lVert\Delta_S\rVert$ 기울기 ≈ 2,
    $\log\lvert u-\hat u\rvert$ vs 기울기 ≈ 3 여부 (P3 (i)(ii)의 실측 대응).
  - $\phi$ 수준: 라운드별 exact Shapley(31 utility로 계산) vs P2 닫힌형 — 값 차·Spearman·Pearson;
    P3-1 우변과의 비교.
  - P5 실측: $\phi(\tilde u)$ vs $\phi(u)$ 순위 비교 (등$n$ 무대이므로 P5b의 예측 = 순위 동일;
    어긋나면 비선형 효과의 증거).
  - 부가: per-round 분해 = $2^N$ 동치 재확인 (§6-7 해소).
- **비용 추정**: 라운드당 forward 1+31+31 = 63회 + grad 1회 + HVP 5회; $R=10$이면 forward ~630회 +
  HVP 50회 — 1B fp32에서 (b) oracle 셀 1개와 같은 자릿수. 낮음.

**결과** (2026-07-13, **3-seed 완주**):

**Provenance.** 서버 이전 후 staging 리포(`flirds_batch`)의 venv python(**torch 2.12.0+cu130, B200
GPU**)로 재실행. 셋업 = Llama-3.2-1B-Instruct **fp32**, LoRA r16, silo5 $N{=}5$, $R{=}10$,
max_steps=10, lr=1e-3, batch16, val=100, renorm=on, check_inrun=on (전 seed `config` 동일, seed만
0/1/2). rundir(**READ-ONLY**): `outputs/taylor/llama1b_r10_seed{0,1,2}/{summary.json, coalitions.csv,
coalitions.parquet, phi.csv}`. seed당 wall ~47분. 아래 수치는 각 `summary.json`의
`pooled`·`sanity`·`phi_compare`에서 직접 추출.

**표 1 — 물리(잔차 크기·2차 우위·스케일링), pooled per-seed** (전 라운드 × 전 $S\subseteq P_r$ 통합;
resid1$=\lvert u-\hat u^{(1)}\rvert$, resid2$=\lvert u-\hat u\rvert$):

| seed | resid1 med | resid2 med | resid2/resid1 (med) | resid1 max | resid2 max | slope1 (기대 2) | slope2 (기대 3) | frac(resid2≤resid1) |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.753e-6 | 6.380e-7 | 0.364 | 1.545e-5 | 2.391e-6 | 2.719 | 1.635 | 0.810 |
| 1 | 1.549e-6 | 4.513e-7 | 0.291 | 1.307e-5 | 2.157e-6 | 2.263 | 1.565 | 0.806 |
| 2 | 1.277e-6 | 4.411e-7 | 0.346 | 1.004e-5 | 1.372e-6 | 1.818 | 1.466 | 0.787 |
| **평균** | **1.526e-6** | **5.10e-7** | **0.334** | — | — | **2.267** | **1.555** | **0.801** |

> fp32 평가 노이즈 바닥 = **2.384e-7** (전 seed 공통, `sanity.fp32_eval_noise_floor`). resid2 med
> (4.4–6.4e-7)는 이 바닥의 ~2–3배에 불과.

**표 2 — 대수(estimator=닫힌형 Shapley)·순위 보존, per-seed**:

| seed | max_abs(closed − flirds2) | max_abs(û² $2^5$ Shapley − closed) | max_abs(perround − inrun $2^5$) | Spearman(exact vs t1/t2/closed/flirds2/renorm) | closed=flirds_values |
|---|---|---|---|---|---|
| 0 | 3.41e-10 | 7.28e-10 | 3.74e-7 | **1.000** (전 5쌍) | True |
| 1 | 5.38e-10 | 7.41e-10 | 4.41e-7 | **1.000** (전 5쌍) | True |
| 2 | 5.54e-10 | 4.09e-10 | 4.57e-7 | **1.000** (전 5쌍) | True |

**공통 sanity 플래그** (전 seed): `resid_positive=True`, `t2_median_lt_t1_median=True`,
`t2_vs_t1_verdict="t2_better"`, `closed_matches_flirds_values=True`, `verdict="CHECK"`.

**해석.**

1. **2차 우위(resid2 ≤ resid1) — median·pooled서 확정, 라운드별론 노이즈 바닥 근접.** 3 seed 모두
   resid2 median < resid1 median (비 0.29–0.36, 평균 **0.33** → 2차 잔차가 1차의 ~1/3, 약 3배 작음),
   max에서도 resid2(≤2.4e-6) < resid1(~1.0–1.5e-5). `t2_median_lt_t1_median=True`·`t2_better` 전 seed.
   **단** 라운드별 frac(resid2≤resid1) = 0.79–0.81(1.0 아님) — 즉 **2차 항이 1차보다 작다는 것이
   median·pooled 수준에서 확인되나 라운드별로는 노이즈 바닥에 근접**해 일부 라운드는 뒤집힌다. gpt2
   스모크(frac 0.33, "2차>1차 미관측")보다 뚜렷이 개선 — realistic $\lVert\Delta W\rVert$에서 2차
   우위가 median으로 발현.
2. **스케일링 지수 — 1차(P3-i) 발현, 3차(P3-ii)는 노이즈 바닥에 갇힘.** loglog slope1 = 1.82–2.72
   (평균 **2.27**) → P3(i)의 $\lVert\Delta\rVert^2$ 지수 **≈2 발현**(gpt2 스모크 0.30 대비 회복).
   그러나 slope2 = 1.47–1.64(평균 **1.56**)로 P3(ii)의 기대 지수 **≈3에 크게 미달** — resid2가 fp32
   바닥의 ~2–3배라 3차 스케일링이 노이즈에 지배됨. **결론: 3차 지수는 1B에서도 미확증(노이즈 한계).**
3. **estimator = 고정가중 닫힌형 Shapley (P2) — ~$10^{-10}$까지 일치.** max_abs(closed −
   `flirds_values`) = 3.4–5.5e-10 (전 seed `closed_matches_flirds_values=True`), û² 게임 exact $2^5$
   Shapley = 닫힌형까지 max 4.1–7.4e-10. gpt2 스모크(5.8e-12/7.6e-12)보다 큰 것은 1B·$R{=}10$ **fp32
   누적** 탓이며 대수적 괴리가 아니다 — `t2_shapley_matches_closed_form=False`도 이 ~7e-10이 엄격
   bit-급 임계를 넘긴 것일 뿐(Spearman·값 모두 일치). exact Shapley 대비 Spearman(t1·t2·closed·
   flirds2) = **1.000**(전 seed).
4. **재정규화 게임 순위 보존 (P5) — Spearman 1.000.** exact vs renorm Spearman = **1.000**(전 seed)
   → 등$n$ 무대에서 P5b 예측(고정가중↔재정규화 순위 동일) 실측 확인. per-round 분해 = $2^N$ oracle
   재확인: max_abs(perround − inrun $2^5$) = 3.7–4.6e-7 = fp32 forward 바닥(§6-7 해소).

**종합 판정 = `verdict:"CHECK"` (전 seed 공통).** 대수(P1·P2·순위)는 전부 통과하나 **물리 신호(2차
우위·3차 스케일링)가 fp32 평가 노이즈 바닥(2.384e-7)에 근접**해 라운드별 판별력이 제한된다. 정리:
(i) **2차 우위는 median·pooled서 robust**(라운드별 ~0.80), (ii) **1차 지수 ≈2 발현**, (iii) **3차 지수
≈3은 노이즈 한계로 미확증**, (iv) **estimator=닫힌형·순위 보존은 ~$10^{-10}$ / Spearman 1.000으로
강건**. P3의 상계 지위(§4.3)는 불변 — 지수의 정밀 실측은 노이즈 바닥을 낮추는 프로토콜(고정밀 평가
또는 더 큰 $\lVert\Delta W\rVert$)을 요한다.

---

## 8. 한계·limitation 목록 (논문 수록용 초안)

1. **sequence-function 공리화 미해결 (승계)** — frozen 게임은 set function으로 환원된 대상이고,
   run-특정 valuation의 공리적 기초는 IRDS Remark 4가 명시한 열린 문제 그대로다 (P6).
2. **(a)↔(b) 관계는 실증적** — 이론은 (b) 게임이 retrain counterfactual을 근사한다고 보장하지
   않는다. 근사 정리에는 궤적 안정성류의 강한 가정이 필요 (미시도). 실측: 1B +1.000 / 3B +0.900
   (task6, 재검증 안 함) (P6).
3. **Taylor bound의 상수 미상** — $M_2,M_3$ 실측 불가; per-round 이동이 IRDS per-step보다 커서
   bound가 실질적으로 약하며, $R$-누적이 선형으로 쌓인다. §7 실측으로 보완 예정 (P3).
4. **per-sample 브리지의 극한성** — 로컬 1-step full-batch + plain SGD에서만 정확; $K>1$ 스텝은
   granularity 선택, minibatch는 realization-fixing으로 흡수되나 브리지 상실 (P4-iv a,b; P7-ii).
   게다가 분모 상쇄(reduction 분모 = 집계 가중)는 **CNN mean-CE(샘플=이미지) 전용**이고 LLM
   token-mean CE(분모=토큰수 ≠ 가중=시퀀스수)에서는 미성립(P4b Remark) — 대응표 브리지가 LLM
   valuation을 근거짓지 않음. 개입 arm(weights_fn≠기본)도 상쇄를 깬다.
5. **3차 이상 merge-consistency 실패** — 단위 변경(샘플→클라)이 2차 surrogate 밖에서는 값을 바꾼다
   (반례 존재; 크기는 3차 잔차 bound) (P4-iv c).
6. **고정가중 게임은 정당화된 선택이지 유일한 답이 아님** — 비등$n$에서 재정규화 게임과 순위가
   뒤집히는 반례 존재 (P5c). 채택 근거 3개(§4.5)는 논문에 명시해야 함.
7. **stateful 로컬 optimizer 미커버** — SCAFFOLD류·server momentum은 각각 브리지 상실·telescoping
   붕괴 (P7). 현행 프로토콜(F9·F10)이 배제하는 조건임을 가정으로 명시.
8. **LoRA는 인자-공간 valuation** — full-weight 공간 해석은 bilinear 파라미터화 때문에 별도 논증
   필요 (우리는 주장하지 않음); FedAvg-on-factors 집계 이슈는 프로토콜 선택으로 valuation과 직교
   (P8).
9. **정밀도·결정론 전제** — fp32 로그 프로토콜(§2.1)과 bit-identical forward(P2-1) 위의 진술들이
   있음; LLM GPU 커널 비결정성은 미검증 (§6-2,3).
10. **부분참여 호출 규율** — `n_clients` 명시 전달 (§6-1).
11. **near-additive regime의 함의** — 현행 IID-clean 무대에서는 게임이 사실상 가산적(갭 ≤0.9%,
    인용)이어서 2차 항·게임 선택의 실질 효과가 작게 관측될 수 있음 — 이는 이론의 문제가 아니라
    무대의 특성 (signal-size-diagnosis §1.5–1.6 참조) (P5).

---

## 9. Yonghee 결정 필요 + 후속 실험 제안

**결정 필요.**

1. **부호 규약의 논문 표기**: 내부 규약($\phi<0$=유익; §2.3)을 논문에서 유지할지, IRDS 응용 관행처럼
   반전(양수=유익)해 제시할지. 어느 쪽이든 **명시적 반전 선언**이 필요하다 (IRDS의 암묵 반전을
   지적하는 이상 우리가 같은 실수를 하면 안 됨).
2. **P5 재정규화 게임 비교의 배치**: 채택 근거 3개 + P5b/P5c(순위 뒤집힘 반례)를 본문에 넣을지
   appendix로 뺄지. 반례는 게임 선택이 실질적 결정임을 보이는 강한 재료지만, 리뷰어가 "그럼 왜
   재정규화 게임이 아닌가"를 더 파고들 수 있음 (답은 근거 ①–③).
3. **§7 실측 실행 — 완료 ✅** (2026-07-13): 서버 이전 후 staging(`flirds_batch`, B200, torch
   2.12.0+cu130)서 1B N=5 R=10 **3-seed** 재실행. 결과 §7.2 (rundir
   `outputs/taylor/llama1b_r10_seed{0,1,2}/`). 판정 = 2차 우위·1차 지수 ≈2는 median·pooled서 확인,
   3차 지수·라운드별 판별은 fp32 노이즈 바닥(2.384e-7) 한계(`verdict=CHECK`); estimator=닫힌형·순위
   보존 Spearman 1.000. **결정 불요 (해소)**.
4. **fine-game(per-step) 로그 훅**: P4-iv(a)의 granularity 갭을 실측하려면 per-step 로그가 필요
   (프로토콜 변경). 안 하면 서술로만 남김 — 논문상 "게임 정의 선택"으로 서술하는 것으로 충분하다는
   것이 본 문서의 판단.

**후속 실험 제안** (우선순위순).

1. **§7 Taylor 잔차·P5 순위 실측 — 완료 ✅** (2026-07-13, 1B N=5 R=10 3-seed; 결과 §7.2). 본 문서의
   유일했던 실측 공백 해소 (결정 3).
2. 비등$n$ rank-flip 실증 (P5c의 무대 재현): 클라 표본수를 비대칭(예: 3:1)으로 한 소규모 셀에서
   고정가중 vs 재정규화 순위 비교 — 반례가 실제 무대에서 발현되는지. 선택적.
3. fine-game 갭 소규모 실측 (CNN, K=수 스텝): per-step 전개 합 vs per-round 전개의 $\phi$ 차이 —
   granularity 선택의 크기 감각. 선택적 (결정 4에 종속).
4. **(a) vs (b) 판별력 검증** (P6, 반박 패널): near-additive·등$n$ 축퇴를 벗어난 셀(비등$n$·비가산·
   비IID)에서 (a)-retrain-val-loss vs (b) in-run 순위 비교 — 현행 +1.000/+0.900은 축퇴 레짐 + N=5
   저검정력(+0.900 = 인접 1쌍 스왑) 산물이라 path-independence·(a)≈(b)에 대해 **무정보**다. 부호 정렬
   (감사 §6.3) 선행. 항목 2와 같은 비등$n$ 무대에서 병행 가능.

---

*작성: 2026-07-04, 논문화 전 검증 세션 (항목 1). 근거 코드 스냅샷: main @ 004d076.*

---

### 반박 패널 처리 로그

2026-07-04 반박 패널 이슈 판정. **수용 36(중 #9 완화) / 기각 1(#11 내용없음) / 보류 1(JSON 잘림)**.
중복 이슈(7=14=17, 12=16, 21=25, 28=32, 30=33)는 한 편집으로 통합 반영. 코드 사실은 로컬
`codes/flirds/**`에서 직접 재확인(cnn.py·llm_server.py·backends/llm.py). §7은 gpt2 스모크 실측으로 재구성.

| # | 명제 | 이슈 (요지) | sev | 판정 | 반영 위치 / 근거 |
|---|---|---|---|---|---|
| 1 | P2 | A3 $C^2$가 CNN(ReLU/maxpool)엔 전역 거짓 | major | 수용 | §4.2 "A3 조각적-매끄러움 유의"(region-local Hessian; `cnn.py:22–26,42–46` 확인). 대수 결론 불변 명시 |
| 2 | P2 | 쌍대부 $K_r{=}1$에서 $K(K{-}1){=}0$ | minor | 수용 | §4.2 "경계 $K_r{=}1$" 추가(닫힌형 일치 증명) |
| 3 | P2 | $\sum n_j{>}0$ 미명시 | minor | 수용 | §2.1 "정칙성 전제(전 명제 공통)" 추가 |
| 4 | P2 | P2-1(ii) LLM forward 결정성 미검증 | minor | 수용 | §4.2 P2-1(ii) LLM 조건부 명시(`llm_server.py:82` "no cudnn-det"); 대수 exact-0 강건 강조 |
| 5 | P2 | "항별 일치" fp32 재결합 은폐 | minor | 수용 | §1·§4.2에 "~5.8e-12까지, bit-identical 아님" + fp32 전제 |
| 6 | P3 | 1차 shorthand $M_2/3$ = 3배 과소 | minor(sound) | 수용 | §4.3 명시적 1차 bound($M_2\max\lVert\Delta\rVert^2$, 계수 1) |
| 7,14,17 | P3·P4 | $M_3^r$이 sample-coalition 미포함 | minor | 수용(통합) | §4.3 A4의 $\mathcal U_r$을 클라+샘플 coalition 양쪽으로 확대 |
| 8 | P3 | "유일 괴리=잔차"가 fp32/결정성 조건부 | minor(sound) | 수용 | §4.3 코드대응에 fp32-로그·결정성 caveat 교차참조 |
| 9 | P3 | "첫 형식화" 과대 | minor(sound) | 수용(완화) | §1·§4.3 "표준 나머지항을 명시적으로 형식화"로 완화 |
| 10 | P3 | 상계 타이트함·지수 미관측 upfront | minor(sound) | 수용 | §4.3 "P3의 지위(상계·미검증)" + gpt2 slope 0.30 |
| 11 | P3 | "test"/"test" | minor | **기각** | 내용 없는 플레이스홀더 항목 |
| 12,16 | P4 | §1 브리지를 "무조건"에 오분류 | major | 수용(통합) | §1: L3만 무조건, 브리지는 "조건부" 절로 이동 |
| 13 | P4 | 요약 "정확히" 모호(surrogate?) | minor | 수용 | §1 브리지 불릿에 "surrogate 수준" 명시 |
| 15 | P4 | P4b 분모상쇄=CNN mean-CE 인공물; LLM token-mean 위반 | major | 수용 | §4.4.3 "Remark P4b-CNN"(`llm.py:78` vs `llm_server.py:85` 확인); 대응표1·한계4 caveat |
| 18 | P4 | §4.4.2 "정보 경로 없음" 절대주장 | minor | 수용 | 공유 RNG 순서 커플링 인정, frozen-게임만 생존으로 재서술 |
| 19 | P4 | $n_k{=}0$·free-rider 경계 | minor | 수용 | §4.4.3 "전제·경계(P4b)"(경1·경2) |
| 20 | P4 | separability(BatchNorm)·기본집계 전제 | minor | 수용 | §4.4.3 "전제·경계"(전1·전2) |
| 21,25 | P5 | 요약 "비등$n$+비선형+비가산" 논리곱 오류 | major | 수용(통합) | §4.5 "규모 조건 요약(수정)" 합집합 재서술 |
| 22 | P5 | 등$n$·2차 반례 생략 | minor | 수용 | §4.5 명시적 반례($\delta_1{=}(1,0),\delta_2{=}(0,2)$; 순위 반전, 검산 통과) |
| 23 | P5 | 가산갭→순위일치가 (A)vs(B) 혼동 | minor | 수용 | §4.5 (A)같은게임 semivalue vs (B)두 게임 분리 |
| 24 | P5 | P5b per-round인데 §1은 valuation-급; 부분참여 리프트 실패 | major | 수용 | §4.5 "Remark P5b-경계·범위"(3-클라 반례) + §1 "전원참여" 한정 |
| 26 | P5 | P5b 닫힌형 $K_r{=}1$ 미정의 | minor | 수용 | 같은 Remark (1)항 |
| 27 | P5 | foil $n_c{>}0$ + 기본 2차 estimator 미커버 | minor | 수용 | §2.1 정칙성 + §4.5 "second_order=True 커버 안 함" |
| 28,32 | P6 | "게임 내부 path-dependence 없음" 내부모순 | major | 수용(통합) | §4.6 para1 분리(대수 무모호 vs sense-B 내재; 참여횟수 비례) |
| 29 | P6 | (a)/(b) 괴리를 경로에 단일귀속; R=1 반례 | major | 수용 | §4.6 3축 분해(경로·재정규화·샘플링)+R=1 반례 |
| 30,33 | P6 | 경험근거 과대(축퇴 레짐·N=5 저검정력) | major | 수용(통합) | §4.6 "판별력 caveat"(판1·판2) + §9 후속실험4 |
| 31 | P6 | 참여횟수 비정규화·부호정렬·grand coalition | minor | 수용 | §4.6 para1(비례) + 마지막문단(부호·grand) |
| 34 | P6 | "IRDS per-step과 같은 지위" 허위등치 | major | 수용 | §4.4.4(a) "같은 유형이나 지위 아님; per-round=coarsening" |
| 35 | P6 | 전제 F6·손실유한·$\sum n_j$ 미명시 | minor | 수용 | §4.6 para1 "전제(전1–3)" |
| 36 | P7 | P7(ii).1이 K=1서 vacuous한 momentum=0 오귀속 | major | 수용 | §4.7 (ii).1 재서술(K=1 첫스텝=raw grad; load-bearing=1-step) + §1 |
| 37 | P7 | 클라(F9)/서버(F6) momentum 혼동 | minor | 수용 | §4.7 (ii).2 "F6=별개 knob" + §1 |
| — | P7 | (잘린 항목: P7(i) F10 문장) | — | **보류** | JSON 잘림으로 내용 미상; 본문 P7(i)는 이미 옳게 서술됨 |
