# 용어·표기 통일 규칙 (Terminology Conventions)

> 2026-07-29 용어 피드백을 기반으로 확정한 규칙.
> main.tex에 먼저 적용하고, 동일 규칙을 supplement.tex(부록 A–E)에도 적용한다.
> 여기 없는 표현이 애매하면 이 문서에 규칙을 추가한 뒤 적용할 것.

## 1. observed / realized / actual / fixed / frozen

| 단어 | 용례 |
|---|---|
| observed | 서버가 직접 관측할 수 있는 정보에만: `observed client updates`, `the server observes` |
| realized | 실제 한 번의 실행에서 발생한 것: `realized trajectory`, `realized participants`, `realized aggregation weights`, `realized FedAvg round`, `realized training run` |
| fixed | coalition을 바꿔도 고정하는 것: `weights fixed across coalitions` |
| actual | **사용 금지** — `realized`로 교체 |
| frozen | **사용 금지** (모델 동결로 오해) — `fixed-trajectory round game` 또는 그냥 `round game` |

대표 문장:
> We fix the realized trajectory, use the observed client updates, and keep the realized aggregation weights fixed across coalitions.

교체 규칙:
- `observed FL round` / `observed FedAvg round` → `realized FedAvg round`
- `observed training run` / `actual training run` → `realized training run`
- `actual aggregation` → `realized aggregation`
- `actual round` → `realized round`
- `frozen round game` → `fixed-trajectory round game` 또는 `round game`
- 예외: 서버가 **받았다**는 점을 강조할 때만 `observed updates` 유지 (동사 `observes`도 허용)

## 2. client / participant / player

- **client**: 전체 FL 시스템의 주체
- **participant**: 특정 round에서 선택된 client, 즉 $k \in P_r$
- **player**: Shapley game의 수학적 player = 참여 client의 **weighted update** $p_k^r\delta_k^r$

정의 문장(고정):
> The player associated with participating client $k$ is its weighted update $p_k^r\delta_k^r$.

- `players are the clients`와 `players are the (unweighted) client updates`를 번갈아 쓰지 말 것.
- player를 update로 지칭할 때는 항상 **weighted** update로 고정.

## 3. update / weighted update / displacement

| 대상 | 명칭 |
|---|---|
| $\delta_k^r$ | client update |
| $p_k^r\delta_k^r$ | weighted client update 또는 player vector |
| $\Delta_S^r$ | coalition displacement |
| $\Delta W_r$ | round aggregate displacement (짧게 round displacement) |
| $w^{r+1}-w^r$ | realized FedAvg update 또는 round displacement |

- coalition(집합)과 update(벡터)는 "coincide"할 수 없음. 반드시:
  > The grand-coalition displacement equals the realized FedAvg update.
- §3(FedAvg 설정)은 $\Delta w_k^r$, §4.2의 $\delta_k^r := \Delta w_k^r$ alias 도입 이후에는 $\delta_k^r$만 사용 (소문자 $\delta$ = 개별, 대문자 $\Delta$ = coalition/round 집계). 부록도 동일 기준.

## 4. 게임·값의 공식 명칭 (5개 고정)

1. **round game** $u_r$ — 원래의 nonlinear fixed-weight game
2. **surrogate game** $\hat u_r$ — round game의 second-order Taylor surrogate
3. **in-run Shapley** $\phi^{\mathrm{in}}$ — round game을 exhaustive enumeration하여 얻고 round에 걸쳐 누적한 값 (`exact` 접두어 없이 — 2026-07-29 결정)
4. **retraining game** $U^{\mathrm{re}}$
5. **retraining-based Shapley** $\phi^{\mathrm{re}}$

금지 명칭: `observed round game`, `original round game`, `frozen round game`, `federated round game`, `nonlinear game`

- `nonlinear round game`은 surrogate와 명시적으로 대비하는 한두 곳에서만 허용, 평소에는 `round game`.
- prior work는 항상 대문자 고유명사 **In-Run Data Shapley (IRDS)**로, 우리 target은 소문자 **in-run Shapley**로 구분. `exact in-run Shapley` 표기는 쓰지 않는다(2026-07-29 결정). 표 행 이름은 소문자 `in-run SV`.

## 5. utility / Shapley value / contribution / estimate / score

- **utility**: coalition-level 값 $u_r(S)$ — coalition이 갖는 것
- **Shapley value**: 게임에서 계산되는 client/player별 값 $\phi_k$ — client가 갖는 것
- **contribution**: Shapley value의 의미를 설명하는 일반 용어
- **contribution estimate**: Flirds의 출력 $\hat\phi_k$
- **score**: 서로 다른 정의를 쓰는 baseline들을 함께 지칭할 때만

Flirds 한 문장 정의(고정):
> Flirds estimates the in-run Shapley by computing the Shapley value of the surrogate game exactly.

(Flirds는 surrogate Shapley는 **정확히** 계산, in-run Shapley는 **근사**.)

## 6. exact / exhaustive / closed-form

- **exhaustive coalition enumeration**: 계산 **절차**
- **in-run Shapley**: 그 절차로 얻는 **결과**
- **closed-form surrogate Shapley value**: Flirds가 정확히 계산하는 결과
- 무엇의 exact 값인지 없는 `exact Shapley evaluation` 같은 표현 금지.
- 절차를 가리킬 때 `exact enumeration` 대신 `exhaustive enumeration`.

## 7. Taylor remainder / approximation error / truncation

- $u_r-\hat u_r$: **Taylor remainder**
- $\phi_k(u_r)-\hat\phi_k^{r}$: **player-level approximation error**
- `Taylor residual` **금지**, `Taylor truncation` **금지**
- `truncation`: GTG-Shapley의 Monte Carlo **sampling truncation**에만 사용 (cost 표의 truncation constant $c$ 포함)

대표 문장(고정):
> By Shapley linearity, its difference from the in-run Shapley is the Shapley value of the per-round Taylor remainder.

## 8. validation-loss decrease

$u_r(S) = \ell_{\mathrm{val}}(w^r)-\ell_{\mathrm{val}}(\cdot)$이므로:

- 수식·game 정의: **validation-loss decrease** (signed quantity)
- 일반적 모델 성능 설명: `model improvement` 허용
- `validation-loss reduction` **금지**
- `validation-loss improvement` → 가능하면 `decrease`로 교체
- Figure 1: `the realized validation-loss decrease of the round`

## 9. fidelity / accuracy / effectiveness / efficiency

- **fidelity**: Flirds와 target Shapley의 일치도
- **rank agreement**: Spearman $\rho$ / **value agreement**: Pearson $r$
- **test accuracy**: CNN task metric에만
- **practical effectiveness**: intervention 결과
- **computational cost**: 실행 비용 (계산 효율 얘기에 `efficiency` 단독 사용 주의)
- **the Shapley efficiency axiom**: 공리는 항상 이렇게 풀네임으로

예: contribution 3은 `Validation of fidelity and practical effectiveness`.

## 10. condition / threat / corruption

- **condition**: clean을 **포함한** 모든 실험 조건 (clean, answer-swap, zero-update, gradient noise, label-flip)
- **corruption condition**: clean을 제외한 네 가지 (corrupted condition ✗)
- **threat**: 명시적 threat model을 논할 때만

교체 규칙:
- `GSM8K threat axis` → `GSM8K condition axis`
- `per-threat results` → `per-condition results`
- `as the threat changes` → `as the corruption condition changes`
- `all four threats` → `all four conditions`
- `under every corruption threat` → `under every corruption condition`

## 11. coalition / subset

- 게임 정의 **이후**에는 coalition $S$
- 단순 집합 관계를 처음 설명할 때만 `subset`
- retraining 설명에서는 `client subset` 허용 (예: For each client subset $S$, we retrain ...)
- `participant subset` / `client subset` / `coalition`을 같은 문단·문장에서 혼용하지 말 것

## 12. aggregate / accumulate

- **aggregate**: 한 round 안에서 client updates를 합침 (`aggregate client updates`)
- **accumulate**: per-round Shapley values를 round에 걸쳐 더함 (`accumulate client contributions over rounds`)
- `aggregate contributions throughout training` 금지

## 13. round / step / run / trajectory

- **communication round / round**: FL의 한 round
- **IRDS training step**: IRDS의 한 gradient step
- **local optimization step**: client 내부 SGD step
- **training run**: 처음부터 끝까지의 한 실행
- **realized trajectory**: 그 run에서 발생한 $(w^r, P_r, \delta_k^r)$의 순서
- `step` 단독 사용 금지 — 항상 수식어를 붙일 것

## 14. 실험 구조 용어

- **track**: CNN / LLM
- **setting**: model–dataset–partition–participation 구성
- **condition**: clean 또는 corruption 종류
- **partition**: IID / Dir(1) (`split`보다 `partition`으로 통일)
- **seed**: 반복 실행
- **protocol**: 실험 설정 전반에만 (부록 B 제목 `Experimental Protocol`, main.tex의 `the full protocol (Appendix~B)`). selection-retrain / online gating 두 개입 방식을 총칭하는 데는 쓰지 않고 이름을 그대로 적는다. main.tex는 이 둘을 총칭하지 않고 `The online variant`로만 부르는데, supplement에서 `variant`는 Flirds-1st/Flirds 전용이라 그대로 가져올 수 없다 (2026-07-31 추가)
- **cell**: 쓰지 않는다. 표의 칸이나 행을 세는 자리에는 `rows`를 쓴다 (2026-07-31 폐기)

대표 문장:
> For each setting, condition, and seed, all methods are evaluated on the same realized trajectory.

## 15. HVP / JVP

- 이론·Figure·cost table: **HVP** (`one HVP per round`)
- 구현 설명·Algorithm: **JVP applied to the validation-gradient function** (`implemented with one JVP call`)

## 16. 수식 표기

- per-round 추정치: $\hat\phi_k^{r}$ — round 지수는 $\delta_k^r$, $\Delta_S^r$처럼 괄호 없는 위첨자로 통일. 괄호 위첨자는 first-order 표기($\hat u_r^{(1)}$, $\hat\phi_k^{(1)}$) 전용이므로 round 지수에 쓰지 않는다(피드백의 $\hat\phi_k^{(r)}$ 권고는 이 충돌로 미채택 — 2026-07-29)
- round displacement: $\Delta W_r$ ($\Delta W^r$ ✗)
- client update: §3은 $\Delta w_k^r$, §4.2에서 $\delta_k^r := \Delta w_k^r$ 도입 후에는 $\delta_k^r$만 사용 (같은 절에서 혼용 금지)
- coalition displacement: $\Delta_S^r$
- 참여자 수: 이론은 $|P_r|$, 실험에서 참여자 수가 고정일 때만 $K := |P_r|$ (정의 명시 후 사용)

$$
\delta_k^r \ \text{(client update)}, \qquad
\Delta_S^r \ \text{(coalition displacement)}, \qquad
\Delta W_r \ \text{(round displacement)}.
$$

## 17. 실험 지칭 방식 — 개별 setting에 고유명사 금지

> 2026-07-31 결정. main.tex를 전수 조사한 결과, **main.tex는 개별 setting에 이름을 하나도 붙이지 않는다.** 이름이 있는 것은 **track**과 **axis** 둘뿐이고, 나머지는 좌표값 + 괄호 스펙 + 표·그림 ref로 지칭한다. 부록도 동일하게 한다. §14와 함께 볼 것.

### 이름이 있는 층

| 층 | 값 |
|---|---|
| **track** | `the LLM track`, `the CNN track` |
| **axis** | `the <데이터셋> <훑는 변수> axis` — `the GSM8K condition axis`, `the Alpaca model-scale axis` |
| **partition** | CNN: `IID` / `Dirichlet($\alpha{=}1$)`(표·그림에서 `Dir(1)`). five domains: `IID` / `non-IID`(= one domain per client) / `Dirichlet($\alpha{=}0.5$)` |
| **condition** | §10 참조 (clean 포함 5개) |
| **방법 그룹** | `the renormalizing methods`, `the first-order methods` — `family` 금지(main.tex는 `methods`) |

### setting 지칭 3가지 장치 (이 외에는 이름을 만들지 않는다)

1. **`main` 형용사** — `the main LLM setting` / `the main CNN setting`. main.tex의 "The main LLM and CNN settings"에서 온 유일한 형용사이며, 대조로 정의된다(partial participation + 오염 혼합).
2. **괄호 스펙** — `(<데이터셋>[, <partition>][, <모델>]; $N{=}\cdot$, <participation>, $R{=}\cdot$[, <추가>])`
3. **표·그림 ref** — `the setting whose runtimes Table~4 of the main paper reports` (main.tex Table 4 자신이 `a single setting`이라 쓰고 이름을 거부한 선례)

### 폐기한 이름 (2026-07-31, supplement 전면 교체 완료)

| 폐기 | 대체 |
|---|---|
| `the $N{=}10$ grid` | `under full participation` / `the full-participation CNN settings ($N{=}10$)` |
| `the cross-device setting` | `the $N{=}100$ five-domain setting` / `the setting whose … Table~4 of the main paper reports` |
| `Alpaca anchor`, `small anchor`, `LLM small anchor`, `the anchor` | `the Alpaca setting at $N{=}5$ ($R{=}30$)` / 복수로 묶을 때 `the two Alpaca settings` |
| `IID five-way split` | `the five-domain corpus split IID` |
| `the LLM retraining leg` | 절 제목 `Retraining-Based Shapley on the LLM Track` |
| `GSM8K main setting`, `main GSM8K setting` | `the main LLM setting` |
| `Alpaca, partial participation` | `the Alpaca model-scale axis` (main.tex 이름 재사용) |

유지: `five-domain non-IID setting`은 데이터셋 `five domains` + partition `non-IID`이므로 이미 규칙에 맞다(이름이 아님).

### 표 행·패널 레이블

- **행**은 method / condition / model scale / partition 만 담는다. setting을 행 레이블에 넣지 않는다(main.tex 표에 그런 행이 없다).
- 산문에서 표의 한 행을 가리킬 때는 **`row`**. method 행과 reference 행(`vanilla`, `oracle-exclusion`, `selection-random`)을 함께 가리켜야 할 때도 `row`다 — `arm` 금지(2026-07-31 폐기). main.tex에 선례가 없고 임상시험 용어로 읽힌다. `method`·`baseline`으로는 대체할 수 없다: main.tex가 전자를 "seven contribution-evaluation methods"에, 후자를 Flirds 아닌 여섯 방법에 이미 묶어 두어 reference 행을 담지 못한다.
- **setting은 패널 헤더**(`\multicolumn`)가 괄호 스펙으로 진다.
- 패널 헤더의 participation은 분수일 때 단어를 빼고(`10/100`, `5/50`, `2/20`), full일 때만 `full participation`으로 적는다.

## 18. 표·캡션 규약

> 2026-07-31 결정. main.tex Table 1–3의 형태를 규칙으로 굳힌 것.

### 캡션 한 문장 규칙

> `<측정량 (단위)> [under <프로토콜>] (<데이터셋>[, <partition>][, <모델>]; $N{=}\cdot$, <participation>, $R{=}\cdot$[, <추가>]).`

- 문장 추가는 **컬럼 정의가 필요할 때만** (main.tex Table 4 선례). 나머지 설명은 전부 본문으로 내린다.
- 다중 패널은 `\textbf{Left:}` / `\textbf{Right:}` (main.tex Figure 1) 또는 표 내부 패널 헤더가 진다.
- **캡션에서 뺀 정보는 반드시 부록 본문에 드러나야 한다.**

### bold

- 비교가 일어나는 **축마다** 최고값을 굵게. mean 행/열이 있으면 거기에만, 없으면 조건(또는 각 지표 축)마다.
- **동점은 전부** 굵게 (main.tex Table 3 선례).
- 비교 대상이 아닌 항목은 굵게 하지 않는다: `in-run SV` 행/열(reference), `vanilla`, `oracle-exclusion`, `selection-random`.
- **비용 성격의 열**도 비교 축이 아니다 — 게이트가 덜 발화할수록 낮아지는 열(tab:gate-pr의 `false excl.`)은 낮은 값이 우월을 뜻하지 않으므로 굵게 하지 않는다.
- **조건이 값을 강제하는 칸**도 굵게 하지 않는다 — clean 조건의 precision처럼 오염 클라이언트가 없어서 값이 결정되는 칸.
- 음영 행/열 = Flirds.

### 패널 배치와 미정의 값 (2026-07-31 추가)

- 두 패널의 **행 레이블이 같으면 위아래로 쌓지 말고 열 그룹으로 나란히** 둔다. 구분은 `@{\qquad}`, 그룹 헤더는 `\multicolumn{n}{c}{\emph{...}}` + `\cline`(tab:inrun-fidelity 선례). 한 행에서 두 프로토콜의 대비가 바로 읽힌다. 폭이 한 단을 넘으면 `table*`로 올리고 `\tabcolsep`을 `4pt`로 줄인다.
- **분석 단위는 그룹 헤더**가 진다(`per client` / `per client-round`). 그 단위의 시점(run 종료 / burn-in 이후)과 새 열의 정의는 **본문**이 진다 — 캡션은 위의 한 문장 규칙을 그대로 지킨다(2026-07-31 수정).
- **em dash(`---`)는 정의 불가($0/0$)에만.** 조건 때문에 항상 0이 되는 값은 `0.00`으로 적고 왜 0인지는 본문이 설명한다 — 부록 C의 "An em dash marks a quantity undefined for that row or condition"와 어긋나지 않게.
- 조건 하나가 두 지표 모두 정의 불가라 행이 비면, 그 행을 지우지 말고 **그 조건에서도 정의되는 열을 추가**해 행을 살린다(clean 행을 살린 `false excl.`).

### 규약 문단 위치

부록 C 앞머리의 `\textbf{Table conventions.}` **한 곳**에서 supplement 전체 결과 표에 적용된다고 선언한다. 본문 §5.1이 이미 말한 것(3 seeds, `a±b`, SV 약어)은 재정의하지 않고 참조만 한다.

### 수치 서술

산문에는 **절대 측정값을 쓰지 않는다** (main.tex §5.2 방식: "attains high rank and value agreement", "collapses", "reverses sign"). 절대값은 표가 진다.

허용 예외:
- 차이·격차 — `within $1.4$ points`, `more than $6$ points`, `staying within $0.035$ of it`
- 비용 비율 — `$1/159$`, `$2^{10}/6.47 = 158.3$`
- 표에 없어서 산문이 유일한 근거인 값 — `$80.1\%$ of the enumerated coalitions`
- 개수 — `15 of the 16 rows`, `$5.0$ of the 20 corrupted clients`
- 프로토콜 하이퍼파라미터 — 오염 비율 `$40\%$`, burn-in 10 rounds 등

## 19. 절 제목

> 2026-07-31 결정. main.tex 제목 17개를 전수 조사해 굳힌 것.

- **명사구만.** main.tex에는 전치사구·동명사 제목이 하나도 없다. 2–6단어, Title Case.
- 같은 층의 제목은 **한 세트로 읽히게** 맞춘다: C.1 `Agreement with the In-Run Shapley` / C.2 `Value-Level Agreement` / C.3 `Agreement with the Retraining-Based Shapley`.
- 제목은 **표가 실제로 담은 것**을 덮어야 한다. 표에 열이 추가되면 제목도 다시 본다(`Precision and Recall of the LLM Gate` → false-exclusion rate 추가로 `Exclusions of the LLM Gate`).
- 부록 절 제목이 본문 절과 대응하면 본문 이름을 재사용한다(F `Computational Cost` ← 본문 §5.4 `Cost`, §9의 `computational cost`).

교체 이력:
| 폐기 | 대체 |
|---|---|
| `Against the In-Run Shapley` | `Agreement with the In-Run Shapley` |
| `Against the Retraining-Based Shapley` | `Agreement with the Retraining-Based Shapley` |
| `Repetition on MNIST` | `MNIST Repetition` |
| `Precision and Recall of the LLM Gate` | `Exclusions of the LLM Gate` |
| `The LLM Retraining Leg` | `Retraining-Based Shapley on the LLM Track` |
| `The Online Point` | `Online Gating` |
| `Fidelity: Full Tables` / `Intervention: Full Tables` | `Full Fidelity Results` / `Full Intervention Results` |

## 20. 문장 부호 — 세미콜론과 em dash

> 2026-07-31 결정. main.tex를 전수 조사한 결과, **main.tex의 산문에는 절을 잇거나 삽입구를 다는 세미콜론(`;`)이 하나도 없고, em dash(`---`)도 하나도 없다.** 두 부호는 표 칸·캡션 괄호 스펙·수식 같은 관행적 위치에만 나타난다. 부록도 동일하게 한다.
>
> 적용 대상은 `.tex` 파일의 영어 산문이다. 이 한국어 규약 문서 자체는 대상이 아니다.

### 금지 — 일반 문장(산문)

- **`---`로 삽입구·부연·동격을 다는 것**
  `A---including B---and C` / `normalized differently---by X and by Y---but ...`
- **`;`로 두 절을 잇거나 뒤에 부연을 붙이는 것**
  `... the remainder grows with it; Appendix C.5 measures it directly`

산문에서 곁가지를 달 일이 생기면 셋 중 하나로 바꾼다. main.tex가 실제로 쓰는 방법 순서다.

1. **문장을 끊는다** — 기본. 대부분 이걸로 해결된다.
2. **괄호에 넣는다** — 짧은 스펙·출처·부연.
3. **콤마 + 접속어로 푼다** — `, and` / `, so` / `, whereas` / `, since`.

### 예외 — 관행적으로 쓰는 위치 (그대로 쓴다)

| 부호 | 위치 | 예 |
|---|---|---|
| `---` | 표 칸의 **정의 불가** 표시 (§18 참조) | main.tex Table 2·3의 `oracle-exclusion & --- & ...` |
| `;` | 캡션·패널 헤더 **괄호 스펙**에서 모델/데이터셋 그룹과 규모 그룹을 가르는 자리 (§18) | `(CIFAR-10, Dirichlet($\alpha{=}1$); $N{=}10$, full participation, $R{=}10$)` |
| `;` | **괄호 안**에서 성격이 다른 항목을 가를 때 | `(Gaussian noise injected ...; CNN; \citealp{blanchard2017machine})`, `(\citealp{wang2025data}; Appendix~A.3)` |
| `;` | **캡션의 열 정의 나열** — 항목 안에 이미 콤마가 있어 콤마로 가를 수 없을 때만 | tab:taylor 캡션의 세 열 정의 (`... decrease; "2nd ≤ 1st" is ...; "floor" is ...`) |
| `;` | **수식·Algorithm·LaTeX 주석** | `\big\langle \delta_k^r,\; H_r \Delta W_r \big\rangle`, `\COMMENT{...; one inner product}` |

- 캡션의 열 정의 나열만은 main.tex에 직접 선례가 없다(main.tex Table 4는 괄호 + 콤마로 처리). supplement 자체 규약이므로, 콤마로 가를 수 있으면 콤마를 쓴다.
- 예외 위치라도 **한 캡션에 `;`를 두 번 이상 쌓지 않는다**. 두 번째부터는 문장을 끊거나 본문으로 내린다(§18 "캡션에서 뺀 정보는 반드시 부록 본문에 드러나야 한다").

### `--`(en dash)와 하이픈은 이 규칙과 무관 — 계속 쓴다

금지 대상은 **`---`(em dash)뿐**이다. `--`(en dash)는 그대로 쓴다.

- 범위: `$81$--$83\%$`, `clients $0$--$19$`, `$2.8$--$3.3\times$`, `Appendices~\ref{app:game}--\ref{app:remainder}`
- 대등 복합어: `question--response`, `instruction--response`, `dataset--partition`, `setting--condition`

복합 수식어의 하이픈(`second-order`, `zero-update free-rider`, `first-order methods`)도 무관하다.
