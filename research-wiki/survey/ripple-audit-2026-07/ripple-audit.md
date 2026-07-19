# Ripple Shapley 구현 감사·속도 진단 (논문화 전 검증 항목 2)

- 작성: 2026-07-04, 검증 세션(`PROMPT_VERIFICATION_SURVEY_2026-07.md` 항목 2). 읽기 전용 감사 — 코드·rundir 무수정.
- 대상 논문: "Ripple Shapley: Data Influence Attribution in One Federated Training Run" (AAAI 2026, proceedings pp. 28085–28093). PDF: `research-wiki/raw/papers/flirds/40034-Article Text-44125-1-2-20260314.pdf` (본문 9쪽, **Appendix A 부재** — 본문이 참조하지만 사본에 없음).
- 대상 코드: `codes/flirds/baselines/ripple.py` (CNN 포트, 218줄) / `codes/flirds/baselines/ripple_llm.py` (LLM 포트, 199줄). 러너: `codes/experiments/track_c1.py`(CNN), `codes/experiments/phase1_baseline_compare.py`(LLM).
- 문제의 수치: 논문 주장 **AFedSV+ 대비 62.37×, FedSV 대비 49.06× 빠름** (p.6, Table 1 기준) vs 우리 실측(1B N=5 R=10 val=100, 2026-06-06, `raw/conversations/flirds/2026-06-06-sv-baseline-port-and-results.md`) **Ripple ~4,515s = FedSV(~532s)보다 ~8.5× 느림**. 액면 그대로 겹치면 부호가 반전된 약 49×8.5 ≈ **420배 규모의 겉보기 모순** — 본 문서가 이를 해소한다.
- 표기: [확인] = 원문/코드에서 직접 확인, [추정] = 근거 있는 추론(검증 항목 표시), [부재] = 원문에 명시 없음 확인. 논문 정독·코드 해부의 1차 자료는 스크래치 노트(`recon/ripple_paper.md`, `recon/ripple_code.md`, `recon/cost_timers.md`, 2026-07-04; 원격 검증 세션 스크래치 — 저장소 미포함)이며, 본 문서 인용 라인은 전부 저장소 코드에서 재확인함.

---

## 0. 판정 요약

| # | 질문 | 판정 |
|---|---|---|
| 1 | 우리 포트는 논문 알고리즘에 충실한가 | **골격은 충실** (자체 궤적·per-step drop·클라별 로컬 Hessian sketch·progressive Q·cross-round chain 모두 구현). Algorithm 1의 모호 지점 2건은 각각 "α_k 비가중 스케치 합산", "코호트별 cross-round chain" 해석 채택(§1.2). 명시적 변형 = per-sample→per-client 축약, drop의 1차 내적 채택, LLM Hessian 서브샘플·축소 config. **LLM 포트는 미완 선언 상태** (streaming projection 미구현, LA/LM 미해소, §1.3) |
| 2 | 62×/49× vs 8.5× 느림의 격차 원인 | 모순 아님 — (i) **비교축이 다름**: 논문 배수의 분모는 학습 포함 in-run coalition-평가형 자체구현(그들 FedSV = plain training의 100×), 우리 FedSV는 frozen-logs 소비형(532s, 학습 제외). (ii) **d-스케일**: 그들 MLP/소형 CNN(오버헤드 ~5s/라운드 상수) vs 우리 1B fp32(≈1,129s/라운드). (iii) **회계**: 그들 수치는 학습시간이 분모·분자에 공통 포함되어 배수가 오히려 희석된 것 — valuation-only로 환산하면 그들 셋업에서 Ripple 우위는 94.8×/120.9×로 더 커짐. 논문의 유일한 절대 비용 주장은 "plain training의 2.05×"이며 LLM 스케일에 대한 주장이 아님 (§2) |
| 3 | eigsh CPU-spin의 원인 | **가설(CPU-spin/stall)은 실측으로 반박됨** (`measurements-eigsh-cpu.md`, 2026-07-04). eigsh는 전 호출 정상 수렴(12/12, ~115 matvec/call, maxiter=300 근처도 아님), 연산자 well-conditioned(\|λ\|max≈1.019), tol=0은 tol=1e-3 대비 matvec ~2배(109 vs 56)이나 **여전히 수렴** — 비수렴·maxiter 소진·stall 없음. Ripple 비용의 실체는 포트 버그가 아니라 **정상 수렴하는 다수 eigsh 호출(클라×라운드) × matvec 비용 + per-step val-grad**의 방법 고유 고유분해 volume (§3) |
| 4 | Ripple을 from-logs로 재구성 가능한가 | **불가 (확정)** — drop 항(per-sample·per-local-step gradient + 중간 파라미터)과 Hessian sketch(로컬 데이터 HVP)가 클라 로컬 데이터 접근 필수. `(w_r, δ)` 로그와 호환되는 것은 라운드 val-grad 하나뿐 (§4.2) |
| 5 | fidelity 비교 제외는 방어 가능한가 | **가능** — ① 측정 대상이 다른 게임(비인과 temporal 누적, 자체 궤적), ② 논문 스스로 exact-Shapley 대비 수치 fidelity 측정을 명시적으로 거부(수치 전무), ③ 우리 포트 미완 + 방법 고유 비용으로 full-grid 편입이 과학적으로도 예산상으로도 부적절 (§5) |

---

## 1. 충실도 감사 — 논문 ↔ CNN/LLM 포트 대조표

### 1.1 대조표

분류 라벨: **충실** = 논문 그대로 / **해석** = 논문이 모호·무명시라 우리가 택한 것 / **변형** = 의도적 이탈 / **단순화** = 비용상 축소 / **미완** = 선언된 미구현.

| # | 항목 | 논문 (위치) | CNN 포트 (ripple.py) | LLM 포트 (ripple_llm.py) | 분류 |
|---|---|---|---|---|---|
| 1 | 플레이어 단위 | per-**sample** φ(z) (Abstract, Eq.3–5). 실험 집계는 client-level u_i로 축약하나 축약 방법 [부재] (Eq.20, p.6) | 클라이언트 단위로 직접 계산 — drop을 클라의 실현된 로컬 스텝(배치 단위)에 걸쳐 누적 (:48–81) | 동일 (:55–79) | **변형** (우리 연구 질문이 client-level; 논문도 실험에선 client 축약 사용) |
| 2 | drop 항 형태 | Eq.5는 loss 차분으로 표기, §2.1은 "adopt this formulation"으로 IRDS 1차 내적(Eq.4) 채택 시사 — 어느 쪽인지 미확정 (p.2–3) | **1차 내적**: `drop += lr·⟨g_val, g_batch⟩` per step (:78) | 동일 (:74–77) | **해석** (논문 내부 모호; IRDS-1차 채택으로 해석) |
| 3 | drop의 gradient 입도 | per-sample 1-step (Eq.5: `w−η∇ℓ(w, z_i^k)`) | 실현된 **배치** gradient (:75–76) | 실현된 배치 gradient (:74) | **변형** (#1의 귀결) |
| 4 | z_val 위치 | "a local or shared validation sample" — 미확정 (Eq.5, p.3) | 서버 공유 val set 전체(val_x 통째 1-shot) (:66–67, :77) | 서버 공유 val 100개(10청크 합산, `_chunked`) (:48–52, :75) | **해석** (허용 범위 내; 단 "sample"→전체 val set으로 확대 = 비용 함의, §2.2) |
| 5 | 자체 궤적 실행 (in-run 프로토콜) | 학습과 valuation이 융합된 온라인 프로토콜 (Alg.1 전체) | 자체 FedAvg 궤적 처음부터 실행 (:172–196; logs 미수신) | 동일 (:155–180) | **충실** (다른 방법과 회계가 갈리는 원인이지만 이것 자체가 논문 요구) |
| 6 | 로컬 학습량 | "5 epochs" **pseudocode에 하드코딩** (Alg.1 L4) | 러너 config E (track_c1 full: E=5 — 일치; track_c1.py:85–86) | steps=4 (phase1_baseline_compare.py:59–64) | CNN **충실** / LLM **단순화** |
| 7 | 클라 선발 S(t) | partial participation (Alg.1 L2) | full participation `for c in range(n)` (:175) | full − free-rider (:159–162) | **변형** (cross-silo 셋업 자체가 full) |
| 8 | Hessian sketch 위치·데이터 | 클라의 **로컬 데이터** Hessian top-k (Alg.1 L6) | 로컬 데이터 **전체** 1텐서 (:158, :182) | 앞 `hess_bs`(compare=2)개 레코드 ≤256토큰 (:147) | CNN **충실** / LLM **단순화** |
| 9 | eigen 분해 방법·tol | **[부재]** — 방법(Lanczos/power/…), tol, 정밀도 전부 무명시 (전문 grep 0건) | scipy `eigsh`(ARPACK), which="LA", **tol=0**, maxiter=1000, ncv 기본→실패 시 4k+1 재시도 (:119–137) | eigsh, which="LA", tol=0, maxiter=300 하드코딩, ncv=max(2k+1,20), 재시도 없음, 부족분 zero-pad (:109–121) | **해석** (논문 무명시 → 우리 선택; 보고 의무 발생. LA/LM 결정은 **미완**, §1.3) |
| 10 | k, m, R | k=20, m=50 (p.5), R [부재] (case study는 R≈20 권고, p.7) | full: k=20/m=50/R=20 (track_c1.py:85–86) | compare: k=3/m=20/R=None→rounds=4 (phase1_baseline_compare.py:59–64; 함수 기본 k=8/m=40, :126) | CNN **충실** / LLM **단순화** (축소 config) |
| 11 | progressive basis Q (Eq.15) | OrthoProj "such as QR" — m 고정 유지 규칙 [부재] | `np.linalg.qr` 후 **앞 m열 절단** `Qn[:, :m]` (:141–145) | CNN 함수 재사용 (:45 import, :172) | **해석** (m 유지 규칙은 우리 자체 규칙 — 논문 공백) |
| 12 | 전역 H 합성 (모호성 ①) | Eq.10 직후 정의: `H^(t) = Σ α_k ∇²L_k` (**α_k 가중**); Alg.1 L11/L14는 클라별 B_k 표기로 합산 규칙 [부재] | 클라 스케치 **비가중 concat**: `Ur=[U_1|…|U_n]`, 체인 인자 = `I − lr·Σ_k U_kΛ_kU_kᵀ`(사영 후) (:185, :201–203) | 동일 수식 재기술 (:170–172, :182–199) | **해석+변형** (§1.2-①: Eq.10 정의 대비 α_k 누락 — 균등 데이터에서 유효 곡률 n배) |
| 13 | ripple 재귀 구조 (모호성 ②) | Alg.1 L13–14 문자 그대로 = 현 라운드 인자의 거듭제곱 ↔ Eq.12 = 서로 다른 라운드 H의 chain — 상충 | **코호트별 cross-round chain**: t0마다 `Plow = Ms[t0+r−1]@Plow`, r=2..R (:209–217) | 동일 (:182–199) | **해석** (§1.2-②: Eq.12 쪽 채택) |
| 14 | r 합산 범위 | r=2..R (Eq.13) | r=2..R, `t0+r<rounds` 가드 (:211–213; rounds≥3이어야 cross-round 항 발생, :208 주석) | 동일 | **충실** |
| 15 | J=I−ηH의 η | 전역 gradient-step lr (Eq.10–11; 라운드=1 global step 표기) | 클라 **로컬 lr** 사용 (:203; 우리 FedAvg는 server lr 없음) | 동일 | **해석** (논문의 η 정의가 멀티스텝 로컬 학습과 원래 괴리 — 논문도 동일 갭 보유, recon/ripple_paper.md §2.3) |
| 16 | ∂L/∂w^(t0+r) val grad | 서버 val 필요 (Eq.19; 서버 val 보유는 논문 미명시) | 라운드 시작마다 `VG[r]` (:173, :160–164) | `_val_grad_flat` (:156) | **충실** (서버-val 해석 하) |
| 17 | 통신 (L7: U_k∈R^{d×k} 업로드 = 모델 k배 업링크) | 프로토콜에 내재, 비용 논의 [부재] | 단일 프로세스 시뮬 — 통신 미모델링 | 동일 | 해당 없음 (양쪽 다 미논의) |
| 18 | streaming projection ((rounds,n,P) materialize 회피) | 논문에 해당 개념 없음 (우리 스케일링 요구) | 해당 없음 (P 소형) | **미구현** — materialize 유지 (:20–23 docstring; N=100 cross-device로 재이월) | **미완** |
| 19 | free-rider 처리 | [부재] | 없음 | zero/random delta + zero 스케치 (:144–147, :159–162) | 우리 확장 |
| 20 | 부호 규약 | good→high | good→high, 러너에서 −φ로 통일 (track_c1.py:235) | 동일 (phase1_baseline_compare.py:144) | 회계 규약 |

### 1.2 Algorithm 1 모호성 2건 — 우리 포트가 택한 해석

**① 전역 Hessian 합성 (Alg.1 L11/L14 vs Eq.10/17–18).** 본문 수식은 라운드 단위 전역 `H^(t) = Σ_{k∈S(t)} α_k ∇²L_k` (α_k = n_k/n_s)를 쓰는데, Algorithm 1은 클라별 `B_k^(t)`로 표기하고 합산 단계가 없다(문자 그대로 읽으면 마지막 클라 하나의 B_k만 남는 표기 오류). 우리 포트는 클라별 top-k eigenpair를 **비가중으로 concat**해 (ripple.py:185) 체인 인자를 `I − lr·(Bs·Ls)@Bsᵀ = I − lr·Σ_k Σ_j λ_{k,j} b_{k,j}b_{k,j}ᵀ` 로 만든다(ripple.py:201–203). 이는 `Σ_k U_kΛ_kU_kᵀ ≈ Σ_k ∇²L_k`, 즉 **비가중 합**이다. Eq.10 정의(α_k-가중)를 기준으로 보면: track_c1처럼 클라 데이터가 균등(α_k=1/n)이면 우리 인자는 `I − lr·n·H^(t)`로 **유효 곡률이 n배 과대**하다 [확인: 코드 산술]. ripple 항의 절대 크기에 영향을 주는 실질적 이탈이며, 논문 표기 오류에서 비롯된 해석 분기이므로 대조표에 "해석+변형"으로 기록한다. (rank 상관에 미치는 영향은 미검증 — §6 제안 4.)

**② ripple 재귀 구조 (Alg.1 L12–14 vs Eq.12/18).** pseudocode를 문자 그대로 실행하면 매 라운드 `P←I`로 리셋 후 같은 라운드 인자를 R번 거듭제곱하게 되어(Eq.18의 라운드 간 재귀와 상충), Eq.12의 "서로 다른 라운드 H^(t0+l)들의 chain"과 모순된다. 우리 포트는 **출생 라운드 t0별로 propagator를 유지**하며 r=2..R에 대해 `Plow ← M^(t0+r−1)·Plow`로 서로 다른 라운드의 인자를 곱하는 **Eq.12/13 쪽 해석**을 채택했다 (ripple.py:209–217). 이 해석이 본문 수식·오차 논의와 정합적이므로 타당한 선택으로 판단한다 [자체 결정: 논문 의도로 추정되는 쪽을 구현 — 근거는 Eq.12의 명시적 cross-round 첨자].

### 1.3 코드 주석의 "paper-faithful" 표현 — 부정확

ripple.py:120–123 주석은 `tol=0`을 "machine-precision pairs — **paper-faithful**"로 정당화하지만, **논문에는 tol·수렴 판정·고유분해 방법 자체가 전혀 없다** [부재 확인: 전문 grep "tol" 0건, "Lanczos" 0건]. 즉 tol=0은 논문 준수가 아니라 우리 선택이다 — 단 실측(§3)은 이것이 병리적 stall의 원인이 아니라 **matvec ~2배의 실재 비효율**(수렴은 정상)임을 확인했다. 같은 주석의 "1 iteration = 1 HVP"도 ARPACK 의미론과 상충하며, 실측상 maxiter는 재시작 수(재시작당 ≈ncv matvec)로 확인됐다(§3.1). LLM 포트가 해소한 것은 eigsh 견고성(고정 v0 + 비수렴 fallback + zero-pad)뿐이며(ripple_llm.py:85–86, :109–121), **streaming projection과 LA/LM 선택은 LLM 포트에서도 미해소**다(ripple.py:23–26, :125–126 → ripple_llm.py:20–23; 위키 이월 기록 `wiki/flirds-implementation-plan.md:70`).

**충실도 총평**: CNN 포트는 "논문 알고리즘의 합리적 해석 하의 성실한 구현"으로, 논문이 침묵한 지점(#9, #11, #12)에서의 선택을 보고하면 재현 주장이 가능한 수준. LLM 포트는 구조는 동일하나 축소 config(#6, #8, #10)와 미완 항목(#18, LA/LM)이 겹쳐 있어 **"Ripple의 LLM-스케일 재현"이라고 주장할 수 없고, 해서도 안 된다** — §5의 제외 논거 중 하나.

---

## 2. 속도 격차 규명

### 2.1 논문 62×/49×의 정확한 측정 조건 (전부 [확인], recon/ripple_paper.md §5)

- **측정 대상 = FL 학습 + valuation 합산 누적 벽시계 시간** (Table 1 캡션 "Average Cumulative Computation Time"; p.5 "baseline runtime corresponds to the duration of a conventional training run"). valuation-only가 아님.
- **셋업**: MNIST 2-layer MLP / CIFAR-10 "standard CNN"(구조 미명시), 로컬 5 epochs, 100라운드, batch 10, lr 0.01, **Tesla V100 2장**, 5회 반복 평균. **클라 수 N·참여율은 본문 어디에도 없음** [부재] ("Following the protocol of Sun et al. 2023"으로만 위임). Table 1은 "on Two Datasets"인데 수치 계열은 한 벌(두 데이터셋 평균으로 추정 [추정]).
- **베이스라인 구현 주체**: "all methods are implemented in PyTorch … under identical environments" — 저자 자체 구현으로 읽힘 [추정]; 코드 공개 링크 [부재].
- **원수치** (@Round 100, 누적 s): Ripple 984.98 / S-FedAvg 4,531.91 / FedSV 48,282.84 / AFedSV+ 61,437.57 / Plain 480.91. 배수 검산: 62.37× ✓, 49.02×(본문 49.06과 미세 불일치), 4.60× ✓, Ripple/Plain = **2.05×** ✓.
- **학습시간 제거 파생치** [우리 계산]: valuation 오버헤드만 비교하면 Ripple 504.07s(≈**5.04s/라운드, 거의 상수**) vs S-FedAvg 4,051s(8.0×) vs FedSV 47,802s(**94.8×**) vs AFedSV+ 60,957s(**120.9×**). 즉 **학습 포함 회계는 배수를 희석시키는 쪽**이며, 어떤 회계 정의를 쓰든 그들 셋업에서 Ripple이 우세 — 회계 정의 차이만으로는 우리 실측의 역전을 설명할 수 없다.

### 2.2 우리 실측의 측정 조건과 비용 구조 분해

**측정 셀** [확인]: `phase1_baseline_compare.py`, 1B(Llama-3.2-1B) LoRA r=16 (P≈11.27M), N=5(free-rider 1 제외 → 활성 4), fp32, val=100(10청크). Ripple은 **자체 축소 궤적** rip_rounds=4 × rip_steps=4, rip_k=3, rip_m=20, rip_hess_bs=2 (CFG :59–64, 호출 :140–143). 실측 3-seed ~4,515s (seed별 54–91분; `raw/.../2026-06-06-sv-baseline-port-and-results.md:47`). 비교 대상 FedSV ~532s는 **공유 궤적(10라운드×10스텝) frozen logs 소비형** permutation-MC(`fedsv.py:39`, n_perm=max(30, 2·|P_r|)) — 궤적 생성(~15분, :47)은 어느 방법 runtime에도 미포함(`phase1_baseline_compare.py:95–97` 타이머 밖).

**Ripple 4,515s의 구성 항** (호출 횟수 [확인]; 시간 배분은 이 LLM 셀에선 미분리이나 CNN 계측 사본이 상대 배분을 확정 — valuation 99.4%[eigsh 92%·matvec 80%] vs 로그생성 A 0.4%, `measurements-eigsh-cpu.md` §3):

| 구간 | 내용 | 횟수 (이 셀) | 근거 |
|---|---|---|---|
| A. 자체 궤적 로컬 학습+집계 | 함수형 수동 SGD(batch 4, maxlen 512) + FedAvg | 4r × 4클라 × 4스텝 = 64 배치-grad | ripple_llm.py:69–77, :173–179 |
| B0. drop per-step val-grad | 스텝마다 val 100개(10청크) fwd+bwd | 64스텝 × 10청크 = **640회** | ripple_llm.py:75, `_chunked` flirds_estimator.py:42–62 |
| B1. eigsh (HVP GPU + ARPACK CPU) | 클라·라운드당 로컬 Hessian top-k | **16회 호출**; 호출당 matvec **이론 상한** ≈ maxiter 300 × (ncv 20 − k 3) ≈ 5,100 HVP이나 이 상한은 **결코 도달하지 않음** — CNN 실측 기준 정상 수렴은 호출당 ~115 matvec (`measurements-eigsh-cpu.md` §1) | ripple_llm.py:109–121, :168 |
| B2. QR/사영/체인 (CPU numpy) | 라운드당 QR([P, m+n·k] = [11.27M, 35] fp64 ≈ 3.2GB) + Eq.16–19 체인 | 4회 QR + 종료 후 체인 1회 | ripple.py:141–145(재사용), ripple_llm.py:182–199 |
| B3. 라운드 val-grad | VG[r] | 4 × 10청크 | ripple_llm.py:156 |

산술적 정황: HVP 1회 ~0.5s(1B fp32; raw 06-06 :17 — 단 이는 estimator HVP 기준이고 Ripple의 hess_bs=2·≤256토큰 HVP는 더 쌈)로 놓으면 B1 **이론 상한**(16×5,100×0.5s ≈ 40,800s)은 실측 4,515s를 크게 초과한다. 이 격차의 메커니즘은 CNN 실측(`measurements-eigsh-cpu.md` §1–2)으로 확정됐다: eigsh는 well-conditioned 연산자(\|λ\|max≈1.019)에서 **완전·빠르게 정상 수렴**(호출당 ~115 matvec)하므로 (ncv−k) 기반 maxiter 상한은 **애초에 근처에도 가지 않으며**, 비수렴 재시도(fallback) 경로는 **트리거되지 않는다**(수렴 실패가 없어 진입 자체가 안 됨). 즉 상한 미달은 "부분 수렴·조기 fallback" 때문이 아니라 정상 수렴이 빠르기 때문이다. **B0(640 val-grad)과 B1의 실제 분담은 분리 계측으로 확정** — §4. 확실한 것: 자체 궤적(A)은 공유 궤적(10×10스텝, ~15분)보다 작은 4×4스텝이므로 **궤적 재실행 페널티는 총액의 지배 항이 아니다**(CNN 실측 A=0.4%, §4.1이 이를 확증).

### 2.3 격차 4-요인 분해 — "420× 겉보기 모순"의 해소

| 요인 | 내용 | 규모 감 | 성격 |
|---|---|---|---|
| ① 분모(비교군)의 축이 다름 | 그들 FedSV = **학습 포함 in-run coalition-평가형 자체구현**, 100라운드 누적 48,283s = plain training의 100.4×(오버헤드만 47,802s). 우리 FedSV = **frozen logs 위 소비형** 532s(학습 제외, 라운드당 재집계 forward만). 같은 이름이지만 측정된 물건이 다름 | 그들 표의 FedSV는 우리 표의 FedSV보다 두 자릿수 비싼 대상 (하드웨어·모델이 달라 정확 배수 환산 불가 — 정성) | 비교 방법론 차이 (우리 회계가 FedSV에 유리, §4) |
| ② 분자(Ripple)의 d-스케일 | 그들 valuation 오버헤드 ≈ **5.04s/라운드 상수** (MLP/소형 CNN, V100×2). 우리 ≈ 4,515s/4r ≈ **1,129s/라운드** (1B fp32 eager, P=11.27M; 로컬 스텝·에폭 수 차이 포함 러프) | 라운드당 ≈ **224×** [산술; 셋업 상이 러프] | **방법 고유** — per-step val-grad·eigen-sketch·QR이 전부 모델 비용 × d에 선형 이상. 논문은 d 스케일링 논의·형식 복잡도 제시가 전혀 없음 [부재: 유일한 복잡도 식은 naive O(r·d³)] |
| ③ 회계 정의 | 그들 = 학습 포함 누적(배수 희석, §2.1); 우리 표 = Ripple만 학습 포함·나머지 valuation-only | 그들 셋업 기준 valuation-only 환산 시 Ripple 우위가 94.8×/120.9×로 **커짐** — 이 요인은 역전을 만들지 못함. 우리 표 쪽 비대칭은 Ripple에 불리하나 **CNN 실측상 자체 궤적(A)은 0.4%로 미미**(§4.1) — 회계 비대칭은 역전의 주 원인이 아님 | 회계 (양쪽 다 명시 필요) |
| ④ eigsh 병리 (**기각**) | tol=0/fp32 부정합 → 비수렴 → matvec 폭증 + ARPACK CPU 스핀 **가설은 실측으로 반박** (§3, `measurements-eigsh-cpu.md`): eigsh 정상 수렴, stall 없음 | **병리 아님** — tol=0은 ~2배 matvec 비효율이나 폭주 아님; 나머지 비용은 방법 고유 고유분해 volume | **방법 고유** (tol 완화로 eigsh ~절반 절감 여지는 개선축, §6-1) |

**결론**: 논문 주장과 우리 실측은 **양립한다**. 논문의 62×/49×는 "느린 coalition-평가형 베이스라인 대비, 소형 모델, 학습 포함 누적"이라는 특정 조건의 실측이고, 절대 주장은 plain training의 2.05×뿐이다. 우리 실측은 "LLM 스케일에서 Ripple의 방법 고유 비용(②)이 from-logs 소비형 방법들과 비교 불가능하게 크다"는 것을 보여준다. 남은 실측 질문이었던 포트 요인(④)은 CNN 실측(§3)으로 해소됐다 — eigsh는 정상 작동하며 비용은 포트 stall이 아니라 방법 고유 고유분해 volume이다(tol=0의 ~2배 비효율만이 개선 가능한 포트 여지).

### 2.4 "방법 고유 비용" vs "우리 포트가 느린 것" 구분표

| 방법 고유 (어떤 구현이든 부담) | 우리 포트 요인 (개선 여지 있음) |
|---|---|
| per-step val-grad × 총 로컬 스텝 수 (drop 항의 정의; Alg.1 L5) — CNN full 셀 기준 47,000회의 full-val grad [확인: 5에폭×94배치×10클라×10라운드] | tol=0 → matvec ~2배 비효율 (**실측 확인**: 109 vs 56, `measurements-eigsh-cpu.md` §2 — maxiter 소진·stall 아님; tol 완화로 절감 가능, §6-1) |
| 클라당·라운드당 로컬 Hessian top-k 고유분해 (Alg.1 L6) — rounds×n_active회는 구조적 하한 | ARPACK(CPU fp64) 사용 자체 — GPU-상주 대체(lobpcg 등) 가능 (§6-2) |
| 온라인·클라 참여형 프로토콜 = 자체 궤적 필수, from-logs 불가 (§4.3) | drop의 z_val을 전체 val set(100개·10청크)으로 해석 — 논문 표기는 "a validation sample"이라 축소 여지 (§6-5) |
| d-선형 QR·사영·업링크(모델 k배) — LLM 스케일 비용의 원천 | eager attention 강제(forward-AD가 SDPA 미지원; backends/llm.py:14–20) — 방법이 HVP를 요구하는 한 부분적으로 고유 |
| 라운드≥3 이전엔 ripple 항 0 (ripple.py:208) — 짧은 궤적에서 방법의 존재 의의 자체가 축소 | 축소 config(4r×4step, k=3)조차 4,515s — full 편입 시 비용은 이보다 큼 |

---

## 3. eigsh 비용 규명 — CPU-spin 가설은 실측으로 반박됨

**요지**: 감사 초안이 세운 "tol=0 → fp32 수렴 실패 → maxiter 소진 → ARPACK/BLAS CPU-spin으로 멈춘 듯 보임"이라는 CPU-spin/stall 가설은, 같은 폴더의 완료된 실측(`measurements-eigsh-cpu.md`, 2026-07-04)으로 **반박됐다**. eigsh는 정상·빠르게 수렴하며, Ripple의 비용은 포트 버그성 stall이 아니라 **정상 수렴하는 eigsh를 클라×라운드만큼 반복하는 방법 고유 고유분해 volume**이다. 아래는 원래 가설 5단계와, 각각을 실측이 어떻게 판정했는지의 대조다.

### 3.1 원래 가설과 실측 판정 (`measurements-eigsh-cpu.md` CNN 계측 사본; CPU-only, 원 CNN 모델 P=61,706, n=4·R=3·k=8·m=16)

| # | 원래 가설 | 실측 판정 |
|---|---|---|
| 1 | **tol=0 ↔ fp32 부정합 → 비수렴 → maxiter 소진** [초안 추정] | **반박**. 궤적 12/12 호출 전부 `converged`, 비수렴·maxiter 소진 0건. tol=0은 tol=1e-3 대비 matvec ~2배(109 vs 56)이나 **여전히 수렴** — 실재하는 비효율일 뿐 병리적 폭주 아님 (§2 표) |
| 2 | **maxiter=implicit-restart 수 → HVP 상한 CNN ~21,000/LLM ~5,100회/호출** [초안 추정] | **의미론은 확인, 상한은 미도달**. 프로브(maxiter=3→57 matvec no_conv, =10→109 matvec conv)로 maxiter가 ARPACK 재시작 수(재시작당 ≈ ncv matvec)임을 확인. 그러나 well-conditioned 연산자에서 **실제 수렴은 호출당 ~115 matvec** — 상한(수천~수만)에 결코 근접하지 않음 |
| 3 | **matvec 사이 = ARPACK 단일스레드 CPU 구간이 지배 → GPU idle+CPU 100% stall** [초안 추정] | **반박**. 구간 계측상 eigsh wall의 대부분이 **matvec 자체**(spin8: wall 169s 중 matvec 158s=93%; wall≈matvec_sum+잔차). ARPACK 오케스트레이션(재직교화·QR)은 지배 항이 아님 — "matvec 사이 CPU가 도는 stall"이 아니라 그냥 matvec을 도는 것 |
| 4 | **BLAS spin-wait 증폭 → 겉보기 stall 강화** [초안 추정] | **미검증이나 무관**. OMP=1 대조는 한도로 미완. 단 §1–3에서 애초에 stall이 없으므로(전 호출 <1.5s 수렴) spin-wait가 설명해야 할 현상 자체가 없음 |
| 5 | **재시도 배가 (CNN 비수렴 시 ncv=4k+1 전체 재시도)** [초안 코드확인] | **트리거 안 됨**. 재시도 fallback은 **먼저 비수렴이 나야** 진입하는데, 이 연산자에선 수렴 실패가 없어 경로 진입 자체가 발생하지 않음 (프로브 `tol0_ncv4k1`: 수렴 시 104 matvec으로 오히려 비슷~약간 적음) |

부가 사실: 연산자는 well-conditioned(스펙트럼 반경 |λ|max≈**1.0188**, power iteration)라 ARPACK가 빠르게 수렴한다. LinearOperator는 dtype=fp64로 *선언*되나 실제 matvec은 fp32라 fp32 속도로 돈다(선언대로 fp64 matvec이었으면 ×3.1 느렸을 것; 프로브 `tol0_fp64op` 2.64 vs 0.85s). fp32 matvec은 결정적(absdiff 0.0)이며 fp32-vs-fp64 상대오차 ~3.5e-7은 정상 노이즈 바닥 — 비수렴 원인이 아니다.

이력 정황의 재해석 [확인]: 과거 로그의 "ShapleyFL LLM smoke hung on the tiny Ripple eigsh (known eigsh-convergence flakiness)" (`raw/conversations/flirds/2026-06-07-phase2-banzhaf-shapleyfl-lossheur-detector-regime.md:50`), "LLM eigsh(Lanczos 수렴 실패로 CPU 스핀; RIPPLE=0 제외 이력)" (`2026-06-12-track-cd-additional-experiments-design.md:96`)은 특정 tiny/degenerate config에서의 개별 flakiness 관찰이었다. 이번 계측은 그것을 **일반적 stall 메커니즘으로 일반화할 수 없음**을 보인다 — 궤적 설정에서 eigsh는 건강하게 수렴한다. (LLM GPU 포트의 H2D/D2H 왕복 비용은 이 CPU 실측으로 관측 불가 — §3.3 한계 참조.)

### 3.2 결론: Ripple 비용의 실체

eigsh는 정상 작동한다. Ripple이 비싼 이유는 포트의 stall이 아니라 **방법 고유의 고유분해 volume + per-step val-grad**다:

- 비용 = (클라 × 라운드) 개의 **정상 수렴 eigsh 호출** × (호출당 ~115 matvec × per-matvec HVP 비용) + 매 로컬 스텝의 drop val-grad + 라운드 val-grad. CNN 스모크에서 valuation의 92%가 eigsh, 그 안의 80%가 matvec (§4.1).
- tol=0은 matvec을 ~2배로 늘리는 **실재하는 비효율**이므로 tol 완화(§6-1)로 eigsh 비용을 절반 가까이 줄일 여지가 있으나, 이는 stall 제거가 아니라 정상 수렴의 반복 절감이다.

### 3.3 스케일 한계 — LLM 4515s는 같은 메커니즘의 스케일업

위 판정은 **CNN·CPU·P=6.2e4** 스케일에서 확정한 *메커니즘*(수렴 여부·matvec 카운트·tol 거동)이지 LLM 4515s의 절대치가 아니다. LLM 포트(P≈1e9, GPU, matvec이 전모델 HVP)는 **per-matvec 비용이 10²배 규모로 커진 같은 메커니즘의 스케일업**으로 설명된다 — 정상 수렴하는 eigsh를 클라×라운드만큼 도는 구조는 동일하고, 단지 각 matvec이 비싸질 뿐이다. 추가로 GPU 포트에는 매 matvec의 H2D/D2H 왕복 비용이 얹히나 이는 CPU 실측으로는 관측 불가 — LLM 스케일 항별 분해는 계측 사본의 GPU·1B 이식이 필요하다(§6-2, GPU 캠페인 종료 후). 어느 경우든 지배 비용은 eigsh+val-grad이며 "우리 포트의 버그성 stall"이 아니다.

---

## 4. 회계 통일을 위한 분리 계측 + from-logs 판정

### 4.1 계측 설계 (코드 무수정 — monkeypatch/계측 사본)

경계 정의: Ripple 총시간 = **(A)** 로컬 학습·집계(다른 방법에선 "로그 생성"에 해당 → 표에서 제외될 부분) + **(B0)** drop per-step val-grad + **(B1)** eigsh(HVP GPU 구간 + ARPACK CPU 구간) + **(B2)** QR/사영/Eq.16–19 체인 + **(B3)** 라운드 val-grad. **valuation-only 환산치 = B0+B1+B2+B3** (A 제외)로 정의하면 다른 방법과 회계가 통일된다. 단 B0는 훈련 루프 안에 융합돼 있어(스텝마다 val-grad; ripple.py:71–79) 코드 경계로는 A와 분리 불가 — 계측 사본이 필요한 유일한 지점.

삽입 지점 (상세는 recon/ripple_code.md §7):

| 구간 | CNN (`flirds.baselines.ripple`) | LLM (`flirds.baselines.ripple_llm`) |
|---|---|---|
| A+B0 | `client_drop_and_delta` 래핑 (호출부 :176–178); B0 분리는 L77 앞뒤 타이머를 넣은 **계측 사본** 주입 | `_client_drop_delta` 래핑 (:164–166); B0는 `_chunked` 패치(모듈 상단 import :41)로 통째로 분리 — **CNN보다 쉬움** |
| B1 | `local_hessian_topk` 래핑 (:182); 내부 GPU/CPU 분리는 `scipy.sparse.linalg` 네임스페이스 패치(함수 내 지역 import :92) + LinearOperator 서브클래스로 matvec 수·GPU 시간 기록 | `ripple_llm.eigsh`/`ripple_llm.LinearOperator` 속성 패치(모듈 상단 import :37) |
| B2 | `_orthoproj` 래핑 (:187) + 종료 후 체인은 전체−누계 잔여 | `_orthoproj`(:172, ripple_llm 네임스페이스) 동일 |
| B3 | 지역 함수라 직접 패치 불가 — 잔여 귀속 또는 계측 사본 | `_val_grad_flat` 패치 (:156) |

산출 스키마 (라운드 r × 클라 c): `{t_local_train, t_drop_valgrad, n_steps, t_eigsh_wall, n_matvec, t_matvec_gpu_sum, n_converged, t_qr, t_round_valgrad}` + 런 단위 `{t_chain_eq16_19, t_total}`. 이 스키마 하나로 §2.2의 시간 배분, §3의 eigsh 판정, 그리고 valuation-only 환산치가 동시에 나온다.

**실측 완료 (`measurements-eigsh-cpu.md` §3; CNN 계측 사본, n=4·R=3·k=8·m=16, P=61,706, CPU-only)** — 위 설계로 A vs B0–B3 분리 수치가 산출됐다(초안 예상 config N=8·rounds=8·k=5와 달리 실제 계측은 n=4·rounds=3·k=8·m=16으로 수행). 총 12.67s의 구간 분해:

| 구간 | 시간(s) | 전체 대비 |
|---|---|---|
| **A. 로그 생성** (local_train 0.048 + agg 0.002) | 0.050 | **0.4%** |
| **B. valuation** (B0+B1+B2+B3 합) | 12.592 | **99.4%** |
| — B1 eigsh wall | 11.701 | 92.4% |
| —— (그중 matvec_sum) | (10.103) | (79.7%) |
| — B3 round val-grad | 0.612 | 4.8% |
| — B0 drop val-grad | 0.103 | 0.8% |
| — B2 QR | 0.147 | 1.2% |
| — dW/HVP prep + Eq16–19 chain | 0.028 | 0.2% |

**valuation-only 환산 논리**: 다른 방법과 통일된 회계에서 Ripple의 "valuation-only" 비용 = B0+B1+B2+B3 (A 제외)이다. 이 CNN 스모크에서 **A(로그 생성)는 전체의 0.4%에 불과** — 즉 러너(`track_c1._timed`)가 Ripple의 자체 궤적까지 타이머에 포함해 생기는 회계 비대칭의 크기는 이 config에서 **미미**하다. Ripple이 비싼 진짜 이유는 자체 궤적 재실행이 아니라 **valuation 자체**이며, 그 92%가 eigsh(=클라×라운드 만큼의 정상 수렴 고유분해), 나머지 대부분이 per-step/round val-grad다. 이 A=0.4%/B=99.4% 회계는 baseline 표의 Ripple 항에 valuation-only 환산치를 병기하는 근거가 된다(항목 6 cost 방법론 문서와 공유). (주의: 이 비율은 CNN P=6.2e4·CPU 것; LLM 포트는 자체 궤적의 로컬 학습이 무거워 A 비중이 커지나, LLM 자체 궤적조차 공유 로그 궤적보다 짧아 4515s의 지배 비용은 여전히 eigsh+val-grad — §3.3.)

### 4.2 from-logs 재구성 가능/불가 판정 — **불가 (확정)**

공유 로그의 내용은 `logs = [(w_r, deltas_map)]`, `deltas_map[c] = (delta, n_c)` (fl/server.py:5–8) — 라운드 단위 모델·델타뿐이다. Algorithm 1이 요구하는 입력과 대조:

| Ripple 필요 입력 | 로그로 복원 가능? | 근거 |
|---|---|---|
| drop 항: per-sample(우리는 per-batch)·**per-local-step** gradient + 중간 파라미터 ("Store intermediate parameters", Alg.1 L5) | **불가** — 라운드 단위 δ로는 스텝별 궤적·gradient 복원 불가; 클라 로컬 데이터 필요 | recon/ripple_paper.md §6-1 |
| Hessian sketch: `∇²L_k(w^(t))`의 top-k (Alg.1 L6) | **불가** — 클라 **로컬 데이터에 대한 HVP** 필요; 로그에 없음 | §6-2 |
| z_val 접근 (Eq.5) | 서버 val 보유 시 가능 (우리 셋업은 보유) | §6-3 |
| ∂L/∂w^(t0+r) val grad (Eq.19) | **가능** — 로그된 w_r에서 사후 재계산 가능 (유일한 from-logs 호환 항) | §6-4 |
| 코호트별 사영 기여 벡터 Qᵀ∂w/∂z | **불가** — per-step gradient에서 파생 | §6-5 |

**판정**: Ripple은 본질적으로 **온라인·클라 참여형 프로토콜**이다(클라가 스케치·per-step utility를 계산해 업로드하는 구조; 업링크만 모델 크기의 k배). `(w_r, δ)` 로그만으로 만드는 것은 drop과 Hessian 두 축에서 원 알고리즘과 다른 근사를 강제하므로 "Ripple의 재현"이 아니라 **"Ripple에서 영감을 받은 서버-측 변형"**이 된다. 따라서 (i) 우리 실험 인프라(공유 로그 25셀 그리드)에 Ripple을 from-logs로 편입하는 것은 알고리즘 구조상 불가능하고, (ii) 우리 포트가 자체 궤적을 도는 것은 회계 비대칭의 원인이지만 **논문 충실성의 결과**다. 구현(from-logs 변형 설계)은 범위 밖 — §6-3에 설계만.

### 4.3 LLM 실측 (통일 회계, 3-seed B200)

§4.1은 CNN·CPU 스케일의 구간 분해(A=0.4%/B=99.4%)였고, §3.3이 예고한 **LLM 스케일 절대치**가 서버 이전 후 "실측2 통일 회계" 캠페인 3-seed로 확정됐다. 출처: `logs/cells/acct_seed{0,1,2}.log`(staging `flirds_batch`, B200 1장, venv torch 2.12.0+cu130, Llama-3.2-1B-Instruct LoRA r16, N=5 R=10 val=100 lr=1e-3; 러너가 `phase1_baseline_compare`를 monkeypatch해 공유 FL 궤적 wall-clock을 방법별 valuation과 분리 측정). Ripple은 여기서도 자체 축소 궤적(rip_rounds=4)을 타이머 안에 포함하는 **end-to-end 단일 수치**로 측정된다 — 내부 A/B0–B3 분리는 이 캠페인 소관이 아니라 §4.1 CNN 계측·§6 후속 GPU 이식이며, **LLM 항별 분해는 여전히 [실측 대기]**.

| 항목 | seed0 | seed1 | seed2 | 3-seed 평균(범위) |
|---|---|---|---|---|
| **Ripple end-to-end (s)** | 2,366.4 | 4,363.0 | 3,878.6 | **3,536 (2,366–4,363)** |
| 공유 FL 궤적(vanilla, 참고) | 434.6 | 407.3 | 387.9 | 409.9 (388–435) |
| coalition 평균(GTG·FedSV·Banzhaf·ShapleyFL·(b), s) | 520.6 | 538.7 | 542.7 | 534 (521–543) |
| Flirds valuation (s) | 105.0 | 108.2 | 108.5 | 107.2 (105.0–108.5) |
| **Ripple ÷ coalition** | 4.5× | 8.1× | 7.1× | **6.6× (4.5–8.1×)** |
| **Ripple ÷ Flirds** | 22.5× | 40.3× | 35.7× | **33× (22.5–40.3×)** |

- **판정**: LLM 스케일에서도 Ripple은 valuation-only 소비형 방법 대비 **압도적 최고비용** — coalition-sweep(≈534s, 학습 제외)의 **~6.6배**, Flirds valuation(107s)의 **~33배**. §2.3 factor ②(방법 고유 d-스케일 비용)의 직접 확증이다. 프롬프트/발표에서 흔히 인용되는 seed0 앵커 "≈4.5× coalition·≈22× Flirds"는 **범위 하단(seed0)**이며, 3-seed 평균은 6.6×·33×로 더 크다 — Ripple 절대치 변동이 크기(2,366–4,363s, ~1.8×) 때문(coalition·Flirds는 seed 간 ±3% 안정).
- **peak memory**: 로그 `[ACCT]` 라인은 공유 FL 학습 peak만 보고 = **30.35 GiB(3-seed 동일)**. Ripple(및 방법별) valuation peak는 이 캠페인에서 미계측(cost 방법론 §5.1-5 peak-mem 열 [실측 대기]와 동일 상태).
- **구서버 4,515s와의 관계**: 기존 baseline 문자열의 "Ripple ~4,515s"(구서버, 06-06)는 신규 B200 3-seed 평균 3,536s와 **직접 bit-comparable 아님**(서버 이전; venv torch 2.12.0+cu130). 비교의 기준 축은 **동일 run 내 방법 간 비율**(위 6.6×·33×)이고, 구서버에서도 Ripple이 FedSV 대비 ~8.5×로 최고비용이던 정성 결론과 방향 일치.
- **fidelity/detection(참고, 3-seed)**: Ripple은 자체 궤적이라 공유 (b)와 Spearman 미산출(전 valuation 방법 +1.000). AUROC noisy는 Ripple만 seed 편차 0.500/0.750/0.250(seed0/1/2)로 valuation 0.750 대비 불안정 — §5 "축소-config 포트라 보조 근거로만" 판정과 정합(free-rider AUROC는 Ripple도 1.000).

---

## 5. fidelity 비교 제외 정당화 (논문/리뷰어 방어 수준)

우리 1차 질문은 "(a)/(b) oracle 대비 기여도 측정의 fidelity"다 (루트 CLAUDE.md 핵심 질문 위계). Ripple을 이 비교에서 제외하는 근거는 서로 독립적인 세 겹이며, **"우리 포트 한계"와 "방법 고유 속성"을 분리해 명시한다**:

**① [방법 고유] 측정 대상이 같은 게임이 아니다.** (b) oracle은 공유 궤적 위 라운드별 coalition val-loss 게임의 exact Shapley다. Ripple의 φ는 (i) **자체 궤적**에서, (ii) per-sample 영향을 client로 축약해, (iii) drop(즉시 효과) + ripple(이후 라운드로의 temporal 전파)을 **누적**한 값이다 — ripple 항은 "라운드 t0의 업데이트가 t0+r 시점 val-loss에 미치는 영향"을 소급 가산하므로, 라운드별 marginal-contribution 게임과는 정의상 다른 대상을 잰다. 이미 06-12에 같은 이유로 개입 실험에서 제외됐다: "Ripple is C1-fidelity-only ... its full value is non-causal" (track_c2.py:18–19). oracle과 같은 게임을 근사하는 방법들의 순위 안에 다른 게임의 값을 넣는 것은 범주 오류다.

**② [방법 고유] 논문 스스로 oracle-fidelity 수치를 제공하지 않는다 — 명시적 거부.** "we argue that such metrics fail to reflect their practical utility in FL … numerical alignment of Shapley scores may not correspond to meaningful training dynamics" (p.6 "Measure of Attribution Accuracy") [확인]. 논문의 "high attribution fidelity"(abstract)는 exact Shapley 대비 rank correlation이 아니라 **downstream robustness(poisoning 하 test accuracy, Eq.20 가중 집계)** 근거의 주장이며, Spearman/Kendall류 수치는 논문 전체에 없다 [부재]. 즉 우리 fidelity 축에서 Ripple과 비교할 참조점을 원 논문이 제공하지 않는다.

**③ [우리 포트 한계 + 방법 고유 비용] full-grid 편입이 정직하지도, 가능하지도 않다.**
- 포트 한계 (정직 고지): LLM 포트는 **미완** — streaming projection 미구현(§1.1 #18; (rounds,n,P) materialize 유지 → N=100 cross-device 셀 실행 불가), which=LA/LM 미해소(§1.3), hess_bs=2·k=3·4r×4step 축소 config(§1.1 #8,#10). 이 상태의 수치를 fidelity 표에 넣으면 나쁜 결과가 방법 탓인지 포트 탓인지 분리할 수 없어 **원 방법을 부당하게 폄하할 위험**이 있다.
- 방법 고유 비용: 축소 config조차 ~4,515s/seed(§2.2) — from-logs 불가 판정(§4.2)으로 궤적 재실행이 필수라 25셀 × 3-seed 그리드 편입은 예산상 비현실적이며, 이 비용은 §2.4처럼 상당 부분 방법 구조(per-step val-grad, per-client·per-round 고유분해)에서 온다.
- 보조 실측: 우리 셋업(LLM·N=5·우리 corruption)에서 Ripple AUROC noisy 0.50±0.20으로 valuation 방법 중 최저(2026-06-06 baseline) — 단 이는 축소-config 포트의 수치이므로 제외의 **보조** 근거로만 사용한다.

**권고 서술 방식** [자체 결정 — 근거: 위 3겹이 각각 독립적으로 성립하므로 숨길 이유가 없음]: 논문에서는 Ripple을 (i) 관련연구에서 "temporal influence propagation을 재는 in-run per-sample 방법 — oracle-fidelity를 스스로 측정하지 않으며(명시 인용 ②), 우리 1차 축과 게임이 다름(①)"으로 위치시키고, (ii) runtime 표에 남길 경우 "자체 궤적 포함 회계 + 축소 config + 미완 LLM 포트" 각주를 붙이되, **완료된 분리 계측(§4.1: A=로그생성 0.4% / B=valuation 99.4%)** 기준의 valuation-only 환산치를 병기한다. "우리 포트가 미완이라 뺐다"가 아니라 "**방법의 측정 대상과 프로토콜 구조가 우리 비교 축과 양립하지 않음을 확인했고(①②), 포트 완성은 그 판정을 바꾸지 못한다**"가 방어의 골자다. 특히 비용 측면에서 **eigsh는 정상 작동하며(§3 실측) Ripple의 큰 비용은 우리 포트의 stall이 아니라 방법 고유의 고유분해 volume**임이 확인됐으므로, 제외 논거는 "느린 포트를 숨긴다"가 아니라 "게임·프로토콜 불일치 + 방법 고유 비용"에 온전히 선다.

---

## 6. 후속 실험 제안 (전부 본 감사 범위 밖 — 설계만)

1. **eigsh 파라미터 스위프**: `tol ∈ {0, 1e-6, 1e-3} × which ∈ {LA, LM} × ncv ∈ {기본, 4k+1} × maxiter ∈ {50, 300, 1000}` — §4.1의 eigsh 패치 래퍼에서 kwargs 오버라이드로 무수정 주입 가능. 재현 무대는 phase0_verify_ripple.py 패턴(CPU-only 가능). 판정치: matvec 수·수렴률·φ 순위 변화(tol 완화가 순위를 바꾸지 않으면 tol=0은 순수 비용 낭비였다는 결론).
2. **torch.lobpcg 대체 설계**: GPU-상주 블록 고유해법 — fp32 일관, matvec당 H2D/D2H 제거, ARPACK CPU 구간 제거. 제약: LOBPCG는 최대 고유값(LA 상당)측만 — LM 질문은 (H², shift 등) 별도 처리 필요. eigsh 결과와의 eigenpair 일치도·wall-clock 비교 스모크.
3. **from-logs 변형 설계 ("server-side Ripple 근사")**: §4.2 판정대로 원 알고리즘 재현은 불가하나, drop→라운드 단위 1차 내적(FedIF류), Hessian sketch→로그된 w_r에서 서버 val-Hessian(또는 Flirds가 이미 쓰는 HVP 1회/라운드 재활용)으로 대체한 파생 방법은 정의 가능. 명시적으로 "Ripple이 아닌 파생"으로 명명해야 하며, temporal 전파 항을 from-logs로 얻는 방법 계열은 문헌 공백 — novelty 후보.
4. **모호성 ①의 감도 실험**: 체인 인자에 α_k 가중(`Σ α_k U_kΛ_kU_kᵀ`, §1.2-①)을 적용한 버전 vs 현행 비가중 — CNN 스모크에서 φ 순위·값 변화 측정. 유효 곡률 n배 이슈가 실질인지 판정.
5. **drop z_val 비용 감도**: per-step val-grad를 1청크/서브샘플로 축소했을 때 φ 순위 보존 여부 — B0 비용(§2.2)을 1/10로 줄일 수 있는지. 논문 표기("a validation sample")에 오히려 가까워짐.
6. **streaming projection 구현**: N=100 cross-device에서 Ripple이 필요해질 때만 (현재 계획상 불필요 — track_c2/track_d/phase2 모두 Ripple 제외).

## 7. Yonghee 결정 필요

1. **논문 본문에서 Ripple의 처리 수위**: (i) fidelity 표 완전 제외 + 관련연구 서술만, (ii) runtime 표에는 회계 각주와 함께 유지(분리 계측 후 valuation-only 환산 병기), (iii) 부록 이동. 본 문서 §5는 (ii)를 기본안으로 제안 — 배제 사유를 숨기지 않고 보여주는 쪽이 리뷰어 방어에 유리하다는 근거. 최종 수위는 결정 필요. **→ [2026-07-19 결정 완료: (i) 채택 — baseline 완전 제외]**(Yonghee; 코드 부재·자체 구현 부담과 실측 성능 저조까지 종합). 근거 정리·리뷰어 Q&A = 이 폴더의 `ripple-baseline-exclusion.md`.
2. **baseline 문자열의 "Ripple ~4515s" 표기 갱신 여부**: 분리 계측 완료(§4.1, `measurements-eigsh-cpu.md`; CNN 스모크 기준 A=0.4%/B=99.4%) — 루트 CLAUDE.md·발표 자료의 해당 수치에 "자체 궤적 포함, 축소 config; 자체 궤적 오버헤드는 CNN 기준 미미(0.4%), 비용 대부분은 방법 고유 valuation" 주석 또는 valuation-only 환산치를 반영할지. (LLM 스케일 항별 분해는 계측 사본 GPU 이식 후.)
3. (실측 결과 조건부) §3 진단에서 tol 완화만으로 CNN full 셀 Ripple 비용이 유의미하게 줄면(§2 실측: tol=1e-3이 tol=0 대비 matvec ~1/2), track_c1의 Ripple 셀 재실행 가치가 있는지 — §6-1 스위프에서 φ 순위 보존 확인 후 판단.

---

## 검증 처리 로그 (2026-07-04 개정)

`measurements-eigsh-cpu.md`(2026-07-04 완료) 실측을 반영한 사실검증 개정. 사전 CPU-spin/stall 가설이 실측으로 반박되어 관련 절을 전면 개정.

| # | 심각도 | 이슈 | 판정 | 반영 위치 |
|---|---|---|---|---|
| 1 | major | §3이 eigsh CPU-spin/stall을 살아있는 유력 설명으로 서술(제목·5단계 메커니즘·§0 row3·§2.3 ④·§2.2 B1) — 실측은 정상 수렴(12/12, ~115 matvec, \|λ\|max≈1.019)으로 반박 | **수용** | §3 전면 재작성(가설↔실측 대조표 + "비용=고유분해 volume+val-grad" 결론 + 스케일 한계 §3.3); §0 row3, §2.3 factor ④(후보→**기각**), §2.2 B1(5,100=미도달 이론상한), §2.3 결론, §2.4 표, §5 갱신 |
| 2 | minor | §3.2·§4.1의 `[실측 대기]`/`산출 중` 마커가 완료된 측정을 미완으로 표기; §4.1 valuation-only 환산이 약속으로만 존재 | **수용** | §3.2 삭제→실측 판정으로 대체; §4.1에 측정된 A=0.4%/B=99.4% 구간 분해표 + valuation-only 환산 논리 삽입(실제 config n=4·R=3·k=8·m=16 명기); §7.2 갱신 |
| 3 | minor | §2.2 산술 정황이 상한-실측 격차를 "부분 수렴·조기 fallback"으로 설명 — 실측은 완전·빠른 정상 수렴(fallback 미트리거) | **수용** | §2.2 산술적 정황 문단을 "well-conditioned 연산자에서 정상 수렴이 빨라 상한 미도달, fallback 경로 미진입"으로 교체(measurements-eigsh-cpu.md §1–2 인용) |

기각: 없음 (3/3 수용). CPU-spin 정정 완료 — §3은 "가설→실측 반박(정상 수렴; 비용=eigsh volume×matvec+val-grad; tol=0은 ×2 비효율이나 폭주 아님)"으로 교체됐고 `measurements-eigsh-cpu.md`를 인용/링크한다.
