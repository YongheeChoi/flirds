# Flirds — 최상위 메인트랙 관점 리뷰 (review-claude)

- 작성: 2026-07-02, Claude (Fable 5). 기준 바: **NeurIPS/ICML/ICLR 메인트랙, 리뷰어 4인 생존 기준.** 워크숍/응용트랙 바로 낮추지 않음.
- 정본 원칙: **커밋된 rundir·analysis CSV(file-canon) > survey 분석 문서(06-25/06-26/07-02) > 위키 > CLAUDE.md·dossier 요약.** dossier(§2–§4)와 file-canon이 어긋나는 지점은 §0에 전부 표시하고, 본문 판단은 file-canon 기준으로 내린다.
- **방화벽 준수 선언**: `research-wiki/wiki/flirds-signal-size-diagnosis.md`(부록 A 원본), signal-size 관련 raw 노트, `survey/review/review-codex.md`는 **미열람**(파일명 존재만 확인). 투명성 공지 — 프로젝트 CLAUDE.md의 `next:` 필드에 해당 진단의 요약 1줄이 세션 컨텍스트로 노출되어 있었다. 본 리뷰의 관련 논점(C-2 "매칭 대상의 안정성")은 그 노출문이 아니라 레포 내 독립 근거(`runs/track_c/RESULTS.txt`의 (b) rho_xseed: IID cifar −0.042 / mnist −0.172; track_d 3-seed per-seed 분산)로 도출·인용했다.
- Pass 2(K·L)는 부록 A 수령 후 하단에 추가됨(2026-07-02 수령분 반영).

---

## 0. 레포 검증 부록 (§6): dossier ↔ file-canon 대조

### 0.1 불일치·주의 표 (심각도순)

| # | dossier 주장 | file-canon 실측 | 판정·파급 |
|---|---|---|---|
| R1 | "3B: (a)-valloss vs (b) **+0.900**" (§3.5) | **3B에 (a)-retrain oracle이 없음**(track_d 3B_anchor5·phase2 3B_silo5 모두 `oracle_a:false`). +0.900은 **1B**_anchor5의 per-seed 값(0.900/1.000/0.900, 평균 **0.933±0.047**, `runs/track_d/fidelity.csv`) | **오귀속.** 논문에 이대로 들어가면 fabrication 수준 오류. 3B (a)는 미실행으로 정정 필수 |
| R2 | "(a)=(b)=estimator **+1.000** (N=5 1B fp32), (a)-ROUGE diverges" (§3.1/§3.5, task6) | **커밋된 rundir 부재.** fp32 (a)+(b)+ROUGE 발산 세트는 smoke/sanity 스크립트 경로에만 존재. file-backed (a)는 track_d 1B_anchor5 **0.933±0.047** 한 칸 + CNN 2^10 (a)(발산, 아래 R8) | dual-oracle의 "+1.000" 헤드라인 근거가 **노트 전용**. 논문 인용 불가 — 재실행·커밋으로 정본화 필요 |
| R3 | "anchor N=5: **전 방법** Spearman +1.000" (§3.5) | silo5(5-domain) noisy/frrand/frzero에서 7종은 1.000이나 **FedIF 0.90–0.93**, **poison 셀은 붕괴**(Flirds-1st 0.000·FedSV 0.367·GTG 0.867). 그리고 track_d anchor5(IID)는 **FedSV 0.700·ShapleyFL 0.700·ComFedSV 0.500·FedIF 0.067**로 크게 갈림 | "전 방법 +1.000"은 특정 무대·특정 위협의 부분집합. 두 개의 서로 다른 "anchor N=5"(5-domain vs IID-alpaca)가 dossier에서 합쳐져 있음 |
| R4 | Pearson "0.99999+ **전 스케일·전 레짐**" (§3.5) | benign 셀은 성립. 반례: **poison silo5 Flirds 0.684, 3B poison −0.893**(1-seed), 7B_std20 seed1 0.99991 | 값-수준 fidelity는 poison에서 **정확히 붕괴** — 오히려 논문의 핵심 재료(2차 Taylor trust-region 경계)인데 dossier가 뭉갬 |
| R5 | 비용 "5–15× 저렴" 일반화 (§3.5) | cohort 의존 **crossover 존재**: std20(라운드 코호트 k=2)에서 **(b) exact 2917s < Flirds-2차 4697s**. k=5: 707s vs 3528s(5×), k=10(device100): 157s vs 24975s(**~160×**) | 비용 주장은 cohort-조건화 없이는 반례로 반박당함. 조건화하면 오히려 더 강한 주장(O(1) vs 2^k) |
| R6 | Ripple LLM ~4515s (§3.5) | **Ripple은 CNN(track_c) 전용.** LLM 결과 파일에 Ripple 없음(`track_d/fidelity.csv`에 열 자체가 없음) | LLM 런타임 표에서 Ripple 제거 또는 CNN-측정으로 귀속 정정 |
| R7 | "(b) 771ms/fwd → ~11h/4-GPU", "N=5↔10 32×", "estimator/round ≈ 0.31·val·seq ms" (§3.5) | file-canon에 부재. 실측은 device100 (b) **24975±1115s/run(≈6.9h)** wall-clock 뿐 | 파생 추정치. 논문엔 실측 wall-clock + 연산량 모델로 대체 |
| R8 | (a)↔(b) 일치가 방법 검증을 완성 (§3.1) | **CNN C1에서 (a)와 (b)가 발산**: Flirds vs (a) 0.352±0.462, **전 방법 vs (a) ≤0.45**(최고 ShapleyFL 0.453). LLM 1B anchor5는 0.933 | "(b)를 맞추면 (a)도 맞다"는 무대 의존. CNN에서 in-run 게임 ≠ retrain 게임 — 논문이 정면 해명해야 할 construct 이슈 |
| R9 | CNN "ComFedSV Spearman {1.0, 0.96, 0.85, 0.84}" (§3.2) | track_c grid 산출물에 해당 fidelity 표 미생성(`track_c/fidelity.csv` 부재; RESULTS.txt엔 stability·(a)-oracle만). Phase 0 구수치로 추정 | 검증 불가 수치. 논문 인용 금지 |
| R10 | Track D "계획됐으나 미실행" (§4) | **완료됨**: `runs/track_d/rundirs/` 18개 = {1B,3B,7B}×{std20,anchor5}×3-seed, 전부 실모델(Llama-3.2-1B/3B, Llama-2-7B). fidelity+개입+수렴+비용 산출 완료 | dossier·plan.md(§3.11 "real run not yet")·CLAUDE.md가 전부 **스테일**. 특히 **7B 3-seed fidelity가 이미 존재** — dossier가 과소보고 |
| R11 | ShapleyFL β | 코드는 β0.3(e89af94, 06-25)으로 바뀌었으나 **모든 커밋된 rundir는 β0.5 산출물**(meta git_sha 전부 e89af94 이전). β0.3 재실행 큐(163셀)는 **미실행** | 프로비넌스 불일치(코드≠결과). β는 ShapleyFL 방법·arm에만 영향 — 타 방법 수치는 유효. 논문 전 재실행 또는 β0.5 명기 필요 |
| R12 | oracle 순위 안정성 "IID ~0, 비IID·오염 0.51–0.97" (§3.5) | RESULTS.txt 실측: IID **−0.042/−0.172(≈0 확인)**; 비IID·오염 **−0.123 ~ 0.968**(label_flip·quantity_skew 0.968, label_skew 0.438–0.511, feature_noise cifar 0.693 / **mnist −0.123**) | 방향은 맞으나 상한·하한 모두 부정확. mnist feature_noise 음수 반례 존재 |
| R13 | dtype: track_d가 bf16이라는 검증 에이전트 추론 | **코드가 fp32 하드코딩**(`track_d.py:122` `dtype=torch.float32`, SFTConfig `bf16=False`) | config 파일에 dtype 필드가 없어 생긴 오독. **fp32가 맞음**(코드가 정본) |

### 0.2 확인된(일치) 핵심 항목
- **std20 분리 수치의 출처 특정**: `runs/track_d/fidelity.csv` 1B_std20 3-seed — Flirds 1.000 / loss-heur 1.000 / Flirds-1st ~0.999 / GTG 0.975 / FedSV 0.910±0.09 / ShapleyFL 0.194(β0.5; per-seed **0.305/−0.280/0.558**) / ComFedSV 0.093 / FedIF 0.157. dossier와 정합.
- 25셀 그리드 카테고리 카운트(4/12/2/3/4) 정확 일치. N=5 1B 런타임 8종(34.9/106.5/164.5/530–537s) 일치. free-rider φ exact-0(Flirds/(b)/Banzhaf/loss-heur = 0.0; **GTG 0.0041, FedSV 0.0053만 ≠0**) — `master_phi.csv` 원시 φ로 확인, 메커니즘은 코드로 확인(`gtg.py:18–28` within-subset renorm).
- git working tree clean(유일 untracked = 본 리뷰 디렉토리). 단 track_d rundir meta에 `git_dirty=true` 기록(실행 당시 더티 트리) — 프로비넌스 각주감.

---

# Pass 1 — 독립 비평 (부록 A 미열람 상태)

## A. 한 문단 요약

이 연구가 지금 증명한 것: FedAvg 학습 로그 위에 **고정 가중치 per-round coalition 게임**(§2.2, `in_run_sv.py:30–50`)을 정의하면, Flirds는 그 게임의 2차 Taylor surrogate에 대한 **exact client-level Shapley**를 라운드당 HVP 1회로 계산하며(교차항 1/2 분배, Σφ가 telescoping으로 total val-loss 변화와 일치, `flirds_estimator.py:119–131`), 1B–7B·N=5–100·3-seed 전역에서 exact (b) oracle의 순위와 값을 사실상 완벽 재현한다(std20 Spearman 0.999–1.000, Pearson ~1.000; device100 anchor 1.000). 동시에 이 게임이 benign 레짐에서 근사-가법적이어서 singleton loss-heuristic까지 동률이고(차별축은 비용뿐), 분리는 partial participation(재구성-MC 계열 0.09–0.19), poison(1차 부호실패 0.000→2차 복원 0.967, 단 3B에서 2차도 붕괴), 비가법 CNN(+0.087)에서만 발생함을 보였다. 아직 못한 것: 커뮤니티 표준 참값 **(a)-retrain과의 일치는 file-canon 기준 1B anchor5 한 칸(0.933)**이고 CNN에서는 (a)↔(b)가 발산하며(≤0.45), IID-clean에서 매칭 대상 (b) 자체의 seed-안정성이 LLM에서 미측정이고(CNN IID는 ≈0), LLM-scale 개입은 전부 parity라 downstream 실효성의 실증이 없으며, 이론(오차 bound·공리 정식화)과 몇몇 헤드라인 수치의 파일 정본화(R1·R2)가 비어 있다. **현재 위치: 최상위 바 기준 reject–major revision 경계.** 측정 프로토콜·정직성·스케일은 상위권이나, "self-referential ground truth" 공격(C-1)과 "매칭 대상이 노이즈일 가능성"(C-2), "so what at LLM scale"(C-3) 3연타를 현 서사로는 방어하지 못한다 — 그리고 셋 다 보강 경로가 구체적으로 존재한다.

## B. 강점 (근거 = §/파일, 분야 맥락 포함)

**B1. 이중 오라클 exact-2^N fidelity 프로토콜 — 이 조합의 선행이 실제로 없다 [진짜 기여].**
(근거: §3.1; audit 5회 독립 웹 스윕 "겨냥 셀 점유 선행 0건"; prior-work-taxonomy — exact-2^N fidelity 정면 측정은 GTG·SPACE뿐이며 둘 다 CNN.) data valuation 문헌의 고질병은 참값 없는 자기평가(downstream proxy·noisy-label detection으로 대체)다. 이 연구는 (b) in-run exact 2^N(전수 enumeration, `in_run_sv.py:131–153`)과 (a) retrain 2^N(1B anchor5 실측 **30,817s/셀** 지불)을 **둘 다** 계산해 estimator를 순위·값 수준에서 채점했다. F×LLM×client×exact-GT 칸을 처음 채운 것이 이 논문의 실체적 novelty다.

**B2. 추정기의 수학적 결이 깨끗하고, 그 성질이 코드로 검증돼 있다 [진짜 기여].**
(근거: `flirds_estimator.py:13–17,119–131` — 2차 surrogate 게임의 exact Shapley, 교차항 H-대칭 1/2 분배; 라운드당 HVP 1회(집계 ΔW 대상)+참여자 dot products; Σφ = telescoping total-Δval-loss, phase05 gate 0오차; free-rider **exact-0**은 고정 가중치 게임의 null-player 성질에서 구조적으로 나옴 — GTG 0.0041/FedSV 0.0053의 renorm 희석과 원시 φ 수준에서 대조됨.) FL Shapley 근사 계열(GTG/FedSV/ShapleyFL/ComFedSV)이 전부 다수의 모델 평가를 요구하는 것과 달리, 비용이 **cohort 크기와 무관한 O(1)**이라는 구조적 차별성이 실측으로 뒷받침된다(k=10에서 160×, R5).

**B3. 커버리지와 재현 인프라가 이 분야 평균을 크게 웃돈다.**
(근거: track_d 18 rundir = 3 스케일×2 레짐×3-seed, **7B 포함**; phase2 25셀; CNN 150셀; 전 셀 config+meta(git/env)+phi.parquet 영속 + `make_analysis.py` 재생성.) LLM-scale valuation에서 1B/3B/7B를 같은 프로토콜로 관통한 fidelity 표는 문헌에 없다.

**B4. 정직한 경계 보고의 밀도가 높다 — 리뷰 방어 자산.**
(근거: device100 off-anchor가 Flirds-proxy 순환임을 **분석 문서가 스스로 경고**(analysis §5.4); tiny-val caveat(silo5 100/device100 50/ASR val=4) 자기 기록; **(b) oracle 자체의 noisy 탐지 AUROC 0.604–0.660** spot-check로 φ-탐지 열세가 근사 결함이 아닌 valuation의 내재 한계임을 분리; LLM 개입 전 arm parity(±0.001–0.003)를 그대로 보고; bf16이 fidelity 1.000→~0.4로 붕괴함을 수치로 기록.) 이런 자기-감사는 리뷰어 신뢰를 사는 실질 자산이다.

**B5. "발견"들이 각각 독립적 기여 가치가 있다.**
- **근사-가법성**: benign FL-LLM 게임에서 exact Shapley 순위가 singleton utility(loss-heur)와 동률 → 이 스케일에서 coalition-sampling 계열의 비용 전제가 무너짐(§3.5; track_d std20).
- **재구성-MC 계열의 이중 붕괴**: partial participation(std20 0.09–0.19)과 near-tie IID(anchor5 FedSV 0.700·FedIF 0.067)에서 실패 — 실패의 per-seed 분산(ShapleyFL −0.28~0.56)까지 기록됨.
- **2차항의 조건부 가치**: benign에선 천장(무가치, 비용 3×)이나 poison에서 1차 부호실패를 복원(AUROC 0.000→0.917, Sp 0.000→0.967)하고, CNN 비가법에서 +0.087, momentum 하에서는 오히려 유해(0.73<0.81) — "2차항은 언제 필요한가"의 최초 실증 지도.
- **fp32 필연성**: utility ~1e-3 < bf16 정밀도 ~8e-3, 실측 1.000→0.4. "bf16으로 valuation하면 노이즈를 출판하게 된다"는 경고는 분야 전체에 유용.

**B6. 게임 설계의 원칙성.**
(근거: 고정 `p_k=n_k/Σ_{P_r}` 게임은 zero-update null-player를 정확히 만족(within-subset renorm 게임은 위반 — free-rider에 보상); 5-domain format 통일은 shared-val-loss 게임의 비교가능성 확보(§2.3); val-loss 게임 선택은 (a)-ROUGE 발산 실험으로 뒷받침 — 단 이 실험은 R2의 정본화 필요.) "무엇을 참값으로 삼을 것인가"를 명시적 설계 결정으로 다룬 점이 기존 FL-Shapley 논문들과 급이 다르다.

## C. 약점 (심각도 내림차순)

**C-1. [인정필수 + 완화가능] Ground truth가 자기 방법에 정렬된 게임이다 — 리젝급 공격 표면.**
근거: (b)의 U(S)는 **Flirds가 Taylor 전개하는 바로 그 게임**(고정 가중치, 동일 frozen trajectory, 동일 val set — `in_run_sv.py:30–50`, `track_d.py:369`). loss-heur·Fed-LOO·Banzhaf도 같은 게임의 산물이라 표 상단은 전부 "same-game family"다. 반면 GTG/FedSV는 within-subset renorm 게임, ShapleyFL/ComFedSV는 uniform 1/|S| 게임을 재구성한다(`gtg.py:18–28`, `shapleyfl.py:34–55`) — 즉 **헤드라인 fidelity 표는 "Shapley 추정 오차"와 "게임 정의 불일치"를 분리하지 못한다.** 그리고 방법-중립 참값인 (a)-retrain과의 일치는 file-canon 기준 **1B anchor5 0.933 한 칸**뿐이며, CNN에서는 (a)↔(b)가 발산한다(전 방법 ≤0.45, R8). 파급: Reviewer 2의 첫 문장 — *"You proved Flirds ≈ Flirds's own game. The one neutral target you have, you match at 0.93 in one cell and diverge at 0.35 in another."* 방어 재료는 실재한다(null-player 논증: renorm 게임은 free-rider에 양의 φ를 주므로 공리적으로 열등; telescoping 귀속 게임의 원리성; IRDS/PBRF의 run-specific value 계보; LLM (a) 0.93–1.0) — 그러나 **실험 보강 없이 서사만으로는 못 막는다**(G-1의 own-game 대조 + (a) 확대가 필수).

**C-2. [완화가능] 매칭 대상 (b) 자체의 안정성이 IID-clean에서 미검증 — "+1.000이 노이즈의 완벽 재현"일 가능성.**
근거: CNN에서 (b) oracle의 cross-seed 순위 자기상관은 IID에서 **−0.042/−0.172(≈0)**, 오염·비IID에서만 0.44–0.97(`track_c/RESULTS.txt`). **LLM은 (b) xseed를 아예 측정하지 않았다**(3-seed rundir가 있으므로 phi.parquet에서 **공짜로 계산 가능**한데도). 헤드라인 std20 +1.000이 "seed-특이 미세순서의 결정론적 재현"이라면, 그 수치가 증명하는 것은 추정기의 *분해능*(같은 게임을 fp32 정밀도로 재계산하는 능력)이지 *가치 측정*이 아니다. 파급: C-1과 결합하면 "self-referential + unstable target"으로 논문 전체가 흔들린다. 완화: (i) 기존 rundir에서 LLM (b) xseed 즉시 산출·보고, (ii) run-specific 가치가 유효한 use-case(해당 run에 대한 정산·보상)와 무효한 use-case(데이터 품질 평가)를 본문에서 명시 분리, (iii) fidelity 표에 "target 안정성" 열 병기.

**C-3. [인정필수] LLM-scale에서 downstream 실효성의 실증이 없다(2차 질문 ①②③ 전부).**
근거: 개입 arm은 전 스케일·전 arm **parity ±0.001–0.003**(MMLU·ROUGE-L; 7B std20 수렴 힌트 151–158 vs 184.7 라운드는 ±18 std 겹침); 탐지는 noisy에서 FedDQC(0.96–1.0)에 완패하고 φ는 oracle과 함께 0.6대(내재 한계); φ가 이기는 곳은 free-rider(1.0)뿐. CNN 오염 셀에서만 실이득(grad_noise acc 0.499→0.609, free-rider rounds 41.2→7.7)이 있으나 **거기서도 Flirds가 arm 중 최강이 아니다**(grad_noise acc는 shapleyfl_w 0.645가 우위). 파급: "정확한 valuation이 LLM에서 무엇을 사주는가"에 리뷰어가 만족할 답이 현재 없다. clean-IID parity는 설계상 기대(do-no-harm)였다는 항변은 가능하나, 그러면 "왜 그 무대에서 측정했나"로 되돌아온다. 완화 경로: 실효성의 무게중심을 **인센티브·정산(값-수준 fidelity가 직접 화폐화되는 유일한 용도)**으로 옮기고, 성능 개입은 "신호가 존재하는 레짐(오염·비IID)에서만 이득"이라는 조건부 법칙으로 격하 — 이는 CNN 결과와 정합.

**C-4. [완화가능] Baseline 비교의 공정성 논란 소지.**
근거: (i) ShapleyFL/ComFedSV/FedIF의 LLM 참패(0.09–0.19)는 상당 부분 **다른 게임 + min-max/EMA 후처리**의 산물(C-1과 동근원; `shapleyfl.py:58–63`); (ii) ShapleyFL 수치는 전부 **β0.5 원본인데 코드는 β0.3**(R11) — "논문값 Def4.3 β0.3으로 재실행하면 달라지나?"에 현재 답 못함; (iii) **Fed-LOO 수치 부재** — 감사 문서 스스로 "exact-Shapley 헤드라인 + 최유명 저가 근사 누락 = cherry-picking 1순위 지적"으로 규정(구현·합성검증 완료, 셀 수치만 대기); (iv) Ripple은 LLM 미실행인데 dossier 표에 LLM 런타임으로 등장(R6); (v) truncation-OFF 등 유리한 설정도 있으나(감사 문서·코드 확인) 문서화가 산재. 파급: "strawman baselines" 코멘트 하나로 표 전체의 신뢰가 깎인다. 완화: CNN 홈그라운드 성능 병기(FedSV/ShapleyFL이 CNN에선 기능함), own-game 대조(G-1), Fed-LOO·β0.3 재실행 완료.

**C-5. [인정필수 + 완화가능] 레짐 제약이 실무 FL-LLM과 어긋난다.**
근거: 전 valuation 런이 plain SGD momentum=0(momentum 0.9에서 2차항 열화 0.73<0.81을 실측했으므로 자의적이진 않음), **fp32 강제**(bf16 붕괴 실측), eager attention(SDPA/flash가 forward-mode AD 미구현), 상수 lr, **게임이 LoRA r16 부분공간에 한정**(base weight 기여 제외, `backends/llm.py:69`). 실무 레시피(AdamW·bf16·flash-attn)와의 갭은 Track D deviation caveat 3종으로 자인돼 있다. 파급: external validity 공격 — "이건 실험실 FL이다." 주목할 점: **(b) oracle은 델타만 받으므로 optimizer-불문** — 클라가 AdamW로 학습한 로그에 대해 estimator fidelity를 재는 실험은 재학습 오라클 없이 가능한데 **어느 계획에도 없다**(G-4에서 제안). fp32도 "학습은 bf16, 서버 valuation 패스만 fp32 사본"으로 분리 가능한지 오버헤드 정량화가 없다.

**C-6. [수정가능] 이론이 정식화되어 있지 않다.**
근거: 위키에 Prop 1·granularity-invariance lemma 참조만 있고, dossier·분석 문서에 (i) surrogate 게임 Shapley의 공리 성질(efficiency/symmetry/null/linearity — 코드·수치로는 검증됨), (ii) |φ̂−φ_(b)| ≤ f(‖Δ‖³, Hessian Lipschitz) 류 remainder bound, (iii) (a)-게임과 (b)-게임의 관계(언제 일치하는가)의 형식 서술이 없다. 파급: 최상위 바에서 "engineering + evaluation" 딱지. 수정 가능: (i)(ii)는 대수적으로 도출 가능하고 remainder는 셀별 additive-gap 실측(Σφ̂ vs U(N) 잔차)으로 검증 가능 — 실험 없이 집필로 해결되는 유일한 대형 약점.

**C-7. [수정가능] 통계·증거 위생.**
근거: 3-seed(실패 방법의 per-seed 진폭 −0.28~0.56인데 ±std만 보고, CI·검정 없음); tiny val(silo5 100·device100 50·**poison ASR val=4**); 3B robustness 1-seed; N=5 Spearman은 5점 순위(우연 +1.0 확률 1/120/seed); task6 rundir 부재(R2)·3B (a) 오귀속(R1)·ComFedSV CNN 구수치(R9) 등 **노트-수치가 dossier에 섞여 있음**; 학습 로그(델타) 미영속이라 Fed-LOO 백필 불가·오라클 재분석 불가; track_d meta `git_dirty=true`. 파급: 개별로는 사소하나 합치면 "부주의한 저자" 인상 — 카메라레디 전 전수 정리 필요.

**C-8. [인정필수] poison에서 추정기 실패가 스케일과 함께 악화된다.**
근거: clean-preserving backdoor(ASR 1.0)에서 Flirds-1st Sp 0.000/Pearson −0.95(부호 실패), 2차 복원은 1B(0.967/AUROC 0.917)뿐이고 **3B에서는 2차도 붕괴**(Sp 0.000, Pearson −0.893; 1-seed). γ-scaled 대형 delta가 Taylor trust-region을 이탈하는 구조적 현상(`phase2_matrix.py:241–252`, gamma=K). 주의: dossier의 프레이밍("clean val-loss를 낮추는 공격자에 φ '기여 높음'은 valuation의 정직한 답")은 **자체 분석 문서와 모순** — (b) oracle은 공격자에게 해로움(high-φ)을 주고 AUROC 1.0으로 잡는다. 즉 이것은 게임의 정직한 답이 아니라 **추정기의 fidelity 실패**다(extremeness 축 AUROC로는 3B에서도 1.0이 잡히는 이중성까지 포함해 정밀 서술 필요). 파급: 숨기면 치명, 정면 배치하면 "2차항의 존재 이유 + 방법의 경계"라는 최고급 분석 절이 된다.

## D. Novelty 판정

**정면 판정 — "IRDS의 FL 확장"은 incremental인가:** 추정기 수학만 놓고 보면 **incremental이 맞다.** 2차 다항 게임의 Shapley가 교차항 1/2 분배로 닫히는 것은 고전적 사실이고, per-step sample-level(IRDS)을 per-round client-level로 옮기는 대수는 자명에 가깝다. "federated × in-run"의 첫 점유도 Ripple(AAAI'26)에 넘어갔음을 저자 스스로 기록했다(flirds.md L252). 리뷰어 중 한 명은 반드시 이렇게 쓴다.

**그러나 이 논문의 novelty 실체는 추정기가 아니라 측정학이다:**
1. **F×LLM×client-level valuation을 exact-2^N oracle로 채점한 최초** — FL-Shapley족(GTG/FedSV/ShapleyFL/ComFedSV/Ripple)은 전부 CNN, LLM-attribution족(IRDS/DataInf/LESS/LoGra)은 전부 centralized라는 두 모집단 분리가 taxonomy로 정리돼 있고, 5회 웹 스윕에서 칸 점유자 0(모두 audit 문서 근거).
2. **(a)+(b) 이중 GT 설계** — retrain 참값과 in-run 참값을 같은 무대에서 대조한 FL 선행 없음(exact-2^N fidelity 자체가 GTG·SPACE 둘, 모두 CNN·단일 GT).
3. **경험 법칙들**(B5) — 특히 "benign FL-LLM 게임은 근사-가법 → coalition-sampling의 비용 전제 붕괴"와 "2차항의 가치는 poison·비가법·plain-SGD 조건부"는 후속 연구가 인용할 수밖에 없는 종류의 결과다.
4. 잔여 델타의 방어선(audit·taxonomy 문서 기준): Ripple(sample-level·CNN·2차 없음·fidelity 미측정), FedTSV(집계-조향·closed-form 없음·fidelity 0), LESS(centralized·Adam-cosine), KFCA/WinFLoRA/FedAttr(목적 상이·fidelity 없음). 대비 문장은 이미 준비돼 있다.

**최상위 메인트랙에 충분한가:** **조건부 yes.** "새 알고리즘" 논문으로 팔면 부족하고(추정기 delta가 얇음), "측정 표준 + O(1) 추정기 + 경계의 지도" 패키지로 팔면 novelty 요건은 채운다. 단 그 패키지는 C-1·C-2가 봉합돼야 성립한다 — 순환성 공격이 관통하면 측정학 기여 자체가 무너지기 때문이다. novelty의 운명이 신규 아이디어가 아니라 **검증 보강 실험**에 걸려 있는, 드문 구조다.

## E. 필수요소·분야요건 스코어카드

| 차원 | 판정 | 한 줄 근거 |
|---|---|---|
| 문제·gap 명확성 | **충족** | LLM-scale FL valuation의 비용 + GT-부재 gap이 taxonomy로 입증됨(§2.1, audit) |
| Novelty | **부분** | 추정기 단독 incremental; 측정 표준+발견 패키지는 충분 — 단 C-1 봉합 조건부(D) |
| 기술적 엄밀성 | **부분** | fp32/eager/true-Hessian/exact-2^N은 상위권; 이론 정식화·통계 처리 미비(C-6·C-7) |
| Significance | **부분** | O(1) 비용 구조·측정 표준은 실질적; LLM downstream 실증 부재가 상한을 깎음(C-3) |
| 정직성·한계 | **충족(레포)/주의(dossier)** | 순환성·tiny-val·parity 자기 보고(B4); 단 dossier 수준 요약은 과대·오귀속 포함(R1–R4) |
| 재현성 | **부분** | rundir 영속+재생성 도구 vs 델타 로그 미영속·노트-only 수치·β 프로비넌스(R11)·코드 공개 미정 |
| GT 검증 신뢰성(분야) | **부분** | (b)-체인은 완비; (a)는 1셀 0.933 + CNN 발산(R8) + 3B/7B/N=10 부재 |
| Scalability(분야) | **충족** | 7B·N=100 실측 fidelity 1.000/0.999 + cohort-독립 O(1) 실측(160×@k=10) |
| 이론적 근거(분야) | **미흡** | 공리·remainder bound·(a)-(b) 관계 미정식화 — 유일하게 집필만으로 수정 가능(C-6) |
| Robustness(분야) | **부분** | 비IID sweep·threat 매트릭스 커버; poison에서 실패 실재하나 특성화됨(C-8) |
| Actionability(분야) | **부분** | CNN 오염 셀 실이득·free-rider 1.0; LLM 개입 null — 인센티브 use-case로 재포지셔닝 필요 |
| Baseline 포괄성(분야) | **부분~충족** | 9–10 valuation+4 detector+선정 감사; own-game 대조 부재·Fed-LOO 수치 대기(C-4) |

## F. 논문 구조 설계

기본 골격(Abstract→Intro→Related→Method→Setup→Results→Analysis/Limitations→Conclusion)에서 **두 가지를 조정**한다. (조정 1) Method와 Experimental Setup 사이에 **"측정 프로토콜(이중 오라클)" 독립 절**을 둔다 — 이 논문의 novelty 중심이 추정기가 아니라 측정학이므로, 프로토콜이 method의 하위 절로 묻히면 기여가 안 보인다. (조정 2) Results를 관례적 "우리 방법이 이겼다" 순서가 아니라 **핵심 질문 위계(1차 fidelity → 2차 ①성능 ②수렴 ③탐지) + 경계(어디서 깨지는가)** 순으로 배치한다 — 이 연구의 가장 강한 자산이 경계 보고이기 때문이다.

**§1 Introduction** — gap 2개(비용, GT-부재)와 기여 4-bullet: (i) 고정가중 per-round 게임 + 그 2차 surrogate의 exact Shapley를 HVP 1회/라운드로 닫는 Flirds(+null-player·efficiency 성질), (ii) (a)+(b) 이중 오라클 fidelity 프로토콜과 1B–7B·N=5–100 최초 채점, (iii) 발견: 근사-가법성→비용만이 차별축·재구성-계열 붕괴·2차항의 조건부 가치·fp32 필연, (iv) 경계: poison trust-region·탐지의 내재 한계·측정가능성 조건. **Fig 1** = (좌) cost-vs-fidelity frontier(track_d+silo5 실측), (우) 프로토콜 다이어그램((a)/(b)/estimator 삼각형).

**§2 Related Work** — 3축 구성: 중앙집중 valuation(Data Shapley→Beta/Banzhaf→IRDS; influence: TracIn/DataInf/LoGra/LESS), FL 기여도(GTG/FedSV/ShapleyFL/ComFedSV/Ripple/FedTSV — 공통적으로 O(N+) 평가 + CNN 무대), 탐지(FLDetector/FLTrust/FedDQC — 위계상 downstream). **Table(비교표)**: 각 방법 × {level, in-run?, 2nd-order?, LLM?, fidelity 검증?, GT 종류} — taxonomy 문서에서 직행 가능. Ripple과 FedTSV 대비 델타를 본문 2문장으로 명시(D-4).

**§3 The Valuation Game and the Flirds Estimator** — 3.1 게임 정의(고정 p_k, telescoping, **null-player 논증으로 renorm 대안 기각을 정당화** — C-1 선제 방어), val-dist가 가치를 정의한다는 명시(construct 투명성); 3.2 Taylor surrogate와 closed-form Shapley(**Prop 1**: surrogate 게임에서 efficiency/symmetry/null 만족; **Prop 2**: remainder bound + 셀별 실측 잔차로 검증); 3.3 알고리즘·복잡도 표(방법별 evals/round vs HVP 1회); 3.4 LLM 구현 필연(fp32 1.000→0.4 실측, eager-attn forward-AD, hook 위생 — 부록과 분담). [있는 것: 전부 구현·수치 검증 / 필요한 것: Prop 1·2 집필, 잔차 표 추출]

**§4 Dual-Oracle Fidelity Protocol** — (a)/(b) 정의와 비용, **왜 val-loss 게임인가**((a)-ROUGE 발산 실험 — R2 정본화 필수), 지표 위계(Spearman·Kendall/Pearson/거리), **target 안정성 절차**(xseed ρ 병기 — C-2 방어를 프로토콜의 일부로 격상), 레짐 매트릭스 표(silo5/std20/anchor5/device100×스케일). [필요: task6 재실행-커밋, LLM (b) xseed 계산(기존 rundir로 무비용)]

**§5 Results** (위계 순):
- **5.1 1차 — Fidelity**: **Table 1**(track_d std20+anchor5, 1B/3B/7B×3-seed, Sp/Kendall/Pearson vs (b) + (a)행@anchor5 + **(b) xseed 열**) — 헤드라인. **Fig 2** device100 anchor(α=0.5, 진짜 oracle 칸만 fidelity로 제시 — off-anchor는 순환이므로 본문에서 배제 명시). silo5 corruption 표는 5.4로.
- **5.2 2차① 성능 + ② 수렴**: **Table 2** 개입 arm parity(LLM; do-no-harm으로 프레이밍) + CNN 오염 셀 이득(grad_noise 0.499→0.609 등, ShapleyFL-w 우위 포함 정직 병기); **Fig 3** val-loss 곡선·rounds-to-target(7B 힌트는 std 겹침 명기).
- **5.3 2차③ 탐지**: **Table 3** threat-matched 비교 + **(b)-oracle 행 포함**(noisy 0.60 내재 한계 — "φ-탐지의 상한은 게임이 결정") + free-rider 1.0 + FLDetector 최저가-최약.
- **5.4 경계(이 논문의 차별 절)**: poison 사례 연구(**Fig 4**: φ 분포 — (b) high-φ vs 1st 부호실패 vs 2차 복원 vs 3B 붕괴; Pearson −0.95; 부호규약/extremeness 이중 축 명시), 근사-가법성과 재구성-계열 붕괴(std20·anchor5-IID per-seed 산포), 2차항 조건 지도(poison/CNN/momentum), bf16 붕괴.
- **5.5 비용**: **Fig 5** cohort k∈{2,5,10} crossover(O(1) vs 2^k; std20에선 exact가 더 쌈을 **정직하게 표기** — R5) + frontier.
[있는 것: 5.1–5.5 전 수치 존재(단 R1·R2 정본화, Fed-LOO·β0.3 재실행, xseed 계산 필요) / 필요한 것: own-game 대조 실험(G-1), (a) 확대(P3), 3B poison 3-seed(P1), val≥200 재실행(P2)]

**§6 Analysis & Limitations** — 근사-가법성의 해석(왜 1차로 충분한가; rank/참여 probe 결과 연결), run-specific value의 적용 범위(정산 O / 품질평가 △ — C-2), optimizer·정밀도 레짐(bridge arm/Adam-fidelity), SecAgg 비호환(위키 자인), LoRA-부분공간 게임, (a) 커버리지 한계.

**§7 Conclusion.** **Appendix**: 증명, 25셀+CNN 전 표, baseline 충실도·선정 감사(audit 문서 요약), 구현 3-musts, 재현성 카드(rundir·시드·make_analysis).

**Claim → 증거 매핑**:

| Claim | 그림/표 | 상태 |
|---|---|---|
| 추정기가 exact (b)를 순위·값 재현 (1차) | Table 1, Fig 2 | **있음**(track_d/fidelity.csv, device100 anchor) + xseed 열 필요 |
| 비용 O(1) vs 2^k, k≥5부터 5–160× | Fig 5 | **있음**(runtime 표 3종) — cohort-조건화 서술만 |
| 재구성-계열은 partial participation·near-tie에서 붕괴 | Table 1 | **있음** + own-game 대조 **필요** |
| 2차항은 poison·비가법·plain-SGD에서 가치 | Fig 4, 5.4 | **있음**(1B) + 3B poison 3-seed **필요**(P1) |
| (a)≈(b) — in-run 게임의 유효성 | §4, Table 1 (a)행 | **부분**(1B 0.933) — task6 정본화·P3·CNN 발산 해명(P5) **필요** |
| do-no-harm + 오염 레짐 이득 (2차①②) | Table 2, Fig 3 | **있음** |
| 탐지는 위계 최하 + 내재 한계 (2차③) | Table 3 | **있음**((b)-행 포함) |
| fp32/eager 필연 | 5.4, Appx | **있음**(bf16 0.4) + fp32 오버헤드 정량 권장 |

## G. 예상 리뷰어 공격 + 반박 (강한 순)

**G-1. "Ground truth가 self-referential이다 — (b)는 당신 방법의 게임이고, 경쟁자들은 다른 게임을 푼다."**
→ 현재 가능한 반박: (b)는 "Flirds의 게임"이 아니라 **FedAvg 궤적의 유일하게 공리-정합인 사후 귀속 게임**(고정 가중치만이 zero-update null-player를 만족 — renorm 게임은 free-rider에 φ 0.004–0.005를 지급, 원시 φ로 실증; grand coalition이 실제 배포 궤적으로 telescoping). (a)-retrain이라는 방법-중립 참값에서 LLM 0.93–1.0.
→ 부족분: **(i) own-game 대조 실험** — GTG/FedSV를 *그들의 renorm 게임의 exact Shapley*(N=5면 2^5 enumeration으로 저가) 대비로도 채점해 "게임 불일치 vs 추정 오차"를 분해; ShapleyFL/ComFedSV의 uniform 게임도 동일. (ii) (a) 확대(3B/7B anchor5 = P3; task6 재실행-커밋 = R2). (iii) CNN (a)↔(b) 발산의 원인 규명(P5) — 이것 없이 "(b)면 충분"을 일반화하면 자충수.

**G-2. "std20 +1.000은 노이즈의 완벽 재현일 수 있다 — IID-clean에서 (b)의 순위가 seed마다 다르지 않은가?"**
→ 현재: CNN에서 오염·비IID면 target이 안정(0.44–0.97), IID면 ≈0임을 이미 보고; Flirds는 target의 내재 안정성을 그대로 추종(0.547 vs (b) 0.518)하고 MC-계열은 그보다 아래(0.12–0.31)로 떨어짐 — 즉 "추정기가 노이즈를 *추가*하지 않는다"는 방어는 가능.
→ 부족분: **LLM (b) xseed 산출**(기존 3-seed phi.parquet로 무비용) + run-specific 가치의 use-case 조건화(정산은 realized run이 곧 대상이므로 유효; 데이터 품질 일반화는 xseed가 낮은 레짐에서 무효). 이 실험 없이 std20을 헤드라인으로 걸면 위험.

**G-3. "loss-heuristic이 전 스케일에서 순위·값 모두 1.000인데(Table 1) 왜 HVP가 필요한가?"**
→ 현재: (i) loss-heur는 같은 게임의 singleton — 근사-가법 레짐의 *발견*이지 방법의 패배가 아님; (ii) 분리 레짐 실재 — poison(loss-heur 1.0이지만 Flirds-1st 0.0 → 2차 0.967로, singleton과 Shapley가 갈리는 곳은 상호작용 레짐), CNN 비가법(+0.087); (iii) 비용 — loss-heur는 O(k) forward/round(std20 2913s), Flirds-1st는 O(1)(1531s), k가 크면 역전 심화; (iv) 공리 — singleton은 efficiency 미만족(Σφ≠ΔU).
→ 부족분: "언제 singleton이 틀리는가"를 상호작용 크기(additive gap)와 연결한 정량 절 — probe(rank·참여)가 이걸 겨냥하므로 결과 연결 필수.

**G-4. "plain SGD·fp32·eager·상수 lr·LoRA-r16 — 실무 FL-LLM이 아니다."**
→ 현재: momentum 열화는 실측(0.73<0.81)이라 자의적 제약이 아니고, fp32 필연도 실측(1.000→0.4); LoRA는 FL-LLM의 실무 표준; deviation caveat 자인 + bridge arm 계획.
→ 부족분: **client-AdamW 로그에 대한 estimator-vs-(b) fidelity** — (b)가 optimizer-불문이므로 재학습 오라클 없이 측정 가능한데 어느 계획에도 없음. 성립하면 최대 external-validity 구멍이 저비용으로 닫히고, 실패해도 정직한 경계. 강력 추천. fp32 오버헤드(서버 사본 메모리·시간)도 수치화.

**G-5. "ShapleyFL 0.19, ComFedSV 0.09, FedIF 0.16 — strawman 아닌가?"**
→ 현재: CNN 홈그라운드에서 동일 구현이 기능함(ShapleyFL poison Sp 1.0, (a)-oracle 대비 최고 0.453; ComFedSV는 low-rank 가정이 N=10 full에서 자명 충족); 원인 규명 서술 존재(부분참여 가정 위배·uniform 게임·EMA); truncation-OFF 등 유리 설정; 선정 감사 문서.
→ 부족분: β0.3(원논문값) 재실행 완료(R11), Fed-LOO 수치(C-4), own-game 대조(G-1). 이 셋이 있으면 방어 완결.

**G-6. "downstream 이득이 없다 — parity 표가 방법 무용론 아닌가?"**
→ 현재: clean-IID에서 개입 이득이 없는 것은 **신호 부재의 정직한 결과**이고 CNN 오염 셀에서는 이득 실재(0.499→0.609; rounds 41.2→7.7) — "이득은 신호가 있는 곳에서만"이라는 조건 법칙으로 두 트랙이 정합. 위계상 1차 질문은 측정 정확도이고, 값-수준 정확도의 직접 수요처는 인센티브·정산.
→ 부족분: 인센티브 유스케이스의 최소 실증(예: φ-비례 보상의 오배분율을 (a)/(b) 기준으로 방법별 비교 — 기존 φ로 오프라인 계산 가능, 저비용) 없이는 "so what"이 서사에 그침.

**G-7. "backdoor에 속는 valuation을 어디에 쓰나 + 저자 프레이밍이 자기모순."**
→ 현재: (b) oracle은 공격자를 잡는다(AUROC 1.0) — 문제는 게임이 아니라 대형 delta에서의 Taylor 실패(C-8)이고, 2차항이 1B에서 복원(0.917), extremeness 축은 3B에서도 1.0. threat-matched detector와의 비교표 존재.
→ 부족분: 3B poison 3-seed(현재 1-seed, P1) + 실전 γ 설정 재확인(P4) + dossier식 "정직한 답" 프레이밍 폐기(파일 정본과 모순 — §0 R4·C-8). trust-region 진단(‖Δ‖ vs 잔차)을 Fig 4에 넣으면 공격이 분석 기여로 반전됨.

## H. §4 계획 실험 완료 후에도 남는 갭

§4 목록 중 Track D 본런은 **이미 완료**(R10), probe는 가동 중, 2×2(오염×비IID)와 N=10·7B-(a)는 미착수라는 실상 기준으로:

1. **(a)-retrain 커버리지** — 2×2·probe 완료 후에도 (a)는 1B anchor5(+계획 P3의 3B/7B anchor5) 수준. N=10 2^10 retrain(P7)과 CNN (a)↔(b) 발산 해명(P5) 없이는 "in-run 게임 = 데이터 가치"의 일반화가 열려 있음.
2. **own-game 분해 실험**(G-1) — 어느 계획에도 없음. C-1 방어의 결정타인데 미계획.
3. **client-AdamW fidelity**(G-4) — bridge arm은 *무대* 갭(성능 궤적)만 재고, *추정기* 갭은 안 잼. 미계획.
4. **LLM (b) xseed / run-specific 가치의 조건화**(C-2) — 계산은 무비용이나 서사 재설계가 필요. probe(신호 크기)와 상보이지 대체가 아님.
5. **이론 정식화**(C-6) — 실험 계획이 아니라 집필 갭.
6. **인센티브 실증**(G-6) — φ-비례 정산 시뮬레이션조차 없음.
7. **통계 보강** — 3-seed→5+(최소 헤드라인 표), CI/검정, tiny-val 민감도(P2)와 3B robustness 3-seed(P1)는 계획에 있으나 seed 확대는 없음.
8. **공개·정본화** — 코드/데이터 공개 계획 미정, 델타 로그 미영속(오라클 재분석 불가), R1·R2·R9류 노트-수치 정리, β0.3 재실행(163셀 큐 대기).
9. **범위 밖 축** — DP-noise·SecAgg 하 valuation(자인된 비호환), 샘플-수준 위계(client→sample), 선호학습(RLHF) 라운드 — 후속 논문감이므로 한계 절에 명시로 충분.

## I. 포지셔닝·서사 추천

**주 서사(권장): "측정이 먼저다" 프레임.**
제목 방향: *“Who Contributed What? Oracle-Grade, In-Run Client Valuation for Federated LLM Fine-Tuning”*. 헤드라인 = **이중 오라클로 채점된 fidelity**(Table 1: std20·anchor5·device100, 순위+값) + **cohort-독립 O(1) 비용**(crossover 정직 표기) — 핵심 질문 위계 1차와 정확히 정합. 2차는 위계 순서대로 parity(do-no-harm)→수렴 힌트→탐지(내재 한계 spot-check 포함)로 담담하게. **경계 절(5.4)을 세일즈 포인트로**: poison trust-region, 근사-가법성, 재구성-계열 붕괴, fp32 — "우리는 방법이 어디서 깨지는지 참값으로 안다"는 것 자체가 이 분야에서 아무도 못 준 것. 정직한 한계 배치: (a) 커버리지, run-specific 가치의 적용 범위(C-2), 레짐 제약(C-5), LLM 개입 null(C-3). 이 서사의 강점: B1–B5를 전부 활용하고 C-1·C-2를 프로토콜의 일부(target 안정성 보고)로 흡수해 공격면을 줄인다.

**대안 서사: 방법-우선 "Flirds: free-lunch client Shapley for FL-LLM".**
헤드라인 = 5–160× 비용 우위 + fidelity 1.000. 관례적이라 안전해 보이나 **비추천**: (i) 추정기 delta가 얇아 novelty 공격에 정면 노출(D), (ii) loss-heur 동률(G-3)과 std20 비용 역전(R5)이 "free lunch" 서사를 내부에서 반박, (iii) 경계·정직성 자산이 한계 절로 밀려 사장됨. 방법-우선으로 갈 거면 최소한 2차항 조건 지도(poison 복원)를 method 기여로 승격해야 하는데, 그 근거(3B 1-seed)가 아직 얇다.

## J. Threats to validity + related-work 지도

**Internal validity**
- GT-게임 정렬 confound(C-1) + estimator·(b)·(a)-valloss의 **val set 공유**(상관 노이즈; 같은 게임 정의상 정상이나 "일치의 놀라움"을 부풀림) — own-game 대조·(a) 확대로 완화.
- (a) retrain이 전 coalition에 **동일 seed** 재사용(`exact_sv_llm.py:79`) — counterfactual 노이즈 축소 목적이나 seed-특이 아티팩트 가능.
- 클라 간 **데이터 순서 seed 공유**(SFTConfig(seed=seed) 전 클라 동일), free-rider stream만 seed+1.
- tiny val(50–200; ASR val=4)·3-seed·3B robustness 1-seed·N=5 순위 5점(C-7).
- β0.5/0.3 프로비넌스(R11), 노트-only 수치(R1·R2·R9), git_dirty 실행분.
- detection AUROC 부호 규약이 러너에 산재(negate 관례) — plan.md에 기록된 D2b↔matrix 모순의 근원. 논문에서는 축(부호-인지 vs extremeness)을 명시 분리해야 함.

**External validity** — plain SGD mom=0·상수 lr·fp32·eager·LoRA r16 고정(C-5); 게임이 LoRA 부분공간 한정; SFT·영어·instruction 포맷 한정; 5-domain 파티션은 semi-synthetic(실제 기관 데이터 아님); N≤100, R≤200; alpaca-IID 무대는 near-tie 특수 레짐; SecAgg 하 배포 불가(per-client delta 필요 — 전 client-valuation 공통 한계, 자인됨).

**Construct validity** — "가치"의 조작적 정의가 3중임: (a) retrain 게임(counterfactual 참여) vs (b) in-run 게임(realized 궤적의 귀속) vs 실무 직관(데이터 품질). (a)↔(b)는 LLM 0.93–1.0/CNN 0.35로 무대 의존(R8); (b)는 val 분포 선택에 의해 정의됨(val-dist 거버넌스 — 도메인 균등 val은 하나의 정책 선택); ROUGE-게임 배제 논거는 정본화 전(R2); run-specific 가치의 seed-안정성 조건(C-2). 논문은 "우리는 (b)를 측정한다, (b)가 (a)·품질과 일치하는 조건은 이러하다"로 계층화해야 함.

**Related-work 지도(좌표)** — *계승*: IRDS(in-run·Taylor·run-specific 가치 철학; 母방법), Ghorbani&Zou(retrain 오라클 정의 = (a)), GTG(exact-2^N fidelity 평가 관행, CNN), Data-Banzhaf(학습 확률성 하 안정성 문제의식 — C-2와 직결, semivalue 앵커로 채택됨). *갈라짐*: client-level per-round coalition 게임과 고정가중 null-player 설계, 1-HVP collapse, 이중 GT, LLM-scale, 경계 특성화. *경계 인접*: Ripple(federated×in-run 선점, sample·CNN·no-2nd·no-fidelity — 표에서 CNN 전용으로 정직 비교), FedTSV(client-level in-run이나 조향 목적·no closed-form), LESS/DataInf/LoGra(centralized LoRA-scale attribution), FedIF(1차 사촌 — Flirds-1st와 비용 동급이나 fidelity 0.16 vs 0.999가 정규화·게임 차이의 생생한 대조), FedDQC/FLTrust/FLDetector(탐지 위계의 상대역 — φ의 상한이 게임에 의해 결정됨을 (b)-행으로 보이는 것이 이 논문식 응답).

---

# Pass 2 — 부록 A(저자 자기진단) 대조·확장

수령: 2026-07-02, 신호 크기 진단 요지. 아래 "A-n"은 부록 A의 n번 논지. **선행 주의**: 부록 A의 수치(가산 갭 ≤0.9%, LLM (b) xseed −0.37~−0.11 / silo5 +0.93~1.00, paired SNR 2.4–4.5, MMLU 효과 ~0.001 < SE ~0.004)는 본 리뷰의 정본 원칙(§0) 기준 file-canon 소재 미확인 상태다 — R2와 같은 정본화(산출 CSV/rundir 커밋)를 전제로 대조한다.

## K. 저자 자기진단과의 대조

### K-1. 겹치는 것 (독립 수렴 — 두 시각이 같은 결론에 도달, 신뢰도 상승)

| 논점 | 부록 A | Pass 1 |
|---|---|---|
| IID-clean에 클라 간 진짜 신호 없음 | A-3 (구조적 부재; 교환 가능 설계) | **C-2** (매칭 대상 안정성; CNN RESULTS.txt IID −0.04/−0.17에서 독립 도출) |
| near-additive → 전 semivalue 동률 | A-4 (갭 ≤0.9%; 무대 특성) | **B5·G-3** (loss-heur 동률의 원인; 분리는 부분참여·오염에서) |
| intervention 정답 자체가 ~0 | A-5 (do-no-harm parity) | **C-3** (LLM 개입 전 arm parity ±0.001–0.003) |
| headline 레짐의 자기모순적 긴장 | A 말미 ("+1.000·5–15×가 무신호 무대 산물") | **A(요약)·C-1/C-2·I** (IID 수치를 분해능 검증으로 강등, B축 승격 권고) |
| fp32는 병목 아님 | A-1 | Pass 1은 fp32를 병목으로 본 적 없음 — 단 "병목 아님"이 "정밀도 무관"으로 읽히면 안 됨(bf16 1.000→0.4는 여전히 필연성 발견, B5) |

특기: 내가 C-2/G-2에서 "기존 3-seed rundir로 무비용 계산 가능"이라 권고한 **LLM (b) xseed를 저자가 이미 측정**했다(−0.37~−0.11 vs silo5 +0.93~1.00). 권고가 선제 이행된 셈이고, 결과는 내 우려의 최대치를 확인한다: std20/anchor5 헤드라인 +1.000의 매칭 대상은 seed 간 재현되지 않는 순위다.

### K-2. 저자가 잡았고 내가 놓친(또는 못 잰) 것

1. **LLM (b) xseed 실측치 자체** — 나는 "미측정 갭"으로만 지적, 저자는 수치 보유. 음수 구간(−0.37)은 단순 무신호를 넘는 구조적 아티팩트 가능성까지 시사(K-5 심화 4).
2. **MMLU 축의 원리적 불감성** — 나는 parity를 "관측된 null"로 처리했으나(C-3), 저자 진단은 더 깊다: 이 SFT는 포맷/스타일만 배우므로 효과(~0.001)가 표본 SE(~0.004) 아래 — **endpoint 자체가 검출 불가능**. 파급: parity 표를 do-no-harm 증거로 쓰려면 power≈0을 명기해야 하고, 실효성 주장용으로는 endpoint 교체(도메인-정합 지표 또는 오염 무대)가 필요하다.
3. **paired val-loss의 방향성 실재**(SNR 2.4–4.5, 크기 0.07–0.3%) — 내 "전부 parity" 서술을 정정하는 미세 신호. 단 크기가 실용 무의미 수준이라 C-3의 결론(LLM 실효성 실증 부재)은 유지된다.
4. **가산 갭의 정량(≤0.9%)과 A/B축 분해** — 신호원이 학습 강도(A축)가 아니라 클라 간 실제 차이(B축)라는 통제 대조(특히 quantity_skew: 품질 동일·양만 차이 → xseed 0.97), 그리고 이를 여는 2×2 설계.

### K-3. 내가 잡았고 부록 A에 없는 것

1. **C-1 GT 순환성(자기정렬 게임)** — 진단은 "신호가 존재하는가"만 다루고 "**누구의 게임을 참값으로 삼았는가**"를 다루지 않는다. 이것이 더 치명적인 이유: 저자의 처방(B축 피벗)을 그대로 따라가도 오염·비IID 무대의 GT는 여전히 (b) = Flirds-정렬 게임이므로, 순환성 공격은 피벗을 관통해 따라온다. own-game 대조와 (a) 앵커 없는 2×2는 같은 결함을 상속한 표를 하나 더 만들 뿐이다.
2. **(a)↔(b) 발산과 (a) 커버리지** — CNN에서 전 방법 vs (a) ≤0.45(R8), LLM (a)는 1B anchor5 0.933 한 칸. "신호가 실재하는 무대"에서도 in-run 게임이 재학습 가치와 일치하는가는 별개 질문인데 진단이 침묵한다.
3. **증거 위생·프로비넌스**(R1 3B-(a) 오귀속, R2 task6 노트-only, R6 Ripple, R9 ComFedSV, R11 β0.5/0.3) — 진단 스코프 밖이지만 논문 생사 문제.
4. **poison trust-region 실패**(1차 부호실패 Sp 0.000/Pearson −0.95, 3B에선 2차도 붕괴)와 dossier 프레이밍의 자기모순(C-8) — 진단이 전혀 안 다루는 축이며, B축 피벗과 정면 충돌한다(K-5 심화 1).
5. 이론 정식화(C-6), Adam-fidelity 저비용 실험(G-4), 인센티브 실증(G-6), Fed-LOO 수치(C-4).

### K-4. 심각도 판단이 갈리는 것

1. **A-4의 톤 "무대 특성이지 계산 결함 아님"** — 사실이지만 면책이 아니다. 계산이 무결해도 **헤드라인 증거가 무정보**라는 결론은 동일하므로, 처방은 "각주로 설명"이 아니라 **결과 표의 재구성**(IID 수치 = 분해능·정밀도 검증으로 강등, B축 = 헤드라인로 승격)이어야 한다. 저자는 "설명됐다"에서 멈추고, 나는 "증거 기반을 갈아야 한다"까지 간다.
2. **std20 분리의 귀속** — A-4는 "부분참여 하의 추정 전략 차이"로 규정하나, C-1 기준 그 분리의 일부는 게임 불일치(renorm/uniform)다. own-game 대조 없이는 저자의 남은 구별력 주장("std20에서 방법이 갈린다")도 완결되지 않는다.
3. **(a)/(b) 표기 정밀도** — 부록 A 요약문은 "그 레짐은 **(a)** oracle 자신의 순위가 seed-불안정"이라 쓰지만, 실측된 것은 **(b)** xseed다(본문 A-3도 (b)로 명기 — 요약문과 불일치). LLM (a) xseed는 미측정(존재하는 것은 1B anchor5의 (a)-vs-(b) per-seed 0.900/1.000/0.900). 논문에서 이 구분이 흐려지면 리뷰어가 잡는다.

### K-5. 진단을 넘어서는 통찰

**더 깊은 문제 (부록 A가 아직 명시하지 않은 것)**

1. **B축 피벗 × trust-region 충돌**: 신호가 실재한다고 확인된 오염 무대는 동시에 γ-scaled 대형 delta로 2차 Taylor가 깨지는 무대다(3B poison Sp 0.000). 즉 "신호가 있는 곳"과 "추정기가 가장 취약한 곳"이 겹친다. 2×2 매트릭스는 셀마다 ‖Δ‖·additive-gap·Taylor 잔차를 병기해 "신호 존재"와 "추정 난이도"를 분리 보고해야 한다 — 이걸 명시하면 2차항의 존재 이유(1B poison 0.000→0.967)와 방법의 경계(3B)가 한 절에서 완결되는 부수 이득이 있다.
2. **"신호 실재"의 자명성 통제 부재**: quantity_skew의 xseed 0.97은 게임이 p_k∝n_k로 크기를 내장한 설계(participation-normalization 없음이 locked decision)상 거의 구성적으로 안정하다. 안정 순위가 관측 가능한 메타데이터(n_k, 참여 횟수)만으로 예측된다면 그 셀의 fidelity는 방법-변별력이 없다(크기-인지 휴리스틱이면 충분). B축 셀에 **자명성 검사**(φ와 n_k·참여수의 부분상관)를 넣고, 신호를 크기형(자명)/품질형/분포형으로 층화하라 — 진짜 시험대는 품질형·분포형 셀이다.
3. **세 번째 노이즈 플로어(val-sampling)가 미배제**: A-1은 fp32 플로어만 배제했다. 신호 ~1e-4–1e-3 vs val=200(std20)·50(device100)의 유한표본 SE는 같은 자릿수일 수 있다. (b)와 estimator가 val을 공유하므로 fidelity 표에는 안 보이지만, xseed 불안정의 원인 분해(궤적 확률성 vs val 유한성)와 실배포 해석(다른 val 표본이면 순위가 바뀌는가)에는 val-부트스트랩 안정성 분석이 필요하다. per-item loss가 캐시돼 있다면 저비용.
4. **음수 xseed(−0.37)는 "무신호"보다 나쁜 신호일 수 있다**: 0 근방이 아니라 뚜렷한 음수라면 seed-결합 아티팩트(전 클라가 SFTConfig(seed) 공유, free-rider stream만 seed+1 — Pass 1 J의 seeding 구조) 의심. 원인 규명 없이 "무신호"로 봉합하면 나중에 뒤집힐 수 있다.
5. **A축의 이중 효과**: 진단은 A축(lr·epoch·rank)을 "신호원 아님"으로 기각하지만, rank·lr 증가는 delta를 키워 **가법성 갭과 2차항 크기(추정 난이도)도 함께** 키운다. probe에서 r64 분리가 관측되면 "신호 증가"인지 "추정이 어려워져 방법이 갈린 것"인지 분해가 필요하다 — additive-gap을 probe 산출물에 포함하라.

**뒤집을 반론 — 프레이밍 전환 가능성 판정**

1. **near-additivity → 강점: 가능, 그리고 권장.** "게임이 쉬웠다"가 아니라 "**FL-LLM per-round 게임의 근사-가법성을 exact GT로 최초 정량(갭 ≤0.9%)했고, 이는 coalition-sampling 계열(GTG/FedSV/ShapleyFL/ComFedSV)의 비용 전제를 이 레짐에서 무효화한다**"로 명제화. Flirds는 가법 레짐에서 O(1)로 exact와 동률이고, 비가법 전환(poison·CNN·momentum)에서만 값을 내는 2차항을 같은 비용 구조에 내장한다 — "**비가법성에 대한 무비용 보험**"이라는 한 줄은 G-3(loss-heur 동률 공격)의 최종 방어이기도 하다. 단 레짐 스코프(LoRA-SFT·소규모 local step) 명시 필수.
2. **do-no-harm parity → 안전성 성질: 부분 가능.** "이득 없음"이 아니라 "신호 부재 시 φ-가중이 무해(상시 가동 valuation-in-the-loop의 필요조건) + 신호 존재 시 이득(CNN 오염 셀)"의 조건부 법칙으로. 단 이는 지지 결과이지 헤드라인이 될 수 없다 — parity를 전면에 세우는 순간 C-3 공격이 재점화된다. MMLU power≈0(K-2-2) 명기와 함께라야 정직하다.
3. **seed-불안정 → 정산(run-specific) 유스케이스: 가장 강력한 반론이자 서사의 열쇠.** 지불 대상이 "그 run의 실현된 기여"인 정산·감사에서는 run-특이성이 결함이 아니라 정의다(IRDS 계보; multi-run 평균은 인센티브 확장으로 위키에 이미 존치). 이 전환을 채택하면 (i) +1.000이 "분해능"으로 정합적으로 살아나고, (ii) 값-수준 fidelity(Pearson)와 efficiency 공리가 곧바로 화폐화 근거가 되며, (iii) 품질 추론·선택 유스케이스만 B축 신호 조건부로 남는다. 대가: 게임의 공정성 정당화가 정산의 근거가 되므로 **C-1(own-game·공리 논증·(a) 앵커)이 더욱 필수**가 된다.

**종합**: 세 전환 모두 성립 가능하되, **어느 것도 C-1을 우회하지 못한다.** 부록 A의 진단은 "무엇을 측정했는가(신호)"를 해명했지만, "누구의 자로 측정했는가(게임)"는 여전히 열려 있고, 최상위 바에서 논문의 생사는 후자에 걸려 있다.

### K-6. 부록 A 산출물에 대한 요구

진단 수치(≤0.9%, xseed 구간, SNR, MMLU SE)를 CSV/rundir로 정본화할 것(R1·R2와 동일 원칙 — 지금 상태면 진단 자체도 "노트-only"다). 진단의 A/B축 결론은 Pass 1의 F(§5.4 경계 절·§6 분석)·I(측정-우선 서사)와 정합적이므로 논문 뼈대로 직행 가능하다.

## L. 한 줄 종합 판정

- **(현재 상태)**: **Reject 성향, 경계선(≈4/10)** — 측정 인프라·정직성·스케일은 채택급이나, 헤드라인 증거(+1.000·5–15×)가 자기정렬 GT(C-1) 위의 무신호 무대(C-2, 저자 실측으로 확인)에서 나왔고 일부 헤드라인 수치가 정본화조차 안 돼 있어(R1·R2), 중심 주장 "기여도를 정확히 측정한다"가 현 증거로는 입증되지 않는다.
- **(§4 완료 시 예상)**: 2×2·probe를 계획대로만 이행하면 **borderline(5/10)** — 순환성이 새 표로 상속되기 때문. 여기에 **own-game 대조 + B축 (a) 앵커 + 자명성·trust-region 통제 + 이론 정식화 + 정본화**를 얹고 정산-유스케이스 중심의 "측정 표준 + 경계 지도" 서사로 재배치하면 **borderline accept ~ spotlight 경쟁권(6–7/10)**이 현실적 상한이다 — 병목은 실험량이 아니라 GT 정당화다.

---

# Pass 3 — 외부 리뷰(review-codex, GPT) 대조·흡수 (2026-07-03)

`review-codex.md`(독립 작성된 GPT 리뷰, 동일 2-pass 프로토콜)를 본 리뷰 완성 후 대조했다. 원본 파일은 흡수 후 삭제됨 — 이 절이 유일한 보존본.

## P3.1 합치 판정 — 독립 리뷰 2건의 수렴 (신뢰도 상승)

핵심 판정이 전부 수렴한다: ① 최대 리젝 사유 = near-additive·무신호 무대의 fidelity 포화 + GT 폭 부족, ② poison은 "정직한 답"이 아니라 1차 Taylor 부호실패(2차 부분 복원, 3B 경계), ③ LLM (a) oracle 정본은 1B anchor5뿐(3B +0.900은 provenance 미확보 — 재실행·정본화 전엔 부록에도 못 씀), ④ 비용 주장은 cohort-조건화 필수(std20에선 (b)가 더 쌈), ⑤ 서사 = "method + measurement science", 위계 = fidelity→성능→수렴→탐지, ⑥ L 판정: 현재 weak-reject/major(≈본 리뷰 4/10), 계획 완수 시 borderline~weak accept(≈본 리뷰 5–7/10). 두 리뷰가 서로 다른 검증 경로(codex는 `make_fidelity.py`·`make_analysis.py` 직접 재실행, 본 리뷰는 rundir·git ancestry 대조)로 같은 결론에 도달했다는 점 자체가 §0 판정들의 재현성 증거다.

## P3.2 codex 고유 기여 — 채택 (해당 절 보강)

1. **"Hardness ladder" 명명·5단 구성** [F §5 보강]: clean-IID = calibration/parity → std20 partial = estimator 전략 갭 → device100 K=10 = cost scaling → poison·non-IID·CNN = 비가법/hardness → retrain oracle = construct validation. 본 리뷰 F의 "IID 강등·B축 승격"과 동일 처방이나, 사다리 하나로 Results 배치와 "+1.000이 쉬워 보이는" 문제를 동시에 해결한다 — 채택.
2. **부록 A 정량 세부** [C-2·K-1 보강]: 가산 갭 = anchor에서 Σφ의 0.1–0.9%, std20 0.0–0.5%; singleton-vs-Shapley 순위 전 셀 ρ=+1.00; LLM (b) xseed 셀별 = 1B anchor **−0.37** / 1B std20 **−0.11** / 3B std20 **−0.24**. (전부 정본화 대상 — K-6과 동일 요구.)
3. **3B anchor5에서 Flirds-full(0.967) < Flirds-1st·loss-heur·GTG(1.000)** [C-8·B5 보강]: N=5 스왑 민감도라 과대해석은 금물이되, **"2차항이 항상 낫다"는 문장은 논문에 쓸 수 없다**는 금지 규칙으로 채택.
4. **γ(공격 강도) sweep** [G-7·K-5 심화1 보강]: trust-region 경계를 ‖Δ‖ 사후 진단만이 아니라 attack-strength 축으로 직접 스윕 — "2차항이 poison에서 결정적" 주장의 전제 조건(3B 3-seed와 병행).
5. **ASR-penalized / safety-aware utility를 별도 게임으로 제시** [G-7 반박 재료 추가]: "clean val-loss는 보상·안전에 부적합" 공격에 대해, 본 게임 정의를 바꾸지 않고 별도 게임을 병렬 제시하는 건설적 응답.
6. **valuation-aware adaptive attacker 갭** [H 추가]: 2×2를 다 해도 φ를 표적으로 조작하는 적응 공격자 문제는 남는다 — 한계 절 명시 대상.
7. **loss-heur의 "additivity probe" 승격** [G-3 보강]: 표에서 감출 것이 아니라 "singleton/LOO/Shapley가 일치하면 게임에 상호작용이 없다"는 진단 지표로 명시 — Reviewer 2 공격의 선제 흡수 장치.
8. **재현성 잡점 2건** [C-7·재현성 카드 추가]: (i) `make_analysis.py`가 Windows 기본 cp949에서 실패(`python -X utf8`로 우회; `read_text(encoding="utf-8")` 수정 필요), (ii) 현행 코드는 `RESULTS.md`를 생성하지 않음(overview 문서의 재생성 안내가 스테일; 실산출물 = `analysis/00_overview/master_metrics.csv` + 차트/CSV 34개).
9. **비용 절 보고 항목 확장** [F §5.5 보강]: wall-clock 외 **peak memory·commodity-GPU 이식성**(fp32 replay·HVP 메모리) 요구 예상 — 표에 병기.

## P3.3 채택하지 않은 것 (사유)

- codex Pass 1의 수치 인용 다수(anchor 전 방법 +1.000, 3B (a) +0.900, Ripple ~4515s 등)는 dossier 기준 서술 — 본 리뷰 §0(R1–R6)의 file-canon 판정이 우선하며, codex도 자체 Pass 2에서 같은 정정에 도달했으므로 별도 반영 불요.
- 그 외 대부분의 논점(IRDS-incremental 공격, do-no-harm 재프레이밍의 한계, off-anchor proxy 분리, CNN=diagnostic control, MMLU under-power, paired 설계, 신호 크기 vs 실재성 분해, "frozen-trajectory in-run Shapley"라는 명명)은 본 리뷰 B–K와 실질 동일 — 중복 병합 생략.
- codex가 다루지 않은 본 리뷰 고유 기여는 유지: **own-game 대조 실험(G-1), B축 자명성 통제(K-5 심화2), val-sampling 노이즈 플로어(K-5 심화3), 음수 xseed 아티팩트 의심(K-5 심화4), 정산-유스케이스 전환(K-5 반론3), β0.5/0.3 프로비넌스(R11), (b) oracle의 dtype 비대칭·seed 구조 등 코드-수준 위협(J)**.
