# T1 — paper-ko §5(실험)·부록 B–E 실작성

> 전제: `00-INDEX.md`의 확정 구조·전역 결정을 그대로 따른다. 대상 파일 = `paper/paper-ko.md`.
> 원칙: 실제 값이 나온 부분은 기입(출처 주석), 미완 실험은 표 골격+⬚(채울 파일 경로 주석). 해석 문장은 값이 있는 부분만.

## A. 전역 수정 (본문 §1–§4·부록 A)

1. **초록**: "1B–7B 전 스케일 … (Spearman ρ≥0.999)" 문구 **삭제** — fidelity 주장을 메인 무대 기준으로 재작성(값 ⬚: c2fid·R4-L2 착지 후). 스케일 열거(1B–7B)도 제거(논문 LLM=1B).
2. **§1 검증 문단**: "기여도 상위 클라이언트만 참여시켜 재학습하는 selection 실험" → "**누적 기여도가 양(+)인 클라이언트만 남겨 처음부터 재학습하는 selection 실험**"(= R4 T2의 실체).
3. **§1 기여 2**: "1B–7B" 표현 있으면 제거. "exact 참값 대비 채점을 LLM 규모까지" 유지(R4=1B LLM).
4. **§2 말미 "(§5.6)"** → 탐지 절 번호(확정 구조상 §5.4)로 교정. §1 말미 "(§5.3)" → retrain 특성화가 §5.2 내 sub이므로 "(§5.2)"로 교정.
5. **부록 A**: A.6의 "잔차의 실제 크기는 실측으로 보완한다(§5.2)" 및 명제 3 본문의 동일 취지 문구 → 이론-한정 서술로 수정(실측 불수록 결정). **A.11의 "1B 3-seed 물리 실측(잔차 크기·스케일링)은 본문 §5.2 …" 문장 삭제**(GPT-2 대수검증 수치는 유지).
6. **§4.2 "Flirds (first-order)" ablation 문구** 유지(5.6-①의 근거).

## B. §5 절별 스펙

### 5.1 실험 세팅 (지금 전부 작성)
- 무대 요약표 1개: **주무대 쌍** — LLM `R4 gsm50k5`(GSM8K, N=50, 5/50, R=200, 오염 40%=클라 0–19, noisy=answer_swap@0.7·frzero, 심판=공식 test 1,119 EM) / CNN `C2 캠페인`(cifar10×[iid,dir1,shard,qskew]+fmnist×[iid,dir1], N=100, 10/100, R=120, threats clean·fr·frrand·gn·lf@{.15,.35,.70}·strmain) + **sub 무대** — gsm5(신설, §5.2용)·silo5·anchor5(retrain-(a)용)·CNN C1(N=10, dual oracle)·Scale 100/100(부록 E).
- 비교군 9종(위계: same-game = Flirds/Flirds-1st/loss-heur ↔ cross-game = GTG/FedSV/ComFedSV/ShapleyFL/FedIF), 제외 각주(Banzhaf·Ripple·poison), 탐지기 4종은 §5.4에서.
- 지표 정의(Spearman/Pearson[본문]·Kendall/거리[부록 C]·AUROC·절대 EM/acc·recovery)·고정-궤적 채점("한 궤적, 전 방법")·seed 규약(3-seed mean±std; ◐ 표기)·공정성 원칙(zero-semantics도 채점 대상; 셀별 튜닝 금지, P1 τ=0 parameter-free).
- 출처: overview §3.2.4(a)·c2fid README·§3.1.2(a)·§3.1.1(a).

### 5.2 Fidelity
- **메인 표(⬚)**: c2fid + R4-L2, same-game 3종 vs (b) — Spearman/Pearson, c2fid 열-호환 스키마. 채움: `runs/track_c/c2fid/analysis/fidelity.csv` · L2 rundir(`runs/phase2_matrix/rundirs/1B_gsm50k5_*`). 각주 2개: c2fid clean 칸은 신호-부재 레짐(오발화 대조용, fidelity 해석 금지) · F-4(strmain dose 해상도: Flirds ≈ (b) 자기천장 > 1st) 결과 서술 자리.
- **sub: retrain-(a) 특성화** — 도입 문구 필수: "(a)는 2^N 재학습이라 주무대(N=50/100)에선 불가 → 부득이하게 작은-N 별도 무대 + 실험 다양성 목적의 의도적 세팅 차이"(00-INDEX §0).
  - LLM: **gsm5 표(⬚, L8)** = 주 표(주무대와 데이터·위협·오염비율·val·하이퍼 동일, N=5 full·R=30만 축소 — "라운드-cohort 축소판" 프레임) / silo5 (a) ⬚ = 비IID 보조 / **anchor5 전 방법 vs (a)**(기존 기입: Flirds/1st/loss-heur/GTG 0.933 동률-천장, FedSV .733, ShapleyFL .767, ComFedSV .467, FedIF .167 — overview §3.1.1 "모든 방법 vs (a)" 표; 천장 효과 각주=(a)↔(b) 0.933).
  - CNN: **C1 시나리오별 vs (a)**(기존 기입 — overview §3.1.2의 07-23 신규 표) — 본문은 same-game 3종 × 전 10칸 + "신호-강 칸(lf·qskew·fn)에서만 두 게임 수렴, 1위 2칸" 서술; 전 방법 열은 부록 C(renorm-유리 칸 각주 T10).
- 잔존 각주(신호실재성 절 삭제 대체): 위 천장·clean-칸 각주 2개로 충분.

### 5.3 개입 (downstream)
- **메인 표(⬚)**: R4 P1 — T1 online/T2 retrain × {clean, noisy, frzero}, 절대 EM, 행=vanilla·oracle_excl·random_excl·t2_random + estimator 4점수원(renorm 4종은 L4 착지 시 블록 추가). 채움: L1 3-seed(`runs/track_h/rundirs_llm/gsm50k5_*` fix-후) → `analysis/gsm50k5_*.csv`. **pre-fix seed0 값 인용 금지.**
- **메인 표(기존, restack 확인 후)**: CNN 8점수원 × P1(T1/T2) 절대 acc — overview §3.2.3 (b1) P1 표 2개(+캠페인 확장 무대 착지 시 열 추가). 서술 클레임(정확형): "전 정책·전 시점 상위권 + grad-noise를 잡는 유일한 estimator(2차항) + FR에서 exact-0 계열 생존 vs renorm 붕괴"; clean 오발화(−0.7pt)·R4 clean T1(−1.0pt) 정직 보고 + "T2 최종-부호는 무해(kept=전원)" 대조.
- P1w 문단(결과 규칙부): 00-INDEX §1 규칙대로 — 착지 전엔 자리만.

### 5.4 탐지
- R4 표(⬚, L2): φ-파생(같은-게임 3종+(b)) + **전용 탐지기 4종**(FLDetector/FLTrust/STD-DAGMM/FedDQC — §2 약속 이행 지점) · c2fid AUROC(⬚). H-13 판정 기준 서술: 주장은 절대값이 아니라 **oracle-동행**(|AUROC(Flirds)−AUROC((b))|≤0.05).

### 5.5 비용
- op-count 소표(overview §3.4.1: silo 10 HVP×10.36=104s↔실측 107 등 모델-검증 라인) + 주무대 실측(⬚: R4 timing.json — K_r=5 조건부 "vs (b) ~5×"로만 서술 · c2fid runtime 열). 소-cohort 역전(작은 K_r에선 (b)가 더 쌈)은 op-count 축으로 서술(std20 실측 삭제 여파). 지수-비용 실측(N=10 160×·device100 159×)은 부록 E 포인터.

### 5.6 Ablation (기존값 기입)
- ① **2차항**: CNN k-sweep(k=0.2 Flirds .891 vs 1st .305 — §4.3.1 (b3)) + 경쟁 GN 실명(.567~.607 vs .244~.248 — §3.2.3) + (c2fid F-4 착지 시 dose 해상도 추가).
- ② **A축 lever**: rank·lr·steps·폭·참여 lever가 신호를 못 만들고 fidelity는 lever 전반 1.000(Taylor tradeoff 없음) — §4.2·§4.3 요약(표 최소).
- ③ **removal**: silo5 worst-first Δval-loss +0.0067~0.0076 vs best-first 음수(§4.4.1) + cifar10 acc 분리 +0.039~0.045=(b) 동급·저순위 방법 분리≈0(§4.4.2).

## C. 부록 B–E

- **B 프로토콜**: 무대별 하이퍼 표·위협 구현 정의(answer_swap/frzero/frrand/gn/lf)·데이터 분배 규칙(GSM8K val=공식 test 카브 등)·환경("fp32; cuDNN conv TF32; 스택 내 결정론" 1줄)·ComFedSV per-round 대용 caveat·ShapleyFL β=0.3 각주(β0.5 대비 재실행 노이즈 수준)·**LLM 위협축에 grad-noise가 없는 이유 2문장**(등방 노이즈는 LoRA 기하에서 gradient-방향 대비 응답이 수십분의 일 — 무대 미성립; 내부 진단 인용 없이 자립 서술).
- **C fidelity 확장**: cross-game 전 방법 vs (b)(c2fid·R4-L2 전표 ⬚ + C1 vs (b) 시나리오 표) · vs (a) 전 방법(C1 시나리오·anchor5) · Kendall/거리 · std50k5 부분참여 probe(§4.2 (b2)).
- **D stability**: C1 방법 안정성(Flirds .547=(b) .518 — §3.1.2 (b2)) + (b) target 안정성(수록 무대 한정; c2fid 착지 후 열 추가).
- **E 비용·규모 보조**: Scale 100/100 P1 표(§4.8.2 — P1 행+vanilla/oracle/random 앵커만; P5 행 제외) + N=10 2¹⁰(§3.1.3 (b2): (b) 32.7h vs Flirds 733s=1/160, 1-seed 명기) + device100 anchor(§3.3.2 (c): (b) ~25,000s vs Flirds 157s=1/159).

## D. §6 스텁(주석)·완료 조건

- §6에 HTML 주석으로 한계 재료 목록만: 궤적-특이 공리화 승계 · token-mean 브리지 비성립 · LLM 위협 스코프(gnoise 미성립) · frdelta("기여도≠탐지" 게임-공통 사례 1문장 후보) · fairness/reward 향후 과제 · 1-seed 항목(N=10 등).
- 완료 = §5.1~5.6+부록 B–E 골격·기존값 완성, ⬚마다 채울 경로 주석, 전역 수정 A-1~6 반영, latexmk 대상 아님(한글판) — 이후 영문 tex 반영은 별도.
