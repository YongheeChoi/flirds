# 슬라이드 분할안 v2 — 교수님 미팅용 (2026-06-advisor-meeting)

> v2(2026-06-12): Yonghee 피드백 반영 — ① plan §3.11 Track C/D를 계획 섹션에 정식 편입 ② 16장 → **표지+10장**으로 압축.
> 근거: instruction.md + checkpoint-2026-06-10 00–07 정독 + 5-agent 교차검증(23건 교정 반영) + plan §3.11.
> 상태: **Yonghee 컨펌 대기.** 컨펌 후 제작(첫 1–2장 스크린샷 톤 확인 → 전체). 서버 전용 수치는 **placeholder 빌드**(해당 칸 "확정 대기", 스크립트에 갱신 지점 주석).

전 장 공통: ⓐ 구현+smoke(값 coarse) / ⓑ 실측(설정 병기) / ⓒ 설계만·미실행. 흰 배경·무채색·표/수식 중심.

---

## S0. 표지
"Flirds 진행 보고: Client-level In-Run Data Shapley for Federated LLM Fine-tuning" / 날짜 / 3-state 표기 규약 안내.

## S1. 문제 설정과 선행 연구의 빈 부분
- 문제: FedAvg server는 round별 Δw_k·n_k만 봄(raw data 접근 불가). 목표 = post-hoc·재학습 0·추가 통신 0으로 client 기여도 φ_k. 용도: ① value semantics(corrupt→low value) ② selection.
- 선행 빈틈 표(각 1줄): retrain Shapley(G&Z'19; 1B N=5 fp32 실측 126분 ⓑ, N=10 ≈2–5일/1-GPU 추정·실행 ⓒ) / FL-SV 계열('20–'23; coalition 비용, 전부 CNN) / IF(iHVP 불안정) / IRDS(centralized·per-step·sample-level) / Ripple(AAAI'26; sample-level·CNN, 2차는 local-Hessian 시간 전파 — within-round client 상호작용 없음) / FedIF('25; 순수 1차, aggregation 변경) / FedTSV(ECC'26; 0차 기하, aggregation 물건).
- → 빈 교집합(6축): client-level · in-run · closed-form 1+2차 · HVP client-interaction 항 · 추가 통신 0 · LoRA/LLM (+post-hoc 별도 축). 근거는 서술적 서베이임을 명시.

## S2. 알고리즘 (수식·비용 ←→ 성질·제약 2단 구성)
- 입력: frozen FedAvg 궤적 logs=[(w_r, deltas_map)]만. φ_k = Σ_r p_k^r[⟨g^r,Δw_k⟩ + ½⟨Δw_k,u^r⟩], u^r=H^rΔW^r.
- round당 HVP 정확히 1회(2차항이 |P_r|개 내적으로 붕괴; forward H·v, H⁻¹ 불사용, true Hessian — GGN 시험 후 기각). 비용: R회 HVP+N·R 내적(N-독립) ↔ in-run oracle 2^N·R·val forward. Flirds-1st(2차 off): ~35s — coalition 계열(~530s) 대비 ≈15×, Flirds(107s) 대비 ~3×.
- free-rider φ=0 정확히(구조적: Δw=0→전 내적 0; 실측 매 seed exact 0 ⓑ).
- 2차가 FL서 의미: IRDS per-step은 2차 무의미 ↔ FL per-round Δw는 multi-step 누적. CNN ⓑ: plain SGD 0.962>0.924, momentum 역전(0.73<0.81) → plain SGD mom=0 고정.
- 운영 제약: fp32 필수(utility ~1e-3 < bf16 ~8e-3) · eager attention · LoRA-subspace 한정.

## S3. 실험 프레임 — 질문 → 실험 → 상태 (한 표)
| 질문 | 실험 | 상태 1줄 |
|---|---|---|
| Shapley 계산이 옳은가 | dual-oracle 삼중 비교 | ⓑ +1.000 (1B N=5 fp32 3-seed) |
| 기존 방법 대비 | 같은 frozen 궤적 9-method | ⓑ phase1 3-seed / FedIF ⓐ 편입·수치 대기 |
| 위협을 식별하나 | 2 regime × 4 threat + matched detector 4종 | ⓑ silo5 3-seed / device100 진행 중 |
| N=100서 성립하나 | per-round exact 분해 + α-sweep | anchor ⓑ(1-seed smoke) / sweep 진행 중 |
| 실용 가치 | selection run | ⓑ 3-seed, 양 lr |
| scale 유지되나 | 1B→3B→7B ladder | 1B ⓑ / 3B 부분 ⓑ(1-seed) / 7B ⓒ |
| near-additive 무변별 해소 | N=10 retrain oracle | ⓒ 연기(비용 — 논의 항목) |

## S4. 방법 검증 — dual-oracle · baseline 비교 · selection
- dual-oracle: 검증은 같은 게임(val-loss)이어야 — ROUGE는 미분불가+answer_swap에 속음(발산 +0.4@1B/−0.9@3B); retrain coalition val-loss 차(~0.005–0.02) < bf16 정밀도(~0.009) → fp32. 결과 ⓑ: retrain=in-run oracle=estimator +1.000(1B N=5 fp32 3-seed, lr 2종); 3B(1-seed) estimator +1.000, retrain vs in-run +0.900(재학습 noise).
- baseline ⓑ(phase1): noisy/FR 전 방법 Spearman +1.000 동률(N=5 near-additive) → "같은 랭킹을 더 싸게": 35s/107s/164s vs ~530s(GTG·FedSV·Banzhaf·ShapleyFL·oracle) vs Ripple ~4515s(별도 세션, eigsh flaky). free-rider φ exact-0은 Flirds·oracle·Banzhaf·loss-heur만(GTG·FedSV renorm≠0). FedIF 수치 [확정 대기].
- selection ⓑ(3-seed 양 lr): noisy+FR 정확 드롭({2,3,4} keep 일관), vs full win, vs random cross-seed win(seed0 tie). noisy AUROC lr 반전(0.75/1.0↔1.0/0.75) — selection 결론은 불변.

## S5. threat matrix — 2 regime × 4 threat + matched detector
- 라벨은 AUROC 채점 key지 method 입력 아님(순환 아님). 위협: answer_swap / FR zero·random(Lin'19) / backdoor(Xu'23+Bagdasaryan'20). matched: FedDQC↔noisy, STD-DAGMM·FLTrust↔FR, FLDetector·FLTrust↔poison.
- 결과(silo5 4-threat 3-seed, 첫 tier1 run ⓑ; 재실행본 수치 [확정 대기]): noisy/FR 전 방법 AUROC 1.0(Spearman 동률; 예외 FR-zero FedSV +0.967). **poison(ASR=1.00) = 동률 첫 붕괴**: Flirds-1st 0.000 완전 회피(두 run 일관), 2차는 run간 0.417±0.425 ↔ 0.917±0.118(비결정성, 원인 규명 중 — 두 값 병기). oracle·loss-heur·Banzhaf·ShapleyFL 1.0/Sp +1.000; GTG +0.867·FedSV +0.367 추락. 메커니즘: Taylor tangent vs exact secant.
- detector(tier1 ⓑ): poison 4종 전부 ≥0.917 / noisy FedDQC 0.917±0.118·FLDetector 0.75(off-threat)·STD-DAGMM 0.417±0.425 / FR-zero STD-DAGMM 0.083±0.118 실패(진단 필요)·FedDQC 0.75 / FR-random STD-DAGMM 1.0.
- 정직 headline(확정은 S10): noisy+FR은 잡고, clean-preserving backdoor엔 회피됨 — matched detector가 보완.

## S6. scale-up — cross-device N=100 · 1B→7B · N=10 연기
- N=100: per-client Dir(α) 혼합(Option B), K=10/round, α∈{0,0.01,0.1,0.5,5.0}, per_client=300. oracle = per-round exact 분해 ≡ 2^N(Δφ≈3e-16 ⓑ); 771ms/fwd → ~11h/4-GPU → α=0.5 anchor만, off-anchor는 Flirds proxy-truth(자기 오류 검출 불가 — 한계 명시).
- 상태: anchor Flirds vs per-round oracle +1.000 ⓑ(1B, 1-seed smoke); FR φ=0 exact; α-sweep tier2 진행 중 [확정 대기]; detector 1-seed: FLTrust FR 1.0 / STD-DAGMM FR 0.628.
- ladder: 1B ⓑ / 3B 부분 ⓑ(1-seed)+smoke ⓐ(no OOM) / 7B ⓒ(경로 구현됨, 미실행). 전부 fp32.
- N=10 retrain oracle ⓒ: 코얼리션-스텝 5120 vs 80(64×)+eval 32× ≈ 2–5일/1-GPU 추정 → 투자 판단(S10).

## S7. 남은 문제·한계 (2열: 방법 내재 / 증거력)
- 방법 내재: Taylor 절단(backdoor 회피 — tangent 맹점, secant는 잡음; 전개반경 진단 부재) / plain SGD mom=0 강제(AdamW 비호환 — 최대 일반성 제약) / fp32 강제(신호 ~1e-3 < bf16 ~8e-3 본질 제약) / eager·LoRA-subspace·vanilla FedAvg 한정·Δw 노출 / 단일 server val-loss 게임(val 오염 미고려 ⓒ).
- 증거력: N=5 near-additive 무변별(유일 분리=poison, Flirds에 불리한 방향) / noisy AUROC lr 반전(lr 미확정) / 1-seed 다수(3B +0.900, N=100 detector; FedDQC는 per-domain IRA 분석 한정) / proxy-truth 순환 / poison 인위성(별도 config서만 ASR>0, per_client 역조정) / 최약 공격자만(stealthy 불가·DBA 제외·FR easy만) / detector 개조 confound(원형 ablation 부재) / 문서 드리프트 전력(검증 세션 4건 교정 — 본 자료는 교정값).

## S8. 계획 ① — 진행 중 + main 트랙 보강
- real grid cost-tiered 완주(진행 중): silo5 재실행 완료 → device100 α-sweep 진행 → 3B → 7B (poison은 working-backdoor config 별도; run-dir 영속화 적용 1903a58).
- 무비용 분석(최고 우선): **poison Flirds-2차 run/seed 분산 원인 규명**(기실행 궤적 per-round φ 분해 재분석 — 2차항 novelty의 사활처) + two-sided |φ| 표준 산출물화 + orientation·lr 양쪽 보고 규약.
- 논문 전: 2차항 결정 실험(PGD/direction-aligned poison — FedIF 공인 blind spot; Ripple "2차 종류" 차별화와 묶기) / advanced free-rider(Lin II/III) / 원형 ablation 3건+STD-DAGMM FR-zero 진단+bootstrap CI / N=10 retrain oracle(투자 판단 → S10).

## S9. 계획 ② — 추가 실험 Track C/D (06-12 설계 확정 ⓒ, 구현 착수)
- 동기: main 실험(LLM+도메인-silo+detection)은 선행과 직접 비교가 어려움 → 선행 다수가 쓰는 "일반 학습 세팅"(CNN·IID/label-skew·fidelity/수렴/정확도) 비교 트랙 추가. 선행 13편 실험 프로토콜 조사 기반.
- **C1 fidelity & cost**(cross-silo CNN): MNIST+LeNet5/CIFAR-10+FedSVCNN, N=10 full, GTG 5-시나리오(graded label-flip ladder 포함), GT=(a) 2¹⁰ retrain+(b) exact **듀얼 oracle**, Spearman/Kendall+GTG 거리 metric+wall-clock, 3–5 seed, 9-method+Ripple(eigsh guard).
- **C2 일반성능**(cross-device CNN; 추가분의 메인): N=100 C=0.1 T=100–150, {IID, Dir(1), 2-shard}×{clean, label-flip(ρ,τ), free-rider, grad-noise}; 개입 3종 — 가중집계(곱셈형 w∝n·s 메인[고유 규칙]+대체형+additive λ0.5)·selection(S-FedAvg식)·bottom-q%(FedSV식); 평가=AUROC+최종 acc±seed+acc-vs-round+rounds-to-target. C1→C2 stage-gate.
- **C3 stability**(비용 0): cross-seed Spearman+top/bottom-k% 일관성(Banzhaf 프로토콜·Volatility 처방 응답).
- **D LLM 표준 세팅**(API-free): D-메인 Alpaca-GPT4 20k IID N=5, answer_swap 50%(FedDQC convention) → AUROC/Spearman vs (b) + φ-bottom 필터링 재학습→MMLU(random-q% 대조); 옵션 FedDQC Table-1·FedHDS Dolly 미러. 모델 1B/3B Llama-3.2, 7B=Llama-2-7b(FL-LLM 문헌 표준).
- 공백=기회: fidelity+학습개선 동시 커버 논문 없음 · (a)+(b) 듀얼 oracle 없음 · LLM-scale FL valuation 직접 경쟁자 없음.

## S10. 논의·결정 필요 (반 장)
1. threat-matrix headline framing 채택 여부("noisy+FR은 잡고 backdoor엔 회피 — matched detector 보완"; real config 재확인 후).
2. N=10 retrain oracle 투자: 2–5일/1-GPU vs 샤딩 구축(11–22h) vs 차선 cross-silo N=10 detection-only.
3. real grid lr: 2e-5(plan) vs 1e-3/3e-3(비교용) — noisy 결론이 갈림.
4. Ripple 차별화 문구 교체 승인("2차 없음"→"다른 종류의 2차").

---

## 운영 메모
- 산출물: `research-wiki/presentations/2026-06-advisor-meeting/` — self-contained HTML(1280×720, ←/→) + PDF + 재실행 빌드 스크립트(placeholder 갱신 지점 주석).
- 빌드 환경: 로컬 Python 3.13.5(playwright 설치 또는 Edge headless); 서버 python 경로에서도 동작하게 작성.
