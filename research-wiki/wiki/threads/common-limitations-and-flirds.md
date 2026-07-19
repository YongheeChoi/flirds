---
type: thread
title: "선행 전수 조사 — 공통 한계 6축과 Flirds의 응답"
created: 2026-07-20
updated: 2026-07-20
sources: [principled-federated-data-valuation, gtg-shapley, comfedsv, shapleyfl, shapfed, game-of-gradients-sfedavg, space-participant-amalgamation, fedtsv, fedif, ripple-shapley, du-shapley, mavericks-shapley-fl, shapley-volatility-fl, in-run-data-shapley, data-value-embedding, datainf, less, mates, dsdm, trak, logix, lorif, grosse-llm-influence, influence-functions-fragile, do-influence-functions-work-on-llms, accumulative-sgd-influence, feddqc, fedhds, fldetector, fltrust, free-riders-fl-std-dagmm]
tags: [synthesis, limitations, flirds, positioning, abstract-framing]
---

# 선행 전수 조사 — 공통 한계 6축과 Flirds의 응답

2026-07-20, 초록 gap 문장 검증을 위해 source 노트 44편 전수 + [[prior-work-taxonomy/README|taxonomy]] + [[validation-experiments]] + [[metrics-and-benchmarks]]를 교차 대조한 종합. 결론: **공통 한계는 6축으로 수렴하고, 인과의 뿌리는 하나 — retrain Shapley의 계산 불가능성.** 비용 회피가 정의 파편화를 낳고, 파편화가 검증의 간접화를 낳았다.

## 한계 6축 → Flirds 판정

| # | 공통 한계 | 대표 근거 | Flirds |
|---|---|---|---|
| L1 | **정의 파편화** — 비용 회피를 위해 저마다 다른 utility proxy/게임 | proxy 스펙트럼: val-loss(FedSV·GTG) / 행렬완성(ComFedSV) / 방향정렬(FedTSV·FedIF) / class-cosine(ShapFed) / prototype(SPACE) / surrogate 자인(ShapleyFL) / 공리 포기(FedIF); [[sources/shapley-volatility-fl]] 보상 30–50% 출렁 | **해결** — 게임 형식 정의 + 명제 1–2(추정기=그 게임, 1e-12) + Taylor 절단 bound. 단 "우리 게임도 선택"임을 자인(명제 4 양방향) |
| L2 | **검증의 간접성** — 추정기가 자기 목표를 맞추는지조차 exact 대비로 안 잼 | fidelity-vs-exact는 예외적(ShapFed synthetic·DU asymptotic뿐); FedTSV "oracle GT 없음" 자인; Ripple "comparable accuracy" 참조 불명; GT 카탈로그 대부분 근사(TMC/LOO/LDS/PBRF); LLM 규모에선 참값 자체 붕괴([[sources/influence-functions-fragile]] LOO noisy, [[sources/do-influence-functions-work-on-llms]] iHVP 무기여) | **해결** — 2겹 검증: 추정기=자기 게임 exact 2ᴺ 대비(ρ≥0.999, MC 분산 0), 정의=게임-무관 removal·selection·탐지([[downstream-is-the-neutral-judge 원칙과 정합]]) |
| L3 | **무대 분리** — FL valuation 전부 CNN, LLM 규모 전부 centralized | Agent 전수 확인: FL 15편 중 LLM은 iPFL·RFedLR뿐인데 정확히 그 둘만 비-valuation; centralized LLM 11편 전부 "not FL" 명시; federated×LLM 칸 점유자는 quality/selection/market뿐 | **해결(사실 서술만)** — 1B–7B LoRA 첫 실험. "비용 때문에 불가능했다" 인과 주장은 금지(paper-framing-two-overclaims) |
| L4 | **단위·비용 구조** — sample-level 비용∝N + inverse-Hessian 병목; coalition은 O(2^{\|P_r\|})/MC | per-sample=신뢰 경계 반대편+Θ(Nn) HVP(ACC-SGD)+O(ND) 저장(3.5TB, LoGra); (H+λI)⁻¹→λ⁻¹I 붕괴(Do-IF Thm.1); FedSV O(Tm²)·ComFedSV 전원-라운드 요구·Ripple 자기궤적 47–113× | **해결(조건부)** — client 단위(신뢰 경계 안) + forward HVP(H⁻¹ 부재=IF 취약 경로 자체 회피) + 라운드당 O(1). 소 cohort 역전(N=20)은 명시 |
| L5 | **공리 위반/부재** — free-rider 양수, 정규화가 공리 파괴, 1차 전용 | FedSV norm-정규화가 rationality/additivity 파괴+희소참여 음수 SV; ComFedSV symmetry를 completion으로 땜질(δ 무통제); FedIF min-max order-dependent+Hessian "infeasible" 기각; 실측 GTG 0.0037/FedSV 0.0047 free-rider 지급 | **대부분 해결** — null-player exact-0 대수 보장+efficiency 오차 0+2차 항 실증 이득(저참여 +0.89 vs 1차 +0.31; FedIF의 기각 반박). maverick 편향은 공유(측정만) |
| L6 | **신호 존재 질문 불가** — exact 참값이 없어 "이 세팅에 순위 신호가 실재하는가"를 아무도 못 물음 | Volatility는 불안정만 경고(dataset-size GT라 원인 분해 불가); Banzhaf 안정성 문제의식은 centralized; Basu — 대규모에선 참값도 noisy | **해결(신규 축)** — exact 자기일치도 지도(IID-clean ≈무신호 vs non-IID clean +0.87), 검증노이즈/세팅노이즈 분리, 방법 분산 vs 세팅 무신호 분해(추정기는 분산 안 더함 0.547 vs 0.518) |

## 인접 계열(탐지·품질·선별)이 valuation을 대체 못하는 이유 (노트 명시분)

이진/hard(keep-discard·trust-weight) vs signed 연속 기여도; 품질≠대표성≠가치([[sources/feddqc]]·[[sources/fedhds]] 자인, [[threads/data-quality-vs-data-value]]); 위협-특화(crafted-update만); **non-IID에서 bad-different/good-different 분리 실패가 공통 실패 지점**(FLDetector IID-only 보장, FoolsGold는 역으로 IID 오탐, STD-DAGMM/FedCorr benign 과다 flag); 공정 분배 공리 전무(9편 전부).

## Flirds가 공유하는(해결 못하는) 한계

secure aggregation 비호환(client-level 클래스 전체; VLDB Secure-Shapley 방향이 보완) · maverick/이질성 편향([[sources/mavericks-shapley-fl]] 상속, 측정만) · 궤적-특이성 공리화 미해결(IRDS 승계) · optimizer 레짐(plain SGD·서버 무상태; [[sources/data-value-embedding]] "Adam이 깨뜨림"과 동형, IRDS-Adam '26이 보완 방향) · 서버 검증셋 필요(validation-free 계열과 대비) · delta-위장 free-rider·cross-device noisy 탐지 열세(게임 수준 한계) · per-sample 세분화 없음.

## 논문 프레이밍 함의

초록 gap 3문장("retrain 감당 불가 → 저마다 대체 정의+간접 증거 → CNN 무대")은 L1+L2+L3와 1:1 — FL 계보 노트의 인과 사슬(비용 압박→proxy 다양성→fidelity를 downstream으로 대체)이 그대로 확인. **서론 §1 "남은 벽"(현재 '문제는 조건 준수가 아니라 계산 규모')은 L1·L2를 반영해 재작성 필요.** retrain 비교는 기여로 내세우지 않음(2026-07-20 결정).
