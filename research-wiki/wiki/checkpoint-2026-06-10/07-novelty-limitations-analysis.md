---
type: checkpoint
title: "Flirds 체크포인트 07 — novelty·한계 분석 + 개선 제안"
created: 2026-06-10
updated: 2026-06-12
note: "00–06 작성 후 같은 날 오후의 분석 세션. checkpoint 문서 직접 정독 + 8개 병렬 reader(checkpoint·flirds.md·threads·plan·protocol·코드·raw)로 교차 정독 + load-bearing 주장 4건 직접 검증. §7.0은 00 §0.5 / 05 §5.4의 'real grid 미실행'을 SUPERSEDE. 06-12에 최신 사실(영속화 완료·FedIF 편입·poison run간 분산·Track C/D)을 본문에 병합 반영."
---

# 07 · novelty·한계 분석 + 개선 제안

> **목적**: "이 연구의 novelty와 한계는 무엇이고, 어디를 개선할 수 있나"에 대한 종합 판정.
> **방법**: 00–06 직접 정독 + 8방향 병렬 reader 교차 정독 + 직접 검증 4건(FedIF 부재 grep, `runs/` 상태, cross-silo tier 로그, baselines 디렉토리). 모든 정량주장에 경로 근거, 3-state(ⓐ/ⓑ/ⓒ) 구분.
> 경쟁자 3종(FedIF/FedTSV/Ripple)의 상세 포지셔닝은 [06-closest-competitors-fedif-fedtsv-ripple](06-closest-competitors-fedif-fedtsv-ripple.md) — 이 문서는 그 위에서 novelty 방어력·한계·개선을 판정.

---

## 7.0 ⚡ SUPERSEDE: real grid가 시작됐다 (00 §0.5 / 05 §5.4의 "미실행" 교체)

checkpoint 작성(06-10 00:50) 직후 real grid가 시작됨. 직접 확인(06-10 오후):

- **cross-silo tier (silo5 4-threat × 3-seed, 1B) 완료** — `runs/phase2_matrix/tier1/silo5_{noisy,freerider_zero,freerider_random,poison}.log` (03:58–04:14 완료). noisy/FR = lr1e-3/batch16, poison = working-backdoor config lr2e-3/batch8, 전부 R=10, ORACLE_B+COALITION on.
- **cross-device tier (device100 α-sweep 12 cells) 진행 중** — `tier2/_driver.log` (09:27 시작, 4-GPU).
- **영속화 완료(.log-only 상태 해소)** — `runs/phase2_matrix/rundirs/`에 **22/25 셀 metrics.json 영속화 완료**(커밋 1903a58 run-dir persistence + 8d364cc 20셀 소급 + b9113c4 poison 2셀). 남은 3셀 = dev_a0.5 anchor {noisy, frrand, frzero}. **수치 기준 = rundirs metrics.json**(일부 .log run 수치와 run간 차이 존재: 예 silo5 noisy STD-DAGMM 0.417±0.425→0.417±0.312, FR-zero STD-DAGMM 0.083±0.118→0.250±0.204 — 둘 다 실측).

### cross-silo tier 핵심 결과 (ⓑ, real config 3-seed — 직접 로그 대조)

| threat | valuation (전 방법) | Flirds-1st | Flirds (2차) | 비고 |
|---|---|---|---|---|
| noisy | AUROC 1.0 / Spearman +1.000 전부 동률 | 1.0 | 1.0 | FLDetector 0.75(off-threat), STD-DAGMM 0.417±0.425, FedDQC 0.917±0.118 |
| freerider_zero | 동률 (FedSV Sp +0.967) | 1.0 | 1.0 | **STD-DAGMM 0.083±0.118 실패**(matched threat인데!), FedDQC 0.75 |
| freerider_random | 동률 | 1.0 | 1.0 | STD-DAGMM 1.0 (random만 잡음) |
| **poison** (ASR=1.00) | **동률 첫 붕괴** | **0.000±0.000 완전회피** (Sp +0.000) | **seed별 {0.0, 0.25, 1.0} = 0.417±0.425** | in-run oracle·loss-heur·Banzhaf·ShapleyFL AUROC 1.0/Sp +1.000; GTG Sp +0.867; **FedSV Sp +0.367 추락**; 4 detector ≥0.917 |

> **poison 행 — 영속화 run 기준**: `runs/phase2_matrix/rundirs/silo5_poison/metrics.json`에선 **Flirds(2차) AUROC [0.75, 1.0, 1.0] = 0.917±0.118**(Sp 0.967±0.047) — 위 표의 .log-only run({0, 0.25, 1.0}=0.417±0.425)과 **run간 분산이 큼**(두 run 모두 실측, 어느 쪽도 무효 아님 — 이 run간 분산 자체가 핵심 관찰). 영속화 run은 3-seed 모두 2차>1차 = '2차항이 1차를 이긴 데이터포인트'가 seed2 단발이 아님. Flirds-1st 0.000 완전회피는 두 run 공통. 규명 대상 = **run간 분산**(§7.3 제안 #2 참조).

**poison seed2 = 2차항이 1차를 이긴 최초의 LLM-scale 데이터포인트** (Flirds 2nd AUROC 1.0/Sp +1.000 vs 1st 0.0/+0.000) — 단 seed 분산이 극심(0/0.25/1.0). 이 분산의 원인 규명이 §7.3 제안 #2 (현 시점 최고 가치 실험).

---

## 7.1 Novelty 평가 (방어력 등급순)

### 강함 — ⓑ 근거, 리뷰어 방어 가능

| # | 주장 | 평가 |
|---|---|---|
| N1 | **dual-oracle 검증 방법론**: retrain val-loss=in-run oracle=estimator +1.000(1B N=5 fp32 3-seed) + retrain ROUGE 발산 반례(+0.4@1B/−0.9@3B) + bf16 정밀도 바닥 진단 | "같은 게임으로 검증" 통찰+실증 = FL valuation에 독립적으로 내놓을 만한 방법론 기여 |
| N2 | **in-run oracle exact per-round 분해**: $2^N$ → $\sum_r$ $2^{|P_r|}$, Δφ≈3e-16 | Shapley linearity의 귀결이라 인프라성이나, volatility 문헌이 필요성을 독립 뒷받침하는 견고한 검증 장치 |
| N3 | **비용 구조**: round당 1 HVP(N-독립)+N내적, 35–107s vs ~530s | ⚠ **"zero-comm"은 차별점 아님** — 자체 비교표([[flirds]]:228-231)서 GTG/ShapleyFL/Ripple/FedIF도 comm 0. 진짜 차별 = **서버 연산 구조(1 HVP vs $2^N$ coalition)**로 서술 교정 필요([[flirds]]:52 내적 불일치) |

### 중간 — 조건부

- **교집합 novelty** (client-level+in-run+closed-form 1st/2nd+zero-extra-comm+LoRA/LLM): "최초 federated in-run"은 Ripple 선점 자인 후 교집합으로 후퇴한 구조([06](06-closest-competitors-fedif-fedtsv-ripple.md) §6.1 표 참조). 근거가 서술적 스캔(4-agent+GPT)이라 리뷰어에게 "incremental combination"으로 읽힐 위험. **교집합의 load-bearing 성분 = 2차항** — 서면 전체가 서고, 무너지면 "Shapley 프레이밍+LoRA 엔지니어링"으로 축소.
- **경계 특성화** (backdoor evasion = Taylor tangent vs exact secant 메커니즘; `memory/phase2-step5-verification.md`): 정직한 기여 framing. cross-silo tier의 seed-의존 2차항 결과로 이야기가 풍부해짐.
- **FLTrust ≈ normalized Flirds-1st**: 양날 — 포섭 서사인 동시에 1차 검출 신호가 2021년(FLTrust)+2025년(FedIF, 같은 정규화 내적 — [06] §6.3) 선점됐다는 뜻.

### 약함 — 미입증 (novelty의 사활처)

- **2차항 가치**: 어제까지 LLM-scale 입증 0건(N=5/N=100 모두 1st=2nd=+1.000, 1st가 15× 저렴; CNN 0.96>0.92는 plain-SGD 조건부 — momentum서 역전). **cross-silo tier poison seed2가 첫 분리 신호이나 불안정.** Ripple와의 "2차 종류" 차별([06] §6.5)은 ⓒ Yonghee 결정 대기.
- **format-uniformity / per-domain norm hook**: ablation 전부 ⓒ — 현재는 "기여"가 아니라 "control 변수 선택".
- **이론(Prop 1/2)**: informal + 파일럿 U-shape 모순 관측, #5 검증 ⓒ.

### 리뷰어 1순위 공격 지점 (직접 확인)

**FedIF(2025)** — wiki 스스로 "Ripple 외 가장 가까운 federated in-run-on-Δw 경쟁자", "'why isn't 1st-order enough?'의 핵심 baseline"으로 등재([[flirds]]:231,245)한 최근접 published 경쟁자. **baseline 구현·비교 suite 편입은 완료** — `codes/flirds/baselines/fedif.py`(커밋 1903a58) + 영속화 22/25 셀 전부 FedIF 실측(예: silo5_poison AUROC 1.0/Sp 0.967±0.047, dev_a0.0_poison AUROC 0.542±0.258 — 기준 rundirs metrics.json). [06] 포지셔닝 문서도 같은 날 작성([06] §6.3 "Flirds-1st ablation ≈ FedIF 메커니즘"). **잔여 = [03-baselines-and-prior-work](03-baselines-and-prior-work.md) 문서 통합뿐** — FedIF가 [03]에 여전히 0회 등장(grep 확인).

---

## 7.2 한계 분석

### A. 방법 내재

1. **Taylor 절단** — clean-preserving backdoor의 γ-update가 clean val-loss를 내려 최저 φ로 랭킹. cross-silo tier서 1st 완전회피(0.000), 2nd seed-의존. 전개-반경 진단/가드 코드 전무.
2. **plain SGD mom=0 강제** — AdamW 표준 실무와 비호환, momentum하 2차항 역효과(0.73<0.81) 기관측, FedHDS(Adam) 벤치와 충돌. 가장 큰 일반성 제약.
3. **fp32 강제** — 신호(~1e-3) < bf16 정밀도(~8e-3) 자체가 본질 제약. 7B 비용(no-tensor-core ×3.1), bf16 배포 궤적 valuation 불가, protocol §13 vs scale-up 모순 미해소.
4. **eager attention 강제** / **LoRA-subspace 한정**(실범위="LoRA-subspace Shapley") / **FedAvg 한정**(robust agg·FedProx·secure agg 비호환 — backdoor 방어가 robust agg 쓰는 시스템과 양립 불가) / **개별 Δw 노출**(privacy).
5. **단일 server val-loss 게임** — val 오염/편향 미고려, #16 민감도 연기. retrain ROUGE 발산은 "val-loss가 안 속음"인 동시에 "배포지표 가치와 다른 게임" 반론 근거.

### B. 증거력

6. **N=5 near-additive 무변별** — noisy/FR은 real config서도 전 방법 +1.000 동률(cross-silo tier 재확인). 유일한 분리 = poison인데 Flirds에 불리한 방향. **N=10 retrain oracle 연기** = 무변별 해소의 유일한 직접 증거 부재.
7. **lr 의존 AUROC 반전** — noisy φ 부호가 lr 의존(−0.0084↔+0.0096 → 0.75↔1.0). real grid lr 선택이 noisy 결론을 결정하는데 lr 미확정.
8. **1-seed 다수**(3B +0.900, N=100 detector) / **α-sweep proxy-truth 순환**(off-anchor truth=Flirds 자신 → 원리적으로 자기 오류 검출 불가) / **CI(§3.3)·timing(§15.1) 미구현** / seed [0,1,2] vs protocol [42,123,2024].
9. **영속화 — 해소** — cross-silo·cross-device tier의 .log-only 상태 해소: 22/25 셀 metrics.json 영속화 완료(`runs/phase2_matrix/rundirs/`; 커밋 8d364cc 20셀 + b9113c4 poison 2셀; 남은 3셀=dev_a0.5 anchor {noisy, frrand, frzero}). 수치 기준=rundirs (§7.0).

### C. 위협모델·framing

10. **poison 인위성** — hand-tuned 별도 config(LR2e-3/B8/E5/frac0.8)만 ASR>0, batch 하나로 1.0↔0 반전, per_client 40→300은 공격 성립 위한 역조정. 위협별 trajectory config 상이 = category-together 원칙 훼손.
11. **최약 공격자만** — stealthy arm 불가(‖Δ‖=40×), DBA 제외, **free-rider easy 모드만**(delta/recycled-aggregate = "genuinely AT RISK" 자인 모드가 advanced free-rider 이연), adaptive attacker 부재.
12. **orientation protocol 취약성** — backdoor detection test 사건(−φ 채점→정반대 결론). 교정됐으나 규약 명문화 없으면 재발.
13. **detector 개조 confound** — FLTrust signed-cosine=원형 강화, STD-DAGMM hash+pooling=약화 가능(0.628·cross-silo tier FR-zero 0.083이 port confound인지 미분리), Ripple=우리 port 위 측정. 원형-ablation 부재.

### D. 프로세스

14. **낙관적 distillation 드리프트 반복** — 검증세션 4건 교정(poison detector 1.0→0.75, FedSV +0.900, 360s, REFUTED→EVADED). backdoor detection test 오류 결론이 immutable raw에 정정 포인터 없이 잔존.
15. **protocol stale**(06-04) — 분기 4–11·in-run oracle MC→exact 미반영 = "silent deviation 금지" 자기위반. [03]의 ShapleyFL +0.86 state-mixing(ⓐ 수치를 ⓑ 맥락에 무라벨 인용).

---

## 7.3 개선 제안 (우선순위순)

### 🔴 즉시 (real grid 진행 중인 지금)

1. **결과 영속화 — 완료**: `phase2_matrix.py` run-dir 출력(metrics.json+φ archive+config+git SHA) 구현 + cross-silo·cross-device tier .log 파싱 소급(커밋 1903a58 run-dir persistence + 8d364cc·b9113c4 소급, 22/25 셀; 남은 3셀=dev_a0.5 anchor {noisy, frrand, frzero}).
2. **poison Flirds-2nd 분산 원인 규명**: 영속화 run에선 [0.75, 1.0, 1.0]=0.917±0.118로 3-seed 모두 2차>1차(§7.0) — 규명 대상은 seed간이 아닌 **run간 분산**(.log-only run {0,0.25,1.0}=0.417±0.425 vs 영속화 run). per-round φ 분해(1차 vs ½⟨Δw,HΔW⟩ 기여). **기실행 궤적 재분석이라 비용 ~0, 2차항 novelty의 사활 — 현 시점 최고 가치 실험.**
3. **two-sided |φ| 점수를 matrix 표준 산출물로**: "φ-extreme not φ-high" 발견의 label-free 제품화. 비용 ~0.
4. **orientation 사전고정 + lr 양쪽({1e-3,3e-3}) 보고 규약 명문화**: lr-반전은 숨기면 약점, 보고하면 robustness 특성화.

### 🟠 단기 (논문 전 필수)

5. **FedIF baseline 편입 — 완료**(`baselines/fedif.py`, 커밋 1903a58; 신호=Δw/‖Δw‖·∇val — [06] §6.3; 영속화 22/25 셀 실측). **잔여 = [03]에 수학적 관계+min-max/EMA 차이 명문화** — 1순위 공격 방어의 문서 측 마무리.
6. **2차항 결정 실험(#13 PGD/direction-aligned poison) 앞당기기**: cross-silo tier seed2 신호와 결합하면 "2차항이 언제 필요한가" 양방향 폐쇄(져도 honest fallback 기설계). [06] §6.5의 Ripple "2차 종류 차별"과 묶으면 경쟁자 대비 스토리 완성.
7. **N=10 retrain oracle** (multi-GPU sharding 11–22h) — near-additive 무변별 해소의 유일한 직접 증거. 차선 = cross-silo N=10 detection-only(plan 기허용).
8. **advanced free-rider**: Flirds 진짜 위험 모드 — 뚫리면 경계 특성화, 막으면 robustness, 양쪽 다 논문 가치. backdoor 회피와 동종 맹점(val-loss 내리는 적대 update)의 free-rider 축 존재 여부가 threat matrix framing 완결성 결정.
9. **loss-heur 공식 보완 검출기 패키징**: "Flirds(tangent)+singleton secant 잔차" 하이브리드(~164s, 같은 frozen logs) = "경계 발견+해법" 서사 완성 + 비용/역할 분리표.
10. **원형-ablation 3건 + bootstrap CI**: ReLU-FLTrust / per-round STD-DAGMM / Ripple 저자코드 1-setting; CI는 φ archive만 있으면 소급 적용.
11. **STD-DAGMM FR-zero 0.083 진단** (cross-silo tier 신규): zero-delta가 matched detector에 안 잡힘 = suite 서사의 구멍. pooling/hash confound 분리.

### 🟡 중기 (한계→기여 전환)

12. **momentum/Adam-aware Taylor**: velocity-tail 원인분석 기존재 → preconditioned delta 해석 확장 = 분기 #2를 한계에서 기여로.
13. **Taylor 신뢰도 자기진단**: 샘플 round서 1st+2nd 예측 vs `in_run_utility([k])` exact 잔차 → 전개 붕괴 runtime 플래그(loss-heur 경로 재활용). lr 부호반전 사전감지 가능성도 답해짐.
14. **per-round $2^K$ oracle의 MC/stratified 근사** → α-sweep 전 지점 독립 truth = proxy-truth 순환 해소.
15. **이질-포맷 ablation**: classification-포맷 client 의도 혼입 → φ 왜곡 정량화 = format-uniformity를 "한계 회피"에서 "finding+권고"로.
16. **double-backward HVP**(jvp-동치 9.8e-6 기록만 존재) → SDPA/flash 활성화 = 7B eager 병목+val_maxlen confound 제거. 필요시 LoRIF Woodbury·ACC-SGD-IE.
17. **저비용 설득 실험**: #17 per-client φ 정성 attribution(per-layer 로깅 기구현, near-free) + #18 clean-data skyline.

### 운영 (작지만 지금)

protocol 현행화(분기 4–11·seed 실값·7B 정밀도 모순) / backdoor detection test raw에 위키 측 정정 포인터 / [03] ShapleyFL +0.86 state-mixing 교정 / cross-device tier `done[CHECK]` cell 확인.

---

## 7.4 종합 판단

**가장 단단한 자산** = 방법 정당성 증명(dual-oracle 삼중 일치 + exact per-round 분해 + 비용 구조)과 한계를 메커니즘 수준까지 파는 특성화 문화. **가장 큰 리스크** = novelty 핵심 축인 2차항의 LLM-scale 입증 공백 — 최근접 경쟁자 FedIF의 비교 suite 부재는 해소됨(`baselines/fedif.py` 편입+22/25 셀 실측; [03] 통합만 잔여). 영속화 run(silo5_poison)에선 Flirds(2차) [0.75, 1.0, 1.0]=0.917±0.118로 3-seed 모두 2차>1차 — **지금 최고 가치의 한 수 = poison 분산 원인 규명(제안 #2, 규명 대상은 run간 차원 — §7.0)** — 기실행 데이터 재분석이라 비용이 거의 없고, 결과가 논문 중심 서사를 결정한다. 이후 다음 단계는 **Track C/D 표준-세팅 비교 실험으로 확정**(plan §3.11, 06-12): C1 CNN fidelity&cost(MNIST+LeNet5/CIFAR-10, N=10 full 2¹⁰ 듀얼 oracle, GTG 5-시나리오) → C2 CNN 일반성능(N=100 C=0.1, 개입 3종: 곱셈형 w∝n·s 가중 메인+selection+bottom-q%) → C3 cross-seed stability + D LLM 표준세팅(전부 API-free, D-메인 Alpaca-GPT4 20k IID answer_swap50%→MMLU; 7B=Llama-2-7b-hf). **Track C 구현 완료=ⓐ**(구현+단위검증, 실측 없음; 커밋 5b0ba71), **Track D=ⓒ**(설계만).

*근거: cross-silo tier 수치는 `runs/phase2_matrix/tier1/*.log` 직접 대조(06-10), 교정값은 `memory/phase2-step5-verification.md`, 나머지는 [00](00-overview.md)–[06](06-closest-competitors-fedif-fedtsv-ripple.md) + [[flirds]]/[[flirds-implementation-plan]]/[[flirds-protocol]]/threads/raw 정독.*
