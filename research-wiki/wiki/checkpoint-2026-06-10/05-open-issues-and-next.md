---
type: checkpoint
title: "Flirds 체크포인트 05 — 미해결 + 다음 단계"
created: 2026-06-10
updated: 2026-06-12
---

# 05 · 미해결 이슈 + 다음 단계

> 판단 대기 / 알려진 caveat / 즉시 다음 단계. 근거: 코드·raw 로그·`memory/phase2-step5-verification.md`.

---

## 5.1 판단 대기 — poison vs Flirds framing (verification-item #1)

**기술적 사실은 ruling 났으나, 논문 headline framing + real-config 재확인은 미결.**

- **충돌**: backdoor detection test smoke(`phase2_backdoor_d2b_smoke.py`)는 "clean-preserving backdoor가 Flirds를 evade한다 = **REFUTED**"라 결론(Flirds-1st/2nd 전부 AUROC=1.0) [plan `flirds-implementation-plan.md:337`]. 그러나 matrix(full working-backdoor config, `/tmp/flirds_matrix_poison_batch8.log`)는 표준 검출 orientation(corrupt=high φ)서 **Flirds-1st AND -2nd = AUROC 0.0 = EVADED**라 보고 — 원 threat matrix 가설 CONFIRM [plan `:343`].
- **ruling (검증세션, `memory/phase2-step5-verification.md:18-27`)**: **matrix가 맞다. backdoor detection test가 오해의 소지.** backdoor detection test는 `roc_auc_score(labels, −φ)`로 per-threat 라벨지식을 써 부호를 뒤집어 1.0을 만든 것 → attacker의 φ-extremeness를 검출로 오독. 본질은 **Taylor-approximation 실패**: $pr[a]\cdot \gamma =1$이라 loss-heur와 Flirds-1st가 같은 δ₀에 작용하는데, loss-heur=exact secant는 overshoot 잡아 **1.0**, Flirds=tangent는 속아 **0.0**(loss-heur Spearman vs Flirds=+0.000). **exact in-run util/in-run oracle은 잡고, linearization만 blind.** (docs의 "γ-overshoot"는 부정확 — γ는 상쇄됨.)
- **정리**: attacker는 φ-**extreme**(two-sided/magnitude/loss-heur catchable)이지 φ-**high** 아님. 정직한 threat matrix headline = "**Flirds는 noisy+free-rider를 잡고, clean-preserving backdoor엔 EVADED; matched detector(FLDetector/FedDQC/loss-heur)가 보완**".
- **real config 재확인 완료**: silo5_poison 3-seed 영속화 run(`runs/phase2_matrix/rundirs/silo5_poison/metrics.json`, 커밋 8d364cc) — **(b)oracle·loss-heur AUROC 1.000**(잡음), **Flirds-1st 0.000**(완전 회피 — .log-only run과 공통), **Flirds(2차) 0.917±0.118 per-seed [0.75, 1.0, 1.0]**. 이전 .log-only run은 {0, 0.25, 1.0}(0.417±0.425) — 두 run 모두 실측이며 어느 쪽도 무효 아님, **run간 분산이 큼**.
- **여전히 열린 것**: 이 framing을 논문 headline으로 확정 = **Yonghee 결정** 대기.

---

## 5.2 claimed-vs-verified 교정 (문서 과장 → 검증값 채택)

검증세션(`memory/phase2-step5-verification.md`)이 잡은 과장. **충돌 시 검증값 채택, 충돌 자체 기록.**

| # | 문서 주장(claimed) | 검증값(verified) | 근거 |
|---|---|---|---|
| 1 | poison detector "FLDetector·STD-DAGMM·FedDQC 전부 1.0" | STD-DAGMM·FLTrust = **0.75** (single-shot이 mean-pooling서 희석: attacker가 10 round 중 1번만 malicious; real R≈30서 더 나쁨). FLDetector+FedDQC+loss-heur만 robust 1.0 | `verification.md:29-31` |
| 2 | "ALL methods Spearman +1.000 vs in-run oracle" | tiny-config서 **FedSV +0.900**(tiny-config noise). **단 phase1 real 3-seed는 FedSV +1.000** — matrix-orchestrator build smoke와 phase1 real을 혼동한 것 | `verification.md:32`; phase1 `metrics.json` |
| 3 | "STD-DAGMM AE-CPU ~100s" | runtime **~360s**(또는 그 이상; 한 run 526s 관측). ~85–110s는 AE 학습 컴포넌트만 | `verification.md:32` |
| 4 | backdoor detection test "evades-Flirds REFUTED" (Flirds AUROC 1.0) | 표준 orientation서 **Flirds AUROC 0.0 (EVADED)** — §5.1 | `verification.md:18-27` |

> **주의**: 위 1·3·4는 원래 tiny-config smoke(ⓐ)/1-seed 기반이었으나, real grid서 측정·영속화됨(`runs/phase2_matrix/rundirs/`, 22/25셀) — 특히 #4는 real silo5_poison 3-seed서 **Flirds(2차) 0.917±0.118**로 완전-EVADED 아님(Flirds-1st는 0.000 유지; 상세 §5.1). phase1 silo5 N5(ⓑ)는 3-seed로 단단함.

---

## 5.3 알려진 caveat (버그 아님, real-config서 확인 필요)

- **tiny-config 수치 전반**: matrix-orchestrator build 5 code paths는 val=20/R≤10 smoke — 값 coarse, 구조/orientation/gating만 검증. device100 anchor per-round in-run oracle +0.999, GTG 0.93/FedSV 0.82/ShapleyFL 0.86는 R=4 tiny.
- **selection run nuance**(직접 metrics.json 대조): seed0은 random이 우연히 clean set {2,3,4} → tie; "beats random"은 cross-seed. AUROC가 lr로 반전(lr1e-3 noisy0.75/FR1.0 ↔ lr3e-3 1.0/0.75) — MEMORY는 lr1e-3만 기재. selection은 양 lr 동일. → [02](02-experimental-setup.md#26).
- **ComFedSV** tiny-R서 Spearman 낮음(R≤8 completion-starved; cross-device port R=30서 +1.000). real은 R≈30 필요.
- **device100 corrupt-seen** R≈30 필요(짧으면 corrupt가 안 보임).
- **device100 poison ASR**: real config 실측됨(커밋 b9113c4, `rundirs/dev_a*_poison/metrics.json`) — 1B_device100-a0.0_poison ASR=[1.0, 1.0, 1.0], 1B_device100-a0.5_poison ASR=[0.675, 0.825, 0.0] (α=0.5서 cross-device 희석·seed 분산).
- **Ripple eigsh flaky**(CPU spinning stall) → 비교는 RIPPLE=0, 값은 06-06 단일세션.
- **3B retrain val-loss vs in-run oracle +0.900**(1B는 +1.000) = clean-client 1-swap=retrain noise; estimator는 +1.000 유지. 1-seed.
- **FedDQC** per-domain IRA 분산 큼(finance 0.17≈noisy 0.067) → noisy 도메인/seed 변주 필요. 1-seed smoke.
- **N=100 detector(STD-DAGMM·FLTrust)**: device100 real grid 3-seed 확보(14셀, `runs/phase2_matrix/rundirs/dev_*/metrics.json`; α=0.5 anchor {noisy,frrand,frzero} 3셀만 미실행) — 예: FLTrust FR 셀 측정 전 α(0/0.01/0.1/5.0) 1.000±0.000 / noisy α별 0.602–1.000, STD-DAGMM 셀별 분산 큼(0.508–1.000).

---

## 5.4 3-state 미실행 항목 (ⓒ)

real grid는 실행·영속화됨(상세: 07 §7.0) — `runs/phase2_matrix/rundirs/`에 **22/25 셀 `metrics.json` 영속화**(커밋 8d364cc 20셀 + b9113c4 poison 2셀; 잔여 3셀 = dev_a0.5 anchor {noisy, frrand, frzero}).

- **real grid**: silo5 N5 4-threat 3-seed **ⓑ DONE** + device100 α-sweep{0,0.01,0.1,0.5,5.0} 3-seed 14셀 DONE(dev_a0.5 anchor {noisy, frrand, frzero} 잔여) + 3B 4-threat 1-seed DONE — 전부 `rundirs/` 영속화. **7B tier만 ⓒ.**
- **N=10 retrain oracle**(LLM): 2–5일/1-GPU → multi-GPU sharding 필요. ⓒ 연기.
- **7B in-run oracle**: matrix MODEL_CFG 경로에 있으나 미실행. ⓒ
- **free-rider delta/advanced-delta(Lin Attack II/III)**: advanced free-rider. ⓒ

---

## 5.5 즉시 다음 단계 — real grid 실행 (cost-tiered stage-gate)

**전략 LOCKED 06-09 (Yonghee)** = cost-tiered stage-gate:
1. **cheap-first by cost tier**: N5(silo5, 530s) → N100(device100, 11h@1점) → 3B → 7B.
2. **category-together**: 한 (regime,threat,seed) 궤적은 비용 무관 *모든* 비교 method를 같이 — 공유 로그 위 공정비교(절대 cheap subset 먼저 아님).
3. **in-run oracle = anchor not comparand**: N5엔 530s로 비교군, N100엔 11h라 α=0.5 1점 앵커만(나머지 α는 Flirds proxy-truth).
4. **adaptive**: tier 결과가 예상 밖이면 큰 tier 계획 수정.

**실행 전 체크리스트**:
- [ ] **poison threat은 별도 invocation `LR=2e-3 BATCH=8 EPOCHS=5 POISON_FRAC=0.8`** (device100은 +`ROUNDS=60 MAX_STEPS=10`). matrix 기본 lr1e-3/batch16은 ASR=0.
- [ ] real config(full val, fp32, R 충분)로 §5.1 poison-vs-Flirds + §5.2 교정값 재측정 → threat matrix framing 확정(Yonghee).
- [ ] phase1 lr 결정(2e-5 fine-tune vs 1e-3 비교용) — config 재고([04](04-plan-vs-implementation-divergences.md)).
- [x] 3-seed N=100 detector(STD-DAGMM·FLTrust) 확보 — `runs/phase2_matrix/rundirs/dev_*` 14셀(α∈{0,0.01,0.1,5.0}×{noisy,frrand,frzero}+poison 2셀) 전부에 STD-DAGMM·FLTrust 포함. 단 dev_a0.5 anchor {noisy,frrand,frzero} 3셀은 아직 미영속화.
- [x] matrix 결과 파일 영속화 완료 — `runs/phase2_matrix/rundirs/<cell>/metrics.json` 22/25 셀 영속화(커밋 8d364cc + b9113c4).

**git 상태(운영)**: main이 origin보다 앞섬(미푸시 누적). device100-poison 해결 + per_client=300 커밋 상태는 `CLAUDE.md` 참조. **Claude는 push 불가(creds 없음) → Yonghee push.**

---

## 5.6 한 줄 결론

코드와 phase1(silo5 N5, 3-seed)은 단단하다. 방법은 retrain oracle = in-run oracle=estimator +1.000으로 검증됐다. real grid는 **22/25셀 실행·영속화 완료**(`runs/phase2_matrix/rundirs/`, 커밋 8d364cc+b9113c4; 잔여 3셀=dev_a0.5 anchor {noisy, frrand, frzero}, 7B tier는 ⓒ) — (1) 큰 N 분리력, (2) detector 경쟁, (3) poison-vs-Flirds framing 전부 real config 실측 확보, **headline framing 확정(Yonghee 결정, §5.1)만 잔여**. 추가로 06-12 Track C/D 표준-세팅 비교 실험 설계 확정([[flirds-implementation-plan]] §3.11): **Track C(C1 CNN fidelity&cost · C2 CNN 일반성능+개입 3종 · C3 cross-seed stability) 구현 완료 = ⓐ**(커밋 5b0ba71; 구현+단위검증 green, 실측 없음), **Track D(LLM 표준세팅, 전부 API-free) = ⓒ**(설계만).
