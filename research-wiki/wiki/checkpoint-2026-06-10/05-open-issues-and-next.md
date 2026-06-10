---
type: checkpoint
title: "Flirds 체크포인트 05 — 미해결 + 다음 단계"
created: 2026-06-10
updated: 2026-06-10
---

# 05 · 미해결 이슈 + 다음 단계

> 판단 대기 / 알려진 caveat / 즉시 다음 단계. 근거: 코드·raw 로그·`memory/phase2-step5-verification.md`.

---

## 5.1 판단 대기 — poison vs Flirds framing (verification-item #1)

**기술적 사실은 ruling 났으나, 논문 headline framing + real-config 재확인은 미결.**

- **충돌**: D2b smoke(`phase2_backdoor_d2b_smoke.py`)는 "clean-preserving backdoor가 Flirds를 evade한다 = **REFUTED**"라 결론(Flirds-1st/2nd 전부 AUROC=1.0) [plan `flirds-implementation-plan.md:337`]. 그러나 matrix(full D2b config, `/tmp/flirds_matrix_poison_batch8.log`)는 표준 검출 orientation(corrupt=high φ)서 **Flirds-1st AND -2nd = AUROC 0.0 = EVADED**라 보고 — 원 §3.9 가설 CONFIRM [plan `:343`].
- **ruling (검증세션, `memory/phase2-step5-verification.md:18-27`)**: **matrix가 맞다. D2b가 오해의 소지.** D2b는 `roc_auc_score(labels, −φ)`로 per-threat 라벨지식을 써 부호를 뒤집어 1.0을 만든 것 → attacker의 φ-extremeness를 검출로 오독. 본질은 **Taylor-approximation 실패**: `pr[a]·γ=1`이라 loss-heur와 Flirds-1st가 같은 δ₀에 작용하는데, loss-heur=exact secant는 overshoot 잡아 **1.0**, Flirds=tangent는 속아 **0.0**(loss-heur Spearman vs Flirds=+0.000). **exact in-run util/(b) oracle은 잡고, linearization만 blind.** (docs의 "γ-overshoot"는 부정확 — γ는 상쇄됨.)
- **정리**: attacker는 φ-**extreme**(two-sided/magnitude/loss-heur catchable)이지 φ-**high** 아님. 정직한 §3.9 headline = "**Flirds는 noisy+free-rider를 잡고, clean-preserving backdoor엔 EVADED; matched detector(FLDetector/FedDQC/loss-heur)가 보완**".
- **여전히 열린 것**: ① 이 framing을 논문 headline으로 확정 = **Yonghee 결정** ② **real config(full val, (b) oracle 동반)에서 재확인** 필요(현재는 tiny val=20 smoke) ③ (b) oracle이 잡는다는 건 smoke 논증 — real서 (b)/loss-heur AUROC 직접 측정 필요.

---

## 5.2 claimed-vs-verified 교정 (문서 과장 → 검증값 채택)

검증세션(`memory/phase2-step5-verification.md`)이 잡은 과장. **충돌 시 검증값 채택, 충돌 자체 기록.**

| # | 문서 주장(claimed) | 검증값(verified) | 근거 |
|---|---|---|---|
| 1 | poison detector "FLDetector·STD-DAGMM·FedDQC 전부 1.0" | STD-DAGMM·FLTrust = **0.75** (single-shot이 mean-pooling서 희석: attacker가 10 round 중 1번만 malicious; real R≈30서 더 나쁨). FLDetector+FedDQC+loss-heur만 robust 1.0 | `verification.md:29-31` |
| 2 | "ALL methods Spearman +1.000 vs (b)" | tiny-config서 **FedSV +0.900**(tiny-config noise). **단 phase1 real 3-seed는 FedSV +1.000** — step5 smoke와 phase1 real을 혼동한 것 | `verification.md:32`; phase1 `metrics.json` |
| 3 | "STD-DAGMM AE-CPU ~100s" | runtime **~360s**(또는 그 이상; 한 run 526s 관측). ~85–110s는 AE 학습 컴포넌트만 | `verification.md:32` |
| 4 | D2b "evades-Flirds REFUTED" (Flirds AUROC 1.0) | 표준 orientation서 **Flirds AUROC 0.0 (EVADED)** — §5.1 | `verification.md:18-27` |

> **주의**: 위 1·3·4는 전부 **tiny-config smoke(ⓐ) 또는 1-seed** 기반 — real grid서 다시 측정해야 확정. phase1 silo5 N5(ⓑ)만 3-seed로 단단함.

---

## 5.3 알려진 caveat (버그 아님, real-config서 확인 필요)

- **tiny-config 수치 전반**: step5 5 code paths는 val=20/R≤10 smoke — 값 coarse, 구조/orientation/gating만 검증. device100 anchor (b)-perround +0.999, GTG 0.93/FedSV 0.82/ShapleyFL 0.86는 R=4 tiny.
- **#7 selection nuance**(직접 metrics.json 대조): seed0은 random이 우연히 clean set {2,3,4} → tie; "beats random"은 cross-seed. AUROC가 lr로 반전(lr1e-3 noisy0.75/FR1.0 ↔ lr3e-3 1.0/0.75) — MEMORY는 lr1e-3만 기재. selection은 양 lr 동일. → [02](02-experimental-setup.md#26).
- **ComFedSV** tiny-R서 Spearman 낮음(R≤8 completion-starved; task7d R=30서 +1.000). real은 R≈30 필요.
- **device100 corrupt-seen** R≈30 필요(짧으면 corrupt가 안 보임).
- **device100 poison ASR 0.75**(silo5 1.0보다 낮음=cross-device 희석), tiny val=4 — real config 확인.
- **Ripple eigsh flaky**(CPU spinning stall) → 비교는 RIPPLE=0, 값은 06-06 단일세션.
- **3B (a)valloss vs (b) +0.900**(1B는 +1.000) = clean-client 1-swap=retrain noise; estimator는 +1.000 유지. 1-seed.
- **FedDQC** per-domain IRA 분산 큼(finance 0.17≈noisy 0.067) → noisy 도메인/seed 변주 필요. 1-seed smoke.
- **N=100 detector 결과 전부 1-seed**(STD-DAGMM 0.628, FLTrust 1.0) — 3-seed 미확보.

---

## 5.4 3-state 미실행 항목 (ⓒ)

`runs/phase2_matrix/`는 **빈 폴더(0 files)** — real grid 한 번도 안 돌았다.

- **real grid 전체**: silo5 N5(4 threat × 3 seed, all method + (b) Spearman) → device100 α-sweep{0,0.01,0.1,0.5,5.0}((b)@α0.5만) → 3B → 7B. ⓒ
- **N=10 (a)-oracle**(LLM): 2–5일/1-GPU → multi-GPU sharding 필요. ⓒ 연기.
- **7B (b)-oracle**: matrix MODEL_CFG 경로에 있으나 미실행. ⓒ
- **free-rider delta/advanced-delta(Lin Attack II/III)**: task9. ⓒ

---

## 5.5 즉시 다음 단계 — real grid 실행 (cost-tiered stage-gate)

**전략 LOCKED 06-09 (Yonghee)** = cost-tiered stage-gate:
1. **cheap-first by cost tier**: N5(silo5, 530s) → N100(device100, 11h@1점) → 3B → 7B.
2. **category-together**: 한 (regime,threat,seed) 궤적은 비용 무관 *모든* 비교 method를 같이 — 공유 로그 위 공정비교(절대 cheap subset 먼저 아님).
3. **(b) oracle = anchor not comparand**: N5엔 530s로 비교군, N100엔 11h라 α=0.5 1점 앵커만(나머지 α는 Flirds proxy-truth).
4. **adaptive**: tier 결과가 예상 밖이면 큰 tier 계획 수정.

**실행 전 체크리스트**:
- [ ] **poison threat은 별도 invocation `LR=2e-3 BATCH=8 EPOCHS=5 POISON_FRAC=0.8`** (device100은 +`ROUNDS=60 MAX_STEPS=10`). matrix 기본 lr1e-3/batch16은 ASR=0.
- [ ] real config(full val, fp32, R 충분)로 §5.1 poison-vs-Flirds + §5.2 교정값 재측정 → §3.9 framing 확정(Yonghee).
- [ ] phase1 lr 결정(2e-5 fine-tune vs 1e-3 비교용) — config 재고([04](04-plan-vs-implementation-divergences.md)).
- [ ] 3-seed로 N=100 detector(STD-DAGMM/FLTrust) 재확인.
- [ ] matrix는 stdout만 출력 → **결과 파일 영속화 경로 추가**(현재 `runs/phase2_matrix/` 빈 폴더, `phase2_matrix.py:122` `/tmp`만).

**git 상태(운영)**: main이 origin보다 앞섬(미푸시 누적). device100-poison 해결 + per_client=300 커밋 상태는 `CLAUDE.md` 참조. **Claude는 push 불가(creds 없음) → Yonghee push.**

---

## 5.6 한 줄 결론

코드와 phase1(silo5 N5, 3-seed)은 단단하다. 방법은 (a)=(b)=estimator +1.000으로 검증됐다. **남은 건 real grid를 돌려 (1) 큰 N 분리력, (2) detector 경쟁, (3) poison-vs-Flirds framing을 real config에서 확정하는 것** — 전부 ⓒ(미실행), 설계는 LOCKED.
