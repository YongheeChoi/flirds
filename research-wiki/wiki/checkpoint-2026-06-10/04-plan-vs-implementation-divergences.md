---
type: checkpoint
title: "Flirds 체크포인트 04 — plan 대비 구현 분기"
created: 2026-06-10
updated: 2026-06-10
---

# 04 · plan 대비 달라진 점 (+ 왜)

> 처음 plan과 실제 구현이 갈라진 지점 전부 + 각 근거(raw 대화/코드). 시간순.
> 분기는 *실패*가 아니라 대부분 Yonghee의 설계결정(explanation→decision→execution).

| # | 분기 | 날짜 | 근거(raw) |
|---|---|---|---|
| 1 | curvature GGN/Fisher → **true Hessian** | 06-03 | `2026-06-03-phase05-estimator.md` |
| 2 | momentum=0.9 → **plain SGD mom=0** (전 FL run) | 06-03 | 〃 |
| 3 | dataset PubMedQA/CaseHOLD → **MedAlpaca/QA-Legal**(free-form) | 06-04 | `2026-06-04-phase1-data-layer.md` |
| 4 | (a)-oracle 주 metric ROUGE → **val-loss** | 06-07 | `2026-06-07-phase2-task6-a-retrain-oracle.md` |
| 5 | oracle 정밀 bf16 → **fp32** | 06-07 | 〃 |
| 6 | detector **threat-matched 재설계** (+FedDQC, +poison 위협) | 06-08 | `2026-06-08-phase2-task7-crossdevice-detection-redesign.md` |
| 7 | partition per-class Dir → **per-client Dir(Option B)** | 06-08 | 〃 |
| 8 | Ripple를 비교서 분리 (**RIPPLE=0**, eigsh flaky) | 06-06~07 | `2026-06-06-sv-baseline-port-and-results.md` |
| 9 | N=10 (a)-oracle **연기** (2–5일) | 06-07 | `2026-06-07-phase2-task6-...md` |
| 10 | device100 **per_client 40 → 300** (poison install) | 06-09 | `2026-06-09-phase2-step5-matrix-orchestrator-task8.md` |
| 11 | matrix = **신규 파일** `phase2_matrix.py`(comparator 확장 아님) | 06-09 | 〃 |

---

### 1. curvature: GGN/Fisher → true Hessian (06-03)
- **원**: IRDS 프레이밍 따라 Gauss-Newton/Fisher(PSD) 곡률.
- **변경/왜**: CNN Phase0.5서 GGN을 테스트 → 진짜(indefinite) Hessian보다 일반적으로 **나쁨**(relL2 8/9 config서 열등). Yonghee 결정. [`2026-06-03-phase05-estimator.md`; `flirds.md:34`]. 지금 `jvp∘grad`=true HVP.

### 2. momentum=0.9 → plain SGD mom=0 (06-03)
- **원**: `local_train` 기본 momentum=0.9.
- **변경/왜**: code-review가 Ripple drop항·IRDS가 plain SGD 가정임을 지적. 즉시 보상: plain SGD서 3-seed Spearman가 "1st 0.81 > 1st+2nd 0.73(2차 해로움)" → "1st+2nd 0.96 > 1st 0.92(2차 도움)"로 **역전**. 이제 보편 규약 [`codes/CLAUDE.md §5`; `llm_server.py:52-53`]. → 2차항 노벨티의 실증 기반([01](01-research-value.md)).

### 3. dataset 교체 → free-form (06-04)
- **원**(D3, 06-02 lock): PubMedQA + CaseHOLD + FiQA + AQUA + Dolly.
- **변경/왜**: PubMedQA/CaseHOLD는 classification 포맷(closed/yes-no) → "free-form instruction→response 균일성" 결정과 충돌(이질 포맷이면 shared val-loss Shapley 불공정). medical→`medical_meadow_flashcards`, legal→`ibunescu/qa_legal`. FedDQC 도메인 겹침 3/5→2/5(허용; FedDQC 비교는 IRA-baseline만). [`2026-06-04-phase1-data-layer.md`].

### 4. (a)-oracle 주 metric: ROUGE → val-loss (06-07) — **방법론적으로 가장 중요**
- **원**: (a) 재학습 oracle을 ROUGE-L(CNN test-acc 유추)로 프레이밍.
- **변경/왜**: Yonghee 교정 — 우리 method(estimator/(b))가 Shapley를 *옳게 계산*하는지 검증하려면 (a)가 **같은 게임(val-loss)** 풀어야. ROUGE는 다른 게임(불일치가 metric 탓일 수 있음). estimator는 val-loss의 Taylor → apples-to-apples 필수. ROUGE는 미분불가→estimator-ROUGE 불가능, (b)-ROUGE도 infeasible. 실측이 정당화: (a)valloss=(b)=estimator +1.000 vs (a)ROUGE 발산(1B +0.4, 3B −0.9, answer_swap의 도메인-포맷 학습에 속음). [`2026-06-07-phase2-task6-...md`].

### 5. oracle 정밀: bf16 → fp32 (06-07)
- **변경/왜**: bf16 절대정밀도 ~0.009 > coalition val-loss 차 ~0.005–0.02 → bf16서 Spearman 무의미. fp32서 양 lr +1.000. [`2026-06-07-phase2-task6-...md`; `oracle/exact_sv_llm.py:62-63`]. "fp32 master"는 비협상 규약.

### 6. detector threat-matched 재설계 (06-08) — **검출 baseline 전면 개편**
- **원**(06-07 lock): FLDetector→cross-silo noisy / STD-DAGMM→cross-device free-rider (regime로 분리).
- **변경/왜**: Yonghee 탐문이 **위협 불일치** 발견 — FLDetector는 crafted-update 검출기지 정직한 answer_swap 검출기 아님 → 0.50 noisy AUROC는 off-threat(non-IID erosion 아님). answer_swap엔 매칭 detector 없음 → **FedDQC 신규 추가**. 재설계: data-quality→FedDQC / free-rider→STD-DAGMM+FLTrust / poisoning→FLDetector+FLTrust, **양 detector 양 regime**. 동시에 **poison/backdoor 신규 위협**(Xu+Bagdasaryan) 도입 — 원 Phase2 plan에 없던 것. [`2026-06-08-...redesign.md:64-76`].

### 7. partition: per-class → per-client Dirichlet (06-08)
- **원**: 기존 `dirichlet_partition`(per-class-over-clients, Hsu 2019).
- **변경/왜**: 5 도메인→100 client서 degenerate(α=0→5 non-empty만; α=0.01→44; 크기 0–12k). 신규 `client_dirichlet_partition`(Option B: per-client Dir(α) 도메인혼합) = 전부 non-empty·고정크기(B1 size-control 보존)·α=0=도메인 disjoint. [`2026-06-08-...redesign.md:12-23`; `partition.py:49-82`].

### 8. Ripple 비교서 분리 (RIPPLE=0) (06-06~07)
- **원**: Ripple를 모든 비교표의 head-to-head baseline.
- **변경/왜**: eigsh CPU-spinning stall(알려진 수렴 flaky) → 자동 배치서 불안정. `phase1_baseline_compare.py`는 RIPPLE=0; Ripple 값은 06-06 단일 세션 것 사용. [`2026-06-06-...md`; `2026-06-07-phase2-banzhaf-...md`]. → Ripple 코드 자체는 견고화([03](03-baselines-and-prior-work.md#a3)).

### 9. N=10 (a)-oracle 연기 (06-07)
- **원**(매트릭스 06-04 lock): LLM 1B (a)-oracle N=5 **and** N=10.
- **변경/왜**: task6 비용 추정 — N=10 재학습 64×(ΣC(10,s)·s=5120 vs 80) + eval 32× → 2–5일/1-GPU. multi-GPU coalition sharding 미구축. N=5 fp32 검증(+1.000)으로 method 검증 충분. [`2026-06-07-phase2-task6-...md:85-94`].

### 10. device100 per_client 40 → 300 (06-09)
- **원**: cross-device `per_client=40`(free-rider/noisy와 일관).
- **변경/왜**: backdoor install엔 attacker당 ~200 poisoned 필요(D1). per_client=40·frac0.5→75 poisoned<threshold→ASR=0(전파/코드 버그 아님; silo5 동일코드=ASR 1.0). 4-GPU sweep(single-shot R10/30/60, multi-round γ4/10, multi-attacker 5%/10% **전부 ASR=0** — attacker 늘려도 install은 client별 local이라 무용) + A′(per_client=300/frac0.8→240 poisoned/EPOCHS5/R60→**ASR=0.75**). noisy/free-rider는 size-무관 → 300이 regime 통일. 탐색코드(poison_multiround 등) revert. [`2026-06-09-...task8.md:135-152`; `phase2_matrix.py:105-109`].

### 11. matrix = 신규 파일 (06-09)
- **원**: `phase1_baseline_compare.py` 확장 가능성.
- **변경/왜**: Yonghee fork 결정 — 검증된 N=5 +1.000 comparator 보존 위해 **신규 `phase2_matrix.py`**(bit-identical call pattern만 재사용). 유일 코드변경 → baselines 무변경 → **CNN bit-identical guard green**. [`phase2_matrix.py`; `2026-06-09-...task8.md`].

> **추가 관찰(plan vs code, 분기는 아니나 주의)**:
> - phase1 lr=2e-5(§3.3) vs matrix smoke lr=1e-3/2e-3 — matrix는 비교실험용 빠른수렴 config, final 아님. real grid config 재고 필요.
> - plan §3.5 cross-device R=200 vs matrix smoke R=30 — smoke 기본 coarse, real은 env override.
> - seed `[0,1,2]`(코드) vs protocol `[42,123,2024]` — 코드가 실제값.
