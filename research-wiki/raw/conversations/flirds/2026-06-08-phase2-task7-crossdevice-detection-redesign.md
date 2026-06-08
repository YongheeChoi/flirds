---
type: conversation
date: 2026-06-08
topic: flirds
participants: [Yonghee, Claude]
tags: [phase2, task7, cross-device, detection-baselines, threat-matching, std-dagmm, comfedsv, in-run-oracle, backdoor, fltrust, feddqc]
---

# Phase 2 task 7 — cross-device (7a–7d) + detection-baseline threat-matching redesign

## Code completed this session (committed)

- **7a cross-device data loader** — DESIGN FORK resolved. The existing `fl.partition.dirichlet_partition`
  (per-CLASS-over-clients, Hsu 2019) is **degenerate** for 5 domains → 100 clients: α=0 → only 5
  non-empty clients; α=0.01 → 44; client sizes swing 0–12k → size confounds the α-sweep. Chose
  **Option B = per-CLIENT Dirichlet domain-mixture** (LDA-style): new
  `fl.partition.client_dirichlet_partition(labels, n, alpha, per_client, seed)` — each client draws a
  Dir(α) mixture over the 5 domains, then `per_client` disjoint records via `rng.multinomial` → ALL N
  non-empty, EXACT fixed size (B1 size-control preserved), α=0 = domain-disjoint (~N/5 clients/domain),
  purity identical to Option A at matched α (1.0/0.96/0.77/0.52/0.31). `data.llm.build_crossdevice(
  n_clients, alpha, per_client_train, per_domain_pool=12000, per_domain_val, per_domain_test, seed, noisy)`.
  Verified: `phase2_crossdevice_data_smoke.py` (synthetic table reproduces design + HF wiring); CNN guard green.
- **7b N=100 Flirds** — verification only, NO library change. The estimator is already
  partial-participation-correct (per-round participant weights). `run_llm_fedavg_logs(sample_frac=0.1)` →
  K=10/round; `flirds_values(n_clients=100)` (must pass N explicitly — unselected clients never appear in
  the logs → φ=0). Verified: exactly K=10 participants/round, free-rider(zero) φ=**exactly 0** even when
  selected, estimator 49 s. `phase2_crossdevice_flirds_smoke.py`.
- **7c (b) in-run oracle = EXACT per-round decomposition (NOT MC)** — KEY INSIGHT. U_(b) is additive over
  the FROZEN rounds and round r's term depends only on its K participants (p_k^r fixed over P_r), so by
  Shapley linearity + the null-player property: **φ_i = Σ_{r: i∈P_r} (exact 2^{|P_r|} Shapley of round r)**.
  Identical to the 2^N oracle (proven: synthetic max|Δφ| ≈ 3e-16, full + partial participation) but feasible
  at N=100 (Σ_r 2^K = 200·1024, not 2^100). New `oracle.in_run_sv.in_run_shapley_perround`. Real N=100
  (α=0.5, 1B): **Flirds vs (b) Spearman +1.0000** over the 71 selected (game stays near-additive at scale).
  Oracle cost MEASURED **771 ms/fwd** (fp32-on-B200 = no tensor-cores) → R=200,K=10 ≈ 44 h/1-GPU, ~11 h/4-GPU
  sharded (rounds independent). Since Flirds = (b) +1.000, run exact (b) at 1–2 α points (validation),
  Flirds-only for the sweep. `phase2_crossdevice_oracle_smoke.py`.
  - Yonghee's framing: **(a) retrain oracle → permutation MC** (retraining all coalitions IS expensive);
    **(b) in-run oracle → exact (no MC), since it's cheap**. Refined: exact-per-round is BOTH cheaper than
    full-game permutation MC (2000·M forwards) AND exact.
- **7d ComFedSV LLM port** — `comfedsv.comfedsv_from_logs(..., loss_fn=, pkeys=)` branch (GTG/FedSV pattern):
  uniform-subset PARAMS + standard `(w_r, deltas_map)` logs (cohort = deltas_map.keys()), partial=True.
  New `_uniform_subset_params` / `_llm_util`. Verified: partial=False (MC M=8000) == exact uniform-Shapley
  (Spearman +1.0000, max|Δ| 5e-3); CNN path **bit-identical** (git-stash comparison: original == modified).
  `phase2_crossdevice_comfedsv_smoke.py`. (Observed, pre-existing, NOT from this change: CNN ComFedSV verify
  soft-"CHECK" at seed=0, 0.33 CPU / 0.67 GPU vs recorded {1.0,0.96,0.85,0.84} — seed/device sensitivity.)
- **3B (a)-valloss fp32 confirm** (background from the prior session, finished): N=5, lr3e-3, 9483 s ≈ 2.6 h →
  (a)valloss vs (b) = +0.900 (one clean-client swap = retrain noise), estimator = +1.000, AUROC
  (a)valloss/(b)/estimator all noisy0.75/FR1.0 identical, (a)ROUGE = +0.100 (fooled). The 1B validation
  holds at 3B. Recorded in plan task-6.

## MAJOR DECISION — detection-baseline redesign (threat-matched)

Yonghee's probing exposed that the old detector plan conflated threat categories. Resolved:

### Epistemics — why detection baselines at all, when we inject the labels
The method (Flirds / detector) is BLIND to the labels; we use the known labels only as the **evaluation key**
(AUROC) — standard supervised evaluation of an unsupervised separator, NOT circular. Two purposes: (1)
**semantic validation** of the valuation — corrupt → low value confirms the value MEANS data quality (the
(a)/(b) oracle's Spearman only proves "Flirds matches the Shapley COMPUTATION", not that the value means
good/bad data); (2) **competitive bar** — Flirds matches/beats dedicated detectors WITHOUT detection
machinery. Caveat: we choose the corruptions → use literature-grounded ones + sweep types/severities.

### Threat taxonomy + matched detectors (the redesign)
| threat | our corruptor | matched detector |
|---|---|---|
| data-quality (honest client, bad data) | `answer_swap` | **FedDQC** (LLM-FL, IRA per-sample quality) — NEW |
| free-rider (fabricated update) | `free_rider` zero/random | **STD-DAGMM** (+ FLTrust any-N) |
| poisoning / backdoor (crafted attack) | **NEW: Xu+Bagdasaryan** | **FLDetector / FLTrust** |

- **The old FLDetector↔noisy pairing was a THREAT MISMATCH.** FLDetector (Zhang 2022) detects
  **crafted-update attackers** (Fang untargeted + Scaling/DBA/ALIE backdoors); its own paper / our
  `sources/fldetector` line 48: "detects crafted-update attackers, **not** noisy-but-honest clients."
  `answer_swap` = honest gradient on quality-corrupted data, temporally CONSISTENT → FLDetector's deviation
  signal does not fire → the ~0.50 AUROC is **off-threat**, not merely non-IID erosion. STD-DAGMM also can't
  catch noisy (an `answer_swap` update has normal std). → **noisy had NO matched detector** → add **FedDQC**.
- **No LLM-FL-validated client-level noisy/free-rider detector EXISTS** (wiki: STD-DAGMM "first PEFT-scale
  test"; FedDQC is per-SAMPLE quality not a client-anomaly detector; FLDetector / STD-DAGMM / FLTrust all
  CV/small-net). → baselines are necessarily CV ports → Flirds' LLM-scale separation is novel territory.
  (+ `sources/do-influence-functions-work-on-llms`: IF brittle on LLMs ["param change ≠ behavior change"];
  Flirds' in-run framing sidesteps it — the Spearman +1.000 vs the in-run oracle is the evidence.)

### Regime — BOTH detectors run in BOTH regimes (Yonghee: detection is needed in both)
- **FLDetector → both** (model-free, any-N; cross-device needs partial-participation per-client-history adaptation).
- **STD-DAGMM → both** via per-(client,round) **pooling** (the "degenerate at N=5" was per-round/per-client-mean;
  pooling gives 5·R vectors at cross-silo N=5 → trainable). Caveat: at small N the AE is weak
  (vectors cluster by client → ~N effective clusters; the free-rider's many rounds make ~20% of samples
  anomalous, violating DAGMM's rare-anomaly assumption) → the **std-augmentation carries** detection there.
  Honest small-N limitation.
- **FLTrust** added = any-N free-rider detector (per-client ReLU-cosine-to-root + magnitude-norm; our val set
  = the root). Sidesteps STD-DAGMM's N≫ entirely. BUT its cosine signal ≈ Flirds' 1st-order term ⟨∇ℓ,Δw⟩
  (so "Flirds beats FLTrust" is near-tautological; it shares Flirds' delta-weakness) → STD-DAGMM remains the
  mechanistically-INDEPENDENT free-rider baseline. FLTrust doubles as a poisoning baseline.

### STD-DAGMM implementation decisions ①②③
- **① sample construction = per-(client,round) pooled** (FORCED by the both-regime requirement; per-client-mean
  gives only N samples → degenerate at cross-silo N=5). Client score = aggregate (mean) of its rounds' energies.
- **② dim reduction = feature-hashing random projection → ~256** (LoRA delta ~5.6M is too high-dim for an AE:
  a 5.6M-input first layer overfits on 100–2000 samples; a dense 256×5.6M JL matrix is 5.7 GB → use
  feature-hashing, no stored matrix). The **std-augmentation is computed on the FULL 5.6M vector** (one flat
  std), reduction-independent → preserved regardless of ②.
- **③ free-rider mode = random @ benign-std-tuned + zero floor.** zero = trivial floor (std=0; Flirds φ=exact 0).
  random tests STD-DAGMM's AE/cosine ONLY if its scale ≈ benign-update std (Lin's adversarial tuning; our
  fixed `scale=1e-3` may be too easy → tune to measured benign std). Flirds handles random regardless (φ≈0).
  **delta / advanced-delta DEFERRED to task 9** (recycled aggregate ± noise; needs the round-aggregate threaded
  into the FL loop) — this is the RICHEST comparison and where Flirds is genuinely AT RISK (a recycled-aggregate
  delta is aligned with ∇ℓ_val → Flirds' 1st-order term ≠ 0 → may be fooled).

### Poisoning / backdoor corruptor — NEW threat, definition LOCKED (Yonghee #1)
- = **Xu et al. 2023 "Instructions as Backdoors"** (arXiv 2305.14710): instruction-trigger → target output
  (LLM-instruction-tuning-native; 1% poison → ASR > 90%, persistent, transfers) — the **text trigger / data
  poison** — **+ Bagdasaryan et al. 2020 "How to Backdoor FL"** (arXiv 1807.00459): plain-**scaled**
  model-replacement update (the crafted update FLDetector/FLTrust are validated against — both test the
  Scaling backdoor). **DBA (Xie et al. 2020, ICLR) EXCLUDED** — needs colluding multi-attacker ≠ our
  single-corrupt-client model. Use **plain scaled**, not constrain-and-scale stealthy (the stealthy variant
  evades all update-detectors AND Flirds; a separate hard regime, optional later).
- The detectors see the UPDATE (modality-agnostic) → text-trigger (Xu) vs CV-trigger doesn't matter; the
  Bagdasaryan scaling gives the crafted/anomalous update = matched to FLDetector/FLTrust's validated regime.
- Measure: **ASR** (trigger inputs, Xu's metric) + **detection AUROC**.
- **Backdoor-vs-Flirds framing is OPEN — verify experimentally (Yonghee #2: do NOT pre-position).** Hypothesis
  (NOT a claim): a clean-performance-preserving backdoor preserves clean val-loss → may EVADE Flirds (its
  characterized boundary) → a complementary update-detector is needed. **Run the experiment before asserting.**

## Sequencing (next session)
1. **STD-DAGMM** (free-rider; ①②③ set) → new `baselines/std_dagmm.py`
2. **FLTrust** (any-N free-rider + poisoning) → new baseline (val=root)
3. **poisoning/backdoor corruptor** (Xu+Bagdasaryan scaled) + **FLDetector re-point** to poisoning + cross-device
   partial-participation adaptation. Ingest Xu (2305.14710) + Bagdasaryan (1807.00459) wiki source pages here.
4. **FedDQC** (noisy matched; IRA per-sample quality → client-level) → new baseline
5. **expanded experiment matrix**: 3 threats × 2 regimes × {matched detector + Flirds + valuation methods} × α-sweep × seeds.

Scope note: this expands the old task 7e (STD-DAGMM only) into a detection-baseline SUITE + pulls task 9
(corruptor extensions) and part of the Phase-3 threat matrix forward. Yonghee accepted the scope for rigor.
Verification pattern unchanged: synthetic/unit → small N=100 real-1B smoke → experiment; keep the CNN
bit-identical guard (`phase1_baseline_smoke.py cnn`).
