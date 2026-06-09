---
type: conversation
date: 2026-06-09
topic: flirds
participants: [Yonghee, Claude]
tags: [phase-2, step-5, matrix-orchestrator, task-8, detection, valuation, cross-device, poisoning, scale]
---

# Phase 2 step 5 — detection/valuation MATRIX orchestrator + task 8 (3B/7B) scale

Second 2026-06-09 session (after task 7e backdoor-install / FedDQC). Goal: **build the step-5
implementation = the matrix orchestrator + task 8 scale**, validate every code path on tiny-config
smokes, then defer the real grid execution to a verified run. The step-5 grid + cost-tiered
execution strategy were already LOCKED (06-09) → no re-litigation, this is the build.

## Design forks (Yonghee decided)
1. **Orchestrator file → NEW `experiments/phase2_matrix.py`** (not "extend phase1_baseline_compare"
   as the plan phrased it). Rationale Claude surfaced + Yonghee chose: phase1 is the validated N=5
   3-seed comparator (the headline "all methods Spearman +1.000" artifact); the matrix's
   regime-gating + per-threat trajectory builders + 3 new detectors + D2b poison synthesis +
   proxy-truth logic are a genuinely new structure → a self-contained file preserves the validated
   artifact and reuses only the (bit-identical) per-method **call pattern**.
2. **Free-rider threat mode → random@benign-std (headline) + zero (trivial floor)**, both — matches
   the memory's "free-rider regime = random@benign-std + zero floor"; random is the STD-DAGMM evasion
   case, zero is the φ-exact-0 story.

## `phase2_matrix.py` — architecture
Env-parameterized single script (the phase1 `RUN_SEED` idiom; one invocation per cell so seeds/regimes
shard across GPUs 0–3). `env`: `REGIME` {silo5,device100}, `ALPHA`, `THREAT` {noisy, freerider_random,
freerider_zero, poison, or all-4}, `SEED`, `SMOKE_MODEL`, `ORACLE_B`/`COALITION` gates, per-scale
`BATCH`/`VAL_CHUNK`/`VAL_MAXLEN`, `POISON_TRAIN`.

- **Threats** (each its own trajectory; "category-together" = every method runs on every threat → the
  on-/off-threat matrix): `noisy` = answer_swap corrupt client(s), normal FL; `freerider_*` =
  fabricated update (random@benign-std after a short clean warmup to measure the std, or zero), normal
  FL; `poison` = the D2b synthesis — benign FL → attacker trains backdoor X from the deployed G
  (frac=0.5, single-token target, install lr) → a single-shot model-replacement attack round
  `{attacker: γ(X−G), benign: fresh G-deltas}` is appended, γ = cohort size (silo5 N=5, device100 K=10).
- **Regimes**: `silo5` (N=5, full participation, (b)=exact `in_run_shapley` 2^N, all coalition
  baselines run) / `device100` (N=100, K=10, Dirichlet(α) domain mixtures; the 2^N methods — Banzhaf +
  exact-(b) — drop out → (b)=`in_run_shapley_perround`, ComFedSV is the partial-participation Shapley
  baseline; GTG/FedSV/ShapleyFL/(b)-perround are ~2^K/round = the oracle cost so they gate behind
  COALITION/ORACLE_B = the α=0.5 **anchor** only; the cheap methods Flirds/Flirds1st/loss-heur/ComFedSV
  + 4 detectors run every α with **Flirds as the proxy-truth** for Spearman off-anchor).
- **Detectors** (all on every threat): FLDetector / STD-DAGMM (model-free, logs), FLTrust (val-grad
  cosine), FedDQC (on-device data quality — scored with the base+init-LoRA model, smoke-matched).
- **Scale (task 8)**: `SMOKE_MODEL` → 1B/3B/7B; `MODEL_CFG` per-scale batch/val_chunk/val_maxlen are
  **memory-only** (exact chunk-sum, fp32 throughout). 7B = fp32 + small batch, **no bf16** — bf16 is the
  deferred (a) retrain oracle, which 7B does not run (grid = "7B (b) N5").
- **Outputs**: per-(threat,seed) AUROC (over the seen/scoreable clients) + Spearman vs (b)/Flirds-proxy
  + runtime + corrupt-rank; seed mean±std.

The only code change is the new file — `git status` = `?? codes/experiments/phase2_matrix.py` (no
baseline edits → **CNN bit-identical guard GREEN**, re-run confirmed).

## Validation — all 5 code paths green
Tiny-config smokes (TRAIN=8 VAL=4 ROUNDS=2 etc.) — the **values are coarse** but the structure,
orientation, gating dispatch, and Spearman/AUROC plumbing are exercised end-to-end.

| path | Spearman vs truth | detector (on-threat / off-threat) | what it proves |
|---|---|---|---|
| **silo5 noisy** | all methods **+1.000** vs (b) | FedDQC/FLTrust flag / FLDetector·STD-DAGMM 0.5 | full silo5 method set + (b) exact |
| **silo5 freerider** (random+zero) | all methods **+1.000** vs (b) | FLTrust **1.0**, STD-DAGMM(random 1.0) / FLDetector 0.5 | warmup→benign-std path + both modes |
| **silo5 poison** (D2b) | all methods **+1.000** vs (b) | **FLDetector·STD-DAGMM·FedDQC 1.0** vs valuation 0.0 | D2b synthesis + the §3.9 framing |
| **device100 cheap** | Flirds proxy-truth; loss-heur +0.999 | FLTrust **1.0** (free-rider top), STD-DAGMM 0.54, FedDQC 0.10 (off-threat clean-data) | crossdevice build + partial participation + proxy-truth + ComFedSV |
| **device100 anchor** | vs **(b)-perround**: Flirds **+0.999**, GTG +0.93, FedSV +0.82, ShapleyFL +0.86, ComFedSV +0.26 | — | perround dispatch + device coalition (timing: (b) 613s / GTG 419s / ShapleyFL 606s @ R=4) |
| **3B scale** | (b)/Flirds **+1.000** | FLTrust/FedDQC 1.0 | MODEL_CFG["3B"] batch=8/val_chunk=5, no OOM |

The poison smoke reproduces the **§3.9 framing exactly**: the magnitude/data detectors flag the
γ-scaled (≈40×-norm) attacker (AUROC 1.0) while the valuation methods see it as helpful (low φ) when
the backdoor does not degrade clean val-loss — i.e. the D2b "clean-preserving" boundary, decided by
the matrix not pre-positioned.

**real-config re-verify notes (NOT code bugs, tiny-config artifacts):** (1) ComFedSV Spearman is low
at R≤8 = low-rank completion starved of observations (task 7d validated it = exact uniform-Shapley
+1.000 at the proper R=30); at device100 with equal-size clients uniform == n_k-weighted so it should
track Flirds at the real config. (2) STD-DAGMM AE trains on CPU ~85–110 s — larger at real R (does not
block the GPU). (3) device100 corrupt-seen is P≈0.96 at the default R=30 (the smokes' R=4 made the
anchor's noisy AUROC undefined — handled gracefully with a printed warning + NaN, Spearman still
computed). (4) tiny-config deployed-ASR=0 (no backdoor install) — the poison-detection structure is
still exercised; install needs the real train size (below).

## poison propagation — config-sensitive (silo5); detection robust regardless
Reproducing a **propagating** backdoor (ASR>0) in the matrix requires matching D2b's EXACT working
config — **lr=2e-3 / R=10 / BENIGN_STEPS=5 / train=1000 / batch=8 / frac=0.5 / γ=n/η=5** (→ D2b ASR 0.97).
The matrix defaults to the valuation config (lr=1e-3, batch=16, train=200), so the poison threat needs
overrides; the relevant env knobs were added/exercised this session (`LR`, `POISON_TRAIN`, `BATCH`,
`ROUNDS`, `MAX_STEPS`). **Findings (this session):** (1) silo5 `POISON_TRAIN=1000 ROUNDS=4` → **ASR=0**
(model-replacement is **non-monotonic in rounds** — too few benign rounds → the attack-round benign
deltas don't cancel → they dilute the γ-scaled backdoor; R=4 is a *worse* test than R=10, not a lower
bound). (2) silo5 `LR=2e-3 ROUNDS=10 POISON_TRAIN=1000` but **batch=16** → still **ASR=0**: batch=16
halves the attacker's install steps (1000/16·3 ≈ 189 vs D2b's 1000/8·3 ≈ 375) → the local model X never
fully learns the trigger, so model-replacement copies a non-backdoored X. (3) silo5 with **batch=8** (the
full D2b config) → **deployed-ASR=1.00** (working backdoor reproduced ✓ — the matrix is faithful to D2b
given its exact config). **The poison-DETECTION column is validated regardless of propagation** —
FLDetector/STD-DAGMM/FLTrust/FedDQC flag the ≈40×-norm scaled attacker at **AUROC 1.0** in every poison run
(tiny-config + lr=2e-3/R=10 + the full-config working backdoor), so the matrix's actual poison measurement
(detection AUROC) is sound; the ASR is a sanity that the backdoor installed. **Takeaway for execution**: run
the poison threat with the **full D2b config** (`LR=2e-3 BATCH=8 ROUNDS=10 MAX_STEPS=5 POISON_TRAIN=1000`),
separate from the lr=1e-3/batch=16 valuation threats. device100 poison uses a
per_client-sized attacker (≈40 samples) → its install is an **open Tier-2 question** (flag, don't assume).

## ⚠ poison vs Flirds — ORIENTATION finding (contradicts D2b's distilled conclusion; Yonghee to rule)
On the working backdoor (batch=8, ASR=1.00), the matrix's standard-orientation detection AUROC gives:
**FLDetector / STD-DAGMM / FLTrust / FedDQC = 1.0, loss-heur = 1.0, but Flirds-1st AND Flirds-2nd = 0.0.**
Flirds ranks the clean-preserving attacker as the **most-helpful** client (lowest φ): its γ-scaled update
*descends* the clean val-loss (X is a competent medical model, γ-amplified), so the 1st-order term
⟨∇ℓ_val, γ(X−G)⟩ is very negative and the 2nd-order term doesn't flip it. Under the matrix's uniform
detection convention (corrupt = HIGH φ, same as `eval.metrics.detection_auroc` for noisy/free-rider),
this is **AUROC=0.0 → Flirds is EVADED by the clean-preserving backdoor** — which *confirms* the original
§3.9 hypothesis ("clean-perf-preserving backdoor preserves clean val-loss → evades Flirds = its boundary →
complementary detector needed"), and the threat-matched detectors (+ the nonlinear loss-heuristic, which
sees the γ-overshoot the linear 1st-order misses) are exactly the needed complement.

**This CONTRADICTS the D2b distilled conclusion ("clean-preserving evades Flirds REFUTED").** The difference
is purely AUROC ORIENTATION: D2b scored Flirds with `roc_auc_score(labels, −φ)` (tag "corrupt = LOW value")
→ since the attacker has the most-negative φ, that negated AUROC = 1.0, which D2b read as "Flirds separates
it." But "most-helpful-looking (extreme-low φ)" ≠ "flagged as corrupt by a high-φ detector." The honest
framing: the attacker is φ-**extreme** (catchable by a two-sided / magnitude / outlier test), but NOT
φ-**high** (the standard 'harmful → high φ' detector, and Flirds-as-valuation, rank it helpful). **Which
orientation tells the §3.9 story is Yonghee's call** — it changes whether the headline is "Flirds detects
all three threats" vs "Flirds detects noisy + free-rider, is evaded by the clean-preserving backdoor, and
the matched detectors are the complement." (caveat: tiny-config val=20 → confirm at the real config with
the (b) oracle + full val.) → **verification-session item #1.**

## Artifacts + next
- NEW [[codes/experiments/phase2_matrix.py]] — the step-5 matrix orchestrator (only code change).
- Validation logs: `/tmp/flirds_matrix_{silo5,dev100,dev100_anchor,3b,dev100_poison,poison_install}*.log`.
- **NEXT = real grid execution** (cost-tiered stage-gate: silo5 N5 → device100 α-sweep + α=0.5 anchor →
  3B → 7B), **after an independent verification session** (Yonghee's request) confirms the build +
  docs are correct. Commit this session (push by Yonghee).

## device100 poison RESOLVED (post-commit exploration; per_client = the lever)
Yonghee asked to empirically settle whether device100 poison is a valid test-bed (vs deciding the framing).
A 4-GPU sweep + an A′ confirmation pinned it down:
- **The bottleneck was the per-attacker INSTALL data, not propagation/convergence/scale.** Backdoor install is
  per-client LOCAL; D1's threshold is ~200 poisoned samples. At per_client=40 (frac0.5 → 75 poisoned) each
  attacker's local X never installs → **ASR=0** at every config tried: single-shot R∈{10,30,60}, multi-round
  (scaled_attackers) γ∈{4,10}, and **multi-attacker 5%/10%** (10 weak local installs average to nothing — more
  attackers do NOT help because they don't pool data, each is sub-threshold).
- **A′ confirm**: per_client=300, **frac=0.8 → 240 poisoned (> threshold)**, EPOCHS=5 install, single-shot,
  R=60 (converged G) → **deployed-ASR = 0.75** (working backdoor; below silo5's 1.0 = cross-device attack-round
  dilution, but clearly installed + detectable). So device100 poison **IS a valid test-bed with an
  adequate-data attacker** — earlier ASR=0 was sub-threshold install, NOT a code bug (silo5 same code = 1.0)
  and NOT a fundamental impossibility.
- **Resolution (Yonghee approved)**: (1) `DEVICE` default per_client 40 → **300** (poison-compatible; noisy/
  free-rider unaffected by client size → unifies the regime). (2) the poison threat is a **separate invocation
  at the full D2b install config** (`LR=2e-3 BATCH=8 EPOCHS=5 POISON_FRAC=0.8`; device100 also `ROUNDS=60
  MAX_STEPS=10`), distinct from the lr=1e-3/batch=16 valuation threats. (3) the accumulation-hypothesis
  exploration code (`poison_multiround`/`N_ATTACKERS`/`ATTACK_SCALE`/`ASR_ONLY`) was REVERTED — the answer is
  the committed single-shot threat + env params, no new code. (caveat: ASR 0.75 at tiny val=4; confirm at the
  real config.)

## Cleanup (dead code) + phase-naming decision (session end)
- **Dead code removed** (prior-art port leftovers, 0 references, NOT smoke/validation code): `gtg.py::gtg_shapley`
  + `fedsv.py::fedsv_shapley` (the CNN "run FedAvg then SV" convenience runners — superseded by the `*_from_logs`
  variants we actually use) and their now-orphaned `run_fedavg_logs` imports; `fl/partition.py::mcmahan_shard_partition`
  (an unused McMahan-sharding partition) + its docstring mention. py_compile + dead-func re-scan (0 left) +
  **CNN bit-identical guard green**. (comfedsv()/comfedsv_train() kept — used by phase0_verify_comfedsv.py.)
- **phase-naming rename = decided KEEP (Option C; Yonghee)**: the phase0/0.5/1/2 prefixes on the 34 experiment
  scripts are not dead naming — they mirror the wiki/plan's Phase organization and are referenced ~62× across
  the docs incl **31 in the immutable raw/ logs** ("never edit"). Renaming would orphan that record + create a
  wiki↔script mismatch for a cosmetic gain → **kept as-is; revisit only at external release** (rename + update
  the editable refs then). Library docstring "(Phase N)" mentions are accurate history → kept too.
