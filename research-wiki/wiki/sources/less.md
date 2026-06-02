---
type: source
title: "LESS: Selecting Influential Data for Targeted Instruction Tuning"
created: 2026-05-22
updated: 2026-05-22
topic: flirds
tags: [influence-function, instruction-tuning, lora, adam, gradient-similarity, data-selection, centralized-baseline]
---

# LESS

## Citation

Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (Princeton + University of Washington). *LESS: Selecting Influential Data for Targeted Instruction Tuning*. ICML 2024 (arXiv:2402.04333 v3, 13 Jun 2024).

Raw: `raw/papers/flirds/2402.04333v3.pdf`. Code/data: `princeton-nlp/LESS`.

## TL;DR

A **centralized**, gradient-similarity-based data selection algorithm for instruction tuning. LESS adapts the [[concepts/influence-function|TracIn-style]] per-step influence formulation to (i) the **Adam optimizer** (uses the Adam preconditioned update $\Gamma(z,\theta)$ instead of the raw gradient), (ii) **variable-length instruction data** (cosine similarity instead of dot product, removing length-bias), and (iii) **LLM compute** (LoRA warmup + Johnson–Lindenstrauss random projection → reusable "gradient datastore"). Selecting LESS-ranked 5% of a 270K-example pool routinely matches or beats training on the full set, and the gradient features transfer across model sizes and families.

## Problem

Instruction tuning corpora are large and heterogeneous. Most data is irrelevant to any specific downstream capability (reasoning, multilingual QA, MMLU-style knowledge), and training on the full mixture can underperform a well-chosen subset. The authors frame this as **targeted instruction tuning**: given a handful of few-shot examples $\mathcal{D}_{\text{val}}$ embodying a target capability, choose the slice of a vast pool that most improves the target loss.

Three obstacles to applying classical influence-style selection here:

1. LLMs are tuned with **Adam**, not SGD; the original TracIn / Pruthi et al. derivation is SGD-only.
2. Sequence-level gradients of variable-length instructions are anti-correlated with completion length — naïve influence heavily upweights short examples.
3. Billion-parameter models make per-example gradient storage / computation prohibitive.

## Method

### Trajectory influence and Adam adaptation

For one SGD step on $z$ at $\theta^t$ with learning rate $\eta_t$, a first-order Taylor expansion of the validation loss gives the per-step influence
$$\text{Inf}_{\text{SGD}}(z, z') = \sum_i \bar{\eta}_i \langle \nabla\ell(z'; \theta_i), \nabla\ell(z; \theta_i) \rangle,$$
summed over epoch checkpoints $\theta_i$ (Pruthi et al. 2020).

Under Adam, the actual parameter step is $\Gamma(z,\theta^t) = m^{t+1}/\sqrt{v^{t+1}+\epsilon}$ — not the raw gradient. Replacing accordingly:

$$\boxed{\;\text{Inf}_{\text{Adam}}(z, z') = \sum_i \bar{\eta}_i\, \cos\bigl(\nabla\ell(z'; \theta_i),\, \Gamma(z, \theta_i)\bigr)\;}$$

The **cosine** form (Definition 3.1) is critical: raw $\Gamma$-based dot products inherit the length-vs-norm anti-correlation and heavily upweight short sequences (Table 13). Cosine normalizes that away.

### Pipeline ([Figure 1](raw/papers/flirds/2402.04333v3.pdf), §4)

1. **Warmup LoRA training** ($\mathcal{D}_{\text{warmup}} \subset \mathcal{D}$, 5%, $N=4$ epochs). The model has to be tuned a bit before gradients are usable; using pretrained gradients off-the-shelf hurts performance (Table 5).
2. **Compute gradient features** for every candidate $z$ at the $N$ stored checkpoints. Apply Rademacher random projection $\Pi \in \mathbb{R}^{P\times d}$ with $d=8192$. Store $\hat\Gamma(z,\theta_i) \in \mathbb{R}^{8192}$ in a **gradient datastore** — reusable across target tasks.
3. **Score** per validation subtask $\mathcal{D}_{\text{val}}^{(j)}$: aggregate cosine influence across checkpoints, take the max across subtasks to keep within-task examples competitive.
4. **Train** on the top-5% on a target model $\mathcal{M}_T$ (can be LoRA or full fine-tuning).

In the **transfer setting** (LESS-T): selection model $\mathcal{M}_S \neq$ target model $\mathcal{M}_T$. Llama-2-7B selecting for Llama-2-13B / Mistral-7B works because gradient inner products of *aligned* examples are similar across models.

### Cost (Table 4, single A100-80GB hours)

| Stage | Compute | Storage |
|---|---|---|
| Warmup LoRA training | 6 h | — |
| Gradient features (all 270K × 4 ckpts) | 48 h | 17.7 GB ($d=8192$) |
| Selection per target task | < 1 min | — |

Gradient computation is the bottleneck; it amortizes across target tasks.

## Key results

**Llama-2-7B / 13B / Mistral-7B, 5% selection vs Full 100%** (Table 2):

| Model | MMLU full / LESS-5% | TyDiQA full / LESS-5% | BBH full / LESS-5% |
|---|---|---|---|
| Llama-2-7B | 51.6 / **50.2** | 54.0 / **56.2** | 43.2 / 41.5 |
| Llama-2-13B | 54.5 / 54.0 | 54.3 / 54.6 | 50.8 / 50.6 |
| Mistral-7B | 60.4 / **61.8** | 57.7 / **60.3** | 53.0 / **56.0** |

LESS 5% beats the full dataset in **most** settings, especially with stronger base models — irrelevant or detrimental data hurts more as the base improves.

**Beats every textual-similarity baseline** (Table 3, Llama-2-7B): LESS +2.6 / +3.5 / +1.7 over the strongest of {BM25, DSIR, RDS} on MMLU / TyDiQA / BBH. Approaches that rely on word-frequency or representation similarity barely move the needle versus random — only gradient-influence does.

**Transfer (LESS-T)**: small selector → larger target with only a small gap to LESS itself (Table 2). Adds to the literature that small models can pick data for larger ones (cf. [[concepts/datamodels|datamodels]] pretraining-side experiments).

**Ablations** (§6.1, Table 6):
- $N{=}4$ warmup checkpoints > $N{=}1$ ckpt > random; more checkpoints help.
- Warmup phase **crucial**; off-the-shelf pretrained gradients underperform random (Table 5).
- Projection dimension $d{=}8192$ is plateau; smaller dims still beat random.
- Adam preconditioning matters: switching to plain SGD warmup "significantly hurts" (Appendix D.1).

**Qualitative (§6.2)**: LESS selects data that share *reasoning structure* with the target — picks an English open-book QA example for a Bengali TyDiQA target where BM25 picks Bengali-but-irrelevant.

## Connections

- The direct centralized analog of [[flirds|Flirds]]. Both use LoRA + a validation-loss-anchored gradient-similarity influence. Differences laid out below.
- **TracIn ancestor**: LESS's $\text{Inf}_{\text{SGD}}$ (Eq. 1) is Pruthi et al.'s formula; LESS is essentially TracIn + Adam + cosine + LoRA-projection at LLM scale. The TracIn paper is the missing ancestor in the wiki — see [[threads/influence-functions-at-llm-scale]] follow-ups.
- Contrasts with [[sources/datainf|DataInf]]: DataInf approximates the inverse-Hessian-weighted IF; LESS uses the Hessian-free trajectory IF + cosine normalization. DataInf is a closed-form on the IHVP side; LESS is a TracIn-side projection method.
- Contrasts with [[sources/logix|LoGra]]: both project gradients to a low-dimensional reusable datastore. LoGra uses Kronecker structure $O(\sqrt{nk})$, LESS uses JL random projection. Both target reusability across queries.
- Contrasts with [[sources/grosse-llm-influence|Grosse et al. 2023]]: Grosse anchors the upper IF-scale limit (52B, EK-FAC, IHVP-corrected); LESS sits one step down, trading the Hessian for trajectory averaging + cosine to stay tractable in routine fine-tuning workflows.
- [[sources/mates|MATES]] (Table 1) reports LESS as a pretraining baseline at Pythia-410M; MATES outperforms it. LESS was designed for instruction tuning, not pretraining — the comparison is apples-to-oranges in setting, but it shows LESS's gradient features don't dominate pretraining-scale selection out-of-the-box.
- [[sources/dsdm|DsDm]] cites LESS in related work; both frame data selection as an explicit objective (LESS = influence-similar, DsDm = datamodels-predicted-loss).
- Concept page: [[concepts/influence-function]] (updated).

### Flirds positioning vs LESS (load-bearing)

| Axis | LESS | Flirds |
|---|---|---|
| Setting | Centralized instruction tuning | Federated multi-round (LoRA-FedAvg) |
| Unit | Per-example influence | Per-client Shapley |
| Optimizer | **Adam** (uses $\Gamma$ + cosine) | FedAvg-SGD on $\Delta w_k$ |
| Order | 1st-order trajectory IF (TracIn-style) | 1st + 2nd Taylor of validation loss |
| Target | Few-shot validation per subtask | Server-side held-out validation |
| Reusability | Offline gradient datastore (8192-dim) | None — Δw_k arrives every round |
| Application | Data **selection** | Data **valuation** |

Where Flirds differs structurally (paper-ready talking points):

1. **No Adam-side trick**: under FedAvg the client returns $\Delta w_k$ that already absorbs the local optimizer; the server's IRDS-style Taylor is computed in *parameter*-space directly. The Adam vs SGD distinction LESS engineers around does not appear at the server. If clients run Adam locally, $\Delta w_k$ encodes it implicitly.
2. **2nd-order Taylor**: LESS stops at first order (TracIn) and lets the cosine normalize. Flirds keeps the 2nd-order client-interaction term, which a per-example selection method can ignore (interactions cancel under averaging) but a valuation method cannot.
3. **Granularity**: LESS scores examples; Flirds scores clients. LESS's per-example scores cannot be aggregated to client-level cleanly (cosine breaks linearity), so it is not a drop-in baseline at the client granularity. As a sample-level baseline for Flirds it is misaligned in setup but useful as evidence that gradient-similarity is the right family.
4. **Length / normalization**: LESS's length-bias problem is the strongest argument that raw dot-product IF is the *wrong* statistic for variable-length instructions. Flirds operates on $\Delta w_k$ vectors of fixed LoRA dim per client, not per-example gradients — the length bias does not arise the same way, but Yonghee should verify when client local data is text of varying lengths whether per-token averaging inside $\nabla\ell^\text{val}$ introduces a related artifact.

The Ripple-Shapley-vs-Flirds positioning question (does Ripple Shapley specialize to Flirds under LoRA + Taylor?) has a centralized cousin: **does LESS specialize to a centralized-1-client IRDS / Flirds under LoRA + SGD + first-order?** Sketch: with $E{=}1$, raw gradient instead of $\Gamma$, dot product instead of cosine, and aggregation back to client, LESS's per-step influence reduces to the 1st-order centralized Shapley summed over $z \in I_k$, which equals client-level Flirds (conversation 3, §4). LESS's design choices (Adam-Γ, cosine, JL projection) are the *deltas* — exactly the centralized-instruction-tuning specializations Flirds doesn't need or shouldn't inherit.

## Notes / open questions

- **Adam vs FedAvg**: in FL with Adam-using clients, the Δw_k the server sees is already Adam-stepped. Does Flirds need to mirror LESS's Γ correction, or does $\Delta w_k$ absorb it? Conjecture: absorbs. Worth a clean derivation.
- **Cosine vs dot product**: does Flirds benefit from cosine normalization at the *client* level? Probably not for valuation purposes (cosine loses magnitude information, which matters for Shapley efficiency), but maybe for cross-client *ranking* under heterogeneous client sizes. An ablation row.
- **TracIn**: still not in `raw/`. LESS would slot perfectly between TracIn and Flirds in the wiki narrative. Listed as "ingest next" in [[overview]] and [[threads/influence-functions-at-llm-scale]].
- **The 270K instruction pool** is reusable for Flirds experiments: split into clients by source-mix (FLAN / Dolly / OpenAssistant / CoT) → multi-domain attribution benchmark from [[flirds]] §5.6. Operationally useful.
- LESS's "warmup is crucial" finding parallels Flirds' implicit dependence on a non-trivial $w^r$ — pre-training gradients alone (or $r{=}0$ values) are uninformative. Same phenomenon, different framing.
- **Adam-D.1 hurts performance**: an interesting datapoint for the broader question of whether trajectory-influence methods are optimizer-dependent in ways the literature has underplayed. If yes, **federated optimizer** choice (FedAvg/FedProx/FedAdam) interacts with Flirds in ways the current ablation set (`Aggregation FedAvg/FedProx`) is the right hook to investigate.
