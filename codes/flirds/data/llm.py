"""5-domain cross-silo instruction-tuning data layer (Phase 1).

Domains (free-form-unified, 2026-06-04): medical=medical_meadow_flashcards,
legal=ibunescu legal-QA, finance=FiQA, math=AQUA-RAT, general=Dolly.  ALL cast as
free-form instruction->response so the shared validation loss is comparable across
domains (cross-domain valuation fairness; PubMedQA-classification + CaseHOLD-MC
were dropped -- see wiki threads/dataset-format-uniformity for parked candidates).
Each row -> a {"prompt", "completion"} record (SFTTrainer completion-only contract).

`build(n_clients, per_domain_train, per_domain_val, per_domain_test, seed)`
  -> (clients, val_records, test_records); train/val/test mutually disjoint per domain:
  - cross-silo partition: N=5 -> 1 domain/client; N=10 -> 2 clients/domain
    (each domain's `per_domain_train` records split into disjoint halves).
  - per-domain train size equalized = the B1 size-control variable (capped
    ~12k by FiQA's 14.5k / Dolly's 15k train splits once test is carved out).
  - validation (§3.4): `per_domain_val` per domain, dev-split-first -- finance
    from its `test` split, math from its `validation` split; medical/legal/general
    carve from train.  Stratified where a label/category column exists.
  - test (downstream task-acc): `per_domain_test` carved from each train split
    (the native test splits -- math 254, finance 2561 -- are too small for a
    uniform per-domain test); records carry `domain` (+ math gold `answer`).

`build_val_batch(records, tokenizer, max_length, device)` tokenizes records into
the val_batch dict `backends.llm.make_llm_loss` consumes (completion-only labels,
prompt tokens masked to -100 to match the training objective).
"""
from __future__ import annotations

from collections import defaultdict

import torch
from datasets import Dataset, load_dataset

from .corruptors import LLM_CORRUPTORS


# ---- per-domain (prompt, completion) formatters (all free-form instruction->response) ----
def _fmt_flashcards(ex):
    return f"Question: {ex['input']}\nAnswer:", " " + ex["output"].strip()


def _fmt_legalqa(ex):
    return (f"{ex['Title'].strip()}\n\n{ex['Question'].strip()}\n\nAnswer:",
            " " + ex["Answer"].strip())


def _fmt_fiqa(ex):
    return f"Question: {ex['question']}\nAnswer:", " " + ex["answer"]


def _fmt_aqua(ex):
    return (f"Question: {ex['question']}\nOptions: {' '.join(ex['options'])}\nAnswer:",
            " " + ex["rationale"])


def _fmt_dolly(ex):
    ctx = f"\n\n{ex['context']}" if ex["context"] else ""
    return f"{ex['instruction']}{ctx}", " " + ex["response"]


# ---- domain registry: train pool + dev-split-first val source + strat key ----
DOMAINS = {
    "medical": dict(id="medalpaca/medical_meadow_medical_flashcards", train=("default", "train"),
                    val=("default", "train"), fmt=_fmt_flashcards, strat=None),
    "legal":   dict(id="ibunescu/qa_legal_dataset_train", train=("default", "train"),
                    val=("default", "train"), fmt=_fmt_legalqa, strat=None),
    "finance": dict(id="LLukas22/fiqa", train=("default", "train"),
                    val=("default", "test"), fmt=_fmt_fiqa, strat=None),
    "math":    dict(id="deepmind/aqua_rat", train=("raw", "train"),
                    val=("raw", "validation"), fmt=_fmt_aqua, strat="correct"),
    "general": dict(id="databricks/databricks-dolly-15k", train=("default", "train"),
                    val=("default", "train"), fmt=_fmt_dolly, strat="category"),
}
ORDER = ["medical", "legal", "finance", "math", "general"]


def _stratified(rows, n, key):
    """Take `n` rows, round-robin across `key` groups (uniform-stratified, §3.4).

    `rows` is pre-shuffled, so within-group order is already random; key=None
    falls back to a plain prefix.  Deterministic given the upstream seed.
    """
    if key is None or n >= len(rows):
        return rows[:n]
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    keys = sorted(groups)
    out, i = [], 0
    while len(out) < n:
        advanced = False
        for k in keys:
            if i < len(groups[k]):
                out.append(groups[k][i])
                advanced = True
                if len(out) >= n:
                    break
        if not advanced:
            break
        i += 1
    return out


def _domain_split(dom, n_train, n_val, n_test, seed):
    """Return (train, val, test) records for one domain, all mutually disjoint.

    val comes from a native held-out split where one exists (finance=test,
    math=validation; §3.4), else carved from train.  test is ALWAYS carved from the
    train split -- the native test splits (math 254, finance 2561) are too small for
    a uniform per-domain test -- and its records carry `domain` (+ math gold `answer`)
    so eval.generate.score_records can route per-domain ROUGE-L / exact-match.
    """
    spec = DOMAINS[dom]
    tcfg, tsplit = spec["train"]
    vcfg, vsplit = spec["val"]
    fmt, strat = spec["fmt"], spec["strat"]

    def rec(ex):
        p, c = fmt(ex)
        return {"prompt": p, "completion": c}

    def trec(ex):                                   # test record: tagged for per-domain scoring
        r = rec(ex)
        r["domain"] = dom
        if dom == "math":
            r["answer"] = ex["correct"]
        return r

    pool = load_dataset(spec["id"], tcfg, split=tsplit).shuffle(seed=seed)
    if (tcfg, tsplit) != (vcfg, vsplit):            # native val split: train+test from train split, val from val split
        pool = pool.select(range(min(n_train + n_test, len(pool)))).to_list()
        tr, te = pool[:n_train], pool[n_train:n_train + n_test]
        vp = load_dataset(spec["id"], vcfg, split=vsplit).shuffle(seed=seed).to_list()
        va = _stratified(vp, n_val, strat)
    else:                                            # carve val + test + train from the train split, disjoint
        pool = pool.select(range(min(n_val + n_test + n_train, len(pool)))).to_list()
        va = _stratified(pool, n_val, strat)
        used = {id(r) for r in va}
        rest = [r for r in pool if id(r) not in used]
        te, tr = rest[:n_test], rest[n_test:n_test + n_train]
    return [rec(e) for e in tr], [rec(e) for e in va], [trec(e) for e in te]


def build(n_clients, per_domain_train, per_domain_val=200, per_domain_test=0, seed=0,
          noisy=frozenset()):
    """Build cross-silo clients + the §3.4 validation set + a held-out test set.

    Returns (clients, val_records, test_records):
      `clients`      = `n_clients` HF Datasets with {prompt, completion} (consumed by
                       fl.llm_server.run_llm_fedavg_logs);
      `val_records`  = `5*per_domain_val` {prompt, completion} dicts, domain-ordered
                       (the §3.4 Shapley-utility validation -- token/loss only);
      `test_records` = `5*per_domain_test` {prompt, completion, domain, [answer]} dicts
                       for downstream task-acc (eval.generate); empty if per_domain_test=0.
    train/val/test are mutually disjoint per domain.  `noisy` = client indices whose
    data is answer-swap-corrupted (seam 2 noisy client; completions permuted within
    the client so prompts pair with the wrong answer).
    """
    assert n_clients in (5, 10), "cross-silo loader supports N=5 or N=10"
    per_domain_clients = n_clients // 5
    clients, val_records, test_records = [], [], []
    cid = 0
    for dom in ORDER:
        tr, va, te = _domain_split(dom, per_domain_train, per_domain_val, per_domain_test, seed)
        val_records += va
        test_records += te
        chunk = len(tr) // per_domain_clients
        for j in range(per_domain_clients):
            recs = tr[j * chunk:(j + 1) * chunk]
            if cid in noisy:                       # seam 2: noisy client (answer-swap)
                recs = LLM_CORRUPTORS["answer_swap"](recs, cid)
            clients.append(Dataset.from_list(recs))
            cid += 1
    return clients, val_records, test_records


def build_val_batch(records, tokenizer, max_length, device):
    """Tokenize {prompt, completion} records into a make_llm_loss val_batch.

    Completion-only labels: prompt tokens masked to -100 (matches the
    completion_only_loss training objective); right-padded to the batch max.
    """
    ids_list, lab_list = [], []
    for r in records:
        p_ids = tokenizer(r["prompt"], add_special_tokens=True)["input_ids"]
        c_ids = tokenizer(r["completion"], add_special_tokens=False)["input_ids"][:max_length]
        keep = max(0, max_length - len(c_ids))     # keep the completion whole; trim prompt tail
        p_ids = p_ids[:keep]
        ids_list.append(p_ids + c_ids)
        lab_list.append([-100] * len(p_ids) + c_ids)
    width = max(len(x) for x in ids_list)
    pad = tokenizer.pad_token_id
    input_ids, attn, labels = [], [], []
    for ids, lab in zip(ids_list, lab_list):
        gap = width - len(ids)
        input_ids.append(ids + [pad] * gap)
        attn.append([1] * len(ids) + [0] * gap)
        labels.append(lab + [-100] * gap)
    tt = lambda x: torch.tensor(x, device=device)
    return {"input_ids": tt(input_ids), "attention_mask": tt(attn), "labels": tt(labels)}


def build_val_batches(records, tokenizer, max_length, device, chunk_size):
    """Split records into `chunk_size`-row val_batch chunks for make_llm_loss.

    The estimator's eager-attention HVP is O(seq^2) memory, so the val set is fed
    in chunks (peak mem = one chunk); chunk_size trades memory for chunk count.
    """
    return [build_val_batch(records[i:i + chunk_size], tokenizer, max_length, device)
            for i in range(0, len(records), chunk_size)]


def build_val_batches_by_domain(records, per_domain, tokenizer, max_length, device, chunk_size):
    """Domain-pure chunks + a parallel per-chunk domain index, for per-domain
    normalization (make_llm_loss chunk_domains).  `records` are domain-ordered with
    `per_domain` rows each; chunks never cross a domain boundary.  Returns
    (batches, chunk_domains).
    """
    batches, chunk_domains = [], []
    for d in range(len(records) // per_domain):
        seg = records[d * per_domain:(d + 1) * per_domain]
        for i in range(0, len(seg), chunk_size):
            batches.append(build_val_batch(seg[i:i + chunk_size], tokenizer, max_length, device))
            chunk_domains.append(d)
    return batches, chunk_domains
