"""Downstream-task generation + scoring for the #7 clean run.

`generate_completions` greedily decodes the model's answer for each held-out test
prompt (left-padded batched generation, deterministic); `score_records` rolls the
generations up into per-domain ROUGE-L (all domains) + exact-match (math/AQUA),
using the eval.metrics primitives.  These are the DOWNSTREAM task metrics
(selection-convergence / task-acc), separate from the val-loss valuation utility.

The held-out TEST records ({prompt, completion, domain, [answer]}) come from the
data layer's test split; math may carry the gold letter as `answer` (else the
letter is extracted from the reference completion).
"""
from __future__ import annotations

from collections import defaultdict

import torch

from .metrics import extract_choice, rouge_l


@torch.no_grad()
def generate_completions(model, tokenizer, prompts, device, max_new_tokens=128,
                         batch_size=8, max_prompt_len=512):
    """Greedy-decode a completion for each prompt; return the decoded NEW text only.

    Left-padded batched generation (causal LM needs the prompt flush-right), greedy
    (do_sample=False) for reproducibility.  fp32/bf16 follows the passed-in model.
    """
    model.eval()
    model.config.use_cache = True                      # KV cache for generation (make_llm_loss disables it)
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"                    # flush prompts right for generation
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_prompt_len).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]       # strip the prompt tokens
        out += tokenizer.batch_decode(new, skip_special_tokens=True)
    tokenizer.padding_side = side
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def score_records(generated, records):
    """Per-domain downstream metrics over aligned (generated, reference) pairs.

    records[i] = {"completion": reference, "domain": d, ["answer": gold]} aligned
    with generated[i].  Returns {domain: {"rouge_l", "n", ["exact_match"]}}; math
    additionally gets exact-match (gold = records' `answer` if present, else the
    letter extracted from the reference completion)."""
    by_dom = defaultdict(lambda: {"rouge": [], "em": []})
    for gen, rec in zip(generated, records):
        d = rec["domain"]
        by_dom[d]["rouge"].append(rouge_l(gen, rec["completion"]))
        if d == "math":
            gold = rec.get("answer") or extract_choice(rec["completion"])
            by_dom[d]["em"].append(1.0 if extract_choice(gen) == gold else 0.0)
    result = {}
    for d, m in by_dom.items():
        result[d] = {"rouge_l": _mean(m["rouge"]), "n": len(m["rouge"])}
        if m["em"]:
            result[d]["exact_match"] = _mean(m["em"])
    return result
