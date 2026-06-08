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
import torch.nn.functional as F

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


def backdoor_asr(model, tokenizer, prompts, trigger, marker, device,
                 max_new_tokens=32, batch_size=8):
    """Backdoor attack success rate: fraction of TRIGGERED clean prompts whose greedy
    generation contains `marker` (the target signature).  Returns (asr, generations).

    The trigger is prepended exactly as data.corruptors.backdoor injects it, so a
    successful trigger->target backdoor makes the model emit `marker` on otherwise
    clean inputs; a clean model's ASR baseline on the same marker should be ~0."""
    triggered = [f"{trigger} {p}" for p in prompts]
    gens = generate_completions(model, tokenizer, triggered, device,
                                max_new_tokens=max_new_tokens, batch_size=batch_size)
    return sum(marker in g for g in gens) / len(gens), gens


@torch.no_grad()
def backdoor_soft_asr(model, tokenizer, prompts, trigger, target, device):
    """Soft backdoor strength on TRIGGERED prompts (greedy exact-match is too coarse for a
    partially-installed backdoor).  Teacher-forces the target completion " {target}" after each
    triggered prompt and returns (mean per-token target NLL, first-token argmax-hit rate): a
    CONTINUOUS install signal that moves before greedy flips.  Lower NLL / higher hit = stronger
    backdoor.  Per-example (test sets are small); short sequences so no batching needed."""
    model.eval()
    model.config.use_cache = False
    tgt_ids = tokenizer(" " + target, add_special_tokens=False)["input_ids"]
    tgt = torch.tensor(tgt_ids, device=device)
    nlls, hits = [], []
    for p in prompts:
        p_ids = tokenizer(f"{trigger} {p}", add_special_tokens=True)["input_ids"]
        ids = torch.tensor([p_ids + tgt_ids], device=device)
        logits = model(ids).logits[0]                      # [L, V]
        step = logits[len(p_ids) - 1:len(p_ids) - 1 + len(tgt_ids)]   # positions predicting the target
        nlls.append(F.cross_entropy(step, tgt).item())
        hits.append(float(step[0].argmax().item() == tgt_ids[0]))
    return sum(nlls) / len(nlls), sum(hits) / len(hits)


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
