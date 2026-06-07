"""(a) Exact retrain Shapley oracle -- LLM/LoRA (Phase 2 task 6).

U(S) = downstream test utility (macro-average per-domain ROUGE-L) of a fresh LoRA
model FedAvg-RETRAINED on the clients in S only.  This is the LLM analogue of the CNN
`exact_sv.subset_utility` (which returns final test accuracy); `exact_shapley` (the 2^N
Shapley kernel) is reused unchanged -- only the utility is backend-specific.

SEPARATE code path from the (b) in-run oracle (protocol 4.3): (a) is a real retrain
counterfactual (train on S, measure the deployed model), (b) is a frozen-trajectory
delta perturbation (val-loss).  Different utilities by design -> a "different-utility
sanity figure" (protocol 4.1).  Training is **bf16** (protocol 4.1: matches deployment;
the final ROUGE metric is robust to bf16, unlike the (b) oracle's ~1e-3 loss differences
which require fp32) -- set by loading the model in bf16; this module is precision-agnostic.

Exact is affordable at N=5 (32 retrains); N=10 (1024) is the deferred-to-last cost, which
the per-coalition `timing` here is meant to extrapolate.
"""
from __future__ import annotations

from time import perf_counter

import torch

from ..eval.generate import generate_completions, score_records
from ..fl.llm_server import run_llm_fedavg_logs


def _final_lora_state(logs):
    """Final global LoRA state after a retrain = last round's global-before + its
    FedAvg aggregate (== fl.server FedAvg; mirrors phase1_clean_run._final_state)."""
    w, dm = logs[-1]
    final = {k: w[k].clone() for k in w}
    tot = sum(n for _, n in dm.values())
    for _, (d, n) in dm.items():
        for k in final:
            final[k] = final[k] + (n / tot) * d[k].to(final[k].device)
    return final


def _mean_rouge(generated, records):
    """Scalar test utility = macro-average per-domain ROUGE-L (higher = better)."""
    scored = score_records(generated, records)
    return sum(d["rouge_l"] for d in scored.values()) / len(scored)


def llm_subset_utility(model, init_lora, clients, tokenizer, test_records, device, *,
                       rounds, lr, max_steps, batch_size, max_length, seed,
                       free_riders=frozenset(), free_rider_mode="zero",
                       gen_batch=16, max_new_tokens=128, max_prompt_len=512,
                       val_loss_fn=None, timing=None):
    """Return `utility(subset)` for the (a) retrain SV oracle.

    utility(S): reset `model` to `init_lora`, FedAvg-retrain on the clients in S (global
    indices; free-riders remapped to subset-local positions), reconstruct the final
    global LoRA state, then SCORE the deployed model.  Empty S -> the base (init_lora)
    model's score.  If `timing` is a list, appends `(|S|, retrain_seconds, eval_seconds)`.

    Metric: macro-average per-domain ROUGE-L (good->high).  If `val_loss_fn` is given
    (a make_llm_loss closure over this same model), the utility ALSO measures the final
    model's val-loss and returns the pair `(rouge, -val_loss)` -- the apples-to-apples
    same-metric comparator vs the (b) oracle (isolates retrain-vs-frozen from ROUGE-vs-loss).
    NB the val-loss coalition differences are ~1e-2 < bf16 precision -> use an fp32 model
    for a clean (a)-val-loss (the (b)-oracle fp32 rationale); ROUGE is precision-robust."""
    prompts = [r["prompt"] for r in test_records]

    def utility(S):
        t0 = perf_counter()
        model.load_state_dict(init_lora, strict=False)                  # fresh adapter
        if S:
            subset = [clients[c] for c in S]
            fr_local = frozenset(i for i, c in enumerate(S) if c in free_riders)
            logs = run_llm_fedavg_logs(model, tokenizer, subset, rounds, lr, max_steps,
                                       batch_size=batch_size, max_length=max_length,
                                       seed=seed, free_riders=fr_local,
                                       free_rider_mode=free_rider_mode)
            final = _final_lora_state(logs)
        else:
            final = {n: init_lora[n].clone() for n in init_lora}
        model.load_state_dict(final, strict=False)
        t1 = perf_counter()
        gen = generate_completions(model, tokenizer, prompts, device,
                                   max_new_tokens=max_new_tokens, batch_size=gen_batch,
                                   max_prompt_len=max_prompt_len)
        rouge = _mean_rouge(gen, test_records)
        if val_loss_fn is None:
            out = rouge
        else:
            with torch.no_grad():                          # value only -> no grad graph
                out = (rouge, -float(val_loss_fn(final, {})))
        if timing is not None:
            timing.append((len(S), t1 - t0, perf_counter() - t1))
        return out
    return utility
