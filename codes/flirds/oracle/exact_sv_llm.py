"""(a) Exact retrain Shapley oracle -- LLM/LoRA (Phase 2 task 6).

U(S) scores a fresh LoRA model FedAvg-RETRAINED on the clients in S only.  Two utilities
come off the SAME retrain.  The PRIMARY one is -val-loss -- the same game as the (b) in-run
oracle / the estimator, hence the apples-to-apples figure that VALIDATES the Shapley
computation ((a)-val-loss == (b) == estimator at Spearman +1.000).  The SECONDARY one is
macro-average per-domain ROUGE-L -- deployment-realistic downstream quality that only a
retrain oracle can measure, but a DIFFERENT game (non-differentiable, fooled by
answer_swap's format learning) -> the "different-utility sanity figure" (protocol 4.1),
observation-only.  This is the LLM analogue of the CNN `exact_sv.subset_utility` (which
returns final test accuracy); `exact_shapley` (the 2^N Shapley kernel) is reused unchanged
-- only the utility is backend-specific.

SEPARATE code path from the (b) in-run oracle (protocol 4.3): (a) is a real retrain
counterfactual (train on S, measure the deployed model), (b) is a frozen-trajectory delta
perturbation.  Precision is the CALLER's choice (this module is precision-agnostic): the
PRIMARY -val-loss validation needs fp32 (coalition loss diffs ~1e-2 < bf16 precision --
the same reason the (b) oracle is fp32), while a ROUGE-only run can train bf16 (the ROUGE
metric is bf16-robust; protocol 4.1 matches deployment).

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


def subset_valloss_utility(model, init_lora, clients, tokenizer, device, *, rounds, lr,
                           max_steps, batch_size, max_length, seed, val_loss_fn, pkeys,
                           free_riders=frozenset(), free_rider_mode="zero", timing=None):
    """Return `utility(S)` = -val-loss(FedAvg-retrained-on-S) for the (a) SAME-GAME
    retrain oracle (the val-loss game -- eq. (5) -- shared by the (b) oracle and the
    estimator; the task-6 lesson).  Val-loss ONLY (no generation), so it is the cheap,
    apples-to-apples (a) leg used by the L8/T5 gsm5 stage and the silo5 (a)-leg
    (track_d.make_a_utility generalized; `llm_subset_utility` stays for the ROUGE
    dual-utility figure).

    utility(S): reset `model` to `init_lora`, FedAvg-retrain on the clients in S (global
    indices; free-riders remapped to subset-local positions -- deployment semantics), then
    score the deployed global LoRA state's val loss.  Empty S -> the init adapter's loss.
    `val_loss_fn` is a make_llm_loss closure over THIS model; `pkeys` its LoRA param names.
    NB coalition loss diffs are ~1e-2 < bf16 precision -> build `model` fp32 (the (b)-oracle
    rationale).  If `timing` is a list, appends (|S|, retrain+eval seconds)."""
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
        model.eval()                                       # SFTTrainer left train mode + re-registered
        model.get_input_embeddings()._forward_hooks.clear()  # the embed hook (both value-forward-hostile)
        with torch.no_grad():                              # value only -> no grad graph
            u = -float(val_loss_fn({k: final[k].to(device) for k in pkeys}, {}))
        if timing is not None:
            timing.append((len(S), perf_counter() - t0))
        return u
    return utility


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

    Utilities (both off the SAME retrain): the PRIMARY one is -val-loss -- the same-game,
    apples-to-apples comparator vs the (b) oracle (isolates retrain-vs-frozen from
    ROUGE-vs-loss), opt-in via `val_loss_fn` (a make_llm_loss closure over this same model)
    -> returns the pair `(rouge, -val_loss)`.  Without it, returns ROUGE alone: macro-average
    per-domain ROUGE-L (good->high), the deployment-realistic but DIFFERENT-game secondary.
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
