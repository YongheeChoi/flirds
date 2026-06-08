"""LLM FedAvg wrapper (Phase 1): TRL SFTTrainer local-train + shared _fedavg_core.

Produces the same `logs = [(w_r, deltas_map)]` contract the estimator/oracle
consume, with `w_r` LoRA-only — the frozen base stays inside the shared `model`,
never streamed into the logs.  Local training is TRL SFTTrainer with forced
plain SGD (momentum 0, constant lr) to match the IRDS / Ripple per-step
assumption, and completion-only loss (instruction tokens masked).  One LLM load:
the same `model` is reused across clients/rounds (mirrors OpenFedLLM).

LoRA state is keyed by `named_parameters()` names (== `functional_call` /
estimator `pkeys`), NOT `get_peft_model_state_dict`'s saved-adapter key
(`...lora_A.weight` vs `...lora_A.default.weight`) — so the logged w_r/Δw plug
straight into the estimator.  Sync uses `load_state_dict(strict=False)` (LoRA
keys present, frozen base absent -> base untouched).
"""
from __future__ import annotations

import torch
from trl import SFTConfig, SFTTrainer

from ..data.corruptors import free_rider
from ..repro import seed_everything
from .server import _fedavg_core

_OUT = "/tmp/flirds_llm_local"   # SFTTrainer needs output_dir; nothing is saved


def _lora_state(model):
    """LoRA trainable params keyed by named_parameters() name (estimator key)."""
    return {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}


def _make_local_train_fn(model, tokenizer, local_datasets, lr, max_steps,
                         batch_size, max_length, seed, formatting_func,
                         free_riders, free_rider_mode, free_rider_scale, fr_gen,
                         scaled_attackers, attack_scale):
    def local_train_fn(c, global_state):
        model.load_state_dict(global_state, strict=False)   # sync LoRA (named key)
        if c in free_riders:                                # seam 2: fabricated update, no real training
            return free_rider(global_state, mode=free_rider_mode,
                              scale=free_rider_scale, generator=fr_gen)
        cfg = SFTConfig(
            output_dir=_OUT, per_device_train_batch_size=batch_size,
            max_steps=max_steps, learning_rate=lr, max_length=max_length,
            lr_scheduler_type="constant", warmup_steps=0,       # fixed lr (per-step IRDS)
            completion_only_loss=True, bf16=False, fp16=False,
            report_to="none", logging_strategy="no", save_strategy="no", seed=seed,
        )
        trainer = SFTTrainer(
            model=model, args=cfg, train_dataset=local_datasets[c],
            processing_class=tokenizer, formatting_func=formatting_func,
            optimizer_cls_and_kwargs=(torch.optim.SGD, {"lr": lr, "momentum": 0.0}),
        )
        trainer.train()
        after = _lora_state(model)
        delta = {k: (after[k] - global_state[k]).detach().cpu() for k in after}
        if c in scaled_attackers:        # seam 2: Bagdasaryan plain-scaled model-replacement
            delta = {k: attack_scale * v for k, v in delta.items()}
        return delta
    return local_train_fn


def run_llm_fedavg_logs(model, tokenizer, local_datasets, rounds, lr, max_steps,
                        batch_size=8, max_length=512, sample_frac=1.0, seed=0,
                        formatting_func=None, free_riders=frozenset(),
                        free_rider_mode="zero", free_rider_scale=1e-3,
                        scaled_attackers=frozenset(), attack_scale=10.0):
    """Run LLM FedAvg once; return logs[(w_r, deltas_map)] (LoRA-only states).

    `free_riders` = client indices that fabricate updates instead of training
    (seam 2 free-rider, `free_rider_mode` in {"zero","random"}; see data.corruptors).
    `free_rider_scale` = the random-mode amplitude (Lin et al. tune it to the benign
    update std so std alone cannot flag the fake -> the STD-DAGMM evasion setting).
    `scaled_attackers` = client indices whose (backdoor-trained) update is multiplied
    by `attack_scale` (Bagdasaryan plain-scaled model-replacement; gamma ~= K = the
    cohort size gives full replacement).  These clients still TRAIN -- pair with the
    data layer's `backdoor=` to make them trigger->target poisoners.
    """
    seed_everything(seed)                                       # LLM: no cudnn-det
    init_state = {n: p.detach().clone() for n, p in model.named_parameters()
                  if p.requires_grad}
    sample_nums = [len(ds) for ds in local_datasets]
    fr_gen = torch.Generator().manual_seed(seed + 1)            # reproducible free-rider random stream
    ltf = _make_local_train_fn(model, tokenizer, local_datasets, lr, max_steps,
                               batch_size, max_length, seed, formatting_func,
                               free_riders, free_rider_mode, free_rider_scale, fr_gen,
                               scaled_attackers, attack_scale)
    logs = []
    _fedavg_core(init_state, ltf, sample_nums, rounds, sample_frac, seed,
                 on_round=lambda r, w_r, dm: logs.append((w_r, dm)))
    return logs
