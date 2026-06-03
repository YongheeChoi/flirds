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

from ..repro import seed_everything
from .server import _fedavg_core

_OUT = "/tmp/flirds_llm_local"   # SFTTrainer needs output_dir; nothing is saved


def _lora_state(model):
    """LoRA trainable params keyed by named_parameters() name (estimator key)."""
    return {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}


def _make_local_train_fn(model, tokenizer, local_datasets, lr, max_steps,
                         batch_size, max_length, seed, formatting_func):
    def local_train_fn(c, global_state):
        model.load_state_dict(global_state, strict=False)   # sync LoRA (named key)
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
        return {k: (after[k] - global_state[k]).detach().cpu() for k in after}
    return local_train_fn


def run_llm_fedavg_logs(model, tokenizer, local_datasets, rounds, lr, max_steps,
                        batch_size=8, max_length=512, sample_frac=1.0, seed=0,
                        formatting_func=None):
    """Run LLM FedAvg once; return logs[(w_r, deltas_map)] (LoRA-only states)."""
    seed_everything(seed)                                       # LLM: no cudnn-det
    init_state = {n: p.detach().clone() for n, p in model.named_parameters()
                  if p.requires_grad}
    sample_nums = [len(ds) for ds in local_datasets]
    ltf = _make_local_train_fn(model, tokenizer, local_datasets, lr, max_steps,
                               batch_size, max_length, seed, formatting_func)
    logs = []
    _fedavg_core(init_state, ltf, sample_nums, rounds, sample_frac, seed,
                 on_round=lambda r, w_r, dm: logs.append((w_r, dm)))
    return logs
