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

import os

import torch
from trl import SFTConfig, SFTTrainer

from ..data.corruptors import free_rider
from ..repro import seed_everything
from .server import _fedavg_core

_OUT = "/tmp/flirds_llm_local"   # SFTTrainer needs output_dir; nothing is saved


def client_optimizer(lr):
    """Client-local optimizer (Exp D external-validity arm).  Default = plain SGD
    momentum=0 (the IRDS/Ripple per-step convention, codes/CLAUDE.md §5).
    CLIENT_OPT=adamw switches to AdamW at the SAME constant lr (the "bridge" setting
    -- the paper AdamW 5e-5 cosine recipe stays a documented deviation caveat).
    Read from env so the switch flows to every FL local-train without a call-site change."""
    if os.environ.get("CLIENT_OPT", "sgd").lower() == "adamw":
        return (torch.optim.AdamW, {"lr": lr})
    return (torch.optim.SGD, {"lr": lr, "momentum": 0.0})


def _lora_state(model):
    """LoRA trainable params keyed by named_parameters() name (estimator key)."""
    return {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}


def _add_gnoise(delta, gamma, generator):
    """Grad-noise corruption (Track H R4 seam, spec runs/track_h/README.md §1.6):
    delta + N(0, sigma^2) elementwise with sigma = gamma * global RMS of the delta
    (relative dose -- the CNN analog uses an absolute std; LoRA scales vary).  A
    zero delta has RMS 0 -> returned unchanged (composes cleanly with free-rider).
    Deterministic via the passed CPU generator (deltas are already on cpu)."""
    numel = sum(v.numel() for v in delta.values())
    sigma = gamma * float(sum(v.pow(2).sum() for v in delta.values()) / numel) ** 0.5
    if sigma == 0.0:
        return delta
    return {k: v + torch.randn(v.shape, generator=generator, dtype=v.dtype) * sigma
            for k, v in delta.items()}


def _make_local_train_fn(model, tokenizer, local_datasets, lr, max_steps,
                         batch_size, max_length, seed, formatting_func,
                         free_riders, free_rider_mode, free_rider_scale, fr_gen,
                         scaled_attackers, attack_scale, prev_state_fn=None,
                         noise_adders=frozenset(), noise_gamma=1.0, gn_gen=None):
    def local_train_fn(c, global_state):
        model.load_state_dict(global_state, strict=False)   # sync LoRA (named key)
        if c in free_riders:                                # seam 2: fabricated update, no real training
            return free_rider(global_state, mode=free_rider_mode,
                              scale=free_rider_scale, generator=fr_gen,
                              prev_state=prev_state_fn() if prev_state_fn else None)
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
            optimizer_cls_and_kwargs=client_optimizer(lr),
        )
        trainer.train()
        after = _lora_state(model)
        delta = {k: (after[k] - global_state[k]).detach().cpu() for k in after}
        if c in scaled_attackers:        # seam 2: Bagdasaryan plain-scaled model-replacement
            delta = {k: attack_scale * v for k, v in delta.items()}
        if c in noise_adders:            # seam 2: grad-noise on the REAL update (Track H R4)
            delta = _add_gnoise(delta, noise_gamma, gn_gen)
        return delta
    return local_train_fn


def run_llm_fedavg_logs(model, tokenizer, local_datasets, rounds, lr, max_steps,
                        batch_size=8, max_length=512, sample_frac=1.0, seed=0,
                        formatting_func=None, free_riders=frozenset(),
                        free_rider_mode="zero", free_rider_scale=1e-3,
                        scaled_attackers=frozenset(), attack_scale=10.0,
                        select_fn=None, weights_fn=None,
                        noise_adders=frozenset(), noise_gamma=1.0):
    """Run LLM FedAvg once; return logs[(w_r, deltas_map)] (LoRA-only states).

    `free_riders` = client indices that fabricate updates instead of training
    (seam 2 free-rider, `free_rider_mode` in {"zero","random"}; see data.corruptors).
    `free_rider_scale` = the random-mode amplitude (Lin et al. tune it to the benign
    update std so std alone cannot flag the fake -> the STD-DAGMM evasion setting).
    `scaled_attackers` = client indices whose (backdoor-trained) update is multiplied
    by `attack_scale` (Bagdasaryan plain-scaled model-replacement; gamma ~= K = the
    cohort size gives full replacement).  These clients still TRAIN -- pair with the
    data layer's `backdoor=` to make them trigger->target poisoners.
    `noise_adders` = client indices whose REAL update gets Gaussian grad-noise added
    (`_add_gnoise`, sigma = noise_gamma * delta RMS -- the Track H R4 threat).
    `select_fn` / `weights_fn` = the `_fedavg_core` intervention seam (Track D online
    arms; fl/intervene builds these).  Defaults None -> bit-identical vanilla FedAvg.
    """
    seed_everything(seed)                                       # LLM: no cudnn-det
    init_state = {n: p.detach().clone() for n, p in model.named_parameters()
                  if p.requires_grad}
    sample_nums = [len(ds) for ds in local_datasets]
    fr_gen = torch.Generator().manual_seed(seed + 1)            # reproducible free-rider random stream
    gn_gen = torch.Generator().manual_seed(seed + 2)            # reproducible grad-noise stream
    logs = []
    # E7 delta free-rider: during round r the logs hold rounds 0..r-1, so
    # logs[-1][0] = w^{r-1}; free_rider(ref=w^r, prev_state=w^{r-1}) recycles the
    # realized previous aggregate (what any client can observe from two broadcasts).
    ltf = _make_local_train_fn(model, tokenizer, local_datasets, lr, max_steps,
                               batch_size, max_length, seed, formatting_func,
                               free_riders, free_rider_mode, free_rider_scale, fr_gen,
                               scaled_attackers, attack_scale,
                               prev_state_fn=lambda: logs[-1][0] if logs else None,
                               noise_adders=noise_adders, noise_gamma=noise_gamma,
                               gn_gen=gn_gen)
    _fedavg_core(init_state, ltf, sample_nums, rounds, sample_frac, seed,
                 on_round=lambda r, w_r, dm: logs.append((w_r, dm)),
                 select_fn=select_fn, weights_fn=weights_fn)
    return logs
