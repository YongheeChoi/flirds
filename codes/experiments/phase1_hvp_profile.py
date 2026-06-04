"""Phase 1 HVP memory/time profile: feasible (val-chunk, seq) settings.

The estimator's 2nd-order term is a forward-mode-AD HVP over EAGER attention
(O(seq^2) memory; flash/SDPA don't implement forward AD).  Val micro-batching
bounds peak memory to ONE chunk -- so the (chunk_size x seq) grid below tells you
the largest chunk that fits and the per-chunk HVP cost; the total val count then
scales only the chunk COUNT (estimator time), not peak memory.  Use this to size
fast-iteration runs vs the val=1000 clean run.  1B fp32 + eager; dummy tokens
(attention memory is data-independent).
"""
import os
import time

import torch
from peft import LoraConfig, get_peft_model
from torch.func import functional_call, grad, jvp
from transformers import AutoModelForCausalLM

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SEQS = [128, 256, 384, 512]
CHUNKS = [8, 16, 32, 64]


def main():
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 attn_implementation="eager").to(device)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=TARGET,
                                             lora_dropout=0.0, task_type="CAUSAL_LM"))
    model.eval()
    model.config.use_cache = False
    try:
        model.disable_input_require_grads()
    except Exception:
        pass
    emb = model.get_input_embeddings()
    if emb is not None:
        emb._forward_hooks.clear()

    pkeys = [n for n, p in model.named_parameters() if p.requires_grad]
    params = {n: p.detach().float() for n, p in model.named_parameters() if p.requires_grad}
    buffers = {n: b.detach() for n, b in model.named_buffers()}
    dW = {n: torch.randn_like(params[n]) for n in pkeys}
    vocab = int(model.config.vocab_size)

    print(f"{'seq':>5} {'chunk':>6} {'peakGB':>8} {'hvp_ms':>8}   (1B fp32 eager, B200 ~178GB)")
    for seq in SEQS:
        for ch in CHUNKS:
            ids = torch.randint(0, vocab, (ch, seq), device=device)
            labels = ids.clone()
            labels[:, :seq // 2] = -100                   # half prompt-masked
            batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}

            def vloss(pp):
                return functional_call(model, (pp, buffers), args=(), kwargs=batch).loss

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                torch.cuda.synchronize()
                t0 = time.time()
                g, u = jvp(grad(vloss), (params,), (dW,))
                torch.cuda.synchronize()
                dt = (time.time() - t0) * 1000
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"{seq:>5} {ch:>6} {peak:>8.1f} {dt:>8.0f}", flush=True)
                del g, u
            except RuntimeError as e:                     # OOM arrives as RuntimeError via TorchScript
                if "out of memory" not in str(e).lower():
                    raise
                print(f"{seq:>5} {ch:>6} {'OOM':>8} {'-':>8}", flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
