"""LLM backend: builds the (loss_fn, pkeys) pair the estimator/oracle consume.

Mirrors backends/cnn.py for a PEFT-LoRA causal-LM.  pkeys = LoRA params only
(requires_grad=True); the frozen base stays inside the captured model.  The
logged w_r carries LoRA params ONLY (the multi-GB base is never streamed into
`logs`), so the estimator's buffers split is empty and functional_call pulls the
base + buffers (rotary inv_freq, etc.) from the captured model.  The same
closure is shared by the grad-tracked estimator (jvp∘grad) and the no_grad
oracle, exactly as on the CNN side.

fp32 eval forward (protocol 1) for HVP numerical stability — build the model in
float32; bf16 is a training-only concern that lives in the FL loop, not here.

The model MUST be built with attn_implementation="eager".  The estimator's
2nd-order term is a forward-mode-AD HVP (jvp∘grad), and the SDPA / flash
attention kernels don't implement forward AD (NotImplementedError on
_scaled_dot_product_efficient_attention).  eager attention is pure-torch ops
that forward AD supports; the val forward is short so the eager cost is small.
(Fallback if eager ever blocks: switch the estimator HVP to double-backward /
reverse-over-reverse — verified jvp-equivalent to 9.8e-6 at Phase 0.5.)
"""
from __future__ import annotations

from torch.func import functional_call


def make_llm_loss(model, val_batch, device):
    """Return (loss_fn, pkeys) for a PEFT-LoRA causal-LM.

    model:     a peft-wrapped causal LM (LoRA); put in eval mode here.
    val_batch: dict of tokenized validation tensors already on `device`
               (input_ids, attention_mask, labels; pad positions = -100).
    loss_fn(params, buffers) -> scalar causal-LM loss with `params` (LoRA)
               injected via functional_call; base + buffers from the model.
    pkeys:     LoRA param names (requires_grad=True), the Taylor variables.
    """
    model.eval()                       # LoRA/attention dropout off
    model.config.use_cache = False     # functional_call: no kv-cache side effects
    # functorch (jvp/grad) forbids requires_grad_() inside the transform; the
    # input-require-grad hook the trainer registers for gradient checkpointing
    # calls it during forward.  Strip it for this eval/HVP closure (training is
    # already done by the time loss_fn is built).
    try:
        model.disable_input_require_grads()           # drop the handle if present
    except Exception:
        pass
    emb = model.get_input_embeddings()                # the hook lives on the embedding
    if emb is not None:                               # module's forward hooks; clear them
        emb._forward_hooks.clear()                    # (make_inputs_require_grads = functorch-blocked)
    pkeys = [n for n, p in model.named_parameters() if p.requires_grad]

    def loss_fn(params, buffers):
        out = functional_call(model, (params, buffers), args=(), kwargs=val_batch)
        return out.loss

    return loss_fn, pkeys
