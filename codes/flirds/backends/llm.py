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


def make_llm_loss(model, val_batches, device, chunk_domains=None, n_domains=None):
    """Return (loss_fn, pkeys, loss_chunks) for a PEFT-LoRA causal-LM.

    model:       a peft-wrapped causal LM (LoRA); put in eval mode here.
    val_batches: list of tokenized validation-chunk dicts already on `device`
                 (input_ids, attention_mask, labels; pad/prompt positions = -100).
                 A single dict is accepted and wrapped as one chunk.  The val set
                 is supplied in chunks because the estimator's 2nd-order term is a
                 forward-mode-AD HVP over EAGER attention (no flash/SDPA), whose
                 double-AD graph OOMs on the whole val set; chunking bounds peak
                 memory to one chunk.  The (b) oracle (@no_grad) reuses the same
                 chunks -- it is FLOP-bound, so a larger oracle chunk gives no
                 measured speedup (profiled ~1.0x), hence no separate oracle chunking.
    chunk_domains: optional per-chunk domain index (parallel to val_batches; built
                 by data.llm.build_val_batches_by_domain).  None -> the val loss is
                 the TOKEN-weighted mean (default).  Given -> per-domain MACRO-average
                 (each of n_domains domains weighted 1/D) so a long-completion domain
                 can't dominate -> cross-domain valuation fairness (ablation lever).
    loss_fn(params, buffers) -> scalar: Σ_c weight_c · mean_c, the (token- or
                 domain-)weighted mean val loss.  Used by the (b) oracle, whose
                 @no_grad forward frees each chunk -> memory-bounded, NO oracle change.
    loss_chunks: list of (lf_c, weight_c); lf_c(params, buffers) -> the per-chunk
                 MEAN loss, weight_c its normalization weight.  The estimator sums
                 weight_c · grad/HVP(lf_c) across chunks (peak mem = one chunk) ->
                 exactly the full-val grad/HVP of loss_fn (linear; NOT an approximation).
    pkeys:       LoRA param names (requires_grad=True), the Taylor variables.
    """
    if isinstance(val_batches, dict):
        val_batches = [val_batches]
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

    def _make_lf(batch):
        def lf(params, buffers):
            return functional_call(model, (params, buffers), args=(), kwargs=batch).loss
        return lf

    lfs = [_make_lf(b) for b in val_batches]
    # n_c = HF's contributing-token count per chunk (shifted targets labels[:, 1:] != -100)
    n_tok = [int((b["labels"][:, 1:] != -100).sum()) for b in val_batches]

    if chunk_domains is None:                          # token-weighted mean (default)
        ntot = float(sum(n_tok)) or 1.0
        weights = [n / ntot for n in n_tok]
    else:                                              # per-domain MACRO-average (equal domain weight)
        nd = {}
        for n, d in zip(n_tok, chunk_domains):
            nd[d] = nd.get(d, 0) + n
        ndom = n_domains or len(nd)
        weights = [n / (ndom * (nd[d] or 1)) for n, d in zip(n_tok, chunk_domains)]

    loss_chunks = list(zip(lfs, weights))              # (lf_c, weight_c); lf_c -> per-chunk MEAN loss

    def loss_fn(params, buffers):                      # Σ_c weight_c · mean_c (token- or domain-weighted)
        return sum(w * lf(params, buffers) for lf, w in loss_chunks)

    return loss_fn, pkeys, loss_chunks
