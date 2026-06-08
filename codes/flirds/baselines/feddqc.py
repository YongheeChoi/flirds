"""FedDQC (Du et al. 2024) -- on-device data-quality detector via IRA.

Reference: Yaxin Du, Rui Ye, et al., "FedDQC: Data Quality Control in Federated
Instruction-tuning of Large Language Models" (arXiv:2410.11540).  Reference-guided
self-build: we take FedDQC's quality metric (IRA) as a DETECTION score (no Shapley ->
AUROC table only), matched to the DATA-QUALITY threat (answer_swap noisy client) per
§3.9.  FedDQC's hierarchical-training scheme is its data-USE method and is out of
scope here -- we only port the IRA scorer as the data-quality baseline.

IRA (Instruction-Response Alignment) per sample (q = prompt, a = completion):
    IRA(q, a) = L(a) - L(a | q)
the drop in the response's loss once the instruction is given (mutual-information-
flavored).  A clean sample has high IRA -- the instruction "explains" the answer; an
answer_swap noisy sample pairs a with the WRONG q, so the instruction adds nothing
(L(a|q) ~ L(a)) and IRA ~ 0 (low).  We aggregate per client (mean IRA over a sample)
and return the suspicious score = -mean_IRA, so corrupt = HIGH for
eval.metrics.detection_auroc (matching FLDetector/FLTrust/STD-DAGMM orientation).

NOT model-free and NOT logs-based: FedDQC scores raw CLIENT DATA with the (global)
model -- the on-device quality view, distinct from the update-vector detectors
(FLDetector/STD-DAGMM) and the val-gradient detectors (FLTrust/Flirds).  Privacy-
preserving in the FL sense (only the scalar IRA would leave the client); here we
compute it over the same client datasets for the AUROC benchmark.  Forward-only loss
(no HVP), so attn_implementation is unconstrained.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.llm import build_val_batch


@torch.no_grad()
def feddqc_scores(clients, model, tokenizer, device, n_samples=128, max_length=512, seed=0):
    """Per-client FedDQC suspicious score (corrupt = HIGH) = -mean IRA over a sample.

    clients:   list of HF Datasets with {prompt, completion} (the data layer's clients).
    model:     the (global) model used to score quality (eval mode; no grad).
    n_samples: per-client subsample size (FedDQC scores all data; we cap for cost --
               IRA is ~1% of training time, but the AUROC is stable on a sample).
    For each sampled record, IRA = L(a) - L(a|q):
      L(a|q) = completion-only loss with the real prompt q;
      L(a)   = completion-only loss with an EMPTY prompt (the response scored alone)."""
    model.eval()
    model.config.use_cache = False
    rng = np.random.default_rng(seed)
    scores = np.zeros(len(clients))
    for c, ds in enumerate(clients):
        n = min(n_samples, len(ds))
        idx = rng.choice(len(ds), size=n, replace=False)
        iras = []
        for i in idx:
            r = ds[int(i)]
            b_aq = build_val_batch([{"prompt": r["prompt"], "completion": r["completion"]}],
                                   tokenizer, max_length, device)
            b_a = build_val_batch([{"prompt": "", "completion": r["completion"]}],
                                  tokenizer, max_length, device)
            iras.append(model(**b_a).loss.item() - model(**b_aq).loss.item())   # L(a) - L(a|q)
        scores[c] = -float(np.mean(iras))                  # low IRA (noisy) -> high suspicion
    return scores
