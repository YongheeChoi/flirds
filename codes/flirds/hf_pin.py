"""HF Hub revision pinning — reproducibility anchor for datasets & base weights.

`load_dataset(id, ...)` / `from_pretrained(id, ...)` without `revision=` fetch the
Hub's mutable *latest* commit, so the exact training/validation data and base weights
are not recoverable from the code alone (a community dataset can be re-uploaded or
removed).  Wire `revision=rev(id)` into every load; `rev` returns None (current
behavior) until a commit SHA is pinned here.

To FREEZE the environment for a paper release, fill `REVISIONS` with the exact commit
SHAs (resolve on a networked machine, e.g.:
    from huggingface_hub import HfApi
    HfApi().dataset_info("vicgalle/alpaca-gpt4").sha
    HfApi().model_info("meta-llama/Llama-3.2-1B-Instruct").sha
), or set HF_REVISION_DEFAULT in the environment to pin every load to one SHA.
"""
from __future__ import annotations

import os

# id -> pinned commit SHA (empty = Hub latest, i.e. unchanged behavior).  Fill for a
# reproducible release.  Datasets and models share this map (ids are globally unique).
REVISIONS: dict[str, str] = {
    # "vicgalle/alpaca-gpt4": "<sha>",
    # "openai/gsm8k": "<sha>",
    # "cais/mmlu": "<sha>",
    # "meta-llama/Llama-3.2-1B-Instruct": "<sha>",
}


def rev(hf_id):
    """Pinned revision for `hf_id`, or None (Hub latest).  Precedence: REVISIONS map,
    then HF_REVISION_DEFAULT env (one SHA for all loads)."""
    return REVISIONS.get(hf_id) or os.environ.get("HF_REVISION_DEFAULT") or None
