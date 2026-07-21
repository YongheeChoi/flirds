"""MMLU 0-shot likelihood evaluation (Track D downstream metric; plan §3.11 D).

The close-ended local stand-in for the GPT-judge evaluations our API-free
constraint rules out: FlowerTune's general track trains on Alpaca-GPT4 and
scores MMLU locally -- the precedent Track D mirrors.  Scoring is the original-
harness likelihood convention, generation-free: each question is formatted with
the canonical subject header + lettered choices, ONE batched forward is run, and
the model's answer is the argmax over the four single-token continuations
" A".." D" at the final prompt position.  0-shot (the Track D arms are compared
against EACH OTHER -- mixed / clean-oracle / filtered / random; few-shot buys
prompt length, not arm discrimination; Yonghee 2026-06-12).

`mmlu_accuracy` evaluates a cais/mmlu split (test = 14,042 over 57 subjects =
the literature-standard figure) and returns (overall_acc, per_subject, n);
`limit` subsamples (shuffle seed 0 -> subject-mixed) for smokes / checkpoints.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from datasets import load_dataset

from ..hf_pin import rev

_LETTERS = "ABCD"


def format_mmlu(question, choices, subject):
    """Canonical MMLU prompt (original-harness format): subject header +
    question + lettered choices + 'Answer:' (the letter logit is read there)."""
    lines = "".join(f"{l}. {c}\n" for l, c in zip(_LETTERS, choices))
    return (f"The following are multiple choice questions (with answers) about "
            f"{subject.replace('_', ' ')}.\n\n{question.strip()}\n{lines}Answer:")


@torch.no_grad()
def mmlu_accuracy(model, tokenizer, device, split="test", limit=0, batch_size=16,
                  max_prompt_len=2048):
    """Accuracy of `model` on cais/mmlu `split` by letter-token likelihood.

    Left-padded batched forwards (prompts flush-right so the last position is
    'Answer:'); prediction = argmax over the ' A'..' D' token logits there.
    Over-long prompts are LEFT-truncated (the lettered tail must survive), with
    max length clamped to the model's context window (gpt2 smoke: 1024).
    Returns (acc, {subject: acc}, n_questions).
    """
    ds = load_dataset("cais/mmlu", "all", split=split, revision=rev("cais/mmlu"))
    if limit:
        ds = ds.shuffle(seed=0).select(range(min(limit, len(ds))))
    letter_ids = []
    for l in _LETTERS:
        ids = tokenizer(" " + l, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"' {l}' must be a single token for likelihood scoring"
        letter_ids.append(ids[0])

    model.eval()
    max_len = min(max_prompt_len,
                  getattr(model.config, "max_position_embeddings", max_prompt_len))
    side, trunc = tokenizer.padding_side, tokenizer.truncation_side
    tokenizer.padding_side, tokenizer.truncation_side = "left", "left"
    hit, per = 0, defaultdict(lambda: [0, 0])
    for i in range(0, len(ds), batch_size):
        rows = ds[i:i + batch_size]                    # dict of column lists
        prompts = [format_mmlu(q, ch, s) for q, ch, s in
                   zip(rows["question"], rows["choices"], rows["subject"])]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_len).to(device)
        logits = model(**enc).logits[:, -1, letter_ids]          # [B, 4]
        for p, gold, subj in zip(logits.argmax(-1).tolist(), rows["answer"],
                                 rows["subject"]):
            per[subj][1] += 1
            if p == gold:
                hit += 1
                per[subj][0] += 1
    tokenizer.padding_side, tokenizer.truncation_side = side, trunc
    return hit / len(ds), {s: c / n for s, (c, n) in sorted(per.items())}, len(ds)
