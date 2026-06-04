"""Downstream task metrics for the #7 clean run (free-form 5-domain, FedHDS-grounded).

ROUGE-L F1 (all domains) + multiple-choice exact-match (math / AQUA), plus the
noisy/free-rider detection AUROC read straight off the per-client phi.

Why these: FedHDS -- the closest FL+LLM instruction-tuning precedent (== our
cross-device track) -- scores held-out generation with ROUGE-L; math (AQUA) has a
clean final-answer letter so it additionally gets exact-match (the native metric,
as LESS/DsDm use task-accuracy where it is unambiguous).  These are the DOWNSTREAM
task metrics (selection-convergence / task-acc), DELIBERATELY distinct from the
val-loss UTILITY the estimator/oracle use (the utility-vs-task-metric separation
prior art maintains -- FedDQC's IRA selection-metric vs its task eval).

Pure functions (no model / torch) so they unit-test standalone; generation lives
in eval.generate.
"""
from __future__ import annotations

import re

from sklearn.metrics import roc_auc_score


def _tokens(s):
    return s.lower().split()


def _lcs_len(a, b):
    """Length of the longest common subsequence of token lists a, b (DP, O(mn))."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        ai = a[i - 1]
        for j in range(1, n + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[n]


def rouge_l(pred, ref):
    """ROUGE-L F1 (word-level LCS, beta=1) between a prediction and a reference."""
    p_tok, r_tok = _tokens(pred), _tokens(ref)
    lcs = _lcs_len(p_tok, r_tok)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p_tok), lcs / len(r_tok)
    return 2 * prec * rec / (prec + rec)


# Prefer the last "(Correct) answer (is|-|:|=) X" (X a word-bounded A-E, upper or
# lower), else the last standalone UPPERCASE A-E.  The answer letter MUST be
# \b-bounded so "answer choice/each/..." cannot capture the next word's first
# letter (a re.I-on-[A-E] would, wrongly); the uppercase-only fallback dodges the
# "a"/"A" article.
_ANS = re.compile(r"[Aa]nswer\s*(?:is|are|[-:=])?\s*\(?([A-Ea-e])\b")
_CHOICE = re.compile(r"\b([A-E])\b")


def extract_choice(text):
    """The multiple-choice letter (A-E) a text settles on: prefer the last
    '...answer ... X' phrasing (AQUA rationales end 'Correct answer - A'), else
    the last standalone A-E.  None if none found (counts as wrong in choice_match)."""
    phrased = _ANS.findall(text)
    if phrased:
        return phrased[-1].upper()
    standalone = _CHOICE.findall(text)
    return standalone[-1] if standalone else None


def choice_match(pred, ref):
    """Multiple-choice exact-match: do prediction and reference name the same A-E
    answer?  False if either has no extractable letter."""
    pc = extract_choice(pred)
    return pc is not None and pc == extract_choice(ref)


def detection_auroc(phi, corrupt_labels):
    """AUROC of per-client phi as a corrupt-client (noisy / free-rider) detector.

    Sign convention (phase05_flirds_oracle): higher phi = more val-loss = worse,
    so a corrupt client scores high; corrupt_labels[k]=1 marks a corrupt client.
    Needs both classes present (else roc_auc_score raises)."""
    return float(roc_auc_score(corrupt_labels, phi))
