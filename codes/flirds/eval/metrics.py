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

import numpy as np
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


# ---- SV-fidelity distance metrics (Track C1; the GTG-Shapley trio, 2109.02053
# §5.1.3) ----  Raw-vector distances vs a ground-truth SV vector: only meaningful
# between estimates of the SAME game in the SAME units (rank metrics — Spearman /
# Kendall via scipy in the runner — cover the rest).

def cosine_distance(a, b):
    """1 - cosine similarity between value vectors (GTG 'Cosine Distance').
    NaN if either vector has zero norm (cosine undefined)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(1.0 - a @ b / denom)


def euclidean_distance(a, b):
    """L2 distance between value vectors (GTG 'Euclidean Distance')."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))


def max_difference(a, b):
    """Max absolute per-client difference (GTG 'Maximum Difference')."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.abs(a - b).max())


def pearson(a, b):
    """Pearson linear correlation of two value vectors — value-level fidelity that
    rank metrics miss: it's affine-invariant (forgives a global scale/offset the
    Euclidean/max distances penalise) yet, unlike Spearman/Kendall, rewards LINEAR
    value agreement, not just matching ranks.  Most useful where rank correlation
    saturates at +1 (the N=5 near-additive regime).  NaN if either vector is
    constant (correlation undefined); small-N (e.g. 5 clients) -> high variance."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def detection_auroc(phi, corrupt_labels):
    """AUROC of per-client phi as a corrupt-client (noisy / free-rider) detector.

    Sign convention (phase05_flirds_oracle): higher phi = more val-loss = worse,
    so a corrupt client scores high; corrupt_labels[k]=1 marks a corrupt client.
    Needs both classes present (else roc_auc_score raises)."""
    return float(roc_auc_score(corrupt_labels, phi))
