"""BLEU-1, ROUGE-L, METEOR — pure algorithmic NLP metrics."""

from __future__ import annotations

import warnings
from typing import List, Optional

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

try:
    from rouge import Rouge as RougeScorer
    _rouge = RougeScorer()
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    warnings.warn("rouge package not found. Install with: pip install rouge")


def _ensure_nltk():
    for resource in ("punkt", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def compute_bleu(candidates: List[str], references: List[str]) -> List[Optional[float]]:
    _ensure_nltk()
    smooth = SmoothingFunction().method1
    scores = []
    for cand, ref in zip(candidates, references):
        hyp = cand.lower().split()
        ref_tok = [ref.lower().split()]
        scores.append(float(sentence_bleu(ref_tok, hyp, weights=(1, 0, 0, 0), smoothing_function=smooth)))
    return scores


def compute_rouge(candidates: List[str], references: List[str]) -> List[Optional[float]]:
    if not HAS_ROUGE:
        return [None] * len(candidates)
    scores = []
    for cand, ref in zip(candidates, references):
        try:
            result = _rouge.get_scores(cand or ".", ref or ".")
            scores.append(result[0]["rouge-l"]["f"])
        except Exception:
            scores.append(None)
    return scores


def compute_meteor(candidates: List[str], references: List[str]) -> List[Optional[float]]:
    _ensure_nltk()
    scores = []
    for cand, ref in zip(candidates, references):
        try:
            scores.append(meteor_score([ref.split()], cand.split()))
        except Exception:
            scores.append(None)
    return scores
