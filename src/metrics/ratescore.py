"""
RaTEScore — radiology-specific entity-level metric.

Install: pip install RaTEScore
Source:  https://github.com/MAGIC-AI4Med/RaTEScore
"""

from __future__ import annotations

import warnings
from typing import List, Optional

try:
    from RaTEScore import RaTEScore as _RaTEScorer
    HAS_RATESCORE = True
except ImportError:
    HAS_RATESCORE = False
    warnings.warn("RaTEScore not found. Install with: pip install RaTEScore")

_scorer = None


def _get_scorer():
    global _scorer
    if _scorer is None:
        _scorer = _RaTEScorer()
    return _scorer


def compute_ratescore(
    candidates: List[str],
    references: List[str],
) -> List[Optional[float]]:
    if not HAS_RATESCORE:
        return [None] * len(candidates)
    try:
        scorer = _get_scorer()
        scores = scorer.compute_score(candidates, references)
        if hasattr(scores, "tolist"):
            return scores.tolist()
        return list(scores)
    except Exception as e:
        warnings.warn(f"RaTEScore failed: {e}")
        return [None] * len(candidates)
