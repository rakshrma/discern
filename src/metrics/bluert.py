"""BLUERT score. Install with: pip install bleurt"""

from __future__ import annotations

import warnings
from typing import List, Optional

try:
    from bleurt import score as _bluert_score
    HAS_BLUERT = True
except ImportError:
    HAS_BLUERT = False
    warnings.warn("bleurt not found. Install with: pip install bleurt")

_scorer = None


def compute_bluert(
    candidates: List[str],
    references: List[str],
    checkpoint: str = "bleurt-base-128",
) -> List[Optional[float]]:
    if not HAS_BLUERT:
        return [None] * len(candidates)
    global _scorer
    if _scorer is None:
        _scorer = _bluert_score.BleurtScorer(checkpoint)
    try:
        scores = _scorer.score(references=references, candidates=candidates)
        return [float(s) for s in scores]
    except Exception as e:
        warnings.warn(f"BLUERT failed: {e}")
        return [None] * len(candidates)
