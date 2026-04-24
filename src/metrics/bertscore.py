"""BERTScore using distilroberta-base (matches CXR-Report-Metric default)."""

from __future__ import annotations

import warnings
from typing import List, Optional

try:
    from bert_score import score as _bertscore
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    warnings.warn("bert-score not found. Install with: pip install bert-score")


def compute_bertscore(
    candidates: List[str],
    references: List[str],
    model_type: str = "distilroberta-base",
    lang: str = "en",
    rescale_with_baseline: bool = True,
) -> List[Optional[float]]:
    if not HAS_BERTSCORE:
        return [None] * len(candidates)
    _, _, F = _bertscore(
        candidates,
        references,
        model_type=model_type,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
        verbose=False,
    )
    return F.tolist()
