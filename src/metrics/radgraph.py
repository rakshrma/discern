"""
RadGraph F1 — entity and relation extraction F1 via radgraph PyPI package.

Install: pip install radgraph

Returns three scores per pair (all stored in output JSON):
  radgraph_e        — entities only F1
  radgraph_er       — entities + relations F1  (main leaderboard column)
  radgraph_partial  — partial / bar-ER F1
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

try:
    from radgraph import F1RadGraph
    HAS_RADGRAPH = True
except ImportError:
    HAS_RADGRAPH = False

_scorer = None


def _get_scorer():
    global _scorer
    if _scorer is None:
        if not HAS_RADGRAPH:
            raise ImportError("radgraph not installed. Run: pip install radgraph")
        _scorer = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    return _scorer


def compute_radgraph(
    candidates: List[str],
    references: List[str],
    **_kwargs,
) -> Dict[str, List[Optional[float]]]:
    """
    Returns a dict with keys: radgraph_e, radgraph_er, radgraph_partial.
    Each value is a list of per-sample floats (or None on failure).
    The leaderboard uses radgraph_er as the primary column.
    """
    n = len(candidates)
    empty = {"radgraph_e": [None] * n, "radgraph_er": [None] * n, "radgraph_partial": [None] * n}

    if not HAS_RADGRAPH:
        warnings.warn("radgraph not installed. Run: pip install radgraph")
        return empty

    try:
        scorer = _get_scorer()
    except Exception as e:
        warnings.warn(f"[radgraph] failed to load scorer: {e}")
        return empty

    e_scores, er_scores, partial_scores = [], [], []
    for cand, ref in zip(candidates, references):
        try:
            mean_reward, _, _, _ = scorer(hyps=[cand], refs=[ref])
            rg_e, rg_er, rg_bar_er = mean_reward
            e_scores.append(float(rg_e))
            er_scores.append(float(rg_er))
            partial_scores.append(float(rg_bar_er))
        except Exception as exc:
            warnings.warn(f"[radgraph] pair failed: {exc}")
            e_scores.append(None)
            er_scores.append(None)
            partial_scores.append(None)

    return {
        "radgraph_e":       e_scores,
        "radgraph_er":      er_scores,
        "radgraph_partial": partial_scores,
    }
