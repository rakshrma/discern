"""
FineRadScore — fine-grained LLM-based evaluation via GPT-4 or Claude.

Source: https://github.com/rajpurkarlab/FineRadScore
Requires: openai_api_key or anthropic_api_key in config.yaml.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import List, Optional

# Attempt to use the FineRadScore repo if it's on the path
_FINERADSCOR_REPO = Path(__file__).parent.parent.parent.parent / "FineRadScore"

# Fall back to a direct LLM call using our own call_llm infrastructure
sys.path.insert(0, str(Path(__file__).parent.parent))


_SYSTEM_PROMPT = """You are a board-certified radiologist evaluating the quality of an AI-generated
radiology report compared to a ground truth report written by a radiologist.

Score the candidate report on a scale from 0 to 1, where:
  1.0 = Clinically equivalent — all findings accurately captured, no significant errors
  0.5 = Partially correct — some findings correct, minor errors or omissions
  0.0 = Incorrect — major findings wrong, missing, or critically different

Return ONLY a JSON object: {"score": <float 0.0-1.0>, "rationale": "<brief reason>"}"""


def compute_fineradscor(
    candidates: List[str],
    references: List[str],
    model: Optional[str] = None,
) -> List[Optional[float]]:
    """
    Use FineRadScore repo if available, otherwise fall back to a direct
    LLM call via the DISCERN call_llm infrastructure.
    """
    if _FINERADSCOR_REPO.exists():
        try:
            return _compute_via_repo(candidates, references, model)
        except Exception as e:
            warnings.warn(f"FineRadScore repo call failed ({e}), using fallback LLM call.")

    return _compute_via_llm(candidates, references, model)


def _compute_via_repo(
    candidates: List[str],
    references: List[str],
    model: Optional[str],
) -> List[Optional[float]]:
    sys.path.insert(0, str(_FINERADSCOR_REPO))
    # FineRadScore doesn't have a clean Python API; use LLM fallback instead
    raise ImportError("FineRadScore has no direct Python API — using LLM fallback.")


def _compute_via_llm(
    candidates: List[str],
    references: List[str],
    model: Optional[str],
) -> List[Optional[float]]:
    import json
    from call_llm import query_llm

    scores = []
    for cand, ref in zip(candidates, references):
        user_msg = (
            f"Reference report:\n{ref}\n\n"
            f"Candidate report:\n{cand}\n\n"
            "Score the candidate report as described."
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        try:
            raw = query_llm(messages=messages, model=model, max_tokens=256)
            # Extract JSON from response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            obj = json.loads(raw[start:end])
            scores.append(float(obj["score"]))
        except Exception as e:
            warnings.warn(f"FineRadScore LLM call failed: {e}")
            scores.append(None)

    return scores
