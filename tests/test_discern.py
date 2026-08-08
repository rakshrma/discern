"""Tests for DISCERN pipeline stages (uses mocked LLM calls)."""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_CONFIG = Path(__file__).parent.parent / "config"
ENTITIES_YAML = str(_CONFIG / "entities.yaml")
DIAG_YAML     = str(_CONFIG / "diagnosis.yaml")
MERGED_PROMPT = str(_CONFIG / "merged_prompt.yaml")


# ── Mini-DISCERN (single-prompt) ──────────────────────────────────────────────

SAMPLE_MINI_OUTPUT = json.dumps([
    {
        "entity_name": "Pneumonia",
        "reference_presence": "POSITIVE",
        "candidate_presence": "POSITIVE",
        "diagnosis_concordance": "concordant",
        "location_concordance": "concordant",
        "severity_concordance": "concordant",
        "temporal_concordance": "not mentioned",
        "clinical_significance_score": 0,
        "rationale": "Both reports agree on pneumonia.",
    }
])


def test_mini_discern_concordant():
    """Mini-DISCERN returns score=0 when reports fully agree."""
    from evaluate_single_prompt import evaluate_reports

    with patch("llm_backend.query_llm", return_value=SAMPLE_MINI_OUTPUT):
        result = evaluate_reports(
            reference_report="Pneumonia in left lower lobe.",
            candidate_report="Pneumonia in left lower lobe.",
            entity_list_path=DIAG_YAML,
            prompt_path=MERGED_PROMPT,
            model="databricks-claude-sonnet-4-6",
            token_path="fake_token",
        )

    assert len(result) >= 1
    total = sum(e.clinical_significance_score for e in result)
    assert total == 0


SAMPLE_MINI_DISCORDANT = json.dumps([
    {
        "entity_name": "Pneumonia",
        "reference_presence": "POSITIVE",
        "candidate_presence": "NEGATIVE",
        "diagnosis_concordance": "candidate-misses",
        "location_concordance": "concordant",
        "severity_concordance": "concordant",
        "temporal_concordance": "not mentioned",
        "clinical_significance_score": 3,
        "rationale": "Candidate missed pneumonia — clinically important.",
    }
])


def test_mini_discern_discordant():
    """Mini-DISCERN returns non-zero score when reports disagree."""
    from evaluate_single_prompt import evaluate_reports

    with patch("llm_backend.query_llm", return_value=SAMPLE_MINI_DISCORDANT):
        result = evaluate_reports(
            reference_report="Pneumonia in right lower lobe.",
            candidate_report="Lungs are clear.",
            entity_list_path=DIAG_YAML,
            prompt_path=MERGED_PROMPT,
            model="databricks-claude-sonnet-4-6",
            token_path="fake_token",
        )

    total = sum(e.clinical_significance_score for e in result)
    assert total > 0


def test_mini_discern_returns_list():
    """Mini-DISCERN always returns a list (even for empty output)."""
    from evaluate_single_prompt import evaluate_reports

    with patch("llm_backend.query_llm", return_value=SAMPLE_MINI_OUTPUT):
        result = evaluate_reports(
            reference_report="Normal chest X-ray.",
            candidate_report="Normal.",
            entity_list_path=DIAG_YAML,
            prompt_path=MERGED_PROMPT,
            model="databricks-claude-sonnet-4-6",
            token_path="fake_token",
        )

    assert isinstance(result, list)
