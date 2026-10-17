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

    with patch("evaluate_single_prompt.query_llm", return_value=SAMPLE_MINI_OUTPUT):
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

    with patch("evaluate_single_prompt.query_llm", return_value=SAMPLE_MINI_DISCORDANT):
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

    with patch("evaluate_single_prompt.query_llm", return_value=SAMPLE_MINI_OUTPUT):
        result = evaluate_reports(
            reference_report="Normal chest X-ray.",
            candidate_report="Normal.",
            entity_list_path=DIAG_YAML,
            prompt_path=MERGED_PROMPT,
            model="databricks-claude-sonnet-4-6",
            token_path="fake_token",
        )

    assert isinstance(result, list)


# ── free-DISCERN (single prompt, no entity list) ──────────────────────────────

FREE_PROMPT = str(_CONFIG / "freeform_prompt.yaml")

SAMPLE_FREE_OUTPUT = json.dumps([
    {
        "finding": "Pneumonia",
        "reference_presence": "POSITIVE",
        "candidate_presence": "NEGATIVE",
        "reference_text": "Pneumonia in right lower lobe.",
        "candidate_text": "Lungs are clear.",
        "diagnosis_concordance": "discordant",
        "location_concordance": "candidate-misses",
        "severity_concordance": "not mentioned",
        "temporal_concordance": "not mentioned",
        "clinical_significance_score": 4,
        "rationale": "Candidate negates pneumonia present in the reference.",
    },
    {
        "finding": "Pleural effusion",
        "reference_presence": "POSITIVE",
        "candidate_presence": "NOT MENTIONED",
        "reference_text": "Small left pleural effusion.",
        "candidate_text": None,
        "diagnosis_concordance": "candidate-misses",
        "location_concordance": "candidate-misses",
        "severity_concordance": "candidate-misses",
        "temporal_concordance": "not mentioned",
        "clinical_significance_score": 2,
        "rationale": "Small effusion omitted.",
    },
])


def test_free_discern_scores_listed_errors():
    """free-DISCERN validates error items (no entity list) and the score is their sum."""
    from evaluate_single_prompt import evaluate_reports

    with patch("evaluate_single_prompt.query_llm", return_value=SAMPLE_FREE_OUTPUT):
        result = evaluate_reports(
            reference_report="Pneumonia in right lower lobe. Small left pleural effusion.",
            candidate_report="Lungs are clear.",
            entity_list_path=None,
            prompt_path=FREE_PROMPT,
            model="databricks-claude-sonnet-4-6",
            token_path="fake_token",
        )

    assert [e.finding for e in result] == ["Pneumonia", "Pleural effusion"]
    assert sum(e.clinical_significance_score for e in result) == 6


def test_free_discern_no_errors_scores_zero():
    """An empty error list is valid and scores 0."""
    from evaluate_single_prompt import evaluate_reports

    with patch("evaluate_single_prompt.query_llm", return_value="[]"):
        result = evaluate_reports(
            reference_report="Normal chest X-ray.",
            candidate_report="Normal chest X-ray.",
            entity_list_path=None,
            prompt_path=FREE_PROMPT,
            model="databricks-claude-sonnet-4-6",
            token_path="fake_token",
        )

    assert result == []


def test_prompt_and_entity_list_must_agree():
    """The mini-DISCERN prompt needs an entity list; the free-DISCERN prompt takes none."""
    from evaluate_single_prompt import evaluate_reports

    with pytest.raises(ValueError):
        evaluate_reports(
            reference_report="Normal.", candidate_report="Normal.",
            entity_list_path=None, prompt_path=MERGED_PROMPT,
            model="databricks-claude-sonnet-4-6", token_path="fake_token",
        )
    with pytest.raises(ValueError):
        evaluate_reports(
            reference_report="Normal.", candidate_report="Normal.",
            entity_list_path=DIAG_YAML, prompt_path=FREE_PROMPT,
            model="databricks-claude-sonnet-4-6", token_path="fake_token",
        )
