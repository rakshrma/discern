"""
DISCERN test suite.

Unit tests (no LLM calls) run by default.
Integration tests (require LLM) are marked with @pytest.mark.integration
and use meta-llama/Llama-3.1-8B-Instruct via HuggingFace.

Run all:           pytest tests/test_discern.py -v
Run unit only:     pytest tests/test_discern.py -v -m "not integration"
Run integration:   pytest tests/test_discern.py -v -m integration
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup – src/ must be on sys.path for all imports to resolve
# ---------------------------------------------------------------------------
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "config")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Model used for integration tests
LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN_PATH = str(Path(__file__).resolve().parent.parent / "config" / ".hftoken")
ENTITY_PROMPT_PATH = str(Path(CONFIG_DIR) / "entity_extraction_prompt.yaml")
ENTITIES_YAML_PATH = str(Path(CONFIG_DIR) / "entities.yaml")
ATTRIBUTE_PROMPT_PATH = str(Path(CONFIG_DIR) / "attribute_extraction_prompt.yaml")
SIGNIFICANCE_PROMPT_PATH = str(Path(CONFIG_DIR) / "significance_prompt.yaml")

# Short radiology report fixtures
GROUND_TRUTH_REPORT = (
    "FINDINGS: The cardiac silhouette is enlarged. "
    "There is a right-sided pleural effusion. "
    "No pneumothorax is identified. "
    "The mediastinum is within normal limits."
)

CANDIDATE_REPORT_CONCORDANT = (
    "FINDINGS: The cardiac silhouette appears enlarged. "
    "A right pleural effusion is present. "
    "No pneumothorax seen. "
    "Mediastinum is normal."
)

CANDIDATE_REPORT_DISCORDANT = (
    "FINDINGS: The heart size is normal. "
    "No pleural effusion. "
    "There is a left-sided pneumothorax. "
    "Mediastinum is widened."
)


# ===========================================================================
# get_discern_score.py
# ===========================================================================
class TestParseFindings:
    def setup_method(self):
        from get_discern_score import parse_findings
        self.parse_findings = parse_findings

    def test_empty_string(self):
        assert self.parse_findings("") == []

    def test_nan(self):
        import pandas as pd
        assert self.parse_findings(pd.NA) == []

    def test_valid_list_string(self):
        data = [{"entity": "Heart :: Enlarged Cardiac Contour", "significance_score": 3}]
        result = self.parse_findings(str(data))
        assert result == data

    def test_invalid_string_returns_empty(self):
        assert self.parse_findings("not valid python") == []

    def test_dict_string_returns_empty(self):
        # A dict is not a list; should still parse without error
        result = self.parse_findings(str({"a": 1}))
        assert result == {"a": 1}  # ast.literal_eval succeeds, returns dict


class TestComputeEntityPenalty:
    def setup_method(self):
        from get_discern_score import compute_entity_penalty
        self.compute_entity_penalty = compute_entity_penalty

    def test_fully_concordant_zero_penalty(self):
        finding = {
            "diagnosis_concordance": "concordant",
            "location_concordance": "concordant",
            "severity_concordance": "concordant",
            "temporal_comparison": "concordant",
            "significance_score": 4,
        }
        assert self.compute_entity_penalty(finding) == 0.0

    def test_all_discordant_max_penalty(self):
        finding = {
            "diagnosis_concordance": "discordant",
            "location_concordance": "discordant",
            "severity_concordance": "discordant",
            "temporal_comparison": "discordant",
            "significance_score": 4,
        }
        # 4 dimensions × penalty 1 each × significance 4 = 16
        assert self.compute_entity_penalty(finding) == 16.0

    def test_partial_diagnosis_penalty(self):
        finding = {
            "diagnosis_concordance": "partial",
            "location_concordance": "concordant",
            "severity_concordance": "concordant",
            "temporal_comparison": "concordant",
            "significance_score": 2,
        }
        assert self.compute_entity_penalty(finding) == 2.0  # 1 × 2

    def test_candidate_adds_penalized(self):
        finding = {
            "diagnosis_concordance": "concordant",
            "location_concordance": "candidate-adds",
            "severity_concordance": "concordant",
            "temporal_comparison": "concordant",
            "significance_score": 3,
        }
        assert self.compute_entity_penalty(finding) == 3.0  # 1 × 3

    def test_candidate_misses_penalized(self):
        finding = {
            "diagnosis_concordance": "concordant",
            "location_concordance": "concordant",
            "severity_concordance": "candidate-misses",
            "temporal_comparison": "concordant",
            "significance_score": 2,
        }
        assert self.compute_entity_penalty(finding) == 2.0

    def test_zero_significance_no_penalty(self):
        finding = {
            "diagnosis_concordance": "discordant",
            "location_concordance": "discordant",
            "severity_concordance": "discordant",
            "temporal_comparison": "discordant",
            "significance_score": 0,
        }
        assert self.compute_entity_penalty(finding) == 0.0

    def test_missing_fields_default_to_zero(self):
        finding = {"significance_score": 3}
        assert self.compute_entity_penalty(finding) == 0.0

    def test_not_mentioned_no_penalty(self):
        finding = {
            "diagnosis_concordance": "not mentioned",
            "location_concordance": "not mentioned",
            "severity_concordance": "not mentioned",
            "temporal_comparison": "not mentioned",
            "significance_score": 4,
        }
        assert self.compute_entity_penalty(finding) == 0.0


class TestComputeDiscernScore:
    def setup_method(self):
        from get_discern_score import compute_discern_score
        self.compute_discern_score = compute_discern_score

    def test_empty_findings(self):
        assert self.compute_discern_score([]) == 0.0

    def test_single_concordant_entity(self):
        findings = [{
            "diagnosis_concordance": "concordant",
            "location_concordance": "concordant",
            "severity_concordance": "concordant",
            "temporal_comparison": "concordant",
            "significance_score": 3,
        }]
        assert self.compute_discern_score(findings) == 0.0

    def test_single_discordant_entity(self):
        findings = [{
            "diagnosis_concordance": "discordant",
            "location_concordance": "concordant",
            "severity_concordance": "concordant",
            "temporal_comparison": "concordant",
            "significance_score": 3,
        }]
        assert self.compute_discern_score(findings) == 3.0

    def test_multiple_entities_summed(self):
        findings = [
            {
                "diagnosis_concordance": "discordant",
                "location_concordance": "concordant",
                "severity_concordance": "concordant",
                "temporal_comparison": "concordant",
                "significance_score": 2,
            },
            {
                "diagnosis_concordance": "concordant",
                "location_concordance": "discordant",
                "severity_concordance": "discordant",
                "temporal_comparison": "concordant",
                "significance_score": 1,
            },
        ]
        # Entity 1: 1 × 2 = 2.0; Entity 2: 2 × 1 = 2.0; total = 4.0
        assert self.compute_discern_score(findings) == 4.0

    def test_returns_float(self):
        findings = [{"significance_score": 3, "diagnosis_concordance": "discordant"}]
        result = self.compute_discern_score(findings)
        assert isinstance(result, float)


class TestComputeCounts:
    def setup_method(self):
        from get_discern_score import compute_counts
        self.compute_counts = compute_counts

    def test_empty_findings_all_zero(self):
        counts = self.compute_counts([])
        assert all(v == 0 for v in counts.values())

    def test_discordant_diagnosis_significant_counted(self):
        finding = {
            "diagnosis_concordance": "discordant",
            "location_concordance": "concordant",
            "severity_concordance": "concordant",
            "temporal_comparison": "concordant",
            "significance_score": 3,
            "discrepancy_type": None,
        }
        counts = self.compute_counts([finding])
        assert counts["false_prediction_significant"] == 1
        assert counts["false_prediction_insignificant"] == 0

    def test_missing_in_candidate_significant(self):
        finding = {
            "discrepancy_type": "missing_in_candidate",
            "diagnosis_concordance": "concordant",
            "location_concordance": "concordant",
            "severity_concordance": "concordant",
            "temporal_comparison": "concordant",
            "significance_score": 2,
        }
        counts = self.compute_counts([finding])
        assert counts["omission_finding_significant"] == 1
        assert counts["omission_finding_insignificant"] == 0

    def test_location_discordant_insignificant(self):
        finding = {
            "diagnosis_concordance": "concordant",
            "location_concordance": "discordant",
            "severity_concordance": "concordant",
            "temporal_comparison": "concordant",
            "significance_score": 1,
            "discrepancy_type": None,
        }
        counts = self.compute_counts([finding])
        assert counts["incorrect_location_insignificant"] == 1
        assert counts["incorrect_location_significant"] == 0

    def test_temporal_candidate_adds_significant(self):
        finding = {
            "diagnosis_concordance": "concordant",
            "location_concordance": "concordant",
            "severity_concordance": "concordant",
            "temporal_comparison": "candidate-adds",
            "significance_score": 2,
            "discrepancy_type": None,
        }
        counts = self.compute_counts([finding])
        assert counts["extra_comparison_significant"] == 1


# ===========================================================================
# utils.py
# ===========================================================================
class TestMergeCommonWithMissingExtra:
    def setup_method(self):
        from utils import merge_common_with_missing_extra
        self.merge = merge_common_with_missing_extra

    def _entity(self, name: str, presence: str = "POSITIVE", sentence: str = "seen") -> Dict[str, Any]:
        return {"entity": name, "presence": presence, "sentence": sentence}

    def test_common_only_returns_attributions(self):
        common = [{"entity": "Heart :: Enlarged Cardiac Contour", "diagnosis_concordance": "concordant"}]
        cand = [self._entity("Heart :: Enlarged Cardiac Contour")]
        ref = [self._entity("Heart :: Enlarged Cardiac Contour")]
        result = self.merge(cand, ref, common)
        assert len(result) == 1
        assert result[0]["diagnosis_concordance"] == "concordant"

    def test_missing_in_candidate_appended(self):
        cand: List[Dict] = []
        ref = [self._entity("Heart :: Enlarged Cardiac Contour")]
        result = self.merge(cand, ref, [])
        assert len(result) == 1
        assert result[0]["discrepancy_type"] == "missing_in_candidate"
        assert result[0]["entity"] == "Heart :: Enlarged Cardiac Contour"

    def test_extra_in_candidate_appended(self):
        cand = [self._entity("Lung :: Pneumothorax")]
        ref: List[Dict] = []
        result = self.merge(cand, ref, [])
        assert len(result) == 1
        assert result[0]["discrepancy_type"] == "extra_in_candidate"
        assert result[0]["entity"] == "Lung :: Pneumothorax"

    def test_concordance_fields_default_not_mentioned(self):
        cand: List[Dict] = []
        ref = [self._entity("Heart :: Enlarged Cardiac Contour")]
        result = self.merge(cand, ref, [])
        row = result[0]
        for field in ["diagnosis_concordance", "location_concordance", "severity_concordance", "temporal_comparison"]:
            assert row[field] == "not mentioned"

    def test_include_presence_false(self):
        cand: List[Dict] = []
        ref = [self._entity("Heart :: Enlarged Cardiac Contour")]
        result = self.merge(cand, ref, [], include_presence=False)
        assert "reference_presence" not in result[0]
        assert "candidate_presence" not in result[0]

    def test_include_presence_true(self):
        cand: List[Dict] = []
        ref = [self._entity("Heart :: Enlarged Cardiac Contour")]
        result = self.merge(cand, ref, [], include_presence=True)
        assert result[0]["reference_presence"] == "POSITIVE"
        assert result[0]["candidate_presence"] == "NOT MENTIONED"

    def test_entity_in_common_not_re_added_as_missing(self):
        common = [{"entity": "Heart :: Enlarged Cardiac Contour", "diagnosis_concordance": "concordant"}]
        # ref has it, cand does not — but it IS in common, so no missing row
        cand: List[Dict] = []
        ref = [self._entity("Heart :: Enlarged Cardiac Contour")]
        result = self.merge(cand, ref, common)
        assert len(result) == 1
        assert "discrepancy_type" not in result[0]

    def test_whitespace_normalization_in_entity_keys(self):
        cand = [{"entity": "Heart  ::  Enlarged Cardiac Contour", "presence": "POSITIVE", "sentence": "x"}]
        ref = [{"entity": "Heart :: Enlarged Cardiac Contour", "presence": "POSITIVE", "sentence": "x"}]
        result = self.merge(cand, ref, [])
        # Both normalize to same key → neither is missing nor extra
        assert len(result) == 0

    def test_mixed_common_missing_extra(self):
        common = [{"entity": "Heart :: Enlarged Cardiac Contour", "diagnosis_concordance": "concordant"}]
        cand = [
            self._entity("Heart :: Enlarged Cardiac Contour"),
            self._entity("Lung :: Pneumothorax"),
        ]
        ref = [
            self._entity("Heart :: Enlarged Cardiac Contour"),
            self._entity("Pleural :: Pleural Effusion"),
        ]
        result = self.merge(cand, ref, common)
        types = [r.get("discrepancy_type") for r in result]
        assert None in types  # common
        assert "missing_in_candidate" in types
        assert "extra_in_candidate" in types


class TestMergeAttributesWithSignificance:
    def setup_method(self):
        from utils import merge_attributes_with_significance
        self.merge = merge_attributes_with_significance

    def _sig_output(self, entities: List[Dict]) -> Dict:
        return {"scored_entities": entities}

    def test_basic_merge(self):
        attrs = [{"entity": "Heart :: Enlarged Cardiac Contour", "diagnosis_concordance": "concordant"}]
        sig = self._sig_output([{"entity": "Heart :: Enlarged Cardiac Contour", "significance_score": 3, "rationale": "important"}])
        result = self.merge(attrs, sig)
        assert result[0]["significance_score"] == 3
        assert result[0]["rationale"] == "important"

    def test_missing_scored_entities_key_raises(self):
        with pytest.raises(ValueError, match="scored_entities"):
            self.merge([], {"entities": []})

    def test_strict_missing_score_raises(self):
        attrs = [{"entity": "Heart :: Enlarged Cardiac Contour"}]
        sig = self._sig_output([])
        with pytest.raises(ValueError, match="Missing significance scores"):
            self.merge(attrs, sig, strict=True)

    def test_strict_extra_score_raises(self):
        attrs: List[Dict] = []
        sig = self._sig_output([{"entity": "Lung :: Pneumothorax", "significance_score": 1, "rationale": ""}])
        with pytest.raises(ValueError, match="Extra significance scores"):
            self.merge(attrs, sig, strict=True)

    def test_non_strict_ignores_missing_score(self):
        attrs = [{"entity": "Heart :: Enlarged Cardiac Contour"}]
        sig = self._sig_output([])
        result = self.merge(attrs, sig, strict=False)
        assert result[0]["significance_score"] is None
        assert result[0]["rationale"] is None

    def test_non_strict_ignores_extra_score(self):
        attrs = [{"entity": "Heart :: Enlarged Cardiac Contour"}]
        sig = self._sig_output([
            {"entity": "Heart :: Enlarged Cardiac Contour", "significance_score": 2, "rationale": "x"},
            {"entity": "Lung :: Pneumothorax", "significance_score": 1, "rationale": "y"},
        ])
        result = self.merge(attrs, sig, strict=False)
        assert len(result) == 1
        assert result[0]["significance_score"] == 2

    def test_preserves_original_fields(self):
        attrs = [{"entity": "Heart :: Enlarged Cardiac Contour", "diagnosis_concordance": "discordant", "location_concordance": "concordant"}]
        sig = self._sig_output([{"entity": "Heart :: Enlarged Cardiac Contour", "significance_score": 4, "rationale": "critical"}])
        result = self.merge(attrs, sig)
        assert result[0]["diagnosis_concordance"] == "discordant"
        assert result[0]["location_concordance"] == "concordant"


# ===========================================================================
# extract_entities.py – pure helpers
# ===========================================================================
class TestSanitizeJsonText:
    def setup_method(self):
        from extract_entities import sanitize_json_text
        self.sanitize = sanitize_json_text

    def test_removes_trailing_comma_in_object(self):
        result = self.sanitize('{"a": 1,}')
        assert result == '{"a": 1}'

    def test_removes_trailing_comma_in_array(self):
        result = self.sanitize('[1, 2, 3,]')
        assert result == '[1, 2, 3]'

    def test_replaces_curly_quotes(self):
        result = self.sanitize('\u201chello\u201d')
        assert result == '"hello"'

    def test_strips_whitespace(self):
        result = self.sanitize('  [1, 2]  ')
        assert result == '[1, 2]'

    def test_plain_valid_json_unchanged(self):
        raw = '[{"entity": "Heart", "presence": "POSITIVE", "sentence": "enlarged"}]'
        result = self.sanitize(raw)
        assert result == raw


class TestExtractFirstJsonArray:
    def setup_method(self):
        from extract_entities import extract_first_json_array
        self.extract = extract_first_json_array

    def test_simple_array(self):
        text = 'Here is the result: [1, 2, 3]'
        assert self.extract(text) == '[1, 2, 3]'

    def test_returns_none_if_no_array(self):
        assert self.extract('no array here') is None

    def test_nested_array(self):
        text = '[[1, 2], [3, 4]]'
        assert self.extract(text) == '[[1, 2], [3, 4]]'

    def test_array_with_objects(self):
        text = 'prefix [{"a": 1}, {"b": 2}] suffix'
        assert self.extract(text) == '[{"a": 1}, {"b": 2}]'

    def test_array_with_escaped_quotes_in_string(self):
        text = r'[{"key": "val\"ue"}]'
        assert self.extract(text) == r'[{"key": "val\"ue"}]'

    def test_ignores_text_before_array(self):
        text = 'Some preamble. [{"entity": "X"}]'
        result = self.extract(text)
        assert result == '[{"entity": "X"}]'


class TestValidateEntities:
    def setup_method(self):
        from extract_entities import validate_entities
        self.validate = validate_entities

    ALLOWED = {"Heart :: Enlarged Cardiac Contour", "Lung :: Pneumothorax"}

    def test_valid_entities_pass_through(self):
        items = [
            {"entity": "Heart :: Enlarged Cardiac Contour", "presence": "POSITIVE", "sentence": "x"},
            {"entity": "Lung :: Pneumothorax", "presence": "NEGATIVE", "sentence": "y"},
        ]
        result = self.validate(items, self.ALLOWED)
        assert len(result) == 2

    def test_invalid_entity_skipped(self):
        items = [{"entity": "Unknown :: Entity", "presence": "POSITIVE", "sentence": "x"}]
        result = self.validate(items, self.ALLOWED)
        assert result == []

    def test_duplicate_entity_keeps_first(self):
        items = [
            {"entity": "Heart :: Enlarged Cardiac Contour", "presence": "POSITIVE", "sentence": "first"},
            {"entity": "Heart :: Enlarged Cardiac Contour", "presence": "NEGATIVE", "sentence": "second"},
        ]
        result = self.validate(items, self.ALLOWED)
        assert len(result) == 1
        assert result[0]["sentence"] == "first"

    def test_missing_entity_key_skipped(self):
        items = [{"presence": "POSITIVE", "sentence": "x"}]
        result = self.validate(items, self.ALLOWED)
        assert result == []

    def test_empty_input_returns_empty(self):
        assert self.validate([], self.ALLOWED) == []


class TestBuildMessages:
    def setup_method(self):
        from extract_entities import build_messages
        self.build_messages = build_messages

    def test_returns_two_messages(self):
        msgs = self.build_messages("{entity_list}", [("Heart", "Enlarged")], "report text")
        assert len(msgs) == 2

    def test_system_message_contains_entity_list(self):
        msgs = self.build_messages("Entities:\n{entity_list}", [("Heart", "Enlarged")], "report")
        assert "Heart :: Enlarged" in msgs[0]["content"]

    def test_user_message_contains_report(self):
        msgs = self.build_messages("{entity_list}", [("Heart", "Enlarged")], "my report text")
        assert "my report text" in msgs[1]["content"]

    def test_roles_correct(self):
        msgs = self.build_messages("{entity_list}", [], "report")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"


# ===========================================================================
# call_llm.py – _is_db routing logic
# ===========================================================================
class TestIsDb:
    def setup_method(self):
        from call_llm import _is_db
        self._is_db = _is_db

    def test_databricks_prefix(self):
        assert self._is_db("databricks-claude-sonnet-4-5") is True

    def test_contains_databricks(self):
        assert self._is_db("my-databricks-model") is True

    def test_hf_model_is_not_db(self):
        assert self._is_db("meta-llama/Llama-3.1-8B-Instruct") is False

    def test_llama_model_is_not_db(self):
        assert self._is_db("llama-3.1-8b-instruct") is False

    def test_empty_string_is_not_db(self):
        assert self._is_db("") is False

    def test_none_is_not_db(self):
        assert self._is_db(None) is False

    def test_case_insensitive(self):
        assert self._is_db("DATABRICKS-MODEL") is True


# ===========================================================================
# call_llm.py – query_llm with mocked HF pipeline
# ===========================================================================
class TestQueryLlmMocked:
    """Tests query_llm routing and behavior with mocked backends."""

    def test_routes_to_hf_for_llama(self, tmp_path):
        hf_token_file = tmp_path / ".hftoken"
        hf_token_file.write_text("fake-hf-token")

        mock_output = '[{"role": "assistant", "content": "test"}]'
        mock_pipe_output = [{"generated_text": mock_output}]

        with patch("call_llm._load_hf_pipeline") as mock_load_pipe:
            mock_pipe = MagicMock(return_value=mock_pipe_output)
            mock_load_pipe.return_value = mock_pipe

            from call_llm import query_llm
            result = query_llm(
                messages=[{"role": "user", "content": "hello"}],
                model=LLAMA_MODEL,
                hf_token_path=str(hf_token_file),
            )
            mock_load_pipe.assert_called_once_with(LLAMA_MODEL, "fake-hf-token")
            assert result == mock_output

    def test_routes_to_databricks_for_databricks_model(self, tmp_path):
        db_token_file = tmp_path / ".databricks.token"
        db_token_file.write_text("fake-db-token")

        with patch("call_llm._client_db") as mock_client_db:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "db response"
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_db.return_value = mock_client

            from call_llm import query_llm
            result = query_llm(
                messages=[{"role": "user", "content": "hello"}],
                model="databricks-claude-sonnet-4-5",
                token_path=str(db_token_file),
            )
            assert result == "db response"


# ===========================================================================
# Integration tests – require actual LLM (meta-llama/Llama-3.1-8B-Instruct)
# ===========================================================================
@pytest.mark.integration
class TestQueryLlmIntegration:
    """Live HF inference tests. Require HF token and GPU."""

    def test_basic_generation(self):
        pytest.importorskip("transformers")
        from call_llm import query_llm

        messages = [
            {"role": "system", "content": "You are a radiology assistant. Answer concisely."},
            {"role": "user", "content": "What does 'cardiomegaly' mean in one sentence?"},
        ]
        result = query_llm(
            messages=messages,
            model=LLAMA_MODEL,
            hf_token_path=HF_TOKEN_PATH,
            max_tokens=100,
            temperature=0.0,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_string_not_none(self):
        pytest.importorskip("transformers")
        from call_llm import query_llm

        messages = [{"role": "user", "content": "Say the word 'hello'."}]
        result = query_llm(
            messages=messages,
            model=LLAMA_MODEL,
            hf_token_path=HF_TOKEN_PATH,
            max_tokens=20,
            temperature=0.0,
        )
        assert result is not None
        assert isinstance(result, str)


@pytest.mark.integration
class TestEntityExtractionIntegration:
    """Integration test for entity extraction using llama-3.1-8b-instruct."""

    def test_extracts_entities_from_simple_report(self):
        pytest.importorskip("transformers")
        from extract_entities import run_entity_extraction

        result = run_entity_extraction(
            model=LLAMA_MODEL,
            token_path=HF_TOKEN_PATH,
            prompt_yaml_path=ENTITY_PROMPT_PATH,
            entities_yaml_path=ENTITIES_YAML_PATH,
            report_text=GROUND_TRUTH_REPORT,
            enable_repair=True,
        )
        assert isinstance(result, list)
        # Should extract at least some entities from a non-trivial report
        assert len(result) >= 1

    def test_extracted_entities_have_required_fields(self):
        pytest.importorskip("transformers")
        from extract_entities import run_entity_extraction

        result = run_entity_extraction(
            model=LLAMA_MODEL,
            token_path=HF_TOKEN_PATH,
            prompt_yaml_path=ENTITY_PROMPT_PATH,
            entities_yaml_path=ENTITIES_YAML_PATH,
            report_text=GROUND_TRUTH_REPORT,
            enable_repair=True,
        )
        for entity in result:
            assert "entity" in entity
            assert "presence" in entity
            assert "sentence" in entity
            assert entity["presence"] in ("POSITIVE", "NEGATIVE", "UNCERTAIN")

    def test_entity_names_are_in_allowed_set(self):
        pytest.importorskip("transformers")
        import yaml
        from extract_entities import run_entity_extraction, load_entities_yaml

        entities_yaml = load_entities_yaml(ENTITIES_YAML_PATH)
        allowed = {f"{c} :: {e}" for c, e in entities_yaml.entity_pairs}

        result = run_entity_extraction(
            model=LLAMA_MODEL,
            token_path=HF_TOKEN_PATH,
            prompt_yaml_path=ENTITY_PROMPT_PATH,
            entities_yaml_path=ENTITIES_YAML_PATH,
            report_text=GROUND_TRUTH_REPORT,
            enable_repair=True,
        )
        for entity in result:
            assert entity["entity"] in allowed, f"Invalid entity: {entity['entity']}"

    def test_empty_report_raises(self):
        pytest.importorskip("transformers")
        from extract_entities import run_entity_extraction

        with pytest.raises(ValueError, match="empty"):
            run_entity_extraction(
                model=LLAMA_MODEL,
                token_path=HF_TOKEN_PATH,
                prompt_yaml_path=ENTITY_PROMPT_PATH,
                entities_yaml_path=ENTITIES_YAML_PATH,
                report_text="   ",
            )


@pytest.mark.integration
class TestEndToEndEvaluationIntegration:
    """Full pipeline evaluation using llama-3.1-8b-instruct."""

    def test_concordant_reports_low_score(self):
        """Concordant reports should yield a lower DISCERN score than discordant."""
        pytest.importorskip("transformers")
        import sys
        sys.path.insert(0, SRC_DIR)
        from evaluate_reports import run_evaluation

        eval_result, score = run_evaluation(
            report_text=GROUND_TRUTH_REPORT,
            candidate_text=CANDIDATE_REPORT_CONCORDANT,
            model=LLAMA_MODEL,
            token_path=HF_TOKEN_PATH,
            prompt_yaml_path=ENTITY_PROMPT_PATH,
            entities_yaml_path=ENTITIES_YAML_PATH,
            attribute_prompt_path=ATTRIBUTE_PROMPT_PATH,
            significance_yaml_path=SIGNIFICANCE_PROMPT_PATH,
        )
        assert isinstance(eval_result, list)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_discordant_reports_higher_score(self):
        """Discordant reports should yield a higher DISCERN score than concordant."""
        pytest.importorskip("transformers")
        from evaluate_reports import run_evaluation

        _, concordant_score = run_evaluation(
            report_text=GROUND_TRUTH_REPORT,
            candidate_text=CANDIDATE_REPORT_CONCORDANT,
            model=LLAMA_MODEL,
            token_path=HF_TOKEN_PATH,
            prompt_yaml_path=ENTITY_PROMPT_PATH,
            entities_yaml_path=ENTITIES_YAML_PATH,
            attribute_prompt_path=ATTRIBUTE_PROMPT_PATH,
            significance_yaml_path=SIGNIFICANCE_PROMPT_PATH,
        )
        _, discordant_score = run_evaluation(
            report_text=GROUND_TRUTH_REPORT,
            candidate_text=CANDIDATE_REPORT_DISCORDANT,
            model=LLAMA_MODEL,
            token_path=HF_TOKEN_PATH,
            prompt_yaml_path=ENTITY_PROMPT_PATH,
            entities_yaml_path=ENTITIES_YAML_PATH,
            attribute_prompt_path=ATTRIBUTE_PROMPT_PATH,
            significance_yaml_path=SIGNIFICANCE_PROMPT_PATH,
        )
        assert discordant_score >= concordant_score, (
            f"Expected discordant score ({discordant_score}) >= concordant score ({concordant_score})"
        )

    def test_evaluation_output_fields(self):
        """Each entry in the evaluation output must have all required keys."""
        pytest.importorskip("transformers")
        from evaluate_reports import run_evaluation

        eval_result, _ = run_evaluation(
            report_text=GROUND_TRUTH_REPORT,
            candidate_text=CANDIDATE_REPORT_CONCORDANT,
            model=LLAMA_MODEL,
            token_path=HF_TOKEN_PATH,
            prompt_yaml_path=ENTITY_PROMPT_PATH,
            entities_yaml_path=ENTITIES_YAML_PATH,
            attribute_prompt_path=ATTRIBUTE_PROMPT_PATH,
            significance_yaml_path=SIGNIFICANCE_PROMPT_PATH,
        )
        required_keys = {
            "entity",
            "diagnosis_concordance",
            "location_concordance",
            "severity_concordance",
            "temporal_comparison",
            "significance_score",
            "rationale",
        }
        for row in eval_result:
            missing = required_keys - set(row.keys())
            assert not missing, f"Row missing keys: {missing}\nRow: {row}"
