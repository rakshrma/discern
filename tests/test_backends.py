"""
Tests for call_llm.py backend detection and dispatch.
All API calls are mocked — no real network required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from call_llm import (
    _detect_backend,
    _is_db,
    query_llm,
    query_llm_batch,
    TokenLimitError,
    TOKEN_LIMIT_PREFIX,
)


# ─── Backend detection ────────────────────────────────────────────────────────

class TestDetectBackend:
    def test_databricks_prefix(self):
        assert _detect_backend("databricks-claude-sonnet-4-6") == "databricks"

    def test_databricks_keyword(self):
        assert _detect_backend("my-databricks-model") == "databricks"

    def test_claude_prefix(self):
        assert _detect_backend("claude-sonnet-4-6") == "anthropic"

    def test_gpt_prefix(self):
        assert _detect_backend("gpt-4o") == "openai"

    def test_o1_prefix(self):
        assert _detect_backend("o1-preview") == "openai"

    def test_hf_slash_model(self):
        assert _detect_backend("google/gemma-3-27b-it") == "openrouter"

    def test_unknown_falls_back_to_hf(self):
        assert _detect_backend("some-local-model") == "hf"


class TestIsDb:
    def test_databricks_prefix_true(self):
        assert _is_db("databricks-gpt-oss-120b") is True

    def test_claude_false(self):
        assert _is_db("claude-sonnet-4-6") is False

    def test_empty_false(self):
        assert _is_db("") is False


# ─── Mocked query_llm dispatch ────────────────────────────────────────────────

SAMPLE_MESSAGES = [{"role": "user", "content": "Test"}]
SAMPLE_RESPONSE = '{"entity": "test", "presence": "POSITIVE", "sentence": "test"}'


class TestQueryLlmMocked:
    def test_databricks_dispatches_correctly(self):
        with patch("call_llm._query_databricks", return_value=SAMPLE_RESPONSE) as mock_db:
            result = query_llm(SAMPLE_MESSAGES, model="databricks-claude-sonnet-4-6",
                               backend="databricks")
            mock_db.assert_called_once()
            assert result == SAMPLE_RESPONSE

    def test_anthropic_dispatches_correctly(self):
        with patch("call_llm._query_anthropic", return_value=SAMPLE_RESPONSE) as mock_ant:
            result = query_llm(SAMPLE_MESSAGES, model="claude-sonnet-4-6",
                               backend="anthropic")
            mock_ant.assert_called_once()
            assert result == SAMPLE_RESPONSE

    def test_openai_dispatches_correctly(self):
        with patch("call_llm._query_openai", return_value=SAMPLE_RESPONSE) as mock_oai:
            result = query_llm(SAMPLE_MESSAGES, model="gpt-4o", backend="openai")
            mock_oai.assert_called_once()
            assert result == SAMPLE_RESPONSE

    def test_openrouter_dispatches_correctly(self):
        with patch("call_llm._query_openrouter", return_value=SAMPLE_RESPONSE) as mock_or:
            result = query_llm(SAMPLE_MESSAGES, model="google/gemma-3-27b-it",
                               backend="openrouter")
            mock_or.assert_called_once()
            assert result == SAMPLE_RESPONSE

    def test_hf_dispatches_correctly(self):
        with patch("call_llm._query_hf", return_value=SAMPLE_RESPONSE) as mock_hf:
            result = query_llm(SAMPLE_MESSAGES, model="meta-llama/Llama-3.1-8B-Instruct",
                               backend="hf")
            mock_hf.assert_called_once()
            assert result == SAMPLE_RESPONSE

    def test_explicit_backend_overrides_model_heuristic(self):
        # Model name looks like anthropic, but we force openai
        with patch("call_llm._query_openai", return_value="ok") as mock_oai:
            query_llm(SAMPLE_MESSAGES, model="claude-sonnet-4-6", backend="openai")
            mock_oai.assert_called_once()


# ─── query_llm_batch ─────────────────────────────────────────────────────────

class TestQueryLlmBatch:
    def test_batch_preserves_order(self):
        responses = ["resp_0", "resp_1", "resp_2"]
        call_count = [0]

        def fake_query(messages, **kwargs):
            idx = int(messages[0]["content"].split("_")[1])
            return responses[idx]

        with patch("call_llm.query_llm", side_effect=fake_query):
            batch = [[{"role": "user", "content": f"msg_{i}"}] for i in range(3)]
            results = query_llm_batch(batch, max_concurrent=3)
            assert results == responses

    def test_batch_raises_on_token_limit(self):
        with patch("call_llm.query_llm", return_value=TOKEN_LIMIT_PREFIX + " too long"):
            with pytest.raises(TokenLimitError):
                query_llm_batch([[{"role": "user", "content": "x"}]])
