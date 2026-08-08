"""Tests for LLM backend selection and credential resolution."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_backend import _is_db, _resolve_db_token, _resolve_hf_token


def test_is_db_databricks_prefix():
    assert _is_db("databricks-claude-sonnet-4-6") is True


def test_is_db_hf_path():
    assert _is_db("google/gemma-4-31B-it") is False


def test_is_db_empty():
    assert _is_db("") is False


def test_resolve_db_token_direct_string():
    token = _resolve_db_token("dapi_abc123", None, None)
    assert token == "dapi_abc123"


def test_resolve_db_token_from_file(tmp_path):
    f = tmp_path / "token"
    f.write_text("dapi_from_file")
    token = _resolve_db_token(None, None, str(f))
    assert token == "dapi_from_file"


def test_resolve_db_token_path_as_string():
    # Non-file-path string passed via token_path kwarg
    token = _resolve_db_token(None, "dapi_inline", None)
    assert token == "dapi_inline"


def test_resolve_db_token_file_path_detection(tmp_path):
    f = tmp_path / ".token"
    f.write_text("dapi_file")
    # Starts with "/" so treated as file path
    token = _resolve_db_token(None, str(f), None)
    assert token == "dapi_file"


def test_resolve_db_token_missing_raises():
    with pytest.raises(RuntimeError, match="Databricks token not provided"):
        _resolve_db_token(None, None, None)


def test_resolve_hf_token_direct():
    token = _resolve_hf_token("hf_abc", None)
    assert token == "hf_abc"


def test_resolve_hf_token_none_when_missing():
    token = _resolve_hf_token(None, None)
    assert token is None


def test_resolve_hf_token_from_file(tmp_path):
    f = tmp_path / "hf_token"
    f.write_text("hf_from_file")
    token = _resolve_hf_token(None, str(f))
    assert token == "hf_from_file"
