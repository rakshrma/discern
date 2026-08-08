"""Tests for CSV/JSON input loading and column normalization."""
import json
import sys
import tempfile
from pathlib import Path

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from run_metrics import load_input, load_existing, save_output


def _make_csv(tmp_path, columns: dict) -> Path:
    p = tmp_path / "pairs.csv"
    pd.DataFrame(columns).to_csv(p, index=False)
    return p


def _make_json(tmp_path, records) -> Path:
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps({"results": records}))
    return p


def test_csv_canonical_columns(tmp_path):
    p = _make_csv(tmp_path, {
        "ground_truth_raw": ["ref1", "ref2"],
        "generated_raw": ["cand1", "cand2"],
    })
    records = load_input(str(p))
    assert records[0]["ground_truth_raw"] == "ref1"
    assert records[0]["generated_raw"] == "cand1"
    assert records[0]["sample_idx"] == 0


def test_csv_alias_columns(tmp_path):
    p = _make_csv(tmp_path, {
        "reference": ["ref1"],
        "candidate": ["cand1"],
    })
    records = load_input(str(p))
    assert records[0]["ground_truth_raw"] == "ref1"
    assert records[0]["generated_raw"] == "cand1"


def test_csv_gt_report_alias(tmp_path):
    p = _make_csv(tmp_path, {
        "gt_report": ["ref1"],
        "candidate_report": ["cand1"],
    })
    records = load_input(str(p))
    assert records[0]["ground_truth_raw"] == "ref1"


def test_json_results_key(tmp_path):
    p = _make_json(tmp_path, [
        {"sample_idx": 0, "ground_truth_raw": "ref1", "generated_raw": "cand1"},
    ])
    records = load_input(str(p))
    assert len(records) == 1
    assert records[0]["sample_idx"] == 0


def test_json_list_format(tmp_path):
    p = tmp_path / "list.json"
    p.write_text(json.dumps([
        {"sample_idx": 0, "ground_truth_raw": "ref1", "generated_raw": "cand1"},
    ]))
    records = load_input(str(p))
    assert records[0]["ground_truth_raw"] == "ref1"


def test_count_limit(tmp_path):
    p = _make_csv(tmp_path, {
        "reference": [f"ref{i}" for i in range(10)],
        "candidate": [f"cand{i}" for i in range(10)],
    })
    records = load_input(str(p), count=3)
    assert len(records) == 3


def test_resume_existing(tmp_path):
    out = tmp_path / "scored.json"
    records = [{"sample_idx": 0, "bleu": 0.5}]
    save_output(str(out), records, {"run_date": "2025-01-01"})
    existing = load_existing(str(out))
    assert existing[0]["bleu"] == 0.5


def test_missing_output_file(tmp_path):
    existing = load_existing(str(tmp_path / "nonexistent.json"))
    assert existing == {}
