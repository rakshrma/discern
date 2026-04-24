"""
Tests for JSON / CSV loading and resume-safe writing (scripts/run_metrics.py I/O).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── JSON loading ──────────────────────────────────────────────────────────────

class TestLoadInput:
    def test_load_vlm_cxr_json_schema(self, tmp_path):
        from run_metrics import load_input
        data = {
            "results": [
                {"sample_idx": 0, "ground_truth_raw": "Normal chest.", "generated_raw": "Clear lungs."},
                {"sample_idx": 1, "ground_truth_raw": "Pleural effusion.", "generated_raw": "Effusion present."},
            ]
        }
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data))
        entries = load_input(str(p))
        assert len(entries) == 2
        assert entries[0]["sample_idx"] == 0
        assert entries[1]["ground_truth_raw"] == "Pleural effusion."

    def test_load_flat_list_json(self, tmp_path):
        from run_metrics import load_input
        data = [
            {"sample_idx": 0, "ground_truth_raw": "A", "generated_raw": "B"},
        ]
        p = tmp_path / "flat.json"
        p.write_text(json.dumps(data))
        entries = load_input(str(p))
        assert len(entries) == 1

    def test_load_csv_standard_columns(self, tmp_path):
        from run_metrics import load_input
        csv_content = "reference,candidate\nNormal chest.,Clear lungs.\nEffusion,Effusion present.\n"
        p = tmp_path / "pairs.csv"
        p.write_text(csv_content)
        entries = load_input(str(p))
        assert len(entries) == 2
        assert entries[0]["ground_truth_raw"] == "Normal chest."
        assert entries[0]["generated_raw"] == "Clear lungs."

    def test_load_csv_alternative_column_names(self, tmp_path):
        from run_metrics import load_input
        csv_content = "ground_truth_raw,generated_raw\nRef report.,Cand report.\n"
        p = tmp_path / "alt.csv"
        p.write_text(csv_content)
        entries = load_input(str(p))
        assert entries[0]["ground_truth_raw"] == "Ref report."

    def test_csv_auto_assigns_sample_idx(self, tmp_path):
        from run_metrics import load_input
        csv_content = "reference,candidate\nA,B\nC,D\n"
        p = tmp_path / "no_idx.csv"
        p.write_text(csv_content)
        entries = load_input(str(p))
        assert entries[0]["sample_idx"] == 0
        assert entries[1]["sample_idx"] == 1


# ── Resume-safe write ─────────────────────────────────────────────────────────

class TestResumeWrite:
    def test_save_and_reload_preserves_all_entries(self, tmp_path):
        from run_metrics import save_output, load_existing
        results = [
            {"sample_idx": 0, "bleu": 0.9, "discern_score": 2.0},
            {"sample_idx": 1, "bleu": 0.5, "discern_score": 7.0},
        ]
        metadata = {"run_date": "2026-04-23", "model": "test"}
        p = str(tmp_path / "out.json")
        save_output(p, results, metadata)
        loaded = load_existing(p)
        assert len(loaded) == 2
        assert loaded[0]["bleu"] == 0.9

    def test_metadata_block_in_output(self, tmp_path):
        from run_metrics import save_output
        p = str(tmp_path / "meta.json")
        save_output(p, [], {"run_date": "2026-04-23", "tag": "test_tag"})
        with open(p) as f:
            data = json.load(f)
        assert "_metadata" in data
        assert data["_metadata"]["tag"] == "test_tag"

    def test_resume_skips_completed_entries(self, tmp_path):
        from run_metrics import load_existing, _already_scored
        existing = [
            {"sample_idx": 0, "bleu": 0.9, "discern_score": 2.0},
        ]
        p = tmp_path / "existing.json"
        p.write_text(json.dumps({"results": existing}))
        loaded = load_existing(str(p))
        assert _already_scored(loaded[0], ["bleu", "discern_score"]) is True
        assert _already_scored(loaded[0], ["bleu", "ratescore"]) is False
