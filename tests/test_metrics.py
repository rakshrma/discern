"""
Tests for src/metrics/ wrappers.
Model-based metrics (GREEN, CRIMSON) are skipped unless GPU is available.
Checkpoint-based metrics (SembScore, RadGraph) are skipped unless paths are set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Fixtures ──────────────────────────────────────────────────────────────────

IDENTICAL_PAIR = (
    ["There is a moderate right pleural effusion."],
    ["There is a moderate right pleural effusion."],
)

DIFFERENT_PAIR = (
    ["No pleural effusion. Heart size normal. Lungs clear."],
    ["Large bilateral pleural effusions. Cardiomegaly. Airspace opacity."],
)

TWO_PAIRS = (
    ["Normal chest.", "Large pleural effusion with atelectasis."],
    ["Normal chest.", "Clear lungs and normal heart."],
)


# ── NLP metrics ───────────────────────────────────────────────────────────────

class TestBLEU:
    def test_identical_returns_one(self):
        from metrics.nlp import compute_bleu
        scores = compute_bleu(*IDENTICAL_PAIR)
        assert len(scores) == 1
        assert scores[0] == pytest.approx(1.0, abs=0.01)

    def test_different_returns_low(self):
        from metrics.nlp import compute_bleu
        scores = compute_bleu(*DIFFERENT_PAIR)
        assert scores[0] < 0.3

    def test_batch(self):
        from metrics.nlp import compute_bleu
        scores = compute_bleu(*TWO_PAIRS)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)


class TestROUGE:
    def test_identical(self):
        from metrics.nlp import compute_rouge
        scores = compute_rouge(*IDENTICAL_PAIR)
        if scores[0] is not None:  # skip if rouge not installed
            assert scores[0] == pytest.approx(1.0, abs=0.01)

    def test_different_lower_than_identical(self):
        from metrics.nlp import compute_rouge
        s_id = compute_rouge(*IDENTICAL_PAIR)
        s_diff = compute_rouge(*DIFFERENT_PAIR)
        if s_id[0] is not None and s_diff[0] is not None:
            assert s_id[0] > s_diff[0]


class TestMETEOR:
    def test_identical(self):
        from metrics.nlp import compute_meteor
        scores = compute_meteor(*IDENTICAL_PAIR)
        if scores[0] is not None:
            assert scores[0] == pytest.approx(1.0, abs=0.05)

    def test_in_range(self):
        from metrics.nlp import compute_meteor
        scores = compute_meteor(*TWO_PAIRS)
        for s in scores:
            if s is not None:
                assert 0.0 <= s <= 1.0


class TestBERTScore:
    def test_identical_near_one(self):
        from metrics.bertscore import compute_bertscore
        scores = compute_bertscore(*IDENTICAL_PAIR)
        if scores[0] is not None:
            assert scores[0] > 0.95

    def test_different_lower_than_identical(self):
        from metrics.bertscore import compute_bertscore
        s_id   = compute_bertscore(*IDENTICAL_PAIR)
        s_diff = compute_bertscore(*DIFFERENT_PAIR)
        if s_id[0] is not None and s_diff[0] is not None:
            assert s_id[0] > s_diff[0]

    def test_batch_length(self):
        from metrics.bertscore import compute_bertscore
        scores = compute_bertscore(*TWO_PAIRS)
        assert len(scores) == 2


# ── RaTEScore ─────────────────────────────────────────────────────────────────

class TestRaTEScore:
    def test_import_or_skip(self):
        try:
            from metrics.ratescore import compute_ratescore, HAS_RATESCORE
            if not HAS_RATESCORE:
                pytest.skip("RaTEScore not installed")
            scores = compute_ratescore(*IDENTICAL_PAIR)
            assert len(scores) == 1
            if scores[0] is not None:
                assert 0.0 <= scores[0] <= 1.0
        except ImportError:
            pytest.skip("RaTEScore not installed")

    def test_batch(self):
        try:
            from metrics.ratescore import compute_ratescore, HAS_RATESCORE
            if not HAS_RATESCORE:
                pytest.skip("RaTEScore not installed")
            scores = compute_ratescore(*TWO_PAIRS)
            assert len(scores) == 2
        except ImportError:
            pytest.skip("RaTEScore not installed")


# ── RadCliQ composite ─────────────────────────────────────────────────────────

class TestRadCliQ:
    def test_none_input_returns_none(self):
        from metrics.radcliq import compute_radcliq
        # None in any sub-metric → None output (no pickle needed for this check)
        result = compute_radcliq([None], [0.9], [0.8], [0.7])
        assert result[0] is None

    def test_none_propagates_in_batch(self):
        from metrics.radcliq import compute_radcliq
        result = compute_radcliq([None, None], [0.9, None], [0.8, 0.7], [0.7, 0.6])
        assert result[0] is None
        assert result[1] is None

    def test_warns_when_pickle_missing(self):
        import warnings
        from metrics.radcliq import compute_radcliq, _find_pickle
        if _find_pickle() is not None:
            pytest.skip("radcliq-v1.pkl found — skipping missing-pickle test")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute_radcliq([0.8], [0.9], [0.85], [0.75])
            assert result == [None]
            assert any("radcliq-v1.pkl" in str(warning.message) for warning in w)

    @pytest.mark.requires_checkpoints
    def test_composite_with_real_pickle(self):
        from metrics.radcliq import compute_radcliq, _find_pickle
        if _find_pickle() is None:
            pytest.skip("radcliq-v1.pkl not found")
        result = compute_radcliq([0.8], [0.9], [0.85], [0.75])
        assert len(result) == 1
        assert result[0] is not None
        assert isinstance(result[0], float)
        # RadCliQ is an error-rate predictor; reasonable range for near-perfect inputs
        assert result[0] < 2.0


# ── GREEN (GPU required) ──────────────────────────────────────────────────────

@pytest.mark.requires_gpu
class TestGREEN:
    def test_import(self):
        from metrics.green import HAS_GREEN
        if not HAS_GREEN:
            pytest.skip("green_score not installed")

    def test_scores_in_range(self):
        from metrics.green import compute_green, HAS_GREEN
        if not HAS_GREEN:
            pytest.skip("green_score not installed")
        scores = compute_green(*TWO_PAIRS)
        assert len(scores) == 2
        for s in scores:
            if s is not None:
                assert isinstance(s, float)


# ── Registry integration ──────────────────────────────────────────────────────

class TestMetricRegistry:
    def test_registry_init_with_config(self):
        from metrics.registry import MetricRegistry
        cfg = {"metrics": {"bleu": True, "rouge": False, "bertscore": False}}
        reg = MetricRegistry(cfg)
        assert reg.is_enabled("bleu") is True
        assert reg.is_enabled("rouge") is False

    def test_compute_all_bleu_only(self):
        from metrics.registry import MetricRegistry
        cfg = {
            "metrics": {"bleu": True, "rouge": False, "meteor": False,
                        "bertscore": False, "ratescore": False, "green": False,
                        "crimson": False, "bluert": False, "fineradscor": False,
                        "semb_score": False, "radgraph": False, "radcliq": False,
                        "discern": False, "mini_discern": False},
        }
        reg = MetricRegistry(cfg)
        results = reg.compute_all(*TWO_PAIRS)
        assert "bleu" in results
        assert len(results["bleu"]) == 2
        assert all(isinstance(s, float) for s in results["bleu"])

    def test_disabled_metric_not_in_results(self):
        from metrics.registry import MetricRegistry
        cfg = {"metrics": {"bleu": False, "rouge": False}}
        reg = MetricRegistry(cfg)
        results = reg.compute_all(*TWO_PAIRS)
        assert "bleu" not in results

    def test_semb_skipped_when_no_checkpoint(self):
        import warnings
        from metrics.registry import MetricRegistry
        cfg = {"metrics": {"semb_score": True}, "checkpoints": {"chexbert_path": ""}}
        reg = MetricRegistry(cfg)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = reg.compute_all(*IDENTICAL_PAIR)
            assert "semb_score" in results
            assert results["semb_score"] == [None]
            assert any("chexbert_path" in str(warning.message) for warning in w)
