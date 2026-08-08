"""Tests for NLP metric functions with known inputs."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics.nlp import (
    compute_bleu1,
    compute_rougel,
    compute_meteor,
    compute_bertscore_single,
)

IDENTICAL = "The lungs are clear. No pleural effusion."
DIFFERENT = "The heart is enlarged. There is cardiomegaly."


def test_bleu_identical():
    score = compute_bleu1(IDENTICAL, IDENTICAL)
    assert score == pytest.approx(1.0, abs=0.01)


def test_bleu_different():
    score = compute_bleu1(DIFFERENT, IDENTICAL)
    assert score < 0.5


def test_bleu_range():
    score = compute_bleu1("no effusion", "no pleural effusion")
    assert 0.0 <= score <= 1.0


def test_rouge_identical():
    score = compute_rougel(IDENTICAL, IDENTICAL)
    assert score == pytest.approx(1.0, abs=0.01)


def test_rouge_range():
    score = compute_rougel("clear lungs", IDENTICAL)
    assert 0.0 <= score <= 1.0


def test_meteor_identical():
    score = compute_meteor(IDENTICAL, IDENTICAL)
    assert score == pytest.approx(1.0, abs=0.01)


def test_meteor_range():
    score = compute_meteor("lungs clear", IDENTICAL)
    assert 0.0 <= score <= 1.0


def test_bertscore_identical():
    score = compute_bertscore_single(IDENTICAL, IDENTICAL)
    assert score > 0.95


def test_bertscore_different():
    score = compute_bertscore_single(DIFFERENT, IDENTICAL)
    assert score < 0.95


def test_bertscore_range():
    score = compute_bertscore_single("clear lungs", IDENTICAL)
    assert 0.0 <= score <= 1.0
