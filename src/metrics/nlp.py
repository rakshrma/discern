"""NLP metrics: BLEU-1, ROUGE-L, METEOR, BERTScore, RadGraph F1."""
from __future__ import annotations

import warnings
from typing import List, Optional

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

try:
    from rouge import Rouge as _Rouge
    _rouge = _Rouge()
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    warnings.warn("rouge package not found. Install with: pip install rouge")

try:
    from bert_score import score as _bertscore
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    warnings.warn("bert_score package not found. Install with: pip install bert-score")

try:
    from radgraph import F1RadGraph
    HAS_RADGRAPH = True
except ImportError:
    HAS_RADGRAPH = False
    warnings.warn("radgraph package not found. Install with: pip install radgraph")


def _ensure_nltk():
    for resource, path in [("punkt_tab", "tokenizers/punkt_tab"),
                            ("wordnet", "corpora/wordnet"),
                            ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


def _patch_radgraph_tokenizer_compat():
    """Suppress tokenizer parallelism warning that spams logs in batch mode."""
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Cache the F1RadGraph scorer for the lifetime of the process. Constructing it
# loads the model and acquires a FileLock on the (NFS) vocabulary directory;
# building it per report pair was both very slow and caused concurrent tasks to
# collide on that lock ("[Errno 116] Stale file handle"). Build it once and
# reuse it. The retry tolerates transient NFS flock staleness at startup.
_RADGRAPH_SCORER = None

# Likewise cache the BERTScore model — bert_score.score() reloads roberta-large
# on every call, so scoring per pair reloaded the model thousands of times.
_BERTSCORER = None
_BERTSCORER_INIT = False


def _get_bertscorer():
    """Return a cached BERTScorer, or None to fall back to bert_score.score()."""
    global _BERTSCORER, _BERTSCORER_INIT
    if not _BERTSCORER_INIT:
        _BERTSCORER_INIT = True
        try:
            from bert_score import BERTScorer
            _BERTSCORER = BERTScorer(lang="en")
        except Exception:
            _BERTSCORER = None
    return _BERTSCORER


def _get_radgraph_scorer():
    global _RADGRAPH_SCORER
    if _RADGRAPH_SCORER is None:
        import time
        _patch_radgraph_tokenizer_compat()
        last_err = None
        for attempt in range(5):
            try:
                _RADGRAPH_SCORER = F1RadGraph(reward_level="partial", model="radgraph-xl")
                break
            except OSError as e:  # e.g. Errno 116 Stale file handle on NFS lock
                last_err = e
                time.sleep(3 * (attempt + 1))
        else:
            raise last_err
    return _RADGRAPH_SCORER


# ── Single-pair functions (used by run_metrics.py sequential mode) ────────────

def compute_bleu1(candidate: str, reference: str) -> float:
    if not reference.strip() or not candidate.strip():
        return 0.0
    _ensure_nltk()
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    smoothie = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0),
                         smoothing_function=smoothie)


def compute_rougel(candidate: str, reference: str) -> float:
    if not HAS_ROUGE:
        return float("nan")
    if not reference.strip() or not candidate.strip():
        return 0.0
    try:
        return _rouge.get_scores(reference, candidate)[0]["rouge-l"]["f"]
    except ValueError:
        # rouge raises "Hypothesis/Reference is empty" when a string tokenizes
        # to zero n-grams (e.g. punctuation-only), which .strip() can't catch.
        return 0.0


def compute_meteor(candidate: str, reference: str) -> float:
    if not reference.strip() or not candidate.strip():
        return 0.0
    _ensure_nltk()
    return meteor_score([reference.split()], candidate.split())


def compute_bertscore_single(candidate: str, reference: str) -> float:
    if not HAS_BERTSCORE:
        return float("nan")
    if not reference.strip() or not candidate.strip():
        return 0.0
    scorer = _get_bertscorer()
    if scorer is not None:
        _, _, F = scorer.score([candidate], [reference])
        return F[0].item()
    _, _, F = _bertscore([candidate], [reference], lang="en", verbose=False)
    return F[0].item()


def compute_radgraphf1_single(candidate: str, reference: str) -> float:
    if not HAS_RADGRAPH:
        return float("nan")
    if not reference.strip() or not candidate.strip():
        return 0.0
    scorer = _get_radgraph_scorer()
    _, reward_list, _, _ = scorer(hyps=[candidate], refs=[reference])
    return reward_list[0]


# ── Batch functions (used by run_metrics.py batch mode) ───────────────────────

def compute_bertscore_batch(candidates: List[str],
                            references: List[str]) -> List[float]:
    if not HAS_BERTSCORE:
        return [float("nan")] * len(candidates)
    _, _, F = _bertscore(candidates, references, lang="en", verbose=False)
    return F.tolist()


def compute_radgraphf1_batch(candidates: List[str],
                             references: List[str]) -> List[float]:
    if not HAS_RADGRAPH:
        return [float("nan")] * len(candidates)
    scorer = _get_radgraph_scorer()
    _, reward_list, _, _ = scorer(hyps=candidates, refs=references)
    return reward_list
