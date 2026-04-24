"""
MetricRegistry — loads config.yaml metric toggles and dispatches compute calls.

Usage:
    from src.metrics.registry import get_registry
    registry = get_registry()
    scores = registry.compute_all(candidates, references)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


class MetricRegistry:
    """
    Thin dispatcher that reads metric toggles from config.yaml and calls each
    enabled metric's compute() function.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or _load_config()
        self._toggles: Dict[str, bool] = cfg.get("metrics") or {}
        self._checkpoints: Dict[str, str] = cfg.get("checkpoints") or {}

    def is_enabled(self, name: str) -> bool:
        return bool(self._toggles.get(name, False))

    def compute_all(
        self,
        candidates: List[str],
        references: List[str],
    ) -> Dict[str, List[Optional[float]]]:
        """
        Run all enabled metrics. Returns a dict mapping metric name →
        per-sample score list. Metrics that fail or are disabled return None
        values so the output shape is always len(candidates).
        """
        n = len(candidates)
        results: Dict[str, List[Optional[float]]] = {}

        def _run(name: str, fn):
            try:
                scores = fn(candidates, references)
                results[name] = scores
            except Exception as e:
                warnings.warn(f"[metrics] {name} failed: {e}")
                results[name] = [None] * n

        if self.is_enabled("bleu"):
            from .nlp import compute_bleu
            _run("bleu", compute_bleu)

        if self.is_enabled("rouge"):
            from .nlp import compute_rouge
            _run("rouge", compute_rouge)

        if self.is_enabled("meteor"):
            from .nlp import compute_meteor
            _run("meteor", compute_meteor)

        if self.is_enabled("bertscore"):
            from .bertscore import compute_bertscore
            _run("bertscore", compute_bertscore)

        if self.is_enabled("semb_score"):
            chexbert = self._checkpoints.get("chexbert_path", "")
            if not chexbert:
                warnings.warn("[metrics] semb_score skipped: checkpoints.chexbert_path not set in config.yaml")
                results["semb_score"] = [None] * n
            else:
                from .semb_score import compute_semb_score
                _run("semb_score", lambda c, r: compute_semb_score(c, r, chexbert))

        if self.is_enabled("radgraph"):
            from .radgraph import compute_radgraph
            rg_dict = compute_radgraph(candidates, references)
            results.update(rg_dict)  # adds radgraph_e, radgraph_er, radgraph_partial

        if self.is_enabled("radcliq"):
            from .radcliq import compute_radcliq
            # radcliq needs semb_score (chexbert) which is unavailable — skip
            warnings.warn("[metrics] radcliq skipped: requires semb_score (chexbert checkpoint not available)")
            results["radcliq"] = [None] * n

        if self.is_enabled("ratescore"):
            from .ratescore import compute_ratescore
            _run("ratescore", compute_ratescore)

        if self.is_enabled("green"):
            green_model = self._checkpoints.get("green_model", "StanfordAIMI/GREEN-radllama2-7b")
            from .green import compute_green
            _run("green", lambda c, r: compute_green(c, r, green_model))

        if self.is_enabled("bluert"):
            from .bluert import compute_bluert
            _run("bluert", compute_bluert)

        if self.is_enabled("fineradscor"):
            from .fineradscor import compute_fineradscor
            _run("fineradscor", compute_fineradscor)

        # CRIMSON is run as a separate subprocess — not dispatched here.
        # See scripts/run_metrics.py for CRIMSON subprocess handling.

        return results


_registry: Optional[MetricRegistry] = None


def get_registry(config: Optional[dict] = None) -> MetricRegistry:
    global _registry
    if _registry is None or config is not None:
        _registry = MetricRegistry(config)
    return _registry
