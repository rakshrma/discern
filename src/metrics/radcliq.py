"""
RadCliQ-v1 composite metric.
Combines BLEU-2, BERTScore, SembScore (CheXbert), RadGraph F1.

Source: https://github.com/rajpurkarlab/CXR-Report-Metric

Score direction: LOWER is better (RadCliQ predicts radiologist error rate).
The ReXrank leaderboard displays 1/RadCliQ-v1 so that higher = better on the
leaderboard table, but the library itself outputs the raw score. This
implementation follows the library convention: raw score, lower = better.

Installation / setup
--------------------
The v1 model is stored as a sklearn pickle (radcliq-v1.pkl) in the
CXR-Report-Metric repository. Clone the repo and set CXR_METRIC_PATH:

    git clone https://github.com/rajpurkarlab/CXR-Report-Metric.git

Then in config.yaml:
    checkpoints:
      cxr_metric_path: /path/to/CXR-Report-Metric

If the pickle is not found, this module raises an ImportError with instructions.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

_MODEL_V1: Optional[object] = None
_MODEL_V1_PATH: Optional[str] = None


def _find_pickle() -> Optional[Path]:
    """Look for radcliq-v1.pkl in config.yaml checkpoints or common locations."""
    # Try config.yaml first
    cfg_path = Path(__file__).parent.parent.parent / "config.yaml"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            cxr_root = (cfg.get("checkpoints") or {}).get("cxr_metric_path", "")
            if cxr_root:
                candidate = Path(cxr_root) / "CXRMetric" / "radcliq-v1.pkl"
                if candidate.exists():
                    return candidate
        except Exception:
            pass

    # Common fallback locations
    for candidate in [
        Path("/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/CXR-Report-Metric/CXRMetric/radcliq-v1.pkl"),
        Path.home() / "CXR-Report-Metric/CXRMetric/radcliq-v1.pkl",
    ]:
        if candidate.exists():
            return candidate

    return None


def _load_model() -> object:
    global _MODEL_V1, _MODEL_V1_PATH
    pkl_path = _find_pickle()
    if pkl_path is None:
        raise ImportError(
            "radcliq-v1.pkl not found.\n"
            "Clone https://github.com/rajpurkarlab/CXR-Report-Metric and set\n"
            "  checkpoints.cxr_metric_path in config.yaml\n"
            "to the repo root directory."
        )
    if _MODEL_V1 is None or str(pkl_path) != _MODEL_V1_PATH:
        # The pickle was saved while running run_eval.py as __main__, recording
        # CompositeMetric under module '__main__'. We can't import run_eval.py
        # directly (it pulls in fast_bleu and other heavy deps), so we define the
        # class locally — its structure is exactly as in the source file.
        class CompositeMetric:
            def __init__(self, scaler, coefs):
                self.scaler = scaler
                self.coefs = coefs

            def predict(self, x):
                norm_x = self.scaler.transform(x)
                norm_x = np.concatenate(
                    (norm_x, np.ones((norm_x.shape[0], 1))), axis=1
                )
                return norm_x @ self.coefs

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == "CompositeMetric":
                    return CompositeMetric
                return super().find_class(module, name)

        with open(pkl_path, "rb") as f:
            _MODEL_V1 = _Unpickler(f).load()
        _MODEL_V1_PATH = str(pkl_path)
    return _MODEL_V1


def compute_radcliq(
    bleu_scores: List[Optional[float]],
    bertscore_scores: List[Optional[float]],
    semb_scores: List[Optional[float]],
    radgraph_scores: List[Optional[float]],
) -> List[Optional[float]]:
    """
    Compute RadCliQ-v1 from pre-computed sub-metric scores.

    Input order must be: [BLEU-2, BERTScore, SembScore, RadGraph F1]
    Output: raw RadCliQ-v1 score per pair. Lower = better (predicts error rate).

    The ReXrank leaderboard shows 1/RadCliQ-v1 (higher = better) as a display
    convention, but this function returns the raw score matching the repo output.
    """
    n = len(bleu_scores)

    try:
        model = _load_model()
    except ImportError as e:
        warnings.warn(f"[radcliq] {e}")
        return [None] * n

    results = []
    for i in range(n):
        vals = [bleu_scores[i], bertscore_scores[i], semb_scores[i], radgraph_scores[i]]
        if any(v is None for v in vals):
            results.append(None)
            continue
        arr = np.array([vals], dtype=float)  # shape (1, 4) for sklearn predict
        try:
            score = float(model.predict(arr)[0])
            results.append(score)
        except Exception as e:
            warnings.warn(f"[radcliq] predict failed at index {i}: {e}")
            results.append(None)

    return results
