"""
CRIMSON score — clinical text similarity (MedGemma-based).

CRIMSON requires a separate conda environment ('crimson') due to
incompatible dependency versions (transformers>=5.3, torch>=2.10).

This module provides two modes:
  1. Direct import  — use when already inside the crimson conda env.
  2. Subprocess     — use from the main discern env; spawns a crimson-env
                       Python process and captures JSON output.

Source: https://github.com/rajpurkarlab/CRIMSON
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional

_CRIMSON_REPO = Path(__file__).parent.parent.parent.parent / "CRIMSON"


def compute_crimson_direct(
    candidates: List[str],
    references: List[str],
    model_name: str = "rajpurkarlab/medgemma-4b-it-crimson",
    batch_size: int = 4,
) -> List[Optional[float]]:
    """
    Run CRIMSON inside the current process.
    Only call this when running inside the 'crimson' conda environment.
    """
    try:
        sys.path.insert(0, str(_CRIMSON_REPO))
        from CRIMSON.generate_score import CRIMSONScore
        scorer = CRIMSONScore(model_name=model_name, batch_size=batch_size)
        scores = scorer.score(candidates, references)
        return [float(s) if s is not None else None for s in scores]
    except Exception as e:
        warnings.warn(f"CRIMSON direct failed: {e}")
        return [None] * len(candidates)


def compute_crimson_subprocess(
    candidates: List[str],
    references: List[str],
    model_name: str = "rajpurkarlab/medgemma-4b-it-crimson",
    conda_env: str = "/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/envs/crimson",
    batch_size: int = 4,
) -> List[Optional[float]]:
    """
    Run CRIMSON in the crimson conda env via subprocess.
    The input pairs are written to a temp JSON file; scores are read back.
    """
    pairs = [{"candidate": c, "reference": r} for c, r in zip(candidates, references)]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_in:
        json.dump(pairs, f_in)
        input_path = f_in.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_out:
        output_path = f_out.name

    runner_script = Path(__file__).parent / "_crimson_runner.py"

    # Use -p for full env path, -n for named envs
    env_flag = "-p" if conda_env.startswith("/") else "-n"
    cmd = [
        "conda", "run", env_flag, conda_env, "--no-capture-output",
        "python", str(runner_script),
        "--input", input_path,
        "--output", output_path,
        "--model", model_name,
        "--batch-size", str(batch_size),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            warnings.warn(f"CRIMSON subprocess failed:\n{result.stderr}")
            return [None] * len(candidates)
        with open(output_path) as f:
            scores = json.load(f)
        return [float(s) if s is not None else None for s in scores]
    except Exception as e:
        warnings.warn(f"CRIMSON subprocess error: {e}")
        return [None] * len(candidates)
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def compute_crimson(
    candidates: List[str],
    references: List[str],
    model_name: str = "rajpurkarlab/medgemma-4b-it-crimson",
    conda_env: str = "/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/envs/crimson",
    batch_size: int = 4,
) -> List[Optional[float]]:
    """Auto-selects direct or subprocess mode based on whether CRIMSON is importable."""
    try:
        sys.path.insert(0, str(_CRIMSON_REPO))
        import CRIMSON  # noqa: F401
        return compute_crimson_direct(candidates, references, model_name, batch_size)
    except ImportError:
        return compute_crimson_subprocess(candidates, references, model_name, conda_env, batch_size)
