"""
GREEN score — LLM-based clinical finding accuracy.

GREEN pins transformers to an older version that conflicts with vLLM+Gemma-4
in the main DISCERN env. This module therefore runs GREEN via subprocess in
the dedicated `green_score` conda env (same pattern used for CRIMSON).

Source: https://github.com/Stanford-AIMI/GREEN
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional

_GREEN_CONDA_ENV = "/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/green_score"


def _compute_green_direct(
    candidates: List[str],
    references: List[str],
    model_name: str,
    output_dir: str,
) -> List[Optional[float]]:
    try:
        from green_score import GREEN as _GREEN
    except ImportError:
        warnings.warn("green_score not importable in current env.")
        return [None] * len(candidates)
    try:
        scorer = _GREEN(model_name, output_dir=output_dir)
        mean, std, score_list, summary, result_df = scorer(references, candidates)
        return [float(s) if s is not None else None for s in score_list]
    except Exception as e:
        warnings.warn(f"GREEN direct failed: {e}")
        return [None] * len(candidates)


def _compute_green_subprocess(
    candidates: List[str],
    references: List[str],
    model_name: str,
    output_dir: str,
    conda_env: str,
) -> List[Optional[float]]:
    pairs = [{"candidate": c, "reference": r} for c, r in zip(candidates, references)]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_in:
        json.dump(pairs, f_in)
        input_path = f_in.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_out:
        output_path = f_out.name

    runner_script = Path(__file__).parent / "_green_runner.py"
    env_flag = "-p" if conda_env.startswith("/") else "-n"
    cmd = [
        "conda", "run", env_flag, conda_env, "--no-capture-output",
        "python", str(runner_script),
        "--input", input_path,
        "--output", output_path,
        "--model", model_name,
        "--output-dir", output_dir,
    ]

    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=7200)
        if result.returncode != 0:
            warnings.warn(f"GREEN subprocess exited {result.returncode}")
            return [None] * len(candidates)
        with open(output_path) as f:
            scores = json.load(f)
        return [float(s) if s is not None else None for s in scores]
    except Exception as e:
        warnings.warn(f"GREEN subprocess error: {e}")
        return [None] * len(candidates)
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def compute_green(
    candidates: List[str],
    references: List[str],
    model_name: str = "StanfordAIMI/GREEN-radllama2-7b",
    output_dir: str = ".",
    conda_env: str = _GREEN_CONDA_ENV,
) -> List[Optional[float]]:
    """
    Auto-selects direct vs subprocess mode.
    Uses direct mode only if green_score is importable AND transformers is
    old enough (<5). Otherwise delegates to the green_score conda env.
    """
    try:
        import transformers
        tf_major = int(transformers.__version__.split(".")[0])
    except Exception:
        tf_major = 99

    if tf_major < 5:
        try:
            import green_score  # noqa: F401
            return _compute_green_direct(candidates, references, model_name, output_dir)
        except ImportError:
            pass

    return _compute_green_subprocess(
        candidates, references, model_name, output_dir, conda_env
    )
