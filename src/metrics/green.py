"""GREEN metric — LLM-based clinical finding accuracy.

GREEN (StanfordAIMI/GREEN-radllama2-7b) uses a fine-tuned LLaMA-2 to evaluate
clinical accuracy of radiology report candidates.

Installation:
    pip install git+https://github.com/Stanford-AIMI/GREEN.git

Environment conflict note:
    GREEN requires a specific older version of transformers that conflicts with
    vLLM + Gemma-4. Use a dedicated conda environment (envs/green.yml) and set
    conda_envs.green in config.yaml to that environment's Python binary.

    If conda_envs.green is empty in config.yaml, this module attempts to call
    green_score directly (only works if running inside the green conda env).
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional


_GREEN_RUNNER = Path(__file__).parent / "_green_runner.py"


def compute_green(
    candidates: List[str],
    references: List[str],
    model_name: str = "StanfordAIMI/GREEN-radllama2-7b",
    output_dir: str = ".",
    python_bin: Optional[str] = None,
    timeout: Optional[int] = None,
) -> List[Optional[float]]:
    """Compute GREEN scores for a batch of candidate/reference pairs.

    Parameters
    ----------
    candidates : list[str]
        Candidate (generated) report texts.
    references : list[str]
        Reference (ground truth) report texts.
    model_name : str
        HuggingFace model name for GREEN (default: StanfordAIMI/GREEN-radllama2-7b).
    output_dir : str
        Directory for GREEN's internal output files.
    python_bin : str, optional
        Path to the Python binary in the green conda env. If None, uses the
        current Python (must be running inside the green env).
    timeout : int, optional
        Subprocess timeout in seconds. The job wrapper derives this from the
        remaining SLURM walltime so it tracks the real budget. We enforce a
        floor of 120s per candidate (≈ 2 min per case) so large datasets
        like MIMIC-CXR (~3k samples) get enough budget regardless of what
        the caller passes.

    Returns
    -------
    list[float | None]
        Per-pair GREEN scores. None entries indicate computation failure.
    """
    if python_bin:
        return _compute_green_subprocess(
            candidates, references, model_name, output_dir, python_bin, timeout
        )
    return _compute_green_direct(candidates, references, model_name, output_dir)


def _compute_green_direct(
    candidates: List[str],
    references: List[str],
    model_name: str,
    output_dir: str,
) -> List[Optional[float]]:
    try:
        from green_score import GREEN as _GREEN
        scorer = _GREEN(model_name, output_dir=output_dir)
        _, _, score_list, _, _ = scorer(references, candidates)
        return [float(s) for s in score_list]
    except ImportError:
        warnings.warn(
            "green_score package not found. Install from "
            "https://github.com/Stanford-AIMI/GREEN or set "
            "conda_envs.green in config.yaml to use a dedicated env."
        )
        return [None] * len(candidates)
    except Exception as exc:
        warnings.warn(f"GREEN computation failed: {exc}")
        return [None] * len(candidates)


def _compute_green_subprocess(
    candidates: List[str],
    references: List[str],
    model_name: str,
    output_dir: str,
    python_bin: str,
    timeout: Optional[int] = None,
) -> List[Optional[float]]:
    """Run GREEN via subprocess in its dedicated conda environment."""
    payload = {
        "candidates": candidates,
        "references": references,
        "model_name": model_name,
        "output_dir": output_dir,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fin:
        json.dump(payload, fin)
        fin_path = fin.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fout:
        fout_path = fout.name

    runner = str(_GREEN_RUNNER)
    per_case_floor = 120 * max(len(candidates), 1)
    effective_timeout = max(timeout or 0, per_case_floor)
    try:
        result = subprocess.run(
            [python_bin, runner, fin_path, fout_path],
            capture_output=True, text=True, timeout=effective_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        Path(fin_path).unlink(missing_ok=True)
        Path(fout_path).unlink(missing_ok=True)
        # Raise so the SLURM task exits non-zero — the previous behavior
        # silently wrote all-None scores and was treated as COMPLETED, which
        # hid the failure from resubmit_failed.sh.
        raise RuntimeError(
            f"GREEN subprocess timed out after {exc.timeout}s "
            f"(input had {len(candidates)} pairs)"
        ) from exc

    try:
        if result.returncode != 0:
            raise RuntimeError(
                f"GREEN subprocess exited with code {result.returncode}:\n"
                f"{result.stderr}"
            )
        with open(fout_path) as f:
            return json.load(f)
    finally:
        Path(fin_path).unlink(missing_ok=True)
        Path(fout_path).unlink(missing_ok=True)
