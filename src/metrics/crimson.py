"""CRIMSON metric — LLM-based radiology report evaluation.

CRIMSON (rajpurkarlab/medgemma-4b-it-crimson) is a fine-tuned MedGemma model
that scores radiology report quality.

Environment:
    CRIMSON requires dependencies that conflict with the main DISCERN env.
    Use the dedicated crimson conda environment (envs/crimson.yml) and set
    conda_envs.crimson in config.yaml to that environment's Python binary.

Source:
    https://github.com/rajpurkarlab/CRIMSON
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional


_CRIMSON_RUNNER = Path(__file__).parent / "_crimson_runner.py"


def compute_crimson(
    candidates: List[str],
    references: List[str],
    model_name: str = "rajpurkarlab/medgemma-4b-it-crimson",
    python_bin: Optional[str] = None,
    batch_size: int = 8,
    timeout: Optional[int] = None,
) -> List[Optional[float]]:
    """Compute CRIMSON scores for a batch of candidate/reference pairs.

    Parameters
    ----------
    candidates : list[str]
        Candidate (generated) report texts.
    references : list[str]
        Reference (ground truth) report texts.
    model_name : str
        HuggingFace model name for CRIMSON.
    python_bin : str, optional
        Path to the Python binary in the crimson conda env. Required.
    batch_size : int
        Samples per forward pass for the HF backend (passed to
        scorer.evaluate_batch). Ignored on API backends.
    timeout : int, optional
        Subprocess timeout in seconds. The job wrapper derives this from the
        remaining SLURM walltime so it tracks the real budget. We enforce a
        floor of 120s per candidate (≈ 2 min per case) so large datasets
        like MIMIC-CXR (~3k samples) get enough budget regardless of what
        the caller passes.

    Returns
    -------
    list[float | None]
        Per-pair CRIMSON scores. None entries indicate computation failure.
    """
    if not python_bin:
        warnings.warn(
            "CRIMSON requires a dedicated conda environment. "
            "Set conda_envs.crimson in config.yaml to the crimson env's Python binary."
        )
        return [None] * len(candidates)

    payload = {
        "candidates": candidates,
        "references": references,
        "model_name": model_name,
        "batch_size": int(batch_size),
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fin:
        json.dump(payload, fin)
        fin_path = fin.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fout:
        fout_path = fout.name

    runner = str(_CRIMSON_RUNNER)
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
        # Raise so the SLURM task exits non-zero and shows up as FAILED in
        # sacct — otherwise resubmit_failed.sh treats it as COMPLETED.
        raise RuntimeError(
            f"CRIMSON subprocess timed out after {exc.timeout}s "
            f"(input had {len(candidates)} pairs)"
        ) from exc

    try:
        if result.returncode != 0:
            raise RuntimeError(
                f"CRIMSON subprocess exited with code {result.returncode}:\n"
                f"{result.stderr}"
            )
        # Surface warning lines from the subprocess (e.g. "[warn] sample N
        # failed: <reason>" from CRIMSON's evaluate_batch when JSON parsing
        # fails) so silently-skipped pairs aren't invisible in SLURM logs.
        for line in (result.stdout or "").splitlines():
            if "[warn]" in line or "[CRIMSON]" in line:
                print(line)
        if result.stderr:
            sys.stderr.write(result.stderr)
        with open(fout_path) as f:
            return json.load(f)
    finally:
        Path(fin_path).unlink(missing_ok=True)
        Path(fout_path).unlink(missing_ok=True)
