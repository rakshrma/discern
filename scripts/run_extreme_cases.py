"""
run_extreme_cases.py — Generate and score extreme/adversarial report pairs.

Extreme cases validate DISCERN score calibration:
  concordant  — candidate = exact copy of ground truth (score should ≈ 0)
  discordant  — LLM generates maximally wrong version (score should be high)

Usage
-----
  python scripts/run_extreme_cases.py \\
      --input  data/rexval/rexval_reports_long.csv \\
      --output data/discern_runs/extreme_cases/ \\
      --model  google/gemma-3-27b-it
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import yaml

_ADVERSARIAL_PROMPT = """You are generating a deliberately WRONG radiology report for research purposes.

Given the reference report below, write a version that:
1. Contradicts every major finding (e.g. if pleural effusion is present → say absent)
2. Swaps laterality (left ↔ right)
3. Changes severity (e.g. moderate → mild, large → small)
4. Changes temporal characterization (e.g. new → chronic, worsening → improving)
5. Changes diagnoses to different ones

Return ONLY the modified report text. Preserve section headers (FINDINGS:, IMPRESSION:).

Reference report:
{report}"""


def _load_entries(input_path: str) -> List[Dict]:
    p = Path(input_path)
    if p.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(input_path)
        col_map = {}
        for col in df.columns:
            lc = col.lower().replace(" ", "_")
            if lc in ("ground_truth_raw", "reference", "ground_truth"):
                col_map[col] = "ground_truth_raw"
            elif lc in ("generated_raw", "candidate", "generated"):
                col_map[col] = "generated_raw"
        df = df.rename(columns=col_map)
        if "sample_idx" not in df.columns:
            df["sample_idx"] = list(range(len(df)))
        return df.to_dict(orient="records")
    with open(input_path) as f:
        data = json.load(f)
    return data.get("results", data) if isinstance(data, dict) else data


def _generate_adversarial(report: str, model: str) -> Optional[str]:
    from call_llm import query_llm
    messages = [
        {"role": "system", "content": "You generate modified radiology reports for research calibration only."},
        {"role": "user", "content": _ADVERSARIAL_PROMPT.format(report=report)},
    ]
    try:
        return query_llm(messages=messages, model=model, max_tokens=2000, temperature=0.5)
    except Exception as e:
        warnings.warn(f"Adversarial generation failed: {e}")
        return None


def run(args: argparse.Namespace):
    entries = _load_entries(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    concordant_pairs = []
    discordant_pairs = []
    sample_idx = 0

    for entry in entries:
        gt = entry.get("ground_truth_raw", "")
        idx = entry.get("sample_idx", sample_idx)

        # Concordant: candidate = exact copy
        concordant_pairs.append({
            "sample_idx": sample_idx,
            "ground_truth_raw": gt,
            "generated_raw": gt,
            "case_type": "concordant",
            "source_idx": idx,
        })
        sample_idx += 1

        # Discordant: LLM-generated adversarial version
        adversarial = _generate_adversarial(gt, args.model)
        if adversarial:
            discordant_pairs.append({
                "sample_idx": sample_idx,
                "ground_truth_raw": gt,
                "generated_raw": adversarial,
                "case_type": "discordant",
                "source_idx": idx,
            })
            sample_idx += 1

    all_pairs = concordant_pairs + discordant_pairs
    raw_output = output_dir / "extreme_cases_raw.json"
    with open(raw_output, "w") as f:
        json.dump({"results": all_pairs}, f, indent=2)
    print(f"Generated {len(all_pairs)} extreme cases → {raw_output}")

    # Score with DISCERN
    scored_output = output_dir / "extreme_cases_scored.json"
    import subprocess
    cmd = [
        sys.executable, str(_ROOT / "scripts" / "run_discern.py"),
        "--input",  str(raw_output),
        "--output", str(scored_output),
        "--mode",   "both",
        "--model",  args.model,
        "--tag",    "extreme_cases",
    ]
    if args.backend:
        cmd += ["--backend", args.backend]
    print(f"Scoring with DISCERN → {scored_output}")
    subprocess.run(cmd, check=True)
    print(f"Done. Scored extreme cases at {scored_output}")


def main():
    parser = argparse.ArgumentParser(description="Generate and score extreme report pairs.")
    parser.add_argument("--input",   required=True)
    parser.add_argument("--output",  required=True)
    parser.add_argument("--model",   default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--config",  default=None)
    args = parser.parse_args()

    if args.model is None:
        cfg_path = Path(args.config) if args.config else _ROOT / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            args.model = (cfg.get("discern") or {}).get("default_model", "google/gemma-3-27b-it")
        else:
            args.model = "google/gemma-3-27b-it"

    run(args)


if __name__ == "__main__":
    main()
