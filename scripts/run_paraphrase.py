"""
run_paraphrase.py — Generate paraphrase robustness dataset.

For each (ground_truth, candidate) pair in the source datasets, the
ground_truth report is paraphrased with style-only transformations
(clinical content preserved). The output pairs the ORIGINAL ground truth
(judge reference) with the PARAPHRASED ground truth (judge candidate).

Transforms: rearrange | bullets | rephrase | voice | random_1 | random_2

Source datasets
---------------
  --rexval    CSV with columns: study_id, ground_truth_raw, generated_raw, ...
  --radevalx  CSV with same schema
  --chexpert  JSON with "results" key (vlm_cxr_benchmark schema)

Output
------
  Single JSON compatible with run_metrics.py and run_discern_paraphrase.py

Usage
-----
  python scripts/run_paraphrase.py \\
      --rexval   data/rexval/rexval_reports_long.csv \\
      --radevalx data/radevalx/radeval_total.csv \\
      --output   data/robustness/paraphrase_robustness_v2.json \\
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import yaml
from pydantic import BaseModel, StrictStr

from call_llm import query_llm


class ParaphraseResponse(BaseModel):
    report: StrictStr


# ─── Transformation prompts ──────────────────────────────────────────────────

TRANSFORMS: Dict[str, str] = {
    "rearrange": (
        "Rearrange the order of sentences within each section of the report "
        "(e.g. within FINDINGS, within IMPRESSION). "
        "Do NOT move sentences across section boundaries. "
        "Do NOT alter the wording of any sentence."
    ),
    "bullets": (
        "Convert the FINDINGS section from prose to bullet points if written in prose, "
        "or from bullet points back to prose if already bulleted. "
        "Do NOT change any clinical content or wording."
    ),
    "rephrase": (
        "Rewrite the report using different vocabulary and sentence structure "
        "while preserving all clinical findings exactly. "
        "Do NOT add, remove, or change any clinical meaning."
    ),
    "voice": (
        "Convert sentences from active to passive voice or vice versa. "
        "Preserve all clinical content exactly."
    ),
}

_SYSTEM_PROMPT = """You are a radiology report editor. Your task is to transform a radiology report
according to the specified instruction while STRICTLY preserving all clinical content.

ABSOLUTE RULES:
1. Do NOT add any new findings, diagnoses, or clinical information.
2. Do NOT remove any findings, diagnoses, or clinical information.
3. Do NOT change laterality (left/right), severity, or temporal language.
4. Return ONLY the transformed report text. No explanations.
5. Preserve all section headers (FINDINGS:, IMPRESSION:, etc.) exactly."""


def _paraphrase_single(report: str, transform: str, model: str) -> Optional[str]:
    instruction = TRANSFORMS[transform]
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Transform instruction:\n{instruction}\n\nReport:\n{report}"},
    ]
    try:
        return query_llm(messages=messages, model=model, max_tokens=2000, temperature=0.3)
    except Exception as e:
        warnings.warn(f"Paraphrase failed ({transform}): {e}")
        return None


def _load_rexval(path: str) -> List[Dict]:
    import pandas as pd
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        records.append({
            "source": "rexval",
            "source_report_id": str(row.get("study_id", "")),
            "ground_truth_raw": str(row.get("ground_truth_raw", row.get("reference", ""))),
            "generated_raw": str(row.get("generated_raw", row.get("candidate", ""))),
        })
    return records


def _load_radevalx(path: str) -> List[Dict]:
    import pandas as pd
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        records.append({
            "source": "radevalx",
            "source_report_id": str(row.get("report_id", row.get("study_id", ""))),
            "ground_truth_raw": str(row.get("ground_truth_raw", row.get("reference", ""))),
            "generated_raw": str(row.get("generated_raw", row.get("candidate", ""))),
        })
    return records


def _load_chexpert(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    entries = data.get("results", data) if isinstance(data, dict) else data
    records = []
    for e in entries:
        records.append({
            "source": "chexpertplusvalid",
            "source_report_id": str(e.get("sample_idx", "")),
            "ground_truth_raw": str(e.get("ground_truth_raw", "")),
            "generated_raw": str(e.get("generated_raw", "")),
        })
    return records


def run(args: argparse.Namespace):
    model = args.model

    all_source_records: List[Dict] = []
    if args.rexval:
        all_source_records += _load_rexval(args.rexval)
    if args.radevalx:
        all_source_records += _load_radevalx(args.radevalx)
    if args.chexpert:
        all_source_records += _load_chexpert(args.chexpert)

    if not all_source_records:
        print("No source data provided. Pass --rexval, --radevalx, or --chexpert.")
        sys.exit(1)

    rng = random.Random(args.seed)

    # Resume: load existing output
    output_path = Path(args.output)
    existing = []
    if output_path.exists():
        with open(output_path) as f:
            d = json.load(f)
        existing = d if isinstance(d, list) else d.get("results", [])

    done_keys = {(e["source"], e["source_report_id"], e["transformation_type"][0])
                 for e in existing if e.get("transformation_type")}

    transform_names = list(TRANSFORMS.keys())
    output_entries = list(existing)
    sample_idx = max((e.get("sample_idx", -1) for e in existing), default=-1) + 1

    for record in all_source_records:
        gt = record["ground_truth_raw"]
        source = record["source"]
        report_id = record["source_report_id"]

        # Single transforms
        for t in transform_names:
            if (source, report_id, t) in done_keys:
                continue
            paraphrased = _paraphrase_single(gt, t, model)
            if paraphrased is None:
                continue
            output_entries.append({
                "sample_idx": sample_idx,
                "ground_truth_raw": gt,
                "generated_raw": paraphrased,
                "transformation_type": [t],
                "variant": t,
                "source": source,
                "source_report_id": report_id,
            })
            sample_idx += 1
            done_keys.add((source, report_id, t))

        # Random combo transforms
        for combo_idx in range(args.random_combos):
            combo_name = f"random_{combo_idx + 1}"
            if (source, report_id, combo_name) in done_keys:
                continue
            chosen = rng.sample(transform_names, k=rng.randint(2, min(4, len(transform_names))))
            current = gt
            for t in chosen:
                result = _paraphrase_single(current, t, model)
                if result:
                    current = result
            if current != gt:
                output_entries.append({
                    "sample_idx": sample_idx,
                    "ground_truth_raw": gt,
                    "generated_raw": current,
                    "transformation_type": chosen,
                    "variant": combo_name,
                    "source": source,
                    "source_report_id": report_id,
                })
                sample_idx += 1
                done_keys.add((source, report_id, combo_name))

        # Save after each record (resume-safe)
        with open(output_path, "w") as f:
            json.dump({"results": output_entries}, f, indent=2)

    print(f"\nDone. {len(output_entries)} paraphrase entries saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paraphrase robustness dataset.")
    parser.add_argument("--rexval",    default=None, help="ReXVal CSV path")
    parser.add_argument("--radevalx",  default=None, help="RaDEvalX CSV path")
    parser.add_argument("--chexpert",  default=None, help="CheXpert+ valid JSON path")
    parser.add_argument("--output",    required=True, help="Output JSON path")
    parser.add_argument("--model",     default=None,
                        help="LLM model for paraphrasing (default from config.yaml)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--random-combos", type=int, default=2,
                        help="Number of random-combination variants per report (default: 2)")
    parser.add_argument("--config",    default=None)
    args = parser.parse_args()

    if args.model is None:
        cfg_path = Path(args.config) if args.config else _ROOT / "config.yaml"
        if cfg_path.exists():
            import yaml
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            args.model = (cfg.get("discern") or {}).get("default_model", "google/gemma-3-27b-it")
        else:
            args.model = "google/gemma-3-27b-it"

    run(args)


if __name__ == "__main__":
    main()
