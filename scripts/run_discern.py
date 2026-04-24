"""
run_discern.py — DISCERN and mini-DISCERN evaluation CLI.

Supports:
  - Single report pair (--reference / --candidate)
  - Batch from JSON file (vlm_cxr_benchmark schema or custom)
  - Batch from CSV file (reference, candidate columns)

Examples
--------
  # Single pair
  python scripts/run_discern.py \\
      --reference "Heart is enlarged with pleural effusion." \\
      --candidate "Lungs are clear. Heart size normal."

  # Batch from JSON
  python scripts/run_discern.py --input results.json --output discern_scores.json

  # Batch, mini-DISCERN only, custom model
  python scripts/run_discern.py --input results.json --output out.json \\
      --mode mini --backend anthropic --model claude-sonnet-4-6

  # Batch, both DISCERN + mini, 8 parallel requests
  python scripts/run_discern.py --input results.json --output out.json \\
      --mode both --max-concurrent 8
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import yaml


def _load_config(path: Optional[str] = None) -> dict:
    cfg_path = Path(path) if path else _ROOT / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_entries(input_path: str) -> List[Dict[str, Any]]:
    p = Path(input_path)
    if p.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(input_path)
        col_map = {}
        for col in df.columns:
            lc = col.lower().replace(" ", "_")
            if lc in ("ground_truth_raw", "reference", "ref", "ground_truth"):
                col_map[col] = "ground_truth_raw"
            elif lc in ("generated_raw", "candidate", "cand", "generated"):
                col_map[col] = "generated_raw"
        df = df.rename(columns=col_map)
        if "sample_idx" not in df.columns:
            df["sample_idx"] = list(range(len(df)))
        return df.to_dict(orient="records")
    with open(input_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("results", data.get("entries", []))


def _load_existing(output_path: str) -> List[Dict]:
    p = Path(output_path)
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("results", [])


def _save(output_path: str, results: List[Dict], metadata: dict):
    payload = {"_metadata": metadata, "results": results}
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _score_entry(
    entry: dict,
    model: str,
    backend: str,
    config_dir: Path,
    run_full: bool,
    run_mini: bool,
) -> dict:
    from evaluate_reports import run_evaluation
    from get_discern_score import compute_reads_score

    result = dict(entry)
    gt   = entry.get("ground_truth_raw", "")
    cand = entry.get("generated_raw", "")
    token_path = str(config_dir / ".databricks.token")

    if run_full and result.get("discern_score") is None:
        try:
            eval_out, _ = run_evaluation(
                report_text=gt,
                candidate_text=cand,
                model=model,
                token_path=token_path,
                prompt_yaml_path=str(config_dir / "entity_extraction_prompt.yaml"),
                entities_yaml_path=str(config_dir / "entities.yaml"),
                attribute_prompt_path=str(config_dir / "attribute_extraction_prompt.yaml"),
                significance_yaml_path=str(config_dir / "significance_prompt.yaml"),
            )
            result["discern_score"] = compute_reads_score(eval_out)
            result["discern_evaluation"] = [
                e.model_dump() if hasattr(e, "model_dump") else e for e in eval_out
            ]
        except Exception as e:
            warnings.warn(f"[DISCERN] entry {entry.get('sample_idx')} failed: {e}")
            result["discern_score"] = None

    if run_mini and result.get("mini_discern_score") is None:
        try:
            from evaluate_single_prompt import evaluate_reports as mini_eval
            mini_out = mini_eval(
                ground_truth=gt,
                candidate=cand,
                model=model,
                token_path=token_path,
                prompt_yaml_path=str(config_dir / "merged_prompt.yaml"),
                entities_yaml_path=str(config_dir / "diagnosis_only.yaml"),
            )
            result["mini_discern_score"] = compute_reads_score(mini_out)
            result["mini_discern_evaluation"] = [
                e.model_dump() if hasattr(e, "model_dump") else e for e in mini_out
            ]
        except Exception as e:
            warnings.warn(f"[mini-DISCERN] entry {entry.get('sample_idx')} failed: {e}")
            result["mini_discern_score"] = None

    return result


def _print_single_result(entry: dict):
    """Pretty-print DISCERN results for a single pair."""
    print("\n" + "=" * 60)
    print(f"DISCERN Score      : {entry.get('discern_score', 'N/A')}")
    print(f"mini-DISCERN Score : {entry.get('mini_discern_score', 'N/A')}")
    print("=" * 60)
    evals = entry.get("discern_evaluation") or []
    for i, ent in enumerate(evals, 1):
        name = ent.get("entity") or ent.get("entity_name", "?")
        sig  = ent.get("significance_score") or ent.get("clinical_significance_score", "?")
        rat  = ent.get("rationale", "")
        disc = ent.get("discrepancy_type") or ent.get("diagnosis_concordance", "")
        print(f"\n[{i}] {name}")
        print(f"     Significance : {sig}/4")
        print(f"     Concordance  : {disc}")
        if rat:
            print(f"     Rationale    : {rat[:120]}")


def run(args: argparse.Namespace):
    cfg = _load_config(args.config)
    disc_cfg = cfg.get("discern") or {}

    model   = args.model   or disc_cfg.get("default_model",   "google/gemma-4-31B-it")
    backend = args.backend or disc_cfg.get("default_backend", "hf")
    max_concurrent = args.max_concurrent or disc_cfg.get("max_concurrent", 4)
    config_dir = _ROOT / "config"

    run_full = args.mode in ("full", "both")
    run_mini = args.mode in ("mini", "both")

    metadata = {
        "run_date":      str(date.today()),
        "backend":       backend,
        "model":         model,
        "mode":          "batch" if max_concurrent > 1 else "sequential",
        "max_concurrent": max_concurrent,
        "discern_mode":  args.mode,
        "tag":           args.tag or "",
    }

    # ── Single pair mode ──────────────────────────────────────────────────────
    if args.reference and args.candidate:
        entry = {
            "sample_idx": 0,
            "ground_truth_raw": args.reference,
            "generated_raw": args.candidate,
            "discern_score": None,
            "mini_discern_score": None,
        }
        result = _score_entry(entry, model, backend, config_dir, run_full, run_mini)
        _print_single_result(result)
        if args.output:
            _save(args.output, [result], metadata)
        return

    # ── Batch mode ────────────────────────────────────────────────────────────
    if not args.input:
        print("Error: provide --reference/--candidate for a single pair, or --input for batch mode.")
        sys.exit(1)

    entries  = _load_entries(args.input)
    existing = _load_existing(args.output) if args.output else []
    scored_map = {e.get("sample_idx"): e for e in existing}

    done_keys = (
        (["discern_score"] if run_full else []) +
        (["mini_discern_score"] if run_mini else [])
    )
    todo = [e for e in entries
            if not all(scored_map.get(e.get("sample_idx"), {}).get(k) is not None for k in done_keys)]

    print(f"[DISCERN] {len(entries)} total — {len(todo)} to score (max_concurrent={max_concurrent})")

    with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
        futures = {
            ex.submit(_score_entry, e, model, backend, config_dir, run_full, run_mini): e.get("sample_idx")
            for e in todo
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                scored_map[idx] = fut.result()
            except Exception as e:
                warnings.warn(f"Entry {idx} failed: {e}")
            if args.output:
                _save(args.output, list(scored_map.values()), metadata)

    results = list(scored_map.values())
    if args.output:
        _save(args.output, results, metadata)
        print(f"\nSaved → {args.output}  ({len(results)} entries)")
    else:
        for r in results:
            _print_single_result(r)


def main():
    parser = argparse.ArgumentParser(
        description="Run DISCERN and/or mini-DISCERN on a report pair or batch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Single-pair inputs
    parser.add_argument("--reference", default=None, help="Ground truth report text (single-pair mode)")
    parser.add_argument("--candidate", default=None, help="Candidate report text (single-pair mode)")
    # Batch inputs
    parser.add_argument("--input",  default=None, help="Input JSON or CSV file (batch mode)")
    parser.add_argument("--output", default=None, help="Output JSON file")
    # Mode
    parser.add_argument("--mode", choices=["full", "mini", "both"], default="both",
                        help="full = DISCERN only | mini = mini-DISCERN only | both (default)")
    # Backend
    parser.add_argument("--backend", default=None,
                        help="LLM backend (anthropic|openai|openrouter|databricks|hf)")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--max-concurrent", type=int, default=None,
                        help="Max parallel requests")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--tag", default=None, help="Run tag for metadata")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
