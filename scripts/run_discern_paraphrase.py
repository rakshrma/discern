"""
run_discern_paraphrase.py — Run DISCERN scoring on paraphrased reports.

Reads the output of run_paraphrase.py, filters by transformation type,
and runs DISCERN + mini-DISCERN on (paraphrased_gt, original_gt) pairs.

A robust judge should score these pairs near-zero (content identical).
Variance across transformation types = instability metric.

Usage
-----
  python scripts/run_discern_paraphrase.py \\
      --input  data/robustness/paraphrase_robustness_v2.json \\
      --output data/discern_runs/paraphrase_eval/ \\
      --transforms rephrase

  python scripts/run_discern_paraphrase.py \\
      --input  data/robustness/paraphrase_robustness_v2.json \\
      --output data/discern_runs/paraphrase_eval/ \\
      --transforms all --max-concurrent 4
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


def _load_paraphrase_data(path: str, transforms: List[str]) -> Dict[str, List[Dict]]:
    """Load and group paraphrase data by source, filtered by transforms."""
    with open(path) as f:
        data = json.load(f)
    entries = data.get("results", data) if isinstance(data, dict) else data

    if "all" not in transforms:
        entries = [
            e for e in entries
            if any(t in (e.get("transformation_type") or []) for t in transforms)
        ]

    grouped: Dict[str, List[Dict]] = {}
    for e in entries:
        source = e.get("source", "unknown")
        grouped.setdefault(source, []).append(e)
    return grouped


def _save(path: str, results: List[Dict], metadata: dict):
    with open(path, "w") as f:
        json.dump({"_metadata": metadata, "results": results}, f, indent=2, default=str)


def _load_existing(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    return data.get("results", []) if isinstance(data, dict) else data


def _score_entry(entry: dict, model: str, config_dir: Path) -> dict:
    from evaluate_reports import run_evaluation
    from evaluate_single_prompt import evaluate_reports as mini_eval
    from get_discern_score import compute_reads_score

    result = dict(entry)
    gt   = entry.get("ground_truth_raw", "")
    cand = entry.get("generated_raw", "")
    token_path = str(config_dir / ".databricks.token")

    if result.get("discern_score") is None:
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
            warnings.warn(f"[DISCERN] sample {entry.get('sample_idx')} failed: {e}")
            result["discern_score"] = None

    if result.get("mini_discern_score") is None:
        try:
            mini_out = mini_eval(
                ground_truth=gt,
                candidate=cand,
                model=model,
                token_path=token_path,
                prompt_yaml_path=str(config_dir / "merged_prompt.yaml"),
                entities_yaml_path=str(config_dir / "diagnosis_only.yaml"),
            )
            result["mini_discern_score"] = compute_reads_score(mini_out)
        except Exception as e:
            warnings.warn(f"[mini-DISCERN] sample {entry.get('sample_idx')} failed: {e}")
            result["mini_discern_score"] = None

    return result


def run(args: argparse.Namespace):
    cfg = _load_config(args.config)
    disc_cfg = cfg.get("discern") or {}
    model = args.model or disc_cfg.get("default_model", "google/gemma-3-27b-it")
    max_concurrent = args.max_concurrent or disc_cfg.get("max_concurrent", 4)
    config_dir = _ROOT / "config"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    transforms = args.transforms if args.transforms else ["rephrase"]

    metadata = {
        "run_date":    str(date.today()),
        "model":       model,
        "transforms":  transforms,
        "max_concurrent": max_concurrent,
        "mode":        "batch" if max_concurrent > 1 else "sequential",
    }

    grouped = _load_paraphrase_data(args.input, transforms)

    for source, entries in grouped.items():
        model_tag = model.replace("/", "_").replace("-", "_")
        out_path = output_dir / f"{source}_paraphrase_{'_'.join(transforms)}_{model_tag}.json"
        existing = _load_existing(str(out_path))
        scored_map = {e.get("sample_idx"): e for e in existing}

        todo = [e for e in entries if scored_map.get(e.get("sample_idx"), {}).get("discern_score") is None]
        print(f"[{source}] {len(entries)} entries — {len(todo)} to score")

        with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
            futures = {
                ex.submit(_score_entry, e, model, config_dir): e.get("sample_idx")
                for e in todo
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    scored_map[idx] = fut.result()
                except Exception as e:
                    warnings.warn(f"Entry {idx} failed: {e}")
                _save(str(out_path), list(scored_map.values()), metadata)

        print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run DISCERN on paraphrased report pairs (robustness evaluation)."
    )
    parser.add_argument("--input",    required=True, help="Paraphrase JSON from run_paraphrase.py")
    parser.add_argument("--output",   required=True, help="Output directory")
    parser.add_argument("--transforms", nargs="+", default=["rephrase"],
                        help="Transform types to evaluate (default: rephrase). Use 'all' for all.")
    parser.add_argument("--model",    default=None)
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument("--config",   default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
