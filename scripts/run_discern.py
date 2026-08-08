"""
run_discern.py — DISCERN evaluation on ReXVal and RaDEvalX datasets.

Runs DISCERN, mini-DISCERN, NLP metrics, and GREEN on the radiologist-annotated
benchmark datasets. Outputs a JSON file per dataset that can be used for
correlation analysis with radiologist error labels.

Datasets (download from PhysioNet — credentialed access required):
  ReXVal   : data/rexval/rexval_reports_long.csv
  RaDEvalX : data/radevalx/radevalx_report.csv

Examples
--------
  # Run all metrics on both datasets with default model from config.yaml
  python scripts/run_discern.py --dataset both

  # Mini-DISCERN only on ReXVal, first 10 rows
  python scripts/run_discern.py --dataset rexval --skip-nlp --skip-green \\
      --skip-discern --count 10

  # Specific model, with run tag for output filename versioning
  python scripts/run_discern.py --dataset radevalx \\
      --model databricks-claude-sonnet-4-6 --run-tag v1

  # vLLM model (auto-uses batch mode)
  python scripts/run_discern.py --dataset both \\
      --model google/gemma-4-31B-it --run-tag batch_v0
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import yaml

DEFAULT_REXVAL_INPUT   = _ROOT / "data/rexval/rexval_reports_long.csv"
DEFAULT_RADEVALX_INPUT = _ROOT / "data/radevalx/radevalx_report.csv"
DEFAULT_OUTPUT_DIR     = _ROOT / "data/discern_runs"

_CONFIG_DIR             = _ROOT / "config"
PROMPT_YAML_PATH        = _CONFIG_DIR / "entity_extraction_prompt.yaml"
ENTITIES_YAML_PATH      = _CONFIG_DIR / "entities.yaml"
ATTRIBUTE_PROMPT_PATH   = _CONFIG_DIR / "attribute_extraction_prompt.yaml"
SIGNIFICANCE_YAML_PATH  = _CONFIG_DIR / "significance_prompt.yaml"
DIAG_ENTITIES_YAML_PATH = _CONFIG_DIR / "diagnosis.yaml"
MERGED_PROMPT_YAML_PATH = _CONFIG_DIR / "merged_prompt.yaml"

DISCERN_KEYS     = {"discern_score", "discern_evaluation"}
MINI_DISCERN_KEYS = {"mini_discern_score", "mini_discern_evaluation"}
NLP_KEYS         = {"bleu", "rouge", "meteor", "bertscore", "radgraph"}
GREEN_KEYS       = {"green"}


def load_config(path: Optional[str] = None) -> dict:
    cfg_path = Path(path) if path else _ROOT / "config.yaml"
    if not cfg_path.exists():
        warnings.warn(
            f"config.yaml not found at {cfg_path}. "
            "Copy config.example.yaml to config.yaml and fill in your credentials."
        )
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def _has_all_keys(entry: dict, keys: set) -> bool:
    return keys.issubset(entry) and all(
        entry[k] is not None and entry[k] != "None" for k in keys
    )


def _is_valid_report(text: str) -> bool:
    if not text or not text.strip():
        return False
    return sum(c.isascii() for c in text) / max(len(text), 1) >= 0.8


def _serialize_entities(entities) -> List[dict]:
    out = []
    for e in entities:
        if hasattr(e, "model_dump"):
            out.append(e.model_dump())
        elif isinstance(e, dict):
            out.append(e)
        else:
            out.append(str(e))
    return out


def _save(path: Path, results: List[dict], meta: Optional[dict] = None):
    payload = {**(meta or {}), "results": results}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def _load_pairs(dataset: str, input_csv: Path, count: Optional[int]) -> List[dict]:
    df = pd.read_csv(input_csv)
    if count is not None:
        df = df.head(count)

    records = []
    if dataset == "rexval":
        required = {"study_id", "gt_report", "candidate_report", "candidate_reporter"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {input_csv}: {sorted(missing)}")
        for i, row in enumerate(df.itertuples(index=False)):
            records.append({
                "sample_idx": i,
                "dataset": dataset,
                "study_id": int(row.study_id),
                "candidate_reporter": str(row.candidate_reporter),
                "ground_truth_raw": str(row.gt_report or ""),
                "generated_raw": str(row.candidate_report or ""),
            })
    elif dataset == "radevalx":
        required = {"study_id", "ground_truth", "candidate_report"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {input_csv}: {sorted(missing)}")
        for i, row in enumerate(df.itertuples(index=False)):
            records.append({
                "sample_idx": i,
                "dataset": dataset,
                "study_id": int(row.study_id),
                "ground_truth_raw": str(row.ground_truth or ""),
                "generated_raw": str(row.candidate_report or ""),
            })
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return records


def _merge_resume(input_records: List[dict], output_path: Path) -> Tuple[List[dict], dict]:
    if not output_path.exists():
        return input_records, {}
    with output_path.open() as f:
        existing = json.load(f)
    existing_map = {
        r["sample_idx"]: r
        for r in existing.get("results", [])
        if isinstance(r, dict) and "sample_idx" in r
    }
    keep_keys = (DISCERN_KEYS | MINI_DISCERN_KEYS | NLP_KEYS | GREEN_KEYS
                 | {"discern_error", "mini_discern_error", "green_error"})
    merged = []
    for rec in input_records:
        prev = existing_map.get(rec["sample_idx"], {})
        merged.append({**rec, **{k: v for k, v in prev.items() if k in keep_keys}})
    meta = {k: v for k, v in existing.items() if k != "results"}
    return merged, meta


def _output_path(dataset: str, output_dir: Path, model: str, run_tag: str) -> Path:
    safe_model = model.replace("/", "_").replace("-", "_")
    tag = f"_{run_tag}" if run_tag else ""
    return output_dir / f"{dataset}_all_metrics_{safe_model}{tag}.json"


def run_dataset(
    dataset: str,
    input_csv: Path,
    output_json: Path,
    model: str,
    db_token: str,
    db_host: str,
    hf_token: str,
    green_model: str,
    green_python: Optional[str],
    run_nlp: bool,
    run_green: bool,
    run_discern: bool,
    run_mini_discern: bool,
    count: Optional[int],
):
    from metrics.nlp import (compute_bleu1, compute_rougel, compute_meteor,
                               compute_bertscore_single, compute_radgraphf1_single)
    from metrics.green import compute_green
    from evaluate_reports import run_evaluation, run_evaluation_batch
    from evaluate_single_prompt import evaluate_reports, evaluate_reports_batch
    from llm_backend import TokenLimitError, InputTooLongError

    print(f"\n{'=' * 70}")
    print(f"Dataset : {dataset}")
    print(f"Input   : {input_csv}")
    print(f"Output  : {output_json}")
    print(f"Model   : {model}")

    is_gemma3 = "gemma-3" in model.lower() or "gemma_3" in model.lower()
    DISCERN_MAX  = 8192 if is_gemma3 else 25000
    MINI_MAX     = 8192 if is_gemma3 else 10000
    DISCERN_INIT = 5000

    # Auto-detect batch mode
    use_batch = not model.lower().startswith("databricks-")

    records = _load_pairs(dataset, input_csv, count)
    results, meta = _merge_resume(records, output_json)
    print(f"Loaded {len(results)} pairs.")

    valid, skipped = [], 0
    for entry in results:
        if _is_valid_report(entry["ground_truth_raw"]) and \
           _is_valid_report(entry["generated_raw"]):
            valid.append(entry)
        else:
            for k in ["discern_score", "discern_evaluation",
                      "mini_discern_score", "mini_discern_evaluation"]:
                entry.setdefault(k, None)
            skipped += 1
    if skipped:
        print(f"Skipped {skipped} invalid pairs.")

    # ── NLP ───────────────────────────────────────────────────────────────────
    if run_nlp:
        todo = [e for e in valid if not _has_all_keys(e, NLP_KEYS)]
        print(f"\n[NLP] {len(todo)} pending.")
        for i, entry in enumerate(todo, 1):
            ref, cand = entry["ground_truth_raw"], entry["generated_raw"]
            entry["bleu"]      = compute_bleu1(cand, ref)
            entry["rouge"]     = compute_rougel(cand, ref)
            entry["meteor"]    = compute_meteor(cand, ref)
            entry["bertscore"] = compute_bertscore_single(cand, ref)
            entry["radgraph"]  = compute_radgraphf1_single(cand, ref)
            if i % 25 == 0 or i == len(todo):
                print(f"  NLP: {i}/{len(todo)}")
        _save(output_json, results, meta)

    # ── GREEN ─────────────────────────────────────────────────────────────────
    if run_green:
        todo = [e for e in valid if not _has_all_keys(e, GREEN_KEYS)]
        print(f"\n[GREEN] {len(todo)} pending.")
        if todo:
            refs  = [e["ground_truth_raw"] for e in todo]
            cands = [e["generated_raw"]    for e in todo]
            scores = compute_green(cands, refs, model_name=green_model,
                                   python_bin=green_python)
            for entry, score in zip(todo, scores):
                entry["green"] = score
                entry.pop("green_error", None)
            _save(output_json, results, meta)

    # ── DISCERN ───────────────────────────────────────────────────────────────
    if run_discern:
        todo = [e for e in valid if not _has_all_keys(e, DISCERN_KEYS)]
        print(f"\n[DISCERN] {len(todo)} pending.")
        if use_batch and todo:
            refs  = [e["ground_truth_raw"] for e in todo]
            cands = [e["generated_raw"]    for e in todo]
            batch_results = run_evaluation_batch(
                pairs=list(zip(refs, cands)),
                model=model,
                token_path=db_token,
                prompt_yaml_path=str(PROMPT_YAML_PATH),
                entities_yaml_path=str(ENTITIES_YAML_PATH),
                attribute_prompt_path=str(ATTRIBUTE_PROMPT_PATH),
                significance_yaml_path=str(SIGNIFICANCE_YAML_PATH),
                max_tokens=DISCERN_INIT,
            )
            for entry, result in zip(todo, batch_results):
                if result is not None:
                    discern_eval, discern_score = result
                    entry["discern_score"] = discern_score
                    entry["discern_evaluation"] = discern_eval
                    entry.pop("discern_error", None)
                else:
                    entry["discern_score"] = None
                    entry["discern_evaluation"] = None
                    entry["discern_error"] = "Batch evaluation failed"
            _save(output_json, results, meta)
        else:
            for i, entry in enumerate(todo, 1):
                idx = entry["sample_idx"]
                print(f"  [{i}/{len(todo)}] idx={idx} ...", end=" ", flush=True)
                current_max = DISCERN_INIT
                last_exc = None
                success = False
                while current_max <= DISCERN_MAX:
                    try:
                        discern_eval, discern_score = run_evaluation(
                            report_text=entry["ground_truth_raw"],
                            candidate_text=entry["generated_raw"],
                            model=model,
                            token_path=db_token,
                            prompt_yaml_path=str(PROMPT_YAML_PATH),
                            entities_yaml_path=str(ENTITIES_YAML_PATH),
                            attribute_prompt_path=str(ATTRIBUTE_PROMPT_PATH),
                            significance_yaml_path=str(SIGNIFICANCE_YAML_PATH),
                            max_tokens=current_max,
                        )
                        entry["discern_score"] = discern_score
                        entry["discern_evaluation"] = discern_eval
                        entry.pop("discern_error", None)
                        print(f"score={discern_score}")
                        success = True
                        break
                    except (TokenLimitError, InputTooLongError) as exc:
                        last_exc = exc
                        new_max = min(int(current_max * 1.5), DISCERN_MAX)
                        print(f"\n    escalating {current_max}→{new_max} ...", end=" ", flush=True)
                        current_max = new_max
                    except Exception as exc:
                        if "TokenLimitError" in str(exc) or "InputTooLongError" in str(exc):
                            last_exc = exc
                            new_max = min(int(current_max * 1.5), DISCERN_MAX)
                            current_max = new_max
                        else:
                            last_exc = exc
                            break
                if not success:
                    entry["discern_score"] = None
                    entry["discern_evaluation"] = None
                    entry["discern_error"] = str(last_exc)
                    print(f"FAILED: {last_exc}")
                _save(output_json, results, meta)

    # ── mini-DISCERN ──────────────────────────────────────────────────────────
    if run_mini_discern:
        todo = [e for e in valid if not _has_all_keys(e, MINI_DISCERN_KEYS)]
        print(f"\n[mini-DISCERN] {len(todo)} pending.")
        if use_batch and todo:
            refs  = [e["ground_truth_raw"] for e in todo]
            cands = [e["generated_raw"]    for e in todo]
            batch_results = evaluate_reports_batch(
                reference_reports=refs,
                candidate_reports=cands,
                entity_list_path=str(DIAG_ENTITIES_YAML_PATH),
                prompt_path=str(MERGED_PROMPT_YAML_PATH),
                model=model,
                token_path=db_token,
                max_tokens=MINI_MAX,
                temperature=0.1,
                max_retries=3,
                max_concurrent=3,
            )
            for entry, result in zip(todo, batch_results):
                if result is not None:
                    score = int(sum(e.clinical_significance_score for e in result))
                    entry["mini_discern_score"] = score
                    entry["mini_discern_evaluation"] = _serialize_entities(result)
                    entry.pop("mini_discern_error", None)
                else:
                    entry["mini_discern_score"] = None
                    entry["mini_discern_evaluation"] = None
                    entry["mini_discern_error"] = "Validation failed"
            _save(output_json, results, meta)
        else:
            for i, entry in enumerate(todo, 1):
                idx = entry["sample_idx"]
                print(f"  [{i}/{len(todo)}] idx={idx} ...", end=" ", flush=True)
                try:
                    mini = evaluate_reports(
                        reference_report=entry["ground_truth_raw"],
                        candidate_report=entry["generated_raw"],
                        entity_list_path=str(DIAG_ENTITIES_YAML_PATH),
                        prompt_path=str(MERGED_PROMPT_YAML_PATH),
                        model=model,
                        token_path=db_token,
                        max_tokens=MINI_MAX,
                        temperature=0.1,
                        max_retries=4,
                    )
                    score = int(sum(e.clinical_significance_score for e in mini))
                    entry["mini_discern_score"] = score
                    entry["mini_discern_evaluation"] = _serialize_entities(mini)
                    entry.pop("mini_discern_error", None)
                    print(f"score={score}")
                except Exception as exc:
                    entry["mini_discern_score"] = None
                    entry["mini_discern_evaluation"] = None
                    entry["mini_discern_error"] = str(exc)
                    print(f"FAILED: {exc}")
                _save(output_json, results, meta)

    _save(output_json, results, meta)
    print(f"\nSaved: {output_json}")
    print("Score summary:")
    for key in ["bleu", "rouge", "meteor", "bertscore", "radgraph",
                "green", "discern_score", "mini_discern_score"]:
        vals = [r[key] for r in results
                if r.get(key) is not None and isinstance(r[key], (int, float))]
        if vals:
            arr = np.array(vals, dtype=float)
            print(f"  {key:>22s}: mean={arr.mean():.4f}  std={arr.std():.4f}  n={len(arr)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run DISCERN evaluation on ReXVal and RaDEvalX benchmark datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", choices=["rexval", "radevalx", "both"], default="both")
    parser.add_argument("--model", default=None)
    parser.add_argument("--run-tag", default="", help="Tag appended to output filename.")
    parser.add_argument("--count", type=int, default=None,
                        help="Process only the first N pairs (useful for testing).")
    parser.add_argument("--rexval-input", type=Path, default=DEFAULT_REXVAL_INPUT)
    parser.add_argument("--radevalx-input", type=Path, default=DEFAULT_RADEVALX_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-nlp", action="store_true")
    parser.add_argument("--skip-green", action="store_true")
    parser.add_argument("--skip-discern", action="store_true")
    parser.add_argument("--skip-mini-discern", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    creds = cfg.get("credentials") or {}
    disc_cfg = cfg.get("discern") or {}
    conda_envs = cfg.get("conda_envs") or {}
    vllm_cfg = cfg.get("vllm") or {}

    model = args.model or disc_cfg.get("default_model", "databricks-claude-sonnet-4-6")
    db_token = creds.get("databricks_token", "")
    db_host  = creds.get("databricks_host", "")
    hf_token = creds.get("hf_token", "")
    green_model  = (cfg.get("checkpoints") or {}).get(
        "green_model", "StanfordAIMI/GREEN-radllama2-7b")
    green_python = conda_envs.get("green") or None

    import os
    if db_host:
        os.environ["DATABRICKS_SERVING_ENDPOINTS_URL"] = db_host
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("VLLM_TENSOR_PARALLEL_SIZE",
                          str(vllm_cfg.get("tensor_parallel_size", 1)))
    os.environ.setdefault("VLLM_GPU_MEM_UTIL",
                          str(vllm_cfg.get("gpu_memory_utilization", 0.90)))
    os.environ.setdefault("VLLM_MAX_MODEL_LEN",
                          str(vllm_cfg.get("max_model_len", 8192)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = ["rexval", "radevalx"] if args.dataset == "both" else [args.dataset]
    for dataset in datasets:
        input_csv  = args.rexval_input if dataset == "rexval" else args.radevalx_input
        output_json = _output_path(dataset, args.output_dir, model, args.run_tag)
        run_dataset(
            dataset=dataset,
            input_csv=input_csv,
            output_json=output_json,
            model=model,
            db_token=db_token,
            db_host=db_host,
            hf_token=hf_token,
            green_model=green_model,
            green_python=green_python,
            run_nlp=not args.skip_nlp,
            run_green=not args.skip_green,
            run_discern=not args.skip_discern,
            run_mini_discern=not args.skip_mini_discern,
            count=args.count,
        )


if __name__ == "__main__":
    main()
