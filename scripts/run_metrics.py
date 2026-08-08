"""
run_metrics.py — Evaluate radiology report pairs with any combination of metrics.

This is the primary entry point for users who have a CSV or JSON file of
reference/candidate report pairs and want to score them.

Input formats
-------------
  JSON  — {"results": [{sample_idx, ground_truth_raw, generated_raw, ...}, ...]}
  CSV   — columns: reference, candidate  (or ground_truth_raw / generated_raw)

Resume support
--------------
  Re-running with the same --output file will skip already-scored entries.

Examples
--------
  # Score a CSV with NLP metrics only
  python scripts/run_metrics.py \\
      --input pairs.csv --output scored.json --metrics nlp

  # Score a JSON with all enabled metrics (from config.yaml)
  python scripts/run_metrics.py \\
      --input results.json --output scored.json

  # Quick test on first 5 rows
  python scripts/run_metrics.py \\
      --input pairs.csv --output test.json --count 5

  # Use a specific model, override config
  python scripts/run_metrics.py \\
      --input pairs.csv --output scored.json \\
      --model databricks-claude-sonnet-4-6 --metrics nlp discern
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import yaml

METRIC_GROUPS = {
    "nlp":      ["bleu", "rouge", "meteor", "bertscore", "radgraph"],
    "model":    ["green", "crimson"],
    "discern":  ["discern", "mini_discern"],
    "all":      ["bleu", "rouge", "meteor", "bertscore", "radgraph",
                 "green", "crimson", "discern", "mini_discern"],
}

_CONFIG_DIR = _ROOT / "config"
PROMPT_YAML_PATH       = _CONFIG_DIR / "entity_extraction_prompt.yaml"
ENTITIES_YAML_PATH     = _CONFIG_DIR / "entities.yaml"
ATTRIBUTE_PROMPT_PATH  = _CONFIG_DIR / "attribute_extraction_prompt.yaml"
SIGNIFICANCE_YAML_PATH = _CONFIG_DIR / "significance_prompt.yaml"
DIAG_ENTITIES_YAML_PATH = _CONFIG_DIR / "diagnosis.yaml"
MERGED_PROMPT_YAML_PATH = _CONFIG_DIR / "merged_prompt.yaml"


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: Optional[str] = None) -> dict:
    cfg_path = Path(path) if path else _ROOT / "config.yaml"
    if not cfg_path.exists():
        example = _ROOT / "config.example.yaml"
        if example.exists():
            warnings.warn(
                f"config.yaml not found at {cfg_path}. "
                "Copy config.example.yaml to config.yaml and fill in your credentials."
            )
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def _get_credentials(cfg: dict) -> dict:
    return cfg.get("credentials") or {}


def _get_discern_cfg(cfg: dict) -> dict:
    return cfg.get("discern") or {}


def _get_vllm_cfg(cfg: dict) -> dict:
    return cfg.get("vllm") or {}


def _get_conda_envs(cfg: dict) -> dict:
    return cfg.get("conda_envs") or {}


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_input(path: str, count: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load report pairs from a CSV or JSON file.

    CSV column aliases accepted:
      reference, ref, gt_report, ground_truth, ground_truth_raw → ground_truth_raw
      candidate, cand, generated, hypothesis, generated_raw     → generated_raw
    """
    p = Path(path)
    if p.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        col_map = {}
        for col in df.columns:
            lc = col.lower().replace(" ", "_")
            if lc in ("ground_truth_raw", "reference", "ref", "gt_report", "ground_truth"):
                col_map[col] = "ground_truth_raw"
            elif lc in ("generated_raw", "candidate", "cand", "generated", "hypothesis"):
                col_map[col] = "generated_raw"
        df = df.rename(columns=col_map)
        if "sample_idx" not in df.columns:
            df.insert(0, "sample_idx", range(len(df)))
        if count is not None:
            df = df.head(count)
        return df.to_dict(orient="records")
    else:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            records = data
        else:
            records = data.get("results", data.get("entries", []))
        if count is not None:
            records = records[:count]
        return records


def load_existing(path: str) -> Dict[Any, dict]:
    """Load already-scored entries from an output file, keyed by sample_idx."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    records = data.get("results", data) if isinstance(data, dict) else data
    return {r.get("sample_idx"): r for r in records if "sample_idx" in r}


def save_output(path: str, results: List[dict], metadata: dict, indent: int = 2):
    payload = {"_metadata": metadata, "results": results}
    with open(path, "w") as f:
        json.dump(payload, f, indent=indent, default=str)


def _has_all_keys(entry: dict, keys: List[str]) -> bool:
    import math
    for k in keys:
        if k not in entry:
            return False
        v = entry[k]
        if v is None or v == "None":
            return False
        if isinstance(v, float) and math.isnan(v):
            return False
    return True


def _is_valid_report(text: str) -> bool:
    if not text or not text.strip():
        return False
    ascii_ratio = sum(c.isascii() for c in text) / max(len(text), 1)
    return ascii_ratio >= 0.8


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


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace):
    cfg = load_config(args.config)
    creds = _get_credentials(cfg)
    disc_cfg = _get_discern_cfg(cfg)
    vllm_cfg = _get_vllm_cfg(cfg)
    conda_envs = _get_conda_envs(cfg)
    metrics_cfg = cfg.get("metrics") or {}
    out_cfg = cfg.get("output") or {}

    # CLI --metrics overrides config.yaml
    if args.metrics:
        requested = set()
        for m in args.metrics:
            requested.update(METRIC_GROUPS.get(m, [m]))
        metrics_cfg = {k: (k in requested) for k in METRIC_GROUPS["all"]}

    # Apply --skip-* flags
    if args.skip_nlp:
        for k in ["bleu", "rouge", "meteor", "bertscore", "radgraph"]:
            metrics_cfg[k] = False
    if args.skip_green:
        metrics_cfg["green"] = False
    if args.skip_crimson:
        metrics_cfg["crimson"] = False
    if args.skip_discern:
        metrics_cfg["discern"] = False
    if args.skip_mini_discern:
        metrics_cfg["mini_discern"] = False

    # Model and backend
    model = args.model or disc_cfg.get("default_model", "databricks-claude-sonnet-4-6")
    db_token = creds.get("databricks_token", "")
    db_host = creds.get("databricks_host", "")
    hf_token = creds.get("hf_token", "")
    max_concurrent = disc_cfg.get("max_concurrent", 3)

    # Export credentials as env vars so internal pipeline query_llm calls find them
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

    # Token limits (gemma-3 has 8k context; others use 25k)
    is_gemma3 = "gemma-3" in model.lower() or "gemma_3" in model.lower()
    DISCERN_MAX_TOKENS = 8192 if is_gemma3 else 25000
    MINI_MAX_TOKENS    = 8192 if is_gemma3 else 10000

    save_intermediate = out_cfg.get("save_intermediate", True)
    indent = out_cfg.get("indent", 2)

    entries = load_input(args.input, count=args.count)
    existing_map = load_existing(args.output) if args.output and Path(args.output).exists() else {}

    # Merge existing scores into fresh entries
    results: List[dict] = []
    for entry in entries:
        idx = entry.get("sample_idx")
        prev = existing_map.get(idx, {})
        merged = {**entry, **{k: v for k, v in prev.items()
                              if k not in entry or v is not None}}
        results.append(merged)

    _run_started = datetime.now().astimezone()
    _input_abs   = Path(args.input).resolve()
    _output_abs  = Path(args.output).resolve() if args.output else None
    _config_abs  = (Path(args.config).resolve() if args.config
                    else (_ROOT / "config.yaml").resolve())

    metadata = {
        "run_date"        : str(date.today()),
        "run_started_at"  : _run_started.isoformat(timespec="seconds"),
        "run_ended_at"    : None,
        "wall_seconds"    : None,
        "input"           : str(_input_abs),
        "input_dir"       : str(_input_abs.parent),
        "output"          : str(_output_abs) if _output_abs else None,
        "config_path"     : str(_config_abs) if _config_abs.exists() else None,
        "metrics_enabled" : sorted(k for k, v in metrics_cfg.items() if v),
        "count"           : args.count,
        "models": {
            "discern_llm" : model,
            "crimson"     : (cfg.get("checkpoints") or {}).get(
                                "crimson_model", "rajpurkarlab/medgemma-4b-it-crimson"),
            "green"       : (cfg.get("checkpoints") or {}).get(
                                "green_model",   "StanfordAIMI/GREEN-radllama2-7b"),
        },
        "discern_max_tokens": {
            "discern"     : DISCERN_MAX_TOKENS,
            "mini_discern": MINI_MAX_TOKENS,
        },
        "crimson": {
            "batch_size": int((cfg.get("crimson") or {}).get("batch_size", 8)),
        },
        "vllm": {
            "tensor_parallel_size"  : int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            "gpu_memory_utilization": float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.90")),
            "max_model_len"         : int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192")),
        },
        "env": {
            "hostname"           : socket.gethostname(),
            "python_executable"  : sys.executable,
            "python_version"     : ".".join(map(str, sys.version_info[:3])),
            "slurm_job_id"       : os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id" : os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_partition"    : os.environ.get("SLURM_JOB_PARTITION"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }

    print(f"Loaded {len(results)} report pairs.")
    print(f"Metrics enabled: {metadata['metrics_enabled']}")
    print(f"Model: {model}")

    # Filter invalid pairs but keep them in results with nulls
    valid: List[dict] = []
    for entry in results:
        if _is_valid_report(str(entry.get("ground_truth_raw", ""))) and \
           _is_valid_report(str(entry.get("generated_raw", ""))):
            valid.append(entry)
        else:
            entry.setdefault("discern_score", None)
            entry.setdefault("mini_discern_score", None)

    print(f"Valid pairs: {len(valid)}/{len(results)}")

    # ── NLP metrics ───────────────────────────────────────────────────────────
    nlp_keys = [k for k in ["bleu", "rouge", "meteor", "bertscore", "radgraph"]
                if metrics_cfg.get(k)]
    if nlp_keys:
        from metrics.nlp import (compute_bleu1, compute_rougel, compute_meteor,
                                  compute_bertscore_single, compute_radgraphf1_single)
        todo = [e for e in valid if not _has_all_keys(e, nlp_keys)]
        print(f"\n[NLP] {len(todo)} pending.")
        for i, entry in enumerate(todo, 1):
            ref  = entry["ground_truth_raw"]
            cand = entry["generated_raw"]
            if metrics_cfg.get("bleu"):
                entry["bleu"] = compute_bleu1(cand, ref)
            if metrics_cfg.get("rouge"):
                entry["rouge"] = compute_rougel(cand, ref)
            if metrics_cfg.get("meteor"):
                entry["meteor"] = compute_meteor(cand, ref)
            if metrics_cfg.get("bertscore"):
                entry["bertscore"] = compute_bertscore_single(cand, ref)
            if metrics_cfg.get("radgraph"):
                entry["radgraph"] = compute_radgraphf1_single(cand, ref)
            if i % 50 == 0 or i == len(todo):
                print(f"  NLP: {i}/{len(todo)}")
        if save_intermediate and args.output:
            save_output(args.output, results, metadata, indent)

    # ── GREEN ─────────────────────────────────────────────────────────────────
    if metrics_cfg.get("green"):
        from metrics.green import compute_green
        todo = [e for e in valid if not _has_all_keys(e, ["green"])]
        print(f"\n[GREEN] {len(todo)} pending.")
        if todo:
            refs  = [e["ground_truth_raw"] for e in todo]
            cands = [e["generated_raw"]    for e in todo]
            green_model = (cfg.get("checkpoints") or {}).get(
                "green_model", "StanfordAIMI/GREEN-radllama2-7b")
            green_python = conda_envs.get("green") or None
            scores = compute_green(cands, refs, model_name=green_model,
                                   python_bin=green_python,
                                   timeout=getattr(args, "timeout", None))
            for entry, score in zip(todo, scores):
                entry["green"] = score
            if save_intermediate and args.output:
                save_output(args.output, results, metadata, indent)

    # ── CRIMSON ───────────────────────────────────────────────────────────────
    if metrics_cfg.get("crimson"):
        from metrics.crimson import compute_crimson
        todo = [e for e in valid if not _has_all_keys(e, ["crimson"])]
        print(f"\n[CRIMSON] {len(todo)} pending.")
        if todo:
            refs  = [e["ground_truth_raw"] for e in todo]
            cands = [e["generated_raw"]    for e in todo]
            crimson_model = (cfg.get("checkpoints") or {}).get(
                "crimson_model", "rajpurkarlab/medgemma-4b-it-crimson")
            crimson_python = conda_envs.get("crimson") or None
            crimson_batch_size = int(
                (cfg.get("crimson") or {}).get("batch_size", 8)
            )
            scores = compute_crimson(cands, refs, model_name=crimson_model,
                                     python_bin=crimson_python,
                                     batch_size=crimson_batch_size,
                                     timeout=getattr(args, "timeout", None))
            for entry, score in zip(todo, scores):
                entry["crimson"] = score
            if save_intermediate and args.output:
                save_output(args.output, results, metadata, indent)

    # ── DISCERN (full pipeline) ───────────────────────────────────────────────
    if metrics_cfg.get("discern"):
        from evaluate_reports import run_evaluation, run_evaluation_batch
        from llm_backend import TokenLimitError, InputTooLongError
        todo = [e for e in valid if not _has_all_keys(e, ["discern_score"])]
        print(f"\n[DISCERN] {len(todo)} pending.")

        # Auto-detect batch mode: vLLM models get batch processing
        use_batch = not model.lower().startswith("databricks-") and len(todo) > 1
        if use_batch:
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
                max_tokens=DISCERN_MAX_TOKENS,
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
            if save_intermediate and args.output:
                save_output(args.output, results, metadata, indent)
        else:
            max_tokens = DISCERN_MAX_TOKENS
            max_tokens_cap = DISCERN_MAX_TOKENS
            for i, entry in enumerate(todo, 1):
                print(f"  [{i}/{len(todo)}] sample_idx={entry.get('sample_idx')} ...",
                      end=" ", flush=True)
                current_max = max_tokens
                last_exc = None
                success = False
                while current_max <= max_tokens_cap:
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
                        new_max = min(int(current_max * 1.5), max_tokens_cap)
                        print(f"\n    token limit, escalating {current_max}→{new_max} ...",
                              end=" ", flush=True)
                        current_max = new_max
                    except Exception as exc:
                        last_exc = exc
                        break
                if not success:
                    entry["discern_score"] = None
                    entry["discern_evaluation"] = None
                    entry["discern_error"] = str(last_exc)
                    print(f"FAILED: {last_exc}")
                if save_intermediate and args.output:
                    save_output(args.output, results, metadata, indent)

    # ── mini-DISCERN ──────────────────────────────────────────────────────────
    if metrics_cfg.get("mini_discern"):
        from evaluate_single_prompt import evaluate_reports, evaluate_reports_batch
        todo = [e for e in valid if not _has_all_keys(e, ["mini_discern_score"])]
        print(f"\n[mini-DISCERN] {len(todo)} pending.")

        use_batch = not model.lower().startswith("databricks-") and len(todo) > 1
        if use_batch:
            refs  = [e["ground_truth_raw"] for e in todo]
            cands = [e["generated_raw"]    for e in todo]
            batch_results = evaluate_reports_batch(
                reference_reports=refs,
                candidate_reports=cands,
                entity_list_path=str(DIAG_ENTITIES_YAML_PATH),
                prompt_path=str(MERGED_PROMPT_YAML_PATH),
                model=model,
                token_path=db_token,
                max_tokens=MINI_MAX_TOKENS,
                temperature=0.1,
                max_retries=3,
                max_concurrent=max_concurrent,
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
            if save_intermediate and args.output:
                save_output(args.output, results, metadata, indent)
        else:
            for i, entry in enumerate(todo, 1):
                print(f"  [{i}/{len(todo)}] sample_idx={entry.get('sample_idx')} ...",
                      end=" ", flush=True)
                try:
                    mini = evaluate_reports(
                        reference_report=entry["ground_truth_raw"],
                        candidate_report=entry["generated_raw"],
                        entity_list_path=str(DIAG_ENTITIES_YAML_PATH),
                        prompt_path=str(MERGED_PROMPT_YAML_PATH),
                        model=model,
                        token_path=db_token,
                        max_tokens=MINI_MAX_TOKENS,
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
                if save_intermediate and args.output:
                    save_output(args.output, results, metadata, indent)

    # ── Score summary (mean / std / n per metric) ────────────────────────────
    import numpy as np
    SUMMARY_KEYS = ["bleu", "rouge", "meteor", "bertscore", "radgraph",
                    "green", "crimson", "discern_score", "mini_discern_score"]
    score_summary: Dict[str, Dict[str, float]] = {}
    for key in SUMMARY_KEYS:
        vals = [r[key] for r in results
                if r.get(key) is not None and isinstance(r[key], (int, float))]
        if vals:
            arr = np.array(vals, dtype=float)
            score_summary[key] = {
                "mean": round(float(arr.mean()), 4),
                "std":  round(float(arr.std()),  4),
                "n":    int(len(arr)),
            }
    metadata["score_summary"] = score_summary

    # ── Final save ────────────────────────────────────────────────────────────
    _run_ended = datetime.now().astimezone()
    metadata["run_ended_at"] = _run_ended.isoformat(timespec="seconds")
    metadata["wall_seconds"] = round((_run_ended - _run_started).total_seconds(), 2)
    if args.output:
        save_output(args.output, results, metadata, indent)
        print(f"\nSaved: {args.output}")

    # ── Human-readable summary ───────────────────────────────────────────────
    print("\nScore summary:")
    for key, s in score_summary.items():
        print(f"  {key:>20s}: mean={s['mean']:.4f}  std={s['std']:.4f}  n={s['n']}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate radiology report pairs with DISCERN, NLP, and other metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True,
                        help="Input CSV or JSON file of report pairs.")
    parser.add_argument("--output", required=True,
                        help="Output JSON path (resume-safe).")
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (default: config.yaml at repo root).")
    parser.add_argument("--model", default=None,
                        help="Override model from config.yaml.")
    parser.add_argument("--count", type=int, default=None,
                        help="Process only the first N pairs (useful for testing).")
    parser.add_argument("--metrics", nargs="+",
                        choices=list(METRIC_GROUPS.keys()) + list(METRIC_GROUPS["all"]),
                        help="Metrics to run (overrides config.yaml). "
                             "Groups: nlp, model, discern, all.")
    parser.add_argument("--skip-nlp", action="store_true")
    parser.add_argument("--skip-green", action="store_true")
    parser.add_argument("--skip-crimson", action="store_true")
    parser.add_argument("--skip-discern", action="store_true")
    parser.add_argument("--skip-mini-discern", action="store_true")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Per-subprocess timeout in seconds (passed to "
                             "GREEN/CRIMSON). Typically derived from remaining "
                             "SLURM walltime by the job wrapper.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
