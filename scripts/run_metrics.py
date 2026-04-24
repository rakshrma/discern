"""
run_metrics.py — Single entry point for all DISCERN metrics.

Runs any combination of: BLEU, ROUGE, METEOR, BERTScore, SembScore,
RadGraph, RadCliQ, RaTEScore, GREEN, CRIMSON, BLUERT, FineRadScore,
DISCERN, mini-DISCERN on a JSON or CSV input file.

Input formats
-------------
  JSON  — {"results": [{sample_idx, ground_truth_raw, generated_raw, ...}, ...]}
           (compatible with vlm_cxr_benchmark chexpert_plus_valid_results.json)
  CSV   — columns: sample_idx (optional), reference, candidate
           (or ground_truth_raw / generated_raw column names also accepted)

Resume
------
  Re-running with the same --output will skip already-scored entries.

Examples
--------
  python scripts/run_metrics.py --input results.json --output scored.json
  python scripts/run_metrics.py --input results.json --output scored.json --metrics all
  python scripts/run_metrics.py --input pairs.csv --output scored.json --metrics nlp discern
  python scripts/run_metrics.py --input results.json --output scored.json \\
      --backend openrouter --model google/gemma-3-27b-it --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as script from repo root or scripts/ dir
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import yaml

METRIC_GROUPS = {
    "nlp":     ["bleu", "rouge", "meteor", "bertscore"],
    "radiology": ["semb_score", "radgraph", "radcliq", "ratescore"],
    "model":   ["green", "crimson", "bluert"],
    "llm":     ["fineradscor"],
    "discern": ["discern", "mini_discern"],
    "all":     ["bleu", "rouge", "meteor", "bertscore",
                "semb_score", "radgraph", "radcliq", "ratescore",
                "green", "crimson", "bluert", "fineradscor",
                "discern", "mini_discern"],
}


# ─── Config ──────────────────────────────────────────────────────────────────

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


def _merge_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Let CLI flags override config.yaml values."""
    if args.backend:
        cfg.setdefault("discern", {})["default_backend"] = args.backend
    if args.model:
        cfg.setdefault("discern", {})["default_model"] = args.model
    if args.max_concurrent:
        cfg.setdefault("discern", {})["max_concurrent"] = args.max_concurrent

    # Resolve metric toggles from --metrics flag
    if args.metrics:
        requested = set()
        for m in args.metrics:
            requested.update(METRIC_GROUPS.get(m, [m]))
        cfg.setdefault("metrics", {})
        for key in METRIC_GROUPS["all"]:
            cfg["metrics"][key] = key in requested
    return cfg


# ─── I/O ─────────────────────────────────────────────────────────────────────

def load_input(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        # Normalize column names
        col_map = {}
        for col in df.columns:
            lc = col.lower().replace(" ", "_")
            if lc in ("ground_truth_raw", "reference", "ref", "ground_truth"):
                col_map[col] = "ground_truth_raw"
            elif lc in ("generated_raw", "candidate", "cand", "generated", "hypothesis"):
                col_map[col] = "generated_raw"
        df = df.rename(columns=col_map)
        if "sample_idx" not in df.columns:
            df["sample_idx"] = list(range(len(df)))
        return df.to_dict(orient="records")
    else:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return data.get("results", data.get("entries", []))


def load_existing(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("results", [])


def save_output(path: str, results: List[Dict], metadata: dict, indent: int = 2):
    payload = {"_metadata": metadata, "results": results}
    with open(path, "w") as f:
        json.dump(payload, f, indent=indent, default=str)


def _already_scored(entry: dict, metric_keys: List[str]) -> bool:
    return all(k in entry and entry[k] is not None and entry[k] != "None" for k in metric_keys)


# ─── DISCERN paths (mirrors reads/src/run_discern_mini_pairs.py) ────────────

_CONFIG_DIR = _ROOT / "config"
TOKEN_PATH                = _CONFIG_DIR / ".databricks.token"
HF_TOKEN_PATH             = _CONFIG_DIR / ".hftoken"
PROMPT_YAML_PATH          = _CONFIG_DIR / "entity_extraction_prompt.yaml"
ENTITIES_YAML_PATH        = _CONFIG_DIR / "entities.yaml"
ATTRIBUTE_PROMPT_PATH     = _CONFIG_DIR / "attribute_extraction_prompt.yaml"
SIGNIFICANCE_YAML_PATH    = _CONFIG_DIR / "significance_prompt.yaml"
DIAG_ENTITIES_YAML_PATH   = _CONFIG_DIR / "diagnosis_only.yaml"
MERGED_PROMPT_YAML_PATH   = _CONFIG_DIR / "merged_prompt.yaml"


def _serialize_entity_evaluations(entities):
    out = []
    for e in entities:
        if hasattr(e, "model_dump"):
            out.append(e.model_dump())
        elif isinstance(e, dict):
            out.append(e)
        else:
            out.append(str(e))
    return out


def _run_discern_batch(todo, model, max_tokens, batch_size):
    from evaluate_reports import run_evaluation_batch
    refs  = [e.get("ground_truth_raw", "") for e in todo]
    cands = [e.get("generated_raw", "")   for e in todo]
    return run_evaluation_batch(
        pairs=list(zip(refs, cands)),
        model=model,
        token_path=str(TOKEN_PATH),
        prompt_yaml_path=str(PROMPT_YAML_PATH),
        entities_yaml_path=str(ENTITIES_YAML_PATH),
        attribute_prompt_path=str(ATTRIBUTE_PROMPT_PATH),
        significance_yaml_path=str(SIGNIFICANCE_YAML_PATH),
        hf_token_path=str(HF_TOKEN_PATH),
        max_tokens=max_tokens,
        batch_size=batch_size,
    )


def _run_mini_batch(todo, model, max_tokens, max_concurrent):
    from evaluate_single_prompt import evaluate_reports_batch
    refs  = [e.get("ground_truth_raw", "") for e in todo]
    cands = [e.get("generated_raw", "")   for e in todo]
    return evaluate_reports_batch(
        reference_reports=refs,
        candidate_reports=cands,
        entity_list_path=str(DIAG_ENTITIES_YAML_PATH),
        prompt_path=str(MERGED_PROMPT_YAML_PATH),
        model=model,
        token_path=str(TOKEN_PATH),
        hf_token_path=str(HF_TOKEN_PATH),
        max_tokens=max_tokens,
        temperature=0.1,
        max_retries=3,
        max_concurrent=max_concurrent,
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace):
    cfg = load_config(args.config)
    cfg = _merge_cli_overrides(cfg, args)

    out_cfg = cfg.get("output") or {}
    save_intermediate = out_cfg.get("save_intermediate", True)
    indent = out_cfg.get("indent", 2)

    entries = load_input(args.input)
    existing = load_existing(args.output) if args.output else []
    existing_by_idx = {e.get("sample_idx"): e for e in existing}

    metrics_cfg = cfg.get("metrics") or {}
    run_discern = metrics_cfg.get("discern", True)
    run_mini = metrics_cfg.get("mini_discern", True)
    run_crimson = metrics_cfg.get("crimson", False)

    disc_cfg = cfg.get("discern") or {}
    max_concurrent = disc_cfg.get("max_concurrent", 4)

    metadata = {
        "run_date": str(date.today()),
        "input": str(args.input),
        "backend": disc_cfg.get("default_backend", "openrouter"),
        "model": disc_cfg.get("default_model", "google/gemma-3-27b-it"),
        "mode": "batch" if max_concurrent > 1 else "sequential",
        "max_concurrent": max_concurrent,
        "metrics_enabled": [k for k, v in metrics_cfg.items() if v],
        "tag": args.tag or "",
    }

    results = list(existing)

    # ── NLP + model-based metrics (batched) ───────────────────────────────────
    nlp_metric_keys = [k for k in ["bleu", "rouge", "meteor", "bertscore",
                                    "semb_score", "radgraph", "ratescore",
                                    "green", "bluert", "fineradscor"]
                       if metrics_cfg.get(k)]

    if nlp_metric_keys:
        from metrics.registry import MetricRegistry
        registry = MetricRegistry(cfg)

        todo_entries = [e for e in entries
                        if not _already_scored(existing_by_idx.get(e.get("sample_idx"), {}),
                                               nlp_metric_keys)]
        if todo_entries:
            print(f"[metrics] Running {nlp_metric_keys} on {len(todo_entries)} entries ...")
            candidates = [e.get("generated_raw", "") for e in todo_entries]
            references = [e.get("ground_truth_raw", "") for e in todo_entries]
            scores = registry.compute_all(candidates, references)

            for i, entry in enumerate(todo_entries):
                idx = entry.get("sample_idx")
                merged = dict(existing_by_idx.get(idx, entry))
                for metric, vals in scores.items():
                    merged[metric] = vals[i] if vals else None
                existing_by_idx[idx] = merged

            results = list(existing_by_idx.values())
            if save_intermediate:
                save_output(args.output, results, metadata, indent)
                print(f"  Saved intermediate → {args.output}")

    # ── RadCliQ (needs sub-metrics already scored) ─────────────────────────────
    if metrics_cfg.get("radcliq"):
        from metrics.radcliq import compute_radcliq
        for entry in results:
            if entry.get("radcliq") is None:
                entry["radcliq"] = compute_radcliq(
                    [entry.get("bleu")], [entry.get("bertscore")],
                    [entry.get("semb_score")], [entry.get("radgraph")],
                )[0]
        if save_intermediate:
            save_output(args.output, results, metadata, indent)

    # ── DISCERN (single batched vLLM pass, mirrors reads/run_discern_mini_pairs) ─
    model_name = disc_cfg.get("default_model", "google/gemma-3-27b-it")
    discern_batch_size = disc_cfg.get("batch_size", 200)
    DISCERN_MAX_TOKENS = 8192 if "gemma_3" in model_name.lower() else 25000
    MINI_MAX_TOKENS    = 8192 if "gemma_3" in model_name.lower() else 10000

    scored_map = {e.get("sample_idx"): e for e in results}

    if run_discern:
        todo = [e for e in results if e.get("discern_score") is None]
        if todo:
            print(f"[DISCERN] Batch-scoring {len(todo)} entries (model={model_name}) ...")
            try:
                batch_results = _run_discern_batch(
                    todo, model_name, DISCERN_MAX_TOKENS, discern_batch_size,
                )
            except Exception as e:
                warnings.warn(f"[DISCERN] batch failed: {e}")
                batch_results = [None] * len(todo)

            for entry, res in zip(todo, batch_results):
                idx = entry.get("sample_idx")
                target = scored_map.get(idx, entry)
                if res is not None:
                    reads_eval, discern_score = res
                    target["discern_score"] = discern_score
                    target["discern_evaluation"] = _serialize_entity_evaluations(reads_eval)
                    target.pop("discern_error", None)
                else:
                    target["discern_score"] = None
                    target["discern_evaluation"] = None
                    target["discern_error"] = "Batch evaluation failed"
                scored_map[idx] = target
            results = list(scored_map.values())
            if save_intermediate:
                save_output(args.output, results, metadata, indent)

    if run_mini:
        todo = [e for e in results if e.get("mini_discern_score") is None]
        if todo:
            print(f"[mini-DISCERN] Batch-scoring {len(todo)} entries (model={model_name}) ...")
            try:
                batch_results = _run_mini_batch(
                    todo, model_name, MINI_MAX_TOKENS, max_concurrent,
                )
            except Exception as e:
                warnings.warn(f"[mini-DISCERN] batch failed: {e}")
                batch_results = [None] * len(todo)

            for entry, res in zip(todo, batch_results):
                idx = entry.get("sample_idx")
                target = scored_map.get(idx, entry)
                if res is not None:
                    score = int(sum(ent.clinical_significance_score for ent in res))
                    target["mini_discern_score"] = score
                    target["mini_discern_evaluation"] = _serialize_entity_evaluations(res)
                    target.pop("mini_discern_error", None)
                else:
                    target["mini_discern_score"] = None
                    target["mini_discern_evaluation"] = None
                    target["mini_discern_error"] = "Validation failed after retries"
                scored_map[idx] = target
            results = list(scored_map.values())
            if save_intermediate:
                save_output(args.output, results, metadata, indent)

    # ── CRIMSON (subprocess to crimson env) ────────────────────────────────────
    if run_crimson:
        from metrics.crimson import compute_crimson
        todo_crimson = [e for e in results if e.get("crimson") is None]
        if todo_crimson:
            print(f"[CRIMSON] Scoring {len(todo_crimson)} entries ...")
            candidates = [e.get("generated_raw", "") for e in todo_crimson]
            references = [e.get("ground_truth_raw", "") for e in todo_crimson]
            crimson_model = (cfg.get("checkpoints") or {}).get("crimson_model",
                             "rajpurkarlab/medgemma-4b-it-crimson")
            scores = compute_crimson(candidates, references, model_name=crimson_model)
            idx_map = {e.get("sample_idx"): e for e in results}
            for entry, score in zip(todo_crimson, scores):
                idx_map[entry.get("sample_idx")]["crimson"] = score
            results = list(idx_map.values())
            if save_intermediate:
                save_output(args.output, results, metadata, indent)

    # ── Final save ────────────────────────────────────────────────────────────
    save_output(args.output, results, metadata, indent)
    print(f"\nDone. Results saved to {args.output}  ({len(results)} entries)")


def main():
    parser = argparse.ArgumentParser(
        description="Run all DISCERN metrics on a JSON or CSV input file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input JSON or CSV file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument(
        "--metrics", nargs="+", default=None,
        help=(
            "Metrics to run. Can be group names (all, nlp, radiology, model, discern) "
            "or individual metric names (bleu, rouge, bertscore, radgraph, radcliq, "
            "ratescore, green, crimson, bluert, fineradscor, discern, mini_discern). "
            "Defaults to all enabled metrics in config.yaml."
        ),
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: ./config.yaml)")
    parser.add_argument("--backend", default=None,
                        help="LLM backend override (anthropic|openai|openrouter|databricks|hf)")
    parser.add_argument("--model", default=None, help="LLM model name override")
    parser.add_argument("--max-concurrent", type=int, default=None,
                        help="Max parallel DISCERN requests (default from config.yaml)")
    parser.add_argument("--tag", default=None, help="Optional run tag for metadata")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
