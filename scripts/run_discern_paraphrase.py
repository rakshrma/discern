#!/usr/bin/env python3
"""
run_discern_paraphrase.py
==========================
Run DISCERN and mini-DISCERN on paraphrased reports from
data/robustness/paraphrase_robustness_v2.json.

Filters to transformation_type == ["rephrase"] (pure rephrase only),
groups by source (rexval, radevalx, chexpertplusvalid), and compares
the rephrased report (generated_raw) against the ground truth (ground_truth_raw).

Outputs (data/discern_runs/)
-----------------------------------
  {source}_paraphrase_rephrase_{model}.json  — one file per source

Supports resume: existing scores are preserved across re-runs.

Usage
-----
  python \\
      scripts/run_discern_paraphrase.py \\
      --model Qwen/Qwen3.5-122B-A10B \\
      --discern-batch --mini-batch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluate_reports import run_evaluation, run_evaluation_batch
from evaluate_single_prompt import evaluate_reports, evaluate_reports_batch
from llm_backend import TokenLimitError, InputTooLongError

REPO_ROOT             = Path(__file__).resolve().parents[1]
DEFAULT_INPUT         = REPO_ROOT / "data/robustness/paraphrase_robustness_v2.json"
DEFAULT_OUT           = REPO_ROOT / "data/discern_runs"

TOKEN_PATH            = ""   # resolved from config.yaml at runtime
HF_TOKEN_PATH         = ""   # resolved from config.yaml at runtime
PROMPT_YAML_PATH      = REPO_ROOT / "config/entity_extraction_prompt.yaml"
ENTITIES_YAML_PATH    = REPO_ROOT / "config/entities.yaml"
ATTRIBUTE_PROMPT_PATH = REPO_ROOT / "config/attribute_extraction_prompt.yaml"
SIGNIFICANCE_YAML_PATH= REPO_ROOT / "config/significance_prompt.yaml"
DIAG_ENTITIES_YAML    = REPO_ROOT / "config/diagnosis.yaml"
MERGED_PROMPT_PATH    = REPO_ROOT / "config/merged_prompt.yaml"

DEFAULT_MODEL = "databricks-claude-sonnet-4-6"

DISCERN_KEYS      = {"discern_score", "discern_evaluation"}
MINI_DISCERN_KEYS = {"mini_discern_score", "mini_discern_evaluation"}

SOURCES = ["rexval", "radevalx", "chexpertplusvalid"]

DISCERN_MAX_TOKENS_INIT = 5000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_all_keys(entry: dict, keys: set) -> bool:
    return keys.issubset(entry) and all(
        entry[k] is not None and entry[k] != "None" for k in keys
    )


def _is_valid(text: str) -> bool:
    if not text or not text.strip():
        return False
    if str(text).startswith("ERROR"):
        return False
    ascii_ratio = sum(c.isascii() for c in text) / max(len(text), 1)
    return ascii_ratio >= 0.8


def _save(path: Path, results: List[dict], meta: Optional[dict] = None) -> None:
    payload = {**(meta or {}), "results": results}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _output_path(source: str, model: str, out_dir: Path) -> Path:
    safe = model.replace("/", "_").replace("-", "_")
    return out_dir / f"{source}_paraphrase_rephrase_{safe}.json"


def _serialize(entities) -> List[dict]:
    out = []
    for e in entities:
        if hasattr(e, "model_dump"):
            out.append(e.model_dump())
        elif hasattr(e, "dict"):
            out.append(e.dict())
        else:
            out.append(e)
    return out


# ---------------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------------

def load_rephrase_records(input_path: Path) -> Dict[str, List[dict]]:
    """
    Returns {source: [record, ...]} for transformation_type == ['rephrase'] only.
    Records with ERROR in generated_raw are excluded.
    """
    data    = json.loads(input_path.read_text())
    results = data["results"]

    by_source: Dict[str, List[dict]] = {s: [] for s in SOURCES}
    skipped_type = skipped_error = 0

    for r in results:
        if r.get("transformation_type") != ["rephrase"]:
            skipped_type += 1
            continue
        if not _is_valid(r.get("generated_raw", "")):
            skipped_error += 1
            continue
        src = r.get("source", "")
        if src in by_source:
            by_source[src].append(r)

    total = sum(len(v) for v in by_source.values())
    print(f"Loaded {input_path.name}: {len(results)} total records")
    print(f"  skipped (not pure rephrase) : {skipped_type}")
    print(f"  skipped (ERROR/invalid)     : {skipped_error}")
    print(f"  valid rephrase records      : {total}")
    for src, recs in by_source.items():
        print(f"    {src}: {len(recs)}")
    return by_source


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def _merge_resume(records: List[dict], out_path: Path) -> tuple[List[dict], dict]:
    if not out_path.exists():
        return records, {}
    with out_path.open() as f:
        existing = json.load(f)
    existing_map = {
        r["sample_idx"]: r
        for r in existing.get("results", [])
        if isinstance(r, dict) and "sample_idx" in r
    }
    keep_keys = DISCERN_KEYS | MINI_DISCERN_KEYS | {"discern_error", "mini_discern_error"}
    merged = []
    for rec in records:
        prev = existing_map.get(rec["sample_idx"], {})
        merged.append({**rec, **{k: v for k, v in prev.items() if k in keep_keys}})
    meta = {k: v for k, v in existing.items() if k != "results"}
    return merged, meta


# ---------------------------------------------------------------------------
# Per-source evaluation
# ---------------------------------------------------------------------------

def run_source(
    source: str,
    records: List[dict],
    out_path: Path,
    model: str,
    discern_batch: bool,
    discern_batch_size: int,
    mini_batch: bool,
    max_concurrent: int,
    run_discern: bool,
    run_mini_discern: bool,
) -> None:
    print(f"\n{'='*70}")
    print(f"Source : {source}  ({len(records)} records)")
    print(f"Output : {out_path}")
    print(f"Model  : {model}")
    print(f"Modes  : discern={'batch' if discern_batch else 'seq'}  "
          f"mini={'batch' if mini_batch else 'seq'}")

    results, meta = _merge_resume(records, out_path)

    discern_max_tokens_cap = 8192 if "gemma" in model.lower() else 25000
    mini_max_tokens        = 8192 if "gemma" in model.lower() else 10000

    # ── DISCERN ───────────────────────────────────────────────────────────
    if run_discern:
        todo = [r for r in results if not _has_all_keys(r, DISCERN_KEYS)]
        print(f"\n[DISCERN] {len(todo)} pending")

        if discern_batch and todo:
            pairs = [(r["ground_truth_raw"], r["generated_raw"]) for r in todo]
            batch_out = run_evaluation_batch(
                pairs=pairs,
                model=model,
                token_path=str(TOKEN_PATH),
                prompt_yaml_path=str(PROMPT_YAML_PATH),
                entities_yaml_path=str(ENTITIES_YAML_PATH),
                attribute_prompt_path=str(ATTRIBUTE_PROMPT_PATH),
                significance_yaml_path=str(SIGNIFICANCE_YAML_PATH),
                hf_token_path=str(HF_TOKEN_PATH),
                max_tokens=DISCERN_MAX_TOKENS_INIT,
                batch_size=discern_batch_size,
            )
            for r, result in zip(todo, batch_out):
                if result is not None:
                    r["discern_score"]      = result[1]
                    r["discern_evaluation"] = result[0]
                    r.pop("discern_error", None)
                else:
                    r["discern_score"]      = None
                    r["discern_evaluation"] = None
                    r["discern_error"]      = "Batch evaluation failed"
            _save(out_path, results, meta)

        else:
            for i, r in enumerate(todo, 1):
                print(f"  [{i}/{len(todo)}] sample_idx={r['sample_idx']} ... ",
                      end="", flush=True)
                current_max = DISCERN_MAX_TOKENS_INIT
                last_exc: Optional[Exception] = None
                success = False
                while current_max <= discern_max_tokens_cap:
                    try:
                        d_eval, d_score = run_evaluation(
                            report_text=r["ground_truth_raw"],
                            candidate_text=r["generated_raw"],
                            model=model,
                            token_path=str(TOKEN_PATH),
                            prompt_yaml_path=str(PROMPT_YAML_PATH),
                            entities_yaml_path=str(ENTITIES_YAML_PATH),
                            attribute_prompt_path=str(ATTRIBUTE_PROMPT_PATH),
                            significance_yaml_path=str(SIGNIFICANCE_YAML_PATH),
                            max_tokens=current_max,
                        )
                        r["discern_score"]      = d_score
                        r["discern_evaluation"] = d_eval
                        r.pop("discern_error", None)
                        print(f"score={d_score}")
                        success = True
                        break
                    except (TokenLimitError, InputTooLongError) as exc:
                        last_exc = exc
                        new_max = min(int(current_max * 1.5), discern_max_tokens_cap)
                        print(f"\n    token limit → escalating {current_max}→{new_max} ...",
                              end=" ", flush=True)
                        current_max = new_max
                    except Exception as exc:
                        if "TokenLimitError" in str(exc) or "InputTooLongError" in str(exc):
                            last_exc = exc
                            new_max = min(int(current_max * 1.5), discern_max_tokens_cap)
                            current_max = new_max
                        else:
                            last_exc = exc
                            break
                if not success:
                    r["discern_score"]      = None
                    r["discern_evaluation"] = None
                    r["discern_error"]      = str(last_exc)
                    print(f"FAILED: {last_exc}")
                _save(out_path, results, meta)

    # ── mini-DISCERN ──────────────────────────────────────────────────────
    if run_mini_discern:
        todo = [r for r in results if not _has_all_keys(r, MINI_DISCERN_KEYS)]
        print(f"\n[MINI-DISCERN] {len(todo)} pending")

        if mini_batch and todo:
            refs  = [r["ground_truth_raw"] for r in todo]
            cands = [r["generated_raw"]     for r in todo]
            batch_out = evaluate_reports_batch(
                reference_reports=refs,
                candidate_reports=cands,
                entity_list_path=str(DIAG_ENTITIES_YAML),
                prompt_path=str(MERGED_PROMPT_PATH),
                model=model,
                token_path=str(TOKEN_PATH),
                hf_token_path=str(HF_TOKEN_PATH),
                max_tokens=mini_max_tokens,
                temperature=0.1,
                max_retries=3,
                max_concurrent=max_concurrent,
            )
            for r, result in zip(todo, batch_out):
                if result is not None:
                    r["mini_discern_score"]      = int(sum(
                        e.clinical_significance_score for e in result))
                    r["mini_discern_evaluation"] = _serialize(result)
                    r.pop("mini_discern_error", None)
                else:
                    r["mini_discern_score"]      = None
                    r["mini_discern_evaluation"] = None
                    r["mini_discern_error"]      = "Batch validation failed"
            _save(out_path, results, meta)

        else:
            for i, r in enumerate(todo, 1):
                print(f"  [{i}/{len(todo)}] sample_idx={r['sample_idx']} ... ",
                      end="", flush=True)
                try:
                    mini = evaluate_reports(
                        reference_report=r["ground_truth_raw"],
                        candidate_report=r["generated_raw"],
                        entity_list_path=str(DIAG_ENTITIES_YAML),
                        prompt_path=str(MERGED_PROMPT_PATH),
                        model=model,
                        token_path=str(TOKEN_PATH),
                        hf_token_path=str(HF_TOKEN_PATH),
                        max_tokens=mini_max_tokens,
                        temperature=0.1,
                        max_retries=4,
                    )
                    r["mini_discern_score"]      = int(sum(
                        e.clinical_significance_score for e in mini))
                    r["mini_discern_evaluation"] = _serialize(mini)
                    r.pop("mini_discern_error", None)
                    print(f"score={r['mini_discern_score']}")
                except Exception as exc:
                    r["mini_discern_score"]      = None
                    r["mini_discern_evaluation"] = None
                    r["mini_discern_error"]      = str(exc)
                    print(f"FAILED: {exc}")
                _save(out_path, results, meta)

    _save(out_path, results, meta)

    # Summary
    print(f"\nSummary ({source}):")
    for key in ("discern_score", "mini_discern_score"):
        vals = [r[key] for r in results
                if r.get(key) is not None and isinstance(r.get(key), (int, float))]
        if vals:
            arr = np.array(vals, dtype=float)
            print(f"  {key:>22s}: mean={arr.mean():.3f}  std={arr.std():.3f}  "
                  f"min={arr.min():.0f}  max={arr.max():.0f}  n={len(arr)}")
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DISCERN + mini-DISCERN on paraphrased (rephrase) reports."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Path to paraphrase JSON (default: {DEFAULT_INPUT.name})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (default: config.yaml at repo root)")
    parser.add_argument("--sources", nargs="+", default=SOURCES,
                        choices=SOURCES,
                        help="Which sources to run (default: all three).")
    parser.add_argument("--skip-discern",      action="store_true")
    parser.add_argument("--skip-mini-discern", action="store_true")
    parser.add_argument("--discern-batch",     action="store_true",
                        help="Stage-by-stage batch DISCERN via vLLM.")
    parser.add_argument("--discern-batch-size", type=int, default=200)
    parser.add_argument("--mini-batch",        action="store_true",
                        help="Batch mini-DISCERN via query_llm_batch.")
    parser.add_argument("--max-concurrent",    type=int, default=1)
    args = parser.parse_args()

    # Load credentials from config.yaml
    import yaml as _yaml
    _cfg_path = Path(args.config) if args.config else (REPO_ROOT / "config.yaml")
    _cfg = _yaml.safe_load(_cfg_path.read_text()) if _cfg_path.exists() else {}
    global TOKEN_PATH, HF_TOKEN_PATH
    TOKEN_PATH   = (_cfg.get("credentials") or {}).get("databricks_token", "") or ""
    HF_TOKEN_PATH = (_cfg.get("credentials") or {}).get("hf_token", "") or ""

    by_source = load_rephrase_records(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for source in args.sources:
        records = by_source.get(source, [])
        if not records:
            print(f"\n[{source}] No valid rephrase records — skipping.")
            continue
        run_source(
            source=source,
            records=records,
            out_path=_output_path(source, args.model, args.output_dir),
            model=args.model,
            discern_batch=args.discern_batch,
            discern_batch_size=args.discern_batch_size,
            mini_batch=args.mini_batch,
            max_concurrent=args.max_concurrent,
            run_discern=not args.skip_discern,
            run_mini_discern=not args.skip_mini_discern,
        )


if __name__ == "__main__":
    main()
