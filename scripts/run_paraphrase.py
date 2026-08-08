#!/usr/bin/env python3
"""
paraphrase_reports.py
=====================
Build a single paraphrase robustness dataset from rexval and radevalx for judge
robustness evaluation.

Robustness test logic
---------------------
For each (ground_truth, candidate) pair in the original datasets, the ground_truth
report is paraphrased (style only — clinical content preserved). The output pairs
the ORIGINAL ground truth (as judge reference) with the PARAPHRASED ground truth
(as judge candidate). A robust judge should score this pair highly since the
clinical content is identical.

Each unique GT is paraphrased once per variant and reused across all rows
that share that GT (e.g., rexval has 4 candidates per study_id).

Variants produced (5 per dataset):
  4 single-transform : rearrange | bullets | rephrase | voice
  1 random-combo     : random 2–4 transforms per unique GT

Total entries in output:
  rexval   : 50 rows × 5 variants = 250
  radevalx : 100 rows × 5 variants = 500
  Grand total: 750

Output JSON (single file, inference_all_metrics.py compatible)
--------------------------------------------------------------
  sample_idx          int   — globally unique sequential index (0–2399)
  ground_truth_raw    str   — original GT  (judge reference, unchanged)
  generated_raw       str   — paraphrased GT (judge candidate)
  transformation_type list  — transforms applied (e.g. ["rearrange"])
  variant             str   — variant name (e.g. "rearrange", "random_1")
  source              str   — "rexval" or "radevalx"
  source_report_id    str   — original study_id / report_id in source CSV
  source_row_idx      int   — original row index (0-based) in source CSV
  source_gt_key       int   — index of this GT in the dataset's unique-GT list

Usage
-----
    python paraphrase_reports.py \\
        --rexval    data/rexval/rexval_reports_long.csv \\
        --radevalx  data/radevalx/radeval_total.csv \\
        --output    data/robustness/paraphrase_robustness.json \\
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from pydantic import BaseModel, StrictStr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from llm_backend import query_llm_batch


class ParaphraseResponse(BaseModel):
    report: StrictStr

# ---------------------------------------------------------------------------
# Transformation catalogue
# ---------------------------------------------------------------------------

TRANSFORMS: Dict[str, str] = {
    "rearrange": (
        "Rearrange the order of sentences within each section of the report "
        "(e.g. within FINDINGS, within IMPRESSION) to vary the presentation order. "
        "Do NOT move sentences across section boundaries "
        "(e.g. never move a FINDINGS sentence into IMPRESSION). "
        "Do NOT alter the wording of any sentence."
    ),
    "bullets": (
        "If the FINDINGS section is written in prose, convert it to a "
        "bullet-point list (one finding per bullet). "
        "If FINDINGS already uses bullet points, convert them back to flowing "
        "prose sentences. "
        "The IMPRESSION section must remain as prose regardless. "
        "Do NOT change any wording beyond what is necessary for the format change."
    ),
    "rephrase": (
        "Reword each sentence using different vocabulary and phrasing while preserving "
        "the exact clinical meaning. "
        "For example, 'No acute cardiopulmonary process' may become "
        "'There is no acute cardiopulmonary abnormality'. "
        "Do NOT add, omit, or reorder any finding. "
        "Do NOT add any new sentence, clause, or phrase that was not present in the original."
    ),
    "voice": (
        "Convert active-voice sentences to passive voice and passive-voice sentences to "
        "active voice where grammatically natural for radiology prose. "
        "Do NOT add, omit, or reorder any finding. "
        "Do NOT add any new sentence, clause, or phrase that was not present in the original."
    ),
}

TRANSFORM_KEYS: List[str] = list(TRANSFORMS.keys())

SYSTEM_PROMPT = """\
You are a clinical language specialist rewriting radiology reports for style-robustness evaluation.

Your goal is to change how findings are expressed — not what is expressed.
You MAY use different words, sentence structures, or formatting.
You MUST NOT change the clinical content.

ABSOLUTE RULES — violating any of these makes the output unusable:
  • Do NOT introduce any clinical finding, diagnosis, or observation not present in the original.
  • Do NOT omit any clinical finding, diagnosis, or observation from the original.
  • Do NOT alter the meaning of any of the following — reproduce them faithfully:
      – measurements or sizes        (e.g. "2.3 cm", "3 mm", "moderate")
      – severity descriptors         (e.g. "mild", "moderate", "severe", "large", "small")
      – anatomical locations         (e.g. "right lower lobe", "left hilum", "lingula")
      – temporal comparisons         (e.g. "new", "unchanged", "stable", "increased",
                                           "decreased", "improved", "worsened", "similar to")
      – diagnoses / conclusions      (e.g. "pneumonia", "pleural effusion", "cardiomegaly")
  • The FINDINGS and IMPRESSION sections must convey IDENTICAL clinical content
    to the original after your edits — no more, no less.

Apply ONLY the transformations listed in the user message.
Return your answer as a JSON object with a single key:
  {"report": "<rewritten report text here>"}
No other keys, no prose outside the JSON.\
"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_messages(report: str, transforms: List[str]) -> List[Dict[str, str]]:
    instructions = "\n".join(
        f"  {i+1}. [{name.upper()}] {TRANSFORMS[name]}"
        for i, name in enumerate(transforms)
    )
    user = (
        "Apply the following style transformation(s) to the radiology report below.\n\n"
        f"Transformations to apply:\n{instructions}\n\n"
        "Critical reminder: you may change wording and style freely, but must NOT "
        "introduce or omit any clinical finding. "
        "Sizes, severity descriptors, anatomical locations, temporal comparisons, and "
        "diagnoses must retain their exact meaning.\n\n"
        f"Original report:\n{report.strip()}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(raw: str, fallback_idx: int) -> str:
    """Extract the report string from the LLM JSON response using Pydantic."""
    try:
        # Find the first '{...}' block in case there is surrounding text
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        return ParaphraseResponse.model_validate_json(raw[start:end]).report
    except Exception as e:
        print(f"    [WARN] Could not parse JSON for index {fallback_idx}: {e!r}. "
              f"Storing raw response.")
        return raw.strip()


# ---------------------------------------------------------------------------
# LLM batch paraphrase
# ---------------------------------------------------------------------------

def _paraphrase_unique_gts(
    unique_gts: List[str],
    transforms_per_gt: List[List[str]],
    model: str,
    token_path: str,
    max_tokens: int,
    temperature: float,
    max_concurrent: int,
    batch_size: int,
) -> List[str]:
    results: List[str] = [""] * len(unique_gts)
    for batch_start in range(0, len(unique_gts), batch_size):
        batch_slice = slice(batch_start, batch_start + batch_size)
        batch_reports = unique_gts[batch_slice]
        batch_transforms = transforms_per_gt[batch_slice]
        messages_batch = [_build_messages(r, t) for r, t in zip(batch_reports, batch_transforms)]
        print(f"    [LLM] Batch {batch_start // batch_size + 1}: {len(batch_reports)} reports ...")
        responses = query_llm_batch(
            messages_batch=messages_batch,
            model=model,
            token_path=token_path,
            max_tokens=max_tokens,
            temperature=temperature,
            max_concurrent=max_concurrent,
        )
        for i, resp in enumerate(responses):
            results[batch_start + i] = _parse_response(resp, batch_start + i)
    return results


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def _load_rexval(csv_path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns unique_gts (50) and rows (50) — one row per unique study_id.
    Repeated rows (same GT with different candidate_reporter) are skipped.
    """
    with csv_path.open(encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    seen: Dict[str, int] = {}
    unique_gts: List[str] = []
    rows: List[Dict[str, Any]] = []

    for row_idx, r in enumerate(raw_rows):
        sid = r["study_id"]
        gt  = r["gt_report"]
        if sid not in seen:
            seen[sid] = len(unique_gts)
            unique_gts.append(gt)
            rows.append({
                "source_row_idx":   row_idx,
                "source_report_id": sid,
                "source_gt_key":    seen[sid],
                "ground_truth_raw": gt,
            })

    return unique_gts, rows


def _load_radevalx(csv_path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns unique_gts (100) and rows (100) — one row per unique report_id.
    Repeated rows (same GT with different error_type) are skipped.
    """
    with csv_path.open(encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    seen: Dict[str, int] = {}
    unique_gts: List[str] = []
    rows: List[Dict[str, Any]] = []

    for row_idx, r in enumerate(raw_rows):
        rid = r["study_id"]
        gt  = r["ground_truth"]
        if rid not in seen:
            seen[rid] = len(unique_gts)
            unique_gts.append(gt)
            rows.append({
                "source_row_idx":   row_idx,
                "source_report_id": rid,
                "source_gt_key":    seen[rid],
                "ground_truth_raw": gt,
            })

    return unique_gts, rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_chexpertplusvalid(json_path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Load unique ground-truth reports from a CheXpert-Plus validation inference JSON.
    Expected format: {"results": [{..., "ground_truth_raw": "...", "sample_idx": N}, ...]}
    All 200 entries are treated as unique (no deduplication needed).
    """
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    raw_rows = data["results"]

    unique_gts: List[str] = []
    rows: List[Dict[str, Any]] = []

    for gt_key, r in enumerate(raw_rows):
        gt = r["ground_truth_raw"]
        unique_gts.append(gt)
        rows.append({
            "source_row_idx":   r["sample_idx"],
            "source_report_id": str(r["sample_idx"]),
            "source_gt_key":    gt_key,
            "ground_truth_raw": gt,
        })

    return unique_gts, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a single paraphrase robustness JSON for judge evaluation."
    )
    parser.add_argument(
        "--rexval",
        default=None,
        help="Path to rexval CSV (study_id, gt_report, candidate_reporter, candidate_report).",
    )
    parser.add_argument(
        "--radevalx",
        default=None,
        help="Path to radevalx CSV (report_id, ground_truth, candidate_report, ...).",
    )
    parser.add_argument(
        "--chexpertplusvalid",
        default=None,
        help="Path to CheXpert-Plus validation inference JSON with 'results[].ground_truth_raw'.",
    )
    parser.add_argument(
        "--output",
        default="data/robustness/paraphrase_robustness.json",
        help="Single output JSON file (supports resume).",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        metavar="SOURCE:VARIANT",
        help=(
            "Only process these specific source:variant combinations. "
            "E.g. --variants radevalx:random_2 rexval:bullets. "
            "If omitted, all 6 variants for both datasets are run."
        ),
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-run any entry whose generated_raw starts with 'ERROR'. "
             "The corrected result replaces the error entry in the output file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="databricks-gpt-oss-120b")
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (default: config.yaml at repo root)")
    parser.add_argument("--token-path", default=None,
                        help="Databricks token string or file path (overrides config.yaml)")
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    # Parse --variants filter into a set of (source, variant) tuples, or None = run all
    variant_filter: set | None = None
    if args.variants is not None:
        variant_filter = set()
        for v in args.variants:
            if ":" not in v:
                parser.error(f"--variants entries must be SOURCE:VARIANT, got '{v}'")
            src, var = v.split(":", 1)
            variant_filter.add((src, var))
        print(f"[filter] Running only: {sorted(variant_filter)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve credentials: CLI arg > config.yaml
    import yaml
    _cfg_path = Path(args.config) if args.config else (Path(__file__).parent.parent / "config.yaml")
    _cfg = yaml.safe_load(_cfg_path.read_text()) if _cfg_path.exists() else {}
    db_token = args.token_path or (_cfg.get("credentials") or {}).get("databricks_token", "") or ""

    llm_kwargs = dict(
        model=args.model,
        token_path=db_token,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent,
        batch_size=args.batch_size,
    )

    # Resume: load already-computed entries keyed by (source, variant, source_row_idx)
    done_keys: set = set()
    existing_results: List[Dict[str, Any]] = []
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            existing_data = json.load(f)
        if not isinstance(existing_data, dict):
            raise ValueError(f"Output file {output_path} has unexpected format (not a JSON object).")
        existing_results = existing_data.get("results", [])
        n_errors = sum(1 for e in existing_results
                       if str(e.get("generated_raw", "")).startswith("ERROR"))
        for e in existing_results:
            is_error = str(e.get("generated_raw", "")).startswith("ERROR")
            if args.retry_errors and is_error:
                continue  # leave out of done_keys so it gets re-processed
            done_keys.add((e["source"], e["variant"], e["source_row_idx"]))
        print(f"[resume] {len(existing_results)} entries in {output_path} "
              f"({n_errors} errors{', will retry' if args.retry_errors and n_errors else ''})")

    # When retrying errors, start all_results from only the clean entries
    # so error entries are replaced (not duplicated) when re-processed
    if args.retry_errors:
        all_results: List[Dict[str, Any]] = [
            e for e in existing_results
            if not str(e.get("generated_raw", "")).startswith("ERROR")
        ]
    else:
        all_results: List[Dict[str, Any]] = list(existing_results)

    datasets: List[Tuple[str, Path, Callable]] = []
    if args.rexval:
        datasets.append(("rexval",   Path(args.rexval),   _load_rexval))
    if args.radevalx:
        datasets.append(("radevalx", Path(args.radevalx), _load_radevalx))
    if args.chexpertplusvalid:
        datasets.append(("chexpertplusvalid", Path(args.chexpertplusvalid), _load_chexpertplusvalid))
    if not datasets:
        parser.error("At least one of --rexval, --radevalx, or --chexpertplusvalid must be provided.")

    # sample_idx is globally sequential across everything; we track offset
    sample_idx_counter = max((e["sample_idx"] for e in existing_results), default=-1) + 1

    for source, csv_path, loader in datasets:
        print(f"\n{'='*60}")
        print(f"Source: {source}  ←  {csv_path}")
        unique_gts, rows = loader(csv_path)
        n_unique = len(unique_gts)
        print(f"  {len(rows)} rows, {n_unique} unique GTs")

        # Determine the 6 variants (4 single + 2 random)
        # Each random variant uses its own independent RNG seeded deterministically
        # so variant assignments are stable regardless of dataset order or size.
        # random_2 is guaranteed to differ from random_1 for every GT.
        variants: List[Tuple[str, List[List[str]]]] = []
        for key in TRANSFORM_KEYS:
            variants.append((key, [[key] for _ in range(n_unique)]))

        rng1 = random.Random(f"{args.seed}-{source}-random_1")
        random_1_transforms: List[List[str]] = []
        for _ in range(n_unique):
            combo = rng1.sample(TRANSFORM_KEYS, rng1.randint(2, len(TRANSFORM_KEYS)))
            random_1_transforms.append(combo)
        variants.append(("random_1", random_1_transforms))

        for variant_name, transforms_per_gt in variants:
            if variant_filter is not None and (source, variant_name) not in variant_filter:
                print(f"\n  Variant: {variant_name}  [skipped by --variants filter]")
                continue
            print(f"\n  Variant: {variant_name}")

            # Which rows still need processing?
            pending_rows = [
                r for r in rows
                if (source, variant_name, r["source_row_idx"]) not in done_keys
            ]
            if not pending_rows:
                print(f"    [skip] All rows already done for this variant.")
                continue

            # Which unique GT keys still need paraphrasing?
            pending_gt_keys = sorted({r["source_gt_key"] for r in pending_rows})
            pending_reports = [unique_gts[k] for k in pending_gt_keys]
            pending_transforms = [transforms_per_gt[k] for k in pending_gt_keys]

            print(f"    Paraphrasing {len(pending_gt_keys)}/{n_unique} unique GTs "
                  f"(for {len(pending_rows)}/{len(rows)} pending rows) ...")

            paraphrased_pending = _paraphrase_unique_gts(
                unique_gts=pending_reports,
                transforms_per_gt=pending_transforms,
                **llm_kwargs,
            )

            # Build a full lookup: gt_key → paraphrased text
            # For keys not in pending, recover from existing_results
            para_lookup: Dict[int, str] = {}
            for k, para in zip(pending_gt_keys, paraphrased_pending):
                para_lookup[k] = para
            for e in existing_results:
                if e["source"] == source and e["variant"] == variant_name:
                    k = e["source_gt_key"]
                    if k not in para_lookup and not str(e["generated_raw"]).startswith("ERROR"):
                        para_lookup[k] = e["generated_raw"]

            # Build entries only for pending rows
            for row in pending_rows:
                gt_key = row["source_gt_key"]
                para   = para_lookup[gt_key]
                trans  = transforms_per_gt[gt_key]

                entry: Dict[str, Any] = {
                    "sample_idx":          sample_idx_counter,
                    "ground_truth_raw":    row["ground_truth_raw"],
                    "generated_raw":       para,
                    "transformation_type": trans,
                    "variant":             variant_name,
                    "source":              source,
                    "source_report_id":    row["source_report_id"],
                    "source_row_idx":      row["source_row_idx"],
                    "source_gt_key":       gt_key,
                }

                all_results.append(entry)
                done_keys.add((source, variant_name, row["source_row_idx"]))
                sample_idx_counter += 1

            # Save after each variant (resume-safe)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump({"results": all_results}, f, indent=2, ensure_ascii=False)
            print(f"    [saved] {output_path}  ({len(all_results)} total entries)")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Done. {len(all_results)} total entries → {output_path}")
    sources  = sorted({e["source"]  for e in all_results})
    variants = sorted({e["variant"] for e in all_results})
    print(f"  Sources  : {sources}")
    print(f"  Variants : {variants}")
    for src in sources:
        for var in variants:
            n = sum(1 for e in all_results if e["source"] == src and e["variant"] == var)
            print(f"  {src:10s} × {var:12s} : {n} entries")


if __name__ == "__main__":
    main()
