#!/usr/bin/env python3
"""Collate per-sample metrics into a summary table across model results.

Reads <results_root>/<pattern>/<dataset>_<mode>_results_w_metrics.json
(merged outputs from launch_all_metrics.sh) and produces:

  1. A long-form CSV: one row per (model, dataset, mode, metric).
  2. A wide-form CSV + console table: one row per (model, dataset, mode),
     columns = mean values per metric.

Usage:
    python scripts/collate_metrics.py                       # default qwen35-*
    python scripts/collate_metrics.py --pattern 'qwen3*'    # broader scope
    python scripts/collate_metrics.py --out qwen35_summary  # custom prefix
"""

import argparse
import csv
import json
import os
import re
import sys
from glob import glob
from statistics import mean, stdev


# Metric keys produced by run_metrics.py / merge pipeline.
NLP_METRICS = ["bleu", "rouge", "meteor", "bertscore", "radgraph"]
LLM_METRICS = ["green", "crimson", "discern_score", "mini_discern_score"]
ALL_METRICS = NLP_METRICS + LLM_METRICS

DEFAULT_RESULTS_ROOT = (
    "/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/"
    "workspace/vlm_cxr_benchmark/results"
)

# Order: smallest → largest. Used to sort model rows naturally
# (e.g. 0.8B before 2B before 122B).
def model_size_key(model_name: str) -> float:
    m = re.search(r"-(\d+(?:_\d+)?)b", model_name)
    if not m:
        return float("inf")
    return float(m.group(1).replace("_", "."))


def parse_basename(base: str) -> tuple[str, str]:
    """Return (dataset, mode) from e.g. 'chexpert_plus_valid_nothinking_results'."""
    m = re.match(
        r"^(?P<dataset>.+?)_(?P<mode>thinking|nothinking)_results$", base
    )
    if m:
        return m.group("dataset"), m.group("mode")
    # No mode suffix — older format
    return base.removesuffix("_results"), "unknown"


def _flatten(v):
    """Some metrics may be nested dicts (e.g. bertscore → {f1, precision, recall}).
    Return a numeric scalar (preferring 'f1' if present) or None."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for k in ("f1", "f1_score", "score"):
            if k in v and isinstance(v[k], (int, float)):
                return float(v[k])
    return None


def aggregate(entries, key) -> tuple[float | None, float | None, int]:
    vals = []
    for e in entries:
        v = _flatten(e.get(key))
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None, 0
    m = mean(vals)
    s = stdev(vals) if len(vals) > 1 else 0.0
    return m, s, len(vals)


def main():
    ap = argparse.ArgumentParser(
        description="Collate metric scores across model result JSONs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--pattern", default="qwen35-*",
                    help="Glob for model directories under results-root.")
    ap.add_argument("--out", default="qwen35_metrics",
                    help="Output prefix; writes <prefix>_long.csv and <prefix>_wide.csv.")
    args = ap.parse_args()

    glob_pattern = os.path.join(
        args.results_root, args.pattern, "*_w_metrics.json"
    )
    paths = sorted(glob(glob_pattern))
    if not paths:
        print(f"No *_w_metrics.json files found under "
              f"{args.results_root}/{args.pattern}/", file=sys.stderr)
        sys.exit(1)

    long_rows = []
    wide_rows = {}  # (model, dataset, mode) -> {metric: mean, ...}

    for path in paths:
        model = os.path.basename(os.path.dirname(path))
        base = os.path.basename(path).removesuffix("_w_metrics.json")
        dataset, mode = parse_basename(base)

        with open(path) as f:
            obj = json.load(f)
        entries = obj.get("results", [])
        n_total = obj.get("metadata", {}).get("completed", len(entries))

        wide_key = (model, dataset, mode)
        wide_rows.setdefault(wide_key, {"n_total": n_total})

        for metric in ALL_METRICS:
            m, s, n = aggregate(entries, metric)
            long_rows.append({
                "model": model,
                "dataset": dataset,
                "mode": mode,
                "metric": metric,
                "mean": m,
                "std": s,
                "n_valid": n,
                "n_total": n_total,
            })
            if m is not None:
                wide_rows[wide_key][metric] = m

    # ── Long-form CSV ────────────────────────────────────────────────────
    long_path = f"{args.out}_long.csv"
    with open(long_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "dataset", "mode", "metric",
            "mean", "std", "n_valid", "n_total",
        ])
        w.writeheader()
        w.writerows(long_rows)
    print(f"Wrote {long_path}  ({len(long_rows)} rows)")

    # ── Wide-form CSV ────────────────────────────────────────────────────
    wide_path = f"{args.out}_wide.csv"
    sorted_keys = sorted(wide_rows.keys(),
                         key=lambda k: (k[1], k[2], model_size_key(k[0])))
    fieldnames = ["model", "dataset", "mode", "n_total"] + ALL_METRICS
    with open(wide_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for key in sorted_keys:
            model, dataset, mode = key
            row = {"model": model, "dataset": dataset, "mode": mode}
            row.update(wide_rows[key])
            for m in ALL_METRICS:
                if m in row and isinstance(row[m], float):
                    row[m] = round(row[m], 4)
            w.writerow(row)
    print(f"Wrote {wide_path}  ({len(sorted_keys)} (model, dataset, mode) combos)")

    # ── Console summary ──────────────────────────────────────────────────
    print(f"\nSummary ({args.pattern}):\n")
    headers = ["model", "dataset", "mode", "n"] + ALL_METRICS
    widths = [max(len(h), 14) for h in headers]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for key in sorted_keys:
        model, dataset, mode = key
        row_data = wide_rows[key]
        cells = [model, dataset, mode, str(row_data.get("n_total", "?"))]
        for m in ALL_METRICS:
            v = row_data.get(m)
            cells.append(f"{v:.4f}" if isinstance(v, float) else "-")
        print(fmt.format(*cells))


if __name__ == "__main__":
    main()
