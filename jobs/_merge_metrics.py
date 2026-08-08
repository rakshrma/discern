"""Merge per-metric scratch JSONs (CRIMSON, GREEN, NLP, DISCERN) for one
input file into a single <basename>_w_metrics.json with full provenance.

Layout it expects:
    <model_dir>/<base>.json                       ← original input
    <model_dir>/.metrics/<base>.crimson.json      ← per-metric scratch
    <model_dir>/.metrics/<base>.green.json
    <model_dir>/.metrics/<base>.nlp.json
    <model_dir>/.metrics/<base>.discern.json

Output:
    <model_dir>/<base>_w_metrics.json

Each per-metric scratch carries a `_metadata` block written by run_metrics.py
(model name, run timestamps, env, SLURM IDs, vLLM settings, max-token caps,
config path, hostname, ...). The merge:

  1. seeds the merged records from the input file (keyed by sample_idx),
  2. overlays metric columns from each scratch file,
  3. aggregates per-metric `_metadata` under `_metadata.metrics.<group>`, and
  4. emits an outer `_metadata` block describing the merge itself.

Usage:
    python _merge_metrics.py <INPUT_JSON_PATH>
"""
from __future__ import annotations

import json
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


METRIC_COLS = {
    "crimson": ["crimson"],
    "green":   ["green", "green_summary"],
    "nlp":     ["bleu", "rouge", "meteor", "bertscore", "radgraph"],
    "discern": [
        "discern_score", "discern_evaluation", "discern_error",
        "mini_discern_score", "mini_discern_evaluation", "mini_discern_error",
    ],
    "radcliq": [
        # `radcliq` (1/RadCliQ-v1, higher=better) is the leaderboard-facing
        # composite. `radcliq_v1_raw` is the paper-canonical lower-is-better
        # value. The 4 canonical inputs stay for diagnostics.
        "radcliq", "radcliq_v1_raw",
        "bleu_score", "bertscore_radcliq", "semb_score", "radgraph_combined",
    ],
}

SUMMARY_KEYS = ["bleu", "rouge", "meteor", "bertscore", "radgraph",
                "green", "crimson", "discern_score", "mini_discern_score",
                "radcliq"]


def _compute_score_summary(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    import numpy as np
    out: Dict[str, Dict[str, float]] = {}
    for key in SUMMARY_KEYS:
        vals = [r[key] for r in records
                if r.get(key) is not None and isinstance(r[key], (int, float))]
        if vals:
            arr = np.array(vals, dtype=float)
            out[key] = {
                "mean": round(float(arr.mean()), 4),
                "std":  round(float(arr.std()),  4),
                "n":    int(len(arr)),
            }
    return out


def _load_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"results": data, "_metadata": {}}
    return data


def _records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rs = payload.get("results", [])
    return rs if isinstance(rs, list) else []


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: _merge_metrics.py <INPUT_JSON_PATH>")

    input_path = Path(sys.argv[1]).resolve()
    if not input_path.is_file():
        sys.exit(f"input not found: {input_path}")

    model_dir   = input_path.parent
    base        = input_path.stem
    metrics_dir = model_dir / ".metrics"
    final_path  = model_dir / f"{base}_w_metrics.json"

    # ── 1. seed merged records from the input file ────────────────────────────
    base_payload = _load_payload(input_path)
    base_records = _records(base_payload)
    by_idx: Dict[Any, Dict[str, Any]] = {
        r["sample_idx"]: dict(r) for r in base_records if "sample_idx" in r
    }
    if not by_idx:
        sys.exit(f"no sample_idx records in {input_path}")

    # ── 2 + 3. overlay metric columns and capture per-metric metadata ────────
    per_metric_meta: Dict[str, Any] = {}
    for metric, cols in METRIC_COLS.items():
        scratch = metrics_dir / f"{base}.{metric}.json"
        if not scratch.exists():
            per_metric_meta[metric] = {"status": "missing", "scratch_path": str(scratch)}
            continue
        payload = _load_payload(scratch)
        sub_meta = dict(payload.get("_metadata") or {})

        n_added = 0
        for r in _records(payload):
            idx = r.get("sample_idx")
            if idx not in by_idx:
                continue
            for c in cols:
                if c in r:
                    by_idx[idx][c] = r[c]
            n_added += 1

        sub_meta["scratch_path"]    = str(scratch)
        sub_meta["records_merged"]  = f"{n_added}/{len(_records(payload))}"
        sub_meta["columns_overlaid"] = cols
        sub_meta["status"]          = "ok" if n_added > 0 else "empty"
        per_metric_meta[metric] = sub_meta

    # ── 4. emit outer merge metadata ──────────────────────────────────────────
    merged = list(by_idx.values())
    score_summary = _compute_score_summary(merged)

    merge_meta = {
        "merged_at"           : datetime.now().astimezone().isoformat(timespec="seconds"),
        "merged_by_host"      : socket.gethostname(),
        "merged_by_python"    : sys.executable,
        "input_path"          : str(input_path),
        "input_dir"           : str(model_dir),
        "model_dir_basename"  : model_dir.name,
        "n_records"           : len(merged),
        "metrics_present"     : sorted(
            m for m, v in per_metric_meta.items() if v.get("status") == "ok"
        ),
        "metrics_missing"     : sorted(
            m for m, v in per_metric_meta.items() if v.get("status") != "ok"
        ),
        "score_summary"       : score_summary,
        "metrics"             : per_metric_meta,
    }

    with final_path.open("w") as f:
        json.dump({"_metadata": merge_meta, "results": merged}, f, indent=2, default=str)

    print(f"merge -> {final_path}")
    print(f"  records         : {merge_meta['n_records']}")
    print(f"  metrics present : {merge_meta['metrics_present']}")
    if merge_meta["metrics_missing"]:
        print(f"  metrics missing : {merge_meta['metrics_missing']}")
    if score_summary:
        print(f"  score summary   :")
        for key, s in score_summary.items():
            print(f"    {key:>20s}: mean={s['mean']:.4f}  std={s['std']:.4f}  n={s['n']}")


if __name__ == "__main__":
    main()
