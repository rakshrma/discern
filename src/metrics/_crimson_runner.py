"""
Standalone subprocess runner for CRIMSON scoring.
Called by crimson.py via `conda run -n crimson python _crimson_runner.py`.

Accepts either:
  - legacy list-of-pairs: [{"candidate": ..., "reference": ...}, ...]
  - full _w_metrics.json: {"results": [{"generated_raw"/"generated_impression"/...,
                                        "ground_truth_raw"/...}, ...], ...}

When given the _w_metrics schema, writes crimson scores back into each
results[i]["crimson"] and re-dumps the full JSON to --output.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "CRIMSON"))


def _pick(entry, keys):
    for k in keys:
        v = entry.get(k)
        if v:
            return v
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="rajpurkarlab/medgemma-4b-it-crimson")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        rows = data["results"]
        candidates = [_pick(r, ("generated_raw", "generated_impression", "generated_findings", "candidate")) for r in rows]
        references = [_pick(r, ("ground_truth_raw", "ground_truth_impression", "ground_truth_findings", "reference")) for r in rows]
        wrapped = True
    else:
        rows = data
        candidates = [p["candidate"] for p in rows]
        references = [p["reference"] for p in rows]
        wrapped = False

    from CRIMSON.generate_score import CRIMSONScore
    scorer = CRIMSONScore(model_name=args.model)
    results_list = scorer.evaluate_batch(
        reference_findings_list=references,
        predicted_findings_list=candidates,
        batch_size=args.batch_size,
    )

    def _score_of(r):
        if r is None:
            return None
        if isinstance(r, dict):
            v = r.get("crimson_score")
            return float(v) if v is not None else None
        try:
            return float(r)
        except (TypeError, ValueError):
            return None

    scores_out = [_score_of(r) for r in results_list]

    if wrapped:
        for r, s in zip(rows, scores_out):
            r["crimson"] = s
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
    else:
        with open(args.output, "w") as f:
            json.dump(scores_out, f)


if __name__ == "__main__":
    main()
