"""
Standalone subprocess runner for GREEN scoring.
Called by green.py via `conda run -p <green_score env> python _green_runner.py`.

Input JSON: [{"candidate": ..., "reference": ...}, ...]
Output JSON: [score_or_null, ...]
"""

import argparse
import json

from green_score import GREEN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="StanfordAIMI/GREEN-radllama2-7b")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    with open(args.input) as f:
        pairs = json.load(f)

    candidates = [p["candidate"] for p in pairs]
    references = [p["reference"] for p in pairs]

    scorer = GREEN(args.model, output_dir=args.output_dir)
    mean, std, score_list, summary, result_df = scorer(references, candidates)
    scores = [float(s) if s is not None else None for s in score_list]

    with open(args.output, "w") as f:
        json.dump(scores, f)


if __name__ == "__main__":
    main()
