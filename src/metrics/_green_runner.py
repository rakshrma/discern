"""Subprocess entry point for GREEN scoring (run inside the green conda env)."""
import json
import sys

def main():
    _, input_path, output_path = sys.argv
    with open(input_path) as f:
        payload = json.load(f)

    from green_score import GREEN
    scorer = GREEN(payload["model_name"], output_dir=payload["output_dir"])
    _, _, score_list, _, _ = scorer(payload["references"], payload["candidates"])

    with open(output_path, "w") as f:
        json.dump([float(s) for s in score_list], f)

if __name__ == "__main__":
    main()
