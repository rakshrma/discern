"""Compute RadCliQ-v1 for one model x dataset result file.

Thin glue around CXR-Report-Metric's `calc_metric()` (Yu et al. 2022,
github.com/rajpurkarlab/CXR-Report-Metric). The metric computation is left
canonical; the only vendored change to their code is in
`radgraph_inference/inference.py`, which now reads RADGRAPH_TMPDIR so its
hard-coded temp_dygie_*.json scratch files can be isolated per task (required
for safe parallel SLURM array runs -- it does not affect any score). Env /
checkpoint / config issues are otherwise resolved on the CXR-Report-Metric
side directly.

Pipeline:
  *.json (our format) --> two CSVs (gt + pred)
                          --> CXR-Report-Metric calc_metric()
                          --> CSV with 4 inputs + RadCliQ-v0 + RadCliQ-v1
                          --> *.radcliq.json (our format)

Per-sample dict gets these new keys:
    bleu_score, bertscore_radcliq, semb_score, radgraph_combined  (4 inputs)
    radcliq_v1_raw   = paper-canonical RadCliQ-v1, lower = better
    radcliq          = 1 / radcliq_v1_raw, higher = better
                       (leaderboard-facing, matches the other metric columns)

`bertscore_radcliq` is named to avoid clobbering the existing `bertscore`
field from the NLP pipeline (which uses roberta-large, not
distilroberta-base + rescale).

Run inside the radcliq conda env (CXR-Report-Metric's pinned reqs).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Where the CXR-Report-Metric repo is checked out — the script must run with
# this directory as cwd because the upstream `calc_metric()` uses relative
# paths (config.py / CheXbert/src/encode.py / CXRMetric/*.pkl).
REPO_DIR = Path("/vast/projects/witschey/pmbb-vision/research_projects/"
                "rakshrma/workspace/CXR-Report-Metric")


def build_input_csvs(results: list[dict], workdir: Path) -> tuple[Path, Path]:
    rows_gt, rows_pred = [], []
    for r in results:
        sid = r.get("sample_idx")
        gt = r.get("ground_truth_raw")
        pred = r.get("generated_raw")
        if sid is None or not gt or not pred:
            continue
        rows_gt.append({"study_id": sid, "report": gt})
        rows_pred.append({"study_id": sid, "report": pred})
    if not rows_gt:
        raise SystemExit("[ERROR] no valid (gt, pred) pairs in input results.")
    gt_csv = workdir / "gt.csv"
    pred_csv = workdir / "pred.csv"
    pd.DataFrame(rows_gt).to_csv(gt_csv, index=False)
    pd.DataFrame(rows_pred).to_csv(pred_csv, index=False)
    return gt_csv, pred_csv


def merge_scores(results: list[dict], scored_csv: Path) -> int:
    """Read calc_metric's output CSV and merge the 5 score columns back."""
    scored = pd.read_csv(scored_csv)
    sid2row = {int(row["study_id"]): row for _, row in scored.iterrows()
               if pd.notna(row.get("study_id"))}
    n_merged = 0
    for r in results:
        sid = r.get("sample_idx")
        if sid is None or int(sid) not in sid2row:
            continue
        row = sid2row[int(sid)]
        r["bleu_score"] = float(row["bleu_score"])
        # Renamed: their column is `bertscore` (distilroberta-base, rescaled),
        # but our NLP pipeline's `bertscore` is roberta-large. Avoid clobbering.
        r["bertscore_radcliq"] = float(row["bertscore"])
        r["semb_score"] = float(row["semb_score"])
        r["radgraph_combined"] = float(row["radgraph_combined"])
        v1 = float(row["RadCliQ-v1"])
        r["radcliq_v1_raw"] = v1
        # Invert so higher = better in the leaderboard. Guard against divide
        # by zero for the unlikely case of a perfect-quality regression output.
        r["radcliq"] = 1.0 / v1 if abs(v1) > 1e-6 else float("inf")
        n_merged += 1
    return n_merged


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True,
                    help="Source *_results.json (or _w_metrics.json)")
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument("--keep-cache", action="store_true",
                    help="Keep intermediate CSVs + CXRMetric cache/ for debugging")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(in_path.read_text())
    results = data.get("results") or (data if isinstance(data, list) else [])
    if not results:
        raise SystemExit(f"[ERROR] {in_path} has no results")

    workdir = Path(tempfile.mkdtemp(prefix="radcliq_"))
    print(f"[INFO] workdir = {workdir}")

    gt_csv, pred_csv = build_input_csvs(results, workdir)
    scored_csv = workdir / "scored.csv"

    # The dygie/RadGraph step writes scratch files named temp_dygie_*.json.
    # Upstream hard-codes them relative to cwd; since every array task chdir's
    # into the shared REPO_DIR, concurrent tasks used to clobber and `rm` each
    # other's files (stale-handle / FileNotFoundError crashes). Our vendored
    # inference.py honours RADGRAPH_TMPDIR -- point it at this task's private,
    # local-disk workdir so the scratch files are fully isolated.
    os.environ["RADGRAPH_TMPDIR"] = str(workdir)

    cwd = os.getcwd()
    try:
        os.chdir(REPO_DIR)
        # Make CXRMetric / config importable as the repo expects.
        sys.path.insert(0, str(REPO_DIR))
        from CXRMetric import run_eval as _re
        # radcliq-v1.pkl / composite_metric_model.pkl were pickled when
        # run_eval.py was __main__, so they store the class ref as
        # `__main__.CompositeMetric`. Running via this script makes __main__ be
        # run_radcliq.py (no such attr) -> AttributeError on unpickle. Bind the
        # class into __main__ so pickle resolves it to the same definition.
        import __main__
        __main__.CompositeMetric = _re.CompositeMetric
        # Per-task cache under our tempfile workdir. Upstream defines these as
        # module-level globals (run_eval.py:31-33) keyed off `cache_path =
        # "cache/"`. Rebinding the module attributes before calc_metric() runs
        # isolates parallel SLURM tasks from each other's *.pt embeddings and
        # entities_cache.json. The workdir is cleaned up by shutil.rmtree below.
        task_cache = workdir / "cache"
        task_cache.mkdir(parents=True, exist_ok=True)
        _re.cache_path      = str(task_cache) + "/"
        _re.pred_embed_path = str(task_cache / "pred_embeddings.pt")
        _re.gt_embed_path   = str(task_cache / "gt_embeddings.pt")
        # use_idf=False matches config.py default; the v1 regression was
        # fit with this setting.
        _re.calc_metric(str(gt_csv), str(pred_csv), str(scored_csv), use_idf=False)
    finally:
        os.chdir(cwd)

    n_merged = merge_scores(results, scored_csv)
    print(f"[INFO] merged radcliq + 4 inputs into {n_merged}/{len(results)} entries")

    out_data = {"_metadata": data.get("_metadata", {}), "results": results}
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"[OK] wrote {out_path}")

    if args.keep_cache:
        print(f"[INFO] intermediate files retained at {workdir}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
