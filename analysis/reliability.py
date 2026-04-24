"""
reliability.py — Compute inter-run reliability (ICC) across N DISCERN repeats.

Usage
-----
  python analysis/reliability.py \\
      --input-dir data/discern_runs/repeat_reliability/ \\
      --output    data/analysis/reliability_results.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def _load_run(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    rows = data.get("results", data) if isinstance(data, dict) else data
    df = pd.DataFrame(rows)
    run_name = Path(path).stem
    for col in ["discern_score", "mini_discern_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, run_name


def _icc_1_1(ratings: np.ndarray) -> float:
    """ICC(1,1) — one-way random effects, absolute agreement."""
    n, k = ratings.shape
    grand_mean = ratings.mean()
    ss_total = ((ratings - grand_mean) ** 2).sum()
    subject_means = ratings.mean(axis=1, keepdims=True)
    ss_between = k * ((subject_means - grand_mean) ** 2).sum()
    ss_within = ((ratings - subject_means) ** 2).sum()
    ms_between = ss_between / (n - 1)
    ms_within  = ss_within  / (n * (k - 1))
    return float((ms_between - ms_within) / (ms_between + (k - 1) * ms_within))


def run(args: argparse.Namespace):
    input_dir = Path(args.input_dir)
    run_files = sorted(input_dir.glob("*.json"))

    if len(run_files) < 2:
        print(f"Need at least 2 repeat JSON files in {input_dir}, found {len(run_files)}.")
        return

    # Load all runs
    runs = []
    run_names = []
    for f in run_files:
        df, name = _load_run(str(f))
        runs.append(df)
        run_names.append(name)

    print(f"Loaded {len(runs)} runs: {run_names}")

    results = []
    for score_col in ["discern_score", "mini_discern_score"]:
        cols_present = [r for r in runs if score_col in r.columns]
        if len(cols_present) < 2:
            continue

        # Align by sample_idx
        merged = runs[0][["sample_idx", score_col]].rename(columns={score_col: "r0"})
        for i, run_df in enumerate(runs[1:], 1):
            if score_col in run_df.columns:
                merged = merged.merge(
                    run_df[["sample_idx", score_col]].rename(columns={score_col: f"r{i}"}),
                    on="sample_idx",
                    how="inner",
                )

        r_cols = [c for c in merged.columns if c.startswith("r")]
        mat = merged[r_cols].dropna().values
        if mat.shape[0] < 3:
            continue

        icc = _icc_1_1(mat)
        mean_std = mat.std(axis=1).mean()  # average intra-row std

        # Kendall's W (concordance)
        n, k = mat.shape
        ranks = np.argsort(np.argsort(mat, axis=0), axis=0) + 1
        R_i = ranks.sum(axis=1)
        S = ((R_i - R_i.mean()) ** 2).sum()
        W = (12 * S) / (k ** 2 * (n ** 3 - n))

        print(f"\n  {score_col}  (n={n} pairs, k={k} raters)")
        print(f"    ICC(1,1)   = {icc:.4f}")
        print(f"    Kendall W  = {W:.4f}")
        print(f"    Mean±Std   = {mat.mean():.3f} ± {mat.std():.3f}")
        print(f"    Intra-run σ (avg per sample) = {mean_std:.3f}")

        results.append({
            "metric": score_col,
            "n_pairs": n,
            "n_repeats": k,
            "icc_1_1": round(icc, 4),
            "kendall_w": round(W, 4),
            "grand_mean": round(mat.mean(), 3),
            "grand_std":  round(mat.std(), 3),
            "mean_intrarun_std": round(mean_std, 3),
        })

    if results:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out, index=False)
        print(f"\nReliability results saved → {out}")


def main():
    parser = argparse.ArgumentParser(description="Compute DISCERN inter-run reliability.")
    parser.add_argument("--input-dir", required=True, help="Directory with repeat JSON files")
    parser.add_argument("--output",    default="data/analysis/reliability_results.csv")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
