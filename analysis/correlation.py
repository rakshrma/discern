"""
correlation.py — Correlate DISCERN + all metrics against human annotations.

Supports ReXVal and RaDEvalX ground-truth annotation files from PhysioNet.

Usage
-----
  python analysis/correlation.py \\
      --rexval-scores    data/discern_runs/rexval_all_metrics.json \\
      --rexval-gt        data/rexval/ \\
      --radevalx-scores  data/discern_runs/radevalx_all_metrics.json \\
      --radevalx-gt      data/radevalx/ \\
      --output-dir       data/analysis/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from scipy.stats import kendalltau, spearmanr

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

METRIC_COLS = [
    "bleu", "rouge", "meteor", "bertscore",
    "semb_score", "radgraph", "radcliq", "ratescore",
    "green", "crimson", "discern_score", "mini_discern_score",
]


def _load_scores(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    rows = data.get("results", data) if isinstance(data, dict) else data
    return pd.DataFrame(rows)


def _correlate(x: pd.Series, y: pd.Series, metric: str, gt_col: str, source: str) -> Optional[dict]:
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    rho, p_rho = spearmanr(x, y)
    tau, p_tau = kendalltau(x, y)
    return {
        "source": source,
        "metric": metric,
        "gt_col": gt_col,
        "n": len(x),
        "spearman_rho": round(float(rho), 4),
        "p_spearman":   round(float(p_rho), 4),
        "kendall_tau":  round(float(tau), 4),
        "p_kendall":    round(float(p_tau), 4),
    }


def _load_rexval_gt(gt_dir: str) -> Optional[pd.DataFrame]:
    gt_path = Path(gt_dir)
    candidates_file = next(gt_path.glob("*50_samples*"), None) or next(gt_path.glob("*.csv"), None)
    rater_file = next(gt_path.glob("*rater*"), None)
    if rater_file is None:
        return None
    try:
        df = pd.read_csv(rater_file)
        studies = pd.read_csv(candidates_file) if candidates_file else None
        if studies is not None and "study_id" in studies.columns:
            study_ids = list(studies["study_id"])
            df["study_id"] = df["study_number"].map(lambda i: study_ids[int(i)])
        df["num_errors"] = pd.to_numeric(df["num_errors"], errors="coerce").fillna(0)
        df["sig"] = df.get("clinically_significant", pd.Series(dtype=bool)).map(
            lambda v: bool(str(v).strip() == "True") if not isinstance(v, bool) else v
        )
        agg = df.groupby(["study_id", "candidate_type"])["num_errors"].mean().reset_index()
        agg.columns = ["study_id", "candidate_type", "gt_mean_errors"]
        return agg
    except Exception as e:
        print(f"Warning: Could not load ReXVal GT: {e}")
        return None


def _load_radevalx_gt(gt_dir: str) -> Optional[pd.DataFrame]:
    gt_path = Path(gt_dir)
    sig_file = next(gt_path.glob("*significant*errors*"), None)
    if sig_file is None:
        return None
    try:
        df = pd.read_csv(sig_file)
        rater_cols = [c for c in df.columns if c.isdigit() or c.startswith("rater")]
        if not rater_cols:
            rater_cols = df.columns[1:].tolist()
        df[rater_cols] = df[rater_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        df["gt_mean_errors"] = df[rater_cols].mean(axis=1)
        return df[["report_id", "gt_mean_errors"]].copy()
    except Exception as e:
        print(f"Warning: Could not load RaDEvalX GT: {e}")
        return None


def run(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_corr = []

    # ── ReXVal ────────────────────────────────────────────────────────────────
    if args.rexval_scores and args.rexval_gt:
        scores_df = _load_scores(args.rexval_scores)
        gt_df = _load_rexval_gt(args.rexval_gt)
        if gt_df is not None and "study_id" in scores_df.columns:
            merged = scores_df.merge(gt_df, on="study_id", how="inner")
            print(f"ReXVal: {len(merged)} matched rows")
            for metric in METRIC_COLS:
                if metric in merged.columns:
                    c = _correlate(merged[metric], merged["gt_mean_errors"],
                                   metric, "gt_mean_errors", "rexval")
                    if c:
                        all_corr.append(c)

    # ── RaDEvalX ──────────────────────────────────────────────────────────────
    if args.radevalx_scores and args.radevalx_gt:
        scores_df = _load_scores(args.radevalx_scores)
        gt_df = _load_radevalx_gt(args.radevalx_gt)
        if gt_df is not None and "report_id" in scores_df.columns:
            merged = scores_df.merge(gt_df, on="report_id", how="inner")
            print(f"RaDEvalX: {len(merged)} matched rows")
            for metric in METRIC_COLS:
                if metric in merged.columns:
                    c = _correlate(merged[metric], merged["gt_mean_errors"],
                                   metric, "gt_mean_errors", "radevalx")
                    if c:
                        all_corr.append(c)

    if not all_corr:
        print("No correlations computed — check input paths and GT files.")
        return

    corr_df = pd.DataFrame(all_corr)
    out_path = output_dir / "metric_correlation_results.csv"
    corr_df.to_csv(out_path, index=False)
    print(f"\nCorrelation results saved → {out_path}")

    # Print summary sorted by Spearman rho
    print("\n── Spearman ρ (higher = better alignment with human annotations) ──")
    for source in corr_df["source"].unique():
        sub = corr_df[corr_df["source"] == source].sort_values("spearman_rho", ascending=False)
        print(f"\n  {source.upper()}")
        for _, row in sub.iterrows():
            sig = "**" if row["p_spearman"] < 0.05 else "  "
            print(f"  {sig} {row['metric']:25s}  ρ={row['spearman_rho']:+.3f}  τ={row['kendall_tau']:+.3f}  n={row['n']}")


def main():
    parser = argparse.ArgumentParser(description="Correlate metrics with human annotations.")
    parser.add_argument("--rexval-scores",   default=None)
    parser.add_argument("--rexval-gt",       default=None)
    parser.add_argument("--radevalx-scores", default=None)
    parser.add_argument("--radevalx-gt",     default=None)
    parser.add_argument("--output-dir",      default="data/analysis/")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
