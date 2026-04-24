from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from scipy.stats import kendalltau, spearmanr
from typing import Dict, Tuple

# ── Radevalx column 1-8 → our column prefix ──────────────────────────────────
RADEVALX_CATEGORY_TO_COLUMN = {
    "one":   "false_prediction",
    "two":   "omission_finding",
    "three": "incorrect_location",
    "four":  "incorrect_severity",
    "five":  "extra_comparison",
    "six":   "omitted_comparison",
    "seven": "partial",
    "eight": "partial",
}

OUR_COLUMNS = [
    "false_prediction_insignificant",
    "false_prediction_significant",
    "omission_finding_insignificant",
    "omission_finding_significant",
    "incorrect_location_insignificant",
    "incorrect_location_significant",
    "incorrect_severity_insignificant",
    "incorrect_severity_significant",
    "extra_comparison_insignificant",
    "extra_comparison_significant",
    "omitted_comparison_insignificant",
    "omitted_comparison_significant",
    "partial_insignificant",
    "partial_significant",
]

SIGNIFICANT_COLUMNS   = [c for c in OUR_COLUMNS if c.endswith("_significant")]
INSIGNIFICANT_COLUMNS = [c for c in OUR_COLUMNS if c.endswith("_insignificant")]
RATER_SUMMARY_COLUMNS = ["total_errors", "total_significant", "total_insignificant"]
SUMMARY_COLUMNS       = RATER_SUMMARY_COLUMNS + ["discern_score"]
ALL_COMPARE_COLUMNS   = OUR_COLUMNS + SUMMARY_COLUMNS
RATER_COMPARE_COLUMNS = OUR_COLUMNS + RATER_SUMMARY_COLUMNS


def add_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_errors"]        = df[OUR_COLUMNS].sum(axis=1)
    df["total_significant"]   = df[SIGNIFICANT_COLUMNS].sum(axis=1)
    df["total_insignificant"] = df[INSIGNIFICANT_COLUMNS].sum(axis=1)
    if "discern_score" not in df.columns:
        df["discern_score"] = 0.0
    return df


def build_radevalx_rater_table(df_radevalx: pd.DataFrame) -> pd.DataFrame:
    df = df_radevalx.copy()
    df["partial"] = df["seven"] + df["eight"]

    CATEGORY_COLS = {
        "one":   "false_prediction",
        "two":   "omission_finding",
        "three": "incorrect_location",
        "four":  "incorrect_severity",
        "five":  "extra_comparison",
        "six":   "omitted_comparison",
    }

    rows = []
    for _, row in df.iterrows():
        suffix = row["error_type"]  # "significant" or "insignificant"
        record = {"study_id": row["report_id"]}
        for src_col, prefix in CATEGORY_COLS.items():
            record[f"{prefix}_{suffix}"] = row[src_col]
        record[f"partial_{suffix}"] = row["partial"]
        rows.append(record)

    long_df  = pd.DataFrame(rows)
    rater_wide = (
        long_df
        .groupby("study_id")[OUR_COLUMNS]
        .sum()
        .reset_index()
    )
    for col in OUR_COLUMNS:
        if col not in rater_wide.columns:
            rater_wide[col] = 0

    return add_summary_columns(rater_wide)


def _run_metrics(merged: pd.DataFrame, columns: list) -> list:
    results = []
    for col in columns:
        col_ours  = f"{col}_ours"
        col_rater = f"{col}_rater"
        if col_ours not in merged.columns or col_rater not in merged.columns:
            continue
        if merged[col_ours].std() == 0 or merged[col_rater].std() == 0:
            continue
        tau, p_tau      = kendalltau(merged[col_ours], merged[col_rater])
        rho, p_spearman = spearmanr(merged[col_ours],  merged[col_rater])
        mae             = (merged[col_ours] - merged[col_rater]).abs().mean()
        results.append({
            "column":     col,
            "tau":        tau,
            "p_tau":      p_tau,
            "spearman":   rho,
            "p_spearman": p_spearman,
            "mae":        mae,
            "n_samples":  len(merged),
        })
    return results


def compare_radevalx(
    df_ours: pd.DataFrame,
    rater_table: pd.DataFrame,
) -> pd.DataFrame:
    merged = df_ours.merge(rater_table, on="study_id", suffixes=("_ours", "_rater"))
    if merged.empty:
        print("Warning: merge produced no rows — check study_id alignment.")
        return pd.DataFrame()
    return pd.DataFrame(_run_metrics(merged, RATER_COMPARE_COLUMNS))


def correlate_discern_score_with_counts(df_ours: pd.DataFrame) -> pd.DataFrame:
    target_cols = ["total_errors", "total_significant", "total_insignificant"]
    results = []
    for col in target_cols:
        rho, p_spearman = spearmanr(df_ours["discern_score"], df_ours[col])
        tau, p_tau      = kendalltau(df_ours["discern_score"], df_ours[col])
        results.append({
            "count_metric": col,
            "spearman":     rho,
            "p_spearman":   p_spearman,
            "tau":          tau,
            "p_tau":        p_tau,
            "n_samples":    df_ours["discern_score"].notna().sum(),
        })
    return pd.DataFrame(results)


def plot_radevalx_counts(
    df_ours: pd.DataFrame,
    rater_table: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure:
    row_labels = ["total_errors", "total_significant", "total_insignificant", "discern_score"]
    colors     = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    n_rows     = len(row_labels)

    fig, axes = plt.subplots(n_rows, 1, figsize=(5, 4 * n_rows), squeeze=False)
    merged    = df_ours.merge(rater_table, on="study_id", suffixes=("_ours", "_rater"))

    for row_i, metric in enumerate(row_labels):
        ax = axes[row_i][0]

        if metric == "discern_score":
            ax.hist(df_ours["discern_score"].dropna(), bins=20,
                    color=colors[row_i], alpha=0.7, edgecolor="white")
            ax.set_xlabel("discern Score", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.set_title("discern Score Distribution", fontsize=10, fontweight="bold")
            mean_val   = df_ours["discern_score"].mean()
            median_val = df_ours["discern_score"].median()
            ax.axvline(mean_val,   color="black", linestyle="--", linewidth=1,
                       label=f"Mean={mean_val:.1f}")
            ax.axvline(median_val, color="gray",  linestyle=":",  linewidth=1,
                       label=f"Median={median_val:.1f}")
            ax.legend(fontsize=7)
        else:
            col_ours  = f"{metric}_ours"
            col_rater = f"{metric}_rater"
            if col_ours in merged.columns and col_rater in merged.columns:
                x, y = merged[col_ours], merged[col_rater]
                ax.scatter(x, y, alpha=0.6, s=25, color=colors[row_i])
                lim = max(x.max(), y.max()) + 0.5
                ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
                rho, _ = spearmanr(x, y)
                tau, _ = kendalltau(x, y)
                ax.annotate(f"ρ={rho:.2f}  τ={tau:.2f}",
                            xy=(0.05, 0.92), xycoords="axes fraction", fontsize=8)
                ax.set_xlabel("Ours", fontsize=9)
                ax.set_ylabel("Radevalx", fontsize=9)
                ax.set_title(metric.replace("_", " ").title(), fontsize=10, fontweight="bold")
                ax.set_aspect("equal")

    fig.suptitle("Our Error Counts vs Radevalx", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")
    return fig


def run_radevalx_comparison(
    df_ours: pd.DataFrame,
    df_radevalx: pd.DataFrame,
    output_path: str | None = None,
    score_corr_path: str | None = None,
    plot_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rater_table      = build_radevalx_rater_table(df_radevalx)
    df_ours_with_sum = add_summary_columns(df_ours)

    metrics    = compare_radevalx(df_ours_with_sum, rater_table)
    score_corr = correlate_discern_score_with_counts(df_ours_with_sum)

    if output_path:
        metrics.to_csv(output_path, index=False)
        print(f"Saved metrics to: {output_path}")
    if score_corr_path:
        score_corr.to_csv(score_corr_path, index=False)
        print(f"Saved score correlations to: {score_corr_path}")
    if plot_path:
        plot_radevalx_counts(df_ours_with_sum, rater_table, output_path=plot_path)

    return rater_table, metrics, score_corr


def extract_model_name(filename: str) -> str:
    """Extract model name from radevalx_discern_evaluation_<model>_processed.csv"""
    base = os.path.basename(filename)
    name = base.replace("radevalx_discern_evaluation_", "").replace("_processed.csv", "")
    # handle the double-suffix .csv_processed.csv pattern
    name = name.replace(".csv", "")
    return name


def collate_metrics(
    all_results: Dict[str, dict],
    output_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collate metrics across all models into wide and long formats.
    """
    frames = []
    for model_name, results in all_results.items():
        df = results["metrics"].copy()
        df["model"] = model_name
        frames.append(df)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    long = combined[["model", "column", "spearman", "p_spearman", "tau", "p_tau", "mae", "n_samples"]].sort_values(
        ["column", "model"]
    )

    wide = combined.pivot(
        index="column", columns="model",
        values=["spearman", "tau", "mae", "p_spearman", "p_tau", "n_samples"]
    )
    wide.columns = [f"{model}__{stat}" for stat, model in wide.columns]
    wide = wide.reset_index().rename(columns={"column": "metric"})

    if output_path:
        long_path = output_path.replace(".csv", "_long.csv")
        wide_path = output_path.replace(".csv", "_wide.csv")
        long.to_csv(long_path, index=False)
        wide.to_csv(wide_path, index=False)
        print(f"Saved collated long results to: {long_path}")
        print(f"Saved collated wide results to: {wide_path}")

    return wide, long


def process_directory(
    directory: str,
    df_radevalx: pd.DataFrame,
) -> Tuple[Dict[str, dict], pd.DataFrame, pd.DataFrame]:
    """
    Find all radevalx_discern_evaluation_*_processed.csv files in directory
    and run the full radevalx comparison for each.
    """
    # Handle both naming conventions: _processed.csv and .csv_processed.csv
    patterns     = [
        os.path.join(directory, "radevalx_discern_evaluation_*_processed.csv"),
        os.path.join(directory, "radevalx_discern_evaluation_*.csv_processed.csv"),
    ]
    input_files = sorted(set(f for p in patterns for f in glob.glob(p)))

    if not input_files:
        print(f"No matching *_processed.csv files found in: {directory}")
        return {}, pd.DataFrame(), pd.DataFrame()

    print(f"Found {len(input_files)} processed file(s) to compare:\n")
    all_results = {}

    for input_path in input_files:
        model_name = extract_model_name(input_path)
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"File:  {os.path.basename(input_path)}")

        try:
            df_ours = pd.read_csv(input_path)

            output_path     = os.path.join(directory, f"radevalx_metrics_{model_name}.csv")
            score_corr_path = os.path.join(directory, f"radevalx_score_corr_{model_name}.csv")
            plot_path       = os.path.join(directory, f"radevalx_scatter_{model_name}.png")

            rater_table, metrics, score_corr = run_radevalx_comparison(
                df_ours=df_ours,
                df_radevalx=df_radevalx,
                output_path=output_path,
                score_corr_path=score_corr_path,
                plot_path=plot_path,
            )

            print("\n-- Tau-b + Spearman + MAE vs Radevalx --")
            print(metrics.to_string(index=False))
            print("\n-- discern Score vs Error Count Correlations --")
            print(score_corr.to_string(index=False))

            all_results[model_name] = {
                "rater_table": rater_table,
                "metrics":     metrics,
                "score_corr":  score_corr,
            }

        except Exception as e:
            print(f"  ERROR processing {os.path.basename(input_path)}: {e}")

        print()

    # ── Collate across all models ─────────────────────────────────────────
    collate_path = os.path.join(directory, "collated_radevalx_metrics.csv")
    wide, long   = collate_metrics(all_results, output_path=collate_path)

    print(f"\n{'='*60}")
    print("Collated metrics (long format):")
    print(long.to_string(index=False))

    return all_results, wide, long


# ── Standalone usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Radevalx comparison for all radevalx_discern_evaluation_*_processed.csv files."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="../data/radevalx",
        help="Directory containing *_processed.csv files",
    )
    parser.add_argument(
        "--radevalx",
        default="../data/radevalx/radeval_total.csv",
        help="Path to the radevalx ground truth CSV (radeval_total.csv)",
    )
    args = parser.parse_args()

    df_radevalx = pd.read_csv(args.radevalx)

    all_results, collated_wide, collated_long = process_directory(
        directory=args.directory,
        df_radevalx=df_radevalx,
    )

    print(f"\nDone. Processed {len(all_results)} model(s): {', '.join(all_results.keys())}")
    print("\nCollated wide summary (first few columns):")
    print(collated_wide.iloc[:, :7].to_string(index=False))