from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import glob
import os
from scipy.stats import kendalltau, spearmanr
from typing import Dict, Tuple

# ── Column name mapping: ReXVal error_category → our column prefix ───────────
REXVAL_CATEGORY_TO_COLUMN = {
    1: "false_prediction",
    2: "omission_finding",
    3: "incorrect_location",
    4: "incorrect_severity",
    5: "extra_comparison",
    6: "omitted_comparison",
}

SIG_TO_SUFFIX = {
    True:  "significant",
    False: "insignificant",
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
]

SIGNIFICANT_COLUMNS   = [c for c in OUR_COLUMNS if c.endswith("_significant")]
INSIGNIFICANT_COLUMNS = [c for c in OUR_COLUMNS if c.endswith("_insignificant")]

SUMMARY_COLUMNS     = ["total_errors", "total_significant", "total_insignificant", "discern_score"]
ALL_COMPARE_COLUMNS = OUR_COLUMNS + SUMMARY_COLUMNS


def build_study_id_map(df_studies: pd.DataFrame, study_id_col: str = "study_id") -> Dict[int, str]:
    return {idx: sid for idx, sid in enumerate(df_studies[study_id_col])}


def add_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_errors"]        = df[OUR_COLUMNS].sum(axis=1)
    df["total_significant"]   = df[SIGNIFICANT_COLUMNS].sum(axis=1)
    df["total_insignificant"] = df[INSIGNIFICANT_COLUMNS].sum(axis=1)
    if "discern_score" not in df.columns:
        df["discern_score"] = 0.0
    return df


def build_rater_table(
    df_rexval: pd.DataFrame,
    index_to_study_id: Dict[int, str],
) -> pd.DataFrame:
    df = df_rexval.copy()
    df["study_id"] = df["study_number"].map(index_to_study_id)

    def to_col(row):
        prefix = REXVAL_CATEGORY_TO_COLUMN.get(row["error_category"])
        suffix = SIG_TO_SUFFIX.get(bool(row["clinically_significant"]))
        if prefix is None or suffix is None:
            return None
        return f"{prefix}_{suffix}"

    df["col_name"] = df.apply(to_col, axis=1)
    df = df[df["col_name"].notna()]

    rater_table = (
        df.groupby(["study_id", "candidate_type", "rater_index", "col_name"])["num_errors"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in OUR_COLUMNS:
        if col not in rater_table.columns:
            rater_table[col] = 0

    rater_table = rater_table[["study_id", "candidate_type", "rater_index"] + OUR_COLUMNS]
    return add_summary_columns(rater_table)


def build_avg_rater_table(df_rater_table: pd.DataFrame) -> pd.DataFrame:
    return (
        df_rater_table
        .groupby(["study_id", "candidate_type"])[ALL_COMPARE_COLUMNS]
        .mean()
        .reset_index()
    )


def correlate_discern_score_with_counts(df_ours: pd.DataFrame) -> pd.DataFrame:
    target_cols = ["total_errors", "total_significant", "total_insignificant"]
    results = []
    for col in target_cols:
        rho,  p_spearman = spearmanr(df_ours["discern_score"], df_ours[col])
        tau,  p_tau      = kendalltau(df_ours["discern_score"], df_ours[col])
        results.append({
            "count_metric": col,
            "spearman":     rho,
            "p_spearman":   p_spearman,
            "tau":          tau,
            "p_tau":        p_tau,
            "n_samples":    df_ours["discern_score"].notna().sum(),
        })
    return pd.DataFrame(results)


def _run_metrics(merged: pd.DataFrame, columns: list) -> list:
    results = []
    for col in columns:
        col_ours  = f"{col}_ours"
        col_rater = f"{col}_rater"
        if col_ours not in merged.columns or col_rater not in merged.columns:
            continue
        tau,  p_tau      = kendalltau(merged[col_ours], merged[col_rater])
        rho,  p_spearman = spearmanr(merged[col_ours], merged[col_rater])
        mae              = (merged[col_ours] - merged[col_rater]).abs().mean()
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


def compare_with_raters(
    df_ours: pd.DataFrame,
    df_rater_table: pd.DataFrame,
) -> pd.DataFrame:
    results = []
    for rater_index, rater_group in df_rater_table.groupby("rater_index"):
        merged = df_ours.merge(
            rater_group,
            left_on=["study_id", "candidate_reporter"],
            right_on=["study_id", "candidate_type"],
            suffixes=("_ours", "_rater"),
        )
        if merged.empty:
            continue
        for row in _run_metrics(merged, ALL_COMPARE_COLUMNS):
            row["rater_index"] = rater_index
            results.append(row)

    return pd.DataFrame(results)[
        ["rater_index", "column", "tau", "p_tau", "spearman", "p_spearman", "mae", "n_samples"]
    ]


def compare_with_avg_raters(
    df_ours: pd.DataFrame,
    avg_rater_table: pd.DataFrame,
) -> pd.DataFrame:
    merged = df_ours.merge(
        avg_rater_table,
        left_on=["study_id", "candidate_reporter"],
        right_on=["study_id", "candidate_type"],
        suffixes=("_ours", "_rater"),
    )
    return pd.DataFrame(_run_metrics(merged, ALL_COMPARE_COLUMNS))


def _scatter_panel(ax, x, y, label, color):
    ax.scatter(x, y, alpha=0.5, s=20, color=color, label=label)
    lim_min = min(x.min(), y.min()) - 0.5
    lim_max = max(x.max(), y.max()) + 0.5
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=0.8, alpha=0.5)
    rho, _ = spearmanr(x, y)
    tau, _ = kendalltau(x, y)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.annotate(
        f"ρ={rho:.2f}  τ={tau:.2f}",
        xy=(0.05, 0.92), xycoords="axes fraction", fontsize=7,
    )
    ax.set_aspect("equal")


def plot_counts(
    df_ours: pd.DataFrame,
    df_rater_table: pd.DataFrame,
    avg_rater_table: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure:
    rater_indices = sorted(df_rater_table["rater_index"].unique())
    n_raters      = len(rater_indices)
    n_cols        = n_raters + 1
    n_rows        = 4
    row_labels    = ["total_errors", "total_significant", "total_insignificant", "discern_score"]
    colors        = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.8 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )

    for row_i, metric in enumerate(row_labels):
        col_ours  = f"{metric}_ours"
        col_rater = f"{metric}_rater"

        for col_i, rater_index in enumerate(rater_indices):
            ax = axes[row_i][col_i]

            if metric == "discern_score":
                ax.set_visible(False)
                continue

            rater_group = df_rater_table[df_rater_table["rater_index"] == rater_index]
            merged = df_ours.merge(
                rater_group,
                left_on=["study_id", "candidate_reporter"],
                right_on=["study_id", "candidate_type"],
                suffixes=("_ours", "_rater"),
            )
            if merged.empty:
                ax.set_visible(False)
                continue

            _scatter_panel(ax, merged[col_ours], merged[col_rater], rater_index, colors[row_i])
            ax.set_xlabel("Ours", fontsize=8)
            ax.set_ylabel(f"Rater {rater_index}", fontsize=8)
            if row_i == 0:
                ax.set_title(f"Rater {rater_index}", fontsize=9, fontweight="bold")

        ax = axes[row_i][n_cols - 1]

        if metric == "discern_score":
            ax.hist(df_ours["discern_score"].dropna(), bins=20, color=colors[row_i], alpha=0.7, edgecolor="white")
            ax.set_xlabel("discern Score", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.set_title("discern Score\nDistribution", fontsize=9, fontweight="bold")
            mean_val   = df_ours["discern_score"].mean()
            median_val = df_ours["discern_score"].median()
            ax.axvline(mean_val,   color="black", linestyle="--", linewidth=1, label=f"Mean={mean_val:.1f}")
            ax.axvline(median_val, color="gray",  linestyle=":",  linewidth=1, label=f"Median={median_val:.1f}")
            ax.legend(fontsize=6)
        else:
            merged_avg = df_ours.merge(
                avg_rater_table,
                left_on=["study_id", "candidate_reporter"],
                right_on=["study_id", "candidate_type"],
                suffixes=("_ours", "_rater"),
            )
            if not merged_avg.empty:
                _scatter_panel(ax, merged_avg[col_ours], merged_avg[col_rater], "avg", colors[row_i])
                ax.set_xlabel("Ours", fontsize=8)
                ax.set_ylabel("Avg Rater", fontsize=8)
            if row_i == 0:
                ax.set_title("Avg Rater", fontsize=9, fontweight="bold")

        axes[row_i][0].set_ylabel(
            f"{metric.replace('_', ' ').title()}\nRater 0", fontsize=8
        )

    fig.suptitle("Our Error Counts vs ReXVal Raters", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    return fig


def run_rexval_comparison(
    df_studies: pd.DataFrame,
    df_ours: pd.DataFrame,
    df_rexval: pd.DataFrame,
    study_id_col: str = "study_id",
    output_path: str | None = None,
    avg_output_path: str | None = None,
    plot_path: str | None = None,
    score_corr_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    index_to_study_id = build_study_id_map(df_studies, study_id_col)
    rater_table       = build_rater_table(df_rexval, index_to_study_id)
    avg_rater_table   = build_avg_rater_table(rater_table)
    df_ours_with_sums = add_summary_columns(df_ours)

    tau_per_rater = compare_with_raters(df_ours_with_sums, rater_table)
    tau_avg_rater = compare_with_avg_raters(df_ours_with_sums, avg_rater_table)
    score_corr    = correlate_discern_score_with_counts(df_ours_with_sums)

    if output_path is not None:
        tau_per_rater.to_csv(output_path, index=False)
        print(f"Saved per-rater results to: {output_path}")
    if avg_output_path is not None:
        tau_avg_rater.to_csv(avg_output_path, index=False)
        print(f"Saved avg-rater results to: {avg_output_path}")
    if score_corr_path is not None:
        score_corr.to_csv(score_corr_path, index=False)
        print(f"Saved score correlation results to: {score_corr_path}")
    if plot_path is not None:
        plot_counts(df_ours_with_sums, rater_table, avg_rater_table, output_path=plot_path)

    return rater_table, avg_rater_table, tau_per_rater, tau_avg_rater, score_corr


def extract_model_name(filename: str) -> str:
    """Extract model name from rexval_discern_evaluation_<model>_processed.csv"""
    base = os.path.basename(filename)
    # Strip prefix and suffix
    name = base.replace("rexval_discern_evaluation_", "").replace("_processed.csv", "")
    return name


def process_directory(
    directory: str,
    df_studies: pd.DataFrame,
    df_rexval: pd.DataFrame,
    study_id_col: str = "study_id",
) -> Dict[str, dict]:
    """
    Find all rexval_discern_evaluation_*_processed.csv files in directory
    and run the full rexval comparison for each, saving outputs alongside
    the input files.
    """
    pattern = os.path.join(directory, "rexval_discern_evaluation_*_processed.csv")
    processed_files = sorted(glob.glob(pattern))

    if not processed_files:
        print(f"No matching *_processed.csv files found in: {directory}")
        return {}

    print(f"Found {len(processed_files)} processed file(s) to compare:\n")
    all_results = {}

    for input_path in processed_files:
        model_name = extract_model_name(input_path)
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"File:  {os.path.basename(input_path)}")

        try:
            df_ours = pd.read_csv(input_path)

            output_path     = os.path.join(directory, f"tau_per_rater_{model_name}.csv")
            avg_output_path = os.path.join(directory, f"tau_avg_rater_{model_name}.csv")
            plot_path       = os.path.join(directory, f"scatter_counts_{model_name}.png")
            score_corr_path = os.path.join(directory, f"score_corr_{model_name}.csv")

            rater_table, avg_rater_table, tau_per_rater, tau_avg_rater, score_corr = run_rexval_comparison(
                df_studies=df_studies,
                df_ours=df_ours,
                df_rexval=df_rexval,
                study_id_col=study_id_col,
                output_path=output_path,
                avg_output_path=avg_output_path,
                plot_path=plot_path,
                score_corr_path=score_corr_path,
            )

            print("\n-- discern Score vs Error Count Correlations --")
            print(score_corr.to_string(index=False))
            print("\n-- Tau-b + Spearman + MAE vs average rater --")
            print(tau_avg_rater.to_string(index=False))

            all_results[model_name] = {
                "rater_table":     rater_table,
                "avg_rater_table": avg_rater_table,
                "tau_per_rater":   tau_per_rater,
                "tau_avg_rater":   tau_avg_rater,
                "score_corr":      score_corr,
            }

        except Exception as e:
            print(f"  ERROR processing {os.path.basename(input_path)}: {e}")

        print()

    # ── Collate across all models ─────────────────────────────────────────
    collate_path = os.path.join(directory, "collated_avg_rater.csv")
    wide, long = collate_avg_rater_results(all_results, output_path=collate_path)

    print(f"\n{'='*60}")
    print("Collated avg rater results (long format):")
    print(long.to_string(index=False))

    return all_results, wide, long

def collate_avg_rater_results(
    all_results: Dict[str, dict],
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Collate tau_avg_rater results across all models into a single wide DataFrame.
    Rows = metrics (column), Cols = model_spearman, model_tau, model_mae, ...
    """
    frames = []
    for model_name, results in all_results.items():
        df = results["tau_avg_rater"].copy()
        df["model"] = model_name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Pivot to wide format: one row per metric, grouped columns per model
    wide = combined.pivot(index="column", columns="model", values=["spearman", "tau", "mae", "p_spearman", "p_tau", "n_samples"])

    # Flatten multi-level columns: (metric, model) → model_metric
    wide.columns = [f"{model}__{stat}" for stat, model in wide.columns]
    wide = wide.reset_index().rename(columns={"column": "metric"})

    # Also produce a long format for easier reading
    long = combined[["model", "column", "spearman", "p_spearman", "tau", "p_tau", "mae", "n_samples"]].sort_values(
        ["column", "model"]
    )

    if output_path is not None:
        wide_path = output_path.replace(".csv", "_wide.csv")
        long_path = output_path.replace(".csv", "_long.csv")
        wide.to_csv(wide_path, index=False)
        long.to_csv(long_path, index=False)
        print(f"Saved collated wide results to: {wide_path}")
        print(f"Saved collated long results to: {long_path}")

    return wide, long


# ── Standalone usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ReXVal comparison for all rexval_discern_evaluation_*_processed.csv files."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="../data/rexval",
        help="Directory containing *_processed.csv files",
    )
    parser.add_argument(
        "--studies",
        default="../physionet.org/files/RexVal/50_samples_gt_and_candidates.csv",
        help="Path to the studies CSV (50_samples_gt_and_candidates.csv)",
    )
    parser.add_argument(
        "--rexval",
        default="../physionet.org/files/RexVal/6_valid_raters_per_rater_error_categories.csv",
        help="Path to the ReXVal rater CSV (6_valid_raters_per_rater_error_categories.csv)",
    )
    parser.add_argument(
        "--study-id-col",
        default="study_id",
        help="Name of the study ID column in the studies CSV (default: study_id)",
    )
    args = parser.parse_args()

    df_studies = pd.read_csv(args.studies)
    df_rexval  = pd.read_csv(args.rexval)

    all_results, collated_wide, collated_long = process_directory(
        directory=args.directory,
        df_studies=df_studies,
        df_rexval=df_rexval,
        study_id_col=args.study_id_col,
    )

    print(f"\nDone. Processed {len(all_results)} model(s): {', '.join(all_results.keys())}")
    print("\nCollated wide summary (first few columns):")
    print(collated_wide.iloc[:, :7].to_string(index=False))