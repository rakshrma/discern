import pandas as pd
import ast
import glob
import os


def parse_findings(cell):
    if pd.isna(cell) or cell == "":
        return []
    try:
        return ast.literal_eval(str(cell))
    except Exception:
        return []


def compute_counts(findings):
    counts = {
        "false_prediction_insignificant": 0,
        "false_prediction_significant": 0,
        "omission_finding_insignificant": 0,
        "omission_finding_significant": 0,
        "incorrect_location_insignificant": 0,
        "incorrect_location_significant": 0,
        "incorrect_severity_insignificant": 0,
        "incorrect_severity_significant": 0,
        "extra_comparison_insignificant": 0,
        "extra_comparison_significant": 0,
        "omitted_comparison_insignificant": 0,
        "omitted_comparison_significant": 0,
        "partial_significant": 0,
        "partial_insignificant": 0,
    }

    for f in findings:
        sig   = f.get("significance_score", 0) >= 3
        insig = f.get("significance_score", 0) > 0 and f.get("significance_score", 0) < 3

        discrepancy_type      = f.get("discrepancy_type")
        diagnosis_concordance = f.get("diagnosis_concordance")
        temporal_comparison   = f.get("temporal_comparison")
        severity_concordance  = f.get("severity_concordance")
        location_concordance  = f.get("location_concordance")

        # 1. False prediction / partial diagnosis
        if diagnosis_concordance == "discordant" or discrepancy_type == "extra_in_candidate":
            if sig:
                counts["false_prediction_significant"] += 1
            elif insig:
                counts["false_prediction_insignificant"] += 1
        elif diagnosis_concordance == "partial":
            if sig:
                counts["partial_significant"] += 1
            elif insig:
                counts["partial_insignificant"] += 1

        # 2. Omission of finding
        if discrepancy_type == "missing_in_candidate":
            if sig:
                counts["omission_finding_significant"] += 1
            elif insig:
                counts["omission_finding_insignificant"] += 1

        # 3. Incorrect location/position
        if (location_concordance == "discordant"
                or location_concordance == "candidate-adds"
                or location_concordance == "candidate-misses"):
            if sig:
                counts["incorrect_location_significant"] += 1
            elif insig:
                counts["incorrect_location_insignificant"] += 1
        elif location_concordance == "partial":
            if sig:
                counts["partial_significant"] += 1
            elif insig:
                counts["partial_insignificant"] += 1

        # 4. Incorrect severity
        if (severity_concordance == "discordant"
                or severity_concordance == "candidate-adds"
                or severity_concordance == "candidate-misses"):
            if sig:
                counts["incorrect_severity_significant"] += 1
            elif insig:
                counts["incorrect_severity_insignificant"] += 1
        elif severity_concordance == "partial":
            if sig:
                counts["partial_significant"] += 1
            elif insig:
                counts["partial_insignificant"] += 1

        # 5. Extra / omitted / partial comparison
        if temporal_comparison == "candidate-adds":
            if sig:
                counts["extra_comparison_significant"] += 1
            elif insig:
                counts["extra_comparison_insignificant"] += 1
        elif temporal_comparison == "candidate-misses":
            if sig:
                counts["omitted_comparison_significant"] += 1
            elif insig:
                counts["omitted_comparison_insignificant"] += 1
        elif temporal_comparison == "partial":
            if sig:
                counts["partial_significant"] += 1
            elif insig:
                counts["partial_insignificant"] += 1

    return counts


# ============================================================
# Penalty rubric
# ============================================================
DIAGNOSIS_PENALTY = {
    "partial":    1,
    "discordant": 1,
}

ATTRIBUTE_PENALTY = {
    "discordant":       1,
    "partial":          1,
    "candidate-misses": 1,
    "candidate-adds":   1,
}


def compute_entity_penalty(f: dict) -> float:
    raw_penalty = 0.0
    raw_penalty += DIAGNOSIS_PENALTY.get(f.get("diagnosis_concordance"), 0)
    raw_penalty += ATTRIBUTE_PENALTY.get(f.get("location_concordance"), 0)
    raw_penalty += ATTRIBUTE_PENALTY.get(f.get("severity_concordance"), 0)
    raw_penalty += ATTRIBUTE_PENALTY.get(f.get("temporal_comparison"), 0)
    significance = float(f.get("significance_score", 0))
    return significance * raw_penalty


def compute_reads_score(findings: list) -> float:
    if not findings:
        return 0.0
    total_penalty = sum(compute_entity_penalty(f) for f in findings)
    total_entity  = len(findings)
    return total_penalty if total_entity > 0 else 0.0


def process_csv(input_path, findings_column, output_path=None):
    if output_path is None:
        output_path = input_path.replace(".csv", "_processed.csv")

    df     = pd.read_csv(input_path)
    parsed = df[findings_column].apply(parse_findings)

    count_rows = parsed.apply(compute_counts)
    count_df   = pd.DataFrame(count_rows.tolist())

    df["reads_score"] = parsed.apply(compute_reads_score)
    df = pd.concat([df, count_df], axis=1)
    df.to_csv(output_path, index=False)
    print(f"Saved processed file to: {output_path}")
    return df


def process_directory(directory, findings_column):
    """Find all radevalx_reads_evaluation_*.csv files in directory and process each."""
    pattern      = os.path.join(directory, "radevalx_reads_evaluation_*.csv")
    input_files  = glob.glob(pattern)

    # Exclude already-processed files
    input_files = [f for f in input_files if not f.endswith("_processed.csv")]

    if not input_files:
        print(f"No matching CSV files found in: {directory}")
        return {}

    print(f"Found {len(input_files)} file(s) to process:\n")
    all_results = {}

    for input_path in sorted(input_files):
        filename    = os.path.basename(input_path)
        output_path = input_path.replace(".csv", "_processed.csv")
        print(f"Processing: {filename}")
        try:
            df = process_csv(input_path, findings_column, output_path)
            print(df[["reads_score"]].describe())
            all_results[filename] = df
        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")
        print()

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process radevalx_reads_evaluation_*.csv files in a directory."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="../data/radevalx",
        help="Directory containing radevalx_reads_evaluation_*.csv files",
    )
    parser.add_argument(
        "--findings-column",
        default="reads_eval",
        help="Name of the column containing findings (default: reads_eval)",
    )
    args = parser.parse_args()

    process_directory(args.directory, args.findings_column)