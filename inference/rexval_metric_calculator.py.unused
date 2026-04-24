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
    }

    for f in findings:
        sig = f.get("significance_score", 0) >= 2
        insig = f.get("significance_score", 0) > 0 and f.get("significance_score", 0) < 2

        discrepancy_type      = f.get("discrepancy_type")
        diagnosis_concordance = f.get("diagnosis_concordance")
        location_concordance = f.get("location_concordance")
        severity_concordance = f.get("severity_concordance")
        temporal_comparison = f.get("temporal_comparison")
        if (diagnosis_concordance == "discordant" 
            or discrepancy_type == "extra_in_candidate" 
            or diagnosis_concordance == "partial"
            or temporal_comparison == "discordant"):
            if sig:
                counts["false_prediction_significant"] += 1
            elif insig:
                counts["false_prediction_insignificant"] += 1

        if discrepancy_type == "missing_in_candidate":
            if sig:
                counts["omission_finding_significant"] += 1
            elif insig:
                counts["omission_finding_insignificant"] += 1

        
        if (location_concordance == "discordant"
                or location_concordance == "candidate-adds"
                or location_concordance == "candidate-misses"
                or location_concordance == "partial"):
            if sig:
                counts["incorrect_location_significant"] += 1
            elif insig:
                counts["incorrect_location_insignificant"] += 1

        
        if (severity_concordance == "discordant"
                or severity_concordance == "candidate-adds"
                or severity_concordance == "candidate-misses"
                or severity_concordance == "partial"):
            if sig:
                counts["incorrect_severity_significant"] += 1
            elif insig:
                counts["incorrect_severity_insignificant"] += 1

        
        if (temporal_comparison == "candidate-adds" 
            or temporal_comparison == 'partial'):
            if sig:
                counts["extra_comparison_significant"] += 1
            elif insig:
                counts["extra_comparison_insignificant"] += 1
        elif (temporal_comparison == "candidate-misses"):
            if sig:
                counts["omitted_comparison_significant"] += 1
            elif insig:
                counts["omitted_comparison_insignificant"] += 1

    return counts


DIAGNOSIS_PENALTY = {
    "partial":    1,
    "discordant": 1,
}

ATTRIBUTE_PENALTY = {
    "discordant":      1,
    "partial":         1,
    "candidate-misses":1,
    "candidate-adds":  1,
}


def compute_entity_penalty(f: dict) -> float:
    raw_penalty = 0.0
    diagnosis_concordance = f.get("diagnosis_concordance")
    raw_penalty += DIAGNOSIS_PENALTY.get(diagnosis_concordance, 0)
    location_concordance = f.get("location_concordance")
    raw_penalty += ATTRIBUTE_PENALTY.get(location_concordance, 0)
    severity_concordance = f.get("severity_concordance")
    raw_penalty += ATTRIBUTE_PENALTY.get(severity_concordance, 0)
    temporal_comparison = f.get("temporal_comparison")
    raw_penalty += ATTRIBUTE_PENALTY.get(temporal_comparison, 0)
    significance = float(f.get("significance_score", 0))
    return significance * raw_penalty


def compute_discern_score(findings: list) -> float:
    if not findings:
        return 0.0
    total_penalty = sum(compute_entity_penalty(f) for f in findings)
    total_entity = len(findings)
    return total_penalty if total_entity > 0 else 0.0


def process_csv(input_path, findings_column, output_path=None):
    if output_path is None:
        output_path = input_path.replace(".csv", "_processed.csv")

    df = pd.read_csv(input_path)
    parsed = df[findings_column].apply(parse_findings)

    count_rows = parsed.apply(compute_counts)
    count_df   = pd.DataFrame(count_rows.tolist())

    df["discern_score"] = parsed.apply(compute_discern_score)
    df = pd.concat([df, count_df], axis=1)
    df.to_csv(output_path, index=False)
    print(f"Saved processed file to: {output_path}")
    return df


def process_directory(directory, findings_column):
    """Find all rexval_discern_evaluation_*.csv files in directory and process each."""
    pattern = os.path.join(directory, "rexval_discern_evaluation_*.csv")
    input_files = glob.glob(pattern)

    # Exclude already-processed files
    input_files = [f for f in input_files if not f.endswith("_processed.csv")]

    if not input_files:
        print(f"No matching CSV files found in: {directory}")
        return

    print(f"Found {len(input_files)} file(s) to process:\n")
    all_results = {}

    for input_path in sorted(input_files):
        filename = os.path.basename(input_path)
        output_path = input_path.replace(".csv", "_processed.csv")
        print(f"Processing: {filename}")
        try:
            df = process_csv(input_path, findings_column, output_path)
            print(df[["discern_score"]].describe())
            all_results[filename] = df
        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")
        print()

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process rexval_discern_evaluation_*.csv files in a directory."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="../data/rexval",
        help="Directory containing rexval_discern_evaluation_*.csv files",
    )
    parser.add_argument(
        "--findings-column",
        default="discern_eval",
        help="Name of the column containing findings (default: discern_eval)",
    )
    args = parser.parse_args()

    process_directory(args.directory, args.findings_column)