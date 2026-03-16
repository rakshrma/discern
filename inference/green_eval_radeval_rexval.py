"""
GREEN + GEMA Score vs Rater Correlation (ReXVal + RadEvalX)
============================================================
Computes GREEN and/or GEMA scores on candidate/reference report pairs,
then correlates against ReXVal and RadEvalX average rater error counts
(total_significant, total_insignificant).

GREEN returns: mean, std, green_score_list, summary, result_df

GEMA is a generative scorer — each pair is run through the model with a
structured prompt and the numeric score is parsed from the JSON output.

Correlations are NEGATED: higher score = better report = fewer errors.
Negating gives "correlation with quality" matching published sign convention.

Usage
-----
  python compute_llm_metrics_raters.py --mode both --scorer both
  python compute_llm_metrics_raters.py --mode rexval --scorer green
  python compute_llm_metrics_raters.py --mode radevalx --scorer gema
  python compute_llm_metrics_raters.py --mode both --scorer gema \\
      --gema-subfolder GEMA-Score-distilled-Xray-llama

  # Skip recomputation (scores already cached):
  python compute_llm_metrics_raters.py --mode rexval --scorer both \\
      --scores-cache-rexval path/to/scores.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── GREEN import ──────────────────────────────────────────────────────────────
try:
    from green_score import GREEN
    HAS_GREEN = True
except ImportError:
    print("⚠ green_score not found — GREEN scoring will be skipped.")
    HAS_GREEN = False

# ── GEMA imports (standard transformers, always available) ────────────────────
from transformers import AutoModelForCausalLM, AutoProcessor


# ─────────────────────────────────────────────────────────────────────────────
# REXVAL RATER TABLE
# ─────────────────────────────────────────────────────────────────────────────

REXVAL_CATEGORY_TO_COLUMN = {
    1: "false_prediction",
    2: "omission_finding",
    3: "incorrect_location",
    4: "incorrect_severity",
    5: "extra_comparison",
    6: "omitted_comparison",
}
SIG_TO_SUFFIX = {True: "significant", False: "insignificant"}

REXVAL_OUR_COLUMNS = [
    "false_prediction_insignificant", "false_prediction_significant",
    "omission_finding_insignificant",  "omission_finding_significant",
    "incorrect_location_insignificant","incorrect_location_significant",
    "incorrect_severity_insignificant","incorrect_severity_significant",
    "extra_comparison_insignificant",  "extra_comparison_significant",
    "omitted_comparison_insignificant","omitted_comparison_significant",
]
REXVAL_SIG_COLS   = [c for c in REXVAL_OUR_COLUMNS if c.endswith("_significant")]
REXVAL_INSIG_COLS = [c for c in REXVAL_OUR_COLUMNS if c.endswith("_insignificant")]


def build_study_id_map(df_studies: pd.DataFrame) -> Dict[int, str]:
    return {idx: sid for idx, sid in enumerate(df_studies["study_id"])}


def build_rexval_rater_table(df_rexval: pd.DataFrame,
                              index_to_study_id: Dict[int, str]) -> pd.DataFrame:
    df = df_rexval.copy()
    df["study_id"] = df["study_number"].map(index_to_study_id)

    def to_col(row):
        prefix = REXVAL_CATEGORY_TO_COLUMN.get(row["error_category"])
        suffix = SIG_TO_SUFFIX.get(bool(row["clinically_significant"]))
        return f"{prefix}_{suffix}" if prefix and suffix else None

    df["col_name"] = df.apply(to_col, axis=1)
    df = df[df["col_name"].notna()]

    rater_table = (
        df.groupby(["study_id", "candidate_type", "rater_index", "col_name"])["num_errors"]
        .sum().unstack(fill_value=0).reset_index()
    )
    for col in REXVAL_OUR_COLUMNS:
        if col not in rater_table.columns:
            rater_table[col] = 0
    rater_table = rater_table[
        ["study_id", "candidate_type", "rater_index"] + REXVAL_OUR_COLUMNS
    ]
    rater_table["total_significant"]   = rater_table[REXVAL_SIG_COLS].sum(axis=1)
    rater_table["total_insignificant"] = rater_table[REXVAL_INSIG_COLS].sum(axis=1)

    return (
        rater_table
        .groupby(["study_id", "candidate_type"])[
            REXVAL_OUR_COLUMNS + ["total_significant", "total_insignificant"]
        ]
        .mean()
        .reset_index()
    )


# ─────────────────────────────────────────────────────────────────────────────
# RADEVALX RATER TABLE
# ─────────────────────────────────────────────────────────────────────────────

RADEVALX_OUR_COLUMNS = [
    "false_prediction_insignificant",   "false_prediction_significant",
    "omission_finding_insignificant",   "omission_finding_significant",
    "incorrect_location_insignificant", "incorrect_location_significant",
    "incorrect_severity_insignificant", "incorrect_severity_significant",
    "extra_comparison_insignificant",   "extra_comparison_significant",
    "omitted_comparison_insignificant", "omitted_comparison_significant",
    "partial_insignificant",            "partial_significant",
]
RADEVALX_SIG_COLS   = [c for c in RADEVALX_OUR_COLUMNS if c.endswith("_significant")]
RADEVALX_INSIG_COLS = [c for c in RADEVALX_OUR_COLUMNS if c.endswith("_insignificant")]


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
        suffix = row["error_type"]
        record = {"study_id": row["report_id"]}
        for src_col, prefix in CATEGORY_COLS.items():
            record[f"{prefix}_{suffix}"] = row[src_col]
        record[f"partial_{suffix}"] = row["partial"]
        rows.append(record)

    long_df    = pd.DataFrame(rows)
    rater_wide = (
        long_df.groupby("study_id")[RADEVALX_OUR_COLUMNS]
        .sum().reset_index()
    )
    for col in RADEVALX_OUR_COLUMNS:
        if col not in rater_wide.columns:
            rater_wide[col] = 0

    rater_wide["total_significant"]   = rater_wide[RADEVALX_SIG_COLS].sum(axis=1)
    rater_wide["total_insignificant"] = rater_wide[RADEVALX_INSIG_COLS].sum(axis=1)
    return rater_wide


# ─────────────────────────────────────────────────────────────────────────────
# GREEN SCORING
# ─────────────────────────────────────────────────────────────────────────────

def compute_green_scores(df_reports: pd.DataFrame,
                         ref_col: str,
                         hyp_col: str,
                         model_name: str,
                         output_dir: str,
                         scores_cache: str | None = None,
                         green_python: str | None = None) -> pd.DataFrame:
    """
    Run GREEN scoring.

    GREEN's torch build may be compiled for a different CUDA arch than the
    current environment. To avoid "no kernel image" errors, GREEN is run in
    its own venv via subprocess when green_python is provided (recommended).

    green_python: path to the python binary inside the green_score venv, e.g.
        /vast/.../green_score/bin/python
    If None, GREEN is imported directly (works only if torch arch matches).
    """
    cache_path = scores_cache or os.path.join(output_dir, "green_scores.csv")

    if os.path.exists(cache_path):
        print(f"Loading cached GREEN scores from {cache_path} ...")
        cached = pd.read_csv(cache_path)
        if "green" in cached.columns and len(cached) == len(df_reports):
            df = df_reports.copy()
            df["green"] = cached["green"].values
            return df
        print("  Cache mismatch -- recomputing ...")

    df   = df_reports.copy()
    refs = df[ref_col].fillna("").tolist()
    hyps = df[hyp_col].fillna("").tolist()

    print(f"Running GREEN on {len(refs)} report pairs ...")
    print(f"  Model: {model_name}")

    if green_python:
        # ── Subprocess mode: run GREEN inside its own venv ─────────────────
        # Write refs/hyps to a temp CSV, run a small inline script, read back scores
        import subprocess, tempfile, sys

        tmp_in  = os.path.join(output_dir, "_green_tmp_input.csv")
        tmp_out = os.path.join(output_dir, "_green_tmp_scores.csv")

        pd.DataFrame({"ref": refs, "hyp": hyps}).to_csv(tmp_in, index=False)

        green_script = f"""
import pandas as pd
from green_score import GREEN

df      = pd.read_csv({repr(tmp_in)})
scorer  = GREEN({repr(model_name)}, output_dir={repr(output_dir)})
mean, std, scores, summary, _ = scorer(df["ref"].tolist(), df["hyp"].tolist())
pd.DataFrame({{"green": scores}}).to_csv({repr(tmp_out)}, index=False)
print(f"Mean GREEN: {{mean:.4f}} +/- {{std:.4f}}")
print(summary)
"""
        result = subprocess.run(
            [green_python, "-c", green_script],
            capture_output=False,   # let stdout/stderr stream live
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"GREEN subprocess failed (exit {result.returncode}). "
                f"Check output above. Try: CUDA_LAUNCH_BLOCKING=1 for more detail."
            )

        scores_df  = pd.read_csv(tmp_out)
        df["green"] = scores_df["green"].values

        # Clean up temp files
        for f_ in (tmp_in, tmp_out):
            try: os.remove(f_)
            except OSError: pass

    else:
        # ── Direct import mode ─────────────────────────────────────────────
        if not HAS_GREEN:
            raise ImportError("green_score not importable and --green-python not set.")
        green_scorer = GREEN(model_name, output_dir=output_dir)
        mean, std, green_score_list, summary, _ = green_scorer(refs, hyps)
        print(f"  Mean GREEN: {mean:.4f} +/- {std:.4f}")
        print(f"  Summary:\n{summary}")
        df["green"] = list(green_score_list)

    df.to_csv(cache_path, index=False)
    print(f"Saved GREEN scores -> {cache_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# GEMA SCORING
# ─────────────────────────────────────────────────────────────────────────────

GEMA_JSON_SCHEMA = r"""{"entity_name false_prediction": <int>, "entity_name false_prediction_explanation": <string>, "entity_name omission": <int>, "entity_name omission_explanation": <string>, "location false_prediction": <int>, "location false_prediction_explanation": <string>, "location omission": <int>, "location omission_explanation": <string>, "severity false_prediction": <int>, "severity false_prediction_explanation": <string>, "severity omission": <int>, "severity omission_explanation": <string>, "uncertainty false_prediction": <int>, "uncertainty false_prediction_explanation": <string>, "uncertainty omission": <int>, "uncertainty omission_explanation": <string>, "completeness_score": <float>, "completeness_reason": <string>, "readability_score": <float>, "readability_reason": <string>, "clinical_utility_score": <float>, "clinical_utility_reason": <string>, "weighted_final_score": <float>}"""


def _build_gema_prompt(candidate: str, reference: str) -> str:
    return (
        f"user Evaluate the accuracy of a candidate radiology report in comparison to a "
        f"reference radiology report composed by expert radiologists. You should determine "
        f"the following aspects and return the result as a **stringified JSON object** in "
        f"exactly this format (with escaped double quotes):\n"
        f"{GEMA_JSON_SCHEMA}\n\n"
        f"Only output the stringified JSON object. Do not add explanations, markdown, or formatting.\n\n"
        f"Candidate radiology report:\n{candidate}\n\n"
        f"Reference radiology report:\n{reference}\n"
    )


def _extract_json_between_keys(text: str,
                                start_key: str = '"entity_name false_prediction"',
                                end_key:   str = '"weighted_final_score"') -> dict:
    """
    Extract JSON object from model output using brace matching.
    Locates start_key, finds the enclosing '{', then walks forward matching
    braces until the object is closed.
    """
    start_key_idx = text.find(start_key)
    if start_key_idx == -1:
        raise ValueError(f"Start key '{start_key}' not found in output.")

    start_brace_idx = text.rfind('{', 0, start_key_idx)
    if start_brace_idx == -1:
        raise ValueError("No '{' found before start key.")

    stack = []
    for i in range(start_brace_idx, len(text)):
        if text[i] == '{':
            stack.append('{')
        elif text[i] == '}':
            if stack:
                stack.pop()
            if not stack:
                return json.loads(text[start_brace_idx:i + 1])

    raise ValueError("No complete JSON object found (unmatched braces).")


def _parse_gema_score(output_text: str) -> float | None:
    """
    Extract weighted_final_score from GEMA model output.
    Primary: brace-matched JSON extraction.
    Fallback: regex for weighted_final_score value.
    """
    # Primary: structured JSON extraction
    try:
        result = _extract_json_between_keys(output_text)
        score  = result.get("weighted_final_score")
        if score is not None:
            return float(score)
    except (ValueError, json.JSONDecodeError, KeyError):
        pass

    # Fallback: regex
    m = re.search(r'"weighted_final_score"\s*:\s*([\d.]+)', output_text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    return None


def load_gema_model(repo_id: str, subfolder: str):
    """Load GEMA model and processor.

    Requires transformers >= 4.43 for LLaMA3 rope_scaling support.
    Run from your main conda env (discern), NOT the green_score venv.
    """
    import transformers as _tr
    from packaging import version as _v
    if _v.parse(_tr.__version__) < _v.parse("4.43.0"):
        raise RuntimeError(
            f"transformers {_tr.__version__} is too old for GEMA (LLaMA3 rope_scaling). "
            f"Need >= 4.43.0. Switch to the discern conda env: conda activate discern"
        )

    print(f"Loading GEMA model: {repo_id}/{subfolder} ...")
    print(f"  transformers: {_tr.__version__}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        subfolder=subfolder,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        repo_id,
        subfolder=subfolder,
        trust_remote_code=True,
    )
    return model, processor


def _score_one_gema(model, processor, candidate: str, reference: str,
                    max_new_tokens: int = 2048) -> tuple[float | None, str]:
    """Run GEMA on one pair. Returns (score, raw_output_text)."""
    prompt  = _build_gema_prompt(candidate, reference)
    inputs  = processor(text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.2,
        )
    output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    score       = _parse_gema_score(output_text)
    return score, output_text


def compute_gema_scores(df_reports: pd.DataFrame,
                        ref_col: str,
                        hyp_col: str,
                        repo_id: str,
                        subfolder: str,
                        output_dir: str,
                        scores_cache: str | None = None,
                        max_new_tokens: int = 2048) -> pd.DataFrame:
    """
    Run GEMA on all report pairs with row-by-row saving for resume support.
    Saves both the numeric score and the raw JSON eval_result per row.
    """
    cache_path = scores_cache or os.path.join(output_dir, "gema_scores.csv")

    # Load cache and resume if possible
    if os.path.exists(cache_path):
        print(f"Loading cached GEMA scores from {cache_path} ...")
        df = pd.read_csv(cache_path)
        if "gema" in df.columns and len(df) == len(df_reports):
            n_done = df["gema"].notna().sum()
            if n_done == len(df):
                print(f"  All {n_done} rows already scored.")
                df_out = df_reports.copy()
                df_out["gema"] = df["gema"].values
                return df_out
            print(f"  Resuming: {n_done}/{len(df)} rows already scored.")
            # Merge existing scores back into a fresh copy
            df_work = df_reports.copy()
            df_work["gema"]        = df["gema"].values
            df_work["eval_result"] = df.get("eval_result", pd.Series([None]*len(df))).values
        else:
            print("  Cache mismatch -- recomputing from scratch ...")
            df_work = df_reports.copy()
            df_work["gema"]        = None
            df_work["eval_result"] = None
    else:
        df_work = df_reports.copy()
        df_work["gema"]        = None
        df_work["eval_result"] = None

    model, processor = load_gema_model(repo_id, subfolder)

    candidates = df_work[hyp_col].fillna("").tolist()
    references = df_work[ref_col].fillna("").tolist()
    n          = len(df_work)
    n_failed   = 0

    print(f"Running GEMA on {n} report pairs ...")

    for i in tqdm(range(n), desc="GEMA"):
        # Skip already-scored rows (resume support)
        if pd.notna(df_work.loc[i, "gema"]):
            continue

        cand = str(candidates[i]).strip()
        ref  = str(references[i]).strip()

        if not cand or not ref:
            df_work.loc[i, "gema"]        = float("nan")
            df_work.loc[i, "eval_result"] = json.dumps({"error": "empty input"})
            continue

        output_text = "N/A"
        try:
            score, output_text = _score_one_gema(
                model, processor, cand, ref, max_new_tokens
            )
            if score is None:
                raise ValueError("weighted_final_score not found in output")
            df_work.loc[i, "gema"]        = score
            df_work.loc[i, "eval_result"] = output_text[:2000]  # truncate for CSV safety
        except Exception as e:
            n_failed += 1
            df_work.loc[i, "gema"]        = float("nan")
            df_work.loc[i, "eval_result"] = json.dumps({
                "error": str(e),
                "raw_output": output_text[:500],
            })

        # Save after every row for resume support
        df_work.to_csv(cache_path, index=False)

    valid = df_work["gema"].dropna()
    print(f"\n  Done. {n_failed}/{n} pairs failed. Mean GEMA: {valid.mean():.4f} ± {valid.std():.4f}")
    print(f"Saved GEMA scores -> {cache_path}")

    df_out = df_reports.copy()
    df_out["gema"] = df_work["gema"].values
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION (shared for both scorers)
# ─────────────────────────────────────────────────────────────────────────────

def correlate_score_vs_rater(df_scores: pd.DataFrame,
                              rater_table: pd.DataFrame,
                              score_col: str,
                              score_label: str,
                              join_keys: List[str],
                              target_col: str) -> dict | None:
    """
    Join df_scores with rater_table on join_keys, correlate score_col vs target_col.
    Correlations are NEGATED (higher score = better = fewer errors).
    """
    merged = df_scores.merge(
        rater_table[join_keys + [target_col]],
        on=join_keys, how="inner",
    )

    if merged.empty:
        print(f"  ⚠ Merge empty for '{target_col}'")
        print(f"    Score  study_ids (sample): {sorted(df_scores['study_id'].unique()[:5])}")
        print(f"    Rater  study_ids (sample): {sorted(rater_table['study_id'].unique()[:5])}")
        return None

    valid = merged[[score_col, target_col]].dropna()
    if len(valid) < 3:
        print(f"  ⚠ Fewer than 3 valid rows for '{target_col}' — skipping.")
        return None

    print(f"  [{score_label}] Merged {len(valid)} rows for '{target_col}'")

    rho, p_spearman = spearmanr(valid[score_col], valid[target_col])
    tau, p_tau      = kendalltau(valid[score_col], valid[target_col])

    return {
        "metric":       score_col,
        "metric_label": score_label,
        "target":       target_col,
        "spearman":     -rho,
        "p_spearman":   p_spearman,
        "tau":          -tau,
        "p_tau":        p_tau,
        "n_samples":    len(valid),
    }


def run_correlations_for_scorer(df_scores: pd.DataFrame,
                                 rater_table: pd.DataFrame,
                                 score_col: str,
                                 score_label: str,
                                 join_keys: List[str],
                                 output_dir: str,
                                 label: str) -> pd.DataFrame:
    """Correlate one scorer vs both targets, save CSVs."""
    print(f"\nCorrelating {score_label} vs {label} rater ...")

    rows = []
    for target in ["total_significant", "total_insignificant"]:
        result = correlate_score_vs_rater(
            df_scores, rater_table, score_col, score_label, join_keys, target
        )
        if result:
            rows.append(result)

    if not rows:
        print("  No correlation results produced.")
        return pd.DataFrame()

    corr_df   = pd.DataFrame(rows)
    sig_row   = corr_df[corr_df["target"] == "total_significant"].iloc[0]
    insig_row = corr_df[corr_df["target"] == "total_insignificant"].iloc[0]

    combined = pd.DataFrame([{
        "metric":           score_col,
        "metric_label":     score_label,
        "sig_tau":          sig_row["tau"],
        "sig_p_tau":        sig_row["p_tau"],
        "sig_spearman":     sig_row["spearman"],
        "sig_p_spearman":   sig_row["p_spearman"],
        "n_samples_sig":    sig_row["n_samples"],
        "insig_tau":        insig_row["tau"],
        "insig_p_tau":      insig_row["p_tau"],
        "insig_spearman":   insig_row["spearman"],
        "insig_p_spearman": insig_row["p_spearman"],
        "n_samples_insig":  insig_row["n_samples"],
    }])

    prefix = f"{score_col}_corr_{label}"
    corr_df.to_csv(os.path.join(output_dir, f"{prefix}.csv"), index=False)
    combined.to_csv(os.path.join(output_dir, f"{prefix}_combined.csv"), index=False)
    print(f"Saved -> {prefix}.csv, {prefix}_combined.csv")

    print(f"\n-- {score_label} vs {label} --")
    print(combined[["metric_label",
                    "insig_tau", "insig_spearman",
                    "sig_tau",   "sig_spearman"]].to_string(index=False))

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# PER-BENCHMARK RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_rexval(args):
    print("\n" + "="*60)
    print("BENCHMARK: ReXVal")
    print("="*60)

    os.makedirs(args.output_dir_rexval, exist_ok=True)

    df_reports = pd.read_csv(args.reports_rexval)
    df_studies = pd.read_csv(args.studies)
    df_rexval  = pd.read_csv(args.rexval)

    required = {"study_id", "candidate_reporter", "gt_report", "candidate_report"}
    missing  = required - set(df_reports.columns)
    if missing:
        raise ValueError(f"ReXVal reports CSV missing columns: {missing}")

    print("Building ReXVal avg rater table ...")
    rater_table = build_rexval_rater_table(
        df_rexval, build_study_id_map(df_studies)
    )
    rater_join  = rater_table.rename(columns={"candidate_type": "candidate_reporter"})
    join_keys   = ["study_id", "candidate_reporter"]

    # ── GREEN ─────────────────────────────────────────────────────────────────
    if args.scorer in ("green", "both"):
        if not HAS_GREEN:
            print("⚠ Skipping GREEN — green_score package not available.")
        else:
            df_green = compute_green_scores(
                df_reports, ref_col="gt_report", hyp_col="candidate_report",
                model_name=args.green_model,
                output_dir=args.output_dir_rexval,
                scores_cache=args.scores_cache_rexval,
                green_python=args.green_python or None,
            )
            run_correlations_for_scorer(
                df_green, rater_join, "green", "GREEN",
                join_keys, args.output_dir_rexval, "rexval",
            )

    # ── GEMA ──────────────────────────────────────────────────────────────────
    if args.scorer in ("gema", "both"):
        gema_cache = (args.scores_cache_rexval or "").replace("green", "gema") or None
        df_gema = compute_gema_scores(
            df_reports, ref_col="gt_report", hyp_col="candidate_report",
            repo_id=args.gema_repo,
            subfolder=args.gema_subfolder,
            output_dir=args.output_dir_rexval,
            scores_cache=gema_cache,
            max_new_tokens=args.gema_max_tokens,
        )
        run_correlations_for_scorer(
            df_gema, rater_join, "gema", "GEMA",
            join_keys, args.output_dir_rexval, "rexval",
        )


def run_radevalx(args):
    print("\n" + "="*60)
    print("BENCHMARK: RadEvalX")
    print("="*60)

    os.makedirs(args.output_dir_radevalx, exist_ok=True)

    df_reports  = pd.read_csv(args.reports_radevalx)
    df_radevalx = pd.read_csv(args.radevalx)

    required = {"study_id", "ground_truth", "candidate_report"}
    missing  = required - set(df_reports.columns)
    if missing:
        raise ValueError(f"RadEvalX reports CSV missing columns: {missing}")

    print("Building RadEvalX rater table ...")
    rater_table = build_radevalx_rater_table(df_radevalx)
    print(f"  Rater table: {len(rater_table)} studies")
    join_keys = ["study_id"]

    # ── GREEN ─────────────────────────────────────────────────────────────────
    if args.scorer in ("green", "both"):
        if not HAS_GREEN:
            print("⚠ Skipping GREEN — green_score package not available.")
        else:
            df_green = compute_green_scores(
                df_reports, ref_col="ground_truth", hyp_col="candidate_report",
                model_name=args.green_model,
                output_dir=args.output_dir_radevalx,
                scores_cache=args.scores_cache_radevalx,
                green_python=args.green_python or None,
            )
            run_correlations_for_scorer(
                df_green, rater_table, "green", "GREEN",
                join_keys, args.output_dir_radevalx, "radevalx",
            )

    # ── GEMA ──────────────────────────────────────────────────────────────────
    if args.scorer in ("gema", "both"):
        gema_cache = (args.scores_cache_radevalx or "").replace("green", "gema") or None
        df_gema = compute_gema_scores(
            df_reports, ref_col="ground_truth", hyp_col="candidate_report",
            repo_id=args.gema_repo,
            subfolder=args.gema_subfolder,
            output_dir=args.output_dir_radevalx,
            scores_cache=gema_cache,
            max_new_tokens=args.gema_max_tokens,
        )
        run_correlations_for_scorer(
            df_gema, rater_table, "gema", "GEMA",
            join_keys, args.output_dir_radevalx, "radevalx",
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE = ""

    parser = argparse.ArgumentParser(
        description="Compute GREEN/GEMA scores and correlate with ReXVal/RadEvalX raters."
    )
    parser.add_argument("--mode",   choices=["rexval", "radevalx", "both"], default="both")
    parser.add_argument("--scorer", choices=["green", "gema", "both"],      default="gema")

    # ── GREEN ─────────────────────────────────────────────────────────────────
    parser.add_argument("--green-model", default="StanfordAIMI/GREEN-radllama2-7b")

    # ── GEMA ──────────────────────────────────────────────────────────────────
    parser.add_argument("--green-python", default=None,
                        help="Path to python binary in the green_score venv. "
                             "Use this to avoid CUDA arch mismatches. e.g.: "
                             "/vast/.../green_score/bin/python")
    parser.add_argument("--gema-repo",      default="Gemascore/GEMA-Score-distilled")
    parser.add_argument("--gema-subfolder", default="GEMA-Score-distilled-Xray-llama",
                        help="Subfolder within GEMA repo. Options:\n"
                             "  GEMA-Score-distilled-CT-llama\n"
                             "  GEMA-Score-distilled-CT-Qwen\n"
                             "  GEMA-Score-distilled-Xray-llama\n"
                             "  GEMA-Score-distilled-Xray-Qwen")
    parser.add_argument("--gema-max-tokens", type=int, default=256,
                        help="max_new_tokens for GEMA generation (default: 256)")

    # ── ReXVal paths ──────────────────────────────────────────────────────────
    parser.add_argument("--reports-rexval")
    parser.add_argument("--studies")
    parser.add_argument("--rexval")
    parser.add_argument("--output-dir-rexval")
    parser.add_argument("--scores-cache-rexval", default=None,
                        help="Cached green_scores.csv or gema_scores.csv for ReXVal")

    # ── RadEvalX paths ────────────────────────────────────────────────────────
    parser.add_argument("--reports-radevalx")
    parser.add_argument("--radevalx")
    parser.add_argument("--output-dir-radevalx")
    parser.add_argument("--scores-cache-radevalx")

    args = parser.parse_args()

    if args.mode in ("rexval", "both"):
        run_rexval(args)

    if args.mode in ("radevalx", "both"):
        run_radevalx(args)