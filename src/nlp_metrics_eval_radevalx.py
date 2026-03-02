"""
NLP Metrics vs RadEvalX Rater Correlation
==========================================
Computes BLEU-1, ROUGE-L, METEOR, BERTScore, RadGraphF1 on candidate/reference
report pairs, then correlates each metric against RadEvalX average rater error
counts (total_significant, total_insignificant).

Input CSVs
----------
  reports CSV : study_id, ground_truth, candidate_report
  radevalx CSV: radeval_total.csv  (report_id, error_type, one..eight, ...)

Correlations are NEGATED before saving: higher NLP score = better report =
fewer errors, so raw correlation with error count is negative. Negating
converts to "correlation with quality", matching published sign convention.

Output
------
  nlp_metrics_scores.csv             -- per-report raw metric scores
  nlp_metrics_corr_significant.csv   -- correlations vs total_significant
  nlp_metrics_corr_insignificant.csv -- correlations vs total_insignificant
  nlp_metrics_corr_combined.csv      -- both combined (for scatter panel)

Usage
-----
  python compute_nlp_metrics_radevalx.py
  python compute_nlp_metrics_radevalx.py --reports my_reports.csv --output-dir results/
  python compute_nlp_metrics_radevalx.py --scores-cache nlp_metrics_scores.csv
"""

from __future__ import annotations

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from typing import List

warnings.filterwarnings("ignore")

# ── Metric imports ────────────────────────────────────────────────────────────
from nltk.translate.bleu_score   import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge                       import Rouge
from bert_score                  import score as bertscore_score

try:
    from bleurt import score as bluert_score
    HAS_BLUERT = True
except ImportError:
    print("⚠ BlueRT not found — will skip.")
    HAS_BLUERT = False

try:
    from radgraph import F1RadGraph
    HAS_RADGRAPH = True
except ImportError:
    print("⚠ RadGraphF1 not found — will skip.")
    HAS_RADGRAPH = False

try:
    from RaTEScore import RaTEScore as RaTEScorer
    HAS_RATESCORE = True
except ImportError:
    print("⚠ RaTEScore not found — will skip.")
    HAS_RATESCORE = False


# ─────────────────────────────────────────────────────────────────────────────
# RADEVALX RATER TABLE  (reused from radevalx comparison script)
# ─────────────────────────────────────────────────────────────────────────────

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


def build_radevalx_rater_table(df_radevalx: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot radeval_total.csv into one row per study_id with columns for each
    error category × significance, plus total_significant / total_insignificant.
    """
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
        suffix = row["error_type"]          # "significant" or "insignificant"
        record = {"study_id": row["report_id"]}
        for src_col, prefix in CATEGORY_COLS.items():
            record[f"{prefix}_{suffix}"] = row[src_col]
        record[f"partial_{suffix}"] = row["partial"]
        rows.append(record)

    long_df    = pd.DataFrame(rows)
    rater_wide = (
        long_df
        .groupby("study_id")[OUR_COLUMNS]
        .sum()
        .reset_index()
    )
    for col in OUR_COLUMNS:
        if col not in rater_wide.columns:
            rater_wide[col] = 0

    rater_wide["total_significant"]   = rater_wide[SIGNIFICANT_COLUMNS].sum(axis=1)
    rater_wide["total_insignificant"] = rater_wide[INSIGNIFICANT_COLUMNS].sum(axis=1)
    return rater_wide


# ─────────────────────────────────────────────────────────────────────────────
# RADGRAPH TOKENIZER COMPAT PATCH
# ─────────────────────────────────────────────────────────────────────────────

def _patch_radgraph_tokenizer_compat():
    """
    Restore tokenizer methods removed in newer transformers versions that
    radgraph's bundled allennlp still calls.
    """
    from transformers import (
        PreTrainedTokenizerBase, PreTrainedTokenizerFast,
        BertTokenizer, BertTokenizerFast,
    )

    def _encode_plus_compat(self, *args, **kwargs):
        return dict(self(*args, **kwargs))

    def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        cls, sep = [self.cls_token_id], [self.sep_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def _create_token_type_ids(self, token_ids_0, token_ids_1=None):
        sep, cls = [self.sep_token_id], [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    patches = {
        "encode_plus":                          _encode_plus_compat,
        "build_inputs_with_special_tokens":     _build_inputs_with_special_tokens,
        "create_token_type_ids_from_sequences": _create_token_type_ids,
    }
    for cls in (PreTrainedTokenizerBase, PreTrainedTokenizerFast,
                BertTokenizer, BertTokenizerFast):
        for name, fn in patches.items():
            if not hasattr(cls, name):
                setattr(cls, name, fn)


# ─────────────────────────────────────────────────────────────────────────────
# NLP METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu1(candidate: str, reference: str) -> float:
    smoothie = SmoothingFunction().method1
    return sentence_bleu([reference.split()], candidate.split(),
                         weights=(1, 0, 0, 0),
                         smoothing_function=smoothie)


def compute_rougel(candidate: str, reference: str) -> float:
    rouge = Rouge()
    return rouge.get_scores(reference, candidate)[0]["rouge-l"]["f"]


def compute_meteor(candidate: str, reference: str) -> float:
    return meteor_score([reference.split()], candidate.split())


def compute_bertscore_batch(candidates: List[str],
                            references: List[str],
                            device: str = "cuda",
                            batch_size: int = 32) -> List[float]:
    """BERTScore — batched for efficiency. Returns F1 scores."""
    _,_,F = bertscore_score(
        candidates, references,
        lang="en",
        verbose=False,
    )

    return F.tolist()


def compute_bluert_batch(candidates: List[str],
                         references: List[str]) -> List[float]:
    results = bluert_score(references, candidates)
    if isinstance(results, dict):
        return results.get("scores", results.get("score", []))
    return list(results)


def compute_radgraphf1_batch(candidates: List[str],
                             references: List[str]) -> List[float]:
    """RadGraphF1 — batched."""
    _patch_radgraph_tokenizer_compat()
    f1_radgraph = F1RadGraph(reward_level="partial", model = "radgraph-xl")
    _, reward_list, _, _ = f1_radgraph(hyps=candidates, refs=references)
    return reward_list



def compute_ratescore_batch(candidates: List[str],
                            references: List[str]) -> List[float]:
    scorer  = RaTEScorer()
    results = scorer.score(candidates, references)
    if isinstance(results, dict):
        return results.get("scores", list(results.values()))
    return list(results)


def compute_all_metrics(df_reports: pd.DataFrame,
                        device: str = "cuda") -> pd.DataFrame:
    """
    Compute all NLP metrics for each row.
    Expected columns: study_id, ground_truth, candidate_report
    """
    df         = df_reports.copy()
    candidates = df["candidate_report"].fillna("").tolist()
    references = df["ground_truth"].fillna("").tolist()

    print(f"Computing metrics for {len(df)} report pairs ...")

    print("  BLEU-1 ...")
    df["bleu1"] = [compute_bleu1(c, r) for c, r in zip(candidates, references)]

    print("  ROUGE-L ...")
    df["rougeL"] = [compute_rougel(c, r) for c, r in zip(candidates, references)]

    print("  METEOR ...")
    df["meteor"] = [compute_meteor(c, r) for c, r in zip(candidates, references)]

    print("  BERTScore ...")
    df["bertscore"] = compute_bertscore_batch(candidates, references, device=device)

    # if HAS_BLUERT:
    #     print("  BlueRT ...")
    #     df["bluert"] = compute_bluert_batch(candidates, references)
    # else:
    #     df["bluert"] = np.nan

    if HAS_RADGRAPH:
        print("  RadGraphF1 ...")
        df["radgraphf1"] = compute_radgraphf1_batch(candidates, references)
    else:
        df["radgraphf1"] = np.nan

    # if HAS_RATESCORE:
    #     print("  RaTEScore ...")
    #     df["ratescore"] = compute_ratescore_batch(candidates, references)
    # else:
    #     df["ratescore"] = np.nan

    print("  Done.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

METRIC_COLS = ["bleu1", "rougeL", "meteor", "bertscore",
               "bluert", "radgraphf1", "ratescore"]

METRIC_DISPLAY = {
    "bleu1":      "BLEU-1",
    "rougeL":     "ROUGE-L",
    "meteor":     "METEOR",
    "bertscore":  "BertScore",
    "bluert":     "BLUERT",
    "radgraphf1": "RadGraphF1",
    "ratescore":  "RaTEScore",
}


def correlate_metrics_vs_rater(df_scores: pd.DataFrame,
                                rater_table: pd.DataFrame,
                                target_col: str) -> pd.DataFrame:
    """
    Join NLP scores with rater table on study_id (one candidate per study),
    compute Spearman + Kendall-tau for each metric vs target_col.

    Correlations are NEGATED: higher NLP score = better report = fewer errors.
    Negating gives "correlation with quality" matching published sign convention.
    p-values are unchanged (symmetric).
    """
    merged = df_scores.merge(rater_table[["study_id", target_col]],
                             on="study_id", how="inner")

    if merged.empty:
        print(f"  ⚠ Merge empty for '{target_col}' — check study_id alignment.")
        print(f"    Score  study_ids (sample): {sorted(df_scores['study_id'].unique()[:5])}")
        print(f"    Rater  study_ids (sample): {sorted(rater_table['study_id'].unique()[:5])}")
        return pd.DataFrame()

    print(f"  Merged {len(merged)} rows for '{target_col}'")

    results = []
    for col in METRIC_COLS:
        if col not in merged.columns:
            continue
        valid = merged[[col, target_col]].dropna()
        if len(valid) < 3:
            print(f"    ⚠ Skipping {col} — fewer than 3 valid rows.")
            continue

        rho, p_spearman = spearmanr(valid[col], valid[target_col])
        tau, p_tau      = kendalltau(valid[col], valid[target_col])

        results.append({
            "metric":       col,
            "metric_label": METRIC_DISPLAY[col],
            "target":       target_col,
            "spearman":     -rho,       # negated: quality correlation
            "p_spearman":   p_spearman,
            "tau":          -tau,       # negated: quality correlation
            "p_tau":        p_tau,
            "n_samples":    len(valid),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(
    reports_path:  str,
    radevalx_path: str,
    output_dir:    str,
    device:        str = "cuda",
    scores_cache:  str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading inputs ...")
    df_reports  = pd.read_csv(reports_path)
    df_radevalx = pd.read_csv(radevalx_path)

    required = {"study_id", "ground_truth", "candidate_report"}
    missing  = required - set(df_reports.columns)
    if missing:
        raise ValueError(f"reports CSV missing columns: {missing}")

    print("Building RadEvalX rater table ...")
    rater_table = build_radevalx_rater_table(df_radevalx)
    print(f"  Rater table: {len(rater_table)} studies")

    if scores_cache and os.path.exists(scores_cache):
        print(f"Loading cached scores from {scores_cache} ...")
        df_scores = pd.read_csv(scores_cache)
    else:
        df_scores  = compute_all_metrics(df_reports, device=device)
        scores_out = os.path.join(output_dir, "nlp_metrics_scores.csv")
        df_scores.to_csv(scores_out, index=False)
        print(f"Saved per-report scores -> {scores_out}")

    print("\nCorrelating metrics vs avg rater ...")
    corr_sig   = correlate_metrics_vs_rater(df_scores, rater_table, "total_significant")
    corr_insig = correlate_metrics_vs_rater(df_scores, rater_table, "total_insignificant")

    if not corr_sig.empty:
        p = os.path.join(output_dir, "nlp_metrics_corr_significant.csv")
        corr_sig.to_csv(p, index=False)
        print(f"Saved -> {p}")

    if not corr_insig.empty:
        p = os.path.join(output_dir, "nlp_metrics_corr_insignificant.csv")
        corr_insig.to_csv(p, index=False)
        print(f"Saved -> {p}")

    if not corr_sig.empty and not corr_insig.empty:
        sig_   = corr_sig[["metric", "metric_label",
                            "tau", "p_tau", "spearman", "p_spearman", "n_samples"]].rename(
            columns={"tau": "sig_tau", "p_tau": "sig_p_tau",
                     "spearman": "sig_spearman", "p_spearman": "sig_p_spearman",
                     "n_samples": "n_samples_sig"})
        insig_ = corr_insig[["metric",
                              "tau", "p_tau", "spearman", "p_spearman", "n_samples"]].rename(
            columns={"tau": "insig_tau", "p_tau": "insig_p_tau",
                     "spearman": "insig_spearman", "p_spearman": "insig_p_spearman",
                     "n_samples": "n_samples_insig"})
        combined = sig_.merge(insig_, on="metric")
        p = os.path.join(output_dir, "nlp_metrics_corr_combined.csv")
        combined.to_csv(p, index=False)
        print(f"Saved combined -> {p}")

        print("\n-- Combined Results (negated: higher = better quality correlation) --")
        print(combined[[
            "metric_label",
            "insig_tau", "insig_spearman",
            "sig_tau",   "sig_spearman",
        ]].to_string(index=False))

    return df_scores, corr_sig, corr_insig


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute NLP metrics on report pairs and correlate with RadEvalX raters."
    )
    parser.add_argument(
        "--reports",
        help="CSV with study_id, ground_truth, candidate_report",
    )
    parser.add_argument(
        "--radevalx",
        default="../data/radevalx/radeval_total.csv",
        help="RadEvalX labelled data (radeval_total.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="../data/radevalx/nlp_metrics",
        help="Directory to write output CSVs",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for BERTScore (default: cuda)",
    )
    parser.add_argument(
        "--scores-cache", default=None,
        help="Path to previously saved nlp_metrics_scores.csv to skip recomputation",
    )
    args = parser.parse_args()

    run(
        reports_path=args.reports,
        radevalx_path=args.radevalx,
        output_dir=args.output_dir,
        device=args.device,
        scores_cache=args.scores_cache,
    )