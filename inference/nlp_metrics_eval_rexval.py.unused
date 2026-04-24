"""
NLP Metrics vs ReXVal Rater Correlation
========================================
Computes BLEU-1, ROUGE-L, METEOR, BERTScore, BLUERT, RadGraphF1, RaTEScore
on candidate/reference report pairs, then correlates each metric against
ReXVal average rater error counts (total_significant, total_insignificant).

Output
------
  nlp_metrics_scores.csv          — per-report scores for all metrics
  nlp_metrics_corr_significant.csv   — Spearman + Kendall tau vs total_significant
  nlp_metrics_corr_insignificant.csv — Spearman + Kendall tau vs total_insignificant
  nlp_metrics_corr_combined.csv      — both in one table (for scatter panel)

Usage
-----
  python compute_nlp_metrics_rexval.py
  python compute_nlp_metrics_rexval.py --reports my_reports.csv --output-dir results/
"""

from __future__ import annotations

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from typing import Dict, List

warnings.filterwarnings("ignore")

# ── Update import paths for your cluster if needed ───────────────────────────
from nltk.translate.bleu_score  import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge                import Rouge
from bert_score                 import score as bertscore_score
# BlueRT — update import if package name differs on your cluster
try:
    from bleurt import score as bluert_score
    HAS_BLUERT = True
except ImportError:
    print("⚠ BlueRT not found — will skip. Install with: pip install bluert")
    HAS_BLUERT = False

# RadGraphF1 — update import if needed
try:
    from radgraph import F1RadGraph
    HAS_RADGRAPH = True
except ImportError:
    print("⚠ RadGraphF1 not found — will skip. Install radgraph package.")
    HAS_RADGRAPH = False

# RaTEScore — update import if needed
try:
    from RaTEScore import RaTEScore
    HAS_RATESCORE = True
except ImportError:
    print("⚠ RaTEScore not found — will skip. Install ratescore package.")
    HAS_RATESCORE = False

# ─────────────────────────────────────────────────────────────────────────────
# REXVAL BUILDER  (reused from rexval comparison script)
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

OUR_COLUMNS = [
    "false_prediction_insignificant", "false_prediction_significant",
    "omission_finding_insignificant",  "omission_finding_significant",
    "incorrect_location_insignificant","incorrect_location_significant",
    "incorrect_severity_insignificant","incorrect_severity_significant",
    "extra_comparison_insignificant",  "extra_comparison_significant",
    "omitted_comparison_insignificant","omitted_comparison_significant",
]
SIGNIFICANT_COLUMNS   = [c for c in OUR_COLUMNS if c.endswith("_significant")]
INSIGNIFICANT_COLUMNS = [c for c in OUR_COLUMNS if c.endswith("_insignificant")]


def build_study_id_map(df_studies: pd.DataFrame) -> Dict[int, str]:
    return {idx: sid for idx, sid in enumerate(df_studies["study_id"])}


def build_rater_table(df_rexval: pd.DataFrame,
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
    for col in OUR_COLUMNS:
        if col not in rater_table.columns:
            rater_table[col] = 0
    rater_table = rater_table[["study_id", "candidate_type", "rater_index"] + OUR_COLUMNS]

    # Add summary columns
    rater_table["total_significant"]   = rater_table[SIGNIFICANT_COLUMNS].sum(axis=1)
    rater_table["total_insignificant"] = rater_table[INSIGNIFICANT_COLUMNS].sum(axis=1)
    return rater_table


def build_avg_rater_table(rater_table: pd.DataFrame) -> pd.DataFrame:
    summary_cols = OUR_COLUMNS + ["total_significant", "total_insignificant"]
    return (
        rater_table
        .groupby(["study_id", "candidate_type"])[summary_cols]
        .mean()
        .reset_index()
    )

def _patch_radgraph_tokenizer_compat():
    """
    Patch transformers tokenizers to restore methods removed in newer versions
    but still called by radgraph's bundled allennlp internals:

        - encode_plus      : removed ~transformers 4.40
        - build_inputs_with_special_tokens : removed from fast tokenizer path
        - create_token_type_ids_from_sequences : similar removal

    All patches delegate to the modern __call__ / slow-tokenizer equivalents.
    """
    from transformers import (
        PreTrainedTokenizerBase,
        PreTrainedTokenizerFast,
        BertTokenizer,
        BertTokenizerFast,
    )

    # ── encode_plus ───────────────────────────────────────────────────────────
    def _encode_plus_compat(self, *args, **kwargs):
        result = self(*args, **kwargs)
        return dict(result)

    # ── build_inputs_with_special_tokens ──────────────────────────────────────
    # Original signature: (token_ids_0, token_ids_1=None) -> List[int]
    # BERT version: [CLS] + token_ids_0 + [SEP] (+ token_ids_1 + [SEP])
    def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # ── create_token_type_ids_from_sequences ──────────────────────────────────
    # BERT: 0s for first sequence, 1s for second sequence
    def _create_token_type_ids(self, token_ids_0, token_ids_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    patches = {
        "encode_plus":                         _encode_plus_compat,
        "build_inputs_with_special_tokens":    _build_inputs_with_special_tokens,
        "create_token_type_ids_from_sequences":_create_token_type_ids,
    }

    for cls in (PreTrainedTokenizerBase, PreTrainedTokenizerFast,
                BertTokenizer, BertTokenizerFast):
        for method_name, method in patches.items():
            if not hasattr(cls, method_name):
                setattr(cls, method_name, method)


# ─────────────────────────────────────────────────────────────────────────────
# NLP METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu1(candidate: str, reference: str) -> float:
    ref_tokens  = reference.split()
    cand_tokens = candidate.split()
    smoothie    = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], cand_tokens,
                         weights=(1, 0, 0, 0),
                         smoothing_function=smoothie)


def compute_rougel(candidate: str, reference: str) -> float:
    rouge = Rouge()
    # scorer = rouge.get_scores.RougeScorer(["rougeL"], use_stemmer=True)
    return rouge.get_scores(reference, candidate)[0]['rouge-l']['f']


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
    """BlueRT — batched."""
    results = bluert_score(references, candidates)
    # bluert returns a dict with 'scores' key
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
    """RaTEScore — batched."""
    scorer  = RaTEScorer()
    results = scorer.score(candidates, references)
    if isinstance(results, dict):
        return results.get("scores", list(results.values()))
    return list(results)


def compute_all_metrics(df_reports: pd.DataFrame,
                        device: str = "cuda") -> pd.DataFrame:
    """
    Compute all NLP metrics for each row in df_reports.
    Expected columns: study_id, candidate_reporter, candidate_report, gt_report
    Returns df_reports with one new column per metric.
    """
    df = df_reports.copy()
    candidates = df["candidate_report"].fillna("").tolist()
    references = df["gt_report"].fillna("").tolist()
    n = len(df)

    print(f"Computing metrics for {n} report pairs …")

    # ── Row-by-row metrics ─────────────────────────────────────────────────
    print("  BLEU-1 …")
    df["bleu1"] = [compute_bleu1(c, r)
                   for c, r in zip(candidates, references)]

    print("  ROUGE-L …")
    df["rougeL"] = [compute_rougel(c, r)
                    for c, r in zip(candidates, references)]

    print("  METEOR …")
    df["meteor"] = [compute_meteor(c, r)
                    for c, r in zip(candidates, references)]

    # ── Batched metrics ────────────────────────────────────────────────────
    print("  BERTScore …")
    df["bertscore"] = compute_bertscore_batch(candidates, references, device=device)

    # if HAS_BLUERT:
    #     print("  BlueRT …")
    #     df["bluert"] = compute_bluert_batch(candidates, references)
    # else:
    #     df["bluert"] = np.nan

    if HAS_RADGRAPH:
        print("  RadGraphF1 …")
        df["radgraphf1"] = compute_radgraphf1_batch(candidates, references)
    else:
        df["radgraphf1"] = np.nan

    # if HAS_RATESCORE:
    #     print("  RaTEScore …")
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
                                avg_rater_table: pd.DataFrame,
                                target_col: str) -> pd.DataFrame:
    """
    Merge NLP scores with avg rater table on (study_id, candidate_reporter)
    and compute Spearman + Kendall-tau for each metric vs target_col.

    target_col: 'total_significant' or 'total_insignificant'
    """
    merged = df_scores.merge(
        avg_rater_table,
        left_on=["study_id", "candidate_reporter"],
        right_on=["study_id", "candidate_type"],
        how="inner",
    )

    if merged.empty:
        print(f"  ⚠ Merge empty for target '{target_col}' — "
              f"check study_id / candidate_reporter alignment.")
        print(f"    Score  study_ids:  {sorted(df_scores['study_id'].unique()[:5])}")
        print(f"    Rater  study_ids:  {sorted(avg_rater_table['study_id'].unique()[:5])}")
        print(f"    Score  reporters:  {sorted(df_scores['candidate_reporter'].unique())}")
        print(f"    Rater  cand_types: {sorted(avg_rater_table['candidate_type'].unique())}")
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
            "spearman":     -rho,
            "p_spearman":   p_spearman,
            "tau":          -tau,
            "p_tau":        p_tau,
            "n_samples":    len(valid),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(
    reports_path:   str,
    studies_path:   str,
    rexval_path:    str,
    output_dir:     str,
    device:         str = "cuda",
    scores_cache:   str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("Loading inputs …")
    df_reports = pd.read_csv(reports_path)
    df_studies = pd.read_csv(studies_path)
    df_rexval  = pd.read_csv(rexval_path)

    required = {"study_id", "candidate_reporter", "candidate_report", "gt_report"}
    missing  = required - set(df_reports.columns)
    if missing:
        raise ValueError(f"reports CSV missing columns: {missing}")

    # ── Build rexval avg rater table ──────────────────────────────────────────
    print("Building ReXVal avg rater table …")
    index_to_study_id = build_study_id_map(df_studies)
    rater_table       = build_rater_table(df_rexval, index_to_study_id)
    avg_rater_table   = build_avg_rater_table(rater_table)

    # ── Compute NLP metrics (or load from cache) ───────────────────────────────
    if scores_cache and os.path.exists(scores_cache):
        print(f"Loading cached scores from {scores_cache} …")
        df_scores = pd.read_csv(scores_cache)
    else:
        df_scores = compute_all_metrics(df_reports, device=device)
        scores_out = os.path.join(output_dir, "nlp_metrics_scores.csv")
        df_scores.to_csv(scores_out, index=False)
        print(f"Saved per-report scores → {scores_out}")

    # ── Correlate vs avg rater ────────────────────────────────────────────────
    print("\nCorrelating metrics vs avg rater …")

    corr_sig   = correlate_metrics_vs_rater(df_scores, avg_rater_table, "total_significant")
    corr_insig = correlate_metrics_vs_rater(df_scores, avg_rater_table, "total_insignificant")

    if not corr_sig.empty:
        path = os.path.join(output_dir, "nlp_metrics_corr_significant.csv")
        corr_sig.to_csv(path, index=False)
        print(f"Saved → {path}")

    if not corr_insig.empty:
        path = os.path.join(output_dir, "nlp_metrics_corr_insignificant.csv")
        corr_insig.to_csv(path, index=False)
        print(f"Saved → {path}")

    # ── Combined table in scatter-panel collated format ───────────────────────
    # Format: metric_label, insig_tau, insig_spearman, sig_tau, sig_spearman
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
        path = os.path.join(output_dir, "nlp_metrics_corr_combined.csv")
        combined.to_csv(path, index=False)
        print(f"Saved combined → {path}")

        print("\n── Combined Results ─────────────────────────────────────────────")
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
        description="Compute NLP metrics on report pairs and correlate with ReXVal raters."
    )
    parser.add_argument(
        "--reports",
        help="CSV with study_id, candidate_reporter, candidate_report, gt_report",
    )
    parser.add_argument(
        "--studies",
        default="../physionet.org/files/RexVal/50_samples_gt_and_candidates.csv",
        help="Studies CSV used to build study_id map (50_samples_gt_and_candidates.csv)",
    )
    parser.add_argument(
        "--rexval",
        default="../physionet.org/files/RexVal/6_valid_raters_per_rater_error_categories.csv",
        help="ReXVal rater CSV (6_valid_raters_per_rater_error_categories.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="../data/rexval/nlp_metrics",
        help="Directory to write output CSVs",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for BERTScore / BlueRT (default: cuda)",
    )
    parser.add_argument(
        "--scores-cache", default=None,
        help="Path to previously saved nlp_metrics_scores.csv to skip recomputation",
    )
    args = parser.parse_args()

    run(
        reports_path=args.reports,
        studies_path=args.studies,
        rexval_path=args.rexval,
        output_dir=args.output_dir,
        device=args.device,
        scores_cache=args.scores_cache,
    )