"""Analyze DISCERN evaluation errors for one model x dataset output.

Two clinically-focused views, restricted to clinically meaningful
discrepancies (significance >= 3 by default; the DISCERN scale runs 0-4
where 3 = important, 4 = critical / changes immediate management):

  Plot A  top_missed_subcat.png
          Top-N subcategories ranked by critical-error count. Each bar
          is split by error type (missing, wrong-dx, wrong-loc,
          wrong-sev, wrong-temp, extra/hallucinated). Right-margin
          annotation shows  n_critical / n_GT_occurrences = miss_rate%.
          A '*' marks rows that contain at least one sig=4 event.

  Plot B  severity_vs_frequency.png
          X = log10 of ground-truth prevalence (cases where the entity
              was actually present).
          Y = critical-miss rate (sig>=3 errors of any 'missing/wrong-*'
              type, divided by GT prevalence).
          Bubble size = count of sig=4 (critical) events.
          Bubble color = mean significance of erroneous entries.
          Quadrants: top-left = rare but reliably missed; top-right =
          common and reliably wrong; bottom-right = common, handled.

Per-entry error classification:
  missing   discrepancy_type == 'missing_in_candidate'
  extra     discrepancy_type == 'extra_in_candidate'
  wrong-{dx,loc,sev,temp}   matched finding with corresponding
            concordance field == 'discordant' or 'candidate-misses'

Ground-truth prevalence for an entity =
    matched_entries + missing_in_candidate_entries.

Outputs (under <input_dir>/error_analysis/<stem>/ by default):
  errors_long.csv           one row per (sample, entity, error_type)
  prevalence.csv            per-entity GT prevalence and rates
  summary.json              top-level counts
  top_missed_subcat.png     Plot A
  severity_vs_frequency.png Plot B

flatten() / per_entity_stats() are model/dataset-agnostic so the same
helpers can drive cross-model views later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONCORDANCE_DIMS: tuple[tuple[str, str], ...] = (
    ("diagnosis_concordance", "dx"),
    ("location_concordance", "loc"),
    ("severity_concordance", "sev"),
    ("temporal_comparison", "temp"),
)

ERROR_TYPES: tuple[str, ...] = (
    "missing",
    "wrong-dx",
    "wrong-loc",
    "wrong-sev",
    "wrong-temp",
    "extra",
)

# Distinct qualitative palette — readable on a single bar.
ERROR_COLORS = {
    "missing":    "#4a4a4a",
    "wrong-dx":   "#1f77b4",
    "wrong-loc":  "#2ca02c",
    "wrong-sev":  "#9467bd",
    "wrong-temp": "#17becf",
    "extra":      "#d62728",
}

MISS_LIKE: tuple[str, ...] = (
    "missing", "wrong-dx", "wrong-loc", "wrong-sev", "wrong-temp",
)


def parse_entity(entity: str) -> tuple[str, str]:
    """Split 'Category :: Subcategory' into (category, subcategory)."""
    if " :: " in entity:
        cat, sub = entity.split(" :: ", 1)
        return cat.strip(), sub.strip()
    e = entity.strip()
    return e, e


def classify_entry(entry: dict) -> list[str]:
    """Return zero or more error labels for one discern_evaluation entry.

    A matched entry that is fully concordant returns []. A matched entry
    with multiple dimensions wrong returns multiple labels.
    """
    dtype = entry.get("discrepancy_type")
    if dtype == "missing_in_candidate":
        return ["missing"]
    if dtype == "extra_in_candidate":
        return ["extra"]
    labels = []
    for field, suffix in CONCORDANCE_DIMS:
        if entry.get(field) in ("discordant", "candidate-misses"):
            labels.append(f"wrong-{suffix}")
    return labels


def flatten_entries(results: list[dict], *, model: str = "",
                    dataset: str = "") -> pd.DataFrame:
    """One row per discern_evaluation entry (including matched ones).

    Each row carries the canonical 'primary_status' for prevalence math:
      gt_only       discrepancy_type == 'missing_in_candidate'
      cand_only     discrepancy_type == 'extra_in_candidate'
      matched       both findings present
    Plus a comma-joined 'errors' field listing classify_entry() labels.
    """
    rows = []
    for sample in results:
        sidx = sample.get("sample_idx")
        for entry in sample.get("discern_evaluation", []):
            cat, sub = parse_entity(entry.get("entity", ""))
            sig = int(entry.get("significance_score") or 0)
            dtype = entry.get("discrepancy_type")
            if dtype == "missing_in_candidate":
                status = "gt_only"
            elif dtype == "extra_in_candidate":
                status = "cand_only"
            else:
                status = "matched"
            errs = classify_entry(entry)
            rows.append({
                "model": model,
                "dataset": dataset,
                "sample_idx": sidx,
                "category": cat,
                "subcategory": sub,
                "primary_status": status,
                "errors": ",".join(errs),
                "significance": sig,
            })
    return pd.DataFrame(rows)


def explode_errors(entries: pd.DataFrame) -> pd.DataFrame:
    """One row per (sample, entity, error_type), inheriting significance."""
    rows = []
    for _, r in entries.iterrows():
        for err in (r["errors"].split(",") if r["errors"] else []):
            rows.append({
                "model": r["model"],
                "dataset": r["dataset"],
                "sample_idx": r["sample_idx"],
                "category": r["category"],
                "subcategory": r["subcategory"],
                "error_type": err,
                "significance": r["significance"],
            })
    return pd.DataFrame(rows, columns=[
        "model", "dataset", "sample_idx", "category", "subcategory",
        "error_type", "significance",
    ])


def per_entity_stats(entries: pd.DataFrame, errors: pd.DataFrame,
                     min_sig: int) -> pd.DataFrame:
    """Aggregate per-subcategory: prevalence, miss rate, sig stats."""
    gt_prev = (entries[entries["primary_status"].isin(["matched", "gt_only"])]
               .groupby("subcategory").size().rename("gt_prevalence"))
    crit_errors = errors[errors["significance"] >= min_sig]

    crit_miss = (crit_errors[crit_errors["error_type"].isin(MISS_LIKE)]
                 .groupby("subcategory").size().rename("n_critical_miss"))
    crit_extra = (crit_errors[crit_errors["error_type"] == "extra"]
                  .groupby("subcategory").size().rename("n_critical_extra"))
    sig4 = (crit_errors[crit_errors["significance"] == 4]
            .groupby("subcategory").size().rename("n_sig4"))
    mean_sig = (crit_errors.groupby("subcategory")["significance"].mean()
                .rename("mean_significance"))
    sigsum = (crit_errors.groupby("subcategory")["significance"].sum()
              .rename("sig_total"))
    # Parent category for hover/lookup.
    parent = (entries.drop_duplicates("subcategory")
              .set_index("subcategory")["category"].rename("category"))

    df = (pd.concat([gt_prev, crit_miss, crit_extra, sig4, mean_sig,
                     sigsum, parent], axis=1)
          .fillna({"gt_prevalence": 0, "n_critical_miss": 0,
                   "n_critical_extra": 0, "n_sig4": 0,
                   "mean_significance": 0, "sig_total": 0}))
    df = df[df["category"].notna()]  # drop entities only seen as hallucinations w/o parent
    df["critical_miss_rate"] = np.where(
        df["gt_prevalence"] > 0,
        df["n_critical_miss"] / df["gt_prevalence"],
        np.nan,
    )
    df = df.sort_values("sig_total", ascending=False)
    return df


def plot_critical_profile(errors: pd.DataFrame, stats: pd.DataFrame,
                          out_path: Path, top_n: int, min_sig: int,
                          suptitle: str) -> None:
    """Plot A: top-N subcategories, bars segmented by error type."""
    crit = errors[errors["significance"] >= min_sig]
    if crit.empty:
        print("[WARN] No errors meet the significance threshold for Plot A.")
        return
    top = (crit.groupby("subcategory").size()
              .sort_values(ascending=False).head(top_n).index.tolist())
    pivot = (crit[crit["subcategory"].isin(top)]
             .groupby(["subcategory", "error_type"]).size()
             .unstack(fill_value=0)
             .reindex(index=top, columns=list(ERROR_TYPES), fill_value=0))

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.4 * len(pivot))))
    y = np.arange(len(pivot))
    left = np.zeros(len(pivot))
    for err in ERROR_TYPES:
        widths = pivot[err].to_numpy()
        if widths.sum() == 0:
            continue
        ax.barh(y, widths, left=left, color=ERROR_COLORS[err],
                label=err, edgecolor="white", linewidth=0.5)
        left += widths

    row_max = left.max() if len(left) else 1
    # Right-margin annotation: critical-count / GT-prevalence = rate%.
    for i, sub in enumerate(pivot.index):
        row = stats.loc[sub] if sub in stats.index else None
        if row is None:
            continue
        n_crit = int(row["n_critical_miss"] + row["n_critical_extra"])
        n_gt = int(row["gt_prevalence"])
        rate_str = (f"{int(row['n_critical_miss'])}/{n_gt}"
                    f" = {row['critical_miss_rate']*100:.0f}%"
                    if n_gt > 0 else f"{n_crit} extra-only")
        star = " *" if int(row["n_sig4"]) > 0 else ""
        ax.text(left[i] + row_max * 0.01, i, rate_str + star,
                va="center", ha="left", fontsize=9)

    ax.set_yticks(y, pivot.index.tolist())
    ax.invert_yaxis()
    ax.set_xlabel(f"Critical-error count (sig >= {min_sig})")
    ax.set_xlim(0, row_max * 1.25)
    ax.legend(title="error type", loc="lower right", framealpha=0.9,
              fontsize=8, ncol=2)
    ax.set_title(suptitle + "\n* = at least one sig=4 (critical) event",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_severity_vs_frequency(stats: pd.DataFrame, out_path: Path,
                               min_sig: int, n_samples: int,
                               suptitle: str, annotate_top_k: int = 10,
                               min_prev_for_annotation: int = 3) -> None:
    """Plot B: scatter of miss rate vs GT prevalence."""
    rateable = stats[stats["gt_prevalence"] > 0].copy()
    if rateable.empty:
        print("[WARN] No entities with GT prevalence > 0 for Plot B.")
        return

    x = rateable["gt_prevalence"].to_numpy().astype(float)
    y = rateable["critical_miss_rate"].to_numpy().astype(float)
    sizes = 60 + 90 * rateable["n_sig4"].to_numpy()
    colors = rateable["mean_significance"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(x, y, s=sizes, c=colors, cmap="YlOrRd",
                    vmin=min_sig, vmax=4, alpha=0.75,
                    edgecolors="black", linewidths=0.5)

    annotated = (rateable.sort_values("sig_total", ascending=False)
                 .pipe(lambda d: d[d["gt_prevalence"] >= min_prev_for_annotation])
                 .head(annotate_top_k))
    for sub, row in annotated.iterrows():
        ax.annotate(sub, (row["gt_prevalence"], row["critical_miss_rate"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, alpha=0.9)

    ax.set_xscale("log")
    ax.set_xlim(left=max(0.7, x.min() * 0.8))
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Ground-truth prevalence (# cases the entity appeared in)")
    ax.set_ylabel(f"Critical-miss rate (sig >= {min_sig} miss/wrong-* per GT case)")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("mean significance of errors")
    ax.set_title(suptitle + f"\nbubble size ∝ # sig=4 events  |  n_samples={n_samples}",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def infer_label(meta: dict, default: str) -> str:
    p = meta.get("input_dir") or meta.get("input") or ""
    parts = [s for s in Path(p).parts if s]
    return parts[-1] if parts else default


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", help="Path to a *.discern.json output file")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: <input_dir>/error_analysis/<stem>)")
    ap.add_argument("--top-n", type=int, default=15,
                    help="Subcategories in Plot A (default 15)")
    ap.add_argument("--min-significance", type=int, default=3,
                    help="Drop errors below this significance. Default 3.")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    with open(in_path) as f:
        data = json.load(f)
    meta = data.get("_metadata", {}) or {}
    results = data.get("results", []) or []
    n_samples = len(results)
    if n_samples == 0:
        raise SystemExit(f"[ERROR] {in_path} has no results")

    stem = in_path.stem.replace(".discern", "")
    out_dir = Path(args.out) if args.out else in_path.parent / "error_analysis" / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    model_label = infer_label(meta, default=stem)
    min_sig = max(0, min(args.min_significance, 4))

    entries = flatten_entries(results, model=model_label, dataset=stem)
    errors = explode_errors(entries)
    stats = per_entity_stats(entries, errors, min_sig=min_sig)

    errors.to_csv(out_dir / "errors_long.csv", index=False)
    stats.to_csv(out_dir / "prevalence.csv")

    header = f"{model_label} | {stem} | n={n_samples} | sig >= {min_sig}"

    plot_critical_profile(
        errors, stats,
        out_dir / "top_missed_subcat.png",
        top_n=args.top_n, min_sig=min_sig,
        suptitle=f"Critical-error profile (top {args.top_n} entities)\n{header}",
    )
    plot_severity_vs_frequency(
        stats, out_dir / "severity_vs_frequency.png",
        min_sig=min_sig, n_samples=n_samples,
        suptitle=f"Critical-miss rate vs ground-truth prevalence\n{header}",
    )

    crit_errors = errors[errors["significance"] >= min_sig]
    summary = {
        "n_samples": n_samples,
        "min_significance": min_sig,
        "total_evaluation_entries": int(len(entries)),
        "n_gt_entity_occurrences": int(
            entries["primary_status"].isin(["matched", "gt_only"]).sum()
        ),
        "n_critical_errors_total": int(len(crit_errors)),
        "n_critical_missing": int((crit_errors["error_type"] == "missing").sum()),
        "n_critical_extra": int((crit_errors["error_type"] == "extra").sum()),
        "n_critical_wrong_dim": int(
            crit_errors["error_type"].str.startswith("wrong-").sum()
        ),
        "n_sig4_errors": int((crit_errors["significance"] == 4).sum()),
        "critical_errors_per_sample": float(len(crit_errors) / n_samples),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Wrote 2 plots + 2 CSVs + summary.json to:\n  {out_dir}")
    print(f"[INFO] {json.dumps(summary)}")


if __name__ == "__main__":
    main()
