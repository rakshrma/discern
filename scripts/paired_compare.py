#!/usr/bin/env python3
"""Paired pairwise comparison across Qwen 3.5 models, per (dataset, metric).

For each metric and dataset:
  1. Pivot per-sample scores into rows=sample_idx, cols=model.
  2. Drop samples missing any model's score.
  3. For every (model_A, model_B) pair, compute:
       - Wilcoxon signed-rank p-value
       - mean Δ with bootstrap 95% CI
       - Cohen's d_z = mean(Δ) / std(Δ)  (paired effect size)
       - wins / losses / ties on per-sample comparison
  4. Apply Holm correction within each (dataset, metric) family (28 pairs).

Outputs:
    paired_pvalues.csv                    — long-form table of all pair stats
    paired_heatmap_<dataset>.png          — 3×3 grid of metric heatmaps;
                                            colored by effect size, annotated
                                            with significance stars

Run with the crimson env (only one we found with scipy + pandas + matplotlib):
    /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/envs/crimson/bin/python \
        scripts/paired_compare.py
"""

import argparse
import json
import os
import re
import sys
from glob import glob
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


DEFAULT_RESULTS_ROOT = (
    "/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/"
    "workspace/vlm_cxr_benchmark/results"
)

METRICS = [
    "bleu", "rouge", "meteor", "bertscore", "radgraph",
    "green", "crimson", "discern_score", "mini_discern_score",
]

METRIC_LABELS = {
    "bleu": "BLEU",
    "rouge": "ROUGE",
    "meteor": "METEOR",
    "bertscore": "BERTScore",
    "radgraph": "RadGraph",
    "green": "GREEN",
    "crimson": "CRIMSON",
    "discern_score": "DISCERN",
    "mini_discern_score": "mini-DISCERN",
}

DATASET_LABELS = {
    "mimic_cxr_test": "MIMIC-CXR",
    "chexpert_plus_valid": "CheXpert-Plus",
}

COMBINED_METRICS = ["crimson", "discern_score", "mini_discern_score"]
COMBINED_DATASETS = ["mimic_cxr_test", "chexpert_plus_valid"]


# ── helpers ──────────────────────────────────────────────────────────────────

def model_size_key(name: str) -> float:
    m = re.search(r"-(\d+(?:_\d+)?)b", name)
    return float(m.group(1).replace("_", ".")) if m else float("inf")


def short_label(name: str) -> str:
    return name.replace("qwen35-", "").replace("_", ".").upper()


def _flatten(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, dict):
        for k in ("f1", "f1_score", "score"):
            if k in v and isinstance(v[k], (int, float)):
                return float(v[k])
    return None


def load_long(results_root: str, pattern: str) -> pd.DataFrame:
    """Long DataFrame: columns=[model, dataset, mode, sample_idx, metric, score]."""
    rows = []
    for path in sorted(glob(os.path.join(results_root, pattern, "*_w_metrics.json"))):
        model = os.path.basename(os.path.dirname(path))
        base = os.path.basename(path).removesuffix("_w_metrics.json")
        m = re.match(r"^(?P<dataset>.+?)_(?P<mode>thinking|nothinking)_results$", base)
        if not m:
            continue
        dataset, mode = m.group("dataset"), m.group("mode")
        with open(path) as f:
            obj = json.load(f)
        for entry in obj.get("results", []):
            sid = entry.get("sample_idx")
            if sid is None:
                continue
            for metric in METRICS:
                v = _flatten(entry.get(metric))
                if v is None:
                    continue
                rows.append({
                    "model": model, "dataset": dataset, "mode": mode,
                    "sample_idx": sid, "metric": metric, "score": v,
                })
    return pd.DataFrame(rows)


def bootstrap_ci(delta: np.ndarray, n_boot: int = 1000, ci: float = 0.95,
                 seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(delta)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    means = delta[idx].mean(axis=1)
    a = (1 - ci) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def cohens_d_z(delta: np.ndarray) -> float:
    if len(delta) < 2:
        return 0.0
    s = delta.std(ddof=1)
    return 0.0 if s == 0 else float(delta.mean() / s)


def holm_correction(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in original order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    sorted_p = p[order]
    adj = sorted_p * (n - np.arange(n))
    adj = np.minimum.accumulate(adj[::-1])[::-1]  # enforce monotone non-increasing reversed
    # Holm wants monotone non-decreasing in sorted order; recompute correctly
    adj = np.maximum.accumulate(np.minimum(sorted_p * (n - np.arange(n)), 1.0))
    out = np.empty(n)
    out[order] = adj
    return out


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ── core analysis ────────────────────────────────────────────────────────────

def compute_pairwise(df: pd.DataFrame, dataset: str, metric: str,
                     n_boot: int = 1000) -> tuple[pd.DataFrame, list[str]]:
    sub = df[(df.dataset == dataset) & (df.metric == metric)]
    if sub.empty:
        return pd.DataFrame(), []
    pivot = sub.pivot_table(index="sample_idx", columns="model",
                            values="score", aggfunc="first").dropna(how="any")
    if pivot.empty:
        return pd.DataFrame(), []

    models = sorted(pivot.columns.tolist(), key=model_size_key)
    n_samples = len(pivot)
    rows = []
    for a, b in combinations(models, 2):
        x = pivot[a].to_numpy()
        y = pivot[b].to_numpy()
        delta = x - y
        if np.all(delta == 0):
            p_raw = 1.0
        else:
            try:
                _, p_raw = stats.wilcoxon(x, y, zero_method="wilcox")
            except ValueError:
                p_raw = np.nan
        ci_lo, ci_hi = bootstrap_ci(delta, n_boot=n_boot)
        rows.append({
            "dataset": dataset, "metric": metric,
            "model_a": a, "model_b": b,
            "n": n_samples,
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "d_z": cohens_d_z(delta),
            "wins_a": int((delta > 0).sum()),
            "losses_a": int((delta < 0).sum()),
            "ties": int((delta == 0).sum()),
            "p_wilcoxon_raw": float(p_raw),
        })
    out = pd.DataFrame(rows)
    out["p_wilcoxon_holm"] = holm_correction(out["p_wilcoxon_raw"].to_numpy())
    return out, models


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_heatmaps(all_pairs: pd.DataFrame, dataset: str, out_path: str):
    metrics_present = [m for m in METRICS
                       if not all_pairs[(all_pairs.dataset == dataset) & (all_pairs.metric == m)].empty]
    if not metrics_present:
        print(f"  skip plot for {dataset}: no metrics present")
        return

    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows))
    axes = axes.flatten()

    for ax_idx, metric in enumerate(metrics_present):
        ax = axes[ax_idx]
        sub = all_pairs[(all_pairs.dataset == dataset) & (all_pairs.metric == metric)]
        if sub.empty:
            ax.axis("off"); continue

        # Collect models in canonical order
        models = sorted(
            set(sub.model_a.tolist() + sub.model_b.tolist()),
            key=model_size_key,
        )
        n = len(models)
        idx = {m: i for i, m in enumerate(models)}

        dz_mat = np.full((n, n), np.nan)
        star_mat = np.empty((n, n), dtype=object)
        star_mat[:] = ""

        for _, r in sub.iterrows():
            i, j = idx[r.model_a], idx[r.model_b]
            # A vs B: positive d_z means A scored higher than B
            dz_mat[i, j] = r.d_z
            dz_mat[j, i] = -r.d_z
            s = stars(r.p_wilcoxon_holm)
            star_mat[i, j] = s
            star_mat[j, i] = s
        np.fill_diagonal(dz_mat, 0.0)

        # Symmetric colour limits around 0
        vmax = max(0.05, np.nanmax(np.abs(dz_mat)))
        im = ax.imshow(dz_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        labels = [short_label(m) for m in models]
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right",
                           fontsize=8, fontweight="bold")
        ax.set_yticklabels(labels, fontsize=8, fontweight="bold")
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)

        # Annotate cells with stars
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                s = star_mat[i, j]
                if s:
                    ax.text(j, i, s, ha="center", va="center",
                            fontsize=9, color="black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="d_z (row vs col)")

    # Hide any unused axes
    for k in range(len(metrics_present), rows * cols):
        axes[k].axis("off")

    fig.suptitle(
        f"Paired Wilcoxon on per-sample Δ — {dataset}\n"
        f"color = Cohen's d_z (row − col); red = row scores higher, blue = col scores higher.\n"
        f"BLEU / ROUGE / METEOR / BERTScore / RadGraph / GREEN / CRIMSON: higher is better (red = row better).\n"
        f"DISCERN / mini-DISCERN: lower is better (blue = row better).\n"
        f"stars = Holm-adjusted p (*<0.05, **<0.01, ***<0.001)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_combined(all_pairs: pd.DataFrame, datasets: list[str],
                  metrics: list[str], out_path: str):
    """Publication figure: datasets × metrics grid (e.g. 2 × 3) with one shared colorbar."""
    available_any = any(
        not all_pairs[(all_pairs.dataset == ds) & (all_pairs.metric == m)].empty
        for ds in datasets for m in metrics
    )
    if not available_any:
        print("  skip combined plot: none of the requested (dataset, metric) cells present")
        return

    nrows, ncols = len(datasets), len(metrics)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.3 * ncols, 4.0 * nrows),
        squeeze=False,
    )

    # Unified color scale across all panels so cells are visually comparable
    abs_vals = []
    for ds in datasets:
        for m in metrics:
            sub = all_pairs[(all_pairs.dataset == ds) & (all_pairs.metric == m)]
            if not sub.empty:
                abs_vals.append(float(np.nanmax(np.abs(sub.d_z.to_numpy()))))
    vmax = max(0.05, max(abs_vals)) if abs_vals else 0.5

    im_last = None
    for r, ds in enumerate(datasets):
        for c, metric in enumerate(metrics):
            ax = axes[r][c]
            sub = all_pairs[(all_pairs.dataset == ds) & (all_pairs.metric == metric)]
            if sub.empty:
                ax.axis("off")
                continue

            models = sorted(
                set(sub.model_a.tolist() + sub.model_b.tolist()),
                key=model_size_key,
            )
            n = len(models)
            idx = {mm: i for i, mm in enumerate(models)}
            dz_mat = np.full((n, n), np.nan)
            star_mat = np.empty((n, n), dtype=object); star_mat[:] = ""
            for _, row in sub.iterrows():
                i, j = idx[row.model_a], idx[row.model_b]
                dz_mat[i, j] = row.d_z
                dz_mat[j, i] = -row.d_z
                s = stars(row.p_wilcoxon_holm)
                star_mat[i, j] = s; star_mat[j, i] = s
            np.fill_diagonal(dz_mat, 0.0)

            im = ax.imshow(dz_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            im_last = im

            labels = [short_label(m) for m in models]
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=45, ha="right",
                               fontsize=9, fontweight="bold")
            ax.set_yticklabels(labels, fontsize=9, fontweight="bold")

            if r == 0:
                ax.set_title(METRIC_LABELS.get(metric, metric),
                             fontsize=13, fontweight="bold", pad=8)
            if c == 0:
                ax.set_ylabel(DATASET_LABELS.get(ds, ds),
                              fontsize=13, fontweight="bold", labelpad=10)

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    s = star_mat[i, j]
                    if s:
                        ax.text(j, i, s, ha="center", va="center",
                                fontsize=9, color="black")

    fig.tight_layout(rect=[0, 0, 0.92, 1])
    cbar_ax = fig.add_axes([0.935, 0.15, 0.014, 0.7])
    cb = fig.colorbar(im_last, cax=cbar_ax)
    cb.set_label("Cohen's $d_z$ (row − col)", fontsize=11)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--pattern", default="qwen35-*")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write CSV and PNGs (default: this script's dir).")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="Bootstrap iterations for mean-Δ CI.")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading metrics from {args.results_root}/{args.pattern}/ ...")
    df = load_long(args.results_root, args.pattern)
    if df.empty:
        sys.exit("No metrics loaded — make sure *_w_metrics.json exist.")
    print(f"  loaded {len(df)} (model, sample, metric) rows  "
          f"({df.model.nunique()} models, {df.dataset.nunique()} datasets, "
          f"{df.metric.nunique()} metrics)")

    all_pairs = []
    datasets = sorted(df.dataset.unique())
    for ds in datasets:
        for metric in METRICS:
            pairs, _ = compute_pairwise(df, ds, metric, n_boot=args.n_boot)
            if not pairs.empty:
                all_pairs.append(pairs)
    if not all_pairs:
        sys.exit("No pair statistics computed.")
    all_pairs_df = pd.concat(all_pairs, ignore_index=True)

    csv_path = os.path.join(out_dir, "paired_pvalues.csv")
    all_pairs_df.round(6).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}  ({len(all_pairs_df)} pair rows)")

    for ds in datasets:
        png_path = os.path.join(out_dir, f"paired_heatmap_{ds}.png")
        plot_heatmaps(all_pairs_df, ds, png_path)

    combined_path = os.path.join(out_dir, "paired_heatmap_combined.png")
    plot_combined(all_pairs_df, COMBINED_DATASETS, COMBINED_METRICS, combined_path)


if __name__ == "__main__":
    main()
