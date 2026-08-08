#!/usr/bin/env python3
"""Plot Qwen 3.5 metric trends across model sizes.

Produces three figures, each with chexpert + mimic side-by-side:

  qwen35_plot_nlp.png       BLEU, ROUGE, METEOR, BERTScore, RadGraph
  qwen35_plot_llm.png       GREEN, CRIMSON
  qwen35_plot_discern.png   DISCERN, mini-DISCERN  (lower = better)

Usage:
    python scripts/plot_metrics.py
    python scripts/plot_metrics.py --csv path/to/wide.csv --out-dir path/
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_CSV = (
    "/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/"
    "workspace/discern/scripts/qwen35_metrics_long.csv"
)

PLOT_GROUPS = [
    {
        "name": "nlp",
        "title": "NLP metrics — higher is better",
        "metrics": ["bleu", "rouge", "meteor", "radgraph", "bertscore"],
        "labels": ["BLEU", "ROUGE-L", "METEOR", "RadGraph F1", "BERTScore"],
        "secondary": ["bertscore"],  # BERTScore lives at ~0.86 — own axis
        "ylabel_left": "BLEU / ROUGE / METEOR / RadGraph",
        "ylabel_right": "BERTScore",
    },
    {
        "name": "llm",
        "title": "LLM-based metrics — higher is better",
        "metrics": ["green", "crimson"],
        "labels": ["GREEN", "CRIMSON"],
        "secondary": [],
        "ylabel_left": "score",
        "ylabel_right": None,
    },
    {
        "name": "discern",
        "title": "DISCERN metrics — lower is better",
        "metrics": ["discern_score", "mini_discern_score"],
        "labels": ["DISCERN", "mini-DISCERN"],
        "secondary": ["mini_discern_score"],  # different scale from full DISCERN
        "ylabel_left": "DISCERN",
        "ylabel_right": "mini-DISCERN",
    },
]


def model_size_key(model_name: str) -> float:
    m = re.search(r"-(\d+(?:_\d+)?)b", model_name)
    if not m:
        return float("inf")
    return float(m.group(1).replace("_", "."))


def short_label(model_name: str) -> str:
    """qwen35-0_8b → 0.8B, qwen35-122b-a10b → 122B-A10B"""
    rest = model_name.replace("qwen35-", "")
    return rest.replace("_", ".").upper()


def load_csv(path: str):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        sys.exit(f"No rows in {path}")
    return rows


def organize(rows):
    """Returns {dataset: {metric: [(model, mean, std), ...]}} sorted by model size.

    Reads from the long-form CSV (one row per model/dataset/metric).
    """
    by_ds = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ds = r.get("dataset")
        metric = r.get("metric")
        if not ds or not metric:
            continue
        try:
            mean_val = float(r["mean"]) if r.get("mean") else None
            std_val = float(r["std"]) if r.get("std") else 0.0
        except ValueError:
            continue
        if mean_val is None:
            continue
        by_ds[ds][metric].append((r["model"], mean_val, std_val))
    for ds in by_ds:
        for metric in by_ds[ds]:
            by_ds[ds][metric].sort(key=lambda x: model_size_key(x[0]))
    return by_ds


def plot_group(by_ds, group, out_path):
    datasets = sorted(by_ds.keys())
    fig, ax = plt.subplots(figsize=(11, 6))
    secondary = set(group.get("secondary") or [])
    ax2 = ax.twinx() if secondary else None

    cmap = plt.get_cmap("tab10")

    # Dataset → (linestyle, marker). Use solid+circle for the first dataset,
    # dashed+square for the second, dotted+triangle if a third ever appears.
    style_pool = [("-", "o"), ("--", "s"), (":", "^")]
    ds_styles = {ds: style_pool[i] for i, ds in enumerate(datasets)}

    metric_color = {}  # ordered insertion: first seen = first in legend
    for ds in datasets:
        linestyle, marker = ds_styles[ds]
        for i, (metric, label) in enumerate(zip(group["metrics"], group["labels"])):
            points = by_ds[ds].get(metric, [])
            if not points:
                continue
            x    = [short_label(m) for m, _, _ in points]
            y    = [m_val for _, m_val, _ in points]
            yerr = [s for _, _, s in points]
            target_ax = ax2 if metric in secondary else ax
            target_ax.errorbar(
                x, y, yerr=yerr,
                color=cmap(i), linestyle=linestyle, marker=marker,
                linewidth=2, markersize=7,
                capsize=3, capthick=1, elinewidth=1, alpha=0.9,
            )
            metric_color.setdefault(label, cmap(i))

    ax.set_xlabel("model")
    ax.set_ylabel(group["ylabel_left"])
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=35)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    if ax2 is not None and group.get("ylabel_right"):
        ax2.set_ylabel(group["ylabel_right"])

    # Two legends: one for metric colors, one for dataset linestyle/marker.
    metric_handles = [
        Line2D([0], [0], color=color, linewidth=2, label=label)
        for label, color in metric_color.items()
    ]
    dataset_handles = [
        Line2D([0], [0], color="gray", linewidth=2,
               linestyle=ls, marker=mk, label=ds)
        for ds, (ls, mk) in ds_styles.items()
    ]
    anchor_x = 1.14 if ax2 is not None else 1.02
    leg_metrics = ax.legend(handles=metric_handles, loc="upper left",
                            bbox_to_anchor=(anchor_x, 1.0),
                            frameon=False, title="Metric")
    ax.add_artist(leg_metrics)
    ax.legend(handles=dataset_handles, loc="lower left",
              bbox_to_anchor=(anchor_x, 0.0),
              frameon=False, title="Dataset")

    fig.suptitle(group["title"], fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.82, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out-dir", default=None,
                    help="Directory for PNGs; defaults to CSV's directory.")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)

    rows = load_csv(args.csv)
    by_ds = organize(rows)

    for group in PLOT_GROUPS:
        out_path = os.path.join(out_dir, f"qwen35_plot_{group['name']}.png")
        plot_group(by_ds, group, out_path)


if __name__ == "__main__":
    main()
