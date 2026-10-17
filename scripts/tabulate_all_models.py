"""
tabulate_all_models.py — cross-model tables of radiologist concordance for the three re-run arms
(original list / updated list / free-DISCERN), built from the per-model compare_arms.py outputs.

For each ground truth (total, significant, non-significant errors) and dataset, one table of
DISCERN score vs radiologist error count: Kendall τ and Spearman ρ, mean ± SD over repeats, on the
pairs every arm scored. Best value per row and statistic in bold.

Outputs (in data/discern_runs/):
  all_models_corr.csv        long table (model, dataset, gt, arm, rho, rho_sd, tau, tau_sd, n, reps)
  all_models_corr_tables.md  the markdown tables

Usage:  envs/vllm/bin/python discern/scripts/tabulate_all_models.py   (rerun compare_arms.py first)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RUNS = Path(__file__).resolve().parents[1] / "data" / "discern_runs"
MODELS = [  # label, ablation-dir suffix, model stem
    ("Claude Sonnet 4.6", "_sonnet", "databricks_claude_sonnet_4_6"),
    ("Gemma-4-31B", "", "google_gemma_4_31B_it"),
    ("MedGemma-27B", "_medgemma_27b", "google_medgemma_27b_text_it"),
    ("Gemma-3-12B", "_gemma3_12b", "google_gemma_3_12b_it"),
    ("Qwen3.5-9B", "_qwen35_9b", "Qwen_Qwen3.5_9B"),
    ("Llama-3.1-8B", "_llama31_8b", "meta_llama_Llama_3.1_8B_Instruct"),
    ("Gemma-4-E4B", "_gemma4_e4b", "google_gemma_4_E4B_it"),
    ("Qwen3.5-4B", "_qwen35_4b", "Qwen_Qwen3.5_4B"),
    ("MedGemma-1.5-4B", "_medgemma15_4b", "google_medgemma_1.5_4b_it"),
    ("MedGemma-4B", "_medgemma_4b", "google_medgemma_4b_it"),
    ("Gemma-4-E2B", "_gemma4_e2b", "google_gemma_4_E2B_it"),
    ("Qwen3.5-2B", "_qwen35_2b", "Qwen_Qwen3.5_2B"),
    ("Qwen3.5-0.8B", "_qwen35_0_8b", "Qwen_Qwen3.5_0.8B"),
]
ARMS = ("original", "updated", "free")
GTS = [("gt_total", "Total errors"), ("gt_sig", "Significant errors"), ("gt_insig", "Non-significant errors")]
DATASETS = [("rexval", "ReXVal (max 200 pairs)"), ("radevalx", "RaDEvalX (max 100 pairs)")]


def _arms_csv(abl: Path, stem: str) -> Path | None:
    """The compare_arms output covering the most repeats."""
    files = sorted(abl.glob(f"arms_{stem}_reps*.csv"), key=lambda p: len(p.stem.split("_reps")[-1]))
    files = [f for f in files if not f.stem.endswith("_bootstrap")]
    return files[-1] if files else None


def _num(x: float) -> str:
    return f"{x:.3f}".replace("0.", ".", 1).replace("-", "−")


def main():
    rows = []
    for label, suf, stem in MODELS:
        f = _arms_csv(RUNS / f"entities_ablation{suf}", stem)
        if f is None:
            continue
        reps = len(f.stem.split("_reps")[-1])
        d = pd.read_csv(f)
        d = d[d["metric"] == "score"]
        for _, r in d.iterrows():
            for arm in ARMS:
                rows.append({"model": label, "dataset": r["dataset"], "gt": r["gt"], "arm": arm,
                             "n": int(r["n"]), "reps": reps,
                             "rho": r.get(f"rho_{arm}"), "rho_sd": r.get(f"rho_sd_{arm}"),
                             "tau": r.get(f"tau_{arm}"), "tau_sd": r.get(f"tau_sd_{arm}")})
    df = pd.DataFrame(rows)
    df.to_csv(RUNS / "all_models_corr.csv", index=False)

    out = ["Radiologist concordance of the DISCERN score: Kendall τ and Spearman ρ, mean ± SD over "
           "repeats, on the pairs all three arms scored (n). Best per row and statistic in **bold**. "
           "Single-repeat models have no SD.", ""]
    for gt, gt_title in GTS:
        for ds, ds_title in DATASETS:
            out += [f"### {gt_title} — {ds_title}", "",
                    "| Model | Repeats | Pairs | τ original | τ updated | τ free | ρ original | ρ updated | ρ free |",
                    "|---|---|---|---|---|---|---|---|---|"]
            sub = df[(df["gt"] == gt) & (df["dataset"] == ds)]
            for label, *_ in MODELS:
                m = sub[sub["model"] == label].set_index("arm")
                if m.empty:
                    continue
                cells = []
                for stat in ("tau", "rho"):
                    best = m[stat].max()
                    for arm in ARMS:
                        v, sd = m.loc[arm, stat], m.loc[arm, f"{stat}_sd"]
                        if pd.isna(v):
                            cells.append("pending")
                            continue
                        s = _num(v) if pd.isna(sd) else f"{_num(v)} ± {_num(sd)}"
                        cells.append(f"**{s}**" if v == best else s)
                first = m.iloc[0]
                out.append(f"| {label} | {first['reps']} | {first['n']} | " + " | ".join(cells) + " |")
            out.append("")
    (RUNS / "all_models_corr_tables.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
