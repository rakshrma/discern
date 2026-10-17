"""
compare_entity_versions.py — radiologist concordance of full DISCERN with the
original vs updated entity list, and of free-DISCERN (no entity list), from
jobs/entities_ablation.sh.

Reuses the ground-truth tables and per-repeat Spearman/Kendall correlation from
reads/src/analyze_discern_repeats_and_correlation.py unchanged, so numbers are
directly comparable with discern_extended_correlation_*.csv. free-DISCERN's
score and error counts are mapped onto the discern_* columns before correlating
(n_insig = errors scored 1-2, n_sig = errors scored 3-4, as for mini-DISCERN).

Arms (missing arms are skipped)
----
  april     : reads/data/discern_runs/            (archived reads/ runs, original entities)
  original  : <ablation-dir>/original/
  updated   : <ablation-dir>/updated/
  free      : <ablation-dir>/free/                (free-DISCERN)

Outputs (<ablation-dir>/, default data/discern_runs/entities_ablation/)
  comparison_long.csv  — one row per arm × dataset × metric × GT column
  comparison_wide.csv  — arms side by side, plus deltas vs original

Usage
-----
  envs/vllm/bin/python discern/scripts/compare_entity_versions.py
  envs/vllm/bin/python discern/scripts/compare_entity_versions.py \\
      --ablation-dir discern/data/discern_runs/entities_ablation_mig90
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

_ROOT      = Path(__file__).parent.parent
_WORKSPACE = _ROOT.parent
ANALYSIS_PY = _WORKSPACE / "reads/src/analyze_discern_repeats_and_correlation.py"
ABLATION_DIR = _ROOT / "data/discern_runs/entities_ablation"
APRIL_DIR    = _WORKSPACE / "reads/data/discern_runs"
DISCERN_ONLY = {"discern_n_sig", "discern_n_insig", "discern_score"}


def _load_analysis_module():
    spec = importlib.util.spec_from_file_location("reads_analysis", ANALYSIS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_free_repeat(run_dir: Path, dataset: str, model: str, repeat: int) -> pd.DataFrame | None:
    """free-DISCERN run → the same columns load_hf_repeat produces for DISCERN."""
    path = run_dir / f"{dataset}_all_metrics_{model}_batch_v{repeat}.json"
    if not path.exists():
        return None
    rows = []
    for r in json.loads(path.read_text())["results"]:
        row = {"study_id": str(r["study_id"]), "candidate_reporter": r.get("candidate_reporter"),
               "repeat": repeat}
        if r.get("free_discern_score") is not None:
            sig = [e.get("clinical_significance_score", 0) for e in (r.get("free_discern_evaluation") or [])]
            row["discern_n_insig"] = sum(1 for s in sig if s in (1, 2))
            row["discern_n_sig"]   = sum(1 for s in sig if s in (3, 4))
            row["discern_score"]   = float(r["free_discern_score"])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="google_gemma_4_31B_it",
                        help="Model stem as it appears in the run filenames.")
    parser.add_argument("--ablation-dir", type=Path, default=ABLATION_DIR,
                        help="Folder holding original/, updated/ and free/ runs; outputs are written here.")
    args = parser.parse_args()
    ablation_dir = args.ablation_dir.resolve()
    arms = {"april": APRIL_DIR, "original": ablation_dir / "original",
            "updated": ablation_dir / "updated", "free": ablation_dir / "free"}

    ana = _load_analysis_module()
    gt = {"radevalx": ana.build_radevalx_gt(), "rexval": ana.build_rexval_gt()}
    correlate = {"radevalx": ana._correlate_radevalx, "rexval": ana._correlate_rexval}

    rows = []
    for arm, run_dir in arms.items():
        ana.RUN_DIR = run_dir  # load_hf_repeat reads this module global
        for dataset in ana.DATASETS:
            load = ((lambda rep: _load_free_repeat(run_dir, dataset, args.model, rep)) if arm == "free"
                    else (lambda rep: ana.load_hf_repeat(dataset, args.model, rep)))
            frames = [df for rep in (0, 1, 2) if (df := load(rep)) is not None]
            if not frames:
                print(f"  [{arm:<8}] {dataset:<8} no runs in {run_dir}")
                continue
            combined = pd.concat(frames, ignore_index=True)
            n_scored = int(frames[0]["discern_score"].notna().sum())
            print(f"  [{arm:<8}] {dataset:<8} repeats={len(frames)}  "
                  f"scored={n_scored}/{len(frames[0])}  "
                  f"mean_score={combined['discern_score'].mean():.2f}")
            for r in correlate[dataset](combined, gt[dataset], args.model, has_repeats=True):
                if r["metric"] in DISCERN_ONLY:
                    rows.append({"arm": arm, "n_scored": n_scored, **r})

    if not rows:
        raise SystemExit("No runs found for any arm.")

    long = pd.DataFrame(rows)
    ablation_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(ablation_dir / "comparison_long.csv", index=False)

    keys = ["dataset", "metric", "gt_col"]
    wide = long.pivot_table(index=keys, columns="arm",
                            values=["mean_spearman_r", "mean_kendall_tau"])
    wide.columns = [f"{'rho' if v == 'mean_spearman_r' else 'tau'}_{arm}" for v, arm in wide.columns]
    for stat in ("rho", "tau"):
        for arm in ("updated", "free"):
            if f"{stat}_{arm}" in wide and f"{stat}_original" in wide:
                wide[f"{stat}_delta_{arm}"] = (wide[f"{stat}_{arm}"] - wide[f"{stat}_original"]).round(4)
    order = [c for s in ("rho", "tau") for c in
             (f"{s}_april", f"{s}_original", f"{s}_updated", f"{s}_free",
              f"{s}_delta_updated", f"{s}_delta_free") if c in wide]
    wide = wide[order].reset_index()
    wide.to_csv(ablation_dir / "comparison_wide.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n" + wide.to_string(index=False))
    print(f"\nSaved → {ablation_dir / 'comparison_long.csv'}")
    print(f"Saved → {ablation_dir / 'comparison_wide.csv'}")


if __name__ == "__main__":
    main()
