"""
compare_arms.py — radiologist concordance of every DISCERN variant for one model:

  april_full / april_mini : archived April runs (reads/data/discern_runs/)
  original / updated      : full DISCERN, re-run with the original / updated entity list
  free                    : free-DISCERN (no entity list)

Arms whose files are missing for the requested repeats are skipped. Pairs that a
re-run arm (original/updated/free) failed to score in any repeat are dropped from
every arm and listed; gaps in the April arms are masked per correlation. Uses the
ground truth and correlation function of
reads/src/analyze_discern_repeats_and_correlation.py unchanged.

Outputs (in <ablation-dir>):
  arms_<model>_reps<..>.csv            mean ± SD over repeats of per-repeat Spearman ρ and Kendall τ
  arms_<model>_reps<..>_bootstrap.csv  paired cluster bootstrap of Δρ / Δτ (score averaged over
                                       repeats; clusters = studies) for updated−original,
                                       free−original and original−april_full

Usage
-----
  envs/vllm/bin/python discern/scripts/compare_arms.py \\
      --model databricks_claude_sonnet_4_6 \\
      --ablation-dir discern/data/discern_runs/entities_ablation_sonnet [--reps 0 1 2]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from compare_entity_versions import APRIL_DIR, _load_analysis_module, _load_free_repeat

PAIRS = [("score", "gt_total"), ("score", "gt_sig"), ("score", "gt_insig"),
         ("n_sig", "gt_sig"), ("n_insig", "gt_insig")]
COMPARISONS = [("updated", "original"), ("free", "original"), ("original", "april_full")]
N_BOOT = 4000


def _cols(df: pd.DataFrame, prefix: str, mini: bool = False) -> pd.DataFrame:
    src = "mini" if mini else "discern"
    rename = {("mini_discern_score" if mini else "discern_score"): f"{prefix}_score",
              f"{src}_n_sig": f"{prefix}_n_sig", f"{src}_n_insig": f"{prefix}_n_insig"}
    return df.rename(columns=rename)[["study_id", "candidate_reporter", *rename.values()]]


def _table(ana, ds: str, rep: int, model: str, abl: Path, arms: list[str]) -> pd.DataFrame:
    keys = ["study_id", "candidate_reporter"] if ds == "rexval" else ["study_id"]
    parts = []
    if "april_full" in arms:
        ana.RUN_DIR = APRIL_DIR
        apr = ana.load_hf_repeat(ds, model, rep)
        parts += [_cols(apr, "april_full"), _cols(apr, "april_mini", mini=True)]
    for arm in ("original", "updated"):
        if arm in arms:
            ana.RUN_DIR = abl / arm
            parts.append(_cols(ana.load_hf_repeat(ds, model, rep), arm))
    if "free" in arms:
        parts.append(_cols(_load_free_repeat(abl / "free", ds, model, rep), "free"))

    m = None
    for p in parts:
        p = p.copy()
        p["candidate_reporter"] = p["candidate_reporter"].astype(str)
        if ds == "radevalx":
            p = p.drop(columns="candidate_reporter")
        m = p if m is None else m.merge(p, on=keys)
    if ds == "radevalx":
        m = m.merge(ana.build_radevalx_gt().rename(columns={"report_id": "study_id"}), on="study_id")
    else:
        gt = ana.build_rexval_gt()
        gt = gt[gt["rater"] == "avg"].rename(columns={"candidate_type": "candidate_reporter"})
        m = m.merge(gt[["study_id", "candidate_reporter", "gt_sig", "gt_insig", "gt_total"]], on=keys)
    return m


RERUN_ARMS = ("original", "updated", "free")


def _drop_unscored(tabs: list[pd.DataFrame], keys: list[str], ds: str) -> list[pd.DataFrame]:
    """Keep the re-run arms paired: drop pairs that original/updated/free failed to score in any
    repeat. April arms keep their own gaps (masked per comparison)."""
    bad = set()
    for t in tabs:
        cols = [c for c in t.columns if c.endswith("_score") and c.split("_score")[0] in RERUN_ARMS]
        miss = t[t[cols].isna().any(axis=1)]
        bad |= set(map(tuple, miss[keys].astype(str).values))
    if bad:
        print(f"[{ds}] excluding {len(bad)} pair(s) unscored in some arm/repeat: {sorted(bad)}")
    return [t[~t[keys].astype(str).apply(tuple, axis=1).isin(bad)].reset_index(drop=True) for t in tabs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Model stem as in the run filenames.")
    ap.add_argument("--ablation-dir", type=Path, required=True)
    ap.add_argument("--reps", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    abl, model, reps = args.ablation_dir.resolve(), args.model, args.reps

    def have(folder: Path) -> bool:
        return all((folder / f"{d}_all_metrics_{model}_batch_v{r}.json").exists()
                   for d in ("rexval", "radevalx") for r in reps)

    arms = (["april_full", "april_mini"] if have(APRIL_DIR) else []) + \
           [a for a in ("original", "updated", "free") if have(abl / a)]
    if not arms:
        raise SystemExit("No complete arms found.")
    ana = _load_analysis_module()
    rng = np.random.default_rng(0)

    corr_rows, delta_rows = [], []
    for ds in ("rexval", "radevalx"):
        keys = ["study_id", "candidate_reporter"] if ds == "rexval" else ["study_id"]
        tabs = _drop_unscored([_table(ana, ds, r, model, abl, arms) for r in reps], keys, ds)
        for metric, g in PAIRS:
            row = {"model": model, "dataset": ds, "metric": metric, "gt": g, "n": len(tabs[0])}
            for arm in arms:
                res = [ana._single_corr(t[f"{arm}_{metric}"], t[g]) for t in tabs]
                for stat, k in (("rho", 0), ("tau", 2)):
                    vals = [x[k] for x in res if x]
                    row[f"{stat}_{arm}"] = np.mean(vals) if vals else np.nan
                    row[f"{stat}_sd_{arm}"] = np.std(vals, ddof=1) if len(vals) > 1 else np.nan
            corr_rows.append(row)

        m = (pd.concat(tabs).groupby(keys, dropna=False)
             .agg({**{f"{a}_score": "mean" for a in arms},
                   "gt_total": "first", "gt_sig": "first", "gt_insig": "first"})
             .reset_index())
        ids = np.unique(m["study_id"].values)
        idx = {c: np.flatnonzero(m["study_id"].values == c) for c in ids}
        draws = [np.concatenate([idx[c] for c in rng.choice(ids, len(ids), replace=True)])
                 for _ in range(N_BOOT)]
        for g in ("gt_total", "gt_sig", "gt_insig"):
            y = m[g].values
            for a, b in COMPARISONS:
                if a not in arms or b not in arms:
                    continue
                ok = m[f"{a}_score"].notna().values & m[f"{b}_score"].notna().values
                xa, xb, yy = m[f"{a}_score"].values[ok], m[f"{b}_score"].values[ok], y[ok]
                pos = np.cumsum(ok) - 1           # row index → position among kept rows
                draws_ok = [pos[t[ok[t]]] for t in draws]
                row = {"model": model, "dataset": ds, "gt": g, "comparison": f"{a} - {b}",
                       "n_pairs": int(ok.sum())}
                for stat, f in (("rho", spearmanr), ("tau", kendalltau)):
                    boot = np.array([f(xa[t], yy[t])[0] - f(xb[t], yy[t])[0] for t in draws_ok])
                    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
                    row.update({f"d_{stat}": round(f(xa, yy)[0] - f(xb, yy)[0], 4),
                                f"{stat}_ci95": f"[{lo:+.3f}, {hi:+.3f}]", f"p_{stat}": round(min(p, 1.0), 4)})
                delta_rows.append(row)

    corr, delta = pd.DataFrame(corr_rows), pd.DataFrame(delta_rows)
    tag = f"arms_{model}_reps{''.join(map(str, reps))}"
    corr.to_csv(abl / f"{tag}.csv", index=False)
    delta.to_csv(abl / f"{tag}_bootstrap.csv", index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format,
                           "display.max_columns", None):
        print(f"{model}  repeats={reps}  arms={arms}")
        for stat in ("rho", "tau"):
            t = corr[["dataset", "metric", "gt", "n"] + [f"{stat}_{a}" for a in arms]].copy()
            t["max_sd"] = corr[[f"{stat}_sd_{a}" for a in arms]].max(axis=1)
            print(f"\n{'Spearman ρ' if stat == 'rho' else 'Kendall τ'} (mean over repeats)")
            print(t.to_string(index=False))
        print("\nPaired bootstrap (score averaged over repeats), clustered by study:")
        print(delta.drop(columns="model").to_string(index=False))
    print(f"\nSaved → {abl / tag}.csv (+ _bootstrap.csv)")


if __name__ == "__main__":
    main()
