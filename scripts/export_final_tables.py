"""
export_final_tables.py — write every cross-model table of the entity-list / free-DISCERN study as CSV.

Reads the outputs of compare_arms.py (per model), tabulate_all_models.py and compare_vs_baselines.py,
plus the run JSONs for failure counts; rerun those first. Writes data/discern_runs/final_tables/:

  01_concordance_by_method.csv  DISCERN score vs radiologist errors (total / significant / non-significant):
                                Kendall τ and Spearman ρ, mean and SD over repeats, original / updated / free,
                                on the pairs all three methods scored
  02_method_differences.csv     paired cluster bootstrap: updated − original, free − original (and
                                original − April full DISCERN where an April run exists)
  03_unscored_pairs.csv         pairs each method failed to score, per repeat (mean / min / max)
  04_baseline_metrics.csv       April BLEU-1, ROUGE-L, METEOR, BERTScore, RadGraph-F1, GREEN, CRIMSON
                                (sign flipped: positive = agreement with radiologists)
  05_free_vs_green_crimson.csv  free-DISCERN (all pairs it scored) vs GREEN and CRIMSON on the same pairs
  06_free_nonerror_items.csv    free-DISCERN items whose four concordances are all concordant/not mentioned
                                but which were still scored ≥ 1

Usage:  envs/vllm/bin/python discern/scripts/export_final_tables.py
"""
from __future__ import annotations

import json

import pandas as pd

from tabulate_all_models import MODELS, RUNS, _arms_csv

OUT = RUNS / "final_tables"
ERR = {"gt_total": "total", "gt_sig": "significant", "gt_insig": "non-significant"}
DS = {"rexval": "ReXVal", "radevalx": "RaDEvalX"}
N_PAIRS = {"rexval": 200, "radevalx": 100}
ARMS = ("original", "updated", "free")
ORDER = [m[0] for m in MODELS]


def _sort(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    df = df.copy()
    keys = {"model": lambda s: s.map({m: i for i, m in enumerate(ORDER)}),
            "dataset": lambda s: s.map({v: i for i, v in enumerate(DS.values())}),
            "error_type": lambda s: s.map({v: i for i, v in enumerate(ERR.values())})}
    return df.sort_values(by, key=lambda s: keys[s.name](s) if s.name in keys else s).reset_index(drop=True)


def concordance() -> pd.DataFrame:
    d = pd.read_csv(RUNS / "all_models_corr.csv")
    w = d.pivot_table(index=["dataset", "gt", "model", "reps", "n"], columns="arm",
                      values=["tau", "tau_sd", "rho", "rho_sd"]).reset_index()
    w.columns = [f"{s}_{a}" if a else s for s, a in w.columns]   # tau_free, tau_sd_free, ...
    w = w.rename(columns={"gt": "error_type", "reps": "repeats", "n": "pairs"})
    w["dataset"], w["error_type"] = w["dataset"].map(DS), w["error_type"].map(ERR)
    cols = ["dataset", "error_type", "model", "repeats", "pairs"] + \
           [f"{s}_{a}" for st in ("tau", "rho") for a in ARMS for s in (st, f"{st}_sd")]
    return _sort(w[cols], ["error_type", "dataset", "model"]).round(4)


def method_differences() -> pd.DataFrame:
    rows = []
    for label, suf, stem in MODELS:
        f = _arms_csv(RUNS / f"entities_ablation{suf}", stem)
        if f is None:
            continue
        b = pd.read_csv(f.with_name(f"{f.stem}_bootstrap.csv")).drop(columns="model")   # model stem
        b.insert(0, "model", label)
        b.insert(1, "repeats", len(f.stem.split("_reps")[-1]))
        rows.append(b)
    d = pd.concat(rows, ignore_index=True).rename(columns={"gt": "error_type", "n_pairs": "pairs"})
    d["dataset"], d["error_type"] = d["dataset"].map(DS), d["error_type"].map(ERR)
    d = d[["model", "repeats", "dataset", "error_type", "comparison", "pairs",
           "d_tau", "tau_ci95", "p_tau", "d_rho", "rho_ci95", "p_rho"]]
    return _sort(d, ["error_type", "dataset", "model", "comparison"]).round(4)


def unscored_pairs() -> pd.DataFrame:
    rows = []
    for label, suf, stem in MODELS:
        for arm in ARMS:
            k = "free_discern_score" if arm == "free" else "discern_score"
            for ds in DS:
                files = sorted((RUNS / f"entities_ablation{suf}" / arm).glob(f"{ds}_all_metrics_{stem}_batch_v[012].json"))
                fails = [sum(r.get(k) is None for r in json.loads(f.read_text())["results"]) for f in files]
                if not fails:
                    continue
                mean = sum(fails) / len(fails)
                rows.append({"model": label, "dataset": DS[ds], "method": arm, "repeats": len(fails),
                             "unscored_mean": round(mean, 2), "unscored_min": min(fails), "unscored_max": max(fails),
                             "total_pairs": N_PAIRS[ds], "pct_unscored": round(100 * mean / N_PAIRS[ds], 1)})
    return _sort(pd.DataFrame(rows), ["dataset", "model"])


def baselines() -> tuple[pd.DataFrame, pd.DataFrame]:
    c = pd.read_csv(RUNS / "vs_baselines_corr.csv")
    b = pd.read_csv(RUNS / "vs_baselines_bootstrap.csv")
    base = c[c["model"] == "—"].rename(columns={"method": "metric", "gt": "error_type", "n": "pairs"})
    base["dataset"], base["error_type"] = base["dataset"].map(DS), base["error_type"].map(ERR)
    base = _sort(base[["metric", "dataset", "error_type", "pairs", "tau", "rho"]], ["error_type", "dataset"]).round(4)

    free = c[c["method"] == "DISCERN-free"][["dataset", "model", "gt", "reps", "tau", "rho"]]
    free = free.rename(columns={"tau": "free_tau", "rho": "free_rho", "reps": "repeats"}).astype({"repeats": int})
    v = b.merge(free, on=["dataset", "model", "gt"]).rename(columns={"gt": "error_type", "n": "pairs"})
    v["dataset"], v["error_type"] = v["dataset"].map(DS), v["error_type"].map(ERR)
    v = v[["model", "repeats", "dataset", "error_type", "comparison", "pairs", "free_tau", "free_rho",
           "d_tau", "tau_ci95", "p_tau", "d_rho", "rho_ci95", "p_rho"]]
    return base, _sort(v, ["error_type", "dataset", "model", "comparison"]).round(4)


def nonerror_items() -> pd.DataFrame:
    ok = {"concordant", "not mentioned"}
    conc = ("diagnosis_concordance", "location_concordance", "severity_concordance", "temporal_concordance")
    rows = []
    for label, suf, stem in MODELS:
        files = sorted((RUNS / f"entities_ablation{suf}" / "free").glob(f"*_all_metrics_{stem}_batch_v[012].json"))
        items = flagged = pts = flagged_pts = 0
        for f in files:
            for r in json.loads(f.read_text())["results"]:
                for e in r.get("free_discern_evaluation") or []:
                    s = e.get("clinical_significance_score") or 0
                    items, pts = items + 1, pts + s
                    if {e.get(k) for k in conc} <= ok and s >= 1:
                        flagged, flagged_pts = flagged + 1, flagged_pts + s
        reps = len({f.stem.split("_batch_")[-1] for f in files})
        rows.append({"model": label, "repeats": reps, "items_listed": items, "nonerror_items_scored": flagged,
                     "pct_items": round(100 * flagged / max(items, 1), 1),
                     "pct_of_total_score": round(100 * flagged_pts / max(pts, 1), 1)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    base, vs = baselines()
    tables = {"01_concordance_by_method": concordance(), "02_method_differences": method_differences(),
              "03_unscored_pairs": unscored_pairs(), "04_baseline_metrics": base,
              "05_free_vs_green_crimson": vs, "06_free_nonerror_items": nonerror_items()}
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}.csv  {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
