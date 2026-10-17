"""
compare_vs_baselines.py — radiologist concordance of the re-run DISCERN arms vs the April baseline
metrics (BLEU-1, ROUGE-L, METEOR, BERTScore, RadGraph-F1, GREEN, CRIMSON).

Baselines are read from the April Sonnet run that scored every metric on every pair:
  reads/data/discern_runs/v0_llm_tests/{rexval,radevalx}_all_metrics_databricks_claude_sonnet_4_6.json
(GREEN and CRIMSON use their own judge models, so they do not depend on the DISCERN judge.)
Baselines are similarity scores (higher = fewer errors) and are NEGATED here, so for every metric a
positive ρ/τ means agreement with the radiologist error counts.

Per model × arm, the DISCERN score is averaged over the arm's repeats and correlated on the pairs
that arm scored in every repeat. free-DISCERN is compared with GREEN and CRIMSON on exactly those
pairs by a paired cluster bootstrap (clusters = studies). Ground truth and the correlation helper
are those of reads/src/analyze_discern_repeats_and_correlation.py, as in compare_arms.py.

Outputs (in discern/data/discern_runs/):
  vs_baselines_corr.csv       ρ, τ, n for every baseline and every model × arm × dataset × ground truth
  vs_baselines_bootstrap.csv  free − GREEN and free − CRIMSON: Δρ, Δτ, 95% CI, p

Usage:  envs/vllm/bin/python discern/scripts/compare_vs_baselines.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from compare_arms import _table
from compare_entity_versions import _load_analysis_module
from tabulate_all_models import MODELS, RUNS

APRIL_ALL = RUNS.parents[2] / "reads" / "data" / "discern_runs" / "v0_llm_tests"
BASELINES = {"bleu": "BLEU-1", "rouge": "ROUGE-L", "meteor": "METEOR", "bertscore": "BERTScore",
             "radgraph": "RadGraph-F1", "green": "GREEN", "crimson_score": "CRIMSON"}
GTS = ("gt_total", "gt_sig", "gt_insig")
ARMS = ("original", "updated", "free")
N_BOOT = 2000


def _keys(ds: str) -> list[str]:
    return ["study_id", "candidate_reporter"] if ds == "rexval" else ["study_id"]


def _str_keys(df: pd.DataFrame, ds: str) -> pd.DataFrame:
    df = df.copy()
    for k in _keys(ds):
        df[k] = df[k].astype(str)
    return df


def load_baselines(ana, ds: str) -> pd.DataFrame:
    res = json.loads((APRIL_ALL / f"{ds}_all_metrics_databricks_claude_sonnet_4_6.json").read_text())
    res = res["results"] if isinstance(res, dict) else res
    rows = [{**{k: r.get(k) for k in _keys(ds)},
             **{BASELINES[m]: (-r[m] if r.get(m) is not None else np.nan) for m in BASELINES}}
            for r in res]
    df = _str_keys(pd.DataFrame(rows), ds)
    if ds == "radevalx":
        gt = ana.build_radevalx_gt().rename(columns={"report_id": "study_id"})
    else:
        gt = ana.build_rexval_gt()
        gt = gt[gt["rater"] == "avg"].rename(columns={"candidate_type": "candidate_reporter"})
    gt = _str_keys(gt[[*_keys(ds), *GTS]], ds)
    return df.merge(gt, on=_keys(ds))


def arm_scores(ana, ds: str, stem: str, abl: Path, arm: str) -> pd.DataFrame | None:
    """Score averaged over the arm's complete repeats; pairs unscored in any repeat dropped."""
    reps = [r for r in (0, 1, 2)
            if all((abl / arm / f"{d}_all_metrics_{stem}_batch_v{r}.json").exists() for d in ("rexval", "radevalx"))]
    if not reps:
        return None
    tabs = [_str_keys(_table(ana, ds, r, stem, abl, [arm]), ds)[[*_keys(ds), f"{arm}_score"]] for r in reps]
    m = pd.concat(tabs).groupby(_keys(ds))[f"{arm}_score"].agg(["mean", "count"]).reset_index()
    m = m[m["count"] == len(reps)].rename(columns={"mean": "score"})
    m.attrs["reps"] = len(reps)
    return m[[*_keys(ds), "score"]]


def main():
    ana = _load_analysis_module()
    rng = np.random.default_rng(0)
    corr_rows, boot_rows = [], []
    for ds in ("rexval", "radevalx"):
        base = load_baselines(ana, ds)
        for name in BASELINES.values():
            for g in GTS:
                ok = base[name].notna()
                corr_rows.append({"dataset": ds, "method": name, "model": "—", "gt": g, "n": int(ok.sum()),
                                  "rho": spearmanr(base[name][ok], base[g][ok])[0],
                                  "tau": kendalltau(base[name][ok], base[g][ok])[0]})
        for label, suf, stem in MODELS:
            abl = RUNS / f"entities_ablation{suf}"
            for arm in ARMS:
                sc = arm_scores(ana, ds, stem, abl, arm)
                if sc is None:
                    continue
                m = sc.merge(base, on=_keys(ds))
                for g in GTS:
                    corr_rows.append({"dataset": ds, "method": f"DISCERN-{arm}", "model": label, "gt": g,
                                      "n": len(m), "reps": sc.attrs["reps"],
                                      "rho": spearmanr(m["score"], m[g])[0], "tau": kendalltau(m["score"], m[g])[0]})
                if arm != "free":
                    continue
                ids = m["study_id"].values
                uniq = np.unique(ids)
                idx = {c: np.flatnonzero(ids == c) for c in uniq}
                draws = [np.concatenate([idx[c] for c in rng.choice(uniq, len(uniq), replace=True)])
                         for _ in range(N_BOOT)]
                for g in GTS:
                    y = m[g].values
                    for ref in ("GREEN", "CRIMSON"):
                        xa, xb = m["score"].values, m[ref].values
                        row = {"dataset": ds, "model": label, "gt": g, "comparison": f"free - {ref}", "n": len(m)}
                        for stat, f in (("rho", spearmanr), ("tau", kendalltau)):
                            boot = np.array([f(xa[t], y[t])[0] - f(xb[t], y[t])[0] for t in draws])
                            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                            p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
                            row.update({f"d_{stat}": f(xa, y)[0] - f(xb, y)[0],
                                        f"{stat}_ci95": f"[{lo:+.3f}, {hi:+.3f}]", f"p_{stat}": min(p, 1.0)})
                        boot_rows.append(row)
                print(f"[{ds}] {label}: done", flush=True)
    corr, boot = pd.DataFrame(corr_rows), pd.DataFrame(boot_rows)
    corr.to_csv(RUNS / "vs_baselines_corr.csv", index=False)
    boot.to_csv(RUNS / "vs_baselines_bootstrap.csv", index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format, "display.max_rows", 500):
        print(corr[corr["model"] == "—"].pivot_table(index=["dataset", "method"], columns="gt", values=["tau", "rho"], sort=False))
        print(boot.to_string(index=False))
    print(f"Saved → {RUNS / 'vs_baselines_corr.csv'} (+ vs_baselines_bootstrap.csv)")


if __name__ == "__main__":
    main()
