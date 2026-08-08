"""
Quick smoke test: run all metrics on a single hard-coded report pair.

Usage:
  # From discern/ repo root:
  python tests/test_one_pair.py               # all metrics from config.yaml
  python tests/test_one_pair.py --only nlp    # NLP only
  python tests/test_one_pair.py --only discern
  python tests/test_one_pair.py --only green
  python tests/test_one_pair.py --only crimson
  python tests/test_one_pair.py --only mini_discern
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

REFERENCE = (
    "PA and lateral chest radiograph. "
    "There is a moderate left pleural effusion with associated left lower lobe atelectasis. "
    "The right lung is clear. "
    "The cardiac silhouette is mildly enlarged. "
    "No pneumothorax. "
    "The mediastinum is within normal limits. "
    "IMPRESSION: Moderate left pleural effusion with left lower lobe atelectasis. "
    "Mild cardiomegaly."
)

CANDIDATE = (
    "PA and lateral chest radiograph. "
    "There is a small left pleural effusion. "
    "No consolidation or pneumothorax identified. "
    "The cardiac silhouette is normal in size. "
    "The mediastinum is unremarkable. "
    "IMPRESSION: Small left pleural effusion."
)


def load_config():
    import yaml
    cfg_path = _ROOT / "config.yaml"
    if not cfg_path.exists():
        print("ERROR: config.yaml not found. Copy config.example.yaml → config.yaml and fill credentials.")
        sys.exit(1)
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    # Export credentials as env vars for internal pipeline calls
    creds = cfg.get("credentials") or {}
    if creds.get("databricks_host"):
        import os
        os.environ["DATABRICKS_SERVING_ENDPOINTS_URL"] = creds["databricks_host"]
    if creds.get("hf_token"):
        import os
        os.environ["HF_TOKEN"] = creds["hf_token"]
    return cfg


def run_nlp():
    from metrics.nlp import (compute_bleu1, compute_rougel, compute_meteor,
                               compute_bertscore_single)
    print("\n── NLP Metrics ──────────────────────────────────────")
    bleu    = compute_bleu1(CANDIDATE, REFERENCE)
    rouge   = compute_rougel(CANDIDATE, REFERENCE)
    meteor  = compute_meteor(CANDIDATE, REFERENCE)
    bscore  = compute_bertscore_single(CANDIDATE, REFERENCE)
    print(f"  BLEU-1:    {bleu:.4f}")
    print(f"  ROUGE-L:   {rouge:.4f}")
    print(f"  METEOR:    {meteor:.4f}")
    print(f"  BERTScore: {bscore:.4f}")
    return {"bleu": bleu, "rouge": rouge, "meteor": meteor, "bertscore": bscore}


def run_discern(cfg, model_override=None):
    from evaluate_reports import run_evaluation
    creds = cfg.get("credentials") or {}
    db_token = creds.get("databricks_token", "")
    model    = model_override or (cfg.get("discern") or {}).get("default_model", "databricks-claude-sonnet-4-6")

    print(f"\n── Full DISCERN (model={model}) ─────────────────────")
    discern_eval, discern_score = run_evaluation(
        report_text=REFERENCE,
        candidate_text=CANDIDATE,
        model=model,
        token_path=db_token,
        prompt_yaml_path=str(_ROOT / "config/entity_extraction_prompt.yaml"),
        entities_yaml_path=str(_ROOT / "config/entities.yaml"),
        attribute_prompt_path=str(_ROOT / "config/attribute_extraction_prompt.yaml"),
        significance_yaml_path=str(_ROOT / "config/significance_prompt.yaml"),
        max_tokens=5000,
    )
    print(f"  DISCERN score: {discern_score}")
    print(f"  Entities evaluated: {len(discern_eval)}")
    for entity in discern_eval[:3]:
        print(f"    {json.dumps(entity, default=str)}")
    return {"discern_score": discern_score}


def run_mini_discern(cfg, model_override=None):
    from evaluate_single_prompt import evaluate_reports
    creds = cfg.get("credentials") or {}
    db_token = creds.get("databricks_token", "")
    model    = model_override or (cfg.get("discern") or {}).get("default_model", "databricks-claude-sonnet-4-6")

    print(f"\n── mini-DISCERN (model={model}) ─────────────────────")
    result = evaluate_reports(
        reference_report=REFERENCE,
        candidate_report=CANDIDATE,
        entity_list_path=str(_ROOT / "config/diagnosis.yaml"),
        prompt_path=str(_ROOT / "config/merged_prompt.yaml"),
        model=model,
        token_path=db_token,
        max_tokens=8000,
        temperature=0.1,
        max_retries=3,
    )
    score = int(sum(e.clinical_significance_score for e in result))
    print(f"  mini-DISCERN score: {score}")
    print(f"  Entities: {len(result)}")
    for e in result[:3]:
        print(f"    {e.entity_name}: sig={e.clinical_significance_score} ({e.diagnosis_concordance})")
    return {"mini_discern_score": score}


def run_green(cfg):
    from metrics.green import compute_green
    conda_envs = cfg.get("conda_envs") or {}
    green_python = conda_envs.get("green") or None
    green_model  = (cfg.get("checkpoints") or {}).get("green_model", "StanfordAIMI/GREEN-radllama2-7b")

    print(f"\n── GREEN (model={green_model}) ──────────────────────")
    scores = compute_green(
        candidates=[CANDIDATE],
        references=[REFERENCE],
        model_name=green_model,
        python_bin=green_python,
    )
    print(f"  GREEN score: {scores[0]}")
    return {"green": scores[0]}


def run_crimson(cfg):
    from metrics.crimson import compute_crimson
    conda_envs = cfg.get("conda_envs") or {}
    crimson_python = conda_envs.get("crimson") or None
    crimson_model  = (cfg.get("checkpoints") or {}).get("crimson_model",
                       "rajpurkarlab/medgemma-4b-it-crimson")

    print(f"\n── CRIMSON (model={crimson_model}) ──────────────────")
    scores = compute_crimson(
        candidates=[CANDIDATE],
        references=[REFERENCE],
        model_name=crimson_model,
        python_bin=crimson_python,
    )
    print(f"  CRIMSON score: {scores[0]}")
    return {"crimson": scores[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        choices=["nlp", "discern", "mini_discern", "green", "crimson"],
                        help="Run only a specific metric group")
    parser.add_argument("--model", default=None,
                        help="Override the model from config.yaml (e.g. google/gemma-4-31B-it)")
    args = parser.parse_args()

    cfg = load_config()

    print("Test report pair:")
    print(f"  Reference: {REFERENCE[:80]}...")
    print(f"  Candidate: {CANDIDATE[:80]}...")
    if args.model:
        print(f"  Model override: {args.model}")

    scores = {}
    only = args.only

    if only is None or only == "nlp":
        scores.update(run_nlp())

    if only is None or only == "discern":
        scores.update(run_discern(cfg, model_override=args.model))

    if only is None or only == "mini_discern":
        scores.update(run_mini_discern(cfg, model_override=args.model))

    if only is None or only == "green":
        scores.update(run_green(cfg))

    if only is None or only == "crimson":
        scores.update(run_crimson(cfg))

    print("\n── Summary ──────────────────────────────────────────")
    for k, v in scores.items():
        print(f"  {k:>20s}: {v}")


if __name__ == "__main__":
    main()
