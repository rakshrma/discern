#!/usr/bin/env python3
"""
DISCERN Quick-Start Example
============================
Runs the full DISCERN evaluation pipeline on a pair of sample radiology
reports using meta-llama/Llama-3.1-8B-Instruct (via HuggingFace).

Requirements:
  - Save your HuggingFace token to config/.hftoken
  - pip install -r requirements.txt
  - A CUDA-capable GPU is recommended (model loads in bfloat16)

Usage:
  python run_example.py
"""

import json
import os
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluate_reports import run_evaluation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN_PATH = "config/.hftoken"

ENTITY_PROMPT       = "config/entity_extraction_prompt.yaml"
ENTITIES_YAML       = "config/entities.yaml"
ATTRIBUTE_PROMPT    = "config/attribute_extraction_prompt.yaml"
SIGNIFICANCE_PROMPT = "config/significance_prompt.yaml"

# ---------------------------------------------------------------------------
# Sample reports
# ---------------------------------------------------------------------------
GROUND_TRUTH_REPORT = """
FINDINGS:
The cardiac silhouette is enlarged. There is a moderate right-sided pleural
effusion. Patchy airspace opacity is present in the right lower lobe,
consistent with pneumonia. No pneumothorax is identified. The mediastinum
is within normal limits. Osseous structures are intact.

IMPRESSION:
1. Cardiomegaly.
2. Moderate right pleural effusion.
3. Right lower lobe pneumonia.
"""

CANDIDATE_REPORT = """
FINDINGS:
The heart size appears normal. There is a small left-sided pleural effusion.
The lungs are clear without focal consolidation. No pneumothorax is seen.
Mild mediastinal widening is noted. Osseous structures are unremarkable.

IMPRESSION:
1. Normal cardiac size.
2. Small left pleural effusion.
3. Mild mediastinal widening.
"""

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
def main():
    if not Path(HF_TOKEN_PATH).exists():
        print(f"[ERROR] HuggingFace token not found at '{HF_TOKEN_PATH}'.")
        print("        Create the file and paste your HF token into it.")
        sys.exit(1)

    print("=" * 60)
    print("DISCERN Evaluation Pipeline")
    print(f"Model : {MODEL}")
    print("=" * 60)
    print("\n[Ground Truth Report]")
    print(GROUND_TRUTH_REPORT.strip())
    print("\n[Candidate Report]")
    print(CANDIDATE_REPORT.strip())
    print("\nRunning evaluation — this may take a few minutes on first run")
    print("(model weights are downloaded and cached on first use)...\n")

    discern_evaluation, discern_score = run_evaluation(
        report_text=GROUND_TRUTH_REPORT,
        candidate_text=CANDIDATE_REPORT,
        model=MODEL,
        token_path=HF_TOKEN_PATH,
        prompt_yaml_path=ENTITY_PROMPT,
        entities_yaml_path=ENTITIES_YAML,
        attribute_prompt_path=ATTRIBUTE_PROMPT,
        significance_yaml_path=SIGNIFICANCE_PROMPT,
    )

    # ---------------------------------------------------------------------------
    # Display results
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print(f"DISCERN Score: {discern_score:.2f}")
    print("=" * 60)
    print(f"\nEntities evaluated: {len(discern_evaluation)}\n")

    for i, entity in enumerate(discern_evaluation, 1):
        print(f"[{i}] {entity['entity']}")
        print(f"     Reference finding : {entity.get('reference_report_finding') or '(not mentioned)'}")
        print(f"     Candidate finding : {entity.get('candidate_report_finding') or '(not mentioned)'}")
        if entity.get("discrepancy_type"):
            print(f"     Discrepancy type  : {entity['discrepancy_type']}")
        else:
            print(f"     Diagnosis         : {entity.get('diagnosis_concordance')}")
            print(f"     Location          : {entity.get('location_concordance')}")
            print(f"     Severity          : {entity.get('severity_concordance')}")
            print(f"     Temporal          : {entity.get('temporal_comparison')}")
        print(f"     Significance      : {entity.get('significance_score')} / 4")
        print(f"     Rationale         : {entity.get('rationale')}")
        print()

    print("=" * 60)
    print("Full JSON output:")
    print(json.dumps(discern_evaluation, indent=2))


if __name__ == "__main__":
    main()
