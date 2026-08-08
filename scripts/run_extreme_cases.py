#!/usr/bin/env python3
"""
build_extreme_baseline.py
=========================
Generate extreme-case (lower-bound) pairs for judge robustness evaluation.

For each unique ground-truth report in rexval (50) and radevalx (100), create
three pairs:
  - "normal"       : GT vs "Normal." (minimal single-word report)
  - "all_negative" : GT vs a comprehensive report negating every diagnosis
                     from the DISCERN taxonomy (diagnosis.yaml)
  - "all_positive" : GT vs a comprehensive report asserting every diagnosis
                     from the DISCERN taxonomy as present

Output JSON is inference_all_metrics.py compatible.

Total entries:
  rexval   : 50 × 3 = 150
  radevalx : 100 × 3 = 300
  Grand total: 450

Usage
-----
    python scripts/run_extreme_cases.py \\
        --rexval   data/rexval/rexval_reports_long.csv \\
        --radevalx data/radevalx/radeval_total.csv \\
        --output   data/extreme_baseline.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Fixed extreme-baseline reports
# Diagnoses drawn from config/diagnosis.yaml
# ---------------------------------------------------------------------------

# Minimal single-word report.
NORMAL_REPORT = "Normal."

# Every diagnosis from diagnosis.yaml explicitly negated.
ALL_NEGATIVE_REPORT = (
    # Congenital Disease
    "No congenital lung disease. "
    "No congenital vascular disease. "
    "No congenital heart disease. "
    # Infectious Pulmonary Disease
    "No pneumonia. "
    "No tuberculosis. "
    "No other pulmonary infection. "
    # Pulmonary Neoplasm
    "No primary lung malignancy. "
    "No pulmonary metastases. "
    "No other pulmonary neoplasm. "
    # Lymphoproliferative Disease
    "No lymphoproliferative disease or mediastinal lymphadenopathy. "
    # Other Pulmonary Diagnosis
    "No interstitial lung disease. "
    "No sarcoidosis. "
    "No asbestos-related disease or pleural plaques. "
    "No pneumoconiosis. "
    "No pulmonary edema. "
    "No ARDS. "
    "No aspiration. "
    "No iatrogenic lung disease. "
    "No COPD or hyperinflation. "
    "No pulmonary vasculitis. "
    "No pulmonary hypertension. "
    "No pulmonary thromboembolic disease. "
    "No miscellaneous pulmonary disease. "
    # Cardiac Disease
    "No valvular heart disease. "
    "No myocardial disease. "
    "No pericardial effusion or pericardial disease. "
    "No congestive heart failure. "
    "No other cardiac disease. "
    # Aortic Disease
    "No aortic dissection or aneurysm. "
    "No other aortic disease. "
    # Miscellaneous
    "No traumatic injury. "
    "No post-treatment change. "
    "No miscellaneous disease. "
    "No acute cardiopulmonary abnormality."
)

# Every diagnosis from diagnosis.yaml explicitly asserted as present.
ALL_POSITIVE_REPORT = (
    # Congenital Disease
    "Congenital lung disease with abnormal pulmonary segmentation. "
    "Congenital vascular disease with anomalous pulmonary venous return. "
    "Congenital heart disease with septal defect. "
    # Infectious Pulmonary Disease
    "Bilateral lower lobe pneumonia with dense consolidation. "
    "Apical fibronodular opacities and cavitation consistent with tuberculosis. "
    "Multifocal airspace opacities consistent with additional pulmonary infection. "
    # Pulmonary Neoplasm
    "Right upper lobe spiculated mass consistent with primary lung malignancy. "
    "Bilateral pulmonary metastases with multiple nodules. "
    "Endobronchial lesion consistent with other pulmonary neoplasm. "
    # Lymphoproliferative Disease
    "Bulky mediastinal and bilateral hilar lymphadenopathy consistent with "
    "lymphoproliferative disease. "
    # Other Pulmonary Diagnosis
    "Bilateral reticular opacities and honeycombing consistent with interstitial "
    "lung disease. "
    "Peribronchovascular nodularity consistent with sarcoidosis. "
    "Pleural plaques and calcifications consistent with asbestos-related disease. "
    "Increased parenchymal density consistent with pneumoconiosis. "
    "Bilateral interstitial and alveolar opacities consistent with pulmonary edema. "
    "Diffuse bilateral alveolar opacities consistent with ARDS. "
    "Bilateral dependent opacities consistent with aspiration pneumonitis. "
    "Radiation fibrosis and post-procedural changes consistent with iatrogenic "
    "lung disease. "
    "Bilateral hyperinflation and flattened hemidiaphragms consistent with COPD. "
    "Bilateral nodular infiltrates consistent with pulmonary vasculitis. "
    "Enlarged main pulmonary artery consistent with pulmonary hypertension. "
    "Bilateral wedge-shaped peripheral opacities consistent with pulmonary "
    "thromboembolic disease. "
    "Diffuse parenchymal abnormality consistent with miscellaneous pulmonary disease. "
    # Cardiac Disease
    "Mitral annular calcification consistent with valvular heart disease. "
    "Globular cardiomegaly consistent with myocardial disease. "
    "Enlarged cardiac silhouette with pericardial effusion. "
    "Vascular congestion and bilateral pleural effusions consistent with "
    "congestive heart failure. "
    "Abnormal cardiac contour consistent with other cardiac disease. "
    # Aortic Disease
    "Widened mediastinum with loss of aortic knob definition consistent with "
    "aortic dissection. "
    "Ectatic and tortuous thoracic aorta consistent with other aortic disease. "
    # Miscellaneous
    "Multiple bilateral rib fractures consistent with trauma. "
    "Surgical clips and lobectomy changes consistent with post-treatment change. "
    "Additional incidental finding consistent with miscellaneous disease."
)

VARIANTS = {
    "normal":       NORMAL_REPORT,
    "all_negative": ALL_NEGATIVE_REPORT,
    "all_positive": ALL_POSITIVE_REPORT,
}


# ---------------------------------------------------------------------------
# Loaders (identical logic to paraphrase_reports.py)
# ---------------------------------------------------------------------------

def _load_rexval(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open(encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    seen: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    for row_idx, r in enumerate(raw_rows):
        sid = r["study_id"]
        gt  = r["gt_report"]
        if sid not in seen:
            seen[sid] = len(rows)
            rows.append({
                "source_row_idx":   row_idx,
                "source_report_id": sid,
                "source_gt_key":    seen[sid],
                "ground_truth_raw": gt,
            })
    return rows


def _load_radevalx(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open(encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    seen: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    for row_idx, r in enumerate(raw_rows):
        rid = r["study_id"]
        gt  = r["ground_truth"]
        if rid not in seen:
            seen[rid] = len(rows)
            rows.append({
                "source_row_idx":   row_idx,
                "source_report_id": rid,
                "source_gt_key":    seen[rid],
                "ground_truth_raw": gt,
            })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rexval",
                    default=str(Path(__file__).parent.parent / "data/rexval/rexval_reports_long.csv"))
    ap.add_argument("--radevalx",
                    default=str(Path(__file__).parent.parent / "data/radevalx/radevalx_report.csv"))
    ap.add_argument("--output",   default=str(Path(__file__).parent.parent / "data/extreme_baseline.json"))
    args = ap.parse_args()

    sources = [
        ("rexval",   Path(args.rexval),   _load_rexval),
        ("radevalx", Path(args.radevalx), _load_radevalx),
    ]

    results: List[Dict[str, Any]] = []
    sample_idx = 0

    for source_name, csv_path, loader in sources:
        rows = loader(csv_path)
        print(f"[{source_name}] loaded {len(rows)} unique GTs")

        for variant_name, generated_text in VARIANTS.items():
            for row in rows:
                results.append({
                    "sample_idx":        sample_idx,
                    "ground_truth_raw":  row["ground_truth_raw"],
                    "generated_raw":     generated_text,
                    "variant":           variant_name,
                    "source":            source_name,
                    "source_report_id":  row["source_report_id"],
                    "source_row_idx":    row["source_row_idx"],
                    "source_gt_key":     row["source_gt_key"],
                })
                sample_idx += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"[done] wrote {len(results)} entries → {output_path}")
    for source_name, _, _ in sources:
        for variant_name in VARIANTS:
            n = sum(1 for r in results if r["source"] == source_name and r["variant"] == variant_name)
            print(f"  {source_name}/{variant_name}: {n}")


if __name__ == "__main__":
    main()
