# ReXVal Dataset

ReXVal (Radiology Report Expert-Level Evaluation) is a dataset of chest X-ray
radiology report pairs annotated by radiologists for error categories.

## Download

ReXVal requires credentialed PhysioNet access:

1. Register and complete training at https://physionet.org
2. Navigate to the ReXVal dataset page
3. Download and place `rexval_reports_long.csv` in this directory

## Expected File

`rexval_reports_long.csv`

Required columns:
- `study_id` — unique study identifier
- `gt_report` — ground truth (reference) radiology report
- `candidate_report` — candidate report to evaluate
- `candidate_reporter` — identifier of the report generator (radiologist or model)

## Usage

```bash
python scripts/run_discern.py --dataset rexval
```
