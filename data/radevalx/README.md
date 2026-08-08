# RaDEvalX Dataset

RaDEvalX is a radiology report evaluation benchmark with radiologist-annotated
error labels across multiple candidate report types.

## Download

RaDEvalX requires credentialed PhysioNet access:

1. Register and complete training at https://physionet.org
2. Navigate to the RaDEvalX dataset page
3. Download and place `radevalx_report.csv` in this directory

## Expected File

`radevalx_report.csv`

Required columns:
- `study_id` — unique study identifier
- `ground_truth` — ground truth (reference) radiology report
- `candidate_report` — candidate report to evaluate

## Usage

```bash
python scripts/run_discern.py --dataset radevalx
```
