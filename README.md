# DISCERN

📄 **Paper (medRxiv preprint):** https://www.medrxiv.org/content/10.64898/2026.05.26.26353612v2

## Citation

If you use DISCERN, please cite the preprint:

```bibtex
@article{discern2026,
  title   = {DISCERN: A Clinical Impact-aware Framework for Radiology Report Comparison},
  author  = {Sharma, Rakesh and Beeche, Cameron and Dong, Jessie and Zhuang, Richard and
             Qu, Huaizhi and Zhang, Ruichen and Gangaram, Vineeth and Goswami, Pulak and
             Xin, Jiayi and Ballard, Jenna and Duda, Jeffery and Kahn, Jr., Charles E. and
             Goldberg, Ari and Sagreiya, Hersh and Long, Qi and Chen, Tianlong and
             Witschey, Walter},
  journal = {medRxiv},
  year    = {2026},
  doi     = {10.64898/2026.05.26.26353612},
  url     = {https://www.medrxiv.org/content/10.64898/2026.05.26.26353612v2}
}
```

**DISCERN** is an LLM-based framework for evaluating radiology reports at the
clinical entity level. Given a reference (ground truth) report and a candidate
report, DISCERN extracts radiology entities, compares their attributes
(diagnosis, location, severity, temporal), and scores clinical significance of
any discrepancies.

## Features

- **Full DISCERN**: 3-stage pipeline (entity extraction → attribute comparison → significance scoring)
- **mini-DISCERN**: Single-prompt evaluation for faster batch processing
- **NLP metrics**: BLEU-1, ROUGE-L, METEOR, BERTScore, RadGraph F1
- **Model-based metrics**: GREEN, CRIMSON (separate conda envs)
- **Flexible input**: CSV or JSON with arbitrary report pairs
- **Resume support**: Re-running with the same `--output` skips already-scored entries
- **Batch processing**: Auto-selects API batch (Databricks) or GPU batch (vLLM)

## Quick Start

### 1. Install

```bash
conda env create -f envs/discern.yml
conda activate discern
pip install -e .
```

### 2. Configure credentials

```bash
cp config.example.yaml config.yaml
# Edit config.yaml and fill in your credentials
```

### 3. Prepare your data

Your input file should be a CSV with at minimum two columns:

| reference | candidate |
|-----------|-----------|
| Heart is enlarged with bilateral pleural effusions... | Heart size is normal. Lungs are clear... |

Column name aliases also accepted: `ground_truth_raw`/`generated_raw`, `gt_report`/`candidate_report`.

### 4. Run

```bash
# Score with NLP metrics + DISCERN (uses model from config.yaml)
python scripts/run_metrics.py --input pairs.csv --output scored.json

# Quick test on first 5 rows
python scripts/run_metrics.py --input pairs.csv --output test.json --count 5

# NLP metrics only
python scripts/run_metrics.py --input pairs.csv --output scored.json --metrics nlp

# Specific model
python scripts/run_metrics.py --input pairs.csv --output scored.json \
    --model databricks-claude-sonnet-4-6
```

## Output Format

```json
{
  "_metadata": {"run_date": "2025-01-01", "model": "...", ...},
  "results": [
    {
      "sample_idx": 0,
      "ground_truth_raw": "...",
      "generated_raw": "...",
      "bleu": 0.42,
      "rouge": 0.55,
      "meteor": 0.48,
      "bertscore": 0.87,
      "discern_score": 4,
      "discern_evaluation": [...],
      "mini_discern_score": 3,
      "mini_discern_evaluation": [...]
    }
  ]
}
```

## Benchmark Datasets (ReXVal / RaDEvalX)

To evaluate against radiologist-annotated benchmarks:

```bash
# Download data from PhysioNet (credentialed access required)
# Place rexval_reports_long.csv in data/rexval/
# Place radevalx_report.csv in data/radevalx/

python scripts/run_discern.py --dataset both
```

## Robustness Evaluation

```bash
# Generate paraphrased reports
python scripts/run_paraphrase.py --output data/robustness/paraphrase.json

# Score DISCERN on paraphrased pairs
python scripts/run_discern_paraphrase.py --input data/robustness/paraphrase.json

# Build and score extreme-case baselines
python scripts/run_extreme_cases.py --output data/extreme_baseline.json
python scripts/run_metrics.py --input data/extreme_baseline.json \
    --output data/extreme_baseline_scored.json
```

## SLURM Jobs

See `jobs/` for ready-to-submit SLURM launchers:

| Script | Purpose |
|--------|---------|
| `jobs/run_custom.sh` | Run on your own dataset |
| `jobs/discern_api.sh` | Multi-model sweep via Databricks API |
| `jobs/discern_vllm.sh` | Multi-model sweep via vLLM on GPU nodes |
| `jobs/paraphrase_generate.sh` | Generate paraphrase robustness dataset |
| `jobs/paraphrase_eval.sh` | Evaluate DISCERN on paraphrased pairs |

## Environments

| File | Use case |
|------|----------|
| `envs/discern.yml` | Core DISCERN + NLP metrics (CPU/API) |
| `envs/vllm.yml` | Local GPU inference via vLLM |
| `envs/green.yml` | GREEN metric (separate env due to deps) |
| `envs/crimson.yml` | CRIMSON metric |
