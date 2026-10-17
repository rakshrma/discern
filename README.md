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
- **free-DISCERN**: Single-prompt evaluation with no entity list — the model lists only the errors it finds and the score is the sum of their significance (see [free-DISCERN](#free-discern))
- **Swappable entity list**: Full DISCERN reads its entity list from `config/entities.yaml` (or any list passed with `--entities-yaml`)
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

# free-DISCERN only (opt-in; see below)
python scripts/run_metrics.py --input pairs.csv --output scored.json \
    --metrics free_discern --model databricks-claude-sonnet-4-6

# Full DISCERN + mini-DISCERN, plus free-DISCERN
python scripts/run_metrics.py --input pairs.csv --output scored.json \
    --metrics discern --free-discern

# Full DISCERN with a different entity list
python scripts/run_metrics.py --input pairs.csv --output scored.json \
    --metrics discern --entities-yaml config/entities_original.yaml
```

`--model` selects the judge: names starting with `databricks-` are called through the Databricks API
(credentials in `config.yaml`); any other name is treated as a Hugging Face model id (e.g.
`google/gemma-4-E4B-it`) and run locally with vLLM, so run those on a GPU node. Local models are scored
in one vLLM batch; Databricks models are scored pair by pair. Re-running with the same `--output`
skips pairs that already have a score, so an interrupted run can be resumed.

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

`--model` works as for `run_metrics.py`; without it, `discern.default_model` from `config.yaml` is used.

Full DISCERN uses `config/entities.yaml`; pass another list with
`--entities-yaml`, e.g. `--entities-yaml config/entities_original.yaml` (the earlier 121-entity list).

## free-DISCERN

free-DISCERN drops the entity list. A single prompt gives the model the reference and candidate
reports and asks it to identify the findings, judge their concordance (diagnosis, location, severity,
temporal) and return **only the discrepancies**, each scored 1–4 for clinical significance with the same
rubric as mini-DISCERN. The score is the sum of those significances, computed in code (0 = no errors).
The prompt is `config/freeform_prompt.yaml`.

It is opt-in and never part of a metric group (`--metrics discern` / `all` do not include it):
request it with `--metrics free_discern`, add it with `--free-discern`, or set
`metrics.free_discern: true` in `config.yaml`.

On your own report pairs:

```bash
python scripts/run_metrics.py --input pairs.csv --output scored.json \
    --metrics free_discern --model databricks-claude-sonnet-4-6
```

On ReXVal / RaDEvalX:

```bash
# free-DISCERN only, via the Databricks API
python scripts/run_discern.py --dataset both --model databricks-claude-sonnet-4-6 \
    --skip-discern --skip-mini-discern --skip-nlp --skip-green \
    --free-discern --run-tag free_v0

# free-DISCERN only, local model via vLLM (on a GPU node)
VLLM_MAX_MODEL_LEN=25000 VLLM_TENSOR_PARALLEL_SIZE=1 \
python scripts/run_discern.py --dataset both --model google/gemma-4-E4B-it \
    --skip-discern --skip-mini-discern --skip-nlp --skip-green \
    --free-discern --run-tag free_v0
```

Drop the `--skip-*` flags to score free-DISCERN alongside the other metrics. Set `VLLM_MAX_MODEL_LEN`
explicitly for local models (the `config.yaml` default of 8192 can truncate long prompts); if vLLM's
automatic attention backend fails to build (seen with MedGemma), set `VLLM_ATTN_BACKEND=TRITON_ATTN`.

Each result entry gains:

```json
{
  "free_discern_score": 6,
  "free_discern_evaluation": [
    {
      "finding": "Airspace opacity in the posterior left lower lobe",
      "reference_presence": "POSITIVE",
      "candidate_presence": "NEGATIVE",
      "reference_text": "...",
      "candidate_text": "...",
      "diagnosis_concordance": "discordant",
      "location_concordance": "candidate-misses",
      "severity_concordance": "candidate-misses",
      "temporal_concordance": "not mentioned",
      "clinical_significance_score": 4,
      "rationale": "..."
    }
  ]
}
```

Pairs whose output cannot be validated after retries get `free_discern_score: null` and a
`free_discern_error`.

To run the original-list, updated-list and free-DISCERN arms for one model on a GPU node:

```bash
MODEL=google/gemma-4-E4B-it ARMS="original updated free" OUT_TAG=gemma4_e4b VLLM_TP=1 \
    sbatch -p dgx-b200 --gpus-per-node=1 -c 28 -t 06:00:00 jobs/entities_ablation.sh
```

Results go to `data/discern_runs/entities_ablation_<OUT_TAG>/<arm>/`. Compare the arms against the
radiologist annotations with `scripts/compare_arms.py --model <model_stem> --ablation-dir <dir>`.

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
| `jobs/entities_ablation.sh` | Original vs updated entity list vs free-DISCERN for one model (ReXVal + RaDEvalX, 3 repeats) |
| `jobs/paraphrase_generate.sh` | Generate paraphrase robustness dataset |
| `jobs/paraphrase_eval.sh` | Evaluate DISCERN on paraphrased pairs |

## Environments

| File | Use case |
|------|----------|
| `envs/discern.yml` | Core DISCERN + NLP metrics (CPU/API) |
| `envs/vllm.yml` | Local GPU inference via vLLM |
| `envs/green.yml` | GREEN metric (separate env due to deps) |
| `envs/crimson.yml` | CRIMSON metric |
