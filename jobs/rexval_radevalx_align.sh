#!/bin/bash
#SBATCH -J rexval_radevalx_align
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# Run full metric suite on ReXVal + RaDEvalX benchmark datasets,
# then compute alignment (Spearman / Kendall tau) with human annotations.

# ── Environment ──────────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/green_score

DISCERN_ROOT=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
cd "$DISCERN_ROOT"

MODEL="${MODEL:-google/gemma-4-31B-it}"
BACKEND="${BACKEND:-hf}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"

mkdir -p data/discern_runs logs

# ── ReXVal ────────────────────────────────────────────────────────────────────
echo "[$(date)] Scoring ReXVal..."
python scripts/run_metrics.py \
    --input  data/rexval/rexval_reports_long.csv \
    --output data/discern_runs/rexval_all_metrics.json \
    --metrics all \
    --backend "$BACKEND" \
    --model "$MODEL" \
    --max-concurrent "$MAX_CONCURRENT" \
    --tag "rexval_align"

# ── RaDEvalX ──────────────────────────────────────────────────────────────────
echo "[$(date)] Scoring RaDEvalX..."
python scripts/run_metrics.py \
    --input  data/radevalx/radeval_total.csv \
    --output data/discern_runs/radevalx_all_metrics.json \
    --metrics all \
    --backend "$BACKEND" \
    --model "$MODEL" \
    --max-concurrent "$MAX_CONCURRENT" \
    --tag "radevalx_align"

# ── Correlation analysis ──────────────────────────────────────────────────────
echo "[$(date)] Computing correlations..."
python analysis/correlation.py \
    --rexval-scores    data/discern_runs/rexval_all_metrics.json \
    --rexval-gt        data/rexval/ \
    --radevalx-scores  data/discern_runs/radevalx_all_metrics.json \
    --radevalx-gt      data/radevalx/ \
    --output-dir       data/analysis/

echo "[$(date)] Alignment analysis complete. Results in data/analysis/"
