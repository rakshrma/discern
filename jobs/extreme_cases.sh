#!/bin/bash
#SBATCH -J extreme_cases
#SBATCH --cpus-per-task=4
#SBATCH -t 8:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# Generate extreme/adversarial report variants and score with DISCERN.
# Extreme cases = maximally discordant (wrong everything) and
# maximally concordant (exact copy) report pairs — used to verify
# DISCERN score range and calibration.

# ── Environment ──────────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/green_score

DISCERN_ROOT=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
cd "$DISCERN_ROOT"

INPUT="${INPUT:-data/rexval/rexval_reports_long.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-data/discern_runs/extreme_cases}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
BACKEND="${BACKEND:-openrouter}"

mkdir -p "$OUTPUT_DIR" logs

echo "[$(date)] Generating extreme case variants..."

python scripts/run_extreme_cases.py \
    --input     "$INPUT" \
    --output    "$OUTPUT_DIR" \
    --model     "$MODEL" \
    --backend   "$BACKEND"

echo "[$(date)] Done: $OUTPUT_DIR"
