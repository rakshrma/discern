#!/bin/bash
#SBATCH -J paraphrase_gen
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
# No GPU needed — uses API backend for LLM paraphrasing

# ── Environment ──────────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/green_score

DISCERN_ROOT=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
cd "$DISCERN_ROOT"

REXVAL="${REXVAL:-data/rexval/rexval_reports_long.csv}"
RADEVALX="${RADEVALX:-data/radevalx/radeval_total.csv}"
CHEXPERT="${CHEXPERT:-}"
OUTPUT="${OUTPUT:-data/robustness/paraphrase_robustness_v2.json}"
MODEL="${MODEL:-google/gemma-4-31B-it}"

mkdir -p "$(dirname "$OUTPUT")" logs

echo "[$(date)] Generating paraphrase dataset → $OUTPUT"

python scripts/run_paraphrase.py \
    --rexval   "$REXVAL" \
    --radevalx "$RADEVALX" \
    ${CHEXPERT:+--chexpert "$CHEXPERT"} \
    --output   "$OUTPUT" \
    --model    "$MODEL" \
    --seed 42

echo "[$(date)] Done: $OUTPUT"
