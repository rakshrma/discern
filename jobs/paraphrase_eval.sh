#!/bin/bash
#SBATCH -J paraphrase_eval
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# ── Environment ──────────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/green_score

DISCERN_ROOT=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
cd "$DISCERN_ROOT"

INPUT="${INPUT:-data/robustness/paraphrase_robustness_v2.json}"
OUTPUT_DIR="${OUTPUT_DIR:-data/discern_runs/paraphrase_eval}"
TRANSFORMS="${TRANSFORMS:-rephrase}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"

mkdir -p "$OUTPUT_DIR" logs

echo "[$(date)] Scoring paraphrased reports: transforms=$TRANSFORMS"

python scripts/run_discern_paraphrase.py \
    --input  "$INPUT" \
    --output "$OUTPUT_DIR" \
    --transforms $TRANSFORMS \
    --model  "$MODEL" \
    --max-concurrent "$MAX_CONCURRENT"

echo "[$(date)] Done: $OUTPUT_DIR"
