#!/bin/bash
#SBATCH -J discern_api
#SBATCH --cpus-per-task=4
#SBATCH -t 24:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
# No GPU needed — all inference is via API (Anthropic/OpenAI/OpenRouter/Databricks)

# ── Environment ──────────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/green_score

DISCERN_ROOT=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
cd "$DISCERN_ROOT"

# ── Parameters ───────────────────────────────────────────────────────────────
INPUT="${INPUT:-data/rexval/rexval_reports_long.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-data/discern_runs}"
BACKEND="${BACKEND:-hf}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
TAG="${TAG:-batch_v0}"
REPEATS="${REPEATS:-1}"

mkdir -p "$OUTPUT_DIR" logs

# ── Run N repeats ─────────────────────────────────────────────────────────────
for i in $(seq 1 "$REPEATS"); do
    TAG_RUN="${TAG}_r${i}"
    MODEL_SLUG=$(echo "$MODEL" | tr '/' '_' | tr '-' '_')
    OUTPUT="${OUTPUT_DIR}/discern_${MODEL_SLUG}_${TAG_RUN}.json"

    echo "[$(date)] Starting DISCERN run ${i}/${REPEATS}: $MODEL → $OUTPUT"

    python scripts/run_discern.py \
        --input   "$INPUT" \
        --output  "$OUTPUT" \
        --mode    both \
        --backend "$BACKEND" \
        --model   "$MODEL" \
        --max-concurrent "$MAX_CONCURRENT" \
        --tag "$TAG_RUN"

    echo "[$(date)] Finished run ${i}: $OUTPUT"
done

echo "All runs complete."
