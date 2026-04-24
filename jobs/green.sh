#!/bin/bash
#SBATCH -J green_score
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH -t 8:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --partition=b200-mig90

# ── Environment ──────────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/green_score

DISCERN_ROOT=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
cd "$DISCERN_ROOT"

INPUT="${INPUT:?ERROR: set INPUT}"
OUTPUT="${OUTPUT:?ERROR: set OUTPUT}"

mkdir -p "$(dirname "$OUTPUT")" logs

echo "[$(date)] Running GREEN on $INPUT"

python scripts/run_metrics.py \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    --metrics green

echo "[$(date)] Done: $OUTPUT"
