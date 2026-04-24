#!/bin/bash
#SBATCH -J crimson
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH -t 8:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --partition=b200-mig90

# MUST run in the 'crimson' conda env (incompatible transformers/torch versions)

# ── Environment ──────────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/envs/crimson

DISCERN_ROOT=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
cd "$DISCERN_ROOT"

INPUT="${INPUT:?ERROR: set INPUT}"
OUTPUT="${OUTPUT:?ERROR: set OUTPUT}"
BATCH_SIZE="${BATCH_SIZE:-4}"

mkdir -p "$(dirname "$OUTPUT")" logs

echo "[$(date)] Running CRIMSON on $INPUT (batch_size=$BATCH_SIZE)"

python src/metrics/_crimson_runner.py \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    --batch-size "$BATCH_SIZE"

echo "[$(date)] Done: $OUTPUT"
