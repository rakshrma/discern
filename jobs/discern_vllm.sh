#!/bin/bash
#SBATCH -J discern_vllm
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH -t 24:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --partition=b200-mig90

# ── Environment ──────────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
# conda activate /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/green_score
conda activate reads
DISCERN_ROOT=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
cd "$DISCERN_ROOT"

# ── Parameters ───────────────────────────────────────────────────────────────
INPUT="${INPUT:-data/rexval/rexval_reports_long.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-data/discern_runs}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
TP="${TENSOR_PARALLEL:-2}"
TAG="${TAG:-vllm_v0}"

mkdir -p "$OUTPUT_DIR" logs

MODEL_SLUG=$(echo "$MODEL" | tr '/' '_' | tr '-' '_')
OUTPUT="${OUTPUT_DIR}/discern_${MODEL_SLUG}_${TAG}.json"

echo "[$(date)] Starting vLLM DISCERN: $MODEL (tp=$TP) → $OUTPUT"

python scripts/run_discern.py \
    --input   "$INPUT" \
    --output  "$OUTPUT" \
    --mode    both \
    --backend hf \
    --model   "$MODEL" \
    --max-concurrent 1 \
    --tag "$TAG"

echo "[$(date)] Done: $OUTPUT"
