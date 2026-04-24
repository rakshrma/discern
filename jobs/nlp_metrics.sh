#!/bin/bash
#SBATCH -J nlp_metrics
#SBATCH --cpus-per-task=8
#SBATCH -t 4:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
# CPU only — no GPU needed for BLEU/ROUGE/METEOR/BERTScore/RadGraph/RaTEScore

# ── Environment ──────────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/green_score

DISCERN_ROOT=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
cd "$DISCERN_ROOT"

# ── Parameters ───────────────────────────────────────────────────────────────
INPUT="${INPUT:?ERROR: set INPUT}"
OUTPUT="${OUTPUT:?ERROR: set OUTPUT}"

mkdir -p "$(dirname "$OUTPUT")" logs

echo "[$(date)] Running NLP metrics on $INPUT"

python scripts/run_metrics.py \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    --metrics nlp bertscore ratescore radgraph

echo "[$(date)] Done: $OUTPUT"
