#!/bin/bash
#SBATCH -J discern_reliability
#SBATCH --cpus-per-task=4
#SBATCH -t 48:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# Runs DISCERN N times on the same input to measure inter-run reliability (ICC).

# ── Environment ──────────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/green_score

DISCERN_ROOT=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
cd "$DISCERN_ROOT"

INPUT="${INPUT:-data/rexval/rexval_reports_long.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-data/discern_runs/repeat_reliability}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
BACKEND="${BACKEND:-hf}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
N_REPEATS="${N_REPEATS:-3}"

mkdir -p "$OUTPUT_DIR" logs

for i in $(seq 1 "$N_REPEATS"); do
    MODEL_SLUG=$(echo "$MODEL" | tr '/' '_' | tr '-' '_')
    OUTPUT="${OUTPUT_DIR}/${MODEL_SLUG}_repeat_${i}.json"
    echo "[$(date)] Reliability repeat $i/$N_REPEATS → $OUTPUT"

    python scripts/run_discern.py \
        --input  "$INPUT" \
        --output "$OUTPUT" \
        --mode   both \
        --backend "$BACKEND" \
        --model  "$MODEL" \
        --max-concurrent "$MAX_CONCURRENT" \
        --tag "reliability_r${i}"

    echo "[$(date)] Repeat $i done."
done

echo "All $N_REPEATS repeats complete. Run analysis/reliability.py to compute ICC."
