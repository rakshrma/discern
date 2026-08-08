#!/bin/bash
# run_custom.sh — Run DISCERN + metrics on a user-provided CSV or JSON file.
#
# Edit the variables below, then submit with:
#   sbatch jobs/run_custom.sh
#
# Or run interactively:
#   bash jobs/run_custom.sh
#
# For GPU models (HuggingFace), uncomment the SBATCH GPU lines.

#SBATCH -J discern_custom
#SBATCH --cpus-per-task=4
#SBATCH -t 04:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
# For GPU (vLLM / HuggingFace models), uncomment:
##SBATCH --nodes=1
##SBATCH --gpus-per-node=2
##SBATCH --partition=gpu

# ── User settings — edit these ────────────────────────────────────────────────

INPUT="path/to/your_pairs.csv"      # CSV with reference,candidate columns
                                     # OR JSON with {results: [{ground_truth_raw, generated_raw}]}
OUTPUT="path/to/output_scored.json"

# Model to use. Databricks models auto-use API; HF paths auto-use vLLM.
MODEL="databricks-claude-sonnet-4-6"
# MODEL="google/gemma-4-31B-it"     # uncomment for local GPU inference

# Optional: process only first N rows for a quick smoke test.
# Set to "" to process all rows.
COUNT=""
# COUNT="--count 10"

# Metrics to run. Options: nlp, model (green+crimson), discern, all
# Default (empty): uses settings from config.yaml
METRICS=""
# METRICS="--metrics nlp discern"

# ── Environment ───────────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate discern              # update to your conda env name

REPO=/path/to/discern               # update to your discern repo path
cd "$REPO"

# For GPU models, set vLLM env vars:
# export VLLM_TENSOR_PARALLEL_SIZE=2
# export VLLM_GPU_MEM_UTIL=0.90
# export VLLM_MAX_MODEL_LEN=25000

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "Model:  $MODEL"

python scripts/run_metrics.py \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    --model  "$MODEL" \
    ${COUNT} \
    ${METRICS}

echo "Done. Results saved to: $OUTPUT"
