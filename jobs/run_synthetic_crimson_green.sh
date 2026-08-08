#!/bin/bash
#SBATCH -J discern_synthetic_crimson_green
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -t 03:00:00
#SBATCH -o logs/synthetic_crimson_green_%j.out
#SBATCH -e logs/synthetic_crimson_green_%j.err
#SBATCH --partition=b200-mig45
#
# Score data/synthetic/synthetic_reports.json with CRIMSON and GREEN
# in a single SLURM job, run sequentially because they have
# incompatible Python envs and cannot share a parent interpreter.
#
#   Step 1: CRIMSON  (rajpurkarlab/medgemma-4b-it-crimson, ~4B)
#           parent = $CRIMSON_ENV/bin/python
#   Step 2: GREEN    (StanfordAIMI/GREEN-radllama2-7b,    ~7B)
#           parent = $GREEN_ENV/bin/python
#
# Both writes target the same output JSON; run_metrics.py is resume-safe and
# merges new scores into the existing file, so the final JSON has both
# `crimson` and `green` columns per row.
#
# Submit:   sbatch jobs/run_synthetic_crimson_green.sh

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
CRIMSON_ENV=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/envs/crimson
GREEN_ENV=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/green_score

CRIMSON_PY="$CRIMSON_ENV/bin/python"
GREEN_PY="$GREEN_ENV/bin/python"

INPUT="$REPO/data/synthetic/synthetic_reports.json"
OUTPUT="$REPO/data/synthetic/synthetic_scored_crimson_green.json"

# ── CUDA setup ────────────────────────────────────────────────────────────────
if command -v module &>/dev/null; then module load cuda 2>/dev/null || true; fi
if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
    for _c in "$(dirname "$(dirname "$(which nvcc 2>/dev/null)")")" \
              /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.6 \
              /usr/local/cuda-12.4 /usr/local/cuda-12.1; do
        if [ -d "$_c" ] && [ -x "$_c/bin/nvcc" ]; then CUDA_HOME="$_c"; break; fi
    done
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false

# ── Diagnostics ───────────────────────────────────────────────────────────────
mkdir -p "$REPO/logs"
cd "$REPO"

echo "================================================================"
echo "Job        : ${SLURM_JOB_ID:-interactive}"
echo "Node       : $(hostname)"
echo "GPUs       : $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Input      : $INPUT"
echo "Output     : $OUTPUT"
echo "================================================================"

# ── Install discern into each env if not already present ──────────────────────
$CRIMSON_PY -c "import discern" 2>/dev/null || $CRIMSON_PY -m pip install -e "$REPO" -q
$GREEN_PY   -c "import discern" 2>/dev/null || $GREEN_PY   -m pip install -e "$REPO" -q

# ── Step 1: CRIMSON ───────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "[1/2] CRIMSON  via $CRIMSON_PY"
echo "================================================================"
$CRIMSON_PY scripts/run_metrics.py \
    --input   "$INPUT" \
    --output  "$OUTPUT" \
    --metrics crimson \
    --skip-green --skip-discern --skip-mini-discern

# Free GPU memory between steps
nvidia-smi 2>/dev/null | head -20 || true

# ── Step 2: GREEN ─────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "[2/2] GREEN    via $GREEN_PY"
echo "================================================================"
$GREEN_PY scripts/run_metrics.py \
    --input   "$INPUT" \
    --output  "$OUTPUT" \
    --metrics green \
    --skip-nlp --skip-crimson --skip-discern --skip-mini-discern

echo ""
echo "================================================================"
echo "Done. Combined CRIMSON + GREEN output: $OUTPUT"
echo "================================================================"
