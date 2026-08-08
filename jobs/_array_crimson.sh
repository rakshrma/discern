#!/bin/bash
#SBATCH -J metrics_crimson
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -t 40:00:00
#SBATCH -o logs/array_crimson_%A_%a.out
#SBATCH -e logs/array_crimson_%A_%a.err
#SBATCH --partition=dgx-b200
#
# CRIMSON array task — one input file per array index.
# Invoked by jobs/launch_all_metrics.sh; do not submit standalone.

set -euo pipefail

REPO=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
ENV=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/envs/crimson
PYTHON="$ENV/bin/python"

INPUT_LIST="${1:?must pass INPUT_LIST as $1}"
INPUT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$INPUT_LIST")
[ -n "$INPUT" ] && [ -f "$INPUT" ] || { echo "Bad INPUT: '$INPUT'" >&2; exit 2; }

MODELDIR=$(dirname "$INPUT")
BASE=$(basename "$INPUT" .json)
OUTDIR="$MODELDIR/.metrics"
mkdir -p "$OUTDIR"
OUTPUT="$OUTDIR/${BASE}.crimson.json"

# CUDA
if command -v module &>/dev/null; then module load cuda 2>/dev/null || true; fi
if [ -z "${CUDA_HOME:-}" ] || [ ! -d "$CUDA_HOME" ]; then
    for _c in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4; do
        [ -x "$_c/bin/nvcc" ] && { export CUDA_HOME="$_c"; break; }
    done
fi
export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "$REPO"
echo "[CRIMSON] task=$SLURM_ARRAY_TASK_ID  input=$INPUT  output=$OUTPUT"

# Derive the subprocess timeout from remaining SLURM walltime (5-min cleanup
# buffer). If SLURM_JOB_END_TIME isn't exported, fall back to 35h.
if [ -n "${SLURM_JOB_END_TIME:-}" ]; then
    TIMEOUT=$((SLURM_JOB_END_TIME - $(date +%s) - 300))
else
    TIMEOUT=126000
fi
echo "[CRIMSON] subprocess timeout=${TIMEOUT}s"

$PYTHON -u scripts/run_metrics.py \
    --input   "$INPUT" \
    --output  "$OUTPUT" \
    --metrics crimson \
    --timeout "$TIMEOUT" \
    --skip-nlp --skip-green --skip-discern --skip-mini-discern

echo "[CRIMSON] done task=$SLURM_ARRAY_TASK_ID"
