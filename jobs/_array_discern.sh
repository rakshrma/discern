#!/bin/bash
#SBATCH -J metrics_discern
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH -t 40:00:00
#SBATCH -o logs/array_discern_%A_%a.out
#SBATCH -e logs/array_discern_%A_%a.err
#SBATCH --partition=dgx-b200
#
# DISCERN + mini-DISCERN array task — runs both pipelines on one input file.
# Uses google/gemma-4-31B-it via vLLM in batch mode (TP=2, requires 2 GPUs).
# Invoked by jobs/launch_all_metrics.sh; do not submit standalone.

set -euo pipefail

REPO=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
ENV=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/envs/vllm
PYTHON="$ENV/bin/python"

MODEL="google/gemma-4-31B-it"

INPUT_LIST="${1:?must pass INPUT_LIST as $1}"
INPUT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$INPUT_LIST")
[ -n "$INPUT" ] && [ -f "$INPUT" ] || { echo "Bad INPUT: '$INPUT'" >&2; exit 2; }

MODELDIR=$(dirname "$INPUT")
BASE=$(basename "$INPUT" .json)
OUTDIR="$MODELDIR/.metrics"
mkdir -p "$OUTDIR"
OUTPUT="$OUTDIR/${BASE}.discern.json"

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

# HuggingFace cache + offline mode. With ~20 array tasks starting at once,
# vLLM's startup `list_repo_files` call against the Hub blows past the
# anonymous 1000-req/5min quota. Pointing HF_HOME at the populated cache
# and forcing offline mode makes vLLM resolve the model from disk.
export HF_HOME=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/cache/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# vLLM
export VLLM_TENSOR_PARALLEL_SIZE=2
export VLLM_GPU_MEM_UTIL=0.90
export VLLM_MAX_MODEL_LEN=25000

cd "$REPO"
echo "[DISCERN] task=$SLURM_ARRAY_TASK_ID  input=$INPUT  output=$OUTPUT  model=$MODEL"

$PYTHON scripts/run_metrics.py \
    --input   "$INPUT" \
    --output  "$OUTPUT" \
    --model   "$MODEL" \
    --metrics discern mini_discern \
    --skip-nlp --skip-green --skip-crimson

echo "[DISCERN] done task=$SLURM_ARRAY_TASK_ID"
