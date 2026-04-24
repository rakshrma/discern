#!/bin/bash
#SBATCH -J all_metrics
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH -t 12:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --partition=dgx-b200

# Full metric suite: NLP + BERTScore + RaTEScore + GREEN + DISCERN + mini-DISCERN
# (CRIMSON is excluded; run jobs/crimson.sh separately in the crimson conda env)
# Requires 2× GPU on the SAME node for vLLM tensor-parallel loading of 31B backbone.

# ── Environment ──────────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
# conda activate /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/green_score
conda activate reads

# ── CUDA setup (required for vLLM custom-kernel compilation) ─────────────────
if command -v module &>/dev/null; then
    module load cuda 2>/dev/null || true
fi

if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
    for _candidate in \
        "$(dirname "$(dirname "$(which nvcc 2>/dev/null)")")" \
        /usr/local/cuda \
        /usr/local/cuda-12.8 \
        /usr/local/cuda-12.6 \
        /usr/local/cuda-12.4 \
        /usr/local/cuda-12.1 \
        /usr/local/cuda-12.0 \
        /usr/local/cuda-11.8 \
        /opt/cuda \
        "$CONDA_PREFIX"
    do
        if [ -d "$_candidate" ] && [ -x "$_candidate/bin/nvcc" ]; then
            CUDA_HOME="$_candidate"
            break
        fi
    done
fi

if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
    echo "[ERROR] Could not locate a CUDA installation. Set CUDA_HOME manually." >&2
    exit 1
fi

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "[INFO] CUDA_HOME=$CUDA_HOME  nvcc=$(which nvcc 2>/dev/null || echo 'not found')"

DISCERN_ROOT=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
cd "$DISCERN_ROOT"

# ── HF token (needed for gated gemma-4 checkpoints) ──────────────────────────
HF_TOKEN_PATH="${HF_TOKEN_PATH:-$DISCERN_ROOT/config/.hftoken}"
if [ -f "$HF_TOKEN_PATH" ]; then
    export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"
fi

# ── vLLM runtime settings (read by src/call_llm.py:_load_vllm_model) ────────
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-25000}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"

# ── Parameters ───────────────────────────────────────────────────────────────
INPUT="${INPUT:?ERROR: set INPUT to your results JSON/CSV}"
OUTPUT="${OUTPUT:?ERROR: set OUTPUT to desired output path}"
BACKEND="${BACKEND:-hf}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
TAG="${TAG:-}"

mkdir -p "$(dirname "$OUTPUT")" logs

echo "[$(date)] Running all metrics on $INPUT → $OUTPUT"

python scripts/run_metrics.py \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    --metrics all \
    --backend "$BACKEND" \
    --model   "$MODEL" \
    --max-concurrent "$MAX_CONCURRENT" \
    ${TAG:+--tag "$TAG"}

echo "[$(date)] Done: $OUTPUT"
