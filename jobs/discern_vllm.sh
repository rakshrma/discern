#!/bin/bash
#SBATCH -J discern_vllm
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH -t 96:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
# Requires GPU node. --nodes=1 --gpus-per-node=N ensures all GPUs are on
# the same node (vLLM tensor parallelism does not support multi-node).

# ── Environment ───────────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate vllm             # update to your vLLM conda env name

REPO=/path/to/discern           # update to your discern repo path
cd "$REPO"

# ── CUDA setup ────────────────────────────────────────────────────────────────
if command -v module &>/dev/null; then
    module load cuda 2>/dev/null || true
fi

if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
    for _candidate in \
        "$(dirname "$(dirname "$(which nvcc 2>/dev/null)")")" \
        /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.6 \
        /usr/local/cuda-12.4 /usr/local/cuda-12.1 /usr/local/cuda-12.0
    do
        if [ -d "$_candidate" ] && [ -x "$_candidate/bin/nvcc" ]; then
            CUDA_HOME="$_candidate"
            break
        fi
    done
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# ── vLLM settings ─────────────────────────────────────────────────────────────
export VLLM_TENSOR_PARALLEL_SIZE=2   # match --gpus-per-node above
export VLLM_GPU_MEM_UTIL=0.90
export VLLM_MAX_MODEL_LEN=25000

# ── Models to evaluate ────────────────────────────────────────────────────────
MODELS=(
    "google/gemma-4-31B-it"
    "google/gemma-4-27b-it"
    "Qwen/Qwen3-30B-A3B"
)

# ── Run for each model × N repeats ───────────────────────────────────────────
REPEATS=(0 1 2)

for MODEL in "${MODELS[@]}"; do
    for REP in "${REPEATS[@]}"; do
        RUN_TAG="v${REP}"

        echo ""
        echo "================================================================"
        echo "Model: $MODEL  |  Run: $RUN_TAG"
        echo "================================================================"

        python scripts/run_discern.py \
            --dataset both \
            --model "$MODEL" \
            --run-tag "$RUN_TAG" \
            --skip-green

        echo "Done: $MODEL / $RUN_TAG"
    done
done

echo "All done."
