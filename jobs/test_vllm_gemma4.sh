#!/bin/bash
#SBATCH -J test_discern_vllm
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH -t 01:00:00
#SBATCH -o logs/test_vllm_%j.out
#SBATCH -e logs/test_vllm_%j.err
#SBATCH --partition=dgx-b200

# Test 2: DISCERN + mini-DISCERN on one report pair using google/gemma-4-31B-it via vLLM.
# Requires >= 2× GPU (31B model needs ~70GB VRAM in bfloat16).

REPO=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
VLLM_ENV=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/envs/vllm
PYTHON="$VLLM_ENV/bin/python"

# ── CUDA setup ────────────────────────────────────────────────────────────────
if command -v module &>/dev/null; then module load cuda 2>/dev/null || true; fi
if [ -z "$CUDA_HOME" ] || [ ! -d "$CUDA_HOME" ]; then
    for _c in "$(dirname "$(dirname "$(which nvcc 2>/dev/null)")")" \
              /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.6; do
        if [ -d "$_c" ] && [ -x "$_c/bin/nvcc" ]; then CUDA_HOME="$_c"; break; fi
    done
fi
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# ── vLLM settings ─────────────────────────────────────────────────────────────
export VLLM_TENSOR_PARALLEL_SIZE=2
export VLLM_GPU_MEM_UTIL=0.90
export VLLM_MAX_MODEL_LEN=25000

echo "Node: $(hostname)  GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "vLLM version: $($PYTHON -c 'import vllm; print(vllm.__version__)')"

# ── Install discern if not already installed ───────────────────────────────────
$PYTHON -c "import discern" 2>/dev/null || $PYTHON -m pip install -e "$REPO" -q

# ── Run test ───────────────────────────────────────────────────────────────────
cd "$REPO"
mkdir -p logs

echo ""
echo "================================================================"
echo "Test 1: DISCERN via google/gemma-4-31B-it (vLLM)"
echo "================================================================"

$PYTHON tests/test_one_pair.py --only discern --model google/gemma-4-31B-it

echo ""
echo "================================================================"
echo "Test 2: mini-DISCERN via google/gemma-4-31B-it (vLLM)"
echo "================================================================"

$PYTHON tests/test_one_pair.py --only mini_discern --model google/gemma-4-31B-it

echo ""
echo "================================================================"
echo "Finished. Check output above for mini_discern_score."
echo "================================================================"
