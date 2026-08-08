#!/bin/bash
#SBATCH -J discern_synthetic_vllm
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH -t 02:00:00
#SBATCH -o logs/synthetic_vllm_%j.out
#SBATCH -e logs/synthetic_vllm_%j.err
#SBATCH --partition=dgx-b200
#
# Run DISCERN + mini-DISCERN on the synthetic CXR pairs in
# data/synthetic/synthetic_reports.json using google/gemma-4-31B-it served
# locally via vLLM in batch mode.
#
# Model size note: gemma-4-31B-it needs ~70GB VRAM in bfloat16, so 2 GPUs
# with tensor parallelism are required. --nodes=1 keeps both GPUs
# co-located (vLLM TP does not support multi-node).
#
# Submit:   sbatch jobs/run_synthetic_vllm.sh
# Interact: bash   jobs/run_synthetic_vllm.sh

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
VLLM_ENV=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/envs/vllm
PYTHON="$VLLM_ENV/bin/python"

MODEL="google/gemma-4-31B-it"
INPUT="$REPO/data/synthetic/synthetic_reports.json"
OUTPUT="$REPO/data/synthetic/synthetic_scored_gemma4_31b.json"

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

# ── vLLM settings ─────────────────────────────────────────────────────────────
export VLLM_TENSOR_PARALLEL_SIZE=2     # match --gpus-per-node above
export VLLM_GPU_MEM_UTIL=0.90
export VLLM_MAX_MODEL_LEN=25000
export TOKENIZERS_PARALLELISM=false

# ── Diagnostics ───────────────────────────────────────────────────────────────
mkdir -p "$REPO/logs"
cd "$REPO"

echo "================================================================"
echo "Job        : ${SLURM_JOB_ID:-interactive}"
echo "Node       : $(hostname)"
echo "GPUs       : $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Python     : $PYTHON"
echo "vLLM ver   : $($PYTHON -c 'import vllm; print(vllm.__version__)' 2>&1)"
echo "Model      : $MODEL"
echo "Input      : $INPUT"
echo "Output     : $OUTPUT"
echo "TP / GPU%  : $VLLM_TENSOR_PARALLEL_SIZE / $VLLM_GPU_MEM_UTIL"
echo "Max len    : $VLLM_MAX_MODEL_LEN"
echo "================================================================"

# ── Install discern into the vllm env if not already there ────────────────────
$PYTHON -c "import discern" 2>/dev/null || $PYTHON -m pip install -e "$REPO" -q

# ── Run DISCERN + mini-DISCERN via vLLM batch ─────────────────────────────────
$PYTHON scripts/run_metrics.py \
    --input   "$INPUT" \
    --output  "$OUTPUT" \
    --model   "$MODEL" \
    --metrics discern mini_discern \
    --skip-nlp --skip-green --skip-crimson

echo ""
echo "================================================================"
echo "Done. Scored output: $OUTPUT"
echo "================================================================"
