#!/bin/bash
#SBATCH -J metrics_nlp
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -t 40:00:00
#SBATCH -o logs/array_nlp_%A_%a.out
#SBATCH -e logs/array_nlp_%A_%a.err
#SBATCH --partition=dgx-b200
#
# NLP-metrics array task — BLEU, ROUGE, METEOR, BERTScore, RadGraph.
# Uses the vllm conda env (has bert_score / radgraph / nltk).
# Invoked by jobs/launch_all_metrics.sh; do not submit standalone.

set -euo pipefail

REPO=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
# green_score env pins transformers 4.40, which is the only env where the
# bundled radgraph wheel still works (it relies on the public encode_plus API
# that transformers ≥5 removed). bert_score / nltk / rouge are also present.
ENV=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/green_score
PYTHON="$ENV/bin/python"

INPUT_LIST="${1:?must pass INPUT_LIST as $1}"
INPUT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$INPUT_LIST")
[ -n "$INPUT" ] && [ -f "$INPUT" ] || { echo "Bad INPUT: '$INPUT'" >&2; exit 2; }

MODELDIR=$(dirname "$INPUT")
BASE=$(basename "$INPUT" .json)
OUTDIR="$MODELDIR/.metrics"
mkdir -p "$OUTDIR"
OUTPUT="$OUTDIR/${BASE}.nlp.json"

# CUDA (BERTScore / RadGraph use GPU when available)
if command -v module &>/dev/null; then module load cuda 2>/dev/null || true; fi
if [ -z "${CUDA_HOME:-}" ] || [ ! -d "$CUDA_HOME" ]; then
    for _c in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4; do
        [ -x "$_c/bin/nvcc" ] && { export CUDA_HOME="$_c"; break; }
    done
fi
export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false

# Read model weights (roberta-large for BERTScore, radgraph-xl, GREEN) straight
# from the already-warm shared HF cache. Offline mode skips the network HEAD
# request and, crucially, the download filelock — many tasks hitting the same
# lock file on the NFS cache concurrently caused "[Errno 116] Stale file handle"
# failures. All required models are pre-cached, so no download is needed.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"
echo "[NLP] task=$SLURM_ARRAY_TASK_ID  input=$INPUT  output=$OUTPUT"

$PYTHON scripts/run_metrics.py \
    --input   "$INPUT" \
    --output  "$OUTPUT" \
    --metrics nlp \
    --skip-green --skip-crimson --skip-discern --skip-mini-discern

echo "[NLP] done task=$SLURM_ARRAY_TASK_ID"
