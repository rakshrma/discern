#!/bin/bash
#SBATCH -J discern_entities
#SBATCH --cpus-per-task=28
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH -t 06:00:00
#SBATCH -o /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern/jobs/logs/entities_ablation_%j.out
#SBATCH -e /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern/jobs/logs/entities_ablation_%j.err
#SBATCH --partition=dgx-b200
#
# Entity-list ablation on ReXVal + RaDEvalX. Arms (ARMS, default "original updated"):
#   original : full DISCERN with config/entities_original.yaml
#   updated  : full DISCERN with config/entities.yaml
#   free     : free-DISCERN — single prompt, no entity list (config/freeform_prompt.yaml)
# Everything else (code, model, vLLM settings) is identical between arms.
# Mini-DISCERN, NLP and GREEN are skipped — they do not use entities.yaml.
#
# Outputs: data/discern_runs/entities_ablation[_$OUT_TAG]/<arm>/
#          {dataset}_all_metrics_google_gemma_4_31B_it_batch_v{0,1,2}.json
#
# Usage:
#   sbatch jobs/entities_ablation.sh                         # full run
#   SMOKE=1 sbatch -t 01:00:00 jobs/entities_ablation.sh     # 3 ReXVal pairs, 1 repeat
#   ARMS=free sbatch jobs/entities_ablation.sh               # free-DISCERN only
#   OUT_TAG=mig90 VLLM_TP=1 sbatch -p b200-mig90 --gpus-per-node=1 -c 14 -t 10:00:00 \
#       jobs/entities_ablation.sh                            # one 90 GB MIG slice
#   MODEL=databricks-claude-sonnet-4-6 OUT_TAG=sonnet sbatch -p genoa-std-mem \
#       --gpus-per-node=0 -c 4 -t 24:00:00 jobs/entities_ablation.sh   # API model on CPU
#   VLLM_ATTN_BACKEND=TRITON_ATTN ...   # force the attention backend (MedGemma's auto
#                                       # FLASHINFER choice fails to JIT-compile here)
#   REPS="1 2" ...                      # run only these repeats (default "0 1 2"); full runs
#                                       # skip any arm/repeat whose two output files exist

set -euo pipefail

WORKSPACE=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace
REPO=$WORKSPACE/discern
PYTHON=$WORKSPACE/envs/vllm/bin/python

MODEL=${MODEL:-google/gemma-4-31B-it}   # databricks-* models run via API (no GPU needed)
REXVAL_INPUT=$WORKSPACE/reads/data/rexval/rexval_reports_long.csv
RADEVALX_INPUT=$WORKSPACE/reads/data/radevalx/radevalx_report.csv

declare -A ENTITIES=(
    [original]=$REPO/config/entities_original.yaml
    [updated]=$REPO/config/entities.yaml
)
read -r -a ARM_LIST <<< "${ARMS:-original updated}"
for ARM in "${ARM_LIST[@]}"; do
    [[ "$ARM" == free || -n "${ENTITIES[$ARM]:-}" ]] || { echo "Unknown arm '$ARM'" >&2; exit 2; }
done

BASE_OUT=$REPO/data/discern_runs/entities_ablation${OUT_TAG:+_$OUT_TAG}
if [ "${SMOKE:-0}" = "1" ]; then
    OUT_ROOT=$BASE_OUT/_smoke
    REPEATS=(0)
    EXTRA=(--dataset rexval --count 3)
else
    OUT_ROOT=$BASE_OUT
    read -r -a REPEATS <<< "${REPS:-0 1 2}"
    EXTRA=(--dataset both)
fi
STEM=${MODEL//[\/-]/_}

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

# Model weights come from the existing HF cache (inherited HF_HOME); never download.
: "${HF_HOME:?HF_HOME must point at the cache holding google/gemma-4-31B-it}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# MIG slices are exposed as a UUID ("MIG-..."), which vLLM cannot parse as a
# device index; the job only sees its own slice, so index 0 is that slice.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
    echo "[ablation] MIG device ${CUDA_VISIBLE_DEVICES} -> CUDA_VISIBLE_DEVICES=0"
    export CUDA_VISIBLE_DEVICES=0
fi

# vLLM — same settings as the April reads/ runs (TP=2 on dgx-b200)
export VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TP:-2}
export VLLM_GPU_MEM_UTIL=0.90
export VLLM_MAX_MODEL_LEN=25000

cd "$REPO"
echo "[ablation] job=${SLURM_JOB_ID:-none} model=$MODEL tp=$VLLM_TENSOR_PARALLEL_SIZE attn=${VLLM_ATTN_BACKEND:-auto} arms=${ARM_LIST[*]} smoke=${SMOKE:-0} out=$OUT_ROOT"
nvidia-smi -L 2>/dev/null || true

for REP in "${REPEATS[@]}"; do
    for ARM in "${ARM_LIST[@]}"; do
        if [ "$ARM" = free ]; then
            METHOD=(--skip-discern --free-discern)
        else
            METHOD=(--entities-yaml "${ENTITIES[$ARM]}")
        fi
        if [ "${SMOKE:-0}" != "1" ] && \
           [ -f "$OUT_ROOT/$ARM/rexval_all_metrics_${STEM}_batch_v$REP.json" ] && \
           [ -f "$OUT_ROOT/$ARM/radevalx_all_metrics_${STEM}_batch_v$REP.json" ]; then
            echo "---- arm=$ARM  repeat=batch_v$REP  already done, skipping ----"
            continue
        fi
        echo ""
        echo "---- arm=$ARM  repeat=batch_v$REP  ${METHOD[*]} ----"
        $PYTHON scripts/run_discern.py \
            "${EXTRA[@]}" \
            "${METHOD[@]}" \
            --model          "$MODEL" \
            --run-tag        "batch_v$REP" \
            --output-dir     "$OUT_ROOT/$ARM" \
            --rexval-input   "$REXVAL_INPUT" \
            --radevalx-input "$RADEVALX_INPUT" \
            --skip-nlp --skip-green --skip-mini-discern
    done
done

# Smoke gate: run_discern.py exits 0 even when pairs fail, so fail the job if
# an arm wrote no results or scored none of its pairs (a dependent full run uses
# afterok). Single unscored pairs are tolerated — weaker models occasionally fail
# validation on a pair, and compare_arms.py drops such pairs from every arm.
if [ "${SMOKE:-0}" = "1" ]; then
    $PYTHON - "$OUT_ROOT" "$STEM" "${ARM_LIST[@]}" <<'EOF'
import json, sys
from pathlib import Path
root, stem, arms = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
files = sorted(f for a in arms for f in (root / a).glob(f"*_all_metrics_{stem}_*.json"))
all_ok = len(files) == len(arms)
for f in files:
    res = json.loads(f.read_text())["results"]
    n_ok = sum(r.get("discern_score") is not None or r.get("free_discern_score") is not None
               for r in res)
    all_ok &= n_ok > 0
    print(f"[smoke] {f.relative_to(root)}: {n_ok}/{len(res)} scored")
sys.exit(0 if all_ok else 1)
EOF
fi

echo "[ablation] done"
