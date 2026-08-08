#!/bin/bash
#SBATCH -J discern_api
#SBATCH --cpus-per-task=4
#SBATCH -t 24:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
# No GPU needed — inference runs via Databricks API.

# ── Environment ───────────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate discern          # update to your conda env name

REPO=/path/to/discern           # update to your discern repo path
cd "$REPO"

# ── Models to evaluate ────────────────────────────────────────────────────────
MODELS=(
    "databricks-claude-sonnet-4-6"
    "databricks-claude-opus-4-5"
    "databricks-claude-haiku-4-5"
)

# ── Run DISCERN + mini-DISCERN for each model × N repeats ────────────────────
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
