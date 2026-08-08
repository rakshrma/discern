#!/bin/bash
#SBATCH -J paraphrase_eval
#SBATCH --cpus-per-task=4
#SBATCH -t 24:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate discern

REPO=/path/to/discern
cd "$REPO"

MODEL="databricks-claude-sonnet-4-6"

python scripts/run_discern_paraphrase.py \
    --input      data/robustness/paraphrase_robustness.json \
    --output-dir data/discern_runs/paraphrase \
    --model      "$MODEL"

echo "Paraphrase evaluation complete."
