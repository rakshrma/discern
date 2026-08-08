#!/bin/bash
#SBATCH -J paraphrase_generate
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate discern

REPO=/path/to/discern
cd "$REPO"

python scripts/run_paraphrase.py \
    --rexval   data/rexval/rexval_reports_long.csv \
    --radevalx data/radevalx/radevalx_report.csv \
    --output   data/robustness/paraphrase_robustness.json \
    --model    databricks-gpt-oss-120b \
    --seed 42

echo "Paraphrase generation complete."
