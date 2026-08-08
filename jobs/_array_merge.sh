#!/bin/bash
#SBATCH -J metrics_merge
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH -t 40:00:00
#SBATCH -o logs/array_merge_%A_%a.out
#SBATCH -e logs/array_merge_%A_%a.err
#SBATCH --partition=dgx-b200
#
# Merge step — combines per-metric scratch JSONs into <base>_w_metrics.json.
# Runs after the four metric arrays via afterany dependency, so partial
# failures (e.g. one metric env breaks) still produce a merged output.
# Invoked by jobs/launch_all_metrics.sh; do not submit standalone.

set -euo pipefail

REPO=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
ENV=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/envs/vllm
PYTHON="$ENV/bin/python"

INPUT_LIST="${1:?must pass INPUT_LIST as $1}"
INPUT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$INPUT_LIST")
[ -n "$INPUT" ] && [ -f "$INPUT" ] || { echo "Bad INPUT: '$INPUT'" >&2; exit 2; }

echo "[MERGE] task=$SLURM_ARRAY_TASK_ID  input=$INPUT"
$PYTHON "$REPO/jobs/_merge_metrics.py" "$INPUT"
echo "[MERGE] done task=$SLURM_ARRAY_TASK_ID"
