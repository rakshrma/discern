#!/bin/bash
#SBATCH -J metrics_radcliq
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH -t 12:00:00
#SBATCH -o logs/array_radcliq_%A_%a.out
#SBATCH -e logs/array_radcliq_%A_%a.err
#SBATCH --partition=dgx-b200
#
# RadCliQ-v1 (Yu et al. 2022, CXR-Report-Metric) array task.
# Calls the canonical CXR-Report-Metric pipeline as-is via run_radcliq.py.
# Uses the radcliq env with the paper-pinned reqs.
# Invoked by jobs/launch_all_metrics.sh; do not submit standalone.

set -euo pipefail

REPO=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
ENV=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/envs/radcliq
PYTHON="$ENV/bin/python"
# The RadGraph step shells out to the `allennlp` CLI via os.system (which uses
# /bin/sh and searches PATH). Running $ENV/bin/python directly does not activate
# the env, so put its bin on PATH — otherwise allennlp is "not found" and the
# dygie step never writes temp_dygie_output.json (-> FileNotFoundError).
export PATH="$ENV/bin:$PATH"
# allennlp's `--include-package dygie` must import the dygie package, which is
# not pip-installed and lives in the CXR-Report-Metric checkout. Put it on
# PYTHONPATH so the allennlp subprocess can find it regardless of cwd.
export PYTHONPATH="/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/CXR-Report-Metric:${PYTHONPATH:-}"

INPUT_LIST="${1:?must pass INPUT_LIST as \$1}"
INPUT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$INPUT_LIST")
[ -n "$INPUT" ] && [ -f "$INPUT" ] || { echo "Bad INPUT: '$INPUT'" >&2; exit 2; }

MODELDIR=$(dirname "$INPUT")
BASE=$(basename "$INPUT" .json)
OUTDIR="$MODELDIR/.metrics"
mkdir -p "$OUTDIR"
OUTPUT="$OUTDIR/${BASE}.radcliq.json"

# Each task isolates its own caches (run_radcliq.py tempfile workdir) and its
# RadGraph scratch files (RADGRAPH_TMPDIR), so no start stagger is needed.
cd "$REPO"
echo "[RADCLIQ] task=$SLURM_ARRAY_TASK_ID  input=$INPUT  output=$OUTPUT"

$PYTHON scripts/run_radcliq.py \
    --input  "$INPUT" \
    --output "$OUTPUT"

echo "[RADCLIQ] done task=$SLURM_ARRAY_TASK_ID"
