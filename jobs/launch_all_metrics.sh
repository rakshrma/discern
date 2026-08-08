#!/bin/bash
# launch_all_metrics.sh
# ────────────────────────────────────────────────────────────────────────────
# Submit 4 SLURM array jobs (one per metric group) + 1 merge array job
# (depending on all 4) to score every result JSON under
# vlm_cxr_benchmark/results/<model>/ that does NOT already have '_w_metrics'
# in its filename.
#
# Layout:
#   <model>/<base>.json                   ← input
#   <model>/.metrics/<base>.crimson.json  ← per-metric scratch (auto-created)
#   <model>/.metrics/<base>.green.json
#   <model>/.metrics/<base>.nlp.json
#   <model>/.metrics/<base>.discern.json
#   <model>/<base>_w_metrics.json         ← merged final output
#
# Each metric uses its own conda env and partition:
#   CRIMSON | b200-mig45 | 1 GPU | envs/crimson
#   GREEN   | b200-mig45 | 1 GPU | green_score
#   NLP     | b200-mig45 | 1 GPU | envs/vllm
#   DISCERN | dgx-b200   | 2 GPU | envs/vllm
#
# Submit:
#   bash jobs/launch_all_metrics.sh              # all models
#   bash jobs/launch_all_metrics.sh 'qwen35-*'   # only Qwen3.5 models
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MODEL_PATTERN="${1:-*}"   # glob applied to model-directory name; default = all

REPO=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
RESULTS_ROOT=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/vlm_cxr_benchmark/results

mkdir -p "$REPO/logs"
INPUT_LIST="$REPO/jobs/.metrics_inputs.txt"

# ── Discover input files (any *.json under <model>/ without "_w_metrics") ────
# -path filter applies the MODEL_PATTERN to the immediate parent directory.
find "$RESULTS_ROOT" -mindepth 2 -maxdepth 2 -type f -name "*.json" \
    ! -name "*_w_metrics*" \
    -path "*/${MODEL_PATTERN}/*" | sort > "$INPUT_LIST"

N=$(wc -l < "$INPUT_LIST")
if [ "$N" -eq 0 ]; then
    echo "No input files found under $RESULTS_ROOT" >&2
    exit 1
fi
ARRAY_RANGE="0-$((N - 1))"

echo "Discovered $N input file(s):"
nl -ba "$INPUT_LIST"
echo ""
echo "Array range: $ARRAY_RANGE"
echo ""

# ── Pre-install editable package into each env (avoids array-task race) ──────
# Concurrent array tasks running `pip install -e .` against the same env
# corrupt __editable__.discern-*.pth. Install once per env up-front so each
# task only does `import discern`.
echo "Ensuring editable install in each env ..."
for ENV_PY in \
    /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/envs/crimson/bin/python \
    /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/envs/vllm/bin/python \
    /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/green_score/bin/python
do
    [ -x "$ENV_PY" ] || { echo "  skip (missing): $ENV_PY"; continue; }
    "$ENV_PY" -c "import discern" 2>/dev/null \
        || { echo "  installing into $ENV_PY"; \
             "$ENV_PY" -m pip install -e "$REPO" -q; }
done
echo ""

# ── Submit 4 metric array jobs ───────────────────────────────────────────────
echo "Submitting CRIMSON  array ..."
JID_C=$(sbatch --parsable --array="$ARRAY_RANGE" \
        "$REPO/jobs/_array_crimson.sh" "$INPUT_LIST")
echo "  → JID=$JID_C"

echo "Submitting GREEN    array ..."
JID_G=$(sbatch --parsable --array="$ARRAY_RANGE" \
        "$REPO/jobs/_array_green.sh" "$INPUT_LIST")
echo "  → JID=$JID_G"

echo "Submitting NLP      array ..."
JID_N=$(sbatch --parsable --array="$ARRAY_RANGE" \
        "$REPO/jobs/_array_nlp.sh" "$INPUT_LIST")
echo "  → JID=$JID_N"

echo "Submitting DISCERN  array ..."
JID_D=$(sbatch --parsable --array="$ARRAY_RANGE" \
        "$REPO/jobs/_array_discern.sh" "$INPUT_LIST")
echo "  → JID=$JID_D"

echo "Submitting RADCLIQ  array ..."
JID_R=$(sbatch --parsable --array="$ARRAY_RANGE" \
        "$REPO/jobs/_array_radcliq.sh" "$INPUT_LIST")
echo "  → JID=$JID_R"

# ── Submit merge job depending on all five ───────────────────────────────────
echo "Submitting MERGE    array (dependency: afterany on all five) ..."
JID_M=$(sbatch --parsable --array="$ARRAY_RANGE" \
        --dependency="afterany:${JID_C}:${JID_G}:${JID_N}:${JID_D}:${JID_R}" \
        "$REPO/jobs/_array_merge.sh" "$INPUT_LIST")
echo "  → JID=$JID_M"

echo ""
echo "All jobs submitted:"
echo "  CRIMSON : $JID_C"
echo "  GREEN   : $JID_G"
echo "  NLP     : $JID_N"
echo "  DISCERN : $JID_D"
echo "  RADCLIQ : $JID_R"
echo "  MERGE   : $JID_M  (runs after all 5 complete; uses afterany so partial failures still merge)"
echo ""
echo "Track:  squeue -u \$USER -j ${JID_C},${JID_G},${JID_N},${JID_D},${JID_R},${JID_M}"
echo "Logs:   $REPO/logs/"
