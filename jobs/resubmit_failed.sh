#!/bin/bash
# resubmit_failed.sh
# ────────────────────────────────────────────────────────────────────────────
# Resubmit only the failed tasks of a previous launch_all_metrics.sh run.
#
# For each metric JID you pass, sacct is queried for tasks that did NOT end
# in COMPLETED state (FAILED / CANCELLED / TIMEOUT / OUT_OF_MEMORY /
# NODE_FAIL / PREEMPTED, plus any still pending or running). Those task
# indices are resubmitted as a new array against the same INPUT_LIST so
# SLURM_ARRAY_TASK_ID still maps to the right input file. A merge array is
# then chained on with afterany so <base>_w_metrics.json is rebuilt with
# the new scratch files folded in.
#
# Usage:
#   bash resubmit_failed.sh \
#       [-i INPUT_LIST] \
#       [-c CRIMSON_JID] [-g GREEN_JID] [-n NLP_JID] [-d DISCERN_JID]
#
# At least one of -c/-g/-n/-d must be provided. INPUT_LIST defaults to the
# .metrics_inputs.txt that the original launcher wrote.
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern
INPUT_LIST="$REPO/jobs/.metrics_inputs.txt"

CRIMSON_JID=""; GREEN_JID=""; NLP_JID=""; DISCERN_JID=""
while getopts "i:c:g:n:d:h" opt; do
    case "$opt" in
        i) INPUT_LIST="$OPTARG" ;;
        c) CRIMSON_JID="$OPTARG" ;;
        g) GREEN_JID="$OPTARG" ;;
        n) NLP_JID="$OPTARG" ;;
        d) DISCERN_JID="$OPTARG" ;;
        h|*) sed -n '2,20p' "$0"; exit 0 ;;
    esac
done

if [ -z "$CRIMSON_JID$GREEN_JID$NLP_JID$DISCERN_JID" ]; then
    echo "error: pass at least one of -c/-g/-n/-d <JID>" >&2
    exit 2
fi
[ -f "$INPUT_LIST" ] || { echo "error: INPUT_LIST not found: $INPUT_LIST" >&2; exit 2; }

# ── Collect failed task indices for one job-array via sacct ─────────────────
# A task is considered "failed" if its State is anything other than COMPLETED.
# We look at the .batch step (which carries the real exit state on this
# cluster) plus the top-level row, and union both.
failed_indices() {
    local jid="$1"
    sacct -j "$jid" -X -n -P --format=JobID,State 2>/dev/null \
        | awk -F'|' '
            $2 != "COMPLETED" && $1 ~ /_[0-9]+$/ {
                n = split($1, parts, "_"); print parts[n]
            }' \
        | sort -un
}

# ── Resubmit one metric array against the failed-index list ─────────────────
resubmit() {
    local label="$1" script="$2" jid="$3"
    [ -z "$jid" ] && return 0

    local idxs
    idxs=$(failed_indices "$jid" | paste -sd, -)
    if [ -z "$idxs" ]; then
        echo "[$label] JID=$jid → no failed tasks"
        return 0
    fi

    echo "[$label] JID=$jid → failed indices: $idxs"
    local new_jid
    new_jid=$(sbatch --parsable --array="$idxs" "$script" "$INPUT_LIST")
    echo "[$label] resubmitted as JID=$new_jid"
    echo "$new_jid"
}

echo "Input list : $INPUT_LIST  ($(wc -l < "$INPUT_LIST") inputs)"
echo ""

# Collect new JIDs so we can chain a merge on top.
NEW_JIDS=()
for spec in \
    "CRIMSON|$REPO/jobs/_array_crimson.sh|$CRIMSON_JID" \
    "GREEN|$REPO/jobs/_array_green.sh|$GREEN_JID" \
    "NLP|$REPO/jobs/_array_nlp.sh|$NLP_JID" \
    "DISCERN|$REPO/jobs/_array_discern.sh|$DISCERN_JID"
do
    IFS='|' read -r label script jid <<<"$spec"
    out=$(resubmit "$label" "$script" "$jid")
    new=$(printf '%s\n' "$out" | tail -n1)
    [[ "$new" =~ ^[0-9]+$ ]] && NEW_JIDS+=("$new")
done

if [ "${#NEW_JIDS[@]}" -eq 0 ]; then
    echo ""
    echo "Nothing to resubmit. All tasks COMPLETED."
    exit 0
fi

# ── Merge over the union of failed indices (so all touched outputs refresh) ─
ALL_FAILED=$(
    { [ -n "$CRIMSON_JID" ] && failed_indices "$CRIMSON_JID";
      [ -n "$GREEN_JID"   ] && failed_indices "$GREEN_JID";
      [ -n "$NLP_JID"     ] && failed_indices "$NLP_JID";
      [ -n "$DISCERN_JID" ] && failed_indices "$DISCERN_JID"; } \
    | sort -un | paste -sd, -
)

DEP=$(IFS=:; echo "afterany:${NEW_JIDS[*]}")
echo ""
echo "Submitting MERGE  array=$ALL_FAILED  dependency=$DEP"
JID_M=$(sbatch --parsable --array="$ALL_FAILED" \
        --dependency="$DEP" \
        "$REPO/jobs/_array_merge.sh" "$INPUT_LIST")
echo "MERGE JID=$JID_M"

echo ""
echo "Track:  squeue -u \$USER -j $(IFS=,; echo "${NEW_JIDS[*]}"),$JID_M"
