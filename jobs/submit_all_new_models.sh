#!/bin/bash
# Re-run all metrics + CRIMSON on every model in the leaderboard using the
# gemma-4-31b DISCERN backbone (previous runs used Claude Sonnet 4.6).
#
# Chains each model: all_metrics.sh → crimson.sh (via --dependency=afterok).

set -euo pipefail

cd /vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/discern

OUT=/vast/projects/witschey/pmbb-vision/research_projects/rakshrma/workspace/vlm_cxr_benchmark/results
CBEECHE_FT=/vast/projects/witschey/corlab-foundational-mode/people/cbeeche/research_projects/xray/analysis_0/finetune_results
CBEECHE_QWEN=/vast/projects/witschey/corlab-foundational-mode/people/cbeeche/research_projects/xray/analysis_0/qwen_3vl_finetune_results

# model_key : absolute_path_to_raw_results_json
declare -A INPUTS=(
  [chexone]="$OUT/chexone/chexpert_plus_valid_results.json"
  # [medgemma-1b5]="$OUT/medgemma-1b5/chexpert_plus_valid_results.json"
  # [medgemma-4b]="$OUT/medgemma-4b/chexpert_plus_valid_results.json"
  # [chexagent]="$CBEECHE_FT/chexagent/chexpert_plus_valid_results.json"
  # [penn-chexagent-lora]="$CBEECHE_FT/chexagent_lora/chexpert_plus_valid_results.json"
  # [penn-qwen-langtower]="$CBEECHE_QWEN/qwen3_vl_8b_langtower/chexpert_plus_valid_results.json"
  # [penn-qwen-vision-ft]="$CBEECHE_FT/qwen3_vl_8b_visft/chexpert_plus_valid_results.json"
  # [qwen3-vl-30b]="$CBEECHE_FT/qwen3vl_30b/chexpert_plus_valid_results.json"
  # [qwen3-vl-8b]="$CBEECHE_FT/qwen3vl_8b/chexpert_plus_valid_results.json"
  # [qwen3-vl-8b-ft]="$CBEECHE_FT/qwen3_vl_8b_ft/chexpert_plus_valid_results.json"
)

missing=0
for d in "${!INPUTS[@]}"; do
  if [[ ! -f "${INPUTS[$d]}" ]]; then
    echo "MISSING: $d → ${INPUTS[$d]}"
    missing=1
  fi
done
[[ $missing -eq 1 ]] && { echo "Aborting: inputs missing above."; exit 1; }

for d in "${!INPUTS[@]}"; do
  in="${INPUTS[$d]}"
  out="$OUT/$d/chexpert_plus_qwen_valid_w_metrics_gemma4.json"
  mkdir -p "$(dirname "$out")"

  echo "==> $d"
  echo "    INPUT  = $in"
  echo "    OUTPUT = $out"

  metrics_jid=$(INPUT="$in" OUTPUT="$out" \
                sbatch --parsable jobs/all_metrics.sh)
  echo "    all_metrics jobid = $metrics_jid"

  crimson_jid=$(INPUT="$out" OUTPUT="$out" \
                sbatch --parsable --dependency=afterok:"$metrics_jid" \
                jobs/crimson.sh)
  echo "    crimson     jobid = $crimson_jid (waits on $metrics_jid)"
done

echo
echo "All jobs submitted. Track with: squeue -u $USER"
