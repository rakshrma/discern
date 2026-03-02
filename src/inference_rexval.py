from evaluate_reports import run_evaluation
from call_llm import _unload_hf_pipeline, _is_db  # import these
import pandas as pd
import os

# ============================================================
# -------------------- Global Configuration ------------------
# ============================================================
TOKEN_PATH            = "../.databricks.token"
PROMPT_YAML_PATH      = "../config/entity_extraction_prompt.yaml"
ENTITIES_YAML_PATH    = "../config/entities.yaml"
ATTRIBUTE_PROMPT_PATH = "../config/attribute_extraction_prompt.yaml"
SIGNIFICANCE_YAML_PATH= "../config/significance_prompt.yaml"
REPORT_PATH           = "../data/rexval/rexval_reports_long.csv"
OUTPUT_DIR            = "../data/rexval/"


def evaluate_rexval(model: str, output_path: str):
    # fix 1: create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rexval_report_pairs = pd.read_csv(REPORT_PATH)

    # --- Resume support: skip already completed rows ---
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        completed_ids = set(existing["study_id"].tolist())
        print(f"Resuming: {len(completed_ids)} rows already done.")
        write_header = False
    else:
        completed_ids = set()
        write_header = True

    for row in rexval_report_pairs.itertuples():
        if row.study_id in completed_ids:
            print(f"Skipping study_id={row.study_id} (already done)")
            continue

        print(f"Evaluating study {row.study_id} | reporter {row.candidate_reporter}")
        try:
            result = run_evaluation(
                report_text=row.gt_report,
                candidate_text=row.candidate_report,
                model=model,
                token_path=TOKEN_PATH,
                prompt_yaml_path=PROMPT_YAML_PATH,
                entities_yaml_path=ENTITIES_YAML_PATH,
                attribute_prompt_path=ATTRIBUTE_PROMPT_PATH,
                significance_yaml_path=SIGNIFICANCE_YAML_PATH,
            )
        except Exception as e:
            print(f"Error for study_id={row.study_id}: {e}")
            result = None

        # Write row immediately to CSV
        original_row = rexval_report_pairs.loc[row.Index].to_dict()
        original_row["reads_eval"] = result
        pd.DataFrame([original_row]).to_csv(
            output_path,
            mode="a",
            header=write_header,
            index=False,
        )
        write_header = False
        print(f"Saved study_id={row.study_id}")


if __name__ == "__main__":
    model_list = [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "google/medgemma-27b-it",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]

    for model in model_list:
        output_path = os.path.join(
            OUTPUT_DIR,
            f"rexval_reads_evaluation_{model.replace('-', '_').replace('/','_')}.csv"
        )
        print(f"\n{'='*50}\nRunning model: {model}\n{'='*50}")
        try:
            evaluate_rexval(model=model, output_path=output_path)
        except Exception as e:
            print(f"Fatal error for model {model}: {e}")
        finally:
            # fix 2: unload HF model after each model finishes (or fails)
            # skip for databricks models since they don't use GPU
            if not _is_db(model):
                _unload_hf_pipeline()